// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recognizes a subtask, reads its Pareto candidates, and routes to the first one.
//!
//! A small recognizer model labels the current step with a subtask ID. Only then does
//! the router read that subtask's Pareto candidates, order them with the configured
//! metric, and append the selected candidate's skill reminder to the conversation.
//! When recognition is missing, uncertain, or unmatched, the router sends the request
//! to the flagship model without reading any candidates.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use switchyard_protocol::{
    ContentBlock, LlmRequest, Message, ModelId, Request, Role, WireFormat, completion_text,
};

use super::util::prompts::insert_message;
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity, RoutingOutcome};
use crate::{LibsyError, Result};

/// Upper bound on remembered predictions. A conversation dropped at the bound
/// starts over without a previous prediction.
const MAX_TRACKED_CONVERSATIONS: usize = 1_024;

/// The packaged recognition prompt.
pub const DEFAULT_RSI_RECOGNIZER_PROMPT: &str = include_str!("../prompts/rsi/recognize.md");

/// How the router orders the Pareto candidates of a recognized subtask.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RsiSelection {
    /// Try the highest success-rate candidate first.
    HighestSuccess,
    /// Try the lowest cost candidate first. This is the default.
    #[default]
    LowestCost,
    /// Try the lowest latency candidate first.
    LowestLatency,
}

/// One candidate model in a subtask's Pareto set.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubtaskCandidate {
    /// Configured target model ID.
    pub model_id: ModelId,
    /// Routing cost metric. `RsiSelection::LowestCost` tries the smallest first.
    pub cost: f64,
    /// Routing latency in milliseconds. `RsiSelection::LowestLatency` tries the
    /// smallest first.
    pub latency_ms: f64,
    /// Share of steps this model is expected to answer well, from 0 to 1.
    /// `RsiSelection::HighestSuccess` tries the largest first.
    pub success_rate: f64,
    /// Skill reminder injected while this candidate's subtask is active.
    /// Missing or blank means no reminder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_content: Option<String>,
    /// Version of `skill_content`. When several candidates carry a skill, the
    /// highest version is injected. Missing or unparsable reads as `0.0`.
    #[serde(default)]
    pub skill_version: f64,
    /// Whether this candidate may supply a skill. Invalid candidates are
    /// skipped when picking the newest skill. Routing still considers them.
    #[serde(default = "default_true")]
    pub is_valid: bool,
}

fn default_true() -> bool {
    true
}

/// The subtask facts the recognizer reads before labeling a step.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubtaskSummary {
    /// Stable subtask identifier.
    pub subtask_id: String,
    /// Human-readable subtask name shown to the recognizer.
    #[serde(default)]
    pub subtask_name: String,
    /// What the subtask does, its inputs, its expected output, and its typical
    /// failure points. The recognizer uses it to decide membership.
    #[serde(default)]
    pub definition: String,
}

/// Host-supplied subtask dictionary.
///
/// Implementations may read a database, a cloud dictionary, or an in-memory table.
/// [`SubtaskCatalog::subtasks`] runs before recognition so the recognizer can choose
/// from every subtask. [`SubtaskCatalog::candidates`] runs after recognition and
/// returns the Pareto set of the recognized subtask only.
#[async_trait]
pub trait SubtaskCatalog: Send + Sync {
    /// Returns every subtask the recognizer may label a step with.
    async fn subtasks(&self) -> Result<Vec<SubtaskSummary>>;
    /// Returns the Pareto candidates of one subtask.
    async fn candidates(&self, subtask_id: &str) -> Result<Vec<SubtaskCandidate>>;
}

/// Configuration for [`RsiRouter`].
#[derive(Clone)]
pub struct RsiRouterConfig {
    /// Model that labels the current step.
    pub recognizer: ModelId,
    /// Model used when no subtask route applies.
    pub flagship: ModelId,
    /// Execution models the dictionary may name.
    pub candidate_models: Vec<ModelId>,
    /// Replaces the packaged recognition prompt.
    pub recognizer_prompt: String,
    /// Lowest confidence that still enters the candidate route.
    pub confidence_threshold: f64,
    /// Candidates whose success rate is below this value are not selectable.
    /// `0.0` keeps every candidate.
    pub min_success_rate: f64,
    /// How many recent steps the recognizer sees.
    pub history_steps: usize,
    /// Candidate ordering policy.
    pub selection: RsiSelection,
    /// Most output tokens the recognizer may use.
    pub max_output_tokens: u64,
    /// Subtask dictionary.
    pub catalog: Arc<dyn SubtaskCatalog>,
    /// When set, every recognized step is served by this model instead of
    /// running candidate selection. The dictionary still supplies the skill
    /// of the candidate matching this model, when one exists.
    pub fixed_target: Option<ModelId>,
    /// Whether the selected candidate's skill reminder is appended to the
    /// request. When false, the request goes out without it.
    pub skill_injected: bool,
}

impl std::fmt::Debug for RsiRouterConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RsiRouterConfig")
            .field("recognizer", &self.recognizer)
            .field("flagship", &self.flagship)
            .field("candidate_models", &self.candidate_models)
            .field("confidence_threshold", &self.confidence_threshold)
            .field("min_success_rate", &self.min_success_rate)
            .field("history_steps", &self.history_steps)
            .field("selection", &self.selection)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("fixed_target", &self.fixed_target)
            .field("skill_injected", &self.skill_injected)
            .finish_non_exhaustive()
    }
}

impl RsiRouterConfig {
    fn validate(&self) -> Result<()> {
        if self.recognizer.as_str().trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router recognizer must not be empty".to_string(),
            });
        }
        if self.flagship.as_str().trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router flagship must not be empty".to_string(),
            });
        }
        if self.candidate_models.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router requires at least one candidate model".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router confidence_threshold must be between 0 and 1".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.min_success_rate) {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router min_success_rate must be between 0 and 1".to_string(),
            });
        }
        if self.history_steps == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router history_steps must be greater than zero".to_string(),
            });
        }
        if self.max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router max_output_tokens must be greater than zero".to_string(),
            });
        }
        if self.recognizer_prompt.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router recognizer_prompt must not be empty".to_string(),
            });
        }
        if self
            .fixed_target
            .as_ref()
            .is_some_and(|fixed| fixed.as_str().trim().is_empty())
        {
            return Err(LibsyError::AlgorithmError {
                message: "rsi_router fixed_target must not be empty".to_string(),
            });
        }
        let mut seen = std::collections::HashSet::new();
        for model in &self.candidate_models {
            if !seen.insert(model.as_str()) {
                return Err(LibsyError::AlgorithmError {
                    message: format!("rsi_router candidate model {model} is repeated"),
                });
            }
        }
        Ok(())
    }
}

/// How a request is tied to its conversation when remembering the previous
/// subtask prediction.
#[derive(Clone, Hash, PartialEq, Eq)]
enum PredictionScope {
    /// A request with a session or sub-agent identity.
    Identity(RoutingIdentity),
    /// A request without one, keyed by the first user message. That text stays
    /// constant across the turns of one conversation.
    Conversation(u64),
}

/// Where a conversation's current subtask skill reminder sits.
struct SkillPosition {
    subtask_id: String,
    /// Message count when the subtask first appeared. Later steps insert the
    /// reminder after this many messages, keeping it at the subtask's start.
    message_index: usize,
}

/// Routes each step by recognized subtask, falling back to a flagship model.
pub struct RsiRouter {
    config: RsiRouterConfig,
    /// Previous subtask prediction for each conversation, shown to the
    /// recognizer on the next step.
    last_predictions: Mutex<HashMap<PredictionScope, String>>,
    /// Current subtask skill reminder position for each conversation.
    skill_positions: Mutex<HashMap<PredictionScope, SkillPosition>>,
}

/// Picks the conversation key for the previous prediction.
fn prediction_scope(request: &Request) -> PredictionScope {
    if let Some(identity) = RoutingIdentity::from_request(request) {
        return PredictionScope::Identity(identity);
    }
    let first_user = request
        .llm_request
        .messages
        .iter()
        .find(|message| message.role == Role::User)
        .and_then(|message| message.text_content("\n"))
        .unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    first_user.hash(&mut hasher);
    PredictionScope::Conversation(hasher.finish())
}

/// What the candidate filter produced for one subtask.
enum CandidateSelection {
    Selected(SubtaskCandidate, Vec<ModelId>),
    NoConfiguredModels,
    NoCandidateAboveThreshold,
}

/// The Pareto set as it entered selection, with the filter outcome per model.
fn pareto_evidence(
    candidates: &[SubtaskCandidate],
    selected: Option<&ModelId>,
    config: &RsiRouterConfig,
) -> serde_json::Value {
    serde_json::Value::Array(
        candidates
            .iter()
            .map(|candidate| {
                let configured = config.candidate_models.contains(&candidate.model_id);
                let above_min_success_rate = candidate.success_rate >= config.min_success_rate;
                serde_json::json!({
                    "model_id": candidate.model_id,
                    "cost": candidate.cost,
                    "latency_ms": candidate.latency_ms,
                    "success_rate": candidate.success_rate,
                    "configured": configured,
                    "above_min_success_rate": above_min_success_rate,
                    "selected": selected == Some(&candidate.model_id),
                })
            })
            .collect(),
    )
}

/// Picks the skill to inject: the highest `skill_version` that carries content.
///
/// Invalid candidates are skipped. Equal versions keep the serving candidate's
/// skill, so a dictionary without versions behaves as before.
fn newest_skill<'a>(candidates: &'a [SubtaskCandidate], serving: &ModelId) -> Option<&'a str> {
    candidates
        .iter()
        .filter(|candidate| {
            candidate.is_valid
                && candidate
                    .skill_content
                    .as_deref()
                    .is_some_and(|skill| !skill.trim().is_empty())
        })
        .max_by(|left, right| {
            left.skill_version
                .total_cmp(&right.skill_version)
                .then_with(|| (left.model_id == *serving).cmp(&(right.model_id == *serving)))
        })
        .and_then(|candidate| candidate.skill_content.as_deref())
        .map(str::trim)
}

impl RsiRouter {
    /// Builds an RSI router.
    ///
    /// Returns an error when the configured thresholds, prompts, or candidate list
    /// cannot produce a usable route.
    pub fn new(config: RsiRouterConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            last_predictions: Mutex::new(HashMap::new()),
            skill_positions: Mutex::new(HashMap::new()),
        })
    }

    fn last_prediction(&self, request: &Request) -> Option<String> {
        self.last_predictions
            .lock()
            .get(&prediction_scope(request))
            .cloned()
    }

    fn remember_prediction(&self, request: &Request, subtask_id: &str) {
        let scope = prediction_scope(request);
        let mut predictions = self.last_predictions.lock();
        if predictions.len() >= MAX_TRACKED_CONVERSATIONS
            && !predictions.contains_key(&scope)
            && let Some(evicted) = predictions.keys().next().cloned()
        {
            predictions.remove(&evicted);
        }
        predictions.insert(scope, subtask_id.to_string());
    }

    fn recognition_prompt(
        &self,
        request: &Request,
        subtasks: &[SubtaskSummary],
        last_prediction: Option<&str>,
    ) -> String {
        let mut prompt = self.config.recognizer_prompt.trim().to_string();
        prompt.push_str("\n\nAvailable subtasks:\n");
        for entry in subtasks {
            prompt.push_str("- id: ");
            prompt.push_str(&entry.subtask_id);
            prompt.push('\n');
            if !entry.subtask_name.trim().is_empty() {
                prompt.push_str("  name: ");
                prompt.push_str(entry.subtask_name.trim());
                prompt.push('\n');
            }
            if !entry.definition.trim().is_empty() {
                prompt.push_str("  definition: ");
                prompt.push_str(&clip(entry.definition.trim(), 500));
                prompt.push('\n');
            }
        }
        prompt.push_str(
            "\nIf the current step does not match any listed subtask, return an empty subtask_id.\n\n",
        );
        prompt.push_str(&recognition_context(
            request,
            self.config.history_steps,
            last_prediction,
        ));
        prompt
    }

    fn recognition_request(&self, request: &Request, prompt: String) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some(self.config.recognizer.to_string()),
                messages: vec![Message::text(Role::User, prompt)],
                output: switchyard_protocol::OutputParams {
                    max_output_tokens: Some(self.config.max_output_tokens),
                    ..Default::default()
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }

    async fn recognize(
        &self,
        driver: &Driver,
        request: &Request,
        subtasks: &[SubtaskSummary],
        last_prediction: Option<&str>,
    ) -> (Option<RecognizedSubtask>, RecognitionTrace) {
        let prompt = self.recognition_prompt(request, subtasks, last_prediction);
        let started = Instant::now();
        let response = driver
            .call_model_with_error_recovery(
                self.recognition_request(request, prompt.clone()),
                vec![self.config.recognizer.clone()],
                true,
            )
            .await;
        let latency = started.elapsed();
        let trace = |response, subtask_id, confidence, error| RecognitionTrace {
            model: self.config.recognizer.clone(),
            prompt: prompt.clone(),
            response,
            subtask_id,
            confidence,
            latency,
            error,
        };
        let response = match response {
            Ok(response) => response,
            Err(error) => return (None, trace(None, None, None, Some(error.to_string()))),
        };
        let aggregate = match response.llm_response.into_agg().await {
            Ok(aggregate) => aggregate,
            Err(error) => return (None, trace(None, None, None, Some(error.to_string()))),
        };
        let text = completion_text(&aggregate);
        match RecognizedSubtask::parse(&text) {
            Some(parsed) => {
                let traced = trace(
                    Some(text),
                    Some(parsed.subtask_id.clone()),
                    Some(parsed.confidence),
                    None,
                );
                (Some(parsed), traced)
            }
            None => (
                None,
                trace(
                    Some(text),
                    None,
                    None,
                    Some("recognizer output had no valid subtask".to_string()),
                ),
            ),
        }
    }

    /// The target a failed step falls back to. A configured fixed target pins
    /// every request, so it replaces the flagship here too.
    fn fallback_target(&self) -> ModelId {
        self.config
            .fixed_target
            .clone()
            .unwrap_or_else(|| self.config.flagship.clone())
    }

    fn route_to_fallback(
        &self,
        driver: &Driver,
        request: Request,
        reason: &str,
        recognition: Option<&RecognizedSubtask>,
        recognition_trace: Option<&RecognitionTrace>,
    ) -> RoutingOutcome {
        self.route_to_fallback_with_pareto(
            driver,
            request,
            reason,
            recognition,
            recognition_trace,
            None,
        )
    }

    fn route_to_fallback_with_pareto(
        &self,
        driver: &Driver,
        mut request: Request,
        reason: &str,
        recognition: Option<&RecognizedSubtask>,
        recognition_trace: Option<&RecognitionTrace>,
        candidates: Option<&[SubtaskCandidate]>,
    ) -> RoutingOutcome {
        let target = self.fallback_target();
        // The fallback target takes the same newest valid skill when the
        // candidates were already read. Earlier fallbacks have no subtask to
        // take one from.
        let skill = if self.config.skill_injected {
            candidates.and_then(|candidates| newest_skill(candidates, &target))
        } else {
            None
        };
        let skill_injected = match (skill, recognition) {
            (Some(skill), Some(recognition)) => {
                self.inject_skill_reminder(&mut request, &recognition.subtask_id, skill)
            }
            _ => false,
        };
        let decision = if self.config.fixed_target.is_some() {
            "fixed_target"
        } else {
            "flagship"
        };
        let mut evidence = serde_json::json!({
            "source": "rsi_router",
            "decision": decision,
            "reason": reason,
            "confidence_threshold": self.config.confidence_threshold,
            "min_success_rate": self.config.min_success_rate,
            "subtask_id": recognition.map(|recognition| recognition.subtask_id.as_str()),
            "confidence": recognition.map(|recognition| recognition.confidence),
            "skill_injected": skill_injected,
            "recognizer": recognition_trace.map(RecognitionTrace::evidence),
        });
        match &self.config.fixed_target {
            Some(fixed) => evidence["fixed_target"] = serde_json::json!(fixed),
            None => evidence["selection"] = serde_json::json!(self.selection_label()),
        }
        if let Some(candidates) = candidates {
            evidence["pareto_candidates"] = pareto_evidence(candidates, None, &self.config);
        }
        driver.set_evidence(evidence);
        RoutingOutcome::route_to(target, Vec::new(), request)
    }

    fn selection_label(&self) -> &'static str {
        match self.config.selection {
            RsiSelection::HighestSuccess => "highest_success",
            RsiSelection::LowestCost => "lowest_cost",
            RsiSelection::LowestLatency => "lowest_latency",
        }
    }

    /// Filters the subtask's candidates to configured models and the minimum
    /// success rate, orders them by the selection metric, and returns the first
    /// one with the rest as fallbacks.
    fn select_candidate(&self, candidates: &[SubtaskCandidate]) -> CandidateSelection {
        let configured: Vec<&SubtaskCandidate> = candidates
            .iter()
            .filter(|candidate| self.config.candidate_models.contains(&candidate.model_id))
            .collect();
        if configured.is_empty() {
            return CandidateSelection::NoConfiguredModels;
        }
        let mut candidates: Vec<&SubtaskCandidate> = configured
            .into_iter()
            .filter(|candidate| candidate.success_rate >= self.config.min_success_rate)
            .collect();
        if candidates.is_empty() {
            return CandidateSelection::NoCandidateAboveThreshold;
        }
        match self.config.selection {
            RsiSelection::HighestSuccess => {
                candidates.sort_by(|left, right| right.success_rate.total_cmp(&left.success_rate))
            }
            RsiSelection::LowestCost => {
                candidates.sort_by(|left, right| left.cost.total_cmp(&right.cost))
            }
            RsiSelection::LowestLatency => {
                candidates.sort_by(|left, right| left.latency_ms.total_cmp(&right.latency_ms))
            }
        }
        let selected = (*candidates[0]).clone();
        let fallbacks = candidates
            .iter()
            .skip(1)
            .map(|candidate| candidate.model_id.clone())
            .collect();
        CandidateSelection::Selected(selected, fallbacks)
    }
}

#[async_trait]
impl Algorithm for RsiRouter {
    fn name(&self) -> &str {
        "rsi_router"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        let subtasks = match self.config.catalog.subtasks().await {
            Ok(subtasks) => subtasks,
            Err(error) => {
                tracing::warn!(
                    %error,
                    "rsi subtask read failed; routing to the flagship model"
                );
                return Ok(self.route_to_fallback(&driver, request, "catalog_failed", None, None));
            }
        };
        if subtasks.is_empty() {
            return Ok(self.route_to_fallback(&driver, request, "empty_catalog", None, None));
        }
        let last_prediction = self.last_prediction(&request);
        let (recognition, recognition_trace) = self
            .recognize(&driver, &request, &subtasks, last_prediction.as_deref())
            .await;
        if let Some(prediction) = &recognition {
            self.remember_prediction(&request, &prediction.subtask_id);
        }
        let Some(recognition) = recognition else {
            return Ok(self.route_to_fallback(
                &driver,
                request,
                "recognition_failed",
                None,
                Some(&recognition_trace),
            ));
        };
        if recognition.subtask_id.is_empty() {
            return Ok(self.route_to_fallback(
                &driver,
                request,
                "no_subtask",
                Some(&recognition),
                Some(&recognition_trace),
            ));
        }
        if recognition.confidence < self.config.confidence_threshold {
            // The subtask is still known, so the flagship can carry its skill.
            let candidates = if self.config.skill_injected
                && subtasks
                    .iter()
                    .any(|subtask| subtask.subtask_id == recognition.subtask_id)
            {
                self.config
                    .catalog
                    .candidates(&recognition.subtask_id)
                    .await
                    .ok()
            } else {
                None
            };
            return Ok(self.route_to_fallback_with_pareto(
                &driver,
                request,
                "low_confidence",
                Some(&recognition),
                Some(&recognition_trace),
                candidates.as_deref(),
            ));
        }
        if !subtasks
            .iter()
            .any(|subtask| subtask.subtask_id == recognition.subtask_id)
        {
            return Ok(self.route_to_fallback(
                &driver,
                request,
                "unknown_subtask",
                Some(&recognition),
                Some(&recognition_trace),
            ));
        }

        // Recognition succeeded. Only now does the router read this subtask's
        // candidates, order them, and select one.
        let candidates = match self
            .config
            .catalog
            .candidates(&recognition.subtask_id)
            .await
        {
            Ok(candidates) => candidates,
            Err(error) => {
                tracing::warn!(
                    subtask_id = %recognition.subtask_id,
                    %error,
                    "rsi candidate read failed; routing to the flagship model"
                );
                return Ok(self.route_to_fallback(
                    &driver,
                    request,
                    "catalog_failed",
                    Some(&recognition),
                    Some(&recognition_trace),
                ));
            }
        };
        // A configured fixed target replaces candidate selection: every
        // recognized step is served by that model. The skill still comes from
        // the candidates, as the highest version with content.
        if let Some(fixed) = self.config.fixed_target.clone() {
            let skill = if self.config.skill_injected {
                newest_skill(&candidates, &fixed)
            } else {
                None
            };
            let skill_injected = if let Some(skill) = skill {
                self.inject_skill_reminder(&mut request, &recognition.subtask_id, skill)
            } else {
                false
            };
            driver.set_evidence(serde_json::json!({
                "source": "rsi_router",
                "decision": "subtask",
                "subtask_id": recognition.subtask_id,
                "confidence": recognition.confidence,
                "confidence_threshold": self.config.confidence_threshold,
                "min_success_rate": self.config.min_success_rate,
                "fixed_target": fixed,
                "selected_model": fixed,
                "skill_injected": skill_injected,
                "candidates": [fixed],
                "pareto_candidates": pareto_evidence(
                    &candidates,
                    Some(&fixed),
                    &self.config,
                ),
                "recognizer": recognition_trace.evidence(),
            }));
            return Ok(RoutingOutcome::route_to(fixed, Vec::new(), request));
        }
        let (selected, fallbacks) = match self.select_candidate(&candidates) {
            CandidateSelection::Selected(selected, fallbacks) => (selected, fallbacks),
            CandidateSelection::NoConfiguredModels => {
                return Ok(self.route_to_fallback_with_pareto(
                    &driver,
                    request,
                    "no_configured_candidates",
                    Some(&recognition),
                    Some(&recognition_trace),
                    Some(&candidates),
                ));
            }
            CandidateSelection::NoCandidateAboveThreshold => {
                return Ok(self.route_to_fallback_with_pareto(
                    &driver,
                    request,
                    "no_candidate_above_min_success_rate",
                    Some(&recognition),
                    Some(&recognition_trace),
                    Some(&candidates),
                ));
            }
        };

        let skill = if self.config.skill_injected {
            newest_skill(&candidates, &selected.model_id)
        } else {
            None
        };
        let skill_injected = if let Some(skill) = skill {
            self.inject_skill_reminder(&mut request, &recognition.subtask_id, skill)
        } else {
            false
        };

        let mut candidate_models = vec![selected.model_id.clone()];
        candidate_models.extend(fallbacks.iter().cloned());
        driver.set_evidence(serde_json::json!({
            "source": "rsi_router",
            "decision": "subtask",
            "subtask_id": recognition.subtask_id,
            "confidence": recognition.confidence,
            "confidence_threshold": self.config.confidence_threshold,
            "min_success_rate": self.config.min_success_rate,
            "selected_model": selected.model_id,
            "cost": selected.cost,
            "latency_ms": selected.latency_ms,
            "success_rate": selected.success_rate,
            "skill_injected": skill_injected,
            "selection": self.selection_label(),
            "candidates": candidate_models,
            "pareto_candidates": pareto_evidence(
                &candidates,
                Some(&selected.model_id),
                &self.config,
            ),
            "recognizer": recognition_trace.evidence(),
        }));
        Ok(RoutingOutcome::route_to(
            selected.model_id,
            fallbacks,
            request,
        ))
    }
}

impl RsiRouter {
    /// Places the subtask's skill reminder at the start of the subtask.
    ///
    /// The subtask's first step appends the reminder. Later steps insert it at
    /// that same position, so the model reads it as context for the subtask
    /// instead of a new instruction for the current step.
    ///
    /// Anthropic-compatible requests carry it as a system message. OpenAI-
    /// compatible requests use a user message.
    ///
    /// Returns whether a reminder was inserted. A conversation that already
    /// carries the same reminder is left alone.
    fn inject_skill_reminder(&self, request: &mut Request, subtask_id: &str, skill: &str) -> bool {
        let reminder = format!(
            "<skill_reminder>当前为子任务id：{subtask_id}，Skill：{skill}</skill_reminder>"
        );
        if request.llm_request.messages.iter().any(|message| {
            message
                .text_content("")
                .is_some_and(|text| text.contains(&reminder))
        }) {
            return false;
        }
        let scope = prediction_scope(request);
        let index = {
            let mut positions = self.skill_positions.lock();
            if positions.len() >= MAX_TRACKED_CONVERSATIONS
                && !positions.contains_key(&scope)
                && let Some(evicted) = positions.keys().next().cloned()
            {
                positions.remove(&evicted);
            }
            // The reminder goes where the subtask's first step put it. A
            // shorter request (history trimmed by the caller) falls back to
            // the end.
            let index = positions
                .get(&scope)
                .filter(|position| position.subtask_id == subtask_id)
                .map(|position| {
                    position
                        .message_index
                        .min(request.llm_request.messages.len())
                })
                .unwrap_or(request.llm_request.messages.len());
            positions.insert(
                scope,
                SkillPosition {
                    subtask_id: subtask_id.to_string(),
                    message_index: index,
                },
            );
            index
        };
        let role = match request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.caller_wire_format.or(metadata.wire_format))
        {
            Some(WireFormat::AnthropicMessages) => Role::System,
            _ => Role::User,
        };
        insert_message(request, index, role, &reminder);
        true
    }
}

/// The recognizer interaction attached to routing evidence for host-side logging.
struct RecognitionTrace {
    model: ModelId,
    prompt: String,
    response: Option<String>,
    subtask_id: Option<String>,
    confidence: Option<f64>,
    latency: Duration,
    error: Option<String>,
}

impl RecognitionTrace {
    fn evidence(&self) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "prompt": self.prompt,
            "response": self.response,
            "subtask_id": self.subtask_id,
            "confidence": self.confidence,
            "latency_ms": self.latency.as_secs_f64() * 1_000.0,
            "error": self.error,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
struct RecognizedSubtask {
    subtask_id: String,
    confidence: f64,
}

impl RecognizedSubtask {
    fn parse(text: &str) -> Option<Self> {
        let start = text.find('{')?;
        let end = text.rfind('}')?;
        if end <= start {
            return None;
        }
        let parsed: Self = serde_json::from_str(&text[start..=end]).ok()?;
        if !parsed.confidence.is_finite() || !(0.0..=1.0).contains(&parsed.confidence) {
            return None;
        }
        Some(Self {
            subtask_id: parsed.subtask_id.trim().to_string(),
            confidence: parsed.confidence,
        })
    }
}

/// Builds the compact context the recognizer reads: the user request, the
/// recent steps, and the previous subtask prediction.
fn recognition_context(
    request: &Request,
    history_steps: usize,
    last_prediction: Option<&str>,
) -> String {
    const MAX_BLOCK_CHARS: usize = 2_000;
    const MAX_TOTAL_CHARS: usize = 8_000;

    let mut sections = Vec::new();
    if let Some(requirement) = request
        .llm_request
        .messages
        .iter()
        .find(|message| message.role == Role::User)
        .and_then(|message| message.text_content("\n"))
    {
        sections.push(format!(
            "USER REQUIREMENT:\n{}",
            clip(&requirement, MAX_BLOCK_CHARS)
        ));
    }

    let recent = recent_steps(&request.llm_request.messages, history_steps);
    if !recent.is_empty() {
        sections.push(format!("RECENT STEPS:\n{recent}"));
    }

    let prediction = last_prediction
        .filter(|prediction| !prediction.is_empty())
        .unwrap_or("(none)");
    sections.push(format!("PREVIOUS SUBTASK PREDICTION:\n{prediction}"));

    clip(&sections.join("\n\n"), MAX_TOTAL_CHARS)
}

fn recent_steps(messages: &[Message], history_steps: usize) -> String {
    let assistant_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.role == Role::Assistant)
        .map(|(index, _)| index)
        .collect();
    let offset = assistant_indices.len().saturating_sub(history_steps);
    let Some(&start) = assistant_indices.get(offset) else {
        return String::new();
    };
    let mut rendered = Vec::new();
    for message in &messages[start..] {
        if matches!(message.role, Role::System | Role::Developer) {
            continue;
        }
        let content = render_content(&message.content);
        if content.is_empty() {
            continue;
        }
        rendered.push(format!(
            "[{}] {}",
            role_label(message.role),
            clip(&content, 1_000)
        ));
    }
    rendered.join("\n")
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        Role::User => "user",
        Role::System => "system",
        Role::Developer => "developer",
    }
}

fn render_content(blocks: &[ContentBlock]) -> String {
    let parts: Vec<String> = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.clone()),
            ContentBlock::Refusal { text } => Some(text.clone()),
            ContentBlock::ToolCall(call) => Some(format!("{} {}", call.name, call.arguments)),
            ContentBlock::ToolResult(result) => Some(format!(
                "{} {}",
                result.tool_call_id,
                render_content(&result.content)
            )),
            _ => None,
        })
        .collect();
    parts.join("\n")
}

fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(limit).collect();
    clipped.push_str("...");
    clipped
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use switchyard_protocol::{
        Category, ContentBlock, LlmRequest, Message, Metadata, Request, Role, Tool, ToolCall,
        ToolDefinition,
    };

    use super::*;
    use crate::RuntimeModels;
    use crate::State;
    use crate::core::testing::{reply, test_drive, test_drive_with_state};

    const RECOGNIZER: &str = "model/recognizer";
    const FLAGSHIP: &str = "model/flagship";
    const CHEAP: &str = "model/cheap";
    const MID: &str = "model/mid";
    const PRICY: &str = "model/pricy";

    struct StaticCatalog {
        subtasks: Mutex<Vec<SubtaskSummary>>,
        candidates: HashMap<String, Vec<SubtaskCandidate>>,
        fail_subtasks: bool,
        fail_candidates: bool,
    }

    impl StaticCatalog {
        fn new(
            subtasks: Vec<SubtaskSummary>,
            candidates: HashMap<String, Vec<SubtaskCandidate>>,
        ) -> Self {
            Self {
                subtasks: Mutex::new(subtasks),
                candidates,
                fail_subtasks: false,
                fail_candidates: false,
            }
        }

        fn failing_subtasks() -> Self {
            Self {
                subtasks: Mutex::new(Vec::new()),
                candidates: HashMap::new(),
                fail_subtasks: true,
                fail_candidates: false,
            }
        }

        fn failing_candidates() -> Self {
            Self {
                subtasks: Mutex::new(vec![summary()]),
                candidates: HashMap::new(),
                fail_subtasks: false,
                fail_candidates: true,
            }
        }
    }

    #[async_trait]
    impl SubtaskCatalog for StaticCatalog {
        async fn subtasks(&self) -> Result<Vec<SubtaskSummary>> {
            if self.fail_subtasks {
                return Err(LibsyError::AlgorithmError {
                    message: "subtask read unavailable".to_string(),
                });
            }
            Ok(self.subtasks.lock().clone())
        }

        async fn candidates(&self, subtask_id: &str) -> Result<Vec<SubtaskCandidate>> {
            if self.fail_candidates {
                return Err(LibsyError::AlgorithmError {
                    message: "candidate read unavailable".to_string(),
                });
            }
            Ok(self.candidates.get(subtask_id).cloned().unwrap_or_default())
        }
    }

    fn config(catalog: Arc<dyn SubtaskCatalog>) -> RsiRouterConfig {
        RsiRouterConfig {
            recognizer: ModelId::from(RECOGNIZER),
            flagship: ModelId::from(FLAGSHIP),
            candidate_models: vec![
                ModelId::from(CHEAP),
                ModelId::from(MID),
                ModelId::from(PRICY),
            ],
            recognizer_prompt: DEFAULT_RSI_RECOGNIZER_PROMPT.trim().to_string(),
            confidence_threshold: 0.7,
            min_success_rate: 0.0,
            history_steps: 3,
            selection: RsiSelection::LowestCost,
            max_output_tokens: 128,
            catalog,
            fixed_target: None,
            skill_injected: true,
        }
    }

    fn summary() -> SubtaskSummary {
        SubtaskSummary {
            subtask_id: "edit-code".to_string(),
            subtask_name: "Edit code".to_string(),
            definition: "Change existing source files in a small, focused patch.".to_string(),
        }
    }

    fn candidate(
        model: &str,
        cost: f64,
        latency_ms: f64,
        success_rate: f64,
        skill: Option<&str>,
        skill_version: f64,
    ) -> SubtaskCandidate {
        SubtaskCandidate {
            model_id: ModelId::from(model),
            cost,
            latency_ms,
            success_rate,
            skill_content: skill.map(str::to_string),
            skill_version,
            is_valid: true,
        }
    }

    /// PRICY has the best success rate, MID the lowest latency, CHEAP the lowest
    /// cost. All skills share version 0, so the serving candidate wins.
    fn catalog() -> StaticCatalog {
        StaticCatalog::new(
            vec![summary()],
            HashMap::from([(
                "edit-code".to_string(),
                vec![
                    candidate(PRICY, 9.0, 800.0, 0.99, Some("Prefer a big patch."), 0.0),
                    candidate(
                        CHEAP,
                        0.5,
                        1_200.0,
                        0.80,
                        Some("Prefer a minimal patch."),
                        0.0,
                    ),
                    candidate(MID, 3.0, 500.0, 0.90, None, 0.0),
                ],
            )]),
        )
    }

    fn catalog_without_skills() -> StaticCatalog {
        StaticCatalog::new(
            vec![summary()],
            HashMap::from([(
                "edit-code".to_string(),
                vec![
                    candidate(PRICY, 9.0, 800.0, 0.99, None, 0.0),
                    candidate(CHEAP, 0.5, 1_200.0, 0.80, None, 0.0),
                    candidate(MID, 3.0, 500.0, 0.90, None, 0.0),
                ],
            )]),
        )
    }

    fn request() -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("switchyard/rsi".to_string()),
                messages: vec![
                    Message::text(Role::User, "Fix the parser"),
                    Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolCall(ToolCall {
                            id: "call-1".to_string(),
                            name: "read_file".to_string(),
                            arguments: serde_json::json!({"path": "parser.rs"}),
                        })],
                    },
                    Message {
                        role: Role::Tool,
                        content: vec![ContentBlock::ToolResult(switchyard_protocol::ToolResult {
                            tool_call_id: "call-1".to_string(),
                            content: vec![ContentBlock::Text {
                                text: "fn parse() {}".to_string(),
                            }],
                            is_error: Some(false),
                        })],
                    },
                ],
                tools: vec![Tool::Function(ToolDefinition {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: serde_json::json!({}),
                    strict: None,
                })],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                ..Metadata::default()
            }),
        }
    }

    fn models() -> RuntimeModels {
        RuntimeModels::new(HashMap::from([(
            Category::Judge,
            vec![ModelId::from(RECOGNIZER)],
        )]))
    }

    fn recognizer_reply(subtask_id: &str, confidence: f64) -> String {
        serde_json::json!({"subtask_id": subtask_id, "confidence": confidence}).to_string()
    }

    fn reminder(request: &Request) -> Option<(Role, String)> {
        request
            .llm_request
            .messages
            .iter()
            .rev()
            .find_map(|message| {
                let text = message.text_content("")?;
                text.contains("<skill_reminder>")
                    .then_some((message.role, text))
            })
    }

    async fn route_with(
        config: RsiRouterConfig,
        recognition: String,
    ) -> (Result<(ModelId, switchyard_protocol::Response)>, String) {
        let captured = Arc::new(Mutex::new(String::new()));
        let capture = captured.clone();
        let router = Arc::new(RsiRouter::new(config).expect("config should be valid"));
        let result = test_drive(
            router,
            request(),
            move |target: ModelId, request: Request| {
                if target == RECOGNIZER {
                    *capture.lock() = request
                        .llm_request
                        .messages
                        .first()
                        .unwrap()
                        .text_content("\n")
                        .unwrap();
                    return std::future::ready(Ok(reply(recognition.clone())));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await;
        let prompt = captured.lock().clone();
        (result, prompt)
    }

    #[tokio::test]
    async fn routes_to_the_lowest_cost_candidate_and_appends_its_skill_reminder() -> Result<()> {
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog()))).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(CHEAP));
        assert!(
            outcome.request.llm_request.instructions.is_empty(),
            "the skill is no longer a system instruction"
        );
        assert_eq!(
            reminder(&outcome.request),
            Some((
                Role::User,
                "<skill_reminder>当前为子任务id：edit-code，Skill：Prefer a minimal patch.</skill_reminder>"
                    .to_string()
            ))
        );
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["subtask_id"], "edit-code");
        assert_eq!(evidence["selected_model"], CHEAP);
        assert_eq!(evidence["cost"], 0.5);
        assert_eq!(evidence["success_rate"], 0.8);
        assert_eq!(evidence["skill_injected"], true);
        assert_eq!(evidence["selection"], "lowest_cost");
        Ok(())
    }

    #[tokio::test]
    async fn injects_the_highest_skill_version_regardless_of_the_serving_model() -> Result<()> {
        let catalog = StaticCatalog::new(
            vec![summary()],
            HashMap::from([(
                "edit-code".to_string(),
                vec![
                    candidate(CHEAP, 0.5, 1_200.0, 0.80, Some("old skill"), 1.0),
                    candidate(PRICY, 9.0, 800.0, 0.99, Some("middle skill"), 2.0),
                    candidate(MID, 3.0, 500.0, 0.90, Some("newest skill"), 3.0),
                ],
            )]),
        );
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog))).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(CHEAP));
        assert_eq!(
            reminder(&outcome.request),
            Some((
                Role::User,
                "<skill_reminder>当前为子任务id：edit-code，Skill：newest skill</skill_reminder>"
                    .to_string()
            ))
        );
        Ok(())
    }

    #[tokio::test]
    async fn skips_invalid_candidates_when_picking_the_newest_skill() -> Result<()> {
        let mut stale = candidate(CHEAP, 0.5, 1_200.0, 0.80, Some("stale skill"), 5.0);
        stale.is_valid = false;
        let catalog = StaticCatalog::new(
            vec![summary()],
            HashMap::from([(
                "edit-code".to_string(),
                vec![
                    stale,
                    candidate(PRICY, 9.0, 800.0, 0.99, Some("middle skill"), 2.0),
                    candidate(MID, 3.0, 500.0, 0.90, Some("current skill"), 3.0),
                ],
            )]),
        );
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog))).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        // Routing still picks the cheapest candidate; only the skill is skipped.
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(CHEAP));
        assert_eq!(
            reminder(&outcome.request),
            Some((
                Role::User,
                "<skill_reminder>当前为子任务id：edit-code，Skill：current skill</skill_reminder>"
                    .to_string()
            ))
        );
        Ok(())
    }

    #[tokio::test]
    async fn skill_injected_false_sends_the_request_without_the_skill_reminder() -> Result<()> {
        let mut rsi_config = config(Arc::new(catalog()));
        rsi_config.skill_injected = false;
        let router = Arc::new(RsiRouter::new(rsi_config).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(CHEAP));
        assert_eq!(reminder(&outcome.request), None);
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["skill_injected"], false);
        Ok(())
    }

    #[tokio::test]
    async fn anthropic_requests_append_the_skill_reminder_as_a_system_message() -> Result<()> {
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog()))).expect("config should be valid"));
        let mut request = request();
        request
            .metadata
            .as_mut()
            .expect("request metadata")
            .caller_wire_format = Some(WireFormat::AnthropicMessages);
        let outcome = test_drive_with_state(
            router,
            request,
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(CHEAP));
        assert_eq!(
            reminder(&outcome.request).map(|(role, _)| role),
            Some(Role::System)
        );
        Ok(())
    }

    #[tokio::test]
    async fn appends_the_skill_reminder_on_every_step() -> Result<()> {
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog()))).expect("config should be valid"));
        for _ in 0..2 {
            let outcome = test_drive_with_state(
                router.clone(),
                request(),
                models(),
                State::default(),
                |target: ModelId, _: Request| {
                    if target == RECOGNIZER {
                        return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                    }
                    std::future::ready(Ok(reply(target.as_str())))
                },
            )
            .await?;
            assert_eq!(outcome.selected_model_id()?, &ModelId::from(CHEAP));
            assert_eq!(
                reminder(&outcome.request).map(|(role, _)| role),
                Some(Role::User)
            );
            let evidence = outcome
                .metadata
                .expect("outcome metadata")
                .evidence
                .expect("evidence");
            assert_eq!(evidence["skill_injected"], true);
        }
        Ok(())
    }

    #[tokio::test]
    async fn keeps_the_skill_reminder_at_the_subtask_start_on_later_steps() -> Result<()> {
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog()))).expect("config should be valid"));
        let respond = |target: ModelId, _: Request| {
            if target == RECOGNIZER {
                return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
            }
            std::future::ready(Ok(reply(target.as_str())))
        };
        let first = test_drive_with_state(
            router.clone(),
            request(),
            models(),
            State::default(),
            respond,
        )
        .await?;
        let start = first.request.llm_request.messages.len() - 1;
        // The caller resends its own transcript, which never carried the
        // injected reminder. It grows by the first step's assistant and user
        // turns.
        let mut next = request();
        next.llm_request
            .messages
            .push(Message::text(Role::Assistant, "Read the parser."));
        next.llm_request
            .messages
            .push(Message::text(Role::User, "Fix it now."));
        let second =
            test_drive_with_state(router, next, models(), State::default(), respond).await?;
        let messages = &second.request.llm_request.messages;
        let positions: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, message)| {
                message
                    .text_content("")
                    .is_some_and(|text| text.contains("<skill_reminder>"))
            })
            .map(|(index, _)| index)
            .collect();
        assert_eq!(positions, vec![start], "one reminder, at the subtask start");
        assert!(start < messages.len() - 1, "not appended at the end");
        let evidence = second
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["skill_injected"], true);
        Ok(())
    }

    #[tokio::test]
    async fn does_not_duplicate_an_existing_skill_reminder() -> Result<()> {
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog()))).expect("config should be valid"));
        let first = test_drive_with_state(
            router.clone(),
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        let existing = first.request.clone();
        let message_count = existing.llm_request.messages.len();
        let second = test_drive_with_state(
            router,
            existing,
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(second.selected_model_id()?, &ModelId::from(CHEAP));
        assert_eq!(second.request.llm_request.messages.len(), message_count);
        assert!(reminder(&second.request).is_some());
        let evidence = second
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["skill_injected"], false);
        Ok(())
    }

    #[tokio::test]
    async fn routes_by_highest_success() -> Result<()> {
        let mut rsi_config = config(Arc::new(catalog()));
        rsi_config.selection = RsiSelection::HighestSuccess;
        let (result, _) = route_with(rsi_config, recognizer_reply("edit-code", 0.95)).await;
        assert_eq!(result?.0, PRICY);
        Ok(())
    }

    #[tokio::test]
    async fn routes_by_lowest_latency_and_injects_no_skill_without_one() -> Result<()> {
        let mut rsi_config = config(Arc::new(catalog_without_skills()));
        rsi_config.selection = RsiSelection::LowestLatency;
        let router = Arc::new(RsiRouter::new(rsi_config).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(MID));
        assert!(
            outcome.request.llm_request.instructions.is_empty(),
            "a candidate without a skill injects nothing"
        );
        assert_eq!(
            reminder(&outcome.request),
            None,
            "a candidate without a skill appends no reminder"
        );
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["skill_injected"], false);
        assert_eq!(evidence["latency_ms"], 500.0);
        assert_eq!(evidence["selection"], "lowest_latency");
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_to_the_flagship_below_the_threshold() -> Result<()> {
        let (result, _) = route_with(
            config(Arc::new(catalog())),
            recognizer_reply("edit-code", 0.4),
        )
        .await;
        assert_eq!(result?.0, FLAGSHIP);
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_for_an_empty_or_unparsable_subtask() -> Result<()> {
        for recognition in ["not json", recognizer_reply("", 0.99).as_str()] {
            let (result, _) =
                route_with(config(Arc::new(catalog())), recognition.to_string()).await;
            assert_eq!(result?.0, FLAGSHIP);
        }
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_when_the_subtask_read_fails_or_is_empty() -> Result<()> {
        let (empty, _) = route_with(
            config(Arc::new(StaticCatalog::new(Vec::new(), HashMap::new()))),
            recognizer_reply("edit-code", 0.99),
        )
        .await;
        assert_eq!(empty?.0, FLAGSHIP);

        let (failed, _) = route_with(
            config(Arc::new(StaticCatalog::failing_subtasks())),
            recognizer_reply("edit-code", 0.99),
        )
        .await;
        assert_eq!(failed?.0, FLAGSHIP);
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_when_the_candidate_read_fails_after_recognition() -> Result<()> {
        let router = Arc::new(
            RsiRouter::new(config(Arc::new(StaticCatalog::failing_candidates())))
                .expect("config should be valid"),
        );
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(FLAGSHIP));
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["reason"], "catalog_failed");
        assert_eq!(evidence["subtask_id"], "edit-code");
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_when_the_recognized_id_is_not_in_the_catalog() -> Result<()> {
        let (result, _) = route_with(
            config(Arc::new(catalog())),
            recognizer_reply("run-tests", 0.99),
        )
        .await;
        assert_eq!(result?.0, FLAGSHIP);
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_when_recognition_fails() -> Result<()> {
        let router = Arc::new(RsiRouter::new(config(Arc::new(catalog()))).unwrap());
        let (selected, _) = test_drive(router, request(), |target: ModelId, _: Request| {
            if target == RECOGNIZER {
                return std::future::ready(Err(switchyard_protocol::LlmClientError::General(
                    "recognizer down".to_string(),
                )));
            }
            std::future::ready(Ok(reply(target.as_str())))
        })
        .await?;
        assert_eq!(selected, FLAGSHIP);
        Ok(())
    }

    #[tokio::test]
    async fn ignores_candidates_that_are_not_configured() -> Result<()> {
        let candidates = HashMap::from([(
            "edit-code".to_string(),
            vec![
                candidate("model/unconfigured", 0.01, 1.0, 0.99, None, 0.0),
                candidate(CHEAP, 0.5, 1_200.0, 0.8, None, 0.0),
                candidate(PRICY, 9.0, 800.0, 0.99, None, 0.0),
            ],
        )]);
        let (result, _) = route_with(
            config(Arc::new(StaticCatalog::new(vec![summary()], candidates))),
            recognizer_reply("edit-code", 0.99),
        )
        .await;
        assert_eq!(result?.0, CHEAP);
        Ok(())
    }

    #[tokio::test]
    async fn removes_candidates_below_the_min_success_rate() -> Result<()> {
        let mut rsi_config = config(Arc::new(catalog()));
        rsi_config.min_success_rate = 0.85;
        let router = Arc::new(RsiRouter::new(rsi_config).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(MID));
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["candidates"], serde_json::json!([MID, PRICY]));
        let pareto = evidence["pareto_candidates"]
            .as_array()
            .expect("pareto candidates");
        assert_eq!(pareto.len(), 3);
        let cheap = pareto
            .iter()
            .find(|candidate| candidate["model_id"] == CHEAP)
            .expect("cheap candidate");
        assert_eq!(cheap["above_min_success_rate"], false);
        assert_eq!(cheap["selected"], false);
        let mid = pareto
            .iter()
            .find(|candidate| candidate["model_id"] == MID)
            .expect("mid candidate");
        assert_eq!(mid["selected"], true);
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_when_every_candidate_is_below_the_min_success_rate() -> Result<()> {
        let mut rsi_config = config(Arc::new(catalog()));
        rsi_config.min_success_rate = 1.0;
        let router = Arc::new(RsiRouter::new(rsi_config).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(FLAGSHIP));
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["reason"], "no_candidate_above_min_success_rate");
        assert_eq!(evidence["min_success_rate"], 1.0);
        let pareto = evidence["pareto_candidates"]
            .as_array()
            .expect("pareto candidates");
        assert_eq!(pareto.len(), 3);
        assert!(
            pareto
                .iter()
                .all(|candidate| candidate["above_min_success_rate"] == false)
        );
        Ok(())
    }

    #[tokio::test]
    async fn flagship_fallback_injects_the_newest_valid_skill() -> Result<()> {
        let mut stale = candidate(CHEAP, 0.5, 1_200.0, 0.80, Some("stale skill"), 3.0);
        stale.is_valid = false;
        let catalog = StaticCatalog::new(
            vec![summary()],
            HashMap::from([(
                "edit-code".to_string(),
                vec![
                    stale,
                    candidate(MID, 3.0, 500.0, 0.90, Some("current skill"), 2.0),
                    candidate(PRICY, 9.0, 800.0, 0.99, None, 0.0),
                ],
            )]),
        );
        let mut rsi_config = config(Arc::new(catalog));
        rsi_config.min_success_rate = 1.0;
        let router = Arc::new(RsiRouter::new(rsi_config).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(FLAGSHIP));
        assert_eq!(
            reminder(&outcome.request),
            Some((
                Role::User,
                "<skill_reminder>当前为子任务id：edit-code，Skill：current skill</skill_reminder>"
                    .to_string()
            ))
        );
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["reason"], "no_candidate_above_min_success_rate");
        assert_eq!(evidence["skill_injected"], true);
        Ok(())
    }

    #[tokio::test]
    async fn low_confidence_fallback_injects_the_newest_valid_skill() -> Result<()> {
        let catalog = StaticCatalog::new(
            vec![summary()],
            HashMap::from([(
                "edit-code".to_string(),
                vec![
                    candidate(CHEAP, 0.5, 1_200.0, 0.80, Some("old skill"), 1.0),
                    candidate(MID, 3.0, 500.0, 0.90, Some("new skill"), 2.0),
                ],
            )]),
        );
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog))).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.5))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(FLAGSHIP));
        assert_eq!(
            reminder(&outcome.request),
            Some((
                Role::User,
                "<skill_reminder>当前为子任务id：edit-code，Skill：new skill</skill_reminder>"
                    .to_string()
            ))
        );
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["reason"], "low_confidence");
        assert_eq!(evidence["skill_injected"], true);
        Ok(())
    }

    #[tokio::test]
    async fn fixed_target_serves_every_recognized_step_with_its_skill() -> Result<()> {
        let mut rsi_config = config(Arc::new(catalog()));
        rsi_config.fixed_target = Some(ModelId::from(CHEAP));
        let router = Arc::new(RsiRouter::new(rsi_config).expect("config should be valid"));
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            |target: ModelId, _: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(CHEAP));
        assert_eq!(
            reminder(&outcome.request),
            Some((
                Role::User,
                "<skill_reminder>当前为子任务id：edit-code，Skill：Prefer a minimal patch.</skill_reminder>"
                    .to_string()
            ))
        );
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["fixed_target"], CHEAP);
        assert_eq!(evidence["selected_model"], CHEAP);
        assert_eq!(evidence["candidates"], serde_json::json!([CHEAP]));
        assert_eq!(evidence["skill_injected"], true);
        Ok(())
    }

    #[tokio::test]
    async fn fixed_target_without_a_skill_or_a_candidate_injects_nothing() -> Result<()> {
        for fixed in [MID, "model/unlisted"] {
            let mut rsi_config = config(Arc::new(catalog_without_skills()));
            rsi_config.fixed_target = Some(ModelId::from(fixed));
            let router = Arc::new(RsiRouter::new(rsi_config).expect("config should be valid"));
            let outcome = test_drive_with_state(
                router,
                request(),
                models(),
                State::default(),
                |target: ModelId, _: Request| {
                    if target == RECOGNIZER {
                        return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                    }
                    std::future::ready(Ok(reply(target.as_str())))
                },
            )
            .await?;
            assert_eq!(outcome.selected_model_id()?, &ModelId::from(fixed));
            assert_eq!(
                reminder(&outcome.request),
                None,
                "no candidate carries a skill, so nothing is injected"
            );
            let evidence = outcome
                .metadata
                .expect("outcome metadata")
                .evidence
                .expect("evidence");
            assert_eq!(evidence["fixed_target"], fixed);
            assert_eq!(evidence["skill_injected"], false);
        }
        Ok(())
    }

    #[tokio::test]
    async fn fixed_target_serves_every_request_without_a_subtask_route() -> Result<()> {
        let mut rsi_config = config(Arc::new(catalog()));
        rsi_config.fixed_target = Some(ModelId::from(CHEAP));
        let (low_confidence, _) =
            route_with(rsi_config.clone(), recognizer_reply("edit-code", 0.4)).await;
        assert_eq!(low_confidence?.0, CHEAP);
        let (unrecognized, _) = route_with(rsi_config.clone(), "not json".to_string()).await;
        assert_eq!(unrecognized?.0, CHEAP);
        let (unknown, _) =
            route_with(rsi_config.clone(), recognizer_reply("run-tests", 0.99)).await;
        assert_eq!(unknown?.0, CHEAP);

        let mut unavailable = config(Arc::new(StaticCatalog::failing_subtasks()));
        unavailable.fixed_target = Some(ModelId::from(CHEAP));
        let (catalog_failed, _) = route_with(unavailable, "not json".to_string()).await;
        assert_eq!(catalog_failed?.0, CHEAP);
        Ok(())
    }

    #[tokio::test]
    async fn recognizer_sees_the_request_recent_steps_and_previous_prediction() -> Result<()> {
        let mut config = config(Arc::new(catalog()));
        config.history_steps = 1;
        let (_, prompt) = route_with(config, recognizer_reply("edit-code", 0.99)).await;
        let requirement = prompt.find("USER REQUIREMENT:").expect("user requirement");
        let recent = prompt.find("RECENT STEPS:").expect("recent steps");
        let previous = prompt
            .find("PREVIOUS SUBTASK PREDICTION:")
            .expect("previous prediction");
        assert!(requirement < recent && recent < previous);
        assert!(prompt.contains("PREVIOUS SUBTASK PREDICTION:\n(none)"));
        assert!(!prompt.contains("CURRENT TOOLS"));
        assert!(!prompt.contains("LATEST TOOL STATE"));
        assert!(prompt.contains("fn parse() {}"));
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("Available subtasks"));
        assert!(prompt.contains("edit-code"));
        assert!(prompt.contains("Edit code"));
        assert!(prompt.contains("Change existing source files"));
        Ok(())
    }

    #[tokio::test]
    async fn recognizer_sees_an_unknown_previous_prediction() -> Result<()> {
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(catalog()))).expect("config should be valid"));
        let prompts = Arc::new(Mutex::new(Vec::new()));
        for (recognition, expected_model) in [("run-tests", FLAGSHIP), ("edit-code", CHEAP)] {
            let prompts = prompts.clone();
            let recognition = recognizer_reply(recognition, 0.99);
            let outcome = test_drive_with_state(
                router.clone(),
                request(),
                models(),
                State::default(),
                move |target: ModelId, request: Request| {
                    if target == RECOGNIZER {
                        prompts.lock().push(
                            request.llm_request.messages[0]
                                .text_content("\n")
                                .expect("recognizer prompt"),
                        );
                        return std::future::ready(Ok(reply(recognition.clone())));
                    }
                    std::future::ready(Ok(reply(target.as_str())))
                },
            )
            .await?;
            assert_eq!(outcome.selected_model_id()?, &ModelId::from(expected_model));
        }
        let prompts = prompts.lock();
        assert!(prompts[0].contains("PREVIOUS SUBTASK PREDICTION:\n(none)"));
        assert!(prompts[1].contains("PREVIOUS SUBTASK PREDICTION:\nrun-tests"));
        Ok(())
    }

    #[test]
    fn rejects_invalid_configuration() {
        let catalog: Arc<dyn SubtaskCatalog> =
            Arc::new(StaticCatalog::new(Vec::new(), HashMap::new()));
        for mutate in [
            |config: &mut RsiRouterConfig| config.confidence_threshold = 1.5,
            |config: &mut RsiRouterConfig| config.min_success_rate = -0.1,
            |config: &mut RsiRouterConfig| config.min_success_rate = 1.5,
            |config: &mut RsiRouterConfig| config.fixed_target = Some(ModelId::from("  ")),
            |config: &mut RsiRouterConfig| config.history_steps = 0,
            |config: &mut RsiRouterConfig| config.max_output_tokens = 0,
            |config: &mut RsiRouterConfig| config.recognizer_prompt = " ".to_string(),
            |config: &mut RsiRouterConfig| config.candidate_models.clear(),
            |config: &mut RsiRouterConfig| {
                config.candidate_models = vec![ModelId::from(CHEAP), ModelId::from(CHEAP)]
            },
        ] {
            let mut config = config(catalog.clone());
            mutate(&mut config);
            assert!(RsiRouter::new(config).is_err());
        }
    }
}
