// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recognizes a subtask, looks up its Pareto candidates, and routes to the cheapest.
//!
//! A small recognizer model labels the current step with a subtask ID. The subtask
//! dictionary then supplies a Pareto set of execution models and an optional skill
//! template. When recognition is missing, uncertain, or unmatched, the router sends
//! the request to the flagship model.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use switchyard_protocol::{
    ContentBlock, LlmRequest, Message, ModelId, Request, Role, completion_text,
};

use super::util::prompts::prepend_system_prompt;
use crate::core::algorithm::{Algorithm, Driver, RoutingOutcome};
use crate::{LibsyError, Result};

/// The packaged recognition prompt.
pub const DEFAULT_RSI_RECOGNIZER_PROMPT: &str = include_str!("../prompts/rsi/recognize.md");

/// How the router orders the Pareto candidates before trying them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RsiSelection {
    /// Try the lowest cost candidate first. This is the default.
    #[default]
    LowestCost,
    /// Keep the order the dictionary supplied.
    CatalogOrder,
}

/// One candidate model in a subtask's Pareto set.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubtaskCandidate {
    /// Configured target model ID.
    pub model_id: ModelId,
    /// Cost in USD per million tokens. The router uses this to order candidates.
    pub cost_per_million_tokens: f64,
}

/// A skill template attached to a subtask.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SkillTemplate {
    /// Version recorded in the step trace.
    pub version: String,
    /// Text prepended to the execution request as a system instruction.
    pub template: String,
}

/// The dictionary entry for one subtask.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubtaskEntry {
    /// Stable subtask identifier.
    pub subtask_id: String,
    /// Human-readable subtask name shown to the recognizer.
    #[serde(default)]
    pub subtask_name: String,
    /// What the subtask covers. The recognizer uses it to decide membership.
    #[serde(default)]
    pub definition: String,
    /// Pareto candidates for this subtask. They may name any configured candidate target.
    pub candidates: Vec<SubtaskCandidate>,
    /// Skill template applied to the execution request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill: Option<SkillTemplate>,
}

impl SubtaskEntry {
    /// Returns an error when the entry cannot produce a usable route.
    pub fn validate(&self) -> Result<()> {
        if self.subtask_id.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "subtask entry has an empty subtask_id".to_string(),
            });
        }
        if self.candidates.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: format!("subtask {} has no candidates", self.subtask_id),
            });
        }
        for candidate in &self.candidates {
            if candidate.model_id.as_str().trim().is_empty() {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "subtask {} has a candidate with an empty model id",
                        self.subtask_id
                    ),
                });
            }
            if !candidate.cost_per_million_tokens.is_finite()
                || candidate.cost_per_million_tokens < 0.0
            {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "subtask {} candidate {} has an invalid cost",
                        self.subtask_id, candidate.model_id
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Host-supplied subtask dictionary.
///
/// Implementations may read a cloud dictionary, a local file, or an in-memory table.
/// The router reads the whole dictionary before each recognition so the recognizer
/// can choose from every subtask.
#[async_trait]
pub trait SubtaskCatalog: Send + Sync {
    /// Returns every subtask in the dictionary.
    async fn entries(&self) -> Result<Vec<SubtaskEntry>>;
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
    /// How many recent steps the recognizer sees.
    pub history_steps: usize,
    /// Candidate ordering policy.
    pub selection: RsiSelection,
    /// Most output tokens the recognizer may use.
    pub max_output_tokens: u64,
    /// Subtask dictionary.
    pub catalog: Arc<dyn SubtaskCatalog>,
}

impl std::fmt::Debug for RsiRouterConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RsiRouterConfig")
            .field("recognizer", &self.recognizer)
            .field("flagship", &self.flagship)
            .field("candidate_models", &self.candidate_models)
            .field("confidence_threshold", &self.confidence_threshold)
            .field("history_steps", &self.history_steps)
            .field("selection", &self.selection)
            .field("max_output_tokens", &self.max_output_tokens)
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

/// Routes each step by recognized subtask, falling back to a flagship model.
pub struct RsiRouter {
    config: RsiRouterConfig,
}

impl RsiRouter {
    /// Builds an RSI router.
    ///
    /// Returns an error when the configured thresholds, prompts, or candidate list
    /// cannot produce a usable route.
    pub fn new(config: RsiRouterConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    fn recognition_prompt(&self, request: &Request, entries: &[SubtaskEntry]) -> String {
        let mut prompt = self.config.recognizer_prompt.trim().to_string();
        prompt.push_str("\n\nAvailable subtasks:\n");
        for entry in entries {
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
        prompt.push_str(&recognition_context(request, self.config.history_steps));
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
        entries: &[SubtaskEntry],
    ) -> (Option<RecognizedSubtask>, RecognitionTrace) {
        let prompt = self.recognition_prompt(request, entries);
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

    fn route_to_flagship(
        &self,
        driver: &Driver,
        request: Request,
        reason: &str,
        recognition: Option<&RecognizedSubtask>,
        recognition_trace: Option<&RecognitionTrace>,
    ) -> RoutingOutcome {
        driver.set_evidence(serde_json::json!({
            "source": "rsi_router",
            "decision": "flagship",
            "reason": reason,
            "confidence_threshold": self.config.confidence_threshold,
            "subtask_id": recognition.map(|recognition| recognition.subtask_id.as_str()),
            "confidence": recognition.map(|recognition| recognition.confidence),
            "recognizer": recognition_trace.map(RecognitionTrace::evidence),
        }));
        RoutingOutcome::route_to(self.config.flagship.clone(), Vec::new(), request)
    }

    fn select_candidate(&self, entry: &SubtaskEntry) -> Option<(SubtaskCandidate, Vec<ModelId>)> {
        let mut candidates: Vec<&SubtaskCandidate> = entry
            .candidates
            .iter()
            .filter(|candidate| self.config.candidate_models.contains(&candidate.model_id))
            .collect();
        if candidates.is_empty() {
            return None;
        }
        if self.config.selection == RsiSelection::LowestCost {
            candidates.sort_by(|left, right| {
                left.cost_per_million_tokens
                    .total_cmp(&right.cost_per_million_tokens)
            });
        }
        let selected = (*candidates[0]).clone();
        let fallbacks = candidates
            .iter()
            .skip(1)
            .map(|candidate| candidate.model_id.clone())
            .collect();
        Some((selected, fallbacks))
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
        let entries = match self.config.catalog.entries().await {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!(
                    %error,
                    "rsi subtask catalog read failed; routing to the flagship model"
                );
                return Ok(self.route_to_flagship(&driver, request, "catalog_failed", None, None));
            }
        };
        if entries.is_empty() {
            return Ok(self.route_to_flagship(&driver, request, "empty_catalog", None, None));
        }
        let (recognition, recognition_trace) = self.recognize(&driver, &request, &entries).await;
        let Some(recognition) = recognition else {
            return Ok(self.route_to_flagship(
                &driver,
                request,
                "recognition_failed",
                None,
                Some(&recognition_trace),
            ));
        };
        if recognition.subtask_id.is_empty() {
            return Ok(self.route_to_flagship(
                &driver,
                request,
                "no_subtask",
                Some(&recognition),
                Some(&recognition_trace),
            ));
        }
        if recognition.confidence < self.config.confidence_threshold {
            return Ok(self.route_to_flagship(
                &driver,
                request,
                "low_confidence",
                Some(&recognition),
                Some(&recognition_trace),
            ));
        }
        let Some(entry) = entries
            .iter()
            .find(|entry| entry.subtask_id == recognition.subtask_id)
        else {
            return Ok(self.route_to_flagship(
                &driver,
                request,
                "unknown_subtask",
                Some(&recognition),
                Some(&recognition_trace),
            ));
        };
        if let Err(error) = entry.validate() {
            tracing::warn!(
                subtask_id = %recognition.subtask_id,
                %error,
                "rsi subtask entry is invalid; routing to the flagship model"
            );
            return Ok(self.route_to_flagship(
                &driver,
                request,
                "invalid_subtask_entry",
                Some(&recognition),
                Some(&recognition_trace),
            ));
        }
        let Some((selected, fallbacks)) = self.select_candidate(entry) else {
            return Ok(self.route_to_flagship(
                &driver,
                request,
                "no_configured_candidates",
                Some(&recognition),
                Some(&recognition_trace),
            ));
        };

        let skill_version = entry
            .skill
            .as_ref()
            .map(|skill| skill.version.clone())
            .filter(|version| !version.trim().is_empty());
        if let Some(skill) = &entry.skill
            && !skill.template.trim().is_empty()
        {
            let prompt = format!(
                "Skill template version {}:\n{}",
                skill.version,
                skill.template.trim()
            );
            prepend_system_prompt(&mut request, &prompt);
        }

        let mut candidate_models = vec![selected.model_id.clone()];
        candidate_models.extend(fallbacks.iter().cloned());
        driver.set_evidence(serde_json::json!({
            "source": "rsi_router",
            "decision": "subtask",
            "subtask_id": recognition.subtask_id,
            "confidence": recognition.confidence,
            "confidence_threshold": self.config.confidence_threshold,
            "selected_model": selected.model_id,
            "cost_per_million_tokens": selected.cost_per_million_tokens,
            "skill_version": skill_version,
            "selection": match self.config.selection {
                RsiSelection::LowestCost => "lowest_cost",
                RsiSelection::CatalogOrder => "catalog_order",
            },
            "candidates": candidate_models,
            "recognizer": recognition_trace.evidence(),
        }));
        Ok(RoutingOutcome::route_to(
            selected.model_id,
            fallbacks,
            request,
        ))
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

/// Builds the compact context the recognizer reads.
fn recognition_context(request: &Request, history_steps: usize) -> String {
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

    let tools: Vec<String> = request
        .llm_request
        .tools
        .iter()
        .filter_map(tool_name)
        .collect();
    sections.push(format!(
        "CURRENT TOOLS:\n{}",
        if tools.is_empty() {
            "(none)".to_string()
        } else {
            tools.join(", ")
        }
    ));

    if let Some(state) = latest_tool_state(&request.llm_request.messages) {
        sections.push(format!(
            "LATEST TOOL STATE:\n{}",
            clip(&state, MAX_BLOCK_CHARS)
        ));
    }

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

fn latest_tool_state(messages: &[Message]) -> Option<String> {
    for message in messages.iter().rev() {
        let mut rendered = Vec::new();
        for block in &message.content {
            match block {
                ContentBlock::ToolCall(call) => rendered.push(format!(
                    "tool call {} ({}) with {}",
                    call.name, call.id, call.arguments
                )),
                ContentBlock::ToolResult(result) => rendered.push(format!(
                    "tool result for {} (error: {}) {}",
                    result.tool_call_id,
                    result.is_error.unwrap_or(false),
                    render_content(&result.content)
                )),
                _ => {}
            }
        }
        if !rendered.is_empty() {
            return Some(rendered.join("\n"));
        }
    }
    None
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

fn tool_name(tool: &switchyard_protocol::Tool) -> Option<String> {
    match tool {
        switchyard_protocol::Tool::Function(definition) => Some(definition.name.clone()),
        switchyard_protocol::Tool::Mcp(toolset) => Some(toolset.name.clone()),
        switchyard_protocol::Tool::Provider { .. } => None,
    }
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
        entries: Mutex<Vec<SubtaskEntry>>,
        fail: bool,
    }

    impl StaticCatalog {
        fn new(entries: Vec<SubtaskEntry>) -> Self {
            Self {
                entries: Mutex::new(entries),
                fail: false,
            }
        }

        fn failing() -> Self {
            Self {
                entries: Mutex::new(Vec::new()),
                fail: true,
            }
        }
    }

    #[async_trait]
    impl SubtaskCatalog for StaticCatalog {
        async fn entries(&self) -> Result<Vec<SubtaskEntry>> {
            if self.fail {
                return Err(LibsyError::AlgorithmError {
                    message: "catalog unavailable".to_string(),
                });
            }
            Ok(self.entries.lock().clone())
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
            history_steps: 3,
            selection: RsiSelection::LowestCost,
            max_output_tokens: 128,
            catalog,
        }
    }

    fn entry() -> SubtaskEntry {
        SubtaskEntry {
            subtask_id: "edit-code".to_string(),
            subtask_name: "Edit code".to_string(),
            definition: "Change existing source files in a small, focused patch.".to_string(),
            candidates: vec![
                SubtaskCandidate {
                    model_id: ModelId::from(PRICY),
                    cost_per_million_tokens: 9.0,
                },
                SubtaskCandidate {
                    model_id: ModelId::from(CHEAP),
                    cost_per_million_tokens: 0.5,
                },
                SubtaskCandidate {
                    model_id: ModelId::from(MID),
                    cost_per_million_tokens: 3.0,
                },
            ],
            skill: Some(SkillTemplate {
                version: "v3".to_string(),
                template: "Prefer a minimal patch.".to_string(),
            }),
        }
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
    async fn routes_to_the_cheapest_candidate_and_applies_the_skill() -> Result<()> {
        let entry = entry();
        let router = Arc::new(
            RsiRouter::new(config(Arc::new(StaticCatalog::new(vec![entry]))))
                .expect("config should be valid"),
        );
        let outcome = test_drive_with_state(
            router,
            request(),
            models(),
            State::default(),
            move |target: ModelId, request: Request| {
                if target == RECOGNIZER {
                    return std::future::ready(Ok(reply(recognizer_reply("edit-code", 0.95))));
                }
                let skill =
                    request.llm_request.instructions.iter().any(|instruction| {
                        instruction.content.iter().any(|block| matches!(
                        block,
                        ContentBlock::Text { text } if text.contains("Prefer a minimal patch.")
                    ))
                    });
                assert!(skill, "skill template should be prepended");
                std::future::ready(Ok(reply(target.as_str())))
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_id()?, &ModelId::from(CHEAP));
        let evidence = outcome
            .metadata
            .expect("outcome metadata")
            .evidence
            .expect("evidence");
        assert_eq!(evidence["subtask_id"], "edit-code");
        assert_eq!(evidence["skill_version"], "v3");
        assert_eq!(evidence["selected_model"], CHEAP);
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_to_the_flagship_below_the_threshold() -> Result<()> {
        let (result, _) = route_with(
            config(Arc::new(StaticCatalog::new(vec![entry()]))),
            recognizer_reply("edit-code", 0.4),
        )
        .await;
        assert_eq!(result?.0, FLAGSHIP);
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_for_an_empty_or_unparsable_subtask() -> Result<()> {
        for recognition in ["not json", recognizer_reply("", 0.99).as_str()] {
            let (result, _) = route_with(
                config(Arc::new(StaticCatalog::new(vec![entry()]))),
                recognition.to_string(),
            )
            .await;
            assert_eq!(result?.0, FLAGSHIP);
        }
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_when_the_catalog_fails_or_is_empty() -> Result<()> {
        let (empty, _) = route_with(
            config(Arc::new(StaticCatalog::new(Vec::new()))),
            recognizer_reply("edit-code", 0.99),
        )
        .await;
        assert_eq!(empty?.0, FLAGSHIP);

        let (failed, _) = route_with(
            config(Arc::new(StaticCatalog::failing())),
            recognizer_reply("edit-code", 0.99),
        )
        .await;
        assert_eq!(failed?.0, FLAGSHIP);
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_when_the_recognized_id_is_not_in_the_catalog() -> Result<()> {
        let (result, _) = route_with(
            config(Arc::new(StaticCatalog::new(vec![entry()]))),
            recognizer_reply("run-tests", 0.99),
        )
        .await;
        assert_eq!(result?.0, FLAGSHIP);
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_when_recognition_fails() -> Result<()> {
        let router =
            Arc::new(RsiRouter::new(config(Arc::new(StaticCatalog::new(vec![entry()])))).unwrap());
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
        let mut entry = entry();
        entry.candidates.push(SubtaskCandidate {
            model_id: ModelId::from("model/unconfigured"),
            cost_per_million_tokens: 0.01,
        });
        let (result, _) = route_with(
            config(Arc::new(StaticCatalog::new(vec![entry]))),
            recognizer_reply("edit-code", 0.99),
        )
        .await;
        assert_eq!(result?.0, CHEAP);
        Ok(())
    }

    #[tokio::test]
    async fn catalog_order_keeps_the_dictionary_order() -> Result<()> {
        let mut config = config(Arc::new(StaticCatalog::new(vec![entry()])));
        config.selection = RsiSelection::CatalogOrder;
        let (result, _) = route_with(config, recognizer_reply("edit-code", 0.99)).await;
        assert_eq!(result?.0, PRICY);
        Ok(())
    }

    #[tokio::test]
    async fn recognizer_sees_only_the_recent_steps() -> Result<()> {
        let mut config = config(Arc::new(StaticCatalog::new(vec![entry()])));
        config.history_steps = 1;
        let (_, prompt) = route_with(config, recognizer_reply("edit-code", 0.99)).await;
        assert!(prompt.contains("fn parse() {}"));
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("Available subtasks"));
        assert!(prompt.contains("edit-code"));
        assert!(prompt.contains("Edit code"));
        assert!(prompt.contains("Change existing source files"));
        Ok(())
    }

    #[test]
    fn rejects_invalid_configuration() {
        let catalog: Arc<dyn SubtaskCatalog> = Arc::new(StaticCatalog::new(Vec::new()));
        for mutate in [
            |config: &mut RsiRouterConfig| config.confidence_threshold = 1.5,
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
