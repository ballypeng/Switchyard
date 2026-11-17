// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quality, latency, and cost (QLC) routing.
//!
//! Candidates that fail a hard requirement are removed first. The rest are scored
//! with configured weights. The best candidate serves the request and the others
//! remain as ordered fallbacks.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Map, Value};
use switchyard_protocol::{ContentBlock, LlmRequest, ModelId, Request};

use crate::core::algorithm::{Algorithm, Driver, RoutingOutcome};
use crate::{LibsyError, Result};

/// One candidate model and its configured values.
#[derive(Clone, Debug)]
pub struct QlcCandidate {
    /// Model this candidate routes to.
    pub target: ModelId,
    /// Quality score in `[0, 1]`. Higher is better.
    pub quality_score: f64,
    /// Latency score in `[0, 1]`. Higher means faster.
    pub latency_score: f64,
    /// Cost score in `[0, 1]`. Higher means cheaper.
    pub cost_score: f64,
    /// Estimated latency in milliseconds, checked against `max_latency_ms`.
    pub estimated_latency_ms: Option<u64>,
    /// Largest context the candidate supports. `None` places no limit.
    pub context_window: Option<u32>,
    /// Whether the candidate accepts image input.
    pub vision: bool,
    /// Whether the candidate supports tool calls.
    pub tool_calling: bool,
    /// Whether the candidate satisfies a local-only deployment requirement.
    pub local: bool,
    /// Whether the candidate is taking traffic. `false` removes it from this round.
    pub available: bool,
}

/// Settings for [`Qlc`].
#[derive(Clone, Debug)]
pub struct QlcConfig {
    /// Candidates in config order. Ties keep this order.
    pub candidates: Vec<QlcCandidate>,
    /// Weight for [`QlcCandidate::quality_score`].
    pub weight_quality: f64,
    /// Weight for [`QlcCandidate::latency_score`].
    pub weight_latency: f64,
    /// Weight for [`QlcCandidate::cost_score`].
    pub weight_cost: f64,
    /// Hard latency ceiling in milliseconds. `None` disables the ceiling.
    pub max_latency_ms: Option<u64>,
    /// Restricts routing to candidates marked [`QlcCandidate::local`].
    pub require_local: bool,
}

/// Scoring weights for one request.
#[derive(Clone, Copy, Debug)]
struct Weights {
    quality: f64,
    latency: f64,
    cost: f64,
}

/// Reads one non-negative weight from the raw request body.
fn request_weight(raw: Option<&Map<String, Value>>, key: &str) -> Option<f64> {
    raw?.get(key)?.as_f64().filter(|value| *value >= 0.0)
}

/// Routes by filtering candidates, then ranking them with a configured score.
pub struct Qlc {
    config: QlcConfig,
    /// Calls in flight for each local candidate, indexed like `config.candidates`.
    in_flight: Vec<AtomicUsize>,
    /// Requests routing to each local candidate right now, indexed like `config.candidates`.
    routing: Vec<AtomicUsize>,
    /// Requests that selected a local candidate but whose call has not ended yet.
    ///
    /// Shared so a decision can release its slot after the expected call duration.
    reserved: Arc<Vec<AtomicUsize>>,
}

impl Qlc {
    /// Builds a QLC router.
    ///
    /// # Errors
    ///
    /// Returns an error when the candidate list is empty or has duplicate models,
    /// a score is outside `[0, 1]`, the weights are invalid, or a latency ceiling
    /// is set without an estimate for every candidate.
    pub fn new(config: QlcConfig) -> Result<Self> {
        if config.candidates.is_empty() {
            return Err(invalid("at least one candidate is required"));
        }
        validate_weights(&config)?;
        let mut seen = HashSet::new();
        for candidate in &config.candidates {
            if !seen.insert(&candidate.target) {
                return Err(invalid(format!(
                    "candidate model {} is listed more than once",
                    candidate.target
                )));
            }
            for (name, score) in [
                ("quality_score", candidate.quality_score),
                ("latency_score", candidate.latency_score),
                ("cost_score", candidate.cost_score),
            ] {
                if !(0.0..=1.0).contains(&score) {
                    return Err(invalid(format!(
                        "candidate {} {name} must be between 0 and 1",
                        candidate.target
                    )));
                }
            }
            if config.max_latency_ms.is_some() && candidate.estimated_latency_ms.is_none() {
                return Err(invalid(format!(
                    "candidate {} needs estimated_latency_ms when max_latency_ms is set",
                    candidate.target
                )));
            }
        }
        let in_flight = config
            .candidates
            .iter()
            .map(|_| AtomicUsize::new(0))
            .collect();
        let routing = config
            .candidates
            .iter()
            .map(|_| AtomicUsize::new(0))
            .collect();
        let reserved = Arc::new(
            config
                .candidates
                .iter()
                .map(|_| AtomicUsize::new(0))
                .collect(),
        );
        Ok(Self {
            config,
            in_flight,
            routing,
            reserved,
        })
    }

    fn score(&self, candidate: &QlcCandidate, in_flight: usize, weights: &Weights) -> f64 {
        let latency_score = effective_latency_score(candidate, in_flight);
        weights.quality * candidate.quality_score
            + weights.latency * latency_score
            + weights.cost * candidate.cost_score
    }

    /// Weights for one request. Request values win; configured values fill the rest.
    fn weights(&self, request: &Request) -> Weights {
        let raw = request.raw_request.as_ref().and_then(Value::as_object);
        let weights = Weights {
            quality: request_weight(raw, "weight_quality").unwrap_or(self.config.weight_quality),
            latency: request_weight(raw, "weight_latency").unwrap_or(self.config.weight_latency),
            cost: request_weight(raw, "weight_cost").unwrap_or(self.config.weight_cost),
        };
        // All-zero weights would make every candidate tie. Use the configured weights.
        if weights.quality + weights.latency + weights.cost > 0.0 {
            weights
        } else {
            Weights {
                quality: self.config.weight_quality,
                latency: self.config.weight_latency,
                cost: self.config.weight_cost,
            }
        }
    }

    fn local_index(&self, model: &ModelId) -> Option<usize> {
        self.config
            .candidates
            .iter()
            .position(|candidate| candidate.local && &candidate.target == model)
    }

    fn passes(&self, candidate: &QlcCandidate, profile: &RequestProfile) -> bool {
        if !candidate.available
            || (profile.needs_vision && !candidate.vision)
            || (profile.needs_tool_calling && !candidate.tool_calling)
            || (self.config.require_local && !candidate.local)
        {
            return false;
        }
        if let Some(window) = candidate.context_window
            && profile.tokens > u64::from(window)
        {
            return false;
        }
        if let Some(limit) = self.config.max_latency_ms
            && candidate
                .estimated_latency_ms
                .is_none_or(|latency| latency > limit)
        {
            return false;
        }
        true
    }
}

#[async_trait]
impl Algorithm for Qlc {
    fn name(&self) -> &str {
        "qlc"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let profile = RequestProfile::of(&request.llm_request);
        let weights = self.weights(&request);
        // Requests that route at the same time all arrive before any model call starts.
        // Count each one while it is routing so they see each other.
        let mut admitted = Vec::new();
        let mut routing = vec![0; self.config.candidates.len()];
        for (index, candidate) in self.config.candidates.iter().enumerate() {
            if candidate.local && self.passes(candidate, &profile) {
                routing[index] = self.routing[index].fetch_add(1, Ordering::AcqRel);
                admitted.push(index);
            }
        }
        let _admissions = RoutingAdmissions {
            qlc: self.as_ref(),
            indices: admitted,
        };
        let in_flight: Vec<usize> = self
            .in_flight
            .iter()
            .map(|count| count.load(Ordering::Acquire))
            .collect();
        let reserved: Vec<usize> = self
            .reserved
            .iter()
            .map(|count| count.load(Ordering::Acquire))
            .collect();
        let concurrency: Vec<usize> = (0..self.config.candidates.len())
            .map(|index| in_flight[index] + routing[index] + reserved[index])
            .collect();
        let mut ranked: Vec<(&QlcCandidate, usize, f64)> = self
            .config
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| self.passes(candidate, &profile))
            .map(|(index, candidate)| {
                (
                    candidate,
                    index,
                    self.score(candidate, concurrency[index], &weights),
                )
            })
            .collect();
        ranked.sort_by(|left, right| right.2.total_cmp(&left.2));

        let Some(&(selected, selected_index, selected_score)) = ranked.first() else {
            return Err(LibsyError::AlgorithmError {
                message: "qlc: no candidate passed the hard filters".to_string(),
            });
        };
        // Keep the selection counted until its model call starts.
        if self.config.candidates[selected_index].local {
            self.reserved[selected_index].fetch_add(1, Ordering::AcqRel);
        }
        let candidates: Vec<_> = self
            .config
            .candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                serde_json::json!({
                    "target": candidate.target.as_str(),
                    "score": self.score(candidate, concurrency[index], &weights),
                    "quality_score": candidate.quality_score,
                    "latency_score": effective_latency_score(candidate, concurrency[index]),
                    "base_latency_score": candidate.latency_score,
                    "concurrency": concurrency[index],
                    "cost_score": candidate.cost_score,
                    "eligible": self.passes(candidate, &profile),
                })
            })
            .collect();
        driver.set_evidence(serde_json::json!({
            "source": "qlc",
            "selected": selected.target.as_str(),
            "score": selected_score,
            "weights": {
                "quality": weights.quality,
                "latency": weights.latency,
                "cost": weights.cost,
            },
            "candidates": candidates,
        }));
        let fallbacks = ranked
            .iter()
            .skip(1)
            .map(|(candidate, _, _)| candidate.target.clone())
            .collect();
        Ok(RoutingOutcome::route_to(
            selected.target.clone(),
            fallbacks,
            request,
        ))
    }

    fn model_call_started(&self, model: &ModelId) {
        if let Some(index) = self.local_index(model) {
            // The request that selected this model reserved a slot. Move it to `in_flight`.
            release_reservation(&self.reserved, index);
            self.in_flight[index].fetch_add(1, Ordering::AcqRel);
        }
    }

    fn model_call_finished(&self, model: &ModelId) {
        if let Some(index) = self.local_index(model) {
            let _ =
                self.in_flight[index].try_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                    count.checked_sub(1)
                });
        }
    }

    fn release_selection(&self, model: &ModelId) {
        let Some(index) = self.local_index(model) else {
            return;
        };
        // A decision does not start a call here. Keep its slot for the expected
        // call duration so concurrent decisions see each other.
        let hold = self.config.candidates[index].estimated_latency_ms.map_or(
            Duration::from_millis(DEFAULT_DECISION_HOLD_MS),
            Duration::from_millis,
        );
        let reserved = Arc::clone(&self.reserved);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    tokio::time::sleep(hold).await;
                    release_reservation(&reserved, index);
                });
            }
            Err(_) => release_reservation(&reserved, index),
        }
    }
}

/// How long a decision without a latency estimate keeps its local slot.
const DEFAULT_DECISION_HOLD_MS: u64 = 1_000;

fn release_reservation(reserved: &[AtomicUsize], index: usize) {
    let _ = reserved[index].try_update(Ordering::AcqRel, Ordering::Acquire, |count| {
        count.checked_sub(1)
    });
}

/// Releases a route's temporary admission slots when the route finishes.
struct RoutingAdmissions<'a> {
    qlc: &'a Qlc,
    indices: Vec<usize>,
}

impl Drop for RoutingAdmissions<'_> {
    fn drop(&mut self) {
        for &index in &self.indices {
            self.qlc.routing[index].fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Each request already running on a local model lowers its latency score by this much.
const CONCURRENCY_PENALTY: f64 = 0.08;

/// The latency score a local candidate has while `in_flight` calls are running.
fn effective_latency_score(candidate: &QlcCandidate, in_flight: usize) -> f64 {
    if candidate.local {
        (candidate.latency_score - CONCURRENCY_PENALTY * in_flight as f64).max(0.0)
    } else {
        candidate.latency_score
    }
}

/// The request facts the hard filters read.
struct RequestProfile {
    needs_vision: bool,
    needs_tool_calling: bool,
    tokens: u64,
}

impl RequestProfile {
    fn of(request: &LlmRequest) -> Self {
        Self {
            needs_vision: request
                .instructions
                .iter()
                .any(|block| has_image(&block.content))
                || request
                    .messages
                    .iter()
                    .any(|message| has_image(&message.content)),
            needs_tool_calling: !request.tools.is_empty()
                || request.messages.iter().any(|message| {
                    message.content.iter().any(|block| {
                        matches!(
                            block,
                            ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
                        )
                    })
                }),
            tokens: estimated_tokens(request),
        }
    }
}

fn has_image(blocks: &[ContentBlock]) -> bool {
    blocks.iter().any(|block| match block {
        ContentBlock::Image { .. } => true,
        ContentBlock::ToolResult(result) => has_image(&result.content),
        _ => false,
    })
}

/// A rough token count: one token per four serialized characters, plus the output budget.
fn estimated_tokens(request: &LlmRequest) -> u64 {
    let chars: usize = [
        serde_json::to_string(&request.instructions),
        serde_json::to_string(&request.messages),
        serde_json::to_string(&request.tools),
    ]
    .into_iter()
    .filter_map(std::result::Result::ok)
    .map(|part| part.chars().count())
    .sum();
    let input = (chars as u64).div_ceil(4);
    input.saturating_add(request.output.max_output_tokens.unwrap_or(0))
}

fn validate_weights(config: &QlcConfig) -> Result<()> {
    let weights = [
        ("weight_quality", config.weight_quality),
        ("weight_latency", config.weight_latency),
        ("weight_cost", config.weight_cost),
    ];
    if weights
        .iter()
        .any(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(invalid("weights must be finite and nonnegative"));
    }

    if !weights.iter().any(|(_, weight)| *weight > 0.0) {
        return Err(invalid("at least one weight must be positive"));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: format!("invalid qlc configuration: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use switchyard_protocol::{
        ImageSource, Message, Role, Tool, ToolCall, ToolDefinition, text_request,
    };

    use crate::core::algorithm::{RuntimeModels, Step};
    use crate::core::state::State;

    fn candidate(target: &str) -> QlcCandidate {
        QlcCandidate {
            target: ModelId::from(target),
            quality_score: 0.5,
            latency_score: 0.5,
            cost_score: 0.5,
            estimated_latency_ms: None,
            context_window: None,
            vision: false,
            tool_calling: false,
            local: false,
            available: true,
        }
    }

    fn config(candidates: Vec<QlcCandidate>) -> QlcConfig {
        QlcConfig {
            candidates,
            weight_quality: 1.0,
            weight_latency: 1.0,
            weight_cost: 1.0,
            max_latency_ms: None,
            require_local: false,
        }
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("qlc".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        }
    }

    async fn outcome(qlc: Qlc, request: Request) -> Result<RoutingOutcome> {
        outcome_of(Arc::new(qlc), request).await
    }

    async fn outcome_of(qlc: Arc<Qlc>, request: Request) -> Result<RoutingOutcome> {
        let mut stream = qlc.run_stream(
            request,
            Arc::new(RuntimeModels::default()),
            State::default(),
        );
        while let Some(step) = stream.next().await {
            if let Step::Done(outcome) = step? {
                return Ok(*outcome);
            }
        }
        Err(LibsyError::MissingFinalResponse)
    }

    fn evidence(outcome: &RoutingOutcome) -> serde_json::Value {
        outcome
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.evidence.clone())
            .expect("qlc records evidence")
    }

    #[tokio::test]
    async fn score_orders_the_selected_model_and_fallbacks() -> Result<()> {
        let mut best = candidate("best");
        best.quality_score = 1.0;
        let mut second = candidate("second");
        second.quality_score = 0.4;
        second.cost_score = 1.0;
        let mut last = candidate("last");
        last.quality_score = 0.0;
        last.cost_score = 0.0;
        let qlc = Qlc::new(config(vec![last, best, second]))?;

        let outcome = outcome(qlc, request()).await?;
        assert_eq!(
            outcome.selected_model_ids,
            ["best", "second", "last"].map(ModelId::from)
        );
        Ok(())
    }

    #[tokio::test]
    async fn outcome_evidence_lists_every_candidate_score() -> Result<()> {
        let mut best = candidate("best");
        best.quality_score = 1.0;
        let mut unavailable = candidate("unavailable");
        unavailable.available = false;
        let qlc = Qlc::new(config(vec![unavailable, best]))?;

        let outcome = outcome(qlc, request()).await?;
        let evidence = outcome
            .metadata
            .and_then(|metadata| metadata.evidence)
            .expect("qlc records evidence");
        assert_eq!(evidence["selected"], "best");
        assert_eq!(evidence["candidates"][0]["target"], "unavailable");
        assert_eq!(evidence["candidates"][0]["eligible"], false);
        assert_eq!(evidence["candidates"][1]["target"], "best");
        assert_eq!(evidence["candidates"][1]["eligible"], true);
        Ok(())
    }

    #[tokio::test]
    async fn local_concurrency_lowers_latency_score() -> Result<()> {
        for concurrency in [0, 1, 2, 10] {
            let mut local = candidate("local");
            local.local = true;
            local.latency_score = 0.20;
            let qlc = Arc::new(Qlc::new(config(vec![local]))?);
            for _ in 0..concurrency {
                qlc.model_call_started(&ModelId::from("local"));
            }

            let outcome = outcome_of(Arc::clone(&qlc), request()).await?;
            let evidence = evidence(&outcome);
            let candidate = &evidence["candidates"][0];
            assert_eq!(candidate["concurrency"], concurrency);
            assert_eq!(candidate["base_latency_score"], 0.20);
            let expected = (0.20 - CONCURRENCY_PENALTY * concurrency as f64).max(0.0);
            let actual = candidate["latency_score"].as_f64().expect("latency score");
            assert!(
                (actual - expected).abs() < 1e-9,
                "{concurrency} concurrent calls: expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn concurrency_penalty_changes_the_selected_model() -> Result<()> {
        let mut local = candidate("local");
        local.local = true;
        local.latency_score = 0.60;
        let mut cloud = candidate("cloud");
        cloud.latency_score = 0.30;
        let qlc = Arc::new(Qlc::new(config(vec![local, cloud]))?);

        let selected = outcome_of(Arc::clone(&qlc), request()).await?;
        assert_eq!(selected.selected_model_id()?, "local");

        for _ in 0..7 {
            qlc.model_call_started(&ModelId::from("local"));
        }
        let selected = outcome_of(Arc::clone(&qlc), request()).await?;
        assert_eq!(selected.selected_model_id()?, "cloud");
        let evidence = evidence(&selected);
        assert_eq!(evidence["candidates"][0]["concurrency"], 7);
        assert_eq!(evidence["candidates"][1]["concurrency"], 0);
        Ok(())
    }

    #[tokio::test]
    async fn finished_calls_free_their_concurrency_slot() -> Result<()> {
        let mut local = candidate("local");
        local.local = true;
        let qlc = Arc::new(Qlc::new(config(vec![local]))?);
        let model = ModelId::from("local");

        // A finished call with no start never pushes the count below zero.
        qlc.model_call_finished(&model);
        qlc.model_call_started(&model);
        qlc.model_call_started(&model);
        qlc.model_call_finished(&model);

        let outcome = outcome_of(Arc::clone(&qlc), request()).await?;
        assert_eq!(evidence(&outcome)["candidates"][0]["concurrency"], 1);
        Ok(())
    }

    #[tokio::test]
    async fn requests_routing_at_the_same_time_count_as_concurrent() -> Result<()> {
        let mut local = candidate("local");
        local.local = true;
        local.latency_score = 0.20;
        let qlc = Arc::new(Qlc::new(config(vec![local]))?);
        // Another request is inside its route call right now.
        qlc.routing[0].fetch_add(1, Ordering::AcqRel);

        let outcome = outcome_of(Arc::clone(&qlc), request()).await?;
        let candidate = &evidence(&outcome)["candidates"][0];
        assert_eq!(candidate["concurrency"], 1);
        let expected = 0.20 - CONCURRENCY_PENALTY;
        let latency = candidate["latency_score"].as_f64().expect("latency score");
        assert!(
            (latency - expected).abs() < 1e-9,
            "unexpected latency: {latency}"
        );
        // This route released its own admission; only the simulated one remains.
        assert_eq!(qlc.routing[0].load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test]
    async fn selection_reservation_counts_until_the_call_starts() -> Result<()> {
        let mut local = candidate("local");
        local.local = true;
        let qlc = Arc::new(Qlc::new(config(vec![local]))?);
        let model = ModelId::from("local");

        // Routing reserves the selected local model.
        let _ = outcome_of(Arc::clone(&qlc), request()).await?;
        assert_eq!(qlc.reserved[0].load(Ordering::Acquire), 1);

        // A route in the gap before the call starts sees the reservation.
        let during = outcome_of(Arc::clone(&qlc), request()).await?;
        assert_eq!(evidence(&during)["candidates"][0]["concurrency"], 1);

        // Starting the call moves the reservation to the in-flight count.
        qlc.model_call_started(&model);
        assert_eq!(qlc.reserved[0].load(Ordering::Acquire), 1);
        assert_eq!(qlc.in_flight[0].load(Ordering::Acquire), 1);

        let after = outcome_of(Arc::clone(&qlc), request()).await?;
        assert_eq!(evidence(&after)["candidates"][0]["concurrency"], 2);
        Ok(())
    }

    #[tokio::test]
    async fn released_decisions_keep_their_slot_for_the_estimated_latency() -> Result<()> {
        let mut local = candidate("local");
        local.local = true;
        local.latency_score = 0.20;
        local.estimated_latency_ms = Some(20);
        let qlc = Arc::new(Qlc::new(config(vec![local]))?);
        let model = ModelId::from("local");

        let _ = outcome_of(Arc::clone(&qlc), request()).await?;
        qlc.release_selection(&model);
        assert_eq!(qlc.reserved[0].load(Ordering::Acquire), 1);

        // The released decision still counts until its expected call would end.
        let during = outcome_of(Arc::clone(&qlc), request()).await?;
        assert_eq!(evidence(&during)["candidates"][0]["concurrency"], 1);
        qlc.release_selection(&model);
        assert_eq!(qlc.reserved[0].load(Ordering::Acquire), 2);

        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(qlc.reserved[0].load(Ordering::Acquire), 0);
        Ok(())
    }

    #[tokio::test]
    async fn routing_admissions_are_released_when_no_candidate_passes() -> Result<()> {
        let mut local = candidate("local");
        local.local = true;
        local.available = false;
        let qlc = Arc::new(Qlc::new(config(vec![local]))?);

        let error = outcome_of(Arc::clone(&qlc), request())
            .await
            .err()
            .expect("all candidates were filtered");
        assert!(error.to_string().contains("no candidate"));
        assert_eq!(qlc.routing[0].load(Ordering::Acquire), 0);
        Ok(())
    }

    #[tokio::test]
    async fn weights_decide_between_quality_and_cost() -> Result<()> {
        let mut quality = candidate("quality");
        quality.quality_score = 1.0;
        let mut cheap = candidate("cheap");
        cheap.cost_score = 1.0;

        let mut settings = config(vec![cheap.clone(), quality.clone()]);
        settings.weight_quality = 10.0;
        let selected = outcome(Qlc::new(settings)?, request()).await?;
        assert_eq!(selected.selected_model_id()?, "quality");

        let mut settings = config(vec![cheap, quality]);
        settings.weight_cost = 10.0;
        let selected = outcome(Qlc::new(settings)?, request()).await?;
        assert_eq!(selected.selected_model_id()?, "cheap");
        Ok(())
    }

    #[tokio::test]
    async fn request_weights_override_configured_weights() -> Result<()> {
        let mut quality = candidate("quality");
        quality.quality_score = 1.0;
        quality.latency_score = 0.0;
        quality.cost_score = 0.0;
        let mut cheap = candidate("cheap");
        cheap.quality_score = 0.0;
        cheap.latency_score = 0.0;
        cheap.cost_score = 1.0;

        let mut settings = config(vec![quality, cheap]);
        settings.weight_quality = 1.0;
        settings.weight_latency = 0.0;
        settings.weight_cost = 0.0;

        let selected = outcome(Qlc::new(settings.clone())?, request()).await?;
        assert_eq!(selected.selected_model_id()?, "quality");

        let mut request = request();
        request.raw_request = Some(serde_json::json!({
            "weight_quality": 0.0,
            "weight_latency": 0.0,
            "weight_cost": 1.0,
        }));
        let selected = outcome(Qlc::new(settings)?, request).await?;
        assert_eq!(selected.selected_model_id()?, "cheap");
        Ok(())
    }

    #[tokio::test]
    async fn request_weights_fill_missing_keys_from_config() -> Result<()> {
        let mut settings = config(vec![candidate("only")]);
        settings.weight_quality = 1.0;
        settings.weight_latency = 2.0;
        settings.weight_cost = 3.0;

        let mut request = request();
        request.raw_request = Some(serde_json::json!({
            "weight_quality": 0.5,
            "weight_latency": -1.0,
            "weight_cost": "high",
        }));
        let outcome = outcome(Qlc::new(settings)?, request).await?;

        let weights = &evidence(&outcome)["weights"];
        assert_eq!(weights["quality"], 0.5);
        assert_eq!(weights["latency"], 2.0);
        assert_eq!(weights["cost"], 3.0);
        Ok(())
    }

    #[tokio::test]
    async fn all_zero_request_weights_use_config_weights() -> Result<()> {
        let mut settings = config(vec![candidate("only")]);
        settings.weight_quality = 1.0;
        settings.weight_latency = 2.0;
        settings.weight_cost = 3.0;

        let mut request = request();
        request.raw_request = Some(serde_json::json!({
            "weight_quality": 0.0,
            "weight_latency": 0.0,
            "weight_cost": 0.0,
        }));
        let outcome = outcome(Qlc::new(settings)?, request).await?;

        let weights = &evidence(&outcome)["weights"];
        assert_eq!(weights["quality"], 1.0);
        assert_eq!(weights["latency"], 2.0);
        assert_eq!(weights["cost"], 3.0);
        Ok(())
    }

    #[tokio::test]
    async fn vision_tool_and_context_filters_remove_candidates() -> Result<()> {
        let mut no_vision = candidate("no_vision");
        no_vision.tool_calling = true;
        let mut no_tools = candidate("no_tools");
        no_tools.vision = true;
        let mut small = candidate("small");
        small.vision = true;
        small.tool_calling = true;
        small.context_window = Some(1);
        let mut good = candidate("good");
        good.vision = true;
        good.tool_calling = true;

        let qlc = Qlc::new(config(vec![no_vision, no_tools, small, good]))?;
        let mut request = request();
        request.llm_request.messages[0]
            .content
            .push(ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example.test/image.png".to_string(),
                    detail: None,
                },
            });
        request.llm_request.tools = vec![Tool::Function(ToolDefinition {
            name: "bash".to_string(),
            description: None,
            parameters: serde_json::json!({}),
            strict: None,
        })];

        let outcome = outcome(qlc, request).await?;
        assert_eq!(outcome.selected_model_ids, [ModelId::from("good")]);
        Ok(())
    }

    #[tokio::test]
    async fn tool_history_also_requires_tool_calling() -> Result<()> {
        let mut no_tools = candidate("no_tools");
        no_tools.cost_score = 1.0;
        let mut tools = candidate("tools");
        tools.tool_calling = true;
        let qlc = Qlc::new(config(vec![no_tools, tools]))?;
        let mut request = request();
        request.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: "bash".to_string(),
                arguments: serde_json::json!({}),
            })],
        });

        let outcome = outcome(qlc, request).await?;
        assert_eq!(outcome.selected_model_ids, [ModelId::from("tools")]);
        Ok(())
    }

    #[tokio::test]
    async fn locality_and_latency_filters_remove_candidates() -> Result<()> {
        let mut remote = candidate("remote");
        remote.cost_score = 1.0;
        remote.estimated_latency_ms = Some(100);
        let mut slow = candidate("slow");
        slow.local = true;
        slow.quality_score = 1.0;
        slow.estimated_latency_ms = Some(5_000);
        let mut local_fast = candidate("local_fast");
        local_fast.local = true;
        local_fast.estimated_latency_ms = Some(500);

        let mut settings = config(vec![remote, slow, local_fast]);
        settings.require_local = true;
        settings.max_latency_ms = Some(1_000);
        let outcome = outcome(Qlc::new(settings)?, request()).await?;
        assert_eq!(outcome.selected_model_ids, [ModelId::from("local_fast")]);
        Ok(())
    }

    #[tokio::test]
    async fn unavailable_candidates_are_removed() -> Result<()> {
        let mut down = candidate("down");
        down.quality_score = 1.0;
        down.available = false;
        let qlc = Qlc::new(config(vec![down, candidate("up")]))?;

        let outcome = outcome(qlc, request()).await?;
        assert_eq!(outcome.selected_model_ids, [ModelId::from("up")]);
        Ok(())
    }

    #[tokio::test]
    async fn equal_scores_keep_config_order() -> Result<()> {
        let qlc = Qlc::new(config(vec![
            candidate("first"),
            candidate("second"),
            candidate("third"),
        ]))?;
        let outcome = outcome(qlc, request()).await?;
        assert_eq!(
            outcome.selected_model_ids,
            ["first", "second", "third"].map(ModelId::from)
        );
        Ok(())
    }

    #[tokio::test]
    async fn no_candidate_passing_the_filters_is_an_error() -> Result<()> {
        let mut candidate = candidate("only");
        candidate.available = false;
        let qlc = Qlc::new(config(vec![candidate]))?;
        let error = outcome(qlc, request())
            .await
            .err()
            .expect("all candidates were filtered");
        assert!(error.to_string().contains("no candidate"));
        Ok(())
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let cases: [(&str, QlcConfig); 5] = [
            ("at least one", config(Vec::new())),
            (
                "listed more than once",
                config(vec![candidate("same"), candidate("same")]),
            ),
            (
                "between 0 and 1",
                config(vec![QlcCandidate {
                    quality_score: 1.5,
                    ..candidate("bad")
                }]),
            ),
            (
                "finite and nonnegative",
                QlcConfig {
                    weight_quality: -1.0,
                    ..config(vec![candidate("bad")])
                },
            ),
            (
                "at least one weight",
                QlcConfig {
                    weight_quality: 0.0,
                    weight_latency: 0.0,
                    weight_cost: 0.0,
                    ..config(vec![candidate("bad")])
                },
            ),
        ];
        for (expected, settings) in cases {
            let message = match Qlc::new(settings) {
                Ok(_) => String::new(),
                Err(error) => error.to_string(),
            };
            assert!(message.contains(expected), "unexpected error: {message}");
        }

        let message = match Qlc::new(QlcConfig {
            max_latency_ms: Some(1_000),
            ..config(vec![candidate("missing-estimate")])
        }) {
            Ok(_) => String::new(),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("needs estimated_latency_ms"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn token_estimate_counts_messages_tools_and_output_budget() {
        let mut request = text_request(Some("qlc".to_string()), "hello");
        request.output.max_output_tokens = Some(100);
        let without_tools = estimated_tokens(&request);
        assert!(without_tools >= 100);

        request.tools = vec![Tool::Function(ToolDefinition {
            name: "bash".to_string(),
            description: Some("run a command".to_string()),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        })];
        assert!(estimated_tokens(&request) > without_tools);
    }
}
