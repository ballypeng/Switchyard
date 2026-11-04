// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quality, latency, and cost (QLC) routing.
//!
//! Candidates that fail a hard requirement are removed first. The rest are scored
//! with configured weights. The best candidate serves the request and the others
//! remain as ordered fallbacks.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
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

/// Routes by filtering candidates, then ranking them with a configured score.
pub struct Qlc {
    config: QlcConfig,
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
        Ok(Self { config })
    }

    fn score(&self, candidate: &QlcCandidate) -> f64 {
        self.config.weight_quality * candidate.quality_score
            + self.config.weight_latency * candidate.latency_score
            + self.config.weight_cost * candidate.cost_score
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
        let mut ranked: Vec<&QlcCandidate> = self
            .config
            .candidates
            .iter()
            .filter(|candidate| self.passes(candidate, &profile))
            .collect();
        ranked.sort_by(|left, right| self.score(right).total_cmp(&self.score(left)));

        let Some(selected) = ranked.first() else {
            return Err(LibsyError::AlgorithmError {
                message: "qlc: no candidate passed the hard filters".to_string(),
            });
        };
        let candidates: Vec<_> = self
            .config
            .candidates
            .iter()
            .map(|candidate| {
                serde_json::json!({
                    "target": candidate.target.as_str(),
                    "score": self.score(candidate),
                    "quality_score": candidate.quality_score,
                    "latency_score": candidate.latency_score,
                    "cost_score": candidate.cost_score,
                    "eligible": self.passes(candidate, &profile),
                })
            })
            .collect();
        driver.set_evidence(serde_json::json!({
            "source": "qlc",
            "selected": selected.target.as_str(),
            "score": self.score(selected),
            "candidates": candidates,
        }));
        let fallbacks = ranked
            .iter()
            .skip(1)
            .map(|candidate| candidate.target.clone())
            .collect();
        Ok(RoutingOutcome::route_to(
            selected.target.clone(),
            fallbacks,
            request,
        ))
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
        let mut stream = Arc::new(qlc).run_stream(
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
