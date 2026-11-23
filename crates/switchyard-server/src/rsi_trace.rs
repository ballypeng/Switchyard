// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable RSI session and step records.
//!
//! Each line holds the full agent context at one step and the routing data for that
//! step. A reader selects records by session ID and step number.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use humantime::format_rfc3339_millis;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use switchyard_costs::CostSource;
use switchyard_protocol::{Message, Role, Usage};

use crate::{ServerError, ServerResult};

/// Serialized record schema version.
pub(crate) const SCHEMA_VERSION: u16 = 1;

/// One step in an RSI session.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TrajectoryStep {
    /// Serialized record schema version.
    pub schema_version: u16,
    /// Request identifier shared by the three RSI logs.
    pub request_id: String,
    /// Session that owns this step, when the request supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// One-based step number within the session.
    pub step: u64,
    /// RFC3339 UTC timestamp.
    pub ts: String,
    /// Public route model the client requested.
    pub route_id: String,
    /// Algorithm that routed the step.
    pub algorithm: String,
    /// Recognized subtask, when one applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtask_id: Option<String>,
    /// Recognizer confidence, when one applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Router decision: `subtask`, `flagship`, or `fixed_target` when a
    /// configured fixed target served a step that had no subtask route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    /// Why the router chose the flagship model, when it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Whether the selected candidate's skill reminder was appended to the step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_injected: Option<bool>,
    /// Model the router selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_model: Option<String>,
    /// Model that actually served the step, after any fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model: Option<String>,
    /// Outcome of this step.
    pub result: StepResult,
    /// Token usage reported for the step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<StepUsage>,
    /// Estimated or provider-reported cost in USD.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    /// Where the cost came from: `estimated` or `provider_reported`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_source: Option<String>,
    /// End-to-end step latency in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    /// The request body the client sent to the execution model for this step,
    /// after model resolution, format conversion, and backend transforms. This is
    /// what the model actually received, so it is the source for offline
    /// analysis. Absent when the step failed before a model call succeeded, or in
    /// records written before this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_request: Option<serde_json::Value>,
    /// Normalized response returned by the execution model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
}

/// The result of one routed step.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct StepResult {
    /// `ok` or `error`.
    pub status: String,
    /// Failure message when the step failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl StepResult {
    pub(crate) fn ok() -> Self {
        Self {
            status: "ok".to_string(),
            error: None,
        }
    }

    pub(crate) fn error(error: impl Into<String>) -> Self {
        Self {
            status: "error".to_string(),
            error: Some(error.into()),
        }
    }
}

/// Token usage recorded for one step.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct StepUsage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

impl From<&Usage> for StepUsage {
    fn from(usage: &Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens.unwrap_or(0),
            cached_input_tokens: usage.cached_input_tokens().unwrap_or(0),
            cache_creation_input_tokens: usage.cache_creation_input_tokens().unwrap_or(0),
            output_tokens: usage.output_tokens.unwrap_or(0),
            reasoning_tokens: usage.reasoning_tokens.unwrap_or(0),
        }
    }
}

pub(crate) fn cost_source(source: CostSource) -> &'static str {
    match source {
        CostSource::Estimated => "estimated",
        CostSource::ProviderReported => "provider_reported",
    }
}

/// A step record without its potentially large context.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct StepSummary {
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub step: u64,
    pub ts: String,
    pub route_id: String,
    pub algorithm: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtask_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_injected: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model: Option<String>,
    pub result: StepResult,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}

impl From<&TrajectoryStep> for StepSummary {
    fn from(step: &TrajectoryStep) -> Self {
        Self {
            request_id: step.request_id.clone(),
            session_id: step.session_id.clone(),
            step: step.step,
            ts: step.ts.clone(),
            route_id: step.route_id.clone(),
            algorithm: step.algorithm.clone(),
            subtask_id: step.subtask_id.clone(),
            confidence: step.confidence,
            decision: step.decision.clone(),
            reason: step.reason.clone(),
            skill_injected: step.skill_injected,
            result: step.result.clone(),
            selected_model: step.selected_model.clone(),
            served_model: step.served_model.clone(),
            cost_usd: step.cost_usd,
            latency_ms: step.latency_ms,
        }
    }
}

/// Append-only writer for one RSI trajectory JSONL file.
pub(crate) struct TrajectoryLog {
    file: fs::File,
    path: PathBuf,
}

impl TrajectoryLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| trajectory_error(&path, error))?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| trajectory_error(&path, error))?;
        Ok(Self { file, path })
    }

    pub(crate) fn append(&mut self, step: &TrajectoryStep) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(step).map_err(std::io::Error::other)?;
        line.push(b'\n');
        self.file.write_all(&line)
    }

    pub(crate) fn read_steps(&self, session_id: &str) -> std::io::Result<Vec<TrajectoryStep>> {
        read_steps(&self.path, session_id)
    }

    pub(crate) fn read_step(
        &self,
        session_id: &str,
        step: u64,
    ) -> std::io::Result<Option<TrajectoryStep>> {
        read_step(&self.path, session_id, step)
    }
}

/// Shared append-only trajectory log.
#[derive(Clone)]
pub(crate) struct SharedTrajectoryLog {
    writer: std::sync::Arc<Mutex<TrajectoryLog>>,
}

impl SharedTrajectoryLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        Ok(Self {
            writer: std::sync::Arc::new(Mutex::new(TrajectoryLog::new(path)?)),
        })
    }

    pub(crate) fn append(&self, step: &TrajectoryStep) {
        if let Err(error) = self.writer.lock().append(step) {
            tracing::warn!(%error, "RSI trajectory append failed");
        }
    }

    pub(crate) fn read_steps(&self, session_id: &str) -> std::io::Result<Vec<TrajectoryStep>> {
        self.writer.lock().read_steps(session_id)
    }

    pub(crate) fn read_step(
        &self,
        session_id: &str,
        step: u64,
    ) -> std::io::Result<Option<TrajectoryStep>> {
        self.writer.lock().read_step(session_id, step)
    }
}

/// One recognizer interaction, written for manual review.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RecognizerRecord {
    pub schema_version: u16,
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub step: u64,
    pub ts: String,
    pub route_id: String,
    pub model: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtask_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Builds a recognizer record from the algorithm's routing evidence.
pub(crate) fn recognizer_record(
    request_id: &str,
    session_id: Option<&str>,
    step: u64,
    route_id: &str,
    evidence: Option<&serde_json::Value>,
) -> Option<RecognizerRecord> {
    let recognizer = evidence?.get("recognizer")?.as_object()?;
    let field = |name: &str| {
        recognizer
            .get(name)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    };
    Some(RecognizerRecord {
        schema_version: SCHEMA_VERSION,
        request_id: request_id.to_string(),
        session_id: session_id.map(str::to_string),
        step,
        ts: timestamp(),
        route_id: route_id.to_string(),
        model: field("model").unwrap_or_default(),
        prompt: field("prompt").unwrap_or_default(),
        response: field("response"),
        subtask_id: field("subtask_id"),
        confidence: recognizer
            .get("confidence")
            .and_then(serde_json::Value::as_f64),
        latency_ms: recognizer
            .get("latency_ms")
            .and_then(serde_json::Value::as_f64),
        error: field("error"),
    })
}

/// One Pareto candidate as it entered selection.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RoutingCandidate {
    pub model_id: String,
    pub cost: f64,
    pub latency_ms: f64,
    pub success_rate: f64,
    pub configured: bool,
    pub above_min_success_rate: bool,
    pub selected: bool,
}

/// One routing decision, written without the step's message context.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RoutingStep {
    pub schema_version: u16,
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub step: u64,
    pub ts: String,
    pub route_id: String,
    pub algorithm: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtask_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_success_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pareto_candidates: Option<Vec<RoutingCandidate>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_injected: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model: Option<String>,
    pub result: StepResult,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<StepUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}

/// Append-only writer for routing decisions.
pub(crate) struct RoutingStepLog {
    file: fs::File,
}

impl RoutingStepLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| trajectory_error(&path, error))?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| trajectory_error(&path, error))?;
        Ok(Self { file })
    }

    pub(crate) fn append(&mut self, record: &RoutingStep) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(record).map_err(std::io::Error::other)?;
        line.push(b'\n');
        self.file.write_all(&line)
    }
}

/// Shared append-only routing decision log.
#[derive(Clone)]
pub(crate) struct SharedRoutingStepLog {
    writer: std::sync::Arc<Mutex<RoutingStepLog>>,
}

impl SharedRoutingStepLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        Ok(Self {
            writer: std::sync::Arc::new(Mutex::new(RoutingStepLog::new(path)?)),
        })
    }

    pub(crate) fn append(&self, record: &RoutingStep) {
        if let Err(error) = self.writer.lock().append(record) {
            tracing::warn!(%error, "RSI routing log append failed");
        }
    }
}

/// Append-only writer for recognizer interactions.
pub(crate) struct RecognizerLog {
    file: fs::File,
}

impl RecognizerLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| trajectory_error(&path, error))?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| trajectory_error(&path, error))?;
        Ok(Self { file })
    }

    pub(crate) fn append(&mut self, record: &RecognizerRecord) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(record).map_err(std::io::Error::other)?;
        line.push(b'\n');
        self.file.write_all(&line)
    }
}

/// Shared append-only recognizer log.
#[derive(Clone)]
pub(crate) struct SharedRecognizerLog {
    writer: std::sync::Arc<Mutex<RecognizerLog>>,
}

impl SharedRecognizerLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        Ok(Self {
            writer: std::sync::Arc::new(Mutex::new(RecognizerLog::new(path)?)),
        })
    }

    pub(crate) fn append(&self, record: &RecognizerRecord) {
        if let Err(error) = self.writer.lock().append(record) {
            tracing::warn!(%error, "RSI recognizer log append failed");
        }
    }
}

/// Returns the one-based step number of the request being routed.
///
/// Each completed step leaves one assistant message in the conversation history.
pub(crate) fn step_number(messages: &[Message]) -> u64 {
    let completed = messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count();
    u64::try_from(completed)
        .unwrap_or(u64::MAX)
        .saturating_add(1)
}

pub(crate) fn timestamp() -> String {
    format_rfc3339_millis(SystemTime::now()).to_string()
}

/// Reads every step for `session_id`, in file order.
pub(crate) fn read_steps(path: &Path, session_id: &str) -> std::io::Result<Vec<TrajectoryStep>> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut steps = Vec::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        if !line.ends_with(b"\n") {
            break;
        }
        let Ok(step) = serde_json::from_slice::<TrajectoryStep>(&line) else {
            continue;
        };
        if step.session_id.as_deref() == Some(session_id) {
            steps.push(step);
        }
    }
    Ok(steps)
}

/// Reads one step for `session_id`.
pub(crate) fn read_step(
    path: &Path,
    session_id: &str,
    step: u64,
) -> std::io::Result<Option<TrajectoryStep>> {
    Ok(read_steps(path, session_id)?
        .into_iter()
        .find(|record| record.step == step))
}

fn trajectory_error(path: &Path, error: std::io::Error) -> ServerError {
    ServerError::new(format!(
        "failed to initialize RSI trajectory log {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::Message;

    fn step(session: &str, number: u64) -> TrajectoryStep {
        TrajectoryStep {
            schema_version: SCHEMA_VERSION,
            request_id: format!("request-{number}"),
            session_id: Some(session.to_string()),
            step: number,
            ts: timestamp(),
            route_id: "switchyard/rsi".to_string(),
            algorithm: "rsi_router".to_string(),
            subtask_id: Some("edit-code".to_string()),
            confidence: Some(0.9),
            decision: Some("subtask".to_string()),
            reason: None,
            skill_injected: Some(true),
            selected_model: Some("model/cheap".to_string()),
            served_model: Some("model/cheap".to_string()),
            result: StepResult::ok(),
            usage: Some(StepUsage {
                input_tokens: 10,
                output_tokens: 2,
                ..StepUsage::default()
            }),
            cost_usd: Some(0.0001),
            cost_source: Some("estimated".to_string()),
            latency_ms: Some(12.5),
            upstream_request: Some(serde_json::json!({
                "model": "model/cheap",
                "messages": [{"role": "user", "content": "Fix the parser"}]
            })),
            response: Some(serde_json::json!({
                "outputs": [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}]
            })),
        }
    }

    #[test]
    fn writes_and_reads_steps_by_session_and_number() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("rsi.jsonl");
        let mut log = TrajectoryLog::new(&path).expect("log");
        log.append(&step("session-a", 1)).expect("append");
        log.append(&step("session-b", 1)).expect("append");
        log.append(&step("session-a", 2)).expect("append");

        let steps = log.read_steps("session-a").expect("read");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].step, 1);
        assert_eq!(steps[1].step, 2);
        assert_eq!(
            steps[1]
                .upstream_request
                .as_ref()
                .expect("upstream request")["messages"][0]["content"],
            "Fix the parser"
        );
        assert_eq!(
            log.read_step("session-a", 2)
                .expect("read")
                .expect("step")
                .subtask_id
                .as_deref(),
            Some("edit-code")
        );
        assert!(log.read_step("session-a", 9).expect("read").is_none());
    }

    #[test]
    fn step_number_counts_completed_assistant_turns() {
        assert_eq!(step_number(&[]), 1);
        assert_eq!(
            step_number(&[
                Message::text(Role::User, "start"),
                Message::text(Role::Assistant, "one"),
                Message::text(Role::User, "next"),
            ]),
            2
        );
        assert_eq!(
            step_number(&[
                Message::text(Role::Assistant, "one"),
                Message::text(Role::Assistant, "two"),
            ]),
            3
        );
    }

    #[test]
    fn summaries_omit_the_context() {
        let summary = StepSummary::from(&step("session-a", 1));
        let json = serde_json::to_value(summary).expect("json");
        assert!(json.get("upstream_request").is_none());
        assert_eq!(json["subtask_id"], "edit-code");
        assert_eq!(json["cost_usd"], 0.0001);
        assert_eq!(json["latency_ms"], 12.5);
    }

    #[test]
    fn writes_recognizer_interactions() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("recognizer.jsonl");
        let mut log = RecognizerLog::new(&path).expect("log");
        log.append(&RecognizerRecord {
            schema_version: SCHEMA_VERSION,
            request_id: "request-1".to_string(),
            session_id: Some("session-a".to_string()),
            step: 1,
            ts: timestamp(),
            route_id: "switchyard/rsi".to_string(),
            model: "model/recognizer".to_string(),
            prompt: "Available subtasks: edit-code".to_string(),
            response: Some(r#"{"subtask_id":"edit-code","confidence":0.9}"#.to_string()),
            subtask_id: Some("edit-code".to_string()),
            confidence: Some(0.9),
            latency_ms: Some(8.0),
            error: None,
        })
        .expect("append");

        let line = fs::read_to_string(&path).expect("read");
        let record: serde_json::Value = serde_json::from_str(line.trim()).expect("json");
        assert_eq!(record["session_id"], "session-a");
        assert_eq!(record["model"], "model/recognizer");
        assert_eq!(record["subtask_id"], "edit-code");
        assert!(record["prompt"].as_str().unwrap().contains("edit-code"));
    }

    #[test]
    fn writes_routing_decisions() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        let mut log = RoutingStepLog::new(&path).expect("log");
        let step = step("session-a", 1);
        log.append(&RoutingStep {
            schema_version: SCHEMA_VERSION,
            request_id: step.request_id,
            session_id: step.session_id,
            step: step.step,
            ts: step.ts,
            route_id: step.route_id,
            algorithm: step.algorithm,
            decision: step.decision,
            reason: step.reason,
            subtask_id: step.subtask_id,
            confidence: step.confidence,
            confidence_threshold: Some(0.7),
            min_success_rate: Some(0.6),
            fixed_target: None,
            selection: Some("lowest_cost".to_string()),
            candidates: Some(vec!["model/cheap".to_string()]),
            pareto_candidates: Some(vec![RoutingCandidate {
                model_id: "model/cheap".to_string(),
                cost: 0.4,
                latency_ms: 900.0,
                success_rate: 0.8,
                configured: true,
                above_min_success_rate: true,
                selected: true,
            }]),
            skill_injected: step.skill_injected,
            selected_model: step.selected_model,
            served_model: step.served_model,
            result: step.result,
            usage: step.usage,
            cost_usd: step.cost_usd,
            cost_source: step.cost_source,
            latency_ms: step.latency_ms,
        })
        .expect("append");

        let line = fs::read_to_string(&path).expect("read");
        let record: serde_json::Value = serde_json::from_str(line.trim()).expect("json");
        assert_eq!(record["session_id"], "session-a");
        assert_eq!(record["step"], 1);
        assert_eq!(record["subtask_id"], "edit-code");
        assert_eq!(record["selected_model"], "model/cheap");
        assert_eq!(record["cost_usd"], 0.0001);
        assert_eq!(record["min_success_rate"], 0.6);
        assert_eq!(record["selection"], "lowest_cost");
        assert_eq!(record["candidates"], serde_json::json!(["model/cheap"]));
        assert_eq!(record["pareto_candidates"][0]["model_id"], "model/cheap");
        assert_eq!(record["pareto_candidates"][0]["success_rate"], 0.8);
        assert!(record.get("messages").is_none());
    }
}
