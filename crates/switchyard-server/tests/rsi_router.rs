// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the RSI router and its step records.

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use libsy::{
    RsiRouter, RsiRouterConfig, RsiSelection, SubtaskCandidate, SubtaskCatalog, SubtaskSummary,
};
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{LlmClientError, LlmResponse, ModelId, Request, Response, text_response};
use switchyard_runner::{ModelCapabilities, Route, Runner, RuntimeModels};
use switchyard_server::{ServerState, build_switchyard_router};
use tower::ServiceExt as _;

const RECOGNIZER: &str = "model/recognizer";
const FLAGSHIP: &str = "model/flagship";
const CHEAP: &str = "openai/gpt-6-luna";
const MID: &str = "openai/gpt-6-sol";
const ROUTE: &str = "switchyard/rsi";

struct TestCatalog;

#[async_trait]
impl SubtaskCatalog for TestCatalog {
    async fn subtasks(&self) -> libsy::Result<Vec<SubtaskSummary>> {
        Ok(vec![SubtaskSummary {
            subtask_id: "edit-code".to_string(),
            subtask_name: "Edit code".to_string(),
            definition: "Change existing source files in a small, focused patch.".to_string(),
        }])
    }

    async fn candidates(&self, subtask_id: &str) -> libsy::Result<Vec<SubtaskCandidate>> {
        if subtask_id != "edit-code" {
            return Ok(Vec::new());
        }
        Ok(vec![
            SubtaskCandidate {
                model_id: ModelId::from(MID),
                cost: 1.2,
                latency_ms: 900.0,
                success_rate: 0.90,
                skill_content: None,
                skill_version: 0.0,
                is_valid: true,
            },
            SubtaskCandidate {
                model_id: ModelId::from(CHEAP),
                cost: 0.4,
                latency_ms: 1_200.0,
                success_rate: 0.95,
                skill_content: Some("Read the nearby tests first.".to_string()),
                skill_version: 1.0,
                is_valid: true,
            },
        ])
    }
}

struct FakeClient;

#[async_trait]
impl switchyard_protocol::RoutedLlmClient for FakeClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let model = request.llm_request.model.as_deref().unwrap_or_default();
        let text = if model == RECOGNIZER {
            r#"{"subtask_id":"edit-code","confidence":0.95}"#.to_string()
        } else {
            "done".to_string()
        };
        let mut aggregate = text_response(Some(model.to_string()), text);
        if model != RECOGNIZER {
            aggregate.usage = switchyard_protocol::Usage {
                input_tokens: Some(100),
                output_tokens: Some(20),
                ..Default::default()
            };
        }
        let messages = serde_json::to_value(&request.llm_request.messages).expect("messages");
        Ok(Response {
            llm_response: LlmResponse::Agg(aggregate),
            // The real client records the body it sent upstream. The stand-in
            // carries the same field so the record plumbing is exercised.
            metadata: Some(switchyard_protocol::Metadata {
                upstream_request: Some(Arc::new(
                    serde_json::json!({ "model": model, "messages": messages }),
                )),
                ..Default::default()
            }),
            upstream_headers: Default::default(),
        })
    }
}

fn app(
    trajectory_log: &std::path::Path,
    routing_log: &std::path::Path,
    recognizer_log: &std::path::Path,
) -> axum::Router {
    app_with_fixed_target(trajectory_log, routing_log, recognizer_log, None)
}

fn app_with_fixed_target(
    trajectory_log: &std::path::Path,
    routing_log: &std::path::Path,
    recognizer_log: &std::path::Path,
    fixed_target: Option<&str>,
) -> axum::Router {
    let algorithm = Arc::new(
        RsiRouter::new(RsiRouterConfig {
            recognizer: ModelId::from(RECOGNIZER),
            flagship: ModelId::from(FLAGSHIP),
            candidate_models: vec![ModelId::from(CHEAP), ModelId::from(MID)],
            recognizer_prompt: libsy::DEFAULT_RSI_RECOGNIZER_PROMPT.trim().to_string(),
            confidence_threshold: 0.7,
            min_success_rate: 0.0,
            history_steps: 3,
            selection: RsiSelection::LowestCost,
            max_output_tokens: 128,
            catalog: Arc::new(TestCatalog),
            fixed_target: fixed_target.map(ModelId::from),
            skill_injected: true,
        })
        .expect("algorithm"),
    );
    let route = Route::new(
        algorithm,
        ClientRouter::single(Arc::new(FakeClient)),
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::default(),
    );
    let runner = Runner::new(vec![(ModelId::from(ROUTE), route)]);
    let state = ServerState::from_runner(runner)
        .expect("state")
        .with_trajectory_log(trajectory_log)
        .expect("trajectory log")
        .with_routing_step_log(routing_log)
        .expect("routing log")
        .with_recognizer_log(recognizer_log)
        .expect("recognizer log");
    build_switchyard_router(state)
}

fn last_record(path: &std::path::Path) -> serde_json::Value {
    let records = std::fs::read_to_string(path).expect("log");
    let line = records
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .expect("record");
    serde_json::from_str(line).expect("json")
}

#[tokio::test]
async fn unresolved_or_missing_model_falls_back_to_the_rsi_route() {
    let dir = tempfile::tempdir().expect("temp dir");
    let app = app(
        &dir.path().join("rsi.jsonl"),
        &dir.path().join("routing.jsonl"),
        &dir.path().join("recognizer.jsonl"),
    );

    // Claude CLI sends its own model name, not the gateway's route ID.
    let unknown = app
        .clone()
        .oneshot(
            HttpRequest::post("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"claude-sonnet-4-5-20250929","max_tokens":64,"messages":[{"role":"user","content":"Fix the parser"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(unknown.status(), StatusCode::OK);
    assert_eq!(
        unknown
            .headers()
            .get("x-model-router-selected-model")
            .and_then(|value| value.to_str().ok()),
        Some(CHEAP)
    );
    let body = axum::body::to_bytes(unknown.into_body(), usize::MAX)
        .await
        .expect("body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(body["content"][0]["text"], "done");

    // A request with no model field also uses the default route.
    let missing = app
        .oneshot(
            HttpRequest::post("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"max_tokens":64,"messages":[{"role":"user","content":"Fix the parser"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(missing.status(), StatusCode::OK);
    assert_eq!(
        missing
            .headers()
            .get("x-model-router-selected-model")
            .and_then(|value| value.to_str().ok()),
        Some(CHEAP)
    );
}

#[tokio::test]
async fn rsi_route_selects_the_cheapest_candidate_and_records_the_step() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = dir.path().join("rsi.jsonl");
    let routing_log = dir.path().join("routing.jsonl");
    let recognizer_log = dir.path().join("recognizer.jsonl");
    let app = app(&log, &routing_log, &recognizer_log);

    let response = app
        .clone()
        .oneshot(
            HttpRequest::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-switchyard-session-id", "session-rsi")
                .body(Body::from(
                    r#"{"model":"switchyard/rsi","messages":[{"role":"user","content":"Fix the parser"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            HttpRequest::get("/v1/rsi/sessions/session-rsi/steps/1")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let step: serde_json::Value = serde_json::from_slice(&body).expect("json");

    assert_eq!(step["session_id"], "session-rsi");
    assert_eq!(step["step"], 1);
    assert_eq!(step["subtask_id"], "edit-code");
    assert_eq!(step["confidence"], 0.95);
    assert_eq!(step["decision"], "subtask");
    assert_eq!(step["skill_injected"], true);
    assert_eq!(step["selected_model"], CHEAP);
    assert_eq!(step["served_model"], CHEAP);
    assert_eq!(step["result"]["status"], "ok");
    // The record carries the model-facing request body, not the caller's body.
    assert_eq!(step["upstream_request"]["model"], CHEAP);
    assert!(step.get("raw_request").is_none());
    assert_eq!(step["usage"]["input_tokens"], 100);
    assert_eq!(step["usage"]["output_tokens"], 20);
    assert!(step["cost_usd"].as_f64().is_some_and(|cost| cost > 0.0));
    assert!(step["latency_ms"].as_f64().is_some());
    assert!(step["response"].to_string().contains("done"));

    let records = std::fs::read_to_string(&routing_log).expect("routing log");
    let record: serde_json::Value = serde_json::from_str(records.trim()).expect("record");
    assert_eq!(record["session_id"], "session-rsi");
    assert_eq!(record["step"], 1);
    assert_eq!(record["subtask_id"], "edit-code");
    assert_eq!(record["selected_model"], CHEAP);
    assert_eq!(record["confidence_threshold"], 0.7);
    assert_eq!(record["min_success_rate"], 0.0);
    assert_eq!(record["selection"], "lowest_cost");
    assert_eq!(record["candidates"], serde_json::json!([CHEAP, MID]));
    let pareto = record["pareto_candidates"]
        .as_array()
        .expect("pareto candidates");
    assert_eq!(pareto.len(), 2);
    let cheap = pareto
        .iter()
        .find(|candidate| candidate["model_id"] == CHEAP)
        .expect("cheap candidate");
    assert_eq!(cheap["cost"], 0.4);
    assert_eq!(cheap["latency_ms"], 1_200.0);
    assert_eq!(cheap["success_rate"], 0.95);
    assert_eq!(cheap["configured"], true);
    assert_eq!(cheap["above_min_success_rate"], true);
    assert_eq!(cheap["selected"], true);
    let mid = pareto
        .iter()
        .find(|candidate| candidate["model_id"] == MID)
        .expect("mid candidate");
    assert_eq!(mid["selected"], false);
    assert!(record["cost_usd"].as_f64().is_some_and(|cost| cost > 0.0));
    assert!(record["latency_ms"].as_f64().is_some());
    assert!(record.get("messages").is_none());
    assert!(record.get("response").is_none());

    let records = std::fs::read_to_string(&recognizer_log).expect("recognizer log");
    let record: serde_json::Value = serde_json::from_str(records.trim()).expect("record");
    assert_eq!(record["session_id"], "session-rsi");
    assert_eq!(record["step"], 1);
    assert_eq!(record["model"], RECOGNIZER);
    assert_eq!(record["subtask_id"], "edit-code");
    assert_eq!(record["confidence"], 0.95);
    assert!(
        record["prompt"]
            .as_str()
            .unwrap()
            .contains("Fix the parser")
    );
    assert!(record["prompt"].as_str().unwrap().contains("Edit code"));
    assert!(record["response"].as_str().unwrap().contains("edit-code"));

    // A request without a session ID still writes all three logs. The shared
    // request_id joins them.
    let response = app
        .clone()
        .oneshot(
            HttpRequest::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"switchyard/rsi","metadata":{"session_id":""},"messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);

    let execution = last_record(&log);
    let routing = last_record(&routing_log);
    let recognizer = last_record(&recognizer_log);
    assert!(execution.get("session_id").is_none());
    assert!(routing.get("session_id").is_none());
    assert!(recognizer.get("session_id").is_none());
    let request_id = execution["request_id"].as_str().expect("request id");
    assert!(!request_id.is_empty());
    assert_eq!(routing["request_id"], request_id);
    assert_eq!(recognizer["request_id"], request_id);
    assert_eq!(execution["step"], 1);
    assert_eq!(routing["step"], 1);
    assert_eq!(recognizer["step"], 1);

    // Claude CLI sends the session ID in the body instead of a header. The body
    // also carries a tool exchange, which the record's upstream body keeps verbatim.
    let response = app
        .oneshot(
            HttpRequest::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"switchyard/rsi","metadata":{"session_id":"claude-body-session"},"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_answer","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-1","content":"42"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let execution = last_record(&log);
    assert_eq!(execution["session_id"], "claude-body-session");
    // The answer call's model-facing body, not the caller's body.
    assert_eq!(execution["upstream_request"]["model"], CHEAP);
    assert!(execution.get("raw_request").is_none());
    assert_eq!(
        last_record(&routing_log)["session_id"],
        "claude-body-session"
    );
    assert_eq!(
        last_record(&recognizer_log)["session_id"],
        "claude-body-session"
    );
}

#[tokio::test]
async fn skill_reminder_stays_at_the_subtask_start_in_the_caller_format() {
    let dir = tempfile::tempdir().expect("temp dir");
    let execution_log = dir.path().join("rsi.jsonl");
    let app = app(
        &execution_log,
        &dir.path().join("routing.jsonl"),
        &dir.path().join("recognizer.jsonl"),
    );

    let response = app
        .clone()
        .oneshot(
            HttpRequest::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-switchyard-session-id", "session-skill")
                .body(Body::from(
                    r#"{"model":"switchyard/rsi","messages":[{"role":"user","content":"Fix the parser"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let first = last_record(&execution_log);
    let messages = first["upstream_request"]["messages"]
        .as_array()
        .expect("messages");
    assert_eq!(messages.last().expect("last message")["role"], "user");
    assert!(
        messages.last().expect("last message")["content"]
            .to_string()
            .contains("<skill_reminder>当前为子任务id：edit-code，Skill：Read the nearby tests first.</skill_reminder>")
    );

    let response = app
        .clone()
        .oneshot(
            HttpRequest::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-switchyard-session-id", "session-skill")
                .body(Body::from(
                    r#"{"model":"switchyard/rsi","messages":[{"role":"user","content":"Fix the parser"},{"role":"assistant","content":"done"},{"role":"user","content":"Continue"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let second = last_record(&execution_log);
    let messages = second["upstream_request"]["messages"]
        .as_array()
        .expect("messages");
    // The caller resends its own transcript, so the reminder goes back to
    // where the subtask started instead of opening a new turn at the end.
    assert_eq!(messages[1]["role"], "user");
    assert!(
        messages[1]["content"]
            .to_string()
            .contains("<skill_reminder>当前为子任务id：edit-code，Skill：Read the nearby tests first.</skill_reminder>")
    );
    assert_eq!(messages.last().expect("last message")["role"], "user");
    assert!(
        !messages.last().expect("last message")["content"]
            .to_string()
            .contains("<skill_reminder>")
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["content"].to_string().contains("<skill_reminder>"))
            .count(),
        1,
        "one reminder per request"
    );

    let response = app
        .oneshot(
            HttpRequest::post("/v1/messages")
                .header("content-type", "application/json")
                .header("x-switchyard-session-id", "session-anthropic")
                .body(Body::from(
                    r#"{"model":"switchyard/rsi","max_tokens":64,"messages":[{"role":"user","content":"Fix the parser"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let third = last_record(&execution_log);
    let messages = third["upstream_request"]["messages"]
        .as_array()
        .expect("messages");
    assert_eq!(messages.last().expect("last message")["role"], "system");
    assert!(
        messages.last().expect("last message")["content"]
            .to_string()
            .contains("<skill_reminder>当前为子任务id：edit-code，Skill：Read the nearby tests first.</skill_reminder>")
    );
}

#[tokio::test]
async fn fixed_target_routes_and_records_the_pin() {
    let dir = tempfile::tempdir().expect("temp dir");
    let execution_log = dir.path().join("rsi.jsonl");
    let routing_log = dir.path().join("routing.jsonl");
    let app = app_with_fixed_target(
        &execution_log,
        &routing_log,
        &dir.path().join("recognizer.jsonl"),
        Some(MID),
    );

    let response = app
        .oneshot(
            HttpRequest::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-switchyard-session-id", "session-fixed")
                .body(Body::from(
                    r#"{"model":"switchyard/rsi","messages":[{"role":"user","content":"Fix the parser"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-model-router-selected-model")
            .and_then(|value| value.to_str().ok()),
        Some(MID)
    );

    let record = last_record(&routing_log);
    assert_eq!(record["fixed_target"], MID);
    assert_eq!(record["selected_model"], MID);
    assert_eq!(record["decision"], "subtask");
    // The skill comes from the subtask candidates, not from the fixed target.
    assert_eq!(record["skill_injected"], true);
    assert_eq!(record["candidates"], serde_json::json!([MID]));
    assert!(record.get("selection").is_none());
    let pareto = record["pareto_candidates"]
        .as_array()
        .expect("pareto candidates");
    let cheap = pareto
        .iter()
        .find(|candidate| candidate["model_id"] == CHEAP)
        .expect("cheap candidate");
    assert_eq!(cheap["selected"], false);

    let execution = last_record(&execution_log);
    assert_eq!(execution["selected_model"], MID);
}
