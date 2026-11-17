// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the RSI router and its step records.

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use libsy::{
    RsiRouter, RsiRouterConfig, RsiSelection, SkillTemplate, SubtaskCandidate, SubtaskCatalog,
    SubtaskEntry,
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
    async fn entries(&self) -> libsy::Result<Vec<SubtaskEntry>> {
        Ok(vec![SubtaskEntry {
            subtask_id: "edit-code".to_string(),
            subtask_name: "Edit code".to_string(),
            definition: "Change existing source files in a small, focused patch.".to_string(),
            candidates: vec![
                SubtaskCandidate {
                    model_id: ModelId::from(MID),
                    cost_per_million_tokens: 1.2,
                },
                SubtaskCandidate {
                    model_id: ModelId::from(CHEAP),
                    cost_per_million_tokens: 0.4,
                },
            ],
            skill: Some(SkillTemplate {
                version: "v-test".to_string(),
                template: "Read the nearby tests first.".to_string(),
            }),
        }])
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
        Ok(Response {
            llm_response: LlmResponse::Agg(aggregate),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

fn app(
    trajectory_log: &std::path::Path,
    routing_log: &std::path::Path,
    recognizer_log: &std::path::Path,
) -> axum::Router {
    let algorithm = Arc::new(
        RsiRouter::new(RsiRouterConfig {
            recognizer: ModelId::from(RECOGNIZER),
            flagship: ModelId::from(FLAGSHIP),
            candidate_models: vec![ModelId::from(CHEAP), ModelId::from(MID)],
            recognizer_prompt: libsy::DEFAULT_RSI_RECOGNIZER_PROMPT.trim().to_string(),
            confidence_threshold: 0.7,
            history_steps: 3,
            selection: RsiSelection::LowestCost,
            max_output_tokens: 128,
            catalog: Arc::new(TestCatalog),
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
    assert_eq!(step["skill_version"], "v-test");
    assert_eq!(step["selected_model"], CHEAP);
    assert_eq!(step["served_model"], CHEAP);
    assert_eq!(step["result"]["status"], "ok");
    assert_eq!(step["messages"][0]["content"][0]["text"], "Fix the parser");
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

    fn last_record(path: &std::path::Path) -> serde_json::Value {
        let records = std::fs::read_to_string(path).expect("log");
        let line = records
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .expect("record");
        serde_json::from_str(line).expect("json")
    }

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

    // Claude CLI sends the session ID in the body instead of a header.
    let response = app
        .oneshot(
            HttpRequest::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"switchyard/rsi","metadata":{"session_id":"claude-body-session"},"messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(last_record(&log)["session_id"], "claude-body-session");
    assert_eq!(
        last_record(&routing_log)["session_id"],
        "claude-body-session"
    );
    assert_eq!(
        last_record(&recognizer_log)["session_id"],
        "claude-body-session"
    );
}
