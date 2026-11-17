// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public factory contracts for configured routes without provider clients.

use std::sync::Arc;

use switchyard_protocol::{Category, Metadata, ModelId, Request, text_request};
use switchyard_runner::{AlgorithmFactory, Runner};

const MISSING_KEY: &str = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET";
const DEPLOYMENT: &str = r#"
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "https://example.test/v1"
api_key_env = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
[targets]
fast = { id = "upstream/fast", llm_client = "answers" }
capable = { id = "upstream/capable", llm_client = "answers" }
[decision_clients.judge]
format = "openai_decisions"
endpoint = "https://example.test/v1/decisions"
api_key_env = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
timeout_ms = 2000
[decision_targets.judge]
id = "upstream/judge"
decision_client = "judge"
[routes.parent]
id = "public/main"
type = "passthrough"
target = "fast"
[routes.parent.subagents]
type = "passthrough"
target = "capable"
[routes.classifier]
id = "public/classify"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "capable"
weak_target = "fast"
[routes.classifier.decision]
cutoff = 0.7
candidates = { a = "capable", b = "fast" }
evidence = { description = "Compare both targets." }
"#;

#[test]
fn full_deployment_builds_without_provider_credentials() {
    assert!(std::env::var_os(MISSING_KEY).is_none());
    let factory = AlgorithmFactory::from_toml(DEPLOYMENT).expect("deployment should parse");
    let algorithms = factory.build().expect("algorithms need no provider keys");
    assert_eq!(
        algorithms.keys().map(String::as_str).collect::<Vec<_>>(),
        ["public/classify", "public/main"]
    );
    let (algorithm, models) = &algorithms["public/classify"];
    assert_eq!(algorithm.name(), "llm_task_classifier");
    assert_eq!(
        models.models_for(&Category::Judge),
        [ModelId::from("upstream/judge")]
    );
    assert_eq!(
        models.models_for(&Category::Capable),
        [ModelId::from("upstream/capable")]
    );
    assert_eq!(
        models.models_for(&Category::Efficient),
        [ModelId::from("upstream/fast")]
    );
    assert_eq!(
        models.models_for(&Category::Any),
        [
            ModelId::from("upstream/fast"),
            ModelId::from("upstream/capable")
        ]
    );

    let error = Runner::from_toml(DEPLOYMENT)
        .err()
        .expect("the native runner still requires configured credentials");
    assert!(error.to_string().contains(MISSING_KEY), "{error}");
}

#[tokio::test]
async fn configured_parent_and_subagent_targets_stay_separate() {
    let mut algorithms = AlgorithmFactory::from_toml(DEPLOYMENT)
        .expect("deployment should parse")
        .build()
        .expect("algorithms should build");
    let (algorithm, models) = algorithms.remove("public/main").expect("public route id");
    assert_eq!(
        models.models_for(&Category::Any),
        [ModelId::from("upstream/fast")]
    );
    assert_eq!(
        models.subagent_models_for(&Category::Any),
        [ModelId::from("upstream/capable")]
    );
    let models = Arc::new(models);
    for (delegated, selected) in [(false, "upstream/fast"), (true, "upstream/capable")] {
        let request = Request {
            llm_request: text_request(Some("public/main".to_string()), "hello"),
            metadata: delegated.then(|| Metadata {
                session_id: Some("task".to_string()),
                agent_id: Some("child".to_string()),
                is_subagent: true,
                is_delegated_work: true,
                ..Metadata::default()
            }),
            ..Request::default()
        };
        let outcome = libsy::drive(
            Arc::clone(&algorithm),
            request,
            Arc::clone(&models),
            |_| async { panic!("passthrough must not dispatch provider calls") },
        )
        .await
        .expect("configured route should resolve");
        assert_eq!(outcome.selected_model_ids, [ModelId::from(selected)]);
        assert!(outcome.response.is_none());
    }
}

const RSI_DEPLOYMENT: &str = r#"
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "https://example.test/v1"
api_key_env = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
[targets]
recognizer = { id = "upstream/recognizer", llm_client = "answers" }
flagship = { id = "upstream/flagship", llm_client = "answers" }
cheap = { id = "upstream/cheap", llm_client = "answers" }
mid = { id = "upstream/mid", llm_client = "answers" }
[routes.rsi]
id = "public/rsi"
type = "rsi_router"
recognizer_target = "recognizer"
flagship_target = "flagship"
candidate_targets = ["cheap", "mid"]
redis_url_env = "SWITCHYARD_RSI_TEST_REDIS_URL"
"#;

const RSI_REDIS_ENV: &str = "SWITCHYARD_RSI_TEST_REDIS_URL";
static RSI_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_rsi_env() -> std::sync::MutexGuard<'static, ()> {
    RSI_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn rsi_deployment_builds_from_configured_targets() {
    let _env = lock_rsi_env();
    // The test owns this variable name, so no other test reads it.
    unsafe {
        std::env::set_var(RSI_REDIS_ENV, "redis://127.0.0.1:6379");
    }
    let factory = AlgorithmFactory::from_toml(RSI_DEPLOYMENT).expect("deployment should parse");
    let algorithms = factory.build().expect("rsi route should build");
    let (algorithm, models) = &algorithms["public/rsi"];
    assert_eq!(algorithm.name(), "rsi_router");
    assert_eq!(
        models.models_for(&Category::Judge),
        [ModelId::from("upstream/recognizer")]
    );
    assert_eq!(
        models.models_for(&Category::Capable),
        [ModelId::from("upstream/flagship")]
    );
    assert_eq!(
        models.models_for(&Category::Any),
        [
            ModelId::from("upstream/flagship"),
            ModelId::from("upstream/cheap"),
            ModelId::from("upstream/mid"),
        ]
    );

    let routes = switchyard_runner::route_targets(RSI_DEPLOYMENT).expect("route targets");
    let route = routes.first().expect("route");
    assert_eq!(route.kind, "rsi_router");
    assert_eq!(
        route
            .answer_targets
            .iter()
            .map(|target| target.name.as_str())
            .collect::<Vec<_>>(),
        ["flagship", "cheap", "mid"]
    );
    assert_eq!(
        route
            .judge_targets
            .iter()
            .map(|target| target.name.as_str())
            .collect::<Vec<_>>(),
        ["recognizer"]
    );
    unsafe {
        std::env::remove_var(RSI_REDIS_ENV);
    }
}

#[test]
fn rsi_deployment_builds_without_redis_url() {
    let _env = lock_rsi_env();
    unsafe {
        std::env::remove_var(RSI_REDIS_ENV);
    }
    let factory = AlgorithmFactory::from_toml(RSI_DEPLOYMENT).expect("deployment should parse");
    let algorithms = factory
        .build()
        .expect("rsi route should build without Redis");
    assert_eq!(algorithms["public/rsi"].0.name(), "rsi_router");
}

#[test]
fn invalid_rsi_configuration_is_rejected() {
    let _env = lock_rsi_env();
    unsafe {
        std::env::set_var(RSI_REDIS_ENV, "redis://127.0.0.1:6379");
    }
    for (source, expected) in [
        (
            RSI_DEPLOYMENT.replace(
                "candidate_targets = [\"cheap\", \"mid\"]",
                "candidate_targets = []",
            ),
            "candidate_targets must not be empty",
        ),
        (
            RSI_DEPLOYMENT.replace(
                "candidate_targets = [\"cheap\", \"mid\"]",
                "candidate_targets = [\"cheap\", \"cheap\"]",
            ),
            "candidate target cheap is repeated",
        ),
        (
            RSI_DEPLOYMENT.replace(
                "recognizer_target = \"recognizer\"",
                "recognizer_target = \"missing\"",
            ),
            "unknown target missing",
        ),
    ] {
        let error = AlgorithmFactory::from_toml(&source)
            .and_then(|factory| factory.build())
            .err()
            .expect("invalid rsi deployment should fail");
        assert!(error.to_string().contains(expected), "{error}");
    }
    unsafe {
        std::env::remove_var(RSI_REDIS_ENV);
    }
}

#[test]
fn invalid_routing_configuration_is_rejected() {
    for (source, expected) in [
        (
            DEPLOYMENT.replacen("target = \"fast\"", "target = \"missing\"", 1),
            "unknown target missing",
        ),
        (
            DEPLOYMENT.replace("id = \"public/classify\"", "id = \"public/main\""),
            "route ids must be unique",
        ),
        (
            DEPLOYMENT.replace(
                "type = \"passthrough\"",
                "type = \"passthrough\"\nbogus = true",
            ),
            "bogus",
        ),
        (
            DEPLOYMENT.replace(
                "weak_target = \"fast\"\n",
                "weak_target = \"fast\"\nunmatched_steps = 2\n",
            ),
            "decision cannot use LLM judge settings",
        ),
    ] {
        let error = AlgorithmFactory::from_toml(&source)
            .and_then(|factory| factory.build())
            .err()
            .expect("invalid deployment should fail");
        assert!(error.to_string().contains(expected), "{error}");
    }
}
