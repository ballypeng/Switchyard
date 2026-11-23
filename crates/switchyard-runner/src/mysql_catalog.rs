// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MySQL-backed subtask catalog for the RSI router.
//!
//! The `subtask` table holds what the recognizer reads. The `candidate_model` table
//! holds one subtask's Pareto candidates together with its skill, and is read after
//! recognition. Rows that cannot produce a usable route are skipped with a warning.

use std::time::Duration;

use async_trait::async_trait;
use libsy::{LibsyError, Result, SubtaskCandidate, SubtaskCatalog, SubtaskSummary};
use mysql_async::prelude::Queryable;
use mysql_async::{Opts, Pool, Row};
use switchyard_protocol::ModelId;
use tracing::warn;

const SUBTASK_SQL: &str = "SELECT subtask_id, subtask_name, definition FROM subtask";
// Column names vary between dictionaries (`latency`/`latency_ms`,
// `success`/`success_rate`), so the row is read by name rather than position.
const CANDIDATE_SQL: &str = "SELECT * FROM candidate_model WHERE subtask_id = ?";

/// A catalog lookup failure.
#[derive(Debug, thiserror::Error)]
pub(crate) enum MySqlCatalogError {
    #[error("mysql URL is invalid: {0}")]
    Url(String),
    #[error("timeout must be greater than zero")]
    Timeout,
}

/// Reads subtask summaries and one subtask's candidates from MySQL.
pub(crate) struct MySqlSubtaskCatalog {
    pool: Pool,
    timeout: Duration,
}

impl MySqlSubtaskCatalog {
    /// Builds a catalog client. No connection is opened until the first read.
    pub(crate) fn new(
        url: &str,
        timeout: Duration,
    ) -> std::result::Result<Self, MySqlCatalogError> {
        let opts =
            Opts::from_url(url).map_err(|error| MySqlCatalogError::Url(error.to_string()))?;
        if timeout.is_zero() {
            return Err(MySqlCatalogError::Timeout);
        }
        Ok(Self {
            pool: Pool::new(opts),
            timeout,
        })
    }

    async fn read_subtasks(&self) -> std::result::Result<Vec<SubtaskSummary>, mysql_async::Error> {
        let mut conn = self.pool.get_conn().await?;
        let rows: Vec<Row> = conn.query(SUBTASK_SQL).await?;
        Ok(rows
            .into_iter()
            .filter_map(|row| match subtask_from_row(&row) {
                Ok(summary) => Some(summary),
                Err(reason) => {
                    warn!(%reason, "skipping an unusable subtask row");
                    None
                }
            })
            .collect())
    }

    async fn read_candidates(
        &self,
        subtask_id: &str,
    ) -> std::result::Result<Vec<SubtaskCandidate>, mysql_async::Error> {
        let mut conn = self.pool.get_conn().await?;
        let rows: Vec<Row> = conn.exec(CANDIDATE_SQL, (subtask_id,)).await?;
        Ok(rows
            .into_iter()
            .filter_map(|row| match candidate_from_row(subtask_id, &row) {
                Ok(candidate) => Some(candidate),
                Err(reason) => {
                    warn!(subtask_id, %reason, "skipping an unusable candidate row");
                    None
                }
            })
            .collect())
    }
}

#[async_trait]
impl SubtaskCatalog for MySqlSubtaskCatalog {
    async fn subtasks(&self) -> Result<Vec<SubtaskSummary>> {
        match tokio::time::timeout(self.timeout, self.read_subtasks()).await {
            Ok(Ok(subtasks)) => Ok(subtasks),
            Ok(Err(error)) => Err(LibsyError::external("subtask catalog read", error)),
            Err(_) => Err(LibsyError::external(
                "subtask catalog read",
                catalog_timeout(),
            )),
        }
    }

    async fn candidates(&self, subtask_id: &str) -> Result<Vec<SubtaskCandidate>> {
        match tokio::time::timeout(self.timeout, self.read_candidates(subtask_id)).await {
            Ok(Ok(candidates)) => Ok(candidates),
            Ok(Err(error)) => Err(LibsyError::external("subtask catalog read", error)),
            Err(_) => Err(LibsyError::external(
                "subtask catalog read",
                catalog_timeout(),
            )),
        }
    }
}

/// A catalog for a route whose MySQL settings are unavailable.
///
/// The route builds and serves requests. Every lookup fails, so the router
/// uses the flagship model until MySQL is configured.
pub(crate) struct UnavailableSubtaskCatalog {
    reason: String,
}

impl UnavailableSubtaskCatalog {
    pub(crate) fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl SubtaskCatalog for UnavailableSubtaskCatalog {
    async fn subtasks(&self) -> Result<Vec<SubtaskSummary>> {
        Err(LibsyError::AlgorithmError {
            message: self.reason.clone(),
        })
    }

    async fn candidates(&self, _subtask_id: &str) -> Result<Vec<SubtaskCandidate>> {
        Err(LibsyError::AlgorithmError {
            message: self.reason.clone(),
        })
    }
}

fn catalog_timeout() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::TimedOut, "mysql query timed out")
}

/// Reads a string column. A missing column, a NULL, or a failed conversion is `None`.
fn string_at(row: &Row, column: &str) -> Option<String> {
    match row.get_opt::<Option<String>, &str>(column) {
        Some(Ok(Some(value))) => Some(value),
        _ => None,
    }
}

/// Reads a numeric column. A missing column, a NULL, or a failed conversion is `None`.
///
/// Integer columns do not convert to `f64` directly, so they are read as
/// integers and widened.
fn number_at(row: &Row, column: &str) -> Option<f64> {
    if let Some(Ok(Some(value))) = row.get_opt::<Option<f64>, &str>(column) {
        return Some(value);
    }
    if let Some(Ok(Some(value))) = row.get_opt::<Option<i64>, &str>(column) {
        return Some(value as f64);
    }
    match row.get_opt::<Option<u64>, &str>(column) {
        Some(Ok(Some(value))) => Some(value as f64),
        _ => None,
    }
}

/// Builds a summary from one `subtask` row.
fn subtask_from_row(row: &Row) -> std::result::Result<SubtaskSummary, String> {
    let subtask_id = string_at(row, "subtask_id").ok_or("subtask_id is missing")?;
    build_summary(
        &subtask_id,
        string_at(row, "subtask_name"),
        string_at(row, "definition"),
    )
}

/// Validates one `subtask` row's values.
fn build_summary(
    subtask_id: &str,
    subtask_name: Option<String>,
    definition: Option<String>,
) -> std::result::Result<SubtaskSummary, String> {
    let subtask_id = subtask_id.trim();
    if subtask_id.is_empty() {
        return Err("subtask_id is empty".to_string());
    }
    Ok(SubtaskSummary {
        subtask_id: subtask_id.to_string(),
        subtask_name: subtask_name.unwrap_or_default(),
        definition: definition.unwrap_or_default(),
    })
}

/// Builds a candidate from one `candidate_model` row. Blank skill content means
/// no skill.
fn candidate_from_row(
    subtask_id: &str,
    row: &Row,
) -> std::result::Result<SubtaskCandidate, String> {
    let model_id = string_at(row, "model_id").ok_or("model_id is missing")?;
    let latency_ms = number_at(row, "latency").or_else(|| number_at(row, "latency_ms"));
    let success_rate = number_at(row, "success").or_else(|| number_at(row, "success_rate"));
    build_candidate(
        subtask_id,
        &model_id,
        number_at(row, "cost"),
        latency_ms,
        success_rate,
        string_at(row, "skill_content"),
        number_at(row, "skill_version"),
        valid_at(row, "is_valid"),
    )
}

/// Reads a validity flag. `0` marks the row invalid; a missing column or NULL
/// keeps it valid.
fn valid_at(row: &Row, column: &str) -> bool {
    match number_at(row, column) {
        Some(value) => value != 0.0,
        None => true,
    }
}

/// Validates one `candidate_model` row's values.
fn build_candidate(
    subtask_id: &str,
    model_id: &str,
    cost: Option<f64>,
    latency_ms: Option<f64>,
    success_rate: Option<f64>,
    skill_content: Option<String>,
    skill_version: Option<f64>,
    is_valid: bool,
) -> std::result::Result<SubtaskCandidate, String> {
    let model_id = model_id.trim();
    if model_id.is_empty() {
        return Err(format!(
            "subtask {subtask_id} has a candidate with an empty model_id"
        ));
    }
    let cost =
        cost.ok_or_else(|| format!("subtask {subtask_id} candidate {model_id} has no cost"))?;
    if !cost.is_finite() || cost < 0.0 {
        return Err(format!(
            "subtask {subtask_id} candidate {model_id} has an invalid cost {cost}"
        ));
    }
    let latency_ms = latency_ms
        .ok_or_else(|| format!("subtask {subtask_id} candidate {model_id} has no latency"))?;
    if !latency_ms.is_finite() || latency_ms < 0.0 {
        return Err(format!(
            "subtask {subtask_id} candidate {model_id} has an invalid latency {latency_ms}"
        ));
    }
    let success_rate = success_rate
        .ok_or_else(|| format!("subtask {subtask_id} candidate {model_id} has no success rate"))?;
    // Dictionaries in the wild store either a 0..1 share or a 0..100
    // percentage. Keep the share as the internal unit.
    let success_rate = if (0.0..=1.0).contains(&success_rate) {
        success_rate
    } else if (1.0..=100.0).contains(&success_rate) {
        success_rate / 100.0
    } else {
        return Err(format!(
            "subtask {subtask_id} candidate {model_id} has a success rate {success_rate} out of range"
        ));
    };
    let skill_content = skill_content
        .map(|skill| skill.trim().to_string())
        .filter(|skill| !skill.is_empty());
    let skill_version = skill_version
        .filter(|version| version.is_finite())
        .unwrap_or(0.0);
    Ok(SubtaskCandidate {
        model_id: ModelId::from(model_id),
        cost,
        latency_ms,
        success_rate,
        skill_content,
        skill_version,
        is_valid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_urls_and_timeouts() {
        assert!(MySqlSubtaskCatalog::new("not-a-url", Duration::from_secs(2)).is_err());
        assert!(
            MySqlSubtaskCatalog::new("postgres://127.0.0.1/db", Duration::from_secs(2)).is_err()
        );
        let zero_timeout = MySqlSubtaskCatalog::new("mysql://127.0.0.1/db", Duration::ZERO);
        assert!(zero_timeout.is_err());
        if let Err(error) = zero_timeout {
            assert!(error.to_string().contains("timeout"), "{error}");
        }
        MySqlSubtaskCatalog::new(
            "mysql://user:secret@127.0.0.1:3306/switchyard",
            Duration::from_secs(2),
        )
        .expect("a plain mysql URL builds the client");
    }

    #[test]
    fn subtask_summaries_need_an_id() {
        build_summary("edit-code", Some("Edit code".to_string()), None).expect("summary");
        let error = build_summary("  ", None, None).expect_err("blank id");
        assert!(error.contains("empty"), "{error}");
    }

    #[test]
    fn candidate_rows_are_validated() {
        let built = |cost, latency, success| {
            build_candidate(
                "edit-code",
                "model/cheap",
                cost,
                latency,
                success,
                None,
                None,
                true,
            )
        };
        let share = built(Some(0.5), Some(500.0), Some(0.9)).expect("candidate");
        assert!((share.success_rate - 0.9).abs() < f64::EPSILON);
        let percentage = built(Some(0.5), Some(500.0), Some(70.0)).expect("candidate");
        assert!((percentage.success_rate - 0.7).abs() < f64::EPSILON);
        assert!(built(None, Some(500.0), Some(0.9)).is_err());
        assert!(built(Some(-1.0), Some(500.0), Some(0.9)).is_err());
        assert!(built(Some(0.5), None, Some(0.9)).is_err());
        assert!(built(Some(0.5), Some(-1.0), Some(0.9)).is_err());
        assert!(built(Some(0.5), Some(500.0), None).is_err());
        assert!(built(Some(0.5), Some(500.0), Some(100.1)).is_err());
        assert!(
            build_candidate(
                "edit-code",
                "  ",
                Some(0.5),
                Some(500.0),
                Some(0.9),
                None,
                None,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn blank_skill_content_means_no_skill() {
        let candidate = build_candidate(
            "edit-code",
            "model/cheap",
            Some(0.5),
            Some(500.0),
            Some(0.9),
            Some("  ".to_string()),
            Some(4.0),
            true,
        )
        .expect("candidate");
        assert_eq!(candidate.skill_content, None);
        assert!((candidate.skill_version - 4.0).abs() < f64::EPSILON);
        assert!(candidate.is_valid);

        let candidate = build_candidate(
            "edit-code",
            "model/cheap",
            Some(0.5),
            Some(500.0),
            Some(0.9),
            Some(" Minimal patch. ".to_string()),
            None,
            false,
        )
        .expect("candidate");
        assert_eq!(candidate.skill_content.as_deref(), Some("Minimal patch."));
        assert!((candidate.skill_version - 0.0).abs() < f64::EPSILON);
        assert!(!candidate.is_valid);
    }

    #[tokio::test]
    async fn an_unavailable_catalog_reports_an_error() {
        let catalog = UnavailableSubtaskCatalog::new("RSI_MYSQL_URL is not set");
        let error = catalog
            .subtasks()
            .await
            .expect_err("unavailable catalog should fail");
        assert!(error.to_string().contains("RSI_MYSQL_URL is not set"));
        assert!(catalog.candidates("edit-code").await.is_err());
    }
}
