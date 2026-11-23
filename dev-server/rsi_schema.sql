-- RSI router dictionary schema.
--
-- The `subtask` table holds what the recognizer reads. The `candidate_model`
-- table holds one subtask's Pareto candidates and is read after recognition.
-- Apply with:
--   mysql --host 127.0.0.1 --user root --default-character-set=utf8mb4 < dev-server/rsi_schema.sql

CREATE DATABASE IF NOT EXISTS rsi
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS rsi.subtask (
  subtask_id   VARCHAR(191) NOT NULL,
  subtask_name VARCHAR(255) NOT NULL DEFAULT '',
  definition   TEXT NOT NULL,
  PRIMARY KEY (subtask_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS rsi.candidate_model (
  subtask_id    VARCHAR(191) NOT NULL,
  model_id      VARCHAR(255) NOT NULL,
  cost          DOUBLE NOT NULL,
  latency_ms    DOUBLE NOT NULL,
  success_rate  DOUBLE NOT NULL,
  total_cnt     BIGINT UNSIGNED DEFAULT NULL,
  skill_content TEXT NULL,
  PRIMARY KEY (subtask_id, model_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
