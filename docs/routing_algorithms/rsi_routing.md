# RSI Router

The RSI router labels the current step with a subtask ID. It reads the subtask
list from MySQL, shows every subtask's ID, name, and definition to a small
recognizer model, and only then reads the recognized subtask's Pareto
candidates. The router orders the candidates with the configured metric and
sends the step to the first one. If the label is missing, weak, or unknown, the
step goes to a flagship model. The flagship model is the large model you use
when the router has no reliable subtask. Setting `fixed_target` pins every
request to one model instead, including those fallback cases.

Use it with `type = "rsi_router"`.

## Use it as the default route

The server sends a request to the first `rsi_router` route in configuration
order when either condition holds:

- The request has no `model` field.
- The `model` field does not match a configured route ID.

This lets clients send their own model names, such as Claude CLI's
`claude-sonnet-4-5`, and still reach the router. A deployment with no
`rsi_router` route rejects requests that name no configured route.

## Configure a route

```toml
schema_version = 1

[llm_clients.cloud]
format = "openai_chat"
base_url = "https://gateway.example.com/v1"
api_key_env = "MODEL_GATEWAY_KEY"

[targets.subtask_recognizer]
id = "small/subtask-model"
llm_client = "cloud"

[targets.flagship]
id = "large/flagship-model"
llm_client = "cloud"

[targets.cheap]
id = "model/cheap"
llm_client = "cloud"

[targets.mid]
id = "model/mid"
llm_client = "cloud"

[routes.agent]
id = "switchyard/rsi"
type = "rsi_router"
recognizer_target = "subtask_recognizer"
flagship_target = "flagship"
candidate_targets = ["cheap", "mid"]
min_success_rate = 0.7
mysql_url_env = "RSI_MYSQL_URL"
```

The server reads `RSI_MYSQL_URL` at startup. The URL may include a username,
password, database name, and TLS parameters:

```text
mysql://user:password@mysql.example.com:3306/switchyard?require_ssl=true
```

Do not put the URL in the TOML file. Keep it in the environment variable.

The same target may appear in both `candidate_targets` and `flagship_target`.
That lets the flagship model also serve subtasks that need it.

## How a step is routed

1. The router reads the `subtask` table from MySQL.
2. The recognizer reads the first user message, the last `history_steps` steps,
   the previous subtask prediction, and every subtask's ID, name, and
   definition. An unknown prediction is still shown on the next step.
3. The recognizer returns JSON:

   ```json
   {"subtask_id": "edit-code", "confidence": 0.92}
   ```

4. If the confidence is at least `confidence_threshold` and the ID is in the
   `subtask` table, the router reads that subtask's candidates from the
   `candidate_model` table. Only now does it touch the candidates.
5. The router filters the candidates to `candidate_targets` and to
   `min_success_rate`, orders them with `selection`, and picks the first one.
   When candidates carry `skill_content`, the router injects the one with the
   highest `skill_version` as a `<skill_reminder>` message at the start of the
   subtask.
6. If any step above fails, the router sends the request to `flagship_target`,
   or to `fixed_target` when one is set. When the subtask was recognized, the
   request still carries the newest valid skill. Recognition failures and
   unknown subtasks carry none.

## MySQL tables

The dictionary lives in two MySQL tables.

`subtask` holds what the recognizer reads:

```sql
CREATE TABLE subtask (
  subtask_id   VARCHAR(191) PRIMARY KEY,
  subtask_name VARCHAR(255) NOT NULL DEFAULT '',
  definition   TEXT NOT NULL
);
```

`definition` states what the subtask does, its input, its expected output, and
its typical failure points. The recognizer sees `subtask_id`, `subtask_name`,
and `definition`, and must choose one of the listed IDs. When the current step
does not fit any definition, it returns an empty `subtask_id`, and the router
uses the flagship model.

`candidate_model` holds the Pareto set of each subtask:

```sql
CREATE TABLE candidate_model (
  subtask_id    VARCHAR(191) NOT NULL,
  model_id      VARCHAR(255) NOT NULL,
  cost          DOUBLE NOT NULL,
  latency       DOUBLE NOT NULL,
  success       DOUBLE NOT NULL,
  skill_content TEXT NULL,
  skill_version DOUBLE NOT NULL DEFAULT 0,
  is_valid      TINYINT(1) NOT NULL DEFAULT 1,
  PRIMARY KEY (subtask_id, model_id)
);
```

`candidates` is the Pareto set for the subtask. It is the short list of models
that are acceptable for this subtask. Each `model_id` must match the `id` of one
entry in `candidate_targets`. The router skips rows that name another model,
rows with a missing or out-of-range `cost`, `latency`, or `success`, and rows
whose `success` is below `min_success_rate`. If every candidate is filtered
out, the step falls back to `flagship_target`.

- `cost` orders the candidates with `selection = "lowest_cost"` (the default).
- `latency`, in milliseconds, orders them with `selection = "lowest_latency"`.
  The column may also be named `latency_ms`.
- `success` (or `success_rate`), a share from 0 to 1 or a percentage from 1 to
  100, orders them with `selection = "highest_success"`.

The first candidate runs the step; the rest are fallback models for the host to
try in order.

## Fixed target

When `fixed_target` is set, the router pins every request to that target. It
replaces `flagship_target` in every fallback case: a failed recognition, an
empty or unknown subtask ID, low confidence, an unavailable catalog, a failed
candidate read, or candidates that are all filtered out. The request is still
served.

Recognition still runs, so a recognized step keeps its `subtask_id` in the logs
and can carry a skill. When recognition succeeds and the candidates are
readable, the skill comes from the candidate with the highest `skill_version`,
the same as without a fixed target. Otherwise the request goes out without a
skill. A fixed target has no fallback models. Routing logs record
`fixed_target` plus the Pareto set when it was read.

When candidates carry `skill_content`, the router injects the one with the
highest `skill_version` among candidates whose `is_valid` is not `0`. A missing
`is_valid` column or NULL counts as valid, and a missing or unreadable version
counts as `0`. When versions tie, the candidate that serves the step wins, so a
dictionary without versions keeps its earlier behavior. Blank content counts as
no skill. `is_valid` only filters skill injection; routing still considers the
row. The injected text is a message like this:

```text
<skill_reminder>当前为子任务id1，Skill：Read the nearby tests first.</skill_reminder>
```

The `1` is the recognized subtask ID.

The reminder goes where the subtask started. On the subtask's first step it
sits at the end of that request. On later steps the router inserts it at the
same position in the conversation, so the first step's work follows it. This
keeps the model from reading the reminder as a new instruction for the current
step.

OpenAI-compatible requests use the `user` role for this message. Their chat
templates may reject a system message that is not at the start. Anthropic-
compatible requests use the `system` role internally. Every request carries one
reminder. If the conversation already carries the same reminder, the router does
not add another. Blank skill content adds nothing. With `skill_injected = false`,
no skill reminder is added at all.

## Fallback to the flagship model

The router uses `flagship_target` for these cases:

- `RSI_MYSQL_URL` is unset or empty.
- The MySQL URL is invalid, or the connection fails.
- A query fails or exceeds `mysql_timeout_ms`.
- The `subtask` table is empty.
- The recognizer call fails.
- The recognizer output is not valid JSON.
- The `subtask_id` is empty.
- `confidence` is below `confidence_threshold`.
- The recognized ID is not in the `subtask` table.
- The subtask has no candidate that is also in `candidate_targets`.
- Every candidate's `success` is below `min_success_rate`.

These failures do not stop the server. The route still starts and serves every
request from the flagship model until MySQL is available. When `fixed_target` is
set, it serves these cases instead.

When the subtask is recognized, the fallback request carries the same newest
valid skill the router would inject for a candidate. This covers a low
confidence step and a recognized subtask whose candidates are all filtered out.
A failed recognition, an empty ID, an unknown subtask, or a failed candidate
read has no skill to inject.

## Logs

There are three independent JSONL logs. Every record carries a `request_id`, a
`ts`, and a one-based `step`. Use `request_id` to join the three logs for one
request.

The logs are enabled by default, relative to the working directory, at:

```text
logs/rsi-execution.jsonl
logs/rsi-routing.jsonl
logs/rsi-recognizer.jsonl
```

Override a path when needed:

```bash
switchyard-server --config routes.toml \
  --trajectory-log-file ./logs/rsi-execution.jsonl \
  --rsi-routing-log-file ./logs/rsi-routing.jsonl \
  --rsi-recognizer-log-file ./logs/rsi-recognizer.jsonl
```

`--dry-run` does not create the default files.

A session ID is optional. The server reads it from an `x-switchyard-session-id`
header first, then from `metadata.session_id` in the request body. When neither
is present, the record has no `session_id`, and `request_id` plus `ts` provide
the correlation.

The step number is the number of assistant messages in the incoming request plus
one. Send the full conversation history so the number stays stable.

### Execution log

`--trajectory-log-file` records the execution model interaction for each step.
It holds:

- `request_id`, optional `session_id`, `step`, `route_id`, and `ts`.
- `upstream_request`: the request body sent to the execution model, exactly as
  the client built it — resolved model id, format conversion, target
  `extra_body`, and caching marks included. This is what the model actually
  received, so it is the source for offline analysis. Present when a model call
  served the step; failed steps record no request body, and the routing log's
  error text carries the upstream complaint.
- The normalized model `response`, including text, tool calls, and stop reason.
  The log leaves out the preserved provider body, which repeats the same
  content.
- `selected_model` and `served_model`.
- `subtask_id`, `confidence`, `decision`, `reason`, and `skill_injected`.
- The step `result`, `usage`, `cost_usd`, `cost_source`, and `latency_ms`.

Streamed responses are accumulated for the log while they are forwarded to the
client. A stream dropped before its terminal event has no execution record.

### Routing log

`--rsi-routing-log-file` records the routing decision only. It holds:

- `request_id`, optional `session_id`, `step`, `route_id`, and `ts`.
- `decision`, `reason`, `subtask_id`, `confidence`, `confidence_threshold`,
  `min_success_rate`, `selection`, `skill_injected`. `fixed_target` appears
  instead of `selection` when a fixed target served the step.
- `candidates`: the ordered eligible model IDs, selected first, then fallbacks.
- `pareto_candidates`: the Pareto set as it entered selection. Each entry has
  `model_id`, `cost`, `latency_ms`, `success_rate`, `configured`,
  `above_min_success_rate`, and `selected`.
- `selected_model` and `served_model`.
- `result`, `usage`, `cost_usd`, `cost_source`, and `latency_ms`.

`candidates` and `pareto_candidates` are present when the router read the
subtask's candidates, including the fallback cases where every candidate is
filtered out. They are absent when recognition failed or the candidate read
failed.

It has no messages, tools, or model response. Use it for quick analysis of which
model each step used and what it cost.

### Recognizer log

`--rsi-recognizer-log-file` records the recognizer interaction. It holds:

- `request_id`, optional `session_id`, `step`, `route_id`, and `ts`.
- The recognizer `model`.
- The full `prompt` sent to the recognizer.
- The raw `response` text.
- The parsed `subtask_id` and `confidence`.
- The recognizer call `latency_ms`.
- `error`, when the call failed or the output had no valid subtask.

The recognizer log is written as soon as routing finishes. The execution and
routing logs are written when the response reaches its terminal event, so they
can include final usage and latency.

Cost uses the bundled rates. Pass `--pricing-file` to replace them. When no rate
is available, `cost_usd` is omitted.

Read execution records through the server:

```text
GET /v1/rsi/sessions/{session_id}/steps
GET /v1/rsi/sessions/{session_id}/steps/{step}
```

The first endpoint returns step summaries with cost and latency. The second
returns one full record, including `upstream_request` and the model response.

Records written by older servers have no `upstream_request`; they store either
the caller's body in `raw_request` or a normalized conversation in `messages`,
`instructions`, and `tools`. In that normalized shape a tool result appears as
a `user` message with a `tool_result` block.

## Configuration keys

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `recognizer_target` | Yes | — | Target that labels the current step. |
| `flagship_target` | Yes | — | Target used when no subtask route applies. Unused when `fixed_target` is set. |
| `candidate_targets` | Yes | — | Targets the dictionary may select. |
| `confidence_threshold` | No | `0.7` | Lowest confidence that enters the candidate route. In `[0, 1]`. |
| `min_success_rate` | No | `0.0` | Candidates with a lower success rate are removed before selection. In `[0, 1]`. `0` keeps every candidate. |
| `history_steps` | No | `3` | Recent steps shown to the recognizer. |
| `selection` | No | `lowest_cost` | `highest_success`, `lowest_cost`, or `lowest_latency`. |
| `fixed_target` | No | unset | When set, every request is served by this target, including unrecognized subtasks and catalog failures. Skill injection still applies when a subtask and its candidates are available. |
| `skill_injected` | No | `true` | When `false`, the selected candidate's skill reminder is not appended to the request. |
| `recognizer_prompt` | No | packaged prompt | Replaces the recognition instructions. |
| `max_output_tokens` | No | `4096` | Maximum tokens for the recognizer output. |
| `mysql_url_env` | Yes | — | Environment variable holding the MySQL URL. |
| `mysql_timeout_ms` | No | `2000` | Timeout for one catalog query. |

## Limits

- The recognizer is an LLM target. It must return the JSON shape above.
- The router reads the `subtask` table on every routed step, and the recognized
  subtask's candidates right after recognition. It keeps no local copy.
- The route needs a MySQL server. A MySQL failure sends the step to the
  flagship model and does not stop the request.
- Step numbers depend on the client sending its conversation history.
- The read endpoints scan the JSONL file. Long-lived deployments should rotate
  the file.
- A step record is written when the response reaches its terminal event. A
  stream dropped before that point has no step record. The recognizer log is
  still written as soon as routing finishes.

## run
cargo build --locked --release -p switchyard-server

.\target\release\switchyard-server.exe --config dev-server\config.toml  --host 127.0.0.1 --port 4000

curl -i http://127.0.0.1:4000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"hello"}]}'
