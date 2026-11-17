# RSI Router

The RSI router labels the current step with a subtask ID. It reads the whole
subtask catalog from Redis, shows every subtask's ID, name, and definition to a
small recognizer model, and sends the step to a model from the selected
subtask's Pareto set. If the label is missing, weak, or unknown, the step goes
to a flagship model. The flagship model is the large model you use when the
router has no reliable subtask.

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
redis_url_env = "RSI_REDIS_URL"
```

The server reads `RSI_REDIS_URL` at startup. The URL may use `redis://` or
`rediss://`. It may include a username, password, and database number:

```text
rediss://user:password@redis.example.com:6380/2
```

Do not put the URL in the TOML file. Keep it in the environment variable.

The same target may appear in both `candidate_targets` and `flagship_target`.
That lets the flagship model also serve subtasks that need it.

## How a step is routed

1. The router reads the whole subtask catalog from Redis.
2. The recognizer reads the first user message, the last `history_steps` steps,
   the tool names, the latest tool call or result, and every subtask's ID, name,
   and definition.
3. The recognizer returns JSON:

   ```json
   {"subtask_id": "edit-code", "confidence": 0.92}
   ```

4. If the confidence is at least `confidence_threshold` and the ID is in the
   catalog, the router uses that subtask.
5. The router picks a model from the subtask's `candidates`.
6. If any step above fails, the router sends the request to `flagship_target`.

The route sends no candidate request when it falls back to the flagship model.

## Dictionary entries

The catalog is one JSON array at Redis key `rsi:subtasks`. Change the key with
`redis_catalog_key`. The value holds every subtask:

```json
[
  {
    "subtask_id": "edit-code",
    "subtask_name": "Edit code",
    "definition": "Change existing source files in a small, focused patch.",
    "candidates": [
      {"model_id": "model/cheap", "cost_per_million_tokens": 0.4},
      {"model_id": "model/mid", "cost_per_million_tokens": 1.2}
    ],
    "skill": {
      "version": "1.4.0",
      "template": "Read the nearby tests before editing the parser."
    }
  }
]
```

The recognizer sees `subtask_id`, `subtask_name`, and `definition`. It must
choose one of the listed IDs. When the current step does not fit any definition,
it returns an empty `subtask_id`, and the router uses the flagship model.

`candidates` is the Pareto set for the subtask. It is the short list of models
that are acceptable for this subtask. Each `model_id` must match the `id` of one
entry in `candidate_targets`. The router ignores entries that name another
model.

With `selection = "lowest_cost"` (the default), the router tries the cheapest
candidate first. With `selection = "catalog_order"`, it keeps the order in the
JSON. The remaining candidates are fallback models for the host to try in order.

When a `skill` is present, its `template` is added to the request as a system
instruction. Its `version` is written to the step record.

## Fallback to the flagship model

The router uses `flagship_target` for these cases:

- `RSI_REDIS_URL` is unset or empty.
- The Redis URL is invalid, or TLS setup fails.
- The Redis connection fails or the read times out.
- The catalog JSON is invalid.
- The catalog is empty.
- The recognizer call fails.
- The recognizer output is not valid JSON.
- The `subtask_id` is empty.
- `confidence` is below `confidence_threshold`.
- The recognized ID is not in the catalog.
- The entry has no candidate that is also in `candidate_targets`.
- The entry has invalid data.

These failures do not stop the server. The route still starts and serves every
request from the flagship model until Redis is available.

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
- The full conversation `messages`, `instructions`, and `tools`.
- The normalized model `response`, including text, tool calls, and stop reason.
  The log leaves out the preserved provider body, which repeats the same
  content.
- `selected_model` and `served_model`.
- `subtask_id`, `confidence`, `decision`, `reason`, and `skill_version`.
- The step `result`, `usage`, `cost_usd`, `cost_source`, and `latency_ms`.

Streamed responses are accumulated for the log while they are forwarded to the
client. A stream dropped before its terminal event has no execution record.

### Routing log

`--rsi-routing-log-file` records the routing decision only. It holds:

- `request_id`, optional `session_id`, `step`, `route_id`, and `ts`.
- `decision`, `reason`, `subtask_id`, `confidence`, and `skill_version`.
- `selected_model` and `served_model`.
- `result`, `usage`, `cost_usd`, `cost_source`, and `latency_ms`.

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
returns one full record, including the messages, tools, and model response.

## Configuration keys

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `recognizer_target` | Yes | — | Target that labels the current step. |
| `flagship_target` | Yes | — | Target used when no subtask route applies. |
| `candidate_targets` | Yes | — | Targets the dictionary may select. |
| `confidence_threshold` | No | `0.7` | Lowest confidence that enters the candidate route. In `[0, 1]`. |
| `history_steps` | No | `3` | Recent steps shown to the recognizer. |
| `selection` | No | `lowest_cost` | `lowest_cost` or `catalog_order`. |
| `recognizer_prompt` | No | packaged prompt | Replaces the recognition instructions. |
| `max_output_tokens` | No | `4096` | Maximum tokens for the recognizer output. |
| `redis_url_env` | Yes | — | Environment variable holding the Redis URL. |
| `redis_catalog_key` | No | `rsi:subtasks` | Redis key holding the catalog array. |
| `redis_timeout_ms` | No | `2000` | Timeout for one catalog read. |

## Limits

- The recognizer is an LLM target. It must return the JSON shape above.
- The router reads the whole catalog on every routed step. It keeps no local copy.
- The route needs a Redis server. A Redis failure sends the step to the
  flagship model and does not stop the request.
- Step numbers depend on the client sending its conversation history.
- The read endpoints scan the JSONL file. Long-lived deployments should rotate
  the file.
- A step record is written when the response reaches its terminal event. A
  stream dropped before that point has no step record. The recognizer log is
  still written as soon as routing finishes.
    
## run
.\target\release\switchyard-server.exe --config dev-server\config.toml  --host 127.0.0.1 --port 4000

curl -i http://127.0.0.1:4000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"hello"}]}'