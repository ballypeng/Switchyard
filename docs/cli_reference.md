# Server CLI Reference

`switchyard-server` runs the standalone Rust proxy for API clients and custom deployments.

Install the standalone binary with `cargo install --locked switchyard-server`.
It reads a native TOML deployment.

### Usage

```bash
switchyard-server --config <deployment.toml> [options]
```

| Option | Default | Purpose |
|---|---|---|
| `--config PATH` | Required | TOML file defining LLM clients, targets, and algorithm routes. |
| `--host HOST` | `0.0.0.0` | Address on which the server listens. |
| `-p, --port PORT` | `4000` | Port on which the server listens. |
| `--backlog BACKLOG` | `65535` | TCP listen backlog configured before accepting traffic. |
| `--shutdown-timeout SHUTDOWN_TIMEOUT` | `30s` | Maximum time active requests may drain during shutdown. |
| `--dry-run` | Off | Validate the deployment without binding a socket. |
| `--routing-log-file PATH` | None | Append durable per-request routing records to this JSONL file. |
| `--pricing-file PATH` | None | Estimate unreported call costs from this JSON file of USD rates per million tokens. Applies to the routing log and the RSI execution log. |
| `--trajectory-log-file PATH` | `logs/rsi-execution.jsonl` | Append RSI execution records (context and model response) to this JSONL file. Enables the `/v1/rsi/sessions/...` read endpoints. |
| `--rsi-recognizer-log-file PATH` | `logs/rsi-recognizer.jsonl` | Append each RSI recognizer input and output to this JSONL file for manual review. |
| `--rsi-routing-log-file PATH` | `logs/rsi-routing.jsonl` | Append each RSI routing decision to this JSONL file, without the message context. |

The three RSI logs use their default paths unless you pass a `--*-log-file`
option. Default paths are relative to the process working directory.
`--dry-run` does not create the default files.

Every RSI log record carries a `request_id` and a `ts`. Use `request_id` to join
the three logs for one request. A `session_id` is used when available from a
request header or from `metadata.session_id` in the request body.
| `--tls-cert PATH` | None | PEM certificate path; requires `--tls-key`. |
| `--tls-key PATH` | None | PEM private-key path; requires `--tls-cert`. |
| `-h, --help` | — | Print command help. |
| `-V, --version` | — | Print the server version. |

Validate a deployment, then start the proxy:

```bash
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

## Related Documentation

- [Getting Started](getting_started.md): installation, configuration, and validation
- [`switchyard-server`](../crates/switchyard-server/README.md): complete TOML schema, TLS, and metrics
