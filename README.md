# llm-cost-dashboard

> Real-time terminal dashboard for LLM token spend -- cost per request, per-model
> breakdown, projected monthly bills, and budget enforcement. Zero external
> services required.

[![CI](https://github.com/Mattbusel/llm-cost-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-cost-dashboard/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/llm-cost-dashboard.svg)](https://crates.io/crates/llm-cost-dashboard)
[![docs.rs](https://docs.rs/llm-cost-dashboard/badge.svg)](https://docs.rs/llm-cost-dashboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built with [ratatui](https://ratatui.rs) and [crossterm](https://github.com/crossterm-rs/crossterm).
Structured logging via [tracing](https://tracing.rs). No database. No network. No cloud account.

---

## What it does

`llm-dash` reads a stream of LLM request records (JSON lines) from a log file or
`stdin`, computes USD cost using a built-in pricing table, and renders a live
terminal dashboard showing:

- **Total spend** (session) and **projected monthly bill** (extrapolated from
  the last hour of activity)
- **Cost by model** -- horizontal bar chart sorted by highest spend
- **Recent requests** -- scrollable table with timestamp, model, token counts,
  cost, and latency
- **Budget gauge** -- visual progress bar with configurable alert threshold and
  hard limit
- **Spend sparkline** -- mini chart of the last 60 request costs

---

## Supported providers and models

| Provider   | Model               | Input ($/1M) | Output ($/1M) |
|------------|---------------------|--------------|---------------|
| Anthropic  | claude-opus-4-6     | $15.00       | $75.00        |
| Anthropic  | claude-sonnet-4-6   | $3.00        | $15.00        |
| Anthropic  | claude-haiku-4-5    | $0.25        | $1.25         |
| OpenAI     | gpt-4o              | $5.00        | $15.00        |
| OpenAI     | gpt-4o-mini         | $0.15        | $0.60         |
| OpenAI     | gpt-4-turbo         | $10.00       | $30.00        |
| OpenAI     | o1-preview          | $15.00       | $60.00        |
| OpenAI     | o3-mini             | $1.10        | $4.40         |
| Google     | gemini-1.5-pro      | $3.50        | $10.50        |
| Google     | gemini-1.5-flash    | $0.075       | $0.30         |

Unknown models automatically fall back to `$5.00/$15.00` input/output pricing.
Lookup is case-insensitive.

---

## Quickstart

### Install from crates.io

```bash
cargo install llm-cost-dashboard
```

### Install from source

```bash
git clone https://github.com/Mattbusel/llm-cost-dashboard
cd llm-cost-dashboard
cargo install --path .
```

### Run with demo data

```bash
llm-dash --demo
```

### Set a monthly budget and tail a log file

```bash
llm-dash --budget 50.0 --log-file requests.log
```

### Pipe from your application

```bash
your-llm-app | llm-dash --budget 25.0
```

---

## Dashboard layout

```
 LLM Cost Dashboard  [q: quit | r: reset | d: demo data | j/k: scroll]
+------------------+--------------------------------------------------+
| Summary          |  Cost by Model (uUSD)                            |
| Total: $0.0142   |  ████████ claude-sonnet-4-6                      |
| Proj:  $0.42/mo  |  ████ gpt-4o-mini                                |
+------------------+  ██ claude-haiku-4-5                             |
| Budget           +--------------------------------------------------+
| ████░░░ 14.2%    |  Recent Requests                                 |
| $8.58 remaining  |  12:34:01  claude-sonnet  847in/312out  $0.0031  |
+------------------+--------------------------------------------------+
| Sparkline: spend over last 60 requests                              |
+--------------------------------------------------------------------+
```

---

## Log file format

Records must be newline-delimited JSON (NDJSON). The four required fields are
`model`, `input_tokens`, `output_tokens`, and `latency_ms`:

```json
{"model":"claude-sonnet-4-6","input_tokens":512,"output_tokens":256,"latency_ms":340}
{"model":"gpt-4o-mini","input_tokens":128,"output_tokens":64,"latency_ms":12}
```

Optional fields:

| Field      | Type   | Default     | Description                              |
|------------|--------|-------------|------------------------------------------|
| `provider` | string | `"unknown"` | Provider name shown in traces            |
| `error`    | string | absent      | Error message; marks request as failed   |

Malformed lines are skipped and logged as warnings -- the dashboard never crashes
on bad input.

---

## CLI reference

```
llm-dash [OPTIONS]

Options:
  --budget <BUDGET>        Monthly budget limit in USD [default: 10.0]
  --log-file <LOG_FILE>    JSON log file to tail for live data
  --demo                   Start with built-in demo data pre-loaded
  -h, --help               Print help
  -V, --version            Print version
```

---

## Keyboard controls

| Key         | Action                      |
|-------------|-----------------------------|
| q / Esc     | Quit                        |
| d           | Load demo data              |
| r           | Reset all data              |
| j / Down    | Scroll requests down        |
| k / Up      | Scroll requests up          |

---

## Environment variables

| Variable   | Description                                                  | Default  |
|------------|--------------------------------------------------------------|----------|
| `RUST_LOG` | Tracing log level (`error`, `warn`, `info`, `debug`)         | `info`   |

Tracing output is written to **stderr** so it does not interfere with piped
stdin/stdout.

---

## Library usage

The crate exposes its core types as a library crate (`llm_cost_dashboard`) for
embedding cost tracking directly in your application:

```rust
use llm_cost_dashboard::{CostLedger, CostRecord};

let mut ledger = CostLedger::new();
let record = CostRecord::new("gpt-4o-mini", "openai", 512, 256, 34);
ledger.add(record).expect("valid record");
println!("total: ${:.6}", ledger.total_usd());
println!("projected/mo: ${:.2}", ledger.projected_monthly_usd(1));
```

### Key types

| Type             | Module  | Description                                       |
|------------------|---------|---------------------------------------------------|
| `CostRecord`     | `cost`  | Single LLM request with computed USD cost         |
| `CostLedger`     | `cost`  | Append-only ledger with aggregation helpers       |
| `ModelStats`     | `cost`  | Per-model aggregated statistics                   |
| `BudgetEnvelope` | `budget`| Hard limit + alert threshold spend tracker        |
| `LogEntry`       | `log`   | Raw log entry (model, tokens, latency)            |
| `RequestLog`     | `log`   | Ordered log with JSON ingestion                   |
| `TraceSpan`      | `trace` | Distributed trace span with cost annotation       |
| `SpanStore`      | `trace` | In-memory span store                              |
| `DashboardError` | `error` | Unified error type                                |
| `App`            | `ui`    | Full TUI application state                        |

---

## Architecture

```
src/
  main.rs          # CLI entry point (clap + tracing init)
  lib.rs           # Public re-exports
  error.rs         # DashboardError (thiserror)
  cost/
    mod.rs         # CostRecord, CostLedger, ModelStats
    pricing.rs     # Static pricing table + lookup/compute_cost
  budget/
    mod.rs         # BudgetEnvelope (hard limit + alert threshold)
  log/
    mod.rs         # LogEntry, RequestLog, IncomingRecord (NDJSON parser)
  trace/
    mod.rs         # TraceSpan, SpanStore (distributed tracing helpers)
  ui/
    mod.rs         # App state + run() event loop
    dashboard.rs   # Full-frame layout compositor
    widgets.rs     # Budget gauge, sparkline, summary panel
    theme.rs       # Centralised colour/style palette

tests/
  unit_tests.rs        # Public-API unit tests (pricing, ledger, budget, log)
  integration_tests.rs # Cross-module integration tests
  integration.rs       # End-to-end app-level tests

benches/
  cost_bench.rs    # Criterion benchmarks for pricing lookup and aggregation
```

---

## Development

```bash
# Run all tests
cargo test

# Run with debug tracing
RUST_LOG=debug cargo run -- --demo

# Lint
cargo clippy --all-targets --all-features -- -D warnings

# Format
cargo fmt

# Benchmarks
cargo bench

# Documentation
cargo doc --open
```

---

## Related projects by @Mattbusel

- [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- Rust async LLM pipeline orchestration
- [rot-signals-api](https://github.com/Mattbusel/rot-signals-api) -- Options signal REST API
- [prompt-observatory](https://github.com/Mattbusel/prompt-observatory) -- LLM interpretability dashboard

---

## License

MIT -- see [LICENSE](LICENSE) for details.
