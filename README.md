# llm-cost-dashboard

> Real-time terminal dashboard for LLM token spend -- cost per request, per-model
> breakdown, projected monthly bills, budget enforcement, anomaly detection, and
> webhook alerting. Zero external services required.

[![CI](https://github.com/Mattbusel/llm-cost-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-cost-dashboard/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/llm-cost-dashboard.svg)](https://crates.io/crates/llm-cost-dashboard)
[![docs.rs](https://docs.rs/llm-cost-dashboard/badge.svg)](https://docs.rs/llm-cost-dashboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built with [ratatui](https://ratatui.rs) and [crossterm](https://github.com/crossterm-rs/crossterm).
Structured logging via [tracing](https://tracing.rs). No database. No network. No cloud account.

---

## What is this?

`llm-cost-dashboard` is a Rust terminal application (TUI) that reads a stream of
LLM request records from a log file or stdin, computes per-request USD costs from a
built-in pricing table, and renders a live dashboard showing total spend, per-model
breakdowns, budget status, and projected monthly bills. It also includes a rolling
Z-score anomaly detector that flags unexpected cost spikes, an OLS linear regression
forecaster that projects spend to end-of-month, and a Slack-compatible webhook
alerter that delivers budget and anomaly alerts with per-kind cooldown
deduplication. Everything runs locally -- no cloud accounts, no telemetry, no
databases.

---

## 5-Minute Quickstart

### Step 1 - Install

```bash
# From crates.io (recommended)
cargo install llm-cost-dashboard

# Or from source
git clone https://github.com/Mattbusel/llm-cost-dashboard
cd llm-cost-dashboard
cargo install --path .
```

### Step 2 - Launch with demo data

```bash
llm-dash --demo
```

You will immediately see a live dashboard with pre-loaded synthetic requests
covering Claude, GPT-4o, Gemini, and o3-mini.

### Step 3 - Set a budget and tail your log file

```bash
llm-dash --budget 50.0 --log-file /var/log/llm-requests.ndjson
```

### Step 4 - Pipe directly from your application

```bash
your-llm-app | llm-dash --budget 25.0
```

### Step 5 - Use the keyboard controls

| Key      | Action                   |
|----------|--------------------------|
| q / Esc  | Quit                     |
| d        | Load demo data           |
| r        | Reset all data           |
| j / Down | Scroll requests down     |
| k / Up   | Scroll requests up       |

That is it. The dashboard updates every 250 ms automatically.

---

## Installation

### Binary via cargo install

```bash
cargo install llm-cost-dashboard
```

The binary is named `llm-dash`.

### Build from source

```bash
git clone https://github.com/Mattbusel/llm-cost-dashboard
cd llm-cost-dashboard
cargo build --release
# Binary at ./target/release/llm-dash
```

### Build without webhook support (smaller binary, no TLS dependency)

```bash
cargo build --release --no-default-features
```

---

## Dashboard layout (ASCII)

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
+------------------+  12:33:58  gpt-4o-mini    512in/128out  $0.0001  |
                   |  12:33:55  claude-haiku   256in/64out   $0.0001  |
+--------------------------------------------------+------------------+
| Sparkline: spend over last 60 requests                              |
| ▁▁▂▁▁▃▁▁▂▄▁▁▂▁▃▄▁▁▂▁▁▂▁▁▂▄▃▁▁▂▁▁▂▁▁▃▁▁▂▁▁▂▁▁▂▄▁▁▁▂▁▁▂▁▁▃▁▁▂▁       |
+--------------------------------------------------------------------+
```

### Layout regions

| Region          | Description                                            |
|-----------------|--------------------------------------------------------|
| Summary         | Session total and extrapolated monthly projection      |
| Budget gauge    | Visual progress bar with alert threshold marker        |
| Cost by model   | Horizontal bar chart sorted by highest spend           |
| Recent requests | Scrollable table (j/k to scroll)                       |
| Sparkline       | Last 60 request costs as a mini chart                  |

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

Malformed lines are skipped and logged as warnings. The dashboard never crashes
on bad input.

---

## Supported providers and models

**100+ models** built-in across 11 providers. Lookup is case-insensitive.
Unknown models fall back to `$5.00/$15.00` automatically.

<details>
<summary><b>Anthropic Claude</b> — 9 models</summary>

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|--------------|
| claude-opus-4-6 | $15.00 | $75.00 |
| claude-sonnet-4-6 | $3.00 | $15.00 |
| claude-haiku-4-5 | $0.25 | $1.25 |
| claude-3-5-sonnet-20241022 | $3.00 | $15.00 |
| claude-3-5-haiku-20241022 | $0.80 | $4.00 |
| claude-3-opus-20240229 | $15.00 | $75.00 |
| claude-3-sonnet-20240229 | $3.00 | $15.00 |
| claude-3-haiku-20240307 | $0.25 | $1.25 |

</details>

<details>
<summary><b>OpenAI</b> — 11 models</summary>

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|--------------|
| gpt-4.5-preview | $75.00 | $150.00 |
| gpt-4o | $5.00 | $15.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4-turbo | $10.00 | $30.00 |
| gpt-4 | $30.00 | $60.00 |
| gpt-3.5-turbo | $0.50 | $1.50 |
| o1 | $15.00 | $60.00 |
| o3 | $10.00 | $40.00 |
| o3-mini | $1.10 | $4.40 |
| o4-mini | $1.10 | $4.40 |

</details>

<details>
<summary><b>Google Gemini</b> — 6 models</summary>

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|--------------|
| gemini-2.5-pro | $1.25 | $10.00 |
| gemini-2.0-flash | $0.10 | $0.40 |
| gemini-2.0-flash-lite | $0.075 | $0.30 |
| gemini-1.5-pro | $3.50 | $10.50 |
| gemini-1.5-flash | $0.075 | $0.30 |
| gemini-1.5-flash-8b | $0.0375 | $0.15 |

</details>

<details>
<summary><b>DeepSeek</b> — 7 models</summary>

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|--------------|
| deepseek-r1 | $0.55 | $2.19 |
| deepseek-v3 | $0.27 | $1.10 |
| deepseek-v2-5 | $0.14 | $0.28 |
| deepseek-chat | $0.27 | $1.10 |
| deepseek-r1-distill-llama-70b | $0.55 | $2.19 |

</details>

<details>
<summary><b>Mistral</b> — 8 models</summary>

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|--------------|
| mistral-large-2411 | $2.00 | $6.00 |
| mistral-small-2501 | $0.10 | $0.30 |
| mistral-nemo | $0.15 | $0.15 |
| codestral-2501 | $0.30 | $0.90 |
| pixtral-large-2411 | $2.00 | $6.00 |
| ministral-8b-2410 | $0.10 | $0.10 |
| ministral-3b-2410 | $0.04 | $0.04 |

</details>

<details>
<summary><b>Meta Llama</b> — 7 models (Together AI / Groq)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|--------------|
| meta-llama/llama-3.1-405b-instruct-turbo | $5.00 | $5.00 |
| meta-llama/llama-3.1-70b-instruct-turbo | $0.88 | $0.88 |
| meta-llama/llama-3.3-70b-instruct-turbo | $0.88 | $0.88 |
| llama-3.3-70b-versatile (Groq) | $0.59 | $0.79 |
| llama-3.1-8b-instant (Groq) | $0.05 | $0.08 |

</details>

<details>
<summary><b>xAI Grok, Cohere, Perplexity, Amazon, Qwen, AI21, Writer</b> — 30+ models</summary>

| Provider | Model | Input ($/1M) | Output ($/1M) |
|----------|-------|-------------|--------------|
| xAI | grok-3 | $3.00 | $15.00 |
| xAI | grok-3-mini | $0.30 | $0.50 |
| xAI | grok-2-1212 | $2.00 | $10.00 |
| Cohere | command-r-plus-08-2024 | $2.50 | $10.00 |
| Cohere | command-r-08-2024 | $0.15 | $0.60 |
| Perplexity | sonar-pro | $3.00 | $15.00 |
| Perplexity | sonar | $1.00 | $1.00 |
| Amazon | amazon.nova-pro-v1:0 | $0.80 | $3.20 |
| Amazon | amazon.nova-lite-v1:0 | $0.06 | $0.24 |
| Amazon | amazon.nova-micro-v1:0 | $0.035 | $0.14 |
| Qwen | qwen-max | $1.60 | $6.40 |
| Qwen | qwen-turbo | $0.05 | $0.20 |
| AI21 | jamba-1.5-large | $2.00 | $8.00 |
| Writer | palmyra-x-004 | $5.00 | $15.00 |

</details>

---

## HTTP API mode

Run the TUI **and** a local HTTP API server simultaneously with `--serve`:

```bash
llm-dash --demo --serve 8080
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/summary` | GET | JSON cost summary (total, by-model, projections) |
| `/api/export.json` | GET | Full ledger as pretty-printed JSON download |
| `/api/export.csv` | GET | Full ledger as CSV download |

Enables dashboard-as-a-service: forward LLM logs to `llm-dash` and poll
aggregated costs from CI scripts or monitoring dashboards.

---

## Webhook alerts

Send budget alerts to Slack, Discord, or any HTTP endpoint:

```bash
llm-dash --budget 50 \
  --webhook-url "https://hooks.slack.com/services/T.../B.../xxx" \
  --webhook-threshold 40 \
  --webhook-format slack
```

Alert kinds fired automatically: `BudgetWarning`, `BudgetExceeded`,
`CostAnomaly`, `DailySpendSpike`. Each kind has a configurable cooldown
(default 5 min) to prevent alert floods.

---

## CLI reference

```
llm-dash [OPTIONS]

Options:
  --budget <BUDGET>             Monthly budget limit in USD [default: 10.0]
  --log-file <LOG_FILE>         NDJSON log file to read on startup
  --demo                        Pre-load built-in demo data
  --serve <PORT>                Start HTTP API server alongside the TUI
  --webhook-url <URL>           Webhook URL for budget alerts (repeatable)
  --webhook-threshold <USD>     Alert threshold in USD (default: 80% of budget)
  --webhook-format <FORMAT>     Payload format: "slack" or "generic"
  -h, --help                    Print help
  -V, --version                 Print version
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset ledger |
| `d` | Load demo data |
| `E` | Export to timestamped CSV / JSON file |
| `j` / `k` | Scroll requests table |
| `↑` / `↓` | Scroll requests table |

---

## Configuration reference

### Environment variables

| Variable      | Description                                              | Default   |
|---------------|----------------------------------------------------------|-----------|
| `RUST_LOG`    | Tracing log level (`error`, `warn`, `info`, `debug`)     | `info`    |

Tracing output is written to **stderr** so it does not interfere with piped
stdin/stdout.

### Feature flags

| Feature    | Default | Description                                            |
|------------|---------|--------------------------------------------------------|
| `webhooks` | on      | Enables `reqwest` + TLS for webhook alert delivery     |

Disable with `--no-default-features` to produce a smaller, TLS-free binary.

---

## Anomaly detection setup

The `CostAnomalyDetector` in `src/anomaly.rs` uses a rolling Z-score algorithm
to flag requests whose cost deviates significantly from recent history.

### How it works

1. A sliding window of the last N request costs is maintained (default: 50).
2. For each new request the Z-score is computed: `(cost - mean) / std_dev`.
3. If `|Z| > threshold` (default: 3.0 sigma) an `AnomalyEvent` is returned.
4. The first two observations never trigger an alert (std dev is undefined).

### Embedding in your application

```rust
use llm_cost_dashboard::anomaly::CostAnomalyDetector;

// Window of 50 recent requests, flag anything beyond 3 standard deviations.
let mut detector = CostAnomalyDetector::new(50, 3.0);

// Feed each completed request cost. Returns Some(AnomalyEvent) on a spike.
if let Some(event) = detector.observe("gpt-4o", request_cost_usd) {
    eprintln!(
        "ANOMALY: model={} cost=${:.6} z={:.2} (mean=${:.6} std=${:.6})",
        event.model, event.cost_usd, event.z_score,
        event.window_mean, event.window_std
    );
}
```

### Tuning

| Parameter     | Guidance                                                      |
|---------------|---------------------------------------------------------------|
| `window_size` | Larger window = more stable baseline; 30-100 is typical       |
| `threshold`   | 2.0 = sensitive (more alerts), 4.0 = conservative (fewer)     |

---

## Forecast setup

The `SpendForecaster` in `src/forecast.rs` fits an OLS linear regression line
to `(unix_timestamp_secs, cumulative_cost_usd)` pairs and projects spend to the
end of the current calendar month.

### How it works

1. Record `(timestamp, cumulative_cost)` pairs as each request completes.
2. Call `forecast(budget_limit)` at any time to get a `ForecastResult`.
3. The result includes projected month-end spend, projected daily rate, days
   until the budget is hit, an R² confidence score, and a trend classification
   (Accelerating / Stable / Decelerating).

### Embedding in your application

```rust
use llm_cost_dashboard::forecast::SpendForecaster;
use std::time::{SystemTime, UNIX_EPOCH};

let mut forecaster = SpendForecaster::new();

// Call this each time a request completes.
fn record_request(forecaster: &mut SpendForecaster, cumulative_usd: f64) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    forecaster.record(ts, cumulative_usd);
}

// Project spend at any time.
if let Some(result) = forecaster.forecast(Some(100.0)) {
    println!("Month-end projection: ${:.2}", result.projected_month_end_usd);
    println!("Daily rate:           ${:.4}/day", result.projected_daily_usd);
    println!("R² confidence:        {:.2}", result.confidence);
    if let Some(days) = result.days_until_budget_hit {
        println!("Budget hit in:        {days:.1} days");
    }
}
```

### Confidence interpretation

| R² range    | Meaning                                        |
|-------------|------------------------------------------------|
| 0.95 – 1.00 | Excellent fit; projection is reliable          |
| 0.80 – 0.95 | Good fit; projection is reasonable             |
| 0.50 – 0.80 | Moderate fit; treat projection as indicative   |
| < 0.50      | Poor fit; spend is noisy or highly variable    |

---

## Webhook alerting configuration

The `WebhookAlerter` in `src/alerting.rs` delivers Slack-compatible JSON
payloads to one or more HTTP(S) URLs with per-alert-kind cooldown
deduplication. Requires the `webhooks` crate feature (on by default).

### Alert kinds

| Kind                | Trigger                                                       |
|---------------------|---------------------------------------------------------------|
| `BudgetExceeded`    | Hard budget limit breached                                    |
| `BudgetWarning`     | Soft threshold (default 80%) crossed                         |
| `CostAnomaly`       | Z-score spike detected by `CostAnomalyDetector`               |
| `DailySpendSpike`   | Today's spend is N× the rolling daily average                |

### Setup

```rust
#[cfg(feature = "webhooks")]
use llm_cost_dashboard::alerting::{Alert, AlertKind, WebhookAlerter};

#[tokio::main]
async fn main() {
    let mut alerter = WebhookAlerter::new(
        vec![
            "https://hooks.slack.com/services/T.../B.../xxx".to_string(),
            "https://discord.com/api/webhooks/...".to_string(),
        ],
        300, // suppress repeat alerts of the same kind for 5 minutes
    );

    // Fire a budget warning.
    let alert = Alert::new(AlertKind::BudgetWarning {
        spent: 85.0,
        limit: 100.0,
        pct: 85.0,
    });
    alerter.fire(alert).await;
}
```

### Slack setup

1. In your Slack workspace go to **Apps > Manage > Custom Integrations > Incoming Webhooks**.
2. Create a new webhook and copy the URL.
3. Pass the URL to `WebhookAlerter::new`.

The payload format is Slack Block Kit compatible and also works with
Mattermost and Discord (Slack-compat mode).

### Cooldown behaviour

Each `AlertKind` variant has a stable cooldown key. A second `BudgetWarning`
fired within `cooldown_secs` of the first will be silently dropped. This
prevents alert floods during sustained budget overruns.

---

## Org Hierarchy Budgets

Model your company's LLM spend as an **org → team → project** tree.
Spend recorded at the project level automatically rolls up to the parent
team and the top-level org.  Any node can trigger a soft alert when its
threshold is crossed; a hard limit blocks spend at that level.

```rust,no_run
use llm_cost_dashboard::budget::hierarchy::{OrgTree, TeamConfig, ProjectConfig};

let mut tree = OrgTree::new("AcmeCorp", 1_000.0, 0.80); // $1k org limit, alert at 80%

tree.add_team(TeamConfig { name: "platform".into(), limit_usd: 400.0, alert_threshold: 0.75 });
tree.add_team(TeamConfig { name: "product".into(),  limit_usd: 500.0, alert_threshold: 0.75 });

tree.add_project(ProjectConfig {
    team: "platform".into(),
    name: "embeddings-prod".into(),
    limit_usd: 200.0,
    alert_threshold: 0.90,
}).unwrap();

// Record $45 spent by platform/embeddings-prod
let alerts = tree.spend("platform", "embeddings-prod", 45.0).unwrap();
for alert in &alerts {
    println!("[BUDGET ALERT] {}: {:.1}% consumed", alert.path, alert.fill * 100.0);
}

// Roll-up summary
let summary = tree.summary();
println!("Org total: ${:.2} / ${:.2}", summary.org_spent_usd, summary.org_limit_usd);
for team in &summary.teams {
    println!("  {} {:.1}%:", team.name, team.fill * 100.0);
    for proj in &team.projects {
        println!("    {} ${:.2}", proj.name, proj.spent_usd);
    }
}

// Find teams burning through budget fastest
let hot_teams = tree.teams_over_threshold(0.70);

// Monthly rollover
tree.reset_all();
```

---

## Library usage

The crate exposes its core types as `llm_cost_dashboard` for embedding cost
tracking directly in your Rust application:

```rust
use llm_cost_dashboard::{CostLedger, CostRecord};

let mut ledger = CostLedger::new();
let record = CostRecord::new("gpt-4o-mini", "openai", 512, 256, 34);
ledger.add(record).expect("valid record");
println!("total: ${:.6}", ledger.total_usd());
println!("projected/mo: ${:.2}", ledger.projected_monthly_usd(1));
```

### Key types

| Type                  | Module     | Description                                       |
|-----------------------|------------|---------------------------------------------------|
| `CostRecord`          | `cost`     | Single LLM request with computed USD cost         |
| `CostLedger`          | `cost`     | Append-only ledger with aggregation helpers       |
| `ModelStats`          | `cost`     | Per-model aggregated statistics                   |
| `BudgetEnvelope`      | `budget`   | Hard limit + alert threshold spend tracker        |
| `OrgTree`             | `budget::hierarchy` | Three-level org→team→project budget tree with automatic spend roll-up |
| `BudgetAlert`         | `budget::hierarchy` | Alert emitted when any hierarchy node crosses its threshold |
| `CostAnomalyDetector` | `anomaly`  | Rolling Z-score spike detector                    |
| `AnomalyEvent`        | `anomaly`  | Event emitted when an anomaly is detected         |
| `SpendForecaster`     | `forecast` | OLS linear regression spend projector             |
| `ForecastResult`      | `forecast` | Month-end projection with confidence and trend    |
| `WebhookAlerter`      | `alerting` | Slack-compatible webhook delivery with cooldown   |
| `Alert`               | `alerting` | Structured alert with id, timestamp, and message  |
| `AlertKind`           | `alerting` | Alert category enum                               |
| `LogEntry`            | `log`      | Raw log entry (model, tokens, latency)            |
| `RequestLog`          | `log`      | Ordered log with JSON ingestion                   |
| `TraceSpan`           | `trace`    | Distributed trace span with cost annotation       |
| `SpanStore`           | `trace`    | In-memory span store                              |
| `DashboardError`      | `error`    | Unified error type                                |
| `App`                 | `ui`       | Full TUI application state                        |

---

## Architecture

```
src/
  main.rs          # CLI entry point (clap + tracing init)
  lib.rs           # Public re-exports
  error.rs         # DashboardError (thiserror)
  anomaly.rs       # CostAnomalyDetector -- rolling Z-score spike detection
  forecast.rs      # SpendForecaster -- OLS linear regression month-end projection
  alerting.rs      # WebhookAlerter -- Slack-compatible alerts with cooldown
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

# Build without webhook TLS dependency
cargo build --release --no-default-features

# Documentation
cargo doc --open
```

---

## Troubleshooting

### Dashboard shows $0.00 for everything

Ensure your log records include non-zero `input_tokens` or `output_tokens` and
that the `model` field matches a known model name (or accepts the fallback
pricing). Run `llm-dash --demo` to confirm the TUI itself is working.

### Malformed JSON lines are silently skipped

By default bad lines emit a `WARN` tracing event. Set `RUST_LOG=warn` or
`RUST_LOG=debug` to see them on stderr:

```bash
RUST_LOG=warn llm-dash --log-file requests.log
```

### Webhook alerts are not firing

1. Confirm the `webhooks` feature is enabled (it is by default):
   `cargo build --features webhooks`
2. Check that the URL is reachable from your machine.
3. Watch the tracing output for `webhook delivery failed` warnings:
   `RUST_LOG=warn llm-dash ...`
4. Verify the cooldown period has elapsed -- the same alert kind will not fire
   more than once per `cooldown_secs` seconds.

### Anomaly detector fires on every request initially

This is expected behaviour while the window is filling up. The detector
requires at least 2 observations and only becomes meaningful after roughly
`window_size / 2` observations, at which point the mean and standard deviation
stabilise.

### Forecaster returns None

`SpendForecaster::forecast` requires at least two `(timestamp, cost)`
observations. Record a second observation before calling `forecast`.

### Binary not found after cargo install

Ensure `~/.cargo/bin` is on your `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

---

## Cost Forecasting & Anomaly Detection

The dashboard includes a real-time rolling-window spend forecaster built on linear
regression, complementing the existing OLS month-end projector:

| Feature | Description |
|---------|-------------|
| `TrendForecaster` | 24-hour rolling window, OLS slope → projected daily + monthly |
| `TrendDirection` | `Up` / `Down` / `Flat` — arrow shown in the Forecast panel |
| Anomaly detection | Flags requests where spend > mean + 2σ of the 24-hour window |
| Budget breach ETA | Estimates time until monthly budget is hit at current rate |
| `SeasonalAdjustment` | Hour-of-day × day-of-week multiplicative factor on forecasts |

The **Forecast** panel is displayed in the TUI left column between the Budget gauge
and the Cache Breakdown panel.

```
 Forecast
 Proj/day:  $0.002341   ↑
 Proj/mo:   $0.7023
 Confidence:  62%
 Anomaly: none
```

---

## API Key Validation

On startup, or on demand, the dashboard can validate your provider API keys against
each provider's live model-list endpoint.  No credentials are stored.

```rust
use llm_cost_dashboard::validator::{MultiValidator, KeyConfig};

let validator = MultiValidator::new(vec![
    KeyConfig { provider: "anthropic".into(), key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default() },
    KeyConfig { provider: "openai".into(),    key: std::env::var("OPENAI_API_KEY").unwrap_or_default() },
    KeyConfig { provider: "google".into(),    key: std::env::var("GOOGLE_API_KEY").unwrap_or_default() },
]);
let results = validator.validate_all().await;
for r in &results {
    println!("{}", r.status_str());
}
```

Requires the `webhooks` feature (enabled by default).  Validator per provider:

| Provider | Endpoint | Auth method |
|----------|----------|-------------|
| Anthropic | `https://api.anthropic.com/v1/models` | `x-api-key` header |
| OpenAI | `https://api.openai.com/v1/models` | `Authorization: Bearer` |
| Google | `https://generativelanguage.googleapis.com/v1/models?key=` | Query param |

---

## Provider Auto-Detection

When ingesting log lines, `LogEntry` can auto-detect the provider from HTTP
response headers supplied alongside the record.  Call
`entry.apply_header_detection(headers)` with an iterator of `(name, value)`
pairs after parsing:

```rust
use llm_cost_dashboard::LogEntry;

let mut entry = LogEntry::new("claude-sonnet-4-6", "unknown", 512, 256, 45);
entry.apply_header_detection([
    ("x-ratelimit-limit-tokens", "50000"),
]);
assert_eq!(entry.effective_provider(), "anthropic");
```

Detection rules applied in priority order:

| Header | Provider |
|--------|----------|
| `x-ratelimit-limit-tokens` | Anthropic |
| `x-goog-request-params` | Google / Gemini |
| `x-request-id` (UUID-shaped) | OpenAI |

The detected provider is stored in `LogEntry::detected_provider` and
automatically upgrades `provider` from `"unknown"` when a match is found.

---

## Data Export

Press **`e`** in the TUI to export the current session's cost data to disk.
Two timestamped files are written to the current working directory:

```
llm-costs-20260322-120000.json
llm-costs-20260322-120000.csv
```

You can also call the export API directly:

```rust
use llm_cost_dashboard::export::{CostExporter, ExportFormat};
use llm_cost_dashboard::CostLedger;

let ledger = CostLedger::new();
let exporter = CostExporter::new(&ledger);
let filename = exporter.export(ExportFormat::Csv).unwrap();
println!("Exported to {filename}");
```

The `export` module also provides lower-level path-based helpers:
`export_csv`, `export_json`, and `export_summary_json`.

---

## Related projects by @Mattbusel

- [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- Rust async LLM pipeline orchestration
- [rot-signals-api](https://github.com/Mattbusel/rot-signals-api) -- Options signal REST API
- [prompt-observatory](https://github.com/Mattbusel/prompt-observatory) -- LLM interpretability dashboard

---

## License

MIT -- see [LICENSE](LICENSE) for details.
