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

## What's New

### Cost Prediction Engine

The new `prediction` module provides a 24×7 pattern-matrix cost predictor trained on historical [`TaggedRequest`] data.

**Key types:** `CostPredictor`, `UsagePattern`, `PatternMatrix`, `WeeklyHeatmap`

**Features:**
- `CostPredictor::learn(requests)` — fills the 168-cell pattern matrix using online mean updates (Welford algorithm); multiple calls are additive
- `CostPredictor::predict_next_hour()` — looks up the current UTC (hour, day-of-week) cell
- `CostPredictor::predict_next_n_hours(n)` — rolling predictions for the next N hours as `Vec<(DateTime<Utc>, f64)>`
- `CostPredictor::confidence(hour, day)` — returns `min(1.0, sample_count / 10.0)`; low-sample cells are flagged as low-confidence
- `WeeklyHeatmap::render_ascii()` — prints a 7×24 ASCII grid using block intensity characters (`.`, `░`, `▒`, `▓`, `█`, `■`)

### Cost Diff Reporter

The new `diff` module compares two named `PeriodSnapshot`s and renders a Markdown diff report.

**Key types:** `CostDiff`, `PeriodSnapshot`, `DiffReport`, `ModelDiff`

**Features:**
- `CostDiff::compare(baseline, current)` — computes absolute/percentage change, new/removed models, per-model diffs sorted by magnitude
- `DiffReport::render_markdown()` — formatted Markdown table with ↑↓ arrows and per-model breakdown section
- **CLI:** `llm-dash --diff <period1> <period2>` — loads log/demo data, buckets by date prefix, and prints the Markdown diff

```
$ llm-dash --demo --diff 2024-01-01 2024-01-08
## Cost Diff: 2024-01-01 → 2024-01-08

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| Total cost | $10.2340 | $14.8900 | ↑ $4.6560 (+45.5%) |
...
```

### Budget Planner

The new `budget::planner` module provides period-aware budget planning with percentage-based allocation splits, reconciliation against actuals, and linear spend forecasting.

**Key types:** `BudgetPlanner`, `BudgetPlan`, `BudgetPeriod`, `BudgetAllocation`, `AllocationStatus`, `BudgetForecast`

**Periods:** `Daily`, `Weekly`, `Monthly`, `Quarterly`

**Status thresholds:**
- `< 80 %` of budget → `OnTrack`
- `80 – 99 %` → `AtRisk { pct_used }`
- `≥ 100 %` → `Exceeded { overage_usd }`

```rust
use llm_cost_dashboard::budget::planner::{BudgetPeriod, BudgetPlanner};
use std::collections::HashMap;

// Split a $1000/month budget
let mut plan = BudgetPlanner::create(
    BudgetPeriod::Monthly,
    1000.0,
    vec![
        ("GPT-4o".to_string(), 60.0),   // $600
        ("Claude".to_string(), 40.0),   // $400
    ],
);

// Fill in actuals mid-month
let mut actuals = HashMap::new();
actuals.insert("GPT-4o".to_string(), 490.0);
actuals.insert("Claude".to_string(), 180.0);
BudgetPlanner::reconcile(&mut plan, &actuals);

// Linear forecast at 50% elapsed
let fc = BudgetPlanner::forecast(&plan, 0.5);
println!("Projected: ${:.2} (on_track={})", fc.projected_total_usd, fc.on_track);
```

---

### Multi-Tenant Cost Isolation

The new `tenant` module provides per-tenant cost tracking, quota enforcement, top-model analysis, and daily spend breakdowns.

**Key types:** `TenantIsolator`, `Tenant`, `TenantLedger`, `TenantReport`

- **Registration** — `add_tenant(Tenant { id, name, quota_usd, tags })`
- **Recording** — `record(tenant_id, cost_usd, model_id)` appends a timestamped event
- **Quota tracking** — `quota_remaining(tenant_id)` returns `Option<f64>` (negative when exceeded)
- **Over-quota scan** — `over_quota()` returns all tenants whose spend exceeds their limit
- **Reports** — `report_all()` produces `Vec<TenantReport>` with top-5 models and daily breakdown per tenant

**CLI flags** (in `llm-dash`):
- `--tenant <id>` — filter all output to a single tenant
- `--tenant-report` — print a summary table of all tenant spend vs. quota

```rust
use llm_cost_dashboard::tenant::{Tenant, TenantIsolator};

let mut isolator = TenantIsolator::new();
isolator.add_tenant(Tenant {
    id: "platform".to_string(),
    name: "Platform Team".to_string(),
    quota_usd: Some(500.0),
    tags: vec!["internal".to_string()],
});

isolator.record("platform", 12.50, "gpt-4o");
isolator.record("platform", 7.00, "claude-3-5-sonnet");

for r in isolator.report_all() {
    println!("{}: ${:.2} used of ${:.2} quota",
        r.tenant.name, r.total_cost_usd,
        r.quota_usd.unwrap_or(0.0));
}
```

---

### Cost Allocation Tagging

The `tagging` module now includes a `TagStore` — an append-only, indexed store of `TaggedRequest` records with structured `CostTag` key/value labels for FinOps cost attribution.

**Key types:** `CostTag`, `TaggedRequest`, `TagStore`, `TagFilter`, `GroupStats`, `TagReport`

```rust
use llm_cost_dashboard::tagging::{CostTag, TagStore, TagFilter, TagReport, TaggedRequest};
use chrono::Utc;

let mut store = TagStore::new();
store.push(TaggedRequest {
    request_id: 1,
    model_id: "gpt-4o-mini".to_string(),
    cost_usd: 0.015,
    tokens_in: 512,
    tokens_out: 256,
    tags: vec![
        CostTag::new("project", "recommendation-engine"),
        CostTag::new("env", "production"),
    ],
    timestamp: Utc::now(),
});

// Query by tag
let results = store.query(&TagFilter {
    key: Some("env".to_string()),
    value: Some("production".to_string()),
    ..Default::default()
});

// Group by tag dimension
let groups = store.group_by("project");

// Top-10 report by cost
let report = TagReport::generate(&store, "project");
```

### Export Engine

The `export` module now includes an `Exporter` that writes `TaggedRequest` slices to CSV, JSON, JSONL, or Markdown via `std::io::Write` (streaming-friendly).

**Key types:** `Exporter`, `ExportError`, `ExportFormat` (extended with `Jsonl`, `Markdown`)

| Format | Description |
|--------|-------------|
| `csv` | Header row + one row per request; all tag keys as extra columns |
| `json` | `{ "requests": [...], "summary": { total_cost, total_tokens, count } }` |
| `jsonl` | One JSON object per line (streaming-friendly) |
| `markdown` | Summary table sorted by cost descending |

**CLI flags:** `--export csv|json|jsonl|markdown --out <file>`

```bash
# Export tagged requests as JSONL
llm-dash --demo --export jsonl --out costs.jsonl

# Export as Markdown to stdout
llm-dash --demo --export markdown
```

```rust
use llm_cost_dashboard::export::{Exporter, ExportFormat};

let mut buf = Vec::new();
Exporter::export(&requests, ExportFormat::Jsonl, &mut buf).unwrap();
```

---

## What is this?

`llm-cost-dashboard` is a Rust terminal application (TUI) that reads a stream of
LLM request records from a log file or stdin, computes per-request USD costs from a
built-in pricing table, and renders a live dashboard showing total spend, per-model
breakdowns, budget status, and projected monthly bills. It includes a rolling
Z-score anomaly detector, an OLS forecaster, a Holt-Winters exponential smoother for
short-horizon projections, side-by-side multi-provider cost comparison across 80+
models, session tracking, CSV/JSON export, and a Slack-compatible webhook alerter.
Everything runs locally -- no cloud accounts, no telemetry, no databases.

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

## FinOps Cost Tagging

The `tagging` module adds structured tag-based cost attribution so teams can
track LLM spend by project, team, cost centre, environment, or any custom
dimension — without changing your existing log format.

### Tag sources (applied in order)

1. **Passthrough fields** — copy log fields directly as tags (e.g. `project`, `team`).
2. **Default tags** — always present (e.g. `env=production`).
3. **Rule-derived tags** — `TagRule` maps field patterns to tag key-value pairs.
4. **Caller overrides** — win on conflict with all of the above.

### Example

```rust
use llm_cost_dashboard::tagging::{TagEngine, TagRule, TagMatch, TaggedLedger};
use std::collections::HashMap;

let mut engine = TagEngine::new();

// Always tag with environment.
engine.add_default_tag("env", "production");

// Pass the "project" field from logs straight through as a tag.
engine.add_passthrough("project");

// Derive provider from the model name.
engine.add_rule(TagRule {
    field: "model".to_string(),
    pattern: TagMatch::Contains("claude".to_string()),
    tag_key: "provider".to_string(),
    tag_value: "anthropic".to_string(),
});

// Resolve tags for an incoming log record.
let mut fields = HashMap::new();
fields.insert("model".to_string(), "claude-sonnet-4-6".to_string());
fields.insert("project".to_string(), "recommendation-engine".to_string());

let tags = engine.resolve(&fields);
assert_eq!(tags.get("provider").map(|s| s.as_str()), Some("anthropic"));
assert_eq!(tags.get("project").map(|s| s.as_str()), Some("recommendation-engine"));
```

### Cost roll-up by tag dimension

```rust
use llm_cost_dashboard::tagging::TaggedLedger;

let mut ledger = TaggedLedger::new();
ledger.add(0.10, unix_ts, tags_for_search_team);
ledger.add(0.20, unix_ts, tags_for_search_team);
ledger.add(0.05, unix_ts, tags_for_billing_team);

let by_team = ledger.by_tag("team");
// Sorted from highest to lowest spend:
for (team, total_usd) in by_team.ranked() {
    println!("{team}: ${total_usd:.4}");
}

if let Some((top, cost)) = by_team.top_spender() {
    println!("Top spender: {top} at ${cost:.4}");
}
```

---

## Model Recommendation Engine

The `recommendations` module analyses your usage patterns and suggests cheaper
alternative models, computing projected monthly savings for each switch.

```rust
use llm_cost_dashboard::{CostLedger, CostRecord};
use llm_cost_dashboard::recommendations::ModelRecommender;

let mut ledger = CostLedger::new();
// ... populate ledger from your request logs ...

let recommender = ModelRecommender::new(&ledger);

// Print all recommendations, sorted by largest projected saving.
for s in recommender.suggest() {
    println!("{}", s.summary_line());
    // Example output: "claude-sonnet-4-6 → claude-haiku-4-5 | save 92% ($18.40/mo)"
}

println!(
    "Total potential saving: ${:.2}/mo",
    recommender.total_projected_monthly_saving()
);
```

**Supported current models:** claude-opus-4-6, claude-sonnet-4-6, gpt-4o,
gpt-4-turbo, o1, o3, gemini-1.5-pro, gpt-4.5-preview, mistral-large-2411, and more.

---

## Team Cost Allocation

The `allocation` module maps requests to team/project buckets using a priority-ordered
rule list and supports chargeback/showback reporting workflows.

```rust
use llm_cost_dashboard::allocation::{AllocationEngine, AllocationRule};

let mut engine = AllocationEngine::new();

// Route sessions whose ID starts with "eng-" to the engineering team.
engine.add_rule(AllocationRule {
    rule_id: "eng-prefix".to_string(),
    team: "engineering".to_string(),
    project: "infra".to_string(),
    session_prefix: Some("eng-".to_string()),
    tag_key: None,
    tag_value: None,
});

// Route by metadata tag for other teams.
engine.add_rule(AllocationRule {
    rule_id: "data-tag".to_string(),
    team: "data-science".to_string(),
    project: "ml-platform".to_string(),
    session_prefix: None,
    tag_key: Some("team".to_string()),
    tag_value: Some("data".to_string()),
});

let allocation = engine.allocate("eng-alice-session-42", &tags);
println!("Team: {} / Project: {}", allocation.team, allocation.project);
```

---

## Automated Export Scheduling

The `scheduler` module runs cron-based scheduled cost exports so you always
have up-to-date CSV/JSON reports without manual intervention.

```rust
use llm_cost_dashboard::scheduler::{ExportSchedule, ScheduledExportFormat, Scheduler};
use llm_cost_dashboard::cost::CostLedger;

let ledger = CostLedger::new();
let mut scheduler = Scheduler::new(ledger);

// Export CSV every Monday at 09:00.
scheduler.add_schedule(ExportSchedule {
    cron: "0 9 * * MON".to_string(),
    format: ScheduledExportFormat::Csv,
    output_dir: "/var/reports/llm".to_string(),
    label: "weekly-csv".to_string(),
});

// Export JSON summary on the 1st of every month.
scheduler.add_schedule(ExportSchedule {
    cron: "0 8 1 * *".to_string(),
    format: ScheduledExportFormat::JsonSummary,
    output_dir: "/var/reports/llm".to_string(),
    label: "monthly-summary".to_string(),
});

// Call this each tick (e.g. every minute) to fire due schedules.
let fired = scheduler.tick(chrono::Utc::now());
for report in fired {
    println!("Wrote: {}", report.path);
}
```

---

## Feature Gallery

### Multi-Provider Cost Comparison

`ProviderComparison` ranks every model in the 80+ model pricing table by projected monthly cost for your actual workload.

```rust,no_run
use llm_cost_dashboard::comparison::{ProviderComparison, WorkloadProfile};

// Derive profile from your real request history.
let profile = WorkloadProfile::from_ledger(&ledger)
    .unwrap_or_else(|| WorkloadProfile::from_rph(1000)); // 1000 req/hr fallback

let cmp = ProviderComparison::compute(&profile);

println!("Cheapest: {} at ${:.2}/mo", cmp.cheapest().model, cmp.cheapest().monthly_cost_usd);
println!("Most expensive: {} at ${:.2}/mo",
    cmp.most_expensive().model, cmp.most_expensive().monthly_cost_usd);
println!("Cost spread: {:.0}x", cmp.cost_spread_ratio());

for proj in cmp.top_n_cheapest(10) {
    println!("  {:<45} ${:8.2}/mo  ${:.4}/1k req  ({}) ",
        proj.model, proj.monthly_cost_usd, proj.cost_per_1k_requests, proj.provider);
}
```

From the CLI:

```bash
llm-dash --demo --compare --workload-rph 1000
```

The TUI shows a "Compare" tab (`c` to switch) with a full ranked table updated from your live request history.

### Holt-Winters Cost Forecasting

`CostForecaster` applies double exponential smoothing to project spend over the next hour, day, week, and month.  Unlike OLS, it adapts to changing spend rates (e.g. a marketing campaign that doubles traffic).

```rust,no_run
use llm_cost_dashboard::forecast::CostForecaster;

let mut forecaster = CostForecaster::new();  // default α=0.3, β=0.1
// Or tune the smoothing parameters:
// let mut forecaster = CostForecaster::new().with_params(0.4, 0.15);

// Feed cumulative spend observations (unix_secs, cumulative_usd).
forecaster.record(1_700_000_000.0, 0.0);
forecaster.record(1_700_003_600.0, 0.50);
forecaster.record(1_700_007_200.0, 1.05);

if let Some(hw) = forecaster.forecast(Some(100.0)) {
    println!("Next hour:  ${:.4}", hw.next_hour_usd);
    println!("Next day:   ${:.2}", hw.next_day_usd);
    println!("Next week:  ${:.2}", hw.next_week_usd);
    println!("Next month: ${:.2}", hw.next_month_usd);
    println!("80%% CI:     [{:.4}, {:.4}]",
        hw.confidence_interval.0, hw.confidence_interval.1);
    if hw.budget_warning {
        eprintln!("WARNING: forecasted spend exceeds 80%% of budget!");
    }
}
```

From the CLI:

```bash
# Print a forecast summary and exit.
llm-dash --demo --forecast
```

The TUI shows a "Projected Spend" panel that updates every tick.  When the forecast exceeds 80 % of the budget the panel turns red.

### Anomaly Detection

The `anomaly` module applies a rolling Z-score to the per-request cost time series.  Any request whose cost deviates more than 3σ from the rolling mean triggers an alert.

### Session Tracking

`session::SessionLedger` groups requests by session ID, enforces per-session budgets, and surfaces per-session spend in the TUI.

### Export

Press `e` in the TUI to dump the current session to `llm-costs-<timestamp>.json` and `llm-costs-<timestamp>.csv` in the working directory.

```bash
# Programmatic export via the HTTP API (--serve mode).
curl http://localhost:8080/api/export.json > session.json
curl http://localhost:8080/api/export.csv  > session.csv
```

### Configuration Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--budget USD` | `10.0` | Monthly hard budget limit |
| `--log-file PATH` | — | NDJSON file to tail |
| `--demo` | off | Pre-load synthetic demo data |
| `--serve PORT` | — | Start HTTP API on given port alongside TUI |
| `--webhook-url URL` | — | Slack/generic webhook destination (repeatable) |
| `--webhook-threshold USD` | 80 % of `--budget` | USD spend level that triggers webhook |
| `--webhook-format` | `generic` | `slack` or `generic` |
| `--compare` | off | Show multi-provider cost comparison panel |
| `--workload-rph N` | derived from log | Requests per hour for comparison projection |
| `--forecast` | off | Print Holt-Winters forecast summary and exit |

### Keyboard Controls (TUI)

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `r` | Reset all data |
| `d` | Load demo data |
| `e` | Export session to JSON + CSV |
| `j` / `Down` | Scroll requests down |
| `k` / `Up` | Scroll requests up |
| `c` | Switch to Compare tab |
| `f` | Switch to Forecast tab |

---

## Budget Alerts

Rule-based budget alerting with cooldown tracking, multi-window cost aggregation,
and flexible delivery channels.

### Alert Rules TOML format

```toml
[[rules]]
name        = "daily-5-usd"
threshold_usd = 5.0
window      = "daily"        # "daily" | "weekly" | "monthly"
cooldown_secs = 3600

[[rules]]
name        = "monthly-50-usd"
threshold_usd = 50.0
window      = "monthly"
cooldown_secs = 86400
```

### CLI usage

```sh
# Load rules and run one check against demo data, then start TUI.
llm-dash --demo --alerts rules.toml
```

### Library usage

```rust,no_run
use std::time::Duration;
use llm_cost_dashboard::alerts::{AlertChannel, AlertEngine, AlertRule, AlertWindow};
use llm_cost_dashboard::cost::CostLedger;

let rules = vec![AlertRule {
    name: "daily-5-usd".to_string(),
    threshold_usd: 5.0,
    window: AlertWindow::Daily,
    channel: AlertChannel::Log,
    cooldown: Duration::from_secs(3600),
}];

let mut engine = AlertEngine::new(rules);
let ledger = CostLedger::new();
let alerts = engine.check(&ledger);

let summary = engine.summary();
println!("Fired: {}  Suppressed: {}  Rules: {}",
    summary.fired_total, summary.suppressed_by_cooldown, summary.rules_count);
```

Supported channels: `AlertChannel::Log`, `AlertChannel::Webhook { url, secret }`,
`AlertChannel::File { path }`.  Webhook delivery includes an optional
`X-Alert-Signature: sha256=<hex>` HMAC header when a secret is configured.

---

## Cost Anomaly Detection

Welford online algorithm Z-score detector — finds cost spikes (overspend) and
dips (underspend / service outage) with O(1) memory and O(1) per-observation cost.

### CLI usage

```sh
# Print anomaly report and exit.
llm-dash --demo --anomaly

# Combine with log data.
llm-dash --log-file requests.ndjson --anomaly
```

### Library usage

```rust,no_run
use chrono::Utc;
use llm_cost_dashboard::anomaly::{AnomalyConfig, AnomalyDetector};

let mut detector = AnomalyDetector::new(AnomalyConfig {
    window_size: 30,     // rolling window of 30 observations
    z_threshold: 3.0,    // flag observations > 3σ from mean
    min_samples: 5,      // require at least 5 samples before firing
});

// Observe individual costs.
let result = detector.observe(0.001);
println!("Anomaly: {}  Z-score: {:.2}", result.is_anomaly, result.z_score);

// Batch analysis with timestamps.
let now = Utc::now();
let observations = vec![
    (now, 0.001), (now, 0.002), (now, 5.0), // spike!
];
let report = detector.analyze(&observations);
println!("Anomaly rate: {:.1}%  Max Z: {:.2}",
    report.anomaly_rate * 100.0, report.max_z_score);
```

---

## Related projects by @Mattbusel

- [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- Rust async LLM pipeline orchestration
- [rot-signals-api](https://github.com/Mattbusel/rot-signals-api) -- Options signal REST API
- [prompt-observatory](https://github.com/Mattbusel/prompt-observatory) -- LLM interpretability dashboard

---

## License

MIT -- see [LICENSE](LICENSE) for details.
