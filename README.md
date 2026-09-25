# llm-cost-dashboard

[![crates.io](https://img.shields.io/crates/v/llm-cost-dashboard.svg)](https://crates.io/crates/llm-cost-dashboard)
[![docs.rs](https://docs.rs/llm-cost-dashboard/badge.svg)](https://docs.rs/llm-cost-dashboard)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A terminal dashboard for LLM spend: feed it a log of your model calls and see cost per request, cost per model, budget used and a projected monthly bill, priced from a built-in table of 83 models. Runs locally, no account or database.

Token prices differ by 100x between models, and bills arrive after the fact. `llm-dash` turns a newline-delimited JSON log of requests (model, input tokens, output tokens, latency) into a live [ratatui](https://ratatui.rs) dashboard, and can also answer one-off questions from the command line: which model would be cheapest for this workload, what will this month cost, where are the spikes, what changed between two days. Everything it does is also available as a Rust library.

## Quick start

```bash
# Latest code from this repository (crates.io has the older 1.0.2 release)
cargo install --git https://github.com/Mattbusel/llm-cost-dashboard

# or the published release
cargo install llm-cost-dashboard
```

The binary is called `llm-dash`.

```bash
llm-dash --demo                               # dashboard with synthetic Claude / GPT-4o / o3-mini traffic
llm-dash --budget 50 --log-file requests.ndjson
```

Your log is one JSON object per line:

```json
{"model":"claude-sonnet-4-6","input_tokens":512,"output_tokens":256,"latency_ms":340}
{"model":"gpt-4o-mini","input_tokens":128,"output_tokens":64,"latency_ms":120,"provider":"openai"}
{"model":"gpt-4o","input_tokens":900,"output_tokens":0,"latency_ms":30000,"error":"timeout"}
```

`model`, `input_tokens`, `output_tokens` and `latency_ms` are required; `provider` and `error` are optional. Malformed lines are skipped with a warning on stderr (`RUST_LOG=warn`). Model names are matched case-insensitively; unknown models are priced at a fallback of $5 / $15 per million tokens.

## The dashboard

Panels: **Summary** (total and projected monthly spend), **Budget** gauge, **Forecast**, **Cache Breakdown**, **Cost by Model** bar chart, **Recent Requests** table, **Savings Opportunities**, and a sparkline of the last 60 request costs. The screen refreshes every 250 ms.

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `d` | Load demo data |
| `r` | Reset all data |
| `e` | Export the session to `llm-costs-<timestamp>.json` and `.csv` in the current directory |
| `j` / `k` or arrow keys | Scroll the requests table |
| `x` | Open the cost explorer (`s` cycles sort order, `Enter` toggles detail, `x` or `Esc` closes it) |

## Command-line reports

These load `--demo` and/or `--log-file` data, print a report and exit without starting the TUI:

```bash
llm-dash --demo --compare                     # rank every priced model by monthly cost for this workload
llm-dash --compare --workload-rph 1000        # same, for a hypothetical 1000 requests/hour
llm-dash --demo --anomaly                     # Z-score cost anomaly report
llm-dash --demo --export-csv costs.csv        # or --export-json costs.json
llm-dash --demo --export markdown             # csv | json | jsonl | markdown, to --out FILE or stdout
```

`--forecast` (Holt-Winters projection) and `--diff <A> <B>` (Markdown diff between two date prefixes) also exist, but see [Status](#status-and-limitations): log lines carry no timestamp, so from the CLI they currently have no time spread to work with.

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--budget <USD>` | `10.0` | Monthly budget limit |
| `--log-file <PATH>` | | NDJSON request log to load at startup |
| `--demo` | off | Pre-load demo data |
| `--serve <PORT>` | | Also start the HTTP API (below) |
| `--webhook-url <URL>` | | Slack or generic webhook for budget alerts (repeatable) |
| `--webhook-threshold <USD>` | 80% of budget | Spend level that fires the webhook |
| `--webhook-format <FORMAT>` | `generic` | `slack` or `generic` |
| `--alerts <RULES_TOML>` | | Load budget alert rules and run a background check loop |
| `--session <NAME>` | | Tag every ingested record with a session id |
| `--export-csv <PATH>`, `--export-json <PATH>` | | Write all records and exit |
| `--export <FORMAT>`, `--out <FILE>` | | Export tagged requests as csv, json, jsonl or markdown and exit |
| `--compare`, `--workload-rph <N>` | `1000` | Multi-provider cost ranking and exit |
| `--forecast` | off | Print a spend forecast and exit (needs at least 3 records) |
| `--anomaly` | off | Print an anomaly report and exit |
| `--diff <A> <B>` | | Compare two date-prefix periods and exit |

`RUST_LOG` controls log verbosity; logs go to stderr.

### HTTP API

```bash
llm-dash --demo --serve 8080
curl localhost:8080/api/summary       # JSON cost summary
curl localhost:8080/api/export.json   # full ledger as JSON
curl localhost:8080/api/export.csv    # full ledger as CSV
```

### Budget alerts

Webhook alerts (Slack or a generic JSON POST) fire when spend crosses `--webhook-threshold`:

```bash
llm-dash --budget 50 --webhook-url "https://hooks.slack.com/services/..." --webhook-threshold 40 --webhook-format slack
```

Rule files for `--alerts` look like this:

```toml
[[rules]]
name = "daily-5-usd"
threshold_usd = 5.0
window = "daily"        # daily | weekly | monthly
cooldown_secs = 3600
```

Webhook delivery needs the default `webhooks` feature; build with `--no-default-features` for a smaller binary without TLS.

## Supported models

83 models across Anthropic, OpenAI, Google, DeepSeek, Mistral, Meta Llama (Together AI and Groq), xAI, Cohere, Perplexity, Amazon Bedrock, Alibaba Qwen, Writer and AI21. Prices are USD per million tokens, from `src/cost/pricing.rs` (last updated 2026-03-22); check them against your provider before relying on the numbers.

<details>
<summary><b>Anthropic / Claude 4 family</b> (3)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `claude-opus-4-6` | $15 | $75 |
| `claude-sonnet-4-6` | $3 | $15 |
| `claude-haiku-4-5` | $0.25 | $1.25 |

</details>

<details>
<summary><b>Anthropic / Claude 3.5 family</b> (3)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `claude-3-5-sonnet-20241022` | $3 | $15 |
| `claude-3-5-haiku-20241022` | $0.8 | $4 |
| `claude-3-5-sonnet-20240620` | $3 | $15 |

</details>

<details>
<summary><b>Anthropic / Claude 3 family</b> (3)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `claude-3-opus-20240229` | $15 | $75 |
| `claude-3-sonnet-20240229` | $3 | $15 |
| `claude-3-haiku-20240307` | $0.25 | $1.25 |

</details>

<details>
<summary><b>OpenAI / GPT-4o family</b> (5)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `gpt-4o` | $5 | $15 |
| `gpt-4o-mini` | $0.15 | $0.6 |
| `gpt-4-turbo` | $10 | $30 |
| `gpt-4.5-preview` | $75 | $150 |
| `chatgpt-4o-latest` | $5 | $15 |

</details>

<details>
<summary><b>OpenAI / o-series reasoning models</b> (6)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `o1` | $15 | $60 |
| `o1-preview` | $15 | $60 |
| `o1-mini` | $1.1 | $4.4 |
| `o3` | $10 | $40 |
| `o3-mini` | $1.1 | $4.4 |
| `o4-mini` | $1.1 | $4.4 |

</details>

<details>
<summary><b>OpenAI / Legacy</b> (3)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `gpt-4` | $30 | $60 |
| `gpt-3.5-turbo` | $0.5 | $1.5 |
| `gpt-3.5-turbo-instruct` | $1.5 | $2 |

</details>

<details>
<summary><b>Google / Gemini 2 family</b> (4)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `gemini-2.5-pro` | $1.25 | $10 |
| `gemini-2.0-flash` | $0.1 | $0.4 |
| `gemini-2.0-flash-lite` | $0.075 | $0.3 |
| `gemini-2.0-flash-thinking` | $0.15 | $0.6 |

</details>

<details>
<summary><b>Google / Gemini 1.5 family</b> (3)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `gemini-1.5-pro` | $3.5 | $10.5 |
| `gemini-1.5-flash` | $0.075 | $0.3 |
| `gemini-1.5-flash-8b` | $0.0375 | $0.15 |

</details>

<details>
<summary><b>DeepSeek</b> (7)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `deepseek-r1` | $0.55 | $2.19 |
| `deepseek-v3` | $0.27 | $1.1 |
| `deepseek-v2-5` | $0.14 | $0.28 |
| `deepseek-chat` | $0.27 | $1.1 |
| `deepseek-coder` | $0.14 | $0.28 |
| `deepseek-r1-distill-llama-70b` | $0.55 | $2.19 |
| `deepseek-r1-distill-qwen-32b` | $0.55 | $2.19 |

</details>

<details>
<summary><b>Mistral</b> (10)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `mistral-large-2411` | $2 | $6 |
| `mistral-large-2407` | $3 | $9 |
| `mistral-small-2501` | $0.1 | $0.3 |
| `mistral-small-2402` | $1 | $3 |
| `mistral-nemo` | $0.15 | $0.15 |
| `codestral-2501` | $0.3 | $0.9 |
| `pixtral-large-2411` | $2 | $6 |
| `pixtral-12b-2409` | $0.15 | $0.15 |
| `ministral-8b-2410` | $0.1 | $0.1 |
| `ministral-3b-2410` | $0.04 | $0.04 |

</details>

<details>
<summary><b>Meta / Llama (via Together AI / Groq)</b> (9)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `meta-llama/llama-3.1-405b-instruct-turbo` | $5 | $5 |
| `meta-llama/llama-3.1-70b-instruct-turbo` | $0.88 | $0.88 |
| `meta-llama/llama-3.1-8b-instruct-turbo` | $0.18 | $0.18 |
| `meta-llama/llama-3.3-70b-instruct-turbo` | $0.88 | $0.88 |
| `meta-llama/llama-3.2-90b-vision-instruct-turbo` | $1.2 | $1.2 |
| `meta-llama/llama-3.2-11b-vision-instruct-turbo` | $0.18 | $0.18 |
| `llama-3.3-70b-versatile` | $0.59 | $0.79 |
| `llama-3.1-70b-versatile` | $0.59 | $0.79 |
| `llama-3.1-8b-instant` | $0.05 | $0.08 |

</details>

<details>
<summary><b>xAI Grok</b> (5)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `grok-3` | $3 | $15 |
| `grok-3-mini` | $0.3 | $0.5 |
| `grok-2-1212` | $2 | $10 |
| `grok-2-vision-1212` | $2 | $10 |
| `grok-beta` | $5 | $15 |

</details>

<details>
<summary><b>Cohere</b> (4)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `command-r-plus-08-2024` | $2.5 | $10 |
| `command-r-08-2024` | $0.15 | $0.6 |
| `command-a-03-2025` | $2.5 | $10 |
| `command-r7b-12-2024` | $0.0375 | $0.15 |

</details>

<details>
<summary><b>Perplexity</b> (4)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `sonar-pro` | $3 | $15 |
| `sonar` | $1 | $1 |
| `sonar-reasoning-pro` | $2 | $8 |
| `sonar-reasoning` | $1 | $5 |

</details>

<details>
<summary><b>Amazon (Bedrock)</b> (5)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `amazon.nova-pro-v1:0` | $0.8 | $3.2 |
| `amazon.nova-lite-v1:0` | $0.06 | $0.24 |
| `amazon.nova-micro-v1:0` | $0.035 | $0.14 |
| `amazon.titan-text-express-v1` | $0.2 | $0.6 |
| `amazon.titan-text-lite-v1` | $0.3 | $0.4 |

</details>

<details>
<summary><b>Alibaba Qwen</b> (5)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `qwen-max` | $1.6 | $6.4 |
| `qwen-plus` | $0.4 | $1.2 |
| `qwen-turbo` | $0.05 | $0.2 |
| `qwen2.5-72b-instruct` | $0.9 | $0.9 |
| `qwen2.5-7b-instruct` | $0.1 | $0.1 |

</details>

<details>
<summary><b>Writer</b> (2)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `palmyra-x-004` | $5 | $15 |
| `palmyra-x-003-instruct` | $1.5 | $2 |

</details>

<details>
<summary><b>AI21 Labs</b> (2)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `jamba-1.5-large` | $2 | $8 |
| `jamba-1.5-mini` | $0.2 | $0.4 |

</details>


## Library usage

```toml
[dependencies]
llm-cost-dashboard = { git = "https://github.com/Mattbusel/llm-cost-dashboard" }
```

The crate is `llm_cost_dashboard`. A ledger of priced requests:

```rust
use llm_cost_dashboard::{CostLedger, CostRecord};

let mut ledger = CostLedger::new();
// model, provider, input tokens, output tokens, latency ms
ledger.add(CostRecord::new("gpt-4o-mini", "openai", 512, 256, 34))?;
ledger.add(CostRecord::new("claude-sonnet-4-6", "anthropic", 1200, 400, 900))?;
println!("total: ${:.6}", ledger.total_usd());
println!("projected per month: ${:.2}", ledger.projected_monthly_usd(1));
```

Which model would be cheapest for this traffic:

```rust
use llm_cost_dashboard::comparison::{ProviderComparison, WorkloadProfile};
use llm_cost_dashboard::CostLedger;

let ledger = CostLedger::new(); // your populated ledger
let profile = WorkloadProfile::from_ledger(&ledger).unwrap_or_else(|| WorkloadProfile::from_rph(1000));
let cmp = ProviderComparison::compute(&profile);
for p in cmp.top_n_cheapest(5) {
    println!("{:<40} ${:>8.2}/mo  ({})", p.model, p.monthly_cost_usd, p.provider);
}
```

Org, team and project budgets with roll-up and alerts:

```rust
use llm_cost_dashboard::budget::hierarchy::{OrgTree, ProjectConfig, TeamConfig};

let mut tree = OrgTree::new("AcmeCorp", 1_000.0, 0.80); // $1k org limit, alert at 80%
tree.add_team(TeamConfig { name: "platform".into(), limit_usd: 400.0, alert_threshold: 0.75 });
tree.add_project(ProjectConfig {
    team: "platform".into(),
    name: "embeddings-prod".into(),
    limit_usd: 200.0,
    alert_threshold: 0.90,
})?;

for alert in tree.spend("platform", "embeddings-prod", 190.0)? {
    println!("[BUDGET ALERT] {}: {:.1}% used", alert.path, alert.fill * 100.0);
}
let summary = tree.summary();
println!("org: ${:.2} of ${:.2}", summary.org_spent_usd, summary.org_limit_usd);
```

Cost spikes, with a rolling Z-score detector:

```rust
use llm_cost_dashboard::anomaly::CostAnomalyDetector;

let mut detector = CostAnomalyDetector::new(50, 3.0); // window of 50, flag beyond 3 sigma
for cost in [0.001, 0.0012, 0.0009, 0.0011, 0.001, 0.25] {
    if let Some(event) = detector.observe("gpt-4o", cost) {
        println!("spike: ${:.4} (z = {:.1})", event.cost_usd, event.z_score);
    }
}
```

Spend forecasting from `(unix_seconds, cumulative_usd)` observations, by linear regression (`SpendForecaster`) or Holt-Winters smoothing (`CostForecaster`):

```rust
use llm_cost_dashboard::forecast::{CostForecaster, SpendForecaster};

let mut ols = SpendForecaster::new();
let mut hw = CostForecaster::new();
for (ts, total) in [(1_700_000_000.0, 0.0), (1_700_003_600.0, 0.50), (1_700_007_200.0, 1.05)] {
    ols.record(ts, total);
    hw.record(ts, total);
}
if let Some(f) = ols.forecast(Some(100.0)) {
    println!("month-end ${:.2}, R^2 {:.2}", f.projected_month_end_usd, f.confidence);
}
if let Some(f) = hw.forecast(Some(100.0)) {
    println!("next day ${:.2}, next month ${:.2}", f.next_day_usd, f.next_month_usd);
}
```

FinOps tags on requests, with grouping and reports:

```rust
use chrono::Utc;
use llm_cost_dashboard::tagging::{CostTag, TagFilter, TagReport, TagStore, TaggedRequest};

let mut store = TagStore::new();
store.push(TaggedRequest {
    request_id: 1,
    model_id: "gpt-4o-mini".to_string(),
    cost_usd: 0.015,
    tokens_in: 512,
    tokens_out: 256,
    tags: vec![CostTag::new("project", "search"), CostTag::new("env", "production")],
    timestamp: Utc::now(),
});
let prod = store.query(&TagFilter { key: Some("env".into()), value: Some("production".into()), ..Default::default() });
let by_project = store.group_by("project");
let report = TagReport::generate(&store, "project");
```

Other public modules include `budget::planner` (period budgets split by percentage, reconciled against actuals), `tenant` (per-tenant quotas and reports), `alerts` (TOML rule engine behind `--alerts`), `alerting` and `webhook` (Slack and generic webhooks with cooldowns), `validator` (checks Anthropic, OpenAI and Google API keys against their model-list endpoints), `recommendations` (cheaper-model suggestions), `session`, `export`, `trends`, `model_compare`, `prediction` and `diff`. See the rustdoc (`cargo doc --open`) for their APIs.

## How it works

```
src/
  main.rs            CLI (clap): TUI launch and the one-shot report flags
  ui/                ratatui app state, event loop, dashboard layout, widgets, cost explorer
  log/               NDJSON parsing into LogEntry
  cost/              CostRecord, CostLedger, pricing table (pricing.rs)
  budget/            budget envelope, org/team/project hierarchy, planner
  comparison.rs      multi-provider monthly cost ranking
  forecast.rs        OLS and Holt-Winters forecasters
  anomaly.rs         rolling and Welford Z-score detectors
  api/               axum server for --serve
  alerting.rs, alerts.rs, webhook/   alert rules and delivery
  export.rs          CSV / JSON / JSONL / Markdown export
  ...                about 60 further analysis modules (tagging, tenants, allocation, carbon, SLA, and more)
tests/, benches/     integration tests and criterion benchmarks
```

## Status and limitations

- Log files are read once at startup; the dashboard does not follow a file as it grows and does not read stdin. Restart `llm-dash` to pick up new lines.
- Log lines have no timestamp field, so every record is stamped with the time it was loaded. As a result `--forecast` on a log file or demo data has no time spread and prints `inf`, and `--diff` finds no records for past dates. The forecasting and diff APIs work when you supply real timestamps from Rust.
- Prices are a static table; verify them against your provider's current pricing.
- The crate is large (about 80 modules). The core path (ledger, pricing, TUI, export, comparison, HTTP API) is what the binary uses; many analysis modules are library-only.
- The test suite currently has failures: `cargo test` reports 967 passed and 9 failed in the library tests, including the Welford anomaly detector (`anomaly::welford_tests`), so CI is red.

```bash
cargo test
cargo bench
RUST_LOG=debug cargo run -- --demo
```

## License

MIT, see [LICENSE](LICENSE).
