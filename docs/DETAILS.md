# llm-cost-dashboard: details

Everything cut from the short [README](../README.md), kept word for word.

<!-- outcome-block -->
**Prices every LLM call from the logs you already have. Before the invoice shows up.**

No proxy. It reads plain log lines and saved OpenAI and Anthropic responses. Spend, budget and forecasts in a terminal or Prometheus.

If it doesn't do what this README says, [open an issue](https://github.com/Mattbusel/llm-cost-dashboard/issues) and I'll fix it.
<!-- outcome-block -->

[![crates.io](https://img.shields.io/crates/v/llm-cost-dashboard.svg)](https://crates.io/crates/llm-cost-dashboard)
[![docs.rs](https://docs.rs/llm-cost-dashboard/badge.svg)](https://docs.rs/llm-cost-dashboard)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitlab.com/mattbusel/llm-cost-dashboard/-/blob/master/LICENSE)
![MSRV 1.88](https://img.shields.io/badge/MSRV-1.88-blue.svg)

<p align="center">
  <img src="https://gitlab.com/mattbusel/llm-cost-dashboard/-/raw/master/assets/dashboard.gif" alt="llm-dash tailing a request log: new requests stream in, a 60,000-token gpt-4o call is flagged as a 26.7x cost anomaly, then the cost explorer sorts by price and opens that call" width="100%">
</p>

```rust
use llm_cost_dashboard::ingest::Ingester;

fn main() -> Result<(), llm_cost_dashboard::DashboardError> {
    let mut costs = Ingester::new();
    // Any mix of your own log lines and saved OpenAI / Anthropic responses.
    costs.ingest_line(r#"{"model":"gpt-4o-mini","input_tokens":5200,"output_tokens":400}"#)?;
    costs.ingest_line(r#"{"model":"claude-sonnet-4-6","usage":{"input_tokens":900,"output_tokens":350,"cache_read_input_tokens":40000}}"#)?;
    for (model, s) in costs.ledger().by_model() {
        println!("{model}: ${:.6} over {} call(s)", s.total_cost_usd, s.request_count);
    }
    println!("total: ${:.6}", costs.ledger().total_usd()); // total: $0.020970
    Ok(())
}
```

(`cargo run --example quickstart` runs exactly this.)

## Why this

| You want | llm-cost-dashboard | Alternatives |
|---|---|---|
| Price calls from logs you already have | Plain NDJSON, saved OpenAI or Anthropic responses, Claude Code logs. Cache reads and 5-minute / 1-hour cache writes priced separately | [ccstat](https://crates.io/crates/ccstat) reads coding-assistant logs only (Claude, Codex, ...) and is a CLI, not a library |
| A price table you can trust and extend | 105 built-in models, tested against [llm-cost-cap](https://crates.io/crates/llm-cost-cap). Load [LiteLLM's](https://github.com/BerriAI/litellm) table (3,680 models) with `--prices` or `load_prices_json` | llm-cost-cap: 16 models, no cache-write pricing, out-of-date Claude Opus 4.5 to 4.7 and Haiku 4.5 prices |
| A forecast and a budget gate | Damped Holt forecast that handles sparse, bursty logs. Org, team and project budgets. Webhook alerts | llm-cost-cap gates one call before it is sent (use both) |
| To watch it live | `llm-dash`: ratatui dashboard that follows the log file, plus `/metrics` for Prometheus and Grafana | [edgequake-llm](https://crates.io/crates/edgequake-llm) tracks cost inside its own client. Your calls have to go through it |
| A small dependency | `default-features = false` leaves serde, chrono, csv and tokio's sync parts. TUI, HTTP server, webhooks and CLI are features | |

If you only need "would this call cost more than $X", [llm-cost-cap](https://crates.io/crates/llm-cost-cap) has no dependencies and looks up about a quarter faster (see Benchmarks).

## Install

**The dashboard (`llm-dash`)**

Linux (x86_64, Ubuntu 20.04+ / Debian 11+). One line. Installs to `~/.local/bin`:

```sh
mkdir -p ~/.local/bin && curl -fsSL https://gitlab.com/mattbusel/llm-cost-dashboard/-/releases/permalink/latest/downloads/llm-dash-linux-x86_64.tar.gz | tar xz --strip-components=1 -C ~/.local/bin --wildcards '*/llm-dash'
```

| Other systems | |
|---|---|
| **Windows** | [Download llm-dash-windows-x86_64.exe](https://gitlab.com/mattbusel/llm-cost-dashboard/-/releases/permalink/latest/downloads/llm-dash-windows-x86_64.exe) and run it. (Unsigned, so SmartScreen may ask: *More info*, then *Run anyway*.) |
| **Any system with Rust** | `cargo binstall llm-cost-dashboard` (prebuilt Linux and Windows binaries) or `cargo install --locked llm-cost-dashboard` |

Shell completions: `llm-dash --completions bash > ~/.local/share/bash-completion/completions/llm-dash` (also zsh, fish, powershell, elvish).

**The library**

```toml
[dependencies]
llm-cost-dashboard = { version = "1.3", default-features = false }   # pricing, ingest, forecasts, exports
```

**In GitLab CI:** fail a pipeline when the projected monthly bill is over budget with the [`llm-spend-check`](https://gitlab.com/explore/catalog/mattbusel/llm-ci) CI/CD component.

## Cargo features

| Feature | Default | What it adds | Extra dependencies |
|---|---|---|---|
| `cli` | on | The `llm-dash` binary (implies `tui` and `server`) | clap, clap_complete, tracing-subscriber, toml |
| `tui` | via `cli` | `ui`: the ratatui dashboard | ratatui, crossterm |
| `server` | via `cli` | `api`: JSON/CSV endpoints and Prometheus `/metrics` | axum, tokio net |
| `webhooks` | on | Slack / generic webhook alerts, signed with HMAC-SHA256 | reqwest (rustls), hmac, sha2 |
| `async-openai` | off | `interop::async_openai`: price `async_openai` chat completion responses | async-openai (types only, no HTTP client) |

With `default-features = false` you still get the ledger, pricing, `ingest`, forecasts, budgets, anomaly detection and exports.

## Use the dashboard in 3 steps

**1. Look around with sample data** (no API key, no setup):

```bash
llm-dash --demo
```

The dashboard above, with 20 sample Claude, GPT and Gemini requests. `x` opens the cost explorer, `q` quits.

**2. Use your own calls.** Have your app write one JSON line per model call to a file. Your own fields or the provider's response both work:

```json
{"model":"gpt-4o-mini","input_tokens":512,"output_tokens":256,"latency_ms":340}
{"model":"gpt-4o","created":1790344980,"usage":{"prompt_tokens":5120,"completion_tokens":256,"prompt_tokens_details":{"cached_tokens":4096}}}
```

Or use Claude Code's session logs: `llm-dash --log-file ~/.claude/projects/<project>/<session>.jsonl`.

**3. Watch that file live, with a monthly budget:**

```bash
llm-dash --log-file requests.ndjson --budget 50
```

New lines show up live. The budget gauge turns yellow at 80% and red past 100%. A call that costs far more than usual shows under **Cost Anomalies**.

## Works with what you already use

- **async-openai**: enable `async-openai` and call `interop::async_openai::record_from_response(&response)`. Cached tokens get the cached rate.
- **Any other client** (reqwest, genai, rig, the Anthropic HTTP API): pass the response JSON to `Ingester::ingest_line`. Reads OpenAI `usage.prompt_tokens` and Anthropic `usage.input_tokens` / `cache_read_input_tokens` / `cache_creation_input_tokens`.
- **LiteLLM's price table**: `llm-dash --prices model_prices_and_context_window.json` or `cost::pricing::load_prices_json`. Your own prices use the same call: `{"model": {"input_usd_per_1m": 0.5, "output_usd_per_1m": 1.5}}`.
- **Prometheus / Grafana / VictoriaMetrics**: `llm-dash --serve 9898` (or `api::router` inside your axum app) exposes `llm_cost_usd_total`, `llm_requests_total`, `llm_tokens_total` and `llm_projected_monthly_cost_usd` at `/metrics`.
- **Claude Code**: reads session logs under `~/.claude/projects`. `cargo run --example claude_code_spend` totals every session at API list prices.
- **llm-cost-cap**: gate a call before sending with it, account for it after with this.

## Examples

| Example | What it shows |
|---|---|
| `quickstart` | The 10-line example above |
| `claude_code_spend` | What your Claude Code sessions would cost at API prices, per model and per day |
| `budget_check` | Price a log, forecast 30 days, exit 1 when over budget (a CI gate) |
| `metrics_server` (`--features server`) | Record calls from your app and serve `/metrics` for Prometheus |

## Benchmarks

`cargo bench --bench vs_competitors`, on an Intel i7-13700KF, Windows 11, Rust 1.91, criterion medians:

| Task | llm-cost-dashboard 1.3.0 | llm-cost-cap 0.1.0 |
|---|---|---|
| Price one call (5 models in rotation) | 24.2 ns | 18.6 ns |
| Parse and price 10,000 mixed log lines (flat, OpenAI response, Claude Code) | 22.5 ms (445,000 lines/s) | no log parser |

llm-cost-cap is faster because it only does exact-match lookups. This one also ignores case, falls back from dated ids (`claude-sonnet-4-5-20250929`) to the undated price, and checks runtime overrides. Before 1.3.0 it was a linear scan at 61.6 ns.

## Sample output

Real output from `llm-dash 1.3.0` on the demo data:

```text
$ llm-dash --demo --compare
Multi-Provider Cost Comparison  (105 models, 22/day requests, 730in/348out avg tokens)

  Model                                           Monthly USD   Daily USD   Per-1k-req USD  Provider
  ----------------------------------------------------------------------------------------------------
  ministral-3b-2410                                    0.0285      0.0009           0.0431  Mistral
  llama-3.1-8b-instant                                 0.0425      0.0014           0.0643  Meta (Llama)
  amazon.nova-micro-v1:0                               0.0490      0.0016           0.0743  AWS Bedrock
  gemini-1.5-flash-8b                                  0.0525      0.0018           0.0796  Google
  ... 79 more rows ...
Cheapest: ministral-3b-2410 ($0.0285/mo)  |  Most expensive: gpt-4.5-preview ($70.5870/mo)  |  Spread: 2480x
```

```text
$ llm-dash --demo --budget 50 --forecast
Holt-Winters Cost Forecast (based on 20 records)

  Next hour:  $0.004256
  Next day:   $0.1111
  Next week:  $0.81
  Next month: $3.49

  80% CI (next hour): [$0.000000, $0.010753]
  Budget status: OK (monthly forecast $3.49 < 80% of $50.00 budget)
```

In the recording, 70 requests cost $0.5845 total. One gpt-4o call with 60,000 input tokens cost $0.3375 alone, 26.7 times that model's average. That's what the anomaly panel flags.

## Your log format

```json
{"model":"claude-sonnet-4-6","input_tokens":512,"output_tokens":256,"latency_ms":340,"timestamp":"2026-09-25T14:03:00Z"}
{"model":"gpt-4o-mini","input_tokens":128,"output_tokens":64,"latency_ms":120,"provider":"openai","ts":1790344980}
{"model":"gpt-4o","input_tokens":900,"output_tokens":0,"latency_ms":30000,"error":"timeout"}
```

`model`, `input_tokens` and `output_tokens` are required. `latency_ms`, `provider`, `error` and `timestamp` are optional.

Or log the API's response object. Tokens come from `prompt_tokens` / `completion_tokens` or a nested `usage` object (OpenAI `usage.prompt_tokens`, Anthropic `usage.input_tokens`). OpenAI's `created` is the timestamp:

```json
{"model":"gpt-4o-mini","created":1790344980,"usage":{"prompt_tokens":512,"completion_tokens":256,"total_tokens":768}}
{"model":"claude-haiku-4-5","usage":{"input_tokens":900,"output_tokens":120},"latency_ms":610}
```

Claude Code logs (`~/.claude/projects/**/*.jsonl`) are read as is. Each request counts once, even though Claude Code repeats it. User turns and bookkeeping lines are skipped. Cache reads and 5-minute / 1-hour cache writes are priced separately.

`timestamp` (or `ts`, `time`, `created_at`, `created`) can be RFC 3339 or Unix seconds or milliseconds. No timestamp means it's dated when read. Bad lines are skipped (`RUST_LOG=warn` shows why). Model names ignore case, and a dated id like `claude-sonnet-4-5-20250929` uses the undated price. Unknown models get a guessed $5 / $15 per million tokens, listed on stderr so you can add real prices with `--prices`.

## The dashboard

Panels, left to right, top to bottom:

| Panel | What it tells you |
|---|---|
| **Summary** | What you spent, and a month at the last hour's pace |
| **Budget** | Spend against `--budget`; yellow at 80%, red past 100% |
| **Forecast** | Per-day and month-end spend from the trend. Uses the last hour's pace until there's enough history, and says so |
| **Prompt Cache** | Cached-token reads and writes (one line when your log has none) |
| **Savings Opportunities** | Cheaper models for your traffic, with the monthly saving |
| **Cost by Model** | One bar per model, with its total |
| **Recent Requests** | The latest calls, newest first |
| **Cost Anomalies** | Calls that cost 2x or more their model's running average |
| **Last 7 days** | Spend per day, today highlighted |
| **Spend over time** | The last 60 request costs |

Refreshes every 250 ms. With no data, the requests panel shows how to feed it. Colors follow your terminal theme. `NO_COLOR` turns them off.

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `d` | Load demo data |
| `r` | Reset all data |
| `e` | Export the session to `llm-costs-<timestamp>.json` and `.csv` in the current directory |
| `j` / `k` or arrow keys | Scroll the requests table |
| `x` | Open the cost explorer (`s` cycles sort order, `Enter` toggles detail, `x` or `Esc` closes it) |

## Command-line reports

These load `--demo` and/or `--log-file`, print a report and exit:

```bash
llm-dash --demo --compare                     # rank every priced model by monthly cost for this workload
llm-dash --compare --workload-rph 1000        # same, for a hypothetical 1000 requests/hour
llm-dash --demo --anomaly                     # Z-score cost anomaly report
llm-dash --demo --export-csv costs.csv        # or --export-json costs.json
llm-dash --demo --export markdown             # csv | json | jsonl | markdown, to --out FILE or stdout
```

```bash
llm-dash --demo --forecast                    # projection of the next hour, day, week and month (damped Holt smoothing)
llm-dash --log-file requests.ndjson --diff 2026-09-01 2026-09-02   # Markdown diff between two date prefixes
```

`--forecast` buckets the log (a day per bucket for 3+ days, an hour for 3+ hours), counts quiet periods as zero, and runs damped Holt. Sparse logs get a real number, not $0.

`--forecast` and `--diff` need timestamps. Without them, `--forecast` says there isn't enough time spread and `--diff` lists the dates it has.

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--budget <USD>` | `10.0` | Monthly budget limit |
| `--log-file <PATH>` | | NDJSON request log to load at startup and keep following |
| `--demo` | off | Pre-load demo data |
| `--serve <PORT>` | | Also start the HTTP API (below) |
| `--webhook-url <URL>` | | Slack or generic webhook for budget alerts (repeatable) |
| `--webhook-threshold <USD>` | 80% of budget | Spend level that fires the webhook |
| `--webhook-format <FORMAT>` | `generic` | `slack` or `generic` |
| `--alerts <RULES_TOML>` | | Check budget alert rules once against the loaded data, then start the dashboard |
| `--prices <FILE>` | | Load extra or corrected prices (LiteLLM JSON or a plain model map) |
| `--bind <IP>` | `127.0.0.1` | Address for `--serve`; use `0.0.0.0` to accept other machines |
| `--completions <SHELL>` | | Print a shell completion script and exit |
| `--session <NAME>` | | Tag every ingested record with a session id |
| `--export-csv <PATH>`, `--export-json <PATH>` | | Write all records and exit |
| `--export <FORMAT>`, `--out <FILE>` | | Export tagged requests as csv, json, jsonl or markdown and exit |
| `--compare`, `--workload-rph <N>` | `1000` | Multi-provider cost ranking and exit |
| `--forecast` | off | Print a spend forecast and exit (needs at least 3 records) |
| `--anomaly` | off | Print an anomaly report and exit |
| `--diff <A> <B>` | | Compare two date-prefix periods and exit |

`RUST_LOG` sets verbosity. Logs go to stderr. Without it, reports print warnings only and the dashboard prints nothing.

### HTTP API

```bash
llm-dash --demo --serve 8080
curl localhost:8080/api/summary       # JSON cost summary
curl localhost:8080/api/export.json   # full ledger as JSON
curl localhost:8080/api/export.csv    # full ledger as CSV
curl localhost:8080/metrics           # Prometheus text format
```

Listens on 127.0.0.1 unless you pass `--bind 0.0.0.0` (1.2.x listened on every interface).

### Budget alerts

Webhooks (Slack or generic JSON POST) fire when spend crosses `--webhook-threshold`:

```bash
llm-dash --budget 50 --webhook-url "https://hooks.slack.com/services/..." --webhook-threshold 40 --webhook-format slack
```

An `--alerts` rule file:

```toml
[[rules]]
name = "daily-5-usd"
threshold_usd = 5.0
window = "daily"        # daily | weekly | monthly
cooldown_secs = 3600
```

## Supported models

105 models across Anthropic, OpenAI, Google, DeepSeek, Mistral, Meta Llama (Together AI and Groq), xAI, Cohere, Perplexity, Amazon Bedrock, Alibaba Qwen, Writer and AI21. USD per million tokens, from `src/cost/pricing.rs` (updated 2026-10-17; a test pins the Claude rows to [Anthropic's pricing page](https://platform.claude.com/docs/en/about-claude/pricing)). Check against your provider before relying on them.

<details>
<summary><b>Anthropic / Claude 5 family (Anthropic list prices, 2026-10-17)</b> (7)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `claude-fable-5-1` | $10 | $50 |
| `claude-fable-5` | $10 | $50 |
| `claude-opus-5-5` | $4 | $20 |
| `claude-opus-5` | $5 | $25 |
| `claude-sonnet-5-5` | $2 | $10 |
| `claude-sonnet-5` | $2 | $10 |
| `claude-haiku-5-5` | $0.10 | $0.50 |

`claude-haiku-5-5` is priced by prompt length. Over 100,000 prompt tokens (cache included), the whole request pays $0.50 / $2.50. `CostRecord` and `price_for_prompt` apply this. Cache reads are $0.10 on `claude-sonnet-5-5` and $0.20 on `claude-opus-5-5` (0.05x input), $0.25 on `claude-fable-5-1` (0.025x).

</details>

<details>
<summary><b>Anthropic / Claude 4 family</b> (10)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `claude-opus-4-8` | $5 | $25 |
| `claude-opus-4-7` | $5 | $25 |
| `claude-opus-4-6` | $5 | $25 |
| `claude-opus-4-5` | $5 | $25 |
| `claude-opus-4-1` | $15 | $75 |
| `claude-opus-4` | $15 | $75 |
| `claude-sonnet-4-5` | $3 | $15 |
| `claude-sonnet-4` | $3 | $15 |
| `claude-sonnet-4-6` | $3 | $15 |
| `claude-haiku-4-5` | $1 | $5 |

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
<summary><b>OpenAI / GPT-5 and GPT-4.1 families</b> (6)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `gpt-5` | $1.25 | $10 |
| `gpt-5-mini` | $0.25 | $2 |
| `gpt-5-nano` | $0.05 | $0.4 |
| `gpt-4.1` | $2 | $8 |
| `gpt-4.1-mini` | $0.4 | $1.6 |
| `gpt-4.1-nano` | $0.1 | $0.4 |

</details>

<details>
<summary><b>OpenAI / GPT-4o family</b> (5)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `gpt-4o` | $2.5 | $10 |
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
| `o3` | $2 | $8 |
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
<summary><b>Google / Gemini 2 family</b> (6)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `gemini-2.5-pro` | $1.25 | $10 |
| `gemini-2.5-flash` | $0.3 | $2.5 |
| `gemini-2.5-flash-lite` | $0.1 | $0.4 |
| `gemini-2.0-flash` | $0.1 | $0.4 |
| `gemini-2.0-flash-lite` | $0.075 | $0.3 |
| `gemini-2.0-flash-thinking` | $0.15 | $0.6 |

</details>

<details>
<summary><b>Google / Gemini 1.5 family</b> (3)</summary>

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| `gemini-1.5-pro` | $1.25 | $5 |
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
| `deepseek-chat` | $0.28 | $0.42 |
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

Everything the dashboard does is also a Rust library ([docs.rs](https://docs.rs/llm-cost-dashboard)).

<details>
<summary>Examples: ledger, cheapest model, budgets, anomalies, forecasts, tags</summary>

```toml
[dependencies]
llm-cost-dashboard = { version = "1.3", default-features = false }
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
# Ok::<(), llm_cost_dashboard::DashboardError>(())
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
# Ok::<(), llm_cost_dashboard::DashboardError>(())
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

Other modules: `budget::planner` (period budgets by percentage, vs actuals), `tenant` (per-tenant quotas), `alerts` (the TOML rules behind `--alerts`), `alerting` and `webhook` (webhooks with cooldowns), `validator` (checks Anthropic, OpenAI and Google keys), `recommendations` (cheaper models), `session`, `export`, `trends`, `model_compare`, `prediction` and `diff`. APIs: `cargo doc --open`.

</details>

## How it works

<details>
<summary>Source layout</summary>

```text
src/
  main.rs            CLI (clap): TUI launch and the one-shot report flags
  ui/                ratatui app state, event loop, dashboard layout, widgets, cost explorer
  log/               NDJSON parsing into LogEntry
  cost/              CostRecord, CostLedger, pricing table (pricing.rs)
  budget/            budget envelope, org/team/project hierarchy, planner
  comparison.rs      multi-provider monthly cost ranking
  ingest.rs          log lines to priced records (the library entry point)
  interop.rs         adapters for other crates (async-openai)
  forecast.rs        OLS and damped-Holt forecasters
  anomaly.rs         rolling and Welford Z-score detectors
  api/               axum server for --serve
  alerting.rs, alerts.rs, webhook/   alert rules and delivery
  export.rs          CSV / JSON / JSONL / Markdown export
  ...                about 60 further analysis modules (tagging, tenants, allocation, carbon, SLA, and more)
tests/, benches/     integration tests and criterion benchmarks
```

</details>

## Status and limitations

- The dashboard follows `--log-file` (polls every half second). No stdin. One-shot reports (`--forecast`, `--diff`, exports) read the file once.
- Time reports are only as good as your `timestamp` field. Lines without one are dated when read.
- `--alerts` checks its rules once, at startup.
- Prices are built in. Checked against LiteLLM's table in October 2026: 63 of 74 shared models agree, the rest are resellers with different prices. Use `--prices` with LiteLLM's file to stay current.
- The crate is big (about 80 modules). The binary uses the core (ledger, pricing, TUI, export, comparison, HTTP API). Many analysis modules are library-only.

```bash
cargo test
cargo bench
RUST_LOG=debug cargo run -- --demo --compare
```

## License

MIT, see [LICENSE](https://gitlab.com/mattbusel/llm-cost-dashboard/-/blob/master/LICENSE).

Source, issues and stars: [GitHub](https://github.com/Mattbusel/llm-cost-dashboard). Mirror and release downloads: [GitLab](https://gitlab.com/mattbusel/llm-cost-dashboard). More of my Rust: [tensorust.com/open-source](https://tensorust.com/open-source/).
