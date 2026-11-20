# llm-cost-dashboard: your provider's billing page is late and does not know which agent spent it

**Why this exists:** LLM spend shows up on the provider's billing page hours or days later, rolled up per key, with no idea which feature, agent or prompt caused it. Your request logs already carry every token count.

**Why this instead of the usual approach:** Hosted observability wants a proxy or an SDK in your request path. llm-cost-dashboard reads the logs you already have (NDJSON request logs, Claude Code transcripts, OpenAI usage exports), prices every call from the dated llm-prices table plus a built-in one with prompt caching and Batch API rates, and shows spend, budgets, a month-end forecast and outlier calls in the terminal or as Prometheus metrics.

**Why adopt it today:** `cargo install --locked llm-cost-dashboard`, run `llm-dash --log-file requests.ndjson --budget 50`, and see today's spend by model before the invoice does.

**Built for:** engineers paying for LLM APIs or Claude Code seats who want spend per model and per day from their own logs, offline.

**Tradeoffs we chose on purpose:**

- List prices, not your negotiated discount: it shows what the call costs at the published rate.
- It reads logs after the fact. To stop a call before it overspends, use llm-budget.

[![crates.io](https://img.shields.io/crates/v/llm-cost-dashboard.svg)](https://crates.io/crates/llm-cost-dashboard) [![docs.rs](https://docs.rs/llm-cost-dashboard/badge.svg)](https://docs.rs/llm-cost-dashboard) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Install

```sh
cargo install --locked llm-cost-dashboard
```

## Use

```bash
llm-dash --demo
llm-dash --log-file requests.ndjson --budget 50
```

[Docs](https://docs.rs/llm-cost-dashboard) · [crates.io](https://crates.io/crates/llm-cost-dashboard) · [Releases](https://gitlab.com/mattbusel/llm-cost-dashboard/-/releases)

More: [docs/DETAILS.md](docs/DETAILS.md)

License: MIT
