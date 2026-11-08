# llm-cost-dashboard

[![crates.io](https://img.shields.io/crates/v/llm-cost-dashboard.svg)](https://crates.io/crates/llm-cost-dashboard) [![docs.rs](https://docs.rs/llm-cost-dashboard/badge.svg)](https://docs.rs/llm-cost-dashboard) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Prices every LLM call from the logs you already have.

105 models built in. Prompt caching priced right.

- Reads OpenAI, Anthropic and Claude Code logs
- Tracks a budget and forecasts the month
- Flags calls that cost far more than usual
- Serves Prometheus metrics
- Runs in the terminal or as a Rust library

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
