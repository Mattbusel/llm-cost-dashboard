# llm-cost-dashboard

Real-time terminal dashboard for LLM token spend. Displays cost per request,
per-model breakdown, monthly projection, and budget tracking in a single-screen
TUI. No external services required.

Built with [ratatui](https://ratatui.rs) and [crossterm](https://github.com/crossterm-rs/crossterm).

## Screenshot

```
 LLM Cost Dashboard  [q: quit | r: reset | d: demo data | j/k: scroll]
+-------------------+----------------------------------------------------+
| Summary           |  Cost by Model (uUSD)                              |
|                   |  ||||||||  claude-sonnet-4-6                       |
| Total spend:      |  ||||       gpt-4o                                 |
|   $0.014200       |  ||         gpt-4o-mini                            |
| Projected/mo:     |  |          claude-haiku-4-5                       |
|   $0.3400         +----------------------------------------------------+
| Requests: 20      |  Recent Requests                                   |
+-------------------+  Time      Model              In     Out    Cost   |
| Budget            |  14:22:01  claude-sonnet-4-6  847    312  $0.0049 |
|                   |  14:22:01  gpt-4o-mini         512    128  $0.0001 |
| |||||||...  70%   |  14:22:01  claude-haiku-4-5    256     64  $0.0001 |
| $3.00 remaining   |  14:22:01  claude-sonnet-4-6  1024   512  $0.0094 |
+-------------------+----------------------------------------------------+
| Spend over time (last 60 requests)                                     |
| ._.,,...^^...,,.                                                       |
+------------------------------------------------------------------------+
 Pipe data: echo '{"model":"claude-sonnet-4-6","input_tokens":512,...}' | llm-dash
```

## Installation

### From crates.io (once published)

```
cargo install llm-cost-dashboard
```

### From source

```
git clone https://github.com/Mattbusel/llm-cost-dashboard
cd llm-cost-dashboard
cargo build --release
```

The release binary is placed at `target/release/llm-dash`.

## Quickstart

Launch with built-in demo data to see the dashboard immediately:

```
llm-dash --demo
```

Point at a JSON log file and set a $25/month budget:

```
llm-dash --log-file requests.log --budget 25.0
```

Stream records from your application in real time:

```
your-app | llm-dash --budget 50.0
```

Send a single request record from the shell:

```bash
echo '{"model":"claude-sonnet-4-6","input_tokens":512,"output_tokens":256,"latency_ms":340}' \
  | llm-dash --log-file /dev/stdin
```

## Architecture

```
JSON log files / --log-file flag / stdin pipe
        |
        |  newline-delimited JSON records
        v
+-------------------+
|  Parser           |
|  (RequestLog /    |
|   IncomingRecord) |
|  ingest_line()    |
+-------------------+
        |
        |  LogEntry -> CostRecord
        v
+-------------------+       +-------------------+
|  CostEngine       |       |  BudgetEnvelope   |
|  (CostLedger)     | ----> |  spend()          |
|  - total_usd()    |       |  remaining()      |
|  - by_model()     |       |  alert_triggered()|
|  - projected_     |       +-------------------+
|    monthly_usd()  |
|  - sparkline_data |
+-------------------+
        |
        v
+-------------------+
|  Ratatui TUI      |
|  - summary pane   |
|  - budget gauge   |
|  - model bar chart|
|  - request table  |
|  - cost sparkline |
+-------------------+
```

## Log Format

The dashboard accepts newline-delimited JSON (NDJSON). Each line represents one
completed LLM request.

Required fields:

| Field           | Type   | Description                          |
|-----------------|--------|--------------------------------------|
| `model`         | string | Model identifier (see pricing table) |
| `input_tokens`  | uint   | Number of prompt tokens              |
| `output_tokens` | uint   | Number of completion tokens          |
| `latency_ms`    | uint   | End-to-end request latency in ms     |

Optional fields:

| Field      | Type   | Default     | Description                              |
|------------|--------|-------------|------------------------------------------|
| `provider` | string | `"unknown"` | Provider name for display                |
| `error`    | string | absent      | Error message; marks request as failed   |

Example lines:

```json
{"model":"claude-sonnet-4-6","input_tokens":512,"output_tokens":256,"latency_ms":340}
{"model":"gpt-4o-mini","input_tokens":128,"output_tokens":64,"latency_ms":12,"provider":"openai"}
{"model":"gpt-4o","input_tokens":0,"output_tokens":0,"latency_ms":5,"error":"rate_limit"}
```

The dashboard skips malformed lines and continues rather than crashing.

## Key Bindings

| Key      | Action                     |
|----------|----------------------------|
| q / Esc  | Quit                       |
| d        | Load demo data             |
| r        | Reset all data             |
| j / Down | Scroll requests table down |
| k / Up   | Scroll requests table up   |

## Supported Models

| Model             | Input (per 1M tokens) | Output (per 1M tokens) |
|-------------------|-----------------------|------------------------|
| claude-opus-4-6   | $15.00                | $75.00                 |
| claude-sonnet-4-6 | $3.00                 | $15.00                 |
| claude-haiku-4-5  | $0.25                 | $1.25                  |
| gpt-4o            | $5.00                 | $15.00                 |
| gpt-4o-mini       | $0.15                 | $0.60                  |
| gpt-4-turbo       | $10.00                | $30.00                 |
| o1-preview        | $15.00                | $60.00                 |
| o3-mini           | $1.10                 | $4.40                  |
| gemini-1.5-pro    | $3.50                 | $10.50                 |
| gemini-1.5-flash  | $0.075                | $0.30                  |

Unknown models fall back to $5.00/$15.00 per 1M tokens.

## Command-Line Reference

```
USAGE:
    llm-dash [OPTIONS]

OPTIONS:
    --budget <BUDGET>        Monthly budget limit in USD [default: 10.0]
    --log-file <LOG_FILE>    JSON log file to load on startup
    --demo                   Pre-load synthetic demo data
    -h, --help               Print help
    -V, --version            Print version
```

## Building from Source

Requires Rust 1.75 or later.

```
git clone https://github.com/Mattbusel/llm-cost-dashboard
cd llm-cost-dashboard
cargo build --release
cargo test
cargo doc --no-deps --open
```

## Contributing

Contributions are welcome. Please open an issue before submitting a large change.

### Adding a new model's pricing

1. Open `src/cost/pricing.rs`.
2. Append a new entry to the `PRICING` constant:
   ```rust
   ("your-model-id", input_usd_per_1m, output_usd_per_1m),
   ```
   Model IDs are matched case-insensitively, so use the canonical lowercase name.
3. Add a test in the `#[cfg(test)]` block at the bottom of `pricing.rs` that
   verifies the rates via `lookup("your-model-id")`.
4. Add the model to the `major_models` array in
   `tests/integration_tests.rs::test_pricing_covers_all_major_models`.
5. Update the **Supported Models** table in this README.
6. Run `cargo test --all-features` to confirm everything passes.

### Running the test suite

```bash
cargo test --all-features
```

### Building documentation

```bash
cargo doc --no-deps --open
```

## License

MIT

## Related Projects

- [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) - Rust LLM pipeline orchestration
- [rot-signals-api](https://github.com/Mattbusel/rot-signals-api) - Options signal REST API
- [prompt-observatory](https://github.com/Mattbusel/prompt-observatory) - LLM interpretability dashboard
