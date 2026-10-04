# Changelog

All notable changes to `llm-cost-dashboard` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-10-04

### Fixed

- **`--forecast` no longer collapses to $0 on sparse logs.** The smoother ran on
  the spend rate between each pair of consecutive requests and extrapolated an
  undamped trend hundreds of steps ahead, so a small downward trend (quiet
  nights, gaps between bursts) drove every horizon past the next hour below
  zero, clamped to $0. On the 40-call, 3.25-day fixture used by the
  `llm-spend-check` CI component it printed "Next month: $0.00"; with the
  same prices it now prints $7.58 (the plain average rate gives $8.69), and
  $8.62 with the corrected prices below (average rate $9.95). The same
  method overshot dense logs: on the demo data it said $44.33 a month where
  the average rate is $2.77; it now says $3.49. The log is now put on a regular
  grid (about a day per bucket for 3+ days of data, an hour for 3+ hours),
  empty buckets count as zero spend, and Holt's method runs with a damped trend
  (0.9), summed per bucket over each horizon. Output format is unchanged.
  A property test checks every horizon is finite, non-negative and
  non-decreasing on random logs.
- **`llm-dash` panicked when a webhook threshold was reached** ("there is no
  reactor running"): alerts were sent with `tokio::spawn` outside any runtime.
  They now go out on a background thread, once per threshold crossing (before,
  every later request would have posted again).
- **Wrong prices corrected** (checked against Anthropic's and OpenAI's price
  lists and LiteLLM's table): `claude-haiku-4-5` $1 / $5 per million tokens
  (was $0.25 / $1.25, the Claude 3 Haiku price), `claude-opus-4-6` $5 / $25
  (was $15 / $75), `gpt-4o` $2.50 / $10 (was $5 / $15), `o3` $2 / $8 (was
  $10 / $40), `gemini-1.5-pro` $1.25 / $5 (was $3.50 / $10.50),
  `deepseek-chat` $0.28 / $0.42 (was $0.27 / $1.10).
- **Webhook signatures were forgeable.** `integration_webhooks::WebhookPayload::sign`
  XOR-folded FNV hashes, which is not a MAC. It is now HMAC-SHA256
  (`sha256=<hex>`, RFC 4231 test vectors in the tests). The alert engine's
  hand-written SHA-256 was replaced by the RustCrypto `hmac` and `sha2` crates.
- **`--serve` showed a snapshot.** Lines the dashboard tailed after start-up
  never reached the HTTP API; they are now priced into the shared ledger too
  (`api::mirror_feed`).
- **`--serve` listened on every network interface**; it now binds 127.0.0.1
  unless you pass `--bind 0.0.0.0`.
- The over-budget warning was logged once per request after the breach (2,889
  identical lines on one real log); it is logged once.
- `cargo binstall` metadata pointed at GitHub-style release URLs; it now uses
  the GitLab release downloads for Linux x86_64 and Windows x86_64 (checked
  with a binstall dry run against 1.2.3).
- `--alerts` help said it started a background check loop; it checks once.

### Added

- **`ingest::Ingester`**: the library path from log lines to priced records,
  with counts of recorded, skipped and unreadable lines.
- **Log formats:** saved OpenAI and Anthropic responses (`prompt_tokens` /
  `completion_tokens`, nested `usage`, `created` as timestamp; OpenAI
  `cached_tokens` are split out and priced at the cached rate) and **Claude
  Code session logs** (each request counted once although Claude Code repeats
  it per content block; 1-hour cache writes priced at 2x input). On a real
  15,639-line session log it found the same 2,889 requests as an independent
  Python count. `latency_ms` is optional.
- **Prices:** 104 models (Claude 5 family, GPT-5, GPT-4.1, Gemini 2.5 Flash
  added), published cached-input rates, dated ids fall back to the undated
  price, run-time overrides with `cost::pricing::set_price` and
  `load_prices_json` / `llm-dash --prices FILE` (LiteLLM's
  `model_prices_and_context_window.json` loads 3,680 models). One-shot reports
  list models that were priced with the fallback guess.
- **Prometheus `/metrics`** in `--serve` mode and `api::router` /
  `api::serve_on` / `api::prometheus_text` for your own axum app.
- **`async-openai` feature**: `interop::async_openai::record_from_response`.
- **`llm-dash --completions <shell>`** (clap_complete).
- Examples: `quickstart`, `claude_code_spend`, `budget_check`, `metrics_server`.
- Cross-check test against llm-cost-cap's price table, property tests
  (proptest) for log parsing, price files and the forecaster, and a criterion
  benchmark against llm-cost-cap (`benches/vs_competitors.rs`).

### Changed

- **Lean library build:** the TUI (`tui`), HTTP server (`server`), webhooks
  (`webhooks`) and CLI (`cli`) are features. Defaults are unchanged
  (`cli` and `webhooks`), so `cargo install` and existing users get the same
  binary; `default-features = false` drops ratatui, crossterm, axum, clap and
  reqwest.
- Price lookup uses a hash map built once: 61.6 ns to 24.2 ns per call.
- `integration_webhooks::WebhookManager::simulate_delivery` / `process_pending`
  and `cost_forecast::ForecastModel::ARIMA` are deprecated: the first sends
  nothing and decides success from a hash, the second is linear
  extrapolation. Their docs now say so.
- README is the crate documentation on docs.rs and every Rust block in it is
  compiled as a doctest.
- Minimum Rust version 1.88 (checked with `cargo +1.88.0 check`). GitLab CI
  runs tests with all and with no default features, clippy `-D warnings`,
  rustdoc `-D warnings`, and an MSRV check. The dead GitHub workflows are gone.

## [1.2.3] - 2026-09-30

- Links point at GitLab.
- Linux x86_64 release builds from GitLab CI; downloads and links now point at GitLab.

## [Unreleased]

## [1.2.2] - 2026-09-26

### Changed

- Summary says what its monthly figure means ("At last hour's pace"), and the
  Forecast panel labels its trend-based numbers, so the two projections no
  longer look like conflicting answers.
- Forecast falls back to the last hour's pace, and says so, until the log
  spans enough time for a trend (it used to show only "--").
- The 7-day trend is now seven labelled bars (day and dollar amount) across
  the full width, with today highlighted.
- Empty panels shrink to one line: Prompt Cache (was Cache Breakdown) when
  the log has no cached tokens, Cost Anomalies when there are none.
- Release binaries for Linux ARM64 (`aarch64-unknown-linux-gnu`);
  `install.sh`, Homebrew and cargo-binstall pick them up.
- GitHub Actions moved off the deprecated Node 20 versions.
- README: the demo GIF opens on the full dashboard and runs about 14 seconds.

## [1.2.1] - 2026-09-25

### Fixed

- Log lines no longer draw over the dashboard: without `RUST_LOG`, the TUI
  logs nothing and the one-shot reports log warnings only (was `info`).
- An empty ledger showed `$-0.0000`; it now shows `$0.0000`.
- The footer suggested piping JSON into `llm-dash`, which it does not read;
  it now lists the keys and the `--log-file` usage.
- Cost by Model labels were cut to three characters; the chart is now
  horizontal with full model names and dollar amounts.

### Changed

- Dashboard: title bar shows request count, spend and budget; empty state
  explains how to load data; left-column panels no longer clip their lines;
  savings suggestions fit the column; body text uses the terminal's default
  colour so light themes stay readable; `NO_COLOR` is respected.
- `--help` has a plain description, examples and the log line format.
- Install: `install.sh`, `install.ps1`, cargo-binstall metadata, Homebrew and
  Scoop packages, winget manifests in `packaging/winget/`.
- README: banner, recorded demo GIF, install table, 3-step guide, real output.

## [1.2.0] - 2026-09-25

### Added

- Prebuilt binaries for Windows, macOS (Apple Silicon and Intel) and Linux on
  every GitHub Release, with `SHA256SUMS.txt`.
- `--log-file` is now followed while the dashboard is open: appended lines are
  ingested live, partial lines wait for their newline, and a truncated file is
  re-read from the start (`tail` module, `ui::run_with_feed`).
- Log lines may carry a `timestamp` (aliases `ts`, `time`, `created_at`): an
  RFC 3339 string or Unix seconds/milliseconds. Records keep that time instead
  of the load time.
- Demo data is spread over the last 24 hours.

### Fixed

- `--forecast` printed `$inf` when records shared a timestamp; observations at
  the same instant are now merged, and a clear error explains when there is not
  enough time spread. Horizon totals integrate the forecast rate instead of
  using the rate at the far end of the horizon. The "80%%" typo is gone.
- `--diff` found nothing because records had no real dates; it now uses log
  timestamps and says which dates exist when a period is empty.
- Z-score anomaly detectors (`anomaly`, `anomaly_detector`, `cost_predictor`)
  never flagged a spike after a perfectly flat history (standard deviation of
  zero). They now score against a small floor (0.1% of the mean).
- Log lines that are JSON arrays are rejected instead of being read positionally.
- `ModelStats` percentiles use the nearest-rank method (p50 of 10..100 is 50).
- Broken doctests in `aggregator`, `session` and `tagging`, a wrong expected
  slope in a benchmark test, and clippy warnings. CI is green again.

### Added

- Production-readiness pass: doc comments verified and completed on every public
  type, field, function, and trait across all modules.
- `ui/mod.rs`: full `#[cfg(test)]` test suite covering `App` state transitions,
  line ingestion, demo-data loading, scroll, and reset.
- CI workflow updated: `cargo doc --no-deps` job added with
  `RUSTDOCFLAGS=-D warnings`; MSRV pinned to 1.75.
- `CHANGELOG.md`: this `[Unreleased]` section.
- Restored `o1-preview` entry to the pricing table (`$15.00/$60.00` per 1M
  tokens) so that `integration_tests.rs` tests that reference this model name
  pass without fallback pricing.

---

## [1.0.0] - 2026-03-17

### Added

- **Structured tracing** throughout the binary and library using the `tracing`
  crate. Key events (startup, log-file ingestion, demo-data loading, budget
  breaches, terminal lifecycle) are now emitted as `info!`, `warn!`, and
  `error!` spans. Control verbosity with `RUST_LOG`.
- **Proper log-file error handling** in `main.rs`: file-read errors now print a
  diagnostic and exit with code 1; malformed JSON lines are skipped with a
  `warn!` log rather than silently discarded.
- **Graceful terminal cleanup** in `ui::run`: `disable_raw_mode` and
  `LeaveAlternateScreen` are attempted even when the event loop returns an
  error, preventing a corrupted terminal state on panic or IO failure.
- **`[profile.release]`** in `Cargo.toml`: `opt-level = 3`, LTO, single codegen
  unit, symbol stripping, and `panic = "abort"` for a smaller, faster binary.
- **`[profile.dev]`** with `debug = true` for easier development.
- **`tempfile`** and **`criterion`** added as dev-dependencies for tests and
  benchmarks respectively.
- **`benches/cost_bench.rs`**: Criterion benchmarks for pricing lookup and
  ledger aggregation (`add`, `by_model`, `sparkline_data`).
- **Comprehensive `tests/`**: `unit_tests.rs`, `integration_tests.rs`, and
  `integration.rs` cover the full public API including edge cases, pricing
  accuracy per model, and cross-module end-to-end paths.
- **`[[bench]]` target** declared in `Cargo.toml` for the new benchmark suite.
- **CI workflow** (`.github/workflows/ci.yml`): `fmt`, `clippy`, `test`
  (Ubuntu + Windows + macOS), `docs`, `msrv` (1.75), and `audit` jobs.
- **Comprehensive README**: what it does, all supported models/providers,
  quickstart, log-format reference, CLI reference, keyboard controls, library
  usage, architecture diagram, and development guide.
- **Cargo.toml metadata**: `homepage`, `documentation`, `readme`, `authors`,
  and `exclude` fields.

### Changed

- `App::record` now uses `if let Err(e)` for both `ledger.add` and
  `budget.spend` instead of `let _ = ...`, enabling warn-level tracing on
  rejection or budget breach.
- `App::new` logs its `budget_usd` parameter at `info` level on creation.
- `App::load_demo_data` logs the count of records being loaded at `info` level.
- `App::reset` logs the reset event at `info` level.
- `event_loop` logs entry, quit-key detection, reset, and demo-load events.
- `ui::run` logs terminal initialisation and restoration.
- Version bumped from `0.2.0` to `1.0.0` to reflect production-ready status.

### Fixed

- Terminal was not always restored when the event loop returned an error
  (cleanup is now unconditional with warnings on cleanup failure).
- Silent discard of log-file read errors (now fatal with a user-facing message
  and exit code 1).

---

## [0.2.0] - 2026-01-15

### Added

- `TraceSpan` and `SpanStore` for distributed-trace-style request tracking with
  cost annotation.
- `BudgetEnvelope::alert_triggered`, `gauge_pct`, and `status` helpers.
- `CostLedger::sparkline_data` for ratatui `Sparkline` integration.
- `RequestLog::to_json` for serialising all entries to pretty-printed JSON.
- Comprehensive inline `#[cfg(test)]` suites for every module.

### Changed

- `DashboardError` variants renamed for clarity (`LogParse` to `LogParseError`,
  `Io` to `IoError`, `Json` to `SerializationError`).

---

## [0.1.0] - 2025-12-01

### Added

- Initial release: live ratatui TUI displaying total spend, cost by model,
  recent requests table, budget gauge, and spend sparkline.
- Built-in pricing table for Anthropic, OpenAI, and Google models.
- NDJSON log-file ingestion and `--demo` mode.
- `CostLedger`, `CostRecord`, `ModelStats`, `BudgetEnvelope`, `LogEntry`,
  `RequestLog`, and `DashboardError` public types.
- `clap`-based CLI with `--budget`, `--log-file`, and `--demo` flags.

[1.0.0]: https://github.com/Mattbusel/llm-cost-dashboard/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/Mattbusel/llm-cost-dashboard/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Mattbusel/llm-cost-dashboard/releases/tag/v0.1.0
