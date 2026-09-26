# Changelog

All notable changes to `llm-cost-dashboard` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
