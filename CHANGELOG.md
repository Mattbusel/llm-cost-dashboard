# Changelog

All notable changes to llm-cost-dashboard are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-17

### Added

- Initial production release.
- Real-time ratatui TUI with five panels: title bar, summary, budget gauge,
  per-model bar chart, recent requests table, spend sparkline, and help bar.
- Cost ledger (`CostLedger`) with per-model aggregation, p99 latency, and
  30-day monthly projection.
- Budget envelope (`BudgetEnvelope`) with hard limit, 80% soft alert threshold,
  and traffic-light status display.
- Static pricing table covering 10 models across Anthropic, OpenAI, and Google.
  Fallback pricing for unknown models.
- Newline-delimited JSON log ingestion via `--log-file` or stdin pipe.
- Graceful error handling on malformed log lines: bad lines are skipped with a
  `LogParseError`; the TUI never panics.
- `DashboardError` unified error type with variants: `Ledger`, `BudgetExceeded`,
  `BudgetAlert`, `UnknownModel`, `LogParseError`, `IoError`, `InvalidPricing`,
  `SerializationError`, `Terminal`.
- Comprehensive test suite: unit tests in every module plus integration tests
  in `tests/integration.rs`.
- CI workflow: `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, and
  `cargo build --release` on both `ubuntu-latest` and `windows-latest`.
- Full `cargo doc` coverage: every public struct, enum, function, method, and
  field carries a `///` doc comment.
- `--demo` flag to pre-load 20 synthetic records across 7 models.
- Key bindings: `q`/`Esc` quit, `r` reset, `d` demo, `j`/`k` scroll.
