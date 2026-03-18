# Contributing to llm-cost-dashboard

Thank you for your interest in contributing! Please read this guide before
opening a pull request.

---

## Prerequisites

- **Rust stable** (1.75 or later) — install via [rustup](https://rustup.rs/)
- No external services required — the dashboard runs entirely in-process

---

## Building

```bash
git clone https://github.com/Mattbusel/llm-cost-dashboard.git
cd llm-cost-dashboard
cargo build --release
```

The release binary is at `target/release/llm-dash`.

---

## Running tests

```bash
cargo test --all-features --locked
```

Run a specific module:

```bash
cargo test cost
cargo test budget
cargo test pricing
```

Run benchmarks:

```bash
cargo bench
```

---

## Adding new model pricing

Pricing is maintained in `src/cost/pricing.rs` in the `PRICING` static slice.
Each entry is a tuple `(model_id, input_usd_per_1m, output_usd_per_1m)`.

To add a model:

1. Open `src/cost/pricing.rs`.
2. Add a new line to `PRICING` with the model ID (lowercase, hyphenated),
   input price, and output price in USD per 1 million tokens.
3. Update the `// Last updated:` comment at the top of the constant with
   today's date.
4. Add a corresponding `test_<model>_pricing` unit test in the same file.
5. Run `cargo test pricing` to verify.

---

## Code style

- **Formatting**: run `cargo fmt --all` before committing.
- **Lints**: the project enforces `-D warnings` via Clippy. Run:
  ```bash
  cargo clippy --all-targets --all-features -- -D warnings
  ```
- **Doc comments**: every public item (function, struct, enum, field) must have
  a `///` doc comment. The crate uses `#![deny(missing_docs)]`.
- **Error handling**: use `DashboardError` variants — do not `panic!` or
  `unwrap()` outside of tests.

---

## Opening a pull request

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b my-feature
   ```
2. Ensure `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test` all
   pass locally.
3. Open a pull request against `main` with a clear description of the change.

CI enforces formatting, Clippy, the full test suite on Ubuntu/Windows/macOS,
rustdoc with `-D warnings`, MSRV (1.75), and a security audit before merging.
