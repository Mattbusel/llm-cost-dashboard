//! Integration tests for llm-cost-dashboard.
//!
//! These tests verify cross-module behaviour: demo mode producing real
//! CostRecord entries and the pricing table covering all major models.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::len_zero)]

use llm_cost_dashboard::{
    cost::pricing::{lookup, FALLBACK_PRICING},
    ui::App,
};

/// Instantiate the app in demo mode and verify it generates at least one
/// cost entry without panicking.
#[test]
fn test_demo_mode_produces_output() {
    let mut app = App::new(50.0);
    app.load_demo_data();

    // The ledger should contain at least one record.
    assert!(
        app.ledger.is_empty() == false,
        "demo mode should populate the ledger with at least one record"
    );

    // Total spend must be strictly positive.
    assert!(
        app.ledger.total_usd() > 0.0,
        "total spend after demo load must be > $0.00"
    );

    // The log should also have been populated (one log entry per ingest).
    // Demo data uses `App::record` directly (bypasses log ingestion), so
    // we just confirm the ledger path is clean.
    assert_eq!(
        app.ledger.len(),
        20,
        "load_demo_data should inject exactly 20 demo records"
    );
}

/// Verify that every major model family has an explicit entry in the pricing
/// table (i.e. lookup does NOT fall back to FALLBACK_PRICING for these).
#[test]
fn test_pricing_covers_all_major_models() {
    let major_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
        "o1-preview",
        "o3-mini",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ];

    for model in &major_models {
        let (i, o) = lookup(model);
        assert!(
            (i, o) != FALLBACK_PRICING || {
                // FALLBACK_PRICING coincidentally matches a real model only if
                // the model's rates happen to equal (5.00, 15.00).  Guard that
                // case by also checking the model is actually present in the
                // PRICING table.
                use llm_cost_dashboard::cost::pricing::PRICING;
                PRICING
                    .iter()
                    .any(|(m, _, _)| m.eq_ignore_ascii_case(model))
            },
            "model '{model}' uses fallback pricing — add it to PRICING"
        );
        assert!(i > 0.0, "model '{model}' input rate must be > 0");
        assert!(o > 0.0, "model '{model}' output rate must be > 0");
    }
}

/// Demo mode should not exceed the budget with the default $50 limit.
#[test]
fn test_demo_mode_does_not_panic_with_low_budget() {
    // Budget will be exceeded but the app must not panic — errors are
    // silently absorbed via `let _ = self.budget.spend(cost)`.
    let mut app = App::new(0.0001);
    app.load_demo_data();
    // If we get here, no panic occurred.
    assert!(app.ledger.is_empty() == false);
}

/// `ingest_line` with valid JSON must add exactly one record to both the
/// log and the ledger and must not panic.
#[test]
fn test_ingest_line_integration() {
    let mut app = App::new(100.0);
    let line = r#"{"model":"gpt-4o-mini","input_tokens":1000,"output_tokens":500,"latency_ms":20}"#;
    app.ingest_line(line)
        .expect("valid JSON line should be ingested");
    assert_eq!(app.log.len(), 1);
    assert_eq!(app.ledger.len(), 1);
    assert!(app.ledger.total_usd() > 0.0);
}

/// Ingesting a malformed JSON line must return an error but must not corrupt
/// the ledger or the log.
#[test]
fn test_ingest_malformed_line_does_not_corrupt_state() {
    let mut app = App::new(100.0);
    // First add a valid record.
    let good = r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50,"latency_ms":10}"#;
    app.ingest_line(good).unwrap();

    // Now try a bad record.
    let bad = "this is not json";
    assert!(app.ingest_line(bad).is_err());

    // The valid record must still be there.
    assert_eq!(app.ledger.len(), 1);
    assert_eq!(app.log.len(), 1);
}

// ── JSON parsing edge cases ───────────────────────────────────────────────────

/// An unknown model name in the JSON is accepted at the log layer; the cost
/// layer silently falls back to mid-range pricing.
#[test]
fn test_ingest_unknown_model_falls_back_to_pricing() {
    let mut app = App::new(100.0);
    let line = r#"{"model":"my-totally-unknown-model-xyz","input_tokens":1000000,"output_tokens":0,"latency_ms":5}"#;
    app.ingest_line(line)
        .expect("unknown model should be accepted");
    assert_eq!(app.ledger.len(), 1);
    // Cost must be positive (fallback pricing: $5.00/1M input).
    assert!(
        app.ledger.total_usd() > 0.0,
        "fallback pricing should produce a positive cost"
    );
    // Cost should equal the fallback input rate of $5.00/1M * 1M = $5.00.
    let expected = 5.00;
    assert!(
        (app.ledger.total_usd() - expected).abs() < 1e-6,
        "fallback cost was {}, expected {expected}",
        app.ledger.total_usd()
    );
}

/// Missing required fields (output_tokens) must produce a LogParseError.
#[test]
fn test_ingest_missing_output_tokens_returns_parse_error() {
    use llm_cost_dashboard::DashboardError;
    let mut app = App::new(100.0);
    let line = r#"{"model":"gpt-4o","input_tokens":100,"latency_ms":10}"#;
    let err = app.ingest_line(line).unwrap_err();
    assert!(
        matches!(err, DashboardError::LogParseError(_)),
        "expected LogParseError, got {err:?}"
    );
    assert!(app.ledger.is_empty());
}

/// A JSON object with all fields present but a string value where a u64 is
/// expected must return a LogParseError.
#[test]
fn test_ingest_invalid_token_type_returns_parse_error() {
    use llm_cost_dashboard::DashboardError;
    let mut app = App::new(100.0);
    // input_tokens is a string, not a number — serde must reject this.
    let line = r#"{"model":"gpt-4o","input_tokens":"lots","output_tokens":50,"latency_ms":10}"#;
    let err = app.ingest_line(line).unwrap_err();
    assert!(
        matches!(err, DashboardError::LogParseError(_)),
        "expected LogParseError for invalid token type, got {err:?}"
    );
    assert!(app.ledger.is_empty());
}

/// Negative token values encoded as signed integers are rejected because the
/// JSON type does not match the expected u64.
#[test]
fn test_ingest_negative_token_value_returns_parse_error() {
    use llm_cost_dashboard::DashboardError;
    let mut app = App::new(100.0);
    let line = r#"{"model":"gpt-4o","input_tokens":-1,"output_tokens":50,"latency_ms":10}"#;
    let err = app.ingest_line(line).unwrap_err();
    assert!(
        matches!(err, DashboardError::LogParseError(_)),
        "negative token count should be rejected"
    );
}

/// An empty JSON object `{}` is missing all required fields and must error.
#[test]
fn test_ingest_empty_json_object_returns_error() {
    let mut app = App::new(100.0);
    assert!(app.ingest_line("{}").is_err());
    assert!(app.ledger.is_empty());
}

/// Multiple valid lines interspersed with bad lines must only count the good ones.
#[test]
fn test_ingest_mixed_valid_invalid_lines() {
    let mut app = App::new(1000.0);
    let good = r#"{"model":"gpt-4o-mini","input_tokens":100,"output_tokens":50,"latency_ms":5}"#;
    let bad = "this is not json at all";

    app.ingest_line(good).unwrap();
    let _ = app.ingest_line(bad);
    app.ingest_line(good).unwrap();
    let _ = app.ingest_line(bad);
    app.ingest_line(good).unwrap();

    assert_eq!(app.ledger.len(), 3, "only valid lines should be counted");
}

// ── Cost calculation accuracy ────────────────────────────────────────────────

/// Verify exact cost for claude-opus-4-6 input: $15.00/1M tokens.
#[test]
fn test_cost_accuracy_claude_opus_input() {
    use llm_cost_dashboard::cost::pricing::compute_cost;
    let cost = compute_cost("claude-opus-4-6", 1_000_000, 0);
    assert!(
        (cost - 15.00).abs() < 1e-9,
        "claude-opus-4-6 1M input cost: {cost}"
    );
}

/// Verify exact cost for claude-opus-4-6 output: $75.00/1M tokens.
#[test]
fn test_cost_accuracy_claude_opus_output() {
    use llm_cost_dashboard::cost::pricing::compute_cost;
    let cost = compute_cost("claude-opus-4-6", 0, 1_000_000);
    assert!(
        (cost - 75.00).abs() < 1e-9,
        "claude-opus-4-6 1M output cost: {cost}"
    );
}

/// Verify exact cost for gemini-1.5-flash input: $0.075/1M tokens.
#[test]
fn test_cost_accuracy_gemini_flash_input() {
    use llm_cost_dashboard::cost::pricing::compute_cost;
    let cost = compute_cost("gemini-1.5-flash", 1_000_000, 0);
    assert!(
        (cost - 0.075).abs() < 1e-9,
        "gemini-1.5-flash 1M input cost: {cost}"
    );
}

/// Verify that combined input + output cost is the sum of its parts.
#[test]
fn test_cost_accuracy_combined_tokens() {
    use llm_cost_dashboard::cost::pricing::compute_cost;
    // gpt-4o-mini: $0.15/1M input + $0.60/1M output
    let combined = compute_cost("gpt-4o-mini", 1_000_000, 1_000_000);
    let expected = 0.15 + 0.60;
    assert!(
        (combined - expected).abs() < 1e-9,
        "combined cost: {combined}, expected {expected}"
    );
}

/// Verify o1-preview output pricing: $60.00/1M tokens.
#[test]
fn test_cost_accuracy_o1_preview_output() {
    use llm_cost_dashboard::cost::pricing::compute_cost;
    let cost = compute_cost("o1-preview", 0, 1_000_000);
    assert!(
        (cost - 60.00).abs() < 1e-9,
        "o1-preview 1M output cost: {cost}"
    );
}

/// Verify gpt-4-turbo combined pricing: $10.00 input + $30.00 output = $40.00.
#[test]
fn test_cost_accuracy_gpt4_turbo_combined() {
    use llm_cost_dashboard::cost::pricing::compute_cost;
    let cost = compute_cost("gpt-4-turbo", 1_000_000, 1_000_000);
    assert!(
        (cost - 40.00).abs() < 1e-9,
        "gpt-4-turbo combined 1M/1M cost: {cost}"
    );
}

// ── TUI rendering panic-safety ────────────────────────────────────────────────

/// Calling `render_summary` with zeroed values must not panic.
#[test]
fn test_tui_render_summary_zero_values_no_panic() {
    use llm_cost_dashboard::ui::App;
    // Just exercise the non-TUI path: build an App, verify rendering state is
    // coherent without actually driving a terminal.
    let app = App::new(10.0);
    assert!(app.ledger.is_empty());
    // Confirm the data methods the TUI calls do not panic on empty input.
    let _ = app.ledger.total_usd();
    let _ = app.ledger.projected_monthly_usd(1);
    let _ = app.ledger.sparkline_data(60);
    let _ = app.ledger.by_model();
    let _ = app.ledger.last_n(200);
    let _ = app.budget.gauge_pct();
    let _ = app.budget.status();
    let _ = app.budget.pct_consumed();
}

/// Calling TUI data methods with a fully populated App must not panic.
#[test]
fn test_tui_render_data_methods_with_demo_data_no_panic() {
    let mut app = App::new(10.0);
    app.load_demo_data();
    let _ = app.ledger.total_usd();
    let _ = app.ledger.projected_monthly_usd(1);
    let data = app.ledger.sparkline_data(60);
    assert!(!data.is_empty());
    let by_model = app.ledger.by_model();
    assert!(!by_model.is_empty());
    let records = app.ledger.last_n(200);
    assert!(!records.is_empty());
}

/// Scroll operations must not panic when the offset exceeds record count.
#[test]
fn test_tui_scroll_past_end_does_not_panic() {
    let mut app = App::new(10.0);
    for _ in 0..5 {
        app.scroll_down();
    }
    // Now scroll_offset is 5, but there are 0 records.
    let records = app.ledger.last_n(200);
    // last_n(200).iter().rev().skip(5) should just yield nothing — no panic.
    let visible: Vec<_> = records
        .iter()
        .rev()
        .skip(app.scroll_offset)
        .take(20)
        .collect();
    assert!(visible.is_empty());
}

/// `by_model` on a ledger with a single model should not panic and return
/// correct aggregation counts.
#[test]
fn test_tui_by_model_single_model_no_panic() {
    use llm_cost_dashboard::cost::{CostLedger, CostRecord};
    let mut ledger = CostLedger::new();
    for _ in 0..3 {
        ledger
            .add(CostRecord::new("gpt-4o-mini", "openai", 100, 50, 10))
            .unwrap();
    }
    let stats = ledger.by_model();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats["gpt-4o-mini"].request_count, 3);
}

/// Resetting the app after loading demo data must return it to a clean state.
#[test]
fn test_reset_after_demo_data() {
    let mut app = App::new(50.0);
    app.load_demo_data();
    assert!(app.ledger.is_empty() == false);
    app.reset();
    assert_eq!(app.ledger.len(), 0);
    assert_eq!(app.log.len(), 0);
    assert_eq!(app.ledger.total_usd(), 0.0);
    assert_eq!(app.budget.spent_usd, 0.0);
    assert_eq!(app.scroll_offset, 0);
}
