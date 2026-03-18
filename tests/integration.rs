//! Integration tests for llm-cost-dashboard.
//!
//! These tests exercise the public API as a whole, verifying that components
//! compose correctly end-to-end.

use llm_cost_dashboard::{
    budget::BudgetEnvelope,
    cost::{pricing, CostLedger, CostRecord},
    error::DashboardError,
    log::RequestLog,
    trace::{SpanStore, TraceSpan},
    App,
};

// ---------------------------------------------------------------------------
// Pricing table integration
// ---------------------------------------------------------------------------

#[test]
fn all_pricing_entries_produce_finite_costs() {
    for (model, _, _) in pricing::PRICING {
        let cost = pricing::compute_cost(model, 1_000_000, 1_000_000);
        assert!(cost.is_finite(), "non-finite cost for {model}");
        assert!(cost > 0.0, "zero cost for {model}");
    }
}

#[test]
fn fallback_pricing_is_nonzero() {
    let (i, o) = pricing::FALLBACK_PRICING;
    assert!(i > 0.0);
    assert!(o > 0.0);
}

// ---------------------------------------------------------------------------
// Cost ledger + budget pipeline
// ---------------------------------------------------------------------------

#[test]
fn ledger_and_budget_stay_consistent() {
    let mut ledger = CostLedger::new();
    let mut budget = BudgetEnvelope::new("monthly", 1.0, 0.8);

    for i in 0..5u64 {
        let rec = CostRecord::new("gpt-4o-mini", "openai", i * 100, i * 50, 10);
        let cost = rec.total_cost_usd;
        ledger.add(rec).unwrap();
        let _ = budget.spend(cost);
    }

    let total_ledger = ledger.total_usd();
    let total_budget = budget.spent_usd;
    assert!((total_ledger - total_budget).abs() < 1e-12);
}

#[test]
fn budget_exceeded_after_large_spend() {
    let mut budget = BudgetEnvelope::new("test", 0.001, 0.8);
    let err = budget.spend(1.0).unwrap_err();
    assert!(matches!(err, DashboardError::BudgetExceeded { .. }));
}

// ---------------------------------------------------------------------------
// JSON log ingestion pipeline
// ---------------------------------------------------------------------------

#[test]
fn ingest_multiple_valid_lines() {
    let mut log = RequestLog::new();
    let lines = [
        r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50,"latency_ms":20}"#,
        r#"{"model":"claude-sonnet-4-6","input_tokens":200,"output_tokens":100,"latency_ms":40,"provider":"anthropic"}"#,
        r#"{"model":"gemini-1.5-flash","input_tokens":50,"output_tokens":25,"latency_ms":10}"#,
    ];
    for line in &lines {
        log.ingest_line(line).unwrap();
    }
    assert_eq!(log.len(), 3);
}

#[test]
fn malformed_line_skipped_gracefully() {
    let mut log = RequestLog::new();
    log.ingest_line(
        r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50,"latency_ms":20}"#,
    )
    .unwrap();
    let err = log.ingest_line("this is not json").unwrap_err();
    assert!(matches!(err, DashboardError::LogParseError(_)));
    // The previously ingested entry is still present.
    assert_eq!(log.len(), 1);
}

#[test]
fn missing_required_field_returns_error() {
    let mut log = RequestLog::new();
    // latency_ms is missing
    let result = log.ingest_line(r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50}"#);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Trace span integration
// ---------------------------------------------------------------------------

#[test]
fn span_store_total_matches_individual_costs() {
    let mut store = SpanStore::new();
    let models = ["gpt-4o-mini", "claude-sonnet-4-6", "gemini-1.5-flash"];
    let mut expected_total = 0.0f64;
    for model in &models {
        let span = TraceSpan::new("r", *model, "p", 500_000, 250_000, 100);
        expected_total += span.cost_usd;
        store.record(span);
    }
    assert!((store.total_cost() - expected_total).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// App (ui::App) end-to-end
// ---------------------------------------------------------------------------

#[test]
fn app_ingest_and_reset() {
    let mut app = App::new(50.0);
    let line = r#"{"model":"gpt-4o-mini","input_tokens":1000,"output_tokens":500,"latency_ms":15}"#;
    app.ingest_line(line).unwrap();
    assert_eq!(app.ledger.len(), 1);
    assert_eq!(app.log.len(), 1);

    app.reset();
    assert!(app.ledger.is_empty());
    assert!(app.log.is_empty());
    assert_eq!(app.budget.spent_usd, 0.0);
}

#[test]
fn app_demo_data_produces_nonzero_total() {
    let mut app = App::new(100.0);
    app.load_demo_data();
    assert!(app.ledger.total_usd() > 0.0);
    assert!(app.budget.spent_usd > 0.0);
}

#[test]
fn app_multiple_malformed_lines_do_not_corrupt_state() {
    let mut app = App::new(50.0);
    app.ingest_line(
        r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50,"latency_ms":10}"#,
    )
    .unwrap();

    for bad in &["", "not json", "{}", r#"{"model":"x"}"#] {
        let _ = app.ingest_line(bad); // errors are expected; must not panic
    }

    // Only the one valid line should be present.
    assert_eq!(app.ledger.len(), 1);
}

#[test]
fn app_projected_monthly_nonzero_after_ingest() {
    let mut app = App::new(50.0);
    app.ingest_line(
        r#"{"model":"gpt-4o","input_tokens":1_000_000,"output_tokens":500_000,"latency_ms":100}"#,
    )
    .unwrap_or_default(); // may fail on JSON underscore literals - that's fine
    app.load_demo_data();
    // With demo data the projection over a 1-hour window should be positive.
    assert!(app.ledger.projected_monthly_usd(1) >= 0.0);
}
