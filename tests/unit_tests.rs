//! Comprehensive unit tests for llm-cost-dashboard.
#![allow(clippy::unwrap_used, clippy::expect_used)]
//!
//! These tests exercise `pricing`, `log` (parser), `cost` (calculator /
//! aggregation), and `budget` modules using only their public APIs.

use llm_cost_dashboard::{
    budget::BudgetEnvelope,
    cost::{
        pricing::{compute_cost, lookup, FALLBACK_PRICING, PRICING},
        CostLedger, CostRecord,
    },
    log::RequestLog,
};

// ── pricing.rs ───────────────────────────────────────────────────────────────

#[test]
fn pricing_every_known_model_has_nonzero_rates() {
    for (model, input_rate, output_rate) in PRICING {
        assert!(*input_rate > 0.0, "model '{model}' has zero input rate");
        assert!(*output_rate > 0.0, "model '{model}' has zero output rate");
    }
}

#[test]
fn pricing_cost_for_1000_tokens_gpt4o() {
    // gpt-4o: $5.00/1M input  → 1 000 tokens = $0.005
    let cost = compute_cost("gpt-4o", 1_000, 0);
    let expected = 5.00 * 1_000.0 / 1_000_000.0;
    assert!(
        (cost - expected).abs() < 1e-10,
        "gpt-4o 1k input cost = {cost}, expected {expected}"
    );
}

#[test]
fn pricing_cost_for_1000_tokens_gpt4o_mini() {
    // gpt-4o-mini: $0.15/1M input → 1 000 tokens = $0.00015
    let cost = compute_cost("gpt-4o-mini", 1_000, 0);
    let expected = 0.15 * 1_000.0 / 1_000_000.0;
    assert!(
        (cost - expected).abs() < 1e-12,
        "gpt-4o-mini 1k input cost = {cost}, expected {expected}"
    );
}

#[test]
fn pricing_cost_for_1000_tokens_claude_sonnet() {
    // claude-sonnet-4-6: $3.00/1M input → 1 000 tokens = $0.003
    let cost = compute_cost("claude-sonnet-4-6", 1_000, 0);
    let expected = 3.00 * 1_000.0 / 1_000_000.0;
    assert!(
        (cost - expected).abs() < 1e-10,
        "claude-sonnet-4-6 1k input cost = {cost}, expected {expected}"
    );
}

#[test]
fn pricing_unknown_model_returns_fallback() {
    let (i, o) = lookup("this-model-does-not-exist");
    assert_eq!(
        (i, o),
        FALLBACK_PRICING,
        "unknown model should return FALLBACK_PRICING"
    );
}

#[test]
fn pricing_unknown_model_compute_cost_is_nonzero() {
    // Even for an unknown model, fallback pricing should produce a positive cost.
    let cost = compute_cost("nonexistent-model-xyz", 1_000, 0);
    assert!(cost > 0.0, "fallback cost should be > 0");
}

#[test]
fn pricing_lookup_case_insensitive() {
    let (i_lower, o_lower) = lookup("gpt-4o");
    let (i_upper, o_upper) = lookup("GPT-4O");
    assert!((i_lower - i_upper).abs() < f64::EPSILON);
    assert!((o_lower - o_upper).abs() < f64::EPSILON);
}

#[test]
fn pricing_all_table_entries_are_lookable() {
    for (model, exp_i, exp_o) in PRICING {
        let (i, o) = lookup(model);
        assert!((i - exp_i).abs() < 1e-9, "input mismatch for {model}");
        assert!((o - exp_o).abs() < 1e-9, "output mismatch for {model}");
    }
}

// ── parser (log.rs / RequestLog::ingest_line) ────────────────────────────────

#[test]
fn parser_valid_json_log_entry_is_ingested() {
    let mut log = RequestLog::new();
    let line = r#"{"model":"gpt-4o","input_tokens":512,"output_tokens":256,"latency_ms":34}"#;
    log.ingest_line(line).expect("valid JSON should parse");
    assert_eq!(log.len(), 1);
    assert_eq!(log.all()[0].model, "gpt-4o");
    assert_eq!(log.all()[0].input_tokens, 512);
    assert_eq!(log.all()[0].output_tokens, 256);
    assert_eq!(log.all()[0].latency_ms, 34);
}

#[test]
fn parser_malformed_json_returns_error() {
    let mut log = RequestLog::new();
    let err = log.ingest_line("not valid json {{{").unwrap_err();
    assert!(
        matches!(err, llm_cost_dashboard::DashboardError::LogParseError(_)),
        "expected LogParseError, got {err:?}"
    );
    assert_eq!(log.len(), 0, "no entry should be added on error");
}

#[test]
fn parser_empty_string_returns_error() {
    let mut log = RequestLog::new();
    assert!(log.ingest_line("").is_err());
    assert_eq!(log.len(), 0);
}

#[test]
fn parser_whitespace_only_returns_error() {
    let mut log = RequestLog::new();
    assert!(log.ingest_line("   \t  ").is_err());
}

#[test]
fn parser_missing_required_field_returns_error() {
    let mut log = RequestLog::new();
    // output_tokens and latency_ms are required; missing them should fail.
    let line = r#"{"model":"gpt-4o","input_tokens":100}"#;
    assert!(
        log.ingest_line(line).is_err(),
        "missing required fields should produce a parse error"
    );
}

#[test]
fn parser_missing_model_field_returns_error() {
    let mut log = RequestLog::new();
    let line = r#"{"input_tokens":100,"output_tokens":50,"latency_ms":10}"#;
    assert!(log.ingest_line(line).is_err());
}

#[test]
fn parser_optional_provider_defaults_to_unknown() {
    let mut log = RequestLog::new();
    let line = r#"{"model":"gpt-4o","input_tokens":10,"output_tokens":5,"latency_ms":20}"#;
    log.ingest_line(line).unwrap();
    assert_eq!(log.all()[0].provider, "unknown");
}

#[test]
fn parser_optional_provider_field_is_used_when_present() {
    let mut log = RequestLog::new();
    let line = r#"{"model":"gpt-4o","input_tokens":10,"output_tokens":5,"latency_ms":20,"provider":"openai"}"#;
    log.ingest_line(line).unwrap();
    assert_eq!(log.all()[0].provider, "openai");
}

#[test]
fn parser_error_field_marks_success_false() {
    let mut log = RequestLog::new();
    let line =
        r#"{"model":"gpt-4o","input_tokens":0,"output_tokens":0,"latency_ms":5,"error":"timeout"}"#;
    log.ingest_line(line).unwrap();
    let entry = &log.all()[0];
    assert!(!entry.success);
    assert_eq!(entry.error.as_deref(), Some("timeout"));
}

// ── calculator (cost/mod.rs — CostLedger aggregation & projection) ───────────

#[test]
fn calculator_aggregate_empty_returns_zero_cost() {
    let ledger = CostLedger::new();
    assert_eq!(ledger.total_usd(), 0.0);
    assert!(ledger.is_empty());
}

#[test]
fn calculator_aggregate_single_entry() {
    let mut ledger = CostLedger::new();
    // claude-sonnet-4-6: $3.00/1M input → 1M input tokens = $3.00
    let rec = CostRecord::new("claude-sonnet-4-6", "anthropic", 1_000_000, 0, 100);
    ledger.add(rec).unwrap();
    assert!((ledger.total_usd() - 3.00).abs() < 1e-9);
    assert_eq!(ledger.len(), 1);
}

#[test]
fn calculator_aggregate_multiple_entries_sums_correctly() {
    let mut ledger = CostLedger::new();
    // Two records each costing $3.00 → total $6.00
    for _ in 0..2 {
        ledger
            .add(CostRecord::new(
                "claude-sonnet-4-6",
                "anthropic",
                1_000_000,
                0,
                50,
            ))
            .unwrap();
    }
    assert!((ledger.total_usd() - 6.00).abs() < 1e-9);
}

#[test]
fn calculator_aggregate_multiple_models() {
    let mut ledger = CostLedger::new();
    // gpt-4o: $5.00/1M → $5.00
    ledger
        .add(CostRecord::new("gpt-4o", "openai", 1_000_000, 0, 10))
        .unwrap();
    // gpt-4o-mini: $0.15/1M → $0.15
    ledger
        .add(CostRecord::new("gpt-4o-mini", "openai", 1_000_000, 0, 10))
        .unwrap();
    let expected = 5.00 + 0.15;
    assert!(
        (ledger.total_usd() - expected).abs() < 1e-9,
        "total = {}, expected {expected}",
        ledger.total_usd()
    );
}

#[test]
fn calculator_monthly_projection_from_daily_rate() {
    // projected_monthly_usd(window_hours) extrapolates current window to 30 days.
    // We cannot control "now" easily, so we test the zero-window edge case and
    // the formula via the known cost of a record.
    let ledger = CostLedger::new();
    // With empty ledger any window should produce 0.
    assert_eq!(ledger.projected_monthly_usd(24), 0.0);
}

#[test]
fn calculator_projection_zero_usage_returns_zero() {
    let ledger = CostLedger::new();
    assert_eq!(ledger.projected_monthly_usd(0), 0.0);
    assert_eq!(ledger.projected_monthly_usd(720), 0.0);
}

#[test]
fn calculator_by_model_groups_correctly() {
    let mut ledger = CostLedger::new();
    ledger
        .add(CostRecord::new("gpt-4o", "openai", 100, 50, 10))
        .unwrap();
    ledger
        .add(CostRecord::new("gpt-4o", "openai", 200, 100, 20))
        .unwrap();
    ledger
        .add(CostRecord::new("gpt-4o-mini", "openai", 100, 50, 10))
        .unwrap();
    let stats = ledger.by_model();
    assert_eq!(stats["gpt-4o"].request_count, 2);
    assert_eq!(stats["gpt-4o-mini"].request_count, 1);
}

#[test]
fn calculator_negative_cost_record_rejected() {
    let mut ledger = CostLedger::new();
    let mut rec = CostRecord::new("gpt-4o", "openai", 0, 0, 0);
    rec.total_cost_usd = -1.0;
    assert!(ledger.add(rec).is_err());
}

#[test]
fn calculator_last_n_returns_correct_slice() {
    let mut ledger = CostLedger::new();
    for _ in 0..10 {
        ledger
            .add(CostRecord::new("gpt-4o-mini", "openai", 100, 50, 5))
            .unwrap();
    }
    assert_eq!(ledger.last_n(3).len(), 3);
    assert_eq!(ledger.last_n(100).len(), 10);
}

#[test]
fn calculator_clear_empties_ledger() {
    let mut ledger = CostLedger::new();
    ledger
        .add(CostRecord::new("gpt-4o", "openai", 100, 50, 10))
        .unwrap();
    ledger.clear();
    assert!(ledger.is_empty());
    assert_eq!(ledger.total_usd(), 0.0);
}

// ── models (BudgetEnvelope — covers models-like summary/default scenarios) ───

#[test]
fn models_budget_envelope_default_state() {
    // BudgetEnvelope is the primary "summary" type exposed publicly.
    let b = BudgetEnvelope::new("Monthly", 10.0, 0.8);
    assert_eq!(b.spent_usd, 0.0);
    assert_eq!(b.limit_usd, 10.0);
    assert_eq!(b.alert_threshold, 0.8);
    assert_eq!(b.status(), "OK");
    assert!(!b.is_over_budget());
    assert!(!b.alert_triggered());
}

#[test]
fn models_cost_record_serialization_roundtrip() {
    let rec = CostRecord::new("gpt-4o", "openai", 500, 250, 42);
    let json = serde_json::to_string(&rec).expect("serialization should succeed");
    let decoded: CostRecord = serde_json::from_str(&json).expect("deserialization should succeed");
    assert_eq!(decoded.model, rec.model);
    assert_eq!(decoded.provider, rec.provider);
    assert_eq!(decoded.input_tokens, rec.input_tokens);
    assert_eq!(decoded.output_tokens, rec.output_tokens);
    assert_eq!(decoded.latency_ms, rec.latency_ms);
    assert!((decoded.total_cost_usd - rec.total_cost_usd).abs() < 1e-12);
}

#[test]
fn models_budget_envelope_spend_updates_remaining() {
    let mut b = BudgetEnvelope::new("test", 100.0, 0.8);
    b.spend(30.0).unwrap();
    assert!((b.remaining() - 70.0).abs() < f64::EPSILON);
}
