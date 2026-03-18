//! Integration tests for malformed JSON ingestion in the log parser.
#![allow(clippy::unwrap_used, clippy::expect_used)]
//!
//! Every test here feeds broken or incomplete JSON to [`RequestLog::ingest_line`]
//! (via the low-level API) and to [`App::ingest_line`] (via the high-level API)
//! and verifies that:
//!
//! 1. The call returns an `Err` rather than panicking.
//! 2. The error variant is [`DashboardError::LogParseError`].
//! 3. No partial state was written to the log or ledger.

use llm_cost_dashboard::{log::RequestLog, ui::App, DashboardError};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assert that ingesting `input` into a fresh [`RequestLog`] returns a
/// [`DashboardError::LogParseError`] and leaves the log empty.
fn assert_parse_error(input: &str) {
    let mut log = RequestLog::new();
    let result = log.ingest_line(input);
    assert!(
        result.is_err(),
        "expected error for input {:?}, got Ok",
        input
    );
    assert!(
        matches!(result.unwrap_err(), DashboardError::LogParseError(_)),
        "expected LogParseError variant for input {:?}",
        input
    );
    assert_eq!(
        log.len(),
        0,
        "log must remain empty after parse error for input {:?}",
        input
    );
}

// ---------------------------------------------------------------------------
// Truncated / syntactically broken JSON
// ---------------------------------------------------------------------------

/// A completely empty string is not valid JSON.
#[test]
fn truncated_empty_string() {
    assert_parse_error("");
}

/// Whitespace only is not valid JSON.
#[test]
fn truncated_whitespace_only() {
    assert_parse_error("   \t\n  ");
}

/// A JSON object that is cut off mid-key.
#[test]
fn truncated_mid_key() {
    assert_parse_error(r#"{"mod"#);
}

/// A JSON object that is cut off after the opening brace.
#[test]
fn truncated_opening_brace_only() {
    assert_parse_error("{");
}

/// A JSON object that has an unclosed string value.
#[test]
fn truncated_unclosed_string_value() {
    assert_parse_error(
        r#"{"model":"gpt-4o-mini","input_tokens":100,"output_tokens":50,"latency_ms":20"#,
    );
}

/// A JSON array instead of a JSON object is rejected.
#[test]
fn json_array_instead_of_object() {
    assert_parse_error(r#"["gpt-4o", 100, 50, 20]"#);
}

/// A plain JSON string is rejected.
#[test]
fn json_bare_string() {
    assert_parse_error(r#""just a string""#);
}

/// A JSON number is rejected.
#[test]
fn json_bare_number() {
    assert_parse_error("42");
}

/// A JSON boolean is rejected.
#[test]
fn json_bare_boolean() {
    assert_parse_error("true");
}

/// JSON null is rejected.
#[test]
fn json_bare_null() {
    assert_parse_error("null");
}

/// Completely unparseable text is rejected.
#[test]
fn garbage_text() {
    assert_parse_error("not json at all !!!");
}

/// An object with a trailing comma (invalid JSON) is rejected.
#[test]
fn trailing_comma_in_object() {
    assert_parse_error(
        r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50,"latency_ms":20,}"#,
    );
}

/// Duplicate keys -- serde_json's default behaviour keeps the last value, so
/// this is actually valid JSON. We test it to ensure no panic even if behaviour
/// changes.
#[test]
fn duplicate_keys_no_panic() {
    // Duplicate `model` key: serde_json keeps the last value ("gpt-4o").
    // Required fields are present, so this must succeed without panicking.
    let line = r#"{"model":"first","model":"gpt-4o","input_tokens":100,"output_tokens":50,"latency_ms":10}"#;
    let mut log = RequestLog::new();
    // Either Ok or Err is acceptable; the important guarantee is no panic.
    let _ = log.ingest_line(line);
}

// ---------------------------------------------------------------------------
// Missing required fields
// ---------------------------------------------------------------------------

/// `model` field is absent.
#[test]
fn missing_model_field() {
    assert_parse_error(r#"{"input_tokens":100,"output_tokens":50,"latency_ms":10}"#);
}

/// `input_tokens` field is absent.
#[test]
fn missing_input_tokens_field() {
    assert_parse_error(r#"{"model":"gpt-4o","output_tokens":50,"latency_ms":10}"#);
}

/// `output_tokens` field is absent.
#[test]
fn missing_output_tokens_field() {
    assert_parse_error(r#"{"model":"gpt-4o","input_tokens":100,"latency_ms":10}"#);
}

/// `latency_ms` field is absent.
#[test]
fn missing_latency_ms_field() {
    assert_parse_error(r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50}"#);
}

/// All required fields are absent; only an unknown field is present.
#[test]
fn only_unknown_field() {
    assert_parse_error(r#"{"unknown_key":"some_value"}"#);
}

/// An empty JSON object has no required fields.
#[test]
fn empty_json_object() {
    assert_parse_error("{}");
}

// ---------------------------------------------------------------------------
// Wrong field types
// ---------------------------------------------------------------------------

/// `input_tokens` is a string instead of a u64.
#[test]
fn wrong_type_input_tokens_string() {
    assert_parse_error(
        r#"{"model":"gpt-4o","input_tokens":"lots","output_tokens":50,"latency_ms":10}"#,
    );
}

/// `output_tokens` is a boolean.
#[test]
fn wrong_type_output_tokens_bool() {
    assert_parse_error(
        r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":true,"latency_ms":10}"#,
    );
}

/// `latency_ms` is null.
#[test]
fn wrong_type_latency_ms_null() {
    assert_parse_error(
        r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50,"latency_ms":null}"#,
    );
}

/// `model` is a number instead of a string.
#[test]
fn wrong_type_model_number() {
    assert_parse_error(r#"{"model":42,"input_tokens":100,"output_tokens":50,"latency_ms":10}"#);
}

/// `input_tokens` is a negative integer.  serde rejects negative values for u64.
#[test]
fn negative_input_tokens_rejected() {
    assert_parse_error(
        r#"{"model":"gpt-4o","input_tokens":-1,"output_tokens":50,"latency_ms":10}"#,
    );
}

/// `output_tokens` is a negative integer.
#[test]
fn negative_output_tokens_rejected() {
    assert_parse_error(
        r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":-50,"latency_ms":10}"#,
    );
}

/// `latency_ms` is a floating-point number; serde_json will attempt to coerce.
/// We do not mandate a specific outcome (ok or err) but the call must not panic.
#[test]
fn float_latency_ms_no_panic() {
    let line = r#"{"model":"gpt-4o","input_tokens":100,"output_tokens":50,"latency_ms":10.5}"#;
    let mut log = RequestLog::new();
    // Either Ok or Err is fine; must not panic.
    let _ = log.ingest_line(line);
}

// ---------------------------------------------------------------------------
// State integrity after errors
// ---------------------------------------------------------------------------

/// A previously ingested valid entry is preserved when subsequent malformed
/// lines are processed.
#[test]
fn valid_entry_preserved_after_errors() {
    let mut log = RequestLog::new();
    let good = r#"{"model":"gpt-4o-mini","input_tokens":512,"output_tokens":256,"latency_ms":20}"#;
    log.ingest_line(good)
        .expect("valid line should be accepted");
    assert_eq!(log.len(), 1);

    let bad_inputs = [
        "",
        "not json",
        "{",
        r#"{"model":"gpt-4o"}"#,
        r#"{"model":42,"input_tokens":100,"output_tokens":50,"latency_ms":10}"#,
    ];
    for bad in &bad_inputs {
        let _ = log.ingest_line(bad);
    }

    assert_eq!(
        log.len(),
        1,
        "valid entry must survive all subsequent malformed inputs"
    );
    assert_eq!(log.all()[0].model, "gpt-4o-mini");
}

/// The App-level `ingest_line` preserves ledger and log integrity across mixed
/// valid/invalid input.
#[test]
fn app_state_integrity_across_mixed_input() {
    let mut app = App::new(100.0);
    let good =
        r#"{"model":"claude-sonnet-4-6","input_tokens":1000,"output_tokens":500,"latency_ms":50}"#;
    app.ingest_line(good)
        .expect("valid line should be accepted");

    let malformed = [
        "",
        "definitely not json",
        "{}",
        r#"{"model":"gpt-4o","input_tokens":-1,"output_tokens":50,"latency_ms":10}"#,
        r#"{"model":"gpt-4o","input_tokens":"many","output_tokens":50,"latency_ms":10}"#,
        r#"{"input_tokens":100,"output_tokens":50,"latency_ms":10}"#,
    ];

    for bad in &malformed {
        let result = app.ingest_line(bad);
        assert!(
            result.is_err(),
            "expected error for malformed input {:?}",
            bad
        );
    }

    assert_eq!(
        app.log.len(),
        1,
        "only the one valid entry should be present"
    );
    assert_eq!(
        app.ledger.len(),
        1,
        "ledger must only contain the one valid record"
    );
    assert!(
        app.ledger.total_usd() > 0.0,
        "valid record cost must be > 0"
    );
}

/// Feeding only malformed lines to a fresh App leaves it in a clean state.
#[test]
fn app_stays_clean_on_all_malformed_input() {
    let mut app = App::new(50.0);
    let malformed = ["not json", "{", "[]", r#"{"model":"gpt-4o"}"#, ""];
    for bad in &malformed {
        let _ = app.ingest_line(bad);
    }
    assert!(
        app.log.is_empty(),
        "log must be empty after only malformed input"
    );
    assert!(
        app.ledger.is_empty(),
        "ledger must be empty after only malformed input"
    );
    assert_eq!(app.budget.spent_usd, 0.0);
}
