//! Fuzz / property tests for the pricing table and cost computation helpers.
//!
//! These tests cover edge cases that normal unit tests miss:
//! - Zero-token inputs
//! - Extremely large token counts (u64::MAX)
//! - Unknown model names (fallback pricing path)
//! - Mismatched input/output token ratios
//! - Determinism and linearity properties across all table entries

use llm_cost_dashboard::cost::pricing::{compute_cost, lookup, FALLBACK_PRICING, PRICING};

// ---------------------------------------------------------------------------
// Table-driven edge-case tests
// ---------------------------------------------------------------------------

/// Every model in the table produces zero cost for zero tokens.
#[test]
fn zero_tokens_always_zero_cost() {
    for (model, _, _) in PRICING {
        let cost = compute_cost(model, 0, 0);
        assert_eq!(
            cost, 0.0,
            "model '{model}' produced non-zero cost for zero tokens: {cost}"
        );
    }
}

/// Unknown model names fall back to FALLBACK_PRICING and produce zero cost for
/// zero tokens.
#[test]
fn unknown_model_zero_tokens_is_zero() {
    let unknowns = [
        "",
        "totally-unknown",
        "gpt-99",
        "claude-99",
        "UNKNOWN_MODEL",
        "model-with-special-chars-!@#",
    ];
    for name in &unknowns {
        let cost = compute_cost(name, 0, 0);
        assert_eq!(
            cost, 0.0,
            "unknown model '{name}' returned non-zero cost for zero tokens: {cost}"
        );
    }
}

/// Unknown model names resolve to FALLBACK_PRICING.
#[test]
fn unknown_model_lookup_returns_fallback() {
    let (fi, fo) = FALLBACK_PRICING;
    let unknowns = ["no-such-model", "gpt-4o-ultra-super", "", "llama-3-8b"];
    for name in &unknowns {
        let (i, o) = lookup(name);
        assert!(
            (i - fi).abs() < f64::EPSILON && (o - fo).abs() < f64::EPSILON,
            "unknown model '{name}' did not return FALLBACK_PRICING: got ({i}, {o})"
        );
    }
}

/// u64::MAX tokens must not panic; the result may be infinite but must not be
/// NaN (NaN would indicate an unhandled arithmetic error).
#[test]
fn max_u64_input_tokens_no_panic_no_nan() {
    for (model, _, _) in PRICING {
        let cost = compute_cost(model, u64::MAX, 0);
        assert!(
            !cost.is_nan(),
            "model '{model}' produced NaN for u64::MAX input tokens"
        );
    }
}

/// u64::MAX tokens on the output side must not panic or produce NaN.
#[test]
fn max_u64_output_tokens_no_panic_no_nan() {
    for (model, _, _) in PRICING {
        let cost = compute_cost(model, 0, u64::MAX);
        assert!(
            !cost.is_nan(),
            "model '{model}' produced NaN for u64::MAX output tokens"
        );
    }
}

/// u64::MAX on both sides must not panic or produce NaN.
#[test]
fn max_u64_both_tokens_no_panic_no_nan() {
    for (model, _, _) in PRICING {
        let cost = compute_cost(model, u64::MAX, u64::MAX);
        assert!(
            !cost.is_nan(),
            "model '{model}' produced NaN for u64::MAX input and output tokens"
        );
    }
}

/// Unknown model with u64::MAX tokens must not panic and must not return NaN.
#[test]
fn unknown_model_max_u64_no_panic_no_nan() {
    let cost = compute_cost("unknown-model-xyz", u64::MAX, u64::MAX);
    assert!(!cost.is_nan(), "unknown model + u64::MAX produced NaN");
}

/// Pure-input cost (output = 0) is always less than or equal to combined cost
/// (output > 0) for the same input count, because output rates are positive.
#[test]
fn combined_cost_ge_input_only_cost() {
    for (model, _, _) in PRICING {
        let input_only = compute_cost(model, 1_000_000, 0);
        let combined = compute_cost(model, 1_000_000, 1_000_000);
        assert!(
            combined >= input_only,
            "model '{model}': combined cost {combined} < input-only {input_only}"
        );
    }
}

/// Pure-output cost (input = 0) is always less than or equal to combined cost.
#[test]
fn combined_cost_ge_output_only_cost() {
    for (model, _, _) in PRICING {
        let output_only = compute_cost(model, 0, 1_000_000);
        let combined = compute_cost(model, 1_000_000, 1_000_000);
        assert!(
            combined >= output_only,
            "model '{model}': combined cost {combined} < output-only {output_only}"
        );
    }
}

/// Combined cost equals the sum of the per-component costs.
#[test]
fn combined_cost_equals_sum_of_parts() {
    for (model, _, _) in PRICING {
        let input_only = compute_cost(model, 500_000, 0);
        let output_only = compute_cost(model, 0, 250_000);
        let combined = compute_cost(model, 500_000, 250_000);
        let sum = input_only + output_only;
        assert!(
            (combined - sum).abs() < 1e-9,
            "model '{model}': combined {combined} != input {input_only} + output {output_only}"
        );
    }
}

/// Models with higher listed output rates always produce higher output cost per
/// million tokens than models with lower output rates.
/// We spot-check a few known pairs from the pricing table.
#[test]
fn higher_output_rate_model_costs_more() {
    // claude-opus-4-6 output: $75.00/1M vs claude-haiku-4-5 output: $1.25/1M
    let opus_cost = compute_cost("claude-opus-4-6", 0, 1_000_000);
    let haiku_cost = compute_cost("claude-haiku-4-5", 0, 1_000_000);
    assert!(
        opus_cost > haiku_cost,
        "opus output cost {opus_cost} should exceed haiku {haiku_cost}"
    );
}

/// A model with more input tokens than output tokens should not produce a cost
/// lower than a model with fewer input tokens, given the same rates.
/// Tests the ratio case: 99:1 input:output vs 1:99 input:output.
#[test]
fn high_input_ratio_vs_high_output_ratio() {
    // gpt-4o-mini: $0.15 input / $0.60 output
    // 990_000 input + 10_000 output vs 10_000 input + 990_000 output
    let input_heavy = compute_cost("gpt-4o-mini", 990_000, 10_000);
    let output_heavy = compute_cost("gpt-4o-mini", 10_000, 990_000);
    // Output rate is 4x the input rate for gpt-4o-mini, so output-heavy
    // should cost considerably more.
    assert!(
        output_heavy > input_heavy,
        "output-heavy cost {output_heavy} should exceed input-heavy {input_heavy}"
    );
}

/// Lookup is deterministic across multiple calls with the same input.
#[test]
fn lookup_is_deterministic() {
    for (model, _, _) in PRICING {
        let (i1, o1) = lookup(model);
        let (i2, o2) = lookup(model);
        assert_eq!(i1, i2, "model '{model}' input rate changed between calls");
        assert_eq!(o1, o2, "model '{model}' output rate changed between calls");
    }
}

/// Lookup is case-insensitive for all table entries.
#[test]
fn lookup_case_insensitive_for_all_entries() {
    for (model, exp_i, exp_o) in PRICING {
        let upper = model.to_uppercase();
        let (i, o) = lookup(&upper);
        assert!(
            (i - exp_i).abs() < 1e-9,
            "uppercase lookup for '{model}' input rate mismatch: {i} vs {exp_i}"
        );
        assert!(
            (o - exp_o).abs() < 1e-9,
            "uppercase lookup for '{model}' output rate mismatch: {o} vs {exp_o}"
        );
    }
}

/// A single token produces a cost strictly between zero and one for all
/// models in the table (sanity-check that rates are in USD/1M not USD/1).
#[test]
fn single_token_cost_is_fractional() {
    for (model, _, _) in PRICING {
        let cost = compute_cost(model, 1, 0);
        assert!(
            cost > 0.0,
            "model '{model}': single input token should have positive cost, got {cost}"
        );
        assert!(
            cost < 1.0,
            "model '{model}': single input token cost {cost} is >= $1 -- rates may be wrong"
        );
    }
}

// ---------------------------------------------------------------------------
// Property tests using proptest
// ---------------------------------------------------------------------------

proptest::proptest! {
    /// Cost is non-negative for any combination of valid token counts.
    #[test]
    fn prop_cost_non_negative_all_models(
        input in 0u64..=10_000_000u64,
        output in 0u64..=10_000_000u64,
        idx in 0usize..16usize,
    ) {
        // Clamp idx to table length to stay in bounds.
        let idx = idx % PRICING.len();
        let cost = compute_cost(PRICING[idx].0, input, output);
        proptest::prop_assert!(
            cost >= 0.0,
            "cost was negative ({cost}) for model '{}' input={input} output={output}",
            PRICING[idx].0
        );
    }

    /// Cost is additive: compute_cost(m, a+b, 0) == compute_cost(m, a, 0) + compute_cost(m, b, 0).
    #[test]
    fn prop_cost_additive_input(
        a in 0u64..=5_000_000u64,
        b in 0u64..=5_000_000u64,
        idx in 0usize..16usize,
    ) {
        let idx = idx % PRICING.len();
        let model = PRICING[idx].0;
        let combined = compute_cost(model, a + b, 0);
        let split = compute_cost(model, a, 0) + compute_cost(model, b, 0);
        proptest::prop_assert!(
            (combined - split).abs() < 1e-6,
            "additivity violated for '{model}': combined={combined} split={split}"
        );
    }

    /// Cost is additive on the output side.
    #[test]
    fn prop_cost_additive_output(
        a in 0u64..=5_000_000u64,
        b in 0u64..=5_000_000u64,
        idx in 0usize..16usize,
    ) {
        let idx = idx % PRICING.len();
        let model = PRICING[idx].0;
        let combined = compute_cost(model, 0, a + b);
        let split = compute_cost(model, 0, a) + compute_cost(model, 0, b);
        proptest::prop_assert!(
            (combined - split).abs() < 1e-6,
            "output additivity violated for '{model}': combined={combined} split={split}"
        );
    }

    /// Unknown model names never produce NaN cost, regardless of token counts.
    #[test]
    fn prop_unknown_model_never_nan(
        input in 0u64..=u64::MAX,
        output in 0u64..=u64::MAX,
    ) {
        let cost = compute_cost("completely-unknown-model-xyz-123", input, output);
        proptest::prop_assert!(!cost.is_nan(), "NaN cost for unknown model");
    }

    /// Zero-token cost is always exactly zero for any model index.
    #[test]
    fn prop_zero_tokens_zero_cost(idx in 0usize..16usize) {
        let idx = idx % PRICING.len();
        let cost = compute_cost(PRICING[idx].0, 0, 0);
        proptest::prop_assert_eq!(cost, 0.0);
    }
}
