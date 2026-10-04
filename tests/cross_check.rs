//! Cross-check prices against llm-cost-cap 0.1, an independent Rust crate
//! with its own built-in table.
//!
//! Where both crates list a model, the input and output prices must match,
//! except for the models whose llm-cost-cap prices are outdated (checked
//! against Anthropic's published price list in September 2026):
//! claude-haiku-4-5 is $1 / $5 per million tokens (llm-cost-cap: $0.80 / $4),
//! and claude-opus-4-5 / 4-6 / 4-7 are $5 / $25 (llm-cost-cap: $15 / $75).

use llm_cost_dashboard::cost::pricing::lookup_known;

const KNOWN_STALE_IN_LLM_COST_CAP: [&str; 4] = ["claude-haiku-4-5", "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7"];

#[test]
fn prices_agree_with_llm_cost_cap() {
    let mut compared = 0;
    for (model, input, output) in llm_cost_cap::MODEL_PRICES {
        let Some((i, o)) = lookup_known(model) else {
            continue;
        };
        if KNOWN_STALE_IN_LLM_COST_CAP.contains(model) {
            assert!(
                (i, o) != (*input, *output),
                "{model}: llm-cost-cap was updated, re-check and drop it from the stale list"
            );
            continue;
        }
        assert!(
            (i - input).abs() < 1e-9 && (o - output).abs() < 1e-9,
            "{model}: this crate {i}/{o}, llm-cost-cap {input}/{output}"
        );
        compared += 1;
    }
    assert!(compared >= 6, "only {compared} models in common");
}
