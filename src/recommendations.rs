//! # Model Recommendation Engine
//!
//! Analyzes usage patterns from the cost ledger and suggests cheaper alternative
//! models, computing projected monthly savings for each recommendation.
//!
//! ## Usage
//!
//! ```rust
//! use llm_cost_dashboard::{CostLedger, CostRecord};
//! use llm_cost_dashboard::recommendations::ModelRecommender;
//!
//! let mut ledger = CostLedger::new();
//! ledger.add(CostRecord::new("claude-sonnet-4-6", "anthropic", 200, 50, 40)).unwrap();
//!
//! let recommender = ModelRecommender::new(&ledger);
//! let suggestions = recommender.suggest();
//! for s in &suggestions {
//!     println!("{}", s.summary_line());
//! }
//! ```

use std::collections::HashMap;

use crate::cost::pricing::lookup;
use crate::cost::CostLedger;

/// A usage pattern extracted from ledger records for a single model.
#[derive(Debug, Clone)]
pub struct UsagePattern {
    /// The model in use.
    pub model: String,
    /// Number of requests observed.
    pub request_count: u64,
    /// Average input tokens per request.
    pub avg_input_tokens: f64,
    /// Average output tokens per request.
    pub avg_output_tokens: f64,
    /// Total USD cost recorded.
    pub total_cost_usd: f64,
    /// Average cost per request in USD.
    pub avg_cost_per_request: f64,
}

/// A recommendation to switch from one model to a cheaper alternative.
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// The model currently being used.
    pub current_model: String,
    /// The cheaper model suggested as an alternative.
    pub suggested_model: String,
    /// Percentage cost saving (0-100).
    pub saving_pct: f64,
    /// Projected monthly savings in USD (based on current usage rates).
    pub projected_monthly_saving_usd: f64,
    /// Current monthly cost projection in USD.
    pub current_monthly_cost_usd: f64,
    /// Suggested monthly cost projection in USD.
    pub suggested_monthly_cost_usd: f64,
    /// Human-readable reason for the suggestion.
    pub reason: String,
}

impl Recommendation {
    /// One-line summary suitable for display in a TUI panel.
    pub fn summary_line(&self) -> String {
        format!(
            "{} → {} | save {:.0}% (${:.2}/mo)",
            short_model(&self.current_model),
            short_model(&self.suggested_model),
            self.saving_pct,
            self.projected_monthly_saving_usd,
        )
    }
}

fn short_model(model: &str) -> &str {
    // Trim to max 20 chars for display.
    if model.len() > 20 {
        &model[..20]
    } else {
        model
    }
}

/// Alternative model candidates keyed by current model.
///
/// Each entry is a list of `(alternative_model, reason)` pairs, ordered from
/// cheapest to most capable.
static ALTERNATIVES: &[(&str, &[(&str, &str)])] = &[
    (
        "claude-opus-4-6",
        &[
            (
                "claude-sonnet-4-6",
                "5x cheaper; handles most tasks equally well",
            ),
            (
                "claude-haiku-4-5",
                "60x cheaper; ideal for short/simple queries",
            ),
        ],
    ),
    (
        "claude-sonnet-4-6",
        &[
            (
                "claude-haiku-4-5",
                "12x cheaper; great for short queries <500 tokens",
            ),
            (
                "gpt-4o-mini",
                "20x cheaper; competitive quality on simple tasks",
            ),
        ],
    ),
    (
        "gpt-4o",
        &[
            (
                "gpt-4o-mini",
                "33x cheaper; handles most chat and completion tasks",
            ),
            ("claude-haiku-4-5", "20x cheaper; fast and cost-effective"),
        ],
    ),
    (
        "gpt-4-turbo",
        &[
            (
                "gpt-4o",
                "2x cheaper; equivalent quality on most benchmarks",
            ),
            ("gpt-4o-mini", "66x cheaper; suitable for simple workloads"),
        ],
    ),
    (
        "o1",
        &[
            (
                "o3-mini",
                "13x cheaper; similar reasoning for most problems",
            ),
            ("gpt-4o", "3x cheaper; good for non-reasoning tasks"),
        ],
    ),
    (
        "o3",
        &[
            (
                "o3-mini",
                "9x cheaper; comparable reasoning on most benchmarks",
            ),
            (
                "gpt-4o-mini",
                "66x cheaper; for tasks not requiring deep reasoning",
            ),
        ],
    ),
    (
        "gemini-1.5-pro",
        &[
            ("gemini-1.5-flash", "46x cheaper; optimised for throughput"),
            ("gemini-2.0-flash", "35x cheaper; newer, faster, cheaper"),
        ],
    ),
    (
        "claude-3-opus-20240229",
        &[
            (
                "claude-sonnet-4-6",
                "5x cheaper; more capable on most tasks",
            ),
            (
                "claude-haiku-4-5",
                "60x cheaper; for high-volume short requests",
            ),
        ],
    ),
    (
        "gpt-4.5-preview",
        &[
            ("gpt-4o", "15x cheaper; production-ready quality"),
            ("gpt-4o-mini", "500x cheaper; for simple tasks"),
        ],
    ),
    (
        "mistral-large-2411",
        &[
            (
                "mistral-small-2501",
                "20x cheaper; good for instruction-following",
            ),
            ("mistral-nemo", "13x cheaper; efficient and capable"),
        ],
    ),
];

/// Analyzes a [`CostLedger`] and produces model switch recommendations.
pub struct ModelRecommender<'a> {
    ledger: &'a CostLedger,
}

impl<'a> ModelRecommender<'a> {
    /// Create a new recommender bound to the given ledger.
    pub fn new(ledger: &'a CostLedger) -> Self {
        Self { ledger }
    }

    /// Extract usage patterns per model from the ledger.
    pub fn patterns(&self) -> Vec<UsagePattern> {
        let mut by_model: HashMap<String, Vec<_>> = HashMap::new();
        for r in self.ledger.records() {
            by_model.entry(r.model.clone()).or_default().push(r);
        }

        by_model
            .into_iter()
            .map(|(model, recs)| {
                let count = recs.len() as u64;
                let total_in: u64 = recs.iter().map(|r| r.input_tokens).sum();
                let total_out: u64 = recs.iter().map(|r| r.output_tokens).sum();
                let total_cost: f64 = recs.iter().map(|r| r.total_cost_usd).sum();
                let avg_cost = if count > 0 {
                    total_cost / count as f64
                } else {
                    0.0
                };
                let avg_in = if count > 0 {
                    total_in as f64 / count as f64
                } else {
                    0.0
                };
                let avg_out = if count > 0 {
                    total_out as f64 / count as f64
                } else {
                    0.0
                };
                UsagePattern {
                    model,
                    request_count: count,
                    avg_input_tokens: avg_in,
                    avg_output_tokens: avg_out,
                    total_cost_usd: total_cost,
                    avg_cost_per_request: avg_cost,
                }
            })
            .collect()
    }

    /// Generate savings recommendations sorted by projected monthly saving (largest first).
    pub fn suggest(&self) -> Vec<Recommendation> {
        let patterns = self.patterns();
        // Estimate requests per month using all-time rate extrapolated over 30 days.
        // We approximate: if ledger has N records total, monthly rate ≈ N (single-session
        // approximation). For real-time use the caller should pass elapsed hours; we use
        // the simpler per-session total here.
        let total_records = self.ledger.len() as f64;
        let mut recs: Vec<Recommendation> = Vec::new();

        for pattern in &patterns {
            let model_lower = pattern.model.to_lowercase();
            let alternatives_opt = ALTERNATIVES
                .iter()
                .find(|(m, _)| m.eq_ignore_ascii_case(&model_lower))
                .map(|(_, alts)| *alts);

            if let Some(alternatives) = alternatives_opt {
                let (cur_in_rate, cur_out_rate) = lookup(&pattern.model);
                // current cost per request using actual avg tokens
                let cur_cost_per_req = (pattern.avg_input_tokens * cur_in_rate
                    + pattern.avg_output_tokens * cur_out_rate)
                    / 1_000_000.0;

                // Fraction of all requests this model accounts for.
                let model_fraction = if total_records > 0.0 {
                    pattern.request_count as f64 / total_records
                } else {
                    0.0
                };
                // Monthly request count estimate for this model (rough).
                let monthly_req = pattern.request_count as f64 * model_fraction * 30.0;
                let current_monthly = cur_cost_per_req * monthly_req;

                for (alt_model, reason) in alternatives.iter() {
                    let (alt_in, alt_out) = lookup(alt_model);
                    let alt_cost_per_req = (pattern.avg_input_tokens * alt_in
                        + pattern.avg_output_tokens * alt_out)
                        / 1_000_000.0;
                    let suggested_monthly = alt_cost_per_req * monthly_req;

                    if alt_cost_per_req >= cur_cost_per_req {
                        // Not actually cheaper — skip.
                        continue;
                    }

                    let saving = current_monthly - suggested_monthly;
                    let saving_pct = if cur_cost_per_req > 0.0 {
                        (1.0 - alt_cost_per_req / cur_cost_per_req) * 100.0
                    } else {
                        0.0
                    };

                    recs.push(Recommendation {
                        current_model: pattern.model.clone(),
                        suggested_model: alt_model.to_string(),
                        saving_pct,
                        projected_monthly_saving_usd: saving.max(0.0),
                        current_monthly_cost_usd: current_monthly,
                        suggested_monthly_cost_usd: suggested_monthly,
                        reason: reason.to_string(),
                    });
                }
            }
        }

        // Sort by largest saving first.
        recs.sort_by(|a, b| {
            b.projected_monthly_saving_usd
                .partial_cmp(&a.projected_monthly_saving_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        recs
    }

    /// Total projected monthly saving across all top recommendations (one per model).
    pub fn total_projected_monthly_saving(&self) -> f64 {
        let suggestions = self.suggest();
        // Take only the best (cheapest) suggestion per current model.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        suggestions
            .iter()
            .filter(|r| seen.insert(r.current_model.clone()))
            .map(|r| r.projected_monthly_saving_usd)
            .sum()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cost::CostRecord;

    fn make_ledger_with(entries: &[(&str, &str, u64, u64)]) -> CostLedger {
        let mut l = CostLedger::new();
        for (model, provider, inp, out) in entries {
            l.add(CostRecord::new(*model, *provider, *inp, *out, 50))
                .unwrap();
        }
        l
    }

    #[test]
    fn test_patterns_groups_by_model() {
        let ledger = make_ledger_with(&[
            ("claude-sonnet-4-6", "anthropic", 500, 100),
            ("claude-sonnet-4-6", "anthropic", 700, 200),
            ("gpt-4o-mini", "openai", 300, 50),
        ]);
        let r = ModelRecommender::new(&ledger);
        let patterns = r.patterns();
        assert_eq!(patterns.len(), 2);
        let sonnet = patterns
            .iter()
            .find(|p| p.model == "claude-sonnet-4-6")
            .unwrap();
        assert_eq!(sonnet.request_count, 2);
    }

    #[test]
    fn test_suggest_returns_recommendations_for_expensive_models() {
        let ledger = make_ledger_with(&[
            ("claude-sonnet-4-6", "anthropic", 500, 100),
            ("claude-sonnet-4-6", "anthropic", 500, 100),
        ]);
        let r = ModelRecommender::new(&ledger);
        let suggestions = r.suggest();
        assert!(
            !suggestions.is_empty(),
            "should suggest cheaper alternatives"
        );
        // All suggestions should be for cheaper models.
        for s in &suggestions {
            assert!(s.saving_pct > 0.0);
        }
    }

    #[test]
    fn test_suggest_sorted_by_saving() {
        let ledger = make_ledger_with(&[
            ("claude-sonnet-4-6", "anthropic", 1000, 500),
            ("gpt-4o", "openai", 1000, 500),
        ]);
        let r = ModelRecommender::new(&ledger);
        let suggestions = r.suggest();
        for w in suggestions.windows(2) {
            assert!(
                w[0].projected_monthly_saving_usd >= w[1].projected_monthly_saving_usd,
                "suggestions should be sorted descending by saving"
            );
        }
    }

    #[test]
    fn test_empty_ledger_no_suggestions() {
        let ledger = CostLedger::new();
        let r = ModelRecommender::new(&ledger);
        assert!(r.suggest().is_empty());
    }

    #[test]
    fn test_summary_line_format() {
        let rec = Recommendation {
            current_model: "claude-sonnet-4-6".into(),
            suggested_model: "claude-haiku-4-5".into(),
            saving_pct: 60.0,
            projected_monthly_saving_usd: 12.5,
            current_monthly_cost_usd: 20.0,
            suggested_monthly_cost_usd: 7.5,
            reason: "test".into(),
        };
        let line = rec.summary_line();
        assert!(line.contains("60%"));
        assert!(line.contains("12.50"));
    }

    #[test]
    fn test_total_projected_saving_non_negative() {
        let ledger = make_ledger_with(&[("claude-sonnet-4-6", "anthropic", 500, 200)]);
        let r = ModelRecommender::new(&ledger);
        assert!(r.total_projected_monthly_saving() >= 0.0);
    }
}
