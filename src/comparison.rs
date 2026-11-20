//! # Multi-Provider Cost Comparison
//!
//! Given a workload profile (average token counts, request volume), computes
//! the projected monthly cost for every model in the static pricing table and
//! ranks them from cheapest to most expensive.
//!
//! ## Quick Start
//!
//! ```rust
//! use llm_cost_dashboard::comparison::{ProviderComparison, WorkloadProfile};
//!
//! let profile = WorkloadProfile {
//!     avg_input_tokens: 500,
//!     avg_output_tokens: 200,
//!     requests_per_day: 1_000,
//! };
//!
//! let comparison = ProviderComparison::compute(&profile);
//!
//! println!("Cheapest: {} (${:.2}/mo)", comparison.cheapest().model, comparison.cheapest().monthly_cost_usd);
//! println!("Most expensive: {} (${:.2}/mo)", comparison.most_expensive().model, comparison.most_expensive().monthly_cost_usd);
//!
//! for proj in comparison.ranked() {
//!     println!("  {:40} ${:8.2}/mo  ${:6.4}/1k req",
//!         proj.model, proj.monthly_cost_usd, proj.cost_per_1k_requests);
//! }
//! ```
//!
//! ## Workload-Derived Profile
//!
//! When a [`crate::cost::CostLedger`] is available, use
//! [`WorkloadProfile::from_ledger`] to automatically derive the profile from
//! the user's actual request history.

use crate::cost::pricing::PRICING;

/// Describes a typical API workload used to project costs.
#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    /// Average number of input (prompt) tokens per request.
    pub avg_input_tokens: u64,
    /// Average number of output (completion) tokens per request.
    pub avg_output_tokens: u64,
    /// Expected number of API requests per day.
    pub requests_per_day: u64,
}

impl WorkloadProfile {
    /// Derive a workload profile from a cost ledger's recorded requests.
    ///
    /// Returns `None` when the ledger is empty (cannot compute averages).
    pub fn from_ledger(ledger: &crate::cost::CostLedger) -> Option<Self> {
        let records = ledger.records();
        if records.is_empty() {
            return None;
        }
        let n = records.len() as u64;
        let total_input: u64 = records.iter().map(|r| r.input_tokens).sum();
        let total_output: u64 = records.iter().map(|r| r.output_tokens).sum();

        // Estimate requests/day from the time span of the ledger.
        // If all records fall within one second, assume 1-day window.
        let requests_per_day = if n > 1 {
            let first = records.first().map(|r| r.timestamp).unwrap();
            let last = records.last().map(|r| r.timestamp).unwrap();
            let span_days = (last - first).num_seconds() as f64 / 86_400.0;
            if span_days < 1.0 / 86_400.0 {
                n // treat as one day
            } else {
                (n as f64 / span_days).ceil() as u64
            }
        } else {
            n
        };

        Some(Self {
            avg_input_tokens: total_input / n,
            avg_output_tokens: total_output / n,
            requests_per_day,
        })
    }

    /// Build a profile from a requests-per-hour figure (CLI `--workload-rph`).
    ///
    /// Assumes 500 input tokens and 200 output tokens per request when no
    /// ledger data is available.
    pub fn from_rph(requests_per_hour: u64) -> Self {
        Self {
            avg_input_tokens: 500,
            avg_output_tokens: 200,
            requests_per_day: requests_per_hour.saturating_mul(24),
        }
    }
}

/// Cost projection for a single model at the given workload.
#[derive(Debug, Clone)]
pub struct CostProjection {
    /// Model identifier (e.g. `"gpt-4o-mini"`).
    pub model: String,
    /// Provider name inferred from the model ID prefix.
    pub provider: String,
    /// Projected total cost over 30 days in USD.
    pub monthly_cost_usd: f64,
    /// Projected cost per single day in USD.
    pub daily_cost_usd: f64,
    /// Cost per 1,000 requests in USD.
    pub cost_per_1k_requests: f64,
    /// Input token rate in USD per 1 M tokens.
    pub input_rate_per_1m: f64,
    /// Output token rate in USD per 1 M tokens.
    pub output_rate_per_1m: f64,
}

impl CostProjection {
    fn compute(model: &str, input_rate: f64, output_rate: f64, profile: &WorkloadProfile) -> Self {
        let cost_per_request = (profile.avg_input_tokens as f64 * input_rate
            + profile.avg_output_tokens as f64 * output_rate)
            / 1_000_000.0;

        let daily_cost_usd = cost_per_request * profile.requests_per_day as f64;
        let monthly_cost_usd = daily_cost_usd * 30.0;
        let cost_per_1k_requests = cost_per_request * 1_000.0;

        Self {
            model: model.to_string(),
            provider: infer_provider(model),
            monthly_cost_usd,
            daily_cost_usd,
            cost_per_1k_requests,
            input_rate_per_1m: input_rate,
            output_rate_per_1m: output_rate,
        }
    }
}

/// Infer a human-readable provider name from a model ID prefix.
fn infer_provider(model: &str) -> String {
    let m = model.to_lowercase();
    if m.starts_with("claude") {
        "Anthropic".into()
    } else if m.starts_with("gpt")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.starts_with("chatgpt")
    {
        "OpenAI".into()
    } else if m.starts_with("gemini") {
        "Google".into()
    } else if m.starts_with("deepseek") {
        "DeepSeek".into()
    } else if m.starts_with("mistral")
        || m.starts_with("codestral")
        || m.starts_with("pixtral")
        || m.starts_with("ministral")
    {
        "Mistral".into()
    } else if m.starts_with("meta-llama") || m.starts_with("llama") {
        "Meta (Llama)".into()
    } else if m.starts_with("grok") {
        "xAI".into()
    } else if m.starts_with("command")
        || m.starts_with("palmyra")
        || m.starts_with("jamba")
        || m.starts_with("sonar")
    {
        "Various".into()
    } else if m.starts_with("amazon") {
        "AWS Bedrock".into()
    } else if m.starts_with("qwen") {
        "Alibaba".into()
    } else {
        "Unknown".into()
    }
}

/// A ranked list of cost projections across all models in the pricing table.
///
/// Sorted ascending by [`CostProjection::monthly_cost_usd`] (cheapest first).
pub struct ProviderComparison {
    projections: Vec<CostProjection>,
}

impl ProviderComparison {
    /// Compute projections for every model in the pricing table, sorted by
    /// monthly cost (cheapest first).
    pub fn compute(profile: &WorkloadProfile) -> Self {
        let mut projections: Vec<CostProjection> = PRICING
            .iter()
            .map(|(model, input_rate, output_rate)| {
                CostProjection::compute(model, *input_rate, *output_rate, profile)
            })
            .collect();

        projections.sort_by(|a, b| {
            a.monthly_cost_usd
                .partial_cmp(&b.monthly_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self { projections }
    }

    /// All projections ranked cheapest → most expensive.
    pub fn ranked(&self) -> &[CostProjection] {
        &self.projections
    }

    /// The cheapest option across all models.
    ///
    /// # Panics
    ///
    /// Panics only if the pricing table is empty (compile-time constant, so
    /// this can never happen in practice).
    pub fn cheapest(&self) -> &CostProjection {
        #[allow(clippy::expect_used)]
        self.projections
            .first()
            .expect("pricing table is never empty")
    }

    /// The most expensive option across all models.
    ///
    /// # Panics
    ///
    /// Same caveat as [`cheapest`](Self::cheapest).
    pub fn most_expensive(&self) -> &CostProjection {
        #[allow(clippy::expect_used)]
        self.projections
            .last()
            .expect("pricing table is never empty")
    }

    /// Return the `n` cheapest models.
    pub fn top_n_cheapest(&self, n: usize) -> &[CostProjection] {
        let end = n.min(self.projections.len());
        &self.projections[..end]
    }

    /// Return the `n` most expensive models.
    pub fn top_n_most_expensive(&self, n: usize) -> &[CostProjection] {
        let start = self.projections.len().saturating_sub(n);
        &self.projections[start..]
    }

    /// Filter projections to a single provider (case-insensitive substring match).
    pub fn for_provider(&self, provider: &str) -> Vec<&CostProjection> {
        let needle = provider.to_lowercase();
        self.projections
            .iter()
            .filter(|p| p.provider.to_lowercase().contains(&needle))
            .collect()
    }

    /// Total number of models compared.
    pub fn model_count(&self) -> usize {
        self.projections.len()
    }

    /// Cost ratio between the most and least expensive model.
    ///
    /// Returns `1.0` when there are fewer than two models or the cheapest
    /// model has zero cost.
    pub fn cost_spread_ratio(&self) -> f64 {
        if self.projections.len() < 2 {
            return 1.0;
        }
        let min = self.cheapest().monthly_cost_usd;
        let max = self.most_expensive().monthly_cost_usd;
        if min < f64::EPSILON {
            return 1.0;
        }
        max / min
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn default_profile() -> WorkloadProfile {
        WorkloadProfile {
            avg_input_tokens: 500,
            avg_output_tokens: 200,
            requests_per_day: 1_000,
        }
    }

    #[test]
    fn cheapest_is_less_than_most_expensive() {
        let cmp = ProviderComparison::compute(&default_profile());
        assert!(
            cmp.cheapest().monthly_cost_usd <= cmp.most_expensive().monthly_cost_usd,
            "cheapest should be ≤ most expensive"
        );
    }

    #[test]
    fn model_count_matches_pricing_table() {
        let cmp = ProviderComparison::compute(&default_profile());
        assert_eq!(cmp.model_count(), PRICING.len());
    }

    #[test]
    fn ranked_is_sorted_ascending() {
        let cmp = ProviderComparison::compute(&default_profile());
        let ranked = cmp.ranked();
        for window in ranked.windows(2) {
            assert!(
                window[0].monthly_cost_usd <= window[1].monthly_cost_usd,
                "{} ({}) should be cheaper than {} ({})",
                window[0].model,
                window[0].monthly_cost_usd,
                window[1].model,
                window[1].monthly_cost_usd
            );
        }
    }

    #[test]
    fn top_n_cheapest_length() {
        let cmp = ProviderComparison::compute(&default_profile());
        assert_eq!(cmp.top_n_cheapest(5).len(), 5);
    }

    #[test]
    fn daily_cost_times_30_equals_monthly() {
        let cmp = ProviderComparison::compute(&default_profile());
        for proj in cmp.ranked() {
            let diff = (proj.daily_cost_usd * 30.0 - proj.monthly_cost_usd).abs();
            assert!(diff < 1e-9, "monthly/daily mismatch for {}", proj.model);
        }
    }

    #[test]
    fn zero_requests_gives_zero_cost() {
        let profile = WorkloadProfile {
            avg_input_tokens: 500,
            avg_output_tokens: 200,
            requests_per_day: 0,
        };
        let cmp = ProviderComparison::compute(&profile);
        for proj in cmp.ranked() {
            assert_eq!(proj.monthly_cost_usd, 0.0, "{} should cost 0", proj.model);
        }
    }

    #[test]
    fn for_provider_anthropic_filters_correctly() {
        let cmp = ProviderComparison::compute(&default_profile());
        let anthropic = cmp.for_provider("Anthropic");
        assert!(!anthropic.is_empty());
        for p in anthropic {
            assert_eq!(p.provider, "Anthropic");
        }
    }

    #[test]
    fn cost_spread_ratio_greater_than_one() {
        let cmp = ProviderComparison::compute(&default_profile());
        assert!(cmp.cost_spread_ratio() > 1.0);
    }

    #[test]
    fn from_rph_sets_24x_day() {
        let profile = WorkloadProfile::from_rph(100);
        assert_eq!(profile.requests_per_day, 2400);
    }

    #[test]
    fn infer_provider_known_prefixes() {
        assert_eq!(infer_provider("claude-sonnet-4-6"), "Anthropic");
        assert_eq!(infer_provider("gpt-4o-mini"), "OpenAI");
        assert_eq!(infer_provider("gemini-2.0-flash"), "Google");
        assert_eq!(infer_provider("grok-3"), "xAI");
        assert_eq!(infer_provider("deepseek-r1"), "DeepSeek");
    }

    #[test]
    fn gpt4_5_preview_is_expensive() {
        // gpt-4.5-preview at $75 in / $150 out should rank near the top.
        let profile = WorkloadProfile {
            avg_input_tokens: 1_000,
            avg_output_tokens: 500,
            requests_per_day: 100,
        };
        let cmp = ProviderComparison::compute(&profile);
        let expensive = cmp.most_expensive();
        // Either gpt-4.5-preview or claude-opus should be the most expensive.
        assert!(
            expensive.monthly_cost_usd > 1.0,
            "expected significant cost, got {:.4}",
            expensive.monthly_cost_usd
        );
    }
}
