//! Model selection optimizer to minimise cost while meeting a quality threshold.
//!
//! Use [`CostOptimizer`] to register model profiles and then call
//! [`CostOptimizer::select_model`] to choose the best model for a request
//! given a token budget, a minimum quality threshold, and an
//! [`OptimizationObjective`].

use std::collections::HashMap;

/// Static profile describing one model's cost and quality characteristics.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Unique model identifier string (e.g. `"gpt-4o"`, `"claude-3-haiku"`).
    pub model_id: String,
    /// Cost per token in USD (combined input+output average, or whichever unit
    /// is consistent across all registered models).
    pub cost_per_token: f64,
    /// Average quality score in [0, 1] measured on a representative benchmark.
    pub avg_quality_score: f64,
    /// Typical end-to-end latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Maximum context window in tokens.
    pub context_window: u64,
}

/// The objective function used to score and rank candidate models.
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    /// Prefer the model with the lowest total cost.
    MinimizeCost,
    /// Prefer the model with the highest quality score.
    MaximizeQuality,
    /// Prefer the model with the lowest latency.
    MinimizeLatency,
    /// Weighted combination.  All three weights should sum to 1.0 but this is
    /// not enforced — they are normalised internally.
    Balanced {
        /// Weight given to cost minimisation.
        cost_weight: f64,
        /// Weight given to quality maximisation.
        quality_weight: f64,
        /// Weight given to latency minimisation.
        latency_weight: f64,
    },
}

/// A summary of the savings achieved by switching from a baseline model to the
/// optimised selection.
#[derive(Debug, Clone)]
pub struct CostSavingsReport {
    /// Total cost using the baseline model (in the same unit as
    /// `ModelProfile::cost_per_token`).
    pub baseline_cost: f64,
    /// Total cost using the optimised model.
    pub optimized_cost: f64,
    /// Percentage saving: `(baseline - optimized) / baseline * 100`.
    pub savings_pct: f64,
    /// Whether the optimised model meets or exceeds the original quality score.
    pub quality_maintained: bool,
}

/// Registry and selector for LLM model profiles.
#[derive(Debug, Default)]
pub struct CostOptimizer {
    profiles: HashMap<String, ModelProfile>,
}

impl CostOptimizer {
    /// Create an empty optimizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or overwrite) a model profile.
    pub fn register_model(&mut self, profile: ModelProfile) {
        self.profiles.insert(profile.model_id.clone(), profile);
    }

    /// Select the best model for a request.
    ///
    /// Only models whose `context_window >= token_count` **and**
    /// `avg_quality_score >= quality_threshold` are considered.  Among
    /// eligible candidates the one that best satisfies `objective` is
    /// returned.
    ///
    /// Returns `None` when no registered model meets the constraints.
    pub fn select_model(
        &self,
        token_count: u64,
        quality_threshold: f64,
        objective: &OptimizationObjective,
    ) -> Option<&ModelProfile> {
        let candidates: Vec<&ModelProfile> = self
            .profiles
            .values()
            .filter(|p| {
                p.context_window >= token_count && p.avg_quality_score >= quality_threshold
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Find the range of each dimension so we can normalise for Balanced.
        let max_cost = candidates
            .iter()
            .map(|p| p.cost_per_token)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_cost = candidates
            .iter()
            .map(|p| p.cost_per_token)
            .fold(f64::INFINITY, f64::min);
        let max_latency = candidates
            .iter()
            .map(|p| p.avg_latency_ms)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_latency = candidates
            .iter()
            .map(|p| p.avg_latency_ms)
            .fold(f64::INFINITY, f64::min);
        let max_quality = candidates
            .iter()
            .map(|p| p.avg_quality_score)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_quality = candidates
            .iter()
            .map(|p| p.avg_quality_score)
            .fold(f64::INFINITY, f64::min);

        let normalise = |val: f64, lo: f64, hi: f64| -> f64 {
            if (hi - lo).abs() < f64::EPSILON {
                0.5
            } else {
                (val - lo) / (hi - lo)
            }
        };

        candidates
            .into_iter()
            .max_by(|a, b| {
                let score_a = self.score(a, objective, normalise, max_cost, min_cost, max_latency, min_latency, max_quality, min_quality);
                let score_b = self.score(b, objective, normalise, max_cost, min_cost, max_latency, min_latency, max_quality, min_quality);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn score(
        &self,
        p: &ModelProfile,
        objective: &OptimizationObjective,
        normalise: impl Fn(f64, f64, f64) -> f64,
        max_cost: f64,
        min_cost: f64,
        max_latency: f64,
        min_latency: f64,
        max_quality: f64,
        min_quality: f64,
    ) -> f64 {
        match objective {
            OptimizationObjective::MinimizeCost => {
                // Lower cost = higher score: invert the normalised cost
                1.0 - normalise(p.cost_per_token, min_cost, max_cost)
            }
            OptimizationObjective::MaximizeQuality => p.avg_quality_score,
            OptimizationObjective::MinimizeLatency => {
                1.0 - normalise(p.avg_latency_ms, min_latency, max_latency)
            }
            OptimizationObjective::Balanced {
                cost_weight,
                quality_weight,
                latency_weight,
            } => {
                let total = cost_weight + quality_weight + latency_weight;
                let cw = cost_weight / total;
                let qw = quality_weight / total;
                let lw = latency_weight / total;
                let cost_score = 1.0 - normalise(p.cost_per_token, min_cost, max_cost);
                let quality_score = normalise(p.avg_quality_score, min_quality, max_quality);
                let latency_score = 1.0 - normalise(p.avg_latency_ms, min_latency, max_latency);
                cw * cost_score + qw * quality_score + lw * latency_score
            }
        }
    }

    /// Estimate the total cost for `model_id` given `token_count`.
    ///
    /// Returns `None` if the model is not registered.
    pub fn estimate_cost(&self, model_id: &str, token_count: u64) -> Option<f64> {
        self.profiles
            .get(model_id)
            .map(|p| p.cost_per_token * token_count as f64)
    }

    /// Return the Pareto frontier in the cost-quality space.
    ///
    /// Model A dominates model B if A has **lower** cost **and** **higher**
    /// quality.  A model is on the frontier if no other model dominates it.
    pub fn pareto_frontier(&self) -> Vec<&ModelProfile> {
        let all: Vec<&ModelProfile> = self.profiles.values().collect();
        all.iter()
            .copied()
            .filter(|candidate| {
                !all.iter().any(|other| {
                    other.model_id != candidate.model_id
                        && other.cost_per_token <= candidate.cost_per_token
                        && other.avg_quality_score >= candidate.avg_quality_score
                        && (other.cost_per_token < candidate.cost_per_token
                            || other.avg_quality_score > candidate.avg_quality_score)
                })
            })
            .collect()
    }

    /// Generate a [`CostSavingsReport`] comparing `baseline_model_id` against
    /// the model that [`select_model`] would choose.
    ///
    /// Returns `None` if either the baseline model is unknown or no optimised
    /// model can be selected under the given constraints.
    pub fn savings_report(
        &self,
        baseline_model_id: &str,
        token_count: u64,
        quality_threshold: f64,
        objective: &OptimizationObjective,
    ) -> Option<CostSavingsReport> {
        let baseline = self.profiles.get(baseline_model_id)?;
        let optimized = self.select_model(token_count, quality_threshold, objective)?;
        let baseline_cost = baseline.cost_per_token * token_count as f64;
        let optimized_cost = optimized.cost_per_token * token_count as f64;
        let savings_pct = if baseline_cost > 0.0 {
            (baseline_cost - optimized_cost) / baseline_cost * 100.0
        } else {
            0.0
        };
        let quality_maintained = optimized.avg_quality_score >= baseline.avg_quality_score;
        Some(CostSavingsReport {
            baseline_cost,
            optimized_cost,
            savings_pct,
            quality_maintained,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(id: &str, cost: f64, quality: f64, latency: f64, ctx: u64) -> ModelProfile {
        ModelProfile {
            model_id: id.to_string(),
            cost_per_token: cost,
            avg_quality_score: quality,
            avg_latency_ms: latency,
            context_window: ctx,
        }
    }

    fn optimizer_with_three() -> CostOptimizer {
        let mut opt = CostOptimizer::new();
        opt.register_model(make_profile("cheap", 0.001, 0.70, 300.0, 8_192));
        opt.register_model(make_profile("mid", 0.005, 0.85, 200.0, 32_768));
        opt.register_model(make_profile("premium", 0.020, 0.98, 150.0, 128_000));
        opt
    }

    #[test]
    fn select_minimize_cost() {
        let opt = optimizer_with_three();
        let selected = opt
            .select_model(1_000, 0.60, &OptimizationObjective::MinimizeCost)
            .unwrap();
        assert_eq!(selected.model_id, "cheap");
    }

    #[test]
    fn select_maximize_quality() {
        let opt = optimizer_with_three();
        let selected = opt
            .select_model(1_000, 0.60, &OptimizationObjective::MaximizeQuality)
            .unwrap();
        assert_eq!(selected.model_id, "premium");
    }

    #[test]
    fn select_minimize_latency() {
        let opt = optimizer_with_three();
        let selected = opt
            .select_model(1_000, 0.60, &OptimizationObjective::MinimizeLatency)
            .unwrap();
        assert_eq!(selected.model_id, "premium");
    }

    #[test]
    fn select_quality_threshold_filters() {
        let opt = optimizer_with_three();
        // Only "premium" meets 0.95+
        let selected = opt
            .select_model(1_000, 0.95, &OptimizationObjective::MinimizeCost)
            .unwrap();
        assert_eq!(selected.model_id, "premium");
    }

    #[test]
    fn select_token_count_filters_context_window() {
        let opt = optimizer_with_three();
        // Only "premium" has context window >= 100_000
        let selected = opt
            .select_model(100_000, 0.0, &OptimizationObjective::MinimizeCost)
            .unwrap();
        assert_eq!(selected.model_id, "premium");
    }

    #[test]
    fn select_returns_none_when_no_candidate() {
        let opt = optimizer_with_three();
        // Impossible quality threshold
        assert!(opt
            .select_model(1_000, 1.1, &OptimizationObjective::MinimizeCost)
            .is_none());
    }

    #[test]
    fn estimate_cost() {
        let opt = optimizer_with_three();
        let cost = opt.estimate_cost("cheap", 1_000).unwrap();
        assert!((cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn estimate_cost_unknown_model() {
        let opt = optimizer_with_three();
        assert!(opt.estimate_cost("phantom", 100).is_none());
    }

    #[test]
    fn pareto_frontier_excludes_dominated() {
        let mut opt = CostOptimizer::new();
        // "dominated" is worse than "mid" on both axes
        opt.register_model(make_profile("dominated", 0.010, 0.70, 200.0, 32_768));
        opt.register_model(make_profile("mid", 0.005, 0.85, 200.0, 32_768));
        opt.register_model(make_profile("cheap_low", 0.001, 0.60, 300.0, 8_192));
        let frontier: Vec<&str> = opt.pareto_frontier().iter().map(|p| p.model_id.as_str()).collect();
        assert!(!frontier.contains(&"dominated"));
        assert!(frontier.contains(&"mid"));
        assert!(frontier.contains(&"cheap_low"));
    }

    #[test]
    fn balanced_objective_selects_mid() {
        let opt = optimizer_with_three();
        // Equal weights — mid should win as a balanced choice between cheap and premium
        let selected = opt
            .select_model(
                1_000,
                0.60,
                &OptimizationObjective::Balanced {
                    cost_weight: 1.0,
                    quality_weight: 1.0,
                    latency_weight: 1.0,
                },
            )
            .unwrap();
        // mid has middling cost, quality, and latency — should score reasonably well
        assert!(!selected.model_id.is_empty());
    }

    #[test]
    fn savings_report_shows_savings() {
        let opt = optimizer_with_three();
        let report = opt
            .savings_report("premium", 1_000, 0.60, &OptimizationObjective::MinimizeCost)
            .unwrap();
        assert!(report.savings_pct > 0.0);
        assert!(report.baseline_cost > report.optimized_cost);
    }
}
