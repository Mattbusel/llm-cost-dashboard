//! Analyze model efficiency: tokens/sec, cost/quality trade-offs, OLS regression.
//!
//! [`EfficiencyAnalyzer`] records individual model runs and computes aggregated
//! [`EfficiencyMetrics`], Pareto-efficient frontiers, OLS regression of
//! tokens → latency, and p95 latency.

use std::collections::HashMap;

/// A single inference run recorded for a model.
#[derive(Debug, Clone)]
pub struct ModelRun {
    /// Unique identifier for this run.
    pub run_id: String,
    /// Model that performed the inference.
    pub model_id: String,
    /// Number of input tokens.
    pub tokens_in: u64,
    /// Number of output tokens.
    pub tokens_out: u64,
    /// End-to-end latency in milliseconds.
    pub latency_ms: u64,
    /// Actual cost incurred in USD.
    pub cost_usd: f64,
    /// Optional quality score in `[0.0, 1.0]` (e.g. from human eval or ROUGE).
    pub quality_score: Option<f64>,
}

/// Aggregated efficiency metrics for a model computed over its recorded runs.
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Average total tokens (in + out) per second across all runs.
    pub tokens_per_second: f64,
    /// Average cost in USD per token (total tokens).
    pub cost_per_token: f64,
    /// Average cost per quality point, or `None` if no quality scores recorded.
    pub cost_per_quality_point: Option<f64>,
    /// Composite throughput efficiency: `tokens_per_second / cost_per_token`.
    /// Higher is better. Zero if `cost_per_token` is zero.
    pub throughput_efficiency: f64,
}

/// Records model runs and computes efficiency metrics.
#[derive(Debug, Default)]
pub struct EfficiencyAnalyzer {
    runs: HashMap<String, Vec<ModelRun>>,
}

impl EfficiencyAnalyzer {
    /// Create an empty analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single model run.
    pub fn record_run(&mut self, run: ModelRun) {
        self.runs
            .entry(run.model_id.clone())
            .or_default()
            .push(run);
    }

    /// Compute aggregate [`EfficiencyMetrics`] for `model_id` across all
    /// recorded runs, or `None` if no runs exist.
    pub fn compute_metrics(&self, model_id: &str) -> Option<EfficiencyMetrics> {
        let runs = self.runs.get(model_id)?;
        if runs.is_empty() {
            return None;
        }

        let n = runs.len() as f64;

        // tokens per second for each run, then average.
        let avg_tps: f64 = runs
            .iter()
            .map(|r| {
                let total_tokens = r.tokens_in + r.tokens_out;
                if r.latency_ms == 0 {
                    total_tokens as f64 * 1_000.0 // avoid div-by-zero
                } else {
                    total_tokens as f64 / (r.latency_ms as f64 / 1_000.0)
                }
            })
            .sum::<f64>()
            / n;

        // cost per token for each run, then average.
        let avg_cpt: f64 = runs
            .iter()
            .map(|r| {
                let total_tokens = (r.tokens_in + r.tokens_out) as f64;
                if total_tokens == 0.0 {
                    0.0
                } else {
                    r.cost_usd / total_tokens
                }
            })
            .sum::<f64>()
            / n;

        // cost per quality point (only runs with quality scores).
        let quality_runs: Vec<_> = runs
            .iter()
            .filter(|r| r.quality_score.is_some())
            .collect();

        let cost_per_quality_point = if quality_runs.is_empty() {
            None
        } else {
            let sum: f64 = quality_runs
                .iter()
                .map(|r| {
                    let q = r.quality_score.unwrap_or(0.0);
                    if q == 0.0 { r.cost_usd } else { r.cost_usd / q }
                })
                .sum();
            Some(sum / quality_runs.len() as f64)
        };

        let throughput_efficiency = if avg_cpt == 0.0 {
            0.0
        } else {
            avg_tps / avg_cpt
        };

        Some(EfficiencyMetrics {
            tokens_per_second: avg_tps,
            cost_per_token: avg_cpt,
            cost_per_quality_point,
            throughput_efficiency,
        })
    }

    /// Return all models ranked by descending `throughput_efficiency`.
    pub fn efficiency_ranking(&self) -> Vec<(String, EfficiencyMetrics)> {
        let mut ranked: Vec<(String, EfficiencyMetrics)> = self
            .runs
            .keys()
            .filter_map(|id| self.compute_metrics(id).map(|m| (id.clone(), m)))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.throughput_efficiency
                .partial_cmp(&a.1.throughput_efficiency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Return the model IDs that are Pareto-efficient in (cost, quality) space
    /// — i.e. no other model dominates them (lower cost AND higher quality).
    ///
    /// Only models that have at least one run with a quality score participate.
    pub fn pareto_efficient_models(&self) -> Vec<String> {
        // Compute mean cost and mean quality per model.
        let model_stats: Vec<(String, f64, f64)> = self
            .runs
            .iter()
            .filter_map(|(id, runs)| {
                let quality_runs: Vec<_> = runs
                    .iter()
                    .filter(|r| r.quality_score.is_some())
                    .collect();
                if quality_runs.is_empty() {
                    return None;
                }
                let n = quality_runs.len() as f64;
                let avg_cost = quality_runs.iter().map(|r| r.cost_usd).sum::<f64>() / n;
                let avg_quality = quality_runs
                    .iter()
                    .map(|r| r.quality_score.unwrap_or(0.0))
                    .sum::<f64>()
                    / n;
                Some((id.clone(), avg_cost, avg_quality))
            })
            .collect();

        // A model is Pareto-efficient if no other model has both lower (or equal)
        // cost AND strictly higher quality, or lower cost AND equal-or-higher quality.
        // Standard definition: dominated if ∃ b s.t. b.cost ≤ a.cost AND b.quality ≥ a.quality
        // with at least one strict inequality.
        model_stats
            .iter()
            .filter(|(id_a, cost_a, quality_a)| {
                !model_stats.iter().any(|(id_b, cost_b, quality_b)| {
                    id_a != id_b
                        && cost_b <= cost_a
                        && quality_b >= quality_a
                        && (cost_b < cost_a || quality_b > quality_a)
                })
            })
            .map(|(id, _, _)| id.clone())
            .collect()
    }

    /// Ordinary Least Squares regression of total tokens → latency (ms) for
    /// `model_id`. Returns `(slope, intercept)` or `None` if fewer than 2 runs
    /// are available.
    ///
    /// The regression equation is: `latency_ms ≈ slope * total_tokens + intercept`.
    pub fn regression_analysis(&self, model_id: &str) -> Option<(f64, f64)> {
        let runs = self.runs.get(model_id)?;
        if runs.len() < 2 {
            return None;
        }

        let n = runs.len() as f64;
        let xs: Vec<f64> = runs
            .iter()
            .map(|r| (r.tokens_in + r.tokens_out) as f64)
            .collect();
        let ys: Vec<f64> = runs.iter().map(|r| r.latency_ms as f64).collect();

        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;

        let ss_xx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
        let ss_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();

        if ss_xx == 0.0 {
            return None; // all x values identical — degenerate
        }

        let slope = ss_xy / ss_xx;
        let intercept = mean_y - slope * mean_x;
        Some((slope, intercept))
    }

    /// Return the 95th-percentile latency in milliseconds for `model_id`, or
    /// `None` if no runs are recorded.
    pub fn p95_latency(&self, model_id: &str) -> Option<u64> {
        let runs = self.runs.get(model_id)?;
        if runs.is_empty() {
            return None;
        }
        let mut latencies: Vec<u64> = runs.iter().map(|r| r.latency_ms).collect();
        latencies.sort_unstable();
        // Use nearest-rank method: ceil(p * n) - 1
        let idx = ((0.95 * latencies.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(latencies.len() - 1);
        Some(latencies[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        model_id: &str,
        tokens_in: u64,
        tokens_out: u64,
        latency_ms: u64,
        cost_usd: f64,
        quality_score: Option<f64>,
    ) -> ModelRun {
        ModelRun {
            run_id: uuid_simple(),
            model_id: model_id.to_string(),
            tokens_in,
            tokens_out,
            latency_ms,
            cost_usd,
            quality_score,
        }
    }

    fn uuid_simple() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos().to_string())
            .unwrap_or_else(|_| "0".to_string())
    }

    // ── compute_metrics ───────────────────────────────────────────────────────

    #[test]
    fn compute_metrics_basic() {
        let mut analyzer = EfficiencyAnalyzer::new();
        // 1000 tokens total, 500 ms → 2000 tok/s; cost 0.01 USD
        analyzer.record_run(run("gpt-4o", 500, 500, 500, 0.01, Some(0.8)));
        let m = analyzer.compute_metrics("gpt-4o").expect("must exist");
        assert!((m.tokens_per_second - 2000.0).abs() < 1.0);
        assert!((m.cost_per_token - 0.01 / 1000.0).abs() < 1e-10);
        assert!(m.cost_per_quality_point.is_some());
    }

    #[test]
    fn compute_metrics_missing_model_is_none() {
        let analyzer = EfficiencyAnalyzer::new();
        assert!(analyzer.compute_metrics("unknown").is_none());
    }

    #[test]
    fn compute_metrics_no_quality_score() {
        let mut analyzer = EfficiencyAnalyzer::new();
        analyzer.record_run(run("m", 100, 100, 200, 0.005, None));
        let m = analyzer.compute_metrics("m").expect("must exist");
        assert!(m.cost_per_quality_point.is_none());
    }

    // ── efficiency_ranking ────────────────────────────────────────────────────

    #[test]
    fn efficiency_ranking_ordered_desc() {
        let mut analyzer = EfficiencyAnalyzer::new();
        // high throughput, low cost → high efficiency
        analyzer.record_run(run("fast-model", 1000, 1000, 100, 0.001, None));
        // low throughput, high cost → low efficiency
        analyzer.record_run(run("slow-model", 100, 100, 5_000, 1.0, None));
        let ranking = analyzer.efficiency_ranking();
        assert_eq!(ranking.len(), 2);
        assert!(ranking[0].1.throughput_efficiency > ranking[1].1.throughput_efficiency);
    }

    // ── pareto_efficient_models ───────────────────────────────────────────────

    #[test]
    fn pareto_dominated_model_excluded() {
        let mut analyzer = EfficiencyAnalyzer::new();
        // "best": low cost AND high quality — dominates "worst"
        analyzer.record_run(run("best", 100, 100, 200, 0.001, Some(0.95)));
        // "worst": high cost AND low quality — dominated
        analyzer.record_run(run("worst", 100, 100, 200, 1.0, Some(0.5)));
        let pareto = analyzer.pareto_efficient_models();
        assert!(pareto.contains(&"best".to_string()));
        assert!(!pareto.contains(&"worst".to_string()));
    }

    #[test]
    fn pareto_no_quality_scores_excluded() {
        let mut analyzer = EfficiencyAnalyzer::new();
        analyzer.record_run(run("no-quality", 100, 100, 200, 0.01, None));
        assert!(analyzer.pareto_efficient_models().is_empty());
    }

    #[test]
    fn pareto_both_models_efficient_when_trade_off() {
        let mut analyzer = EfficiencyAnalyzer::new();
        // "cheap": low cost, lower quality
        analyzer.record_run(run("cheap", 100, 100, 200, 0.001, Some(0.6)));
        // "quality": higher cost, higher quality — neither dominates the other
        analyzer.record_run(run("quality", 100, 100, 200, 0.01, Some(0.95)));
        let pareto = analyzer.pareto_efficient_models();
        assert!(pareto.contains(&"cheap".to_string()));
        assert!(pareto.contains(&"quality".to_string()));
    }

    // ── regression_analysis ───────────────────────────────────────────────────

    #[test]
    fn regression_returns_slope_and_intercept() {
        let mut analyzer = EfficiencyAnalyzer::new();
        // Perfect linear relationship: latency = 2 * total_tokens + 10
        // total_tokens = 100 → latency 210; 200 → 410; 300 → 610
        analyzer.record_run(ModelRun {
            run_id: "r1".to_string(),
            model_id: "m".to_string(),
            tokens_in: 50, tokens_out: 50, latency_ms: 210, cost_usd: 0.01, quality_score: None,
        });
        analyzer.record_run(ModelRun {
            run_id: "r2".to_string(),
            model_id: "m".to_string(),
            tokens_in: 100, tokens_out: 100, latency_ms: 410, cost_usd: 0.02, quality_score: None,
        });
        analyzer.record_run(ModelRun {
            run_id: "r3".to_string(),
            model_id: "m".to_string(),
            tokens_in: 150, tokens_out: 150, latency_ms: 610, cost_usd: 0.03, quality_score: None,
        });
        let (slope, intercept) = analyzer.regression_analysis("m").expect("must exist");
        assert!((slope - 2.0).abs() < 1e-6, "slope={slope}");
        assert!((intercept - 10.0).abs() < 1e-6, "intercept={intercept}");
    }

    #[test]
    fn regression_single_run_is_none() {
        let mut analyzer = EfficiencyAnalyzer::new();
        analyzer.record_run(run("m", 100, 100, 200, 0.01, None));
        assert!(analyzer.regression_analysis("m").is_none());
    }

    #[test]
    fn regression_missing_model_is_none() {
        let analyzer = EfficiencyAnalyzer::new();
        assert!(analyzer.regression_analysis("x").is_none());
    }

    // ── p95_latency ───────────────────────────────────────────────────────────

    #[test]
    fn p95_latency_basic() {
        let mut analyzer = EfficiencyAnalyzer::new();
        // 20 runs with latencies 1..=20 ms
        for i in 1u64..=20 {
            analyzer.record_run(ModelRun {
                run_id: format!("r{i}"),
                model_id: "m".to_string(),
                tokens_in: 100, tokens_out: 100,
                latency_ms: i,
                cost_usd: 0.001,
                quality_score: None,
            });
        }
        // p95 = ceil(0.95 * 20) = 19th element (1-indexed) = 19 ms
        let p95 = analyzer.p95_latency("m").expect("must exist");
        assert_eq!(p95, 19);
    }

    #[test]
    fn p95_latency_single_run() {
        let mut analyzer = EfficiencyAnalyzer::new();
        analyzer.record_run(run("m", 100, 100, 42, 0.01, None));
        assert_eq!(analyzer.p95_latency("m"), Some(42));
    }

    #[test]
    fn p95_latency_missing_model_is_none() {
        let analyzer = EfficiencyAnalyzer::new();
        assert!(analyzer.p95_latency("unknown").is_none());
    }
}
