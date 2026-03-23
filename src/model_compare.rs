//! # Model Comparison Dashboard
//!
//! Computes per-model cost, token, and latency metrics from a slice of
//! [`TaggedRequest`]s and provides ranking, ASCII table rendering, and
//! savings reports relative to a baseline model.
//!
//! ## Quick start
//!
//! ```rust
//! use llm_cost_dashboard::model_compare::{ModelComparison, RankMetric};
//! use llm_cost_dashboard::tagging::{TaggedRequest, CostTag};
//! use chrono::Utc;
//!
//! let requests: Vec<TaggedRequest> = vec![]; // populate from your ledger
//! let cmp = ModelComparison::compute(&requests);
//! let ranked = cmp.rank_by(RankMetric::Cost);
//! println!("{}", cmp.render_table());
//! ```

use crate::tagging::TaggedRequest;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ModelMetrics
// ---------------------------------------------------------------------------

/// Aggregated per-model metrics derived from a set of [`TaggedRequest`]s.
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    /// Model identifier (e.g. `"gpt-4o-mini"`).
    pub model_id: String,
    /// Sum of `cost_usd` across all requests for this model.
    pub total_cost_usd: f64,
    /// Sum of `tokens_in + tokens_out` across all requests.
    pub total_tokens: u64,
    /// `total_cost_usd / (total_tokens / 1000)`, or `0.0` if no tokens.
    pub avg_cost_per_1k_tokens: f64,
    /// Number of requests for this model.
    pub request_count: u64,
    /// 50th-percentile latency in milliseconds (derived from `latency_ms` tag
    /// if present; otherwise `0`).
    pub p50_latency_ms: f64,
    /// 99th-percentile latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Fraction of requests with `error=true` tag (range `[0.0, 1.0]`).
    pub error_rate: f64,
}

// ---------------------------------------------------------------------------
// RankMetric
// ---------------------------------------------------------------------------

/// Dimension used to rank models in [`ModelComparison::rank_by`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankMetric {
    /// Rank by total cost (cheapest first).
    Cost,
    /// Rank by cost-per-1k-tokens (most efficient first).
    TokenEfficiency,
    /// Rank by p50 latency (fastest first).
    Speed,
    /// Rank by error rate (most reliable first).
    Reliability,
}

// ---------------------------------------------------------------------------
// SavingsReport
// ---------------------------------------------------------------------------

/// How much each model costs relative to a baseline.
#[derive(Debug, Clone)]
pub struct SavingsReport {
    /// Baseline model identifier.
    pub baseline: String,
    /// Per-model comparison: `(model_id, pct_change, usd_change)`.
    ///
    /// `pct_change` is positive when the model is *more expensive* than the
    /// baseline.  `usd_change` is the absolute USD difference (positive =
    /// more expensive).
    pub comparisons: Vec<(String, f64, f64)>,
}

// ---------------------------------------------------------------------------
// ModelComparison
// ---------------------------------------------------------------------------

/// Computes and stores per-model metrics from a set of requests.
#[derive(Debug, Clone)]
pub struct ModelComparison {
    metrics: Vec<ModelMetrics>,
}

impl ModelComparison {
    /// Compute metrics for every unique `model_id` in `requests`.
    ///
    /// Models are returned in alphabetical order by `model_id`.
    ///
    /// Latency is read from the `"latency_ms"` tag (parsed as `u64`).  Error
    /// status is read from the `"error"` tag (`"true"` counts as an error).
    pub fn compute(requests: &[TaggedRequest]) -> Self {
        // Accumulate raw data per model.
        struct Acc {
            total_cost: f64,
            total_tokens: u64,
            count: u64,
            latencies: Vec<u64>,
            errors: u64,
        }

        let mut map: HashMap<String, Acc> = HashMap::new();

        for req in requests {
            let acc = map.entry(req.model_id.clone()).or_insert(Acc {
                total_cost: 0.0,
                total_tokens: 0,
                count: 0,
                latencies: Vec::new(),
                errors: 0,
            });

            acc.total_cost += req.cost_usd;
            acc.total_tokens += req.tokens_in as u64 + req.tokens_out as u64;
            acc.count += 1;

            // Latency from tag.
            if let Some(lat_str) = req.tag_value("latency_ms") {
                if let Ok(lat) = lat_str.parse::<u64>() {
                    acc.latencies.push(lat);
                }
            }

            // Error flag from tag.
            if req.tag_value("error") == Some("true") {
                acc.errors += 1;
            }
        }

        let mut metrics: Vec<ModelMetrics> = map
            .into_iter()
            .map(|(model_id, acc)| {
                let avg_cost_per_1k = if acc.total_tokens == 0 {
                    0.0
                } else {
                    acc.total_cost / (acc.total_tokens as f64 / 1000.0)
                };

                let (p50, p99) = percentiles(&acc.latencies);

                let error_rate = if acc.count == 0 {
                    0.0
                } else {
                    acc.errors as f64 / acc.count as f64
                };

                ModelMetrics {
                    model_id,
                    total_cost_usd: acc.total_cost,
                    total_tokens: acc.total_tokens,
                    avg_cost_per_1k_tokens: avg_cost_per_1k,
                    request_count: acc.count,
                    p50_latency_ms: p50,
                    p99_latency_ms: p99,
                    error_rate,
                }
            })
            .collect();

        metrics.sort_by(|a, b| a.model_id.cmp(&b.model_id));

        Self { metrics }
    }

    /// Return a reference to the computed metrics slice.
    pub fn metrics(&self) -> &[ModelMetrics] {
        &self.metrics
    }

    /// Return metrics ranked by the given [`RankMetric`] (best first).
    pub fn rank_by(&self, metric: RankMetric) -> Vec<ModelMetrics> {
        let mut ranked = self.metrics.clone();
        match metric {
            RankMetric::Cost => {
                ranked.sort_by(|a, b| a.total_cost_usd.partial_cmp(&b.total_cost_usd).unwrap_or(std::cmp::Ordering::Equal));
            }
            RankMetric::TokenEfficiency => {
                ranked.sort_by(|a, b| a.avg_cost_per_1k_tokens.partial_cmp(&b.avg_cost_per_1k_tokens).unwrap_or(std::cmp::Ordering::Equal));
            }
            RankMetric::Speed => {
                ranked.sort_by(|a, b| a.p50_latency_ms.partial_cmp(&b.p50_latency_ms).unwrap_or(std::cmp::Ordering::Equal));
            }
            RankMetric::Reliability => {
                ranked.sort_by(|a, b| a.error_rate.partial_cmp(&b.error_rate).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        ranked
    }

    /// Render an ASCII table with all metrics.
    ///
    /// Columns: Model | Requests | Total Cost | Cost/1k | Tokens | P50 ms | P99 ms | Error%
    ///
    /// The best value in each numeric column is marked with `*`.
    pub fn render_table(&self) -> String {
        if self.metrics.is_empty() {
            return "No models to display.\n".to_string();
        }

        // Find best values for each column.
        let best_cost = self
            .metrics
            .iter()
            .map(|m| m.total_cost_usd)
            .fold(f64::INFINITY, f64::min);
        let best_eff = self
            .metrics
            .iter()
            .map(|m| m.avg_cost_per_1k_tokens)
            .fold(f64::INFINITY, f64::min);
        let best_p50 = self
            .metrics
            .iter()
            .map(|m| m.p50_latency_ms)
            .fold(f64::INFINITY, f64::min);
        let best_err = self
            .metrics
            .iter()
            .map(|m| m.error_rate)
            .fold(f64::INFINITY, f64::min);

        let header = format!(
            "{:<30} {:>8} {:>12} {:>10} {:>12} {:>8} {:>8} {:>8}",
            "Model", "Reqs", "Total$", "$/1k", "Tokens", "P50ms", "P99ms", "Err%"
        );
        let sep = "-".repeat(header.len());

        let mut rows = vec![header, sep];

        for m in &self.metrics {
            let cost_mark = if (m.total_cost_usd - best_cost).abs() < 1e-12 { "*" } else { " " };
            let eff_mark = if (m.avg_cost_per_1k_tokens - best_eff).abs() < 1e-12 { "*" } else { " " };
            let p50_mark = if (m.p50_latency_ms - best_p50).abs() < 1e-9 { "*" } else { " " };
            let err_mark = if (m.error_rate - best_err).abs() < 1e-12 { "*" } else { " " };

            rows.push(format!(
                "{:<30} {:>8} {:>11.4}{} {:>9.4}{} {:>12} {:>7.1}{} {:>8.1} {:>7.2}{}",
                m.model_id,
                m.request_count,
                m.total_cost_usd,
                cost_mark,
                m.avg_cost_per_1k_tokens,
                eff_mark,
                m.total_tokens,
                m.p50_latency_ms,
                p50_mark,
                m.p99_latency_ms,
                m.error_rate * 100.0,
                err_mark,
            ));
        }

        rows.join("\n") + "\n"
    }

    /// Compute how much each model costs relative to `baseline_model`.
    ///
    /// Models not in the dataset are omitted.  If `baseline_model` is not
    /// found, the `comparisons` list will be empty and the baseline total
    /// cost will be `0.0`.
    pub fn savings_report(&self, baseline_model: &str) -> SavingsReport {
        let baseline_cost = self
            .metrics
            .iter()
            .find(|m| m.model_id == baseline_model)
            .map_or(0.0, |m| m.total_cost_usd);

        let comparisons = self
            .metrics
            .iter()
            .filter(|m| m.model_id != baseline_model)
            .map(|m| {
                let usd_change = m.total_cost_usd - baseline_cost;
                let pct_change = if baseline_cost == 0.0 {
                    0.0
                } else {
                    (usd_change / baseline_cost) * 100.0
                };
                (m.model_id.clone(), pct_change, usd_change)
            })
            .collect();

        SavingsReport {
            baseline: baseline_model.to_string(),
            comparisons,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute (p50, p99) latencies from a (possibly empty) slice of samples.
fn percentiles(samples: &[u64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let p50 = percentile_val(&sorted, 50);
    let p99 = percentile_val(&sorted, 99);
    (p50 as f64, p99 as f64)
}

fn percentile_val(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (pct * sorted.len()).saturating_sub(1) / 100;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Tests (15+)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tagging::{CostTag, TaggedRequest};
    use chrono::Utc;

    fn make_req(model: &str, cost: f64, tokens_in: u32, tokens_out: u32) -> TaggedRequest {
        TaggedRequest {
            request_id: 0,
            model_id: model.to_string(),
            cost_usd: cost,
            tokens_in,
            tokens_out,
            tags: vec![],
            timestamp: Utc::now(),
        }
    }

    fn make_req_with_tags(
        model: &str,
        cost: f64,
        tokens_in: u32,
        tokens_out: u32,
        tags: Vec<(&str, &str)>,
    ) -> TaggedRequest {
        let tags = tags
            .into_iter()
            .map(|(k, v)| CostTag {
                key: k.to_string(),
                value: v.to_string(),
            })
            .collect();
        TaggedRequest {
            request_id: 0,
            model_id: model.to_string(),
            cost_usd: cost,
            tokens_in,
            tokens_out,
            tags,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn compute_empty_returns_no_metrics() {
        let cmp = ModelComparison::compute(&[]);
        assert!(cmp.metrics().is_empty());
    }

    #[test]
    fn compute_single_model() {
        let reqs = vec![make_req("gpt-4", 0.10, 100, 50)];
        let cmp = ModelComparison::compute(&reqs);
        assert_eq!(cmp.metrics().len(), 1);
        let m = &cmp.metrics()[0];
        assert_eq!(m.model_id, "gpt-4");
        assert!((m.total_cost_usd - 0.10).abs() < 1e-10);
        assert_eq!(m.total_tokens, 150);
        assert_eq!(m.request_count, 1);
    }

    #[test]
    fn compute_two_models() {
        let reqs = vec![make_req("a", 1.0, 1000, 500), make_req("b", 0.5, 500, 250)];
        let cmp = ModelComparison::compute(&reqs);
        assert_eq!(cmp.metrics().len(), 2);
    }

    #[test]
    fn compute_aggregates_multiple_requests_per_model() {
        let reqs = vec![
            make_req("gpt-4", 0.10, 100, 50),
            make_req("gpt-4", 0.20, 200, 100),
        ];
        let cmp = ModelComparison::compute(&reqs);
        let m = &cmp.metrics()[0];
        assert!((m.total_cost_usd - 0.30).abs() < 1e-10);
        assert_eq!(m.request_count, 2);
        assert_eq!(m.total_tokens, 450);
    }

    #[test]
    fn compute_cost_per_1k_tokens() {
        // 1000 tokens, $1.00 → $1.00 / 1k
        let reqs = vec![make_req("m", 1.0, 500, 500)];
        let cmp = ModelComparison::compute(&reqs);
        let m = &cmp.metrics()[0];
        assert!((m.avg_cost_per_1k_tokens - 1.0).abs() < 1e-10);
    }

    #[test]
    fn compute_reads_latency_tag() {
        let reqs = vec![
            make_req_with_tags("m", 0.1, 100, 50, vec![("latency_ms", "200")]),
            make_req_with_tags("m", 0.1, 100, 50, vec![("latency_ms", "400")]),
        ];
        let cmp = ModelComparison::compute(&reqs);
        let m = &cmp.metrics()[0];
        assert!(m.p50_latency_ms > 0.0);
    }

    #[test]
    fn compute_reads_error_tag() {
        let reqs = vec![
            make_req_with_tags("m", 0.1, 100, 50, vec![("error", "true")]),
            make_req("m", 0.1, 100, 50),
        ];
        let cmp = ModelComparison::compute(&reqs);
        let m = &cmp.metrics()[0];
        assert!((m.error_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn metrics_sorted_alphabetically() {
        let reqs = vec![make_req("z-model", 1.0, 100, 50), make_req("a-model", 0.5, 100, 50)];
        let cmp = ModelComparison::compute(&reqs);
        assert_eq!(cmp.metrics()[0].model_id, "a-model");
        assert_eq!(cmp.metrics()[1].model_id, "z-model");
    }

    #[test]
    fn rank_by_cost_cheapest_first() {
        let reqs = vec![make_req("expensive", 10.0, 100, 50), make_req("cheap", 1.0, 100, 50)];
        let cmp = ModelComparison::compute(&reqs);
        let ranked = cmp.rank_by(RankMetric::Cost);
        assert_eq!(ranked[0].model_id, "cheap");
    }

    #[test]
    fn rank_by_reliability_lowest_error_first() {
        let reqs = vec![
            make_req_with_tags("bad", 1.0, 100, 50, vec![("error", "true")]),
            make_req("good", 1.0, 100, 50),
        ];
        let cmp = ModelComparison::compute(&reqs);
        let ranked = cmp.rank_by(RankMetric::Reliability);
        assert_eq!(ranked[0].model_id, "good");
    }

    #[test]
    fn rank_by_token_efficiency() {
        let reqs = vec![
            make_req("pricey", 2.0, 500, 500),  // $2/1k
            make_req("cheap", 0.5, 500, 500),   // $0.5/1k
        ];
        let cmp = ModelComparison::compute(&reqs);
        let ranked = cmp.rank_by(RankMetric::TokenEfficiency);
        assert_eq!(ranked[0].model_id, "cheap");
    }

    #[test]
    fn savings_report_baseline_not_found() {
        let reqs = vec![make_req("a", 1.0, 100, 50)];
        let cmp = ModelComparison::compute(&reqs);
        let report = cmp.savings_report("nonexistent");
        assert_eq!(report.baseline, "nonexistent");
        assert_eq!(report.comparisons.len(), 1);
    }

    #[test]
    fn savings_report_cheaper_model() {
        let reqs = vec![make_req("baseline", 10.0, 1000, 500), make_req("cheaper", 5.0, 1000, 500)];
        let cmp = ModelComparison::compute(&reqs);
        let report = cmp.savings_report("baseline");
        assert_eq!(report.comparisons.len(), 1);
        let (model, pct, usd) = &report.comparisons[0];
        assert_eq!(model, "cheaper");
        assert!(pct < &0.0, "cheaper model should have negative pct_change");
        assert!(usd < &0.0, "cheaper model should have negative usd_change");
    }

    #[test]
    fn savings_report_more_expensive_model() {
        let reqs = vec![make_req("baseline", 5.0, 500, 250), make_req("expensive", 10.0, 500, 250)];
        let cmp = ModelComparison::compute(&reqs);
        let report = cmp.savings_report("baseline");
        let (_, pct, usd) = &report.comparisons[0];
        assert!(pct > &0.0);
        assert!(usd > &0.0);
    }

    #[test]
    fn render_table_non_empty() {
        let reqs = vec![make_req("gpt-4", 1.0, 1000, 500)];
        let cmp = ModelComparison::compute(&reqs);
        let table = cmp.render_table();
        assert!(table.contains("gpt-4"));
        assert!(table.contains("Model"));
    }

    #[test]
    fn render_table_empty() {
        let cmp = ModelComparison::compute(&[]);
        let table = cmp.render_table();
        assert!(table.contains("No models"));
    }

    #[test]
    fn zero_tokens_does_not_panic() {
        let reqs = vec![make_req("m", 0.0, 0, 0)];
        let cmp = ModelComparison::compute(&reqs);
        assert!((cmp.metrics()[0].avg_cost_per_1k_tokens - 0.0).abs() < f64::EPSILON);
    }
}
