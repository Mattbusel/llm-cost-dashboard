//! Model comparison benchmarking.
//!
//! [`BenchmarkSuite`] collects [`BenchmarkResult`] records from external
//! runners, computes per-model statistics (latency percentiles, cost,
//! quality, throughput), and produces a Markdown comparison table.

use std::collections::HashMap;

// ── BenchmarkConfig ───────────────────────────────────────────────────────────

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Model identifiers to benchmark.
    pub model_ids: Vec<String>,
    /// Prompt template strings used as inputs.
    pub prompt_templates: Vec<String>,
    /// Number of measurement iterations per (model, prompt) pair.
    pub iterations: u32,
    /// Number of warm-up rounds whose results are discarded.
    pub warmup_rounds: u32,
}

// ── BenchmarkResult ───────────────────────────────────────────────────────────

/// A single benchmark observation.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Model that produced this result.
    pub model_id: String,
    /// Zero-based index into [`BenchmarkConfig::prompt_templates`].
    pub prompt_idx: usize,
    /// End-to-end latency in milliseconds.
    pub latency_ms: f64,
    /// Input token count.
    pub tokens_in: u32,
    /// Output token count.
    pub tokens_out: u32,
    /// USD cost for this call.
    pub cost_usd: f64,
    /// Quality score in `[0.0, 1.0]`.
    pub quality_score: f64,
}

// ── ModelStats ────────────────────────────────────────────────────────────────

/// Aggregated statistics for a single model across all benchmark results.
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Model identifier.
    pub model_id: String,
    /// Median (p50) latency in milliseconds.
    pub p50_latency: f64,
    /// 95th-percentile latency in milliseconds.
    pub p95_latency: f64,
    /// 99th-percentile latency in milliseconds.
    pub p99_latency: f64,
    /// Mean USD cost per call.
    pub avg_cost: f64,
    /// Mean quality score.
    pub avg_quality: f64,
    /// Throughput in tokens per second (output tokens / total latency).
    pub throughput_tps: f64,
}

impl ModelStats {
    /// Compute statistics for `model_id` from a slice of results.
    ///
    /// Returns a zero-value [`ModelStats`] if `results` is empty.
    pub fn from_results(model_id: &str, results: &[BenchmarkResult]) -> Self {
        if results.is_empty() {
            return Self {
                model_id: model_id.to_owned(),
                p50_latency: 0.0,
                p95_latency: 0.0,
                p99_latency: 0.0,
                avg_cost: 0.0,
                avg_quality: 0.0,
                throughput_tps: 0.0,
            };
        }

        let n = results.len();

        // Latency percentiles
        let mut latencies: Vec<f64> = results.iter().map(|r| r.latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile = |p: f64| -> f64 {
            let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
            latencies[idx.min(n - 1)]
        };

        let p50 = percentile(50.0);
        let p95 = percentile(95.0);
        let p99 = percentile(99.0);

        let avg_cost = results.iter().map(|r| r.cost_usd).sum::<f64>() / n as f64;
        let avg_quality = results.iter().map(|r| r.quality_score).sum::<f64>() / n as f64;

        // Throughput: total output tokens / total latency in seconds
        let total_tokens_out: f64 = results.iter().map(|r| r.tokens_out as f64).sum();
        let total_latency_s: f64 = results.iter().map(|r| r.latency_ms / 1_000.0).sum();
        let throughput_tps = if total_latency_s > 0.0 {
            total_tokens_out / total_latency_s
        } else {
            0.0
        };

        Self {
            model_id: model_id.to_owned(),
            p50_latency: p50,
            p95_latency: p95,
            p99_latency: p99,
            avg_cost,
            avg_quality,
            throughput_tps,
        }
    }
}

// ── BenchmarkSuite ────────────────────────────────────────────────────────────

/// Container for benchmark results and cached per-model statistics.
pub struct BenchmarkSuite {
    /// Benchmark configuration.
    pub config: BenchmarkConfig,
    /// Raw result records.
    pub results: Vec<BenchmarkResult>,
    /// Cached per-model statistics (populated by [`BenchmarkSuite::compute_stats`]).
    pub stats_cache: HashMap<String, ModelStats>,
}

impl BenchmarkSuite {
    /// Create an empty suite with the given configuration.
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            stats_cache: HashMap::new(),
        }
    }

    /// Append a single benchmark result and invalidate the stats cache.
    pub fn add_result(&mut self, r: BenchmarkResult) {
        self.stats_cache.clear();
        self.results.push(r);
    }

    /// (Re-)compute per-model statistics and populate `stats_cache`.
    pub fn compute_stats(&mut self) {
        self.stats_cache.clear();
        let mut by_model: HashMap<String, Vec<BenchmarkResult>> = HashMap::new();
        for r in &self.results {
            by_model.entry(r.model_id.clone()).or_default().push(r.clone());
        }
        for (model_id, model_results) in &by_model {
            let stats = ModelStats::from_results(model_id, model_results);
            self.stats_cache.insert(model_id.clone(), stats);
        }
    }

    /// Return all model statistics sorted by p95 latency (ascending).
    ///
    /// Calls [`BenchmarkSuite::compute_stats`] if the cache is stale.
    pub fn compare_models(&mut self) -> Vec<ModelStats> {
        if self.stats_cache.is_empty() && !self.results.is_empty() {
            self.compute_stats();
        }
        let mut stats: Vec<ModelStats> = self.stats_cache.values().cloned().collect();
        stats.sort_by(|a, b| {
            a.p95_latency
                .partial_cmp(&b.p95_latency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        stats
    }

    /// Return the most cost-efficient model whose average quality exceeds 0.8.
    ///
    /// Returns `None` if no qualifying model exists.
    pub fn winner_by_cost_efficiency(&self) -> Option<&ModelStats> {
        self.stats_cache
            .values()
            .filter(|s| s.avg_quality > 0.8)
            .min_by(|a, b| {
                a.avg_cost
                    .partial_cmp(&b.avg_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Generate a Markdown comparison table of all models.
    pub fn report(&self) -> String {
        let mut stats: Vec<&ModelStats> = self.stats_cache.values().collect();
        stats.sort_by(|a, b| {
            a.p95_latency
                .partial_cmp(&b.p95_latency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut out = String::new();
        out.push_str("| Model | p50 ms | p95 ms | p99 ms | Avg Cost USD | Avg Quality | TPS |\n");
        out.push_str("|-------|--------|--------|--------|-------------|-------------|-----|\n");
        for s in &stats {
            out.push_str(&format!(
                "| {} | {:.1} | {:.1} | {:.1} | {:.6} | {:.3} | {:.1} |\n",
                s.model_id,
                s.p50_latency,
                s.p95_latency,
                s.p99_latency,
                s.avg_cost,
                s.avg_quality,
                s.throughput_tps
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(model: &str, latency: f64, cost: f64, quality: f64) -> BenchmarkResult {
        BenchmarkResult {
            model_id: model.to_owned(),
            prompt_idx: 0,
            latency_ms: latency,
            tokens_in: 100,
            tokens_out: 50,
            cost_usd: cost,
            quality_score: quality,
        }
    }

    #[test]
    fn model_stats_percentiles() {
        let results: Vec<BenchmarkResult> =
            (1..=10).map(|i| make_result("m", i as f64 * 10.0, 0.01, 0.9)).collect();
        let stats = ModelStats::from_results("m", &results);
        assert_eq!(stats.p50_latency, 50.0);
        assert_eq!(stats.p95_latency, 100.0);
    }

    #[test]
    fn suite_compare_and_winner() {
        let cfg = BenchmarkConfig {
            model_ids: vec!["a".into(), "b".into()],
            prompt_templates: vec!["hello".into()],
            iterations: 5,
            warmup_rounds: 1,
        };
        let mut suite = BenchmarkSuite::new(cfg);
        // Model "a": higher quality, higher cost
        for _ in 0..5 {
            suite.add_result(make_result("a", 100.0, 0.05, 0.95));
        }
        // Model "b": lower quality (below 0.8), lower cost
        for _ in 0..5 {
            suite.add_result(make_result("b", 50.0, 0.01, 0.7));
        }
        suite.compute_stats();

        // compare_models returns sorted by p95 latency; "b" is faster
        let ranked = suite.compare_models();
        assert_eq!(ranked[0].model_id, "b");

        // winner_by_cost_efficiency: only "a" has quality > 0.8
        let winner = suite.winner_by_cost_efficiency();
        assert!(winner.is_some());
        assert_eq!(winner.unwrap().model_id, "a");
    }

    #[test]
    fn report_contains_header() {
        let cfg = BenchmarkConfig {
            model_ids: vec!["x".into()],
            prompt_templates: vec![],
            iterations: 1,
            warmup_rounds: 0,
        };
        let mut suite = BenchmarkSuite::new(cfg);
        suite.add_result(make_result("x", 200.0, 0.02, 0.88));
        suite.compute_stats();
        let report = suite.report();
        assert!(report.contains("| Model |"));
        assert!(report.contains("x"));
    }
}
