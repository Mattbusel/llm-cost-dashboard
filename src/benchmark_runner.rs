//! Latency and throughput benchmarking with statistics.

use std::collections::HashMap;
use std::time::Instant;

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Human-readable name for the benchmark.
    pub name: String,
    /// Number of warmup iterations (results discarded).
    pub warmup_iterations: u32,
    /// Number of measurement iterations.
    pub measurement_iterations: u32,
    /// Per-iteration timeout in milliseconds.
    pub timeout_ms: u64,
}

/// A single latency observation.
#[derive(Debug, Clone)]
pub struct LatencySample {
    /// Zero-based iteration index.
    pub iteration: u32,
    /// Elapsed time in microseconds.
    pub duration_us: u64,
    /// Whether the iteration succeeded.
    pub success: bool,
    /// Error message if the iteration failed.
    pub error: Option<String>,
}

/// Aggregated statistics for a benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    /// Benchmark name.
    pub name: String,
    /// Minimum latency in microseconds.
    pub min_us: u64,
    /// Maximum latency in microseconds.
    pub max_us: u64,
    /// Mean latency in microseconds.
    pub mean_us: f64,
    /// Median latency in microseconds.
    pub median_us: f64,
    /// 95th-percentile latency in microseconds.
    pub p95_us: f64,
    /// 99th-percentile latency in microseconds.
    pub p99_us: f64,
    /// Standard deviation of latency in microseconds.
    pub std_dev_us: f64,
    /// Total number of samples (including failures).
    pub samples: usize,
    /// Fraction of iterations that succeeded.
    pub success_rate: f64,
    /// Estimated throughput in requests per second.
    pub throughput_rps: f64,
}

/// Comparison of two benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// Name of the faster benchmark.
    pub faster: String,
    /// Ratio of the slower mean to the faster mean.
    pub speedup_factor: f64,
    /// Difference between mean latencies (a.mean_us - b.mean_us).
    pub mean_diff_us: f64,
    /// True when the p95 latency ranges do not overlap.
    pub is_significant: bool,
}

/// Runs and stores benchmark samples.
pub struct BenchmarkRunner {
    /// Stored samples keyed by benchmark name.
    pub results: HashMap<String, Vec<LatencySample>>,
}

impl BenchmarkRunner {
    /// Create a new `BenchmarkRunner`.
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Add a single sample for the named benchmark.
    pub fn add_sample(&mut self, name: &str, sample: LatencySample) {
        self.results.entry(name.to_string()).or_default().push(sample);
    }

    /// Linear-interpolation percentile over a sorted slice.
    pub fn percentile(sorted_values: &[u64], pct: f64) -> u64 {
        if sorted_values.is_empty() {
            return 0;
        }
        if sorted_values.len() == 1 {
            return sorted_values[0];
        }
        let index = pct / 100.0 * (sorted_values.len() - 1) as f64;
        let lo = index.floor() as usize;
        let hi = index.ceil() as usize;
        if lo == hi {
            return sorted_values[lo];
        }
        let frac = index - lo as f64;
        let lo_val = sorted_values[lo] as f64;
        let hi_val = sorted_values[hi] as f64;
        (lo_val + frac * (hi_val - lo_val)).round() as u64
    }

    /// Compute stats from the stored samples for the named benchmark.
    pub fn compute_stats(&self, name: &str) -> Option<BenchmarkStats> {
        let samples = self.results.get(name)?;
        if samples.is_empty() {
            return None;
        }

        let total = samples.len();
        let success_count = samples.iter().filter(|s| s.success).count();
        let success_rate = success_count as f64 / total as f64;

        // Work only on successful durations for latency stats
        let mut durations: Vec<u64> = samples.iter()
            .filter(|s| s.success)
            .map(|s| s.duration_us)
            .collect();

        if durations.is_empty() {
            return Some(BenchmarkStats {
                name: name.to_string(),
                min_us: 0,
                max_us: 0,
                mean_us: 0.0,
                median_us: 0.0,
                p95_us: 0.0,
                p99_us: 0.0,
                std_dev_us: 0.0,
                samples: total,
                success_rate,
                throughput_rps: 0.0,
            });
        }

        durations.sort_unstable();
        let n = durations.len();
        let min_us = durations[0];
        let max_us = durations[n - 1];
        let mean_us = durations.iter().sum::<u64>() as f64 / n as f64;
        let median_us = Self::percentile(&durations, 50.0) as f64;
        let p95_us = Self::percentile(&durations, 95.0) as f64;
        let p99_us = Self::percentile(&durations, 99.0) as f64;
        let variance = durations.iter()
            .map(|&d| (d as f64 - mean_us).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_dev_us = variance.sqrt();

        // Throughput: success_count / total_time_seconds
        let total_us: u64 = durations.iter().sum();
        let throughput_rps = if total_us > 0 {
            success_count as f64 / (total_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        Some(BenchmarkStats {
            name: name.to_string(),
            min_us,
            max_us,
            mean_us,
            median_us,
            p95_us,
            p99_us,
            std_dev_us,
            samples: total,
            success_rate,
            throughput_rps,
        })
    }

    /// Run a benchmark: first warmup iterations (not recorded), then measurement iterations.
    pub fn run_benchmark<F: Fn(u32) -> Result<(), String>>(
        &mut self,
        config: BenchmarkConfig,
        f: F,
    ) -> BenchmarkStats {
        // Warmup
        for i in 0..config.warmup_iterations {
            let _ = f(i);
        }

        // Measurement
        for i in 0..config.measurement_iterations {
            let start = Instant::now();
            let result = f(i);
            let elapsed_us = start.elapsed().as_micros() as u64;
            let success = result.is_ok();
            let error = result.err();
            self.add_sample(&config.name, LatencySample {
                iteration: i,
                duration_us: elapsed_us,
                success,
                error,
            });
        }

        self.compute_stats(&config.name).unwrap_or(BenchmarkStats {
            name: config.name,
            min_us: 0,
            max_us: 0,
            mean_us: 0.0,
            median_us: 0.0,
            p95_us: 0.0,
            p99_us: 0.0,
            std_dev_us: 0.0,
            samples: 0,
            success_rate: 0.0,
            throughput_rps: 0.0,
        })
    }

    /// Compare two benchmark stats and determine which is faster.
    pub fn compare(a: &BenchmarkStats, b: &BenchmarkStats) -> BenchmarkComparison {
        let mean_diff_us = a.mean_us - b.mean_us;
        let (faster, speedup_factor) = if a.mean_us <= b.mean_us {
            let factor = if a.mean_us > 0.0 { b.mean_us / a.mean_us } else { 1.0 };
            (a.name.clone(), factor)
        } else {
            let factor = if b.mean_us > 0.0 { a.mean_us / b.mean_us } else { 1.0 };
            (b.name.clone(), factor)
        };

        // Significant if p95 ranges don't overlap
        let a_p95_lo = a.mean_us - a.p95_us;
        let a_p95_hi = a.p95_us;
        let b_p95_lo = b.mean_us - b.p95_us;
        let b_p95_hi = b.p95_us;
        let is_significant = a_p95_hi < b_p95_lo || b_p95_hi < a_p95_lo;

        BenchmarkComparison {
            faster,
            speedup_factor,
            mean_diff_us,
            is_significant,
        }
    }

    /// Generate a markdown table report of all benchmarks.
    pub fn suite_report(&self) -> String {
        let mut out = String::new();
        out.push_str("| Name | Samples | Mean (us) | Median (us) | P95 (us) | P99 (us) | Std Dev (us) | Success Rate | Throughput (rps) |\n");
        out.push_str("|------|---------|-----------|-------------|----------|----------|--------------|--------------|------------------|\n");

        let mut names: Vec<&String> = self.results.keys().collect();
        names.sort();

        for name in names {
            if let Some(stats) = self.compute_stats(name) {
                out.push_str(&format!(
                    "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2}% | {:.2} |\n",
                    stats.name,
                    stats.samples,
                    stats.mean_us,
                    stats.median_us,
                    stats.p95_us,
                    stats.p99_us,
                    stats.std_dev_us,
                    stats.success_rate * 100.0,
                    stats.throughput_rps,
                ));
            }
        }
        out
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples(durations: &[u64]) -> Vec<LatencySample> {
        durations.iter().enumerate().map(|(i, &d)| LatencySample {
            iteration: i as u32,
            duration_us: d,
            success: true,
            error: None,
        }).collect()
    }

    #[test]
    fn test_percentile_at_known_positions() {
        let values = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(BenchmarkRunner::percentile(&values, 0.0), 1);
        assert_eq!(BenchmarkRunner::percentile(&values, 100.0), 10);
        let p50 = BenchmarkRunner::percentile(&values, 50.0);
        assert!(p50 >= 5 && p50 <= 6, "p50 should be 5 or 6, got {}", p50);
    }

    #[test]
    fn test_compute_stats_from_samples() {
        let mut runner = BenchmarkRunner::new();
        for s in make_samples(&[100, 200, 300, 400, 500]) {
            runner.add_sample("test", s);
        }
        let stats = runner.compute_stats("test").unwrap();
        assert_eq!(stats.samples, 5);
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 500);
        assert!((stats.mean_us - 300.0).abs() < 1.0);
        assert!((stats.success_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_comparison_finds_faster() {
        let a = BenchmarkStats {
            name: "fast".to_string(),
            min_us: 10, max_us: 50, mean_us: 20.0, median_us: 20.0,
            p95_us: 45.0, p99_us: 50.0, std_dev_us: 5.0,
            samples: 100, success_rate: 1.0, throughput_rps: 50000.0,
        };
        let b = BenchmarkStats {
            name: "slow".to_string(),
            min_us: 100, max_us: 500, mean_us: 200.0, median_us: 200.0,
            p95_us: 450.0, p99_us: 500.0, std_dev_us: 50.0,
            samples: 100, success_rate: 1.0, throughput_rps: 5000.0,
        };
        let cmp = BenchmarkRunner::compare(&a, &b);
        assert_eq!(cmp.faster, "fast");
        assert!(cmp.speedup_factor > 1.0);
    }

    #[test]
    fn test_run_benchmark_populates_results() {
        let mut runner = BenchmarkRunner::new();
        let config = BenchmarkConfig {
            name: "noop".to_string(),
            warmup_iterations: 2,
            measurement_iterations: 10,
            timeout_ms: 5000,
        };
        let stats = runner.run_benchmark(config, |_| Ok(()));
        assert_eq!(stats.samples, 10);
        assert_eq!(stats.success_rate, 1.0);
    }
}
