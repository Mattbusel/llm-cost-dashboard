//! Prometheus-style metrics collection for the LLM cost dashboard.
//!
//! Provides [`MetricsRegistry`] which manages counters, gauges, and histograms
//! and can export them in the Prometheus text exposition format.
//!
//! ## Example
//!
//! ```
//! use llm_cost_dashboard::dashboard_metrics::MetricsRegistry;
//!
//! let reg = MetricsRegistry::new();
//! reg.register_counter("requests_total", "Total number of requests");
//! reg.inc_counter("requests_total", 1.0);
//! let output = reg.export_prometheus();
//! assert!(output.contains("requests_total"));
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// The kind of metric being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Monotonically increasing counter.
    Counter,
    /// Arbitrarily up/down gauge.
    Gauge,
    /// Distribution measured via configurable bucket boundaries.
    Histogram,
}

/// One bucket in a histogram.
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Upper inclusive bound for this bucket (use `f64::INFINITY` for `+Inf`).
    pub upper_bound: f64,
    /// Number of observations falling at or below `upper_bound`.
    pub count: u64,
}

/// The current value held by a [`Metric`].
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Current counter value.
    CounterValue(f64),
    /// Current gauge value.
    GaugeValue(f64),
    /// Histogram state: per-bucket cumulative counts, sum, and total count.
    HistogramValue {
        /// Cumulative counts per bucket.
        buckets: Vec<HistogramBucket>,
        /// Sum of all observed values.
        sum: f64,
        /// Total observation count.
        count: u64,
    },
}

// ---------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------

/// A single named metric with associated metadata and a thread-safe value.
pub struct Metric {
    /// Metric name (must be a valid Prometheus metric name).
    pub name: String,
    /// Human-readable description.
    pub help: String,
    /// Kind of metric.
    pub metric_type: MetricType,
    /// Key/value label pairs attached to this metric series.
    pub labels: HashMap<String, String>,
    /// Current value protected by a mutex.
    pub value: Mutex<MetricValue>,
}

impl std::fmt::Debug for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metric")
            .field("name", &self.name)
            .field("metric_type", &self.metric_type)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MetricsRegistry
// ---------------------------------------------------------------------------

/// Thread-safe registry of named metrics.
///
/// Backed by a [`dashmap::DashMap`] so that concurrent reads and writes from
/// multiple tokio tasks do not block each other.
pub struct MetricsRegistry {
    metrics: dashmap::DashMap<String, Arc<Metric>>,
}

impl MetricsRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self {
            metrics: dashmap::DashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Register a counter metric and return a handle to it.
    ///
    /// If a metric with the same name already exists it is returned unchanged.
    pub fn register_counter(&self, name: &str, help: &str) -> Arc<Metric> {
        self.metrics
            .entry(name.to_string())
            .or_insert_with(|| {
                Arc::new(Metric {
                    name: name.to_string(),
                    help: help.to_string(),
                    metric_type: MetricType::Counter,
                    labels: HashMap::new(),
                    value: Mutex::new(MetricValue::CounterValue(0.0)),
                })
            })
            .clone()
    }

    /// Register a gauge metric and return a handle to it.
    pub fn register_gauge(&self, name: &str, help: &str) -> Arc<Metric> {
        self.metrics
            .entry(name.to_string())
            .or_insert_with(|| {
                Arc::new(Metric {
                    name: name.to_string(),
                    help: help.to_string(),
                    metric_type: MetricType::Gauge,
                    labels: HashMap::new(),
                    value: Mutex::new(MetricValue::GaugeValue(0.0)),
                })
            })
            .clone()
    }

    /// Register a histogram metric with the given bucket boundaries.
    ///
    /// A `+Inf` bucket is appended automatically if not already present.
    pub fn register_histogram(
        &self,
        name: &str,
        help: &str,
        mut buckets: Vec<f64>,
    ) -> Arc<Metric> {
        // Ensure +Inf bucket exists.
        if buckets.last().copied().unwrap_or(0.0) < f64::INFINITY {
            buckets.push(f64::INFINITY);
        }
        let histo_buckets: Vec<HistogramBucket> = buckets
            .iter()
            .map(|&ub| HistogramBucket { upper_bound: ub, count: 0 })
            .collect();

        self.metrics
            .entry(name.to_string())
            .or_insert_with(|| {
                Arc::new(Metric {
                    name: name.to_string(),
                    help: help.to_string(),
                    metric_type: MetricType::Histogram,
                    labels: HashMap::new(),
                    value: Mutex::new(MetricValue::HistogramValue {
                        buckets: histo_buckets,
                        sum: 0.0,
                        count: 0,
                    }),
                })
            })
            .clone()
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Increment a counter by `delta`.  No-op if the metric does not exist or
    /// is not a counter.
    pub fn inc_counter(&self, name: &str, delta: f64) {
        if let Some(metric) = self.metrics.get(name) {
            if let Ok(mut v) = metric.value.lock() {
                if let MetricValue::CounterValue(ref mut c) = *v {
                    *c += delta;
                }
            }
        }
    }

    /// Set a gauge to `value`.  No-op if the metric does not exist or is not
    /// a gauge.
    pub fn set_gauge(&self, name: &str, value: f64) {
        if let Some(metric) = self.metrics.get(name) {
            if let Ok(mut v) = metric.value.lock() {
                if let MetricValue::GaugeValue(ref mut g) = *v {
                    *g = value;
                }
            }
        }
    }

    /// Record one observation in a histogram.  No-op if the metric does not
    /// exist or is not a histogram.
    pub fn observe_histogram(&self, name: &str, value: f64) {
        if let Some(metric) = self.metrics.get(name) {
            if let Ok(mut v) = metric.value.lock() {
                if let MetricValue::HistogramValue {
                    ref mut buckets,
                    ref mut sum,
                    ref mut count,
                } = *v
                {
                    *sum += value;
                    *count += 1;
                    for bucket in buckets.iter_mut() {
                        if value <= bucket.upper_bound {
                            bucket.count += 1;
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    /// Export all metrics in the [Prometheus text exposition format][prom].
    ///
    /// [prom]: https://prometheus.io/docs/instrumenting/exposition_formats/
    pub fn export_prometheus(&self) -> String {
        let mut out = String::new();

        // Collect and sort for deterministic output.
        let mut entries: Vec<(String, Arc<Metric>)> = self
            .metrics
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        for (_, metric) in &entries {
            let type_str = match metric.metric_type {
                MetricType::Counter => "counter",
                MetricType::Gauge => "gauge",
                MetricType::Histogram => "histogram",
            };
            out.push_str(&format!("# HELP {} {}\n", metric.name, metric.help));
            out.push_str(&format!("# TYPE {} {}\n", metric.name, type_str));

            // Build label string.
            let label_str = if metric.labels.is_empty() {
                String::new()
            } else {
                let mut parts: Vec<String> = metric
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                parts.sort();
                format!("{{{}}}", parts.join(","))
            };

            if let Ok(v) = metric.value.lock() {
                match &*v {
                    MetricValue::CounterValue(c) => {
                        out.push_str(&format!("{}{} {}\n", metric.name, label_str, c));
                    }
                    MetricValue::GaugeValue(g) => {
                        out.push_str(&format!("{}{} {}\n", metric.name, label_str, g));
                    }
                    MetricValue::HistogramValue { buckets, sum, count } => {
                        for bucket in buckets {
                            let bound = if bucket.upper_bound == f64::INFINITY {
                                "+Inf".to_string()
                            } else {
                                format!("{}", bucket.upper_bound)
                            };
                            let bucket_labels = if metric.labels.is_empty() {
                                format!("{{le=\"{}\"}}", bound)
                            } else {
                                let mut parts: Vec<String> = metric
                                    .labels
                                    .iter()
                                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                                    .collect();
                                parts.sort();
                                parts.push(format!("le=\"{}\"", bound));
                                format!("{{{}}}", parts.join(","))
                            };
                            out.push_str(&format!(
                                "{}_bucket{} {}\n",
                                metric.name, bucket_labels, bucket.count
                            ));
                        }
                        out.push_str(&format!("{}_sum{} {}\n", metric.name, label_str, sum));
                        out.push_str(&format!("{}_count{} {}\n", metric.name, label_str, count));
                    }
                }
            }
            out.push('\n');
        }

        out
    }

    /// Return a flat list of `(name, value)` pairs for quick inspection.
    ///
    /// For histograms, only the `_sum` value is included.
    pub fn snapshot(&self) -> Vec<(String, f64)> {
        let mut result = Vec::new();
        for entry in self.metrics.iter() {
            let metric = entry.value();
            if let Ok(v) = metric.value.lock() {
                match &*v {
                    MetricValue::CounterValue(c) => result.push((metric.name.clone(), *c)),
                    MetricValue::GaugeValue(g) => result.push((metric.name.clone(), *g)),
                    MetricValue::HistogramValue { sum, .. } => {
                        result.push((format!("{}_sum", metric.name), *sum))
                    }
                }
            }
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_increments() {
        let reg = MetricsRegistry::new();
        reg.register_counter("hits", "Total hits");
        reg.inc_counter("hits", 3.0);
        reg.inc_counter("hits", 2.0);
        let snap = reg.snapshot();
        let val = snap.iter().find(|(n, _)| n == "hits").map(|(_, v)| *v);
        assert_eq!(val, Some(5.0));
    }

    #[test]
    fn test_gauge_set() {
        let reg = MetricsRegistry::new();
        reg.register_gauge("temperature", "Current temperature");
        reg.set_gauge("temperature", 42.0);
        let snap = reg.snapshot();
        let val = snap.iter().find(|(n, _)| n == "temperature").map(|(_, v)| *v);
        assert_eq!(val, Some(42.0));
    }

    #[test]
    fn test_histogram_observe() {
        let reg = MetricsRegistry::new();
        reg.register_histogram("latency", "Request latency ms", vec![10.0, 50.0, 100.0]);
        reg.observe_histogram("latency", 25.0);
        reg.observe_histogram("latency", 75.0);
        let snap = reg.snapshot();
        let sum = snap.iter().find(|(n, _)| n == "latency_sum").map(|(_, v)| *v);
        assert_eq!(sum, Some(100.0));
    }

    #[test]
    fn test_prometheus_export_contains_names() {
        let reg = MetricsRegistry::new();
        reg.register_counter("reqs", "Requests");
        reg.inc_counter("reqs", 7.0);
        let output = reg.export_prometheus();
        assert!(output.contains("# HELP reqs Requests"));
        assert!(output.contains("# TYPE reqs counter"));
        assert!(output.contains("reqs 7"));
    }

    #[test]
    fn test_histogram_prometheus_buckets() {
        let reg = MetricsRegistry::new();
        reg.register_histogram("h", "h help", vec![1.0, 5.0]);
        reg.observe_histogram("h", 0.5);
        let output = reg.export_prometheus();
        assert!(output.contains("h_bucket{le=\"1\"}"));
        assert!(output.contains("h_bucket{le=\"+Inf\"}"));
    }
}
