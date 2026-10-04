//! # Cost Anomaly Detection
//!
//! Detects unusual spending patterns by tracking a rolling mean and standard
//! deviation of request costs per model.  When a request costs more than 2×
//! the rolling mean, a [`CostAnomaly`] is raised with an appropriate severity.
//!
//! ## Usage
//!
//! ```rust
//! use llm_cost_dashboard::cost::anomaly::{AnomalyDetector, AnomalySeverity};
//! use llm_cost_dashboard::cost::CostRecord;
//!
//! let mut detector = AnomalyDetector::new();
//! let record = CostRecord::new("gpt-4o-mini", "openai", 512, 256, 34);
//! if let Some(anomaly) = detector.check(&record) {
//!     println!("Anomaly detected: {} on {} (severity: {:?})", anomaly.actual, anomaly.model, anomaly.severity);
//! }
//! ```

use std::collections::HashMap;

use crate::cost::CostRecord;

/// Severity of a detected cost anomaly.
///
/// Severity is determined by how many multiples of the rolling mean the
/// actual cost exceeds:
///
/// - [`Low`][AnomalySeverity::Low]: 2–3× the rolling mean
/// - [`Medium`][AnomalySeverity::Medium]: 3–5× the rolling mean
/// - [`High`][AnomalySeverity::High]: >5× the rolling mean
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalySeverity {
    /// Cost is 2–3× the rolling mean — worth noting but not urgent.
    Low,
    /// Cost is 3–5× the rolling mean — investigate promptly.
    Medium,
    /// Cost is >5× the rolling mean — requires immediate attention.
    High,
}

impl AnomalySeverity {
    /// Classify a ratio of `actual / expected` into a severity level.
    ///
    /// Returns `None` when the ratio is below 2.0 (no anomaly).
    pub fn classify(ratio: f64) -> Option<Self> {
        if ratio >= 5.0 {
            Some(Self::High)
        } else if ratio >= 3.0 {
            Some(Self::Medium)
        } else if ratio >= 2.0 {
            Some(Self::Low)
        } else {
            None
        }
    }
}

impl std::fmt::Display for AnomalySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
        }
    }
}

/// A detected cost anomaly for a single LLM request.
///
/// Produced by [`AnomalyDetector::check`] when the request cost exceeds the
/// 2× threshold.
#[derive(Debug, Clone)]
pub struct CostAnomaly {
    /// The model that produced the anomalous request.
    pub model: String,
    /// The rolling mean cost that was used as the baseline (USD).
    pub expected: f64,
    /// The actual cost of the request that triggered the anomaly (USD).
    pub actual: f64,
    /// How severe the anomaly is.
    pub severity: AnomalySeverity,
    /// Timestamp when the anomaly was detected (UTC).
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

impl CostAnomaly {
    /// Human-readable ratio string, e.g. `"3.4×"`.
    pub fn ratio_str(&self) -> String {
        if self.expected > 0.0 {
            format!("{:.1}x", self.actual / self.expected)
        } else {
            "∞x".to_string()
        }
    }
}

/// Welford online algorithm state for a single model.
///
/// Maintains a running mean and variance without storing all historical costs.
#[derive(Debug, Default, Clone)]
struct ModelStats {
    /// Number of requests seen so far.
    count: u64,
    /// Running mean (Welford M).
    mean: f64,
    /// Running sum of squared deviations from the mean (Welford S).
    m2: f64,
}

impl ModelStats {
    /// Update the running statistics with a new cost observation.
    fn update(&mut self, cost: f64) {
        self.count += 1;
        let delta = cost - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = cost - self.mean;
        self.m2 += delta * delta2;
    }

    /// Sample standard deviation.  Returns `0.0` for fewer than 2 samples.
    #[allow(dead_code)]
    fn std_dev(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count - 1) as f64).sqrt()
        }
    }
}

/// Tracks per-model rolling statistics and detects cost anomalies.
///
/// Uses Welford's online algorithm to maintain a running mean and variance
/// without storing the full history of request costs, making it suitable for
/// long-running processes.
///
/// A new record must be checked via [`AnomalyDetector::check`] *before*
/// calling `AnomalyDetector::record` so that the current observation is
/// compared against the historical baseline rather than itself.
#[derive(Debug, Default)]
pub struct AnomalyDetector {
    /// Per-model rolling statistics.
    stats: HashMap<String, ModelStats>,
    /// Minimum number of samples required before anomaly detection kicks in.
    ///
    /// Defaults to 3 so that the first few requests cannot spuriously trigger
    /// an alert before a stable mean is established.
    min_samples: u64,
}

impl AnomalyDetector {
    /// Create a new detector.
    ///
    /// Anomaly detection is suppressed until at least 3 requests have been
    /// seen per model (see [`AnomalyDetector::with_min_samples`] to change
    /// this).
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            min_samples: 3,
        }
    }

    /// Override the minimum number of samples required before anomaly
    /// detection activates for a model.
    pub fn with_min_samples(mut self, n: u64) -> Self {
        self.min_samples = n;
        self
    }

    /// Check `record` against the rolling mean for its model.
    ///
    /// Returns a [`CostAnomaly`] if the request cost exceeds 2× the rolling
    /// mean **and** at least [`min_samples`][AnomalyDetector::with_min_samples]
    /// prior requests have been seen for this model.
    ///
    /// The detector's internal statistics are updated *after* the comparison
    /// so that the current cost is not included in its own baseline.
    pub fn check(&mut self, record: &CostRecord) -> Option<CostAnomaly> {
        let cost = record.total_cost_usd;
        let entry = self.stats.entry(record.model.clone()).or_default();

        let anomaly = if entry.count >= self.min_samples && entry.mean > 0.0 {
            let ratio = cost / entry.mean;
            AnomalySeverity::classify(ratio).map(|severity| CostAnomaly {
                model: record.model.clone(),
                expected: entry.mean,
                actual: cost,
                severity,
                detected_at: chrono::Utc::now(),
            })
        } else {
            None
        };

        // Update statistics after the check so the current sample is not its
        // own baseline.
        entry.update(cost);

        anomaly
    }

    /// Return the current rolling mean for `model`, or `None` if no data has
    /// been recorded for that model yet.
    pub fn mean_for(&self, model: &str) -> Option<f64> {
        self.stats.get(model).map(|s| s.mean)
    }

    /// Number of samples recorded for `model`.
    pub fn sample_count_for(&self, model: &str) -> u64 {
        self.stats.get(model).map(|s| s.count).unwrap_or(0)
    }

    /// Reset all accumulated statistics, clearing every model's history.
    pub fn reset(&mut self) {
        self.stats.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::cost::CostRecord;

    fn rec(model: &str, cost: f64) -> CostRecord {
        let mut r = CostRecord::new(model, "test", 0, 0, 0);
        r.total_cost_usd = cost;
        r
    }

    #[test]
    fn no_anomaly_with_fewer_than_min_samples() {
        let mut d = AnomalyDetector::new();
        // Default min_samples = 3; first two records should never trigger.
        assert!(d.check(&rec("gpt-4o", 1.0)).is_none());
        assert!(d.check(&rec("gpt-4o", 1.0)).is_none());
    }

    #[test]
    fn no_anomaly_for_normal_cost() {
        let mut d = AnomalyDetector::new().with_min_samples(3);
        for _ in 0..5 {
            d.check(&rec("gpt-4o", 1.0));
        }
        // 1.5× — below 2× threshold.
        assert!(d.check(&rec("gpt-4o", 1.5)).is_none());
    }

    #[test]
    fn anomaly_low_severity_at_2x() {
        let mut d = AnomalyDetector::new().with_min_samples(3);
        for _ in 0..5 {
            d.check(&rec("gpt-4o", 1.0));
        }
        let anomaly = d.check(&rec("gpt-4o", 2.5)).unwrap();
        assert_eq!(anomaly.severity, AnomalySeverity::Low);
        assert_eq!(anomaly.model, "gpt-4o");
    }

    #[test]
    fn anomaly_medium_severity_at_3x() {
        let mut d = AnomalyDetector::new().with_min_samples(3);
        for _ in 0..5 {
            d.check(&rec("gpt-4o", 1.0));
        }
        let anomaly = d.check(&rec("gpt-4o", 4.0)).unwrap();
        assert_eq!(anomaly.severity, AnomalySeverity::Medium);
    }

    #[test]
    fn anomaly_high_severity_at_5x() {
        let mut d = AnomalyDetector::new().with_min_samples(3);
        for _ in 0..5 {
            d.check(&rec("gpt-4o", 1.0));
        }
        let anomaly = d.check(&rec("gpt-4o", 6.0)).unwrap();
        assert_eq!(anomaly.severity, AnomalySeverity::High);
    }

    #[test]
    fn mean_tracked_per_model() {
        let mut d = AnomalyDetector::new();
        d.check(&rec("gpt-4o", 2.0));
        d.check(&rec("gpt-4o", 4.0));
        // mean = 3.0
        assert!((d.mean_for("gpt-4o").unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn sample_count_increments() {
        let mut d = AnomalyDetector::new();
        d.check(&rec("claude-sonnet-4-6", 1.0));
        d.check(&rec("claude-sonnet-4-6", 1.0));
        assert_eq!(d.sample_count_for("claude-sonnet-4-6"), 2);
        assert_eq!(d.sample_count_for("gpt-4o"), 0);
    }

    #[test]
    fn reset_clears_all_stats() {
        let mut d = AnomalyDetector::new();
        d.check(&rec("gpt-4o", 1.0));
        d.reset();
        assert_eq!(d.sample_count_for("gpt-4o"), 0);
        assert!(d.mean_for("gpt-4o").is_none());
    }

    #[test]
    fn ratio_str_displays_correctly() {
        let anomaly = CostAnomaly {
            model: "gpt-4o".into(),
            expected: 1.0,
            actual: 3.5,
            severity: AnomalySeverity::Medium,
            detected_at: chrono::Utc::now(),
        };
        assert_eq!(anomaly.ratio_str(), "3.5x");
    }

    #[test]
    fn severity_classify_none_below_2x() {
        assert!(AnomalySeverity::classify(1.9).is_none());
    }

    #[test]
    fn severity_classify_low_at_boundary() {
        assert_eq!(AnomalySeverity::classify(2.0).unwrap(), AnomalySeverity::Low);
        assert_eq!(AnomalySeverity::classify(2.9).unwrap(), AnomalySeverity::Low);
    }

    #[test]
    fn severity_classify_medium_at_boundary() {
        assert_eq!(
            AnomalySeverity::classify(3.0).unwrap(),
            AnomalySeverity::Medium
        );
        assert_eq!(
            AnomalySeverity::classify(4.9).unwrap(),
            AnomalySeverity::Medium
        );
    }

    #[test]
    fn severity_classify_high_at_5x() {
        assert_eq!(AnomalySeverity::classify(5.0).unwrap(), AnomalySeverity::High);
    }
}
