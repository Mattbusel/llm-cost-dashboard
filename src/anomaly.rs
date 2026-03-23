//! # Cost Anomaly Detection
//!
//! Rolling Z-score based anomaly detector for per-request cost spikes.
//!
//! [`CostAnomalyDetector`] maintains a sliding window of recent request costs
//! and flags any request whose cost deviates more than `threshold` standard
//! deviations from the rolling mean as anomalous.
//!
//! ## Example
//!
//! ```
//! use llm_cost_dashboard::anomaly::CostAnomalyDetector;
//!
//! let mut detector = CostAnomalyDetector::new(50, 3.0);
//!
//! // Feed observations; get back Some(AnomalyEvent) when a spike is detected.
//! for _ in 0..49 {
//!     detector.observe("gpt-4o-mini", 0.001);
//! }
//! // A cost 10x above the mean should trigger an anomaly.
//! let event = detector.observe("gpt-4o-mini", 0.10);
//! assert!(event.is_some());
//! ```

use std::collections::VecDeque;
use std::time::SystemTime;

use chrono::{DateTime, Utc};

/// A single anomaly event produced when a cost observation falls outside the
/// configured Z-score threshold.
#[derive(Debug, Clone)]
pub struct AnomalyEvent {
    /// Wall-clock time the anomaly was detected.
    pub timestamp: SystemTime,
    /// Model that produced the anomalous request.
    pub model: String,
    /// Per-request cost that triggered the alert (USD).
    pub cost_usd: f64,
    /// Z-score of this observation relative to the rolling window.
    pub z_score: f64,
    /// Rolling window mean at the time of detection.
    pub window_mean: f64,
    /// Rolling window standard deviation at the time of detection.
    pub window_std: f64,
}

/// Rolling Z-score anomaly detector for per-request cost spikes.
///
/// Maintains a sliding window of the most recent `window_size` cost
/// observations and computes an incremental mean and variance using
/// Welford-style online updates (via the sum-of-squares shortcut) to avoid
/// an O(n) pass on every observation.
///
/// An observation is flagged as anomalous when its Z-score exceeds
/// `threshold`.  The window must contain at least two observations before
/// any detection can occur (standard deviation is undefined for n < 2).
pub struct CostAnomalyDetector {
    /// Sliding window of the most recent cost observations.
    window: VecDeque<f64>,
    /// Maximum number of observations retained in the window.
    window_size: usize,
    /// Z-score threshold above which an observation is flagged.  Default: 3.0.
    threshold: f64,
    /// Running sum of all values in the window (Σx).
    sum: f64,
    /// Running sum of squares of all values in the window (Σx²).
    sum_sq: f64,
}

impl CostAnomalyDetector {
    /// Create a new detector.
    ///
    /// # Arguments
    ///
    /// * `window_size` – number of past observations to include in the rolling
    ///   statistics.  Clamped to a minimum of 2 so that standard deviation is
    ///   always meaningful.
    /// * `threshold` – Z-score above which an observation is considered
    ///   anomalous.  A value of `3.0` is a common starting point (flags
    ///   roughly the top 0.15% of a normal distribution).
    pub fn new(window_size: usize, threshold: f64) -> Self {
        let window_size = window_size.max(2);
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            threshold,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Feed a new cost observation and return an [`AnomalyEvent`] if the
    /// observation is anomalous.
    ///
    /// The observation is always added to the window regardless of whether it
    /// is flagged.  When the window is full the oldest value is evicted first.
    ///
    /// Detection requires at least two prior observations (so the first
    /// observation never triggers an event).
    pub fn observe(&mut self, model: &str, cost: f64) -> Option<AnomalyEvent> {
        // Capture statistics *before* adding the new value so the new
        // observation is scored against the existing window distribution.
        let n = self.window.len();
        let event = if n >= 2 {
            let mean = self.mean();
            let std = self.std_dev();
            // Only compute Z-score when standard deviation is non-zero.
            if std > f64::EPSILON {
                let z = (cost - mean) / std;
                if z.abs() > self.threshold {
                    Some(AnomalyEvent {
                        timestamp: SystemTime::now(),
                        model: model.to_string(),
                        cost_usd: cost,
                        z_score: z,
                        window_mean: mean,
                        window_std: std,
                    })
                } else {
                    None
                }
            } else if (cost - mean).abs() > f64::EPSILON {
                // Std-dev is zero (all prior values identical) but the new
                // value differs from the mean — treat as a maximally anomalous
                // event with an infinite z-score represented as f64::INFINITY.
                Some(AnomalyEvent {
                    timestamp: SystemTime::now(),
                    model: model.to_string(),
                    cost_usd: cost,
                    z_score: f64::INFINITY,
                    window_mean: mean,
                    window_std: std,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Evict oldest observation if window is full.
        if self.window.len() == self.window_size {
            if let Some(evicted) = self.window.pop_front() {
                self.sum -= evicted;
                self.sum_sq -= evicted * evicted;
            }
        }

        // Insert new observation.
        self.window.push_back(cost);
        self.sum += cost;
        self.sum_sq += cost * cost;

        event
    }

    /// Rolling mean of the current window.
    ///
    /// Returns `0.0` when the window is empty.
    pub fn mean(&self) -> f64 {
        let n = self.window.len();
        if n == 0 {
            return 0.0;
        }
        self.sum / n as f64
    }

    /// Population standard deviation of the current window.
    ///
    /// Uses the computational formula `sqrt(E[x²] - E[x]²)` derived from the
    /// maintained running sums to avoid O(n) recomputation.  Returns `0.0`
    /// when the window contains fewer than two observations.
    pub fn std_dev(&self) -> f64 {
        let n = self.window.len();
        if n < 2 {
            return 0.0;
        }
        let nf = n as f64;
        let mean = self.sum / nf;
        let variance = (self.sum_sq / nf) - mean * mean;
        // Guard against tiny negative or near-zero values caused by
        // floating-point rounding (e.g. a window of identical values).
        // Use a relative threshold: if variance is negligible compared to
        // mean², treat it as exactly zero.
        let relative_eps = mean * mean * f64::EPSILON * nf;
        if variance <= relative_eps {
            return 0.0;
        }
        variance.sqrt()
    }

    /// Number of observations currently held in the sliding window.
    pub fn window_size(&self) -> usize {
        self.window.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Fill the detector with `n` identical observations and return it.
    fn primed(n: usize, cost: f64) -> CostAnomalyDetector {
        let mut d = CostAnomalyDetector::new(100, 3.0);
        for _ in 0..n {
            d.observe("test-model", cost);
        }
        d
    }

    #[test]
    fn test_new_window_is_empty() {
        let d = CostAnomalyDetector::new(50, 3.0);
        assert_eq!(d.window_size(), 0);
    }

    #[test]
    fn test_window_size_clamped_to_minimum_two() {
        let d = CostAnomalyDetector::new(0, 3.0);
        // Internal window_size field is clamped; observe twice to verify no panic.
        let mut d2 = CostAnomalyDetector::new(1, 3.0);
        d2.observe("m", 0.01);
        d2.observe("m", 0.01);
        assert_eq!(d.window_size(), 0); // empty, nothing observed yet
    }

    #[test]
    fn test_first_observation_never_anomalous() {
        let mut d = CostAnomalyDetector::new(50, 3.0);
        assert!(d.observe("m", 9999.0).is_none());
    }

    #[test]
    fn test_second_observation_never_anomalous_when_std_is_zero() {
        let mut d = CostAnomalyDetector::new(50, 3.0);
        d.observe("m", 0.001);
        // std_dev is 0 for a single element window, so no anomaly fires.
        assert!(d.observe("m", 9999.0).is_none());
    }

    #[test]
    fn test_spike_triggers_anomaly() {
        let mut d = primed(49, 0.001);
        let event = d.observe("gpt-4o-mini", 1.0);
        assert!(event.is_some());
        let ev = event.unwrap();
        assert!(ev.z_score > 3.0);
        assert_eq!(ev.model, "gpt-4o-mini");
    }

    #[test]
    fn test_normal_cost_no_anomaly() {
        let mut d = primed(49, 0.001);
        // A cost equal to the mean should not be flagged.
        let event = d.observe("m", 0.001);
        assert!(event.is_none());
    }

    #[test]
    fn test_mean_correct_after_observations() {
        let mut d = CostAnomalyDetector::new(10, 3.0);
        for i in 1u64..=5 {
            d.observe("m", i as f64);
        }
        // Mean of 1+2+3+4+5 = 3.0
        assert!((d.mean() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_evicts_old_values() {
        let mut d = CostAnomalyDetector::new(3, 3.0);
        d.observe("m", 1.0);
        d.observe("m", 2.0);
        d.observe("m", 3.0);
        // Window is now full: [1,2,3]
        d.observe("m", 4.0);
        // After eviction: [2,3,4], mean = 3.0
        assert!((d.mean() - 3.0).abs() < 1e-9);
        assert_eq!(d.window_size(), 3);
    }

    #[test]
    fn test_std_dev_zero_for_constant_window() {
        let d = primed(10, 0.005);
        assert!(d.std_dev() < 1e-12);
    }

    #[test]
    fn test_anomaly_event_fields_populated() {
        // Build a window with slight variance so window_std > 0 and the
        // anomaly event carries a finite z-score.
        let mut d = CostAnomalyDetector::new(100, 3.0);
        for i in 0..49u64 {
            // Alternate between 0.001 and 0.002 to introduce variance.
            d.observe("test-model", if i % 2 == 0 { 0.001 } else { 0.002 });
        }
        let ev = d.observe("claude-sonnet-4-6", 5.0).unwrap();
        assert_eq!(ev.model, "claude-sonnet-4-6");
        assert!((ev.cost_usd - 5.0).abs() < 1e-9);
        assert!(ev.window_mean > 0.0);
        assert!(ev.window_std > 0.0);
        assert!(ev.z_score > 0.0);
    }
}

// ============================================================================
// AnomalyDetector — Welford online algorithm with full report generation
// ============================================================================

/// Configuration for [`AnomalyDetector`].
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Number of observations that form the rolling window.
    ///
    /// Must be at least 2.  Clamped internally if smaller.
    pub window_size: usize,
    /// Z-score above which an observation is classified as anomalous.
    ///
    /// Both positive (overspend) and negative (underspend) deviations are checked.
    pub z_threshold: f64,
    /// Minimum number of samples before the detector can fire.
    pub min_samples: usize,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            window_size: 30,
            z_threshold: 3.0,
            min_samples: 5,
        }
    }
}

/// Per-observation anomaly assessment.
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Whether this observation is classified as anomalous.
    pub is_anomaly: bool,
    /// Z-score of this observation (signed; positive = overspend).
    pub z_score: f64,
    /// Rolling mean at the time of this observation (USD).
    pub mean_usd: f64,
    /// Rolling standard deviation at the time of this observation (USD).
    pub stddev_usd: f64,
    /// The observed cost value (USD).
    pub current_usd: f64,
}

/// Aggregate report over a batch of observations.
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    /// All `(timestamp, result)` pairs that were classified as anomalous.
    pub anomalies: Vec<(DateTime<Utc>, AnomalyResult)>,
    /// Fraction of observations classified as anomalous (0.0–1.0).
    pub anomaly_rate: f64,
    /// Maximum absolute Z-score observed across all inputs.
    pub max_z_score: f64,
}

/// Z-score anomaly detector using Welford's online algorithm.
///
/// Maintains only the count, mean, and M2 aggregates — no stored history —
/// giving O(1) memory and O(1) per-observation update.
///
/// Detects both positive spikes (overspend) and negative dips
/// (underspend / service outage).
pub struct AnomalyDetector {
    config: AnomalyConfig,
    /// Number of observations fed so far.
    count: usize,
    /// Welford running mean.
    mean: f64,
    /// Welford M2 aggregate (sum of squared deviations from the mean).
    m2: f64,
    /// Sliding window for tracking oldest values so we can age them out.
    ///
    /// We use a VecDeque of (observation, delta_mean) pairs.  When the
    /// window is full we remove the oldest value using the inverse Welford
    /// update.
    window: VecDeque<f64>,
}

impl AnomalyDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: AnomalyConfig) -> Self {
        let window_size = config.window_size.max(2);
        Self {
            config: AnomalyConfig {
                window_size,
                ..config
            },
            count: 0,
            mean: 0.0,
            m2: 0.0,
            window: VecDeque::with_capacity(window_size),
        }
    }

    /// Feed a single cost observation and return the anomaly assessment.
    pub fn observe(&mut self, cost: f64) -> AnomalyResult {
        // If window full, remove oldest value (reverse Welford).
        if self.window.len() == self.config.window_size {
            if let Some(old) = self.window.pop_front() {
                // Reverse Welford update.
                self.count -= 1;
                if self.count == 0 {
                    self.mean = 0.0;
                    self.m2 = 0.0;
                } else {
                    let new_mean = (self.mean * (self.count as f64 + 1.0) - old) / self.count as f64;
                    self.m2 -= (old - self.mean) * (old - new_mean);
                    self.mean = new_mean;
                    if self.m2 < 0.0 {
                        self.m2 = 0.0; // numerical guard
                    }
                }
            }
        }

        // Capture pre-observation stats for scoring.
        let pre_mean = self.mean;
        let pre_stddev = self.current_stddev();
        let pre_count = self.count;

        // Welford update with the new observation.
        self.count += 1;
        let delta = cost - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = cost - self.mean;
        self.m2 += delta * delta2;
        self.window.push_back(cost);

        // Classify.
        let enough = pre_count >= self.config.min_samples.max(2);
        let (z_score, is_anomaly) = if enough && pre_stddev > f64::EPSILON {
            let z = (cost - pre_mean) / pre_stddev;
            (z, z.abs() > self.config.z_threshold)
        } else {
            (0.0, false)
        };

        AnomalyResult {
            is_anomaly,
            z_score,
            mean_usd: pre_mean,
            stddev_usd: pre_stddev,
            current_usd: cost,
        }
    }

    /// Run `observe` on each `(timestamp, cost)` pair and return a full report.
    pub fn analyze(&mut self, observations: &[(DateTime<Utc>, f64)]) -> AnomalyReport {
        let total = observations.len();
        let mut anomalies = Vec::new();
        let mut max_z: f64 = 0.0;

        for (ts, cost) in observations {
            let result = self.observe(*cost);
            if result.z_score.abs() > max_z {
                max_z = result.z_score.abs();
            }
            if result.is_anomaly {
                anomalies.push((*ts, result));
            }
        }

        AnomalyReport {
            anomaly_rate: if total == 0 {
                0.0
            } else {
                anomalies.len() as f64 / total as f64
            },
            max_z_score: max_z,
            anomalies,
        }
    }

    /// Current sample standard deviation from the Welford accumulator.
    fn current_stddev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let variance = self.m2 / (self.count as f64 - 1.0);
        if variance <= 0.0 {
            0.0
        } else {
            variance.sqrt()
        }
    }

    /// Number of observations currently in the sliding window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Current rolling mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Current rolling standard deviation.
    pub fn stddev(&self) -> f64 {
        self.current_stddev()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod welford_tests {
    use super::*;
    use chrono::Utc;

    fn cfg(window: usize, threshold: f64, min_samples: usize) -> AnomalyConfig {
        AnomalyConfig { window_size: window, z_threshold: threshold, min_samples }
    }

    #[test]
    fn test_new_detector_empty() {
        let d = AnomalyDetector::new(AnomalyConfig::default());
        assert_eq!(d.window_len(), 0);
        assert_eq!(d.mean(), 0.0);
        assert_eq!(d.stddev(), 0.0);
    }

    #[test]
    fn test_first_observation_not_anomaly() {
        let mut d = AnomalyDetector::new(AnomalyConfig::default());
        let r = d.observe(100.0);
        assert!(!r.is_anomaly);
    }

    #[test]
    fn test_constant_window_no_anomaly() {
        let mut d = AnomalyDetector::new(cfg(20, 3.0, 5));
        for _ in 0..15 {
            let r = d.observe(0.001);
            assert!(!r.is_anomaly);
        }
    }

    #[test]
    fn test_spike_detected() {
        let mut d = AnomalyDetector::new(cfg(50, 3.0, 5));
        for _ in 0..20 {
            d.observe(0.001);
        }
        let r = d.observe(10.0); // massive spike
        assert!(r.is_anomaly);
        assert!(r.z_score > 3.0);
    }

    #[test]
    fn test_negative_dip_detected() {
        let mut d = AnomalyDetector::new(cfg(50, 3.0, 5));
        for _ in 0..20 {
            d.observe(1.0); // normal cost
        }
        let r = d.observe(0.000001); // near-zero dip
        assert!(r.is_anomaly);
        assert!(r.z_score < -3.0);
    }

    #[test]
    fn test_min_samples_gate() {
        let mut d = AnomalyDetector::new(cfg(50, 1.0, 10));
        for _ in 0..9 {
            d.observe(0.001);
        }
        // 9 observations < min_samples=10; spike should not fire yet.
        let r = d.observe(999.0);
        assert!(!r.is_anomaly);
    }

    #[test]
    fn test_window_size_respected() {
        let mut d = AnomalyDetector::new(cfg(5, 3.0, 2));
        for i in 0..10 {
            d.observe(i as f64);
        }
        assert_eq!(d.window_len(), 5);
    }

    #[test]
    fn test_anomaly_result_fields() {
        let mut d = AnomalyDetector::new(cfg(50, 3.0, 5));
        for _ in 0..20 {
            d.observe(1.0);
        }
        let r = d.observe(100.0);
        assert!(r.current_usd == 100.0);
        assert!(r.mean_usd > 0.0);
        assert!(r.stddev_usd > 0.0);
        assert!(r.is_anomaly);
    }

    #[test]
    fn test_analyze_returns_anomalies() {
        let mut d = AnomalyDetector::new(cfg(50, 3.0, 5));
        let now = Utc::now();
        let observations: Vec<(DateTime<Utc>, f64)> = (0..25)
            .map(|i| (now, if i == 24 { 100.0 } else { 0.001 }))
            .collect();
        let report = d.analyze(&observations);
        assert!(!report.anomalies.is_empty());
        assert!(report.anomaly_rate > 0.0);
    }

    #[test]
    fn test_analyze_anomaly_rate_zero_for_normal_data() {
        let mut d = AnomalyDetector::new(cfg(50, 3.0, 5));
        let now = Utc::now();
        let observations: Vec<(DateTime<Utc>, f64)> = (0..20)
            .map(|_| (now, 0.001))
            .collect();
        let report = d.analyze(&observations);
        assert!(report.anomalies.is_empty());
        assert_eq!(report.anomaly_rate, 0.0);
    }

    #[test]
    fn test_max_z_score_tracked() {
        let mut d = AnomalyDetector::new(cfg(50, 3.0, 5));
        let now = Utc::now();
        let observations: Vec<(DateTime<Utc>, f64)> = (0..25)
            .map(|i| (now, if i == 24 { 1000.0 } else { 0.001 }))
            .collect();
        let report = d.analyze(&observations);
        assert!(report.max_z_score > 0.0);
    }

    #[test]
    fn test_empty_analyze_produces_empty_report() {
        let mut d = AnomalyDetector::new(AnomalyConfig::default());
        let report = d.analyze(&[]);
        assert!(report.anomalies.is_empty());
        assert_eq!(report.anomaly_rate, 0.0);
        assert_eq!(report.max_z_score, 0.0);
    }

    #[test]
    fn test_mean_updates_correctly() {
        let mut d = AnomalyDetector::new(cfg(10, 3.0, 2));
        d.observe(2.0);
        d.observe(4.0);
        d.observe(6.0);
        // Mean should be approximately (2+4+6)/3 = 4.0.
        assert!((d.mean() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_stddev_increases_with_variance() {
        let mut d = AnomalyDetector::new(cfg(20, 3.0, 2));
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            d.observe(v);
        }
        assert!(d.stddev() > 0.0);
    }

    #[test]
    fn test_window_eviction_keeps_window_size_bounded() {
        let mut d = AnomalyDetector::new(cfg(3, 3.0, 2));
        for i in 0..10 {
            d.observe(i as f64);
            assert!(d.window_len() <= 3);
        }
    }
}
