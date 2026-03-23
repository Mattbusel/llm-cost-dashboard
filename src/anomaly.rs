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
