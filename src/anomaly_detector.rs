//! # Anomaly Detector
//!
//! Cost anomaly detection using Z-Score, IQR, and CUSUM methods.
//!
//! Each [`AnomalyDetector`] maintains a rolling window of recent observations.
//! When a new value is added via [`AnomalyDetector::add_observation`] the
//! configured detection method is run and an [`AnomalyAlert`] is returned if
//! the value is anomalous.

use std::collections::VecDeque;

// ── AnomalyMethod ─────────────────────────────────────────────────────────────

/// Statistical method used for anomaly detection.
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyMethod {
    /// Standard-score (Z-score) method.
    ZScore,
    /// Inter-quartile range fence method.
    IQR,
    /// Cumulative sum (CUSUM) control chart.
    CUSUM,
}

// ── AnomalySeverity ───────────────────────────────────────────────────────────

/// Severity classification of a detected anomaly.
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalySeverity {
    /// Score magnitude < 3× threshold.
    Low,
    /// Score magnitude in [3×, 6×) threshold.
    Medium,
    /// Score magnitude ≥ 6× threshold.
    High,
}

impl AnomalySeverity {
    fn from_score(score: f64, threshold: f64) -> Self {
        let ratio = score.abs() / threshold.max(f64::EPSILON);
        if ratio >= 6.0 {
            AnomalySeverity::High
        } else if ratio >= 3.0 {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }
}

// ── AnomalyAlert ──────────────────────────────────────────────────────────────

/// An anomaly that was detected by an [`AnomalyDetector`].
#[derive(Debug, Clone)]
pub struct AnomalyAlert {
    /// The observed value that triggered the alert.
    pub value: f64,
    /// The computed anomaly score (e.g. Z-score, IQR ratio, CUSUM value).
    pub score: f64,
    /// Name of the detection method that fired.
    pub method_name: String,
    /// Severity classification.
    pub severity: AnomalySeverity,
}

// ── AnomalyDetector ───────────────────────────────────────────────────────────

/// Rolling-window cost anomaly detector.
pub struct AnomalyDetector {
    method: AnomalyMethod,
    window: VecDeque<f64>,
    window_size: usize,
    threshold: f64,
    /// Running CUSUM accumulators (high-side, low-side).
    cusum_high: f64,
    cusum_low: f64,
}

impl AnomalyDetector {
    /// Create a new detector.
    ///
    /// - `method`: which statistical test to use.
    /// - `window_size`: maximum number of historical values to retain.
    /// - `threshold`: method-specific detection threshold.
    pub fn new(method: AnomalyMethod, window_size: usize, threshold: f64) -> Self {
        Self {
            method,
            window: VecDeque::with_capacity(window_size),
            window_size,
            threshold,
            cusum_high: 0.0,
            cusum_low: 0.0,
        }
    }

    /// Add an observation and run the configured anomaly check.
    ///
    /// Returns an [`AnomalyAlert`] if the value is anomalous, otherwise `None`.
    /// The value is always appended to the internal window (evicting the oldest
    /// entry if at capacity).
    pub fn add_observation(&mut self, value: f64) -> Option<AnomalyAlert> {
        // Run detection before adding to window so the new value is compared
        // against the existing history.
        let window_slice: Vec<f64> = self.window.iter().copied().collect();

        let result = match self.method {
            AnomalyMethod::ZScore => {
                Self::detect_zscore(&window_slice, value, self.threshold).map(|score| {
                    AnomalyAlert {
                        value,
                        score,
                        method_name: "ZScore".to_string(),
                        severity: AnomalySeverity::from_score(score, self.threshold),
                    }
                })
            }
            AnomalyMethod::IQR => {
                Self::detect_iqr(&window_slice, value, self.threshold).map(|score| {
                    AnomalyAlert {
                        value,
                        score,
                        method_name: "IQR".to_string(),
                        severity: AnomalySeverity::from_score(score, self.threshold),
                    }
                })
            }
            AnomalyMethod::CUSUM => {
                // detect_cusum handles the running accumulators via a slice;
                // we pass a snapshot but manage state ourselves.
                let mean = if window_slice.is_empty() {
                    0.0
                } else {
                    window_slice.iter().sum::<f64>() / window_slice.len() as f64
                };
                self.cusum_high = (self.cusum_high + value - mean).max(0.0);
                self.cusum_low = (self.cusum_low + mean - value).max(0.0);
                let cusum_val = self.cusum_high.max(self.cusum_low);
                if cusum_val > self.threshold {
                    // Reset on detection.
                    self.cusum_high = 0.0;
                    self.cusum_low = 0.0;
                    Some(AnomalyAlert {
                        value,
                        score: cusum_val,
                        method_name: "CUSUM".to_string(),
                        severity: AnomalySeverity::from_score(cusum_val, self.threshold),
                    })
                } else {
                    None
                }
            }
        };

        // Maintain rolling window.
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(value);

        result
    }

    /// Compute Z-score of `value` relative to `window`.
    ///
    /// Returns the Z-score if `|z| > threshold`, otherwise `None`.
    /// Requires at least 2 values in the window; returns `None` otherwise.
    pub fn detect_zscore(window: &[f64], value: f64, threshold: f64) -> Option<f64> {
        if window.len() < 2 {
            return None;
        }
        let n = window.len() as f64;
        let mean = window.iter().sum::<f64>() / n;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        if std_dev < f64::EPSILON {
            return None;
        }
        let z = (value - mean) / std_dev;
        if z.abs() > threshold {
            Some(z)
        } else {
            None
        }
    }

    /// IQR fence detection.
    ///
    /// Returns a score proportional to the distance beyond the fence if
    /// `value` falls outside `[Q1 - threshold*IQR, Q3 + threshold*IQR]`,
    /// otherwise `None`.  Requires at least 4 values.
    pub fn detect_iqr(window: &[f64], value: f64, threshold: f64) -> Option<f64> {
        if window.len() < 4 {
            return None;
        }
        let mut sorted = window.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let q1 = sorted[n / 4];
        let q3 = sorted[3 * n / 4];
        let iqr = q3 - q1;
        let lower = q1 - threshold * iqr;
        let upper = q3 + threshold * iqr;
        if value < lower {
            Some(lower - value)
        } else if value > upper {
            Some(value - upper)
        } else {
            None
        }
    }

    /// Two-sided CUSUM detection (stateless helper; state management is done
    /// inside [`add_observation`]).
    ///
    /// Accumulates the deviation of `value` from the window mean.
    /// Returns the CUSUM statistic if it exceeds `threshold`.
    pub fn detect_cusum(window: &[f64], value: f64, threshold: f64) -> Option<f64> {
        if window.is_empty() {
            return None;
        }
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        // Stateless version: just check single-step deviation.
        let dev = (value - mean).abs();
        if dev > threshold {
            Some(dev)
        } else {
            None
        }
    }

    /// Access the raw observation history.
    pub fn history(&self) -> &VecDeque<f64> {
        &self.window
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_detects_spike() {
        let mut detector = AnomalyDetector::new(AnomalyMethod::ZScore, 20, 2.0);
        // Populate window with stable values.
        for v in [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0_f64] {
            let _ = detector.add_observation(v);
        }
        // Spike should be detected.
        let alert = detector.add_observation(100.0);
        assert!(alert.is_some(), "Expected Z-score spike to be detected");
        let alert = alert.unwrap();
        assert_eq!(alert.method_name, "ZScore");
        assert!(alert.score > 2.0);
    }

    #[test]
    fn test_zscore_no_alert_for_normal_value() {
        let mut detector = AnomalyDetector::new(AnomalyMethod::ZScore, 20, 3.0);
        for v in [10.0_f64; 10] {
            let _ = detector.add_observation(v);
        }
        let alert = detector.add_observation(11.0);
        assert!(alert.is_none(), "Normal value should not trigger alert");
    }

    #[test]
    fn test_iqr_detects_outlier() {
        let mut detector = AnomalyDetector::new(AnomalyMethod::IQR, 20, 1.5);
        // Values tightly clustered.
        for v in [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0_f64] {
            let _ = detector.add_observation(v);
        }
        // Extreme outlier.
        let alert = detector.add_observation(1000.0);
        assert!(alert.is_some(), "Expected IQR outlier to be detected");
        let alert = alert.unwrap();
        assert_eq!(alert.method_name, "IQR");
    }

    #[test]
    fn test_iqr_no_alert_for_normal_value() {
        let mut detector = AnomalyDetector::new(AnomalyMethod::IQR, 20, 1.5);
        for v in [5.0_f64; 10] {
            let _ = detector.add_observation(v);
        }
        let alert = detector.add_observation(5.5);
        assert!(alert.is_none());
    }

    #[test]
    fn test_cusum_detects_trend() {
        // CUSUM accumulates sustained deviations from the mean.
        let mut detector = AnomalyDetector::new(AnomalyMethod::CUSUM, 20, 10.0);
        // Prime the window with baseline values.
        for v in [10.0_f64; 10] {
            let _ = detector.add_observation(v);
        }
        // Gradual upward trend: each +5 above mean.
        let mut fired = false;
        for _ in 0..10 {
            if detector.add_observation(15.0).is_some() {
                fired = true;
                break;
            }
        }
        assert!(fired, "CUSUM should detect a sustained upward trend");
    }

    #[test]
    fn test_history_grows_correctly() {
        let mut detector = AnomalyDetector::new(AnomalyMethod::ZScore, 5, 3.0);
        for i in 0..7 {
            let _ = detector.add_observation(i as f64);
        }
        // Window capped at window_size=5.
        assert_eq!(detector.history().len(), 5);
    }

    #[test]
    fn test_severity_classification() {
        // Low: ratio < 3.
        assert_eq!(AnomalySeverity::from_score(2.0, 1.0), AnomalySeverity::Low);
        // Medium: ratio in [3, 6).
        assert_eq!(AnomalySeverity::from_score(4.0, 1.0), AnomalySeverity::Medium);
        // High: ratio >= 6.
        assert_eq!(AnomalySeverity::from_score(7.0, 1.0), AnomalySeverity::High);
    }

    #[test]
    fn test_detect_zscore_static() {
        let window = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        // Normal value.
        assert!(AnomalyDetector::detect_zscore(&window, 3.5, 2.0).is_none());
        // Spike.
        assert!(AnomalyDetector::detect_zscore(&window, 100.0, 2.0).is_some());
    }

    #[test]
    fn test_detect_iqr_static() {
        let window = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(AnomalyDetector::detect_iqr(&window, 1000.0, 1.5).is_some());
        assert!(AnomalyDetector::detect_iqr(&window, 4.5, 1.5).is_none());
    }
}
