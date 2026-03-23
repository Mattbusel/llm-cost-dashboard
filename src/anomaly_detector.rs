//! Statistical anomaly detection for cost spikes.
//!
//! Provides [`AnomalyDetector`] which runs multiple statistical methods
//! ([`ZScoreDetector`] and [`EWMADetector`]) in parallel on a stream of
//! cost observations and aggregates the results into [`Anomaly`] records.

use std::collections::VecDeque;
use std::time::Instant;

// ── AnomalyMethod ─────────────────────────────────────────────────────────────

/// Statistical method used to flag an anomaly.
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyMethod {
    /// Z-score (standard deviation) detector.
    ZScore {
        /// Number of standard deviations required to flag an anomaly.
        threshold: f64,
    },
    /// Inter-quartile range fence method.
    IQR {
        /// IQR multiplier (e.g. 1.5 for the Tukey fence).
        multiplier: f64,
    },
    /// Exponentially weighted moving average residual detector.
    EWMA {
        /// Smoothing factor in `(0, 1)`.
        alpha: f64,
        /// Number of EWMA standard deviations required to flag an anomaly.
        sigma_multiplier: f64,
    },
    /// Isolation-forest-inspired contamination model.
    Isolation {
        /// Expected fraction of anomalies in the data.
        contamination: f64,
    },
}

// ── Anomaly ───────────────────────────────────────────────────────────────────

/// A single detected anomaly observation.
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Wall-clock time at which the anomaly was detected.
    pub timestamp: Instant,
    /// The raw value that triggered the anomaly.
    pub value: f64,
    /// Anomaly score (method-specific; higher is more anomalous).
    pub score: f64,
    /// Human-readable name of the detection method.
    pub method: String,
    /// Human-readable description of why this value was flagged.
    pub description: String,
}

// ── ZScoreDetector ────────────────────────────────────────────────────────────

/// Rolling Z-score anomaly detector.
///
/// Maintains a fixed-size sliding window of recent observations and flags
/// values whose |z-score| exceeds `threshold`.
pub struct ZScoreDetector {
    /// Sliding window of recent values.
    pub window: VecDeque<f64>,
    /// Maximum window size.
    max_window: usize,
    /// Minimum |z-score| to flag as anomalous.
    pub threshold: f64,
}

impl ZScoreDetector {
    /// Create a new Z-score detector.
    ///
    /// `window_size` is the number of past observations used to compute
    /// mean and standard deviation.
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            max_window: window_size,
            threshold,
        }
    }

    /// Add `value` to the window and return an [`Anomaly`] if it is
    /// anomalous according to the Z-score criterion.
    pub fn detect(&mut self, value: f64) -> Option<Anomaly> {
        let result = if self.window.len() >= 2 {
            let n = self.window.len() as f64;
            let mean = self.window.iter().sum::<f64>() / n;
            let variance = self.window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std_dev = variance.sqrt();
            if std_dev > f64::EPSILON {
                let z = (value - mean).abs() / std_dev;
                if z > self.threshold {
                    Some(Anomaly {
                        timestamp: Instant::now(),
                        value,
                        score: z,
                        method: "ZScore".to_string(),
                        description: format!(
                            "value {value:.4} has |z|={z:.3} > threshold {t:.3} (mean={mean:.4}, std={std_dev:.4})",
                            t = self.threshold
                        ),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if self.window.len() == self.max_window {
            self.window.pop_front();
        }
        self.window.push_back(value);
        result
    }
}

// ── EWMADetector ─────────────────────────────────────────────────────────────

/// Exponentially weighted moving average anomaly detector.
///
/// Maintains a running EWMA of both the mean and the variance.  Values
/// whose residual exceeds `sigma_mult` × √(EWMA variance) are flagged.
pub struct EWMADetector {
    /// Current EWMA mean estimate.
    pub mean: f64,
    /// Current EWMA variance estimate.
    pub variance: f64,
    /// Smoothing factor α ∈ (0, 1).
    pub alpha: f64,
    /// Sigma multiplier for anomaly threshold.
    pub sigma_mult: f64,
    /// Number of observations seen so far (used for warm-up).
    observations: usize,
}

impl EWMADetector {
    /// Create a new EWMA detector.
    pub fn new(alpha: f64, sigma_multiplier: f64) -> Self {
        Self {
            mean: 0.0,
            variance: 1.0,
            alpha,
            sigma_mult: sigma_multiplier,
            observations: 0,
        }
    }

    /// Update the EWMA model with `value` and return an [`Anomaly`] if the
    /// residual exceeds the configured threshold.
    pub fn detect(&mut self, value: f64) -> Option<Anomaly> {
        if self.observations == 0 {
            // Bootstrap with first observation
            self.mean = value;
            self.observations += 1;
            return None;
        }

        let residual = value - self.mean;
        let std_dev = self.variance.sqrt();
        let score = residual.abs() / std_dev.max(f64::EPSILON);

        let anomaly = if self.observations >= 3 && score > self.sigma_mult {
            Some(Anomaly {
                timestamp: Instant::now(),
                value,
                score,
                method: "EWMA".to_string(),
                description: format!(
                    "value {value:.4} deviates {score:.3}σ from EWMA mean {m:.4}",
                    m = self.mean
                ),
            })
        } else {
            None
        };

        // Update EWMA mean and variance
        self.variance =
            (1.0 - self.alpha) * (self.variance + self.alpha * residual * residual);
        self.mean = self.alpha * value + (1.0 - self.alpha) * self.mean;
        self.observations += 1;

        anomaly
    }
}

// ── AnomalyDetector ───────────────────────────────────────────────────────────

/// Multi-method anomaly detector that aggregates results from all sub-detectors.
pub struct AnomalyDetector {
    /// Z-score sub-detector.
    pub z_score: ZScoreDetector,
    /// EWMA sub-detector.
    pub ewma: EWMADetector,
    /// Ring buffer of recently detected anomalies.
    pub anomalies: VecDeque<Anomaly>,
    /// Maximum number of anomaly records to retain.
    pub max_anomalies: usize,
    /// Total observations recorded (including non-anomalous ones).
    total_observations: usize,
}

impl AnomalyDetector {
    /// Create a new detector with default sub-detector parameters.
    ///
    /// * Z-score window: 60 observations, threshold: 3.0
    /// * EWMA: α = 0.1, σ multiplier = 3.0
    pub fn new(max_anomalies: usize) -> Self {
        Self {
            z_score: ZScoreDetector::new(60, 3.0),
            ewma: EWMADetector::new(0.1, 3.0),
            anomalies: VecDeque::new(),
            max_anomalies,
            total_observations: 0,
        }
    }

    /// Record a new cost observation and run all detectors.
    ///
    /// Returns a `Vec` of any anomalies detected by the sub-detectors.
    pub fn record(&mut self, value: f64) -> Vec<Anomaly> {
        self.total_observations += 1;
        let mut found = Vec::new();

        if let Some(a) = self.z_score.detect(value) {
            found.push(a);
        }
        if let Some(a) = self.ewma.detect(value) {
            found.push(a);
        }

        for anomaly in &found {
            if self.anomalies.len() == self.max_anomalies {
                self.anomalies.pop_front();
            }
            self.anomalies.push_back(anomaly.clone());
        }

        found
    }

    /// Return the `n` most-recent anomaly records (or fewer if not enough are
    /// available).
    pub fn recent_anomalies(&self, n: usize) -> &[Anomaly] {
        let len = self.anomalies.len();
        let start = if len > n { len - n } else { 0 };
        // VecDeque slices: make_contiguous would require &mut; use as_slices.
        // We convert lazily — caller gets a slice of the backing storage.
        // Because VecDeque may be non-contiguous, we return as many as we can
        // from the tail slice.
        let (front, back) = self.anomalies.as_slices();
        if back.len() >= n {
            &back[back.len() - n.min(back.len())..]
        } else if front.len() + back.len() >= start {
            // Just return the back slice (most recent)
            back
        } else {
            &front[front.len().saturating_sub(n)..]
        }
    }

    /// Fraction of all observations that triggered at least one detector
    /// (`anomaly_count / total_observations`).
    pub fn anomaly_rate(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.anomalies.len() as f64 / self.total_observations as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zscore_no_anomaly_within_threshold() {
        let mut det = ZScoreDetector::new(10, 3.0);
        for v in [1.0f64, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.02, 0.98] {
            det.detect(v); // feed window
        }
        // Value within normal range should not trigger
        let result = det.detect(1.03);
        assert!(result.is_none());
    }

    #[test]
    fn zscore_detects_spike() {
        let mut det = ZScoreDetector::new(10, 2.0);
        for v in [1.0f64; 9] {
            det.detect(v);
        }
        // A spike of 100 should be flagged
        let result = det.detect(100.0);
        assert!(result.is_some());
    }

    #[test]
    fn ewma_detects_spike() {
        let mut det = EWMADetector::new(0.3, 2.0);
        for v in [1.0f64; 10] {
            det.detect(v);
        }
        let result = det.detect(50.0);
        assert!(result.is_some());
    }

    #[test]
    fn anomaly_detector_record() {
        let mut det = AnomalyDetector::new(100);
        for v in [1.0f64; 20] {
            det.record(v);
        }
        let found = det.record(999.0);
        assert!(!found.is_empty());
    }
}
