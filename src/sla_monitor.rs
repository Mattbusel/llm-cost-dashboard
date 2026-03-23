//! SLA tracking and compliance monitoring.
//!
//! Tracks SLA targets (availability, latency percentiles, error rate, throughput),
//! records observations, and reports compliance status with per-metric reports.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::fmt;

// ── SlaMetricType ─────────────────────────────────────────────────────────────

/// The kind of metric an SLA target applies to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SlaMetricType {
    /// Service availability (proportion of successful requests).
    Availability,
    /// Median (50th percentile) latency in milliseconds.
    P50Latency,
    /// 95th percentile latency in milliseconds.
    P95Latency,
    /// 99th percentile latency in milliseconds.
    P99Latency,
    /// Error rate as a fraction (0.0 = no errors, 1.0 = all errors).
    ErrorRate,
    /// Requests per second.
    Throughput,
}

impl fmt::Display for SlaMetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SlaMetricType::Availability => "Availability",
            SlaMetricType::P50Latency => "P50Latency",
            SlaMetricType::P95Latency => "P95Latency",
            SlaMetricType::P99Latency => "P99Latency",
            SlaMetricType::ErrorRate => "ErrorRate",
            SlaMetricType::Throughput => "Throughput",
        };
        write!(f, "{s}")
    }
}

// ── SlaTarget ─────────────────────────────────────────────────────────────────

/// Definition of a single SLA requirement.
#[derive(Debug, Clone)]
pub struct SlaTarget {
    /// Which metric this target applies to.
    pub metric_type: SlaMetricType,
    /// The target threshold value (e.g. 99.9 for availability %, 200 for P99 ms).
    pub target_value: f64,
    /// Rolling window over which observations are evaluated (seconds).
    pub measurement_window_secs: u64,
    /// Percentage of time (0-100) the metric is allowed to breach the target.
    pub breach_threshold_pct: f64,
}

// ── SlaObservation ────────────────────────────────────────────────────────────

/// A single observed measurement.
#[derive(Debug, Clone)]
pub struct SlaObservation {
    /// Which metric was measured.
    pub metric_type: SlaMetricType,
    /// The measured value.
    pub value: f64,
    /// Unix timestamp of the observation.
    pub timestamp_unix: u64,
    /// Whether the underlying operation succeeded.
    pub success: bool,
}

// ── SlaStatus ─────────────────────────────────────────────────────────────────

/// Compliance status for an SLA target.
#[derive(Debug, Clone, PartialEq)]
pub enum SlaStatus {
    /// The target is being met.
    Compliant,
    /// The target is at risk of being breached.
    AtRisk {
        /// Current measured value.
        current_value: f64,
        /// The SLA target value.
        target: f64,
    },
    /// The target has been breached.
    Breached {
        /// Fraction of the window (0-100) during which the target was violated.
        violation_pct: f64,
    },
}

impl fmt::Display for SlaStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlaStatus::Compliant => write!(f, "Compliant"),
            SlaStatus::AtRisk { current_value, target } => {
                write!(f, "AtRisk(current={current_value:.2}, target={target:.2})")
            }
            SlaStatus::Breached { violation_pct } => {
                write!(f, "Breached(violation={violation_pct:.1}%)")
            }
        }
    }
}

// ── SlaReport ─────────────────────────────────────────────────────────────────

/// A full compliance report for one SLA target.
#[derive(Debug, Clone)]
pub struct SlaReport {
    /// The SLA target this report covers.
    pub target: SlaTarget,
    /// Current compliance status.
    pub status: SlaStatus,
    /// Most recently measured value for this metric.
    pub current_value: f64,
    /// Number of observations within the measurement window.
    pub observations_in_window: usize,
    /// Number of observations that breached the target.
    pub breach_count: usize,
    /// Proportion of successful observations (0-100).
    pub availability_pct: f64,
    /// 50th percentile of values in the window.
    pub p50: f64,
    /// 95th percentile of values in the window.
    pub p95: f64,
    /// 99th percentile of values in the window.
    pub p99: f64,
}

// ── SlaMonitor ────────────────────────────────────────────────────────────────

/// Tracks SLA targets and records observations for compliance evaluation.
pub struct SlaMonitor {
    targets: Vec<SlaTarget>,
    observations: Mutex<VecDeque<SlaObservation>>,
}

const MAX_OBSERVATIONS: usize = 10_000;

impl SlaMonitor {
    /// Create a new monitor with no targets.
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
            observations: Mutex::new(VecDeque::with_capacity(MAX_OBSERVATIONS)),
        }
    }

    /// Register an SLA target.
    pub fn add_target(&mut self, target: SlaTarget) {
        self.targets.push(target);
    }

    /// Record a new observation. Oldest observations are evicted when the
    /// buffer exceeds `MAX_OBSERVATIONS`.
    pub fn record(&self, metric_type: SlaMetricType, value: f64, success: bool) {
        let ts = current_unix_ts();
        let obs = SlaObservation { metric_type, value, timestamp_unix: ts, success };
        if let Ok(mut deq) = self.observations.lock() {
            if deq.len() >= MAX_OBSERVATIONS {
                deq.pop_front();
            }
            deq.push_back(obs);
        }
    }

    /// Compute the compliance status for a given target.
    pub fn compute_status(&self, target: &SlaTarget) -> SlaStatus {
        let window_obs = self.observations_in_window(target);
        if window_obs.is_empty() {
            return SlaStatus::Compliant;
        }

        let breach_count = self.count_breaches(target, &window_obs);
        let violation_pct = breach_count as f64 / window_obs.len() as f64 * 100.0;

        if violation_pct > target.breach_threshold_pct {
            SlaStatus::Breached { violation_pct }
        } else if violation_pct > target.breach_threshold_pct * 0.5 {
            let last = window_obs.last().map(|o| o.value).unwrap_or(0.0);
            SlaStatus::AtRisk { current_value: last, target: target.target_value }
        } else {
            SlaStatus::Compliant
        }
    }

    /// Generate a report for a specific metric type.
    pub fn report(&self, metric_type: SlaMetricType) -> Option<SlaReport> {
        let target = self.targets.iter().find(|t| t.metric_type == metric_type)?;
        let window_obs = self.observations_in_window(target);
        let status = self.compute_status(target);

        let current_value = window_obs.last().map(|o| o.value).unwrap_or(0.0);
        let breach_count = self.count_breaches(target, &window_obs);
        let observations_in_window = window_obs.len();

        let success_count = window_obs.iter().filter(|o| o.success).count();
        let availability_pct = if observations_in_window > 0 {
            success_count as f64 / observations_in_window as f64 * 100.0
        } else {
            100.0
        };

        let mut values: Vec<f64> = window_obs.iter().map(|o| o.value).collect();

        let p50 = Self::percentile(&mut values.clone(), 50.0);
        let p95 = Self::percentile(&mut values.clone(), 95.0);
        let p99 = Self::percentile(&mut values, 99.0);

        Some(SlaReport {
            target: target.clone(),
            status,
            current_value,
            observations_in_window,
            breach_count,
            availability_pct,
            p50,
            p95,
            p99,
        })
    }

    /// Generate reports for all registered targets.
    pub fn all_reports(&self) -> Vec<SlaReport> {
        let types: Vec<SlaMetricType> = self.targets.iter().map(|t| t.metric_type.clone()).collect();
        types.into_iter().filter_map(|m| self.report(m)).collect()
    }

    /// Whether any target is currently in `Breached` status.
    pub fn is_any_breached(&self) -> bool {
        self.targets.iter().any(|t| {
            matches!(self.compute_status(t), SlaStatus::Breached { .. })
        })
    }

    /// Fraction (0-100) of successful observations within the given window.
    pub fn uptime_pct(&self, window_secs: u64) -> f64 {
        let now = current_unix_ts();
        if let Ok(deq) = self.observations.lock() {
            let in_window: Vec<&SlaObservation> = deq.iter()
                .filter(|o| o.timestamp_unix >= now.saturating_sub(window_secs))
                .collect();
            if in_window.is_empty() {
                return 100.0;
            }
            let successes = in_window.iter().filter(|o| o.success).count();
            successes as f64 / in_window.len() as f64 * 100.0
        } else {
            100.0
        }
    }

    /// Compute the `p`-th percentile of `values` (sorted in place).
    pub fn percentile(values: &mut Vec<f64>, p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (values.len() - 1) as f64).round() as usize;
        values[idx.min(values.len() - 1)]
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn observations_in_window(&self, target: &SlaTarget) -> Vec<SlaObservation> {
        let now = current_unix_ts();
        let cutoff = now.saturating_sub(target.measurement_window_secs);
        if let Ok(deq) = self.observations.lock() {
            deq.iter()
                .filter(|o| o.metric_type == target.metric_type && o.timestamp_unix >= cutoff)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    fn count_breaches(&self, target: &SlaTarget, observations: &[SlaObservation]) -> usize {
        observations.iter().filter(|o| is_breach(target, o.value)).count()
    }
}

impl Default for SlaMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine whether an observed `value` constitutes a breach of `target`.
fn is_breach(target: &SlaTarget, value: f64) -> bool {
    match target.metric_type {
        // For availability and throughput, breaching = being *below* target.
        SlaMetricType::Availability | SlaMetricType::Throughput => value < target.target_value,
        // For latency metrics and error rate, breaching = being *above* target.
        _ => value > target.target_value,
    }
}

/// Return the current Unix timestamp in seconds.
fn current_unix_ts() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> SlaMonitor {
        let mut m = SlaMonitor::new();
        m.add_target(SlaTarget {
            metric_type: SlaMetricType::Availability,
            target_value: 99.0,
            measurement_window_secs: 3600,
            breach_threshold_pct: 1.0,
        });
        m.add_target(SlaTarget {
            metric_type: SlaMetricType::P99Latency,
            target_value: 500.0,
            measurement_window_secs: 3600,
            breach_threshold_pct: 5.0,
        });
        m
    }

    #[test]
    fn test_percentile() {
        let mut v = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(SlaMonitor::percentile(&mut v, 50.0), 30.0);
        assert_eq!(SlaMonitor::percentile(&mut v, 0.0), 10.0);
        assert_eq!(SlaMonitor::percentile(&mut v, 100.0), 50.0);
    }

    #[test]
    fn test_empty_observations_compliant() {
        let monitor = make_monitor();
        let status = monitor.compute_status(&monitor.targets[0]);
        assert_eq!(status, SlaStatus::Compliant);
    }

    #[test]
    fn test_record_and_report() {
        let monitor = make_monitor();
        for _ in 0..10 {
            monitor.record(SlaMetricType::P99Latency, 200.0, true);
        }
        let report = monitor.report(SlaMetricType::P99Latency).unwrap();
        assert_eq!(report.observations_in_window, 10);
        assert_eq!(report.breach_count, 0);
        assert!(matches!(report.status, SlaStatus::Compliant));
    }

    #[test]
    fn test_breach_detection() {
        let monitor = make_monitor();
        // Record many high-latency observations — should trigger a breach.
        for _ in 0..100 {
            monitor.record(SlaMetricType::P99Latency, 1000.0, true);
        }
        let report = monitor.report(SlaMetricType::P99Latency).unwrap();
        assert!(matches!(report.status, SlaStatus::Breached { .. }));
    }

    #[test]
    fn test_uptime_pct_all_success() {
        let monitor = make_monitor();
        for _ in 0..5 {
            monitor.record(SlaMetricType::Availability, 100.0, true);
        }
        let uptime = monitor.uptime_pct(3600);
        assert!((uptime - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_is_any_breached_false_when_empty() {
        let monitor = make_monitor();
        assert!(!monitor.is_any_breached());
    }

    #[test]
    fn test_all_reports() {
        let monitor = make_monitor();
        monitor.record(SlaMetricType::Availability, 100.0, true);
        monitor.record(SlaMetricType::P99Latency, 100.0, true);
        let reports = monitor.all_reports();
        assert_eq!(reports.len(), 2);
    }
}
