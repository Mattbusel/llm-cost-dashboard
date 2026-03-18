//! # Budget Envelope
//!
//! Hard budget enforcement with spend tracking, alert thresholds, and
//! remaining-balance queries.
//!
//! A [`BudgetEnvelope`] tracks cumulative spend against a hard limit and an
//! optional soft alert threshold.  It does not perform any I/O; the caller is
//! responsible for persisting state across restarts.

use crate::error::DashboardError;

/// A budget envelope with a hard limit and an optional soft alert threshold.
///
/// # Example
///
/// ```
/// use llm_cost_dashboard::budget::BudgetEnvelope;
///
/// let mut budget = BudgetEnvelope::new("Monthly", 10.0, 0.8);
/// budget.spend(7.5).unwrap();
/// assert_eq!(budget.status(), "OK");
/// budget.spend(1.0).unwrap(); // now at 85% -> WARNING
/// assert_eq!(budget.status(), "WARNING");
/// ```
#[derive(Debug, Clone)]
pub struct BudgetEnvelope {
    /// Hard limit in USD.
    pub limit_usd: f64,
    /// Amount spent so far in USD.
    pub spent_usd: f64,
    /// Fraction of the limit at which the soft alert fires (0.0-1.0).
    ///
    /// Values outside `[0.0, 1.0]` are clamped on construction.
    pub alert_threshold: f64,
    /// Human-readable label for display in the UI.
    pub label: String,
}

impl BudgetEnvelope {
    /// Construct a new envelope.
    ///
    /// `alert_threshold` is clamped to `[0.0, 1.0]`.
    pub fn new(label: impl Into<String>, limit_usd: f64, alert_threshold: f64) -> Self {
        Self {
            limit_usd,
            spent_usd: 0.0,
            alert_threshold: alert_threshold.clamp(0.0, 1.0),
            label: label.into(),
        }
    }

    /// Record a spend of `amount_usd`.
    ///
    /// Returns [`DashboardError::BudgetExceeded`] if the hard limit is
    /// breached after adding `amount_usd`, or [`DashboardError::Ledger`] if
    /// `amount_usd` is negative.
    pub fn spend(&mut self, amount_usd: f64) -> Result<(), DashboardError> {
        if amount_usd < 0.0 {
            return Err(DashboardError::Ledger("negative spend amount".into()));
        }
        self.spent_usd += amount_usd;
        if self.spent_usd > self.limit_usd {
            return Err(DashboardError::BudgetExceeded {
                spent: self.spent_usd,
                limit: self.limit_usd,
            });
        }
        Ok(())
    }

    /// Remaining budget in USD.
    ///
    /// May be negative if the hard limit has been exceeded.
    pub fn remaining(&self) -> f64 {
        self.limit_usd - self.spent_usd
    }

    /// Fraction of the limit consumed (0.0-1.0+).
    ///
    /// Returns `1.0` when `limit_usd <= 0.0` to avoid division by zero.
    pub fn pct_consumed(&self) -> f64 {
        if self.limit_usd <= 0.0 {
            return 1.0;
        }
        self.spent_usd / self.limit_usd
    }

    /// Whether the hard limit has been exceeded (`spent > limit`).
    pub fn is_over_budget(&self) -> bool {
        self.spent_usd > self.limit_usd
    }

    /// Whether the soft alert threshold has been crossed.
    ///
    /// Returns `true` when `pct_consumed() >= alert_threshold`.
    pub fn alert_triggered(&self) -> bool {
        self.pct_consumed() >= self.alert_threshold
    }

    /// Reset spend to zero (start of a new billing period).
    pub fn reset(&mut self) {
        self.spent_usd = 0.0;
    }

    /// Gauge percentage (0-100), clamped, suitable for use with ratatui `Gauge`.
    pub fn gauge_pct(&self) -> u16 {
        (self.pct_consumed() * 100.0).clamp(0.0, 100.0) as u16
    }

    /// Traffic-light status label.
    ///
    /// Returns `"OVER BUDGET"`, `"WARNING"`, or `"OK"`.
    pub fn status(&self) -> &'static str {
        if self.pct_consumed() >= 1.0 {
            "OVER BUDGET"
        } else if self.alert_triggered() {
            "WARNING"
        } else {
            "OK"
        }
    }

    /// Projected monthly spend in USD, given current spend and elapsed hours.
    ///
    /// Returns `0.0` when `elapsed_hours` is zero or negative.
    pub fn projected_monthly(&self, elapsed_hours: f64) -> f64 {
        if elapsed_hours <= 0.0 {
            return 0.0;
        }
        (self.spent_usd / elapsed_hours) * 24.0 * 30.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_remaining_equals_limit() {
        let b = BudgetEnvelope::new("test", 10.0, 0.8);
        assert!((b.remaining() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spend_decreases_remaining() {
        let mut b = BudgetEnvelope::new("test", 10.0, 0.8);
        b.spend(3.0).unwrap();
        assert!((b.remaining() - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_over_budget_returns_error() {
        let mut b = BudgetEnvelope::new("test", 5.0, 0.8);
        assert!(b.spend(6.0).is_err());
    }

    #[test]
    fn test_over_budget_error_variant() {
        let mut b = BudgetEnvelope::new("test", 5.0, 0.8);
        let err = b.spend(6.0).unwrap_err();
        assert!(matches!(err, DashboardError::BudgetExceeded { .. }));
    }

    #[test]
    fn test_over_budget_flag() {
        let mut b = BudgetEnvelope::new("test", 5.0, 0.8);
        let _ = b.spend(6.0);
        assert!(b.is_over_budget());
    }

    #[test]
    fn test_alert_not_triggered_below_threshold() {
        let b = BudgetEnvelope::new("test", 10.0, 0.8);
        assert!(!b.alert_triggered());
    }

    #[test]
    fn test_alert_triggered_at_threshold() {
        let mut b = BudgetEnvelope::new("test", 10.0, 0.8);
        b.spend(8.0).unwrap();
        assert!(b.alert_triggered());
    }

    #[test]
    fn test_alert_triggered_above_threshold() {
        let mut b = BudgetEnvelope::new("test", 10.0, 0.8);
        b.spend(9.5).unwrap();
        assert!(b.alert_triggered());
    }

    #[test]
    fn test_alert_not_triggered_just_below() {
        let mut b = BudgetEnvelope::new("test", 10.0, 0.8);
        b.spend(7.9).unwrap();
        assert!(!b.alert_triggered());
    }

    #[test]
    fn test_reset_clears_spent() {
        let mut b = BudgetEnvelope::new("test", 10.0, 0.8);
        b.spend(5.0).unwrap();
        b.reset();
        assert_eq!(b.spent_usd, 0.0);
        assert!(!b.is_over_budget());
    }

    #[test]
    fn test_pct_consumed_zero_limit_returns_one() {
        let b = BudgetEnvelope::new("test", 0.0, 0.8);
        assert_eq!(b.pct_consumed(), 1.0);
    }

    #[test]
    fn test_gauge_pct_clamped_to_100() {
        let mut b = BudgetEnvelope::new("test", 5.0, 0.8);
        let _ = b.spend(10.0);
        assert_eq!(b.gauge_pct(), 100);
    }

    #[test]
    fn test_gauge_pct_at_50_percent() {
        let mut b = BudgetEnvelope::new("test", 100.0, 0.8);
        b.spend(50.0).unwrap();
        assert_eq!(b.gauge_pct(), 50);
    }

    #[test]
    fn test_status_ok() {
        let b = BudgetEnvelope::new("test", 100.0, 0.8);
        assert_eq!(b.status(), "OK");
    }

    #[test]
    fn test_status_warning() {
        let mut b = BudgetEnvelope::new("test", 100.0, 0.8);
        b.spend(85.0).unwrap();
        assert_eq!(b.status(), "WARNING");
    }

    #[test]
    fn test_status_over_budget() {
        let mut b = BudgetEnvelope::new("test", 100.0, 0.8);
        let _ = b.spend(110.0);
        assert_eq!(b.status(), "OVER BUDGET");
    }

    #[test]
    fn test_negative_spend_returns_error() {
        let mut b = BudgetEnvelope::new("test", 100.0, 0.8);
        assert!(b.spend(-1.0).is_err());
    }

    #[test]
    fn test_projected_monthly_zero_hours_returns_zero() {
        let b = BudgetEnvelope::new("test", 100.0, 0.8);
        assert_eq!(b.projected_monthly(0.0), 0.0);
    }

    #[test]
    fn test_projected_monthly_math() {
        // Spent $1 in 1 hour => $1/hr * 24 * 30 = $720/month
        let mut b = BudgetEnvelope::new("test", 1000.0, 0.8);
        b.spend(1.0).unwrap();
        let proj = b.projected_monthly(1.0);
        assert!((proj - 720.0).abs() < 1e-6);
    }

    #[test]
    fn test_alert_threshold_clamped_above_one() {
        let b = BudgetEnvelope::new("test", 10.0, 1.5);
        assert_eq!(b.alert_threshold, 1.0);
    }

    #[test]
    fn test_alert_threshold_clamped_below_zero() {
        let b = BudgetEnvelope::new("test", 10.0, -0.5);
        assert_eq!(b.alert_threshold, 0.0);
    }
}
