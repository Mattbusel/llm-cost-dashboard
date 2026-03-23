//! # Budget Controller
//!
//! Hard and soft budget limits with spending-pace analysis.
//!
//! Tracks real-time spend against configured limits and can block requests
//! before they push an entity over its hard cap.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// BudgetPeriod
// ---------------------------------------------------------------------------

/// The calendar window over which a budget is measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetPeriod {
    /// Budget resets every day.
    Daily,
    /// Budget resets every week.
    Weekly,
    /// Budget resets every calendar month.
    Monthly,
    /// Budget resets every year.
    Yearly,
}

impl BudgetPeriod {
    /// Approximate number of seconds in this period (used for pace calculations).
    pub fn approximate_secs(&self) -> f64 {
        match self {
            BudgetPeriod::Daily => 86_400.0,
            BudgetPeriod::Weekly => 604_800.0,
            BudgetPeriod::Monthly => 2_592_000.0,   // 30 days
            BudgetPeriod::Yearly => 31_536_000.0,   // 365 days
        }
    }
}

// ---------------------------------------------------------------------------
// BudgetLimit
// ---------------------------------------------------------------------------

/// Combined soft + hard limit configuration for one entity/period.
#[derive(Debug, Clone)]
pub struct BudgetLimit {
    /// Issue a soft warning when spending exceeds this amount.
    pub soft_usd: f64,
    /// Block all further spending once this amount is reached.
    pub hard_usd: f64,
    /// The time window this budget covers.
    pub period: BudgetPeriod,
    /// If true, unspent budget from the previous period carries over.
    pub rollover: bool,
}

impl BudgetLimit {
    /// Create a new limit.
    pub fn new(soft_usd: f64, hard_usd: f64, period: BudgetPeriod) -> Self {
        Self { soft_usd, hard_usd, period, rollover: false }
    }

    /// Enable budget rollover.
    pub fn with_rollover(mut self) -> Self {
        self.rollover = true;
        self
    }
}

// ---------------------------------------------------------------------------
// SpendingPace
// ---------------------------------------------------------------------------

/// Tracks the rate at which an entity is spending relative to its period budget.
pub struct SpendingPace {
    /// How much has been spent so far this period.
    pub current_spend: f64,
    /// When the current period started.
    pub period_start: std::time::Instant,
    /// Total budget for this period.
    pub period_budget: f64,
}

impl SpendingPace {
    /// Create a new pace tracker.
    pub fn new(period_budget: f64) -> Self {
        Self {
            current_spend: 0.0,
            period_start: std::time::Instant::now(),
            period_budget,
        }
    }

    /// Fraction of elapsed time in this period (0.0–1.0).
    fn time_fraction_elapsed(&self, period_secs: f64) -> f64 {
        let elapsed = self.period_start.elapsed().as_secs_f64();
        (elapsed / period_secs).min(1.0).max(f64::EPSILON)
    }

    /// `(current_spend / period_budget) / time_fraction_elapsed`.
    ///
    /// Values > 1.0 mean spending faster than budget allows.
    pub fn pace_ratio(&self, period_secs: f64) -> f64 {
        if self.period_budget <= 0.0 {
            return 0.0;
        }
        let spend_fraction = self.current_spend / self.period_budget;
        spend_fraction / self.time_fraction_elapsed(period_secs)
    }

    /// Extrapolated total spend for the full period at the current run rate.
    pub fn projected_total(&self, period_secs: f64) -> f64 {
        let tf = self.time_fraction_elapsed(period_secs);
        self.current_spend / tf
    }

    /// Returns `true` when the pace ratio exceeds 1.2 (20% over budget pace).
    pub fn is_overpacing(&self, period_secs: f64) -> bool {
        self.pace_ratio(period_secs) > 1.2
    }
}

// ---------------------------------------------------------------------------
// BudgetDecision
// ---------------------------------------------------------------------------

/// The outcome of a `check_and_record` call.
#[derive(Debug)]
pub enum BudgetDecision {
    /// Request is within budget.
    Allow {
        /// USD remaining before the soft threshold.
        remaining_usd: f64,
    },
    /// Soft threshold crossed — warn but still allow.
    SoftWarn {
        /// Human-readable warning message.
        message: String,
        /// USD remaining before the hard cap.
        remaining_usd: f64,
    },
    /// Hard cap reached — request blocked.
    HardBlock {
        /// Reason for blocking.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// BudgetController
// ---------------------------------------------------------------------------

/// Central spend tracker and gate-keeper for multiple entities.
pub struct BudgetController {
    limits: HashMap<String, BudgetLimit>,
    spending: RwLock<HashMap<String, f64>>,
    pace_trackers: RwLock<HashMap<String, SpendingPace>>,
    /// Running count of requests that were hard-blocked.
    pub total_blocked: AtomicU64,
}

impl BudgetController {
    /// Create an empty controller.
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
            spending: RwLock::new(HashMap::new()),
            pace_trackers: RwLock::new(HashMap::new()),
            total_blocked: AtomicU64::new(0),
        }
    }

    /// Register or update a budget limit for `entity`.
    pub fn set_limit(&mut self, entity: &str, limit: BudgetLimit) {
        let budget = limit.hard_usd;
        let period = limit.period.clone();
        self.limits.insert(entity.to_string(), limit);
        // Initialise a pace tracker if none exists.
        if let Ok(mut trackers) = self.pace_trackers.write() {
            trackers.entry(entity.to_string()).or_insert_with(|| {
                SpendingPace::new(budget)
            });
        }
        let _ = period; // used indirectly via limits
    }

    /// Check whether `estimated_cost` is within budget for `entity` and record
    /// it against current spend.  Returns the appropriate [`BudgetDecision`].
    pub fn check_and_record(&self, entity: &str, estimated_cost: f64) -> BudgetDecision {
        let limit = match self.limits.get(entity) {
            Some(l) => l,
            None => {
                // No limit configured → allow.
                return BudgetDecision::Allow { remaining_usd: f64::INFINITY };
            }
        };

        let current = {
            let spending = self.spending.read().unwrap_or_else(|e| e.into_inner());
            *spending.get(entity).unwrap_or(&0.0)
        };

        let projected = current + estimated_cost;

        if projected >= limit.hard_usd {
            self.total_blocked.fetch_add(1, Ordering::Relaxed);
            return BudgetDecision::HardBlock {
                reason: format!(
                    "Hard budget of ${:.4} exceeded for entity '{}' (current: ${:.4}, estimated: ${:.4})",
                    limit.hard_usd, entity, current, estimated_cost
                ),
            };
        }

        // Record the estimated cost.
        if let Ok(mut spending) = self.spending.write() {
            let entry = spending.entry(entity.to_string()).or_insert(0.0);
            *entry += estimated_cost;
        }

        // Update pace tracker.
        if let Ok(mut trackers) = self.pace_trackers.write() {
            if let Some(tracker) = trackers.get_mut(entity) {
                tracker.current_spend += estimated_cost;
            }
        }

        if projected >= limit.soft_usd {
            let remaining = limit.hard_usd - projected;
            BudgetDecision::SoftWarn {
                message: format!(
                    "Soft budget threshold (${:.4}) reached for '{}'. ${:.4} remaining before hard cap.",
                    limit.soft_usd, entity, remaining.max(0.0)
                ),
                remaining_usd: remaining.max(0.0),
            }
        } else {
            BudgetDecision::Allow {
                remaining_usd: limit.soft_usd - projected,
            }
        }
    }

    /// Adjust spend after a request completes with the real cost.
    ///
    /// Call this when the actual cost differs from the estimate passed to
    /// `check_and_record`.
    pub fn record_actual(&self, entity: &str, actual_cost: f64) {
        if let Ok(mut spending) = self.spending.write() {
            let entry = spending.entry(entity.to_string()).or_insert(0.0);
            *entry += actual_cost;
        }
        if let Ok(mut trackers) = self.pace_trackers.write() {
            if let Some(tracker) = trackers.get_mut(entity) {
                tracker.current_spend += actual_cost;
            }
        }
    }

    /// Reset the period spend for `entity` (call at the start of each new
    /// budget period).
    pub fn reset_period(&self, entity: &str) {
        if let Ok(mut spending) = self.spending.write() {
            spending.insert(entity.to_string(), 0.0);
        }
        // Reset the pace tracker.
        if let Ok(mut trackers) = self.pace_trackers.write() {
            if let Some(limit) = self.limits.get(entity) {
                trackers.insert(entity.to_string(), SpendingPace::new(limit.hard_usd));
            }
        }
    }

    /// Returns `(spent, soft_limit, hard_limit)` for `entity`, or `None` if
    /// no limit has been registered.
    pub fn spending_summary(&self, entity: &str) -> Option<(f64, f64, f64)> {
        let limit = self.limits.get(entity)?;
        let spent = self
            .spending
            .read()
            .map(|m| *m.get(entity).unwrap_or(&0.0))
            .unwrap_or(0.0);
        Some((spent, limit.soft_usd, limit.hard_usd))
    }

    /// Returns entities whose spending pace is currently over 120% of the
    /// expected rate.
    pub fn overpacing_entities(&self) -> Vec<String> {
        let trackers = match self.pace_trackers.read() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        trackers
            .iter()
            .filter_map(|(entity, tracker)| {
                let period_secs = self
                    .limits
                    .get(entity)
                    .map(|l| l.period.approximate_secs())
                    .unwrap_or(86_400.0);
                if tracker.is_overpacing(period_secs) {
                    Some(entity.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for BudgetController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn controller_with_limit() -> BudgetController {
        let mut c = BudgetController::new();
        c.set_limit("alice", BudgetLimit::new(5.0, 10.0, BudgetPeriod::Daily));
        c
    }

    #[test]
    fn allow_within_budget() {
        let c = controller_with_limit();
        let dec = c.check_and_record("alice", 2.0);
        assert!(matches!(dec, BudgetDecision::Allow { .. }));
    }

    #[test]
    fn soft_warn_above_soft_threshold() {
        let c = controller_with_limit();
        c.check_and_record("alice", 4.0);
        let dec = c.check_and_record("alice", 1.5);
        assert!(matches!(dec, BudgetDecision::SoftWarn { .. }));
    }

    #[test]
    fn hard_block_at_cap() {
        let c = controller_with_limit();
        c.check_and_record("alice", 9.5);
        let dec = c.check_and_record("alice", 1.0);
        assert!(matches!(dec, BudgetDecision::HardBlock { .. }));
        assert_eq!(c.total_blocked.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn spending_summary_returns_correct_values() {
        let c = controller_with_limit();
        c.check_and_record("alice", 3.0);
        let (spent, soft, hard) = c.spending_summary("alice").unwrap();
        assert!((spent - 3.0).abs() < 0.001);
        assert!((soft - 5.0).abs() < 0.001);
        assert!((hard - 10.0).abs() < 0.001);
    }

    #[test]
    fn reset_period_clears_spend() {
        let c = controller_with_limit();
        c.check_and_record("alice", 8.0);
        c.reset_period("alice");
        let (spent, _, _) = c.spending_summary("alice").unwrap();
        assert!(spent < 0.001);
    }

    #[test]
    fn no_limit_returns_allow() {
        let c = BudgetController::new();
        let dec = c.check_and_record("unknown_entity", 999.0);
        assert!(matches!(dec, BudgetDecision::Allow { .. }));
    }

    #[test]
    fn spending_pace_overpacing() {
        let mut pace = SpendingPace::new(100.0);
        // Simulate spending 90% of budget in essentially 0 elapsed time → overpacing.
        pace.current_spend = 90.0;
        // elapsed ≈ 0 → time_fraction ≈ EPSILON → pace_ratio >> 1.2
        assert!(pace.is_overpacing(86_400.0));
    }
}
