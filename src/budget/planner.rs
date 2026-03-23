//! # Budget Planner
//!
//! Period-aware budget planning with allocation splits, reconciliation against
//! actuals, and spend forecasting.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::budget::planner::{BudgetPeriod, BudgetPlanner};
//! use std::collections::HashMap;
//!
//! let mut plan = BudgetPlanner::create(
//!     BudgetPeriod::Monthly,
//!     1000.0,
//!     vec![
//!         ("GPT-4o".to_string(), 60.0),
//!         ("Claude".to_string(), 40.0),
//!     ],
//! );
//!
//! let mut actuals = HashMap::new();
//! actuals.insert("GPT-4o".to_string(), 550.0);
//! actuals.insert("Claude".to_string(), 200.0);
//! BudgetPlanner::reconcile(&mut plan, &actuals);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Period ─────────────────────────────────────────────────────────────────────

/// The time period covered by a [`BudgetPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BudgetPeriod {
    /// One calendar day.
    Daily,
    /// Seven calendar days.
    Weekly,
    /// One calendar month (~30 days).
    Monthly,
    /// Three calendar months (~90 days).
    Quarterly,
}

impl BudgetPeriod {
    /// Return a human-readable label for the period.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Daily => "Daily",
            Self::Weekly => "Weekly",
            Self::Monthly => "Monthly",
            Self::Quarterly => "Quarterly",
        }
    }
}

// ── Allocation status ─────────────────────────────────────────────────────────

/// Spend status of a single [`BudgetAllocation`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AllocationStatus {
    /// Spend is within acceptable range (< 80 % of budget by default).
    OnTrack,
    /// Spend is elevated; `pct_used` is the fraction consumed (0.0–1.0+).
    AtRisk {
        /// Fraction of the allocation budget already consumed.
        pct_used: f64,
    },
    /// The allocation is over budget.
    Exceeded {
        /// Amount by which the actual spend exceeds the budgeted amount.
        overage_usd: f64,
    },
}

impl AllocationStatus {
    /// Derive a status from `actual_usd` and `budgeted_usd`.
    ///
    /// Thresholds:
    /// - `actual < 80 %` of budget → [`OnTrack`](Self::OnTrack)
    /// - `80 % ≤ actual < 100 %` → [`AtRisk`](Self::AtRisk)
    /// - `actual ≥ 100 %` → [`Exceeded`](Self::Exceeded)
    pub fn from_spend(actual_usd: f64, budgeted_usd: f64) -> Self {
        if budgeted_usd <= 0.0 {
            if actual_usd > 0.0 {
                return Self::Exceeded { overage_usd: actual_usd };
            }
            return Self::OnTrack;
        }
        let pct = actual_usd / budgeted_usd;
        if actual_usd >= budgeted_usd {
            Self::Exceeded { overage_usd: actual_usd - budgeted_usd }
        } else if pct >= 0.80 {
            Self::AtRisk { pct_used: pct }
        } else {
            Self::OnTrack
        }
    }
}

// ── Allocation ────────────────────────────────────────────────────────────────

/// A single named allocation within a [`BudgetPlan`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAllocation {
    /// Human-readable allocation name (e.g. `"GPT-4o"`, `"Claude 3.5"`).
    pub name: String,
    /// Percentage of the total budget assigned to this allocation (0.0–100.0).
    pub pct: f64,
    /// Dollar amount derived from `total_usd * pct / 100`.
    pub usd: f64,
    /// Actual spend recorded during reconciliation.
    pub actual_usd: f64,
    /// `actual_usd - usd` (negative means under-budget).
    pub variance_usd: f64,
    /// Spend status derived during reconciliation.
    pub status: AllocationStatus,
}

// ── Plan ──────────────────────────────────────────────────────────────────────

/// A complete budget plan for a given period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetPlan {
    /// Time granularity this plan covers.
    pub period: BudgetPeriod,
    /// Total budget in US dollars.
    pub total_usd: f64,
    /// Per-allocation breakdowns (mutated in-place by [`BudgetPlanner::reconcile`]).
    pub allocations: Vec<BudgetAllocation>,
}

// ── Forecast ──────────────────────────────────────────────────────────────────

/// End-of-period spend projection computed by [`BudgetPlanner::forecast`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetForecast {
    /// Projected total spend at end of period (linear extrapolation).
    pub projected_total_usd: f64,
    /// `projected_total_usd - plan.total_usd` (positive = over budget).
    pub projected_variance_usd: f64,
    /// `true` if the projected total does not exceed the plan budget.
    pub on_track: bool,
    /// Names of allocations whose projected spend exceeds their budget.
    pub at_risk_allocations: Vec<String>,
}

// ── Planner ───────────────────────────────────────────────────────────────────

/// Stateless helper that creates, reconciles, and forecasts [`BudgetPlan`]s.
pub struct BudgetPlanner;

impl BudgetPlanner {
    /// Construct a [`BudgetPlan`] by splitting `total_usd` according to the
    /// provided percentage allocations.
    ///
    /// `allocations` is a list of `(name, pct)` pairs.  Percentages need not
    /// sum to 100; the USD amounts are computed independently as
    /// `total_usd * pct / 100`.
    ///
    /// All `actual_usd`, `variance_usd` fields are initialised to `0.0` and
    /// `status` is set to [`AllocationStatus::OnTrack`].
    pub fn create(
        period: BudgetPeriod,
        total_usd: f64,
        allocations: Vec<(String, f64)>,
    ) -> BudgetPlan {
        let allocs = allocations
            .into_iter()
            .map(|(name, pct)| BudgetAllocation {
                usd: total_usd * pct / 100.0,
                name,
                pct,
                actual_usd: 0.0,
                variance_usd: 0.0,
                status: AllocationStatus::OnTrack,
            })
            .collect();

        BudgetPlan { period, total_usd, allocations: allocs }
    }

    /// Reconcile `plan` against `actuals` (a map of allocation name → actual USD).
    ///
    /// For each allocation whose name appears in `actuals`:
    /// - Sets `actual_usd`
    /// - Computes `variance_usd = actual_usd - usd`
    /// - Derives `status` via [`AllocationStatus::from_spend`]
    ///
    /// Allocations not present in `actuals` retain their previous values.
    pub fn reconcile(plan: &mut BudgetPlan, actuals: &HashMap<String, f64>) {
        for alloc in &mut plan.allocations {
            if let Some(&actual) = actuals.get(&alloc.name) {
                alloc.actual_usd = actual;
                alloc.variance_usd = actual - alloc.usd;
                alloc.status = AllocationStatus::from_spend(actual, alloc.usd);
            }
        }
    }

    /// Produce a linear end-of-period spend forecast.
    ///
    /// `elapsed_fraction` is the fraction of the period that has elapsed
    /// (0.0 = period start, 1.0 = period end).  Values outside `[0.0, 1.0]`
    /// are clamped to avoid nonsensical projections.
    ///
    /// The projection simply divides each allocation's `actual_usd` by
    /// `elapsed_fraction` to get the expected end-of-period spend.
    pub fn forecast(plan: &BudgetPlan, elapsed_fraction: f64) -> BudgetForecast {
        // Clamp to [ε, 1.0] so we never divide by zero and don't extrapolate
        // backwards in time.
        let frac = elapsed_fraction.clamp(1e-9, 1.0);

        let mut projected_total = 0.0_f64;
        let mut at_risk: Vec<String> = Vec::new();

        for alloc in &plan.allocations {
            let projected_alloc = alloc.actual_usd / frac;
            projected_total += projected_alloc;
            if projected_alloc > alloc.usd {
                at_risk.push(alloc.name.clone());
            }
        }

        let variance = projected_total - plan.total_usd;
        BudgetForecast {
            projected_total_usd: projected_total,
            projected_variance_usd: variance,
            on_track: projected_total <= plan.total_usd,
            at_risk_allocations: at_risk,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_plan() -> BudgetPlan {
        BudgetPlanner::create(
            BudgetPeriod::Monthly,
            1000.0,
            vec![
                ("GPT-4o".to_string(), 60.0),
                ("Claude".to_string(), 40.0),
            ],
        )
    }

    // ── BudgetPeriod ──────────────────────────────────────────────────────────

    #[test]
    fn budget_period_labels() {
        assert_eq!(BudgetPeriod::Daily.label(), "Daily");
        assert_eq!(BudgetPeriod::Weekly.label(), "Weekly");
        assert_eq!(BudgetPeriod::Monthly.label(), "Monthly");
        assert_eq!(BudgetPeriod::Quarterly.label(), "Quarterly");
    }

    // ── AllocationStatus ──────────────────────────────────────────────────────

    #[test]
    fn status_on_track_when_under_80_pct() {
        assert_eq!(AllocationStatus::from_spend(79.0, 100.0), AllocationStatus::OnTrack);
    }

    #[test]
    fn status_at_risk_between_80_and_100_pct() {
        let s = AllocationStatus::from_spend(85.0, 100.0);
        assert!(matches!(s, AllocationStatus::AtRisk { pct_used } if (pct_used - 0.85).abs() < 1e-9));
    }

    #[test]
    fn status_exceeded_when_over_budget() {
        let s = AllocationStatus::from_spend(120.0, 100.0);
        assert!(matches!(s, AllocationStatus::Exceeded { overage_usd } if (overage_usd - 20.0).abs() < 1e-9));
    }

    #[test]
    fn status_exactly_at_budget_is_exceeded() {
        let s = AllocationStatus::from_spend(100.0, 100.0);
        assert!(matches!(s, AllocationStatus::Exceeded { overage_usd } if overage_usd.abs() < 1e-9));
    }

    #[test]
    fn status_zero_budget_with_spend_is_exceeded() {
        let s = AllocationStatus::from_spend(10.0, 0.0);
        assert!(matches!(s, AllocationStatus::Exceeded { .. }));
    }

    // ── BudgetPlanner::create ─────────────────────────────────────────────────

    #[test]
    fn create_splits_budget_correctly() {
        let plan = simple_plan();
        assert_eq!(plan.total_usd, 1000.0);
        assert_eq!(plan.allocations.len(), 2);
        let gpt = &plan.allocations[0];
        assert_eq!(gpt.name, "GPT-4o");
        assert!((gpt.usd - 600.0).abs() < 1e-9);
        assert!((gpt.pct - 60.0).abs() < 1e-9);
        let claude = &plan.allocations[1];
        assert!((claude.usd - 400.0).abs() < 1e-9);
    }

    #[test]
    fn create_initialises_actuals_to_zero() {
        let plan = simple_plan();
        for alloc in &plan.allocations {
            assert_eq!(alloc.actual_usd, 0.0);
            assert_eq!(alloc.variance_usd, 0.0);
            assert_eq!(alloc.status, AllocationStatus::OnTrack);
        }
    }

    #[test]
    fn create_empty_allocations() {
        let plan = BudgetPlanner::create(BudgetPeriod::Daily, 500.0, vec![]);
        assert!(plan.allocations.is_empty());
    }

    #[test]
    fn create_period_stored_correctly() {
        let plan = BudgetPlanner::create(BudgetPeriod::Quarterly, 10_000.0, vec![]);
        assert_eq!(plan.period, BudgetPeriod::Quarterly);
    }

    // ── BudgetPlanner::reconcile ──────────────────────────────────────────────

    #[test]
    fn reconcile_fills_actual_and_variance() {
        let mut plan = simple_plan();
        let mut actuals = HashMap::new();
        actuals.insert("GPT-4o".to_string(), 550.0);
        actuals.insert("Claude".to_string(), 200.0);
        BudgetPlanner::reconcile(&mut plan, &actuals);

        let gpt = &plan.allocations[0];
        assert!((gpt.actual_usd - 550.0).abs() < 1e-9);
        assert!((gpt.variance_usd - (550.0 - 600.0)).abs() < 1e-9);

        let claude = &plan.allocations[1];
        assert!((claude.actual_usd - 200.0).abs() < 1e-9);
        assert!((claude.variance_usd - (200.0 - 400.0)).abs() < 1e-9);
    }

    #[test]
    fn reconcile_sets_status_exceeded() {
        let mut plan = simple_plan();
        let mut actuals = HashMap::new();
        actuals.insert("GPT-4o".to_string(), 700.0); // exceeds 600
        BudgetPlanner::reconcile(&mut plan, &actuals);
        assert!(matches!(plan.allocations[0].status, AllocationStatus::Exceeded { .. }));
    }

    #[test]
    fn reconcile_sets_status_at_risk() {
        let mut plan = simple_plan();
        let mut actuals = HashMap::new();
        actuals.insert("GPT-4o".to_string(), 500.0); // 83% of 600
        BudgetPlanner::reconcile(&mut plan, &actuals);
        assert!(matches!(plan.allocations[0].status, AllocationStatus::AtRisk { .. }));
    }

    #[test]
    fn reconcile_ignores_unknown_names() {
        let mut plan = simple_plan();
        let mut actuals = HashMap::new();
        actuals.insert("Unknown".to_string(), 999.0);
        BudgetPlanner::reconcile(&mut plan, &actuals);
        // No allocations changed.
        for alloc in &plan.allocations {
            assert_eq!(alloc.actual_usd, 0.0);
        }
    }

    // ── BudgetPlanner::forecast ───────────────────────────────────────────────

    #[test]
    fn forecast_on_track_when_underspending() {
        let mut plan = simple_plan();
        let mut actuals = HashMap::new();
        actuals.insert("GPT-4o".to_string(), 100.0);
        actuals.insert("Claude".to_string(), 50.0);
        BudgetPlanner::reconcile(&mut plan, &actuals);
        // Halfway through the period
        let fc = BudgetPlanner::forecast(&plan, 0.5);
        // projected = 200 + 100 = 300, well under 1000
        assert!(fc.on_track);
        assert!((fc.projected_total_usd - 300.0).abs() < 1e-6);
    }

    #[test]
    fn forecast_not_on_track_when_overspending() {
        let mut plan = simple_plan();
        let mut actuals = HashMap::new();
        actuals.insert("GPT-4o".to_string(), 400.0);
        actuals.insert("Claude".to_string(), 300.0);
        BudgetPlanner::reconcile(&mut plan, &actuals);
        // 40% through the period
        let fc = BudgetPlanner::forecast(&plan, 0.4);
        // projected = 1000 + 750 = 1750, over 1000
        assert!(!fc.on_track);
        assert!(fc.projected_variance_usd > 0.0);
    }

    #[test]
    fn forecast_at_risk_allocations_named() {
        let mut plan = simple_plan();
        let mut actuals = HashMap::new();
        actuals.insert("GPT-4o".to_string(), 500.0); // will project to 1000 > 600
        actuals.insert("Claude".to_string(), 50.0);
        BudgetPlanner::reconcile(&mut plan, &actuals);
        let fc = BudgetPlanner::forecast(&plan, 0.5);
        assert!(fc.at_risk_allocations.contains(&"GPT-4o".to_string()));
        assert!(!fc.at_risk_allocations.contains(&"Claude".to_string()));
    }

    #[test]
    fn forecast_clamps_elapsed_fraction_to_epsilon() {
        let plan = simple_plan();
        // Should not panic or return infinity.
        let fc = BudgetPlanner::forecast(&plan, 0.0);
        assert!(fc.projected_total_usd.is_finite());
    }
}
