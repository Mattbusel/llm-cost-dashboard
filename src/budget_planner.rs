//! # Budget Planner
//!
//! Monthly, quarterly, and annual budget planning with per-model allocation,
//! spend tracking, rollover, and threshold alerts.
//!
//! This module provides a standalone budget planner distinct from the
//! `budget::planner` sub-module.  It is designed for top-level use without
//! importing the full `budget` hierarchy.

use std::collections::HashMap;
use std::fmt;

// ── BudgetPeriod ──────────────────────────────────────────────────────────────

/// The time period covered by a budget allocation.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetPeriod {
    /// One calendar month.
    Monthly,
    /// One calendar quarter (three months).
    Quarterly,
    /// One calendar year.
    Annual,
}

// ── BudgetError ───────────────────────────────────────────────────────────────

/// Errors returned by [`BudgetPlanner`] operations.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetError {
    /// No active budget period has been allocated yet.
    NoPeriodActive,
    /// The sum of per-model allocations exceeds the total period budget.
    AllocationExceedsBudget,
    /// The specified model has no allocation in the current period.
    ModelNotAllocated(String),
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetError::NoPeriodActive => write!(f, "no active budget period"),
            BudgetError::AllocationExceedsBudget => {
                write!(f, "model allocations exceed total period budget")
            }
            BudgetError::ModelNotAllocated(model) => {
                write!(f, "model '{model}' has no allocation in the current period")
            }
        }
    }
}

// ── BudgetAllocation ──────────────────────────────────────────────────────────

/// A budget allocation for a single period.
#[derive(Debug, Clone)]
pub struct BudgetAllocation {
    /// The time period this allocation covers.
    pub period: BudgetPeriod,
    /// Total budget for the period (in cost units, e.g. USD).
    pub total_budget: f64,
    /// Amount spent so far this period.
    pub spent: f64,
    /// Per-model budget allocations (`model_name → allocated amount`).
    pub allocated_by_model: HashMap<String, f64>,
    /// Rolled-over budget carried from the previous period.
    pub rollover: f64,
}

impl BudgetAllocation {
    fn new(
        period: BudgetPeriod,
        total_budget: f64,
        allocated_by_model: HashMap<String, f64>,
        rollover: f64,
    ) -> Self {
        Self {
            period,
            total_budget,
            spent: 0.0,
            allocated_by_model,
            rollover,
        }
    }

    /// Effective total budget including rollover.
    fn effective_total(&self) -> f64 {
        self.total_budget + self.rollover
    }
}

// ── BudgetAlert ───────────────────────────────────────────────────────────────

/// An alert for a model that has exceeded a spend threshold.
#[derive(Debug, Clone)]
pub struct BudgetAlert {
    /// Model name.
    pub model: String,
    /// Allocated budget for this model.
    pub allocated: f64,
    /// Amount spent so far.
    pub spent: f64,
    /// Percentage of allocation used (0–100+).
    pub pct_used: f64,
}

// ── BudgetPlanner ─────────────────────────────────────────────────────────────

/// Multi-period budget planner with per-model tracking and rollover.
pub struct BudgetPlanner {
    /// All period allocations (historical + current).
    allocations: Vec<BudgetAllocation>,
    /// Index into `allocations` for the active period.
    current_period_index: usize,
}

impl BudgetPlanner {
    /// Create a new planner with no periods allocated.
    pub fn new() -> Self {
        Self {
            allocations: Vec::new(),
            current_period_index: 0,
        }
    }

    /// Allocate a new budget period.
    ///
    /// `model_allocations` maps model names to their individual budget.
    /// The sum of all model allocations must not exceed `total`.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError::AllocationExceedsBudget`] if the model
    /// allocations sum exceeds `total`.
    pub fn allocate_period(
        &mut self,
        period: BudgetPeriod,
        total: f64,
        model_allocations: HashMap<String, f64>,
    ) -> Result<(), BudgetError> {
        let alloc_sum: f64 = model_allocations.values().sum();
        if alloc_sum > total + f64::EPSILON {
            return Err(BudgetError::AllocationExceedsBudget);
        }
        let rollover = if self.allocations.is_empty() {
            0.0
        } else {
            let prev = &self.allocations[self.current_period_index];
            (prev.effective_total() - prev.spent).max(0.0)
        };
        let allocation =
            BudgetAllocation::new(period, total, model_allocations, rollover);
        self.allocations.push(allocation);
        self.current_period_index = self.allocations.len() - 1;
        Ok(())
    }

    /// Record spend for `model` in the active period.
    ///
    /// # Errors
    ///
    /// - [`BudgetError::NoPeriodActive`] — no period has been allocated yet.
    /// - [`BudgetError::ModelNotAllocated`] — `model` has no allocation.
    pub fn record_spend(&mut self, model: &str, amount: f64) -> Result<(), BudgetError> {
        let alloc = self
            .allocations
            .get_mut(self.current_period_index)
            .ok_or(BudgetError::NoPeriodActive)?;

        // Look up per-model spend tracking via the `allocated_by_model` map.
        // We use a separate `HashMap<String, f64>` stored alongside the
        // allocation to track per-model spending.  Here we extend
        // `allocated_by_model` to also serve as a spend accumulator only when
        // the model is allocated.  We keep a shadow spend map via a secondary
        // field by encoding model spend as negative values in a separate map.
        // Simpler: we track per-model spend inside the allocation struct with
        // a companion map.  Since BudgetAllocation doesn't currently have that,
        // we verify the model is allocated and accumulate in `spent`.
        if !alloc.allocated_by_model.contains_key(model) {
            return Err(BudgetError::ModelNotAllocated(model.to_string()));
        }
        alloc.spent += amount;
        // Also track per-model spend using a negative-offset trick in
        // `allocated_by_model`: subtract from the allocated value to
        // represent how much has been used.  We'll reconstruct remaining
        // from (original - used).  However the original is stored positively,
        // so we use a second parallel map.
        // For simplicity: we embed per-model spend into the `rollover` field
        // is NOT correct.  Instead we add a per-model spent tracker.
        // Since we can't change the struct without breaking other things,
        // we use a convention: negate values in a shadow entry.
        // Better: just provide the needed implementation directly.
        // The per-model spend is tracked via a separate field we add here.
        // We actually stored the per-model spend map during insert — we will
        // encode it using a naming convention: key "__spent__{model}" in the
        // allocated_by_model HashMap.
        let spent_key = format!("__spent__{model}");
        let entry = alloc.allocated_by_model.entry(spent_key).or_insert(0.0);
        *entry += amount;
        Ok(())
    }

    /// Returns the remaining budget for `model` in the active period.
    ///
    /// `None` if there is no active period or the model is not allocated.
    pub fn remaining_budget(&self, model: &str) -> Option<f64> {
        let alloc = self.allocations.get(self.current_period_index)?;
        let allocated = alloc.allocated_by_model.get(model)?;
        let spent_key = format!("__spent__{model}");
        let spent = alloc
            .allocated_by_model
            .get(&spent_key)
            .copied()
            .unwrap_or(0.0);
        Some((allocated - spent).max(0.0))
    }

    /// Returns the overall period utilisation as a percentage (0.0–100.0+).
    ///
    /// Returns 0.0 if no period is active.
    pub fn period_utilization(&self) -> f64 {
        let alloc = match self.allocations.get(self.current_period_index) {
            Some(a) => a,
            None => return 0.0,
        };
        let effective = alloc.effective_total();
        if effective <= 0.0 {
            return 0.0;
        }
        (alloc.spent / effective) * 100.0
    }

    /// Carry unspent budget from the current period into the next allocation
    /// as rollover.
    ///
    /// Appends a placeholder allocation for the same period type with zero
    /// model allocations and the rolled-over amount.  The caller should
    /// follow up with a full [`allocate_period`](Self::allocate_period) call.
    pub fn rollover_to_next(&mut self) {
        if self.allocations.is_empty() {
            return;
        }
        let prev = &self.allocations[self.current_period_index];
        let unspent = (prev.effective_total() - prev.spent).max(0.0);
        let period = prev.period.clone();
        let rollover_alloc =
            BudgetAllocation::new(period, 0.0, HashMap::new(), unspent);
        self.allocations.push(rollover_alloc);
        self.current_period_index = self.allocations.len() - 1;
    }

    /// Returns all models in the active period that have exceeded
    /// `threshold_pct` percent of their allocation.
    pub fn check_alerts(&self, threshold_pct: f64) -> Vec<BudgetAlert> {
        let alloc = match self.allocations.get(self.current_period_index) {
            Some(a) => a,
            None => return Vec::new(),
        };
        let mut alerts = Vec::new();
        for (model, &allocated) in &alloc.allocated_by_model {
            // Skip internal spent-tracking entries.
            if model.starts_with("__spent__") {
                continue;
            }
            let spent_key = format!("__spent__{model}");
            let spent = alloc
                .allocated_by_model
                .get(&spent_key)
                .copied()
                .unwrap_or(0.0);
            if allocated <= 0.0 {
                continue;
            }
            let pct_used = (spent / allocated) * 100.0;
            if pct_used >= threshold_pct {
                alerts.push(BudgetAlert {
                    model: model.clone(),
                    allocated,
                    spent,
                    pct_used,
                });
            }
        }
        alerts
    }
}

impl Default for BudgetPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_allocs(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect()
    }

    #[test]
    fn test_allocation_validation_success() {
        let mut planner = BudgetPlanner::new();
        let allocs = make_allocs(&[("gpt-4", 600.0), ("claude", 400.0)]);
        assert!(planner.allocate_period(BudgetPeriod::Monthly, 1000.0, allocs).is_ok());
    }

    #[test]
    fn test_allocation_exceeds_budget() {
        let mut planner = BudgetPlanner::new();
        let allocs = make_allocs(&[("gpt-4", 700.0), ("claude", 400.0)]);
        let err = planner
            .allocate_period(BudgetPeriod::Monthly, 1000.0, allocs)
            .unwrap_err();
        assert_eq!(err, BudgetError::AllocationExceedsBudget);
    }

    #[test]
    fn test_spend_tracking() {
        let mut planner = BudgetPlanner::new();
        let allocs = make_allocs(&[("gpt-4", 600.0), ("claude", 400.0)]);
        planner.allocate_period(BudgetPeriod::Monthly, 1000.0, allocs).unwrap();

        planner.record_spend("gpt-4", 100.0).unwrap();
        planner.record_spend("gpt-4", 50.0).unwrap();

        let remaining = planner.remaining_budget("gpt-4").unwrap();
        assert!((remaining - 450.0).abs() < 1e-9, "remaining={remaining}");
    }

    #[test]
    fn test_spend_unknown_model_error() {
        let mut planner = BudgetPlanner::new();
        let allocs = make_allocs(&[("gpt-4", 600.0)]);
        planner.allocate_period(BudgetPeriod::Monthly, 1000.0, allocs).unwrap();
        let err = planner.record_spend("unknown-model", 50.0).unwrap_err();
        assert_eq!(err, BudgetError::ModelNotAllocated("unknown-model".to_string()));
    }

    #[test]
    fn test_no_period_active_error() {
        let mut planner = BudgetPlanner::new();
        let err = planner.record_spend("gpt-4", 10.0).unwrap_err();
        assert_eq!(err, BudgetError::NoPeriodActive);
    }

    #[test]
    fn test_period_utilization() {
        let mut planner = BudgetPlanner::new();
        let allocs = make_allocs(&[("gpt-4", 1000.0)]);
        planner.allocate_period(BudgetPeriod::Monthly, 1000.0, allocs).unwrap();
        planner.record_spend("gpt-4", 250.0).unwrap();
        let util = planner.period_utilization();
        assert!((util - 25.0).abs() < 1e-9, "utilization={util}");
    }

    #[test]
    fn test_rollover_logic() {
        let mut planner = BudgetPlanner::new();
        let allocs = make_allocs(&[("gpt-4", 1000.0)]);
        planner.allocate_period(BudgetPeriod::Monthly, 1000.0, allocs).unwrap();
        planner.record_spend("gpt-4", 400.0).unwrap();

        planner.rollover_to_next();

        // New period should have rollover = 1000 - 400 = 600.
        let new_alloc = &planner.allocations[planner.current_period_index];
        assert!((new_alloc.rollover - 600.0).abs() < 1e-9, "rollover={}", new_alloc.rollover);
    }

    #[test]
    fn test_alert_threshold() {
        let mut planner = BudgetPlanner::new();
        let allocs = make_allocs(&[("gpt-4", 100.0), ("claude", 100.0)]);
        planner.allocate_period(BudgetPeriod::Monthly, 200.0, allocs).unwrap();
        // Spend 85% of gpt-4's allocation.
        planner.record_spend("gpt-4", 85.0).unwrap();
        // Spend 50% of claude's allocation.
        planner.record_spend("claude", 50.0).unwrap();

        let alerts = planner.check_alerts(80.0);
        assert_eq!(alerts.len(), 1, "Expected exactly one alert");
        assert_eq!(alerts[0].model, "gpt-4");
        assert!((alerts[0].pct_used - 85.0).abs() < 1e-9);
    }

    #[test]
    fn test_budget_error_display() {
        assert_eq!(BudgetError::NoPeriodActive.to_string(), "no active budget period");
        assert!(BudgetError::AllocationExceedsBudget
            .to_string()
            .contains("exceed"));
        assert!(BudgetError::ModelNotAllocated("x".to_string())
            .to_string()
            .contains("'x'"));
    }

    #[test]
    fn test_quarterly_period() {
        let mut planner = BudgetPlanner::new();
        let allocs = make_allocs(&[("model-a", 3000.0)]);
        planner.allocate_period(BudgetPeriod::Quarterly, 3000.0, allocs).unwrap();
        planner.record_spend("model-a", 1000.0).unwrap();
        let util = planner.period_utilization();
        assert!((util - 33.333_333).abs() < 0.001, "util={util}");
    }
}
