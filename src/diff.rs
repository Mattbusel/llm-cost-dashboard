//! # Cost Diff Reporter
//!
//! Compares cost snapshots across two time periods and produces a structured
//! [`DiffReport`] with absolute/percentage changes and per-model breakdowns.
//!
//! ## Overview
//!
//! [`CostDiff::compare`] accepts a `baseline` and `current` [`PeriodSnapshot`]
//! and produces a [`DiffReport`] that can be rendered as Markdown via
//! [`DiffReport::render_markdown`].
//!
//! ## Example
//!
//! ```rust
//! use std::collections::HashMap;
//! use llm_cost_dashboard::diff::{CostDiff, PeriodSnapshot};
//!
//! let baseline = PeriodSnapshot {
//!     period: "2024-W01".into(),
//!     total_cost_usd: 10.0,
//!     by_model: [("gpt-4o".into(), 10.0)].into(),
//!     request_count: 100,
//! };
//! let current = PeriodSnapshot {
//!     period: "2024-W02".into(),
//!     total_cost_usd: 12.0,
//!     by_model: [("gpt-4o".into(), 12.0)].into(),
//!     request_count: 120,
//! };
//! let report = CostDiff::compare(&baseline, &current);
//! assert!((report.absolute_change_usd - 2.0).abs() < 1e-9);
//! println!("{}", report.render_markdown());
//! ```

use std::collections::HashMap;

// ============================================================================
// Domain types
// ============================================================================

/// A cost snapshot for a single named time period.
#[derive(Debug, Clone)]
pub struct PeriodSnapshot {
    /// Human-readable period label (e.g. `"2024-W01"`, `"2024-01"`, `"7d"`).
    pub period: String,
    /// Total cost in USD across all models.
    pub total_cost_usd: f64,
    /// Per-model cost breakdown (model_id → cost in USD).
    pub by_model: HashMap<String, f64>,
    /// Total number of requests in this period.
    pub request_count: u64,
}

/// Per-model cost change between baseline and current.
#[derive(Debug, Clone)]
pub struct ModelDiff {
    /// Model identifier (e.g. `"claude-3-5-sonnet"`).
    pub model_id: String,
    /// Cost in the baseline period (0.0 if absent).
    pub baseline_usd: f64,
    /// Cost in the current period (0.0 if absent).
    pub current_usd: f64,
    /// `current_usd - baseline_usd`.
    pub change_usd: f64,
    /// Percentage change relative to `baseline_usd`; `f64::INFINITY` when
    /// `baseline_usd` is 0.
    pub pct_change: f64,
}

/// A complete diff between two [`PeriodSnapshot`]s.
#[derive(Debug, Clone)]
pub struct DiffReport {
    /// Label of the baseline period.
    pub baseline_period: String,
    /// Label of the current period.
    pub current_period: String,
    /// `current.total_cost_usd - baseline.total_cost_usd`.
    pub absolute_change_usd: f64,
    /// Percentage change in total cost.
    pub pct_change: f64,
    /// Models present in `current` but not in `baseline`.
    pub new_models: Vec<String>,
    /// Models present in `baseline` but not in `current`.
    pub removed_models: Vec<String>,
    /// Per-model cost deltas, sorted by `|change_usd|` descending.
    pub model_changes: Vec<ModelDiff>,
    /// Request count in the baseline period.
    pub baseline_requests: u64,
    /// Request count in the current period.
    pub current_requests: u64,
}

impl DiffReport {
    /// Render the diff as a Markdown table with ↑↓ arrows.
    ///
    /// Output format:
    /// ```text
    /// ## Cost Diff: baseline → current
    ///
    /// | Metric | Baseline | Current | Change |
    /// |--------|----------|---------|--------|
    /// | Total cost | $10.00 | $12.00 | ↑ $2.00 (+20.0%) |
    /// ...
    ///
    /// ### Per-model breakdown
    /// ...
    /// ```
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!(
            "## Cost Diff: {} → {}\n\n",
            self.baseline_period, self.current_period
        ));

        // Summary table
        md.push_str("| Metric | Baseline | Current | Change |\n");
        md.push_str("|--------|----------|---------|--------|\n");

        let total_change = arrow_fmt(self.absolute_change_usd, self.pct_change);
        md.push_str(&format!(
            "| Total cost | ${:.4} | ${:.4} | {} |\n",
            self.baseline_period_cost(),
            self.current_period_cost(),
            total_change,
        ));

        let req_change = self.current_requests as i64 - self.baseline_requests as i64;
        let req_pct = pct(self.baseline_requests as f64, self.current_requests as f64);
        md.push_str(&format!(
            "| Requests | {} | {} | {} |\n",
            self.baseline_requests,
            self.current_requests,
            arrow_pct(req_change as f64, req_pct),
        ));

        md.push('\n');

        // New / removed models
        if !self.new_models.is_empty() {
            md.push_str(&format!(
                "**New models:** {}\n\n",
                self.new_models.join(", ")
            ));
        }
        if !self.removed_models.is_empty() {
            md.push_str(&format!(
                "**Removed models:** {}\n\n",
                self.removed_models.join(", ")
            ));
        }

        // Per-model table
        if !self.model_changes.is_empty() {
            md.push_str("### Per-model breakdown\n\n");
            md.push_str("| Model | Baseline | Current | Change |\n");
            md.push_str("|-------|----------|---------|--------|\n");
            for m in &self.model_changes {
                let chg = arrow_fmt(m.change_usd, m.pct_change);
                md.push_str(&format!(
                    "| {} | ${:.4} | ${:.4} | {} |\n",
                    m.model_id, m.baseline_usd, m.current_usd, chg,
                ));
            }
            md.push('\n');
        }

        md
    }

    /// Retrieve the total baseline cost from the model_changes fallback.
    fn baseline_period_cost(&self) -> f64 {
        self.model_changes.iter().map(|m| m.baseline_usd).sum::<f64>()
            + self
                .removed_models
                .iter()
                .map(|_| 0.0_f64)
                .sum::<f64>()
    }

    /// Retrieve the total current cost from the model_changes.
    fn current_period_cost(&self) -> f64 {
        self.model_changes.iter().map(|m| m.current_usd).sum::<f64>()
    }
}

// ============================================================================
// CostDiff
// ============================================================================

/// Compares two [`PeriodSnapshot`]s and produces a [`DiffReport`].
pub struct CostDiff;

impl CostDiff {
    /// Compare `baseline` against `current` and produce a full [`DiffReport`].
    pub fn compare(baseline: &PeriodSnapshot, current: &PeriodSnapshot) -> DiffReport {
        let absolute_change_usd = current.total_cost_usd - baseline.total_cost_usd;
        let pct_change = pct(baseline.total_cost_usd, current.total_cost_usd);

        // Union of all model IDs.
        let mut all_models: Vec<String> = baseline
            .by_model
            .keys()
            .chain(current.by_model.keys())
            .cloned()
            .collect();
        all_models.sort_unstable();
        all_models.dedup();

        let mut new_models: Vec<String> = current
            .by_model
            .keys()
            .filter(|k| !baseline.by_model.contains_key(*k))
            .cloned()
            .collect();
        new_models.sort_unstable();

        let mut removed_models: Vec<String> = baseline
            .by_model
            .keys()
            .filter(|k| !current.by_model.contains_key(*k))
            .cloned()
            .collect();
        removed_models.sort_unstable();

        let mut model_changes: Vec<ModelDiff> = all_models
            .iter()
            .map(|model_id| {
                let b = baseline.by_model.get(model_id).cloned().unwrap_or(0.0);
                let c = current.by_model.get(model_id).cloned().unwrap_or(0.0);
                let change_usd = c - b;
                ModelDiff {
                    model_id: model_id.clone(),
                    baseline_usd: b,
                    current_usd: c,
                    change_usd,
                    pct_change: pct(b, c),
                }
            })
            .collect();

        // Sort by magnitude of change descending.
        model_changes.sort_by(|a, b| {
            b.change_usd
                .abs()
                .partial_cmp(&a.change_usd.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        DiffReport {
            baseline_period: baseline.period.clone(),
            current_period: current.period.clone(),
            absolute_change_usd,
            pct_change,
            new_models,
            removed_models,
            model_changes,
            baseline_requests: baseline.request_count,
            current_requests: current.request_count,
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn pct(baseline: f64, current: f64) -> f64 {
    if baseline == 0.0 {
        if current == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (current - baseline) / baseline * 100.0
    }
}

fn arrow_fmt(change: f64, pct_val: f64) -> String {
    if change >= 0.0 {
        if pct_val.is_infinite() {
            format!("↑ ${:.4} (new)", change)
        } else {
            format!("↑ ${:.4} (+{:.1}%)", change, pct_val)
        }
    } else {
        format!("↓ ${:.4} ({:.1}%)", change, pct_val)
    }
}

fn arrow_pct(change: f64, pct_val: f64) -> String {
    if change >= 0.0 {
        if pct_val.is_infinite() {
            format!("↑ {} (new)", change)
        } else {
            format!("↑ {} (+{:.1}%)", change, pct_val)
        }
    } else {
        format!("↓ {} ({:.1}%)", change, pct_val)
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(period: &str, total: f64, models: Vec<(&str, f64)>, reqs: u64) -> PeriodSnapshot {
        PeriodSnapshot {
            period: period.into(),
            total_cost_usd: total,
            by_model: models.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            request_count: reqs,
        }
    }

    // --- absolute change ---

    #[test]
    fn test_absolute_increase() {
        let b = snap("w1", 10.0, vec![("gpt-4o", 10.0)], 100);
        let c = snap("w2", 12.0, vec![("gpt-4o", 12.0)], 120);
        let r = CostDiff::compare(&b, &c);
        assert!((r.absolute_change_usd - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_absolute_decrease() {
        let b = snap("w1", 10.0, vec![("gpt-4o", 10.0)], 100);
        let c = snap("w2", 8.0, vec![("gpt-4o", 8.0)], 80);
        let r = CostDiff::compare(&b, &c);
        assert!((r.absolute_change_usd - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_no_change() {
        let b = snap("w1", 5.0, vec![("m", 5.0)], 50);
        let c = snap("w2", 5.0, vec![("m", 5.0)], 50);
        let r = CostDiff::compare(&b, &c);
        assert!((r.absolute_change_usd).abs() < 1e-9);
        assert!((r.pct_change).abs() < 1e-9);
    }

    // --- percentage change ---

    #[test]
    fn test_pct_increase() {
        let b = snap("w1", 10.0, vec![("m", 10.0)], 10);
        let c = snap("w2", 15.0, vec![("m", 15.0)], 10);
        let r = CostDiff::compare(&b, &c);
        assert!((r.pct_change - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_pct_zero_baseline_infinity() {
        let b = snap("w1", 0.0, vec![], 0);
        let c = snap("w2", 5.0, vec![("m", 5.0)], 5);
        let r = CostDiff::compare(&b, &c);
        assert!(r.pct_change.is_infinite());
    }

    // --- new / removed models ---

    #[test]
    fn test_new_model_detected() {
        let b = snap("w1", 5.0, vec![("a", 5.0)], 10);
        let c = snap("w2", 8.0, vec![("a", 5.0), ("b", 3.0)], 12);
        let r = CostDiff::compare(&b, &c);
        assert_eq!(r.new_models, vec!["b"]);
    }

    #[test]
    fn test_removed_model_detected() {
        let b = snap("w1", 8.0, vec![("a", 5.0), ("b", 3.0)], 10);
        let c = snap("w2", 5.0, vec![("a", 5.0)], 8);
        let r = CostDiff::compare(&b, &c);
        assert_eq!(r.removed_models, vec!["b"]);
    }

    #[test]
    fn test_no_new_or_removed_models() {
        let b = snap("w1", 5.0, vec![("a", 5.0)], 10);
        let c = snap("w2", 6.0, vec![("a", 6.0)], 12);
        let r = CostDiff::compare(&b, &c);
        assert!(r.new_models.is_empty());
        assert!(r.removed_models.is_empty());
    }

    // --- model_changes ---

    #[test]
    fn test_model_changes_count() {
        let b = snap("w1", 10.0, vec![("a", 6.0), ("b", 4.0)], 10);
        let c = snap("w2", 12.0, vec![("a", 7.0), ("b", 5.0)], 10);
        let r = CostDiff::compare(&b, &c);
        assert_eq!(r.model_changes.len(), 2);
    }

    #[test]
    fn test_model_changes_sorted_by_magnitude() {
        let b = snap("w1", 10.0, vec![("a", 1.0), ("b", 9.0)], 10);
        let c = snap("w2", 25.0, vec![("a", 2.0), ("b", 23.0)], 10);
        let r = CostDiff::compare(&b, &c);
        // Model b changed by 14.0, model a by 1.0 → b should come first
        assert_eq!(r.model_changes[0].model_id, "b");
    }

    #[test]
    fn test_model_diff_fields() {
        let b = snap("w1", 10.0, vec![("m", 10.0)], 10);
        let c = snap("w2", 15.0, vec![("m", 15.0)], 10);
        let r = CostDiff::compare(&b, &c);
        let m = &r.model_changes[0];
        assert_eq!(m.model_id, "m");
        assert!((m.baseline_usd - 10.0).abs() < 1e-9);
        assert!((m.current_usd - 15.0).abs() < 1e-9);
        assert!((m.change_usd - 5.0).abs() < 1e-9);
        assert!((m.pct_change - 50.0).abs() < 1e-6);
    }

    // --- render_markdown ---

    #[test]
    fn test_render_markdown_contains_periods() {
        let b = snap("week-1", 10.0, vec![("m", 10.0)], 10);
        let c = snap("week-2", 12.0, vec![("m", 12.0)], 12);
        let r = CostDiff::compare(&b, &c);
        let md = r.render_markdown();
        assert!(md.contains("week-1"));
        assert!(md.contains("week-2"));
    }

    #[test]
    fn test_render_markdown_contains_table_header() {
        let b = snap("a", 1.0, vec![("m", 1.0)], 1);
        let c = snap("b", 2.0, vec![("m", 2.0)], 2);
        let r = CostDiff::compare(&b, &c);
        let md = r.render_markdown();
        assert!(md.contains("| Metric |"));
    }

    #[test]
    fn test_render_markdown_up_arrow_on_increase() {
        let b = snap("a", 5.0, vec![("m", 5.0)], 5);
        let c = snap("b", 10.0, vec![("m", 10.0)], 10);
        let r = CostDiff::compare(&b, &c);
        let md = r.render_markdown();
        assert!(md.contains('↑'));
    }

    #[test]
    fn test_render_markdown_down_arrow_on_decrease() {
        let b = snap("a", 10.0, vec![("m", 10.0)], 10);
        let c = snap("b", 5.0, vec![("m", 5.0)], 5);
        let r = CostDiff::compare(&b, &c);
        let md = r.render_markdown();
        assert!(md.contains('↓'));
    }

    #[test]
    fn test_render_markdown_model_section() {
        let b = snap("a", 5.0, vec![("claude-3", 5.0)], 5);
        let c = snap("b", 7.0, vec![("claude-3", 7.0)], 7);
        let r = CostDiff::compare(&b, &c);
        let md = r.render_markdown();
        assert!(md.contains("claude-3"));
        assert!(md.contains("Per-model breakdown"));
    }

    #[test]
    fn test_render_markdown_new_models_listed() {
        let b = snap("a", 5.0, vec![("old", 5.0)], 5);
        let c = snap("b", 8.0, vec![("old", 5.0), ("new-model", 3.0)], 8);
        let r = CostDiff::compare(&b, &c);
        let md = r.render_markdown();
        assert!(md.contains("new-model"));
        assert!(md.contains("New models"));
    }

    #[test]
    fn test_request_counts_in_report() {
        let b = snap("a", 5.0, vec![("m", 5.0)], 100);
        let c = snap("b", 6.0, vec![("m", 6.0)], 150);
        let r = CostDiff::compare(&b, &c);
        assert_eq!(r.baseline_requests, 100);
        assert_eq!(r.current_requests, 150);
    }
}
