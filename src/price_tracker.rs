//! Track model pricing changes over time with alerts.
//!
//! [`PriceTracker`] maintains a history of [`ModelPrice`] records and can
//! detect changes, rank models by cost, and compute cost indices relative to
//! a baseline model.

use std::collections::HashMap;

/// The pricing for a single model at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelPrice {
    /// Provider-specific model identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
    pub model_id: String,
    /// Cost in USD per 1 000 input tokens.
    pub input_cost_per_1k: f64,
    /// Cost in USD per 1 000 output tokens.
    pub output_cost_per_1k: f64,
    /// Unix timestamp (seconds) when this pricing became effective.
    pub effective_date: u64,
}

/// Direction of a price movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriceDirection {
    /// The combined cost increased.
    Increase,
    /// The combined cost decreased.
    Decrease,
    /// No change detected.
    Unchanged,
}

/// A detected change between two consecutive prices for the same model.
#[derive(Debug, Clone)]
pub struct PriceChange {
    /// Model identifier.
    pub model_id: String,
    /// Previous price record.
    pub old_price: ModelPrice,
    /// New (current) price record.
    pub new_price: ModelPrice,
    /// Percentage change for input cost: `(new - old) / old * 100`.
    pub change_pct_input: f64,
    /// Percentage change for output cost: `(new - old) / old * 100`.
    pub change_pct_output: f64,
    /// Overall direction of the change based on combined cost.
    pub direction: PriceDirection,
}

/// Append-only pricing history with change detection and ranking.
#[derive(Debug, Default)]
pub struct PriceTracker {
    /// Per-model price history in insertion order.
    history: HashMap<String, Vec<ModelPrice>>,
}

impl PriceTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a price record to the history for its model.
    pub fn record_price(&mut self, price: ModelPrice) {
        self.history
            .entry(price.model_id.clone())
            .or_default()
            .push(price);
    }

    /// Return the most recently recorded price for `model_id`, or `None`.
    pub fn current_price(&self, model_id: &str) -> Option<&ModelPrice> {
        self.history.get(model_id)?.last()
    }

    /// Return the full price history for `model_id` in insertion order.
    pub fn price_history(&self, model_id: &str) -> Vec<&ModelPrice> {
        self.history
            .get(model_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Compare the latest and second-to-latest price for each model and return
    /// any detected changes.
    pub fn detect_changes(&self) -> Vec<PriceChange> {
        let mut changes = Vec::new();
        for (model_id, prices) in &self.history {
            if prices.len() < 2 {
                continue;
            }
            let old = &prices[prices.len() - 2];
            let new = &prices[prices.len() - 1];

            let pct_input = if old.input_cost_per_1k == 0.0 {
                0.0
            } else {
                (new.input_cost_per_1k - old.input_cost_per_1k) / old.input_cost_per_1k * 100.0
            };
            let pct_output = if old.output_cost_per_1k == 0.0 {
                0.0
            } else {
                (new.output_cost_per_1k - old.output_cost_per_1k) / old.output_cost_per_1k * 100.0
            };

            let old_combined = old.input_cost_per_1k + old.output_cost_per_1k;
            let new_combined = new.input_cost_per_1k + new.output_cost_per_1k;
            let direction = if (new_combined - old_combined).abs() < f64::EPSILON {
                PriceDirection::Unchanged
            } else if new_combined > old_combined {
                PriceDirection::Increase
            } else {
                PriceDirection::Decrease
            };

            changes.push(PriceChange {
                model_id: model_id.clone(),
                old_price: old.clone(),
                new_price: new.clone(),
                change_pct_input: pct_input,
                change_pct_output: pct_output,
                direction,
            });
        }
        changes
    }

    /// Return the current price with the highest combined (input + output) cost.
    pub fn most_expensive_model(&self) -> Option<&ModelPrice> {
        self.history
            .values()
            .filter_map(|v| v.last())
            .max_by(|a, b| {
                let ca = a.input_cost_per_1k + a.output_cost_per_1k;
                let cb = b.input_cost_per_1k + b.output_cost_per_1k;
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Return the current price with the lowest combined (input + output) cost.
    pub fn cheapest_model(&self) -> Option<&ModelPrice> {
        self.history
            .values()
            .filter_map(|v| v.last())
            .min_by(|a, b| {
                let ca = a.input_cost_per_1k + a.output_cost_per_1k;
                let cb = b.input_cost_per_1k + b.output_cost_per_1k;
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Compute the cost index of `model_id` relative to `baseline_model`.
    ///
    /// Returns `(model combined cost) / (baseline combined cost)`.
    /// Returns `None` if either model has no recorded price or the baseline
    /// combined cost is zero.
    pub fn cost_index(&self, model_id: &str, baseline_model: &str) -> Option<f64> {
        let model = self.current_price(model_id)?;
        let baseline = self.current_price(baseline_model)?;
        let baseline_combined = baseline.input_cost_per_1k + baseline.output_cost_per_1k;
        if baseline_combined == 0.0 {
            return None;
        }
        let model_combined = model.input_cost_per_1k + model.output_cost_per_1k;
        Some(model_combined / baseline_combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_price(model_id: &str, input: f64, output: f64, date: u64) -> ModelPrice {
        ModelPrice {
            model_id: model_id.to_string(),
            input_cost_per_1k: input,
            output_cost_per_1k: output,
            effective_date: date,
        }
    }

    #[test]
    fn record_and_current_price() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("gpt-4o", 5.0, 15.0, 1_000));
        let p = tracker.current_price("gpt-4o").expect("must exist");
        assert_eq!(p.input_cost_per_1k, 5.0);
    }

    #[test]
    fn current_price_missing_model_is_none() {
        let tracker = PriceTracker::new();
        assert!(tracker.current_price("unknown").is_none());
    }

    #[test]
    fn price_history_returns_all_records() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("gpt-4o", 5.0, 15.0, 1_000));
        tracker.record_price(make_price("gpt-4o", 4.0, 12.0, 2_000));
        let hist = tracker.price_history("gpt-4o");
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].input_cost_per_1k, 5.0);
        assert_eq!(hist[1].input_cost_per_1k, 4.0);
    }

    #[test]
    fn price_history_unknown_model_is_empty() {
        let tracker = PriceTracker::new();
        assert!(tracker.price_history("x").is_empty());
    }

    #[test]
    fn detect_changes_price_decrease() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("gpt-4o", 10.0, 30.0, 1_000));
        tracker.record_price(make_price("gpt-4o", 5.0, 15.0, 2_000));
        let changes = tracker.detect_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].direction, PriceDirection::Decrease);
        assert!((changes[0].change_pct_input - (-50.0)).abs() < 1e-6);
    }

    #[test]
    fn detect_changes_price_increase() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("claude", 3.0, 15.0, 1_000));
        tracker.record_price(make_price("claude", 6.0, 20.0, 2_000));
        let changes = tracker.detect_changes();
        assert_eq!(changes[0].direction, PriceDirection::Increase);
    }

    #[test]
    fn detect_changes_single_entry_no_change() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("gpt-4o", 5.0, 15.0, 1_000));
        assert!(tracker.detect_changes().is_empty());
    }

    #[test]
    fn most_expensive_model() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("cheap", 1.0, 2.0, 1_000));
        tracker.record_price(make_price("expensive", 50.0, 100.0, 1_000));
        let m = tracker.most_expensive_model().expect("must exist");
        assert_eq!(m.model_id, "expensive");
    }

    #[test]
    fn cheapest_model() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("cheap", 0.1, 0.2, 1_000));
        tracker.record_price(make_price("expensive", 50.0, 100.0, 1_000));
        let m = tracker.cheapest_model().expect("must exist");
        assert_eq!(m.model_id, "cheap");
    }

    #[test]
    fn cost_index_relative_to_baseline() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("baseline", 2.0, 8.0, 1_000)); // combined = 10
        tracker.record_price(make_price("model", 4.0, 16.0, 1_000));   // combined = 20
        let idx = tracker.cost_index("model", "baseline").expect("must exist");
        assert!((idx - 2.0).abs() < 1e-9);
    }

    #[test]
    fn cost_index_missing_model_is_none() {
        let tracker = PriceTracker::new();
        assert!(tracker.cost_index("x", "y").is_none());
    }

    #[test]
    fn cost_index_zero_baseline_is_none() {
        let mut tracker = PriceTracker::new();
        tracker.record_price(make_price("baseline", 0.0, 0.0, 1_000));
        tracker.record_price(make_price("model", 5.0, 5.0, 1_000));
        assert!(tracker.cost_index("model", "baseline").is_none());
    }

    #[test]
    fn most_expensive_and_cheapest_empty_is_none() {
        let tracker = PriceTracker::new();
        assert!(tracker.most_expensive_model().is_none());
        assert!(tracker.cheapest_model().is_none());
    }
}
