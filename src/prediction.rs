//! # Cost Prediction Engine
//!
//! Predicts future LLM costs based on historical usage patterns learned from
//! [`TaggedRequest`] data.
//!
//! ## Overview
//!
//! [`CostPredictor`] builds a 24×7 pattern matrix where each cell represents
//! the average cost observed at a given (hour-of-day, day-of-week) combination.
//! Predictions are made by looking up the cell for the target timestamp.
//!
//! [`WeeklyHeatmap`] renders the full 7×24 prediction grid as an ASCII
//! heatmap using intensity characters.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::prediction::CostPredictor;
//!
//! let predictor = CostPredictor::new();
//! // Without training data, predictions return 0.0.
//! let p = predictor.predict_next_hour();
//! assert!(p >= 0.0);
//! ```

use chrono::{DateTime, Datelike, Timelike, Utc};

use crate::tagging::TaggedRequest;

// ============================================================================
// Domain types
// ============================================================================

/// Historical usage statistics for one (hour-of-day × day-of-week) cell.
#[derive(Debug, Clone, Default)]
pub struct UsagePattern {
    /// Hour of day (0–23).
    pub hour_of_day: u8,
    /// Day of week (0 = Monday … 6 = Sunday, ISO weekday – 1).
    pub day_of_week: u8,
    /// Cumulative average cost in USD observed for this cell.
    pub avg_cost_usd: f64,
    /// Number of samples that contributed to `avg_cost_usd`.
    pub sample_count: u32,
}

impl UsagePattern {
    fn new(hour: u8, day: u8) -> Self {
        Self {
            hour_of_day: hour,
            day_of_week: day,
            avg_cost_usd: 0.0,
            sample_count: 0,
        }
    }

    /// Update the running average with a new observation.
    fn record(&mut self, cost: f64) {
        self.sample_count += 1;
        // Online Welford-style mean update.
        self.avg_cost_usd +=
            (cost - self.avg_cost_usd) / self.sample_count as f64;
    }
}

/// 24×7 matrix of [`UsagePattern`] cells (indexed `[hour][day]`).
pub struct PatternMatrix([[UsagePattern; 7]; 24]);

impl PatternMatrix {
    fn new() -> Self {
        let mut matrix: [[std::mem::MaybeUninit<UsagePattern>; 7]; 24] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for (h, row) in matrix.iter_mut().enumerate() {
            for (d, cell) in row.iter_mut().enumerate() {
                cell.write(UsagePattern::new(h as u8, d as u8));
            }
        }
        // SAFETY: every element has been initialised above.
        PatternMatrix(unsafe {
            std::mem::transmute::<
                [[std::mem::MaybeUninit<UsagePattern>; 7]; 24],
                [[UsagePattern; 7]; 24],
            >(matrix)
        })
    }

    fn get(&self, hour: u8, day: u8) -> &UsagePattern {
        &self.0[hour as usize % 24][day as usize % 7]
    }

    fn get_mut(&mut self, hour: u8, day: u8) -> &mut UsagePattern {
        &mut self.0[hour as usize % 24][day as usize % 7]
    }
}

// ============================================================================
// CostPredictor
// ============================================================================

/// Predicts future LLM costs based on a learned 24×7 pattern matrix.
pub struct CostPredictor {
    matrix: PatternMatrix,
}

impl CostPredictor {
    /// Create a new predictor with an empty (zero) pattern matrix.
    pub fn new() -> Self {
        Self {
            matrix: PatternMatrix::new(),
        }
    }

    /// Learn from a slice of historical [`TaggedRequest`]s.
    ///
    /// Each request updates the (hour, day-of-week) cell corresponding to its
    /// timestamp.  Multiple calls to `learn` are additive.
    pub fn learn(&mut self, requests: &[TaggedRequest]) {
        for req in requests {
            let hour = req.timestamp.hour() as u8;
            // chrono weekday: Mon=0 … Sun=6
            let day = req.timestamp.weekday().num_days_from_monday() as u8;
            self.matrix.get_mut(hour, day).record(req.cost_usd);
        }
    }

    /// Predict cost for the current (UTC) hour and day-of-week.
    pub fn predict_next_hour(&self) -> f64 {
        let now = Utc::now();
        let hour = now.hour() as u8;
        let day = now.weekday().num_days_from_monday() as u8;
        self.matrix.get(hour, day).avg_cost_usd
    }

    /// Predict costs for the next `n` hours, starting from the current UTC hour.
    ///
    /// Returns a `Vec` of `(timestamp, predicted_cost_usd)` pairs.
    pub fn predict_next_n_hours(&self, n: usize) -> Vec<(DateTime<Utc>, f64)> {
        let now = Utc::now();
        let base_ts = now
            .with_minute(0)
            .and_then(|t| t.with_second(0))
            .and_then(|t| t.with_nanosecond(0))
            .unwrap_or(now);

        (0..n)
            .map(|i| {
                let ts = base_ts + chrono::Duration::hours(i as i64);
                let hour = ts.hour() as u8;
                let day = ts.weekday().num_days_from_monday() as u8;
                let cost = self.matrix.get(hour, day).avg_cost_usd;
                (ts, cost)
            })
            .collect()
    }

    /// Confidence for a (hour, day) cell: `min(1.0, sample_count / 10.0)`.
    ///
    /// Returns 0.0 when no samples have been observed, 1.0 when ≥10 samples
    /// exist.
    pub fn confidence(&self, hour: u8, day: u8) -> f64 {
        let p = self.matrix.get(hour, day);
        (p.sample_count as f64 / 10.0).min(1.0)
    }

    /// Build a [`WeeklyHeatmap`] from the current pattern matrix.
    pub fn weekly_heatmap(&self) -> WeeklyHeatmap {
        let mut grid = [[0.0f64; 24]; 7];
        for day in 0u8..7 {
            for hour in 0u8..24 {
                grid[day as usize][hour as usize] =
                    self.matrix.get(hour, day).avg_cost_usd;
            }
        }
        WeeklyHeatmap { grid }
    }
}

impl Default for CostPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WeeklyHeatmap
// ============================================================================

const DAY_LABELS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/// 7×24 grid of predicted costs, renderable as an ASCII heatmap.
pub struct WeeklyHeatmap {
    /// `grid[day][hour]` — predicted cost in USD.
    pub grid: [[f64; 24]; 7],
}

impl WeeklyHeatmap {
    /// Render the heatmap as an ASCII string using intensity characters.
    ///
    /// Intensity scale (based on fraction of the observed maximum):
    /// ```text
    ///  0.0 – 0.1   ·   (empty / negligible)
    ///  0.1 – 0.2   ░   (very light)
    ///  0.2 – 0.4   ▒   (light)
    ///  0.4 – 0.6   ▓   (medium)
    ///  0.6 – 0.8   █   (heavy)
    ///  0.8 – 1.0   ■   (peak)
    /// ```
    pub fn render_ascii(&self) -> String {
        // Find the maximum value to normalise.
        let max = self
            .grid
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let max = if max <= 0.0 { 1.0 } else { max };

        let hours_header: String = (0..24).map(|h| format!("{h:2}")).collect::<Vec<_>>().join(" ");

        let mut out = String::new();
        out.push_str("     "); // padding for day label
        out.push_str(&hours_header);
        out.push('\n');

        for (day, row) in self.grid.iter().enumerate() {
            out.push_str(&format!("{:3}  ", DAY_LABELS[day]));
            for &cost in row.iter() {
                let ratio = cost / max;
                let ch = match ratio {
                    r if r < 0.1 => '.',
                    r if r < 0.2 => '\u{2591}', // ░
                    r if r < 0.4 => '\u{2592}', // ▒
                    r if r < 0.6 => '\u{2593}', // ▓
                    r if r < 0.8 => '\u{2588}', // █
                    _ => '\u{25A0}',             // ■
                };
                out.push(ch);
                out.push(' ');
            }
            out.push('\n');
        }
        out
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tagging::{CostTag, TaggedRequest};
    use chrono::TimeZone;

    fn req_at(hour: u32, day_offset: i64, cost: f64) -> TaggedRequest {
        // day_offset 0 = 2024-01-01 Monday
        let base = Utc.with_ymd_and_hms(2024, 1, 1, hour, 0, 0).unwrap();
        let ts = base + chrono::Duration::days(day_offset);
        TaggedRequest {
            request_id: 0,
            model_id: "gpt-4o".into(),
            cost_usd: cost,
            tokens_in: 100,
            tokens_out: 50,
            tags: vec![CostTag::new("env", "test")],
            timestamp: ts,
        }
    }

    // --- learn + predict ---

    #[test]
    fn test_no_data_returns_zero() {
        let p = CostPredictor::new();
        // Can't guarantee what hour/day "now" is, but without learning
        // all cells are zero.
        let heatmap = p.weekly_heatmap();
        let all_zero = heatmap.grid.iter().flat_map(|r| r.iter()).all(|&v| v == 0.0);
        assert!(all_zero);
    }

    #[test]
    fn test_learn_single_request() {
        let mut p = CostPredictor::new();
        p.learn(&[req_at(10, 0, 1.5)]);
        // Monday = day_of_week 0, hour 10
        assert!((p.matrix.get(10, 0).avg_cost_usd - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_learn_averages_multiple() {
        let mut p = CostPredictor::new();
        p.learn(&[req_at(8, 0, 1.0), req_at(8, 0, 3.0)]);
        let avg = p.matrix.get(8, 0).avg_cost_usd;
        assert!((avg - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_learn_different_cells() {
        let mut p = CostPredictor::new();
        p.learn(&[req_at(9, 0, 5.0), req_at(9, 1, 2.0)]); // Mon, Tue
        assert!((p.matrix.get(9, 0).avg_cost_usd - 5.0).abs() < 1e-9);
        assert!((p.matrix.get(9, 1).avg_cost_usd - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_count_increments() {
        let mut p = CostPredictor::new();
        p.learn(&[req_at(6, 0, 1.0), req_at(6, 0, 2.0), req_at(6, 0, 3.0)]);
        assert_eq!(p.matrix.get(6, 0).sample_count, 3);
    }

    #[test]
    fn test_learn_additive_across_calls() {
        let mut p = CostPredictor::new();
        p.learn(&[req_at(3, 2, 4.0)]); // Wed
        p.learn(&[req_at(3, 2, 6.0)]);
        let avg = p.matrix.get(3, 2).avg_cost_usd;
        assert!((avg - 5.0).abs() < 1e-9);
    }

    // --- confidence ---

    #[test]
    fn test_confidence_zero_samples() {
        let p = CostPredictor::new();
        assert_eq!(p.confidence(0, 0), 0.0);
    }

    #[test]
    fn test_confidence_partial() {
        let mut p = CostPredictor::new();
        for _ in 0..5 {
            p.learn(&[req_at(0, 0, 1.0)]);
        }
        let c = p.confidence(0, 0);
        assert!((c - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_full_at_ten() {
        let mut p = CostPredictor::new();
        for _ in 0..10 {
            p.learn(&[req_at(0, 0, 1.0)]);
        }
        assert!((p.confidence(0, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_capped_at_one() {
        let mut p = CostPredictor::new();
        for _ in 0..100 {
            p.learn(&[req_at(0, 0, 1.0)]);
        }
        assert!((p.confidence(0, 0) - 1.0).abs() < 1e-9);
    }

    // --- predict_next_n_hours ---

    #[test]
    fn test_predict_n_hours_length() {
        let p = CostPredictor::new();
        let preds = p.predict_next_n_hours(24);
        assert_eq!(preds.len(), 24);
    }

    #[test]
    fn test_predict_n_hours_ascending_timestamps() {
        let p = CostPredictor::new();
        let preds = p.predict_next_n_hours(48);
        for w in preds.windows(2) {
            assert!(w[1].0 > w[0].0);
        }
    }

    #[test]
    fn test_predict_n_hours_zero() {
        let p = CostPredictor::new();
        assert!(p.predict_next_n_hours(0).is_empty());
    }

    // --- WeeklyHeatmap ---

    #[test]
    fn test_heatmap_render_contains_day_labels() {
        let p = CostPredictor::new();
        let heatmap = p.weekly_heatmap();
        let ascii = heatmap.render_ascii();
        for label in &["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] {
            assert!(ascii.contains(label), "missing {label}");
        }
    }

    #[test]
    fn test_heatmap_render_contains_hour_header() {
        let p = CostPredictor::new();
        let ascii = p.weekly_heatmap().render_ascii();
        // Header should contain "0" and "23"
        assert!(ascii.contains('0'));
    }

    #[test]
    fn test_heatmap_all_zero_renders_dots() {
        let p = CostPredictor::new();
        let ascii = p.weekly_heatmap().render_ascii();
        // All zeros → all dots (intensity char for 0 / max=1 → ratio<0.1 → '.')
        assert!(ascii.contains('.'));
    }

    #[test]
    fn test_heatmap_high_value_uses_block() {
        let mut p = CostPredictor::new();
        // Fill Monday hour 0 with a high cost to force a peak block character.
        for _ in 0..15 {
            p.learn(&[req_at(0, 0, 100.0)]);
        }
        let ascii = p.weekly_heatmap().render_ascii();
        assert!(ascii.contains('\u{25A0}') || ascii.contains('\u{2588}'));
    }

    #[test]
    fn test_predict_next_hour_non_negative() {
        let p = CostPredictor::new();
        assert!(p.predict_next_hour() >= 0.0);
    }
}
