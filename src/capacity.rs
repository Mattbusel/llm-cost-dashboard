//! # Capacity Planning
//!
//! Usage forecasting via OLS linear regression and Holt's double exponential
//! smoothing, with budget runway calculations and ASCII capacity reports.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::capacity::{CapacityPlanner, UsagePoint};
//!
//! let mut planner = CapacityPlanner::new();
//! let day = 86_400_u64;
//! for i in 0..14_u64 {
//!     planner.add_observation(UsagePoint {
//!         timestamp: i * day,
//!         tokens: 1_000_000 + i * 100_000,
//!         cost: 10.0 + i as f64 * 1.0,
//!     });
//! }
//! let forecast = planner.linear_forecast(7);
//! assert!(!forecast.forecast_tokens.is_empty());
//! ```

use std::collections::HashMap;

// ── UsagePoint ────────────────────────────────────────────────────────────────

/// A single observed usage data point.
#[derive(Debug, Clone)]
pub struct UsagePoint {
    /// Unix timestamp (seconds) of the observation.
    pub timestamp: u64,
    /// Number of tokens consumed.
    pub tokens: u64,
    /// Cost in USD.
    pub cost: f64,
}

// ── CapacityForecast ──────────────────────────────────────────────────────────

/// Result of a capacity forecast.
#[derive(Debug, Clone)]
pub struct CapacityForecast {
    /// Projected (timestamp, token_count) pairs for each forecast day.
    pub forecast_tokens: Vec<(u64, f64)>,
    /// Projected (timestamp, cost_usd) pairs for each forecast day.
    pub forecast_cost: Vec<(u64, f64)>,
    /// 95% confidence interval half-width for cost: `(lower_bound, upper_bound)`
    /// applied symmetrically around each forecasted cost value.
    pub confidence_interval: (f64, f64),
    /// Estimated number of days until daily cost exceeds the budget, if a
    /// budget was provided to the forecasting call.
    pub days_until_budget_exhausted: Option<f64>,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Aggregate observations into daily buckets.
/// Returns `(day_index, total_tokens, total_cost)` sorted by day.
fn daily_aggregates(points: &[UsagePoint]) -> Vec<(f64, f64, f64)> {
    const DAY: u64 = 86_400;
    let mut buckets: HashMap<u64, (u64, f64)> = HashMap::new();
    for p in points {
        let day = p.timestamp / DAY;
        let e = buckets.entry(day).or_insert((0, 0.0));
        e.0 += p.tokens;
        e.1 += p.cost;
    }
    let mut days: Vec<u64> = buckets.keys().copied().collect();
    days.sort_unstable();

    let min_day = days.first().copied().unwrap_or(0);
    days.iter()
        .map(|&d| {
            let (tok, cost) = buckets[&d];
            ((d - min_day) as f64, tok as f64, cost)
        })
        .collect()
}

/// OLS slope and intercept for `(x, y)` pairs.
/// Returns `(slope, intercept)`.
fn ols(pairs: &[(f64, f64)]) -> (f64, f64) {
    let n = pairs.len() as f64;
    if n < 2.0 {
        return (0.0, pairs.first().map(|p| p.1).unwrap_or(0.0));
    }
    let mean_x: f64 = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y: f64 = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let ss_xy: f64 = pairs
        .iter()
        .map(|p| (p.0 - mean_x) * (p.1 - mean_y))
        .sum();
    let ss_xx: f64 = pairs.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    if ss_xx.abs() < 1e-12 {
        return (0.0, mean_y);
    }
    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

/// Residual standard deviation for OLS predictions.
fn residual_std(pairs: &[(f64, f64)], slope: f64, intercept: f64) -> f64 {
    let n = pairs.len();
    if n < 2 {
        return 0.0;
    }
    let ss_res: f64 = pairs
        .iter()
        .map(|p| {
            let pred = slope * p.0 + intercept;
            (p.1 - pred).powi(2)
        })
        .sum();
    (ss_res / (n - 1) as f64).sqrt()
}

// ── CapacityPlanner ───────────────────────────────────────────────────────────

/// Collects usage observations and produces capacity planning forecasts.
#[derive(Default)]
pub struct CapacityPlanner {
    observations: Vec<UsagePoint>,
}

impl CapacityPlanner {
    /// Create a new, empty planner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new usage observation.
    pub fn add_observation(&mut self, point: UsagePoint) {
        self.observations.push(point);
    }

    /// OLS linear regression forecast.
    ///
    /// Aggregates observations into daily buckets, fits a straight line through
    /// `(day_index, value)`, then projects forward `horizon_days` days.
    /// The 95% confidence interval is `±1.96 * residual_std`.
    pub fn linear_forecast(&self, horizon_days: u32) -> CapacityForecast {
        let daily = daily_aggregates(&self.observations);
        if daily.is_empty() {
            return empty_forecast();
        }

        // Last observed day timestamp.
        const DAY: u64 = 86_400;
        let last_day_idx = daily.last().map(|d| d.0).unwrap_or(0.0);
        let last_ts = self
            .observations
            .iter()
            .map(|p| p.timestamp)
            .max()
            .unwrap_or(0);

        // Build (x, y) pairs for OLS.
        let tok_pairs: Vec<(f64, f64)> = daily.iter().map(|d| (d.0, d.1)).collect();
        let cost_pairs: Vec<(f64, f64)> = daily.iter().map(|d| (d.0, d.2)).collect();

        let (tok_slope, tok_intercept) = ols(&tok_pairs);
        let (cost_slope, cost_intercept) = ols(&cost_pairs);

        let tok_std = residual_std(&tok_pairs, tok_slope, tok_intercept);
        let cost_std = residual_std(&cost_pairs, cost_slope, cost_intercept);
        let ci_half = 1.96 * cost_std;

        let mut forecast_tokens = Vec::new();
        let mut forecast_cost = Vec::new();

        for h in 1..=horizon_days {
            let x = last_day_idx + h as f64;
            let ts = last_ts + h as u64 * DAY;
            let tok = (tok_slope * x + tok_intercept).max(0.0);
            let cost = (cost_slope * x + cost_intercept).max(0.0);
            forecast_tokens.push((ts, tok));
            forecast_cost.push((ts, cost));
        }

        // Suppress unused warning.
        let _ = tok_std;

        CapacityForecast {
            forecast_tokens,
            forecast_cost,
            confidence_interval: (-ci_half, ci_half),
            days_until_budget_exhausted: None,
        }
    }

    /// Holt's double exponential smoothing (level + trend) forecast.
    ///
    /// `alpha` controls the smoothing of the level (0 < alpha < 1); the trend
    /// smoothing coefficient is derived as `alpha * 0.3`.
    pub fn exponential_smoothing_forecast(
        &self,
        horizon_days: u32,
        alpha: f64,
    ) -> CapacityForecast {
        let daily = daily_aggregates(&self.observations);
        if daily.len() < 2 {
            return empty_forecast();
        }

        const DAY: u64 = 86_400;
        let last_ts = self
            .observations
            .iter()
            .map(|p| p.timestamp)
            .max()
            .unwrap_or(0);

        let beta = (alpha * 0.3).min(0.9).max(0.01);
        let alpha = alpha.clamp(0.01, 0.99);

        // Smooth tokens and cost independently.
        let (tok_level, tok_trend) = holt_smooth(&daily.iter().map(|d| d.1).collect::<Vec<_>>(), alpha, beta);
        let (cost_level, cost_trend) = holt_smooth(&daily.iter().map(|d| d.2).collect::<Vec<_>>(), alpha, beta);

        let mut forecast_tokens = Vec::new();
        let mut forecast_cost = Vec::new();

        for h in 1..=horizon_days {
            let ts = last_ts + h as u64 * DAY;
            let tok = (tok_level + tok_trend * h as f64).max(0.0);
            let cost = (cost_level + cost_trend * h as f64).max(0.0);
            forecast_tokens.push((ts, tok));
            forecast_cost.push((ts, cost));
        }

        // Rough CI based on slope uncertainty.
        let ci_half = cost_trend.abs() * 0.3;

        CapacityForecast {
            forecast_tokens,
            forecast_cost,
            confidence_interval: (-ci_half, ci_half),
            days_until_budget_exhausted: None,
        }
    }

    /// Estimate how many days until the daily cost trend exceeds `daily_budget`.
    ///
    /// Uses OLS linear regression on daily costs.  Returns `None` if the trend
    /// is flat or declining, or if there are insufficient data points.
    pub fn budget_runway(&self, daily_budget: f64) -> Option<f64> {
        let daily = daily_aggregates(&self.observations);
        if daily.len() < 2 {
            return None;
        }
        let cost_pairs: Vec<(f64, f64)> = daily.iter().map(|d| (d.0, d.2)).collect();
        let (slope, intercept) = ols(&cost_pairs);

        if slope <= 0.0 {
            // Cost is flat or decreasing — budget will not be exhausted.
            return None;
        }

        let last_x = daily.last().map(|d| d.0).unwrap_or(0.0);
        let current_cost = slope * last_x + intercept;

        if current_cost >= daily_budget {
            // Already over budget.
            return Some(0.0);
        }

        // days_until: solve slope * (last_x + d) + intercept = budget
        let days = (daily_budget - current_cost) / slope;
        if days < 0.0 {
            Some(0.0)
        } else {
            Some(days)
        }
    }

    /// Return which hours of day (0–23) have the highest average token usage.
    ///
    /// Hours are sorted descending by average token count.  Requires
    /// sub-day-resolution timestamps to be meaningful.
    pub fn peak_usage_hours(&self) -> Vec<u8> {
        let mut hour_tokens: HashMap<u8, (u64, u64)> = HashMap::new(); // hour → (total_tokens, count)
        for p in &self.observations {
            let hour = ((p.timestamp % 86_400) / 3600) as u8;
            let e = hour_tokens.entry(hour).or_insert((0, 0));
            e.0 += p.tokens;
            e.1 += 1;
        }

        let mut hourly_avg: Vec<(u8, f64)> = hour_tokens
            .iter()
            .map(|(&h, &(total, count))| (h, total as f64 / count.max(1) as f64))
            .collect();
        hourly_avg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        hourly_avg.into_iter().map(|(h, _)| h).collect()
    }

    /// Week-over-week growth rate in total token usage.
    ///
    /// Returns `(week_2_tokens - week_1_tokens) / week_1_tokens` where week 1
    /// is the oldest 7 days and week 2 is the most recent 7 days.
    /// Returns `None` if there are fewer than 2 weeks of data.
    pub fn growth_rate(&self) -> Option<f64> {
        const WEEK: u64 = 7 * 86_400;
        if self.observations.is_empty() {
            return None;
        }
        let min_ts = self.observations.iter().map(|p| p.timestamp).min().unwrap_or(0);
        let max_ts = self.observations.iter().map(|p| p.timestamp).max().unwrap_or(0);

        if max_ts - min_ts < WEEK * 2 {
            return None;
        }

        let week1_end = min_ts + WEEK;
        let week2_start = max_ts - WEEK;

        let week1_tokens: u64 = self
            .observations
            .iter()
            .filter(|p| p.timestamp < week1_end)
            .map(|p| p.tokens)
            .sum();
        let week2_tokens: u64 = self
            .observations
            .iter()
            .filter(|p| p.timestamp >= week2_start)
            .map(|p| p.tokens)
            .sum();

        if week1_tokens == 0 {
            return None;
        }

        Some((week2_tokens as f64 - week1_tokens as f64) / week1_tokens as f64)
    }

    /// Generate a formatted capacity planning report with an ASCII token chart.
    pub fn to_report(&self, horizon_days: u32) -> String {
        let forecast = self.linear_forecast(horizon_days);
        let growth = self
            .growth_rate()
            .map(|r| format!("{:+.1}%", r * 100.0))
            .unwrap_or_else(|| "N/A".to_string());
        let runway = self
            .budget_runway(100.0)
            .map(|d| format!("{:.1} days", d))
            .unwrap_or_else(|| "N/A".to_string());
        let peak_hours = self.peak_usage_hours();
        let peak_str: Vec<String> = peak_hours.iter().take(3).map(|h| format!("{:02}:00", h)).collect();

        let divider = "─".repeat(60);
        let mut out = String::new();
        out.push_str(&format!("{}\n", divider));
        out.push_str(&format!("{:^60}\n", "CAPACITY PLANNING REPORT"));
        out.push_str(&format!("{}\n", divider));
        out.push_str(&format!("Observations     : {}\n", self.observations.len()));
        out.push_str(&format!("Forecast horizon : {} days\n", horizon_days));
        out.push_str(&format!("WoW growth rate  : {}\n", growth));
        out.push_str(&format!("Budget runway    : {}\n", runway));
        out.push_str(&format!(
            "Peak hours       : {}\n",
            if peak_str.is_empty() { "N/A".to_string() } else { peak_str.join(", ") }
        ));
        out.push_str(&format!("CI (cost ±)      : {:.4}\n", forecast.confidence_interval.1));
        out.push_str(&format!("{}\n", divider));

        // ASCII bar chart of forecast cost.
        if !forecast.forecast_cost.is_empty() {
            out.push_str("Forecast Cost (USD/day):\n");
            let max_cost = forecast
                .forecast_cost
                .iter()
                .map(|(_, c)| *c)
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1.0);

            for (i, (_, cost)) in forecast.forecast_cost.iter().enumerate() {
                let bar_len = ((cost / max_cost) * 30.0) as usize;
                let bar = "#".repeat(bar_len);
                out.push_str(&format!("  Day {:>3}: {:>8.2} |{}\n", i + 1, cost, bar));
            }
            out.push_str(&format!("{}\n", divider));
        }

        out
    }
}

/// Holt's double exponential smoothing.
/// Returns `(final_level, final_trend)`.
fn holt_smooth(series: &[f64], alpha: f64, beta: f64) -> (f64, f64) {
    if series.is_empty() {
        return (0.0, 0.0);
    }
    if series.len() == 1 {
        return (series[0], 0.0);
    }

    let mut level = series[0];
    let mut trend = series[1] - series[0];

    for &val in &series[1..] {
        let prev_level = level;
        level = alpha * val + (1.0 - alpha) * (level + trend);
        trend = beta * (level - prev_level) + (1.0 - beta) * trend;
    }
    (level, trend)
}

fn empty_forecast() -> CapacityForecast {
    CapacityForecast {
        forecast_tokens: Vec::new(),
        forecast_cost: Vec::new(),
        confidence_interval: (0.0, 0.0),
        days_until_budget_exhausted: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_observations(days: u32, tokens_per_day: u64, cost_per_day: f64) -> Vec<UsagePoint> {
        (0..days)
            .map(|i| UsagePoint {
                timestamp: i as u64 * 86_400,
                tokens: tokens_per_day + i as u64 * 1_000,
                cost: cost_per_day + i as f64 * 0.5,
            })
            .collect()
    }

    #[test]
    fn test_ols_on_known_linear_data() {
        // y = 2*x + 5  →  slope≈2, intercept≈5
        let pairs: Vec<(f64, f64)> = (0..10).map(|x| (x as f64, 2.0 * x as f64 + 5.0)).collect();
        let (slope, intercept) = ols(&pairs);
        assert!((slope - 2.0).abs() < 0.01, "slope={}", slope);
        assert!((intercept - 5.0).abs() < 0.01, "intercept={}", intercept);
    }

    #[test]
    fn test_linear_forecast_produces_data() {
        let mut planner = CapacityPlanner::new();
        for obs in linear_observations(14, 1_000_000, 10.0) {
            planner.add_observation(obs);
        }
        let fc = planner.linear_forecast(7);
        assert_eq!(fc.forecast_tokens.len(), 7);
        assert_eq!(fc.forecast_cost.len(), 7);
        // Values should be positive.
        for (_, tok) in &fc.forecast_tokens {
            assert!(*tok > 0.0, "forecasted tokens should be positive");
        }
    }

    #[test]
    fn test_linear_forecast_increasing_trend() {
        let mut planner = CapacityPlanner::new();
        for i in 0..14_u64 {
            planner.add_observation(UsagePoint {
                timestamp: i * 86_400,
                tokens: 1_000_000 + i * 100_000,
                cost: 10.0 + i as f64,
            });
        }
        let fc = planner.linear_forecast(3);
        // Each day should be higher than the previous.
        for w in fc.forecast_cost.windows(2) {
            assert!(
                w[1].1 >= w[0].1,
                "cost should be non-decreasing: {} >= {}",
                w[1].1,
                w[0].1
            );
        }
    }

    #[test]
    fn test_budget_runway_calculation() {
        let mut planner = CapacityPlanner::new();
        // Cost grows by $1/day, starting at $5/day.
        for i in 0..14_u64 {
            planner.add_observation(UsagePoint {
                timestamp: i * 86_400,
                tokens: 1_000,
                cost: 5.0 + i as f64,
            });
        }
        // At day 13 cost is $18.  Budget = $30 → ~12 more days.
        let runway = planner.budget_runway(30.0);
        assert!(runway.is_some(), "runway should have a value");
        let days = runway.unwrap();
        assert!(days > 0.0 && days < 30.0, "runway={}", days);
    }

    #[test]
    fn test_budget_runway_flat_trend_returns_none() {
        let mut planner = CapacityPlanner::new();
        for i in 0..14_u64 {
            planner.add_observation(UsagePoint {
                timestamp: i * 86_400,
                tokens: 1_000,
                cost: 5.0, // constant — slope = 0
            });
        }
        let runway = planner.budget_runway(100.0);
        assert!(runway.is_none(), "flat trend should return None");
    }

    #[test]
    fn test_growth_rate_insufficient_data() {
        let mut planner = CapacityPlanner::new();
        planner.add_observation(UsagePoint { timestamp: 0, tokens: 1000, cost: 1.0 });
        assert!(planner.growth_rate().is_none());
    }

    #[test]
    fn test_growth_rate_positive() {
        let mut planner = CapacityPlanner::new();
        // Week 1: days 0–6, Week 2: days 14–20.
        for i in 0..7_u64 {
            planner.add_observation(UsagePoint {
                timestamp: i * 86_400,
                tokens: 100_000,
                cost: 1.0,
            });
        }
        for i in 14..21_u64 {
            planner.add_observation(UsagePoint {
                timestamp: i * 86_400,
                tokens: 200_000, // double the usage
                cost: 2.0,
            });
        }
        let rate = planner.growth_rate().expect("should have growth rate");
        assert!(rate > 0.0, "expected positive growth rate, got {}", rate);
    }

    #[test]
    fn test_exponential_smoothing_forecast() {
        let mut planner = CapacityPlanner::new();
        for obs in linear_observations(14, 500_000, 5.0) {
            planner.add_observation(obs);
        }
        let fc = planner.exponential_smoothing_forecast(5, 0.3);
        assert_eq!(fc.forecast_tokens.len(), 5);
        assert_eq!(fc.forecast_cost.len(), 5);
    }

    #[test]
    fn test_to_report_contains_key_sections() {
        let mut planner = CapacityPlanner::new();
        for obs in linear_observations(14, 1_000_000, 10.0) {
            planner.add_observation(obs);
        }
        let report = planner.to_report(7);
        assert!(report.contains("CAPACITY PLANNING REPORT"), "missing title");
        assert!(report.contains("Forecast horizon"), "missing horizon");
        assert!(report.contains("Forecast Cost"), "missing cost chart");
    }

    #[test]
    fn test_peak_usage_hours() {
        let mut planner = CapacityPlanner::new();
        // Add observations at hour 14 (2pm) with high usage.
        for day in 0..7_u64 {
            planner.add_observation(UsagePoint {
                timestamp: day * 86_400 + 14 * 3600, // 14:00
                tokens: 1_000_000,
                cost: 10.0,
            });
            planner.add_observation(UsagePoint {
                timestamp: day * 86_400 + 2 * 3600, // 02:00
                tokens: 10_000,
                cost: 0.1,
            });
        }
        let peaks = planner.peak_usage_hours();
        assert!(!peaks.is_empty());
        assert_eq!(peaks[0], 14, "peak hour should be 14, got {:?}", peaks);
    }
}
