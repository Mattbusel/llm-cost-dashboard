//! Predict future costs using linear regression on token usage patterns.

use std::collections::HashMap;

/// A single observed usage data point.
#[derive(Debug, Clone)]
pub struct UsageDataPoint {
    /// Unix epoch milliseconds when the request occurred.
    pub timestamp_ms: u64,
    /// Number of input tokens in the request.
    pub input_tokens: u64,
    /// Number of output tokens in the response.
    pub output_tokens: u64,
    /// Model identifier.
    pub model: String,
    /// Actual cost in USD for this request.
    pub cost_usd: f64,
}

/// Ordinary least squares linear model: `y = slope * x + intercept`.
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Rate of change.
    pub slope: f64,
    /// Y-intercept.
    pub intercept: f64,
    /// Coefficient of determination (0..=1).
    pub r_squared: f64,
}

/// Fit an OLS line to paired `(x, y)` observations.
/// Returns a flat line at the mean if there is insufficient variance in `x`.
pub fn ols_fit(x: &[f64], y: &[f64]) -> LinearModel {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let n = x.len() as f64;
    if n == 0.0 {
        return LinearModel { slope: 0.0, intercept: 0.0, r_squared: 0.0 };
    }

    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let ss_xx: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
    let ss_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean)).sum();
    let ss_yy: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    if ss_xx.abs() < f64::EPSILON {
        return LinearModel { slope: 0.0, intercept: y_mean, r_squared: 0.0 };
    }

    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;

    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();

    let r_squared = if ss_yy.abs() < f64::EPSILON {
        1.0
    } else {
        1.0 - ss_res / ss_yy
    };

    LinearModel { slope, intercept, r_squared }
}

/// Trend summary for a cost series.
#[derive(Debug, Clone)]
pub struct TrendInfo {
    /// Estimated daily cost change (USD/day).
    pub slope_per_day: f64,
    /// `"increasing"`, `"decreasing"`, or `"stable"`.
    pub trend_direction: String,
    /// R² of the fitted model used to derive the trend.
    pub confidence: f64,
}

/// Predictor that fits OLS models to historical usage and extrapolates.
pub struct CostPredictor {
    /// All observed data points.
    pub history: Vec<UsageDataPoint>,
    /// How many trailing days are considered "recent" for predictions.
    pub window_days: u32,
}

impl CostPredictor {
    /// Create a predictor with the given look-back window.
    pub fn new(window_days: u32) -> Self {
        Self {
            history: Vec::new(),
            window_days,
        }
    }

    /// Add a new observation to the history.
    pub fn add_observation(&mut self, dp: UsageDataPoint) {
        self.history.push(dp);
    }

    /// Return references to data points that fall within the last `window_days`.
    pub fn recent_data(&self, now_ms: u64) -> Vec<&UsageDataPoint> {
        let cutoff = now_ms.saturating_sub(self.window_days as u64 * 86_400_000);
        self.history.iter().filter(|dp| dp.timestamp_ms >= cutoff).collect()
    }

    /// Aggregate recent data into (day_index, daily_cost) pairs and fit OLS.
    fn fit_daily_model(&self, now_ms: u64) -> (LinearModel, f64) {
        let recent = self.recent_data(now_ms);
        if recent.is_empty() {
            return (LinearModel { slope: 0.0, intercept: 0.0, r_squared: 0.0 }, 0.0);
        }

        // Bucket by day relative to the earliest point in the window.
        let min_ts = recent.iter().map(|dp| dp.timestamp_ms).min().unwrap_or(now_ms);
        let mut daily: HashMap<i64, f64> = HashMap::new();
        for dp in &recent {
            let day = ((dp.timestamp_ms - min_ts) / 86_400_000) as i64;
            *daily.entry(day).or_insert(0.0) += dp.cost_usd;
        }

        let mut days: Vec<i64> = daily.keys().cloned().collect();
        days.sort_unstable();
        let x: Vec<f64> = days.iter().map(|&d| d as f64).collect();
        let y: Vec<f64> = days.iter().map(|d| daily[d]).collect();

        let base_day = ((now_ms - min_ts) / 86_400_000) as f64;
        (ols_fit(&x, &y), base_day)
    }

    /// Predict daily costs for the next `days_ahead` days.
    /// Returns a vector of `(timestamp_ms, predicted_cost_usd)`.
    pub fn predict_daily_cost(&self, now_ms: u64, days_ahead: u32) -> Vec<(u64, f64)> {
        let (model, base_day) = self.fit_daily_model(now_ms);
        (1..=days_ahead)
            .map(|d| {
                let x = base_day + d as f64;
                let cost = (model.slope * x + model.intercept).max(0.0);
                let ts = now_ms + d as u64 * 86_400_000;
                (ts, cost)
            })
            .collect()
    }

    /// Estimate the total cost over the next 30 days.
    pub fn predict_monthly_cost(&self, now_ms: u64) -> f64 {
        self.predict_daily_cost(now_ms, 30)
            .iter()
            .map(|(_, c)| c)
            .sum()
    }

    /// Describe the cost trend over the recent window.
    pub fn cost_trend(&self, now_ms: u64) -> TrendInfo {
        let (model, _) = self.fit_daily_model(now_ms);
        let direction = if model.slope > 0.01 {
            "increasing"
        } else if model.slope < -0.01 {
            "decreasing"
        } else {
            "stable"
        };
        TrendInfo {
            slope_per_day: model.slope,
            trend_direction: direction.to_string(),
            confidence: model.r_squared,
        }
    }

    /// Compute the Z-score of `dp.cost_usd` relative to the recent window's distribution.
    /// Returns `0.0` if there is insufficient data.
    pub fn anomaly_score(&self, dp: &UsageDataPoint, now_ms: u64) -> f64 {
        let recent = self.recent_data(now_ms);
        if recent.len() < 2 {
            return 0.0;
        }
        let n = recent.len() as f64;
        let mean = recent.iter().map(|d| d.cost_usd).sum::<f64>() / n;
        let variance = recent.iter().map(|d| (d.cost_usd - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        if std < f64::EPSILON {
            return 0.0;
        }
        (dp.cost_usd - mean) / std
    }

    /// Sum costs per model over the recent window.
    pub fn model_cost_breakdown(&self, now_ms: u64) -> HashMap<String, f64> {
        let mut breakdown: HashMap<String, f64> = HashMap::new();
        for dp in self.recent_data(now_ms) {
            *breakdown.entry(dp.model.clone()).or_insert(0.0) += dp.cost_usd;
        }
        breakdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dp(ts_ms: u64, cost: f64, model: &str) -> UsageDataPoint {
        UsageDataPoint {
            timestamp_ms: ts_ms,
            input_tokens: 100,
            output_tokens: 50,
            model: model.to_string(),
            cost_usd: cost,
        }
    }

    #[test]
    fn test_ols_fit_known_data() {
        // y = 2x + 1
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let model = ols_fit(&x, &y);
        assert!((model.slope - 2.0).abs() < 1e-9);
        assert!((model.intercept - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_r_squared_perfect_fit() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let model = ols_fit(&x, &y);
        assert!((model.r_squared - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_prediction_grows_with_positive_slope() {
        let mut predictor = CostPredictor::new(30);
        let now_ms: u64 = 30 * 86_400_000;
        for day in 0..10u64 {
            predictor.add_observation(dp(day * 86_400_000, day as f64 * 1.0, "gpt-4"));
        }
        let preds = predictor.predict_daily_cost(now_ms, 5);
        assert_eq!(preds.len(), 5);
        // Each successive prediction should be >= the previous (positive slope).
        for i in 1..preds.len() {
            assert!(preds[i].1 >= preds[i - 1].1 - 1e-9);
        }
    }

    #[test]
    fn test_model_cost_breakdown_sums_correct() {
        let mut predictor = CostPredictor::new(30);
        let now_ms: u64 = 86_400_000;
        predictor.add_observation(dp(1000, 1.0, "gpt-4"));
        predictor.add_observation(dp(2000, 2.0, "gpt-4"));
        predictor.add_observation(dp(3000, 0.5, "claude"));
        let breakdown = predictor.model_cost_breakdown(now_ms);
        assert!((breakdown["gpt-4"] - 3.0).abs() < 1e-9);
        assert!((breakdown["claude"] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_anomaly_score_high_for_outlier() {
        let mut predictor = CostPredictor::new(30);
        let now_ms: u64 = 86_400_000;
        // Normal cost ~1.0
        for i in 0..10u64 {
            predictor.add_observation(dp(i * 1000, 1.0, "gpt-4"));
        }
        let outlier = dp(11_000, 100.0, "gpt-4");
        let score = predictor.anomaly_score(&outlier, now_ms);
        // Score should be substantially above 0 for an extreme outlier.
        assert!(score > 5.0, "expected high anomaly score, got {}", score);
    }
}
