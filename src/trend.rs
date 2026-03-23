//! # Cost Trend Analyzer
//!
//! Detects and describes cost trends over time using OLS linear regression,
//! CUSUM-based changepoint detection, and day-of-week seasonality analysis.
//!
//! ## Quick start
//!
//! ```rust
//! use llm_cost_dashboard::trend::{TrendAnalyzer, TrendPoint};
//! use chrono::Utc;
//!
//! let points = vec![
//!     TrendPoint { timestamp: Utc::now(), cost_usd: 1.0 },
//! ];
//! let result = TrendAnalyzer::fit(&points);
//! println!("slope: {:.4} $/day", result.slope_usd_per_day);
//! ```

use chrono::{DateTime, Datelike, Utc, Weekday};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TrendPoint
// ---------------------------------------------------------------------------

/// A single daily aggregated cost data point.
#[derive(Debug, Clone)]
pub struct TrendPoint {
    /// UTC timestamp for this data point (typically midnight of the day).
    pub timestamp: DateTime<Utc>,
    /// Aggregated cost in USD for this day.
    pub cost_usd: f64,
}

// ---------------------------------------------------------------------------
// TrendDirection
// ---------------------------------------------------------------------------

/// Qualitative direction of a fitted cost trend.
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Costs are increasing over time.
    Rising {
        /// Average daily cost increase in USD.
        daily_increase_usd: f64,
    },
    /// Costs are decreasing over time.
    Falling {
        /// Average daily cost decrease in USD (positive value).
        daily_decrease_usd: f64,
    },
    /// Costs are approximately flat (slope within ±0.001 USD/day).
    Flat,
}

// ---------------------------------------------------------------------------
// TrendResult
// ---------------------------------------------------------------------------

/// Result of an OLS linear regression fit over [`TrendPoint`]s.
#[derive(Debug, Clone)]
pub struct TrendResult {
    /// Change in cost per day (USD).  Positive = rising, negative = falling.
    pub slope_usd_per_day: f64,
    /// Regression intercept (USD at `t = 0`).
    pub intercept: f64,
    /// Coefficient of determination (R²).  Range `[0.0, 1.0]`.
    pub r_squared: f64,
    /// Qualitative trend direction derived from `slope_usd_per_day`.
    pub trend_direction: TrendDirection,
}

// ---------------------------------------------------------------------------
// SeasonalityReport
// ---------------------------------------------------------------------------

/// Day-of-week average cost breakdown.
#[derive(Debug, Clone)]
pub struct SeasonalityReport {
    /// Average cost per day-of-week: `day_name -> avg_usd`.
    ///
    /// Day names are `"Mon"`, `"Tue"`, `"Wed"`, `"Thu"`, `"Fri"`, `"Sat"`, `"Sun"`.
    pub day_averages: HashMap<String, f64>,
    /// Day of the week with the highest average spend.
    pub highest_spend_day: String,
    /// Day of the week with the lowest average spend.
    pub lowest_spend_day: String,
}

// ---------------------------------------------------------------------------
// TrendAnalyzer
// ---------------------------------------------------------------------------

/// Stateless collection of trend analysis functions.
pub struct TrendAnalyzer;

impl TrendAnalyzer {
    /// Fit an OLS linear regression over `points`.
    ///
    /// The independent variable is elapsed days from the earliest timestamp.
    /// Returns [`TrendResult`] with slope, intercept, R², and direction.
    ///
    /// If fewer than 2 points are provided, slope is `0.0`, intercept is the
    /// cost of the single point (or `0.0` for empty), and R² is `0.0`.
    pub fn fit(points: &[TrendPoint]) -> TrendResult {
        if points.len() < 2 {
            let intercept = points.first().map_or(0.0, |p| p.cost_usd);
            return TrendResult {
                slope_usd_per_day: 0.0,
                intercept,
                r_squared: 0.0,
                trend_direction: TrendDirection::Flat,
            };
        }

        // Convert timestamps to elapsed days from the minimum timestamp.
        let t_min = points
            .iter()
            .map(|p| p.timestamp)
            .min()
            .expect("non-empty slice");

        let xs: Vec<f64> = points
            .iter()
            .map(|p| {
                let delta = p.timestamp.signed_duration_since(t_min);
                delta.num_seconds() as f64 / 86_400.0
            })
            .collect();

        let ys: Vec<f64> = points.iter().map(|p| p.cost_usd).collect();

        let n = xs.len() as f64;
        let x_mean = xs.iter().sum::<f64>() / n;
        let y_mean = ys.iter().sum::<f64>() / n;

        let ss_xy: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();

        let ss_xx: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();

        let slope = if ss_xx.abs() < f64::EPSILON {
            0.0
        } else {
            ss_xy / ss_xx
        };
        let intercept = y_mean - slope * x_mean;

        // R² = 1 - SS_res / SS_tot
        let ss_res: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| {
                let y_hat = slope * x + intercept;
                (y - y_hat).powi(2)
            })
            .sum();
        let ss_tot: f64 = ys.iter().map(|y| (y - y_mean).powi(2)).sum();
        let r_squared = if ss_tot.abs() < f64::EPSILON {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };
        let r_squared = r_squared.max(0.0).min(1.0);

        let flat_threshold = 0.001;
        let trend_direction = if slope > flat_threshold {
            TrendDirection::Rising {
                daily_increase_usd: slope,
            }
        } else if slope < -flat_threshold {
            TrendDirection::Falling {
                daily_decrease_usd: -slope,
            }
        } else {
            TrendDirection::Flat
        };

        TrendResult {
            slope_usd_per_day: slope,
            intercept,
            r_squared,
            trend_direction,
        }
    }

    /// Detect changepoints using the CUSUM algorithm.
    ///
    /// Accumulates the deviation of each point from the mean.  When the
    /// cumulative sum exceeds `sensitivity * std_dev`, a changepoint is
    /// flagged at that timestamp and the accumulator is reset.
    ///
    /// Returns a (possibly empty) list of UTC timestamps where cost behaviour
    /// changed significantly.
    ///
    /// If fewer than 3 points are provided, returns an empty list.
    pub fn detect_changepoints(points: &[TrendPoint], sensitivity: f64) -> Vec<DateTime<Utc>> {
        if points.len() < 3 {
            return vec![];
        }

        let costs: Vec<f64> = points.iter().map(|p| p.cost_usd).collect();
        let n = costs.len() as f64;
        let mean = costs.iter().sum::<f64>() / n;
        let variance = costs.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev < f64::EPSILON {
            return vec![];
        }

        let threshold = sensitivity * std_dev;
        let mut cusum = 0.0_f64;
        let mut changepoints = Vec::new();

        for (i, point) in points.iter().enumerate() {
            cusum += point.cost_usd - mean;
            if cusum.abs() > threshold {
                changepoints.push(point.timestamp);
                // Skip to avoid flagging every subsequent point in a sustained shift.
                cusum = 0.0;
                // Skip ahead a bit to avoid immediate re-triggering.
                if i + 1 < points.len() {
                    // reset handled by the loop continuing with cusum=0
                }
            }
        }

        changepoints
    }

    /// Compute day-of-week seasonality from `points`.
    ///
    /// Returns a [`SeasonalityReport`] with average cost per weekday and the
    /// highest/lowest spend day names.
    ///
    /// If `points` is empty, all averages are `0.0` and both day names are
    /// `"Mon"`.
    pub fn seasonality(points: &[TrendPoint]) -> SeasonalityReport {
        let day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

        let mut sums: HashMap<String, f64> = day_names
            .iter()
            .map(|&d| (d.to_string(), 0.0))
            .collect();
        let mut counts: HashMap<String, u64> = day_names
            .iter()
            .map(|&d| (d.to_string(), 0))
            .collect();

        for point in points {
            let day_name = weekday_name(point.timestamp.weekday());
            *sums.entry(day_name.clone()).or_insert(0.0) += point.cost_usd;
            *counts.entry(day_name).or_insert(0) += 1;
        }

        let day_averages: HashMap<String, f64> = day_names
            .iter()
            .map(|&d| {
                let cnt = *counts.get(d).unwrap_or(&0);
                let sum = *sums.get(d).unwrap_or(&0.0);
                let avg = if cnt == 0 { 0.0 } else { sum / cnt as f64 };
                (d.to_string(), avg)
            })
            .collect();

        let get_avg = |day: &str| -> f64 {
            day_averages.get(day).copied().unwrap_or(0.0)
        };

        let highest_spend_day = day_names
            .iter()
            .max_by(|a, b| {
                get_avg(a).partial_cmp(&get_avg(b)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(&"Mon")
            .to_string();

        let lowest_spend_day = day_names
            .iter()
            .min_by(|a, b| {
                get_avg(a).partial_cmp(&get_avg(b)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(&"Mon")
            .to_string();

        SeasonalityReport {
            day_averages,
            highest_spend_day,
            lowest_spend_day,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn weekday_name(wd: Weekday) -> String {
    match wd {
        Weekday::Mon => "Mon",
        Weekday::Tue => "Tue",
        Weekday::Wed => "Wed",
        Weekday::Thu => "Thu",
        Weekday::Fri => "Fri",
        Weekday::Sat => "Sat",
        Weekday::Sun => "Sun",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Tests (15+)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    fn day(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn point(y: i32, m: u32, d: u32, cost: f64) -> TrendPoint {
        TrendPoint {
            timestamp: day(y, m, d),
            cost_usd: cost,
        }
    }

    // ---- TrendAnalyzer::fit ----

    #[test]
    fn fit_empty_returns_zero_slope() {
        let result = TrendAnalyzer::fit(&[]);
        assert!((result.slope_usd_per_day - 0.0).abs() < f64::EPSILON);
        assert!((result.intercept - 0.0).abs() < f64::EPSILON);
        assert_eq!(result.trend_direction, TrendDirection::Flat);
    }

    #[test]
    fn fit_single_point_returns_zero_slope() {
        let pts = vec![point(2024, 1, 1, 5.0)];
        let result = TrendAnalyzer::fit(&pts);
        assert!((result.slope_usd_per_day - 0.0).abs() < f64::EPSILON);
        assert!((result.intercept - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fit_perfectly_rising_trend() {
        // Cost increases by $1/day for 5 days: slope ≈ 1.0.
        let pts: Vec<TrendPoint> = (0..5)
            .map(|i| TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: i as f64 + 1.0,
            })
            .collect();
        let result = TrendAnalyzer::fit(&pts);
        assert!((result.slope_usd_per_day - 1.0).abs() < 1e-9);
        assert!(matches!(result.trend_direction, TrendDirection::Rising { .. }));
    }

    #[test]
    fn fit_perfectly_falling_trend() {
        let pts: Vec<TrendPoint> = (0..5)
            .map(|i| TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: 10.0 - i as f64,
            })
            .collect();
        let result = TrendAnalyzer::fit(&pts);
        assert!(result.slope_usd_per_day < 0.0);
        assert!(matches!(result.trend_direction, TrendDirection::Falling { .. }));
    }

    #[test]
    fn fit_flat_trend() {
        let pts: Vec<TrendPoint> = (0..5)
            .map(|i| TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: 5.0,
            })
            .collect();
        let result = TrendAnalyzer::fit(&pts);
        assert_eq!(result.trend_direction, TrendDirection::Flat);
    }

    #[test]
    fn fit_r_squared_perfect_linear() {
        let pts: Vec<TrendPoint> = (0..5)
            .map(|i| TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: i as f64 * 2.0,
            })
            .collect();
        let result = TrendAnalyzer::fit(&pts);
        assert!((result.r_squared - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fit_r_squared_in_range() {
        let pts = vec![
            point(2024, 1, 1, 1.0),
            point(2024, 1, 2, 5.0),
            point(2024, 1, 3, 2.0),
            point(2024, 1, 4, 8.0),
        ];
        let result = TrendAnalyzer::fit(&pts);
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    }

    #[test]
    fn fit_rising_daily_increase_matches_slope() {
        let pts: Vec<TrendPoint> = (0..4)
            .map(|i| TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: i as f64 * 3.0,
            })
            .collect();
        let result = TrendAnalyzer::fit(&pts);
        if let TrendDirection::Rising { daily_increase_usd } = result.trend_direction {
            assert!((daily_increase_usd - 3.0).abs() < 1e-9);
        } else {
            panic!("expected Rising");
        }
    }

    // ---- detect_changepoints ----

    #[test]
    fn detect_changepoints_empty() {
        assert!(TrendAnalyzer::detect_changepoints(&[], 1.0).is_empty());
    }

    #[test]
    fn detect_changepoints_too_few_points() {
        let pts = vec![point(2024, 1, 1, 1.0), point(2024, 1, 2, 2.0)];
        assert!(TrendAnalyzer::detect_changepoints(&pts, 1.0).is_empty());
    }

    #[test]
    fn detect_changepoints_flat_no_changepoints() {
        let pts: Vec<TrendPoint> = (0..10)
            .map(|i| TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: 5.0,
            })
            .collect();
        // All values equal → std_dev=0 → no changepoints
        let cps = TrendAnalyzer::detect_changepoints(&pts, 1.0);
        assert!(cps.is_empty());
    }

    #[test]
    fn detect_changepoints_sudden_spike() {
        // 10 days of $1, then a $100 spike, then $1 again.
        let mut pts: Vec<TrendPoint> = (0..10)
            .map(|i| TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: 1.0,
            })
            .collect();
        pts.push(TrendPoint {
            timestamp: day(2024, 1, 11),
            cost_usd: 100.0,
        });
        for i in 12..20 {
            pts.push(TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: 1.0,
            });
        }
        let cps = TrendAnalyzer::detect_changepoints(&pts, 1.0);
        assert!(!cps.is_empty(), "should detect at least one changepoint");
    }

    // ---- seasonality ----

    #[test]
    fn seasonality_empty_all_zeros() {
        let report = TrendAnalyzer::seasonality(&[]);
        for (_, avg) in &report.day_averages {
            assert!((avg - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn seasonality_single_monday() {
        let pts = vec![TrendPoint {
            timestamp: day(2024, 1, 1), // 2024-01-01 is a Monday
            cost_usd: 10.0,
        }];
        let report = TrendAnalyzer::seasonality(&pts);
        assert!((report.day_averages["Mon"] - 10.0).abs() < f64::EPSILON);
        assert_eq!(report.highest_spend_day, "Mon");
    }

    #[test]
    fn seasonality_identifies_highest_day() {
        // Jan 1 2024 = Mon ($5), Jan 2 = Tue ($100)
        let pts = vec![
            TrendPoint { timestamp: day(2024, 1, 1), cost_usd: 5.0 },
            TrendPoint { timestamp: day(2024, 1, 2), cost_usd: 100.0 },
        ];
        let report = TrendAnalyzer::seasonality(&pts);
        assert_eq!(report.highest_spend_day, "Tue");
    }

    #[test]
    fn seasonality_all_days_covered() {
        let report = TrendAnalyzer::seasonality(&[]);
        assert_eq!(report.day_averages.len(), 7);
        for day_name in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] {
            assert!(report.day_averages.contains_key(day_name));
        }
    }

    #[test]
    fn trend_direction_falling_decrease_positive() {
        let pts: Vec<TrendPoint> = (0..5)
            .map(|i| TrendPoint {
                timestamp: day(2024, 1, 1) + Duration::days(i),
                cost_usd: 10.0 - i as f64 * 2.0,
            })
            .collect();
        let result = TrendAnalyzer::fit(&pts);
        if let TrendDirection::Falling { daily_decrease_usd } = result.trend_direction {
            assert!(daily_decrease_usd > 0.0);
        } else {
            panic!("expected Falling");
        }
    }
}
