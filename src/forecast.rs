//! # Spend Forecast Engine
//!
//! Projects cumulative LLM spend forward using two complementary algorithms:
//!
//! 1. **OLS linear regression** — a simple trend line fitted to all
//!    `(timestamp_secs, cumulative_cost_usd)` observations.  Used by
//!    [`SpendForecaster::forecast`].
//!
//! 2. **Holt-Winters exponential smoothing** — a double-exponential smoother
//!    that captures both the level and the trend of the cost series,
//!    reacting faster to recent changes than OLS.  Used by
//!    [`CostForecaster`], which is the recommended algorithm for production
//!    monitoring.  Provides `next_hour`, `next_day`, `next_week`, and
//!    `next_month` projections together with confidence intervals and a
//!    budget-overage warning.
//!
//! ## Quick Start — Holt-Winters
//!
//! ```rust
//! use llm_cost_dashboard::forecast::CostForecaster;
//!
//! let mut f = CostForecaster::new();
//! // Feed hourly cumulative-cost observations (timestamp_secs, cost_usd).
//! let base: f64 = 1_700_000_000.0;
//! let hour = 3_600.0_f64;
//! for i in 0..24_u32 {
//!     // $0.50 / hour spend rate.
//!     f.record(base + i as f64 * hour, i as f64 * 0.50);
//! }
//! if let Some(hw) = f.forecast(Some(50.0)) {
//!     println!("next day: ${:.2}  (CI [{:.2}, {:.2}])",
//!         hw.next_day_usd, hw.confidence_interval.0, hw.confidence_interval.1);
//!     println!("next week: ${:.2}", hw.next_week_usd);
//! }
//! ```
//!
//! ## Quick Start — OLS (legacy)
//!
//! ```
//! use llm_cost_dashboard::forecast::SpendForecaster;
//!
//! let mut f = SpendForecaster::new();
//! // Simulate 3 days of spend observations.
//! let base: f64 = 1_700_000_000.0;
//! f.record(base,             0.00);
//! f.record(base + 86_400.0,  5.50);
//! f.record(base + 172_800.0, 11.00);
//!
//! if let Some(result) = f.forecast(Some(100.0)) {
//!     println!("projected month-end: ${:.2}", result.projected_month_end_usd);
//! }
//! ```

/// The directional trend of spend derived from regression slope changes.
#[derive(Debug, Clone, PartialEq)]
pub enum Trend {
    /// Spend rate is increasing (second-half slope > first-half slope).
    Accelerating,
    /// Spend rate is roughly constant.
    Stable,
    /// Spend rate is slowing down (second-half slope < first-half slope).
    Decelerating,
}

/// The output of a spend forecast computation.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Projected cumulative USD spend at midnight on the last day of the
    /// current calendar month.
    pub projected_month_end_usd: f64,
    /// Projected average daily spend in USD (regression slope × 86 400 s/day).
    pub projected_daily_usd: f64,
    /// Number of days until the budget limit is hit based on the current
    /// regression slope.  `None` when no budget limit was supplied or spend
    /// is already at/above the limit.
    pub days_until_budget_hit: Option<f64>,
    /// Goodness-of-fit confidence in `[0.0, 1.0]` derived from the R²
    /// coefficient of determination.  Values close to `1.0` indicate the
    /// linear model fits the data well.
    pub confidence: f64,
    /// Directional trend inferred by comparing the slope of the first half of
    /// observations against the second half.
    pub trend: Trend,
}

/// Linear-regression spend forecaster.
///
/// Records a time series of `(unix_timestamp_secs, cumulative_cost_usd)` pairs
/// and fits an OLS line to project spend forward to the end of the month.
///
/// At least two distinct observations are required before [`forecast`] can
/// return a result.
///
/// [`forecast`]: SpendForecaster::forecast
pub struct SpendForecaster {
    /// Stored `(timestamp_secs, cumulative_cost_usd)` pairs, sorted by
    /// insertion order (callers are expected to insert chronologically).
    observations: Vec<(f64, f64)>,
}

impl Default for SpendForecaster {
    fn default() -> Self {
        Self::new()
    }
}

impl SpendForecaster {
    /// Create an empty forecaster.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
        }
    }

    /// Append a `(timestamp_secs, cumulative_cost_usd)` observation.
    ///
    /// Observations should be supplied in chronological order.  Internally no
    /// sorting is performed; out-of-order data will produce inaccurate trend
    /// classification but the regression itself is still mathematically valid.
    pub fn record(&mut self, timestamp_secs: f64, cumulative_cost: f64) {
        self.observations.push((timestamp_secs, cumulative_cost));
    }

    /// Number of observations recorded so far.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether no observations have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Project spend to the end of the current calendar month using OLS linear
    /// regression.
    ///
    /// Returns `None` when fewer than two observations are available (regression
    /// is undefined).
    ///
    /// # Arguments
    ///
    /// * `budget_limit` – optional monthly budget cap in USD.  When supplied,
    ///   [`ForecastResult::days_until_budget_hit`] is populated.
    pub fn forecast(&self, budget_limit: Option<f64>) -> Option<ForecastResult> {
        if self.observations.len() < 2 {
            return None;
        }

        let (slope, intercept) = self.linear_regression()?;

        // Seconds per day.
        let spd = 86_400.0_f64;

        // Projected daily spend (USD/day) from the slope.
        let projected_daily_usd = slope * spd;

        // Determine seconds until end-of-month from the most recent timestamp.
        let last_ts = self.observations.last().map(|(t, _)| *t).unwrap_or(0.0);
        let secs_to_month_end = seconds_to_month_end(last_ts);

        // Project cumulative cost at month end.
        let last_cost = self.observations.last().map(|(_, c)| *c).unwrap_or(0.0);
        let projected_month_end_usd = last_cost + slope * secs_to_month_end;

        // Days until budget is hit.
        let days_until_budget_hit = budget_limit.and_then(|limit| {
            if last_cost >= limit || slope <= 0.0 {
                None
            } else {
                let secs_remaining = (limit - last_cost) / slope;
                Some(secs_remaining / spd)
            }
        });

        // R² confidence.
        let confidence = self.r_squared(slope, intercept);

        // Trend: compare slope of first half vs second half.
        let trend = self.classify_trend();

        Some(ForecastResult {
            projected_month_end_usd,
            projected_daily_usd,
            days_until_budget_hit,
            confidence,
            trend,
        })
    }

    /// Perform OLS linear regression on the stored observations.
    ///
    /// Returns `(slope, intercept)` where `slope` is in USD per second and
    /// `intercept` is in USD.  Returns `None` when the observations are
    /// collinear in the x-axis (all timestamps identical).
    fn linear_regression(&self) -> Option<(f64, f64)> {
        let n = self.observations.len() as f64;
        let sum_x: f64 = self.observations.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = self.observations.iter().map(|(_, y)| y).sum();
        let sum_xx: f64 = self.observations.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = self.observations.iter().map(|(x, y)| x * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return None;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;
        Some((slope, intercept))
    }

    /// Compute R² (coefficient of determination) for the given regression line.
    ///
    /// Returns a value in `[0.0, 1.0]`, clamped to handle floating-point
    /// edge cases.
    fn r_squared(&self, slope: f64, intercept: f64) -> f64 {
        let n = self.observations.len() as f64;
        let mean_y: f64 = self.observations.iter().map(|(_, y)| y).sum::<f64>() / n;

        let ss_tot: f64 = self
            .observations
            .iter()
            .map(|(_, y)| (y - mean_y).powi(2))
            .sum();

        if ss_tot < f64::EPSILON {
            // All y values are identical; perfect fit by convention.
            return 1.0;
        }

        let ss_res: f64 = self
            .observations
            .iter()
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    }

    /// Classify spend trend by comparing the OLS slope of the first half of
    /// observations against the second half.
    ///
    /// Falls back to [`Trend::Stable`] when either half has fewer than two
    /// points or regression is undefined.
    fn classify_trend(&self) -> Trend {
        let n = self.observations.len();
        let mid = n / 2;

        let slope_first = {
            let half = SpendForecaster {
                observations: self.observations[..mid].to_vec(),
            };
            half.linear_regression().map(|(s, _)| s)
        };

        let slope_second = {
            let half = SpendForecaster {
                observations: self.observations[mid..].to_vec(),
            };
            half.linear_regression().map(|(s, _)| s)
        };

        match (slope_first, slope_second) {
            (Some(s1), Some(s2)) => {
                // Allow a 10% band around stable.
                let ratio = if s1.abs() > f64::EPSILON { s2 / s1 } else { 1.0 };
                if ratio > 1.10 {
                    Trend::Accelerating
                } else if ratio < 0.90 {
                    Trend::Decelerating
                } else {
                    Trend::Stable
                }
            }
            _ => Trend::Stable,
        }
    }
}

/// Compute the number of seconds remaining until midnight on the last day of
/// the calendar month that contains `unix_ts`.
///
/// Uses a simple leap-year-aware month-length table.  The result is always
/// `>= 0.0`.
fn seconds_to_month_end(unix_ts: f64) -> f64 {
    // Days since Unix epoch.
    let days_since_epoch = (unix_ts / 86_400.0).floor() as i64;

    // Compute year/month/day from days-since-epoch using the proleptic
    // Gregorian calendar algorithm (Fliegel & Van Flandern).
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    let days_in_month = month_days(year, m as u32);

    // Seconds elapsed within the current month.
    let secs_today = unix_ts % 86_400.0;
    let day_of_month = {
        let d = doy - (153 * mp + 2) / 5;
        d + 1 // 1-based
    };

    let days_remaining = days_in_month as i64 - day_of_month;
    (days_remaining as f64 * 86_400.0 + (86_400.0 - secs_today)).max(0.0)
}

/// Number of days in the given (year, month) pair (month is 1-based).
fn month_days(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(year) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

/// Returns `true` for proleptic Gregorian leap years.
fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

// ---------------------------------------------------------------------------
// Holt-Winters double exponential smoothing
// ---------------------------------------------------------------------------

/// Output of a Holt-Winters cost forecast.
///
/// All monetary values are in USD.  The confidence interval is an approximate
/// 80 % prediction interval derived from the RMSE of the smoothed residuals.
#[derive(Debug, Clone)]
pub struct HoltWintersForecast {
    /// Projected incremental spend over the next hour in USD.
    pub next_hour_usd: f64,
    /// Projected incremental spend over the next 24 hours in USD.
    pub next_day_usd: f64,
    /// Projected incremental spend over the next 7 days in USD.
    pub next_week_usd: f64,
    /// Projected incremental spend over the next 30 days in USD.
    pub next_month_usd: f64,
    /// Approximate 80 % prediction interval `(lower, upper)` for the
    /// **next-hour** projection.  Wider intervals indicate higher uncertainty.
    pub confidence_interval: (f64, f64),
    /// `true` when the next-month projection exceeds 80 % of `budget_limit`
    /// (only set when a budget was supplied to [`CostForecaster::forecast`]).
    pub budget_warning: bool,
}

/// Holt-Winters double exponential smoothing forecaster.
///
/// Fits a level-and-trend model to a `(timestamp_secs, cumulative_cost_usd)`
/// time series and projects spend forward in time.  Unlike OLS, it weights
/// recent observations more heavily, making it well-suited for spend series
/// that exhibit changing rates (e.g. business hours vs. weekends).
///
/// ## Parameters
///
/// - `alpha` — Level smoothing factor (`0 < alpha < 1`).  Higher values make
///   the level react faster to recent observations (less smoothing).
///   Default: `0.3`.
/// - `beta` — Trend smoothing factor (`0 < beta < 1`).  Higher values make
///   the trend react faster.  Default: `0.1`.
///
/// Both parameters are configurable via [`CostForecaster::with_params`].
///
/// ## Minimum Observations
///
/// At least **3** observations are required before [`forecast`] returns a
/// result (needed to initialise both level and trend).
///
/// [`forecast`]: CostForecaster::forecast
#[derive(Debug)]
pub struct CostForecaster {
    /// Stored `(timestamp_secs, cumulative_cost_usd)` pairs.
    observations: Vec<(f64, f64)>,
    /// Level smoothing coefficient α.
    alpha: f64,
    /// Trend smoothing coefficient β.
    beta: f64,
}

impl Default for CostForecaster {
    fn default() -> Self {
        Self::new()
    }
}

impl CostForecaster {
    /// Create a forecaster with default smoothing parameters (α=0.3, β=0.1).
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            alpha: 0.3,
            beta: 0.1,
        }
    }

    /// Override the default smoothing parameters.
    ///
    /// Both `alpha` and `beta` are clamped to `(0.0, 1.0)`.
    pub fn with_params(mut self, alpha: f64, beta: f64) -> Self {
        self.alpha = alpha.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        self.beta = beta.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        self
    }

    /// Append a `(timestamp_secs, cumulative_cost_usd)` observation.
    ///
    /// Observations must be supplied in **chronological order**.
    pub fn record(&mut self, timestamp_secs: f64, cumulative_cost: f64) {
        self.observations.push((timestamp_secs, cumulative_cost));
    }

    /// Number of observations recorded so far.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether no observations have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Run the Holt-Winters smoother and project spend forward.
    ///
    /// Returns `None` when fewer than 3 observations are available.
    ///
    /// # Arguments
    ///
    /// * `budget_limit` — optional monthly budget cap in USD used to set
    ///   [`HoltWintersForecast::budget_warning`].
    pub fn forecast(&self, budget_limit: Option<f64>) -> Option<HoltWintersForecast> {
        if self.observations.len() < 3 {
            return None;
        }

        // Work in incremental costs (first differences) rather than cumulative,
        // because Holt-Winters level/trend make more sense on rates than on
        // an ever-increasing cumulative series.
        let increments: Vec<f64> = self
            .observations
            .windows(2)
            .map(|w| {
                let dt = (w[1].0 - w[0].0).max(f64::EPSILON);
                // Normalise to cost-per-second so unequal intervals cancel out.
                (w[1].1 - w[0].1) / dt
            })
            .collect();

        if increments.is_empty() {
            return None;
        }

        // Initialise level and trend from the first two increments.
        let mut level = increments[0];
        let mut trend = if increments.len() >= 2 {
            increments[1] - increments[0]
        } else {
            0.0
        };

        // Residuals for RMSE computation.
        let mut residuals: Vec<f64> = Vec::with_capacity(increments.len());

        // Apply double exponential smoothing (Holt's linear method).
        for &y in &increments {
            let prev_level = level;
            let prev_trend = trend;
            level = self.alpha * y + (1.0 - self.alpha) * (prev_level + prev_trend);
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * prev_trend;
            let forecast_t = prev_level + prev_trend;
            residuals.push(y - forecast_t);
        }

        // RMSE of one-step-ahead residuals.
        let rmse = {
            let ss: f64 = residuals.iter().map(|r| r * r).sum();
            (ss / residuals.len() as f64).sqrt()
        };

        // Typical observation interval in seconds.
        let obs = &self.observations;
        let n = obs.len();
        let avg_interval_secs = if n >= 2 {
            (obs[n - 1].0 - obs[0].0) / (n - 1) as f64
        } else {
            3_600.0 // fallback: assume hourly
        };

        // Project h steps ahead: forecast(h) = level + h * trend (per second).
        // Convert back from cost/second to total cost over the horizon.
        let hour_secs = 3_600.0_f64;
        let day_secs = 86_400.0_f64;
        let week_secs = 7.0 * day_secs;
        let month_secs = 30.0 * day_secs;

        // Steps-ahead for each horizon.
        let steps_hour = (hour_secs / avg_interval_secs).max(1.0);
        let steps_day = (day_secs / avg_interval_secs).max(1.0);
        let steps_week = (week_secs / avg_interval_secs).max(1.0);
        let steps_month = (month_secs / avg_interval_secs).max(1.0);

        // Holt forecast h steps ahead: level + h * trend (still per second).
        let rate_h = |h: f64| -> f64 { (level + h * trend).max(0.0) };

        let next_hour_usd = rate_h(steps_hour) * hour_secs;
        let next_day_usd = rate_h(steps_day) * day_secs;
        let next_week_usd = rate_h(steps_week) * week_secs;
        let next_month_usd = rate_h(steps_month) * month_secs;

        // 80 % prediction interval for next_hour (z_80 ≈ 1.28).
        let uncertainty = 1.28 * rmse * hour_secs;
        let ci_lower = (next_hour_usd - uncertainty).max(0.0);
        let ci_upper = next_hour_usd + uncertainty;

        let budget_warning = budget_limit
            .map(|limit| next_month_usd >= limit * 0.80)
            .unwrap_or(false);

        Some(HoltWintersForecast {
            next_hour_usd,
            next_day_usd,
            next_week_usd,
            next_month_usd,
            confidence_interval: (ci_lower, ci_upper),
            budget_warning,
        })
    }
}

// ---------------------------------------------------------------------------
// New exponential-smoothing forecast engine
// ---------------------------------------------------------------------------

/// Selects which exponential-smoothing algorithm to use when fitting a
/// [`EsCostForecaster`].
#[derive(Debug, Clone)]
pub enum ForecastMethod {
    /// Simple (single) exponential smoothing with smoothing factor `alpha`.
    SimpleExponential {
        /// Level smoothing factor (`0 < alpha < 1`).
        alpha: f64,
    },
    /// Double (Holt's linear) exponential smoothing.
    DoubleExponential {
        /// Level smoothing factor.
        alpha: f64,
        /// Trend smoothing factor.
        beta: f64,
    },
    /// Holt-Winters triple exponential smoothing with multiplicative seasonality.
    HoltWinters {
        /// Level smoothing factor.
        alpha: f64,
        /// Trend smoothing factor.
        beta: f64,
        /// Seasonal smoothing factor.
        gamma: f64,
        /// Number of periods per season (e.g. 7 for weekly, 24 for hourly).
        period: usize,
    },
}

// ---------------------------------------------------------------------------
// SimpleEsModel
// ---------------------------------------------------------------------------

/// Simple (single) exponential smoothing model.
///
/// Maintains a single level estimate; suitable for series without trend or
/// seasonality.
#[derive(Debug, Clone)]
pub struct SimpleEsModel {
    /// Level smoothing coefficient.
    pub alpha: f64,
    /// Current level estimate.
    pub level: f64,
}

impl SimpleEsModel {
    /// Create a model with an initial level of `0.0`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(f64::EPSILON, 1.0 - f64::EPSILON),
            level: 0.0,
        }
    }

    /// Update the level with a new observed value.
    pub fn update(&mut self, value: f64) {
        self.level = self.alpha * value + (1.0 - self.alpha) * self.level;
    }

    /// Forecast `steps` periods ahead (all identical for simple ES).
    pub fn forecast(&self, steps: usize) -> Vec<f64> {
        vec![self.level; steps]
    }
}

// ---------------------------------------------------------------------------
// DoubleEsModel
// ---------------------------------------------------------------------------

/// Double (Holt's linear) exponential smoothing model.
///
/// Tracks both a level and a trend component, allowing it to project series
/// that exhibit a linear drift.
#[derive(Debug, Clone)]
pub struct DoubleEsModel {
    /// Level smoothing coefficient.
    pub alpha: f64,
    /// Trend smoothing coefficient.
    pub beta: f64,
    /// Current level estimate.
    pub level: f64,
    /// Current trend estimate.
    pub trend: f64,
}

impl DoubleEsModel {
    /// Create a model with level and trend initialised to `0.0`.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.clamp(f64::EPSILON, 1.0 - f64::EPSILON),
            beta: beta.clamp(f64::EPSILON, 1.0 - f64::EPSILON),
            level: 0.0,
            trend: 0.0,
        }
    }

    /// Update the model with a new observed value.
    pub fn update(&mut self, value: f64) {
        let prev_level = self.level;
        let prev_trend = self.trend;
        self.level = self.alpha * value + (1.0 - self.alpha) * (prev_level + prev_trend);
        self.trend = self.beta * (self.level - prev_level) + (1.0 - self.beta) * prev_trend;
    }

    /// Forecast `steps` periods ahead using `level + h * trend`.
    pub fn forecast(&self, steps: usize) -> Vec<f64> {
        (1..=steps)
            .map(|h| self.level + h as f64 * self.trend)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// HoltWintersModel
// ---------------------------------------------------------------------------

/// Holt-Winters triple exponential smoothing with **multiplicative** seasonality.
///
/// Suitable for series that have both a trend and a repeating seasonal pattern
/// whose amplitude scales with the level.
#[derive(Debug, Clone)]
pub struct HoltWintersModel {
    /// Level smoothing coefficient.
    pub alpha: f64,
    /// Trend smoothing coefficient.
    pub beta: f64,
    /// Seasonal smoothing coefficient.
    pub gamma: f64,
    /// Season length in periods.
    pub period: usize,
    /// Current level.
    pub level: f64,
    /// Current trend.
    pub trend: f64,
    /// Seasonal indices (length == `period`).
    pub seasonal: Vec<f64>,
}

impl HoltWintersModel {
    /// Create a model with the given parameters.
    ///
    /// `seasonal` must have length `period`; if it is empty or shorter it is
    /// padded with `1.0` (neutral multiplicative factor).
    pub fn new(alpha: f64, beta: f64, gamma: f64, period: usize, seasonal: Vec<f64>) -> Self {
        let p = period.max(1);
        let mut s = seasonal;
        s.resize(p, 1.0);
        Self {
            alpha: alpha.clamp(f64::EPSILON, 1.0 - f64::EPSILON),
            beta: beta.clamp(f64::EPSILON, 1.0 - f64::EPSILON),
            gamma: gamma.clamp(f64::EPSILON, 1.0 - f64::EPSILON),
            period: p,
            level: 0.0,
            trend: 0.0,
            seasonal: s,
        }
    }

    /// Update the model with a new observed value at position `t` (0-based).
    ///
    /// `t` is used to index into the seasonal component.
    pub fn update(&mut self, value: f64, t: usize) {
        let s_idx = t % self.period;
        let seasonal_t = self.seasonal[s_idx];
        let prev_level = self.level;
        let prev_trend = self.trend;

        // Guard against division by zero.
        let denom = if seasonal_t.abs() < f64::EPSILON { 1.0 } else { seasonal_t };

        self.level = self.alpha * (value / denom)
            + (1.0 - self.alpha) * (prev_level + prev_trend);
        self.trend = self.beta * (self.level - prev_level)
            + (1.0 - self.beta) * prev_trend;
        self.seasonal[s_idx] = self.gamma * (value / self.level.max(f64::EPSILON))
            + (1.0 - self.gamma) * seasonal_t;
    }

    /// Forecast `steps` periods ahead (multiplicative seasonality).
    ///
    /// `current_t` is the 0-based index of the last observed period.
    pub fn forecast(&self, steps: usize, current_t: usize) -> Vec<f64> {
        (1..=steps)
            .map(|h| {
                let s_idx = (current_t + h) % self.period;
                (self.level + h as f64 * self.trend) * self.seasonal[s_idx]
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// EsCostForecaster
// ---------------------------------------------------------------------------

/// Cost forecaster built on top of [`SimpleEsModel`], [`DoubleEsModel`], or
/// [`HoltWintersModel`].
///
/// Constructed via [`EsCostForecaster::fit`], which initialises and trains the
/// chosen model on historical cost data in a single call.
pub struct EsCostForecaster {
    /// Fitted forecasts one-step-ahead for each training observation.
    #[allow(dead_code)]
    fitted: Vec<f64>,
    /// Residuals (observed − fitted) for each training observation.
    residuals: Vec<f64>,
    /// Original observations used for training.
    observations: Vec<f64>,
    /// Internal state for stepping ahead at forecast time.
    inner: EsInner,
    /// Length of training data (for Holt-Winters `current_t`).
    trained_len: usize,
}

/// Internal model state after training.
#[derive(Clone)]
enum EsInner {
    Simple(SimpleEsModel),
    Double(DoubleEsModel),
    HoltWinters(HoltWintersModel),
}

impl EsCostForecaster {
    /// Fit a forecaster to `historical_costs` using `method`.
    ///
    /// At least one observation is required; an empty slice produces a
    /// forecaster that returns `0.0` for all future steps.
    pub fn fit(historical_costs: &[f64], method: ForecastMethod) -> Self {
        if historical_costs.is_empty() {
            return Self {
                fitted: Vec::new(),
                residuals: Vec::new(),
                observations: Vec::new(),
                inner: EsInner::Simple(SimpleEsModel::new(0.3)),
                trained_len: 0,
            };
        }

        match method {
            ForecastMethod::SimpleExponential { alpha } => {
                Self::fit_simple(historical_costs, alpha)
            }
            ForecastMethod::DoubleExponential { alpha, beta } => {
                Self::fit_double(historical_costs, alpha, beta)
            }
            ForecastMethod::HoltWinters { alpha, beta, gamma, period } => {
                Self::fit_hw(historical_costs, alpha, beta, gamma, period)
            }
        }
    }

    fn fit_simple(data: &[f64], alpha: f64) -> Self {
        let mut model = SimpleEsModel::new(alpha);
        model.level = data[0];
        let mut fitted = Vec::with_capacity(data.len());
        let mut residuals = Vec::with_capacity(data.len());

        for &y in data {
            let f = model.level;
            fitted.push(f);
            residuals.push(y - f);
            model.update(y);
        }

        Self {
            fitted,
            residuals,
            observations: data.to_vec(),
            inner: EsInner::Simple(model),
            trained_len: data.len(),
        }
    }

    fn fit_double(data: &[f64], alpha: f64, beta: f64) -> Self {
        let mut model = DoubleEsModel::new(alpha, beta);
        model.level = data[0];
        model.trend = if data.len() >= 2 { data[1] - data[0] } else { 0.0 };

        let mut fitted = Vec::with_capacity(data.len());
        let mut residuals = Vec::with_capacity(data.len());

        for &y in data {
            let f = model.level + model.trend;
            fitted.push(f);
            residuals.push(y - f);
            model.update(y);
        }

        Self {
            fitted,
            residuals,
            observations: data.to_vec(),
            inner: EsInner::Double(model),
            trained_len: data.len(),
        }
    }

    fn fit_hw(data: &[f64], alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        let p = period.max(1);

        // Initialise seasonal indices from the first season.
        let season_len = p.min(data.len());
        let season_mean: f64 = data[..season_len].iter().sum::<f64>() / season_len as f64;
        let initial_seasonal: Vec<f64> = data[..season_len]
            .iter()
            .map(|&v| {
                if season_mean.abs() < f64::EPSILON { 1.0 } else { v / season_mean }
            })
            .collect();

        let mut model = HoltWintersModel::new(alpha, beta, gamma, p, initial_seasonal);
        model.level = season_mean.max(f64::EPSILON);
        model.trend = 0.0;

        let mut fitted = Vec::with_capacity(data.len());
        let mut residuals = Vec::with_capacity(data.len());

        for (t, &y) in data.iter().enumerate() {
            let s_idx = t % model.period;
            let f = (model.level + model.trend) * model.seasonal[s_idx];
            fitted.push(f);
            residuals.push(y - f);
            model.update(y, t);
        }

        Self {
            fitted,
            residuals,
            observations: data.to_vec(),
            inner: EsInner::HoltWinters(model),
            trained_len: data.len(),
        }
    }

    /// Project `steps` periods ahead from the end of training data.
    pub fn forecast(&self, steps: usize) -> Vec<f64> {
        if steps == 0 {
            return Vec::new();
        }
        match &self.inner {
            EsInner::Simple(m) => m.forecast(steps),
            EsInner::Double(m) => m.forecast(steps),
            EsInner::HoltWinters(m) => m.forecast(steps, self.trained_len.saturating_sub(1)),
        }
    }

    /// Approximate `(lower, upper)` prediction intervals for each forecast step.
    ///
    /// `z` is the z-score for the desired confidence level (e.g. `1.96` for
    /// ~95 %, `1.28` for ~80 %).
    pub fn confidence_interval(&self, steps: usize, z: f64) -> Vec<(f64, f64)> {
        let point = self.forecast(steps);
        let std = self.residual_std();
        point
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                // Uncertainty grows with horizon (sqrt scaling).
                let margin = z * std * ((i + 1) as f64).sqrt();
                ((p - margin).max(0.0), p + margin)
            })
            .collect()
    }

    /// Mean absolute error on the training data.
    pub fn mae(&self) -> f64 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        self.residuals.iter().map(|r| r.abs()).sum::<f64>() / self.residuals.len() as f64
    }

    /// Mean absolute percentage error on the training data.
    ///
    /// Observations with absolute value `< 1e-9` are skipped to avoid
    /// division by zero.
    pub fn mape(&self) -> f64 {
        let valid: Vec<(f64, f64)> = self
            .observations
            .iter()
            .zip(self.residuals.iter())
            .filter(|(obs, _)| obs.abs() >= 1e-9)
            .map(|(obs, res)| (*obs, *res))
            .collect();

        if valid.is_empty() {
            return 0.0;
        }

        let sum: f64 = valid.iter().map(|(obs, res)| (res / obs).abs()).sum();
        sum / valid.len() as f64
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn residual_std(&self) -> f64 {
        if self.residuals.len() < 2 {
            return 0.0;
        }
        let n = self.residuals.len() as f64;
        let mean = self.residuals.iter().sum::<f64>() / n;
        let variance = self.residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    }
}

// ── ES Forecaster tests ────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod es_tests {
    use super::*;

    fn linear_series(n: usize, slope: f64) -> Vec<f64> {
        (0..n).map(|i| i as f64 * slope).collect()
    }

    #[test]
    fn simple_es_forecast_len() {
        let data = linear_series(10, 1.0);
        let fc = EsCostForecaster::fit(&data, ForecastMethod::SimpleExponential { alpha: 0.3 });
        assert_eq!(fc.forecast(5).len(), 5);
    }

    #[test]
    fn simple_es_mae_finite() {
        let data = linear_series(10, 2.0);
        let fc = EsCostForecaster::fit(&data, ForecastMethod::SimpleExponential { alpha: 0.3 });
        assert!(fc.mae().is_finite());
    }

    #[test]
    fn simple_es_mape_finite() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fc = EsCostForecaster::fit(&data, ForecastMethod::SimpleExponential { alpha: 0.3 });
        assert!(fc.mape().is_finite());
    }

    #[test]
    fn double_es_forecast_increasing_for_positive_trend() {
        let data = linear_series(10, 1.0);
        let fc = EsCostForecaster::fit(
            &data,
            ForecastMethod::DoubleExponential { alpha: 0.5, beta: 0.3 },
        );
        let preds = fc.forecast(3);
        assert_eq!(preds.len(), 3);
        // Each step should be >= the previous for positive trend series.
        for w in preds.windows(2) {
            assert!(w[1] >= w[0], "forecast not increasing: {:?}", preds);
        }
    }

    #[test]
    fn double_es_mae_lower_than_naive_for_linear() {
        // For a perfect linear series, double ES should fit well.
        let data = linear_series(20, 3.0);
        let fc = EsCostForecaster::fit(
            &data,
            ForecastMethod::DoubleExponential { alpha: 0.9, beta: 0.9 },
        );
        // MAE should be finite and non-negative.
        let mae = fc.mae();
        assert!(mae >= 0.0);
        assert!(mae.is_finite());
    }

    #[test]
    fn holt_winters_forecast_len() {
        let data: Vec<f64> = (0..24).map(|i| (i % 7) as f64 + 1.0).collect();
        let fc = EsCostForecaster::fit(
            &data,
            ForecastMethod::HoltWinters { alpha: 0.3, beta: 0.1, gamma: 0.1, period: 7 },
        );
        assert_eq!(fc.forecast(7).len(), 7);
    }

    #[test]
    fn holt_winters_mae_finite() {
        let data: Vec<f64> = (0..24).map(|i| (i % 7) as f64 + 1.0).collect();
        let fc = EsCostForecaster::fit(
            &data,
            ForecastMethod::HoltWinters { alpha: 0.3, beta: 0.1, gamma: 0.1, period: 7 },
        );
        assert!(fc.mae().is_finite());
    }

    #[test]
    fn confidence_interval_len_matches_steps() {
        let data = linear_series(10, 1.0);
        let fc = EsCostForecaster::fit(&data, ForecastMethod::SimpleExponential { alpha: 0.3 });
        let ci = fc.confidence_interval(5, 1.96);
        assert_eq!(ci.len(), 5);
    }

    #[test]
    fn confidence_interval_lower_le_upper() {
        let data = linear_series(10, 1.0);
        let fc = EsCostForecaster::fit(&data, ForecastMethod::SimpleExponential { alpha: 0.3 });
        for (lo, hi) in fc.confidence_interval(5, 1.96) {
            assert!(lo <= hi, "CI lower > upper: {lo} > {hi}");
        }
    }

    #[test]
    fn empty_data_returns_empty_forecast() {
        let fc = EsCostForecaster::fit(&[], ForecastMethod::SimpleExponential { alpha: 0.3 });
        assert!(fc.forecast(5).is_empty() || fc.forecast(5).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forecast_zero_steps_empty() {
        let data = linear_series(5, 1.0);
        let fc = EsCostForecaster::fit(&data, ForecastMethod::SimpleExponential { alpha: 0.3 });
        assert!(fc.forecast(0).is_empty());
    }

    #[test]
    fn simple_es_model_update_and_forecast() {
        let mut m = SimpleEsModel::new(0.5);
        m.update(10.0);
        m.update(20.0);
        let preds = m.forecast(3);
        assert_eq!(preds.len(), 3);
        // All same value for simple ES.
        assert_eq!(preds[0], preds[1]);
    }

    #[test]
    fn double_es_model_update_and_forecast() {
        let mut m = DoubleEsModel::new(0.5, 0.3);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            m.update(v);
        }
        let preds = m.forecast(3);
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn holt_winters_model_update_and_forecast() {
        let seasonal = vec![1.0, 1.2, 0.9, 0.8];
        let mut m = HoltWintersModel::new(0.3, 0.1, 0.1, 4, seasonal);
        m.level = 10.0;
        for (t, v) in [10.0, 12.0, 9.0, 8.0, 11.0, 13.0].iter().enumerate() {
            m.update(*v, t);
        }
        let preds = m.forecast(4, 5);
        assert_eq!(preds.len(), 4);
        assert!(preds.iter().all(|p| p.is_finite()));
    }
}

// ── Legacy tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_forecaster_linear(n: usize, slope_per_day: f64) -> SpendForecaster {
        let mut f = SpendForecaster::new();
        let base: f64 = 1_700_000_000.0; // 2023-11-14-ish
        let spd = 86_400.0;
        for i in 0..n {
            let t = base + i as f64 * spd;
            let cost = i as f64 * slope_per_day;
            f.record(t, cost);
        }
        f
    }

    #[test]
    fn test_new_is_empty() {
        let f = SpendForecaster::new();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn test_record_increases_len() {
        let mut f = SpendForecaster::new();
        f.record(1_700_000_000.0, 0.0);
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_forecast_requires_two_observations() {
        let mut f = SpendForecaster::new();
        f.record(1_700_000_000.0, 0.0);
        assert!(f.forecast(None).is_none());
    }

    #[test]
    fn test_forecast_returns_some_with_two_points() {
        let mut f = SpendForecaster::new();
        f.record(1_700_000_000.0, 0.0);
        f.record(1_700_086_400.0, 5.0);
        assert!(f.forecast(None).is_some());
    }

    #[test]
    fn test_daily_projection_matches_slope() {
        // $5/day linear spend.
        let f = make_forecaster_linear(10, 5.0);
        let result = f.forecast(None).unwrap();
        assert!((result.projected_daily_usd - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_high_r_squared_for_perfect_linear_data() {
        let f = make_forecaster_linear(20, 10.0);
        let result = f.forecast(None).unwrap();
        // Perfect linear data should yield R² ≈ 1.0.
        assert!(result.confidence > 0.99);
    }

    #[test]
    fn test_days_until_budget_hit_none_when_no_limit() {
        let f = make_forecaster_linear(10, 5.0);
        let result = f.forecast(None).unwrap();
        assert!(result.days_until_budget_hit.is_none());
    }

    #[test]
    fn test_days_until_budget_hit_some_when_under_limit() {
        // $5/day, already at $45, limit = $100.  Need $55 more => ~11 days.
        let f = make_forecaster_linear(10, 5.0);
        let result = f.forecast(Some(100.0)).unwrap();
        let days = result.days_until_budget_hit.unwrap();
        assert!((days - 11.0).abs() < 0.5);
    }

    #[test]
    fn test_days_until_budget_hit_none_when_over_limit() {
        // Limit below current spend.
        let f = make_forecaster_linear(10, 5.0);
        let result = f.forecast(Some(1.0)).unwrap();
        assert!(result.days_until_budget_hit.is_none());
    }

    #[test]
    fn test_trend_stable_for_constant_slope() {
        let f = make_forecaster_linear(20, 5.0);
        let result = f.forecast(None).unwrap();
        assert_eq!(result.trend, Trend::Stable);
    }

    #[test]
    fn test_trend_accelerating() {
        let mut f = SpendForecaster::new();
        let base = 1_700_000_000.0;
        let spd = 86_400.0;
        // First half: slow spend (slope ≈ 1/day).
        for i in 0..10 {
            f.record(base + i as f64 * spd, i as f64 * 1.0);
        }
        // Second half: much faster spend (slope ≈ 10/day).
        for i in 10..20 {
            f.record(base + i as f64 * spd, 10.0 + (i - 10) as f64 * 10.0);
        }
        let result = f.forecast(None).unwrap();
        assert_eq!(result.trend, Trend::Accelerating);
    }

    #[test]
    fn test_r_squared_method() {
        let f = make_forecaster_linear(15, 3.0);
        let (slope, intercept) = f.linear_regression().unwrap();
        let r2 = f.r_squared(slope, intercept);
        assert!(r2 > 0.99);
    }

    #[test]
    fn test_projected_month_end_positive() {
        let f = make_forecaster_linear(5, 2.0);
        let result = f.forecast(None).unwrap();
        assert!(result.projected_month_end_usd >= 0.0);
    }

    #[test]
    fn test_default_impl() {
        let f = SpendForecaster::default();
        assert!(f.is_empty());
    }

    // ── Holt-Winters tests ────────────────────────────────────────────────────

    fn make_hw_forecaster_linear(n: usize, rate_per_hour: f64) -> CostForecaster {
        let mut f = CostForecaster::new();
        let base = 1_700_000_000.0_f64;
        let hour = 3_600.0_f64;
        for i in 0..n {
            let t = base + i as f64 * hour;
            let cost = i as f64 * rate_per_hour;
            f.record(t, cost);
        }
        f
    }

    #[test]
    fn hw_requires_three_observations() {
        let mut f = CostForecaster::new();
        let base = 1_700_000_000.0;
        f.record(base, 0.0);
        f.record(base + 3_600.0, 0.5);
        assert!(f.forecast(None).is_none());
        f.record(base + 7_200.0, 1.0);
        assert!(f.forecast(None).is_some());
    }

    #[test]
    fn hw_next_day_greater_than_next_hour() {
        let f = make_hw_forecaster_linear(24, 0.50);
        let result = f.forecast(None).unwrap();
        assert!(
            result.next_day_usd >= result.next_hour_usd,
            "next_day ({:.4}) should be >= next_hour ({:.4})",
            result.next_day_usd,
            result.next_hour_usd
        );
    }

    #[test]
    fn hw_next_week_greater_than_next_day() {
        let f = make_hw_forecaster_linear(24, 0.50);
        let result = f.forecast(None).unwrap();
        assert!(
            result.next_week_usd >= result.next_day_usd,
            "next_week ({:.4}) should be >= next_day ({:.4})",
            result.next_week_usd,
            result.next_day_usd
        );
    }

    #[test]
    fn hw_next_month_greater_than_next_week() {
        let f = make_hw_forecaster_linear(24, 0.50);
        let result = f.forecast(None).unwrap();
        assert!(
            result.next_month_usd >= result.next_week_usd,
            "next_month ({:.4}) should be >= next_week ({:.4})",
            result.next_month_usd,
            result.next_week_usd
        );
    }

    #[test]
    fn hw_confidence_interval_straddles_point_estimate() {
        let f = make_hw_forecaster_linear(24, 0.50);
        let result = f.forecast(None).unwrap();
        let (lo, hi) = result.confidence_interval;
        assert!(lo <= result.next_hour_usd, "CI lower ({lo:.4}) should be <= point estimate ({:.4})", result.next_hour_usd);
        assert!(hi >= result.next_hour_usd, "CI upper ({hi:.4}) should be >= point estimate ({:.4})", result.next_hour_usd);
    }

    #[test]
    fn hw_budget_warning_fires_when_exceeds_80_pct() {
        // 24 h at $0.50/h → $12/day → ~$360/month. Budget = $100 → 80% = $80.
        let f = make_hw_forecaster_linear(24, 0.50);
        let result = f.forecast(Some(100.0)).unwrap();
        // next_month_usd should exceed $80 (80% of $100), triggering warning.
        assert!(result.budget_warning, "budget_warning should be true");
    }

    #[test]
    fn hw_no_budget_warning_when_well_within_budget() {
        // Very low spend rate: $0.001/h → ~$0.72/month. Budget = $1000.
        let f = make_hw_forecaster_linear(24, 0.001);
        let result = f.forecast(Some(1_000.0)).unwrap();
        assert!(!result.budget_warning, "no warning expected for tiny spend vs large budget");
    }

    #[test]
    fn hw_default_is_empty() {
        let f = CostForecaster::default();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn hw_with_params_clamps() {
        // alpha > 1.0 should be clamped.
        let f = CostForecaster::new().with_params(5.0, -1.0);
        // Just verify it doesn't panic and produces a result.
        let mut f2 = f;
        let base = 1_700_000_000.0;
        f2.record(base, 0.0);
        f2.record(base + 3_600.0, 1.0);
        f2.record(base + 7_200.0, 2.0);
        assert!(f2.forecast(None).is_some());
    }

    #[test]
    fn hw_all_projections_non_negative() {
        let f = make_hw_forecaster_linear(10, 2.0);
        let result = f.forecast(None).unwrap();
        assert!(result.next_hour_usd >= 0.0);
        assert!(result.next_day_usd >= 0.0);
        assert!(result.next_week_usd >= 0.0);
        assert!(result.next_month_usd >= 0.0);
        assert!(result.confidence_interval.0 >= 0.0);
    }
}
