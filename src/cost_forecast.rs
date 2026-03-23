//! Time-Series Cost Forecasting
//!
//! Implements multiple forecasting algorithms for projecting LLM spend forward
//! in time. Algorithms include:
//!
//! - **Holt-Winters** triple exponential smoothing (level + trend + season)
//! - **Linear trend** via OLS regression
//! - **EMA** (exponential moving average) with last-observed trend
//! - **ARIMA** placeholder (returns linear extrapolation)
//!
//! The [`CostForecaster`] struct exposes both individual algorithm methods and
//! a [`CostForecaster::forecast`] dispatch method that honours a [`ForecastModel`]
//! variant, plus a [`CostForecaster::best_model`] selector based on AIC.

use std::f64::consts::PI;

// ── Period ────────────────────────────────────────────────────────────────────

/// Granularity of each forecast point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastPeriod {
    /// One calendar day per point.
    Daily,
    /// One calendar week per point.
    Weekly,
    /// One calendar month (30 days) per point.
    Monthly,
    /// One calendar quarter (91 days) per point.
    Quarterly,
}

impl ForecastPeriod {
    /// Nominal duration in seconds for a single period.
    pub fn duration_secs(&self) -> u64 {
        match self {
            ForecastPeriod::Daily => 86_400,
            ForecastPeriod::Weekly => 604_800,
            ForecastPeriod::Monthly => 2_592_000,
            ForecastPeriod::Quarterly => 7_862_400,
        }
    }
}

// ── ForecastPoint ─────────────────────────────────────────────────────────────

/// A single forecasted data point.
#[derive(Debug, Clone)]
pub struct ForecastPoint {
    /// Unix timestamp (seconds) for the start of the period.
    pub period_start: u64,
    /// Unix timestamp (seconds) for the end of the period.
    pub period_end: u64,
    /// Central forecast cost in USD.
    pub forecast_cost: f64,
    /// Lower confidence bound in USD.
    pub lower_bound: f64,
    /// Upper confidence bound in USD.
    pub upper_bound: f64,
    /// Confidence level (0–1, e.g. 0.95 for 95 %).
    pub confidence: f64,
}

// ── SeasonalComponent ─────────────────────────────────────────────────────────

/// Describes a detected seasonal component in a time series.
#[derive(Debug, Clone)]
pub struct SeasonalComponent {
    /// Number of data points in one seasonal cycle.
    pub period: usize,
    /// Fourier amplitudes for harmonics 1..=N.
    pub amplitudes: Vec<f64>,
    /// Phase offset of the dominant harmonic (radians).
    pub phase: f64,
}

// ── ForecastModel ─────────────────────────────────────────────────────────────

/// Forecasting algorithm to apply.
#[derive(Debug, Clone)]
pub enum ForecastModel {
    /// Triple exponential smoothing (Holt-Winters).
    HoltWinters {
        /// Level smoothing (0–1).
        alpha: f64,
        /// Trend smoothing (0–1).
        beta: f64,
        /// Seasonal smoothing (0–1).
        gamma: f64,
        /// Season length in data points.
        season_len: usize,
    },
    /// Simple OLS linear extrapolation.
    LinearTrend {
        /// Slope (cost per data point).
        slope: f64,
        /// Intercept at t=0.
        intercept: f64,
    },
    /// Exponential moving average with trend carry-forward.
    EMA {
        /// Smoothing factor (0–1).
        alpha: f64,
    },
    /// ARIMA (p,d,q) — currently delegates to linear extrapolation.
    ARIMA {
        /// Auto-regressive order.
        p: usize,
        /// Differencing order.
        d: usize,
        /// Moving-average order.
        q: usize,
    },
}

// ── CostForecaster ────────────────────────────────────────────────────────────

/// Stateless cost forecasting engine.
///
/// All methods are pure functions operating on slices of historical data.
/// Create a [`CostForecaster`] with [`CostForecaster::default`] and call the
/// appropriate method.
#[derive(Debug, Clone, Default)]
pub struct CostForecaster;

impl CostForecaster {
    /// Find optimal alpha/beta/gamma for Holt-Winters via grid search on MSE.
    ///
    /// Grid resolution is 0.1 for alpha and beta, 0.1 for gamma.
    /// Returns `(alpha, beta, gamma)`.
    pub fn fit_holt_winters(data: &[f64], season_len: usize) -> (f64, f64, f64) {
        if data.len() < 2 || season_len < 2 {
            return (0.3, 0.1, 0.1);
        }

        let alphas = [0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let betas = [0.05_f64, 0.1, 0.2, 0.3, 0.4, 0.5];
        let gammas = [0.05_f64, 0.1, 0.2, 0.3, 0.4, 0.5];

        let mut best_mse = f64::MAX;
        let mut best = (0.3_f64, 0.1_f64, 0.1_f64);

        for &a in &alphas {
            for &b in &betas {
                for &g in &gammas {
                    let fitted = Self::holt_winters_forecast(data, a, b, g, season_len, 0);
                    if fitted.is_empty() {
                        continue;
                    }
                    let mse = data
                        .iter()
                        .zip(fitted.iter())
                        .map(|(y, yh)| (y - yh).powi(2))
                        .sum::<f64>()
                        / data.len() as f64;
                    if mse < best_mse {
                        best_mse = mse;
                        best = (a, b, g);
                    }
                }
            }
        }
        best
    }

    /// Run Holt-Winters triple exponential smoothing.
    ///
    /// Returns fitted values for the training data followed by `horizon`
    /// out-of-sample forecasts. An empty slice is returned when the data is
    /// too short to initialise seasonal indices.
    pub fn holt_winters_forecast(
        data: &[f64],
        alpha: f64,
        beta: f64,
        gamma: f64,
        season_len: usize,
        horizon: usize,
    ) -> Vec<f64> {
        if data.len() < season_len * 2 || season_len < 2 {
            // Fall back to EMA when there is not enough data.
            return Self::ema_forecast(data, alpha, horizon);
        }

        // Initialise level as mean of first season.
        let mut level: f64 = data[..season_len].iter().sum::<f64>() / season_len as f64;

        // Initialise trend as average first-difference across first two seasons.
        let mut trend: f64 = (0..season_len)
            .map(|i| (data[season_len + i] - data[i]) / season_len as f64)
            .sum::<f64>()
            / season_len as f64;

        // Initialise seasonal indices.
        let mean0: f64 = data[..season_len].iter().sum::<f64>() / season_len as f64;
        let mut seasonal: Vec<f64> = data[..season_len]
            .iter()
            .map(|&x| if mean0.abs() < 1e-12 { 1.0 } else { x / mean0 })
            .collect();

        let mut result: Vec<f64> = Vec::with_capacity(data.len() + horizon);

        for (t, &y) in data.iter().enumerate() {
            let s_idx = t % season_len;
            let prev_level = level;
            let prev_trend = trend;
            let prev_seasonal = seasonal[s_idx];

            // Avoid division by near-zero seasonal index.
            let s_safe = if prev_seasonal.abs() < 1e-12 { 1.0 } else { prev_seasonal };

            level = alpha * (y / s_safe) + (1.0 - alpha) * (prev_level + prev_trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * prev_trend;
            seasonal[s_idx] = gamma * (y / level.max(1e-12)) + (1.0 - gamma) * prev_seasonal;

            result.push((level + trend) * seasonal[s_idx]);
        }

        // Out-of-sample forecast.
        for h in 1..=horizon {
            let s_idx = (data.len() + h - 1) % season_len;
            let forecast = (level + trend * h as f64) * seasonal[s_idx];
            result.push(forecast.max(0.0));
        }

        result
    }

    /// Project a linear trend (OLS) `horizon` steps beyond the training data.
    ///
    /// Returns fitted values for the training period followed by `horizon`
    /// extrapolated values.
    pub fn linear_extrapolation(data: &[f64], horizon: usize) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![0.0; horizon];
        }
        if n == 1 {
            return std::iter::repeat(data[0]).take(1 + horizon).collect();
        }

        let n_f = n as f64;
        let x_mean = (n_f - 1.0) / 2.0;
        let y_mean: f64 = data.iter().sum::<f64>() / n_f;

        let num: f64 = data
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        let den: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();

        let slope = if den.abs() < 1e-12 { 0.0 } else { num / den };
        let intercept = y_mean - slope * x_mean;

        let mut result: Vec<f64> = (0..(n + horizon))
            .map(|i| (intercept + slope * i as f64).max(0.0))
            .collect();

        // Replace training portion with smoothed fitted values.
        for (i, v) in result.iter_mut().enumerate().take(n) {
            *v = (intercept + slope * i as f64).max(0.0);
        }
        result
    }

    /// EMA forecast: smooth the series then carry forward the last observed trend.
    ///
    /// Returns fitted values for the training data followed by `horizon`
    /// constant-trend extrapolations.
    pub fn ema_forecast(data: &[f64], alpha: f64, horizon: usize) -> Vec<f64> {
        if data.is_empty() {
            return vec![0.0; horizon];
        }

        let alpha = alpha.clamp(0.01, 0.99);
        let mut ema = data[0];
        let mut fitted: Vec<f64> = Vec::with_capacity(data.len() + horizon);
        fitted.push(ema);

        for &y in data.iter().skip(1) {
            ema = alpha * y + (1.0 - alpha) * ema;
            fitted.push(ema);
        }

        // Last-period trend for forward projection.
        let last_trend = if data.len() >= 2 {
            fitted[fitted.len() - 1] - fitted[fitted.len() - 2]
        } else {
            0.0
        };

        let last_ema = ema;
        for h in 1..=horizon {
            fitted.push((last_ema + last_trend * h as f64).max(0.0));
        }

        fitted
    }

    /// Compute symmetric confidence intervals using a normal approximation.
    ///
    /// `confidence` is a probability (e.g. 0.95) mapped to a z-score.
    /// Returns `(lower, upper)` pairs parallel to `forecast`.
    pub fn confidence_interval(
        forecast: &[f64],
        residuals: &[f64],
        confidence: f64,
    ) -> Vec<(f64, f64)> {
        if forecast.is_empty() {
            return vec![];
        }
        let confidence = confidence.clamp(0.5, 0.9999);

        // Residual standard deviation.
        let n = residuals.len() as f64;
        let std_dev = if residuals.is_empty() || n < 2.0 {
            forecast.iter().cloned().fold(0.0_f64, f64::max) * 0.1 + 1e-6
        } else {
            let mean = residuals.iter().sum::<f64>() / n;
            let var = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt()
        };

        // z-score via probit approximation (Beasley-Springer-Moro).
        let z = probit(confidence);

        forecast
            .iter()
            .enumerate()
            .map(|(h, &f)| {
                // Widen interval proportionally to horizon (uncertainty growth).
                let se = std_dev * (1.0 + h as f64 * 0.05).sqrt();
                let margin = z * se;
                ((f - margin).max(0.0), f + margin)
            })
            .collect()
    }

    /// Detect the dominant seasonality period in `data` via autocorrelation.
    ///
    /// Returns the lag with the highest ACF value in `[2, data.len()/2]`,
    /// or `None` if no clear season is found (ACF < 0.3).
    pub fn detect_seasonality(data: &[f64]) -> Option<usize> {
        let n = data.len();
        if n < 4 {
            return None;
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        if variance < 1e-12 {
            return None;
        }

        let max_lag = (n / 2).max(2);
        let mut best_lag = 0usize;
        let mut best_acf = 0.0_f64;

        for lag in 2..=max_lag {
            let acf: f64 = (0..(n - lag))
                .map(|i| (data[i] - mean) * (data[i + lag] - mean))
                .sum::<f64>()
                / (n as f64 * variance);
            if acf > best_acf {
                best_acf = acf;
                best_lag = lag;
            }
        }

        if best_acf >= 0.3 {
            Some(best_lag)
        } else {
            None
        }
    }

    /// Generate `horizon` [`ForecastPoint`]s using the given model and period.
    ///
    /// `historical` is a slice of cost values in chronological order.
    /// The first forecast period starts immediately after the last observation,
    /// anchored at Unix epoch 0 for pure relative indexing when a real timestamp
    /// is unavailable.
    pub fn forecast(
        &self,
        historical: &[f64],
        model: &ForecastModel,
        horizon: usize,
        period: ForecastPeriod,
    ) -> Vec<ForecastPoint> {
        if historical.is_empty() || horizon == 0 {
            return vec![];
        }

        // Generate out-of-sample forecasts.
        let forecasted: Vec<f64> = match model {
            ForecastModel::HoltWinters { alpha, beta, gamma, season_len } => {
                let full =
                    Self::holt_winters_forecast(historical, *alpha, *beta, *gamma, *season_len, horizon);
                full.into_iter().skip(historical.len()).collect()
            }
            ForecastModel::LinearTrend { slope, intercept } => {
                let n = historical.len();
                (1..=horizon)
                    .map(|h| (intercept + slope * (n + h - 1) as f64).max(0.0))
                    .collect()
            }
            ForecastModel::EMA { alpha } => {
                let full = Self::ema_forecast(historical, *alpha, horizon);
                full.into_iter().skip(historical.len()).collect()
            }
            ForecastModel::ARIMA { .. } => {
                // Delegate to linear extrapolation.
                let full = Self::linear_extrapolation(historical, horizon);
                full.into_iter().skip(historical.len()).collect()
            }
        };

        // Compute residuals from fitted values on training data (linear fallback).
        let fitted_train = Self::linear_extrapolation(historical, 0);
        let residuals: Vec<f64> = historical
            .iter()
            .zip(fitted_train.iter())
            .map(|(y, yh)| y - yh)
            .collect();

        let cis = Self::confidence_interval(&forecasted, &residuals, 0.95);

        let dur = period.duration_secs();
        let now_offset = historical.len() as u64;

        forecasted
            .iter()
            .enumerate()
            .map(|(i, &fc)| {
                let start = (now_offset + i as u64) * dur;
                let end = start + dur;
                let (lo, hi) = cis.get(i).copied().unwrap_or((fc * 0.9, fc * 1.1));
                ForecastPoint {
                    period_start: start,
                    period_end: end,
                    forecast_cost: fc,
                    lower_bound: lo,
                    upper_bound: hi,
                    confidence: 0.95,
                }
            })
            .collect()
    }

    /// Select the best model for `historical` data using AIC.
    ///
    /// Candidate models: EMA(0.3), LinearTrend, HoltWinters(auto-fit).
    /// Returns the model with the lowest AIC score.
    pub fn best_model(historical: &[f64]) -> ForecastModel {
        if historical.len() < 4 {
            return ForecastModel::EMA { alpha: 0.3 };
        }

        // Evaluate EMA.
        let ema_fitted = Self::ema_forecast(historical, 0.3, 0);
        let ema_aic = aic(&ema_fitted[..historical.len()], historical, 1);

        // Evaluate Linear.
        let lin_fitted = Self::linear_extrapolation(historical, 0);
        let lin_aic = aic(&lin_fitted[..historical.len()], historical, 2);

        // Evaluate Holt-Winters if enough data.
        let season_len = Self::detect_seasonality(historical).unwrap_or(7).max(2);
        let hw_aic = if historical.len() >= season_len * 2 {
            let (a, b, g) = Self::fit_holt_winters(historical, season_len);
            let hw_fitted =
                Self::holt_winters_forecast(historical, a, b, g, season_len, 0);
            let k = 3 + season_len; // alpha + beta + gamma + seasonal indices
            aic(&hw_fitted[..historical.len().min(hw_fitted.len())], historical, k)
        } else {
            f64::MAX
        };

        if ema_aic <= lin_aic && ema_aic <= hw_aic {
            ForecastModel::EMA { alpha: 0.3 }
        } else if lin_aic <= hw_aic {
            // Compute OLS params.
            let n = historical.len() as f64;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean = historical.iter().sum::<f64>() / n;
            let num: f64 = historical
                .iter()
                .enumerate()
                .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
                .sum();
            let den: f64 = (0..historical.len())
                .map(|i| (i as f64 - x_mean).powi(2))
                .sum();
            let slope = if den.abs() < 1e-12 { 0.0 } else { num / den };
            let intercept = y_mean - slope * x_mean;
            ForecastModel::LinearTrend { slope, intercept }
        } else {
            let (a, b, g) = Self::fit_holt_winters(historical, season_len);
            ForecastModel::HoltWinters {
                alpha: a,
                beta: b,
                gamma: g,
                season_len,
            }
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Compute AIC = 2k - 2*ln(L) where L is the Gaussian likelihood.
fn aic(fitted: &[f64], actual: &[f64], k: usize) -> f64 {
    let n = actual.len().min(fitted.len());
    if n == 0 {
        return f64::MAX;
    }
    let mse: f64 = actual
        .iter()
        .zip(fitted.iter())
        .map(|(y, yh)| (y - yh).powi(2))
        .sum::<f64>()
        / n as f64;
    if mse < 1e-15 {
        return 2.0 * k as f64; // perfect fit
    }
    let log_lik = -0.5 * n as f64 * (1.0 + (2.0 * PI * mse).ln());
    2.0 * k as f64 - 2.0 * log_lik
}

/// Rational approximation of the probit function (inverse normal CDF).
/// Accurate to ~1e-4 for p ∈ [0.5, 0.9999].
fn probit(p: f64) -> f64 {
    // Use the Peter Acklam approximation for the inner region.
    let p = p.clamp(0.5001, 0.9999);
    // We only need positive z (one-sided from 0.5), so reflect.
    let q = if p > 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * (1.0 - q).ln()).sqrt();
    let c = [2.515517, 0.802853, 0.010328];
    let d = [1.432788, 0.189269, 0.001308];
    let num = c[0] + t * (c[1] + t * c[2]);
    let den = 1.0 + t * (d[0] + t * (d[1] + t * d[2]));
    let z = t - num / den;
    if p >= 0.5 { z } else { -z }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_data() -> Vec<f64> {
        (0..24).map(|i| 10.0 + i as f64 * 0.5).collect()
    }

    #[test]
    fn linear_extrapolation_trend() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = CostForecaster::linear_extrapolation(&data, 2);
        assert_eq!(result.len(), 7);
        // Out-of-sample should continue the trend upward.
        assert!(result[5] > result[4]);
    }

    #[test]
    fn ema_forecast_length() {
        let data = simple_data();
        let result = CostForecaster::ema_forecast(&data, 0.3, 5);
        assert_eq!(result.len(), data.len() + 5);
    }

    #[test]
    fn holt_winters_returns_non_empty() {
        let data: Vec<f64> = (0..28).map(|i| 10.0 + (i as f64).sin()).collect();
        let result = CostForecaster::holt_winters_forecast(&data, 0.3, 0.1, 0.1, 7, 4);
        assert!(!result.is_empty());
    }

    #[test]
    fn confidence_interval_length_matches() {
        let forecast = vec![10.0, 11.0, 12.0];
        let residuals = vec![0.5, -0.3, 0.2];
        let cis = CostForecaster::confidence_interval(&forecast, &residuals, 0.95);
        assert_eq!(cis.len(), 3);
        for (lo, hi) in &cis {
            assert!(lo <= hi);
        }
    }

    #[test]
    fn detect_seasonality_weekly() {
        // Simulate data with a period-7 seasonal pattern.
        let data: Vec<f64> = (0..42)
            .map(|i| 10.0 + 3.0 * (2.0 * PI * i as f64 / 7.0).sin())
            .collect();
        let s = CostForecaster::detect_seasonality(&data);
        // Should detect some seasonality (may not be exactly 7 with float noise).
        assert!(s.is_some());
    }

    #[test]
    fn forecast_returns_horizon_points() {
        let fc = CostForecaster::default();
        let data = simple_data();
        let model = CostForecaster::best_model(&data);
        let points = fc.forecast(&data, &model, 7, ForecastPeriod::Daily);
        assert_eq!(points.len(), 7);
        for p in &points {
            assert!(p.lower_bound <= p.forecast_cost + 1e-6);
            assert!(p.upper_bound >= p.forecast_cost - 1e-6);
        }
    }

    #[test]
    fn best_model_short_data_returns_ema() {
        let data = vec![1.0, 2.0, 3.0];
        let m = CostForecaster::best_model(&data);
        assert!(matches!(m, ForecastModel::EMA { .. }));
    }
}
