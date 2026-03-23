//! Time-series trend analysis: OLS linear trend, moving averages, exponential smoothing,
//! seasonality detection, STL-style decomposition, and Holt's forecasting.

use std::fmt;

/// A single timestamped observation.
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Unix timestamp (seconds since epoch).
    pub timestamp_unix: u64,
    /// Observed value.
    pub value: f64,
}

/// Direction of a detected trend.
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Slope is meaningfully positive.
    Rising,
    /// Slope is meaningfully negative.
    Falling,
    /// Slope is near zero.
    Flat,
    /// High residual variance relative to trend.
    Volatile,
}

impl fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rising => write!(f, "Rising"),
            Self::Falling => write!(f, "Falling"),
            Self::Flat => write!(f, "Flat"),
            Self::Volatile => write!(f, "Volatile"),
        }
    }
}

/// Detected seasonality period.
#[derive(Debug, Clone, PartialEq)]
pub enum SeasonalityPeriod {
    /// ~24-observation cycle.
    Hourly,
    /// ~168-observation cycle.
    Daily,
    /// ~720-observation cycle.
    Weekly,
    /// ~2160-observation cycle (approximate month).
    Monthly,
    /// No significant seasonality detected.
    None,
}

/// Decomposed time-series components.
#[derive(Debug, Clone)]
pub struct TrendComponents {
    /// Smoothed trend component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Residual (remainder) component.
    pub residual: Vec<f64>,
}

/// Full result of a trend analysis.
#[derive(Debug, Clone)]
pub struct TrendResult {
    /// Detected direction.
    pub direction: TrendDirection,
    /// OLS slope (units per observation).
    pub slope: f64,
    /// OLS coefficient of determination (0–1).
    pub r_squared: f64,
    /// Detected seasonality period.
    pub seasonality: SeasonalityPeriod,
    /// Indices of anomalous observations (|z-score| > 3).
    pub anomaly_indices: Vec<usize>,
    /// Forward forecast values (length = `horizon` passed to [`TrendAnalyzer::forecast`]).
    pub forecast: Vec<f64>,
}

/// Collection of trend-analysis algorithms.
pub struct TrendAnalyzer;

impl TrendAnalyzer {
    /// Ordinary least-squares linear regression.
    ///
    /// Returns `(slope, intercept, r_squared)`.
    pub fn linear_trend(points: &[DataPoint]) -> (f64, f64, f64) {
        let n = points.len();
        if n < 2 {
            return (0.0, points.first().map(|p| p.value).unwrap_or(0.0), 0.0);
        }

        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = points.iter().map(|p| p.value).collect();

        let n_f = n as f64;
        let sum_x: f64 = xs.iter().sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xx: f64 = xs.iter().map(|&x| x * x).sum();
        let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();

        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return (0.0, sum_y / n_f, 0.0);
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n_f;

        // R²
        let mean_y = sum_y / n_f;
        let ss_tot: f64 = ys.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot.abs() < f64::EPSILON {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        (slope, intercept, r_squared)
    }

    /// Simple moving average with the given `window` size.
    ///
    /// The first `window - 1` values use a growing window (left-aligned).
    pub fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
        if values.is_empty() || window == 0 {
            return Vec::new();
        }
        values
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let start = if i + 1 >= window { i + 1 - window } else { 0 };
                let slice = &values[start..=i];
                slice.iter().sum::<f64>() / slice.len() as f64
            })
            .collect()
    }

    /// Exponential moving average (EMA / single exponential smoothing).
    ///
    /// `alpha` in `(0, 1]` — higher alpha = more weight on recent values.
    pub fn exponential_smoothing(values: &[f64], alpha: f64) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        let alpha = alpha.clamp(1e-6, 1.0);
        let mut result = Vec::with_capacity(values.len());
        let mut s = values[0];
        result.push(s);
        for &v in values.iter().skip(1) {
            s = alpha * v + (1.0 - alpha) * s;
            result.push(s);
        }
        result
    }

    /// Holt's double exponential smoothing (level + trend).
    ///
    /// - `alpha`: level smoothing factor
    /// - `beta`: trend smoothing factor
    pub fn double_exponential_smoothing(values: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        let alpha = alpha.clamp(1e-6, 1.0);
        let beta = beta.clamp(1e-6, 1.0);

        let mut result = Vec::with_capacity(values.len());
        let mut level = values[0];
        let mut trend = if values.len() > 1 {
            values[1] - values[0]
        } else {
            0.0
        };

        result.push(level + trend);

        for &v in values.iter().skip(1) {
            let prev_level = level;
            level = alpha * v + (1.0 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;
            result.push(level + trend);
        }
        result
    }

    /// Detect seasonality by computing autocorrelation at lags 24, 168, and 720.
    pub fn detect_seasonality(values: &[f64]) -> SeasonalityPeriod {
        if values.len() < 25 {
            return SeasonalityPeriod::None;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        if variance < f64::EPSILON {
            return SeasonalityPeriod::None;
        }

        let autocorr = |lag: usize| -> f64 {
            if lag >= values.len() {
                return 0.0;
            }
            let n = values.len() - lag;
            let cov: f64 = (0..n)
                .map(|i| (values[i] - mean) * (values[i + lag] - mean))
                .sum::<f64>()
                / n as f64;
            cov / variance
        };

        let ac24 = autocorr(24);
        let ac168 = autocorr(168);
        let ac720 = autocorr(720);

        // Pick the strongest significant lag (threshold 0.3)
        let threshold = 0.3_f64;
        if ac720 > threshold {
            SeasonalityPeriod::Weekly
        } else if ac168 > threshold {
            SeasonalityPeriod::Daily
        } else if ac24 > threshold {
            SeasonalityPeriod::Hourly
        } else {
            // Monthly lag check — just look if ac720 is at least slightly positive
            if ac720 > 0.1 {
                SeasonalityPeriod::Monthly
            } else {
                SeasonalityPeriod::None
            }
        }
    }

    /// Simplified STL decomposition.
    ///
    /// - `trend` = moving average with window = `period`
    /// - `seasonal` = mean residual per cycle position (length-`period` pattern)
    /// - `residual` = value − trend − seasonal
    pub fn stl_decompose(values: &[f64], period: usize) -> TrendComponents {
        if values.is_empty() || period == 0 {
            return TrendComponents {
                trend: Vec::new(),
                seasonal: Vec::new(),
                residual: Vec::new(),
            };
        }

        let trend = Self::moving_average(values, period.min(values.len()));

        // Compute seasonal means per cycle position
        let mut seasonal_sums = vec![0.0_f64; period];
        let mut seasonal_counts = vec![0_usize; period];
        for (i, (&v, &t)) in values.iter().zip(trend.iter()).enumerate() {
            let pos = i % period;
            seasonal_sums[pos] += v - t;
            seasonal_counts[pos] += 1;
        }
        let seasonal_means: Vec<f64> = seasonal_sums
            .iter()
            .zip(seasonal_counts.iter())
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect();

        let seasonal: Vec<f64> = (0..values.len()).map(|i| seasonal_means[i % period]).collect();
        let residual: Vec<f64> = values
            .iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((&v, &t), &s)| v - t - s)
            .collect();

        TrendComponents { trend, seasonal, residual }
    }

    /// Forecast `horizon` future values using Holt's double exponential smoothing.
    ///
    /// Alpha and beta are set to 0.3 and 0.1 respectively for stability.
    pub fn forecast(points: &[DataPoint], horizon: usize) -> Vec<f64> {
        if points.is_empty() || horizon == 0 {
            return Vec::new();
        }
        let values: Vec<f64> = points.iter().map(|p| p.value).collect();
        let alpha = 0.3;
        let beta = 0.1;

        // Compute Holt smoothed series to get final level and trend.
        let n = values.len();
        let mut level = values[0];
        let mut trend = if n > 1 { values[1] - values[0] } else { 0.0 };

        for &v in values.iter().skip(1) {
            let prev_level = level;
            level = alpha * v + (1.0 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;
        }

        (1..=horizon).map(|h| level + trend * h as f64).collect()
    }

    /// Run a full analysis on the supplied data points.
    pub fn analyze(points: &[DataPoint]) -> TrendResult {
        let (slope, _intercept, r_squared) = Self::linear_trend(points);

        // Direction
        let direction = if r_squared < 0.2 {
            TrendDirection::Volatile
        } else if slope > 0.01 {
            TrendDirection::Rising
        } else if slope < -0.01 {
            TrendDirection::Falling
        } else {
            TrendDirection::Flat
        };

        let values: Vec<f64> = points.iter().map(|p| p.value).collect();

        let seasonality = Self::detect_seasonality(&values);

        // Anomaly detection via z-score
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n.max(1.0);
        let std_dev = (values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n.max(1.0)).sqrt();
        let anomaly_indices: Vec<usize> = if std_dev > f64::EPSILON {
            values
                .iter()
                .enumerate()
                .filter(|(_, &v)| ((v - mean) / std_dev).abs() > 3.0)
                .map(|(i, _)| i)
                .collect()
        } else {
            Vec::new()
        };

        let forecast = Self::forecast(points, 10);

        TrendResult {
            direction,
            slope,
            r_squared,
            seasonality,
            anomaly_indices,
            forecast,
        }
    }
}
