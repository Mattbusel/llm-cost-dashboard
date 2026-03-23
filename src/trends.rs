//! # Historical Trend Analytics
//!
//! Aggregates cost events into time-series data and computes trends:
//! - Daily / weekly / monthly spend buckets
//! - Moving averages (7-day, 30-day)
//! - Period-over-period comparison (this week vs last week)
//! - Cost per model over time
//! - Spend acceleration detection

use std::collections::{BTreeMap, HashMap};

// ── Data types ───────────────────────────────────────────────────────────────

/// Aggregated spend for a single calendar day.
#[derive(Debug, Clone, Default)]
pub struct DailySpend {
    /// Calendar date in `YYYY-MM-DD` format.
    pub date: String,
    /// Total USD cost for this day.
    pub total_usd: f64,
    /// Per-model cost breakdown: `model_id -> USD`.
    pub by_model: HashMap<String, f64>,
    /// Number of cost events recorded on this day.
    pub request_count: u64,
}

/// Summary report covering a contiguous time window.
#[derive(Debug, Clone)]
pub struct TrendReport {
    /// Inclusive start date of the current period (`YYYY-MM-DD`).
    pub period_start: String,
    /// Inclusive end date of the current period (`YYYY-MM-DD`).
    pub period_end: String,
    /// Total spend in the current period (USD).
    pub total_spend_usd: f64,
    /// Total spend in the immediately preceding equal-length period (USD).
    pub prev_period_spend_usd: f64,
    /// Period-over-period percentage change.
    ///
    /// `+50.0` means spend increased by 50 %.  `NaN` when previous period had
    /// zero spend (avoid divide-by-zero at the display layer).
    pub period_over_period_pct: f64,
    /// Average daily spend over the current period (USD).
    pub daily_avg_usd: f64,
    /// Day with the highest spend in the current period.
    pub peak_day: Option<DailySpend>,
    /// Day with the lowest spend in the current period (only days with spend).
    pub slowest_day: Option<DailySpend>,
    /// Model with the highest cumulative spend in the current period.
    pub top_model: Option<(String, f64)>,
    /// 7-day moving average ending on the last day of data (USD/day).
    pub moving_avg_7d: f64,
    /// 30-day moving average ending on the last day of data (USD/day).
    pub moving_avg_30d: f64,
    /// `true` when the most recent day's spend exceeds the 7-day moving
    /// average — a simple acceleration signal.
    pub is_accelerating: bool,
    /// Linear projection of total spend through the end of the current
    /// calendar month, based on the daily average so far this month.
    pub projected_month_end_usd: f64,
}

// ── Analyzer ─────────────────────────────────────────────────────────────────

/// Time-series cost accumulator.
///
/// Records are bucketed by calendar day (UTC).  Call [`TrendAnalyzer::record`]
/// for each cost event, then use the reporting methods to query trends.
///
/// # Example
///
/// ```rust
/// use llm_cost_dashboard::trends::TrendAnalyzer;
///
/// let mut analyzer = TrendAnalyzer::new();
/// // Use a fixed timestamp: 2026-01-15 noon UTC = 1768472400 seconds approx.
/// analyzer.record(1_768_472_400, "gpt-4o-mini", 0.05);
/// let report = analyzer.trend_report(7);
/// assert!(report.total_spend_usd >= 0.0);
/// ```
pub struct TrendAnalyzer {
    /// Keyed by `"YYYY-MM-DD"`, BTreeMap for sorted iteration.
    daily_spend: BTreeMap<String, DailySpend>,
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalyzer {
    /// Create an empty analyzer.
    pub fn new() -> Self {
        Self {
            daily_spend: BTreeMap::new(),
        }
    }

    /// Record a cost event.
    ///
    /// `timestamp_secs` is a Unix timestamp (seconds since the epoch, UTC).
    pub fn record(&mut self, timestamp_secs: u64, model: &str, cost_usd: f64) {
        let date = Self::date_string(timestamp_secs);
        let entry = self.daily_spend.entry(date.clone()).or_insert_with(|| DailySpend {
            date,
            ..Default::default()
        });
        entry.total_usd += cost_usd;
        entry.request_count += 1;
        *entry.by_model.entry(model.to_string()).or_insert(0.0) += cost_usd;
    }

    /// Ordered slice of all recorded daily spend entries (oldest first).
    pub fn daily_series(&self) -> Vec<&DailySpend> {
        self.daily_spend.values().collect()
    }

    /// Build a [`TrendReport`] for the most recent `days` calendar days vs
    /// the immediately preceding `days` calendar days.
    ///
    /// The "current period" ends on the most recent calendar day that has any
    /// recorded spend (or today if the analyzer is empty).
    pub fn trend_report(&self, days: u32) -> TrendReport {
        let days = days.max(1) as i64;

        // Determine the last day we have data for, falling back to today.
        let last_date = self
            .daily_spend
            .keys()
            .next_back()
            .cloned()
            .unwrap_or_else(|| Self::today_string());

        // current period: [last_date - (days-1), last_date]
        let period_end = last_date.clone();
        let period_start = Self::subtract_days(&period_end, (days - 1) as u32);

        // previous period: [period_start - days, period_start - 1]
        let prev_end = Self::subtract_days(&period_start, 1);
        let prev_start = Self::subtract_days(&prev_end, (days - 1) as u32);

        // Collect current-period days.
        let current_days: Vec<&DailySpend> = self
            .daily_spend
            .range(period_start.clone()..=period_end.clone())
            .map(|(_, v)| v)
            .collect();

        let prev_days: Vec<&DailySpend> = self
            .daily_spend
            .range(prev_start..=prev_end)
            .map(|(_, v)| v)
            .collect();

        let total_spend: f64 = current_days.iter().map(|d| d.total_usd).sum();
        let prev_total: f64 = prev_days.iter().map(|d| d.total_usd).sum();

        let pop_pct = if prev_total == 0.0 {
            f64::NAN
        } else {
            ((total_spend - prev_total) / prev_total) * 100.0
        };

        let daily_avg = if days > 0 {
            total_spend / days as f64
        } else {
            0.0
        };

        let peak_day = current_days
            .iter()
            .max_by(|a, b| a.total_usd.partial_cmp(&b.total_usd).unwrap_or(std::cmp::Ordering::Equal))
            .map(|d| (*d).clone());

        let slowest_day = current_days
            .iter()
            .filter(|d| d.total_usd > 0.0)
            .min_by(|a, b| a.total_usd.partial_cmp(&b.total_usd).unwrap_or(std::cmp::Ordering::Equal))
            .map(|d| (*d).clone());

        // Top model across the current period.
        let mut model_totals: HashMap<String, f64> = HashMap::new();
        for day in &current_days {
            for (m, &c) in &day.by_model {
                *model_totals.entry(m.clone()).or_insert(0.0) += c;
            }
        }
        let top_model = model_totals
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let ma7 = self.moving_avg_7d();
        let ma30 = self.moving_avg_30d();

        let last_day_spend = self
            .daily_spend
            .get(&last_date)
            .map(|d| d.total_usd)
            .unwrap_or(0.0);
        let is_accelerating = ma7 > 0.0 && last_day_spend > ma7;

        // Month-end projection: daily average * remaining days in month.
        let projected = Self::project_month_end(total_spend, &period_start, &period_end);

        TrendReport {
            period_start,
            period_end,
            total_spend_usd: total_spend,
            prev_period_spend_usd: prev_total,
            period_over_period_pct: pop_pct,
            daily_avg_usd: daily_avg,
            peak_day,
            slowest_day,
            top_model,
            moving_avg_7d: ma7,
            moving_avg_30d: ma30,
            is_accelerating,
            projected_month_end_usd: projected,
        }
    }

    /// 7-day simple moving average (USD/day) ending on the most recent
    /// recorded day.
    pub fn moving_avg_7d(&self) -> f64 {
        self.trailing_avg(7)
    }

    /// 30-day simple moving average (USD/day) ending on the most recent
    /// recorded day.
    pub fn moving_avg_30d(&self) -> f64 {
        self.trailing_avg(30)
    }

    /// Return an ASCII sparkline of daily spend over the last `days` days.
    ///
    /// The sparkline is exactly 20 characters wide and uses the Unicode block
    /// characters `▁▂▃▄▅▆▇█` to represent spend relative to the period max.
    /// Days with no spend render as a space character.
    pub fn sparkline(&self, days: u32) -> String {
        let days = days.max(1) as usize;
        const WIDTH: usize = 20;

        // Collect `days` worth of spend values aligned to today.
        let last_date = self
            .daily_spend
            .keys()
            .next_back()
            .cloned()
            .unwrap_or_else(|| Self::today_string());

        let mut values: Vec<f64> = Vec::with_capacity(days);
        for i in (0..days as u32).rev() {
            let date = Self::subtract_days(&last_date, i);
            values.push(
                self.daily_spend
                    .get(&date)
                    .map(|d| d.total_usd)
                    .unwrap_or(0.0),
            );
        }

        // Subsample / oversample to WIDTH.
        let sampled = Self::resample(&values, WIDTH);

        let max = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max <= 0.0 {
            return " ".repeat(WIDTH);
        }

        const BARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        sampled
            .iter()
            .map(|&v| {
                if v <= 0.0 {
                    ' '
                } else {
                    let idx = ((v / max) * (BARS.len() as f64 - 1.0))
                        .round()
                        .clamp(0.0, (BARS.len() - 1) as f64) as usize;
                    BARS[idx]
                }
            })
            .collect()
    }

    /// Export daily time series as CSV.
    ///
    /// Columns: `date,total_usd,request_count`
    pub fn to_csv(&self) -> String {
        let mut lines = vec!["date,total_usd,request_count".to_string()];
        for d in self.daily_spend.values() {
            lines.push(format!("{},{:.6},{}", d.date, d.total_usd, d.request_count));
        }
        lines.join("\n")
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Convert a Unix timestamp (seconds, UTC) to a `"YYYY-MM-DD"` string.
    fn date_string(timestamp_secs: u64) -> String {
        // Use integer arithmetic to avoid pulling in chrono for this pure fn.
        // Days since Unix epoch.
        let days = timestamp_secs / 86_400;
        // Gregorian calendar conversion (civil date algorithm by Howard Hinnant).
        let z = days as i64 + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = (z - era * 146_097) as u64;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        format!("{y:04}-{m:02}-{d:02}")
    }

    /// Return today's date as `"YYYY-MM-DD"` using the system clock.
    fn today_string() -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self::date_string(now)
    }

    /// Return the date `n` days before `date` (format `"YYYY-MM-DD"`).
    fn subtract_days(date: &str, n: u32) -> String {
        let ts = Self::date_to_unix(date);
        Self::date_string(ts.saturating_sub(n as u64 * 86_400))
    }

    /// Convert a `"YYYY-MM-DD"` string to a Unix timestamp (start of day UTC).
    fn date_to_unix(date: &str) -> u64 {
        // Parse YYYY-MM-DD.
        let parts: Vec<&str> = date.splitn(3, '-').collect();
        if parts.len() != 3 {
            return 0;
        }
        let y: i64 = parts[0].parse().unwrap_or(1970);
        let m: i64 = parts[1].parse().unwrap_or(1);
        let d: i64 = parts[2].parse().unwrap_or(1);
        // Days since Unix epoch via the same civil calendar algorithm.
        let y = if m <= 2 { y - 1 } else { y };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = (y - era * 400) as u64;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) as u64 + 2) / 5 + d as u64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146_097 + doe as i64 - 719_468;
        (days * 86_400) as u64
    }

    /// Simple moving average over the last `n` days of recorded data.
    fn trailing_avg(&self, n: usize) -> f64 {
        let last_date = match self.daily_spend.keys().next_back() {
            Some(d) => d.clone(),
            None => return 0.0,
        };
        let start = Self::subtract_days(&last_date, (n as u32).saturating_sub(1));
        let vals: Vec<f64> = self
            .daily_spend
            .range(start..=last_date)
            .map(|(_, v)| v.total_usd)
            .collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / n as f64
        }
    }

    /// Resample `src` to exactly `target_len` elements using linear
    /// interpolation indices (nearest-neighbour pick).
    fn resample(src: &[f64], target_len: usize) -> Vec<f64> {
        if src.is_empty() || target_len == 0 {
            return vec![0.0; target_len];
        }
        if src.len() == target_len {
            return src.to_vec();
        }
        (0..target_len)
            .map(|i| {
                let src_idx = (i as f64 / (target_len - 1) as f64 * (src.len() - 1) as f64)
                    .round() as usize;
                src[src_idx.min(src.len() - 1)]
            })
            .collect()
    }

    /// Estimate end-of-month spend given `total_so_far` accumulated between
    /// `period_start` and `period_end` (both `"YYYY-MM-DD"`).
    fn project_month_end(total_so_far: f64, period_start: &str, period_end: &str) -> f64 {
        // Days in period.
        let start_ts = Self::date_to_unix(period_start);
        let end_ts = Self::date_to_unix(period_end);
        let elapsed_days = if end_ts >= start_ts {
            ((end_ts - start_ts) / 86_400 + 1).max(1)
        } else {
            1
        };

        let daily_avg = total_so_far / elapsed_days as f64;

        // Days remaining in the month after `period_end`.
        let parts: Vec<&str> = period_end.splitn(3, '-').collect();
        let year: u64 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(2024);
        let month: u64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
        let day_of_month: u64 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
        let days_in_month = Self::days_in_month(year, month);
        let remaining = days_in_month.saturating_sub(day_of_month);

        total_so_far + daily_avg * remaining as f64
    }

    /// Return the number of days in a given year/month.
    fn days_in_month(year: u64, month: u64) -> u64 {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                // Gregorian leap year.
                if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                    29
                } else {
                    28
                }
            }
            _ => 30,
        }
    }

    /// Return `n` days ago as a `"YYYY-MM-DD"` string (relative to today).
    #[allow(dead_code)]
    fn days_ago(n: u32) -> String {
        Self::subtract_days(&Self::today_string(), n)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A fixed "anchor" timestamp: 2024-03-15 00:00:00 UTC.
    const DAY0: u64 = 1_710_460_800;
    const DAY_SECS: u64 = 86_400;

    fn days(n: u64) -> u64 {
        DAY0 + n * DAY_SECS
    }

    // 1. Empty analyzer has no daily entries.
    #[test]
    fn test_empty_daily_series() {
        let a = TrendAnalyzer::new();
        assert!(a.daily_series().is_empty());
    }

    // 2. Single record shows up in the series.
    #[test]
    fn test_single_record_in_series() {
        let mut a = TrendAnalyzer::new();
        a.record(DAY0, "gpt-4o-mini", 1.23);
        assert_eq!(a.daily_series().len(), 1);
    }

    // 3. Two records on the same day merge into one bucket.
    #[test]
    fn test_same_day_accumulation() {
        let mut a = TrendAnalyzer::new();
        a.record(DAY0 + 100, "gpt-4o-mini", 1.0);
        a.record(DAY0 + 200, "gpt-4o-mini", 2.0);
        assert_eq!(a.daily_series().len(), 1);
        assert!((a.daily_series()[0].total_usd - 3.0).abs() < 1e-9);
    }

    // 4. Records on different days produce multiple buckets.
    #[test]
    fn test_different_days_separate_buckets() {
        let mut a = TrendAnalyzer::new();
        a.record(days(0), "m", 1.0);
        a.record(days(1), "m", 2.0);
        a.record(days(2), "m", 3.0);
        assert_eq!(a.daily_series().len(), 3);
    }

    // 5. Per-model breakdown is tracked.
    #[test]
    fn test_per_model_breakdown() {
        let mut a = TrendAnalyzer::new();
        a.record(DAY0, "gpt-4o", 1.0);
        a.record(DAY0, "claude-sonnet-4-6", 2.0);
        let day = &a.daily_series()[0];
        assert!((day.by_model["gpt-4o"] - 1.0).abs() < 1e-9);
        assert!((day.by_model["claude-sonnet-4-6"] - 2.0).abs() < 1e-9);
    }

    // 6. date_string converts a known timestamp correctly.
    #[test]
    fn test_date_string_known_timestamp() {
        // 2024-03-15 00:00:00 UTC
        assert_eq!(TrendAnalyzer::date_string(DAY0), "2024-03-15");
    }

    // 7. subtract_days goes back the right number of days.
    #[test]
    fn test_subtract_days() {
        let result = TrendAnalyzer::subtract_days("2024-03-15", 5);
        assert_eq!(result, "2024-03-10");
    }

    // 8. to_csv header is correct.
    #[test]
    fn test_to_csv_header() {
        let a = TrendAnalyzer::new();
        assert!(a.to_csv().starts_with("date,total_usd,request_count"));
    }

    // 9. to_csv has rows for each day.
    #[test]
    fn test_to_csv_rows() {
        let mut a = TrendAnalyzer::new();
        a.record(days(0), "m", 1.0);
        a.record(days(1), "m", 2.0);
        let lines: Vec<&str> = a.to_csv().lines().collect();
        // 1 header + 2 data rows
        assert_eq!(lines.len(), 3);
    }

    // 10. moving_avg_7d returns 0 on empty analyzer.
    #[test]
    fn test_moving_avg_7d_empty() {
        assert_eq!(TrendAnalyzer::new().moving_avg_7d(), 0.0);
    }

    // 11. moving_avg_7d averages over 7 days.
    #[test]
    fn test_moving_avg_7d_value() {
        let mut a = TrendAnalyzer::new();
        // Record $7 spread over 7 consecutive days ($1/day).
        for i in 0..7u64 {
            a.record(days(i), "m", 1.0);
        }
        // Average = 7 / 7 = 1.0
        let avg = a.moving_avg_7d();
        assert!((avg - 1.0).abs() < 1e-9, "expected 1.0, got {avg}");
    }

    // 12. moving_avg_30d averages over 30 days.
    #[test]
    fn test_moving_avg_30d_value() {
        let mut a = TrendAnalyzer::new();
        for i in 0..30u64 {
            a.record(days(i), "m", 2.0); // $2/day
        }
        // total = 60, divided by 30 window = 2.0
        let avg = a.moving_avg_30d();
        assert!((avg - 2.0).abs() < 1e-9, "expected 2.0, got {avg}");
    }

    // 13. trend_report total_spend_usd is correct.
    #[test]
    fn test_trend_report_total_spend() {
        let mut a = TrendAnalyzer::new();
        for i in 0..7u64 {
            a.record(days(i), "m", 3.0);
        }
        let report = a.trend_report(7);
        assert!((report.total_spend_usd - 21.0).abs() < 1e-9);
    }

    // 14. period_over_period_pct is NaN when previous period is zero.
    #[test]
    fn test_pop_nan_when_no_prev() {
        let mut a = TrendAnalyzer::new();
        a.record(DAY0, "m", 5.0);
        let report = a.trend_report(7);
        assert!(report.period_over_period_pct.is_nan());
    }

    // 15. period_over_period_pct is 100% when spend doubles.
    #[test]
    fn test_pop_pct_doubling() {
        let mut a = TrendAnalyzer::new();
        // prev period: days 0-6 = $1/day = $7 total
        for i in 0..7u64 {
            a.record(days(i), "m", 1.0);
        }
        // current period: days 7-13 = $2/day = $14 total
        for i in 7..14u64 {
            a.record(days(i), "m", 2.0);
        }
        let report = a.trend_report(7);
        assert!(
            (report.period_over_period_pct - 100.0).abs() < 1e-6,
            "expected 100%, got {}",
            report.period_over_period_pct
        );
    }

    // 16. is_accelerating is true when last day > 7d avg.
    #[test]
    fn test_is_accelerating() {
        let mut a = TrendAnalyzer::new();
        for i in 0..6u64 {
            a.record(days(i), "m", 1.0); // low spend
        }
        a.record(days(6), "m", 100.0); // spike on last day
        let report = a.trend_report(7);
        assert!(report.is_accelerating);
    }

    // 17. sparkline is exactly 20 characters wide.
    #[test]
    fn test_sparkline_width() {
        let mut a = TrendAnalyzer::new();
        for i in 0..30u64 {
            a.record(days(i), "m", i as f64 + 1.0);
        }
        let spark = a.sparkline(30);
        assert_eq!(
            spark.chars().count(),
            20,
            "expected 20 chars, got {}",
            spark.chars().count()
        );
    }

    // 18. sparkline on empty analyzer is 20 spaces.
    #[test]
    fn test_sparkline_empty() {
        let a = TrendAnalyzer::new();
        let spark = a.sparkline(7);
        assert_eq!(spark, " ".repeat(20));
    }

    // 19. days_in_month returns correct values including leap year.
    #[test]
    fn test_days_in_month() {
        assert_eq!(TrendAnalyzer::days_in_month(2024, 2), 29); // leap
        assert_eq!(TrendAnalyzer::days_in_month(2023, 2), 28); // non-leap
        assert_eq!(TrendAnalyzer::days_in_month(2024, 1), 31);
        assert_eq!(TrendAnalyzer::days_in_month(2024, 4), 30);
    }

    // 20. round-trip date_to_unix -> date_string is idempotent.
    #[test]
    fn test_date_roundtrip() {
        let dates = ["2024-01-01", "2024-02-29", "2025-12-31", "2026-03-22"];
        for &d in &dates {
            let ts = TrendAnalyzer::date_to_unix(d);
            let result = TrendAnalyzer::date_string(ts);
            assert_eq!(result, d, "round-trip failed for {d}");
        }
    }

    // 21. request_count increments per record call.
    #[test]
    fn test_request_count() {
        let mut a = TrendAnalyzer::new();
        a.record(DAY0, "m", 1.0);
        a.record(DAY0, "m", 1.0);
        a.record(DAY0, "m", 1.0);
        assert_eq!(a.daily_series()[0].request_count, 3);
    }

    // 22. trend_report top_model is the most expensive one.
    #[test]
    fn test_trend_report_top_model() {
        let mut a = TrendAnalyzer::new();
        a.record(DAY0, "cheap-model", 0.10);
        a.record(DAY0, "expensive-model", 9.99);
        let report = a.trend_report(1);
        let (model, _) = report.top_model.unwrap();
        assert_eq!(model, "expensive-model");
    }
}
