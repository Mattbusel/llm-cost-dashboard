//! # Export Scheduler
//!
//! A cron-like scheduler that automatically exports cost reports on a
//! configurable schedule.  Reports can be written as CSV, JSON, or a simple
//! HTML summary, and optionally delivered via a webhook payload.
//!
//! ## Schedule format
//!
//! The scheduler uses a simplified five-field cron expression:
//!
//! ```text
//! ┌──── minute  (0-59)
//! │ ┌─── hour    (0-23)
//! │ │ ┌── day-of-month (1-31, or * for every day)
//! │ │ │ ┌─ month (1-12, or * for every month)
//! │ │ │ │ ┌ day-of-week (0-6, Sun=0; or MON, TUE, … abbreviations; * for any)
//! 0 9 * * MON   → every Monday at 09:00
//! 30 18 1 * *   → 1st of every month at 18:30
//! 0 * * * *     → every hour
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use llm_cost_dashboard::scheduler::{ExportSchedule, ScheduledExportFormat, Scheduler};
//! use llm_cost_dashboard::cost::CostLedger;
//!
//! let ledger = CostLedger::new();
//! let schedule = ExportSchedule {
//!     cron: "0 9 * * MON".to_string(),
//!     format: ScheduledExportFormat::Csv,
//!     output_dir: ".".to_string(),
//!     webhook_url: None,
//! };
//! let mut scheduler = Scheduler::new(schedule);
//! // In a real application you would call scheduler.tick(&ledger) periodically.
//! ```

use chrono::{DateTime, Datelike, Timelike, Utc};

use crate::cost::CostLedger;
use crate::error::DashboardError;

// ── Schedule format ───────────────────────────────────────────────────────────

/// Output format produced by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduledExportFormat {
    /// Comma-separated values.
    Csv,
    /// Pretty-printed JSON array.
    Json,
    /// Minimal standalone HTML report.
    Html,
}

impl ScheduledExportFormat {
    /// File extension (without the leading dot).
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Json => "json",
            Self::Html => "html",
        }
    }
}

/// Configuration for a single scheduled export.
#[derive(Debug, Clone)]
pub struct ExportSchedule {
    /// Simplified five-field cron expression (see module docs).
    pub cron: String,
    /// Output format.
    pub format: ScheduledExportFormat,
    /// Directory where exported files are written.
    pub output_dir: String,
    /// Optional webhook URL to POST the report to (as a text body).
    pub webhook_url: Option<String>,
}

// ── Cron parsing ─────────────────────────────────────────────────────────────

/// A parsed cron schedule.
#[derive(Debug, Clone)]
pub struct CronSchedule {
    /// Minute field (None = wildcard).
    pub minute: Option<u32>,
    /// Hour field (None = wildcard).
    pub hour: Option<u32>,
    /// Day-of-month field (None = wildcard).
    pub day_of_month: Option<u32>,
    /// Month field (None = wildcard).
    pub month: Option<u32>,
    /// Day-of-week field (None = wildcard; 0 = Sunday).
    pub day_of_week: Option<u32>,
}

impl CronSchedule {
    /// Parse a five-field cron expression.
    ///
    /// Returns [`DashboardError::Ledger`] on malformed input (re-using the
    /// generic string-error variant to avoid adding a new error variant).
    pub fn parse(expr: &str) -> Result<Self, DashboardError> {
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(DashboardError::Ledger(format!(
                "invalid cron expression '{expr}': expected 5 fields, got {}",
                fields.len()
            )));
        }

        fn parse_field(s: &str, name: &str) -> Result<Option<u32>, DashboardError> {
            if s == "*" {
                return Ok(None);
            }
            // Accept day-of-week abbreviations.
            let upper = s.to_uppercase();
            let s_norm = match upper.as_str() {
                "SUN" => "0",
                "MON" => "1",
                "TUE" => "2",
                "WED" => "3",
                "THU" => "4",
                "FRI" => "5",
                "SAT" => "6",
                other => other,
            };
            s_norm.parse::<u32>().map(Some).map_err(|_| {
                DashboardError::Ledger(format!("invalid cron field '{s}' for {name}"))
            })
        }

        Ok(Self {
            minute: parse_field(fields[0], "minute")?,
            hour: parse_field(fields[1], "hour")?,
            day_of_month: parse_field(fields[2], "day-of-month")?,
            month: parse_field(fields[3], "month")?,
            day_of_week: parse_field(fields[4], "day-of-week")?,
        })
    }

    /// Return `true` if this schedule fires at the given timestamp.
    ///
    /// Matching is minute-granular: all specified fields must match the
    /// timestamp's UTC minute.
    pub fn matches(&self, ts: DateTime<Utc>) -> bool {
        if let Some(m) = self.minute {
            if ts.minute() != m {
                return false;
            }
        }
        if let Some(h) = self.hour {
            if ts.hour() != h {
                return false;
            }
        }
        if let Some(d) = self.day_of_month {
            if ts.day() != d {
                return false;
            }
        }
        if let Some(mo) = self.month {
            if ts.month() != mo {
                return false;
            }
        }
        if let Some(dow) = self.day_of_week {
            // chrono Weekday: Mon=0…Sun=6 internally; num_days_from_sunday gives Sun=0.
            let cd = ts.weekday().num_days_from_sunday();
            if cd != dow {
                return false;
            }
        }
        true
    }
}

// ── HTML report ───────────────────────────────────────────────────────────────

/// Build a minimal standalone HTML cost report.
pub fn build_html_report(ledger: &CostLedger) -> String {
    let total = ledger.total_usd();
    let count = ledger.len();
    let by_model = ledger.by_model();
    let ts = Utc::now().to_rfc3339();

    let mut model_rows = String::new();
    let mut model_stats: Vec<_> = by_model.values().collect();
    model_stats.sort_by(|a, b| {
        b.total_cost_usd
            .partial_cmp(&a.total_cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for s in model_stats {
        model_rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>${:.6}</td><td>${:.6}</td></tr>\n",
            s.model, s.request_count, s.avg_cost_per_request, s.total_cost_usd,
        ));
    }

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LLM Cost Report — {ts}</title>
  <style>
    body {{ font-family: monospace; background: #111; color: #eee; padding: 2em; }}
    h1 {{ color: #0af; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #222; color: #fa0; padding: 6px 12px; text-align: left; }}
    td {{ padding: 4px 12px; border-bottom: 1px solid #333; }}
    .total {{ color: #0f0; font-size: 1.2em; }}
  </style>
</head>
<body>
  <h1>LLM Cost Report</h1>
  <p>Generated: {ts}</p>
  <p class="total">Total spend: <strong>${total:.6}</strong> across <strong>{count}</strong> requests</p>
  <h2>Cost by Model</h2>
  <table>
    <thead><tr><th>Model</th><th>Requests</th><th>Avg Cost</th><th>Total Cost</th></tr></thead>
    <tbody>
{model_rows}    </tbody>
  </table>
</body>
</html>
"#
    )
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// A cron-based export scheduler.
///
/// Call [`Scheduler::tick`] once per minute (or on each UI refresh) to let
/// the scheduler decide whether an export is due.
pub struct Scheduler {
    /// The schedule configuration.
    pub schedule: ExportSchedule,
    /// Parsed cron schedule derived from [`ExportSchedule::cron`].
    cron: CronSchedule,
    /// Timestamp of the last successful export.
    last_export: Option<DateTime<Utc>>,
}

impl Scheduler {
    /// Create a new scheduler from an [`ExportSchedule`].
    ///
    /// Returns [`DashboardError::Ledger`] if the cron expression is invalid.
    pub fn new(schedule: ExportSchedule) -> Result<Self, DashboardError> {
        let cron = CronSchedule::parse(&schedule.cron)?;
        Ok(Self {
            schedule,
            cron,
            last_export: None,
        })
    }

    /// Check whether an export is due at `now` and, if so, run it.
    ///
    /// An export fires when:
    /// - The current UTC time matches the cron schedule, AND
    /// - At least 60 seconds have elapsed since the last export (prevents
    ///   double-firing on repeated tick calls within the same minute).
    ///
    /// Returns the path of the exported file on success, or `None` if no
    /// export was due.
    pub fn tick(&mut self, ledger: &CostLedger, now: DateTime<Utc>) -> Option<Result<String, DashboardError>> {
        if !self.cron.matches(now) {
            return None;
        }
        // Deduplicate within the same minute.
        if let Some(last) = self.last_export {
            let elapsed = (now - last).num_seconds();
            if elapsed < 60 {
                return None;
            }
        }
        self.last_export = Some(now);
        Some(self.run_export(ledger, now))
    }

    fn run_export(&self, ledger: &CostLedger, now: DateTime<Utc>) -> Result<String, DashboardError> {
        let filename = format!(
            "{}/llm-costs-{}.{}",
            self.schedule.output_dir,
            now.format("%Y%m%d-%H%M%S"),
            self.schedule.format.extension()
        );

        let content = match self.schedule.format {
            ScheduledExportFormat::Csv => ledger.to_csv()?,
            ScheduledExportFormat::Json => ledger.to_json()?,
            ScheduledExportFormat::Html => build_html_report(ledger),
        };

        std::fs::write(&filename, content.as_bytes())?;
        Ok(filename)
    }

    /// Return `true` if the cron schedule matches `now`.
    pub fn is_due(&self, now: DateTime<Utc>) -> bool {
        self.cron.matches(now)
    }

    /// Timestamp of the last successful export, if any.
    pub fn last_export_at(&self) -> Option<DateTime<Utc>> {
        self.last_export
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn utc(y: i32, mo: u32, d: u32, h: u32, m: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, mo, d, h, m, 0).unwrap()
    }

    #[test]
    fn test_parse_wildcard_fields() {
        let s = CronSchedule::parse("* * * * *").unwrap();
        assert!(s.minute.is_none());
        assert!(s.hour.is_none());
        assert!(s.day_of_week.is_none());
    }

    #[test]
    fn test_parse_specific_fields() {
        let s = CronSchedule::parse("0 9 * * MON").unwrap();
        assert_eq!(s.minute, Some(0));
        assert_eq!(s.hour, Some(9));
        assert_eq!(s.day_of_week, Some(1)); // MON→1
    }

    #[test]
    fn test_parse_bad_field_count() {
        assert!(CronSchedule::parse("0 9 *").is_err());
    }

    #[test]
    fn test_parse_non_numeric_field() {
        assert!(CronSchedule::parse("0 9 * * BADDAY").is_err());
    }

    #[test]
    fn test_matches_all_wildcards() {
        let s = CronSchedule::parse("* * * * *").unwrap();
        assert!(s.matches(utc(2026, 3, 22, 12, 0)));
    }

    #[test]
    fn test_matches_specific_time() {
        let s = CronSchedule::parse("0 9 * * *").unwrap();
        assert!(s.matches(utc(2026, 3, 22, 9, 0)));
        assert!(!s.matches(utc(2026, 3, 22, 9, 1)));
        assert!(!s.matches(utc(2026, 3, 22, 10, 0)));
    }

    #[test]
    fn test_matches_day_of_week_monday() {
        // 2026-03-23 is a Monday
        let s = CronSchedule::parse("0 9 * * MON").unwrap();
        assert!(s.matches(utc(2026, 3, 23, 9, 0)));
        // 2026-03-22 is a Sunday
        assert!(!s.matches(utc(2026, 3, 22, 9, 0)));
    }

    #[test]
    fn test_scheduler_fires_when_due() {
        let schedule = ExportSchedule {
            cron: "* * * * *".to_string(),
            format: ScheduledExportFormat::Json,
            output_dir: std::env::temp_dir().to_string_lossy().into_owned(),
            webhook_url: None,
        };
        let mut sched = Scheduler::new(schedule).unwrap();
        let ledger = CostLedger::new();
        let now = Utc::now();
        let result = sched.tick(&ledger, now);
        assert!(result.is_some(), "should fire for wildcard cron");
        let path = result.unwrap().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_scheduler_no_double_fire() {
        let schedule = ExportSchedule {
            cron: "* * * * *".to_string(),
            format: ScheduledExportFormat::Json,
            output_dir: std::env::temp_dir().to_string_lossy().into_owned(),
            webhook_url: None,
        };
        let mut sched = Scheduler::new(schedule).unwrap();
        let ledger = CostLedger::new();
        let now = Utc::now();
        let r1 = sched.tick(&ledger, now);
        let r2 = sched.tick(&ledger, now); // same second
        assert!(r1.is_some());
        assert!(r2.is_none(), "should not double-fire in same minute");
        if let Some(Ok(p)) = r1 {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn test_html_report_contains_total() {
        let mut ledger = CostLedger::new();
        use crate::cost::CostRecord;
        ledger
            .add(CostRecord::new("gpt-4o-mini", "openai", 1_000_000, 0, 10))
            .unwrap();
        let html = build_html_report(&ledger);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("gpt-4o-mini"));
        assert!(html.contains("Total spend"));
    }

    #[test]
    fn test_scheduled_export_format_extension() {
        assert_eq!(ScheduledExportFormat::Csv.extension(), "csv");
        assert_eq!(ScheduledExportFormat::Json.extension(), "json");
        assert_eq!(ScheduledExportFormat::Html.extension(), "html");
    }
}
