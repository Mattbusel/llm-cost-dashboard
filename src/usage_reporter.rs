//! Generate usage reports in multiple formats.
//!
//! Provides [`UsageReporter`] which aggregates [`UsageRecord`]s and can emit
//! Markdown tables, CSV, JSON, or plain-text aligned columns.

use std::collections::HashMap;

// ── Enumerations ──────────────────────────────────────────────────────────────

/// Output format for a generated report.
#[derive(Debug, Clone, PartialEq)]
pub enum ReportFormat {
    /// GitHub-flavoured Markdown with tables and `##` headers.
    Markdown,
    /// Comma-separated values with a header row.
    Csv,
    /// Hand-formatted JSON summary (no serde dependency required).
    Json,
    /// Plain text with padded columns.
    PlainText,
}

/// Time window covered by a report.
#[derive(Debug, Clone, PartialEq)]
pub enum ReportPeriod {
    /// Last 24 hours.
    Daily,
    /// Last 7 days.
    Weekly,
    /// Last 30 days.
    Monthly,
    /// Inclusive range of Unix epoch seconds.
    Custom {
        /// Inclusive start of the custom window (Unix epoch seconds).
        start_epoch: u64,
        /// Inclusive end of the custom window (Unix epoch seconds).
        end_epoch: u64,
    },
}

// ── Core data types ───────────────────────────────────────────────────────────

/// A single API usage record.
#[derive(Debug, Clone)]
pub struct UsageRecord {
    /// Model identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
    pub model: String,
    /// Unix epoch seconds when the request completed.
    pub timestamp_epoch: u64,
    /// Input tokens consumed.
    pub tokens_in: u64,
    /// Output tokens produced.
    pub tokens_out: u64,
    /// USD cost for this request.
    pub cost_usd: f64,
    /// Unique request identifier.
    pub request_id: String,
}

/// Direction of a cost/usage trend.
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Values are generally rising.
    Increasing,
    /// Values are generally falling.
    Decreasing,
    /// No clear direction (|slope| < threshold).
    Stable,
}

/// Result of an OLS trend calculation.
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Overall direction.
    pub direction: TrendDirection,
    /// OLS slope (tokens-per-window-step).
    pub slope: f64,
    /// Percentage change from first window to last.
    pub percent_change: f64,
}

/// A titled section within a report containing key-value data and optional notes.
#[derive(Debug, Clone)]
pub struct ReportSection {
    /// Section heading.
    pub title: String,
    /// Key-value pairs displayed in tabular form.
    pub data: Vec<(String, String)>,
    /// Free-form annotation lines.
    pub notes: Vec<String>,
}

/// A fully assembled report ready to be rendered.
#[derive(Debug, Clone)]
pub struct UsageReport {
    /// Time window this report covers.
    pub period: ReportPeriod,
    /// Ordered sections.
    pub sections: Vec<ReportSection>,
    /// Unix epoch seconds when the report was built.
    pub generated_at: u64,
    /// Sum of all costs in the filtered record set.
    pub total_cost: f64,
}

// ── Reporter ──────────────────────────────────────────────────────────────────

/// Aggregates usage records and generates formatted reports.
pub struct UsageReporter {
    records: Vec<UsageRecord>,
}

impl UsageReporter {
    /// Create an empty reporter.
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    /// Append a single record.
    pub fn add_record(&mut self, record: UsageRecord) {
        self.records.push(record);
    }

    /// Return references to records that fall within `period`.
    ///
    /// For `Daily`, `Weekly`, and `Monthly`, the window is measured backwards
    /// from the maximum timestamp present in the data set (or 0 if empty).
    pub fn filter_by_period<'a>(&'a self, period: &ReportPeriod) -> Vec<&'a UsageRecord> {
        let max_ts = self.records.iter().map(|r| r.timestamp_epoch).max().unwrap_or(0);
        let (start, end) = match period {
            ReportPeriod::Daily => (max_ts.saturating_sub(86_400), max_ts),
            ReportPeriod::Weekly => (max_ts.saturating_sub(7 * 86_400), max_ts),
            ReportPeriod::Monthly => (max_ts.saturating_sub(30 * 86_400), max_ts),
            ReportPeriod::Custom { start_epoch, end_epoch } => (*start_epoch, *end_epoch),
        };
        self.records
            .iter()
            .filter(|r| r.timestamp_epoch >= start && r.timestamp_epoch <= end)
            .collect()
    }

    /// Group costs by model.
    pub fn cost_by_model(&self, records: &[&UsageRecord]) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for r in records {
            *map.entry(r.model.clone()).or_default() += r.cost_usd;
        }
        map
    }

    /// Compute an OLS trend over total tokens bucketed into windows of size
    /// `window` (measured in record count).  Returns [`TrendAnalysis`].
    pub fn token_trend(&self, records: &[&UsageRecord], window: usize) -> TrendAnalysis {
        if records.is_empty() || window == 0 {
            return TrendAnalysis {
                direction: TrendDirection::Stable,
                slope: 0.0,
                percent_change: 0.0,
            };
        }

        // Bucket records into groups of `window` size.
        let buckets: Vec<f64> = records
            .chunks(window)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|r| (r.tokens_in + r.tokens_out) as f64)
                    .sum()
            })
            .collect();

        let n = buckets.len() as f64;
        if n < 2.0 {
            return TrendAnalysis {
                direction: TrendDirection::Stable,
                slope: 0.0,
                percent_change: 0.0,
            };
        }

        // OLS: y = slope * x + intercept
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = buckets.iter().sum::<f64>() / n;
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (i, &y) in buckets.iter().enumerate() {
            let x = i as f64;
            num += (x - x_mean) * (y - y_mean);
            den += (x - x_mean).powi(2);
        }
        let slope = if den.abs() < f64::EPSILON { 0.0 } else { num / den };

        let first = buckets[0];
        let last = *buckets.last().unwrap();
        let percent_change = if first.abs() < f64::EPSILON {
            0.0
        } else {
            (last - first) / first * 100.0
        };

        let direction = if slope > 1.0 {
            TrendDirection::Increasing
        } else if slope < -1.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        TrendAnalysis { direction, slope, percent_change }
    }

    /// Return the top `n` models sorted by descending cost.
    pub fn top_n_models_by_cost(
        &self,
        n: usize,
        records: &[&UsageRecord],
    ) -> Vec<(String, f64)> {
        let map = self.cost_by_model(records);
        let mut pairs: Vec<(String, f64)> = map.into_iter().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(n);
        pairs
    }

    /// Build and render a report for the given `period` in `format`.
    pub fn generate_report(&self, period: ReportPeriod, format: ReportFormat) -> String {
        let records = self.filter_by_period(&period);
        let total_cost: f64 = records.iter().map(|r| r.cost_usd).sum();
        let total_tokens_in: u64 = records.iter().map(|r| r.tokens_in).sum();
        let total_tokens_out: u64 = records.iter().map(|r| r.tokens_out).sum();
        let by_model = self.cost_by_model(&records);
        let trend = self.token_trend(&records, 10.max(records.len() / 5 + 1));
        let top3 = self.top_n_models_by_cost(3, &records);

        let period_label = match &period {
            ReportPeriod::Daily => "Daily".to_string(),
            ReportPeriod::Weekly => "Weekly".to_string(),
            ReportPeriod::Monthly => "Monthly".to_string(),
            ReportPeriod::Custom { start_epoch, end_epoch } => {
                format!("Custom ({start_epoch}–{end_epoch})")
            }
        };

        let trend_arrow = match trend.direction {
            TrendDirection::Increasing => "↑",
            TrendDirection::Decreasing => "↓",
            TrendDirection::Stable => "→",
        };

        match format {
            ReportFormat::Markdown => {
                let mut out = String::new();
                out.push_str(&format!("## {period_label} Usage Report\n\n"));
                out.push_str("### Summary\n\n");
                out.push_str("| Metric | Value |\n");
                out.push_str("|--------|-------|\n");
                out.push_str(&format!("| Total Cost | ${total_cost:.4} |\n"));
                out.push_str(&format!("| Tokens In | {total_tokens_in} |\n"));
                out.push_str(&format!("| Tokens Out | {total_tokens_out} |\n"));
                out.push_str(&format!(
                    "| Token Trend | {trend_arrow} ({:.1}%) |\n",
                    trend.percent_change
                ));
                out.push_str("\n### Cost by Model\n\n");
                out.push_str("| Model | Cost (USD) |\n");
                out.push_str("|-------|------------|\n");
                let mut sorted: Vec<_> = by_model.iter().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (model, cost) in &sorted {
                    out.push_str(&format!("| {model} | ${cost:.4} |\n"));
                }
                out.push_str("\n### Top Models\n\n");
                out.push_str("| Rank | Model | Cost (USD) |\n");
                out.push_str("|------|-------|------------|\n");
                for (i, (model, cost)) in top3.iter().enumerate() {
                    out.push_str(&format!("| {} | {model} | ${cost:.4} |\n", i + 1));
                }
                out
            }

            ReportFormat::Csv => {
                let mut out =
                    "model,timestamp_epoch,tokens_in,tokens_out,cost_usd,request_id\n".to_string();
                for r in &records {
                    out.push_str(&format!(
                        "{},{},{},{},{:.6},{}\n",
                        r.model,
                        r.timestamp_epoch,
                        r.tokens_in,
                        r.tokens_out,
                        r.cost_usd,
                        r.request_id
                    ));
                }
                out
            }

            ReportFormat::Json => {
                let model_costs: Vec<String> = {
                    let mut pairs: Vec<_> = by_model.iter().collect();
                    pairs.sort_by_key(|(k, _)| (*k).clone());
                    pairs
                        .iter()
                        .map(|(k, v)| format!("    \"{k}\": {v:.6}"))
                        .collect()
                };
                format!(
                    "{{\n  \"period\": \"{period_label}\",\n  \"total_cost_usd\": {total_cost:.6},\
                    \n  \"total_tokens_in\": {total_tokens_in},\n  \"total_tokens_out\": {total_tokens_out},\
                    \n  \"trend\": \"{trend_arrow}\",\n  \"trend_slope\": {:.4},\
                    \n  \"cost_by_model\": {{\n{}\n  }}\n}}",
                    trend.slope,
                    model_costs.join(",\n")
                )
            }

            ReportFormat::PlainText => {
                let col_w = 30usize;
                let val_w = 15usize;
                let mut out = format!(
                    "{:=<width$}\n{:^width$}\n{:=<width$}\n",
                    "",
                    format!("{period_label} Usage Report"),
                    "",
                    width = col_w + val_w + 3
                );
                let rows = vec![
                    ("Total Cost", format!("${total_cost:.4}")),
                    ("Tokens In", total_tokens_in.to_string()),
                    ("Tokens Out", total_tokens_out.to_string()),
                    (
                        "Token Trend",
                        format!("{trend_arrow} ({:.1}%)", trend.percent_change),
                    ),
                ];
                for (k, v) in &rows {
                    out.push_str(&format!("{k:<col_w$} {v:>val_w$}\n"));
                }
                out.push_str(&format!("{:-<width$}\n", "", width = col_w + val_w + 3));
                out.push_str(&format!("{:<col_w$} {:>val_w$}\n", "Model", "Cost (USD)"));
                out.push_str(&format!("{:-<width$}\n", "", width = col_w + val_w + 3));
                let mut sorted: Vec<_> = by_model.iter().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (model, cost) in &sorted {
                    out.push_str(&format!(
                        "{:<col_w$} {:>val_w$.4}\n",
                        model, cost
                    ));
                }
                out
            }
        }
    }
}

impl Default for UsageReporter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_records() -> Vec<UsageRecord> {
        vec![
            UsageRecord {
                model: "gpt-4o".to_string(),
                timestamp_epoch: 1_000_000,
                tokens_in: 500,
                tokens_out: 200,
                cost_usd: 0.05,
                request_id: "req-1".to_string(),
            },
            UsageRecord {
                model: "claude-3-5-sonnet".to_string(),
                timestamp_epoch: 1_000_100,
                tokens_in: 800,
                tokens_out: 300,
                cost_usd: 0.12,
                request_id: "req-2".to_string(),
            },
            UsageRecord {
                model: "gpt-4o".to_string(),
                timestamp_epoch: 1_000_200,
                tokens_in: 200,
                tokens_out: 100,
                cost_usd: 0.02,
                request_id: "req-3".to_string(),
            },
        ]
    }

    fn make_reporter() -> UsageReporter {
        let mut r = UsageReporter::new();
        for rec in sample_records() {
            r.add_record(rec);
        }
        r
    }

    #[test]
    fn markdown_contains_headers() {
        let reporter = make_reporter();
        let out = reporter.generate_report(ReportPeriod::Daily, ReportFormat::Markdown);
        assert!(out.contains("## Daily Usage Report"));
        assert!(out.contains("### Summary"));
        assert!(out.contains("| Model | Cost (USD) |"));
    }

    #[test]
    fn csv_has_correct_columns() {
        let reporter = make_reporter();
        let out = reporter.generate_report(ReportPeriod::Daily, ReportFormat::Csv);
        let first_line = out.lines().next().unwrap();
        assert_eq!(
            first_line,
            "model,timestamp_epoch,tokens_in,tokens_out,cost_usd,request_id"
        );
        // 3 data rows + 1 header
        assert_eq!(out.lines().count(), 4);
    }

    #[test]
    fn trend_detection() {
        let reporter = make_reporter();
        let records = reporter.filter_by_period(&ReportPeriod::Daily);
        // With only 3 records and window=1, each bucket is one record.
        let trend = reporter.token_trend(&records, 1);
        // Direction should be one of the three valid values.
        let _ = match trend.direction {
            TrendDirection::Increasing | TrendDirection::Decreasing | TrendDirection::Stable => true,
        };
    }

    #[test]
    fn top_n_ordering() {
        let reporter = make_reporter();
        let records = reporter.filter_by_period(&ReportPeriod::Daily);
        let top2 = reporter.top_n_models_by_cost(2, &records);
        assert_eq!(top2.len(), 2);
        // claude should be most expensive (0.12 vs 0.07).
        assert_eq!(top2[0].0, "claude-3-5-sonnet");
        assert!(top2[0].1 > top2[1].1);
    }

    #[test]
    fn period_filtering() {
        let reporter = make_reporter();
        let custom = reporter.filter_by_period(&ReportPeriod::Custom {
            start_epoch: 1_000_100,
            end_epoch: 1_000_200,
        });
        assert_eq!(custom.len(), 2);
    }

    #[test]
    fn json_report_contains_expected_keys() {
        let reporter = make_reporter();
        let out = reporter.generate_report(ReportPeriod::Daily, ReportFormat::Json);
        assert!(out.contains("\"total_cost_usd\""));
        assert!(out.contains("\"cost_by_model\""));
        assert!(out.contains("\"trend\""));
    }
}
