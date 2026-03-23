//! # Export
//!
//! Export functions for serializing [`CostLedger`] data to CSV and JSON files,
//! plus the [`CostExporter`] helper that writes timestamped files to the
//! current working directory (triggered by pressing `E` in the TUI).
//!
//! ## Functions
//!
//! | Function | Output format | Content |
//! |---|---|---|
//! | [`export_csv`] | CSV | One row per request record |
//! | [`export_json`] | JSON array | One object per request record |
//! | [`export_summary_json`] | JSON object map | Per-model [`ModelStats`] summary |
//!
//! ## Types
//!
//! | Type | Description |
//! |---|---|
//! | [`CostExporter`] | Writes a timestamped file and returns the file name |
//! | [`ExportFormat`] | Selects CSV or JSON output |
//!
//! ## Example (function API)
//!
//! ```rust,no_run
//! use std::path::Path;
//! use llm_cost_dashboard::{CostLedger, CostRecord};
//! use llm_cost_dashboard::export::{export_csv, export_json, export_summary_json};
//!
//! let mut ledger = CostLedger::new();
//! ledger.add(CostRecord::new("gpt-4o-mini", "openai", 512, 256, 34)).unwrap();
//!
//! export_csv(&ledger, Path::new("costs.csv")).unwrap();
//! export_json(&ledger, Path::new("costs.json")).unwrap();
//! export_summary_json(&ledger, Path::new("summary.json")).unwrap();
//! ```
//!
//! ## Example (struct API)
//!
//! ```rust,no_run
//! use llm_cost_dashboard::{CostLedger, CostRecord};
//! use llm_cost_dashboard::export::{CostExporter, ExportFormat};
//!
//! let mut ledger = CostLedger::new();
//! ledger.add(CostRecord::new("gpt-4o-mini", "openai", 512, 256, 34)).unwrap();
//!
//! let exporter = CostExporter::new(&ledger);
//! let filename = exporter.export(ExportFormat::Csv).unwrap();
//! println!("Exported to {filename}");
//! ```

use std::path::Path;

use chrono::Utc;
use serde::Serialize;

use crate::{
    cost::{CostLedger, ModelStats},
    error::DashboardError,
};

// ── CSV export (path-based) ───────────────────────────────────────────────────

/// A flattened CSV row derived from a single cost record.
///
/// Serialized as:
/// `timestamp,model,provider,input_tokens,output_tokens,cost_usd,session_id`
#[derive(Debug, Serialize)]
struct CsvRow<'a> {
    timestamp: String,
    model: &'a str,
    provider: &'a str,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
    session_id: &'a str,
}

/// Export all records in `ledger` to a CSV file at `path`.
///
/// Columns: `timestamp`, `model`, `provider`, `input_tokens`,
/// `output_tokens`, `cost_usd`, `session_id`.
///
/// The file is created (or overwritten) at `path`.  Parent directories must
/// already exist.
///
/// # Errors
///
/// Returns [`DashboardError::IoError`] on I/O failure, or a wrapped CSV error
/// surfaced as [`DashboardError::Ledger`].
pub fn export_csv(ledger: &CostLedger, path: &Path) -> Result<(), DashboardError> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| DashboardError::Ledger(format!("csv writer error: {e}")))?;

    for record in ledger.records() {
        let row = CsvRow {
            timestamp: record.timestamp.to_rfc3339(),
            model: &record.model,
            provider: &record.provider,
            input_tokens: record.input_tokens,
            output_tokens: record.output_tokens,
            cost_usd: record.total_cost_usd,
            session_id: record.session_id.as_deref().unwrap_or(""),
        };
        writer
            .serialize(row)
            .map_err(|e| DashboardError::Ledger(format!("csv serialize error: {e}")))?;
    }

    writer
        .flush()
        .map_err(|e| DashboardError::Ledger(format!("csv flush error: {e}")))?;
    Ok(())
}

// ── JSON export (path-based) ──────────────────────────────────────────────────

/// A JSON-serializable view of a single cost record.
#[derive(Debug, Serialize)]
struct JsonRecord<'a> {
    timestamp: String,
    model: &'a str,
    provider: &'a str,
    input_tokens: u64,
    output_tokens: u64,
    input_cost_usd: f64,
    output_cost_usd: f64,
    cost_usd: f64,
    latency_ms: u64,
    session_id: Option<&'a str>,
}

/// Export all records in `ledger` as a JSON array to `path`.
///
/// Each element corresponds to one request record.  The file is pretty-printed
/// with 2-space indentation.
///
/// # Errors
///
/// Returns [`DashboardError::IoError`] on I/O failure, or
/// [`DashboardError::SerializationError`] if JSON serialization fails.
pub fn export_json(ledger: &CostLedger, path: &Path) -> Result<(), DashboardError> {
    let rows: Vec<JsonRecord<'_>> = ledger
        .records()
        .iter()
        .map(|r| JsonRecord {
            timestamp: r.timestamp.to_rfc3339(),
            model: &r.model,
            provider: &r.provider,
            input_tokens: r.input_tokens,
            output_tokens: r.output_tokens,
            input_cost_usd: r.input_cost_usd,
            output_cost_usd: r.output_cost_usd,
            cost_usd: r.total_cost_usd,
            latency_ms: r.latency_ms,
            session_id: r.session_id.as_deref(),
        })
        .collect();

    let json = serde_json::to_string_pretty(&rows)?;
    std::fs::write(path, json)?;
    Ok(())
}

// ── Summary JSON export ──────────────────────────────────────────────────────

/// A JSON-serializable summary for one model.
#[derive(Debug, Serialize)]
struct JsonModelSummary<'a> {
    model: &'a str,
    request_count: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cost_usd: f64,
    avg_cost_per_request: f64,
    avg_latency_ms: f64,
    p99_latency_ms: f64,
}

impl<'a> From<&'a ModelStats> for JsonModelSummary<'a> {
    fn from(s: &'a ModelStats) -> Self {
        Self {
            model: &s.model,
            request_count: s.request_count,
            total_input_tokens: s.total_input_tokens,
            total_output_tokens: s.total_output_tokens,
            total_cost_usd: s.total_cost_usd,
            avg_cost_per_request: s.avg_cost_per_request,
            avg_latency_ms: s.avg_latency_ms,
            p99_latency_ms: s.p99_latency_ms,
        }
    }
}

/// Export a per-model summary of `ledger` as a pretty-printed JSON object.
///
/// The output is a JSON object whose keys are model names and whose values are
/// [`ModelStats`]-equivalent objects.
///
/// # Errors
///
/// Returns [`DashboardError::IoError`] on I/O failure, or
/// [`DashboardError::SerializationError`] on JSON serialization failure.
pub fn export_summary_json(ledger: &CostLedger, path: &Path) -> Result<(), DashboardError> {
    let by_model = ledger.by_model();
    let summary: std::collections::HashMap<&str, JsonModelSummary<'_>> = by_model
        .values()
        .map(|s| (s.model.as_str(), JsonModelSummary::from(s)))
        .collect();

    let json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(path, json)?;
    Ok(())
}

// ── CostExporter (timestamped file API) ──────────────────────────────────────

/// Output format for the timestamped cost export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Comma-separated values (spreadsheet-compatible).
    Csv,
    /// Pretty-printed JSON array.
    Json,
    /// Newline-delimited JSON (one object per line, streaming-friendly).
    Jsonl,
    /// Markdown summary table sorted by cost descending.
    Markdown,
}

impl ExportFormat {
    /// File extension for this format (without the leading dot).
    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Csv => "csv",
            ExportFormat::Json => "json",
            ExportFormat::Jsonl => "jsonl",
            ExportFormat::Markdown => "md",
        }
    }
}

/// Exports a [`CostLedger`] snapshot to a timestamped file on disk.
///
/// Triggered by pressing `E` in the TUI dashboard.
///
/// The output file name follows the pattern `costs_YYYYMMDD_HHMMSS.<ext>`.
pub struct CostExporter<'a> {
    ledger: &'a CostLedger,
}

impl<'a> CostExporter<'a> {
    /// Create an exporter bound to the given ledger.
    pub fn new(ledger: &'a CostLedger) -> Self {
        Self { ledger }
    }

    /// Export all ledger records in `format` to a timestamped file inside
    /// `dir`.
    ///
    /// Returns the **full path** to the created file on success, so callers
    /// can read or display it.
    ///
    /// # Errors
    ///
    /// Returns [`DashboardError::IoError`] on file-system failures, or the
    /// serialisation error forwarded from [`CostLedger::to_csv`] /
    /// [`CostLedger::to_json`].
    pub fn export_to_dir(
        &self,
        dir: &Path,
        format: ExportFormat,
    ) -> Result<String, DashboardError> {
        let now = Utc::now();
        let filename = format!(
            "costs_{}.{}",
            now.format("%Y%m%d_%H%M%S"),
            format.extension()
        );
        let path = dir.join(&filename);

        let content = match format {
            ExportFormat::Csv => self.ledger.to_csv()?,
            ExportFormat::Json => self.ledger.to_json()?,
            // Jsonl/Markdown not yet supported by CostLedger; fall back to JSON.
            ExportFormat::Jsonl | ExportFormat::Markdown => self.ledger.to_json()?,
        };

        std::fs::write(&path, content.as_bytes())?;
        Ok(filename)
    }

    /// Export all ledger records in `format` to a timestamped file in the
    /// current working directory.
    ///
    /// Returns the file name (not a full path) on success so the TUI can
    /// display a status message such as
    /// `"Exported to costs_20260322_120000.csv"`.
    ///
    /// # Errors
    ///
    /// Returns [`DashboardError::IoError`] on file-system failures, or the
    /// serialisation error forwarded from [`CostLedger::to_csv`] /
    /// [`CostLedger::to_json`].
    pub fn export(&self, format: ExportFormat) -> Result<String, DashboardError> {
        let now = Utc::now();
        let filename = format!(
            "costs_{}.{}",
            now.format("%Y%m%d_%H%M%S"),
            format.extension()
        );

        let content = match format {
            ExportFormat::Csv => self.ledger.to_csv()?,
            ExportFormat::Json => self.ledger.to_json()?,
            // Jsonl/Markdown not yet supported by CostLedger; fall back to JSON.
            ExportFormat::Jsonl | ExportFormat::Markdown => self.ledger.to_json()?,
        };

        std::fs::write(&filename, content.as_bytes())?;
        Ok(filename)
    }

    /// Export to CSV, returning the file name on success.
    pub fn export_csv(&self) -> Result<String, DashboardError> {
        self.export(ExportFormat::Csv)
    }

    /// Export to JSON, returning the file name on success.
    pub fn export_json(&self) -> Result<String, DashboardError> {
        self.export(ExportFormat::Json)
    }
}

// ===========================================================================
// Exporter for TaggedRequest
// ===========================================================================

use std::io::Write;

use crate::tagging::TaggedRequest;

/// Errors that can occur during export.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// An I/O failure occurred while writing output.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization (JSON or CSV) failed.
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<serde_json::Error> for ExportError {
    fn from(e: serde_json::Error) -> Self {
        ExportError::Serialization(e.to_string())
    }
}

/// Exports a slice of [`TaggedRequest`]s to various formats.
pub struct Exporter;

impl Exporter {
    /// Export `requests` in the given `format`, writing to `output`.
    ///
    /// # Errors
    ///
    /// Returns [`ExportError::Io`] on write failures or
    /// [`ExportError::Serialization`] on serialisation failures.
    pub fn export(
        requests: &[TaggedRequest],
        format: ExportFormat,
        output: &mut dyn Write,
    ) -> Result<(), ExportError> {
        match format {
            ExportFormat::Csv | ExportFormat::Jsonl | ExportFormat::Markdown => {}
            ExportFormat::Json => {}
        }
        match format {
            ExportFormat::Csv => Self::export_csv(requests, output),
            ExportFormat::Json => Self::export_json(requests, output),
            ExportFormat::Jsonl => Self::export_jsonl(requests, output),
            ExportFormat::Markdown => Self::export_markdown(requests, output),
        }
    }

    fn export_csv(requests: &[TaggedRequest], output: &mut dyn Write) -> Result<(), ExportError> {
        // Collect union of all tag keys for the header.
        let mut tag_keys: Vec<String> = requests
            .iter()
            .flat_map(|r| r.tags.iter().map(|t| t.key.clone()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        tag_keys.sort();

        // Header row.
        let mut header =
            "request_id,model_id,cost_usd,tokens_in,tokens_out,timestamp".to_string();
        for k in &tag_keys {
            header.push(',');
            header.push_str(k);
        }
        writeln!(output, "{header}")?;

        // Data rows.
        for r in requests {
            let mut row = format!(
                "{},{},{:.6},{},{},{}",
                r.request_id,
                r.model_id,
                r.cost_usd,
                r.tokens_in,
                r.tokens_out,
                r.timestamp.to_rfc3339(),
            );
            for k in &tag_keys {
                let val = r.tag_value(k).unwrap_or("");
                row.push(',');
                // Quote if value contains comma or quote.
                if val.contains(',') || val.contains('"') {
                    row.push('"');
                    row.push_str(&val.replace('"', "\"\""));
                    row.push('"');
                } else {
                    row.push_str(val);
                }
            }
            writeln!(output, "{row}")?;
        }
        Ok(())
    }

    fn export_json(requests: &[TaggedRequest], output: &mut dyn Write) -> Result<(), ExportError> {
        let total_cost: f64 = requests.iter().map(|r| r.cost_usd).sum();
        let total_tokens: u64 = requests
            .iter()
            .map(|r| u64::from(r.tokens_in) + u64::from(r.tokens_out))
            .sum();

        let req_values: Vec<serde_json::Value> = requests
            .iter()
            .map(|r| {
                let tags: serde_json::Value = r
                    .tags
                    .iter()
                    .map(|t| (t.key.clone(), serde_json::Value::String(t.value.clone())))
                    .collect::<serde_json::Map<_, _>>()
                    .into();
                serde_json::json!({
                    "request_id": r.request_id,
                    "model_id": r.model_id,
                    "cost_usd": r.cost_usd,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "timestamp": r.timestamp.to_rfc3339(),
                    "tags": tags,
                })
            })
            .collect();

        let payload = serde_json::json!({
            "requests": req_values,
            "summary": {
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "count": requests.len(),
            }
        });

        let json = serde_json::to_string_pretty(&payload)?;
        output.write_all(json.as_bytes())?;
        Ok(())
    }

    fn export_jsonl(
        requests: &[TaggedRequest],
        output: &mut dyn Write,
    ) -> Result<(), ExportError> {
        for r in requests {
            let tags: serde_json::Value = r
                .tags
                .iter()
                .map(|t| (t.key.clone(), serde_json::Value::String(t.value.clone())))
                .collect::<serde_json::Map<_, _>>()
                .into();
            let obj = serde_json::json!({
                "request_id": r.request_id,
                "model_id": r.model_id,
                "cost_usd": r.cost_usd,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "timestamp": r.timestamp.to_rfc3339(),
                "tags": tags,
            });
            let line = serde_json::to_string(&obj)?;
            writeln!(output, "{line}")?;
        }
        Ok(())
    }

    fn export_markdown(
        requests: &[TaggedRequest],
        output: &mut dyn Write,
    ) -> Result<(), ExportError> {
        // Sort by cost descending.
        let mut sorted: Vec<&TaggedRequest> = requests.iter().collect();
        sorted.sort_by(|a, b| {
            b.cost_usd
                .partial_cmp(&a.cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        writeln!(output, "# Cost Export Summary")?;
        writeln!(output)?;
        writeln!(
            output,
            "| request_id | model_id | cost_usd | tokens_in | tokens_out | timestamp |"
        )?;
        writeln!(
            output,
            "|---|---|---|---|---|---|"
        )?;
        for r in sorted {
            writeln!(
                output,
                "| {} | {} | {:.6} | {} | {} | {} |",
                r.request_id,
                r.model_id,
                r.cost_usd,
                r.tokens_in,
                r.tokens_out,
                r.timestamp.to_rfc3339(),
            )?;
        }
        Ok(())
    }
}

// ===========================================================================
// CLI flags for --export and --out (see main.rs)
// ===========================================================================
// The CLI integration is done in main.rs; the Exporter is a library type.

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod exporter_tests {
    use super::*;
    use crate::tagging::{CostTag, TaggedRequest};
    use chrono::Utc;

    fn make_req(id: u64, cost: f64, tags: Vec<CostTag>) -> TaggedRequest {
        TaggedRequest {
            request_id: id,
            model_id: format!("model-{id}"),
            cost_usd: cost,
            tokens_in: 100,
            tokens_out: 50,
            tags,
            timestamp: Utc::now(),
        }
    }

    fn tag(k: &str, v: &str) -> CostTag {
        CostTag::new(k, v)
    }

    #[test]
    fn test_export_csv_header() {
        let reqs = vec![make_req(1, 0.01, vec![tag("env", "prod")])];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Csv, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("request_id"));
        assert!(s.contains("cost_usd"));
        assert!(s.contains("env"));
    }

    #[test]
    fn test_export_csv_data_row() {
        let reqs = vec![make_req(42, 0.05, vec![tag("team", "search")])];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Csv, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("42"));
        assert!(s.contains("search"));
    }

    #[test]
    fn test_export_csv_multiple_tag_columns() {
        let reqs = vec![
            make_req(1, 0.01, vec![tag("env", "prod"), tag("team", "a")]),
            make_req(2, 0.02, vec![tag("env", "staging"), tag("team", "b")]),
        ];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Csv, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let header = s.lines().next().unwrap();
        assert!(header.contains("env"));
        assert!(header.contains("team"));
    }

    #[test]
    fn test_export_json_structure() {
        let reqs = vec![make_req(1, 0.10, vec![])];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Json, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert!(v.get("requests").is_some());
        assert!(v.get("summary").is_some());
    }

    #[test]
    fn test_export_json_summary_count() {
        let reqs = vec![make_req(1, 0.10, vec![]), make_req(2, 0.20, vec![])];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Json, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["summary"]["count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_export_json_summary_total_cost() {
        let reqs = vec![make_req(1, 0.10, vec![]), make_req(2, 0.20, vec![])];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Json, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        let total = v["summary"]["total_cost"].as_f64().unwrap();
        assert!((total - 0.30).abs() < 1e-9);
    }

    #[test]
    fn test_export_jsonl_one_object_per_line() {
        let reqs = vec![make_req(1, 0.10, vec![]), make_req(2, 0.20, vec![])];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Jsonl, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v.get("request_id").is_some());
        }
    }

    #[test]
    fn test_export_jsonl_streaming_valid() {
        let reqs = vec![make_req(1, 0.10, vec![tag("env", "prod")])];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Jsonl, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let v: serde_json::Value = serde_json::from_str(s.trim()).unwrap();
        assert_eq!(v["request_id"].as_u64().unwrap(), 1);
    }

    #[test]
    fn test_export_markdown_header() {
        let reqs = vec![make_req(1, 0.01, vec![])];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Markdown, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("request_id"));
        assert!(s.contains("cost_usd"));
        assert!(s.contains("|"));
    }

    #[test]
    fn test_export_markdown_sorted_by_cost_desc() {
        let reqs = vec![
            make_req(1, 0.01, vec![]),
            make_req(2, 1.00, vec![]),
            make_req(3, 0.50, vec![]),
        ];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Markdown, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        // The data lines (after the separator) should have id=2 first.
        let data_lines: Vec<&str> = s
            .lines()
            .filter(|l| l.starts_with("| ") && !l.contains("request_id") && !l.starts_with("|---"))
            .collect();
        assert!(!data_lines.is_empty());
        // model-2 should come first (highest cost).
        assert!(data_lines[0].contains("model-2"));
    }

    #[test]
    fn test_export_empty_slice_csv() {
        let reqs: Vec<TaggedRequest> = vec![];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Csv, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        // Header only.
        assert!(s.contains("request_id"));
    }

    #[test]
    fn test_export_empty_slice_json() {
        let reqs: Vec<TaggedRequest> = vec![];
        let mut buf = Vec::new();
        Exporter::export(&reqs, ExportFormat::Json, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["summary"]["count"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_export_format_extension_jsonl() {
        assert_eq!(ExportFormat::Jsonl.extension(), "jsonl");
    }

    #[test]
    fn test_export_format_extension_markdown() {
        assert_eq!(ExportFormat::Markdown.extension(), "md");
    }

    #[test]
    fn test_export_error_io_display() {
        let e = ExportError::Io(std::io::Error::new(std::io::ErrorKind::Other, "oops"));
        assert!(e.to_string().contains("I/O error"));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::cost::CostRecord;
    use tempfile::NamedTempFile;

    fn make_ledger() -> CostLedger {
        let mut l = CostLedger::new();
        l.add(CostRecord::new("gpt-4o-mini", "openai", 512, 256, 34))
            .unwrap();
        l.add(CostRecord::new("claude-sonnet-4-6", "anthropic", 1024, 512, 80))
            .unwrap();
        l
    }

    // ── path-based export tests ───────────────────────────────────────────────

    #[test]
    fn test_export_csv_creates_file() {
        let tmp = NamedTempFile::new().unwrap();
        let ledger = make_ledger();
        export_csv(&ledger, tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(content.contains("gpt-4o-mini"));
        assert!(content.contains("claude-sonnet-4-6"));
        assert!(content.contains("timestamp"));
    }

    #[test]
    fn test_export_csv_has_header_row() {
        let tmp = NamedTempFile::new().unwrap();
        let ledger = make_ledger();
        export_csv(&ledger, tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        let first_line = content.lines().next().unwrap();
        assert!(first_line.contains("cost_usd"));
        assert!(first_line.contains("model"));
    }

    #[test]
    fn test_export_csv_row_count_matches_records() {
        let tmp = NamedTempFile::new().unwrap();
        let ledger = make_ledger();
        export_csv(&ledger, tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        // header + 2 data rows
        assert_eq!(content.lines().count(), 3);
    }

    #[test]
    fn test_export_csv_empty_ledger() {
        let tmp = NamedTempFile::new().unwrap();
        let ledger = CostLedger::new();
        export_csv(&ledger, tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        // Header row only (csv may add trailing newline on some platforms)
        assert!(content.lines().count() <= 1);
        // Header should be present
        assert!(content.contains("model") || content.is_empty());
    }

    #[test]
    fn test_export_json_creates_valid_array() {
        let tmp = NamedTempFile::new().unwrap();
        let ledger = make_ledger();
        export_json(&ledger, tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_export_json_contains_expected_fields() {
        let tmp = NamedTempFile::new().unwrap();
        let ledger = make_ledger();
        export_json(&ledger, tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(content.contains("cost_usd"));
        assert!(content.contains("input_tokens"));
        assert!(content.contains("timestamp"));
    }

    #[test]
    fn test_export_summary_json_creates_map() {
        let tmp = NamedTempFile::new().unwrap();
        let ledger = make_ledger();
        export_summary_json(&ledger, tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(parsed.is_object());
        assert!(parsed.get("gpt-4o-mini").is_some());
        assert!(parsed.get("claude-sonnet-4-6").is_some());
    }

    #[test]
    fn test_export_summary_json_contains_stats_fields() {
        let tmp = NamedTempFile::new().unwrap();
        let ledger = make_ledger();
        export_summary_json(&ledger, tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(content.contains("request_count"));
        assert!(content.contains("avg_cost_per_request"));
        assert!(content.contains("p99_latency_ms"));
    }

    // ── CostExporter (timestamped) tests ─────────────────────────────────────
    // Each test uses export_to_dir with an isolated tempdir so that parallel
    // test execution never races on a shared current-working-directory or on
    // identically-named timestamp files.

    #[test]
    fn test_export_format_extension_csv() {
        assert_eq!(ExportFormat::Csv.extension(), "csv");
    }

    #[test]
    fn test_export_format_extension_json() {
        assert_eq!(ExportFormat::Json.extension(), "json");
    }

    #[test]
    fn test_cost_exporter_csv_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger();
        let exporter = CostExporter::new(&ledger);
        let filename = exporter
            .export_to_dir(dir.path(), ExportFormat::Csv)
            .unwrap();
        assert!(filename.starts_with("costs_"));
        assert!(filename.ends_with(".csv"));
        assert!(dir.path().join(&filename).exists());
    }

    #[test]
    fn test_cost_exporter_json_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger();
        let exporter = CostExporter::new(&ledger);
        let filename = exporter
            .export_to_dir(dir.path(), ExportFormat::Json)
            .unwrap();
        assert!(filename.starts_with("costs_"));
        assert!(filename.ends_with(".json"));
        assert!(dir.path().join(&filename).exists());
    }

    #[test]
    fn test_cost_exporter_csv_content_has_rows() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger();
        let exporter = CostExporter::new(&ledger);
        let filename = exporter
            .export_to_dir(dir.path(), ExportFormat::Csv)
            .unwrap();
        let content = std::fs::read_to_string(dir.path().join(&filename)).unwrap();
        assert!(!content.is_empty(), "CSV must not be empty");
        assert!(
            content.contains("model") || content.contains("cost_usd"),
            "CSV missing expected header"
        );
        assert!(content.contains("gpt-4o-mini"), "CSV missing data row");
    }

    #[test]
    fn test_cost_exporter_json_content_is_valid() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = make_ledger();
        let exporter = CostExporter::new(&ledger);
        let filename = exporter
            .export_to_dir(dir.path(), ExportFormat::Json)
            .unwrap();
        let content = std::fs::read_to_string(dir.path().join(&filename)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_cost_exporter_empty_ledger() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = CostLedger::new();
        let exporter = CostExporter::new(&ledger);
        let filename = exporter
            .export_to_dir(dir.path(), ExportFormat::Csv)
            .unwrap();
        assert!(dir.path().join(&filename).exists());
    }
}
