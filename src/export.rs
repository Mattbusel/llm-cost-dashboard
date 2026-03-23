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
}

impl ExportFormat {
    /// File extension for this format (without the leading dot).
    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Csv => "csv",
            ExportFormat::Json => "json",
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
