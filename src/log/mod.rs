//! # Request Log
//!
//! Ordered append log of raw LLM requests.  Supports filter-by-model,
//! JSON serialization, and ingestion from newline-delimited JSON files.
//!
//! The [`RequestLog`] never panics on malformed input; callers receive a
//! [`crate::error::DashboardError::LogParseError`] and can choose to skip the
//! bad line and continue.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::DashboardError;

/// Raw log entry representing one completed LLM request.
///
/// Entries are created either programmatically via [`LogEntry::new`] or by
/// converting an [`IncomingRecord`] parsed from a JSON log line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Unique identifier for this entry.
    pub id: Uuid,
    /// Wall-clock time of the request (UTC).
    pub timestamp: DateTime<Utc>,
    /// Model identifier, e.g. `"gpt-4o-mini"`.
    pub model: String,
    /// Provider name, e.g. `"openai"`.
    pub provider: String,
    /// Number of input (prompt) tokens consumed.
    pub input_tokens: u64,
    /// Number of output (completion) tokens produced.
    pub output_tokens: u64,
    /// End-to-end request latency in milliseconds.
    pub latency_ms: u64,
    /// Whether the request completed without an error.
    pub success: bool,
    /// Optional error message when `success` is `false`.
    pub error: Option<String>,
}

impl LogEntry {
    /// Construct a successful log entry with the given parameters.
    pub fn new(
        model: impl Into<String>,
        provider: impl Into<String>,
        input: u64,
        output: u64,
        latency_ms: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            model: model.into(),
            provider: provider.into(),
            input_tokens: input,
            output_tokens: output,
            latency_ms,
            success: true,
            error: None,
        }
    }
}

/// The JSON record format expected when tailing a log file.
///
/// Only the four required fields (`model`, `input_tokens`, `output_tokens`,
/// `latency_ms`) are mandatory.  All other fields have sensible defaults.
///
/// Example JSON line:
/// ```json
/// {"model":"gpt-4o-mini","input_tokens":512,"output_tokens":256,"latency_ms":34}
/// ```
#[derive(Debug, Deserialize)]
pub struct IncomingRecord {
    /// Model identifier.
    pub model: String,
    /// Number of input tokens.
    pub input_tokens: u64,
    /// Number of output tokens.
    pub output_tokens: u64,
    /// Request latency in milliseconds.
    pub latency_ms: u64,
    /// Optional provider name; defaults to `"unknown"` when absent.
    #[serde(default)]
    pub provider: Option<String>,
    /// Optional error description; presence implies `success = false`.
    #[serde(default)]
    pub error: Option<String>,
}

impl From<IncomingRecord> for LogEntry {
    fn from(r: IncomingRecord) -> Self {
        let success = r.error.is_none();
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            provider: r.provider.unwrap_or_else(|| "unknown".into()),
            model: r.model,
            input_tokens: r.input_tokens,
            output_tokens: r.output_tokens,
            latency_ms: r.latency_ms,
            success,
            error: r.error,
        }
    }
}

/// Append-only ordered log of [`LogEntry`] records.
///
/// Entries are stored in insertion order.  The log does not perform any
/// deduplication; it is the caller's responsibility to avoid duplicate lines.
#[derive(Debug, Default)]
pub struct RequestLog {
    entries: Vec<LogEntry>,
}

impl RequestLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an already-constructed entry.
    pub fn append(&mut self, entry: LogEntry) {
        self.entries.push(entry);
    }

    /// Parse a single newline-delimited JSON line and append the resulting entry.
    ///
    /// Returns [`DashboardError::LogParseError`] on malformed input so the
    /// caller can surface the error in the UI rather than panicking.
    pub fn ingest_line(&mut self, line: &str) -> Result<(), DashboardError> {
        let record: IncomingRecord = serde_json::from_str(line.trim())
            .map_err(|e| DashboardError::LogParseError(e.to_string()))?;
        self.append(record.into());
        Ok(())
    }

    /// Iterate over entries whose model matches `model` (case-insensitive).
    pub fn filter_by_model<'a>(&'a self, model: &'a str) -> impl Iterator<Item = &'a LogEntry> {
        self.entries
            .iter()
            .filter(move |e| e.model.eq_ignore_ascii_case(model))
    }

    /// Return a slice of all entries in insertion order.
    pub fn all(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize all entries to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, DashboardError> {
        serde_json::to_string_pretty(&self.entries).map_err(Into::into)
    }

    /// Remove all entries from the log.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_append_increases_len() {
        let mut log = RequestLog::new();
        log.append(LogEntry::new("gpt-4o", "openai", 100, 50, 20));
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_all_returns_in_order() {
        let mut log = RequestLog::new();
        log.append(LogEntry::new("a", "p", 1, 1, 1));
        log.append(LogEntry::new("b", "p", 2, 2, 2));
        let all = log.all();
        assert_eq!(all[0].model, "a");
        assert_eq!(all[1].model, "b");
    }

    #[test]
    fn test_filter_by_model_returns_matching() {
        let mut log = RequestLog::new();
        log.append(LogEntry::new("gpt-4o", "openai", 100, 50, 20));
        log.append(LogEntry::new("claude-sonnet-4-6", "anthropic", 100, 50, 20));
        let results: Vec<_> = log.filter_by_model("gpt-4o").collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].model, "gpt-4o");
    }

    #[test]
    fn test_filter_by_model_case_insensitive() {
        let mut log = RequestLog::new();
        log.append(LogEntry::new("GPT-4O", "openai", 100, 50, 20));
        let results: Vec<_> = log.filter_by_model("gpt-4o").collect();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_no_match_returns_empty() {
        let mut log = RequestLog::new();
        log.append(LogEntry::new("gpt-4o", "openai", 100, 50, 20));
        let results: Vec<_> = log.filter_by_model("claude-sonnet-4-6").collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_ingest_valid_json_line() {
        let mut log = RequestLog::new();
        let line =
            r#"{"model":"gpt-4o-mini","input_tokens":512,"output_tokens":256,"latency_ms":34}"#;
        log.ingest_line(line).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log.all()[0].model, "gpt-4o-mini");
        assert_eq!(log.all()[0].input_tokens, 512);
    }

    #[test]
    fn test_ingest_invalid_json_returns_error() {
        let mut log = RequestLog::new();
        assert!(log.ingest_line("not json").is_err());
    }

    #[test]
    fn test_ingest_missing_required_field_returns_error() {
        let mut log = RequestLog::new();
        // missing output_tokens and latency_ms
        let line = r#"{"model":"gpt-4o-mini","input_tokens":512}"#;
        assert!(log.ingest_line(line).is_err());
    }

    #[test]
    fn test_ingest_error_is_log_parse_error_variant() {
        let mut log = RequestLog::new();
        let err = log.ingest_line("bad").unwrap_err();
        assert!(matches!(
            err,
            crate::error::DashboardError::LogParseError(_)
        ));
    }

    #[test]
    fn test_ingest_unknown_model_accepted_gracefully() {
        // Unknown model names are valid at the log layer; cost layer applies fallback pricing.
        let mut log = RequestLog::new();
        let line =
            r#"{"model":"my-custom-model","input_tokens":100,"output_tokens":50,"latency_ms":10}"#;
        log.ingest_line(line).unwrap();
        assert_eq!(log.all()[0].model, "my-custom-model");
    }

    #[test]
    fn test_ingest_with_optional_provider_field() {
        let mut log = RequestLog::new();
        let line = r#"{"model":"gpt-4o","input_tokens":10,"output_tokens":5,"latency_ms":20,"provider":"openai"}"#;
        log.ingest_line(line).unwrap();
        assert_eq!(log.all()[0].provider, "openai");
    }

    #[test]
    fn test_ingest_missing_provider_defaults_to_unknown() {
        let mut log = RequestLog::new();
        let line = r#"{"model":"gpt-4o","input_tokens":10,"output_tokens":5,"latency_ms":20}"#;
        log.ingest_line(line).unwrap();
        assert_eq!(log.all()[0].provider, "unknown");
    }

    #[test]
    fn test_ingest_with_error_field_marks_success_false() {
        let mut log = RequestLog::new();
        let line = r#"{"model":"gpt-4o","input_tokens":0,"output_tokens":0,"latency_ms":5,"error":"timeout"}"#;
        log.ingest_line(line).unwrap();
        assert!(!log.all()[0].success);
        assert_eq!(log.all()[0].error.as_deref(), Some("timeout"));
    }

    #[test]
    fn test_ingest_empty_string_returns_error() {
        let mut log = RequestLog::new();
        assert!(log.ingest_line("").is_err());
    }

    #[test]
    fn test_ingest_whitespace_only_returns_error() {
        let mut log = RequestLog::new();
        assert!(log.ingest_line("   ").is_err());
    }

    #[test]
    fn test_to_json_roundtrip() {
        let mut log = RequestLog::new();
        log.append(LogEntry::new("gpt-4o", "openai", 100, 50, 20));
        let json = log.to_json().unwrap();
        assert!(json.contains("gpt-4o"));
    }

    #[test]
    fn test_clear_empties_log() {
        let mut log = RequestLog::new();
        log.append(LogEntry::new("gpt-4o", "openai", 100, 50, 20));
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_is_empty_on_new_log() {
        assert!(RequestLog::new().is_empty());
    }

    #[test]
    fn test_multiple_ingests_accumulate() {
        let mut log = RequestLog::new();
        for _ in 0..5 {
            let line = r#"{"model":"gpt-4o","input_tokens":10,"output_tokens":5,"latency_ms":10}"#;
            log.ingest_line(line).unwrap();
        }
        assert_eq!(log.len(), 5);
    }
}
