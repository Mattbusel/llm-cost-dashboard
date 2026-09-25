//! # Request Log
//!
//! Ordered append log of raw LLM requests.  Supports filter-by-model,
//! JSON serialization, ingestion from newline-delimited JSON files, and
//! automatic provider detection from HTTP response headers.
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
    /// Provider detected from HTTP response headers, if any.
    ///
    /// When present this overrides the `provider` field for display purposes.
    /// Set via [`LogEntry::apply_header_detection`].
    pub detected_provider: Option<String>,
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
            detected_provider: None,
        }
    }

    /// Inspect HTTP response headers and auto-detect the provider.
    ///
    /// Header rules applied in order:
    /// 1. `x-ratelimit-limit-tokens` present → Anthropic
    /// 2. `x-goog-request-params` present → Google/Gemini
    /// 3. `x-request-id` with a UUID-shaped value → OpenAI
    ///
    /// If a provider is detected it is written to `detected_provider` and also
    /// replaces `provider` when the current `provider` is `"unknown"`.
    ///
    /// `headers` is an iterator of `(header_name, header_value)` pairs.  Both
    /// names and values are compared case-insensitively.
    pub fn apply_header_detection<'a>(
        &mut self,
        headers: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) {
        let mut detected: Option<String> = None;

        for (name, value) in headers {
            let name_lc = name.to_lowercase();
            match name_lc.as_str() {
                "x-ratelimit-limit-tokens" => {
                    detected = Some("anthropic".into());
                    break;
                }
                "x-goog-request-params" => {
                    detected = Some("google".into());
                    break;
                }
                "x-request-id"
                    // OpenAI uses UUID-shaped request IDs; other providers may
                    // also send this header, so we only set it as a fallback.
                    if looks_like_uuid(value) && detected.is_none() => {
                        detected = Some("openai".into());
                    }
                _ => {}
            }
        }

        if let Some(ref p) = detected {
            self.detected_provider = Some(p.clone());
            if self.provider == "unknown" {
                self.provider = p.clone();
            }
        }
    }

    /// Return the effective provider: `detected_provider` if set, else `provider`.
    pub fn effective_provider(&self) -> &str {
        self.detected_provider.as_deref().unwrap_or(&self.provider)
    }
}

/// Returns `true` if `s` looks like a UUID (8-4-4-4-12 hex digits).
fn looks_like_uuid(s: &str) -> bool {
    let s = s.trim();
    if s.len() != 36 {
        return false;
    }
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected_lens = [8, 4, 4, 4, 12];
    parts
        .iter()
        .zip(expected_lens.iter())
        .all(|(part, &len)| part.len() == len && part.chars().all(|c| c.is_ascii_hexdigit()))
}

/// The JSON record format expected when tailing a log file.
///
/// Only the four required fields (`model`, `input_tokens`, `output_tokens`,
/// `latency_ms`) are mandatory.  All other fields have sensible defaults.
///
/// The optional `timestamp` field (aliases `ts`, `time`, `created_at`) accepts
/// an RFC 3339 string, a `YYYY-MM-DD HH:MM:SS` string (UTC), or a Unix time
/// number in seconds or milliseconds.  Lines without one are stamped with the
/// time they were read.
///
/// Example JSON line:
/// ```json
/// {"model":"gpt-4o-mini","input_tokens":512,"output_tokens":256,"latency_ms":34,"timestamp":"2026-09-25T14:03:00Z"}
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
    /// Optional request time.  See the type-level docs for accepted formats.
    #[serde(default, alias = "ts", alias = "time", alias = "created_at")]
    pub timestamp: Option<serde_json::Value>,
}

/// Parse a log-line timestamp value into UTC.
///
/// Accepts RFC 3339 strings, `YYYY-MM-DD HH:MM:SS` / `YYYY-MM-DDTHH:MM:SS`
/// strings interpreted as UTC, bare `YYYY-MM-DD` dates, and Unix time numbers
/// (seconds, or milliseconds when larger than 10^11).  Returns `None` for
/// anything else.
pub fn parse_timestamp(v: &serde_json::Value) -> Option<DateTime<Utc>> {
    fn from_unix(n: f64) -> Option<DateTime<Utc>> {
        if !n.is_finite() || n < 0.0 {
            return None;
        }
        let ms = if n > 1e11 { n } else { n * 1000.0 };
        DateTime::<Utc>::from_timestamp_millis(ms as i64)
    }
    match v {
        serde_json::Value::Number(n) => n.as_f64().and_then(from_unix),
        serde_json::Value::String(s) => {
            let s = s.trim();
            if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
                return Some(dt.with_timezone(&Utc));
            }
            for fmt in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
                if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
                    return Some(ndt.and_utc());
                }
            }
            if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                return d.and_hms_opt(0, 0, 0).map(|ndt| ndt.and_utc());
            }
            s.parse::<f64>().ok().and_then(from_unix)
        }
        _ => None,
    }
}

impl From<IncomingRecord> for LogEntry {
    fn from(r: IncomingRecord) -> Self {
        let success = r.error.is_none();
        let timestamp = r
            .timestamp
            .as_ref()
            .and_then(parse_timestamp)
            .unwrap_or_else(Utc::now);
        Self {
            id: Uuid::new_v4(),
            timestamp,
            provider: r.provider.unwrap_or_else(|| "unknown".into()),
            model: r.model,
            input_tokens: r.input_tokens,
            output_tokens: r.output_tokens,
            latency_ms: r.latency_ms,
            success,
            error: r.error,
            detected_provider: None,
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
        let value: serde_json::Value = serde_json::from_str(line.trim())
            .map_err(|e| DashboardError::LogParseError(e.to_string()))?;
        // Only JSON objects are valid records; serde would otherwise accept a
        // positional array such as `["gpt-4o", 100, 50, 20]`.
        if !value.is_object() {
            return Err(DashboardError::LogParseError(
                "expected a JSON object per line".into(),
            ));
        }
        let record: IncomingRecord = serde_json::from_value(value)
            .map_err(|e| DashboardError::LogParseError(e.to_string()))?;
        if let Some(ts) = &record.timestamp {
            if !ts.is_null() && parse_timestamp(ts).is_none() {
                return Err(DashboardError::LogParseError(format!(
                    "unrecognised timestamp {ts}; use RFC 3339 or Unix seconds"
                )));
            }
        }
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

    /// Return a mutable slice of all entries (used for header-based detection updates).
    pub fn all_mut(&mut self) -> &mut [LogEntry] {
        &mut self.entries
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
    fn test_ingest_uses_timestamp_field() {
        let mut log = RequestLog::new();
        log.ingest_line(r#"{"model":"m","input_tokens":1,"output_tokens":1,"latency_ms":1,"timestamp":"2024-01-08T10:00:00Z"}"#).unwrap();
        log.ingest_line(r#"{"model":"m","input_tokens":1,"output_tokens":1,"latency_ms":1,"ts":1704067200}"#).unwrap();
        log.ingest_line(r#"{"model":"m","input_tokens":1,"output_tokens":1,"latency_ms":1,"ts":1704067200000}"#).unwrap();
        let all = log.all();
        assert_eq!(all[0].timestamp.to_rfc3339(), "2024-01-08T10:00:00+00:00");
        assert_eq!(all[1].timestamp.timestamp(), 1_704_067_200);
        assert_eq!(all[2].timestamp.timestamp(), 1_704_067_200);
    }

    #[test]
    fn test_ingest_bad_timestamp_is_error() {
        let mut log = RequestLog::new();
        let line = r#"{"model":"m","input_tokens":1,"output_tokens":1,"latency_ms":1,"timestamp":"yesterday"}"#;
        assert!(log.ingest_line(line).is_err());
        assert!(log.is_empty());
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

    // ── Provider auto-detection tests ────────────────────────────────────────

    #[test]
    fn test_header_detection_anthropic_via_ratelimit_header() {
        let mut entry = LogEntry::new("claude-sonnet-4-6", "unknown", 100, 50, 10);
        entry.apply_header_detection([("x-ratelimit-limit-tokens", "50000")]);
        assert_eq!(entry.detected_provider.as_deref(), Some("anthropic"));
        assert_eq!(entry.provider, "anthropic"); // upgraded from unknown
    }

    #[test]
    fn test_header_detection_google_via_goog_params() {
        let mut entry = LogEntry::new("gemini-1.5-flash", "unknown", 100, 50, 10);
        entry.apply_header_detection([("x-goog-request-params", "model=gemini")]);
        assert_eq!(entry.detected_provider.as_deref(), Some("google"));
    }

    #[test]
    fn test_header_detection_openai_via_uuid_request_id() {
        let mut entry = LogEntry::new("gpt-4o", "unknown", 100, 50, 10);
        entry
            .apply_header_detection([("x-request-id", "550e8400-e29b-41d4-a716-446655440000")]);
        assert_eq!(entry.detected_provider.as_deref(), Some("openai"));
    }

    #[test]
    fn test_header_detection_non_uuid_request_id_ignored() {
        let mut entry = LogEntry::new("gpt-4o", "unknown", 100, 50, 10);
        entry.apply_header_detection([("x-request-id", "not-a-uuid")]);
        // Should not be detected as openai since it's not a UUID.
        assert!(entry.detected_provider.is_none());
    }

    #[test]
    fn test_header_detection_does_not_overwrite_known_provider() {
        let mut entry = LogEntry::new("gpt-4o", "openai", 100, 50, 10);
        entry.apply_header_detection([("x-ratelimit-limit-tokens", "50000")]);
        // detected_provider is set, but provider stays "openai" (not "unknown").
        assert_eq!(entry.detected_provider.as_deref(), Some("anthropic"));
        assert_eq!(entry.provider, "openai"); // not overwritten
    }

    #[test]
    fn test_header_detection_no_relevant_headers() {
        let mut entry = LogEntry::new("gpt-4o", "unknown", 100, 50, 10);
        entry.apply_header_detection([("content-type", "application/json")]);
        assert!(entry.detected_provider.is_none());
    }

    #[test]
    fn test_effective_provider_uses_detected_when_set() {
        let mut entry = LogEntry::new("claude-sonnet-4-6", "unknown", 100, 50, 10);
        entry.apply_header_detection([("x-ratelimit-limit-tokens", "50000")]);
        assert_eq!(entry.effective_provider(), "anthropic");
    }

    #[test]
    fn test_effective_provider_falls_back_to_provider_field() {
        let entry = LogEntry::new("gpt-4o", "openai", 100, 50, 10);
        assert_eq!(entry.effective_provider(), "openai");
    }

    #[test]
    fn test_looks_like_uuid_valid() {
        assert!(super::looks_like_uuid("550e8400-e29b-41d4-a716-446655440000"));
    }

    #[test]
    fn test_looks_like_uuid_invalid_too_short() {
        assert!(!super::looks_like_uuid("550e8400-e29b-41d4"));
    }

    #[test]
    fn test_looks_like_uuid_invalid_non_hex() {
        assert!(!super::looks_like_uuid("zzzzzzzz-e29b-41d4-a716-446655440000"));
    }

    #[test]
    fn test_detected_provider_serializes_to_json() {
        let mut entry = LogEntry::new("gpt-4o", "unknown", 10, 5, 1);
        entry.apply_header_detection([("x-ratelimit-limit-tokens", "1000")]);
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("detected_provider"));
        assert!(json.contains("anthropic"));
    }
}
