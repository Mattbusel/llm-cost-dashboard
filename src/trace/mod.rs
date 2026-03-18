//! # Trace Spans
//!
//! Lightweight distributed tracing for LLM requests.  Each [`TraceSpan`]
//! records a request correlation ID, model, latency, token counts, and
//! computed cost.  Spans are stored in a [`SpanStore`] in insertion order.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cost::pricing::compute_cost;
use crate::error::DashboardError;

/// A single traced LLM request span.
///
/// Created with [`TraceSpan::new`]; use the builder methods [`TraceSpan::with_tag`]
/// and [`TraceSpan::failed`] to annotate spans before recording them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Unique identifier for this span.
    pub span_id: Uuid,
    /// Caller-supplied request correlation ID.
    pub request_id: String,
    /// Model identifier, e.g. `"gpt-4o"`.
    pub model: String,
    /// Provider name, e.g. `"openai"`.
    pub provider: String,
    /// Wall-clock start time (UTC).
    pub started_at: DateTime<Utc>,
    /// End-to-end latency in milliseconds.
    pub latency_ms: u64,
    /// Number of input (prompt) tokens consumed.
    pub input_tokens: u64,
    /// Number of output (completion) tokens produced.
    pub output_tokens: u64,
    /// Computed cost in USD.
    pub cost_usd: f64,
    /// Whether the request completed without an error.
    pub success: bool,
    /// Arbitrary string tags for filtering and attribution.
    pub tags: Vec<String>,
}

impl TraceSpan {
    /// Create a successful span with cost computed from the pricing table.
    pub fn new(
        request_id: impl Into<String>,
        model: impl Into<String>,
        provider: impl Into<String>,
        input_tokens: u64,
        output_tokens: u64,
        latency_ms: u64,
    ) -> Self {
        let model = model.into();
        let cost = compute_cost(&model, input_tokens, output_tokens);
        Self {
            span_id: Uuid::new_v4(),
            request_id: request_id.into(),
            model,
            provider: provider.into(),
            started_at: Utc::now(),
            latency_ms,
            input_tokens,
            output_tokens,
            cost_usd: cost,
            success: true,
            tags: Vec::new(),
        }
    }

    /// Attach an arbitrary string tag and return `self` (builder pattern).
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Mark the span as failed with an error reason tag.
    ///
    /// Sets `success = false` and appends an `"error:<reason>"` tag.
    pub fn failed(mut self, reason: impl Into<String>) -> Self {
        self.success = false;
        self.tags.push(format!("error:{}", reason.into()));
        self
    }

    /// Serialize the span to a compact JSON string.
    pub fn to_json(&self) -> Result<String, DashboardError> {
        serde_json::to_string(self).map_err(Into::into)
    }
}

/// In-memory append-only store for [`TraceSpan`] records.
///
/// Spans are stored in insertion order.  All retrieval methods are O(n).
#[derive(Debug, Default)]
pub struct SpanStore {
    spans: Vec<TraceSpan>,
}

impl SpanStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a span.
    pub fn record(&mut self, span: TraceSpan) {
        self.spans.push(span);
    }

    /// Find the first span whose `request_id` matches exactly.
    pub fn by_request(&self, request_id: &str) -> Option<&TraceSpan> {
        self.spans.iter().find(|s| s.request_id == request_id)
    }

    /// Return a slice of all spans in insertion order.
    pub fn all(&self) -> &[TraceSpan] {
        &self.spans
    }

    /// Sum of all span costs in USD.
    pub fn total_cost(&self) -> f64 {
        self.spans.iter().map(|s| s.cost_usd).sum()
    }

    /// Number of spans where `success == false`.
    pub fn failure_count(&self) -> usize {
        self.spans.iter().filter(|s| !s.success).count()
    }

    /// Total number of recorded spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the store contains no spans.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_span_records_latency() {
        let s = TraceSpan::new("req1", "gpt-4o", "openai", 100, 50, 42);
        assert_eq!(s.latency_ms, 42);
    }

    #[test]
    fn test_span_cost_computed_from_tokens() {
        let s = TraceSpan::new("req1", "claude-sonnet-4-6", "anthropic", 1_000_000, 0, 100);
        assert!((s.cost_usd - 3.00).abs() < 1e-9);
    }

    #[test]
    fn test_span_unknown_model_uses_fallback() {
        let s = TraceSpan::new("req1", "unknown-model", "provider", 1_000_000, 0, 100);
        assert!(s.cost_usd > 0.0);
    }

    #[test]
    fn test_span_to_json_roundtrip() {
        let s = TraceSpan::new("req1", "gpt-4o", "openai", 100, 50, 42);
        let json = s.to_json().unwrap();
        let deserialized: TraceSpan = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.request_id, "req1");
        assert_eq!(deserialized.latency_ms, 42);
    }

    #[test]
    fn test_span_with_tag() {
        let s = TraceSpan::new("req1", "gpt-4o", "openai", 100, 50, 42).with_tag("production");
        assert!(s.tags.contains(&"production".to_string()));
    }

    #[test]
    fn test_span_failed_marks_success_false() {
        let s = TraceSpan::new("req1", "gpt-4o", "openai", 100, 50, 42).failed("timeout");
        assert!(!s.success);
        assert!(s.tags.iter().any(|t| t.starts_with("error:")));
    }

    #[test]
    fn test_store_record_and_retrieve() {
        let mut store = SpanStore::new();
        store.record(TraceSpan::new("req-abc", "gpt-4o", "openai", 100, 50, 42));
        assert!(store.by_request("req-abc").is_some());
        assert!(store.by_request("req-xyz").is_none());
    }

    #[test]
    fn test_store_total_cost() {
        let mut store = SpanStore::new();
        store.record(TraceSpan::new(
            "r1",
            "claude-sonnet-4-6",
            "anthropic",
            1_000_000,
            0,
            10,
        ));
        store.record(TraceSpan::new(
            "r2",
            "claude-sonnet-4-6",
            "anthropic",
            1_000_000,
            0,
            10,
        ));
        assert!((store.total_cost() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_store_failure_count() {
        let mut store = SpanStore::new();
        store.record(TraceSpan::new("r1", "gpt-4o", "openai", 100, 50, 10).failed("err"));
        store.record(TraceSpan::new("r2", "gpt-4o", "openai", 100, 50, 10));
        assert_eq!(store.failure_count(), 1);
    }

    #[test]
    fn test_store_is_empty_on_new() {
        assert!(SpanStore::new().is_empty());
    }

    #[test]
    fn test_span_zero_tokens_zero_cost() {
        let s = TraceSpan::new("r", "gpt-4o-mini", "openai", 0, 0, 5);
        assert_eq!(s.cost_usd, 0.0);
    }
}
