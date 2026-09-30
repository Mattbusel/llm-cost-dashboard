//! # Cost Ledger
//!
//! Tracks every LLM request with its token counts, latency, and USD cost.
//! Provides aggregation by model, time-window projections, and per-request
//! history.
//!
//! The primary entry point is [`CostLedger`].  Create records with
//! [`CostRecord::new`] and append them with [`CostLedger::add`].

pub mod anomaly;
pub mod pricing;

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::DashboardError;

/// Per-token-type cache breakdown for a single request.
///
/// Claude Prompt Cache tokens (cache reads) are billed at a significant
/// discount vs. ordinary prompt tokens (cache misses / writes).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheBreakdown {
    /// Input tokens that were served from the prompt cache (cache hits).
    ///
    /// These are billed at roughly 10% of the normal input rate for Claude
    /// models.
    pub cache_read_tokens: u64,
    /// Input tokens that populated the prompt cache (cache writes).
    ///
    /// These are typically billed at 125% of the normal input rate for Claude.
    pub cache_write_tokens: u64,
    /// USD cost of cache-read tokens.
    pub cache_read_cost_usd: f64,
    /// USD cost of cache-write tokens.
    pub cache_write_cost_usd: f64,
}

/// A single completed LLM request with token counts and computed USD cost.
///
/// Costs are calculated automatically from the pricing table when the record
/// is constructed with [`CostRecord::new`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecord {
    /// Unique record identifier.
    pub id: Uuid,
    /// Wall-clock timestamp of the request (UTC).
    pub timestamp: DateTime<Utc>,
    /// Model identifier, e.g. `"gpt-4o-mini"`.
    pub model: String,
    /// Provider name, e.g. `"openai"`.
    pub provider: String,
    /// Number of input (prompt) tokens.
    pub input_tokens: u64,
    /// Number of output (completion) tokens.
    pub output_tokens: u64,
    /// Cost of input tokens in USD.
    pub input_cost_usd: f64,
    /// Cost of output tokens in USD.
    pub output_cost_usd: f64,
    /// Total cost in USD (`input_cost_usd + output_cost_usd`).
    pub total_cost_usd: f64,
    /// End-to-end latency in milliseconds.
    pub latency_ms: u64,
    /// Caller-supplied or auto-generated request correlation ID.
    pub request_id: String,
    /// Cache hit/miss breakdown (zero-valued when not applicable).
    pub cache: CacheBreakdown,
    /// Optional session identifier — set via `--session` or the library API.
    ///
    /// When `None`, the record is not associated with any named session.
    pub session_id: Option<String>,
}

impl CostRecord {
    /// Create a record, computing cost automatically from the pricing table.
    ///
    /// If the model is not in the pricing table, fallback pricing is used and
    /// no error is returned at this layer (the caller may emit a
    /// [`DashboardError::UnknownModel`] separately if desired).
    pub fn new(
        model: impl Into<String>,
        provider: impl Into<String>,
        input_tokens: u64,
        output_tokens: u64,
        latency_ms: u64,
    ) -> Self {
        let model = model.into();
        let provider = provider.into();
        let (ir, or_) = pricing::lookup(&model);
        let input_cost = input_tokens as f64 * ir / 1_000_000.0;
        let output_cost = output_tokens as f64 * or_ / 1_000_000.0;
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            model,
            provider,
            input_tokens,
            output_tokens,
            input_cost_usd: input_cost,
            output_cost_usd: output_cost,
            total_cost_usd: input_cost + output_cost,
            latency_ms,
            request_id: Uuid::new_v4().to_string(),
            cache: CacheBreakdown::default(),
            session_id: None,
        }
    }

    /// Attach a session identifier to this record.
    ///
    /// Returns `self` for builder-style chaining.
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Attach cache hit/miss token counts and recompute the total cost.
    ///
    /// Cache-read tokens are billed at 10% of the model's normal input rate.
    /// Cache-write tokens are billed at 125% of the model's normal input rate.
    /// Both are *in addition* to any non-cached `input_tokens` already on the
    /// record.
    pub fn with_cache(
        mut self,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
    ) -> Self {
        let (ir, _) = pricing::lookup(&self.model);
        let read_cost = cache_read_tokens as f64 * ir * 0.10 / 1_000_000.0;
        let write_cost = cache_write_tokens as f64 * ir * 1.25 / 1_000_000.0;
        self.cache = CacheBreakdown {
            cache_read_tokens,
            cache_write_tokens,
            cache_read_cost_usd: read_cost,
            cache_write_cost_usd: write_cost,
        };
        self.total_cost_usd =
            self.input_cost_usd + self.output_cost_usd + read_cost + write_cost;
        self
    }
}

/// Aggregated statistics for a single model.
///
/// Produced by [`CostLedger::by_model`].
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Model identifier.
    pub model: String,
    /// Total number of requests for this model.
    pub request_count: u64,
    /// Sum of all input tokens across all requests.
    pub total_input_tokens: u64,
    /// Sum of all output tokens across all requests.
    pub total_output_tokens: u64,
    /// Sum of all costs in USD.
    pub total_cost_usd: f64,
    /// Mean cost per request in USD.
    pub avg_cost_per_request: f64,
    /// Mean latency in milliseconds.
    pub avg_latency_ms: f64,
    /// 99th-percentile latency in milliseconds.
    pub p99_latency_ms: f64,
}

/// Append-only ledger of all cost records.
///
/// Thread safety: this type is **not** `Send`/`Sync` — wrap it in a `Mutex`
/// if you need to share it across threads.
#[derive(Debug, Default)]
pub struct CostLedger {
    records: Vec<CostRecord>,
}

impl CostLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new record.
    ///
    /// Returns [`DashboardError::Ledger`] if the record has a negative total
    /// cost (which indicates a programming error upstream).
    pub fn add(&mut self, record: CostRecord) -> Result<(), DashboardError> {
        if record.total_cost_usd < 0.0 {
            return Err(DashboardError::Ledger("negative cost".into()));
        }
        self.records.push(record);
        Ok(())
    }

    /// Sum of all recorded costs in USD.
    pub fn total_usd(&self) -> f64 {
        // `+ 0.0` turns the -0.0 an empty f64 sum yields into 0.0, so an empty
        // ledger never prints as "$-0.00".
        self.records.iter().map(|r| r.total_cost_usd).sum::<f64>() + 0.0
    }

    /// Aggregated stats keyed by model name.
    pub fn by_model(&self) -> HashMap<String, ModelStats> {
        let mut map: HashMap<String, Vec<&CostRecord>> = HashMap::new();
        for r in &self.records {
            map.entry(r.model.clone()).or_default().push(r);
        }
        map.into_iter()
            .map(|(model, recs)| {
                let count = recs.len() as u64;
                let total_cost: f64 = recs.iter().map(|r| r.total_cost_usd).sum();
                let total_in: u64 = recs.iter().map(|r| r.input_tokens).sum();
                let total_out: u64 = recs.iter().map(|r| r.output_tokens).sum();
                let avg_cost = if count > 0 {
                    total_cost / count as f64
                } else {
                    0.0
                };
                let mut latencies: Vec<u64> = recs.iter().map(|r| r.latency_ms).collect();
                latencies.sort_unstable();
                let avg_lat = if count > 0 {
                    latencies.iter().sum::<u64>() as f64 / count as f64
                } else {
                    0.0
                };
                let p99 = if latencies.is_empty() {
                    0.0
                } else {
                    // ceil-based index: for N=100 gives index 98 (99th value).
                    let idx = ((latencies.len() as f64 * 0.99).ceil() as usize)
                        .saturating_sub(1)
                        .min(latencies.len() - 1);
                    latencies[idx] as f64
                };
                (
                    model.clone(),
                    ModelStats {
                        model,
                        request_count: count,
                        total_input_tokens: total_in,
                        total_output_tokens: total_out,
                        total_cost_usd: total_cost,
                        avg_cost_per_request: avg_cost,
                        avg_latency_ms: avg_lat,
                        p99_latency_ms: p99,
                    },
                )
            })
            .collect()
    }

    /// Return the last `n` records (oldest first).
    ///
    /// If `n >= self.len()` all records are returned.
    pub fn last_n(&self, n: usize) -> &[CostRecord] {
        let len = self.records.len();
        if n >= len {
            &self.records
        } else {
            &self.records[len - n..]
        }
    }

    /// Records whose timestamp is at or after `ts`.
    pub fn since(&self, ts: DateTime<Utc>) -> Vec<&CostRecord> {
        self.records.iter().filter(|r| r.timestamp >= ts).collect()
    }

    /// Extrapolate a 30-day monthly projection based on spend in the last
    /// `window_hours` hours.
    ///
    /// Returns `0.0` when `window_hours` is zero or the ledger is empty.
    pub fn projected_monthly_usd(&self, window_hours: u64) -> f64 {
        if window_hours == 0 {
            return 0.0;
        }
        let cutoff = Utc::now() - Duration::hours(window_hours as i64);
        let window_total: f64 = self
            .records
            .iter()
            .filter(|r| r.timestamp >= cutoff)
            .map(|r| r.total_cost_usd)
            .sum();
        (window_total / window_hours as f64) * 24.0 * 30.0 + 0.0
    }

    /// Total number of records in the ledger.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the ledger contains no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Remove all records from the ledger.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Return a slice of all records in insertion order.
    ///
    /// This is the primary accessor for export functions that need to iterate
    /// every record in the ledger.
    pub fn records(&self) -> &[CostRecord] {
        &self.records
    }

    /// Sparkline data: the last `n` total_cost values scaled to integer
    /// micro-USD (multiply by 1,000,000) for use with ratatui `Sparkline`.
    pub fn sparkline_data(&self, n: usize) -> Vec<u64> {
        self.last_n(n)
            .iter()
            .map(|r| (r.total_cost_usd * 1_000_000.0) as u64)
            .collect()
    }

    /// 7-day daily spend trend.
    ///
    /// Returns a `Vec` of 7 elements, one per calendar day from oldest (index 0)
    /// to today (index 6).  Each value is the total USD cost for that day.
    /// Days with no data have a value of `0.0`.
    pub fn seven_day_trend(&self) -> [f64; 7] {
        let today = Utc::now().date_naive();
        let mut trend = [0.0f64; 7];
        for record in &self.records {
            let day = record.timestamp.date_naive();
            let delta = (today - day).num_days();
            if (0..7).contains(&delta) {
                // delta 0 = today → index 6, delta 6 → index 0
                trend[(6 - delta) as usize] += record.total_cost_usd;
            }
        }
        trend
    }

    /// Serialize all records to a JSON string.
    ///
    /// Returns a pretty-printed JSON array of [`CostRecord`] objects.
    pub fn to_json(&self) -> Result<String, DashboardError> {
        serde_json::to_string_pretty(&self.records).map_err(Into::into)
    }

    /// Serialize all records to a CSV string.
    ///
    /// The first line is a header row.  Each subsequent line is one
    /// [`CostRecord`].
    pub fn to_csv(&self) -> Result<String, DashboardError> {
        let mut wtr = csv::Writer::from_writer(vec![]);
        // Write header
        wtr.write_record([
            "id",
            "timestamp",
            "model",
            "provider",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "cache_read_cost_usd",
            "cache_write_cost_usd",
            "total_cost_usd",
            "latency_ms",
            "request_id",
        ])
        .map_err(|e| DashboardError::Ledger(e.to_string()))?;
        for r in &self.records {
            wtr.write_record([
                r.id.to_string(),
                r.timestamp.to_rfc3339(),
                r.model.clone(),
                r.provider.clone(),
                r.input_tokens.to_string(),
                r.output_tokens.to_string(),
                r.cache.cache_read_tokens.to_string(),
                r.cache.cache_write_tokens.to_string(),
                format!("{:.10}", r.input_cost_usd),
                format!("{:.10}", r.output_cost_usd),
                format!("{:.10}", r.cache.cache_read_cost_usd),
                format!("{:.10}", r.cache.cache_write_cost_usd),
                format!("{:.10}", r.total_cost_usd),
                r.latency_ms.to_string(),
                r.request_id.clone(),
            ])
            .map_err(|e| DashboardError::Ledger(e.to_string()))?;
        }
        let bytes = wtr
            .into_inner()
            .map_err(|e| DashboardError::Ledger(e.to_string()))?;
        String::from_utf8(bytes).map_err(|e| DashboardError::Ledger(e.to_string()))
    }

}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_record(model: &str, input: u64, output: u64, latency: u64) -> CostRecord {
        CostRecord::new(model, "test", input, output, latency)
    }

    #[test]
    fn test_add_record_increases_len() {
        let mut ledger = CostLedger::new();
        assert_eq!(ledger.len(), 0);
        ledger
            .add(make_record("claude-sonnet-4-6", 100, 50, 100))
            .unwrap();
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn test_total_usd_empty_is_zero() {
        assert_eq!(CostLedger::new().total_usd(), 0.0);
    }

    #[test]
    fn test_total_usd_sums_records() {
        let mut ledger = CostLedger::new();
        ledger
            .add(make_record("claude-sonnet-4-6", 1_000_000, 0, 100))
            .unwrap();
        ledger
            .add(make_record("claude-sonnet-4-6", 1_000_000, 0, 100))
            .unwrap();
        assert!((ledger.total_usd() - 6.00).abs() < 1e-9);
    }

    #[test]
    fn test_negative_cost_rejected() {
        let mut ledger = CostLedger::new();
        let mut r = make_record("claude-sonnet-4-6", 0, 0, 0);
        r.total_cost_usd = -1.0;
        assert!(ledger.add(r).is_err());
    }

    #[test]
    fn test_zero_tokens_cost_is_zero() {
        let rec = make_record("gpt-4o-mini", 0, 0, 0);
        assert_eq!(rec.total_cost_usd, 0.0);
    }

    #[test]
    fn test_known_model_cost_formula() {
        // gpt-4o-mini: $0.15/1M input, $0.60/1M output
        let rec = make_record("gpt-4o-mini", 1_000_000, 1_000_000, 0);
        let expected = 0.15 + 0.60;
        assert!((rec.total_cost_usd - expected).abs() < 1e-9);
    }

    #[test]
    fn test_by_model_groups_correctly() {
        let mut ledger = CostLedger::new();
        ledger.add(make_record("gpt-4o-mini", 100, 50, 10)).unwrap();
        ledger
            .add(make_record("gpt-4o-mini", 200, 100, 20))
            .unwrap();
        ledger
            .add(make_record("claude-sonnet-4-6", 300, 150, 30))
            .unwrap();
        let stats = ledger.by_model();
        assert_eq!(stats["gpt-4o-mini"].request_count, 2);
        assert_eq!(stats["claude-sonnet-4-6"].request_count, 1);
    }

    #[test]
    fn test_by_model_avg_cost() {
        let mut ledger = CostLedger::new();
        ledger.add(make_record("gpt-4o", 1_000_000, 0, 10)).unwrap();
        ledger.add(make_record("gpt-4o", 1_000_000, 0, 10)).unwrap();
        let stats = ledger.by_model();
        let s = &stats["gpt-4o"];
        assert!((s.avg_cost_per_request - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_by_model_token_sums() {
        let mut ledger = CostLedger::new();
        ledger.add(make_record("gpt-4o-mini", 100, 50, 10)).unwrap();
        ledger
            .add(make_record("gpt-4o-mini", 200, 100, 20))
            .unwrap();
        let stats = ledger.by_model();
        let s = &stats["gpt-4o-mini"];
        assert_eq!(s.total_input_tokens, 300);
        assert_eq!(s.total_output_tokens, 150);
    }

    #[test]
    fn test_last_n_returns_at_most_n() {
        let mut ledger = CostLedger::new();
        for _ in 0..20 {
            ledger.add(make_record("gpt-4o-mini", 100, 50, 10)).unwrap();
        }
        assert_eq!(ledger.last_n(5).len(), 5);
    }

    #[test]
    fn test_last_n_more_than_len_returns_all() {
        let mut ledger = CostLedger::new();
        ledger.add(make_record("gpt-4o-mini", 100, 50, 10)).unwrap();
        assert_eq!(ledger.last_n(100).len(), 1);
    }

    #[test]
    fn test_since_filters_by_timestamp() {
        let mut ledger = CostLedger::new();
        let mut old = make_record("gpt-4o-mini", 100, 50, 10);
        old.timestamp = Utc::now() - Duration::hours(2);
        ledger.add(old).unwrap();
        ledger.add(make_record("gpt-4o-mini", 100, 50, 10)).unwrap(); // now
        let cutoff = Utc::now() - Duration::minutes(30);
        let recent = ledger.since(cutoff);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_projected_monthly_zero_window() {
        assert_eq!(CostLedger::new().projected_monthly_usd(0), 0.0);
    }

    #[test]
    fn test_projected_monthly_empty_ledger() {
        assert_eq!(CostLedger::new().projected_monthly_usd(24), 0.0);
    }

    #[test]
    fn test_projected_monthly_math() {
        // Spend $1 right now in a 1-hour window.
        // Projection = ($1 / 1h) * 24h/day * 30 days = $720.
        let mut ledger = CostLedger::new();
        let mut rec = make_record("gpt-4o-mini", 0, 0, 0);
        rec.total_cost_usd = 1.0;
        ledger.add(rec).unwrap();
        let proj = ledger.projected_monthly_usd(1);
        assert!((proj - 720.0).abs() < 1e-6);
    }

    #[test]
    fn test_clear_empties_ledger() {
        let mut ledger = CostLedger::new();
        ledger.add(make_record("gpt-4o-mini", 100, 50, 10)).unwrap();
        ledger.clear();
        assert!(ledger.is_empty());
    }

    #[test]
    fn test_sparkline_data_len() {
        let mut ledger = CostLedger::new();
        for _ in 0..30 {
            ledger.add(make_record("gpt-4o-mini", 100, 50, 10)).unwrap();
        }
        assert_eq!(ledger.sparkline_data(10).len(), 10);
    }

    #[test]
    fn test_p99_latency_correct() {
        let mut ledger = CostLedger::new();
        for i in 1u64..=100 {
            let mut r = make_record("gpt-4o-mini", 100, 50, i);
            r.model = "gpt-4o-mini".into();
            ledger.add(r).unwrap();
        }
        let stats = ledger.by_model();
        // p99 of 1..=100 sorted: ceil(100*0.99)-1 = index 98 => value 99ms
        assert_eq!(stats["gpt-4o-mini"].p99_latency_ms, 99.0);
        assert!(stats["gpt-4o-mini"].p99_latency_ms < 100.0);
    }

    #[test]
    fn test_p99_single_record() {
        let mut ledger = CostLedger::new();
        ledger.add(make_record("gpt-4o-mini", 100, 50, 42)).unwrap();
        let stats = ledger.by_model();
        assert_eq!(stats["gpt-4o-mini"].p99_latency_ms, 42.0);
    }

    #[test]
    fn test_is_empty_on_new_ledger() {
        assert!(CostLedger::new().is_empty());
    }

    #[test]
    fn test_fractional_cost_stored_correctly() {
        // 1 token of gpt-4o-mini input = $0.15 / 1_000_000
        let rec = CostRecord::new("gpt-4o-mini", "openai", 1, 0, 0);
        assert!((rec.input_cost_usd - 0.15 / 1_000_000.0).abs() < 1e-15);
    }
}
