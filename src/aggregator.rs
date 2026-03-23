//! # Time-Bucketed Cost Aggregation
//!
//! Aggregates LLM token usage and cost into configurable time buckets
//! (minute, hour, day, week) with multi-dimensional rollup by model and
//! tenant.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::aggregator::{CostAggregator, BucketGranularity};
//!
//! let agg = CostAggregator::new();
//! agg.record(1_700_000_000, "gpt-4", "acme", 1_000, 500, 0.06);
//! agg.record(1_700_000_060, "gpt-4", "acme", 200, 100, 0.01);
//!
//! let buckets = agg.query(BucketGranularity::Hour, 0, u64::MAX, None, None);
//! assert_eq!(buckets.len(), 1);
//! assert_eq!(buckets[0].request_count, 2);
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// BucketGranularity
// ---------------------------------------------------------------------------

/// The time resolution of a cost bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BucketGranularity {
    /// One-minute buckets.
    Minute,
    /// One-hour buckets.
    Hour,
    /// One-day (24-hour) buckets.
    Day,
    /// One-week (7-day) buckets.
    Week,
}

impl BucketGranularity {
    /// Truncate a Unix timestamp (seconds) to the bucket boundary.
    ///
    /// ```
    /// use llm_cost_dashboard::aggregator::BucketGranularity;
    ///
    /// // 1700000065 is 65 seconds past a minute boundary.
    /// let ts = 1_700_000_065u64;
    /// assert_eq!(BucketGranularity::Minute.bucket_start(ts), ts - 65);
    /// ```
    pub fn bucket_start(self, ts: u64) -> u64 {
        match self {
            Self::Minute => ts - (ts % 60),
            Self::Hour => ts - (ts % 3_600),
            Self::Day => ts - (ts % 86_400),
            Self::Week => ts - (ts % 604_800),
        }
    }

    /// All four granularity levels, in ascending order of coarseness.
    pub fn all() -> [Self; 4] {
        [Self::Minute, Self::Hour, Self::Day, Self::Week]
    }
}

// ---------------------------------------------------------------------------
// BucketKey
// ---------------------------------------------------------------------------

/// Composite key identifying a unique bucket.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BucketKey {
    /// Unix timestamp (seconds) of the bucket start.
    pub period_start: u64,
    /// Model name (e.g. `"gpt-4"`).
    pub model: String,
    /// Tenant / customer identifier.
    pub tenant: String,
    /// The granularity this key belongs to.
    pub granularity: BucketGranularity,
}

// ---------------------------------------------------------------------------
// Bucket
// ---------------------------------------------------------------------------

/// Aggregated usage and cost for a single time bucket.
#[derive(Debug, Clone)]
pub struct Bucket {
    /// Unix timestamp (seconds) of the bucket start.
    pub period_start: u64,
    /// Model name.
    pub model: String,
    /// Tenant identifier.
    pub tenant: String,
    /// Total input tokens recorded in this bucket.
    pub tokens_in: u64,
    /// Total output tokens recorded in this bucket.
    pub tokens_out: u64,
    /// Total cost (USD) recorded in this bucket.
    pub cost: f64,
    /// Number of individual requests recorded.
    pub request_count: u64,
    /// Granularity of this bucket.
    pub granularity: BucketGranularity,
}

impl Bucket {
    fn new(period_start: u64, model: &str, tenant: &str, granularity: BucketGranularity) -> Self {
        Self {
            period_start,
            model: model.to_string(),
            tenant: tenant.to_string(),
            tokens_in: 0,
            tokens_out: 0,
            cost: 0.0,
            request_count: 0,
            granularity,
        }
    }

    fn add(&mut self, tokens_in: u64, tokens_out: u64, cost: f64) {
        self.tokens_in += tokens_in;
        self.tokens_out += tokens_out;
        self.cost += cost;
        self.request_count += 1;
    }
}

// ---------------------------------------------------------------------------
// CostAggregator
// ---------------------------------------------------------------------------

/// Thread-safe time-bucketed cost aggregator.
///
/// Records are stored in an in-process `HashMap` protected by a `Mutex`.  For
/// each call to [`record`](CostAggregator::record), buckets for **all four**
/// granularities are updated simultaneously.
pub struct CostAggregator {
    inner: Mutex<HashMap<BucketKey, Bucket>>,
}

impl Default for CostAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostAggregator {
    /// Create an empty aggregator.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Record a single request's usage across all granularity levels.
    ///
    /// - `ts` — Unix timestamp in seconds when the request completed
    /// - `model` — model identifier (e.g. `"claude-3-5-sonnet"`)
    /// - `tenant` — tenant / customer identifier
    /// - `tokens_in` — prompt token count
    /// - `tokens_out` — completion token count
    /// - `cost` — USD cost of this request
    pub fn record(
        &self,
        ts: u64,
        model: &str,
        tenant: &str,
        tokens_in: u64,
        tokens_out: u64,
        cost: f64,
    ) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        for gran in BucketGranularity::all() {
            let key = BucketKey {
                period_start: gran.bucket_start(ts),
                model: model.to_string(),
                tenant: tenant.to_string(),
                granularity: gran,
            };
            let entry = guard
                .entry(key.clone())
                .or_insert_with(|| Bucket::new(key.period_start, model, tenant, gran));
            entry.add(tokens_in, tokens_out, cost);
        }
    }

    /// Query buckets matching the given filters, sorted by `period_start` ascending.
    ///
    /// - `granularity` — which granularity level to query
    /// - `from_ts` / `to_ts` — inclusive Unix second range
    /// - `model` — optional exact model filter
    /// - `tenant` — optional exact tenant filter
    pub fn query(
        &self,
        granularity: BucketGranularity,
        from_ts: u64,
        to_ts: u64,
        model: Option<&str>,
        tenant: Option<&str>,
    ) -> Vec<Bucket> {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut result: Vec<Bucket> = guard
            .values()
            .filter(|b| {
                b.granularity == granularity
                    && b.period_start >= from_ts
                    && b.period_start <= to_ts
                    && model.map_or(true, |m| b.model == m)
                    && tenant.map_or(true, |t| b.tenant == t)
            })
            .cloned()
            .collect();
        result.sort_by_key(|b| b.period_start);
        result
    }

    /// Aggregate token totals and cost by model from a slice of buckets.
    ///
    /// Returns `HashMap<model, (total_tokens, total_cost)>`.
    pub fn rollup_by_model(&self, buckets: &[Bucket]) -> HashMap<String, (u64, f64)> {
        let mut map: HashMap<String, (u64, f64)> = HashMap::new();
        for b in buckets {
            let entry = map.entry(b.model.clone()).or_default();
            entry.0 += b.tokens_in + b.tokens_out;
            entry.1 += b.cost;
        }
        map
    }

    /// Aggregate token totals and cost by tenant from a slice of buckets.
    ///
    /// Returns `HashMap<tenant, (total_tokens, total_cost)>`.
    pub fn rollup_by_tenant(&self, buckets: &[Bucket]) -> HashMap<String, (u64, f64)> {
        let mut map: HashMap<String, (u64, f64)> = HashMap::new();
        for b in buckets {
            let entry = map.entry(b.tenant.clone()).or_default();
            entry.0 += b.tokens_in + b.tokens_out;
            entry.1 += b.cost;
        }
        map
    }

    /// Return the top-N models by total cost (descending) for the given window.
    pub fn top_n_models(
        &self,
        n: usize,
        granularity: BucketGranularity,
        from_ts: u64,
        to_ts: u64,
    ) -> Vec<(String, f64)> {
        let buckets = self.query(granularity, from_ts, to_ts, None, None);
        let rollup = self.rollup_by_model(&buckets);
        let mut pairs: Vec<(String, f64)> = rollup
            .into_iter()
            .map(|(model, (_tokens, cost))| (model, cost))
            .collect();
        // Sort descending by cost, then alphabetically for determinism.
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        pairs.truncate(n);
        pairs
    }

    /// Return the effective cost per 1 000 tokens for `model` in the window.
    ///
    /// Returns `None` if no data exists for that model in the window.
    pub fn cost_per_1k_tokens(
        &self,
        model: &str,
        granularity: BucketGranularity,
        from_ts: u64,
        to_ts: u64,
    ) -> Option<f64> {
        let buckets = self.query(granularity, from_ts, to_ts, Some(model), None);
        let (total_tokens, total_cost) =
            buckets
                .iter()
                .fold((0u64, 0.0f64), |(tok, cost), b| {
                    (tok + b.tokens_in + b.tokens_out, cost + b.cost)
                });
        if total_tokens == 0 {
            return None;
        }
        Some(total_cost / (total_tokens as f64) * 1_000.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_TS: u64 = 1_700_000_000; // 2023-11-14 22:13:20 UTC

    #[test]
    fn bucket_start_minute() {
        let ts = BASE_TS + 37; // 37 seconds past the minute
        let start = BucketGranularity::Minute.bucket_start(ts);
        assert_eq!(start, BASE_TS + 37 - (BASE_TS + 37) % 60);
    }

    #[test]
    fn bucket_start_hour() {
        let ts = 3_600 * 5 + 999; // 999 seconds into hour 5
        let start = BucketGranularity::Hour.bucket_start(ts);
        assert_eq!(start, 3_600 * 5);
    }

    #[test]
    fn bucket_start_day() {
        let ts = 86_400 * 3 + 7_200; // 2 hours into day 3
        let start = BucketGranularity::Day.bucket_start(ts);
        assert_eq!(start, 86_400 * 3);
    }

    #[test]
    fn bucket_start_week() {
        let ts = 604_800 * 2 + 3_000;
        let start = BucketGranularity::Week.bucket_start(ts);
        assert_eq!(start, 604_800 * 2);
    }

    #[test]
    fn record_and_query_single_bucket() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "gpt-4", "acme", 1_000, 500, 0.06);
        let buckets = agg.query(BucketGranularity::Hour, 0, u64::MAX, None, None);
        assert_eq!(buckets.len(), 1);
        let b = &buckets[0];
        assert_eq!(b.tokens_in, 1_000);
        assert_eq!(b.tokens_out, 500);
        assert!((b.cost - 0.06).abs() < 1e-10);
        assert_eq!(b.request_count, 1);
    }

    #[test]
    fn two_requests_same_bucket_accumulate() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "claude-3", "tenant-a", 100, 50, 0.01);
        agg.record(BASE_TS + 30, "claude-3", "tenant-a", 200, 80, 0.02);
        let buckets = agg.query(BucketGranularity::Minute, 0, u64::MAX, None, None);
        // Same minute bucket.
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].request_count, 2);
        assert_eq!(buckets[0].tokens_in, 300);
    }

    #[test]
    fn different_models_separate_buckets() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "gpt-4", "acme", 100, 50, 0.05);
        agg.record(BASE_TS, "claude-3", "acme", 200, 80, 0.03);
        let buckets = agg.query(BucketGranularity::Hour, 0, u64::MAX, None, None);
        assert_eq!(buckets.len(), 2);
    }

    #[test]
    fn query_filters_by_model() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "gpt-4", "acme", 100, 50, 0.05);
        agg.record(BASE_TS, "claude-3", "acme", 200, 80, 0.03);
        let buckets = agg.query(BucketGranularity::Hour, 0, u64::MAX, Some("gpt-4"), None);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].model, "gpt-4");
    }

    #[test]
    fn query_filters_by_tenant() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "gpt-4", "acme", 100, 50, 0.05);
        agg.record(BASE_TS, "gpt-4", "globex", 200, 80, 0.03);
        let buckets = agg.query(BucketGranularity::Hour, 0, u64::MAX, None, Some("globex"));
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].tenant, "globex");
    }

    #[test]
    fn query_respects_time_range() {
        let agg = CostAggregator::new();
        agg.record(1_000, "m", "t", 1, 1, 0.001);
        agg.record(100_000, "m", "t", 1, 1, 0.001);
        // Query only the first hour.
        let buckets = agg.query(BucketGranularity::Hour, 0, 3_599, None, None);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].period_start, 0);
    }

    #[test]
    fn empty_query_returns_empty_vec() {
        let agg = CostAggregator::new();
        let buckets = agg.query(BucketGranularity::Day, 0, u64::MAX, None, None);
        assert!(buckets.is_empty());
    }

    #[test]
    fn rollup_by_model() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "gpt-4", "a", 100, 50, 0.05);
        agg.record(BASE_TS + 3600, "gpt-4", "a", 200, 100, 0.10);
        agg.record(BASE_TS, "claude-3", "a", 300, 150, 0.06);
        let buckets = agg.query(BucketGranularity::Day, 0, u64::MAX, None, None);
        let rollup = agg.rollup_by_model(&buckets);

        let (gpt_tok, gpt_cost) = rollup["gpt-4"];
        assert_eq!(gpt_tok, (100 + 50) + (200 + 100));
        assert!((gpt_cost - 0.15).abs() < 1e-10);

        let (claude_tok, claude_cost) = rollup["claude-3"];
        assert_eq!(claude_tok, 300 + 150);
        assert!((claude_cost - 0.06).abs() < 1e-10);
    }

    #[test]
    fn rollup_by_tenant() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "m", "acme", 100, 50, 0.05);
        agg.record(BASE_TS, "m", "globex", 200, 80, 0.08);
        let buckets = agg.query(BucketGranularity::Hour, 0, u64::MAX, None, None);
        let rollup = agg.rollup_by_tenant(&buckets);
        assert!((rollup["acme"].1 - 0.05).abs() < 1e-10);
        assert!((rollup["globex"].1 - 0.08).abs() < 1e-10);
    }

    #[test]
    fn top_n_ordering() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "cheap", "t", 100, 50, 0.01);
        agg.record(BASE_TS, "expensive", "t", 100, 50, 1.00);
        agg.record(BASE_TS, "medium", "t", 100, 50, 0.50);

        let top = agg.top_n_models(2, BucketGranularity::Hour, 0, u64::MAX);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "expensive");
        assert_eq!(top[1].0, "medium");
    }

    #[test]
    fn top_n_more_than_available() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "only-model", "t", 100, 50, 0.50);
        let top = agg.top_n_models(10, BucketGranularity::Hour, 0, u64::MAX);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn cost_per_1k_tokens_calculation() {
        let agg = CostAggregator::new();
        // 1000 in + 1000 out = 2000 tokens, $2.00 cost → $1.00 per 1k tokens
        agg.record(BASE_TS, "pricey", "t", 1_000, 1_000, 2.00);
        let cpt = agg
            .cost_per_1k_tokens("pricey", BucketGranularity::Hour, 0, u64::MAX)
            .expect("should exist");
        assert!((cpt - 1.00).abs() < 1e-10, "expected 1.0, got {cpt}");
    }

    #[test]
    fn cost_per_1k_tokens_missing_model() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "model-a", "t", 100, 50, 0.05);
        assert!(agg
            .cost_per_1k_tokens("model-b", BucketGranularity::Hour, 0, u64::MAX)
            .is_none());
    }

    #[test]
    fn buckets_sorted_by_period_start() {
        let agg = CostAggregator::new();
        // Records in reverse chronological order.
        agg.record(BASE_TS + 7_200, "m", "t", 1, 1, 0.01);
        agg.record(BASE_TS, "m", "t", 1, 1, 0.01);
        agg.record(BASE_TS + 3_600, "m", "t", 1, 1, 0.01);
        let buckets = agg.query(BucketGranularity::Hour, 0, u64::MAX, None, None);
        assert!(buckets.windows(2).all(|w| w[0].period_start <= w[1].period_start));
    }

    #[test]
    fn all_granularities_populated_per_record() {
        let agg = CostAggregator::new();
        agg.record(BASE_TS, "m", "t", 100, 50, 0.05);
        for gran in BucketGranularity::all() {
            let buckets = agg.query(gran, 0, u64::MAX, None, None);
            assert_eq!(buckets.len(), 1, "expected bucket for {gran:?}");
        }
    }
}
