//! # FinOps Cost Tagging and Attribution
//!
//! Adds structured tag-based cost attribution so teams can track LLM spend
//! by project, team, cost centre, environment, and any custom dimension.
//!
//! ## Why tagging?
//!
//! A shared LLM deployment serving multiple teams produces a single cost
//! signal.  Without tagging it is impossible to answer:
//!
//! - Which project drove the 40% cost spike last Thursday?
//! - What fraction of the monthly bill belongs to the production environment?
//! - Which team is over their per-sprint LLM budget?
//!
//! Tagging solves this by attaching key-value labels to each [`CostRecord`]
//! as it is ingested and providing a roll-up engine that aggregates cost by
//! any tag dimension.
//!
//! ## Tag sources
//!
//! Tags can come from:
//!
//! 1. **NDJSON log fields** — extra fields in the log line are captured automatically.
//! 2. **Request metadata** — caller-supplied key-value pairs added at call time.
//! 3. **Inference rules** — a [`TagRule`] maps field patterns to tags.
//! 4. **Default tags** — always-present tags like `env=production`.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::tagging::{TagEngine, TagRule, TagMatch, TagSet};
//!
//! let mut engine = TagEngine::new();
//!
//! // Always tag with the environment.
//! engine.add_default_tag("env", "production");
//!
//! // Map model names to cost centres.
//! engine.add_rule(TagRule {
//!     field: "model".to_string(),
//!     pattern: TagMatch::Contains("claude".to_string()),
//!     tag_key: "provider".to_string(),
//!     tag_value: "anthropic".to_string(),
//! });
//! engine.add_rule(TagRule {
//!     field: "model".to_string(),
//!     pattern: TagMatch::Contains("gpt".to_string()),
//!     tag_key: "provider".to_string(),
//!     tag_value: "openai".to_string(),
//! });
//!
//! // Resolve tags for a log record.
//! let mut fields = std::collections::HashMap::new();
//! fields.insert("model".to_string(), "claude-sonnet-4-6".to_string());
//! fields.insert("project".to_string(), "recommendation-engine".to_string());
//!
//! let tags = engine.resolve(&fields);
//! assert_eq!(tags.get("provider"), Some(&"anthropic".to_string()));
//! assert_eq!(tags.get("env"), Some(&"production".to_string()));
//! assert_eq!(tags.get("project"), Some(&"recommendation-engine".to_string()));
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TagSet
// ---------------------------------------------------------------------------

/// An immutable snapshot of key-value tags attached to a cost record.
pub type TagSet = HashMap<String, String>;

// ---------------------------------------------------------------------------
// TagMatch
// ---------------------------------------------------------------------------

/// Pattern used by a [`TagRule`] to match a field value.
#[derive(Debug, Clone)]
pub enum TagMatch {
    /// Exact string equality.
    Exact(String),
    /// The field value contains this substring (case-insensitive).
    Contains(String),
    /// The field value starts with this prefix (case-insensitive).
    Prefix(String),
    /// Always matches — use for rules that apply unconditionally.
    Always,
}

impl TagMatch {
    /// Return `true` if `value` satisfies this match pattern.
    pub fn matches(&self, value: &str) -> bool {
        let lower = value.to_lowercase();
        match self {
            TagMatch::Exact(s) => value == s.as_str(),
            TagMatch::Contains(s) => lower.contains(s.to_lowercase().as_str()),
            TagMatch::Prefix(s) => lower.starts_with(s.to_lowercase().as_str()),
            TagMatch::Always => true,
        }
    }
}

// ---------------------------------------------------------------------------
// TagRule
// ---------------------------------------------------------------------------

/// A rule that derives a tag from a log field value.
///
/// When `pattern` matches the value of `field`, the output tag
/// `tag_key = tag_value` is added to the resolved [`TagSet`].
#[derive(Debug, Clone)]
pub struct TagRule {
    /// The log field to examine (e.g. `"model"`, `"provider"`, `"request_id"`).
    pub field: String,
    /// Pattern that the field value must satisfy.
    pub pattern: TagMatch,
    /// Key of the derived tag.
    pub tag_key: String,
    /// Value of the derived tag.
    pub tag_value: String,
}

// ---------------------------------------------------------------------------
// TagEngine
// ---------------------------------------------------------------------------

/// Resolves a full [`TagSet`] from raw log fields.
///
/// Apply in order:
/// 1. Copy all raw `fields` that are *also* in `passthrough_fields` directly
///    into the tag set (opt-in field pass-through).
/// 2. Apply default tags (always present).
/// 3. Evaluate each [`TagRule`] against the fields.
/// 4. Allow caller-supplied override tags to win on conflict.
///
/// See the [module documentation][self] for a full usage example.
#[derive(Debug, Clone, Default)]
pub struct TagEngine {
    default_tags: TagSet,
    rules: Vec<TagRule>,
    /// Fields whose values are passed through directly as tags.
    passthrough_fields: Vec<String>,
}

impl TagEngine {
    /// Create an empty engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a default tag that is present on every resolved [`TagSet`].
    pub fn add_default_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.default_tags.insert(key.into(), value.into());
    }

    /// Add a rule that derives a tag from a matching log field.
    pub fn add_rule(&mut self, rule: TagRule) {
        self.rules.push(rule);
    }

    /// Pass-through a field: if the field is present in the log record,
    /// copy it as a tag with the same key.
    ///
    /// Useful for fields like `project`, `team`, `cost_centre` that are
    /// already correct in the log and need no derivation.
    pub fn add_passthrough(&mut self, field: impl Into<String>) {
        self.passthrough_fields.push(field.into());
    }

    /// Resolve a full [`TagSet`] from a raw field map.
    ///
    /// `overrides` are applied last and win on conflict with default tags and
    /// rule-derived tags.
    pub fn resolve(&self, fields: &HashMap<String, String>) -> TagSet {
        let mut tags = self.default_tags.clone();

        // 1. Passthrough fields.
        for key in &self.passthrough_fields {
            if let Some(value) = fields.get(key) {
                tags.insert(key.clone(), value.clone());
            }
        }

        // 2. Rule-derived tags.
        for rule in &self.rules {
            if let Some(field_value) = fields.get(&rule.field) {
                if rule.pattern.matches(field_value) {
                    tags.insert(rule.tag_key.clone(), rule.tag_value.clone());
                }
            }
        }

        tags
    }

    /// Resolve with caller-supplied override tags that win on conflict.
    pub fn resolve_with_overrides(
        &self,
        fields: &HashMap<String, String>,
        overrides: &TagSet,
    ) -> TagSet {
        let mut tags = self.resolve(fields);
        for (k, v) in overrides {
            tags.insert(k.clone(), v.clone());
        }
        tags
    }
}

// ---------------------------------------------------------------------------
// CostByTag
// ---------------------------------------------------------------------------

/// Aggregates cost records by a single tag dimension.
#[derive(Debug, Default, Clone)]
pub struct CostByTag {
    /// Total cost in USD keyed by tag value.
    pub totals: HashMap<String, f64>,
    /// Request count keyed by tag value.
    pub counts: HashMap<String, u64>,
}

impl CostByTag {
    /// Record one cost entry against a tag value.
    pub fn record(&mut self, tag_value: impl Into<String>, cost_usd: f64) {
        let key = tag_value.into();
        *self.totals.entry(key.clone()).or_insert(0.0) += cost_usd;
        *self.counts.entry(key).or_insert(0) += 1;
    }

    /// Average cost per request for a given tag value.  Returns `None` if
    /// no records exist for that value.
    pub fn avg_cost(&self, tag_value: &str) -> Option<f64> {
        let total = self.totals.get(tag_value)?;
        let count = self.counts.get(tag_value)?;
        if *count == 0 {
            None
        } else {
            Some(total / *count as f64)
        }
    }

    /// Tag value with the highest total cost.
    pub fn top_spender(&self) -> Option<(&str, f64)> {
        self.totals
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, v)| (k.as_str(), *v))
    }

    /// Sorted list of (tag_value, total_cost_usd) descending by cost.
    pub fn ranked(&self) -> Vec<(String, f64)> {
        let mut v: Vec<(String, f64)> = self.totals
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v
    }
}

// ---------------------------------------------------------------------------
// TaggedLedger
// ---------------------------------------------------------------------------

/// An append-only ledger of tagged cost entries.
///
/// Each entry stores the cost, timestamp, and full [`TagSet`].  Supports
/// slicing by any tag dimension for reporting.
#[derive(Debug, Default, Clone)]
pub struct TaggedLedger {
    entries: Vec<TaggedEntry>,
}

/// A single entry in the [`TaggedLedger`].
#[derive(Debug, Clone)]
pub struct TaggedEntry {
    /// Cost in USD for this request.
    pub cost_usd: f64,
    /// Unix timestamp in seconds.
    pub timestamp_s: i64,
    /// Tags attached to this entry.
    pub tags: TagSet,
}

impl TaggedLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one entry.
    pub fn add(&mut self, cost_usd: f64, timestamp_s: i64, tags: TagSet) {
        self.entries.push(TaggedEntry { cost_usd, timestamp_s, tags });
    }

    /// Aggregate all entries by a single tag dimension.
    pub fn by_tag(&self, tag_key: &str) -> CostByTag {
        let mut agg = CostByTag::default();
        for entry in &self.entries {
            let value = entry.tags.get(tag_key).cloned().unwrap_or_else(|| "untagged".into());
            agg.record(value, entry.cost_usd);
        }
        agg
    }

    /// Total cost across all entries.
    pub fn total_cost_usd(&self) -> f64 {
        self.entries.iter().map(|e| e.cost_usd).sum()
    }

    /// Filter entries to those matching `tag_key = tag_value`.
    pub fn filter_by_tag(&self, tag_key: &str, tag_value: &str) -> Vec<&TaggedEntry> {
        self.entries
            .iter()
            .filter(|e| e.tags.get(tag_key).map(|v| v == tag_value).unwrap_or(false))
            .collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the ledger contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> TagEngine {
        let mut e = TagEngine::new();
        e.add_default_tag("env", "test");
        e.add_passthrough("project");
        e.add_rule(TagRule {
            field: "model".to_string(),
            pattern: TagMatch::Contains("claude".to_string()),
            tag_key: "provider".to_string(),
            tag_value: "anthropic".to_string(),
        });
        e
    }

    #[test]
    fn default_tags_always_present() {
        let e = engine();
        let tags = e.resolve(&HashMap::new());
        assert_eq!(tags.get("env").map(|s| s.as_str()), Some("test"));
    }

    #[test]
    fn passthrough_copies_field() {
        let e = engine();
        let mut fields = HashMap::new();
        fields.insert("project".into(), "billing".into());
        let tags = e.resolve(&fields);
        assert_eq!(tags.get("project").map(|s| s.as_str()), Some("billing"));
    }

    #[test]
    fn rule_derives_tag() {
        let e = engine();
        let mut fields = HashMap::new();
        fields.insert("model".into(), "claude-sonnet-4-6".into());
        let tags = e.resolve(&fields);
        assert_eq!(tags.get("provider").map(|s| s.as_str()), Some("anthropic"));
    }

    #[test]
    fn overrides_win_on_conflict() {
        let e = engine();
        let fields = HashMap::new();
        let mut overrides = HashMap::new();
        overrides.insert("env".into(), "production".into());
        let tags = e.resolve_with_overrides(&fields, &overrides);
        assert_eq!(tags.get("env").map(|s| s.as_str()), Some("production"));
    }

    #[test]
    fn cost_by_tag_aggregates_correctly() {
        let mut ledger = TaggedLedger::new();
        let mut t1 = HashMap::new();
        t1.insert("team".into(), "search".into());
        ledger.add(0.10, 0, t1.clone());
        ledger.add(0.20, 1, t1.clone());
        let mut t2 = HashMap::new();
        t2.insert("team".into(), "billing".into());
        ledger.add(0.05, 2, t2);

        let by_team = ledger.by_tag("team");
        assert!((by_team.totals["search"] - 0.30).abs() < 1e-9);
        assert!((by_team.totals["billing"] - 0.05).abs() < 1e-9);
        let top = by_team.top_spender();
        assert_eq!(top.map(|(k, _)| k), Some("search"));
    }

    #[test]
    fn filter_by_tag_returns_matching() {
        let mut ledger = TaggedLedger::new();
        let mut t = HashMap::new();
        t.insert("env".into(), "production".into());
        ledger.add(1.0, 0, t);
        let mut t2 = HashMap::new();
        t2.insert("env".into(), "staging".into());
        ledger.add(2.0, 1, t2);

        let prod = ledger.filter_by_tag("env", "production");
        assert_eq!(prod.len(), 1);
        assert!((prod[0].cost_usd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn total_cost_sums_all_entries() {
        let mut ledger = TaggedLedger::new();
        ledger.add(0.5, 0, HashMap::new());
        ledger.add(1.5, 1, HashMap::new());
        assert!((ledger.total_cost_usd() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn ranked_sorted_descending() {
        let mut agg = CostByTag::default();
        agg.record("a", 0.5);
        agg.record("b", 2.0);
        agg.record("c", 0.1);
        let ranked = agg.ranked();
        assert_eq!(ranked[0].0, "b");
        assert_eq!(ranked[2].0, "c");
    }
}

// ===========================================================================
// Cost Allocation Tagging — TagStore, TaggedRequest, TagReport
// ===========================================================================

use chrono::{DateTime, Utc};

/// An arbitrary key/value tag for cost attribution.
#[derive(Debug, Clone, PartialEq)]
pub struct CostTag {
    /// Tag key (e.g. `"project"`, `"team"`, `"env"`).
    pub key: String,
    /// Tag value (e.g. `"recommendation-engine"`, `"platform"`, `"production"`).
    pub value: String,
}

impl CostTag {
    /// Create a new `CostTag`.
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// A single tagged inference request with cost and token information.
#[derive(Debug, Clone)]
pub struct TaggedRequest {
    /// Unique numeric request identifier.
    pub request_id: u64,
    /// Model used for this request (e.g. `"gpt-4o-mini"`).
    pub model_id: String,
    /// Total cost in USD.
    pub cost_usd: f64,
    /// Number of input (prompt) tokens.
    pub tokens_in: u32,
    /// Number of output (completion) tokens.
    pub tokens_out: u32,
    /// Arbitrary k/v tags attached to this request.
    pub tags: Vec<CostTag>,
    /// Wall-clock time of the request.
    pub timestamp: DateTime<Utc>,
}

impl TaggedRequest {
    /// Look up the value of the first tag with `key`.
    pub fn tag_value(&self, key: &str) -> Option<&str> {
        self.tags
            .iter()
            .find(|t| t.key == key)
            .map(|t| t.value.as_str())
    }
}

/// Filter criteria for [`TagStore::query`].
#[derive(Debug, Clone, Default)]
pub struct TagFilter {
    /// If set, only requests whose tags contain a tag with this key are returned.
    pub key: Option<String>,
    /// If set, only requests whose tags contain `key=value` are returned.
    /// Ignored unless `key` is also set.
    pub value: Option<String>,
    /// Inclusive lower bound on `timestamp`.
    pub since: Option<DateTime<Utc>>,
    /// Inclusive upper bound on `timestamp`.
    pub until: Option<DateTime<Utc>>,
}

/// Aggregated statistics for a group of requests sharing a tag value.
#[derive(Debug, Clone, Default)]
pub struct GroupStats {
    /// Sum of `cost_usd` across all requests in the group.
    pub total_cost_usd: f64,
    /// Sum of `tokens_in + tokens_out` across all requests in the group.
    pub total_tokens: u64,
    /// Number of requests in the group.
    pub request_count: u64,
    /// `total_cost_usd / request_count`, or `0.0` for empty groups.
    pub avg_cost_usd: f64,
}

/// A top-10-by-cost report for a given tag dimension.
#[derive(Debug, Clone)]
pub struct TagReport {
    /// The tag key this report was generated for.
    pub tag_key: String,
    /// Top 10 `(tag_value, GroupStats)` pairs, sorted by `total_cost_usd` descending.
    pub top_groups: Vec<(String, GroupStats)>,
}

impl TagReport {
    /// Generate a report for `tag_key` from `store`, returning the top 10
    /// tag values by total cost.
    pub fn generate(store: &TagStore, tag_key: &str) -> Self {
        let groups = store.group_by(tag_key);
        let mut sorted: Vec<(String, GroupStats)> = groups.into_iter().collect();
        sorted.sort_by(|a, b| {
            b.1.total_cost_usd
                .partial_cmp(&a.1.total_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(10);
        TagReport {
            tag_key: tag_key.to_string(),
            top_groups: sorted,
        }
    }
}

/// Append-only store of [`TaggedRequest`]s with indexed tag lookup.
///
/// Backed by a `Vec` for append-only storage and a
/// `HashMap<String, Vec<usize>>` index for fast tag-key lookups.
#[derive(Debug, Default)]
pub struct TagStore {
    requests: Vec<TaggedRequest>,
    /// Maps `tag_key` -> list of request indices that have that key.
    index: HashMap<String, Vec<usize>>,
}

impl TagStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a request to the store and update the index.
    pub fn push(&mut self, req: TaggedRequest) {
        let idx = self.requests.len();
        for tag in &req.tags {
            self.index
                .entry(tag.key.clone())
                .or_default()
                .push(idx);
        }
        self.requests.push(req);
    }

    /// Return all requests in insertion order.
    pub fn all(&self) -> &[TaggedRequest] {
        &self.requests
    }

    /// Number of requests in the store.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// True if the store contains no requests.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Query requests matching `filter`.
    ///
    /// All provided criteria are ANDed together.
    pub fn query(&self, filter: &TagFilter) -> Vec<&TaggedRequest> {
        // Start with candidate indices.
        let candidates: Box<dyn Iterator<Item = usize>> = if let Some(key) = &filter.key {
            match self.index.get(key) {
                Some(indices) => Box::new(indices.iter().copied()),
                None => return Vec::new(),
            }
        } else {
            Box::new(0..self.requests.len())
        };

        candidates
            .filter_map(|i| self.requests.get(i))
            .filter(|r| {
                // value filter
                if let (Some(key), Some(val)) = (&filter.key, &filter.value) {
                    if !r.tags.iter().any(|t| &t.key == key && &t.value == val) {
                        return false;
                    }
                }
                // since filter
                if let Some(since) = filter.since {
                    if r.timestamp < since {
                        return false;
                    }
                }
                // until filter
                if let Some(until) = filter.until {
                    if r.timestamp > until {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Group requests by a tag key and aggregate cost/token statistics.
    ///
    /// Requests that do not have the specified tag key are grouped under
    /// the `"(untagged)"` key.
    pub fn group_by(&self, tag_key: &str) -> HashMap<String, GroupStats> {
        let mut groups: HashMap<String, GroupStats> = HashMap::new();

        for req in &self.requests {
            let group_key = req
                .tag_value(tag_key)
                .unwrap_or("(untagged)")
                .to_string();

            let g = groups.entry(group_key).or_default();
            g.total_cost_usd += req.cost_usd;
            g.total_tokens += u64::from(req.tokens_in) + u64::from(req.tokens_out);
            g.request_count += 1;
        }

        // Compute averages.
        for g in groups.values_mut() {
            if g.request_count > 0 {
                g.avg_cost_usd = g.total_cost_usd / g.request_count as f64;
            }
        }

        groups
    }
}

// ── TagStore tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tag_store_tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).unwrap()
    }

    fn req(id: u64, model: &str, cost: f64, tags: Vec<CostTag>) -> TaggedRequest {
        TaggedRequest {
            request_id: id,
            model_id: model.to_string(),
            cost_usd: cost,
            tokens_in: 100,
            tokens_out: 50,
            tags,
            timestamp: ts(1_700_000_000 + id as i64),
        }
    }

    fn tag(k: &str, v: &str) -> CostTag {
        CostTag::new(k, v)
    }

    #[test]
    fn test_push_and_len() {
        let mut store = TagStore::new();
        store.push(req(1, "gpt-4o", 0.01, vec![]));
        store.push(req(2, "claude-3", 0.02, vec![]));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_is_empty() {
        let store = TagStore::new();
        assert!(store.is_empty());
    }

    #[test]
    fn test_query_no_filter_returns_all() {
        let mut store = TagStore::new();
        store.push(req(1, "a", 0.01, vec![]));
        store.push(req(2, "b", 0.02, vec![]));
        let results = store.query(&TagFilter::default());
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_key() {
        let mut store = TagStore::new();
        store.push(req(1, "a", 0.01, vec![tag("env", "prod")]));
        store.push(req(2, "b", 0.02, vec![tag("team", "search")]));
        let filter = TagFilter { key: Some("env".to_string()), ..Default::default() };
        let results = store.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, 1);
    }

    #[test]
    fn test_query_by_key_and_value() {
        let mut store = TagStore::new();
        store.push(req(1, "a", 0.01, vec![tag("env", "prod")]));
        store.push(req(2, "b", 0.02, vec![tag("env", "staging")]));
        let filter = TagFilter {
            key: Some("env".to_string()),
            value: Some("prod".to_string()),
            ..Default::default()
        };
        let results = store.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, 1);
    }

    #[test]
    fn test_query_since() {
        let mut store = TagStore::new();
        let mut r1 = req(1, "a", 0.01, vec![]);
        r1.timestamp = ts(1000);
        let mut r2 = req(2, "b", 0.02, vec![]);
        r2.timestamp = ts(2000);
        store.push(r1);
        store.push(r2);
        let filter = TagFilter { since: Some(ts(1500)), ..Default::default() };
        let results = store.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, 2);
    }

    #[test]
    fn test_query_until() {
        let mut store = TagStore::new();
        let mut r1 = req(1, "a", 0.01, vec![]);
        r1.timestamp = ts(1000);
        let mut r2 = req(2, "b", 0.02, vec![]);
        r2.timestamp = ts(2000);
        store.push(r1);
        store.push(r2);
        let filter = TagFilter { until: Some(ts(1500)), ..Default::default() };
        let results = store.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, 1);
    }

    #[test]
    fn test_group_by_tag() {
        let mut store = TagStore::new();
        store.push(req(1, "a", 0.10, vec![tag("team", "search")]));
        store.push(req(2, "a", 0.20, vec![tag("team", "search")]));
        store.push(req(3, "a", 0.05, vec![tag("team", "billing")]));
        let groups = store.group_by("team");
        let search = &groups["search"];
        assert!((search.total_cost_usd - 0.30).abs() < 1e-9);
        assert_eq!(search.request_count, 2);
        let billing = &groups["billing"];
        assert!((billing.total_cost_usd - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_group_by_untagged() {
        let mut store = TagStore::new();
        store.push(req(1, "a", 0.10, vec![])); // no "team" tag
        let groups = store.group_by("team");
        assert!(groups.contains_key("(untagged)"));
    }

    #[test]
    fn test_group_by_avg_cost() {
        let mut store = TagStore::new();
        store.push(req(1, "a", 0.10, vec![tag("env", "prod")]));
        store.push(req(2, "a", 0.20, vec![tag("env", "prod")]));
        let groups = store.group_by("env");
        let prod = &groups["prod"];
        assert!((prod.avg_cost_usd - 0.15).abs() < 1e-9);
    }

    #[test]
    fn test_tag_report_top_10() {
        let mut store = TagStore::new();
        for i in 0..15u64 {
            store.push(req(i, "a", i as f64 * 0.1, vec![tag("proj", &format!("p{i}"))]));
        }
        let report = TagReport::generate(&store, "proj");
        assert_eq!(report.top_groups.len(), 10);
        // First entry should have the highest cost.
        assert!(
            report.top_groups[0].1.total_cost_usd >= report.top_groups[1].1.total_cost_usd
        );
    }

    #[test]
    fn test_tag_report_tag_key() {
        let store = TagStore::new();
        let report = TagReport::generate(&store, "team");
        assert_eq!(report.tag_key, "team");
    }

    #[test]
    fn test_query_missing_key_returns_empty() {
        let mut store = TagStore::new();
        store.push(req(1, "a", 0.01, vec![tag("env", "prod")]));
        let filter = TagFilter {
            key: Some("nonexistent".to_string()),
            ..Default::default()
        };
        let results = store.query(&filter);
        assert!(results.is_empty());
    }

    #[test]
    fn test_cost_tag_new() {
        let t = CostTag::new("k", "v");
        assert_eq!(t.key, "k");
        assert_eq!(t.value, "v");
    }

    #[test]
    fn test_tagged_request_tag_value() {
        let r = req(1, "a", 0.01, vec![tag("env", "prod"), tag("team", "search")]);
        assert_eq!(r.tag_value("env"), Some("prod"));
        assert_eq!(r.tag_value("missing"), None);
    }

    #[test]
    fn test_total_tokens_in_group() {
        let mut store = TagStore::new();
        let mut r = req(1, "a", 0.01, vec![tag("env", "prod")]);
        r.tokens_in = 200;
        r.tokens_out = 100;
        store.push(r);
        let groups = store.group_by("env");
        assert_eq!(groups["prod"].total_tokens, 300);
    }
}
