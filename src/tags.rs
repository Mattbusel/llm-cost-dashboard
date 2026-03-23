//! # Cost Attribution Tags
//!
//! Tag support for cost records.  A tag is a `key=value` pair that can be
//! attached to any [`TaggedRecord`] to enable attribution of cost to features,
//! environments, user tiers, and any other dimension.
//!
//! ## Tag format
//!
//! Tags are stored as `HashMap<String, String>`.  The canonical wire format is
//! `key=value` pairs; use [`parse_tags`] to convert a slice of `"key=value"`
//! strings into a map.
//!
//! ## Usage
//!
//! ```rust
//! use llm_cost_dashboard::tags::{parse_tags, TagIndex};
//! use llm_cost_dashboard::cost::CostRecord;
//!
//! let rec = CostRecord::new("gpt-4o-mini", "openai", 512, 256, 20);
//! let tags = parse_tags(&["feature=chat", "env=prod", "user_tier=premium"]);
//!
//! let mut index = TagIndex::new();
//! index.insert(rec, tags);
//!
//! let top = index.top_spend_by_tag("feature", 5);
//! for (value, cost) in &top {
//!     println!("{value}: ${cost:.4}");
//! }
//! ```

use std::collections::HashMap;

use crate::cost::CostRecord;

/// A parsed set of tags (key → value).
pub type Tags = HashMap<String, String>;

/// Parse a slice of `"key=value"` strings into a [`Tags`] map.
///
/// Entries that do not contain `=` are silently ignored.  If a key appears
/// multiple times, the last value wins.
pub fn parse_tags(raw: &[&str]) -> Tags {
    let mut map = Tags::new();
    for s in raw {
        if let Some(eq) = s.find('=') {
            let k = s[..eq].trim().to_owned();
            let v = s[eq + 1..].trim().to_owned();
            if !k.is_empty() {
                map.insert(k, v);
            }
        }
    }
    map
}

/// Parse tags from a JSON object's string fields, treating all string values as
/// potential tags.
///
/// Only top-level `key: "string_value"` entries are considered; nested objects
/// are skipped.  This is the entry point used when ingesting newline-delimited
/// JSON log records that carry metadata fields like `feature`, `env`, etc.
pub fn parse_tags_from_json(obj: &serde_json::Map<String, serde_json::Value>) -> Tags {
    let mut map = Tags::new();
    // Reserved fields that are NOT tags.
    const RESERVED: &[&str] = &[
        "model",
        "provider",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "request_id",
        "session_id",
        "timestamp",
        "id",
    ];
    for (k, v) in obj {
        if RESERVED.contains(&k.as_str()) {
            continue;
        }
        if let serde_json::Value::String(s) = v {
            map.insert(k.clone(), s.clone());
        }
    }
    map
}

/// A cost record paired with its attribution tags.
#[derive(Debug, Clone)]
pub struct TaggedRecord {
    /// The underlying cost record.
    pub record: CostRecord,
    /// Attribution tags for this record.
    pub tags: Tags,
}

impl TaggedRecord {
    /// Create a new tagged record.
    pub fn new(record: CostRecord, tags: Tags) -> Self {
        Self { record, tags }
    }

    /// Get the value of a tag by key.
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags.get(key).map(|s| s.as_str())
    }
}

/// An in-memory index of [`TaggedRecord`]s that supports top-N spend queries
/// grouped by any tag key.
#[derive(Debug, Default)]
pub struct TagIndex {
    records: Vec<TaggedRecord>,
}

impl TagIndex {
    /// Create an empty tag index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a cost record with its tags into the index.
    pub fn insert(&mut self, record: CostRecord, tags: Tags) {
        self.records.push(TaggedRecord::new(record, tags));
    }

    /// Total number of tagged records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Return a reference to all tagged records in insertion order.
    pub fn records(&self) -> &[TaggedRecord] {
        &self.records
    }

    /// Aggregate total spend grouped by the values of `tag_key`.
    ///
    /// Returns a map of `tag_value → total_cost_usd`.  Records that do not
    /// carry `tag_key` are grouped under the key `"(untagged)"`.
    pub fn spend_by_tag_value(&self, tag_key: &str) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for r in &self.records {
            let key = r
                .tags
                .get(tag_key)
                .map(|s| s.as_str())
                .unwrap_or("(untagged)")
                .to_owned();
            *map.entry(key).or_insert(0.0) += r.record.total_cost_usd;
        }
        map
    }

    /// Return the top `n` tag values for `tag_key` sorted by spend (largest first).
    ///
    /// Returns a `Vec<(tag_value, total_cost_usd)>`.
    pub fn top_spend_by_tag(&self, tag_key: &str, n: usize) -> Vec<(String, f64)> {
        let mut agg: Vec<(String, f64)> = self.spend_by_tag_value(tag_key).into_iter().collect();
        agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        agg.truncate(n);
        agg
    }

    /// Filter records by an exact tag match (`key=value`).
    pub fn filter_by_tag<'a>(&'a self, key: &str, value: &str) -> Vec<&'a TaggedRecord> {
        self.records
            .iter()
            .filter(|r| r.tags.get(key).map(|v| v == value).unwrap_or(false))
            .collect()
    }

    /// Total spend across all records.
    pub fn total_cost_usd(&self) -> f64 {
        self.records.iter().map(|r| r.record.total_cost_usd).sum()
    }

    /// Number of distinct values seen for `tag_key`.
    pub fn distinct_values(&self, tag_key: &str) -> Vec<String> {
        let mut vals: Vec<String> = self
            .spend_by_tag_value(tag_key)
            .into_keys()
            .collect();
        vals.sort();
        vals
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cost::CostRecord;

    fn rec(model: &str, inp: u64, out: u64) -> CostRecord {
        CostRecord::new(model, "test", inp, out, 10)
    }

    #[test]
    fn test_parse_tags_basic() {
        let tags = parse_tags(&["feature=chat", "env=prod"]);
        assert_eq!(tags.get("feature").map(|s| s.as_str()), Some("chat"));
        assert_eq!(tags.get("env").map(|s| s.as_str()), Some("prod"));
    }

    #[test]
    fn test_parse_tags_ignores_no_equals() {
        let tags = parse_tags(&["not_a_tag", "key=val"]);
        assert!(!tags.contains_key("not_a_tag"));
        assert!(tags.contains_key("key"));
    }

    #[test]
    fn test_parse_tags_empty_key_ignored() {
        let tags = parse_tags(&["=value"]);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_parse_tags_value_with_equals() {
        // Only the first `=` is the delimiter.
        let tags = parse_tags(&["url=http://example.com?a=1"]);
        assert_eq!(
            tags.get("url").map(|s| s.as_str()),
            Some("http://example.com?a=1")
        );
    }

    #[test]
    fn test_tag_index_insert_and_len() {
        let mut idx = TagIndex::new();
        idx.insert(rec("gpt-4o-mini", 100, 50), parse_tags(&["env=prod"]));
        idx.insert(rec("gpt-4o-mini", 200, 100), parse_tags(&["env=staging"]));
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_spend_by_tag_value() {
        let mut idx = TagIndex::new();
        idx.insert(rec("gpt-4o-mini", 1_000_000, 0), parse_tags(&["env=prod"]));
        idx.insert(rec("gpt-4o-mini", 1_000_000, 0), parse_tags(&["env=prod"]));
        idx.insert(rec("gpt-4o-mini", 1_000_000, 0), parse_tags(&["env=staging"]));
        let agg = idx.spend_by_tag_value("env");
        assert!(agg["prod"] > agg["staging"]);
    }

    #[test]
    fn test_top_spend_by_tag_sorted() {
        let mut idx = TagIndex::new();
        idx.insert(rec("gpt-4o-mini", 100, 0), parse_tags(&["feature=search"]));
        idx.insert(rec("gpt-4o-mini", 1_000_000, 0), parse_tags(&["feature=chat"]));
        let top = idx.top_spend_by_tag("feature", 5);
        assert_eq!(top[0].0, "chat");
    }

    #[test]
    fn test_top_spend_by_tag_truncated() {
        let mut idx = TagIndex::new();
        for i in 0..10 {
            idx.insert(
                rec("gpt-4o-mini", 100, 0),
                parse_tags(&[&format!("feature=f{i}")]),
            );
        }
        let top = idx.top_spend_by_tag("feature", 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_untagged_records_grouped() {
        let mut idx = TagIndex::new();
        idx.insert(rec("gpt-4o-mini", 100, 0), Tags::new());
        let agg = idx.spend_by_tag_value("env");
        assert!(agg.contains_key("(untagged)"));
    }

    #[test]
    fn test_filter_by_tag() {
        let mut idx = TagIndex::new();
        idx.insert(rec("gpt-4o-mini", 100, 0), parse_tags(&["env=prod"]));
        idx.insert(rec("gpt-4o-mini", 100, 0), parse_tags(&["env=staging"]));
        let filtered = idx.filter_by_tag("env", "prod");
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_distinct_values() {
        let mut idx = TagIndex::new();
        idx.insert(rec("gpt-4o-mini", 100, 0), parse_tags(&["env=prod"]));
        idx.insert(rec("gpt-4o-mini", 100, 0), parse_tags(&["env=staging"]));
        idx.insert(rec("gpt-4o-mini", 100, 0), parse_tags(&["env=prod"]));
        let vals = idx.distinct_values("env");
        // includes "(untagged)" only if there are untagged recs; here there are none
        assert!(vals.contains(&"prod".to_owned()));
        assert!(vals.contains(&"staging".to_owned()));
    }

    #[test]
    fn test_parse_tags_from_json() {
        let mut obj = serde_json::Map::new();
        obj.insert("model".into(), serde_json::Value::String("gpt-4o".into()));
        obj.insert("feature".into(), serde_json::Value::String("search".into()));
        obj.insert("input_tokens".into(), serde_json::Value::Number(100.into()));
        let tags = parse_tags_from_json(&obj);
        assert!(!tags.contains_key("model"));
        assert!(!tags.contains_key("input_tokens"));
        assert_eq!(tags.get("feature").map(|s| s.as_str()), Some("search"));
    }
}
