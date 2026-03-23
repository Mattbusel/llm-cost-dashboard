//! # Team and Project Cost Allocation
//!
//! Tracks LLM spend broken down by team, project, and user.
//! Supports budget assignment, cost-center reporting, and
//! chargeback/showback workflows.
//!
//! Allocation rules (applied in order):
//! 1. Exact match: `session_id` == configured pattern
//! 2. Prefix match: `session_id` starts with prefix
//! 3. Tag match: request metadata contains a tag
//! 4. Default: assigned to `"unallocated"` bucket

use std::collections::HashMap;

// ── Rule ─────────────────────────────────────────────────────────────────────

/// A single allocation rule that maps incoming requests to a team/project bucket.
///
/// Rules are evaluated in insertion order; the first match wins.
#[derive(Debug, Clone)]
pub struct AllocationRule {
    /// Unique identifier for this rule (e.g. `"eng-infra-prefix"`).
    pub rule_id: String,
    /// Destination team name (e.g. `"engineering"`).
    pub team: String,
    /// Destination project name (e.g. `"infra"`).
    pub project: String,
    /// Session ID prefix that triggers this rule.
    ///
    /// A rule with `session_prefix = Some("eng-")` matches any session whose
    /// ID starts with `"eng-"`.
    pub session_prefix: Option<String>,
    /// Metadata tag key=value pair that triggers this rule.
    ///
    /// If present, the rule fires when `tags[key] == value`.
    pub tag_match: Option<(String, String)>,
    /// Optional budget ceiling in USD for this allocation bucket.
    ///
    /// When set, [`AllocationBucket::is_over_budget`] and
    /// [`AllocationBucket::budget_utilization_pct`] become meaningful.
    pub budget_usd: Option<f64>,
}

// ── Bucket ────────────────────────────────────────────────────────────────────

/// Accumulated cost data for a single team/project pair.
#[derive(Debug, Clone, Default)]
pub struct AllocationBucket {
    /// Team name.
    pub team: String,
    /// Project name.
    pub project: String,
    /// Total USD cost accumulated in this bucket.
    pub total_cost_usd: f64,
    /// Number of cost events recorded in this bucket.
    pub request_count: u64,
    /// Optional budget ceiling in USD (copied from the matching rule).
    pub budget_usd: Option<f64>,
    /// Per-model cost breakdown: `model_id -> total USD`.
    pub models_used: HashMap<String, f64>,
}

impl AllocationBucket {
    /// Percentage of the budget consumed (`0.0`..`100.0+`).
    ///
    /// Returns `None` when no budget is configured.
    pub fn budget_utilization_pct(&self) -> Option<f64> {
        self.budget_usd
            .filter(|&b| b > 0.0)
            .map(|b| (self.total_cost_usd / b) * 100.0)
    }

    /// `true` when `total_cost_usd` exceeds the configured budget.
    ///
    /// Always `false` when no budget is configured.
    pub fn is_over_budget(&self) -> bool {
        self.budget_usd
            .map(|b| self.total_cost_usd > b)
            .unwrap_or(false)
    }

    /// Return the model with the highest accumulated cost in this bucket.
    ///
    /// Returns `None` when no cost has been recorded.
    pub fn top_model(&self) -> Option<(&str, f64)> {
        self.models_used
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(m, &c)| (m.as_str(), c))
    }
}

// ── Allocator ────────────────────────────────────────────────────────────────

/// Rule-based cost allocator that routes LLM spend to team/project buckets.
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use llm_cost_dashboard::allocation::{AllocationRule, CostAllocator};
///
/// let mut allocator = CostAllocator::new();
/// allocator.add_rule(AllocationRule {
///     rule_id: "eng-rule".into(),
///     team: "engineering".into(),
///     project: "backend".into(),
///     session_prefix: Some("eng-".into()),
///     tag_match: None,
///     budget_usd: Some(100.0),
/// });
///
/// let tags = HashMap::new();
/// allocator.record("eng-session-42", "gpt-4o-mini", 0.05, &tags);
///
/// let bucket = allocator.bucket("engineering", "backend").unwrap();
/// assert!((bucket.total_cost_usd - 0.05).abs() < 1e-9);
/// ```
pub struct CostAllocator {
    rules: Vec<AllocationRule>,
    /// Keyed by `"team/project"`.
    buckets: HashMap<String, AllocationBucket>,
}

impl Default for CostAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostAllocator {
    /// Create an empty allocator with no rules.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            buckets: HashMap::new(),
        }
    }

    /// Append a rule.  Rules are evaluated in insertion order.
    pub fn add_rule(&mut self, rule: AllocationRule) {
        self.rules.push(rule);
    }

    /// Record a cost event and allocate it to the matching bucket.
    ///
    /// Matching priority:
    /// 1. First rule whose `session_prefix` is a prefix of `session_id`.
    /// 2. First rule whose `tag_match` key/value is present in `tags`.
    /// 3. Default `"unallocated/unallocated"` bucket.
    pub fn record(
        &mut self,
        session_id: &str,
        model: &str,
        cost_usd: f64,
        tags: &HashMap<String, String>,
    ) {
        let (team, project, budget) = match self.match_rule(session_id, tags) {
            Some(rule) => (rule.team.clone(), rule.project.clone(), rule.budget_usd),
            None => ("unallocated".to_string(), "unallocated".to_string(), None),
        };

        let key = format!("{team}/{project}");
        let bucket = self.buckets.entry(key).or_insert_with(|| AllocationBucket {
            team: team.clone(),
            project: project.clone(),
            budget_usd: budget,
            ..Default::default()
        });

        // Keep the budget value in sync if it was set later or overridden.
        if bucket.budget_usd.is_none() && budget.is_some() {
            bucket.budget_usd = budget;
        }

        bucket.total_cost_usd += cost_usd;
        bucket.request_count += 1;
        *bucket.models_used.entry(model.to_string()).or_insert(0.0) += cost_usd;
    }

    /// Retrieve a specific bucket by team and project name.
    ///
    /// Returns `None` if no cost has been recorded to that bucket yet.
    pub fn bucket(&self, team: &str, project: &str) -> Option<&AllocationBucket> {
        self.buckets.get(&format!("{team}/{project}"))
    }

    /// All buckets sorted by total cost, highest first.
    pub fn all_buckets_ranked(&self) -> Vec<&AllocationBucket> {
        let mut v: Vec<&AllocationBucket> = self.buckets.values().collect();
        v.sort_by(|a, b| {
            b.total_cost_usd
                .partial_cmp(&a.total_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }

    /// Generate a chargeback CSV report.
    ///
    /// Columns: `team,project,total_cost_usd,request_count,budget_usd,utilization_pct,over_budget,top_model`
    pub fn chargeback_csv(&self) -> String {
        let mut lines = vec![
            "team,project,total_cost_usd,request_count,budget_usd,utilization_pct,over_budget,top_model"
                .to_string(),
        ];
        for b in self.all_buckets_ranked() {
            let budget = b
                .budget_usd
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "".to_string());
            let util = b
                .budget_utilization_pct()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "".to_string());
            let over = if b.budget_usd.is_some() {
                b.is_over_budget().to_string()
            } else {
                "".to_string()
            };
            let top = b
                .top_model()
                .map(|(m, _)| m.to_string())
                .unwrap_or_else(|| "".to_string());
            lines.push(format!(
                "{},{},{:.6},{},{},{},{},{}",
                b.team,
                b.project,
                b.total_cost_usd,
                b.request_count,
                budget,
                util,
                over,
                top,
            ));
        }
        lines.join("\n")
    }

    /// Generate a human-readable showback summary (one line per bucket).
    ///
    /// Unlike chargeback, showback is informational only — no financial action
    /// is expected from recipients.
    pub fn showback_summary(&self) -> Vec<String> {
        self.all_buckets_ranked()
            .iter()
            .map(|b| {
                let budget_info = match b.budget_usd {
                    Some(bud) => {
                        let pct = b.budget_utilization_pct().unwrap_or(0.0);
                        format!(" | budget ${bud:.2} ({pct:.1}% used)")
                    }
                    None => " | no budget set".to_string(),
                };
                let top = b
                    .top_model()
                    .map(|(m, c)| format!(" | top model: {m} (${c:.4})"))
                    .unwrap_or_default();
                format!(
                    "[{}/{}] ${:.4} over {} requests{}{}",
                    b.team, b.project, b.total_cost_usd, b.request_count, budget_info, top
                )
            })
            .collect()
    }

    /// Return all buckets whose cost exceeds their configured budget.
    pub fn over_budget_buckets(&self) -> Vec<&AllocationBucket> {
        self.buckets.values().filter(|b| b.is_over_budget()).collect()
    }

    // ── private ────────────────────────────────────────────────────────────

    /// Find the first matching rule for the given session and tags.
    fn match_rule<'a>(
        &'a self,
        session_id: &str,
        tags: &HashMap<String, String>,
    ) -> Option<&'a AllocationRule> {
        // Pass 1: prefix match on session_id (highest priority)
        for rule in &self.rules {
            if let Some(prefix) = &rule.session_prefix {
                if session_id.starts_with(prefix.as_str()) {
                    return Some(rule);
                }
            }
        }
        // Pass 2: tag match
        for rule in &self.rules {
            if let Some((k, v)) = &rule.tag_match {
                if tags.get(k).map(|tv| tv == v).unwrap_or(false) {
                    return Some(rule);
                }
            }
        }
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn empty_tags() -> HashMap<String, String> {
        HashMap::new()
    }

    fn make_rule(
        id: &str,
        team: &str,
        project: &str,
        prefix: Option<&str>,
        tag: Option<(&str, &str)>,
        budget: Option<f64>,
    ) -> AllocationRule {
        AllocationRule {
            rule_id: id.to_string(),
            team: team.to_string(),
            project: project.to_string(),
            session_prefix: prefix.map(|s| s.to_string()),
            tag_match: tag.map(|(k, v)| (k.to_string(), v.to_string())),
            budget_usd: budget,
        }
    }

    // 1. New allocator has no buckets.
    #[test]
    fn test_new_allocator_empty() {
        let a = CostAllocator::new();
        assert!(a.all_buckets_ranked().is_empty());
    }

    // 2. Prefix rule routes session correctly.
    #[test]
    fn test_prefix_rule_matches() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "eng", "backend", Some("eng-"), None, None));
        a.record("eng-session-1", "gpt-4o-mini", 1.0, &empty_tags());
        let b = a.bucket("eng", "backend").unwrap();
        assert!((b.total_cost_usd - 1.0).abs() < 1e-9);
    }

    // 3. Non-matching session goes to unallocated.
    #[test]
    fn test_unallocated_default() {
        let mut a = CostAllocator::new();
        a.record("mystery-session", "gpt-4o", 2.5, &empty_tags());
        let b = a.bucket("unallocated", "unallocated").unwrap();
        assert!((b.total_cost_usd - 2.5).abs() < 1e-9);
    }

    // 4. Tag rule matches when session prefix does not.
    #[test]
    fn test_tag_rule_matches() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule(
            "r1",
            "ml",
            "training",
            None,
            Some(("env", "prod")),
            None,
        ));
        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "prod".to_string());
        a.record("session-abc", "claude-sonnet-4-6", 0.75, &tags);
        let b = a.bucket("ml", "training").unwrap();
        assert!((b.total_cost_usd - 0.75).abs() < 1e-9);
    }

    // 5. Prefix rule takes priority over tag rule.
    #[test]
    fn test_prefix_beats_tag() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule(
            "prefix-rule",
            "eng",
            "api",
            Some("eng-"),
            None,
            None,
        ));
        a.add_rule(make_rule(
            "tag-rule",
            "ml",
            "training",
            None,
            Some(("team", "ml")),
            None,
        ));
        let mut tags = HashMap::new();
        tags.insert("team".to_string(), "ml".to_string());
        // session starts with "eng-" AND has ml tag — prefix should win
        a.record("eng-session-99", "gpt-4o-mini", 1.0, &tags);
        assert!(a.bucket("eng", "api").is_some());
        assert!(a.bucket("ml", "training").is_none());
    }

    // 6. Multiple records accumulate.
    #[test]
    fn test_accumulation() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "ops", "infra", Some("ops-"), None, None));
        a.record("ops-1", "gpt-4o", 1.0, &empty_tags());
        a.record("ops-2", "gpt-4o", 2.0, &empty_tags());
        a.record("ops-3", "gpt-4o", 3.0, &empty_tags());
        let b = a.bucket("ops", "infra").unwrap();
        assert!((b.total_cost_usd - 6.0).abs() < 1e-9);
        assert_eq!(b.request_count, 3);
    }

    // 7. Models used map tracks per-model cost.
    #[test]
    fn test_models_used() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "eng", "search", Some("s-"), None, None));
        a.record("s-1", "gpt-4o-mini", 0.5, &empty_tags());
        a.record("s-2", "claude-sonnet-4-6", 1.5, &empty_tags());
        a.record("s-3", "gpt-4o-mini", 0.5, &empty_tags());
        let b = a.bucket("eng", "search").unwrap();
        assert!((b.models_used["gpt-4o-mini"] - 1.0).abs() < 1e-9);
        assert!((b.models_used["claude-sonnet-4-6"] - 1.5).abs() < 1e-9);
    }

    // 8. top_model returns the costliest model.
    #[test]
    fn test_top_model() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "data", "etl", Some("d-"), None, None));
        a.record("d-1", "cheap-model", 0.10, &empty_tags());
        a.record("d-2", "expensive-model", 9.99, &empty_tags());
        let b = a.bucket("data", "etl").unwrap();
        let (model, _cost) = b.top_model().unwrap();
        assert_eq!(model, "expensive-model");
    }

    // 9. Budget utilization calculation.
    #[test]
    fn test_budget_utilization() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule(
            "r1",
            "eng",
            "ml",
            Some("ml-"),
            None,
            Some(100.0),
        ));
        a.record("ml-session", "gpt-4o", 25.0, &empty_tags());
        let b = a.bucket("eng", "ml").unwrap();
        let pct = b.budget_utilization_pct().unwrap();
        assert!((pct - 25.0).abs() < 1e-9);
        assert!(!b.is_over_budget());
    }

    // 10. is_over_budget triggers correctly.
    #[test]
    fn test_over_budget_detection() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule(
            "r1",
            "ops",
            "batch",
            Some("b-"),
            None,
            Some(10.0),
        ));
        a.record("b-1", "gpt-4o", 15.0, &empty_tags());
        let b = a.bucket("ops", "batch").unwrap();
        assert!(b.is_over_budget());
    }

    // 11. over_budget_buckets returns the correct set.
    #[test]
    fn test_over_budget_buckets_list() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "t1", "p1", Some("t1-"), None, Some(5.0)));
        a.add_rule(make_rule("r2", "t2", "p2", Some("t2-"), None, Some(50.0)));
        a.record("t1-s", "gpt-4o-mini", 10.0, &empty_tags()); // over
        a.record("t2-s", "gpt-4o-mini", 5.0, &empty_tags()); // under
        let over = a.over_budget_buckets();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].team, "t1");
    }

    // 12. all_buckets_ranked orders by cost descending.
    #[test]
    fn test_all_buckets_ranked_order() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "a", "p", Some("a-"), None, None));
        a.add_rule(make_rule("r2", "b", "p", Some("b-"), None, None));
        a.record("a-1", "m", 1.0, &empty_tags());
        a.record("b-1", "m", 5.0, &empty_tags());
        let ranked = a.all_buckets_ranked();
        assert_eq!(ranked[0].team, "b");
        assert_eq!(ranked[1].team, "a");
    }

    // 13. chargeback_csv has the expected header.
    #[test]
    fn test_chargeback_csv_header() {
        let a = CostAllocator::new();
        let csv = a.chargeback_csv();
        assert!(csv.starts_with(
            "team,project,total_cost_usd,request_count,budget_usd,utilization_pct,over_budget,top_model"
        ));
    }

    // 14. chargeback_csv contains data rows.
    #[test]
    fn test_chargeback_csv_rows() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "eng", "api", Some("e-"), None, Some(100.0)));
        a.record("e-1", "gpt-4o-mini", 3.14, &empty_tags());
        let csv = a.chargeback_csv();
        assert!(csv.contains("eng"));
        assert!(csv.contains("api"));
        assert!(csv.contains("gpt-4o-mini"));
    }

    // 15. showback_summary returns one line per bucket.
    #[test]
    fn test_showback_summary_line_count() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "t1", "p1", Some("t1-"), None, None));
        a.add_rule(make_rule("r2", "t2", "p2", Some("t2-"), None, None));
        a.record("t1-s", "m", 1.0, &empty_tags());
        a.record("t2-s", "m", 2.0, &empty_tags());
        assert_eq!(a.showback_summary().len(), 2);
    }

    // 16. No budget — utilization is None and is_over_budget is false.
    #[test]
    fn test_no_budget_utilization_none() {
        let mut b = AllocationBucket::default();
        b.total_cost_usd = 999.0;
        assert!(b.budget_utilization_pct().is_none());
        assert!(!b.is_over_budget());
    }

    // 17. top_model on empty bucket returns None.
    #[test]
    fn test_top_model_empty() {
        let b = AllocationBucket::default();
        assert!(b.top_model().is_none());
    }
}
