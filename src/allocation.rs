//! # Team and Project Cost Allocation
//!
//! Tracks LLM spend broken down by team, project, user, and environment.
//! Supports budget assignment, cost-center reporting, and
//! chargeback/showback workflows.
//!
//! ## Key Types
//!
//! - [`AllocationTag`] - structured label combining team, project, user, and environment
//! - [`AllocationRule`] - maps incoming requests to a team/project bucket via prefix or tag
//! - [`AllocationBucket`] - accumulated cost for a team/project pair
//! - [`CostAllocator`] - rule-based router (legacy, single-level)
//! - [`CostAllocation`] - per-tag spend record
//! - [`AllocationLedger`] - accumulates spend across all tag combinations
//! - [`BudgetHierarchy`] - team → project → user quota cascade
//! - [`AllocationReport`] - breakdown by team showing usage vs budget with % utilisation
//!
//! ## Allocation Flow
//!
//! ```text
//! LogEntry metadata  →  AllocationTag  →  AllocationLedger  →  AllocationReport
//!                                      ↑
//!                                 BudgetHierarchy (enforces quotas)
//! ```

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

// ═══════════════════════════════════════════════════════════════════════════════
// NEW TYPES — Team/Org Cost Allocation (Round 2)
// ═══════════════════════════════════════════════════════════════════════════════

// ── AllocationTag ─────────────────────────────────────────────────────────────

/// Deployment environment for a cost event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum Environment {
    /// Local development environment.
    Dev,
    /// Pre-production / QA environment.
    Staging,
    /// Live production environment.
    #[default]
    Prod,
    /// A custom environment label not covered by the standard variants.
    Custom(String),
}

impl Environment {
    /// Parse a string into an [`Environment`] (case-insensitive).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "dev" | "development" => Self::Dev,
            "staging" | "stage" => Self::Staging,
            "prod" | "production" => Self::Prod,
            other => Self::Custom(other.to_string()),
        }
    }

    /// Return the canonical string label for this environment.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Dev => "dev",
            Self::Staging => "staging",
            Self::Prod => "prod",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A structured label that identifies the cost owner of a single LLM request.
///
/// Tags are parsed from [`crate::log::LogEntry`] metadata fields.  Any field
/// not present in the source log defaults to `"unknown"` / [`Environment::Prod`].
///
/// # Parsing from metadata
///
/// ```rust
/// use std::collections::HashMap;
/// use llm_cost_dashboard::allocation::AllocationTag;
///
/// let mut meta = HashMap::new();
/// meta.insert("team".to_string(), "platform".to_string());
/// meta.insert("project".to_string(), "api-gateway".to_string());
/// meta.insert("user".to_string(), "alice".to_string());
/// meta.insert("env".to_string(), "staging".to_string());
///
/// let tag = AllocationTag::from_metadata(&meta);
/// assert_eq!(tag.team, "platform");
/// assert_eq!(tag.environment.as_str(), "staging");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AllocationTag {
    /// Owning team (e.g. `"platform"`, `"data-science"`).
    pub team: String,
    /// Project within the team (e.g. `"api-gateway"`, `"recommender"`).
    pub project: String,
    /// Individual user or service account (e.g. `"alice"`, `"ci-bot"`).
    pub user: String,
    /// Deployment environment where the request originated.
    pub environment: Environment,
}

impl AllocationTag {
    /// Construct a tag with all fields explicitly supplied.
    pub fn new(
        team: impl Into<String>,
        project: impl Into<String>,
        user: impl Into<String>,
        environment: Environment,
    ) -> Self {
        Self {
            team: team.into(),
            project: project.into(),
            user: user.into(),
            environment,
        }
    }

    /// Parse an [`AllocationTag`] from a key/value metadata map.
    ///
    /// Recognised keys: `team`, `project`, `user`, `env` / `environment`.
    /// Missing keys fall back to `"unknown"` / [`Environment::Prod`].
    pub fn from_metadata(meta: &HashMap<String, String>) -> Self {
        let team = meta
            .get("team")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let project = meta
            .get("project")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let user = meta
            .get("user")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let env_str = meta
            .get("env")
            .or_else(|| meta.get("environment"))
            .map(|s| s.as_str())
            .unwrap_or("prod");
        Self {
            team,
            project,
            user,
            environment: Environment::from_str(env_str),
        }
    }

    /// Return a compact display key: `"team/project/user@env"`.
    pub fn display_key(&self) -> String {
        format!(
            "{}/{}/{}@{}",
            self.team, self.project, self.user, self.environment
        )
    }
}

// ── CostAllocation ────────────────────────────────────────────────────────────

/// A single cost event associated with an [`AllocationTag`].
///
/// This is the unit written into the [`AllocationLedger`].
#[derive(Debug, Clone)]
pub struct CostAllocation {
    /// The tag identifying who owns this cost.
    pub tag: AllocationTag,
    /// USD cost for this event.
    pub cost_usd: f64,
    /// Model used for this request.
    pub model: String,
    /// Token count (input + output) for this request.
    pub tokens: u64,
}

impl CostAllocation {
    /// Create a new allocation event.
    pub fn new(
        tag: AllocationTag,
        cost_usd: f64,
        model: impl Into<String>,
        tokens: u64,
    ) -> Self {
        Self {
            tag,
            cost_usd,
            model: model.into(),
            tokens,
        }
    }
}

// ── AllocationLedger ─────────────────────────────────────────────────────────

/// Accumulated spend per [`AllocationTag`] combination.
///
/// The ledger is a pure accumulator; it never modifies or removes entries.
///
/// # Example
///
/// ```rust
/// use llm_cost_dashboard::allocation::{AllocationTag, AllocationLedger, CostAllocation, Environment};
///
/// let mut ledger = AllocationLedger::new();
/// let tag = AllocationTag::new("eng", "search", "bob", Environment::Prod);
/// ledger.record(CostAllocation::new(tag, 0.12, "gpt-4o-mini", 500));
/// assert!((ledger.total_cost_usd() - 0.12).abs() < 1e-9);
/// ```
#[derive(Debug, Default)]
pub struct AllocationLedger {
    /// Per-tag accumulated costs.
    entries: HashMap<AllocationTag, TagAccumulator>,
}

/// Internal accumulator for one [`AllocationTag`].
#[derive(Debug, Default, Clone)]
struct TagAccumulator {
    total_cost_usd: f64,
    request_count: u64,
    total_tokens: u64,
    /// Per-model cost breakdown.
    models: HashMap<String, f64>,
}

impl AllocationLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cost allocation event.
    pub fn record(&mut self, alloc: CostAllocation) {
        let acc = self.entries.entry(alloc.tag).or_default();
        acc.total_cost_usd += alloc.cost_usd;
        acc.request_count += 1;
        acc.total_tokens += alloc.tokens;
        *acc.models.entry(alloc.model).or_insert(0.0) += alloc.cost_usd;
    }

    /// Total cost across all tags.
    pub fn total_cost_usd(&self) -> f64 {
        self.entries.values().map(|a| a.total_cost_usd).sum()
    }

    /// Total number of cost events recorded.
    pub fn total_request_count(&self) -> u64 {
        self.entries.values().map(|a| a.request_count).sum()
    }

    /// Aggregate spend by team, returning a map of `team -> total USD`.
    pub fn spend_by_team(&self) -> HashMap<String, f64> {
        let mut out: HashMap<String, f64> = HashMap::new();
        for (tag, acc) in &self.entries {
            *out.entry(tag.team.clone()).or_insert(0.0) += acc.total_cost_usd;
        }
        out
    }

    /// Aggregate spend by project within a specific team.
    pub fn spend_by_project(&self, team: &str) -> HashMap<String, f64> {
        let mut out: HashMap<String, f64> = HashMap::new();
        for (tag, acc) in &self.entries {
            if tag.team == team {
                *out.entry(tag.project.clone()).or_insert(0.0) += acc.total_cost_usd;
            }
        }
        out
    }

    /// Aggregate spend by user within a specific team.
    pub fn spend_by_user(&self, team: &str) -> HashMap<String, f64> {
        let mut out: HashMap<String, f64> = HashMap::new();
        for (tag, acc) in &self.entries {
            if tag.team == team {
                *out.entry(tag.user.clone()).or_insert(0.0) += acc.total_cost_usd;
            }
        }
        out
    }

    /// Aggregate spend by environment across all teams.
    pub fn spend_by_environment(&self) -> HashMap<String, f64> {
        let mut out: HashMap<String, f64> = HashMap::new();
        for (tag, acc) in &self.entries {
            *out.entry(tag.environment.to_string()).or_insert(0.0) += acc.total_cost_usd;
        }
        out
    }

    /// All known team names, sorted alphabetically.
    pub fn teams(&self) -> Vec<String> {
        let mut teams: Vec<String> = self
            .entries
            .keys()
            .map(|t| t.team.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        teams.sort();
        teams
    }

    /// Iterate over all (tag, total_cost, request_count, total_tokens) tuples.
    pub fn iter(&self) -> impl Iterator<Item = (&AllocationTag, f64, u64, u64)> {
        self.entries
            .iter()
            .map(|(tag, acc)| (tag, acc.total_cost_usd, acc.request_count, acc.total_tokens))
    }
}

// ── BudgetHierarchy ───────────────────────────────────────────────────────────

/// Per-user quota within a project.
#[derive(Debug, Clone)]
pub struct UserQuota {
    /// User identifier.
    pub user: String,
    /// Monthly spending quota in USD.
    pub quota_usd: f64,
}

/// Per-project budget configuration within a team.
#[derive(Debug, Clone)]
pub struct ProjectBudget {
    /// Project name.
    pub project: String,
    /// Monthly project budget in USD.
    pub budget_usd: f64,
    /// Optional per-user quotas.  Users not listed here are uncapped at project level.
    pub user_quotas: Vec<UserQuota>,
}

impl ProjectBudget {
    /// Create a project budget with no per-user quotas.
    pub fn new(project: impl Into<String>, budget_usd: f64) -> Self {
        Self {
            project: project.into(),
            budget_usd,
            user_quotas: Vec::new(),
        }
    }

    /// Add a per-user quota to this project.
    pub fn with_user_quota(mut self, user: impl Into<String>, quota_usd: f64) -> Self {
        self.user_quotas.push(UserQuota {
            user: user.into(),
            quota_usd,
        });
        self
    }

    /// Lookup the quota for a specific user, returning `None` if uncapped.
    pub fn user_quota(&self, user: &str) -> Option<f64> {
        self.user_quotas
            .iter()
            .find(|q| q.user == user)
            .map(|q| q.quota_usd)
    }
}

/// Per-team budget configuration with cascading project budgets.
#[derive(Debug, Clone)]
pub struct TeamBudget {
    /// Team name.
    pub team: String,
    /// Monthly team-level budget in USD.
    pub budget_usd: f64,
    /// Per-project budgets within this team.
    pub projects: Vec<ProjectBudget>,
}

impl TeamBudget {
    /// Create a team budget with no project-level breakdown.
    pub fn new(team: impl Into<String>, budget_usd: f64) -> Self {
        Self {
            team: team.into(),
            budget_usd,
            projects: Vec::new(),
        }
    }

    /// Add a project budget under this team.
    pub fn with_project(mut self, project: ProjectBudget) -> Self {
        self.projects.push(project);
        self
    }

    /// Return the configured budget for a specific project, or `None` if
    /// that project has no explicit cap (falls back to team-level enforcement).
    pub fn project_budget(&self, project: &str) -> Option<f64> {
        self.projects
            .iter()
            .find(|p| p.project == project)
            .map(|p| p.budget_usd)
    }

    /// Resolve the effective quota for `(project, user)`.
    ///
    /// Resolution order (most-specific wins):
    /// 1. User quota on the matching project.
    /// 2. Project budget.
    /// 3. Team budget (fallback).
    pub fn effective_limit(&self, project: &str, user: &str) -> f64 {
        if let Some(proj) = self.projects.iter().find(|p| p.project == project) {
            if let Some(quota) = proj.user_quota(user) {
                return quota;
            }
            return proj.budget_usd;
        }
        self.budget_usd
    }
}

/// Hierarchical budget store: team budget → project budget → user quota.
///
/// The hierarchy enforces that narrower scopes are bounded by wider ones:
/// the effective limit for any `(team, project, user)` triple is the
/// minimum of all applicable limits in the chain.
///
/// # Example
///
/// ```rust
/// use llm_cost_dashboard::allocation::{BudgetHierarchy, TeamBudget, ProjectBudget};
///
/// let mut hierarchy = BudgetHierarchy::new();
/// let team = TeamBudget::new("engineering", 500.0)
///     .with_project(
///         ProjectBudget::new("backend", 200.0)
///             .with_user_quota("alice", 50.0)
///     );
/// hierarchy.add_team(team);
///
/// assert_eq!(hierarchy.effective_limit("engineering", "backend", "alice"), 50.0);
/// assert_eq!(hierarchy.effective_limit("engineering", "backend", "bob"), 200.0);
/// assert_eq!(hierarchy.effective_limit("engineering", "infra", "bob"), 500.0);
/// ```
#[derive(Debug, Default)]
pub struct BudgetHierarchy {
    teams: HashMap<String, TeamBudget>,
}

impl BudgetHierarchy {
    /// Create an empty hierarchy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or replace a team budget.
    pub fn add_team(&mut self, budget: TeamBudget) {
        self.teams.insert(budget.team.clone(), budget);
    }

    /// Retrieve the budget configuration for a team.
    pub fn team_budget(&self, team: &str) -> Option<&TeamBudget> {
        self.teams.get(team)
    }

    /// Resolve the effective spending limit for a `(team, project, user)` triple.
    ///
    /// Returns `f64::MAX` when no budget has been configured at any level
    /// (i.e. uncapped).
    pub fn effective_limit(&self, team: &str, project: &str, user: &str) -> f64 {
        match self.teams.get(team) {
            Some(tb) => tb.effective_limit(project, user),
            None => f64::MAX,
        }
    }

    /// Check whether the given spend would breach any limit in the cascade.
    ///
    /// Returns `true` if `spend_usd` equals or exceeds the effective limit.
    pub fn is_over_limit(&self, team: &str, project: &str, user: &str, spend_usd: f64) -> bool {
        let limit = self.effective_limit(team, project, user);
        limit < f64::MAX && spend_usd >= limit
    }

    /// All configured team names.
    pub fn team_names(&self) -> Vec<&str> {
        self.teams.keys().map(|s| s.as_str()).collect()
    }
}

// ── AllocationReport ─────────────────────────────────────────────────────────

/// Per-team usage summary within an [`AllocationReport`].
#[derive(Debug, Clone)]
pub struct TeamUsage {
    /// Team name.
    pub team: String,
    /// Total spend by this team in USD.
    pub total_cost_usd: f64,
    /// Configured budget for this team in USD (`None` if uncapped).
    pub budget_usd: Option<f64>,
    /// Percentage of the budget consumed (`0..100+`), or `None` if uncapped.
    pub utilization_pct: Option<f64>,
    /// Number of requests attributed to this team.
    pub request_count: u64,
    /// Top-3 projects by spend: `[(project, cost_usd)]`.
    pub top_projects: Vec<(String, f64)>,
    /// Whether this team is over its configured budget.
    pub over_budget: bool,
}

/// A period-end breakdown of all teams showing usage vs. budget.
///
/// Generated by [`AllocationReport::build`] from an [`AllocationLedger`] and a
/// [`BudgetHierarchy`].
///
/// # Example
///
/// ```rust
/// use llm_cost_dashboard::allocation::{
///     AllocationLedger, AllocationTag, BudgetHierarchy, CostAllocation,
///     AllocationReport, TeamBudget, Environment,
/// };
///
/// let mut ledger = AllocationLedger::new();
/// let tag = AllocationTag::new("eng", "api", "alice", Environment::Prod);
/// ledger.record(CostAllocation::new(tag, 75.0, "gpt-4o", 1000));
///
/// let mut hierarchy = BudgetHierarchy::new();
/// hierarchy.add_team(TeamBudget::new("eng", 100.0));
///
/// let report = AllocationReport::build(&ledger, &hierarchy);
/// assert_eq!(report.teams.len(), 1);
/// assert!((report.teams[0].utilization_pct.unwrap() - 75.0).abs() < 1e-6);
/// ```
#[derive(Debug)]
pub struct AllocationReport {
    /// Per-team usage rows, sorted by total cost descending.
    pub teams: Vec<TeamUsage>,
    /// Grand total spend across all teams.
    pub grand_total_usd: f64,
    /// Grand total request count.
    pub grand_total_requests: u64,
}

impl AllocationReport {
    /// Build an allocation report from a ledger and a budget hierarchy.
    pub fn build(ledger: &AllocationLedger, hierarchy: &BudgetHierarchy) -> Self {
        // Aggregate per-team totals and per-project breakdown.
        let mut team_cost: HashMap<String, f64> = HashMap::new();
        let mut team_requests: HashMap<String, u64> = HashMap::new();
        let mut project_cost: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for (tag, cost, count, _tokens) in ledger.iter() {
            *team_cost.entry(tag.team.clone()).or_insert(0.0) += cost;
            *team_requests.entry(tag.team.clone()).or_insert(0) += count;
            *project_cost
                .entry(tag.team.clone())
                .or_default()
                .entry(tag.project.clone())
                .or_insert(0.0) += cost;
        }

        let grand_total_usd: f64 = team_cost.values().sum();
        let grand_total_requests: u64 = team_requests.values().sum();

        // Build per-team rows.
        let mut teams: Vec<TeamUsage> = team_cost
            .iter()
            .map(|(team, &cost)| {
                let budget_usd = hierarchy
                    .team_budget(team)
                    .map(|tb| tb.budget_usd);
                let utilization_pct = budget_usd
                    .filter(|&b| b > 0.0)
                    .map(|b| (cost / b) * 100.0);
                let over_budget = budget_usd
                    .map(|b| cost > b)
                    .unwrap_or(false);
                let request_count = *team_requests.get(team).unwrap_or(&0);

                // Top-3 projects by spend.
                let mut proj_vec: Vec<(String, f64)> = project_cost
                    .get(team)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                proj_vec.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                proj_vec.truncate(3);

                TeamUsage {
                    team: team.clone(),
                    total_cost_usd: cost,
                    budget_usd,
                    utilization_pct,
                    request_count,
                    top_projects: proj_vec,
                    over_budget,
                }
            })
            .collect();

        teams.sort_by(|a, b| {
            b.total_cost_usd
                .partial_cmp(&a.total_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self {
            teams,
            grand_total_usd,
            grand_total_requests,
        }
    }

    /// Render the report as a human-readable multi-line summary.
    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "Team Allocation Report — Total: ${:.4} across {} requests",
            self.grand_total_usd, self.grand_total_requests
        ));
        lines.push("─".repeat(60));
        for t in &self.teams {
            let budget_str = match t.budget_usd {
                Some(b) => {
                    let pct = t.utilization_pct.unwrap_or(0.0);
                    let flag = if t.over_budget { " [OVER BUDGET]" } else { "" };
                    format!("budget ${b:.2} ({pct:.1}% used){flag}")
                }
                None => "no budget".to_string(),
            };
            lines.push(format!(
                "  {:<20} ${:.4}  {}  ({} requests)",
                t.team, t.total_cost_usd, budget_str, t.request_count
            ));
            for (proj, cost) in &t.top_projects {
                lines.push(format!("    └─ {:<18} ${:.4}", proj, cost));
            }
        }
        lines
    }

    /// Format the report as a CSV string.
    ///
    /// Columns: `team,total_cost_usd,budget_usd,utilization_pct,over_budget,request_count`
    pub fn to_csv(&self) -> String {
        let mut lines =
            vec!["team,total_cost_usd,budget_usd,utilization_pct,over_budget,request_count"
                .to_string()];
        for t in &self.teams {
            let budget = t
                .budget_usd
                .map(|v| format!("{v:.4}"))
                .unwrap_or_default();
            let util = t
                .utilization_pct
                .map(|v| format!("{v:.2}"))
                .unwrap_or_default();
            lines.push(format!(
                "{},{:.6},{},{},{},{}",
                t.team,
                t.total_cost_usd,
                budget,
                util,
                t.over_budget,
                t.request_count,
            ));
        }
        lines.join("\n")
    }
}

// ── TUI helper ────────────────────────────────────────────────────────────────

/// Render rows for a ratatui Teams tab.
///
/// Returns a `Vec<Vec<String>>` where each inner `Vec` is a table row with
/// columns: `["Team", "Spend (USD)", "Budget (USD)", "Utilization %", "Status",
/// "Requests"]`.
///
/// Suitable for direct use with `ratatui::widgets::Table`.
pub fn teams_tab_rows(report: &AllocationReport) -> Vec<Vec<String>> {
    report
        .teams
        .iter()
        .map(|t| {
            let budget = t
                .budget_usd
                .map(|b| format!("${b:.2}"))
                .unwrap_or_else(|| "—".to_string());
            let util = t
                .utilization_pct
                .map(|p| format!("{p:.1}%"))
                .unwrap_or_else(|| "—".to_string());
            let status = if t.over_budget {
                "OVER".to_string()
            } else {
                "OK".to_string()
            };
            vec![
                t.team.clone(),
                format!("${:.4}", t.total_cost_usd),
                budget,
                util,
                status,
                t.request_count.to_string(),
            ]
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

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

    // ── CostAllocator (legacy) ──────────────────────────────────────────────

    #[test]
    fn test_new_allocator_empty() {
        let a = CostAllocator::new();
        assert!(a.all_buckets_ranked().is_empty());
    }

    #[test]
    fn test_prefix_rule_matches() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "eng", "backend", Some("eng-"), None, None));
        a.record("eng-session-1", "gpt-4o-mini", 1.0, &empty_tags());
        let b = a.bucket("eng", "backend").unwrap();
        assert!((b.total_cost_usd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_unallocated_default() {
        let mut a = CostAllocator::new();
        a.record("mystery-session", "gpt-4o", 2.5, &empty_tags());
        let b = a.bucket("unallocated", "unallocated").unwrap();
        assert!((b.total_cost_usd - 2.5).abs() < 1e-9);
    }

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
        a.record("eng-session-99", "gpt-4o-mini", 1.0, &tags);
        assert!(a.bucket("eng", "api").is_some());
        assert!(a.bucket("ml", "training").is_none());
    }

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

    #[test]
    fn test_over_budget_buckets_list() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "t1", "p1", Some("t1-"), None, Some(5.0)));
        a.add_rule(make_rule("r2", "t2", "p2", Some("t2-"), None, Some(50.0)));
        a.record("t1-s", "gpt-4o-mini", 10.0, &empty_tags());
        a.record("t2-s", "gpt-4o-mini", 5.0, &empty_tags());
        let over = a.over_budget_buckets();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].team, "t1");
    }

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

    #[test]
    fn test_chargeback_csv_header() {
        let a = CostAllocator::new();
        let csv = a.chargeback_csv();
        assert!(csv.starts_with(
            "team,project,total_cost_usd,request_count,budget_usd,utilization_pct,over_budget,top_model"
        ));
    }

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

    #[test]
    fn test_showback_summary_line_count() {
        let mut a = CostAllocator::new();
        a.add_rule(make_rule("r1", "t1", "p1", Some("t1-"), None, None));
        a.add_rule(make_rule("r2", "t2", "p2", Some("t2-"), None, None));
        a.record("t1-s", "m", 1.0, &empty_tags());
        a.record("t2-s", "m", 2.0, &empty_tags());
        assert_eq!(a.showback_summary().len(), 2);
    }

    #[test]
    fn test_no_budget_utilization_none() {
        let mut b = AllocationBucket::default();
        b.total_cost_usd = 999.0;
        assert!(b.budget_utilization_pct().is_none());
        assert!(!b.is_over_budget());
    }

    #[test]
    fn test_top_model_empty() {
        let b = AllocationBucket::default();
        assert!(b.top_model().is_none());
    }

    // ── AllocationTag ───────────────────────────────────────────────────────

    #[test]
    fn test_allocation_tag_from_metadata_full() {
        let mut meta = HashMap::new();
        meta.insert("team".to_string(), "platform".to_string());
        meta.insert("project".to_string(), "api-gw".to_string());
        meta.insert("user".to_string(), "alice".to_string());
        meta.insert("env".to_string(), "staging".to_string());
        let tag = AllocationTag::from_metadata(&meta);
        assert_eq!(tag.team, "platform");
        assert_eq!(tag.project, "api-gw");
        assert_eq!(tag.user, "alice");
        assert_eq!(tag.environment, Environment::Staging);
    }

    #[test]
    fn test_allocation_tag_from_metadata_defaults() {
        let meta = HashMap::new();
        let tag = AllocationTag::from_metadata(&meta);
        assert_eq!(tag.team, "unknown");
        assert_eq!(tag.environment, Environment::Prod);
    }

    #[test]
    fn test_allocation_tag_display_key() {
        let tag = AllocationTag::new("eng", "api", "bob", Environment::Dev);
        assert_eq!(tag.display_key(), "eng/api/bob@dev");
    }

    #[test]
    fn test_environment_custom() {
        let env = Environment::from_str("canary");
        assert_eq!(env.as_str(), "canary");
    }

    // ── AllocationLedger ────────────────────────────────────────────────────

    #[test]
    fn test_ledger_accumulates() {
        let mut ledger = AllocationLedger::new();
        let tag = AllocationTag::new("eng", "search", "bob", Environment::Prod);
        ledger.record(CostAllocation::new(tag.clone(), 0.10, "gpt-4o-mini", 100));
        ledger.record(CostAllocation::new(tag, 0.20, "gpt-4o-mini", 200));
        assert!((ledger.total_cost_usd() - 0.30).abs() < 1e-9);
        assert_eq!(ledger.total_request_count(), 2);
    }

    #[test]
    fn test_ledger_spend_by_team() {
        let mut ledger = AllocationLedger::new();
        let t1 = AllocationTag::new("eng", "api", "alice", Environment::Prod);
        let t2 = AllocationTag::new("data", "etl", "bob", Environment::Dev);
        ledger.record(CostAllocation::new(t1, 1.0, "m", 10));
        ledger.record(CostAllocation::new(t2, 2.0, "m", 20));
        let by_team = ledger.spend_by_team();
        assert!((by_team["eng"] - 1.0).abs() < 1e-9);
        assert!((by_team["data"] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_ledger_teams_sorted() {
        let mut ledger = AllocationLedger::new();
        ledger.record(CostAllocation::new(
            AllocationTag::new("zeta", "p", "u", Environment::Prod),
            1.0,
            "m",
            1,
        ));
        ledger.record(CostAllocation::new(
            AllocationTag::new("alpha", "p", "u", Environment::Prod),
            1.0,
            "m",
            1,
        ));
        let teams = ledger.teams();
        assert_eq!(teams, vec!["alpha", "zeta"]);
    }

    // ── BudgetHierarchy ─────────────────────────────────────────────────────

    #[test]
    fn test_budget_hierarchy_user_quota() {
        let mut h = BudgetHierarchy::new();
        let team = TeamBudget::new("engineering", 500.0).with_project(
            ProjectBudget::new("backend", 200.0).with_user_quota("alice", 50.0),
        );
        h.add_team(team);
        assert!((h.effective_limit("engineering", "backend", "alice") - 50.0).abs() < 1e-9);
        assert!((h.effective_limit("engineering", "backend", "bob") - 200.0).abs() < 1e-9);
        assert!((h.effective_limit("engineering", "infra", "bob") - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_budget_hierarchy_unconfigured_team() {
        let h = BudgetHierarchy::new();
        assert_eq!(h.effective_limit("unknown", "proj", "user"), f64::MAX);
    }

    #[test]
    fn test_budget_hierarchy_over_limit() {
        let mut h = BudgetHierarchy::new();
        h.add_team(TeamBudget::new("eng", 100.0));
        assert!(h.is_over_limit("eng", "api", "user", 100.0));
        assert!(!h.is_over_limit("eng", "api", "user", 99.99));
    }

    // ── AllocationReport ────────────────────────────────────────────────────

    #[test]
    fn test_report_build_basic() {
        let mut ledger = AllocationLedger::new();
        let tag = AllocationTag::new("eng", "api", "alice", Environment::Prod);
        ledger.record(CostAllocation::new(tag, 75.0, "gpt-4o", 1000));
        let mut h = BudgetHierarchy::new();
        h.add_team(TeamBudget::new("eng", 100.0));
        let report = AllocationReport::build(&ledger, &h);
        assert_eq!(report.teams.len(), 1);
        let t = &report.teams[0];
        assert_eq!(t.team, "eng");
        assert!((t.total_cost_usd - 75.0).abs() < 1e-9);
        assert!((t.utilization_pct.unwrap() - 75.0).abs() < 1e-6);
        assert!(!t.over_budget);
    }

    #[test]
    fn test_report_over_budget_flag() {
        let mut ledger = AllocationLedger::new();
        let tag = AllocationTag::new("eng", "api", "alice", Environment::Prod);
        ledger.record(CostAllocation::new(tag, 120.0, "gpt-4o", 2000));
        let mut h = BudgetHierarchy::new();
        h.add_team(TeamBudget::new("eng", 100.0));
        let report = AllocationReport::build(&ledger, &h);
        assert!(report.teams[0].over_budget);
    }

    #[test]
    fn test_report_sorted_by_cost() {
        let mut ledger = AllocationLedger::new();
        ledger.record(CostAllocation::new(
            AllocationTag::new("cheap", "p", "u", Environment::Prod),
            1.0,
            "m",
            10,
        ));
        ledger.record(CostAllocation::new(
            AllocationTag::new("expensive", "p", "u", Environment::Prod),
            99.0,
            "m",
            100,
        ));
        let h = BudgetHierarchy::new();
        let report = AllocationReport::build(&ledger, &h);
        assert_eq!(report.teams[0].team, "expensive");
        assert_eq!(report.teams[1].team, "cheap");
    }

    #[test]
    fn test_report_csv_header() {
        let ledger = AllocationLedger::new();
        let h = BudgetHierarchy::new();
        let report = AllocationReport::build(&ledger, &h);
        assert!(report
            .to_csv()
            .starts_with("team,total_cost_usd,budget_usd,utilization_pct,over_budget,request_count"));
    }

    #[test]
    fn test_teams_tab_rows() {
        let mut ledger = AllocationLedger::new();
        ledger.record(CostAllocation::new(
            AllocationTag::new("eng", "api", "alice", Environment::Prod),
            50.0,
            "gpt-4o",
            500,
        ));
        let mut h = BudgetHierarchy::new();
        h.add_team(TeamBudget::new("eng", 100.0));
        let report = AllocationReport::build(&ledger, &h);
        let rows = teams_tab_rows(&report);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], "eng");
        assert_eq!(rows[0][4], "OK");
    }
}
