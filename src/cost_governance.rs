//! Policy-based cost governance with approval workflows and monthly spend tracking.

use std::collections::HashMap;

/// Scope to which a governance policy or spend request applies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PolicyScope {
    /// Applies to all activity.
    Global,
    /// Applies to a named team.
    Team(String),
    /// Applies to a named project.
    Project(String),
    /// Applies to a specific user.
    User(String),
}

impl std::fmt::Display for PolicyScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyScope::Global => write!(f, "global"),
            PolicyScope::Team(t) => write!(f, "team:{}", t),
            PolicyScope::Project(p) => write!(f, "project:{}", p),
            PolicyScope::User(u) => write!(f, "user:{}", u),
        }
    }
}

/// A governance policy that controls how much can be spent and under what conditions.
#[derive(Debug, Clone)]
pub struct GovernancePolicy {
    /// Unique policy identifier.
    pub policy_id: String,
    /// Human-readable policy name.
    pub name: String,
    /// The scope this policy governs.
    pub scope: PolicyScope,
    /// Maximum total spend allowed per calendar month (USD).
    pub monthly_limit_usd: f64,
    /// Maximum spend allowed per individual request (USD).
    pub per_request_limit_usd: f64,
    /// Requests with estimated cost above this threshold require human approval.
    pub requires_approval_above_usd: f64,
    /// Requests with estimated cost at or below this threshold are auto-approved.
    pub auto_approve_below_usd: f64,
}

/// The outcome of evaluating a spend request against active policies.
#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalStatus {
    /// Request was automatically approved (below auto-approve threshold).
    AutoApproved,
    /// Request is awaiting human review.
    PendingReview,
    /// Request was manually approved by `approver`.
    Approved {
        /// Identifier of the person who approved this request.
        approver: String,
    },
    /// Request was rejected.
    Rejected {
        /// Human-readable reason for rejection.
        reason: String,
    },
}

/// A request to spend money on LLM inference.
#[derive(Debug, Clone)]
pub struct SpendRequest {
    /// Unique request identifier.
    pub request_id: String,
    /// Identifier of the user or system making the request.
    pub requester_id: String,
    /// Scope the request falls under.
    pub scope: PolicyScope,
    /// Estimated cost in USD.
    pub estimated_cost_usd: f64,
    /// Human-readable description of what is being requested.
    pub description: String,
    /// Wall-clock timestamp when the request was submitted (ms since epoch).
    pub timestamp: u64,
}

/// A tracked approval decision with optional approved-spend accounting.
#[derive(Debug, Clone)]
struct ApprovalRecord {
    request: SpendRequest,
    status: ApprovalStatus,
}

/// Policy-based engine that evaluates, approves, and tracks spend requests.
#[derive(Debug, Default)]
pub struct GovernanceEngine {
    policies: HashMap<String, GovernancePolicy>,
    records: HashMap<String, ApprovalRecord>,
    /// Approved spend entries: `(scope_string, cost_usd, timestamp_ms)`.
    approved_spend: Vec<(String, f64, u64)>,
}

impl GovernanceEngine {
    /// Create a new, empty governance engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) a governance policy.
    pub fn register_policy(&mut self, policy: GovernancePolicy) {
        self.policies.insert(policy.policy_id.clone(), policy);
    }

    /// Evaluate `request` against all applicable policies.
    ///
    /// Evaluation order:
    /// 1. If any policy rejects on per-request limit → `Rejected`.
    /// 2. If monthly limit would be exceeded → `Rejected`.
    /// 3. If cost ≤ `auto_approve_below_usd` → `AutoApproved` (and spend recorded).
    /// 4. If cost > `requires_approval_above_usd` → `PendingReview`.
    /// 5. Default → `AutoApproved`.
    pub fn evaluate_request(&mut self, request: &SpendRequest) -> ApprovalStatus {
        let applicable = self.applicable_policies(&request.scope);

        // 1. Per-request hard limit.
        for policy in &applicable {
            if request.estimated_cost_usd > policy.per_request_limit_usd {
                let status = ApprovalStatus::Rejected {
                    reason: format!(
                        "Estimated cost ${:.4} exceeds per-request limit ${:.4} (policy: {})",
                        request.estimated_cost_usd, policy.per_request_limit_usd, policy.name
                    ),
                };
                self.store_record(request, status.clone());
                return status;
            }
        }

        // 2. Monthly limit check.
        for policy in &applicable {
            let spent = self.monthly_spend(&policy.scope, request.timestamp);
            if spent + request.estimated_cost_usd > policy.monthly_limit_usd {
                let status = ApprovalStatus::Rejected {
                    reason: format!(
                        "Monthly limit ${:.4} would be exceeded (currently ${:.4}, policy: {})",
                        policy.monthly_limit_usd, spent, policy.name
                    ),
                };
                self.store_record(request, status.clone());
                return status;
            }
        }

        // 3. Auto-approve threshold.
        let auto_threshold = applicable
            .iter()
            .map(|p| p.auto_approve_below_usd)
            .fold(f64::MAX, f64::min);
        if request.estimated_cost_usd <= auto_threshold {
            self.approved_spend.push((
                request.scope.to_string(),
                request.estimated_cost_usd,
                request.timestamp,
            ));
            let status = ApprovalStatus::AutoApproved;
            self.store_record(request, status.clone());
            return status;
        }

        // 4. Requires human review.
        let review_threshold = applicable
            .iter()
            .map(|p| p.requires_approval_above_usd)
            .fold(f64::MAX, f64::min);
        if request.estimated_cost_usd > review_threshold {
            let status = ApprovalStatus::PendingReview;
            self.store_record(request, status.clone());
            return status;
        }

        // 5. Default: auto-approve.
        self.approved_spend.push((
            request.scope.to_string(),
            request.estimated_cost_usd,
            request.timestamp,
        ));
        let status = ApprovalStatus::AutoApproved;
        self.store_record(request, status.clone());
        status
    }

    /// Approve a pending request.  Returns `true` if the request was found and
    /// was in `PendingReview` state.
    pub fn approve(&mut self, request_id: &str, approver: &str) -> bool {
        if let Some(record) = self.records.get_mut(request_id) {
            if record.status == ApprovalStatus::PendingReview {
                let cost = record.request.estimated_cost_usd;
                let scope = record.request.scope.to_string();
                let ts = record.request.timestamp;
                record.status = ApprovalStatus::Approved {
                    approver: approver.to_string(),
                };
                self.approved_spend.push((scope, cost, ts));
                return true;
            }
        }
        false
    }

    /// Reject a pending request.  Returns `true` if the request was found and
    /// was in `PendingReview` state.
    pub fn reject(&mut self, request_id: &str, reason: &str) -> bool {
        if let Some(record) = self.records.get_mut(request_id) {
            if record.status == ApprovalStatus::PendingReview {
                record.status = ApprovalStatus::Rejected {
                    reason: reason.to_string(),
                };
                return true;
            }
        }
        false
    }

    /// Return all requests currently awaiting review.
    pub fn pending_approvals(&self) -> Vec<&SpendRequest> {
        self.records
            .values()
            .filter(|r| r.status == ApprovalStatus::PendingReview)
            .map(|r| &r.request)
            .collect()
    }

    /// Sum of approved spend for `scope` within the current calendar month.
    ///
    /// `now_ms` is the current time in milliseconds since epoch; the "month"
    /// is defined as the 30-day window ending at `now_ms`.
    pub fn monthly_spend(&self, scope: &PolicyScope, now_ms: u64) -> f64 {
        const THIRTY_DAYS_MS: u64 = 30 * 24 * 60 * 60 * 1_000;
        let window_start = now_ms.saturating_sub(THIRTY_DAYS_MS);
        let scope_str = scope.to_string();
        self.approved_spend
            .iter()
            .filter(|(s, _, ts)| s == &scope_str && *ts >= window_start && *ts <= now_ms)
            .map(|(_, cost, _)| cost)
            .sum()
    }

    /// Return `(scope_display, reason)` tuples for every policy scope where the
    /// current monthly spend exceeds the configured limit.
    pub fn policy_violations(&self, now_ms: u64) -> Vec<(String, String)> {
        let mut violations = Vec::new();
        for policy in self.policies.values() {
            let spent = self.monthly_spend(&policy.scope, now_ms);
            if spent > policy.monthly_limit_usd {
                violations.push((
                    policy.scope.to_string(),
                    format!(
                        "Monthly spend ${:.4} exceeds limit ${:.4} (policy: {})",
                        spent, policy.monthly_limit_usd, policy.name
                    ),
                ));
            }
        }
        violations
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    fn applicable_policies(&self, scope: &PolicyScope) -> Vec<&GovernancePolicy> {
        self.policies
            .values()
            .filter(|p| &p.scope == scope || p.scope == PolicyScope::Global)
            .collect()
    }

    fn store_record(&mut self, request: &SpendRequest, status: ApprovalStatus) {
        self.records.insert(
            request.request_id.clone(),
            ApprovalRecord {
                request: request.clone(),
                status,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn global_policy(
        id: &str,
        monthly: f64,
        per_req: f64,
        auto_below: f64,
        review_above: f64,
    ) -> GovernancePolicy {
        GovernancePolicy {
            policy_id: id.to_string(),
            name: format!("policy-{}", id),
            scope: PolicyScope::Global,
            monthly_limit_usd: monthly,
            per_request_limit_usd: per_req,
            requires_approval_above_usd: review_above,
            auto_approve_below_usd: auto_below,
        }
    }

    fn team_policy(team: &str, monthly: f64, per_req: f64, auto_below: f64, review_above: f64) -> GovernancePolicy {
        GovernancePolicy {
            policy_id: format!("team-{}", team),
            name: format!("team-policy-{}", team),
            scope: PolicyScope::Team(team.to_string()),
            monthly_limit_usd: monthly,
            per_request_limit_usd: per_req,
            requires_approval_above_usd: review_above,
            auto_approve_below_usd: auto_below,
        }
    }

    fn req(id: &str, scope: PolicyScope, cost: f64, ts: u64) -> SpendRequest {
        SpendRequest {
            request_id: id.to_string(),
            requester_id: "user-1".to_string(),
            scope,
            estimated_cost_usd: cost,
            description: "test request".to_string(),
            timestamp: ts,
        }
    }

    #[test]
    fn auto_approved_below_threshold() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 50.0, 1.0, 10.0));
        let r = req("r1", PolicyScope::Global, 0.5, 0);
        assert_eq!(engine.evaluate_request(&r), ApprovalStatus::AutoApproved);
    }

    #[test]
    fn pending_review_above_threshold() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 50.0, 1.0, 5.0));
        let r = req("r1", PolicyScope::Global, 10.0, 0);
        assert_eq!(engine.evaluate_request(&r), ApprovalStatus::PendingReview);
    }

    #[test]
    fn rejected_per_request_limit() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 5.0, 1.0, 3.0));
        let r = req("r1", PolicyScope::Global, 10.0, 0);
        match engine.evaluate_request(&r) {
            ApprovalStatus::Rejected { .. } => {}
            other => panic!("Expected Rejected, got {:?}", other),
        }
    }

    #[test]
    fn rejected_monthly_limit() {
        let mut engine = GovernanceEngine::new();
        // Monthly limit of $2, auto-approve below $10 (so no per-request rejection).
        engine.register_policy(global_policy("p1", 2.0, 100.0, 10.0, 50.0));
        // First request: $1.50 → auto-approved, spend = $1.50.
        let r1 = req("r1", PolicyScope::Global, 1.5, 0);
        assert_eq!(engine.evaluate_request(&r1), ApprovalStatus::AutoApproved);
        // Second request: $1.00 → would bring total to $2.50 → rejected.
        let r2 = req("r2", PolicyScope::Global, 1.0, 0);
        match engine.evaluate_request(&r2) {
            ApprovalStatus::Rejected { .. } => {}
            other => panic!("Expected Rejected, got {:?}", other),
        }
    }

    #[test]
    fn approve_pending_request() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 50.0, 1.0, 5.0));
        let r = req("r1", PolicyScope::Global, 10.0, 0);
        engine.evaluate_request(&r);
        assert!(engine.approve("r1", "manager@co"));
        assert!(!engine.approve("r1", "manager@co")); // already approved
    }

    #[test]
    fn reject_pending_request() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 50.0, 1.0, 5.0));
        let r = req("r1", PolicyScope::Global, 10.0, 0);
        engine.evaluate_request(&r);
        assert!(engine.reject("r1", "too expensive"));
        assert!(!engine.reject("r1", "again")); // already rejected
    }

    #[test]
    fn pending_approvals_list() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 50.0, 1.0, 5.0));
        engine.evaluate_request(&req("r1", PolicyScope::Global, 10.0, 0));
        engine.evaluate_request(&req("r2", PolicyScope::Global, 20.0, 0));
        engine.evaluate_request(&req("r3", PolicyScope::Global, 0.5, 0)); // auto-approved
        let pending = engine.pending_approvals();
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn monthly_spend_accumulates() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 500.0, 100.0, 200.0));
        let now = 1_000_000u64;
        engine.evaluate_request(&req("r1", PolicyScope::Global, 10.0, now));
        engine.evaluate_request(&req("r2", PolicyScope::Global, 20.0, now));
        let spent = engine.monthly_spend(&PolicyScope::Global, now);
        assert!((spent - 30.0).abs() < 1e-9);
    }

    #[test]
    fn monthly_spend_ignores_old_entries() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 500.0, 100.0, 200.0));
        const THIRTY_ONE_DAYS_MS: u64 = 31 * 24 * 60 * 60 * 1_000;
        let now = THIRTY_ONE_DAYS_MS + 1_000;
        // Old request (outside 30-day window).
        engine.evaluate_request(&req("r1", PolicyScope::Global, 50.0, 0));
        // Recent request.
        engine.evaluate_request(&req("r2", PolicyScope::Global, 15.0, now));
        let spent = engine.monthly_spend(&PolicyScope::Global, now);
        assert!((spent - 15.0).abs() < 1e-9, "Expected 15.0, got {}", spent);
    }

    #[test]
    fn policy_violations_detected() {
        let mut engine = GovernanceEngine::new();
        // Very small monthly limit so it's easy to exceed.
        engine.register_policy(global_policy("p1", 5.0, 100.0, 50.0, 80.0));
        let now = 1_000u64;
        engine.evaluate_request(&req("r1", PolicyScope::Global, 10.0, now));
        // Force spend past the limit by directly inserting.
        engine.approved_spend.push(("global".to_string(), 10.0, now));
        let violations = engine.policy_violations(now);
        assert!(!violations.is_empty());
    }

    #[test]
    fn team_scope_policy() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(team_policy("ml", 100.0, 20.0, 5.0, 10.0));
        let r = req("r1", PolicyScope::Team("ml".to_string()), 3.0, 0);
        assert_eq!(engine.evaluate_request(&r), ApprovalStatus::AutoApproved);
        let r2 = req("r2", PolicyScope::Team("ml".to_string()), 15.0, 0);
        assert_eq!(engine.evaluate_request(&r2), ApprovalStatus::PendingReview);
    }

    #[test]
    fn approved_request_adds_to_monthly_spend() {
        let mut engine = GovernanceEngine::new();
        engine.register_policy(global_policy("p1", 1000.0, 50.0, 1.0, 5.0));
        let now = 0u64;
        engine.evaluate_request(&req("r1", PolicyScope::Global, 10.0, now));
        engine.approve("r1", "boss");
        let spent = engine.monthly_spend(&PolicyScope::Global, now);
        assert!((spent - 10.0).abs() < 1e-9);
    }

    #[test]
    fn policy_scope_display() {
        assert_eq!(PolicyScope::Global.to_string(), "global");
        assert_eq!(PolicyScope::Team("eng".to_string()).to_string(), "team:eng");
        assert_eq!(PolicyScope::Project("llm".to_string()).to_string(), "project:llm");
        assert_eq!(PolicyScope::User("alice".to_string()).to_string(), "user:alice");
    }
}
