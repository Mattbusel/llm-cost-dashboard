//! Per-user and per-team resource quota management.

use std::collections::HashMap;
use std::fmt;

/// The entity to which a quota applies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuotaScope {
    /// A specific user identified by their ID string.
    User(String),
    /// A specific team identified by their ID string.
    Team(String),
    /// A global quota applying to all traffic.
    Global,
}

impl fmt::Display for QuotaScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", scope_key(self))
    }
}

/// The time period over which a quota resets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaPeriod {
    /// Resets every hour.
    Hourly,
    /// Resets every calendar day (UTC midnight).
    Daily,
    /// Resets every 7-day block.
    Weekly,
    /// Resets every 30-day block.
    Monthly,
}

/// Upper bounds for a single quota scope/period combination.
#[derive(Debug, Clone)]
pub struct QuotaLimit {
    /// Maximum tokens (input + output) allowed per period.
    pub max_tokens: u64,
    /// Maximum number of API requests allowed per period.
    pub max_requests: u64,
    /// Maximum spend in USD allowed per period.
    pub max_cost_usd: f64,
    /// The period over which these limits apply.
    pub period: QuotaPeriod,
}

/// Accumulated usage within the current period.
#[derive(Debug, Clone, Default)]
pub struct QuotaUsage {
    /// Tokens consumed so far this period.
    pub tokens_used: u64,
    /// Requests made so far this period.
    pub requests_made: u64,
    /// Cost incurred so far this period (USD).
    pub cost_incurred: f64,
    /// Unix epoch millisecond at which the current period started.
    pub period_start_ms: u64,
}

/// A single quota entry combining limit, usage, and scope metadata.
#[derive(Debug, Clone)]
pub struct QuotaEntry {
    /// Who the quota belongs to.
    pub scope: QuotaScope,
    /// The enforced limits.
    pub limit: QuotaLimit,
    /// Current-period usage.
    pub usage: QuotaUsage,
}

/// Description of a quota limit that has been exceeded.
#[derive(Debug, Clone)]
pub struct QuotaViolation {
    /// Which scope was violated.
    pub scope: QuotaScope,
    /// Which dimension was violated: `"tokens"`, `"requests"`, or `"cost"`.
    pub dimension: String,
    /// Actual value that would be recorded after the request.
    pub used: f64,
    /// The limit that would be exceeded.
    pub limit: f64,
}

/// Derive the canonical string key for a [`QuotaScope`].
pub fn scope_key(scope: &QuotaScope) -> String {
    match scope {
        QuotaScope::User(id) => format!("user:{}", id),
        QuotaScope::Team(id) => format!("team:{}", id),
        QuotaScope::Global => "global".to_string(),
    }
}

/// Compute the start of the period containing `now_ms`.
pub fn period_start(period: &QuotaPeriod, now_ms: u64) -> u64 {
    match period {
        QuotaPeriod::Hourly => {
            let ms_per_hour = 3_600_000u64;
            (now_ms / ms_per_hour) * ms_per_hour
        }
        QuotaPeriod::Daily => {
            let ms_per_day = 86_400_000u64;
            (now_ms / ms_per_day) * ms_per_day
        }
        QuotaPeriod::Weekly => {
            let ms_per_week = 7 * 86_400_000u64;
            (now_ms / ms_per_week) * ms_per_week
        }
        QuotaPeriod::Monthly => {
            let ms_per_30_days = 30 * 86_400_000u64;
            (now_ms / ms_per_30_days) * ms_per_30_days
        }
    }
}

/// Manages quotas for multiple scopes.
pub struct ResourceQuotaManager {
    /// Keyed by `scope_key(scope)`.
    pub quotas: HashMap<String, QuotaEntry>,
}

impl ResourceQuotaManager {
    /// Create a new, empty quota manager.
    pub fn new() -> Self {
        Self {
            quotas: HashMap::new(),
        }
    }

    /// Register or replace the quota for `scope`.
    pub fn set_quota(&mut self, scope: QuotaScope, limit: QuotaLimit) {
        let key = scope_key(&scope);
        self.quotas.insert(
            key,
            QuotaEntry {
                scope,
                limit,
                usage: QuotaUsage::default(),
            },
        );
    }

    /// If the current period has elapsed, zero out the usage counters.
    pub fn reset_if_needed(&mut self, scope_key: &str, now_ms: u64) {
        if let Some(entry) = self.quotas.get_mut(scope_key) {
            let start = period_start(&entry.limit.period, now_ms);
            if start > entry.usage.period_start_ms {
                entry.usage = QuotaUsage {
                    period_start_ms: start,
                    ..Default::default()
                };
            }
        }
    }

    /// Record usage for `scope` and return any resulting violations.
    pub fn record_usage(
        &mut self,
        scope: &QuotaScope,
        tokens: u64,
        cost_usd: f64,
        now_ms: u64,
    ) -> Vec<QuotaViolation> {
        let key = scope_key(scope);
        self.reset_if_needed(&key, now_ms);

        let violations = if let Some(entry) = self.quotas.get(&key) {
            check_violations(scope, &entry.limit, &entry.usage, tokens, cost_usd)
        } else {
            Vec::new()
        };

        if let Some(entry) = self.quotas.get_mut(&key) {
            entry.usage.tokens_used += tokens;
            entry.usage.requests_made += 1;
            entry.usage.cost_incurred += cost_usd;
            if entry.usage.period_start_ms == 0 {
                entry.usage.period_start_ms = period_start(&entry.limit.period, now_ms);
            }
        }

        violations
    }

    /// Check whether a request would violate any quota, without modifying usage.
    pub fn check_allowed(
        &self,
        scope: &QuotaScope,
        tokens: u64,
        cost_usd: f64,
        now_ms: u64,
    ) -> Vec<QuotaViolation> {
        let key = scope_key(scope);
        if let Some(entry) = self.quotas.get(&key) {
            // Determine effective usage after potential period reset.
            let start = period_start(&entry.limit.period, now_ms);
            let usage = if start > entry.usage.period_start_ms {
                &QuotaUsage::default()
            } else {
                &entry.usage
            };
            check_violations(scope, &entry.limit, usage, tokens, cost_usd)
        } else {
            Vec::new()
        }
    }

    /// Return the percentage utilisation of each dimension for `scope`.
    /// Keys: `"tokens"`, `"requests"`, `"cost"`.
    pub fn utilization(&self, scope: &QuotaScope, now_ms: u64) -> HashMap<String, f64> {
        let key = scope_key(scope);
        let mut map = HashMap::new();
        if let Some(entry) = self.quotas.get(&key) {
            let start = period_start(&entry.limit.period, now_ms);
            let usage = if start > entry.usage.period_start_ms {
                QuotaUsage::default()
            } else {
                entry.usage.clone()
            };
            let token_pct = if entry.limit.max_tokens > 0 {
                usage.tokens_used as f64 / entry.limit.max_tokens as f64 * 100.0
            } else {
                0.0
            };
            let req_pct = if entry.limit.max_requests > 0 {
                usage.requests_made as f64 / entry.limit.max_requests as f64 * 100.0
            } else {
                0.0
            };
            let cost_pct = if entry.limit.max_cost_usd > 0.0 {
                usage.cost_incurred / entry.limit.max_cost_usd * 100.0
            } else {
                0.0
            };
            map.insert("tokens".to_string(), token_pct);
            map.insert("requests".to_string(), req_pct);
            map.insert("cost".to_string(), cost_pct);
        }
        map
    }

    /// Return a summary row for every registered quota.
    /// Each tuple: `(scope_key, token_pct, request_pct, cost_pct)`.
    pub fn quota_report(&self) -> Vec<(String, f64, f64, f64)> {
        self.quotas
            .iter()
            .map(|(key, entry)| {
                let token_pct = if entry.limit.max_tokens > 0 {
                    entry.usage.tokens_used as f64 / entry.limit.max_tokens as f64 * 100.0
                } else {
                    0.0
                };
                let req_pct = if entry.limit.max_requests > 0 {
                    entry.usage.requests_made as f64 / entry.limit.max_requests as f64 * 100.0
                } else {
                    0.0
                };
                let cost_pct = if entry.limit.max_cost_usd > 0.0 {
                    entry.usage.cost_incurred / entry.limit.max_cost_usd * 100.0
                } else {
                    0.0
                };
                (key.clone(), token_pct, req_pct, cost_pct)
            })
            .collect()
    }
}

impl Default for ResourceQuotaManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal helper: compute violations given current usage + proposed addition.
fn check_violations(
    scope: &QuotaScope,
    limit: &QuotaLimit,
    usage: &QuotaUsage,
    tokens: u64,
    cost_usd: f64,
) -> Vec<QuotaViolation> {
    let mut violations = Vec::new();

    let new_tokens = usage.tokens_used + tokens;
    if new_tokens > limit.max_tokens {
        violations.push(QuotaViolation {
            scope: scope.clone(),
            dimension: "tokens".to_string(),
            used: new_tokens as f64,
            limit: limit.max_tokens as f64,
        });
    }

    let new_requests = usage.requests_made + 1;
    if new_requests > limit.max_requests {
        violations.push(QuotaViolation {
            scope: scope.clone(),
            dimension: "requests".to_string(),
            used: new_requests as f64,
            limit: limit.max_requests as f64,
        });
    }

    let new_cost = usage.cost_incurred + cost_usd;
    if new_cost > limit.max_cost_usd {
        violations.push(QuotaViolation {
            scope: scope.clone(),
            dimension: "cost".to_string(),
            used: new_cost,
            limit: limit.max_cost_usd,
        });
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn daily_limit(max_tokens: u64, max_requests: u64, max_cost: f64) -> QuotaLimit {
        QuotaLimit {
            max_tokens,
            max_requests,
            max_cost_usd: max_cost,
            period: QuotaPeriod::Daily,
        }
    }

    #[test]
    fn test_set_and_check_quota_no_violation() {
        let mut mgr = ResourceQuotaManager::new();
        let scope = QuotaScope::User("alice".to_string());
        mgr.set_quota(scope.clone(), daily_limit(1_000, 10, 5.0));
        let violations = mgr.check_allowed(&scope, 100, 0.5, 86_400_000);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_violation_on_exceed() {
        let mut mgr = ResourceQuotaManager::new();
        let scope = QuotaScope::User("bob".to_string());
        mgr.set_quota(scope.clone(), daily_limit(10, 100, 1.0));
        // Record usage up to the limit
        mgr.record_usage(&scope, 10, 0.5, 86_400_000);
        // Now check: adding 1 more token should violate
        let violations = mgr.check_allowed(&scope, 1, 0.0, 86_400_000);
        assert!(violations.iter().any(|v| v.dimension == "tokens"));
    }

    #[test]
    fn test_period_reset_clears_usage() {
        let mut mgr = ResourceQuotaManager::new();
        let scope = QuotaScope::User("carol".to_string());
        mgr.set_quota(scope.clone(), daily_limit(100, 10, 10.0));
        // Record on day 1
        mgr.record_usage(&scope, 90, 9.0, 86_400_000);
        // Advance to day 2 (timestamp = 2 * 86_400_000)
        let key = scope_key(&scope);
        mgr.reset_if_needed(&key, 2 * 86_400_000);
        // Usage should now be reset; no violation for a large request
        let violations = mgr.check_allowed(&scope, 90, 9.0, 2 * 86_400_000);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_utilization_percentages() {
        let mut mgr = ResourceQuotaManager::new();
        let scope = QuotaScope::Team("eng".to_string());
        mgr.set_quota(scope.clone(), daily_limit(1000, 100, 10.0));
        mgr.record_usage(&scope, 500, 5.0, 86_400_000);
        let util = mgr.utilization(&scope, 86_400_000);
        assert!((util["tokens"] - 50.0).abs() < 1e-6);
        assert!((util["cost"] - 50.0).abs() < 1e-6);
        // 1 request out of 100 = 1%
        assert!((util["requests"] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_team_quota_tracking() {
        let mut mgr = ResourceQuotaManager::new();
        let team = QuotaScope::Team("marketing".to_string());
        mgr.set_quota(team.clone(), daily_limit(500, 50, 5.0));
        mgr.record_usage(&team, 100, 1.0, 86_400_000);
        mgr.record_usage(&team, 200, 2.0, 86_400_000);
        let entry = &mgr.quotas[&scope_key(&team)];
        assert_eq!(entry.usage.tokens_used, 300);
        assert_eq!(entry.usage.requests_made, 2);
        assert!((entry.usage.cost_incurred - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_cost_violation() {
        let mut mgr = ResourceQuotaManager::new();
        let scope = QuotaScope::Global;
        mgr.set_quota(scope.clone(), daily_limit(100_000, 10_000, 1.0));
        // Record just under the cost limit
        mgr.record_usage(&scope, 10, 0.99, 86_400_000);
        // Next request pushes cost over 1.0
        let violations = mgr.check_allowed(&scope, 1, 0.02, 86_400_000);
        assert!(violations.iter().any(|v| v.dimension == "cost"));
    }
}
