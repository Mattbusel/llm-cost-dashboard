//! Per-user / per-team hard and soft usage limits with overage handling.
//!
//! [`UsageLimiter`] tracks cumulative token and cost usage for arbitrary
//! [`LimitScope`]s (users, teams, or global) and enforces configurable soft
//! warning and hard block thresholds.  Accumulated usage can be reset with
//! [`UsageLimiter::reset_period`] when a billing period rolls over.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The scope to which a limit applies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LimitScope {
    /// A specific user, identified by their ID string.
    User(String),
    /// A team, identified by its name or ID string.
    Team(String),
    /// A single global limit shared across all scopes.
    Global,
}

/// Configured limits for one scope.
#[derive(Debug, Clone)]
pub struct UsageLimits {
    /// Soft token threshold — triggers a warning when exceeded.
    pub soft_token_limit: u64,
    /// Hard token threshold — blocks further requests when exceeded.
    pub hard_token_limit: u64,
    /// Soft cost threshold in USD — triggers a warning when exceeded.
    pub soft_cost_limit_usd: f64,
    /// Hard cost threshold in USD — blocks further requests when exceeded.
    pub hard_cost_limit_usd: f64,
    /// Duration of a billing period in seconds (informational; reset is manual).
    pub reset_period_secs: u64,
}

/// Status returned after recording usage or checking a scope.
#[derive(Debug, Clone, PartialEq)]
pub enum LimitStatus {
    /// Usage is below all soft limits.
    Ok,
    /// Usage exceeded a soft limit.
    SoftWarning {
        /// Fraction of the soft limit consumed (> 1.0 means exceeded).
        pct_used: f64,
    },
    /// Usage exceeded a hard limit — the request is blocked.
    HardBlocked {
        /// Human-readable reason describing which limit was breached.
        reason: String,
    },
}

/// Accumulated usage for a single scope.
#[derive(Debug, Default, Clone)]
struct Usage {
    tokens: u64,
    cost_usd: f64,
}

/// Thread-safe usage limiter.
#[derive(Debug, Clone)]
pub struct UsageLimiter {
    inner: Arc<Mutex<UsageLimiterInner>>,
}

#[derive(Debug, Default)]
struct UsageLimiterInner {
    limits: HashMap<LimitScope, UsageLimits>,
    usage: HashMap<LimitScope, Usage>,
}

impl UsageLimiter {
    /// Create a new, empty limiter.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(UsageLimiterInner::default())),
        }
    }

    /// Register or update the limits for a scope.
    pub fn set_limits(&self, scope: LimitScope, limits: UsageLimits) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.limits.insert(scope, limits);
    }

    /// Record `tokens` and `cost_usd` consumed by `scope` and return the
    /// resulting [`LimitStatus`].
    ///
    /// If no limits have been configured for this scope, [`LimitStatus::Ok`]
    /// is returned and the usage is still accumulated (so that limits set
    /// later still see historical consumption).
    pub fn record_usage(&self, scope: &LimitScope, tokens: u64, cost_usd: f64) -> LimitStatus {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let usage = guard.usage.entry(scope.clone()).or_default();
        usage.tokens += tokens;
        usage.cost_usd += cost_usd;
        let tokens_now = usage.tokens;
        let cost_now = usage.cost_usd;
        Self::evaluate(&guard, scope, tokens_now, cost_now)
    }

    /// Check the current status for `scope` without recording any new usage.
    pub fn check_status(&self, scope: &LimitScope) -> LimitStatus {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (tokens_now, cost_now) = guard
            .usage
            .get(scope)
            .map(|u| (u.tokens, u.cost_usd))
            .unwrap_or((0, 0.0));
        Self::evaluate(&guard, scope, tokens_now, cost_now)
    }

    /// Reset accumulated usage for `scope` (e.g. at the start of a new billing
    /// period).
    pub fn reset_period(&self, scope: &LimitScope) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(usage) = guard.usage.get_mut(scope) {
            usage.tokens = 0;
            usage.cost_usd = 0.0;
        }
    }

    /// Return all scopes whose current usage exceeds their soft limit, along
    /// with the fraction of the soft limit consumed (value > 1.0 means over).
    pub fn overage_report(&self) -> Vec<(LimitScope, f64)> {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut report = Vec::new();
        for (scope, limits) in &guard.limits {
            let usage = guard.usage.get(scope).cloned().unwrap_or_default();
            let token_pct = if limits.soft_token_limit > 0 {
                usage.tokens as f64 / limits.soft_token_limit as f64
            } else {
                0.0
            };
            let cost_pct = if limits.soft_cost_limit_usd > 0.0 {
                usage.cost_usd / limits.soft_cost_limit_usd
            } else {
                0.0
            };
            let max_pct = token_pct.max(cost_pct);
            if max_pct > 1.0 {
                report.push((scope.clone(), max_pct));
            }
        }
        report
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// Pure function: derive a `LimitStatus` from current usage and configured
    /// limits.  Hard limits take precedence over soft limits.
    fn evaluate(
        inner: &UsageLimiterInner,
        scope: &LimitScope,
        tokens: u64,
        cost_usd: f64,
    ) -> LimitStatus {
        let Some(limits) = inner.limits.get(scope) else {
            return LimitStatus::Ok;
        };

        // Hard block checks (take priority)
        if tokens >= limits.hard_token_limit {
            return LimitStatus::HardBlocked {
                reason: format!(
                    "token limit exceeded: {} >= {} hard limit",
                    tokens, limits.hard_token_limit
                ),
            };
        }
        if cost_usd >= limits.hard_cost_limit_usd {
            return LimitStatus::HardBlocked {
                reason: format!(
                    "cost limit exceeded: ${:.4} >= ${:.4} hard limit",
                    cost_usd, limits.hard_cost_limit_usd
                ),
            };
        }

        // Soft warning checks
        let token_pct = if limits.soft_token_limit > 0 {
            tokens as f64 / limits.soft_token_limit as f64
        } else {
            0.0
        };
        let cost_pct = if limits.soft_cost_limit_usd > 0.0 {
            cost_usd / limits.soft_cost_limit_usd
        } else {
            0.0
        };

        let max_pct = token_pct.max(cost_pct);
        if max_pct >= 1.0 {
            return LimitStatus::SoftWarning { pct_used: max_pct };
        }

        LimitStatus::Ok
    }
}

impl Default for UsageLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_limits(soft_tokens: u64, hard_tokens: u64, soft_cost: f64, hard_cost: f64) -> UsageLimits {
        UsageLimits {
            soft_token_limit: soft_tokens,
            hard_token_limit: hard_tokens,
            soft_cost_limit_usd: soft_cost,
            hard_cost_limit_usd: hard_cost,
            reset_period_secs: 86_400,
        }
    }

    #[test]
    fn ok_when_under_all_limits() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::User("alice".into());
        limiter.set_limits(scope.clone(), simple_limits(1_000, 2_000, 1.0, 2.0));
        let status = limiter.record_usage(&scope, 100, 0.05);
        assert_eq!(status, LimitStatus::Ok);
    }

    #[test]
    fn soft_warning_on_token_overage() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::User("bob".into());
        limiter.set_limits(scope.clone(), simple_limits(500, 2_000, 100.0, 200.0));
        let status = limiter.record_usage(&scope, 600, 0.01);
        match status {
            LimitStatus::SoftWarning { pct_used } => {
                assert!(pct_used > 1.0);
            }
            other => panic!("expected SoftWarning, got {:?}", other),
        }
    }

    #[test]
    fn soft_warning_on_cost_overage() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::Team("engineering".into());
        limiter.set_limits(scope.clone(), simple_limits(100_000, 200_000, 0.50, 2.0));
        let status = limiter.record_usage(&scope, 100, 0.75);
        match status {
            LimitStatus::SoftWarning { pct_used } => {
                assert!(pct_used > 1.0);
            }
            other => panic!("expected SoftWarning, got {:?}", other),
        }
    }

    #[test]
    fn hard_block_on_token_limit() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::User("charlie".into());
        limiter.set_limits(scope.clone(), simple_limits(500, 1_000, 100.0, 200.0));
        let status = limiter.record_usage(&scope, 1_001, 0.01);
        match status {
            LimitStatus::HardBlocked { reason } => {
                assert!(reason.contains("token limit"));
            }
            other => panic!("expected HardBlocked, got {:?}", other),
        }
    }

    #[test]
    fn hard_block_on_cost_limit() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::Global;
        limiter.set_limits(scope.clone(), simple_limits(1_000_000, 2_000_000, 1.0, 5.0));
        let status = limiter.record_usage(&scope, 1, 6.0);
        match status {
            LimitStatus::HardBlocked { reason } => {
                assert!(reason.contains("cost limit"));
            }
            other => panic!("expected HardBlocked, got {:?}", other),
        }
    }

    #[test]
    fn accumulation_across_calls() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::User("dave".into());
        limiter.set_limits(scope.clone(), simple_limits(1_000, 2_000, 10.0, 20.0));
        assert_eq!(limiter.record_usage(&scope, 400, 1.0), LimitStatus::Ok);
        assert_eq!(limiter.record_usage(&scope, 400, 1.0), LimitStatus::Ok);
        // Third call pushes tokens to 1200 (> soft 1_000)
        let status = limiter.record_usage(&scope, 400, 1.0);
        assert!(matches!(status, LimitStatus::SoftWarning { .. }));
    }

    #[test]
    fn reset_period_clears_usage() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::User("eve".into());
        limiter.set_limits(scope.clone(), simple_limits(500, 2_000, 1.0, 5.0));
        limiter.record_usage(&scope, 600, 0.0);
        limiter.reset_period(&scope);
        assert_eq!(limiter.check_status(&scope), LimitStatus::Ok);
    }

    #[test]
    fn check_status_reflects_accumulated() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::Team("ops".into());
        limiter.set_limits(scope.clone(), simple_limits(1_000, 5_000, 1.0, 5.0));
        limiter.record_usage(&scope, 500, 0.1);
        // Check without adding more
        assert_eq!(limiter.check_status(&scope), LimitStatus::Ok);
        limiter.record_usage(&scope, 600, 0.0);
        let s = limiter.check_status(&scope);
        assert!(matches!(s, LimitStatus::SoftWarning { .. }));
    }

    #[test]
    fn overage_report_includes_over_soft() {
        let limiter = UsageLimiter::new();
        let u1 = LimitScope::User("f".into());
        let u2 = LimitScope::User("g".into());
        limiter.set_limits(u1.clone(), simple_limits(100, 500, 1.0, 5.0));
        limiter.set_limits(u2.clone(), simple_limits(100, 500, 1.0, 5.0));
        limiter.record_usage(&u1, 150, 0.0); // over soft
        limiter.record_usage(&u2, 50, 0.0);  // under soft
        let report = limiter.overage_report();
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].0, u1);
    }

    #[test]
    fn ok_when_no_limits_configured() {
        let limiter = UsageLimiter::new();
        let scope = LimitScope::User("nobody".into());
        let status = limiter.record_usage(&scope, 999_999, 9999.0);
        assert_eq!(status, LimitStatus::Ok);
    }
}
