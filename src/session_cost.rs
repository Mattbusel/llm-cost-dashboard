//! # Per-Session Cost Tracking with Budget Enforcement
//!
//! Tracks token usage and cost for individual LLM sessions, enforcing hard
//! budget limits and raising warnings when a configurable percentage of the
//! budget has been consumed.
//!
//! ## Key Types
//!
//! - [`CostSession`] — a single tracked session with budget + usage state.
//! - [`SessionManager`] — concurrent map of active sessions; thread-safe.
//! - [`SessionStatus`] — result returned after each request is recorded.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::session_cost::{SessionBudget, SessionManager};
//!
//! let mgr = SessionManager::new();
//! let id = mgr.create_session("claude-3-haiku", SessionBudget {
//!     max_cost: 1.00,
//!     max_tokens: 100_000,
//!     max_requests: 50,
//!     warn_at_pct: 0.80,
//! });
//! let status = mgr.record(&id, 500, 200, 0.01).unwrap();
//! println!("{status:?}");
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── SessionId ─────────────────────────────────────────────────────────────────

/// Opaque session identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(pub String);

impl SessionId {
    /// Create a new `SessionId` from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── SessionBudget ─────────────────────────────────────────────────────────────

/// Hard limits for a single session.
#[derive(Debug, Clone)]
pub struct SessionBudget {
    /// Maximum total cost in USD.  Exceeded → [`BudgetLimit::Cost`].
    pub max_cost: f64,
    /// Maximum total tokens (in + out).  Exceeded → [`BudgetLimit::Tokens`].
    pub max_tokens: u64,
    /// Maximum number of API requests.  Exceeded → [`BudgetLimit::Requests`].
    pub max_requests: u32,
    /// Fraction of any limit at which a warning is issued (e.g. `0.80` = 80%).
    pub warn_at_pct: f64,
}

// ── SessionUsage ──────────────────────────────────────────────────────────────

/// Accumulated usage for a session.
#[derive(Debug, Clone, Default)]
pub struct SessionUsage {
    /// Number of API requests recorded.
    pub requests: u32,
    /// Total input tokens.
    pub tokens_in: u64,
    /// Total output tokens.
    pub tokens_out: u64,
    /// Total cost in USD.
    pub cost: f64,
    /// Unix timestamp (seconds) when the first request was recorded.
    pub started_at: u64,
    /// Unix timestamp (seconds) of the most recent request.
    pub last_active_at: u64,
}

// ── BudgetLimit / SessionStatus ───────────────────────────────────────────────

/// Identifies which budget dimension was exhausted.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetLimit {
    /// The cost limit was reached.
    Cost,
    /// The token limit was reached.
    Tokens,
    /// The request-count limit was reached.
    Requests,
}

/// The status of a session at a given point in time.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    /// The session is within budget.
    Active,
    /// Usage has crossed `warn_at_pct` of at least one limit.
    WarnThresholdReached(f64 /* fraction of the most-consumed limit */),
    /// A hard budget limit has been reached.
    BudgetExhausted(BudgetLimit),
    /// The session was idle longer than the manager's `max_idle_secs`.
    Expired,
}

// ── SessionError ──────────────────────────────────────────────────────────────

/// Errors returned by [`SessionManager`].
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// The session ID was not found in the manager.
    #[error("session not found: {0}")]
    NotFound(String),
    /// The session has already exhausted its budget.
    #[error("session budget already exhausted")]
    BudgetExhausted,
}

// ── CostSession ───────────────────────────────────────────────────────────────

/// A single tracked session combining budget policy with live usage.
#[derive(Debug, Clone)]
pub struct CostSession {
    /// The session's unique identifier.
    pub id: SessionId,
    /// Budget limits for this session.
    pub budget: SessionBudget,
    /// Accumulated usage counters.
    pub usage: SessionUsage,
    /// The model used for inference in this session.
    pub model: String,
}

impl CostSession {
    /// Create a new session with zero usage.
    pub fn new(id: SessionId, model: impl Into<String>, budget: SessionBudget) -> Self {
        Self {
            id,
            budget,
            usage: SessionUsage::default(),
            model: model.into(),
        }
    }

    /// Record a single API request and return the updated session status.
    ///
    /// Increments all usage counters, then evaluates limits in order:
    /// `Cost → Tokens → Requests`.  The first exceeded limit wins.
    pub fn record_request(
        &mut self,
        tokens_in: u64,
        tokens_out: u64,
        cost: f64,
    ) -> SessionStatus {
        let now = now_unix_secs();
        if self.usage.started_at == 0 {
            self.usage.started_at = now;
        }
        self.usage.last_active_at = now;
        self.usage.tokens_in += tokens_in;
        self.usage.tokens_out += tokens_out;
        self.usage.cost += cost;
        self.usage.requests += 1;

        self.status()
    }

    /// Evaluate the current status without modifying usage.
    pub fn status(&self) -> SessionStatus {
        let total_tokens = self.usage.tokens_in + self.usage.tokens_out;

        // Check hard limits first.
        if self.usage.cost >= self.budget.max_cost {
            return SessionStatus::BudgetExhausted(BudgetLimit::Cost);
        }
        if total_tokens >= self.budget.max_tokens {
            return SessionStatus::BudgetExhausted(BudgetLimit::Tokens);
        }
        if self.usage.requests >= self.budget.max_requests {
            return SessionStatus::BudgetExhausted(BudgetLimit::Requests);
        }

        // Check warn threshold.
        let utilization = self.utilization();
        if utilization >= self.budget.warn_at_pct {
            return SessionStatus::WarnThresholdReached(utilization);
        }

        SessionStatus::Active
    }

    /// Return remaining budget headroom as `(cost_remaining, tokens_remaining, requests_remaining)`.
    pub fn remaining_budget(&self) -> (f64, u64, u32) {
        let total_tokens = self.usage.tokens_in + self.usage.tokens_out;
        let cost_rem = (self.budget.max_cost - self.usage.cost).max(0.0);
        let token_rem = self.budget.max_tokens.saturating_sub(total_tokens);
        let req_rem = self.budget.max_requests.saturating_sub(self.usage.requests);
        (cost_rem, token_rem, req_rem)
    }

    /// Return the utilization fraction (0.0–1.0) as the maximum of the three
    /// normalised dimensions.
    pub fn utilization(&self) -> f64 {
        let total_tokens = self.usage.tokens_in + self.usage.tokens_out;
        let cost_util = if self.budget.max_cost > 0.0 {
            self.usage.cost / self.budget.max_cost
        } else {
            0.0
        };
        let token_util = if self.budget.max_tokens > 0 {
            total_tokens as f64 / self.budget.max_tokens as f64
        } else {
            0.0
        };
        let req_util = if self.budget.max_requests > 0 {
            self.usage.requests as f64 / self.budget.max_requests as f64
        } else {
            0.0
        };
        cost_util.max(token_util).max(req_util)
    }
}

// ── SessionManager ────────────────────────────────────────────────────────────

/// Concurrent map of active [`CostSession`]s.
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<SessionId, CostSession>>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl SessionManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Create and register a new session, returning its [`SessionId`].
    pub fn create_session(&self, model: &str, budget: SessionBudget) -> SessionId {
        let seq = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = SessionId::new(format!("sess-{seq}"));
        let session = CostSession::new(id.clone(), model, budget);
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.clone(), session);
        id
    }

    /// Record a request for an existing session.
    ///
    /// Returns the updated [`SessionStatus`] or [`SessionError::NotFound`] if
    /// the session does not exist.  If the session is already exhausted the call
    /// still records the usage and returns the exhausted status — callers should
    /// gate inference before calling this if they want to prevent over-spend.
    pub fn record(
        &self,
        id: &SessionId,
        tokens_in: u64,
        tokens_out: u64,
        cost: f64,
    ) -> Result<SessionStatus, SessionError> {
        let mut map = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        let session = map
            .get_mut(id)
            .ok_or_else(|| SessionError::NotFound(id.0.clone()))?;
        Ok(session.record_request(tokens_in, tokens_out, cost))
    }

    /// Remove sessions that have been idle for longer than `max_idle_secs`.
    pub fn expire_old_sessions(&self, max_idle_secs: u64) {
        let now = now_unix_secs();
        let mut map = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_id, session| {
            if session.usage.last_active_at == 0 {
                // Never used — keep alive.
                return true;
            }
            now.saturating_sub(session.usage.last_active_at) <= max_idle_secs
        });
    }

    /// Return the number of sessions currently in the map.
    pub fn active_sessions(&self) -> usize {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Sum the total cost across all sessions.
    pub fn total_cost_all_sessions(&self) -> f64 {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| s.usage.cost)
            .sum()
    }

    /// Return a clone of the session for inspection, if it exists.
    pub fn get(&self, id: &SessionId) -> Option<CostSession> {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .cloned()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(max_cost: f64, max_tokens: u64, max_requests: u32) -> SessionBudget {
        SessionBudget {
            max_cost,
            max_tokens,
            max_requests,
            warn_at_pct: 0.80,
        }
    }

    #[test]
    fn active_while_within_budget() {
        let mgr = SessionManager::new();
        let id = mgr.create_session("claude-3", budget(10.0, 100_000, 100));
        let status = mgr.record(&id, 100, 50, 0.01).unwrap();
        assert_eq!(status, SessionStatus::Active);
    }

    #[test]
    fn budget_exhausted_on_cost() {
        let mgr = SessionManager::new();
        let id = mgr.create_session("gpt-4", budget(0.05, 100_000, 100));
        mgr.record(&id, 100, 100, 0.03).unwrap();
        let status = mgr.record(&id, 100, 100, 0.03).unwrap();
        assert_eq!(
            status,
            SessionStatus::BudgetExhausted(BudgetLimit::Cost)
        );
    }

    #[test]
    fn budget_exhausted_on_tokens() {
        let mgr = SessionManager::new();
        let id = mgr.create_session("m", budget(100.0, 500, 100));
        mgr.record(&id, 200, 200, 0.001).unwrap();
        let status = mgr.record(&id, 200, 200, 0.001).unwrap();
        assert_eq!(
            status,
            SessionStatus::BudgetExhausted(BudgetLimit::Tokens)
        );
    }

    #[test]
    fn budget_exhausted_on_requests() {
        let mgr = SessionManager::new();
        let id = mgr.create_session("m", budget(100.0, 1_000_000, 2));
        mgr.record(&id, 10, 10, 0.001).unwrap();
        mgr.record(&id, 10, 10, 0.001).unwrap();
        let status = mgr.record(&id, 10, 10, 0.001).unwrap();
        assert_eq!(
            status,
            SessionStatus::BudgetExhausted(BudgetLimit::Requests)
        );
    }

    #[test]
    fn warn_threshold_fires() {
        let mgr = SessionManager::new();
        let id = mgr.create_session("m", SessionBudget {
            max_cost: 1.0,
            max_tokens: 1_000_000,
            max_requests: 1_000,
            warn_at_pct: 0.80,
        });
        // 0.85 / 1.0 = 85% > 80%
        let status = mgr.record(&id, 100, 100, 0.85).unwrap();
        assert!(
            matches!(status, SessionStatus::WarnThresholdReached(pct) if pct >= 0.80),
            "expected WarnThresholdReached, got {status:?}"
        );
    }

    #[test]
    fn expiry_removes_old_sessions() {
        let mgr = SessionManager::new();
        let id = mgr.create_session("m", budget(10.0, 100_000, 100));
        // Record so last_active_at is set.
        mgr.record(&id, 10, 10, 0.001).unwrap();

        // Force last_active_at to be very old by directly manipulating session via the inner map.
        {
            let mut map = mgr.sessions.lock().unwrap();
            if let Some(session) = map.get_mut(&id) {
                session.usage.last_active_at = 1; // epoch + 1s
            }
        }

        mgr.expire_old_sessions(60); // 60-second window
        assert_eq!(mgr.active_sessions(), 0);
    }

    #[test]
    fn total_cost_aggregates() {
        let mgr = SessionManager::new();
        let id1 = mgr.create_session("m", budget(100.0, 1_000_000, 1_000));
        let id2 = mgr.create_session("m", budget(100.0, 1_000_000, 1_000));
        mgr.record(&id1, 100, 100, 1.50).unwrap();
        mgr.record(&id2, 100, 100, 2.50).unwrap();
        let total = mgr.total_cost_all_sessions();
        assert!((total - 4.0).abs() < 1e-9, "total={total}");
    }

    #[test]
    fn not_found_returns_error() {
        let mgr = SessionManager::new();
        let fake = SessionId::new("does-not-exist");
        assert!(matches!(
            mgr.record(&fake, 10, 10, 0.01),
            Err(SessionError::NotFound(_))
        ));
    }

    #[test]
    fn remaining_budget_decreases() {
        let id = SessionId::new("s");
        let mut session = CostSession::new(id, "m", budget(10.0, 1000, 10));
        session.record_request(100, 100, 2.0);
        let (cost_rem, token_rem, req_rem) = session.remaining_budget();
        assert!((cost_rem - 8.0).abs() < 1e-9);
        assert_eq!(token_rem, 800);
        assert_eq!(req_rem, 9);
    }

    #[test]
    fn utilization_is_max_of_dimensions() {
        let id = SessionId::new("s");
        let mut session = CostSession::new(id, "m", SessionBudget {
            max_cost: 10.0,
            max_tokens: 100,
            max_requests: 100,
            warn_at_pct: 0.80,
        });
        // tokens: 90/100 = 0.9, cost: 0.5/10 = 0.05, requests: 1/100 = 0.01
        session.record_request(45, 45, 0.5);
        let u = session.utilization();
        assert!((u - 0.9).abs() < 1e-9, "utilization={u}");
    }
}
