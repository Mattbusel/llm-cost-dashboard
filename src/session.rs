//! # Session Tracking
//!
//! Per-session budget and cost tracking for LLM requests.
//!
//! A [`Session`] groups related requests together under a named scope with an
//! optional per-session budget.  The [`SessionLedger`] accumulates spend per
//! session and can raise a [`SessionBudgetAlert`] whenever a session exceeds
//! its configured limit.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::session::{Session, SessionLedger};
//!
//! let mut ledger = SessionLedger::new();
//!
//! // Register a session with a $0.50 budget.
//! ledger.register(Session::with_budget("my-experiment", 0.50));
//!
//! // Record spend against the session.
//! if let Some(alert) = ledger.record_spend("my-experiment", 0.30) {
//!     println!("Alert: {}", alert);
//! }
//! ```

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Session ───────────────────────────────────────────────────────────────────

/// A named session that can be used to group and budget LLM requests.
///
/// Sessions are identified by their [`name`][Session::name] field (used as the
/// lookup key in [`SessionLedger`]) and optionally bounded by
/// [`budget_usd`][Session::budget_usd].  A zero or negative budget is treated
/// as "unlimited".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier (auto-generated UUID v4).
    pub id: Uuid,
    /// Human-readable session name used as the lookup key.
    ///
    /// Passed via `--session` on the CLI or set programmatically.
    pub name: String,
    /// Optional per-session budget in USD.  `None` means no limit.
    pub budget_usd: Option<f64>,
    /// When the session was started (UTC).
    pub started_at: DateTime<Utc>,
    /// When the session ended, if it has been closed.
    pub ended_at: Option<DateTime<Utc>>,
}

impl Session {
    /// Create a new open session with no budget limit.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            budget_usd: None,
            started_at: Utc::now(),
            ended_at: None,
        }
    }

    /// Create a new open session with a per-session USD budget limit.
    pub fn with_budget(name: impl Into<String>, budget_usd: f64) -> Self {
        Self {
            budget_usd: Some(budget_usd),
            ..Self::new(name)
        }
    }

    /// Mark the session as ended at the current time.
    pub fn close(&mut self) {
        self.ended_at = Some(Utc::now());
    }

    /// Whether the session has been explicitly closed.
    pub fn is_closed(&self) -> bool {
        self.ended_at.is_some()
    }

    /// Duration the session has been open (from [`started_at`][Session::started_at]
    /// to either [`ended_at`][Session::ended_at] or now).
    pub fn elapsed(&self) -> chrono::Duration {
        let end = self.ended_at.unwrap_or_else(Utc::now);
        end - self.started_at
    }
}

// ── SessionBudgetAlert ────────────────────────────────────────────────────────

/// Raised when a session's cumulative spend exceeds its configured budget.
#[derive(Debug, Clone)]
pub struct SessionBudgetAlert {
    /// Name of the session that exceeded its budget.
    pub session_name: String,
    /// The budget that was configured for the session (USD).
    pub budget_usd: f64,
    /// The total spend accumulated so far this session (USD).
    pub spent_usd: f64,
}

impl SessionBudgetAlert {
    /// How far over budget the session is, in USD.
    pub fn overage_usd(&self) -> f64 {
        (self.spent_usd - self.budget_usd).max(0.0)
    }
}

impl std::fmt::Display for SessionBudgetAlert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Session '{}' exceeded budget: ${:.4} spent of ${:.4} limit (+${:.4} over)",
            self.session_name,
            self.spent_usd,
            self.budget_usd,
            self.overage_usd()
        )
    }
}

// ── SessionEntry (internal) ───────────────────────────────────────────────────

/// Internal bookkeeping for a registered session.
#[derive(Debug, Clone)]
struct SessionEntry {
    session: Session,
    /// Cumulative spend in USD recorded against this session.
    spent_usd: f64,
    /// Number of individual cost events recorded.
    record_count: u64,
}

// ── SessionLedger ─────────────────────────────────────────────────────────────

/// Tracks costs and budgets across multiple named sessions.
///
/// Sessions must be registered with [`SessionLedger::register`] before spend
/// can be recorded against them.  Unknown session names passed to
/// [`SessionLedger::record_spend`] are accepted and a new unbudgeted session is
/// created on-the-fly to avoid data loss.
#[derive(Debug, Default)]
pub struct SessionLedger {
    sessions: HashMap<String, SessionEntry>,
}

impl SessionLedger {
    /// Create an empty session ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a session.
    ///
    /// If a session with the same name already exists it is replaced.
    pub fn register(&mut self, session: Session) {
        let name = session.name.clone();
        self.sessions.insert(
            name,
            SessionEntry {
                session,
                spent_usd: 0.0,
                record_count: 0,
            },
        );
    }

    /// Record `cost_usd` of spend against `session_name`.
    ///
    /// If the session name is not registered a new unlimited session is created
    /// automatically.
    ///
    /// Returns a [`SessionBudgetAlert`] if the session now exceeds its budget,
    /// otherwise returns `None`.
    pub fn record_spend(
        &mut self,
        session_name: &str,
        cost_usd: f64,
    ) -> Option<SessionBudgetAlert> {
        let entry = self.sessions.entry(session_name.to_string()).or_insert_with(|| {
            SessionEntry {
                session: Session::new(session_name),
                spent_usd: 0.0,
                record_count: 0,
            }
        });

        entry.spent_usd += cost_usd;
        entry.record_count += 1;

        // Check budget.
        if let Some(budget) = entry.session.budget_usd {
            if entry.spent_usd > budget {
                return Some(SessionBudgetAlert {
                    session_name: session_name.to_string(),
                    budget_usd: budget,
                    spent_usd: entry.spent_usd,
                });
            }
        }

        None
    }

    /// Return a snapshot of all registered sessions sorted by total spend
    /// (highest first).
    pub fn sessions_by_spend(&self) -> Vec<SessionSnapshot> {
        let mut snaps: Vec<SessionSnapshot> = self
            .sessions
            .values()
            .map(|e| SessionSnapshot {
                name: e.session.name.clone(),
                budget_usd: e.session.budget_usd,
                spent_usd: e.spent_usd,
                record_count: e.record_count,
                is_closed: e.session.is_closed(),
                started_at: e.session.started_at,
            })
            .collect();

        snaps.sort_by(|a, b| {
            b.spent_usd
                .partial_cmp(&a.spent_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        snaps
    }

    /// Return the accumulated spend for `session_name`, or `None` if not found.
    pub fn spent_for(&self, session_name: &str) -> Option<f64> {
        self.sessions.get(session_name).map(|e| e.spent_usd)
    }

    /// Return the request count for `session_name`, or `None` if not found.
    pub fn record_count_for(&self, session_name: &str) -> Option<u64> {
        self.sessions.get(session_name).map(|e| e.record_count)
    }

    /// Total number of registered sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Close a session by name, setting its [`ended_at`][Session::ended_at]
    /// timestamp to now.
    ///
    /// Returns `false` if the session was not found.
    pub fn close_session(&mut self, session_name: &str) -> bool {
        if let Some(entry) = self.sessions.get_mut(session_name) {
            entry.session.close();
            true
        } else {
            false
        }
    }

    /// Remove all sessions and their accumulated spend.
    pub fn reset(&mut self) {
        self.sessions.clear();
    }
}

// ── SessionSnapshot ───────────────────────────────────────────────────────────

/// A read-only point-in-time view of a session's state, suitable for display.
#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    /// Session name.
    pub name: String,
    /// Optional per-session budget (USD).
    pub budget_usd: Option<f64>,
    /// Cumulative spend so far (USD).
    pub spent_usd: f64,
    /// Total number of cost events recorded.
    pub record_count: u64,
    /// Whether the session has been explicitly closed.
    pub is_closed: bool,
    /// When the session was started.
    pub started_at: DateTime<Utc>,
}

impl SessionSnapshot {
    /// Fraction of the session budget consumed, or `None` if no budget is set.
    ///
    /// Values > 1.0 indicate the session is over budget.
    pub fn budget_pct(&self) -> Option<f64> {
        self.budget_usd
            .filter(|&b| b > 0.0)
            .map(|b| self.spent_usd / b)
    }

    /// Remaining budget headroom in USD, or `None` if no budget is set.
    pub fn remaining(&self) -> Option<f64> {
        self.budget_usd.map(|b| (b - self.spent_usd).max(0.0))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_session_new_no_budget() {
        let s = Session::new("test");
        assert_eq!(s.name, "test");
        assert!(s.budget_usd.is_none());
        assert!(!s.is_closed());
    }

    #[test]
    fn test_session_with_budget() {
        let s = Session::with_budget("experiment", 1.0);
        assert_eq!(s.budget_usd, Some(1.0));
    }

    #[test]
    fn test_session_close() {
        let mut s = Session::new("test");
        assert!(!s.is_closed());
        s.close();
        assert!(s.is_closed());
        assert!(s.ended_at.is_some());
    }

    #[test]
    fn test_register_and_record_spend_no_budget() {
        let mut ledger = SessionLedger::new();
        ledger.register(Session::new("sess-a"));
        assert!(ledger.record_spend("sess-a", 0.50).is_none());
        assert!((ledger.spent_for("sess-a").unwrap() - 0.50).abs() < 1e-9);
    }

    #[test]
    fn test_record_spend_triggers_alert_when_over_budget() {
        let mut ledger = SessionLedger::new();
        ledger.register(Session::with_budget("bounded", 1.0));
        ledger.record_spend("bounded", 0.90);
        let alert = ledger.record_spend("bounded", 0.20).unwrap();
        assert_eq!(alert.session_name, "bounded");
        assert!((alert.overage_usd() - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_no_alert_when_spend_below_budget() {
        let mut ledger = SessionLedger::new();
        ledger.register(Session::with_budget("bounded", 10.0));
        assert!(ledger.record_spend("bounded", 1.0).is_none());
        assert!(ledger.record_spend("bounded", 1.0).is_none());
    }

    #[test]
    fn test_unknown_session_auto_created() {
        let mut ledger = SessionLedger::new();
        let alert = ledger.record_spend("unknown-session", 0.50);
        // No budget -> no alert
        assert!(alert.is_none());
        assert_eq!(ledger.session_count(), 1);
        assert!((ledger.spent_for("unknown-session").unwrap() - 0.50).abs() < 1e-9);
    }

    #[test]
    fn test_sessions_by_spend_sorted_descending() {
        let mut ledger = SessionLedger::new();
        ledger.register(Session::new("cheap"));
        ledger.register(Session::new("expensive"));
        ledger.record_spend("cheap", 0.10);
        ledger.record_spend("expensive", 1.00);
        let snaps = ledger.sessions_by_spend();
        assert_eq!(snaps[0].name, "expensive");
        assert_eq!(snaps[1].name, "cheap");
    }

    #[test]
    fn test_record_count_increments() {
        let mut ledger = SessionLedger::new();
        ledger.register(Session::new("s"));
        ledger.record_spend("s", 0.10);
        ledger.record_spend("s", 0.20);
        assert_eq!(ledger.record_count_for("s").unwrap(), 2);
    }

    #[test]
    fn test_close_session() {
        let mut ledger = SessionLedger::new();
        ledger.register(Session::new("s"));
        assert!(ledger.close_session("s"));
        // Closing a non-existent session returns false.
        assert!(!ledger.close_session("nope"));
    }

    #[test]
    fn test_reset_clears_all_sessions() {
        let mut ledger = SessionLedger::new();
        ledger.register(Session::new("a"));
        ledger.register(Session::new("b"));
        ledger.reset();
        assert_eq!(ledger.session_count(), 0);
    }

    #[test]
    fn test_snapshot_budget_pct() {
        let snap = SessionSnapshot {
            name: "s".into(),
            budget_usd: Some(2.0),
            spent_usd: 1.0,
            record_count: 1,
            is_closed: false,
            started_at: Utc::now(),
        };
        assert!((snap.budget_pct().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_snapshot_remaining() {
        let snap = SessionSnapshot {
            name: "s".into(),
            budget_usd: Some(5.0),
            spent_usd: 3.0,
            record_count: 1,
            is_closed: false,
            started_at: Utc::now(),
        };
        assert!((snap.remaining().unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_snapshot_remaining_none_when_no_budget() {
        let snap = SessionSnapshot {
            name: "s".into(),
            budget_usd: None,
            spent_usd: 3.0,
            record_count: 1,
            is_closed: false,
            started_at: Utc::now(),
        };
        assert!(snap.remaining().is_none());
    }

    #[test]
    fn test_alert_display() {
        let alert = SessionBudgetAlert {
            session_name: "test-session".into(),
            budget_usd: 1.0,
            spent_usd: 1.5,
        };
        let s = alert.to_string();
        assert!(s.contains("test-session"));
        assert!(s.contains("exceeded"));
    }
}
