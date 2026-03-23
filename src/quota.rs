//! # Rolling Window Quota Enforcement
//!
//! Per-tenant/model quota enforcement using sliding time windows.
//!
//! ## Overview
//!
//! [`QuotaEnforcer`] stores a [`QuotaWindow`] per `"tenant:model"` key.  Each
//! window is a [`VecDeque`] of `(timestamp, tokens, cost)` events that fall
//! within the relevant time horizon.  On every [`QuotaEnforcer::check_and_record`]
//! call the enforcer:
//!
//! 1. Evicts entries older than the longest applicable window.
//! 2. Checks request-rate (per-minute), token (per-hour), and cost (per-day)
//!    limits in that order.
//! 3. If all limits pass, appends the new event and returns `Allowed`.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::quota::{QuotaConfig, QuotaEnforcer};
//!
//! let cfg = QuotaConfig {
//!     requests_per_minute: 10,
//!     tokens_per_hour: 100_000,
//!     cost_per_day: 5.0,
//!     burst_allowance: 2,
//! };
//! let enforcer = QuotaEnforcer::new(cfg);
//! let result = enforcer.check_and_record("acme:gpt-4o", 500, 0.02);
//! assert!(result.is_allowed());
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ── Config ────────────────────────────────────────────────────────────────────

/// Quota limits for a tenant/model combination.
#[derive(Debug, Clone)]
pub struct QuotaConfig {
    /// Maximum requests per 60-second sliding window.
    pub requests_per_minute: u64,
    /// Maximum tokens per 3600-second sliding window.
    pub tokens_per_hour: u64,
    /// Maximum spend (USD) per 86400-second sliding window.
    pub cost_per_day: f64,
    /// Extra requests allowed above `requests_per_minute` for short bursts.
    pub burst_allowance: u64,
}

impl Default for QuotaConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_hour: 1_000_000,
            cost_per_day: 100.0,
            burst_allowance: 10,
        }
    }
}

// ── Event ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Event {
    ts: Instant,
    tokens: u64,
    cost: f64,
}

// ── QuotaWindow ───────────────────────────────────────────────────────────────

/// Sliding-window event log for a single `tenant:model` key.
#[derive(Debug)]
pub struct QuotaWindow {
    events: VecDeque<Event>,
}

impl QuotaWindow {
    fn new() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }

    /// Remove events older than `horizon`.
    fn evict(&mut self, horizon: Duration) {
        let cutoff = Instant::now() - horizon;
        while let Some(e) = self.events.front() {
            if e.ts < cutoff {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Count requests in the last `window`.
    fn request_count(&self, window: Duration) -> u64 {
        let cutoff = Instant::now() - window;
        self.events.iter().filter(|e| e.ts >= cutoff).count() as u64
    }

    /// Sum tokens in the last `window`.
    fn token_sum(&self, window: Duration) -> u64 {
        let cutoff = Instant::now() - window;
        self.events
            .iter()
            .filter(|e| e.ts >= cutoff)
            .map(|e| e.tokens)
            .sum()
    }

    /// Sum cost in the last `window`.
    fn cost_sum(&self, window: Duration) -> f64 {
        let cutoff = Instant::now() - window;
        self.events
            .iter()
            .filter(|e| e.ts >= cutoff)
            .map(|e| e.cost)
            .sum()
    }

    fn push(&mut self, tokens: u64, cost: f64) {
        self.events.push_back(Event {
            ts: Instant::now(),
            tokens,
            cost,
        });
    }
}

// ── QuotaResult ───────────────────────────────────────────────────────────────

/// Reason a quota check was denied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaDeniedReason {
    /// Requests per minute (+ burst) exceeded.
    RequestRateExceeded,
    /// Tokens per hour exceeded.
    TokensExceeded,
    /// Cost per day exceeded.
    CostExceeded,
}

/// Result of a quota check-and-record operation.
#[derive(Debug, Clone)]
pub enum QuotaResult {
    /// Request is within quota limits.
    Allowed {
        /// Remaining requests this minute (including burst).
        remaining_requests: u64,
        /// Remaining tokens this hour.
        remaining_tokens: u64,
        /// Remaining spend budget today (USD).
        remaining_cost: f64,
    },
    /// Request would exceed a quota limit.
    Denied {
        /// Which limit was violated.
        reason: QuotaDeniedReason,
        /// Seconds to wait before retrying.
        retry_after_secs: u64,
    },
}

impl QuotaResult {
    /// Returns `true` if the result is `Allowed`.
    pub fn is_allowed(&self) -> bool {
        matches!(self, QuotaResult::Allowed { .. })
    }
}

// ── QuotaUsageReport ──────────────────────────────────────────────────────────

/// Current usage vs limits for a single key.
#[derive(Debug, Clone)]
pub struct QuotaUsageReport {
    /// Key this report is for.
    pub key: String,
    /// Requests used this minute.
    pub requests_used: u64,
    /// Request limit per minute (+ burst).
    pub requests_limit: u64,
    /// Request usage as a fraction of the limit (0–1).
    pub requests_pct: f64,
    /// Tokens used this hour.
    pub tokens_used: u64,
    /// Token limit per hour.
    pub tokens_limit: u64,
    /// Token usage as a fraction of the limit (0–1).
    pub tokens_pct: f64,
    /// Cost incurred today (USD).
    pub cost_used: f64,
    /// Daily cost limit (USD).
    pub cost_limit: f64,
    /// Cost usage as a fraction of the limit (0–1).
    pub cost_pct: f64,
}

// ── QuotaEnforcer ─────────────────────────────────────────────────────────────

/// Thread-safe quota enforcer backed by per-key sliding windows.
pub struct QuotaEnforcer {
    config: QuotaConfig,
    windows: Mutex<HashMap<String, QuotaWindow>>,
}

impl QuotaEnforcer {
    /// Create a new enforcer with the given quota configuration.
    pub fn new(config: QuotaConfig) -> Self {
        Self {
            config,
            windows: Mutex::new(HashMap::new()),
        }
    }

    /// Check whether the request fits within quota limits, and if so record it.
    ///
    /// `key` should be a `"tenant:model"` string.  `tokens` is the number of
    /// tokens consumed; `cost` is the USD cost.
    pub fn check_and_record(&self, key: &str, tokens: u64, cost: f64) -> QuotaResult {
        let mut map = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        let window = map.entry(key.to_string()).or_insert_with(QuotaWindow::new);

        let one_day = Duration::from_secs(86_400);
        window.evict(one_day);

        let one_min = Duration::from_secs(60);
        let one_hour = Duration::from_secs(3_600);

        let req_limit = self.config.requests_per_minute + self.config.burst_allowance;
        let reqs_used = window.request_count(one_min);
        if reqs_used >= req_limit {
            return QuotaResult::Denied {
                reason: QuotaDeniedReason::RequestRateExceeded,
                retry_after_secs: 60,
            };
        }

        let tok_used = window.token_sum(one_hour);
        if tok_used + tokens > self.config.tokens_per_hour {
            return QuotaResult::Denied {
                reason: QuotaDeniedReason::TokensExceeded,
                retry_after_secs: 3_600,
            };
        }

        let cost_used = window.cost_sum(one_day);
        if cost_used + cost > self.config.cost_per_day {
            return QuotaResult::Denied {
                reason: QuotaDeniedReason::CostExceeded,
                retry_after_secs: 86_400,
            };
        }

        window.push(tokens, cost);

        QuotaResult::Allowed {
            remaining_requests: req_limit - reqs_used - 1,
            remaining_tokens: self
                .config
                .tokens_per_hour
                .saturating_sub(tok_used + tokens),
            remaining_cost: self.config.cost_per_day - cost_used - cost,
        }
    }

    /// Return a usage report for the given key.  Returns a zero-usage report
    /// if the key has never been seen.
    pub fn usage_report(&self, key: &str) -> QuotaUsageReport {
        let mut map = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        let req_limit = self.config.requests_per_minute + self.config.burst_allowance;

        let (reqs_used, tok_used, cost_used) = if let Some(w) = map.get_mut(key) {
            w.evict(Duration::from_secs(86_400));
            (
                w.request_count(Duration::from_secs(60)),
                w.token_sum(Duration::from_secs(3_600)),
                w.cost_sum(Duration::from_secs(86_400)),
            )
        } else {
            (0, 0, 0.0)
        };

        let safe_pct = |used: f64, limit: f64| {
            if limit == 0.0 {
                0.0
            } else {
                (used / limit).min(1.0)
            }
        };

        QuotaUsageReport {
            key: key.to_string(),
            requests_used: reqs_used,
            requests_limit: req_limit,
            requests_pct: safe_pct(reqs_used as f64, req_limit as f64),
            tokens_used: tok_used,
            tokens_limit: self.config.tokens_per_hour,
            tokens_pct: safe_pct(tok_used as f64, self.config.tokens_per_hour as f64),
            cost_used,
            cost_limit: self.config.cost_per_day,
            cost_pct: safe_pct(cost_used, self.config.cost_per_day),
        }
    }

    /// Remove all window data for keys whose events have fully expired.
    ///
    /// Call periodically (e.g., from a background task) to prevent unbounded
    /// memory growth.
    pub fn cleanup(&self) {
        let horizon = Duration::from_secs(86_400);
        let mut map = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, w| {
            w.evict(horizon);
            !w.events.is_empty()
        });
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> QuotaConfig {
        QuotaConfig {
            requests_per_minute: 3,
            tokens_per_hour: 1_000,
            cost_per_day: 1.0,
            burst_allowance: 0,
        }
    }

    #[test]
    fn allows_within_limits() {
        let e = QuotaEnforcer::new(small_config());
        assert!(e.check_and_record("a:m", 100, 0.10).is_allowed());
    }

    #[test]
    fn denies_on_request_rate() {
        let e = QuotaEnforcer::new(small_config());
        e.check_and_record("a:m", 10, 0.01);
        e.check_and_record("a:m", 10, 0.01);
        e.check_and_record("a:m", 10, 0.01);
        let r = e.check_and_record("a:m", 10, 0.01);
        assert!(!r.is_allowed());
        if let QuotaResult::Denied { reason, .. } = r {
            assert_eq!(reason, QuotaDeniedReason::RequestRateExceeded);
        }
    }

    #[test]
    fn burst_allowance_permits_extra_requests() {
        let cfg = QuotaConfig {
            requests_per_minute: 2,
            burst_allowance: 1,
            ..small_config()
        };
        let e = QuotaEnforcer::new(cfg);
        assert!(e.check_and_record("a:m", 10, 0.01).is_allowed());
        assert!(e.check_and_record("a:m", 10, 0.01).is_allowed());
        assert!(e.check_and_record("a:m", 10, 0.01).is_allowed()); // burst
        assert!(!e.check_and_record("a:m", 10, 0.01).is_allowed()); // over
    }

    #[test]
    fn denies_on_token_limit() {
        let e = QuotaEnforcer::new(small_config());
        // First request uses all tokens.
        e.check_and_record("a:m", 900, 0.01);
        // Second request pushes over.
        let r = e.check_and_record("a:m", 200, 0.01);
        assert!(!r.is_allowed());
        if let QuotaResult::Denied { reason, .. } = r {
            assert_eq!(reason, QuotaDeniedReason::TokensExceeded);
        }
    }

    #[test]
    fn denies_on_cost_limit() {
        let e = QuotaEnforcer::new(small_config());
        e.check_and_record("a:m", 1, 0.90);
        let r = e.check_and_record("a:m", 1, 0.20);
        assert!(!r.is_allowed());
        if let QuotaResult::Denied { reason, .. } = r {
            assert_eq!(reason, QuotaDeniedReason::CostExceeded);
        }
    }

    #[test]
    fn usage_report_zero_for_unknown_key() {
        let e = QuotaEnforcer::new(small_config());
        let r = e.usage_report("unknown:key");
        assert_eq!(r.requests_used, 0);
        assert_eq!(r.tokens_used, 0);
        assert_eq!(r.cost_used, 0.0);
    }

    #[test]
    fn usage_report_reflects_activity() {
        let e = QuotaEnforcer::new(small_config());
        e.check_and_record("t:m", 300, 0.30);
        let r = e.usage_report("t:m");
        assert_eq!(r.requests_used, 1);
        assert_eq!(r.tokens_used, 300);
        assert!((r.cost_used - 0.30).abs() < 1e-9);
    }

    #[test]
    fn cleanup_removes_empty_windows() {
        let e = QuotaEnforcer::new(small_config());
        // Force an entry into the map (check_and_record adds it).
        e.check_and_record("x:y", 1, 0.001);
        // cleanup does NOT remove it because it has a recent event.
        e.cleanup();
        let map = e.windows.lock().unwrap();
        assert!(map.contains_key("x:y"));
    }

    #[test]
    fn remaining_counts_decrease() {
        let e = QuotaEnforcer::new(small_config());
        let r1 = e.check_and_record("a:m", 100, 0.01);
        if let QuotaResult::Allowed { remaining_requests, .. } = r1 {
            // 3 total (no burst), first used = 2 remaining
            assert_eq!(remaining_requests, 2);
        } else {
            panic!("should be allowed");
        }
    }
}
