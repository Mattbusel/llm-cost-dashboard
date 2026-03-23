//! # Event Sourcing Replay
//!
//! Append-only event store with sequence numbers, a [`Projector`] trait, and
//! two built-in projections: [`CostProjection`] and [`LatencyProjection`].
//!
//! ## Overview
//!
//! All domain events are variants of [`AuditEvent`].  They are appended to
//! an [`EventStore`] which assigns a monotonically increasing sequence number
//! to each event.  Any number of [`Projector`] implementations can replay the
//! full or partial history to rebuild derived state (total cost, latency
//! percentiles, etc.).
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::replay::{
//!     AuditEvent, EventStore, CostProjection, replay_all,
//! };
//! use std::sync::Arc;
//!
//! # tokio_test::block_on(async {
//! let store = Arc::new(EventStore::new());
//! store.append(AuditEvent::RequestReceived {
//!     id: "r1".to_string(),
//!     model: "gpt-4o".to_string(),
//!     tokens_in: 100,
//!     timestamp: 0,
//! }).await;
//! store.append(AuditEvent::ResponseCompleted {
//!     id: "r1".to_string(),
//!     tokens_out: 50,
//!     cost: 0.002,
//!     latency_ms: 300,
//! }).await;
//!
//! let mut proj = CostProjection::new();
//! replay_all(&store, &mut proj).await;
//! assert!((proj.total_cost - 0.002).abs() < 1e-9);
//! # });
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ── AuditEvent ────────────────────────────────────────────────────────────────

/// All auditable events in the system.
#[derive(Debug, Clone)]
pub enum AuditEvent {
    /// A new inference request was received.
    RequestReceived {
        /// Unique request identifier.
        id: String,
        /// Model name (e.g. `"gpt-4o"`).
        model: String,
        /// Number of prompt tokens.
        tokens_in: u64,
        /// Unix timestamp in seconds.
        timestamp: u64,
    },
    /// An inference response was produced.
    ResponseCompleted {
        /// Request identifier this response belongs to.
        id: String,
        /// Number of completion tokens.
        tokens_out: u64,
        /// USD cost of this request.
        cost: f64,
        /// End-to-end latency in milliseconds.
        latency_ms: u64,
    },
    /// An error occurred while processing a request.
    ErrorOccurred {
        /// Request identifier.
        id: String,
        /// Human-readable error message.
        error_msg: String,
    },
}

// ── EventStore ────────────────────────────────────────────────────────────────

/// Append-only, thread-safe sequence of [`AuditEvent`]s.
///
/// Each event is tagged with a monotonically increasing `u64` sequence number
/// starting at 1.
pub struct EventStore {
    inner: RwLock<Vec<(u64, AuditEvent)>>,
}

impl EventStore {
    /// Create an empty event store.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Vec::new()),
        }
    }

    /// Append `event` to the store and return its sequence number.
    pub async fn append(&self, event: AuditEvent) -> u64 {
        let mut guard = self.inner.write().await;
        let seq = guard.len() as u64 + 1;
        guard.push((seq, event));
        seq
    }

    /// Return all events with sequence number >= `seq`.
    pub async fn replay_from(&self, seq: u64) -> Vec<(u64, AuditEvent)> {
        let guard = self.inner.read().await;
        guard
            .iter()
            .filter(|(s, _)| *s >= seq)
            .cloned()
            .collect()
    }

    /// Return the total number of events stored.
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    /// Returns `true` if no events have been appended yet.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }
}

impl Default for EventStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Projector trait ───────────────────────────────────────────────────────────

/// Trait for types that can be built by replaying a sequence of [`AuditEvent`]s.
pub trait Projector {
    /// Apply a single event to update the projection's state.
    fn apply(&mut self, event: &AuditEvent);
}

// ── replay_all ────────────────────────────────────────────────────────────────

/// Feed every event in `store` through `projector` in sequence order.
pub async fn replay_all(store: &Arc<EventStore>, projector: &mut dyn Projector) {
    let events = store.replay_from(1).await;
    for (_, event) in &events {
        projector.apply(event);
    }
}

// ── CostProjection ────────────────────────────────────────────────────────────

/// Accumulates total cost and per-model cost breakdown.
#[derive(Debug, Default)]
pub struct CostProjection {
    /// Sum of all response costs seen so far.
    pub total_cost: f64,
    /// Cost broken down by model name.
    pub cost_by_model: HashMap<String, f64>,
    /// Total number of requests received.
    pub request_count: u64,
    /// Map from request-id to model name (used to attribute cost).
    request_model: HashMap<String, String>,
}

impl CostProjection {
    /// Create an empty projection.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Projector for CostProjection {
    fn apply(&mut self, event: &AuditEvent) {
        match event {
            AuditEvent::RequestReceived { id, model, .. } => {
                self.request_count += 1;
                self.request_model.insert(id.clone(), model.clone());
            }
            AuditEvent::ResponseCompleted { id, cost, .. } => {
                self.total_cost += cost;
                let model = self
                    .request_model
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                *self.cost_by_model.entry(model).or_insert(0.0) += cost;
            }
            AuditEvent::ErrorOccurred { .. } => {}
        }
    }
}

// ── LatencyProjection ─────────────────────────────────────────────────────────

/// Tracks observed latencies and exposes p50 / p95 / p99 percentiles.
///
/// Latency values are kept in a sorted `Vec<u64>` (milliseconds); percentiles
/// are recomputed on demand.
#[derive(Debug, Default)]
pub struct LatencyProjection {
    /// All observed latencies in milliseconds, kept sorted.
    latencies: Vec<u64>,
}

impl LatencyProjection {
    /// Create an empty projection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the value at the given percentile (0–100).
    ///
    /// Returns `None` if no latencies have been recorded.
    pub fn percentile(&self, pct: f64) -> Option<u64> {
        if self.latencies.is_empty() {
            return None;
        }
        let idx =
            ((pct / 100.0) * (self.latencies.len() - 1) as f64).round() as usize;
        Some(self.latencies[idx.min(self.latencies.len() - 1)])
    }

    /// 50th percentile latency in milliseconds.
    pub fn p50(&self) -> Option<u64> {
        self.percentile(50.0)
    }

    /// 95th percentile latency in milliseconds.
    pub fn p95(&self) -> Option<u64> {
        self.percentile(95.0)
    }

    /// 99th percentile latency in milliseconds.
    pub fn p99(&self) -> Option<u64> {
        self.percentile(99.0)
    }

    /// Number of latency samples recorded.
    pub fn sample_count(&self) -> usize {
        self.latencies.len()
    }
}

impl Projector for LatencyProjection {
    fn apply(&mut self, event: &AuditEvent) {
        if let AuditEvent::ResponseCompleted { latency_ms, .. } = event {
            // Insertion sort keeps the Vec sorted without an extra pass.
            let pos = self.latencies.partition_point(|&x| x <= *latency_ms);
            self.latencies.insert(pos, *latency_ms);
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn recv(id: &str, model: &str) -> AuditEvent {
        AuditEvent::RequestReceived {
            id: id.to_string(),
            model: model.to_string(),
            tokens_in: 50,
            timestamp: 0,
        }
    }

    fn completed(id: &str, cost: f64, latency_ms: u64) -> AuditEvent {
        AuditEvent::ResponseCompleted {
            id: id.to_string(),
            tokens_out: 20,
            cost,
            latency_ms,
        }
    }

    // ── EventStore ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn append_returns_sequential_seq_numbers() {
        let store = EventStore::new();
        let s1 = store.append(recv("r1", "gpt-4o")).await;
        let s2 = store.append(recv("r2", "gpt-4o")).await;
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
    }

    #[tokio::test]
    async fn replay_from_returns_correct_subset() {
        let store = EventStore::new();
        store.append(recv("r1", "m")).await;
        store.append(recv("r2", "m")).await;
        store.append(recv("r3", "m")).await;

        let all = store.replay_from(1).await;
        assert_eq!(all.len(), 3);

        let tail = store.replay_from(2).await;
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].0, 2);
    }

    #[tokio::test]
    async fn replay_from_beyond_end_returns_empty() {
        let store = EventStore::new();
        store.append(recv("r1", "m")).await;
        let result = store.replay_from(99).await;
        assert!(result.is_empty());
    }

    // ── CostProjection ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn cost_projection_accumulates_correctly() {
        let store = Arc::new(EventStore::new());
        store.append(recv("r1", "gpt-4o")).await;
        store.append(completed("r1", 0.01, 100)).await;
        store.append(recv("r2", "claude-3")).await;
        store.append(completed("r2", 0.02, 200)).await;

        let mut proj = CostProjection::new();
        replay_all(&store, &mut proj).await;

        assert!((proj.total_cost - 0.03).abs() < 1e-9);
        assert_eq!(proj.request_count, 2);
        assert!((proj.cost_by_model["gpt-4o"] - 0.01).abs() < 1e-9);
        assert!((proj.cost_by_model["claude-3"] - 0.02).abs() < 1e-9);
    }

    #[tokio::test]
    async fn cost_projection_ignores_errors() {
        let store = Arc::new(EventStore::new());
        store
            .append(AuditEvent::ErrorOccurred {
                id: "e1".to_string(),
                error_msg: "timeout".to_string(),
            })
            .await;

        let mut proj = CostProjection::new();
        replay_all(&store, &mut proj).await;
        assert_eq!(proj.total_cost, 0.0);
        assert_eq!(proj.request_count, 0);
    }

    // ── LatencyProjection ───────────────────────────────────────────────────

    #[tokio::test]
    async fn latency_projection_percentiles() {
        let store = Arc::new(EventStore::new());
        // Insert 100 latencies: 1ms … 100ms.
        for i in 1u64..=100 {
            store.append(recv(&format!("r{i}"), "m")).await;
            store.append(completed(&format!("r{i}"), 0.001, i)).await;
        }

        let mut proj = LatencyProjection::new();
        replay_all(&store, &mut proj).await;

        assert_eq!(proj.sample_count(), 100);
        // p50 ≈ 50 (within ±1 due to rounding)
        let p50 = proj.p50().unwrap();
        assert!(p50 >= 49 && p50 <= 51, "p50={p50}");
        // p99 ≈ 99
        let p99 = proj.p99().unwrap();
        assert!(p99 >= 98 && p99 <= 100, "p99={p99}");
    }

    #[tokio::test]
    async fn latency_projection_empty() {
        let proj = LatencyProjection::new();
        assert!(proj.p50().is_none());
        assert!(proj.p95().is_none());
        assert!(proj.p99().is_none());
    }

    // ── replay_from partial replay ──────────────────────────────────────────

    #[tokio::test]
    async fn partial_replay_from_seq() {
        let store = Arc::new(EventStore::new());
        store.append(recv("r1", "gpt-4o")).await;
        store.append(completed("r1", 0.05, 100)).await;
        // Second request — we'll replay only from this point.
        let seq = store.append(recv("r2", "claude")).await;
        store.append(completed("r2", 0.10, 200)).await;

        let tail = store.replay_from(seq).await;
        let mut proj = CostProjection::new();
        for (_, ev) in &tail {
            proj.apply(ev);
        }
        // Only r2's cost should be in the projection.
        assert!((proj.total_cost - 0.10).abs() < 1e-9);
    }
}
