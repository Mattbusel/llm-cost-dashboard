//! Webhook Event System with Delivery Tracking
//!
//! Implements a full-featured webhook event pipeline:
//!
//! - [`WebhookEvent`] variants for cost/SLA/budget/anomaly events
//! - FNV-1a-based payload signing via [`WebhookPayload::sign`]
//! - [`WebhookSubscription`] with atomic delivery/failure counters
//! - [`WebhookManager`] for subscription management, event emission,
//!   simulated delivery, and aggregate statistics

use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// ── WebhookEvent ──────────────────────────────────────────────────────────────

/// Events that the dashboard can emit to subscribers.
#[derive(Debug, Clone)]
pub enum WebhookEvent {
    /// Actual spend exceeded a configured threshold.
    CostThresholdExceeded {
        /// Amount spent in USD.
        amount: f64,
        /// Threshold that was exceeded.
        threshold: f64,
    },
    /// An anomaly was detected in a cost or usage metric.
    AnomalyDetected {
        /// Anomaly severity score (higher = more anomalous).
        score: f64,
        /// Name of the metric that triggered the alert.
        metric: String,
    },
    /// Budget utilisation crossed an alert threshold.
    BudgetAlert {
        /// Fraction of budget consumed (0–1).
        pct_used: f64,
    },
    /// A new invoice was generated.
    InvoiceGenerated {
        /// Unique invoice identifier.
        invoice_id: String,
    },
    /// An SLA metric was breached.
    SlaBreached {
        /// Name of the SLA metric that was breached.
        metric: String,
    },
    /// Usage increased sharply.
    UsageSpike {
        /// Percentage increase relative to the baseline period.
        pct_increase: f64,
    },
}

impl WebhookEvent {
    /// A stable lowercase string identifying the event type.
    ///
    /// Used for subscription filtering (`events` field on [`WebhookSubscription`]).
    pub fn event_type(&self) -> &str {
        match self {
            WebhookEvent::CostThresholdExceeded { .. } => "cost_threshold_exceeded",
            WebhookEvent::AnomalyDetected { .. } => "anomaly_detected",
            WebhookEvent::BudgetAlert { .. } => "budget_alert",
            WebhookEvent::InvoiceGenerated { .. } => "invoice_generated",
            WebhookEvent::SlaBreached { .. } => "sla_breached",
            WebhookEvent::UsageSpike { .. } => "usage_spike",
        }
    }
}

// ── WebhookPayload ────────────────────────────────────────────────────────────

/// The payload delivered to a webhook endpoint.
#[derive(Debug, Clone)]
pub struct WebhookPayload {
    /// Unique delivery identifier (UUIDv4).
    pub event_id: String,
    /// The event that triggered this delivery.
    pub event: WebhookEvent,
    /// Unix timestamp (seconds) when the payload was created.
    pub timestamp_unix: u64,
    /// Optional tenant identifier for multi-tenant deployments.
    pub tenant_id: Option<String>,
    /// HMAC-like signature of the serialised payload body.
    pub signature: String,
}

impl WebhookPayload {
    /// Create a new payload, computing the signature from `secret`.
    pub fn new(event: WebhookEvent, tenant_id: Option<String>, secret: &str) -> Self {
        let event_id = Uuid::new_v4().to_string();
        let timestamp_unix = unix_now();
        let body = format!(
            "{}:{}:{}",
            event_id,
            timestamp_unix,
            tenant_id.as_deref().unwrap_or("")
        );
        let signature = Self::sign(&body, secret);
        Self {
            event_id,
            event,
            timestamp_unix,
            tenant_id,
            signature,
        }
    }

    /// Compute an FNV-1a-based HMAC-like signature.
    ///
    /// The key is split into 8-byte blocks; each block is XOR-folded with the
    /// FNV-1a hash of the payload, then all block hashes are XOR-combined.
    /// This is intentionally simple (no crypto library dependency); production
    /// code should replace this with HMAC-SHA256.
    pub fn sign(payload: &str, secret: &str) -> String {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;

        // FNV-1a of the payload.
        let payload_hash: u64 = payload
            .bytes()
            .fold(FNV_OFFSET, |acc, b| (acc ^ b as u64).wrapping_mul(FNV_PRIME));

        // Split secret into 8-byte blocks and XOR-fold with payload hash.
        let secret_bytes = secret.as_bytes();
        let block_size = 8usize;
        let mut combined: u64 = 0;

        let blocks = if secret_bytes.is_empty() {
            1
        } else {
            (secret_bytes.len() + block_size - 1) / block_size
        };

        for block_idx in 0..blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(secret_bytes.len());
            let block = if start < secret_bytes.len() {
                &secret_bytes[start..end]
            } else {
                &[]
            };

            // FNV-1a of the block XOR'd with the payload hash.
            let block_hash: u64 = block
                .iter()
                .fold(FNV_OFFSET ^ payload_hash, |acc, &b| {
                    (acc ^ b as u64).wrapping_mul(FNV_PRIME)
                });
            combined ^= block_hash;
        }

        format!("fnv1a={:016x}", combined)
    }
}

// ── DeliveryAttempt ───────────────────────────────────────────────────────────

/// Record of a single delivery attempt.
#[derive(Debug, Clone)]
pub struct DeliveryAttempt {
    /// 1-based attempt number.
    pub attempt_num: u32,
    /// Unix timestamp of this attempt.
    pub timestamp_unix: u64,
    /// HTTP response code returned by the endpoint (if any).
    pub response_code: Option<u16>,
    /// Whether the attempt was considered successful.
    pub success: bool,
    /// Error description if the attempt failed.
    pub error: Option<String>,
}

// ── WebhookSubscription ───────────────────────────────────────────────────────

/// A registered webhook subscription.
#[derive(Debug)]
pub struct WebhookSubscription {
    /// Unique subscription identifier.
    pub id: String,
    /// Target URL for delivery.
    pub url: String,
    /// Signing secret.
    pub secret: String,
    /// Event types to receive (empty = all events).
    pub events: Vec<String>,
    /// Whether the subscription is currently active.
    pub active: bool,
    /// Unix timestamp when the subscription was created.
    pub created_at: u64,
    /// Total deliveries attempted.
    pub delivery_count: AtomicU64,
    /// Total deliveries that failed.
    pub failure_count: AtomicU64,
}

impl WebhookSubscription {
    fn new(url: &str, secret: &str, events: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            url: url.to_string(),
            secret: secret.to_string(),
            events,
            active: true,
            created_at: unix_now(),
            delivery_count: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
        }
    }

    /// Returns `true` if this subscription wants to receive `event_type`.
    pub fn wants_event(&self, event_type: &str) -> bool {
        self.events.is_empty() || self.events.iter().any(|e| e == event_type)
    }
}

// ── DeliveryStatus ────────────────────────────────────────────────────────────

/// Final outcome of a [`WebhookDelivery`].
#[derive(Debug, Clone)]
pub enum DeliveryStatus {
    /// Not yet processed.
    Pending,
    /// Successfully delivered.
    Delivered,
    /// All attempts failed with the given reason.
    Failed(String),
    /// Delivery abandoned after max retries.
    Abandoned,
}

// ── WebhookDelivery ───────────────────────────────────────────────────────────

/// Tracks all delivery attempts for a single payload to a single subscription.
#[derive(Debug, Clone)]
pub struct WebhookDelivery {
    /// The subscription this delivery targets.
    pub subscription_id: String,
    /// The payload to deliver.
    pub payload: WebhookPayload,
    /// History of all delivery attempts.
    pub attempts: Vec<DeliveryAttempt>,
    /// Final delivery status.
    pub final_status: DeliveryStatus,
}

// ── WebhookStats ──────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`WebhookManager`].
#[derive(Debug, Clone)]
pub struct WebhookStats {
    /// Number of active subscriptions.
    pub total_subscriptions: usize,
    /// Total deliveries ever attempted.
    pub total_deliveries: u64,
    /// Fraction of deliveries that succeeded (0–1).
    pub success_rate: f64,
    /// Average number of attempts per delivery.
    pub avg_attempts: f64,
    /// Deliveries currently pending in the queue.
    pub pending: usize,
}

// ── WebhookManager ────────────────────────────────────────────────────────────

/// Manages webhook subscriptions and delivery queues.
pub struct WebhookManager {
    subscriptions: Arc<DashMap<String, WebhookSubscription>>,
    delivery_queue: Arc<Mutex<VecDeque<WebhookDelivery>>>,
    total_deliveries: AtomicU64,
    total_successes: AtomicU64,
    total_attempts: AtomicU64,
}

impl Default for WebhookManager {
    fn default() -> Self {
        Self::new()
    }
}

impl WebhookManager {
    /// Create a new manager with empty subscription and delivery stores.
    pub fn new() -> Self {
        Self {
            subscriptions: Arc::new(DashMap::new()),
            delivery_queue: Arc::new(Mutex::new(VecDeque::new())),
            total_deliveries: AtomicU64::new(0),
            total_successes: AtomicU64::new(0),
            total_attempts: AtomicU64::new(0),
        }
    }

    /// Register a new webhook subscription.
    ///
    /// Returns the new subscription's unique ID.
    pub fn subscribe(&self, url: &str, secret: &str, events: Vec<String>) -> String {
        let sub = WebhookSubscription::new(url, secret, events);
        let id = sub.id.clone();
        self.subscriptions.insert(id.clone(), sub);
        id
    }

    /// Remove a subscription by ID. Does nothing if not found.
    pub fn unsubscribe(&self, id: &str) {
        self.subscriptions.remove(id);
    }

    /// Emit an event, enqueuing deliveries for all matching active subscriptions.
    ///
    /// The delivery queue is capped at 1 000 entries; older entries are evicted
    /// when the cap is reached.
    pub fn emit(&self, event: WebhookEvent, tenant_id: Option<&str>) {
        let event_type = event.event_type().to_string();

        let mut deliveries: Vec<WebhookDelivery> = self
            .subscriptions
            .iter()
            .filter(|entry| entry.active && entry.wants_event(&event_type))
            .map(|entry| {
                let payload =
                    WebhookPayload::new(event.clone(), tenant_id.map(str::to_string), &entry.secret);
                WebhookDelivery {
                    subscription_id: entry.id.clone(),
                    payload,
                    attempts: Vec::new(),
                    final_status: DeliveryStatus::Pending,
                }
            })
            .collect();

        let mut queue = self.delivery_queue.lock().unwrap_or_else(|e| e.into_inner());
        for delivery in deliveries.drain(..) {
            // Evict oldest entry if at capacity.
            if queue.len() >= 1_000 {
                queue.pop_front();
            }
            queue.push_back(delivery);
        }
    }

    /// Simulate HTTP delivery for `delivery`, recording up to 3 attempts.
    ///
    /// Delivery success is determined by a deterministic pseudo-random function
    /// of the subscription ID and attempt number (no actual HTTP is performed).
    /// In a production implementation this would perform real HTTP POST requests.
    pub fn simulate_delivery(&self, delivery: &mut WebhookDelivery) {
        const MAX_ATTEMPTS: u32 = 3;

        for attempt_num in 1..=MAX_ATTEMPTS {
            self.total_attempts.fetch_add(1, Ordering::Relaxed);

            // Pseudo-random success: hash subscription_id + attempt_num.
            let hash = fnv1a_hash(
                format!("{}:{}", delivery.subscription_id, attempt_num).as_bytes(),
            );
            // ~70 % success rate on first attempt, ~85 % on subsequent.
            let threshold: u64 = if attempt_num == 1 { u64::MAX / 10 * 7 } else { u64::MAX / 100 * 85 };
            let success = hash < threshold;
            let response_code: Option<u16> = if success { Some(200) } else { Some(500) };

            let attempt = DeliveryAttempt {
                attempt_num,
                timestamp_unix: unix_now(),
                response_code,
                success,
                error: if success {
                    None
                } else {
                    Some(format!("server returned {}", response_code.unwrap_or(0)))
                },
            };

            delivery.attempts.push(attempt);

            if success {
                delivery.final_status = DeliveryStatus::Delivered;
                self.total_deliveries.fetch_add(1, Ordering::Relaxed);
                self.total_successes.fetch_add(1, Ordering::Relaxed);

                // Update subscription counters.
                if let Some(sub) = self.subscriptions.get(&delivery.subscription_id) {
                    sub.delivery_count.fetch_add(1, Ordering::Relaxed);
                }
                return;
            }
        }

        // All attempts failed.
        delivery.final_status = DeliveryStatus::Abandoned;
        self.total_deliveries.fetch_add(1, Ordering::Relaxed);

        if let Some(sub) = self.subscriptions.get(&delivery.subscription_id) {
            sub.delivery_count.fetch_add(1, Ordering::Relaxed);
            sub.failure_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Number of deliveries currently pending in the queue.
    pub fn pending_deliveries(&self) -> usize {
        let queue = self.delivery_queue.lock().unwrap_or_else(|e| e.into_inner());
        queue
            .iter()
            .filter(|d| matches!(d.final_status, DeliveryStatus::Pending))
            .count()
    }

    /// Snapshot aggregate statistics.
    pub fn delivery_stats(&self) -> WebhookStats {
        let total_deliveries = self.total_deliveries.load(Ordering::Relaxed);
        let total_successes = self.total_successes.load(Ordering::Relaxed);
        let total_attempts = self.total_attempts.load(Ordering::Relaxed);

        let success_rate = if total_deliveries == 0 {
            0.0
        } else {
            total_successes as f64 / total_deliveries as f64
        };

        let avg_attempts = if total_deliveries == 0 {
            0.0
        } else {
            total_attempts as f64 / total_deliveries as f64
        };

        WebhookStats {
            total_subscriptions: self.subscriptions.len(),
            total_deliveries,
            success_rate,
            avg_attempts,
            pending: self.pending_deliveries(),
        }
    }

    /// Drain and process all pending deliveries in the queue.
    ///
    /// Useful for a background processing task or test helpers.
    pub fn process_pending(&self) {
        let pending: Vec<WebhookDelivery> = {
            let mut queue = self.delivery_queue.lock().unwrap_or_else(|e| e.into_inner());
            let pending: Vec<WebhookDelivery> = queue
                .drain(..)
                .filter(|d| matches!(d.final_status, DeliveryStatus::Pending))
                .collect();
            pending
        };

        for mut delivery in pending {
            self.simulate_delivery(&mut delivery);
            // Re-enqueue completed deliveries for audit trail.
            let mut queue = self.delivery_queue.lock().unwrap_or_else(|e| e.into_inner());
            if queue.len() < 1_000 {
                queue.push_back(delivery);
            }
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    data.iter()
        .fold(FNV_OFFSET, |acc, &b| (acc ^ b as u64).wrapping_mul(FNV_PRIME))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_type_strings() {
        assert_eq!(
            WebhookEvent::CostThresholdExceeded { amount: 1.0, threshold: 0.5 }.event_type(),
            "cost_threshold_exceeded"
        );
        assert_eq!(
            WebhookEvent::BudgetAlert { pct_used: 0.8 }.event_type(),
            "budget_alert"
        );
        assert_eq!(
            WebhookEvent::UsageSpike { pct_increase: 50.0 }.event_type(),
            "usage_spike"
        );
    }

    #[test]
    fn sign_deterministic() {
        let s1 = WebhookPayload::sign("payload", "secret");
        let s2 = WebhookPayload::sign("payload", "secret");
        assert_eq!(s1, s2);
        assert!(s1.starts_with("fnv1a="));
    }

    #[test]
    fn sign_differs_for_different_secrets() {
        let s1 = WebhookPayload::sign("payload", "secret1");
        let s2 = WebhookPayload::sign("payload", "secret2");
        assert_ne!(s1, s2);
    }

    #[test]
    fn subscribe_and_unsubscribe() {
        let mgr = WebhookManager::new();
        let id = mgr.subscribe("https://example.com/hook", "secret", vec![]);
        assert_eq!(mgr.delivery_stats().total_subscriptions, 1);
        mgr.unsubscribe(&id);
        assert_eq!(mgr.delivery_stats().total_subscriptions, 0);
    }

    #[test]
    fn emit_enqueues_for_matching_subscriptions() {
        let mgr = WebhookManager::new();
        mgr.subscribe(
            "https://a.example.com",
            "s",
            vec!["budget_alert".to_string()],
        );
        mgr.subscribe(
            "https://b.example.com",
            "s",
            vec!["anomaly_detected".to_string()],
        );

        mgr.emit(WebhookEvent::BudgetAlert { pct_used: 0.9 }, None);
        // Only the first subscription matches.
        assert_eq!(mgr.pending_deliveries(), 1);
    }

    #[test]
    fn simulate_delivery_updates_stats() {
        let mgr = WebhookManager::new();
        mgr.subscribe("https://example.com", "s", vec![]);
        mgr.emit(WebhookEvent::UsageSpike { pct_increase: 200.0 }, None);
        mgr.process_pending();

        let stats = mgr.delivery_stats();
        assert!(stats.total_deliveries > 0);
    }

    #[test]
    fn queue_cap_at_1000() {
        let mgr = WebhookManager::new();
        mgr.subscribe("https://example.com", "s", vec![]);
        for _ in 0..1_100 {
            mgr.emit(WebhookEvent::BudgetAlert { pct_used: 0.5 }, None);
        }
        // Queue should not exceed 1000.
        let queue = mgr.delivery_queue.lock().unwrap();
        assert!(queue.len() <= 1_000);
    }

    #[test]
    fn wants_event_empty_events_matches_all() {
        let sub = WebhookSubscription::new("https://x.com", "s", vec![]);
        assert!(sub.wants_event("budget_alert"));
        assert!(sub.wants_event("anything"));
    }

    #[test]
    fn wants_event_filtered() {
        let sub = WebhookSubscription::new(
            "https://x.com",
            "s",
            vec!["budget_alert".to_string()],
        );
        assert!(sub.wants_event("budget_alert"));
        assert!(!sub.wants_event("anomaly_detected"));
    }
}
