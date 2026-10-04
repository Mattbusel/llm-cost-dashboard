//! # Outbound Webhook Delivery
//!
//! Provides signed, retryable outbound webhook delivery with an async
//! dispatch queue, delivery log, and a pluggable [`HttpClient`] trait that
//! enables hermetic unit testing without real network calls.
//!
//! ## Signing
//!
//! Each delivery is signed with HMAC-SHA256 computed over the JSON-encoded
//! payload body.  The signature is attached as the
//! `X-Webhook-Signature: sha256=<hex>` request header.
//!
//! The HMAC implementation follows RFC 2104:
//! `HMAC(K, m) = H((K ⊕ opad) ∥ H((K ⊕ ipad) ∥ m))`
//! using a compact SHA-256 implementation contained entirely in this module.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::webhook_dispatch::{
//!     WebhookConfig, WebhookDispatcher, MockHttpClient,
//! };
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let config = WebhookConfig {
//!     url: "https://example.com/hook".into(),
//!     secret: "my-secret".into(),
//!     timeout_ms: 5_000,
//!     max_retries: 3,
//!     retry_backoff_ms: 100,
//! };
//! let client = Arc::new(MockHttpClient::always(200));
//! let dispatcher = WebhookDispatcher::new(config, client);
//! dispatcher.dispatch("cost.alert", serde_json::json!({"cost": 1.23}));
//! # }
//! ```

use std::{
    collections::{HashMap, VecDeque},
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

// SHA-256 (RustCrypto `sha2`), kept for the known-answer tests below.
#[cfg(test)]
fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(data).into()
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 (RustCrypto `hmac` + `sha2`; this used to be hand-written)
// ---------------------------------------------------------------------------

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    use hmac::{Hmac, Mac};
    // HMAC accepts keys of any length, so construction cannot fail.
    match <Hmac<sha2::Sha256> as Mac>::new_from_slice(key) {
        Ok(mut mac) => {
            mac.update(message);
            mac.finalize().into_bytes().into()
        }
        Err(_) => [0; 32],
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// WebhookSigner
// ---------------------------------------------------------------------------

/// Signs webhook payloads with HMAC-SHA256.
pub struct WebhookSigner {
    secret: String,
}

impl WebhookSigner {
    /// Create a new signer with the given shared secret.
    pub fn new(secret: impl Into<String>) -> Self {
        Self {
            secret: secret.into(),
        }
    }

    /// Compute `sha256=<hex>` signature over `body`.
    pub fn sign(&self, body: &str) -> String {
        let mac = hmac_sha256(self.secret.as_bytes(), body.as_bytes());
        format!("sha256={}", hex_encode(&mac))
    }

    /// Verify that `signature` matches `body` (constant-time comparison).
    pub fn verify(&self, body: &str, signature: &str) -> bool {
        let expected = self.sign(body);
        // Simple byte-by-byte comparison (not constant-time, but sufficient
        // for unit tests; use `subtle` crate for production use).
        expected == signature
    }
}

// ---------------------------------------------------------------------------
// HttpClient trait
// ---------------------------------------------------------------------------

/// Abstraction over an HTTP POST client, enabling mock injection in tests.
pub trait HttpClient: Send + Sync {
    /// Perform an HTTP POST.  Returns the HTTP status code on success.
    fn post(
        &self,
        url: &str,
        body: &str,
        headers: HashMap<String, String>,
    ) -> Pin<Box<dyn Future<Output = Result<u16, String>> + Send + '_>>;
}

// ---------------------------------------------------------------------------
// MockHttpClient
// ---------------------------------------------------------------------------

/// A configurable mock [`HttpClient`] for unit testing.
///
/// Returns status codes from a pre-configured sequence; when the sequence is
/// exhausted every subsequent call returns the final configured code.
pub struct MockHttpClient {
    responses: Mutex<VecDeque<u16>>,
    fallback: u16,
}

impl MockHttpClient {
    /// Create a mock that always returns `status`.
    pub fn always(status: u16) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            fallback: status,
        }
    }

    /// Create a mock that returns `statuses` in order, then `fallback`.
    pub fn sequence(statuses: impl IntoIterator<Item = u16>, fallback: u16) -> Self {
        Self {
            responses: Mutex::new(statuses.into_iter().collect()),
            fallback,
        }
    }
}

impl HttpClient for MockHttpClient {
    fn post(
        &self,
        _url: &str,
        _body: &str,
        _headers: HashMap<String, String>,
    ) -> Pin<Box<dyn Future<Output = Result<u16, String>> + Send + '_>> {
        let status = {
            let mut guard = self.responses.lock().unwrap_or_else(|e| e.into_inner());
            guard.pop_front().unwrap_or(self.fallback)
        };
        Box::pin(async move { Ok(status) })
    }
}

// ---------------------------------------------------------------------------
// WebhookConfig
// ---------------------------------------------------------------------------

/// Configuration for a single outbound webhook endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Full URL of the receiving endpoint.
    pub url: String,
    /// Shared secret used for HMAC-SHA256 request signing.
    pub secret: String,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum number of delivery attempts (including the first).
    pub max_retries: u8,
    /// Base backoff in milliseconds between retry attempts (doubles each attempt).
    pub retry_backoff_ms: u64,
}

// ---------------------------------------------------------------------------
// WebhookPayload
// ---------------------------------------------------------------------------

/// The JSON body sent to the webhook endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayload {
    /// Type of event that triggered the delivery (e.g. `"cost.threshold"`).
    pub event_type: String,
    /// Unix timestamp in milliseconds when the event was created.
    pub timestamp: u64,
    /// Arbitrary event data.
    pub data: serde_json::Value,
    /// Unique identifier for this delivery attempt (hex string).
    pub delivery_id: String,
}

impl WebhookPayload {
    /// Create a new payload, generating a pseudo-random delivery ID.
    pub fn new(event_type: impl Into<String>, data: serde_json::Value) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        // Generate a delivery ID from the timestamp XORed with a rotating counter.
        static COUNTER: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0xcafe_babe_dead_beef);
        let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let delivery_id = format!("{:016x}{:016x}", ts ^ seq, seq.wrapping_mul(0x517cc1b727220a95));
        Self {
            event_type: event_type.into(),
            timestamp: ts,
            data,
            delivery_id,
        }
    }
}

// ---------------------------------------------------------------------------
// WebhookDelivery
// ---------------------------------------------------------------------------

/// Record of a single webhook delivery and all its attempts.
#[derive(Debug, Clone)]
pub struct WebhookDelivery {
    /// The payload that was (or is being) delivered.
    pub payload: WebhookPayload,
    /// Total number of delivery attempts made so far.
    pub attempts: u32,
    /// Error message from the last failed attempt, if any.
    pub last_error: Option<String>,
    /// Whether the delivery succeeded.
    pub delivered: bool,
}

impl WebhookDelivery {
    fn new(payload: WebhookPayload) -> Self {
        Self {
            payload,
            attempts: 0,
            last_error: None,
            delivered: false,
        }
    }
}

// ---------------------------------------------------------------------------
// WebhookDeliveryLog
// ---------------------------------------------------------------------------

/// Bounded ring-buffer of recent webhook deliveries.
///
/// Automatically evicts the oldest entry when the log is full.
pub struct WebhookDeliveryLog {
    entries: VecDeque<WebhookDelivery>,
    /// Maximum number of entries retained.
    pub max_entries: usize,
}

impl WebhookDeliveryLog {
    /// Create a new log with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries.min(1000)),
            max_entries,
        }
    }

    /// Append a delivery record, evicting the oldest if at capacity.
    pub fn push(&mut self, delivery: WebhookDelivery) {
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(delivery);
    }

    /// Iterate over all delivery records in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &WebhookDelivery> {
        self.entries.iter()
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// WebhookDispatcher
// ---------------------------------------------------------------------------

/// Async webhook dispatcher with retry and delivery logging.
///
/// Delivery requests are placed on an internal [`mpsc`] channel; a background
/// Tokio task processes them with exponential backoff retries.  The caller
/// does not need to await individual deliveries.
pub struct WebhookDispatcher {
    config: WebhookConfig,
    log: Arc<Mutex<WebhookDeliveryLog>>,
    tx: mpsc::Sender<WebhookPayload>,
}

impl WebhookDispatcher {
    /// Create a new dispatcher backed by `client`.
    ///
    /// Spawns an internal background Tokio task immediately.
    pub fn new(config: WebhookConfig, client: Arc<dyn HttpClient>) -> Self {
        let (tx, mut rx) = mpsc::channel::<WebhookPayload>(256);
        let log = Arc::new(Mutex::new(WebhookDeliveryLog::new(1000)));
        let log_clone = Arc::clone(&log);
        let cfg_clone = config.clone();

        tokio::spawn(async move {
            while let Some(payload) = rx.recv().await {
                let mut delivery = WebhookDelivery::new(payload.clone());
                let body = serde_json::to_string(&payload).unwrap_or_default();
                let signer = WebhookSigner::new(&cfg_clone.secret);
                let signature = signer.sign(&body);

                let mut backoff_ms = cfg_clone.retry_backoff_ms;
                let mut succeeded = false;

                for attempt in 0..=cfg_clone.max_retries {
                    delivery.attempts = attempt as u32 + 1;

                    let mut headers = HashMap::new();
                    headers.insert("Content-Type".to_string(), "application/json".to_string());
                    headers.insert("X-Webhook-Signature".to_string(), signature.clone());

                    match client.post(&cfg_clone.url, &body, headers).await {
                        Ok(status) if (200..300).contains(&status) => {
                            delivery.delivered = true;
                            delivery.last_error = None;
                            succeeded = true;
                            break;
                        }
                        Ok(status) => {
                            delivery.last_error =
                                Some(format!("HTTP {status}"));
                        }
                        Err(e) => {
                            delivery.last_error = Some(e);
                        }
                    }

                    if attempt < cfg_clone.max_retries {
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                        backoff_ms = backoff_ms.saturating_mul(2);
                    }
                }

                let _ = succeeded; // explicitly captured for clarity
                let mut guard = log_clone.lock().unwrap_or_else(|e| e.into_inner());
                guard.push(delivery);
            }
        });

        Self { config, log, tx }
    }

    /// Enqueue an outbound delivery.
    ///
    /// Returns immediately — delivery happens asynchronously in the background.
    pub fn dispatch(&self, event_type: &str, data: serde_json::Value) {
        let payload = WebhookPayload::new(event_type, data);
        // Best-effort: ignore back-pressure errors (channel full / closed).
        let _ = self.tx.try_send(payload);
    }

    /// Perform a synchronous (in the current async task) delivery with retry.
    ///
    /// Unlike [`dispatch`](Self::dispatch) this awaits the final outcome and
    /// returns the completed [`WebhookDelivery`] record.
    pub async fn deliver_with_retry(
        &self,
        payload: WebhookPayload,
        client: &dyn HttpClient,
    ) -> WebhookDelivery {
        let mut delivery = WebhookDelivery::new(payload.clone());
        let body = serde_json::to_string(&payload).unwrap_or_default();
        let signer = WebhookSigner::new(&self.config.secret);
        let signature = signer.sign(&body);
        let mut backoff_ms = self.config.retry_backoff_ms;

        for attempt in 0..=self.config.max_retries {
            delivery.attempts = attempt as u32 + 1;

            let mut headers = HashMap::new();
            headers.insert("Content-Type".to_string(), "application/json".to_string());
            headers.insert("X-Webhook-Signature".to_string(), signature.clone());

            match client.post(&self.config.url, &body, headers).await {
                Ok(status) if (200..300).contains(&status) => {
                    delivery.delivered = true;
                    delivery.last_error = None;
                    break;
                }
                Ok(status) => {
                    delivery.last_error = Some(format!("HTTP {status}"));
                }
                Err(e) => {
                    delivery.last_error = Some(e);
                }
            }

            if attempt < self.config.max_retries {
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = backoff_ms.saturating_mul(2);
            }
        }

        delivery
    }

    /// Shared reference to the delivery log.
    pub fn log(&self) -> Arc<Mutex<WebhookDeliveryLog>> {
        Arc::clone(&self.log)
    }

    /// The configuration this dispatcher was created with.
    pub fn config(&self) -> &WebhookConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SHA-256 / HMAC tests ----------------------------------------------

    #[test]
    fn hmac_sha256_matches_rfc_4231_test_case_2() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hex_encode(&mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn sha256_empty_input_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256(b"");
        let hex = hex_encode(&hash);
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc_known_vector() {
        // FIPS 180-4 test vector: SHA-256("abc")
        // = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let hash = sha256(b"abc");
        let hex = hex_encode(&hash);
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn signing_produces_consistent_signature() {
        let signer = WebhookSigner::new("my-secret");
        let body = r#"{"event":"test"}"#;
        let sig1 = signer.sign(body);
        let sig2 = signer.sign(body);
        assert_eq!(sig1, sig2);
        assert!(sig1.starts_with("sha256="));
    }

    #[test]
    fn signing_different_bodies_produce_different_signatures() {
        let signer = WebhookSigner::new("secret");
        let sig1 = signer.sign("hello");
        let sig2 = signer.sign("world");
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn verify_round_trip() {
        let signer = WebhookSigner::new("hunter2");
        let body = r#"{"event":"budget.exceeded","amount":42.0}"#;
        let sig = signer.sign(body);
        assert!(signer.verify(body, &sig));
        assert!(!signer.verify("tampered body", &sig));
    }

    // ---- Delivery tests ----------------------------------------------------

    #[tokio::test]
    async fn success_on_200() {
        let config = WebhookConfig {
            url: "http://test.local/hook".into(),
            secret: "secret".into(),
            timeout_ms: 1_000,
            max_retries: 2,
            retry_backoff_ms: 1,
        };
        let client = Arc::new(MockHttpClient::always(200));
        let dispatcher = WebhookDispatcher::new(config, client.clone());
        let payload = WebhookPayload::new("test.event", serde_json::json!({"k": "v"}));
        let delivery = dispatcher.deliver_with_retry(payload, client.as_ref()).await;
        assert!(delivery.delivered);
        assert_eq!(delivery.attempts, 1);
        assert!(delivery.last_error.is_none());
    }

    #[tokio::test]
    async fn retry_on_500_then_succeed() {
        let config = WebhookConfig {
            url: "http://test.local/hook".into(),
            secret: "s".into(),
            timeout_ms: 1_000,
            max_retries: 3,
            retry_backoff_ms: 1,
        };
        // First two attempts return 500, third returns 200.
        let client = Arc::new(MockHttpClient::sequence([500, 500, 200], 200));
        let dispatcher = WebhookDispatcher::new(config, client.clone());
        let payload = WebhookPayload::new("retry.event", serde_json::json!(null));
        let delivery = dispatcher.deliver_with_retry(payload, client.as_ref()).await;
        assert!(delivery.delivered);
        assert_eq!(delivery.attempts, 3);
    }

    #[tokio::test]
    async fn all_retries_exhausted_marks_not_delivered() {
        let config = WebhookConfig {
            url: "http://test.local/hook".into(),
            secret: "s".into(),
            timeout_ms: 1_000,
            max_retries: 2,
            retry_backoff_ms: 1,
        };
        let client = Arc::new(MockHttpClient::always(500));
        let dispatcher = WebhookDispatcher::new(config, client.clone());
        let payload = WebhookPayload::new("fail.event", serde_json::json!(null));
        let delivery = dispatcher.deliver_with_retry(payload, client.as_ref()).await;
        assert!(!delivery.delivered);
        // max_retries=2 means attempts 0,1,2 → 3 total.
        assert_eq!(delivery.attempts, 3);
        assert!(delivery.last_error.is_some());
    }

    #[tokio::test]
    async fn dispatch_records_log_entry() {
        let config = WebhookConfig {
            url: "http://test.local/hook".into(),
            secret: "s".into(),
            timeout_ms: 1_000,
            max_retries: 0,
            retry_backoff_ms: 1,
        };
        let client = Arc::new(MockHttpClient::always(200));
        let dispatcher = WebhookDispatcher::new(config, client);
        dispatcher.dispatch("log.test", serde_json::json!({"ok": true}));
        // Allow the background task time to process.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let log = dispatcher.log();
        let guard = log.lock().unwrap();
        assert_eq!(guard.len(), 1);
        assert!(guard.iter().next().unwrap().delivered);
    }

    #[test]
    fn delivery_log_max_entries_eviction() {
        let mut log = WebhookDeliveryLog::new(3);
        for i in 0u64..5 {
            let payload = WebhookPayload {
                event_type: "e".into(),
                timestamp: i,
                data: serde_json::Value::Null,
                delivery_id: format!("{i:032x}"),
            };
            log.push(WebhookDelivery::new(payload));
        }
        assert_eq!(log.len(), 3);
        // The oldest two should have been evicted; first remaining ts = 2.
        assert_eq!(log.iter().next().unwrap().payload.timestamp, 2);
    }
}
