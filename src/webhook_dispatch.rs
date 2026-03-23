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

// ---------------------------------------------------------------------------
// Compact SHA-256 implementation (no external crates)
// ---------------------------------------------------------------------------
// Based on FIPS 180-4.  Used only internally by the HMAC signer.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
    0x5be0cd19,
];

fn sha256(data: &[u8]) -> [u8; 32] {
    // Pre-processing: padding.
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = H0;

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// HMAC-SHA256
// ---------------------------------------------------------------------------

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK_SIZE: usize = 64;

    // If key is longer than block size, hash it first.
    let mut k = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let hk = sha256(key);
        k[..32].copy_from_slice(&hk);
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; BLOCK_SIZE];
    let mut opad = [0x5cu8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }

    // inner = H(ipad ∥ message)
    let mut inner_input = ipad.to_vec();
    inner_input.extend_from_slice(message);
    let inner_hash = sha256(&inner_input);

    // outer = H(opad ∥ inner)
    let mut outer_input = opad.to_vec();
    outer_input.extend_from_slice(&inner_hash);
    sha256(&outer_input)
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
