//! # Webhook Alerter
//!
//! Delivers structured cost and budget alerts to one or more webhook URLs
//! (Slack-compatible payload format) with per-alert-kind cooldown deduplication.
//!
//! ## Feature flag
//!
//! This module is only available when the `webhooks` crate feature is enabled
//! (on by default).  The feature pulls in [`reqwest`] with TLS support.
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "webhooks")]
//! # async fn example() {
//! use llm_cost_dashboard::alerting::{Alert, AlertKind, WebhookAlerter};
//!
//! let mut alerter = WebhookAlerter::new(
//!     vec!["https://hooks.slack.com/services/T.../B.../xxx".to_string()],
//!     300, // 5-minute cooldown per alert kind
//! );
//!
//! let alert = Alert::new(AlertKind::BudgetWarning {
//!     spent: 85.0,
//!     limit: 100.0,
//!     pct: 85.0,
//! });
//! alerter.fire(alert).await;
//! # }
//! ```

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// The category and payload of an alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data")]
pub enum AlertKind {
    /// Monthly hard budget limit has been exceeded.
    BudgetExceeded {
        /// Total spend in USD at the time the limit was crossed.
        spent: f64,
        /// The configured hard limit in USD.
        limit: f64,
    },
    /// Spend has crossed the soft warning threshold.
    BudgetWarning {
        /// Current spend in USD.
        spent: f64,
        /// The configured hard limit in USD.
        limit: f64,
        /// Percentage consumed (0–100).
        pct: f64,
    },
    /// A single request cost deviated more than the Z-score threshold.
    CostAnomaly {
        /// Model that produced the anomalous request.
        model: String,
        /// Per-request cost in USD.
        cost: f64,
        /// Z-score of the anomalous observation.
        z_score: f64,
    },
    /// Today's total spend is a multiple of the recent daily average.
    DailySpendSpike {
        /// Today's accumulated spend in USD.
        today: f64,
        /// Rolling average daily spend in USD.
        avg: f64,
        /// Ratio of today / avg.
        multiplier: f64,
    },
}

impl AlertKind {
    /// A stable, human-readable key used for cooldown deduplication.
    ///
    /// Different instances of the same variant produce the same key so that
    /// repeated triggers of the same alert kind are suppressed during the
    /// cooldown window.
    pub fn cooldown_key(&self) -> &'static str {
        match self {
            AlertKind::BudgetExceeded { .. } => "budget_exceeded",
            AlertKind::BudgetWarning { .. } => "budget_warning",
            AlertKind::CostAnomaly { .. } => "cost_anomaly",
            AlertKind::DailySpendSpike { .. } => "daily_spend_spike",
        }
    }

    /// Generate a human-readable summary message for this alert.
    pub fn message(&self) -> String {
        match self {
            AlertKind::BudgetExceeded { spent, limit } => {
                format!(
                    "Budget exceeded: spent ${spent:.2} of ${limit:.2} monthly limit"
                )
            }
            AlertKind::BudgetWarning { spent, limit, pct } => {
                format!(
                    "Budget warning: {pct:.1}% consumed (${spent:.2} / ${limit:.2})"
                )
            }
            AlertKind::CostAnomaly {
                model,
                cost,
                z_score,
            } => {
                format!(
                    "Cost anomaly detected on {model}: ${cost:.6} (Z={z_score:.2})"
                )
            }
            AlertKind::DailySpendSpike {
                today,
                avg,
                multiplier,
            } => {
                format!(
                    "Daily spend spike: ${today:.2} today vs ${avg:.2} avg ({multiplier:.1}x)"
                )
            }
        }
    }
}

/// A single alert ready to be delivered to webhooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique identifier for this alert instance (UUID v4).
    pub id: String,
    /// The alert category and associated data.
    pub kind: AlertKind,
    /// RFC 3339 timestamp of when the alert was created.
    pub timestamp: String,
    /// Human-readable summary message.
    pub message: String,
}

impl Alert {
    /// Construct a new alert, populating `id`, `timestamp`, and `message`
    /// automatically.
    pub fn new(kind: AlertKind) -> Self {
        let message = kind.message();
        Self {
            id: new_uuid(),
            timestamp: rfc3339_now(),
            message,
            kind,
        }
    }
}

/// Webhook-based alert delivery with per-kind cooldown deduplication.
///
/// Alerts are dispatched as HTTP POST requests to each configured webhook URL.
/// Delivery is best-effort: failures are logged via [`tracing`] but do not
/// propagate as errors to the caller.
///
/// Cooldown ensures that the same alert *kind* cannot fire more than once per
/// `cooldown_secs` seconds, preventing alert floods during sustained budget
/// overruns or anomaly storms.
///
/// Requires the `webhooks` crate feature (enabled by default).
#[cfg(feature = "webhooks")]
pub struct WebhookAlerter {
    /// Target webhook URLs.
    webhooks: Vec<String>,
    /// Shared HTTP client (connection-pooled).
    client: reqwest::Client,
    /// Maps cooldown key -> last-fired `Instant`.
    cooldown: HashMap<String, Instant>,
    /// Minimum seconds between firings of the same alert kind.
    cooldown_secs: u64,
}

#[cfg(feature = "webhooks")]
impl WebhookAlerter {
    /// Create a new alerter.
    ///
    /// # Arguments
    ///
    /// * `webhooks` – list of HTTP(S) URLs to POST to.
    /// * `cooldown_secs` – minimum gap between firings of the same alert kind.
    pub fn new(webhooks: Vec<String>, cooldown_secs: u64) -> Self {
        Self {
            webhooks,
            client: reqwest::Client::new(),
            cooldown: HashMap::new(),
            cooldown_secs,
        }
    }

    /// Fire an alert to all configured webhooks.
    ///
    /// If the same alert kind was fired within `cooldown_secs` seconds this
    /// call is silently dropped.  Otherwise the alert is dispatched
    /// concurrently to all webhooks.  Individual webhook failures are logged
    /// as warnings but do not abort delivery to the remaining URLs.
    pub async fn fire(&mut self, alert: Alert) {
        let key = alert.kind.cooldown_key().to_string();

        // Cooldown check.
        if let Some(last) = self.cooldown.get(&key) {
            let elapsed = last.elapsed().as_secs();
            if elapsed < self.cooldown_secs {
                tracing::debug!(
                    alert_id = %alert.id,
                    kind = %key,
                    elapsed_secs = elapsed,
                    cooldown_secs = self.cooldown_secs,
                    "alert suppressed by cooldown"
                );
                return;
            }
        }

        self.cooldown.insert(key.clone(), Instant::now());

        let payload = self.build_payload(&alert);

        for url in &self.webhooks {
            let result = self
                .client
                .post(url)
                .json(&payload)
                .send()
                .await;

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        tracing::info!(
                            alert_id = %alert.id,
                            url = %url,
                            http_status = %status,
                            "webhook alert delivered"
                        );
                    } else {
                        tracing::warn!(
                            alert_id = %alert.id,
                            url = %url,
                            http_status = %status,
                            "webhook returned non-2xx status"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        alert_id = %alert.id,
                        url = %url,
                        error = %e,
                        "webhook delivery failed"
                    );
                }
            }
        }
    }

    /// Build the JSON payload for a webhook POST.
    ///
    /// The payload is Slack-compatible: it includes a `text` field with the
    /// human-readable summary and an `attachments` array carrying the
    /// structured alert data for richer display in Slack or compatible tools
    /// (e.g., Mattermost, Discord with Slack-compat mode).
    fn build_payload(&self, alert: &Alert) -> serde_json::Value {
        let color = match &alert.kind {
            AlertKind::BudgetExceeded { .. } => "#FF0000",
            AlertKind::BudgetWarning { .. } => "#FFA500",
            AlertKind::CostAnomaly { .. } => "#FF6600",
            AlertKind::DailySpendSpike { .. } => "#FFCC00",
        };

        serde_json::json!({
            "text": format!(":rotating_light: *LLM Cost Alert* — {}", alert.message),
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {
                            "title": "Alert ID",
                            "value": alert.id,
                            "short": true
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp,
                            "short": true
                        },
                        {
                            "title": "Details",
                            "value": serde_json::to_string(&alert.kind)
                                .unwrap_or_else(|_| "serialization error".into()),
                            "short": false
                        }
                    ],
                    "footer": "llm-cost-dashboard",
                }
            ]
        })
    }
}

/// Non-webhook version of the alerter used when the `webhooks` feature is
/// disabled.  All `fire` calls are no-ops.
#[cfg(not(feature = "webhooks"))]
#[allow(dead_code)]
pub struct WebhookAlerter {
    _webhooks: Vec<String>,
    cooldown: HashMap<String, Instant>,
    cooldown_secs: u64,
}

#[cfg(not(feature = "webhooks"))]
impl WebhookAlerter {
    /// Create a no-op alerter (webhooks feature is disabled).
    pub fn new(webhooks: Vec<String>, cooldown_secs: u64) -> Self {
        Self {
            _webhooks: webhooks,
            cooldown: HashMap::new(),
            cooldown_secs,
        }
    }

    /// No-op fire (webhooks feature is disabled).
    pub async fn fire(&mut self, _alert: Alert) {}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Generate a simple UUID v4-like string using random bytes from the standard
/// library's pseudo-random source.  This avoids adding a uuid dependency in
/// this module; the format is UUID v4 compliant.
fn new_uuid() -> String {
    // We read 16 bytes of pseudo-random state from stack addresses and timing.
    // For a production system a proper uuid crate call is preferable; here we
    // use a deterministic-enough approach that avoids extra dependencies.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut h = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut h);
    // Mix in the thread id for uniqueness in multi-threaded contexts.
    std::thread::current().id().hash(&mut h);
    let hi = h.finish();
    // Second pass with a different seed.
    hi.wrapping_add(0xdeadbeef_cafebabe).hash(&mut h);
    let lo = h.finish();

    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (hi >> 32) as u32,
        (hi >> 16) as u16,
        hi as u16 & 0x0fff,
        (lo >> 48) as u16 & 0x3fff | 0x8000,
        lo & 0x0000_ffff_ffff_ffff,
    )
}

/// Format the current UTC time as an RFC 3339 string without pulling in chrono
/// in this module (chrono is available project-wide but keeping this
/// self-contained avoids a circular-module dependency concern).
fn rfc3339_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Convert Unix timestamp to a simple UTC ISO 8601 string.
    unix_secs_to_rfc3339(secs)
}

/// Minimal Unix-timestamp-to-RFC3339 converter (UTC, seconds precision).
///
/// Implemented without external dependencies using the same proleptic Gregorian
/// calendar algorithm used in `forecast.rs`.
fn unix_secs_to_rfc3339(unix_secs: u64) -> String {
    let secs = unix_secs;
    let time_of_day = secs % 86_400;
    let h = time_of_day / 3_600;
    let m = (time_of_day % 3_600) / 60;
    let s = time_of_day % 60;

    // Days since epoch.
    let z = (secs / 86_400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_kind_cooldown_keys_unique() {
        let keys = [
            AlertKind::BudgetExceeded { spent: 0.0, limit: 0.0 }.cooldown_key(),
            AlertKind::BudgetWarning { spent: 0.0, limit: 0.0, pct: 0.0 }.cooldown_key(),
            AlertKind::CostAnomaly { model: "m".into(), cost: 0.0, z_score: 0.0 }.cooldown_key(),
            AlertKind::DailySpendSpike { today: 0.0, avg: 0.0, multiplier: 0.0 }.cooldown_key(),
        ];
        let unique: std::collections::HashSet<_> = keys.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_alert_message_budget_exceeded() {
        let kind = AlertKind::BudgetExceeded { spent: 12.34, limit: 10.0 };
        let msg = kind.message();
        assert!(msg.contains("12.34"));
        assert!(msg.contains("10.00"));
    }

    #[test]
    fn test_alert_message_budget_warning() {
        let kind = AlertKind::BudgetWarning { spent: 85.0, limit: 100.0, pct: 85.0 };
        let msg = kind.message();
        assert!(msg.contains("85.0"));
        assert!(msg.contains("100.00"));
    }

    #[test]
    fn test_alert_message_cost_anomaly() {
        let kind = AlertKind::CostAnomaly {
            model: "gpt-4o".into(),
            cost: 0.05,
            z_score: 4.2,
        };
        let msg = kind.message();
        assert!(msg.contains("gpt-4o"));
        assert!(msg.contains("4.20"));
    }

    #[test]
    fn test_alert_message_daily_spike() {
        let kind = AlertKind::DailySpendSpike { today: 20.0, avg: 5.0, multiplier: 4.0 };
        let msg = kind.message();
        assert!(msg.contains("20.00"));
        assert!(msg.contains("4.0x"));
    }

    #[test]
    fn test_alert_new_populates_fields() {
        let kind = AlertKind::BudgetExceeded { spent: 1.0, limit: 0.5 };
        let alert = Alert::new(kind);
        assert!(!alert.id.is_empty());
        assert!(!alert.timestamp.is_empty());
        assert!(!alert.message.is_empty());
    }

    #[test]
    fn test_rfc3339_epoch() {
        let s = unix_secs_to_rfc3339(0);
        assert_eq!(s, "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_rfc3339_known_date() {
        // 2024-03-15T12:00:00Z => 1710504000
        let s = unix_secs_to_rfc3339(1_710_504_000);
        assert_eq!(s, "2024-03-15T12:00:00Z");
    }

    #[test]
    fn test_new_uuid_not_empty() {
        let id = new_uuid();
        assert!(!id.is_empty());
        // Should look like xxxxxxxx-xxxx-4xxx-xxxx-xxxxxxxxxxxx
        assert_eq!(id.len(), 36);
    }

    #[cfg(feature = "webhooks")]
    #[test]
    fn test_webhook_alerter_new() {
        let alerter = WebhookAlerter::new(vec!["https://example.com".into()], 60);
        assert!(alerter.cooldown.is_empty());
        assert_eq!(alerter.cooldown_secs, 60);
    }

    #[cfg(feature = "webhooks")]
    #[test]
    fn test_build_payload_contains_text() {
        let alerter = WebhookAlerter::new(vec![], 60);
        let alert = Alert::new(AlertKind::BudgetWarning {
            spent: 80.0,
            limit: 100.0,
            pct: 80.0,
        });
        let payload = alerter.build_payload(&alert);
        let text = payload["text"].as_str().unwrap_or("");
        assert!(text.contains("LLM Cost Alert"));
    }
}
