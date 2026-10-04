//! # Budget Alert System
//!
//! Rule-based budget alerting with cooldown tracking, multi-window cost
//! aggregation, and flexible delivery channels (log, webhook, file).
//!
//! ## Design
//!
//! An [`AlertEngine`] holds a set of [`AlertRule`]s.  Callers call
//! [`AlertEngine::check`] periodically (e.g. every minute) with the current
//! [`CostLedger`].  The engine evaluates every rule against the ledger, fires
//! [`Alert`]s for rules whose cost threshold has been exceeded, and suppresses
//! re-firing the same rule for `cooldown` duration.
//!
//! ## Alert channels
//!
//! - [`AlertChannel::Log`] — emits a `tracing::warn!` entry.
//! - [`AlertChannel::Webhook`] — HTTP POST with a JSON body; optional HMAC-SHA256
//!   signature in the `X-Alert-Signature` header.
//! - [`AlertChannel::File`] — appends a JSON line to a file on disk.
//!
//! ## Example
//!
//! ```no_run
//! use std::time::Duration;
//! use llm_cost_dashboard::alerts::{AlertChannel, AlertEngine, AlertRule, AlertWindow};
//! use llm_cost_dashboard::cost::CostLedger;
//!
//! let rules = vec![AlertRule {
//!     name: "daily-5-usd".to_string(),
//!     threshold_usd: 5.0,
//!     window: AlertWindow::Daily,
//!     channel: AlertChannel::Log,
//!     cooldown: Duration::from_secs(3600),
//! }];
//! let mut engine = AlertEngine::new(rules);
//! let ledger = CostLedger::new();
//! let alerts = engine.check(&ledger);
//! println!("fired {} alerts", alerts.len());
//! ```

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write as IoWrite;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{Datelike, Utc};
use serde::{Deserialize, Serialize};

use crate::cost::CostLedger;

// ── AlertWindow ───────────────────────────────────────────────────────────────

/// The time window over which costs are aggregated when evaluating a rule.
#[derive(Debug, Clone, PartialEq)]
pub enum AlertWindow {
    /// Sum costs from the last `Duration` of wall time.
    Rolling(Duration),
    /// Sum today's costs (UTC calendar day).
    Daily,
    /// Sum costs from the last 7 calendar days (UTC).
    Weekly,
    /// Sum costs from the current calendar month (UTC).
    Monthly,
}

// ── AlertChannel ──────────────────────────────────────────────────────────────

/// How an alert is delivered when it fires.
#[derive(Debug, Clone)]
pub enum AlertChannel {
    /// Write a `tracing::warn!` log entry.
    Log,
    /// HTTP POST to a URL with a JSON body.
    ///
    /// When `secret` is `Some`, the body is HMAC-SHA256-signed and the
    /// signature is sent in the `X-Alert-Signature: sha256=<hex>` header.
    Webhook {
        /// Destination URL.
        url: String,
        /// Optional HMAC secret for request signing.
        secret: Option<String>,
    },
    /// Append a JSON line to a file.
    File {
        /// Absolute path to the target file.
        path: String,
    },
}

// ── AlertRule ─────────────────────────────────────────────────────────────────

/// A rule that fires when accumulated cost in `window` exceeds `threshold_usd`.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Stable human-readable identifier for this rule.
    pub name: String,
    /// USD amount that triggers the alert when the window cost exceeds it.
    pub threshold_usd: f64,
    /// Time window to aggregate costs over.
    pub window: AlertWindow,
    /// Where to deliver the alert.
    pub channel: AlertChannel,
    /// Minimum time between successive firings of this rule.
    pub cooldown: Duration,
}

// ── Alert ─────────────────────────────────────────────────────────────────────

/// A fired alert instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Name of the rule that triggered this alert.
    pub rule_name: String,
    /// UTC timestamp of the firing (seconds since Unix epoch).
    pub triggered_at: u64,
    /// Aggregated cost in the evaluation window (USD).
    pub cost_usd: f64,
    /// Rule threshold that was exceeded (USD).
    pub threshold_usd: f64,
    /// Human-readable description.
    pub message: String,
}

impl Alert {
    fn new(rule: &AlertRule, cost_usd: f64) -> Self {
        let triggered_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            rule_name: rule.name.clone(),
            triggered_at,
            cost_usd,
            threshold_usd: rule.threshold_usd,
            message: format!(
                "Alert '{}': cost ${:.4} exceeded threshold ${:.4} over {:?} window",
                rule.name, cost_usd, rule.threshold_usd, rule.window
            ),
        }
    }
}

// ── AlertSummary ──────────────────────────────────────────────────────────────

/// Aggregate statistics returned by [`AlertEngine::summary`].
#[derive(Debug, Clone, Default)]
pub struct AlertSummary {
    /// Total alerts fired since the engine was created.
    pub fired_total: u64,
    /// Alerts suppressed by cooldown since the engine was created.
    pub suppressed_by_cooldown: u64,
    /// Number of rules configured in the engine.
    pub rules_count: usize,
}

// ── HMAC helper ───────────────────────────────────────────────────────────────

/// Compute HMAC-SHA256 over `payload` with `key` and return the hex digest,
/// with the RustCrypto `hmac` and `sha2` crates (a hand-written SHA-256 was
/// used before 1.3.0).
#[cfg(feature = "webhooks")]
fn hmac_sha256_hex(key: &[u8], payload: &[u8]) -> String {
    use hmac::{Hmac, Mac};
    // HMAC accepts keys of any length, so this cannot fail.
    let mut mac = match <Hmac<sha2::Sha256> as Mac>::new_from_slice(key) {
        Ok(m) => m,
        Err(_) => return String::new(),
    };
    mac.update(payload);
    mac.finalize()
        .into_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ── Cost window helper ────────────────────────────────────────────────────────

/// Compute the aggregated cost from `ledger` for the given `window`.
fn window_cost(ledger: &CostLedger, window: &AlertWindow) -> f64 {
    let now = Utc::now();
    match window {
        AlertWindow::Rolling(dur) => {
            let cutoff = now
                - chrono::Duration::from_std(*dur)
                    .unwrap_or(chrono::Duration::MAX);
            ledger
                .records()
                .iter()
                .filter(|r| r.timestamp >= cutoff)
                .map(|r| r.total_cost_usd)
                .sum()
        }
        AlertWindow::Daily => {
            ledger
                .records()
                .iter()
                .filter(|r| {
                    r.timestamp.year() == now.year()
                        && r.timestamp.month() == now.month()
                        && r.timestamp.day() == now.day()
                })
                .map(|r| r.total_cost_usd)
                .sum()
        }
        AlertWindow::Weekly => {
            let cutoff = now - chrono::Duration::days(7);
            ledger
                .records()
                .iter()
                .filter(|r| r.timestamp >= cutoff)
                .map(|r| r.total_cost_usd)
                .sum()
        }
        AlertWindow::Monthly => {
            ledger
                .records()
                .iter()
                .filter(|r| {
                    r.timestamp.year() == now.year() && r.timestamp.month() == now.month()
                })
                .map(|r| r.total_cost_usd)
                .sum()
        }
    }
}

// ── AlertEngine ───────────────────────────────────────────────────────────────

/// Evaluates rules against the cost ledger and fires alerts.
pub struct AlertEngine {
    rules: Vec<AlertRule>,
    /// Maps rule name → instant it last fired.
    last_fired: HashMap<String, Instant>,
    fired_total: u64,
    suppressed_by_cooldown: u64,
}

impl AlertEngine {
    /// Create a new engine with the given rules.
    pub fn new(rules: Vec<AlertRule>) -> Self {
        Self {
            rules,
            last_fired: HashMap::new(),
            fired_total: 0,
            suppressed_by_cooldown: 0,
        }
    }

    /// Check all rules against `ledger` and return the alerts that fired.
    ///
    /// Rules whose cooldown has not elapsed are silently suppressed.
    pub fn check(&mut self, ledger: &CostLedger) -> Vec<Alert> {
        let mut fired = Vec::new();
        for rule in &self.rules {
            let cost = window_cost(ledger, &rule.window);
            if cost <= rule.threshold_usd {
                continue;
            }
            // Cooldown check.
            if let Some(last) = self.last_fired.get(&rule.name) {
                if last.elapsed() < rule.cooldown {
                    self.suppressed_by_cooldown += 1;
                    continue;
                }
            }
            // Fire the alert.
            self.last_fired.insert(rule.name.clone(), Instant::now());
            self.fired_total += 1;
            let alert = Alert::new(rule, cost);
            self.deliver(&alert, rule);
            fired.push(alert);
        }
        fired
    }

    /// Return a summary of alert activity since the engine was created.
    pub fn summary(&self) -> AlertSummary {
        AlertSummary {
            fired_total: self.fired_total,
            suppressed_by_cooldown: self.suppressed_by_cooldown,
            rules_count: self.rules.len(),
        }
    }

    /// Deliver an alert via the channel configured in `rule`.
    fn deliver(&self, alert: &Alert, rule: &AlertRule) {
        match &rule.channel {
            AlertChannel::Log => {
                tracing::warn!(
                    rule = %alert.rule_name,
                    cost_usd = alert.cost_usd,
                    threshold_usd = alert.threshold_usd,
                    message = %alert.message,
                    "budget alert fired"
                );
            }
            AlertChannel::Webhook { url, secret } => {
                self.deliver_webhook(alert, url, secret.as_deref());
            }
            AlertChannel::File { path } => {
                self.deliver_file(alert, path);
            }
        }
    }

    /// Synchronous (blocking) webhook delivery.
    ///
    /// Uses a short-lived `reqwest::blocking::Client` so callers don't need to
    /// be inside a Tokio runtime.  Failures are logged as warnings.
    ///
    /// Requires the `webhooks` crate feature (enabled by default).
    #[cfg_attr(not(feature = "webhooks"), allow(unused_variables))]
    fn deliver_webhook(&self, alert: &Alert, url: &str, secret: Option<&str>) {
        #[cfg(feature = "webhooks")]
        {
            let body = match serde_json::to_string(alert) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to serialize alert for webhook");
                    return;
                }
            };

            let client = match reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to build HTTP client for webhook");
                    return;
                }
            };

            let mut req = client
                .post(url)
                .header("Content-Type", "application/json")
                .body(body.clone());

            if let Some(sec) = secret {
                let sig = hmac_sha256_hex(sec.as_bytes(), body.as_bytes());
                req = req.header("X-Alert-Signature", format!("sha256={sig}"));
            }

            match req.send() {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!(url = %url, "webhook alert delivered");
                }
                Ok(resp) => {
                    tracing::warn!(url = %url, status = %resp.status(), "webhook returned non-2xx");
                }
                Err(e) => {
                    tracing::warn!(url = %url, error = %e, "webhook delivery failed");
                }
            }
        }
        #[cfg(not(feature = "webhooks"))]
        {
            tracing::warn!(
                url = %url,
                "webhook delivery skipped: 'webhooks' feature not enabled"
            );
            let _ = secret; // suppress unused warning
        }
    }

    /// Append a JSON line to a file.
    fn deliver_file(&self, alert: &Alert, path: &str) {
        let line = match serde_json::to_string(alert) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize alert for file");
                return;
            }
        };
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(mut f) => {
                if let Err(e) = writeln!(f, "{line}") {
                    tracing::warn!(path = %path, error = %e, "failed to write alert to file");
                }
            }
            Err(e) => {
                tracing::warn!(path = %path, error = %e, "failed to open alert file");
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::cost::{CostLedger, CostRecord};

    fn ledger_with_cost(total_usd: f64) -> CostLedger {
        let mut ledger = CostLedger::new();
        // gpt-4o-mini at ~$0.15/$0.60 per 1M tokens
        // We'll add a record with enough tokens to hit the desired cost.
        // For simplicity just add multiple small records.
        let per_record = total_usd / 5.0;
        for _ in 0..5 {
            // Use enough input tokens to produce the target cost.
            // At $0.00015/1k input tokens we need (per_record / 0.00015 * 1000) tokens.
            // Instead, use a model that costs more per token (gpt-4o: $5/1M input).
            // 1 input token = $0.000005 => tokens_needed = per_record / 0.000005
            let tokens = (per_record / 0.000005).max(1.0) as u64;
            let _ = ledger.add(CostRecord::new("gpt-4o", "openai", tokens, 0, 1));
        }
        ledger
    }

    fn make_rule(threshold: f64, cooldown: Duration) -> AlertRule {
        AlertRule {
            name: "test-rule".to_string(),
            threshold_usd: threshold,
            window: AlertWindow::Monthly,
            channel: AlertChannel::Log,
            cooldown,
        }
    }

    #[test]
    fn test_no_alert_when_below_threshold() {
        let ledger = CostLedger::new(); // empty
        let mut engine = AlertEngine::new(vec![make_rule(1.0, Duration::from_secs(0))]);
        let alerts = engine.check(&ledger);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_alert_fires_when_threshold_exceeded() {
        let ledger = ledger_with_cost(10.0);
        let mut engine = AlertEngine::new(vec![make_rule(0.01, Duration::from_secs(0))]);
        let alerts = engine.check(&ledger);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].rule_name, "test-rule");
    }

    #[test]
    fn test_alert_cost_matches_window() {
        let ledger = ledger_with_cost(10.0);
        let mut engine = AlertEngine::new(vec![make_rule(0.01, Duration::from_secs(0))]);
        let alerts = engine.check(&ledger);
        assert!(alerts[0].cost_usd > 0.0);
        assert!(alerts[0].threshold_usd == 0.01);
    }

    #[test]
    fn test_cooldown_suppresses_repeated_alert() {
        let ledger = ledger_with_cost(10.0);
        let mut engine = AlertEngine::new(vec![make_rule(0.01, Duration::from_secs(3600))]);
        let first = engine.check(&ledger);
        let second = engine.check(&ledger);
        assert!(!first.is_empty());
        assert!(second.is_empty());
        assert_eq!(engine.summary().suppressed_by_cooldown, 1);
    }

    #[test]
    fn test_zero_cooldown_allows_immediate_re_fire() {
        let ledger = ledger_with_cost(10.0);
        let mut engine = AlertEngine::new(vec![make_rule(0.01, Duration::from_secs(0))]);
        let first = engine.check(&ledger);
        let second = engine.check(&ledger);
        assert!(!first.is_empty());
        assert!(!second.is_empty());
    }

    #[test]
    fn test_multiple_rules_both_fire() {
        let ledger = ledger_with_cost(10.0);
        let rules = vec![
            AlertRule {
                name: "rule-a".to_string(),
                threshold_usd: 0.01,
                window: AlertWindow::Monthly,
                channel: AlertChannel::Log,
                cooldown: Duration::from_secs(0),
            },
            AlertRule {
                name: "rule-b".to_string(),
                threshold_usd: 0.01,
                window: AlertWindow::Monthly,
                channel: AlertChannel::Log,
                cooldown: Duration::from_secs(0),
            },
        ];
        let mut engine = AlertEngine::new(rules);
        let alerts = engine.check(&ledger);
        assert_eq!(alerts.len(), 2);
    }

    #[test]
    fn test_summary_fired_total() {
        let ledger = ledger_with_cost(10.0);
        let mut engine = AlertEngine::new(vec![make_rule(0.01, Duration::from_secs(0))]);
        engine.check(&ledger);
        engine.check(&ledger);
        assert_eq!(engine.summary().fired_total, 2);
    }

    #[test]
    fn test_summary_rules_count() {
        let rules = vec![
            make_rule(1.0, Duration::from_secs(0)),
            make_rule(2.0, Duration::from_secs(0)),
            make_rule(3.0, Duration::from_secs(0)),
        ];
        let engine = AlertEngine::new(rules);
        assert_eq!(engine.summary().rules_count, 3);
    }

    #[test]
    fn test_alert_message_contains_rule_name() {
        let ledger = ledger_with_cost(10.0);
        let rule = AlertRule {
            name: "my-special-rule".to_string(),
            threshold_usd: 0.01,
            window: AlertWindow::Monthly,
            channel: AlertChannel::Log,
            cooldown: Duration::from_secs(0),
        };
        let mut engine = AlertEngine::new(vec![rule]);
        let alerts = engine.check(&ledger);
        assert!(alerts[0].message.contains("my-special-rule"));
    }

    #[test]
    fn test_daily_window_aggregation() {
        let ledger = ledger_with_cost(5.0);
        let rule = AlertRule {
            name: "daily-rule".to_string(),
            threshold_usd: 0.01,
            window: AlertWindow::Daily,
            channel: AlertChannel::Log,
            cooldown: Duration::from_secs(0),
        };
        let mut engine = AlertEngine::new(vec![rule]);
        let alerts = engine.check(&ledger);
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_weekly_window_aggregation() {
        let ledger = ledger_with_cost(5.0);
        let rule = AlertRule {
            name: "weekly-rule".to_string(),
            threshold_usd: 0.01,
            window: AlertWindow::Weekly,
            channel: AlertChannel::Log,
            cooldown: Duration::from_secs(0),
        };
        let mut engine = AlertEngine::new(vec![rule]);
        let alerts = engine.check(&ledger);
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_rolling_window_aggregation() {
        let ledger = ledger_with_cost(5.0);
        let rule = AlertRule {
            name: "rolling-rule".to_string(),
            threshold_usd: 0.01,
            window: AlertWindow::Rolling(Duration::from_secs(3600)),
            channel: AlertChannel::Log,
            cooldown: Duration::from_secs(0),
        };
        let mut engine = AlertEngine::new(vec![rule]);
        let alerts = engine.check(&ledger);
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_file_channel_writes() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        let ledger = ledger_with_cost(5.0);
        let rule = AlertRule {
            name: "file-rule".to_string(),
            threshold_usd: 0.01,
            window: AlertWindow::Monthly,
            channel: AlertChannel::File { path: path.clone() },
            cooldown: Duration::from_secs(0),
        };
        let mut engine = AlertEngine::new(vec![rule]);
        engine.check(&ledger);
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("file-rule"));
    }

    #[cfg(feature = "webhooks")]
    #[test]
    fn test_hmac_sha256_rfc4231_vectors() {
        // RFC 4231 test case 2.
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // RFC 4231 test case 6: key longer than the block size.
        assert_eq!(
            hmac_sha256_hex(&[0xaa; 131], b"Test Using Larger Than Block-Size Key - Hash Key First"),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[cfg(feature = "webhooks")]
    #[test]
    fn test_hmac_sha256_is_deterministic() {
        let sig1 = hmac_sha256_hex(b"secret", b"payload");
        let sig2 = hmac_sha256_hex(b"secret", b"payload");
        assert_eq!(sig1, sig2);
    }

    #[cfg(feature = "webhooks")]
    #[test]
    fn test_hmac_sha256_differs_with_different_key() {
        let sig1 = hmac_sha256_hex(b"key1", b"payload");
        let sig2 = hmac_sha256_hex(b"key2", b"payload");
        assert_ne!(sig1, sig2);
    }

    #[cfg(feature = "webhooks")]
    #[test]
    fn test_hmac_sha256_known_value() {
        // HMAC-SHA256 of "" with key "" should equal
        // b613679a0814d9ec772f95d778c35fc5ff1697c493715653c6c712144292c5ad
        let sig = hmac_sha256_hex(b"", b"");
        assert_eq!(sig, "b613679a0814d9ec772f95d778c35fc5ff1697c493715653c6c712144292c5ad");
    }

    #[test]
    fn test_no_rules_yields_empty_alerts() {
        let ledger = ledger_with_cost(100.0);
        let mut engine = AlertEngine::new(vec![]);
        assert!(engine.check(&ledger).is_empty());
    }
}
