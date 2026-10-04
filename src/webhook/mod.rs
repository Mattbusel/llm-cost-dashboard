//! # Webhook Alerts
//!
//! Sends budget-threshold notifications to external HTTP endpoints such as
//! Slack incoming webhooks, Discord, or any generic JSON webhook.
//!
//! Configure one or more [`WebhookConfig`] entries and call
//! [`fire_budget_alert`] whenever a threshold is crossed.  The function is
//! async and must be awaited inside a Tokio runtime.
//!
//! ## Slack example
//!
//! ```no_run
//! use llm_cost_dashboard::webhook::{WebhookConfig, WebhookFormat, fire_budget_alert};
//!
//! # #[tokio::main]
//! # async fn main() {
//! let cfg = WebhookConfig {
//!     url: "https://hooks.slack.com/services/T.../B.../xxx".into(),
//!     format: WebhookFormat::Slack,
//!     threshold_usd: 8.0,
//! };
//! fire_budget_alert(&cfg, 8.50, 10.0).await.ok();
//! # }
//! ```

use serde::{Deserialize, Serialize};
#[cfg(feature = "webhooks")]
use tracing::{info, warn};

/// Supported webhook payload formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WebhookFormat {
    /// Slack incoming-webhook JSON (`{"text": "..."}`).
    Slack,
    /// Generic JSON (`{"text": "...", "spent_usd": ..., "limit_usd": ...}`).
    Generic,
}

/// Configuration for a single webhook endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Full URL of the webhook endpoint.
    pub url: String,
    /// Payload format to use when posting.
    pub format: WebhookFormat,
    /// Alert fires when `spent_usd` exceeds this value (USD).
    pub threshold_usd: f64,
}

/// Alert payload for a Slack webhook.
#[cfg(feature = "webhooks")]
#[derive(Serialize)]
struct SlackPayload {
    text: String,
}

/// Alert payload for a generic JSON webhook.
#[cfg(feature = "webhooks")]
#[derive(Serialize)]
struct GenericPayload {
    text: String,
    spent_usd: f64,
    limit_usd: f64,
}

/// Blocking version of [`fire_budget_alert`], for callers without a Tokio
/// runtime (the dashboard calls it from a background thread). Same payloads,
/// 10 second timeout. A no-op returning `Ok(())` without the `webhooks`
/// feature.
#[allow(unused_variables)]
pub fn fire_budget_alert_blocking(
    cfg: &WebhookConfig,
    spent_usd: f64,
    limit_usd: f64,
) -> Result<(), String> {
    #[cfg(feature = "webhooks")]
    {
        let msg = format!(
            "LLM Cost Alert: spent ${spent_usd:.4} of ${limit_usd:.4} monthly budget ({:.1}%)",
            (spent_usd / limit_usd) * 100.0,
        );
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| e.to_string())?;
        let req = client.post(&cfg.url);
        let req = match cfg.format {
            WebhookFormat::Slack => req.json(&SlackPayload { text: msg }),
            WebhookFormat::Generic => req.json(&GenericPayload {
                text: msg,
                spent_usd,
                limit_usd,
            }),
        };
        match req.send() {
            Ok(resp) if resp.status().is_success() => {
                info!(url = %cfg.url, "webhook delivered");
                Ok(())
            }
            Ok(resp) => {
                warn!(status = %resp.status(), "webhook returned non-success status");
                Err(format!("webhook HTTP {}", resp.status()))
            }
            Err(e) => {
                warn!(error = %e, "webhook delivery failed");
                Err(e.to_string())
            }
        }
    }
    #[cfg(not(feature = "webhooks"))]
    {
        Ok(())
    }
}

/// Post a budget-alert notification to a webhook endpoint.
///
/// This function is a no-op (and returns `Ok(())`) when the `webhooks` feature
/// is not compiled in; when it *is* compiled the function makes a real HTTP
/// POST with a JSON body.
///
/// `spent_usd` is the current cumulative spend; `limit_usd` is the configured
/// hard limit.
#[allow(unused_variables)]
pub async fn fire_budget_alert(
    cfg: &WebhookConfig,
    spent_usd: f64,
    limit_usd: f64,
) -> Result<(), String> {
    #[cfg(feature = "webhooks")]
    {
        use reqwest::Client;

        let msg = format!(
            "LLM Cost Alert: spent ${spent_usd:.4} of ${limit_usd:.4} monthly budget ({:.1}%)",
            (spent_usd / limit_usd) * 100.0,
        );
        info!(url = %cfg.url, spent_usd, limit_usd, "firing budget webhook");

        let client = Client::new();
        let result = match cfg.format {
            WebhookFormat::Slack => {
                client
                    .post(&cfg.url)
                    .json(&SlackPayload { text: msg })
                    .send()
                    .await
            }
            WebhookFormat::Generic => {
                client
                    .post(&cfg.url)
                    .json(&GenericPayload {
                        text: msg,
                        spent_usd,
                        limit_usd,
                    })
                    .send()
                    .await
            }
        };

        match result {
            Ok(resp) if resp.status().is_success() => {
                info!(status = %resp.status(), "webhook delivered successfully");
                Ok(())
            }
            Ok(resp) => {
                let status = resp.status();
                warn!(status = %status, "webhook returned non-success status");
                Err(format!("webhook HTTP {status}"))
            }
            Err(e) => {
                warn!(error = %e, "webhook delivery failed");
                Err(e.to_string())
            }
        }
    }

    #[cfg(not(feature = "webhooks"))]
    {
        tracing::debug!(url = %cfg.url, "webhooks feature not compiled in; skipping alert");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_config_fields() {
        let cfg = WebhookConfig {
            url: "https://example.com/hook".into(),
            format: WebhookFormat::Generic,
            threshold_usd: 5.0,
        };
        assert_eq!(cfg.threshold_usd, 5.0);
        assert_eq!(cfg.format, WebhookFormat::Generic);
    }

    #[test]
    fn test_webhook_format_slack_variant() {
        let f = WebhookFormat::Slack;
        assert_eq!(f, WebhookFormat::Slack);
    }

    #[tokio::test]
    async fn test_fire_alert_no_panic_on_bad_url() {
        // When webhooks feature is off this is a trivial Ok.
        // When it is on, the bad URL returns an Err without panicking.
        let cfg = WebhookConfig {
            url: "http://127.0.0.1:1".into(), // nothing listening
            format: WebhookFormat::Generic,
            threshold_usd: 1.0,
        };
        // Either Ok (feature off) or Err (feature on, connection refused) — never panics.
        let _ = fire_budget_alert(&cfg, 5.0, 10.0).await;
    }
}
