//! # Notification Dispatcher
//!
//! Multi-channel notification dispatching for cost alerts, budget breaches,
//! and operational events. Supports Webhook, Log, InMemory, Slack, and
//! PagerDuty channels. For HTTP-based channels (Webhook, Slack, PagerDuty)
//! no real HTTP calls are made — dispatch is logged and stored in the InMemory
//! buffer, making this safe to use in unit tests and offline environments.

use std::collections::HashMap;

/// Where a notification should be delivered.
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    /// Generic webhook endpoint.
    Webhook {
        /// The URL to POST to.
        url: String,
        /// Optional HMAC secret for request signing.
        secret: Option<String>,
    },
    /// Emit via the `tracing` log macros.
    Log,
    /// Store in an in-process Vec (useful for testing and auditing).
    InMemory,
    /// Slack incoming webhook.
    Slack {
        /// The Slack webhook URL.
        webhook_url: String,
    },
    /// PagerDuty Events API v2.
    PagerDuty {
        /// The PagerDuty integration routing key.
        routing_key: String,
    },
}

/// Severity level for a notification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Severity {
    /// Informational — no action required.
    Info,
    /// Warning — attention recommended.
    Warning,
    /// Critical — immediate action required.
    Critical,
    /// Recovery — a previously critical condition has resolved.
    Recovery,
}

impl Severity {
    fn as_str(&self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Critical => "critical",
            Severity::Recovery => "recovery",
        }
    }
}

/// A notification record.
#[derive(Debug, Clone)]
pub struct Notification {
    /// Unique notification identifier.
    pub id: String,
    /// The channel this notification was sent on.
    pub channel: NotificationChannel,
    /// Subject / title line.
    pub subject: String,
    /// Full notification body.
    pub body: String,
    /// Severity level.
    pub severity: Severity,
    /// Unix timestamp when the notification was sent successfully, if any.
    pub sent_at: Option<u64>,
    /// Error message if dispatch failed.
    pub error: Option<String>,
    /// Internal retry counter.
    retry_count: u32,
}

/// A template for rendering notification subject and body from variables.
#[derive(Debug, Clone)]
pub struct NotificationTemplate {
    /// Subject template string; use `{{var_name}}` placeholders.
    pub subject_template: String,
    /// Body template string; use `{{var_name}}` placeholders.
    pub body_template: String,
}

impl NotificationTemplate {
    /// Create a new template.
    pub fn new(subject_template: impl Into<String>, body_template: impl Into<String>) -> Self {
        Self {
            subject_template: subject_template.into(),
            body_template: body_template.into(),
        }
    }

    /// Render the template, replacing `{{var}}` placeholders with values from `vars`.
    ///
    /// Returns `(rendered_subject, rendered_body)`.
    pub fn render(&self, vars: &HashMap<String, String>) -> (String, String) {
        (render_template(&self.subject_template, vars), render_template(&self.body_template, vars))
    }
}

fn render_template(template: &str, vars: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (key, value) in vars {
        let placeholder = format!("{{{{{key}}}}}");
        result = result.replace(&placeholder, value);
    }
    result
}

/// Errors returned by the notification dispatcher.
#[derive(Debug, thiserror::Error)]
pub enum NotifError {
    /// The named channel was not found in the dispatcher.
    #[error("notification channel not found")]
    ChannelNotFound,
    /// Sending the notification failed.
    #[error("send failed: {0}")]
    SendFailed(String),
    /// Template rendering produced an invalid result.
    #[error("template error: {0}")]
    TemplateError(String),
}

/// Aggregated statistics from the dispatcher.
#[derive(Debug, Clone, Default)]
pub struct NotifStats {
    /// Total number of dispatch attempts.
    pub total_dispatched: u64,
    /// Number of successful dispatches.
    pub successful: u64,
    /// Number of failed dispatches.
    pub failed: u64,
    /// Count of dispatches broken down by severity label.
    pub by_severity: HashMap<String, u64>,
}

/// Multi-channel notification dispatcher.
///
/// Channels are registered by name. Dispatch routes a notification to the
/// named channel, stores it in the InMemory buffer (for all channel types),
/// and logs via `tracing` for Log channels.
///
/// For HTTP channels (Webhook, Slack, PagerDuty) no real HTTP request is
/// issued — the intent is recorded in the log and stored in-memory.
pub struct NotificationDispatcher {
    /// Registered channels by name.
    channels: HashMap<String, NotificationChannel>,
    /// All dispatched notifications (regardless of channel).
    notifications: Vec<Notification>,
    /// Running statistics.
    stats: NotifStats,
    /// Auto-incrementing notification ID counter.
    next_id: u64,
}

impl NotificationDispatcher {
    /// Create an empty dispatcher.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            notifications: Vec::new(),
            stats: NotifStats::default(),
            next_id: 0,
        }
    }

    /// Register a named channel.
    ///
    /// If a channel with the same name already exists it is replaced.
    pub fn add_channel(&mut self, name: &str, channel: NotificationChannel) {
        self.channels.insert(name.to_string(), channel);
    }

    /// Dispatch a notification to the named channel.
    ///
    /// Returns `Ok(())` on success or a [`NotifError`] on failure.
    pub fn dispatch(
        &mut self,
        name: &str,
        subject: &str,
        body: &str,
        severity: Severity,
    ) -> Result<(), NotifError> {
        let channel = self.channels.get(name).cloned().ok_or(NotifError::ChannelNotFound)?;

        let id = format!("notif-{}", self.next_id);
        self.next_id += 1;

        self.stats.total_dispatched += 1;
        *self.stats.by_severity.entry(severity.as_str().to_string()).or_default() += 1;

        let (sent_at, error) = self.dispatch_to_channel(&channel, subject, body, &severity);

        if error.is_none() {
            self.stats.successful += 1;
        } else {
            self.stats.failed += 1;
        }

        self.notifications.push(Notification {
            id,
            channel,
            subject: subject.to_string(),
            body: body.to_string(),
            severity,
            sent_at,
            error,
            retry_count: 0,
        });

        Ok(())
    }

    /// Internal: perform the actual channel dispatch.
    /// Returns (sent_at, error).
    fn dispatch_to_channel(
        &self,
        channel: &NotificationChannel,
        subject: &str,
        body: &str,
        severity: &Severity,
    ) -> (Option<u64>, Option<String>) {
        match channel {
            NotificationChannel::InMemory => {
                // In-memory stores don't need extra handling — the notification
                // is always stored in self.notifications by the caller.
                (Some(0), None)
            }
            NotificationChannel::Log => {
                match severity {
                    Severity::Critical => tracing::error!(subject, body, "notification dispatched"),
                    Severity::Warning => tracing::warn!(subject, body, "notification dispatched"),
                    Severity::Recovery => tracing::info!(subject, body, "notification dispatched (recovery)"),
                    Severity::Info => tracing::info!(subject, body, "notification dispatched"),
                }
                (Some(0), None)
            }
            NotificationChannel::Webhook { url, secret } => {
                tracing::info!(
                    url,
                    subject,
                    secret_set = secret.is_some(),
                    "would send webhook notification"
                );
                (Some(0), None)
            }
            NotificationChannel::Slack { webhook_url } => {
                tracing::info!(webhook_url, subject, "would send Slack notification");
                (Some(0), None)
            }
            NotificationChannel::PagerDuty { routing_key } => {
                tracing::info!(routing_key, subject, "would send PagerDuty notification");
                (Some(0), None)
            }
        }
    }

    /// Return all stored notifications (from InMemory and simulated channels).
    pub fn dispatched_notifications(&self) -> Vec<&Notification> {
        self.notifications.iter().collect()
    }

    /// Retry all notifications that have an error, up to a maximum of 3 attempts.
    ///
    /// Clears the error on successful retry.
    pub fn retry_failed(&mut self) {
        for notif in self.notifications.iter_mut() {
            if notif.error.is_some() && notif.retry_count < 3 {
                notif.retry_count += 1;
                // In this implementation re-dispatch always succeeds for
                // simulated channels — clear the error to simulate recovery.
                notif.error = None;
                notif.sent_at = Some(0);
                self.stats.failed = self.stats.failed.saturating_sub(1);
                self.stats.successful += 1;
            }
        }
    }

    /// Return current dispatcher statistics.
    pub fn stats(&self) -> NotifStats {
        self.stats.clone()
    }
}

impl Default for NotificationDispatcher {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inmemory_stores_notifications() {
        let mut dispatcher = NotificationDispatcher::new();
        dispatcher.add_channel("alerts", NotificationChannel::InMemory);

        dispatcher.dispatch("alerts", "Test Subject", "Test Body", Severity::Info).unwrap();
        dispatcher.dispatch("alerts", "Warning Subject", "Warning Body", Severity::Warning).unwrap();

        let stored = dispatcher.dispatched_notifications();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].subject, "Test Subject");
        assert_eq!(stored[1].severity, Severity::Warning);
    }

    #[test]
    fn channel_not_found_returns_error() {
        let mut dispatcher = NotificationDispatcher::new();
        let result = dispatcher.dispatch("nonexistent", "S", "B", Severity::Info);
        assert!(matches!(result, Err(NotifError::ChannelNotFound)));
    }

    #[test]
    fn template_renders_placeholders() {
        let template = NotificationTemplate::new(
            "Alert: {{model}} exceeded {{threshold}}%",
            "Model {{model}} cost ${{cost}} (threshold: {{threshold}}%)",
        );
        let mut vars = HashMap::new();
        vars.insert("model".to_string(), "gpt-4".to_string());
        vars.insert("threshold".to_string(), "90".to_string());
        vars.insert("cost".to_string(), "42.50".to_string());

        let (subject, body) = template.render(&vars);
        assert_eq!(subject, "Alert: gpt-4 exceeded 90%");
        assert_eq!(body, "Model gpt-4 cost $42.50 (threshold: 90%)");
    }

    #[test]
    fn template_leaves_unknown_placeholders_unchanged() {
        let template = NotificationTemplate::new("Hello {{name}}", "{{unknown}} var");
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());

        let (subject, body) = template.render(&vars);
        assert_eq!(subject, "Hello World");
        assert_eq!(body, "{{unknown}} var");
    }

    #[test]
    fn stats_count_by_severity() {
        let mut dispatcher = NotificationDispatcher::new();
        dispatcher.add_channel("ch", NotificationChannel::InMemory);

        dispatcher.dispatch("ch", "s", "b", Severity::Info).unwrap();
        dispatcher.dispatch("ch", "s", "b", Severity::Info).unwrap();
        dispatcher.dispatch("ch", "s", "b", Severity::Critical).unwrap();
        dispatcher.dispatch("ch", "s", "b", Severity::Warning).unwrap();

        let stats = dispatcher.stats();
        assert_eq!(stats.total_dispatched, 4);
        assert_eq!(stats.successful, 4);
        assert_eq!(stats.failed, 0);
        assert_eq!(*stats.by_severity.get("info").unwrap_or(&0), 2);
        assert_eq!(*stats.by_severity.get("critical").unwrap_or(&0), 1);
        assert_eq!(*stats.by_severity.get("warning").unwrap_or(&0), 1);
    }

    #[test]
    fn retry_clears_error_state() {
        let mut dispatcher = NotificationDispatcher::new();
        dispatcher.add_channel("ch", NotificationChannel::InMemory);
        dispatcher.dispatch("ch", "s", "b", Severity::Info).unwrap();

        // Manually inject an error into the first notification.
        dispatcher.notifications[0].error = Some("simulated failure".to_string());
        dispatcher.stats.failed += 1;
        dispatcher.stats.successful = dispatcher.stats.successful.saturating_sub(1);

        assert!(dispatcher.notifications[0].error.is_some());

        dispatcher.retry_failed();

        assert!(dispatcher.notifications[0].error.is_none(), "error should be cleared after retry");
        assert_eq!(dispatcher.stats().failed, 0);
    }

    #[test]
    fn retry_respects_max_retry_count() {
        let mut dispatcher = NotificationDispatcher::new();
        dispatcher.add_channel("ch", NotificationChannel::InMemory);
        dispatcher.dispatch("ch", "s", "b", Severity::Critical).unwrap();

        // Inject an error and set retry_count to 3 (already at max).
        dispatcher.notifications[0].error = Some("persistent failure".to_string());
        dispatcher.notifications[0].retry_count = 3;

        dispatcher.retry_failed();

        // Should not have been retried — error remains.
        assert!(dispatcher.notifications[0].error.is_some());
    }

    #[test]
    fn multiple_channels_work_independently() {
        let mut dispatcher = NotificationDispatcher::new();
        dispatcher.add_channel("mem", NotificationChannel::InMemory);
        dispatcher.add_channel("log", NotificationChannel::Log);
        dispatcher.add_channel("slack", NotificationChannel::Slack {
            webhook_url: "https://hooks.slack.com/test".to_string(),
        });

        dispatcher.dispatch("mem", "Memory alert", "body", Severity::Warning).unwrap();
        dispatcher.dispatch("log", "Log alert", "body", Severity::Info).unwrap();
        dispatcher.dispatch("slack", "Slack alert", "body", Severity::Critical).unwrap();

        assert_eq!(dispatcher.dispatched_notifications().len(), 3);
        assert_eq!(dispatcher.stats().total_dispatched, 3);
    }
}
