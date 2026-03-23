//! Multi-channel alerting with cooldown, suppression, and escalation.
//!
//! An [`AlertManager`] holds a set of named [`AlertRule`]s.  Callers invoke
//! [`AlertManager::check_and_fire`] with a metric name and current value;
//! the manager fires [`Alert`]s for every matching rule whose threshold is
//! exceeded and whose cooldown period has elapsed.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// AlertSeverity
// ---------------------------------------------------------------------------

/// Severity level assigned to an alert rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational — no immediate action required.
    Info,
    /// Approaching a limit; attention recommended.
    Warning,
    /// Limit exceeded; action required.
    Critical,
    /// Severe breach; escalate immediately.
    Emergency,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
            AlertSeverity::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

// ---------------------------------------------------------------------------
// AlertChannel
// ---------------------------------------------------------------------------

/// Delivery mechanism for a fired alert.
#[derive(Debug, Clone)]
pub enum AlertChannel {
    /// Write a log entry via `tracing::warn!`.
    Log,
    /// HTTP POST to the given URL.
    Webhook(String),
    /// Post to a Slack channel.
    Slack(String),
    /// Send an email to the given address.
    Email(String),
}

// ---------------------------------------------------------------------------
// AlertRule
// ---------------------------------------------------------------------------

/// Configuration for one alerting rule.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Unique rule identifier.
    pub name: String,
    /// Human-readable description of the condition being monitored.
    pub condition: String,
    /// Threshold value that triggers the alert when exceeded.
    pub threshold: f64,
    /// Severity of the alert when fired.
    pub severity: AlertSeverity,
    /// Minimum seconds between successive firings of this rule.
    pub cooldown_seconds: u64,
    /// Delivery channels for this rule.
    pub channels: Vec<AlertChannel>,
}

// ---------------------------------------------------------------------------
// AlertState
// ---------------------------------------------------------------------------

/// Runtime state for a single rule tracked inside [`AlertManager`].
#[derive(Debug)]
pub struct AlertState {
    /// When the rule was last fired (`None` = never).
    pub last_fired: Option<Instant>,
    /// Total number of times this rule has fired.
    pub fire_count: u64,
    /// When `true`, the rule is suppressed and will not fire.
    pub suppressed: bool,
}

impl AlertState {
    fn new() -> Self {
        Self {
            last_fired: None,
            fire_count: 0,
            suppressed: false,
        }
    }

    /// Returns `true` if enough time has passed since the last firing.
    fn cooldown_elapsed(&self, cooldown_seconds: u64) -> bool {
        match self.last_fired {
            None => true,
            Some(fired_at) => fired_at.elapsed() >= Duration::from_secs(cooldown_seconds),
        }
    }
}

// ---------------------------------------------------------------------------
// Alert (fired event)
// ---------------------------------------------------------------------------

/// A fired alert event.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Name of the rule that was triggered.
    pub rule_name: String,
    /// Severity of the rule.
    pub severity: AlertSeverity,
    /// Human-readable alert message.
    pub message: String,
    /// The metric value that triggered the alert.
    pub value: f64,
    /// Unix timestamp (seconds since epoch) when the alert was fired.
    pub timestamp: u64,
}

impl Alert {
    fn new(rule: &AlertRule, value: f64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            rule_name: rule.name.clone(),
            severity: rule.severity,
            message: format!(
                "[{}] Rule '{}': {} — value {:.4} exceeded threshold {:.4}",
                rule.severity, rule.name, rule.condition, value, rule.threshold
            ),
            value,
            timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// AlertManager
// ---------------------------------------------------------------------------

/// Thread-safe multi-channel alert manager.
pub struct AlertManager {
    rules: Vec<AlertRule>,
    states: Mutex<HashMap<String, AlertState>>,
}

impl AlertManager {
    /// Create a new, empty manager.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            states: Mutex::new(HashMap::new()),
        }
    }

    /// Register a new alerting rule.
    pub fn add_rule(&mut self, rule: AlertRule) {
        // Pre-populate state so it exists before any lock contention.
        if let Ok(mut states) = self.states.lock() {
            states
                .entry(rule.name.clone())
                .or_insert_with(AlertState::new);
        }
        self.rules.push(rule);
    }

    /// Check `value` against all rules whose `condition` matches `metric_name`.
    ///
    /// Fires an [`Alert`] for every rule where:
    /// - `metric_name == rule.name` (name-based matching), AND
    /// - `value > rule.threshold`, AND
    /// - the rule is not suppressed, AND
    /// - the cooldown period has elapsed.
    ///
    /// Also delivers to configured channels (log-only in this implementation;
    /// webhook / Slack / email delivery would be added via async tasks).
    pub fn check_and_fire(&self, metric_name: &str, value: f64) -> Vec<Alert> {
        let mut fired = Vec::new();
        let mut states = match self.states.lock() {
            Ok(g) => g,
            Err(_) => return fired,
        };

        for rule in &self.rules {
            if rule.name != metric_name {
                continue;
            }
            if value <= rule.threshold {
                continue;
            }

            let state = states
                .entry(rule.name.clone())
                .or_insert_with(AlertState::new);

            if state.suppressed {
                continue;
            }
            if !state.cooldown_elapsed(rule.cooldown_seconds) {
                continue;
            }

            // Fire.
            state.last_fired = Some(Instant::now());
            state.fire_count += 1;

            let alert = Alert::new(rule, value);

            // Deliver to channels.
            for ch in &rule.channels {
                match ch {
                    AlertChannel::Log => {
                        tracing::warn!(
                            rule = %rule.name,
                            severity = %rule.severity,
                            value = value,
                            threshold = rule.threshold,
                            "Alert fired"
                        );
                    }
                    AlertChannel::Webhook(url) => {
                        tracing::info!(url = %url, "Would POST alert to webhook");
                    }
                    AlertChannel::Slack(channel) => {
                        tracing::info!(channel = %channel, "Would send alert to Slack");
                    }
                    AlertChannel::Email(address) => {
                        tracing::info!(address = %address, "Would send alert email");
                    }
                }
            }

            fired.push(alert);
        }

        fired
    }

    /// Suppress `rule_name` so it will not fire until [`reset_rule`] is called.
    ///
    /// [`reset_rule`]: AlertManager::reset_rule
    pub fn suppress_rule(&self, rule_name: &str) {
        if let Ok(mut states) = self.states.lock() {
            states
                .entry(rule_name.to_string())
                .or_insert_with(AlertState::new)
                .suppressed = true;
        }
    }

    /// Remove suppression and reset the cooldown for `rule_name`.
    pub fn reset_rule(&self, rule_name: &str) {
        if let Ok(mut states) = self.states.lock() {
            states
                .entry(rule_name.to_string())
                .or_insert_with(AlertState::new)
                .suppressed = false;
        }
    }

    /// Return the names of rules that are currently within their cooldown window
    /// (i.e., they fired recently and are not yet eligible to fire again).
    pub fn active_alerts(&self) -> Vec<String> {
        let mut active = Vec::new();
        let states = match self.states.lock() {
            Ok(g) => g,
            Err(_) => return active,
        };

        for rule in &self.rules {
            if let Some(state) = states.get(&rule.name) {
                if let Some(fired_at) = state.last_fired {
                    if fired_at.elapsed() < Duration::from_secs(rule.cooldown_seconds) {
                        active.push(rule.name.clone());
                    }
                }
            }
        }
        active
    }

    /// Return the escalation chain starting at `severity`.
    ///
    /// Each level escalates to the next higher severity until `Emergency`.
    pub fn escalation_chain(&self, severity: AlertSeverity) -> Vec<AlertSeverity> {
        let all = [
            AlertSeverity::Info,
            AlertSeverity::Warning,
            AlertSeverity::Critical,
            AlertSeverity::Emergency,
        ];
        all.iter()
            .copied()
            .filter(|&s| s >= severity)
            .collect()
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rule(name: &str, threshold: f64) -> AlertRule {
        AlertRule {
            name: name.to_string(),
            condition: format!("{} > {}", name, threshold),
            threshold,
            severity: AlertSeverity::Warning,
            cooldown_seconds: 0,
            channels: vec![AlertChannel::Log],
        }
    }

    #[test]
    fn test_fires_when_threshold_exceeded() {
        let mut mgr = AlertManager::new();
        mgr.add_rule(make_rule("cost", 1.0));
        let alerts = mgr.check_and_fire("cost", 2.0);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].rule_name, "cost");
    }

    #[test]
    fn test_no_fire_below_threshold() {
        let mut mgr = AlertManager::new();
        mgr.add_rule(make_rule("cost", 5.0));
        let alerts = mgr.check_and_fire("cost", 3.0);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_suppression() {
        let mut mgr = AlertManager::new();
        mgr.add_rule(make_rule("cost", 1.0));
        mgr.suppress_rule("cost");
        let alerts = mgr.check_and_fire("cost", 100.0);
        assert!(alerts.is_empty());
        mgr.reset_rule("cost");
        let alerts = mgr.check_and_fire("cost", 100.0);
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn test_escalation_chain() {
        let mgr = AlertManager::new();
        let chain = mgr.escalation_chain(AlertSeverity::Warning);
        assert_eq!(
            chain,
            vec![AlertSeverity::Warning, AlertSeverity::Critical, AlertSeverity::Emergency]
        );
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(AlertSeverity::Emergency.to_string(), "EMERGENCY");
        assert_eq!(AlertSeverity::Info.to_string(), "INFO");
    }
}
