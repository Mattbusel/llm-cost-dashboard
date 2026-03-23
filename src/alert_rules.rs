//! # Alert Rules Engine
//!
//! A rule engine for complex alert conditions including threshold checks,
//! rate-of-change detection, composite logic, and pattern matching.
//!
//! ## Example
//!
//! ```rust
//! use std::collections::HashMap;
//! use llm_cost_dashboard::alert_rules::{
//!     AlertRule, AlertSeverity, Condition, Logic, MetricSnapshot, Operator, RuleEngine,
//! };
//!
//! let mut engine = RuleEngine::new();
//! engine.add_rule(AlertRule {
//!     id: "high-cost".to_string(),
//!     name: "High Daily Cost".to_string(),
//!     condition: Condition::ThresholdExceeded {
//!         metric: "daily_cost_usd".to_string(),
//!         threshold: 50.0,
//!         operator: Operator::Gt,
//!     },
//!     severity: AlertSeverity::Warning,
//!     cooldown_seconds: 300,
//!     message_template: "Daily cost is {{daily_cost_usd}} USD".to_string(),
//!     last_fired: None,
//! });
//!
//! let mut values = HashMap::new();
//! values.insert("daily_cost_usd".to_string(), 75.0);
//! let snapshot = MetricSnapshot {
//!     values,
//!     history: HashMap::new(),
//!     timestamp: 1_000_000,
//! };
//!
//! let firings = engine.evaluate(&snapshot);
//! assert_eq!(firings.len(), 1);
//! ```

use std::collections::{HashMap, VecDeque};

// ── Operator ─────────────────────────────────────────────────────────────────

/// Comparison operator used in threshold conditions.
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    /// Greater than (`>`).
    Gt,
    /// Greater than or equal to (`>=`).
    Gte,
    /// Less than (`<`).
    Lt,
    /// Less than or equal to (`<=`).
    Lte,
    /// Equal to (`==`).
    Eq,
    /// Not equal to (`!=`).
    Ne,
}

impl Operator {
    /// Applies the operator to two f64 values.
    pub fn apply(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            Operator::Gt => lhs > rhs,
            Operator::Gte => lhs >= rhs,
            Operator::Lt => lhs < rhs,
            Operator::Lte => lhs <= rhs,
            Operator::Eq => (lhs - rhs).abs() < f64::EPSILON,
            Operator::Ne => (lhs - rhs).abs() >= f64::EPSILON,
        }
    }
}

// ── Logic ────────────────────────────────────────────────────────────────────

/// Boolean combinator for composite conditions.
#[derive(Debug, Clone, PartialEq)]
pub enum Logic {
    /// Both sub-conditions must be true.
    And,
    /// At least one sub-condition must be true.
    Or,
    /// The left sub-condition is negated (right is ignored for `Not`).
    Not,
}

// ── Condition ────────────────────────────────────────────────────────────────

/// An alert condition expression.
#[derive(Debug, Clone)]
pub enum Condition {
    /// Fires when `metric operator threshold` is true.
    ThresholdExceeded {
        /// Name of the metric to check.
        metric: String,
        /// The value to compare against.
        threshold: f64,
        /// The comparison operator.
        operator: Operator,
    },
    /// Fires when the metric has changed by more than `pct_change` percent
    /// within `window_seconds`.
    RateOfChange {
        /// Name of the metric to check.
        metric: String,
        /// Required percentage change to trigger (absolute value).
        pct_change: f64,
        /// Look-back window in seconds.
        window_seconds: u64,
    },
    /// Combines two conditions with `And`, `Or`, or `Not` logic.
    Composite {
        /// Left-hand side condition.
        left: Box<Condition>,
        /// Right-hand side condition (ignored for `Not`).
        right: Box<Condition>,
        /// Boolean combinator.
        logic: Logic,
    },
    /// Fires when the metric name matches the given substring pattern.
    ///
    /// The pattern is checked as a case-sensitive substring of the metric
    /// value formatted as a string.  In practice this is a simple string
    /// match used for label/tag filtering.
    PatternMatch {
        /// Metric key whose string representation to match.
        metric: String,
        /// Substring pattern to search for.
        pattern: String,
    },
}

// ── AlertSeverity ─────────────────────────────────────────────────────────────

/// Severity level of a fired alert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Informational — no action required.
    Info,
    /// Warning — action may be needed soon.
    Warning,
    /// Critical — immediate action required.
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ── AlertRule ────────────────────────────────────────────────────────────────

/// A named alert rule with a condition, severity, cooldown, and message
/// template.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Unique rule identifier.
    pub id: String,
    /// Human-readable rule name.
    pub name: String,
    /// The condition that triggers this alert.
    pub condition: Condition,
    /// How severe this alert is when fired.
    pub severity: AlertSeverity,
    /// Minimum seconds that must elapse between consecutive firings of this
    /// rule.
    pub cooldown_seconds: u64,
    /// Message template; use `{{metric_name}}` placeholders for current
    /// metric values.
    pub message_template: String,
    /// Unix timestamp (seconds) of the most recent firing, or `None` if
    /// the rule has never fired.
    pub last_fired: Option<u64>,
}

// ── MetricSnapshot ───────────────────────────────────────────────────────────

/// A point-in-time snapshot of metric values with historical context.
#[derive(Debug, Clone)]
pub struct MetricSnapshot {
    /// Current metric values keyed by metric name.
    pub values: HashMap<String, f64>,
    /// Historical time series for rate-of-change checks.  Each entry is a
    /// deque of `(unix_timestamp_seconds, value)` pairs ordered oldest-first.
    pub history: HashMap<String, VecDeque<(u64, f64)>>,
    /// Unix timestamp in seconds at which this snapshot was taken.
    pub timestamp: u64,
}

impl MetricSnapshot {
    /// Creates a new snapshot with no history.
    pub fn new(values: HashMap<String, f64>, timestamp: u64) -> Self {
        Self {
            values,
            history: HashMap::new(),
            timestamp,
        }
    }
}

// ── AlertFiring ──────────────────────────────────────────────────────────────

/// Describes a single rule firing event.
#[derive(Debug, Clone)]
pub struct AlertFiring {
    /// ID of the rule that fired.
    pub rule_id: String,
    /// Name of the rule that fired.
    pub rule_name: String,
    /// Severity of the rule.
    pub severity: AlertSeverity,
    /// Rendered alert message.
    pub message: String,
    /// Unix timestamp (seconds) when the alert fired.
    pub fired_at: u64,
}

// ── format_message ────────────────────────────────────────────────────────────

/// Renders `template` by replacing `{{metric_name}}` placeholders with
/// the current value from `snapshot`.
///
/// Unknown placeholders are left as-is.
pub fn format_message(template: &str, snapshot: &MetricSnapshot) -> String {
    let mut result = template.to_string();
    for (name, value) in &snapshot.values {
        let placeholder = format!("{{{{{}}}}}", name);
        result = result.replace(&placeholder, &format!("{:.4}", value));
    }
    result
}

// ── RuleEngine ────────────────────────────────────────────────────────────────

/// Holds a collection of [`AlertRule`]s and evaluates them against metric
/// snapshots.
#[derive(Debug, Default)]
pub struct RuleEngine {
    /// The rules registered with this engine.
    pub rules: Vec<AlertRule>,
}

impl RuleEngine {
    /// Creates an empty rule engine.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Registers a new rule.
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Evaluates all rules against `snapshot` and returns the list of rules
    /// that fired (respecting cooldowns).
    pub fn evaluate(&mut self, snapshot: &MetricSnapshot) -> Vec<AlertFiring> {
        let now = snapshot.timestamp;
        let mut firings = Vec::new();

        for rule in &mut self.rules {
            // Cooldown check.
            if let Some(last) = rule.last_fired {
                if now.saturating_sub(last) < rule.cooldown_seconds {
                    continue;
                }
            }

            if Self::evaluate_condition(&rule.condition, snapshot) {
                let message = format_message(&rule.message_template, snapshot);
                rule.last_fired = Some(now);
                firings.push(AlertFiring {
                    rule_id: rule.id.clone(),
                    rule_name: rule.name.clone(),
                    severity: rule.severity.clone(),
                    message,
                    fired_at: now,
                });
            }
        }

        firings
    }

    /// Evaluates a single [`Condition`] against the snapshot.
    pub fn evaluate_condition(condition: &Condition, snapshot: &MetricSnapshot) -> bool {
        match condition {
            Condition::ThresholdExceeded {
                metric,
                threshold,
                operator,
            } => {
                if let Some(&value) = snapshot.values.get(metric) {
                    operator.apply(value, *threshold)
                } else {
                    false
                }
            }

            Condition::RateOfChange {
                metric,
                pct_change,
                window_seconds,
            } => {
                let history = match snapshot.history.get(metric) {
                    Some(h) => h,
                    None => return false,
                };
                let current_value = match snapshot.values.get(metric) {
                    Some(&v) => v,
                    None => return false,
                };
                let cutoff = snapshot.timestamp.saturating_sub(*window_seconds);
                // Find the oldest value within the window.
                let baseline = history
                    .iter()
                    .filter(|(ts, _)| *ts >= cutoff)
                    .min_by_key(|(ts, _)| *ts)
                    .map(|(_, v)| *v);

                if let Some(baseline_value) = baseline {
                    if baseline_value.abs() < f64::EPSILON {
                        return false;
                    }
                    let change = ((current_value - baseline_value) / baseline_value).abs() * 100.0;
                    change >= *pct_change
                } else {
                    false
                }
            }

            Condition::Composite { left, right, logic } => match logic {
                Logic::And => {
                    Self::evaluate_condition(left, snapshot)
                        && Self::evaluate_condition(right, snapshot)
                }
                Logic::Or => {
                    Self::evaluate_condition(left, snapshot)
                        || Self::evaluate_condition(right, snapshot)
                }
                Logic::Not => !Self::evaluate_condition(left, snapshot),
            },

            Condition::PatternMatch { metric, pattern } => {
                if let Some(&value) = snapshot.values.get(metric) {
                    format!("{:.4}", value).contains(pattern.as_str())
                        || metric.contains(pattern.as_str())
                } else {
                    false
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_snapshot(kvs: &[(&str, f64)], timestamp: u64) -> MetricSnapshot {
        let values: HashMap<String, f64> = kvs
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        MetricSnapshot::new(values, timestamp)
    }

    fn simple_rule(id: &str, metric: &str, threshold: f64, op: Operator) -> AlertRule {
        AlertRule {
            id: id.to_string(),
            name: format!("Rule {}", id),
            condition: Condition::ThresholdExceeded {
                metric: metric.to_string(),
                threshold,
                operator: op,
            },
            severity: AlertSeverity::Warning,
            cooldown_seconds: 0,
            message_template: format!("{{{{{}}}}} threshold exceeded", metric),
            last_fired: None,
        }
    }

    // ── Threshold ─────────────────────────────────────────────────────────────

    #[test]
    fn threshold_fires_when_exceeded() {
        let mut engine = RuleEngine::new();
        engine.add_rule(simple_rule("r1", "cost", 10.0, Operator::Gt));

        let snap = make_snapshot(&[("cost", 15.0)], 1000);
        let firings = engine.evaluate(&snap);
        assert_eq!(firings.len(), 1);
        assert_eq!(firings[0].rule_id, "r1");
    }

    #[test]
    fn threshold_does_not_fire_below() {
        let mut engine = RuleEngine::new();
        engine.add_rule(simple_rule("r1", "cost", 10.0, Operator::Gt));

        let snap = make_snapshot(&[("cost", 5.0)], 1000);
        let firings = engine.evaluate(&snap);
        assert!(firings.is_empty());
    }

    #[test]
    fn threshold_gte_fires_at_exact_value() {
        let mut engine = RuleEngine::new();
        engine.add_rule(simple_rule("r1", "cost", 10.0, Operator::Gte));

        let snap = make_snapshot(&[("cost", 10.0)], 1000);
        let firings = engine.evaluate(&snap);
        assert_eq!(firings.len(), 1);
    }

    #[test]
    fn threshold_missing_metric_does_not_fire() {
        let mut engine = RuleEngine::new();
        engine.add_rule(simple_rule("r1", "cost", 10.0, Operator::Gt));

        let snap = make_snapshot(&[("other_metric", 100.0)], 1000);
        let firings = engine.evaluate(&snap);
        assert!(firings.is_empty());
    }

    // ── Composite And / Or ────────────────────────────────────────────────────

    #[test]
    fn composite_and_requires_both() {
        let condition = Condition::Composite {
            left: Box::new(Condition::ThresholdExceeded {
                metric: "cost".to_string(),
                threshold: 10.0,
                operator: Operator::Gt,
            }),
            right: Box::new(Condition::ThresholdExceeded {
                metric: "latency".to_string(),
                threshold: 200.0,
                operator: Operator::Gt,
            }),
            logic: Logic::And,
        };

        let snap_both = make_snapshot(&[("cost", 15.0), ("latency", 250.0)], 1000);
        assert!(RuleEngine::evaluate_condition(&condition, &snap_both));

        let snap_one = make_snapshot(&[("cost", 15.0), ("latency", 100.0)], 1000);
        assert!(!RuleEngine::evaluate_condition(&condition, &snap_one));
    }

    #[test]
    fn composite_or_fires_on_either() {
        let condition = Condition::Composite {
            left: Box::new(Condition::ThresholdExceeded {
                metric: "cost".to_string(),
                threshold: 10.0,
                operator: Operator::Gt,
            }),
            right: Box::new(Condition::ThresholdExceeded {
                metric: "latency".to_string(),
                threshold: 200.0,
                operator: Operator::Gt,
            }),
            logic: Logic::Or,
        };

        let snap_left_only = make_snapshot(&[("cost", 15.0), ("latency", 100.0)], 1000);
        assert!(RuleEngine::evaluate_condition(&condition, &snap_left_only));

        let snap_right_only = make_snapshot(&[("cost", 5.0), ("latency", 300.0)], 1000);
        assert!(RuleEngine::evaluate_condition(&condition, &snap_right_only));

        let snap_neither = make_snapshot(&[("cost", 5.0), ("latency", 100.0)], 1000);
        assert!(!RuleEngine::evaluate_condition(&condition, &snap_neither));
    }

    #[test]
    fn composite_not_inverts() {
        let condition = Condition::Composite {
            left: Box::new(Condition::ThresholdExceeded {
                metric: "cost".to_string(),
                threshold: 10.0,
                operator: Operator::Gt,
            }),
            right: Box::new(Condition::ThresholdExceeded {
                metric: "cost".to_string(),
                threshold: 0.0,
                operator: Operator::Gt,
            }),
            logic: Logic::Not,
        };

        let snap_high = make_snapshot(&[("cost", 20.0)], 1000);
        assert!(!RuleEngine::evaluate_condition(&condition, &snap_high));

        let snap_low = make_snapshot(&[("cost", 5.0)], 1000);
        assert!(RuleEngine::evaluate_condition(&condition, &snap_low));
    }

    // ── Cooldown ──────────────────────────────────────────────────────────────

    #[test]
    fn cooldown_suppresses_repeated_firing() {
        let mut engine = RuleEngine::new();
        let mut rule = simple_rule("r1", "cost", 10.0, Operator::Gt);
        rule.cooldown_seconds = 300;
        engine.add_rule(rule);

        let snap1 = make_snapshot(&[("cost", 20.0)], 1000);
        let firings1 = engine.evaluate(&snap1);
        assert_eq!(firings1.len(), 1, "should fire on first evaluation");

        // 100 seconds later — still within cooldown.
        let snap2 = make_snapshot(&[("cost", 20.0)], 1100);
        let firings2 = engine.evaluate(&snap2);
        assert!(firings2.is_empty(), "should be suppressed by cooldown");

        // 400 seconds after first firing — cooldown expired.
        let snap3 = make_snapshot(&[("cost", 20.0)], 1400);
        let firings3 = engine.evaluate(&snap3);
        assert_eq!(firings3.len(), 1, "should fire again after cooldown");
    }

    // ── Rate of Change ────────────────────────────────────────────────────────

    #[test]
    fn rate_of_change_detection() {
        let condition = Condition::RateOfChange {
            metric: "cost".to_string(),
            pct_change: 50.0,
            window_seconds: 60,
        };

        let mut history: VecDeque<(u64, f64)> = VecDeque::new();
        history.push_back((940, 10.0)); // 60 seconds ago: value was 10
        history.push_back((970, 12.0));

        let mut snap = make_snapshot(&[("cost", 20.0)], 1000); // current: 20 (+100%)
        snap.history.insert("cost".to_string(), history);

        assert!(
            RuleEngine::evaluate_condition(&condition, &snap),
            "100% change should exceed 50% threshold"
        );
    }

    #[test]
    fn rate_of_change_no_fire_within_threshold() {
        let condition = Condition::RateOfChange {
            metric: "cost".to_string(),
            pct_change: 50.0,
            window_seconds: 60,
        };

        let mut history: VecDeque<(u64, f64)> = VecDeque::new();
        history.push_back((950, 10.0));

        let mut snap = make_snapshot(&[("cost", 11.0)], 1000); // +10%
        snap.history.insert("cost".to_string(), history);

        assert!(
            !RuleEngine::evaluate_condition(&condition, &snap),
            "10% change should not exceed 50% threshold"
        );
    }

    // ── format_message ────────────────────────────────────────────────────────

    #[test]
    fn format_message_substitutes_placeholders() {
        let snap = make_snapshot(&[("daily_cost_usd", 75.5)], 1000);
        let msg = format_message("Cost is {{daily_cost_usd}} USD", &snap);
        assert!(msg.contains("75.5"), "should contain substituted value");
        assert!(!msg.contains("{{"), "should not contain raw placeholders");
    }

    #[test]
    fn format_message_unknown_placeholder_unchanged() {
        let snap = make_snapshot(&[], 1000);
        let msg = format_message("Cost is {{unknown_metric}} USD", &snap);
        assert!(msg.contains("{{unknown_metric}}"));
    }

    // ── Operator coverage ─────────────────────────────────────────────────────

    #[test]
    fn operators_all_work() {
        assert!(Operator::Gt.apply(2.0, 1.0));
        assert!(!Operator::Gt.apply(1.0, 1.0));
        assert!(Operator::Gte.apply(1.0, 1.0));
        assert!(Operator::Lt.apply(1.0, 2.0));
        assert!(Operator::Lte.apply(1.0, 1.0));
        assert!(Operator::Eq.apply(1.0, 1.0));
        assert!(Operator::Ne.apply(1.0, 2.0));
        assert!(!Operator::Ne.apply(1.0, 1.0));
    }

    // ── PatternMatch ──────────────────────────────────────────────────────────

    #[test]
    fn pattern_match_on_metric_name() {
        let condition = Condition::PatternMatch {
            metric: "gpt4_cost".to_string(),
            pattern: "gpt4".to_string(),
        };
        let snap = make_snapshot(&[("gpt4_cost", 5.0)], 1000);
        assert!(RuleEngine::evaluate_condition(&condition, &snap));
    }

    #[test]
    fn pattern_match_no_match() {
        let condition = Condition::PatternMatch {
            metric: "claude_cost".to_string(),
            pattern: "gpt4".to_string(),
        };
        let snap = make_snapshot(&[("claude_cost", 5.0)], 1000);
        assert!(!RuleEngine::evaluate_condition(&condition, &snap));
    }
}
