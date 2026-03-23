//! SLA tier cost tracking and budget alerting.
//!
//! Tracks LLM request costs broken down by **SLA tier** (Premium, Standard,
//! Economy). Each tier corresponds to a different cost/latency trade-off:
//!
//! | Tier     | Models (examples)                | Characteristics          |
//! |----------|----------------------------------|--------------------------|
//! | Premium  | `gpt-4o`, `claude-opus-4`        | Fastest, most capable, expensive |
//! | Standard | `gpt-4o-mini`, `claude-sonnet-4` | Balanced cost/performance|
//! | Economy  | `gpt-3.5-turbo`, `claude-haiku`  | Cheapest, lower latency  |
//!
//! ## Budget Alerts
//!
//! Each tier can have a per-day or per-month soft budget. When a tier's
//! accumulated cost exceeds its budget, an [`SlaAlert`] is emitted. Premium
//! overages are escalated as warnings.
//!
//! ## Usage
//!
//! ```
//! use llm_cost_dashboard::sla::{SlaTierTracker, SlaConfig, SlaTier};
//!
//! let cfg = SlaConfig::default();
//! let mut tracker = SlaTierTracker::new(cfg);
//!
//! tracker.observe("gpt-4o", 0.05);
//! tracker.observe("gpt-4o-mini", 0.002);
//! tracker.observe("gpt-3.5-turbo", 0.0005);
//!
//! let report = tracker.report();
//! for tier in &report.tiers {
//!     println!("{:?}: ${:.4} ({} requests)", tier.tier, tier.total_cost_usd, tier.request_count);
//! }
//! ```

use std::collections::HashMap;

/// An SLA tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SlaTier {
    /// Fastest, most capable models — highest cost.
    Premium,
    /// Balanced cost and capability.
    Standard,
    /// Cheapest models, suitable for high-volume low-complexity tasks.
    Economy,
    /// Model not matched to any configured tier.
    Unknown,
}

impl SlaTier {
    /// Display name for the tier.
    pub fn display_name(self) -> &'static str {
        match self {
            SlaTier::Premium  => "Premium",
            SlaTier::Standard => "Standard",
            SlaTier::Economy  => "Economy",
            SlaTier::Unknown  => "Unknown",
        }
    }
}

impl std::fmt::Display for SlaTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

/// A budget threshold exceeded event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SlaAlert {
    /// The tier that exceeded its budget.
    pub tier: SlaTier,
    /// The budget that was exceeded (USD).
    pub budget_usd: f64,
    /// The current accumulated cost (USD).
    pub current_cost_usd: f64,
    /// Overage amount: `current - budget`.
    pub overage_usd: f64,
    /// Severity: `"warning"` for Premium, `"info"` for Standard/Economy.
    pub severity: String,
}

/// Budget configuration for one SLA tier.
#[derive(Debug, Clone)]
pub struct TierBudget {
    /// Daily budget in USD. `None` = unlimited.
    pub daily_usd: Option<f64>,
    /// Monthly budget in USD. `None` = unlimited.
    pub monthly_usd: Option<f64>,
}

impl Default for TierBudget {
    fn default() -> Self {
        Self { daily_usd: None, monthly_usd: None }
    }
}

/// Full SLA configuration.
#[derive(Debug, Clone)]
pub struct SlaConfig {
    /// Model prefix/substring → SLA tier mapping. Checked in insertion order.
    pub model_tiers: Vec<(String, SlaTier)>,
    /// Per-tier budget configuration.
    pub budgets: HashMap<SlaTier, TierBudget>,
}

impl Default for SlaConfig {
    fn default() -> Self {
        let model_tiers = vec![
            // Premium.
            ("gpt-4o".to_string(),           SlaTier::Premium),
            ("claude-opus".to_string(),       SlaTier::Premium),
            ("claude-3-opus".to_string(),     SlaTier::Premium),
            ("o1".to_string(),               SlaTier::Premium),
            ("o3".to_string(),               SlaTier::Premium),
            // Standard.
            ("gpt-4o-mini".to_string(),       SlaTier::Standard),
            ("claude-sonnet".to_string(),     SlaTier::Standard),
            ("claude-3-sonnet".to_string(),   SlaTier::Standard),
            ("gpt-4-turbo".to_string(),       SlaTier::Standard),
            // Economy.
            ("gpt-3.5".to_string(),           SlaTier::Economy),
            ("claude-haiku".to_string(),      SlaTier::Economy),
            ("claude-3-haiku".to_string(),    SlaTier::Economy),
            ("gemini-flash".to_string(),      SlaTier::Economy),
            ("mistral-7b".to_string(),        SlaTier::Economy),
        ];

        let mut budgets = HashMap::new();
        budgets.insert(SlaTier::Premium,  TierBudget { daily_usd: Some(50.0),  monthly_usd: Some(500.0) });
        budgets.insert(SlaTier::Standard, TierBudget { daily_usd: Some(100.0), monthly_usd: Some(1000.0) });
        budgets.insert(SlaTier::Economy,  TierBudget { daily_usd: None,         monthly_usd: Some(200.0) });

        Self { model_tiers, budgets }
    }
}

/// Per-tier cost accumulator.
#[derive(Debug, Default, Clone)]
struct TierAccum {
    total_cost: f64,
    count: u64,
    models_seen: Vec<String>,
}

/// Per-tier summary included in the report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TierSummary {
    /// SLA tier.
    pub tier: SlaTier,
    /// Total cost accumulated in this tier (USD).
    pub total_cost_usd: f64,
    /// Number of requests.
    pub request_count: u64,
    /// Average cost per request (USD).
    pub avg_cost_usd: f64,
    /// Fraction of grand total cost.
    pub cost_fraction: f64,
    /// Unique model names seen in this tier.
    pub models: Vec<String>,
    /// Active budget (daily), if configured.
    pub daily_budget_usd: Option<f64>,
    /// Budget utilisation fraction (cost / daily_budget), if budget set.
    pub budget_utilisation: Option<f64>,
}

/// Full SLA tier cost report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SlaReport {
    /// Per-tier summaries, sorted by tier (Premium first).
    pub tiers: Vec<TierSummary>,
    /// Any budget alerts triggered by the current state.
    pub alerts: Vec<SlaAlert>,
    /// Grand total cost (USD).
    pub total_cost_usd: f64,
    /// Total request count.
    pub total_requests: u64,
    /// Premium cost as a fraction of total.
    pub premium_fraction: f64,
}

/// SLA tier cost tracker.
pub struct SlaTierTracker {
    cfg: SlaConfig,
    accum: HashMap<SlaTier, TierAccum>,
}

impl SlaTierTracker {
    /// Create a new tracker.
    pub fn new(cfg: SlaConfig) -> Self {
        Self { cfg, accum: HashMap::new() }
    }

    /// Classify a model name into an [`SlaTier`].
    pub fn classify_model(&self, model: &str) -> SlaTier {
        let model_lower = model.to_lowercase();
        for (prefix, tier) in &self.cfg.model_tiers {
            if model_lower.contains(prefix.to_lowercase().as_str()) {
                return *tier;
            }
        }
        SlaTier::Unknown
    }

    /// Record a request cost for the given model.
    pub fn observe(&mut self, model: &str, cost_usd: f64) {
        let tier = self.classify_model(model);
        let entry = self.accum.entry(tier).or_default();
        entry.total_cost += cost_usd;
        entry.count += 1;
        if !entry.models_seen.contains(&model.to_string()) {
            entry.models_seen.push(model.to_string());
        }
    }

    /// Generate the full SLA tier report and check budgets.
    pub fn report(&self) -> SlaReport {
        let grand_total: f64 = self.accum.values().map(|a| a.total_cost).sum();
        let grand_count: u64 = self.accum.values().map(|a| a.count).sum();

        let tier_order = [SlaTier::Premium, SlaTier::Standard, SlaTier::Economy, SlaTier::Unknown];
        let mut summaries = Vec::new();
        let mut alerts = Vec::new();

        for &tier in &tier_order {
            let acc = self.accum.get(&tier).cloned().unwrap_or_default();
            if acc.count == 0 && grand_count > 0 { continue; }

            let budget = self.cfg.budgets.get(&tier);
            let daily_budget = budget.and_then(|b| b.daily_usd);
            let budget_utilisation = daily_budget.map(|b| if b > 0.0 { acc.total_cost / b } else { 0.0 });

            // Budget alert.
            if let Some(daily) = daily_budget {
                if acc.total_cost > daily {
                    let severity = if tier == SlaTier::Premium { "warning" } else { "info" };
                    alerts.push(SlaAlert {
                        tier,
                        budget_usd: daily,
                        current_cost_usd: acc.total_cost,
                        overage_usd: acc.total_cost - daily,
                        severity: severity.to_string(),
                    });
                }
            }

            summaries.push(TierSummary {
                tier,
                total_cost_usd: acc.total_cost,
                request_count: acc.count,
                avg_cost_usd: if acc.count > 0 { acc.total_cost / acc.count as f64 } else { 0.0 },
                cost_fraction: if grand_total > 0.0 { acc.total_cost / grand_total } else { 0.0 },
                models: acc.models_seen.clone(),
                daily_budget_usd: daily_budget,
                budget_utilisation,
            });
        }

        let premium_cost = self.accum.get(&SlaTier::Premium).map(|a| a.total_cost).unwrap_or(0.0);
        let premium_fraction = if grand_total > 0.0 { premium_cost / grand_total } else { 0.0 };

        SlaReport {
            tiers: summaries,
            alerts,
            total_cost_usd: grand_total,
            total_requests: grand_count,
            premium_fraction,
        }
    }

    /// Set a custom budget for a tier.
    pub fn set_budget(&mut self, tier: SlaTier, daily_usd: Option<f64>, monthly_usd: Option<f64>) {
        self.cfg.budgets.insert(tier, TierBudget { daily_usd, monthly_usd });
    }

    /// Reset all accumulated data.
    pub fn reset(&mut self) {
        self.accum.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tracker() -> SlaTierTracker {
        SlaTierTracker::new(SlaConfig::default())
    }

    #[test]
    fn gpt4o_is_premium() {
        assert_eq!(tracker().classify_model("gpt-4o"), SlaTier::Premium);
    }

    #[test]
    fn gpt4o_mini_is_standard() {
        assert_eq!(tracker().classify_model("gpt-4o-mini"), SlaTier::Standard);
    }

    #[test]
    fn gpt35_is_economy() {
        assert_eq!(tracker().classify_model("gpt-3.5-turbo"), SlaTier::Economy);
    }

    #[test]
    fn unknown_model_is_unknown() {
        assert_eq!(tracker().classify_model("my-local-llm-7b"), SlaTier::Unknown);
    }

    #[test]
    fn claude_opus_is_premium() {
        assert_eq!(tracker().classify_model("claude-opus-4"), SlaTier::Premium);
    }

    #[test]
    fn claude_sonnet_is_standard() {
        assert_eq!(tracker().classify_model("claude-sonnet-4-6"), SlaTier::Standard);
    }

    #[test]
    fn observe_accumulates_per_tier() {
        let mut t = tracker();
        t.observe("gpt-4o", 0.10);
        t.observe("gpt-4o", 0.05);
        let report = t.report();
        let premium = report.tiers.iter().find(|s| s.tier == SlaTier::Premium).unwrap();
        assert!((premium.total_cost_usd - 0.15).abs() < 1e-9);
        assert_eq!(premium.request_count, 2);
    }

    #[test]
    fn cost_fraction_sums_to_one_across_tiers() {
        let mut t = tracker();
        t.observe("gpt-4o", 0.10);
        t.observe("gpt-4o-mini", 0.02);
        t.observe("gpt-3.5-turbo", 0.01);
        let report = t.report();
        let total_frac: f64 = report.tiers.iter().map(|s| s.cost_fraction).sum();
        assert!((total_frac - 1.0).abs() < 1e-9);
    }

    #[test]
    fn alert_fires_when_budget_exceeded() {
        let mut t = tracker();
        // Set a very low premium budget so it triggers immediately.
        t.set_budget(SlaTier::Premium, Some(0.01), None);
        t.observe("gpt-4o", 0.50);
        let report = t.report();
        let premium_alert = report.alerts.iter().find(|a| a.tier == SlaTier::Premium);
        assert!(premium_alert.is_some());
        let alert = premium_alert.unwrap();
        assert_eq!(alert.severity, "warning");
        assert!((alert.overage_usd - 0.49).abs() < 1e-6);
    }

    #[test]
    fn no_alert_within_budget() {
        let mut t = tracker();
        t.set_budget(SlaTier::Premium, Some(100.0), None);
        t.observe("gpt-4o", 0.50);
        let report = t.report();
        assert!(report.alerts.is_empty());
    }

    #[test]
    fn premium_fraction_correct() {
        let mut t = tracker();
        t.observe("gpt-4o", 0.60);
        t.observe("gpt-4o-mini", 0.40);
        let report = t.report();
        assert!((report.premium_fraction - 0.60).abs() < 1e-9);
    }

    #[test]
    fn reset_clears_all_data() {
        let mut t = tracker();
        t.observe("gpt-4o", 1.0);
        t.reset();
        let report = t.report();
        assert_eq!(report.total_requests, 0);
    }

    #[test]
    fn models_seen_tracked() {
        let mut t = tracker();
        t.observe("gpt-4o", 0.01);
        t.observe("gpt-4o", 0.02); // same model, second time
        let report = t.report();
        let premium = report.tiers.iter().find(|s| s.tier == SlaTier::Premium).unwrap();
        // Only one unique model.
        assert_eq!(premium.models.len(), 1);
    }

    #[test]
    fn budget_utilisation_computed() {
        let mut t = tracker();
        t.set_budget(SlaTier::Premium, Some(2.0), None);
        t.observe("gpt-4o", 1.0); // 50% utilisation
        let report = t.report();
        let prem = report.tiers.iter().find(|s| s.tier == SlaTier::Premium).unwrap();
        let util = prem.budget_utilisation.unwrap();
        assert!((util - 0.5).abs() < 1e-9);
    }
}
