//! # Multi-Tenant Cost Isolator
//!
//! Per-tenant cost tracking, quota enforcement, and reporting.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::tenant::{Tenant, TenantIsolator};
//!
//! let mut isolator = TenantIsolator::new();
//!
//! isolator.add_tenant(Tenant {
//!     id: "acme".to_string(),
//!     name: "ACME Corp".to_string(),
//!     quota_usd: Some(500.0),
//!     tags: vec!["enterprise".to_string()],
//! });
//!
//! isolator.record("acme", 12.50, "gpt-4o");
//! isolator.record("acme", 7.00, "claude-3-5-sonnet");
//!
//! let reports = isolator.report_all();
//! assert_eq!(reports[0].total_cost_usd, 19.50);
//! ```

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Tenant ────────────────────────────────────────────────────────────────────

/// A registered tenant (organisation, team, or project).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    /// Unique tenant identifier (slug, e.g. `"acme"` or `"team-platform"`).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Optional hard spending quota in US dollars.  `None` means unlimited.
    pub quota_usd: Option<f64>,
    /// Arbitrary string tags for grouping (e.g. `"enterprise"`, `"internal"`).
    pub tags: Vec<String>,
}

// ── Ledger internals ──────────────────────────────────────────────────────────

/// A single cost event recorded for a tenant.
#[derive(Debug, Clone)]
struct CostEvent {
    cost_usd: f64,
    model_id: String,
    date: NaiveDate,
}

// ── Ledger ────────────────────────────────────────────────────────────────────

/// Per-tenant cost ledger.
///
/// Tracks individual cost events per tenant, providing totals, quota
/// remaining, top-model analysis, and daily breakdowns.
#[derive(Debug, Default)]
pub struct TenantLedger {
    /// Map from tenant_id → list of events.
    events: HashMap<String, Vec<CostEvent>>,
}

impl TenantLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cost event for `tenant_id`.
    ///
    /// `cost_usd` is the amount spent; `model_id` identifies the LLM backend.
    /// The current UTC date is recorded alongside the event.
    pub fn record(&mut self, tenant_id: &str, cost_usd: f64, model_id: &str) {
        let date = chrono::Utc::now().date_naive();
        self.events
            .entry(tenant_id.to_string())
            .or_default()
            .push(CostEvent { cost_usd, model_id: model_id.to_string(), date });
    }

    /// Return the total cost recorded for `tenant_id` (0.0 if unknown).
    pub fn total(&self, tenant_id: &str) -> f64 {
        self.events
            .get(tenant_id)
            .map(|evts| evts.iter().map(|e| e.cost_usd).sum())
            .unwrap_or(0.0)
    }

    /// Return the remaining quota for `tenant_id`, or `None` if no quota is set.
    ///
    /// A negative value means the tenant has exceeded their quota.
    pub fn quota_remaining(&self, tenant_id: &str, quota_usd: Option<f64>) -> Option<f64> {
        quota_usd.map(|q| q - self.total(tenant_id))
    }

    /// Return the top-N models by spend for `tenant_id`, sorted descending.
    pub fn top_models(&self, tenant_id: &str, n: usize) -> Vec<(String, f64)> {
        let events = match self.events.get(tenant_id) {
            Some(e) => e,
            None => return vec![],
        };
        let mut by_model: HashMap<String, f64> = HashMap::new();
        for ev in events {
            *by_model.entry(ev.model_id.clone()).or_insert(0.0) += ev.cost_usd;
        }
        let mut pairs: Vec<(String, f64)> = by_model.into_iter().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(n);
        pairs
    }

    /// Return a daily breakdown of spend for `tenant_id`, sorted ascending by date.
    pub fn daily_breakdown(&self, tenant_id: &str) -> Vec<(NaiveDate, f64)> {
        let events = match self.events.get(tenant_id) {
            Some(e) => e,
            None => return vec![],
        };
        let mut by_day: HashMap<NaiveDate, f64> = HashMap::new();
        for ev in events {
            *by_day.entry(ev.date).or_insert(0.0) += ev.cost_usd;
        }
        let mut pairs: Vec<(NaiveDate, f64)> = by_day.into_iter().collect();
        pairs.sort_by_key(|(d, _)| *d);
        pairs
    }
}

// ── Report ────────────────────────────────────────────────────────────────────

/// Summarised spend report for a single tenant.
#[derive(Debug, Clone, Serialize)]
pub struct TenantReport {
    /// The tenant this report covers.
    pub tenant: Tenant,
    /// Total spend recorded in the ledger for this tenant.
    pub total_cost_usd: f64,
    /// Configured quota (mirrors `tenant.quota_usd`).
    pub quota_usd: Option<f64>,
    /// Fraction of quota consumed (0.0–1.0+); `None` if no quota.
    pub quota_pct_used: Option<f64>,
    /// Top-5 models by spend, sorted descending.
    pub top_models: Vec<(String, f64)>,
    /// Daily spend breakdown, sorted ascending by date.
    pub daily_breakdown: Vec<(NaiveDate, f64)>,
}

// ── Isolator ──────────────────────────────────────────────────────────────────

/// Manages multiple tenants and their associated cost ledgers.
pub struct TenantIsolator {
    tenants: HashMap<String, Tenant>,
    ledger: TenantLedger,
}

impl TenantIsolator {
    /// Create an empty isolator.
    pub fn new() -> Self {
        Self {
            tenants: HashMap::new(),
            ledger: TenantLedger::new(),
        }
    }

    /// Register a tenant.  If a tenant with the same `id` already exists it is
    /// replaced.
    pub fn add_tenant(&mut self, tenant: Tenant) {
        self.tenants.insert(tenant.id.clone(), tenant);
    }

    /// Record a cost event for `tenant_id`.
    ///
    /// If the tenant is not registered the event is still stored in the ledger
    /// (unregistered tenants can be registered later).
    pub fn record(&mut self, tenant_id: &str, cost_usd: f64, model_id: &str) {
        self.ledger.record(tenant_id, cost_usd, model_id);
    }

    /// Return the total cost recorded for `tenant_id`.
    pub fn total(&self, tenant_id: &str) -> f64 {
        self.ledger.total(tenant_id)
    }

    /// Return the quota remaining for `tenant_id`, or `None` if no quota.
    pub fn quota_remaining(&self, tenant_id: &str) -> Option<f64> {
        let quota = self.tenants.get(tenant_id).and_then(|t| t.quota_usd);
        self.ledger.quota_remaining(tenant_id, quota)
    }

    /// Generate a [`TenantReport`] for every registered tenant.
    pub fn report_all(&self) -> Vec<TenantReport> {
        let mut reports: Vec<TenantReport> = self
            .tenants
            .values()
            .map(|tenant| {
                let total = self.ledger.total(&tenant.id);
                let quota_pct_used = tenant.quota_usd.map(|q| {
                    if q <= 0.0 { f64::INFINITY } else { total / q }
                });
                TenantReport {
                    top_models: self.ledger.top_models(&tenant.id, 5),
                    daily_breakdown: self.ledger.daily_breakdown(&tenant.id),
                    total_cost_usd: total,
                    quota_usd: tenant.quota_usd,
                    quota_pct_used,
                    tenant: tenant.clone(),
                }
            })
            .collect();
        reports.sort_by(|a, b| a.tenant.id.cmp(&b.tenant.id));
        reports
    }

    /// Return all registered tenants whose total spend exceeds their quota.
    pub fn over_quota(&self) -> Vec<&Tenant> {
        self.tenants
            .values()
            .filter(|t| {
                if let Some(q) = t.quota_usd {
                    self.ledger.total(&t.id) > q
                } else {
                    false
                }
            })
            .collect()
    }

    /// Return a reference to the tenant with `id`, or `None`.
    pub fn get_tenant(&self, id: &str) -> Option<&Tenant> {
        self.tenants.get(id)
    }

    /// Return all registered tenant IDs.
    pub fn tenant_ids(&self) -> Vec<&str> {
        self.tenants.keys().map(String::as_str).collect()
    }

    /// Print a summary of all tenant reports to stdout.
    pub fn print_report(&self) {
        let reports = self.report_all();
        println!("=== Tenant Cost Report ===");
        for r in &reports {
            println!(
                "  {} ({}): ${:.4} / {} quota",
                r.tenant.name,
                r.tenant.id,
                r.total_cost_usd,
                r.quota_usd.map_or("none".to_string(), |q| format!("${:.2}", q))
            );
            for (model, cost) in &r.top_models {
                println!("    {} -> ${:.4}", model, cost);
            }
        }
    }
}

impl Default for TenantIsolator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tenant(id: &str, quota: Option<f64>) -> Tenant {
        Tenant {
            id: id.to_string(),
            name: format!("Tenant {}", id),
            quota_usd: quota,
            tags: vec![],
        }
    }

    // ── TenantLedger ──────────────────────────────────────────────────────────

    #[test]
    fn ledger_total_unknown_tenant_is_zero() {
        let ledger = TenantLedger::new();
        assert_eq!(ledger.total("no-one"), 0.0);
    }

    #[test]
    fn ledger_record_and_total() {
        let mut ledger = TenantLedger::new();
        ledger.record("t1", 10.0, "gpt-4o");
        ledger.record("t1", 5.0, "claude-3");
        assert!((ledger.total("t1") - 15.0).abs() < 1e-9);
    }

    #[test]
    fn ledger_quota_remaining_with_quota() {
        let mut ledger = TenantLedger::new();
        ledger.record("t1", 30.0, "m");
        let remaining = ledger.quota_remaining("t1", Some(100.0));
        assert!(matches!(remaining, Some(r) if (r - 70.0).abs() < 1e-9));
    }

    #[test]
    fn ledger_quota_remaining_no_quota_is_none() {
        let ledger = TenantLedger::new();
        assert!(ledger.quota_remaining("t1", None).is_none());
    }

    #[test]
    fn ledger_quota_remaining_negative_when_exceeded() {
        let mut ledger = TenantLedger::new();
        ledger.record("t1", 150.0, "m");
        let remaining = ledger.quota_remaining("t1", Some(100.0));
        assert!(matches!(remaining, Some(r) if r < 0.0));
    }

    #[test]
    fn ledger_top_models_sorted_descending() {
        let mut ledger = TenantLedger::new();
        ledger.record("t1", 5.0, "cheap-model");
        ledger.record("t1", 50.0, "expensive-model");
        ledger.record("t1", 20.0, "mid-model");
        let top = ledger.top_models("t1", 10);
        assert_eq!(top[0].0, "expensive-model");
        assert_eq!(top[1].0, "mid-model");
        assert_eq!(top[2].0, "cheap-model");
    }

    #[test]
    fn ledger_top_models_truncated_to_n() {
        let mut ledger = TenantLedger::new();
        for i in 0..10 {
            ledger.record("t1", i as f64, &format!("model-{}", i));
        }
        let top = ledger.top_models("t1", 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn ledger_top_models_unknown_tenant_empty() {
        let ledger = TenantLedger::new();
        assert!(ledger.top_models("nobody", 5).is_empty());
    }

    #[test]
    fn ledger_daily_breakdown_sorted_ascending() {
        // We can only verify it's non-empty and sorted since the date is today.
        let mut ledger = TenantLedger::new();
        ledger.record("t1", 10.0, "m");
        ledger.record("t1", 20.0, "m");
        let breakdown = ledger.daily_breakdown("t1");
        assert_eq!(breakdown.len(), 1);
        assert!((breakdown[0].1 - 30.0).abs() < 1e-9);
    }

    // ── TenantIsolator ────────────────────────────────────────────────────────

    #[test]
    fn isolator_add_and_get_tenant() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("acme", Some(500.0)));
        let t = iso.get_tenant("acme").unwrap();
        assert_eq!(t.name, "Tenant acme");
    }

    #[test]
    fn isolator_record_and_total() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("acme", None));
        iso.record("acme", 25.0, "gpt-4o");
        iso.record("acme", 10.0, "claude-3");
        assert!((iso.total("acme") - 35.0).abs() < 1e-9);
    }

    #[test]
    fn isolator_quota_remaining_some() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("acme", Some(100.0)));
        iso.record("acme", 30.0, "m");
        let rem = iso.quota_remaining("acme");
        assert!(matches!(rem, Some(r) if (r - 70.0).abs() < 1e-9));
    }

    #[test]
    fn isolator_quota_remaining_none_when_no_quota() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("acme", None));
        assert!(iso.quota_remaining("acme").is_none());
    }

    #[test]
    fn isolator_over_quota_returns_exceeded_tenants() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("rich", Some(1000.0)));
        iso.add_tenant(make_tenant("poor", Some(10.0)));
        iso.record("poor", 50.0, "m");
        let over = iso.over_quota();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].id, "poor");
    }

    #[test]
    fn isolator_over_quota_empty_when_none_exceeded() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("t", Some(1000.0)));
        iso.record("t", 5.0, "m");
        assert!(iso.over_quota().is_empty());
    }

    #[test]
    fn isolator_over_quota_skips_unlimited_tenants() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("unlimited", None));
        iso.record("unlimited", 999_999.0, "m");
        assert!(iso.over_quota().is_empty());
    }

    #[test]
    fn isolator_report_all_contains_all_tenants() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("a", None));
        iso.add_tenant(make_tenant("b", None));
        let reports = iso.report_all();
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn isolator_report_quota_pct_used() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("t", Some(200.0)));
        iso.record("t", 100.0, "m");
        let reports = iso.report_all();
        let pct = reports[0].quota_pct_used.unwrap();
        assert!((pct - 0.5).abs() < 1e-9);
    }

    #[test]
    fn isolator_report_top_models_populated() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("t", None));
        iso.record("t", 10.0, "model-a");
        iso.record("t", 50.0, "model-b");
        let reports = iso.report_all();
        assert!(!reports[0].top_models.is_empty());
        assert_eq!(reports[0].top_models[0].0, "model-b");
    }

    #[test]
    fn isolator_tenant_ids_lists_all() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("x", None));
        iso.add_tenant(make_tenant("y", None));
        let mut ids = iso.tenant_ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn isolator_replace_tenant_on_duplicate_id() {
        let mut iso = TenantIsolator::new();
        iso.add_tenant(make_tenant("t", Some(100.0)));
        iso.add_tenant(Tenant {
            id: "t".to_string(),
            name: "Updated".to_string(),
            quota_usd: Some(200.0),
            tags: vec![],
        });
        assert_eq!(iso.get_tenant("t").unwrap().name, "Updated");
        assert_eq!(iso.get_tenant("t").unwrap().quota_usd, Some(200.0));
    }
}
