//! # Multi-Tenant Manager
//!
//! Isolation, quota enforcement, and usage tracking for multiple tenants.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::multi_tenant::{MultiTenantManager, TenantConfig, TenantTier};
//!
//! let manager = MultiTenantManager::new();
//! manager.create_tenant(TenantConfig {
//!     tenant_id: "acme".to_string(),
//!     name: "Acme Corp".to_string(),
//!     tier: TenantTier::Professional,
//!     custom_models: vec![],
//!     allowed_capabilities: vec!["TextGeneration".to_string()],
//!     max_users: 50,
//!     data_retention_days: 90,
//! });
//! manager.record_usage("acme", 1_000, 0.05);
//! let status = manager.check_quota("acme");
//! // Should be Ok or Warning depending on usage.
//! ```

use dashmap::DashMap;
use std::fmt;

// ── TenantTier ────────────────────────────────────────────────────────────────

/// Tier of a tenant subscription, determining quota and rate limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantTier {
    /// Free tier: lowest quota, no SLA.
    Free,
    /// Starter tier: small team usage.
    Starter,
    /// Professional tier: mid-size teams.
    Professional,
    /// Enterprise tier: highest quota and dedicated support.
    Enterprise,
}

impl TenantTier {
    /// Monthly token quota for this tier.
    pub fn monthly_token_quota(&self) -> usize {
        match self {
            Self::Free => 100_000,
            Self::Starter => 1_000_000,
            Self::Professional => 10_000_000,
            Self::Enterprise => 500_000_000,
        }
    }

    /// Rate limit in requests per second.
    pub fn rate_limit_rps(&self) -> f64 {
        match self {
            Self::Free => 1.0,
            Self::Starter => 5.0,
            Self::Professional => 20.0,
            Self::Enterprise => 200.0,
        }
    }
}

impl fmt::Display for TenantTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Free => "Free",
            Self::Starter => "Starter",
            Self::Professional => "Professional",
            Self::Enterprise => "Enterprise",
        };
        write!(f, "{}", s)
    }
}

// ── TenantConfig ──────────────────────────────────────────────────────────────

/// Static configuration for a tenant.
#[derive(Debug, Clone)]
pub struct TenantConfig {
    /// Unique tenant identifier.
    pub tenant_id: String,
    /// Display name.
    pub name: String,
    /// Subscription tier.
    pub tier: TenantTier,
    /// Tenant-specific model IDs they are allowed to use.
    pub custom_models: Vec<String>,
    /// Allowed capability strings (e.g. `["TextGeneration", "CodeGeneration"]`).
    pub allowed_capabilities: Vec<String>,
    /// Maximum number of users permitted.
    pub max_users: u32,
    /// How many days of usage data to retain.
    pub data_retention_days: u32,
}

// ── TenantUsage ───────────────────────────────────────────────────────────────

/// Mutable usage counters for a tenant.
#[derive(Debug, Clone, Default)]
pub struct TenantUsage {
    /// Tenant this usage belongs to.
    pub tenant_id: String,
    /// Tokens consumed today.
    pub tokens_used_today: u64,
    /// Tokens consumed this calendar month.
    pub tokens_used_month: u64,
    /// API requests made today.
    pub requests_today: u64,
    /// API requests made this calendar month.
    pub requests_month: u64,
    /// Estimated cost today in USD.
    pub cost_today_usd: f64,
    /// Estimated cost this month in USD.
    pub cost_month_usd: f64,
    /// Unix timestamp of the most recent API call.
    pub last_active_unix: u64,
}

// ── TenantQuotaStatus ─────────────────────────────────────────────────────────

/// Current quota state for a tenant.
#[derive(Debug, Clone, PartialEq)]
pub enum TenantQuotaStatus {
    /// Well within quota.
    Ok,
    /// Approaching quota; percentage used is provided.
    Warning {
        /// Fraction of quota consumed (0.0–1.0).
        pct_used: f64,
    },
    /// Quota fully consumed; further requests should be rejected.
    Exceeded,
    /// Tenant is administratively suspended.
    Suspended,
}

// ── TenantReport ──────────────────────────────────────────────────────────────

/// Aggregated report for a single tenant.
#[derive(Debug, Clone)]
pub struct TenantReport {
    /// Static configuration snapshot.
    pub config: TenantConfig,
    /// Current usage counters.
    pub usage: TenantUsage,
    /// Current quota status.
    pub quota_status: TenantQuotaStatus,
    /// Fraction of monthly quota consumed (0.0–1.0+).
    pub quota_pct_used: f64,
    /// Simple linear projection of monthly cost based on today's spend.
    pub projected_monthly_cost: f64,
}

// ── MultiTenantManager ────────────────────────────────────────────────────────

/// Manages isolation and accounting for multiple tenants.
pub struct MultiTenantManager {
    configs: DashMap<String, TenantConfig>,
    usages: DashMap<String, TenantUsage>,
    suspended: DashMap<String, bool>,
}

impl Default for MultiTenantManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiTenantManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self {
            configs: DashMap::new(),
            usages: DashMap::new(),
            suspended: DashMap::new(),
        }
    }

    /// Register a new tenant.
    pub fn create_tenant(&self, config: TenantConfig) {
        let usage = TenantUsage {
            tenant_id: config.tenant_id.clone(),
            ..TenantUsage::default()
        };
        self.usages.insert(config.tenant_id.clone(), usage);
        self.configs.insert(config.tenant_id.clone(), config);
    }

    /// Mark a tenant as suspended; quota checks will return [`TenantQuotaStatus::Suspended`].
    pub fn suspend_tenant(&self, tenant_id: &str) {
        self.suspended.insert(tenant_id.to_string(), true);
    }

    /// Permanently remove a tenant and all their usage data.
    pub fn delete_tenant(&self, tenant_id: &str) {
        self.configs.remove(tenant_id);
        self.usages.remove(tenant_id);
        self.suspended.remove(tenant_id);
    }

    /// Record a usage event for `tenant_id`.
    ///
    /// `tokens` is the number of tokens consumed; `cost_usd` is the dollar cost.
    pub fn record_usage(&self, tenant_id: &str, tokens: u64, cost_usd: f64) {
        if let Some(mut usage) = self.usages.get_mut(tenant_id) {
            usage.tokens_used_today = usage.tokens_used_today.saturating_add(tokens);
            usage.tokens_used_month = usage.tokens_used_month.saturating_add(tokens);
            usage.requests_today += 1;
            usage.requests_month += 1;
            usage.cost_today_usd += cost_usd;
            usage.cost_month_usd += cost_usd;
            // Use a simple monotonic counter as a proxy for "now".
            usage.last_active_unix += 1;
        }
    }

    /// Check the current quota status for a tenant.
    pub fn check_quota(&self, tenant_id: &str) -> TenantQuotaStatus {
        if self.suspended.get(tenant_id).map(|v| *v).unwrap_or(false) {
            return TenantQuotaStatus::Suspended;
        }
        let config = match self.configs.get(tenant_id) {
            Some(c) => c,
            None => return TenantQuotaStatus::Suspended,
        };
        let usage = match self.usages.get(tenant_id) {
            Some(u) => u,
            None => return TenantQuotaStatus::Ok,
        };
        let quota = config.tier.monthly_token_quota() as u64;
        if quota == 0 {
            return TenantQuotaStatus::Ok;
        }
        let pct = usage.tokens_used_month as f64 / quota as f64;
        if pct >= 1.0 {
            TenantQuotaStatus::Exceeded
        } else if pct >= 0.80 {
            TenantQuotaStatus::Warning { pct_used: pct }
        } else {
            TenantQuotaStatus::Ok
        }
    }

    /// Reset all tenants' daily usage counters.
    pub fn reset_daily_usage(&self) {
        for mut usage in self.usages.iter_mut() {
            usage.tokens_used_today = 0;
            usage.requests_today = 0;
            usage.cost_today_usd = 0.0;
        }
    }

    /// Reset all tenants' monthly usage counters.
    pub fn reset_monthly_usage(&self) {
        for mut usage in self.usages.iter_mut() {
            usage.tokens_used_month = 0;
            usage.requests_month = 0;
            usage.cost_month_usd = 0.0;
        }
    }

    /// Build a [`TenantReport`] for a single tenant.
    pub fn tenant_report(&self, tenant_id: &str) -> Option<TenantReport> {
        let config = self.configs.get(tenant_id)?.clone();
        let usage = self.usages.get(tenant_id)?.clone();
        let quota_status = self.check_quota(tenant_id);
        let quota = config.tier.monthly_token_quota() as f64;
        let quota_pct_used = if quota > 0.0 {
            usage.tokens_used_month as f64 / quota
        } else {
            0.0
        };
        // Simple projection: assume 30-day month, linear from today's spend.
        let projected_monthly_cost = usage.cost_today_usd * 30.0;
        Some(TenantReport {
            config,
            usage,
            quota_status,
            quota_pct_used,
            projected_monthly_cost,
        })
    }

    /// Build reports for all registered tenants.
    pub fn all_tenant_summary(&self) -> Vec<TenantReport> {
        self.configs
            .iter()
            .filter_map(|r| self.tenant_report(r.key()))
            .collect()
    }

    /// Produce a plain-text cross-tenant cost report ranked by month-to-date spend.
    pub fn cross_tenant_cost_report(&self) -> String {
        let mut entries: Vec<(String, f64)> = self
            .usages
            .iter()
            .map(|r| (r.key().clone(), r.cost_month_usd))
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = String::from("Cross-Tenant Cost Report (Month-to-Date)\n");
        out.push_str("==========================================\n");
        out.push_str(&format!("{:<30} {:>14}\n", "Tenant ID", "Cost (USD)"));
        out.push_str("------------------------------------------\n");
        for (id, cost) in &entries {
            out.push_str(&format!("{:<30} {:>14.6}\n", id, cost));
        }
        let total: f64 = entries.iter().map(|(_, c)| c).sum();
        out.push_str("==========================================\n");
        out.push_str(&format!("{:<30} {:>14.6}\n", "TOTAL", total));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(id: &str, tier: TenantTier) -> TenantConfig {
        TenantConfig {
            tenant_id: id.to_string(),
            name: format!("{} Corp", id),
            tier,
            custom_models: vec![],
            allowed_capabilities: vec!["TextGeneration".to_string()],
            max_users: 10,
            data_retention_days: 30,
        }
    }

    #[test]
    fn create_and_check_quota_ok() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("alpha", TenantTier::Professional));
        assert_eq!(m.check_quota("alpha"), TenantQuotaStatus::Ok);
    }

    #[test]
    fn quota_exceeded_after_heavy_usage() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("beta", TenantTier::Free));
        // Free tier = 100_000 tokens/month
        m.record_usage("beta", 200_000, 1.0);
        assert_eq!(m.check_quota("beta"), TenantQuotaStatus::Exceeded);
    }

    #[test]
    fn quota_warning_at_80_pct() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("gamma", TenantTier::Free));
        // 80 001 / 100 000 >= 0.80
        m.record_usage("gamma", 85_000, 0.5);
        let status = m.check_quota("gamma");
        assert!(matches!(status, TenantQuotaStatus::Warning { .. }));
    }

    #[test]
    fn suspend_tenant() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("delta", TenantTier::Starter));
        m.suspend_tenant("delta");
        assert_eq!(m.check_quota("delta"), TenantQuotaStatus::Suspended);
    }

    #[test]
    fn delete_tenant() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("epsilon", TenantTier::Starter));
        m.delete_tenant("epsilon");
        assert!(m.tenant_report("epsilon").is_none());
    }

    #[test]
    fn reset_daily_usage() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("zeta", TenantTier::Starter));
        m.record_usage("zeta", 5_000, 0.10);
        m.reset_daily_usage();
        let report = m.tenant_report("zeta").unwrap();
        assert_eq!(report.usage.tokens_used_today, 0);
        assert_eq!(report.usage.cost_today_usd, 0.0);
    }

    #[test]
    fn reset_monthly_usage() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("eta", TenantTier::Starter));
        m.record_usage("eta", 5_000, 0.10);
        m.reset_monthly_usage();
        let report = m.tenant_report("eta").unwrap();
        assert_eq!(report.usage.tokens_used_month, 0);
        assert_eq!(report.usage.cost_month_usd, 0.0);
    }

    #[test]
    fn all_tenant_summary_count() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("t1", TenantTier::Free));
        m.create_tenant(make_config("t2", TenantTier::Starter));
        assert_eq!(m.all_tenant_summary().len(), 2);
    }

    #[test]
    fn cross_tenant_report_contains_total() {
        let m = MultiTenantManager::new();
        m.create_tenant(make_config("c1", TenantTier::Starter));
        m.create_tenant(make_config("c2", TenantTier::Professional));
        m.record_usage("c1", 1_000, 0.50);
        m.record_usage("c2", 2_000, 1.00);
        let report = m.cross_tenant_cost_report();
        assert!(report.contains("TOTAL"));
    }

    #[test]
    fn tier_monthly_quotas() {
        assert_eq!(TenantTier::Free.monthly_token_quota(), 100_000);
        assert_eq!(TenantTier::Enterprise.monthly_token_quota(), 500_000_000);
    }

    #[test]
    fn tier_rate_limits() {
        assert!(TenantTier::Enterprise.rate_limit_rps() > TenantTier::Free.rate_limit_rps());
    }
}
