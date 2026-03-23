#![deny(missing_docs)]
//! # llm-cost-dashboard
//!
//! Real-time terminal dashboard for LLM token spend.
//!
//! This crate provides the library components consumed by the `llm-dash`
//! binary.  It can also be used as a library to embed cost tracking in other
//! Rust applications.
//!
//! ## Modules
//!
//! - [`alerting`] - webhook-based alert delivery with cooldown deduplication
//! - [`allocation`] - team/project cost allocation with chargeback/showback workflows
//! - [`anomaly`] - rolling Z-score cost spike detector
//! - [`api`] - optional Axum HTTP API server (`--serve` mode)
//! - [`budget`] - hard budget enforcement, soft alert thresholds, and org→team→project hierarchy ([`budget::hierarchy::OrgTree`])
//! - [`comparison`] - multi-provider side-by-side cost comparison and monthly projections ([`comparison::ProviderComparison`])
//! - [`cost`] - per-request cost records and the append-only ledger
//! - [`cost::pricing`] - static pricing table and cost computation helpers
//! - [`error`] - unified error type
//! - [`export`] - CSV and JSON cost data export (file-based and in-memory)
//! - [`forecast`] - OLS regression and Holt-Winters exponential smoothing cost forecaster
//! - [`log`] - newline-delimited JSON log ingestion with header-based provider detection
//! - [`org`] - multi-tenant organization -> team -> project hierarchy with spend tracking
//! - [`recommendations`] - model recommendation engine with projected monthly savings
//! - [`scheduler`] - cron-based automated export scheduling
//! - [`session`] - per-session budget and cost tracking
//! - [`tagging`] - FinOps cost attribution via structured tag rules and tag-aggregated ledger
//! - [`tags`] - lightweight key=value cost attribution tags with top-N spend queries
//! - [`trace`] - lightweight distributed tracing
//! - [`trends`] - daily time-series aggregation, moving averages, period-over-period comparison, and ASCII sparklines
//! - [`ui`] - ratatui TUI application state and event loop
//! - [`validator`] - API key validation for Anthropic, OpenAI, and Google
//! - [`webhook`] - Slack / generic webhook alerts on budget threshold
//!
//! ## Related Projects
//!
//! - [Reddit-Options-Trader-ROT](https://github.com/Mattbusel/Reddit-Options-Trader-ROT-)
//! - [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator)
//! - [rot-signals-api](https://github.com/Mattbusel/rot-signals-api)

pub mod alerting;
pub mod alerts;
pub mod allocation;
pub mod anomaly;
pub mod api;
pub mod budget;
pub mod comparison;
pub mod cost;
pub mod error;
pub mod export;
pub mod forecast;
pub mod log;
pub mod org;
pub mod recommendations;
pub mod scheduler;
pub mod session;
pub mod tagging;
pub mod tags;
pub mod trace;
pub mod trends;
pub mod ui;
pub mod validator;
pub mod webhook;
pub mod clustering;
pub mod sla;
pub mod carbon;
pub mod tenant;
pub mod prediction;
pub mod diff;
pub mod model_compare;
pub mod trend;
pub mod quota;
pub mod replay;

pub use budget::{
    BudgetAlert, BudgetEnvelope, OrgSummary, OrgTree, ProjectConfig, ProjectSummary, TeamConfig,
    TeamSummary,
};
pub use comparison::{CostProjection, ProviderComparison, WorkloadProfile};
pub use cost::{CacheBreakdown, CostLedger, CostRecord, ModelStats};
pub use error::DashboardError;
pub use export::{CostExporter, ExportFormat};
pub use forecast::{CostForecaster, ForecastResult, HoltWintersForecast, SpendForecaster, Trend,
    EsCostForecaster, ForecastMethod, SimpleEsModel, DoubleEsModel, HoltWintersModel};
pub use log::{LogEntry, RequestLog};
pub use org::{Organization, Project, Team};
pub use tags::{TagIndex, TaggedRecord, Tags};
pub use trace::{SpanStore, TraceSpan};
pub use ui::App;
pub use validator::{
    AnthropicValidator, GoogleValidator, MultiValidator, OpenAiValidator, ValidationResult,
};
pub use webhook::{WebhookConfig, WebhookFormat};
pub use allocation::{
    AllocationBucket, AllocationRule, AllocationTag, AllocationLedger, AllocationReport,
    BudgetHierarchy, CostAllocation, CostAllocator, Environment, ProjectBudget, TeamBudget,
    TeamUsage, teams_tab_rows,
};
pub use trends::{DailySpend, TrendAnalyzer, TrendReport};
pub use alerts::{Alert, AlertChannel, AlertEngine, AlertRule, AlertSummary, AlertWindow};
pub use anomaly::{AnomalyConfig, AnomalyDetector, AnomalyReport, AnomalyResult};
pub use budget::planner::{
    AllocationStatus, BudgetAllocation, BudgetForecast, BudgetPeriod, BudgetPlan, BudgetPlanner,
};
pub use tenant::{Tenant, TenantIsolator, TenantLedger, TenantReport};
pub mod webhook_dispatch;
pub mod aggregator;
pub mod billing;
pub mod capacity;
pub mod visualization;
pub mod alert_rules;
pub mod chargeback;
pub mod session_cost;
pub mod notification;
pub mod benchmark;
pub mod cost_optimizer;
pub mod usage_limiter;
pub mod cost_allocation;
pub mod price_tracker;
pub mod efficiency_analyzer;
pub mod model_lifecycle;
pub mod roi_calculator;
