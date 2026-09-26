#![deny(missing_docs)]
//! # llm-cost-dashboard
//!
//! Price your LLM API calls and total them up: the library behind the
//! `llm-dash` terminal dashboard.
//!
//! ![llm-dash dashboard](https://raw.githubusercontent.com/Mattbusel/llm-cost-dashboard/master/assets/dashboard.gif)
//!
//! Want the dashboard itself? Install the binary with
//! `cargo install llm-cost-dashboard` (or see the
//! [README](https://github.com/Mattbusel/llm-cost-dashboard#install) for
//! Homebrew, Scoop and one-line installers) and run `llm-dash --demo`.
//!
//! ## Quick example
//!
//! ```
//! use llm_cost_dashboard::{CostLedger, CostRecord};
//!
//! let mut ledger = CostLedger::new();
//! // model, provider, input tokens, output tokens, latency in ms
//! ledger.add(CostRecord::new("gpt-4o-mini", "openai", 512, 256, 34))?;
//! ledger.add(CostRecord::new("claude-sonnet-4-6", "anthropic", 1200, 400, 900))?;
//!
//! assert_eq!(ledger.len(), 2);
//! println!("spent ${:.6}", ledger.total_usd());
//! for (model, stats) in ledger.by_model() {
//!     println!("{model}: ${:.6} over {} requests", stats.total_cost_usd, stats.request_count);
//! }
//! # Ok::<(), llm_cost_dashboard::DashboardError>(())
//! ```
//!
//! ## Where to start
//!
//! - [`CostRecord`] and [`CostLedger`]: one priced request, and the running ledger.
//! - [`cost::pricing::compute_cost`]: the price of a call from the built-in table of 83 models.
//! - [`ProviderComparison`]: rank every priced model by monthly cost for a workload.
//! - [`SpendForecaster`] and [`CostForecaster`]: project spend from what you have so far.
//! - [`AnomalyDetector`]: flag cost spikes with a rolling Z-score.
//! - [`OrgTree`]: org, team and project budgets with alerts.
//! - [`ui::App`]: the ratatui dashboard state, if you want to embed the TUI.
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
pub mod tail;
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
pub mod experiment_tracker;
pub mod cost_governance;
pub mod invoice_generator;
pub mod model_registry;
pub mod anomaly_detector;
pub mod budget_planner;
pub mod usage_reporter;
pub mod cost_allocator_v2;
pub mod pricing_engine;
pub mod audit_logger;
pub mod cost_predictor;
pub mod resource_quota;
pub mod token_optimizer;
pub mod benchmark_runner;
pub mod alert_manager;
pub mod dashboard_metrics;
pub mod trend_analyzer;
pub mod cost_attribution;
pub mod capacity_planner;
pub mod sla_monitor;
pub mod cohort_analyzer;
pub mod savings_calculator;
pub mod multi_tenant;
pub mod cost_forecast;
pub mod integration_webhooks;
pub mod budget_controller;
pub mod usage_report;
