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
//! - [`cost`] - per-request cost records and the append-only ledger
//! - [`cost::pricing`] - static pricing table and cost computation helpers
//! - [`error`] - unified error type
//! - [`export`] - CSV and JSON cost data export (file-based and in-memory)
//! - [`forecast`] - OLS linear regression forecaster with trend analysis and seasonal adjustment
//! - [`log`] - newline-delimited JSON log ingestion with header-based provider detection
//! - [`recommendations`] - model recommendation engine with projected monthly savings
//! - [`scheduler`] - cron-based automated export scheduling
//! - [`session`] - per-session budget and cost tracking
//! - [`tagging`] - FinOps cost attribution via structured tag rules and tag-aggregated ledger
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
pub mod allocation;
pub mod anomaly;
pub mod api;
pub mod budget;
pub mod cost;
pub mod error;
pub mod export;
pub mod forecast;
pub mod log;
pub mod recommendations;
pub mod scheduler;
pub mod session;
pub mod tagging;
pub mod trace;
pub mod ui;
pub mod validator;
pub mod webhook;

pub use budget::{
    BudgetAlert, BudgetEnvelope, OrgSummary, OrgTree, ProjectConfig, ProjectSummary, TeamConfig,
    TeamSummary,
};
pub use cost::{CacheBreakdown, CostLedger, CostRecord, ModelStats};
pub use error::DashboardError;
pub use export::{CostExporter, ExportFormat};
pub use forecast::{ForecastResult, SpendForecaster, Trend};
pub use log::{LogEntry, RequestLog};
pub use trace::{SpanStore, TraceSpan};
pub use ui::App;
pub use validator::{
    AnthropicValidator, GoogleValidator, MultiValidator, OpenAiValidator, ValidationResult,
};
pub use webhook::{WebhookConfig, WebhookFormat};
