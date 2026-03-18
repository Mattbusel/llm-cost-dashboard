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
//! - [`budget`] - hard budget enforcement and soft alert thresholds
//! - [`cost`] - per-request cost records and the append-only ledger
//! - [`cost::pricing`] - static pricing table and cost computation helpers
//! - [`error`] - unified error type
//! - [`log`] - newline-delimited JSON log ingestion
//! - [`trace`] - lightweight distributed tracing
//! - [`ui`] - ratatui TUI application state and event loop
//!
//! ## Related Projects
//!
//! - [Reddit-Options-Trader-ROT](https://github.com/Mattbusel/Reddit-Options-Trader-ROT-)
//! - [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator)
//! - [rot-signals-api](https://github.com/Mattbusel/rot-signals-api)

pub mod budget;
pub mod cost;
pub mod error;
pub mod log;
pub mod trace;
pub mod ui;

pub use budget::BudgetEnvelope;
pub use cost::{CostLedger, CostRecord, ModelStats};
pub use error::DashboardError;
pub use log::{LogEntry, RequestLog};
pub use trace::{SpanStore, TraceSpan};
pub use ui::App;
