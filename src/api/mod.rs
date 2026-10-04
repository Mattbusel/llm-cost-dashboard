//! # HTTP API Server
//!
//! Optional HTTP server started when the `--serve` CLI flag is provided.
//!
//! Endpoints:
//! - `GET /api/summary`      – JSON summary of current costs.
//! - `GET /api/export.json`  – Full ledger as JSON download.
//! - `GET /api/export.csv`   – Full ledger as CSV download.
//! - `GET /metrics`          – Prometheus text format (spend, requests and
//!   tokens per model), for Prometheus, Grafana Agent, VictoriaMetrics and
//!   anything else that scrapes the Prometheus format.
//!
//! [`serve`] listens on 127.0.0.1 only; use [`serve_on`] to choose the
//! address (before 1.3.0 `serve` listened on every interface).
//!
//! The server shares ledger state via an `Arc<Mutex<CostLedger>>` so that the
//! TUI and API can coexist in separate Tokio tasks.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::{Arc, Mutex};
//! use llm_cost_dashboard::cost::CostLedger;
//! use llm_cost_dashboard::api::serve;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let ledger = Arc::new(Mutex::new(CostLedger::new()));
//! serve(ledger, 8080).await.unwrap();
//! # }
//! ```

use std::fmt::Write as _;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::Serialize;
use tracing::info;

use crate::cost::CostLedger;
use crate::error::DashboardError;

/// Shared application state for the HTTP handlers.
#[derive(Clone)]
pub struct ApiState {
    ledger: Arc<Mutex<CostLedger>>,
}

/// JSON body returned by `GET /api/summary`.
#[derive(Serialize)]
pub struct SummaryResponse {
    /// Total spend across all recorded requests in USD.
    pub total_usd: f64,
    /// Projected 30-day spend based on the last hour of activity.
    pub projected_monthly_usd: f64,
    /// Total number of recorded requests.
    pub request_count: usize,
    /// 7-day daily spend trend (oldest → today).
    pub seven_day_trend: [f64; 7],
}

/// Start the Axum HTTP server on `port`, serving the shared `ledger`.
///
/// This function runs forever (until the process exits).  Callers should
/// spawn it as a background Tokio task alongside the TUI event loop:
///
/// ```no_run
/// # use std::sync::{Arc, Mutex};
/// # use llm_cost_dashboard::cost::CostLedger;
/// # use llm_cost_dashboard::api::serve;
/// # #[tokio::main]
/// # async fn main() {
/// let ledger = Arc::new(Mutex::new(CostLedger::new()));
/// let ledger_api = Arc::clone(&ledger);
/// tokio::spawn(async move { serve(ledger_api, 8080).await.unwrap() });
/// # }
/// ```
///
/// # Errors
///
/// Returns [`DashboardError::Terminal`] if the TCP listener cannot be bound.
pub async fn serve(
    ledger: Arc<Mutex<CostLedger>>,
    port: u16,
) -> Result<(), DashboardError> {
    serve_on(ledger, SocketAddr::from(([127, 0, 0, 1], port))).await
}

/// Like [`serve`], on any address (`0.0.0.0:8080` to accept connections
/// from other machines).
///
/// # Errors
///
/// Returns [`DashboardError::Terminal`] if the TCP listener cannot be bound.
pub async fn serve_on(
    ledger: Arc<Mutex<CostLedger>>,
    addr: SocketAddr,
) -> Result<(), DashboardError> {
    info!(addr = %addr, "HTTP API server starting");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| DashboardError::Terminal(format!("bind {addr}: {e}")))?;
    axum::serve(listener, router(ledger))
        .await
        .map_err(|e| DashboardError::Terminal(e.to_string()))
}

/// The axum [`Router`] with every endpoint, for mounting inside your own
/// axum application.
pub fn router(ledger: Arc<Mutex<CostLedger>>) -> Router {
    Router::new()
        .route("/api/summary", get(handle_summary))
        .route("/api/export.json", get(handle_export_json))
        .route("/api/export.csv", get(handle_export_csv))
        .route("/metrics", get(handle_metrics))
        .with_state(ApiState { ledger })
}

/// Keep a shared ledger current while the dashboard follows a log file.
///
/// Every line from `lines` is priced into `ledger` (through an
/// [`crate::ingest::Ingester`]) and passed on unchanged through the returned
/// receiver, so the dashboard and the HTTP API see the same requests.
/// Before 1.3.0 the API only showed what was loaded at start-up.
pub fn mirror_feed(
    lines: std::sync::mpsc::Receiver<String>,
    ledger: Arc<Mutex<CostLedger>>,
) -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut ing = crate::ingest::Ingester::new();
        for line in lines {
            if let Ok(true) = ing.ingest_line(&line) {
                if let Some(rec) = ing.ledger().records().last() {
                    let mut l = ledger.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = l.add(rec.clone());
                }
            }
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    rx
}

/// Escape a Prometheus label value (backslash, double quote, newline).
fn label(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// The ledger in the Prometheus text exposition format (version 0.0.4).
///
/// Metrics: `llm_cost_usd_total` and `llm_requests_total` overall and per
/// model and provider, `llm_tokens_total` per model and direction, and
/// `llm_projected_monthly_cost_usd` (last hour's pace).
pub fn prometheus_text(ledger: &CostLedger) -> String {
    let mut by: std::collections::BTreeMap<(&str, &str), (u64, f64, u64, u64)> =
        std::collections::BTreeMap::new();
    for r in ledger.records() {
        let e = by.entry((r.model.as_str(), r.provider.as_str())).or_default();
        e.0 += 1;
        e.1 += r.total_cost_usd;
        e.2 += r.input_tokens + r.cache.cache_read_tokens + r.cache.cache_write_tokens;
        e.3 += r.output_tokens;
    }
    let mut out = String::new();
    let _ = writeln!(out, "# HELP llm_cost_usd_total Spend in US dollars, priced from the model table.");
    let _ = writeln!(out, "# TYPE llm_cost_usd_total counter");
    for ((m, p), (_, cost, _, _)) in &by {
        let _ = writeln!(out, "llm_cost_usd_total{{model=\"{}\",provider=\"{}\"}} {cost}", label(m), label(p));
    }
    let _ = writeln!(out, "# HELP llm_requests_total Requests in the ledger.");
    let _ = writeln!(out, "# TYPE llm_requests_total counter");
    for ((m, p), (n, _, _, _)) in &by {
        let _ = writeln!(out, "llm_requests_total{{model=\"{}\",provider=\"{}\"}} {n}", label(m), label(p));
    }
    let _ = writeln!(out, "# HELP llm_tokens_total Tokens, input (including prompt-cache tokens) and output.");
    let _ = writeln!(out, "# TYPE llm_tokens_total counter");
    for ((m, p), (_, _, input, output)) in &by {
        let _ = writeln!(out, "llm_tokens_total{{model=\"{}\",provider=\"{}\",direction=\"input\"}} {input}", label(m), label(p));
        let _ = writeln!(out, "llm_tokens_total{{model=\"{}\",provider=\"{}\",direction=\"output\"}} {output}", label(m), label(p));
    }
    let _ = writeln!(out, "# HELP llm_projected_monthly_cost_usd 30-day spend at the last hour's pace.");
    let _ = writeln!(out, "# TYPE llm_projected_monthly_cost_usd gauge");
    let _ = writeln!(out, "llm_projected_monthly_cost_usd {}", ledger.projected_monthly_usd(1));
    out
}

async fn handle_metrics(State(state): State<ApiState>) -> Response {
    let ledger = state.ledger.lock().unwrap_or_else(|e| e.into_inner());
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        prometheus_text(&ledger),
    )
        .into_response()
}

async fn handle_summary(State(state): State<ApiState>) -> impl IntoResponse {
    let ledger = state.ledger.lock().unwrap_or_else(|e| e.into_inner());
    Json(SummaryResponse {
        total_usd: ledger.total_usd(),
        projected_monthly_usd: ledger.projected_monthly_usd(1),
        request_count: ledger.len(),
        seven_day_trend: ledger.seven_day_trend(),
    })
}

async fn handle_export_json(State(state): State<ApiState>) -> Response {
    let ledger = state.ledger.lock().unwrap_or_else(|e| e.into_inner());
    match ledger.to_json() {
        Ok(json) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/json"),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"llm-costs.json\"",
                ),
            ],
            json,
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn handle_export_csv(State(state): State<ApiState>) -> Response {
    let ledger = state.ledger.lock().unwrap_or_else(|e| e.into_inner());
    match ledger.to_csv() {
        Ok(csv) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/csv"),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"llm-costs.csv\"",
                ),
            ],
            csv,
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::CostRecord;

    fn make_state() -> ApiState {
        let mut ledger = CostLedger::new();
        ledger
            .add(CostRecord::new("gpt-4o-mini", "openai", 512, 256, 20))
            .unwrap();
        ApiState {
            ledger: Arc::new(Mutex::new(ledger)),
        }
    }

    #[test]
    fn test_state_ledger_accessible() {
        let state = make_state();
        let l = state.ledger.lock().unwrap();
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn test_summary_response_fields() {
        let state = make_state();
        let l = state.ledger.lock().unwrap();
        let resp = SummaryResponse {
            total_usd: l.total_usd(),
            projected_monthly_usd: l.projected_monthly_usd(1),
            request_count: l.len(),
            seven_day_trend: l.seven_day_trend(),
        };
        assert_eq!(resp.request_count, 1);
        assert!(resp.total_usd > 0.0);
        // Today should have a non-zero value in the trend
        let today_val = resp.seven_day_trend[6];
        assert!(today_val > 0.0);
    }

    #[test]
    fn test_csv_export_contains_header() {
        let state = make_state();
        let l = state.ledger.lock().unwrap();
        let csv = l.to_csv().unwrap();
        assert!(csv.contains("model"));
        assert!(csv.contains("gpt-4o-mini"));
    }

    #[test]
    fn test_prometheus_text_format() {
        let mut ledger = CostLedger::new();
        ledger.add(CostRecord::new("gpt-4o", "openai", 1_000_000, 0, 5)).unwrap();
        ledger.add(CostRecord::new("gpt-4o", "openai", 1_000_000, 500_000, 5)).unwrap();
        ledger.add(CostRecord::new("we\"ird\\model", "x", 10, 1, 5)).unwrap();
        let t = prometheus_text(&ledger);
        assert!(t.contains("llm_cost_usd_total{model=\"gpt-4o\",provider=\"openai\"} 10\n"), "{t}");
        assert!(t.contains("llm_requests_total{model=\"gpt-4o\",provider=\"openai\"} 2\n"));
        assert!(t.contains("direction=\"output\"} 500000\n"));
        assert!(t.contains("model=\"we\\\"ird\\\\model\""), "{t}");
        // Every sample line is `name{labels} number` or `name number`.
        for line in t.lines().filter(|l| !l.starts_with('#')) {
            let v = line.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "bad sample: {line}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_router_serves_metrics_over_http() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let state = make_state();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(state.ledger)).await });
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut body = String::new();
        stream.read_to_string(&mut body).await.unwrap();
        assert!(body.starts_with("HTTP/1.1 200"), "{body}");
        assert!(body.contains("llm_requests_total{model=\"gpt-4o-mini\",provider=\"openai\"} 1"));
    }

    #[test]
    fn test_mirror_feed_updates_the_shared_ledger() {
        let shared = Arc::new(Mutex::new(CostLedger::new()));
        let (tx, rx) = std::sync::mpsc::channel();
        let out = mirror_feed(rx, Arc::clone(&shared));
        tx.send(r#"{"model":"gpt-4o","input_tokens":1000000,"output_tokens":0}"#.to_string()).unwrap();
        tx.send("not json".to_string()).unwrap();
        assert!(out.recv().unwrap().contains("gpt-4o"));
        assert_eq!(out.recv().unwrap(), "not json");
        let l = shared.lock().unwrap();
        assert_eq!(l.len(), 1);
        assert!((l.total_usd() - 2.5).abs() < 1e-9);
        assert!(prometheus_text(&l).contains("llm_requests_total{model=\"gpt-4o\",provider=\"unknown\"} 1"));
    }

    #[test]
    fn test_json_export_contains_data() {
        let state = make_state();
        let l = state.ledger.lock().unwrap();
        let json = l.to_json().unwrap();
        assert!(json.contains("gpt-4o-mini"));
    }
}
