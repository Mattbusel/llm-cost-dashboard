//! # llm-dash
//!
//! Binary entry point for the LLM cost dashboard.
//!
//! Run `llm-dash --help` for usage.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::Parser;
use llm_cost_dashboard::export::{export_csv, export_json};
use llm_cost_dashboard::ui::{self, App};
use llm_cost_dashboard::webhook::{WebhookConfig, WebhookFormat};
use tracing::{error, info, warn};

#[derive(Parser)]
#[command(
    name = "llm-dash",
    about = "Real-time LLM token spend dashboard",
    version,
    author
)]
struct Cli {
    /// Monthly budget limit in USD
    #[arg(long, default_value = "10.0")]
    budget: f64,

    /// JSON log file to tail for live data (newline-delimited JSON)
    #[arg(long)]
    log_file: Option<PathBuf>,

    /// Start with demo data pre-loaded
    #[arg(long)]
    demo: bool,

    /// Start an HTTP API server on the given port alongside the TUI.
    ///
    /// Exposes:
    ///   GET /api/summary      – JSON cost summary
    ///   GET /api/export.json  – full ledger as JSON download
    ///   GET /api/export.csv   – full ledger as CSV download
    #[arg(long, value_name = "PORT")]
    serve: Option<u16>,

    /// Slack or generic webhook URL to POST budget alerts to.
    ///
    /// May be specified multiple times for multiple destinations.
    #[arg(long, value_name = "URL")]
    webhook_url: Vec<String>,

    /// USD threshold at which webhook alerts fire (default: 80% of --budget).
    #[arg(long, value_name = "USD")]
    webhook_threshold: Option<f64>,

    /// Webhook payload format: "slack" or "generic" (default: generic).
    #[arg(long, default_value = "generic", value_name = "FORMAT")]
    webhook_format: String,

    /// Export all records to a CSV file at PATH then exit (no TUI).
    ///
    /// If `--log-file` or `--demo` is also given, data is loaded first.
    ///
    /// Example: `llm-dash --log-file requests.ndjson --export-csv costs.csv`
    #[arg(long, value_name = "PATH")]
    export_csv: Option<PathBuf>,

    /// Export all records to a JSON file at PATH then exit (no TUI).
    ///
    /// If `--log-file` or `--demo` is also given, data is loaded first.
    ///
    /// Example: `llm-dash --log-file requests.ndjson --export-json costs.json`
    #[arg(long, value_name = "PATH")]
    export_json: Option<PathBuf>,

    /// Tag all ingested log entries with a session name.
    ///
    /// When set, every [`CostRecord`][llm_cost_dashboard::CostRecord] created
    /// during this run will have its `session_id` field set to SESSION_NAME.
    /// This enables session-level cost aggregation and per-session budget
    /// tracking in the session panel.
    ///
    /// Example: `llm-dash --session experiment-v2 --log-file requests.ndjson`
    #[arg(long, value_name = "SESSION_NAME")]
    session: Option<String>,
}

fn main() {
    // Initialise tracing subscriber; RUST_LOG controls verbosity (default: info).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    info!(
        budget_usd = cli.budget,
        demo = cli.demo,
        serve = ?cli.serve,
        session = ?cli.session,
        "llm-dash starting"
    );

    let mut app = App::new(cli.budget);

    // Attach the session name to the App so all ingested records are tagged.
    if let Some(ref session_name) = cli.session {
        info!(session = %session_name, "session tracking active");
        app.set_session(session_name.clone());
    }

    // Register webhook configurations.
    let webhook_threshold = cli
        .webhook_threshold
        .unwrap_or(cli.budget * 0.80);
    let webhook_format = if cli.webhook_format.eq_ignore_ascii_case("slack") {
        WebhookFormat::Slack
    } else {
        WebhookFormat::Generic
    };
    for url in cli.webhook_url {
        info!(url = %url, threshold_usd = webhook_threshold, "registering webhook");
        app.add_webhook(WebhookConfig {
            url,
            format: webhook_format.clone(),
            threshold_usd: webhook_threshold,
        });
    }

    if cli.demo {
        info!("loading demo data");
        app.load_demo_data();
    }

    if let Some(path) = &cli.log_file {
        info!(path = %path.display(), "loading log file");
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let mut ok = 0usize;
                let mut bad = 0usize;
                for line in content.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    match app.ingest_line(line) {
                        Ok(()) => ok += 1,
                        Err(e) => {
                            warn!(error = %e, "skipping malformed log line");
                            bad += 1;
                        }
                    }
                }
                info!(ok, bad, "log file ingestion complete");
            }
            Err(e) => {
                error!(path = %path.display(), error = %e, "failed to read log file");
                eprintln!("Error reading log file {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    }

    // Handle --export-csv: write CSV to the given path and exit (no TUI).
    if let Some(ref path) = cli.export_csv {
        info!(path = %path.display(), "exporting CSV and exiting");
        match export_csv(&app.ledger, path) {
            Ok(()) => {
                println!("Exported {} records to {}", app.ledger.len(), path.display());
                return;
            }
            Err(e) => {
                error!(error = %e, "CSV export failed");
                eprintln!("Export error: {e}");
                std::process::exit(1);
            }
        }
    }

    // Handle --export-json: write JSON to the given path and exit (no TUI).
    if let Some(ref path) = cli.export_json {
        info!(path = %path.display(), "exporting JSON and exiting");
        match export_json(&app.ledger, path) {
            Ok(()) => {
                println!("Exported {} records to {}", app.ledger.len(), path.display());
                return;
            }
            Err(e) => {
                error!(error = %e, "JSON export failed");
                eprintln!("Export error: {e}");
                std::process::exit(1);
            }
        }
    }

    // If --serve was requested, wrap the ledger in Arc<Mutex<>> and spawn the
    // HTTP server as a background Tokio task, then run the TUI on the main thread.
    if let Some(port) = cli.serve {
        info!(port, "starting HTTP API server alongside TUI");
        // Share the ledger between the HTTP server and the TUI.
        // We clone the existing ledger data into a shared structure.
        let shared_ledger = {
            // Temporarily take the ledger out of `app` by rebuilding it from
            // existing records. We can't move out of `app` while it owns the
            // ledger, so we swap in a fresh one and rebuild the shared copy.
            let mut fresh_ledger = llm_cost_dashboard::cost::CostLedger::new();
            for r in app.ledger.records() {
                let _ = fresh_ledger.add(r.clone());
            }
            Arc::new(Mutex::new(fresh_ledger))
        };

        let ledger_for_api = Arc::clone(&shared_ledger);
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.spawn(async move {
            if let Err(e) = llm_cost_dashboard::api::serve(ledger_for_api, port).await {
                error!(error = %e, "HTTP API server error");
            }
        });

        info!("starting TUI event loop (--serve mode)");
        if let Err(e) = ui::run(app) {
            error!(error = %e, "dashboard terminated with error");
            eprintln!("Dashboard error: {e}");
            std::process::exit(1);
        }
    } else {
        info!("starting TUI event loop");
        if let Err(e) = ui::run(app) {
            error!(error = %e, "dashboard terminated with error");
            eprintln!("Dashboard error: {e}");
            std::process::exit(1);
        }
    }

    info!("llm-dash exited cleanly");
}
