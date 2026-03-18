//! # llm-dash
//!
//! Binary entry point for the LLM cost dashboard.
//!
//! Run `llm-dash --help` for usage.

use std::path::PathBuf;

use clap::Parser;
use llm_cost_dashboard::ui::{self, App};
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
        "llm-dash starting"
    );

    let mut app = App::new(cli.budget);

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

    info!("starting TUI event loop");
    if let Err(e) = ui::run(app) {
        error!(error = %e, "dashboard terminated with error");
        eprintln!("Dashboard error: {e}");
        std::process::exit(1);
    }
    info!("llm-dash exited cleanly");
}
