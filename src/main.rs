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

    /// Show multi-provider cost comparison: rank all models by monthly cost for the
    /// given workload and exit (no TUI).
    ///
    /// The workload is derived from the loaded log data when `--log-file` or
    /// `--demo` is given, or from `--workload-rph` when no log data is present.
    ///
    /// Example: `llm-dash --demo --compare`
    #[arg(long)]
    compare: bool,

    /// Requests per hour for multi-provider cost comparison projections.
    ///
    /// Only used when `--compare` is set and no log data is available.
    /// Default: 1000 requests / hour.
    #[arg(long, value_name = "N", default_value = "1000")]
    workload_rph: u64,

    /// Print a Holt-Winters cost forecast and exit (no TUI).
    ///
    /// Requires at least 3 observations from the loaded log file or demo data.
    /// Projects spend over the next hour, day, week, and month.
    ///
    /// Example: `llm-dash --log-file requests.ndjson --forecast`
    #[arg(long)]
    forecast: bool,

    /// Load budget alert rules from a TOML file and start a background check loop.
    ///
    /// The TOML file must contain an array of `[[rules]]` tables, each with
    /// the fields: `name`, `threshold_usd`, `window` ("daily"|"weekly"|"monthly"),
    /// and `cooldown_secs`.
    ///
    /// Example: `llm-dash --demo --alerts rules.toml`
    #[arg(long, value_name = "RULES_TOML")]
    alerts: Option<std::path::PathBuf>,

    /// Print a cost anomaly report for the current ledger and exit (no TUI).
    ///
    /// Runs the Welford Z-score detector over all loaded records and prints
    /// any detected anomalies with their Z-scores.
    ///
    /// Example: `llm-dash --demo --anomaly`
    #[arg(long)]
    anomaly: bool,
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

    // Handle --compare: print multi-provider cost comparison and exit.
    if cli.compare {
        use llm_cost_dashboard::comparison::{ProviderComparison, WorkloadProfile};

        let profile = WorkloadProfile::from_ledger(&app.ledger)
            .unwrap_or_else(|| {
                info!(rph = cli.workload_rph, "deriving workload profile from --workload-rph");
                WorkloadProfile::from_rph(cli.workload_rph)
            });

        info!(
            avg_input = profile.avg_input_tokens,
            avg_output = profile.avg_output_tokens,
            requests_per_day = profile.requests_per_day,
            "computing provider comparison"
        );

        let cmp = ProviderComparison::compute(&profile);

        println!(
            "\nMulti-Provider Cost Comparison  ({} models, {}/day requests, {}in/{}out avg tokens)\n",
            cmp.model_count(),
            profile.requests_per_day,
            profile.avg_input_tokens,
            profile.avg_output_tokens,
        );
        println!(
            "  {:<45}  {:>12}  {:>10}  {:>15}  {}",
            "Model", "Monthly USD", "Daily USD", "Per-1k-req USD", "Provider"
        );
        println!("  {}", "-".repeat(100));
        for proj in cmp.ranked() {
            println!(
                "  {:<45}  {:>12.4}  {:>10.4}  {:>15.4}  {}",
                proj.model,
                proj.monthly_cost_usd,
                proj.daily_cost_usd,
                proj.cost_per_1k_requests,
                proj.provider,
            );
        }
        println!(
            "\nCheapest: {} (${:.4}/mo)  |  Most expensive: {} (${:.4}/mo)  |  Spread: {:.0}x",
            cmp.cheapest().model,
            cmp.cheapest().monthly_cost_usd,
            cmp.most_expensive().model,
            cmp.most_expensive().monthly_cost_usd,
            cmp.cost_spread_ratio(),
        );
        return;
    }

    // Handle --forecast: print Holt-Winters forecast and exit.
    if cli.forecast {
        use llm_cost_dashboard::forecast::CostForecaster;

        let records = app.ledger.records();
        if records.len() < 3 {
            eprintln!(
                "Error: --forecast requires at least 3 cost records (have {}). \
                 Use --log-file or --demo to load data first.",
                records.len()
            );
            std::process::exit(1);
        }

        let mut forecaster = CostForecaster::new();
        let mut cumulative = 0.0_f64;
        for record in records {
            cumulative += record.total_cost_usd;
            let ts = record.timestamp.timestamp() as f64;
            forecaster.record(ts, cumulative);
        }

        match forecaster.forecast(Some(cli.budget)) {
            Some(hw) => {
                println!("\nHolt-Winters Cost Forecast (based on {} records)\n", records.len());
                println!("  Next hour:  ${:.6}", hw.next_hour_usd);
                println!("  Next day:   ${:.4}", hw.next_day_usd);
                println!("  Next week:  ${:.2}", hw.next_week_usd);
                println!("  Next month: ${:.2}", hw.next_month_usd);
                println!(
                    "\n  80%% CI (next hour): [${:.6}, ${:.6}]",
                    hw.confidence_interval.0,
                    hw.confidence_interval.1
                );
                if hw.budget_warning {
                    eprintln!(
                        "\n  WARNING: forecasted monthly spend (${:.2}) exceeds 80%% of \
                         budget (${:.2})!",
                        hw.next_month_usd,
                        cli.budget
                    );
                } else {
                    println!("  Budget status: OK (monthly forecast ${:.2} < 80%% of ${:.2} budget)",
                        hw.next_month_usd, cli.budget);
                }
            }
            None => {
                eprintln!("Error: insufficient data for Holt-Winters forecast (need >= 3 observations).");
                std::process::exit(1);
            }
        }
        return;
    }

    // Handle --anomaly: print anomaly report for current ledger and exit.
    if cli.anomaly {
        use llm_cost_dashboard::anomaly::{AnomalyConfig, AnomalyDetector};

        let records = app.ledger.records();
        if records.len() < 3 {
            eprintln!(
                "Error: --anomaly requires at least 3 cost records (have {}). \
                 Use --log-file or --demo to load data first.",
                records.len()
            );
            std::process::exit(1);
        }

        let mut detector = AnomalyDetector::new(AnomalyConfig {
            window_size: 30,
            z_threshold: 3.0,
            min_samples: 5,
        });

        let observations: Vec<_> = records
            .iter()
            .map(|r| (r.timestamp, r.total_cost_usd))
            .collect();
        let report = detector.analyze(&observations);

        println!("\nCost Anomaly Report (based on {} records)\n", records.len());
        println!("  Anomaly rate: {:.1}%", report.anomaly_rate * 100.0);
        println!("  Max Z-score:  {:.2}", report.max_z_score);
        println!("  Anomalies detected: {}", report.anomalies.len());

        if !report.anomalies.is_empty() {
            println!("\n  {:<30}  {:>10}  {:>10}  {:>10}  {:>10}",
                "Timestamp", "Cost USD", "Mean USD", "Stddev", "Z-score");
            println!("  {}", "-".repeat(80));
            for (ts, r) in &report.anomalies {
                println!(
                    "  {:<30}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.2}",
                    ts.to_rfc3339(),
                    r.current_usd,
                    r.mean_usd,
                    r.stddev_usd,
                    r.z_score,
                );
            }
        } else {
            println!("\n  No anomalies detected.");
        }
        return;
    }

    // Handle --alerts: load rules from TOML and start a background check loop.
    if let Some(ref alerts_path) = cli.alerts {
        use llm_cost_dashboard::alerts::{AlertChannel, AlertEngine, AlertRule, AlertWindow};
        use std::time::Duration;

        #[derive(serde::Deserialize)]
        struct RuleToml {
            name: String,
            threshold_usd: f64,
            window: String,
            cooldown_secs: u64,
        }

        #[derive(serde::Deserialize)]
        struct RulesFile {
            rules: Vec<RuleToml>,
        }

        info!(path = %alerts_path.display(), "loading alert rules from TOML");
        let content = match std::fs::read_to_string(alerts_path) {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to read alert rules file");
                eprintln!("Error reading alert rules: {e}");
                std::process::exit(1);
            }
        };

        let rules_file: RulesFile = match toml::from_str(&content) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "failed to parse alert rules TOML");
                eprintln!("Error parsing alert rules TOML: {e}");
                std::process::exit(1);
            }
        };

        let rules: Vec<AlertRule> = rules_file
            .rules
            .into_iter()
            .map(|r| {
                let window = match r.window.to_lowercase().as_str() {
                    "daily" => AlertWindow::Daily,
                    "weekly" => AlertWindow::Weekly,
                    "monthly" => AlertWindow::Monthly,
                    other => {
                        warn!(window = %other, "unknown window type, defaulting to Daily");
                        AlertWindow::Daily
                    }
                };
                AlertRule {
                    name: r.name,
                    threshold_usd: r.threshold_usd,
                    window,
                    channel: AlertChannel::Log,
                    cooldown: Duration::from_secs(r.cooldown_secs),
                }
            })
            .collect();

        info!(count = rules.len(), "alert rules loaded");

        let mut engine = AlertEngine::new(rules);

        // Do an immediate check against the current ledger.
        let alerts = engine.check(&app.ledger);
        info!(fired = alerts.len(), "initial alert check complete");
        for a in &alerts {
            println!("ALERT: {}", a.message);
        }

        let summary = engine.summary();
        println!(
            "\nAlert summary: {} fired, {} suppressed by cooldown, {} rules configured",
            summary.fired_total, summary.suppressed_by_cooldown, summary.rules_count
        );

        // Note: in a long-running mode the caller would integrate this check
        // into the TUI event loop. For now we just run one check and fall
        // through to the TUI.
        info!("alert check complete; continuing to TUI");
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
