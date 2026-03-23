//! # UI Layer
//!
//! Terminal user-interface built with [ratatui] and [crossterm].
//!
//! The entry point for embedding consumers is [`App`]; the entry point for
//! running the full interactive TUI is [`run`].

/// Full-screen dashboard layout composer.
pub mod dashboard;
/// Centralised color and style palette.
pub mod theme;
/// Individual widget render functions.
pub mod widgets;

use std::time::Duration;

use chrono::Utc;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{backend::CrosstermBackend, Terminal};
use tracing::{debug, info, warn};

use crate::{
    budget::BudgetEnvelope,
    cost::{
        anomaly::{AnomalyDetector, CostAnomaly},
        CostLedger, CostRecord,
    },
    error::DashboardError,
    log::RequestLog,
    webhook::WebhookConfig,
};

/// Application state -- everything the TUI needs.
///
/// Create via [`App::new`], optionally pre-load data with [`App::load_demo_data`]
/// or [`App::ingest_line`], then pass to [`run`] to launch the interactive TUI.
pub struct App {
    /// The cost ledger tracking every completed request.
    pub ledger: CostLedger,
    /// Raw request log for display and filtering.
    pub log: RequestLog,
    /// Active budget envelope for spend enforcement.
    pub budget: BudgetEnvelope,
    /// Current scroll position in the requests table (rows from top).
    pub scroll_offset: usize,
    /// Whether the event loop should continue running.
    pub running: bool,
    /// Optional webhook configurations for budget alerts.
    pub webhooks: Vec<WebhookConfig>,
    /// Last export status message (shown briefly in the title bar).
    pub last_export_status: Option<String>,
    /// Active session name.  When set, all newly ingested records are tagged
    /// with this identifier (sets [`CostRecord::session_id`]).
    pub current_session: Option<String>,
    /// Rolling anomaly detector (per-model mean and std dev).
    pub anomaly_detector: AnomalyDetector,
    /// Ring buffer of the last 10 detected anomalies (oldest first).
    pub anomalies: Vec<CostAnomaly>,
}

impl App {
    /// Create a new application with the given monthly budget limit in USD.
    ///
    /// The budget alert threshold is set to 80% of the limit.
    pub fn new(budget_usd: f64) -> Self {
        info!(budget_usd, "creating App");
        Self {
            ledger: CostLedger::new(),
            log: RequestLog::new(),
            budget: BudgetEnvelope::new("Monthly", budget_usd, 0.8),
            scroll_offset: 0,
            running: true,
            webhooks: Vec::new(),
            last_export_status: None,
            current_session: None,
            anomaly_detector: AnomalyDetector::new(),
            anomalies: Vec::new(),
        }
    }

    /// Set the active session name.
    ///
    /// All records ingested after this call (via [`App::ingest_line`] or
    /// [`App::record`]) will have their `session_id` set to `session_name`.
    /// Pass `None` to clear the active session.
    pub fn set_session(&mut self, session_name: impl Into<String>) {
        self.current_session = Some(session_name.into());
    }

    /// Clear the active session so newly ingested records are not tagged.
    pub fn clear_session(&mut self) {
        self.current_session = None;
    }

    /// Register a webhook configuration for budget-threshold alerts.
    pub fn add_webhook(&mut self, cfg: WebhookConfig) {
        self.webhooks.push(cfg);
    }

    /// Inject a cost record and update the budget envelope.
    ///
    /// Budget overage is logged as a warning but does not abort the call;
    /// callers that need to react to budget violations should check
    /// [`BudgetEnvelope::is_over_budget`] after recording.
    pub fn record(&mut self, record: CostRecord) {
        let cost = record.total_cost_usd;
        let model = record.model.clone();

        // Run anomaly detection *before* adding to the ledger so the current
        // sample is scored against the historical baseline, not itself.
        if let Some(anomaly) = self.anomaly_detector.check(&record) {
            info!(
                model = %anomaly.model,
                expected = anomaly.expected,
                actual = anomaly.actual,
                severity = %anomaly.severity,
                "cost anomaly detected"
            );
            self.anomalies.push(anomaly);
            // Keep at most 10 anomalies.
            if self.anomalies.len() > 10 {
                self.anomalies.remove(0);
            }
        }

        if let Err(e) = self.ledger.add(record) {
            warn!(error = %e, "cost ledger rejected record");
        }
        if let Err(e) = self.budget.spend(cost) {
            warn!(model = %model, error = %e, "budget limit breached");
        }
        debug!(model = %model, cost_usd = cost, "record ingested");

        // Fire webhook alerts synchronously (best-effort).
        let spent = self.budget.spent_usd;
        let limit = self.budget.limit_usd;
        for cfg in &self.webhooks {
            if spent >= cfg.threshold_usd {
                let cfg_clone = cfg.clone();
                tokio::spawn(async move {
                    if let Err(e) =
                        crate::webhook::fire_budget_alert(&cfg_clone, spent, limit).await
                    {
                        tracing::warn!(error = %e, "webhook alert delivery failed");
                    }
                });
            }
        }
    }

    /// Export the current session data to disk as both JSON and CSV.
    ///
    /// Files are written to the current working directory:
    /// - `llm-costs-<timestamp>.json`
    /// - `llm-costs-<timestamp>.csv`
    ///
    /// Sets [`App::last_export_status`] to a short status message that the TUI
    /// displays in the title bar.
    pub fn export_session(&mut self) {
        use std::fs;
        let ts = Utc::now().format("%Y%m%d-%H%M%S");
        let mut msgs: Vec<String> = Vec::new();

        match self.ledger.to_json() {
            Ok(data) => {
                let path = format!("llm-costs-{ts}.json");
                match fs::write(&path, &data) {
                    Ok(()) => {
                        info!(path, "JSON export written");
                        msgs.push(format!("JSON→{path}"));
                    }
                    Err(e) => {
                        warn!(error = %e, "JSON export write failed");
                        msgs.push(format!("JSON err: {e}"));
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "JSON serialization failed");
                msgs.push(format!("JSON err: {e}"));
            }
        }

        match self.ledger.to_csv() {
            Ok(data) => {
                let path = format!("llm-costs-{ts}.csv");
                match fs::write(&path, &data) {
                    Ok(()) => {
                        info!(path, "CSV export written");
                        msgs.push(format!("CSV→{path}"));
                    }
                    Err(e) => {
                        warn!(error = %e, "CSV export write failed");
                        msgs.push(format!("CSV err: {e}"));
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "CSV serialization failed");
                msgs.push(format!("CSV err: {e}"));
            }
        }

        self.last_export_status = Some(msgs.join(" | "));
    }

    /// Parse and ingest a single newline-delimited JSON line.
    ///
    /// Returns [`DashboardError::LogParseError`] on malformed input; the
    /// dashboard remains in a valid state and the caller may skip or surface
    /// the error.
    pub fn ingest_line(&mut self, line: &str) -> Result<(), DashboardError> {
        self.log.ingest_line(line)?;
        if let Some(last) = self.log.all().last() {
            let mut rec = CostRecord::new(
                &last.model,
                &last.provider,
                last.input_tokens,
                last.output_tokens,
                last.latency_ms,
            );
            // Tag the record with the active session if one is set.
            if let Some(ref session) = self.current_session {
                rec = rec.with_session(session.clone());
            }
            self.record(rec);
        }
        Ok(())
    }

    /// Inject the built-in set of synthetic demo records covering multiple
    /// providers and models so the dashboard renders a realistic layout
    /// immediately after launch with `--demo`.
    pub fn load_demo_data(&mut self) {
        let demos: &[(&str, &str, u64, u64, u64)] = &[
            ("claude-sonnet-4-6", "anthropic", 847, 312, 45),
            ("gpt-4o-mini", "openai", 512, 128, 12),
            ("claude-haiku-4-5", "anthropic", 256, 64, 8),
            ("claude-sonnet-4-6", "anthropic", 1024, 512, 120),
            ("gpt-4o", "openai", 2048, 1024, 340),
            ("gpt-4o-mini", "openai", 400, 200, 15),
            ("claude-sonnet-4-6", "anthropic", 600, 300, 55),
            ("o3-mini", "openai", 800, 400, 200),
            ("gpt-4o-mini", "openai", 300, 150, 10),
            ("claude-haiku-4-5", "anthropic", 128, 32, 5),
            ("claude-sonnet-4-6", "anthropic", 512, 256, 40),
            ("gpt-4o", "openai", 1024, 512, 180),
            ("gemini-1.5-flash", "google", 700, 350, 30),
            ("claude-sonnet-4-6", "anthropic", 900, 450, 95),
            ("gpt-4o-mini", "openai", 600, 300, 22),
            ("claude-haiku-4-5", "anthropic", 200, 100, 7),
            ("o3-mini", "openai", 1500, 750, 280),
            ("claude-sonnet-4-6", "anthropic", 750, 375, 65),
            ("gpt-4o", "openai", 512, 256, 110),
            ("gemini-1.5-pro", "google", 1000, 500, 150),
        ];
        info!(count = demos.len(), "loading demo data");
        for (model, provider, inp, out, lat) in demos {
            self.record(CostRecord::new(*model, *provider, *inp, *out, *lat));
        }
    }

    /// Scroll the requests table down by one row.
    pub fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(1);
    }

    /// Scroll the requests table up by one row.
    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    /// Clear all data and reset scroll position, returning to a blank state.
    pub fn reset(&mut self) {
        info!("resetting all application state");
        self.ledger.clear();
        self.log.clear();
        self.budget.reset();
        self.scroll_offset = 0;
        self.last_export_status = None;
        self.anomaly_detector.reset();
        self.anomalies.clear();
    }
}

/// Run the full TUI event loop, initialising raw-mode and the alternate screen.
///
/// Blocks until the user presses `q` or `Esc`.  Terminal state (raw mode and
/// the alternate screen) is always restored before this function returns,
/// even when an error occurs inside the event loop.
///
/// # Errors
///
/// Returns [`DashboardError::Terminal`] if raw-mode setup or any crossterm
/// operation fails.
pub fn run(mut app: App) -> Result<(), DashboardError> {
    info!("initialising terminal (raw mode + alternate screen)");
    crossterm::terminal::enable_raw_mode().map_err(|e| DashboardError::Terminal(e.to_string()))?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)
        .map_err(|e| DashboardError::Terminal(e.to_string()))?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal =
        Terminal::new(backend).map_err(|e| DashboardError::Terminal(e.to_string()))?;

    let result = event_loop(&mut terminal, &mut app);

    // Always attempt cleanup even if the event loop errored.
    if let Err(e) = crossterm::terminal::disable_raw_mode() {
        warn!(error = %e, "failed to disable raw mode during cleanup");
    }
    if let Err(e) = crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    ) {
        warn!(error = %e, "failed to leave alternate screen during cleanup");
    }

    info!("terminal restored");
    result
}

/// Inner event loop; separated from [`run`] so that terminal teardown in
/// `run` is always reached regardless of whether the loop exits normally or
/// with an error.
///
/// Redraws the dashboard on every 250 ms tick and processes keyboard events.
/// Returns when `app.running` is `false` (set by the quit key handler) or
/// when any terminal operation returns an error.
///
/// # Errors
///
/// Returns [`DashboardError::Terminal`] if a [`crossterm`] draw, poll, or
/// read operation fails.
fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> Result<(), DashboardError> {
    info!("entering event loop");
    while app.running {
        terminal
            .draw(|frame| {
                dashboard::render(
                    frame,
                    &app.ledger,
                    &app.budget,
                    app.scroll_offset,
                    app.last_export_status.as_deref(),
                    &app.anomalies,
                    app.current_session.as_deref(),
                )
            })
            .map_err(|e| DashboardError::Terminal(e.to_string()))?;

        if event::poll(Duration::from_millis(250))
            .map_err(|e| DashboardError::Terminal(e.to_string()))?
        {
            match event::read().map_err(|e| DashboardError::Terminal(e.to_string()))? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        info!("quit key pressed -- stopping event loop");
                        app.running = false;
                    }
                    KeyCode::Char('r') => {
                        info!("reset triggered by user");
                        app.reset();
                    }
                    KeyCode::Char('d') => {
                        info!("loading demo data via keypress");
                        app.load_demo_data();
                    }
                    KeyCode::Char('e') => {
                        info!("export triggered by user");
                        app.export_session();
                    }
                    KeyCode::Char('j') | KeyCode::Down => app.scroll_down(),
                    KeyCode::Char('k') | KeyCode::Up => app.scroll_up(),
                    _ => {}
                },
                _ => {}
            }
        }
    }
    info!("event loop finished");
    Ok(())
}
