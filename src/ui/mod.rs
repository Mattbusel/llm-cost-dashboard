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

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{backend::CrosstermBackend, Terminal};
use tracing::{debug, info, warn};

use crate::{
    budget::BudgetEnvelope,
    cost::{CostLedger, CostRecord},
    error::DashboardError,
    log::RequestLog,
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
        }
    }

    /// Inject a cost record and update the budget envelope.
    ///
    /// Budget overage is logged as a warning but does not abort the call;
    /// callers that need to react to budget violations should check
    /// [`BudgetEnvelope::is_over_budget`] after recording.
    pub fn record(&mut self, record: CostRecord) {
        let cost = record.total_cost_usd;
        let model = record.model.clone();
        if let Err(e) = self.ledger.add(record) {
            warn!(error = %e, "cost ledger rejected record");
        }
        if let Err(e) = self.budget.spend(cost) {
            warn!(model = %model, error = %e, "budget limit breached");
        }
        debug!(model = %model, cost_usd = cost, "record ingested");
    }

    /// Parse and ingest a single newline-delimited JSON line.
    ///
    /// Returns [`DashboardError::LogParseError`] on malformed input; the
    /// dashboard remains in a valid state and the caller may skip or surface
    /// the error.
    pub fn ingest_line(&mut self, line: &str) -> Result<(), DashboardError> {
        self.log.ingest_line(line)?;
        if let Some(last) = self.log.all().last() {
            let rec = CostRecord::new(
                &last.model,
                &last.provider,
                last.input_tokens,
                last.output_tokens,
                last.latency_ms,
            );
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

/// Inner event loop; separated so that terminal teardown is always reached.
fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> Result<(), DashboardError> {
    info!("entering event loop");
    while app.running {
        terminal
            .draw(|frame| dashboard::render(frame, &app.ledger, &app.budget, app.scroll_offset))
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
