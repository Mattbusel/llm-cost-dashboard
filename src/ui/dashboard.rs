//! # Dashboard Layout
//!
//! Top-level dashboard composition.  `render` is called on every tick and
//! assembles the full terminal layout from sub-panels defined in
//! [`crate::ui::widgets`].

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::Modifier,
    text::{Line, Span},
    widgets::{BarChart, Block, Borders, Cell, Paragraph, Row, Sparkline, Table},
    Frame,
};

use crate::budget::BudgetEnvelope;
use crate::cost::{anomaly::CostAnomaly, CostLedger};
use crate::forecast::{ForecastResult, SpendForecaster, Trend};
use crate::ui::{theme::Theme, widgets};

/// Full dashboard rendering — called on every tick.
///
/// # Parameters
///
/// - `export_status`: optional short message shown in the title bar after the
///   user presses `e` to export session data.
/// - `anomalies`: slice of the last 10 detected cost anomalies (oldest first).
/// - `current_session`: active session name, shown in the title bar when set.
pub fn render(
    frame: &mut Frame,
    ledger: &CostLedger,
    budget: &BudgetEnvelope,
    scroll_offset: usize,
    export_status: Option<&str>,
    anomalies: &[CostAnomaly],
    current_session: Option<&str>,
) {
    let area = frame.area();

    // Outer layout: title bar + main + anomalies + trend + sparkline + help bar
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title bar
            Constraint::Min(10),   // main content
            Constraint::Length(5), // anomaly panel (shown when anomalies exist)
            Constraint::Length(5), // 7-day trend panel
            Constraint::Length(3), // per-request sparkline
            Constraint::Length(1), // help bar
        ])
        .split(area);

    render_title(frame, outer[0], export_status, current_session);

    // Main content: left col + right col
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(outer[1]);

    // Left col: summary + budget + forecast + cache breakdown
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(main[0]);

    let total = ledger.total_usd();
    let monthly = ledger.projected_monthly_usd(1);
    widgets::render_summary(frame, left[0], total, monthly, ledger.len());
    widgets::render_budget(frame, left[1], budget);
    render_forecast(frame, left[2], ledger);
    render_cache_breakdown(frame, left[3], ledger);

    // Right col: model bar chart (top) + recent requests table (bottom)
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(main[1]);

    render_model_chart(frame, right[0], ledger);
    render_requests_table(frame, right[1], ledger, scroll_offset);

    // Anomaly panel
    render_anomalies(frame, outer[2], anomalies);

    // 7-day trend panel
    render_trend(frame, outer[3], ledger);

    // Per-request sparkline
    let spark_data = ledger.sparkline_data(60);
    widgets::render_sparkline(frame, outer[4], &spark_data);

    render_help(frame, outer[5]);
}

fn render_title(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    export_status: Option<&str>,
    current_session: Option<&str>,
) {
    let mut spans = vec![
        Span::styled(" LLM Cost Dashboard", Theme::title()),
        Span::styled(
            "  [q: quit | r: reset | d: demo | e: export | j/k: scroll]",
            Theme::dim(),
        ),
    ];
    if let Some(session) = current_session {
        spans.push(Span::styled(
            format!("  [session: {session}]"),
            Theme::warn(),
        ));
    }
    if let Some(status) = export_status {
        spans.push(Span::styled(format!("  Exported: {status}"), Theme::ok()));
    }
    let title = Paragraph::new(Line::from(spans));
    frame.render_widget(title, area);
}

/// Render a compact API key validation status line in the header area.
///
/// Intended for future use when keys are configured at startup.
#[allow(dead_code)]
pub fn render_validation_status(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    statuses: &[(&str, bool)],
) {
    let spans: Vec<Span> = statuses
        .iter()
        .flat_map(|(provider, valid)| {
            let style = if *valid { Theme::ok() } else { Theme::danger() };
            let icon = if *valid { "✓" } else { "✗" };
            vec![
                Span::styled(format!(" {provider}:{icon}"), style),
                Span::raw(" "),
            ]
        })
        .collect();
    let paragraph = Paragraph::new(Line::from(spans));
    frame.render_widget(paragraph, area);
}

fn render_help(frame: &mut Frame, area: ratatui::layout::Rect) {
    let help = Paragraph::new(Line::from(Span::styled(
        " Pipe data: echo '{\"model\":\"claude-sonnet-4-6\",\"input_tokens\":512,\"output_tokens\":256,\"latency_ms\":340}' | llm-dash",
        Theme::dim(),
    )));
    frame.render_widget(help, area);
}

fn render_model_chart(frame: &mut Frame, area: ratatui::layout::Rect, ledger: &CostLedger) {
    let by_model = ledger.by_model();
    // We need owned strings so we collect and format
    let model_data: Vec<(String, u64)> = {
        let mut v: Vec<_> = by_model
            .values()
            .map(|s| {
                let label = s.model.chars().take(16).collect::<String>();
                let val = (s.total_cost_usd * 1_000_000.0) as u64;
                (label, val)
            })
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(8);
        v
    };
    let bar_data: Vec<(&str, u64)> = model_data.iter().map(|(s, v)| (s.as_str(), *v)).collect();
    let chart = BarChart::default()
        .block(
            Block::default()
                .title(" Cost by Model (μUSD) ")
                .borders(Borders::ALL)
                .border_style(Theme::border()),
        )
        .data(&bar_data)
        .bar_width(3)
        .bar_gap(1)
        .bar_style(Theme::ok())
        .value_style(Theme::header());
    frame.render_widget(chart, area);
}

fn render_requests_table(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    ledger: &CostLedger,
    scroll_offset: usize,
) {
    let header = Row::new(vec![
        Cell::from("Time").style(Theme::header()),
        Cell::from("Model").style(Theme::header()),
        Cell::from("In").style(Theme::header()),
        Cell::from("Out").style(Theme::header()),
        Cell::from("CacheR").style(Theme::header()),
        Cell::from("Cost").style(Theme::header()),
        Cell::from("Latency").style(Theme::header()),
    ]);

    let records = ledger.last_n(200);
    let visible: Vec<Row> = records
        .iter()
        .rev()
        .skip(scroll_offset)
        .take(20)
        .map(|r| {
            Row::new(vec![
                Cell::from(r.timestamp.format("%H:%M:%S").to_string()),
                Cell::from(r.model.chars().take(18).collect::<String>()),
                Cell::from(r.input_tokens.to_string()),
                Cell::from(r.output_tokens.to_string()),
                Cell::from(r.cache.cache_read_tokens.to_string()),
                Cell::from(format!("${:.6}", r.total_cost_usd)),
                Cell::from(format!("{}ms", r.latency_ms)),
            ])
        })
        .collect();

    let table = Table::new(
        visible,
        [
            Constraint::Length(10),
            Constraint::Length(19),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(7),
            Constraint::Length(12),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .title(" Recent Requests ")
            .borders(Borders::ALL)
            .border_style(Theme::border()),
    )
    .row_highlight_style(Theme::highlight().add_modifier(Modifier::BOLD));

    frame.render_widget(table, area);
}

/// Render the cost forecast widget showing projected daily/monthly spend and trend.
fn render_forecast(frame: &mut Frame, area: ratatui::layout::Rect, ledger: &CostLedger) {
    // Build a SpendForecaster from all records (cumulative cost over time).
    let mut forecaster = SpendForecaster::new();
    let mut cumulative = 0.0f64;
    for record in ledger.records() {
        cumulative += record.total_cost_usd;
        forecaster.record(record.timestamp.timestamp() as f64, cumulative);
    }
    let fc: Option<ForecastResult> = forecaster.forecast(None);

    let lines = if let Some(ref fc) = fc {
        let (trend_char, trend_style) = match fc.trend {
            Trend::Accelerating => ("↑", Theme::warn()),
            Trend::Stable => ("→", Theme::normal()),
            Trend::Decelerating => ("↓", Theme::ok()),
        };
        vec![
            Line::from(vec![
                Span::styled("Proj/day:  ", Theme::dim()),
                Span::styled(format!("${:.6}", fc.projected_daily_usd), Theme::ok()),
                Span::raw("  "),
                Span::styled(trend_char, trend_style),
            ]),
            Line::from(vec![
                Span::styled("Proj/mo:   ", Theme::dim()),
                Span::styled(
                    format!("${:.4}", fc.projected_month_end_usd),
                    Theme::warn(),
                ),
            ]),
            Line::from(vec![
                Span::styled("Confidence:", Theme::dim()),
                Span::styled(
                    format!(" {:.0}%", fc.confidence * 100.0),
                    Theme::normal(),
                ),
            ]),
        ]
    } else {
        vec![
            Line::from(Span::styled("Proj/day:  --", Theme::dim())),
            Line::from(Span::styled("Proj/mo:   --", Theme::dim())),
            Line::from(Span::styled("(need ≥2 records)", Theme::dim())),
        ]
    };

    let paragraph = Paragraph::new(lines).block(
        Block::default()
            .title(" Forecast ")
            .borders(Borders::ALL)
            .border_style(Theme::border()),
    );
    frame.render_widget(paragraph, area);
}

/// Render the 7-day historical spend trend as a sparkline.
fn render_trend(frame: &mut Frame, area: ratatui::layout::Rect, ledger: &CostLedger) {
    let trend = ledger.seven_day_trend();
    // Scale to integer micro-USD for the Sparkline widget
    let data: Vec<u64> = trend
        .iter()
        .map(|&v| (v * 1_000_000.0) as u64)
        .collect();

    // Build a label showing today's date and the span
    let today = chrono::Utc::now().date_naive();
    let start = today - chrono::Duration::days(6);
    let title = format!(
        " 7-Day Spend Trend ({} → {}) ",
        start.format("%b %d"),
        today.format("%b %d"),
    );

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Theme::border()),
        )
        .data(&data)
        .style(Theme::warn());
    frame.render_widget(sparkline, area);
}

/// Render a cache hit/miss cost breakdown panel.
fn render_cache_breakdown(frame: &mut Frame, area: ratatui::layout::Rect, ledger: &CostLedger) {
    let records = ledger.records();
    let total_cache_read: u64 = records.iter().map(|r| r.cache.cache_read_tokens).sum();
    let total_cache_write: u64 = records.iter().map(|r| r.cache.cache_write_tokens).sum();
    let cache_read_cost: f64 = records.iter().map(|r| r.cache.cache_read_cost_usd).sum();
    let cache_write_cost: f64 = records.iter().map(|r| r.cache.cache_write_cost_usd).sum();

    let lines = vec![
        Line::from(vec![
            Span::styled("Cache reads:  ", Theme::dim()),
            Span::styled(
                format!("{total_cache_read} tok"),
                Theme::ok(),
            ),
        ]),
        Line::from(vec![
            Span::styled("Read cost:    ", Theme::dim()),
            Span::styled(format!("${cache_read_cost:.6}"), Theme::ok()),
        ]),
        Line::from(vec![
            Span::styled("Cache writes: ", Theme::dim()),
            Span::styled(
                format!("{total_cache_write} tok"),
                Theme::warn(),
            ),
        ]),
        Line::from(vec![
            Span::styled("Write cost:   ", Theme::dim()),
            Span::styled(format!("${cache_write_cost:.6}"), Theme::warn()),
        ]),
    ];
    let paragraph = Paragraph::new(lines).block(
        Block::default()
            .title(" Cache Breakdown ")
            .borders(Borders::ALL)
            .border_style(Theme::border()),
    );
    frame.render_widget(paragraph, area);
}
