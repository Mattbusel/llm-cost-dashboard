//! # Dashboard Layout
//!
//! Top-level dashboard composition.  `render` is called on every tick and
//! assembles the full terminal layout from sub-panels defined in
//! [`crate::ui::widgets`].

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::Modifier,
    text::{Line, Span},
    widgets::{BarChart, Block, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

use crate::budget::BudgetEnvelope;
use crate::cost::CostLedger;
use crate::ui::{theme::Theme, widgets};

/// Full dashboard rendering — called on every tick.
pub fn render(
    frame: &mut Frame,
    ledger: &CostLedger,
    budget: &BudgetEnvelope,
    scroll_offset: usize,
) {
    let area = frame.area();

    // Outer layout: title bar + main + footer
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title bar
            Constraint::Min(10),   // main content
            Constraint::Length(3), // sparkline
            Constraint::Length(1), // help bar
        ])
        .split(area);

    render_title(frame, outer[0]);

    // Main content: left col + right col
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(outer[1]);

    // Left col: summary + budget
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(main[0]);

    let total = ledger.total_usd();
    let monthly = ledger.projected_monthly_usd(1);
    widgets::render_summary(frame, left[0], total, monthly, ledger.len());
    widgets::render_budget(frame, left[1], budget);

    // Right col: model bar chart (top) + recent requests table (bottom)
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(main[1]);

    render_model_chart(frame, right[0], ledger);
    render_requests_table(frame, right[1], ledger, scroll_offset);

    // Sparkline
    let spark_data = ledger.sparkline_data(60);
    widgets::render_sparkline(frame, outer[2], &spark_data);

    render_help(frame, outer[3]);
}

fn render_title(frame: &mut Frame, area: ratatui::layout::Rect) {
    let title = Paragraph::new(Line::from(vec![
        Span::styled(" LLM Cost Dashboard", Theme::title()),
        Span::styled(
            "  [q: quit | r: reset | d: demo data | j/k: scroll]",
            Theme::dim(),
        ),
    ]));
    frame.render_widget(title, area);
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
