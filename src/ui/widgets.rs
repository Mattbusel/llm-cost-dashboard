//! # Dashboard Widgets
//!
//! Reusable ratatui widget render functions.  Each function renders a single
//! logical panel of the dashboard and has no side effects beyond writing to
//! the provided `ratatui::Frame`.

use ratatui::{
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Frame,
};

use crate::budget::BudgetEnvelope;
use crate::ui::theme::Theme;

/// Render the budget gauge panel.
pub fn render_budget(frame: &mut Frame, area: Rect, budget: &BudgetEnvelope) {
    let pct = budget.pct_consumed();
    let mut style = Theme::budget_style(pct);
    if crate::ui::theme::no_color() {
        style = style.add_modifier(ratatui::style::Modifier::REVERSED);
    }
    let label = format!(
        "${:.2} of ${:.2}  {:.0}%  {}",
        budget.spent_usd,
        budget.limit_usd,
        pct * 100.0,
        budget.status(),
    );
    let gauge = Gauge::default()
        .block(
            Block::default()
                .title(" Budget ")
                .borders(Borders::ALL)
                .border_style(Theme::border()),
        )
        .gauge_style(style)
        .percent(budget.gauge_pct())
        .label(label);
    frame.render_widget(gauge, area);
}

/// Render a sparkline of recent spend values.
pub fn render_sparkline(frame: &mut Frame, area: Rect, data: &[u64]) {
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(" Spend over time (last 60 requests) ")
                .borders(Borders::ALL)
                .border_style(Theme::border()),
        )
        .data(data)
        .style(Theme::ok());
    frame.render_widget(sparkline, area);
}

/// Render a summary stat paragraph (total spend + projection).
pub fn render_summary(frame: &mut Frame, area: Rect, total: f64, monthly: f64, count: usize) {
    let lines = vec![
        Line::from(vec![
            Span::styled("Spent so far:   ", Theme::dim()),
            Span::styled(format!("${total:.4}"), Theme::ok()),
        ]),
        Line::from(vec![
            Span::styled("Month, 1h pace: ", Theme::dim()),
            Span::styled(format!("${monthly:.2}/mo"), Theme::warn()),
        ]),
        Line::from(vec![
            Span::styled("Requests:       ", Theme::dim()),
            Span::styled(format!("{count}"), Theme::normal()),
        ]),
    ];
    let paragraph = Paragraph::new(lines).block(
        Block::default()
            .title(" Summary ")
            .borders(Borders::ALL)
            .border_style(Theme::border()),
    );
    frame.render_widget(paragraph, area);
}
