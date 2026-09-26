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
use crate::recommendations::ModelRecommender;
use crate::ui::{theme::Theme, widgets};

/// Full dashboard rendering, called on every tick.
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

    render_title(frame, outer[0], ledger, budget, export_status, current_session);

    // Main content: left col + right col
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(outer[1]);

    // Left col: summary + budget + forecast + cache breakdown + savings
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // summary: 3 lines
            Constraint::Length(3), // budget gauge
            Constraint::Length(5), // forecast: 3 lines
            Constraint::Length(6), // cache: 4 lines
            Constraint::Min(3),    // savings: whatever is left
        ])
        .split(main[0]);

    let total = ledger.total_usd();
    let monthly = ledger.projected_monthly_usd(1);
    widgets::render_summary(frame, left[0], total, monthly, ledger.len());
    widgets::render_budget(frame, left[1], budget);
    render_forecast(frame, left[2], ledger);
    render_cache_breakdown(frame, left[3], ledger);
    render_savings_opportunities(frame, left[4], ledger);

    // Right col: model bar chart (top) + recent requests table (bottom)
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(main[1]);

    render_model_chart(frame, right[0], ledger);
    if ledger.is_empty() {
        render_empty_state(frame, right[1]);
    } else {
        render_requests_table(frame, right[1], ledger, scroll_offset);
    }

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
    ledger: &CostLedger,
    budget: &BudgetEnvelope,
    export_status: Option<&str>,
    current_session: Option<&str>,
) {
    let mut spans = vec![
        Span::styled(" llm-dash ", Theme::title()),
        Span::styled(" LLM spend, live", Theme::dim()),
        Span::styled(
            format!(
                "   {} requests   ${:.4} spent   ${:.2} budget",
                ledger.len(),
                ledger.total_usd(),
                budget.limit_usd
            ),
            Theme::normal(),
        ),
    ];
    if let Some(session) = current_session {
        spans.push(Span::styled(format!("   session: {session}"), Theme::warn()));
    }
    if let Some(status) = export_status {
        spans.push(Span::styled(format!("   Exported: {status}"), Theme::ok()));
    }
    let title = Paragraph::new(Line::from(spans));
    frame.render_widget(title, area);
}

/// Shown in place of the requests table before any request has been loaded.
fn render_empty_state(frame: &mut Frame, area: ratatui::layout::Rect) {
    let key = |k: &'static str| Span::styled(k, Theme::header());
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  No requests yet.",
            Theme::normal().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Press ", Theme::dim()),
            key("d"),
            Span::styled(" to load demo data and look around.", Theme::dim()),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  To watch your own traffic, quit and run:",
            Theme::dim(),
        )),
        Line::from(Span::styled(
            "    llm-dash --log-file requests.ndjson --budget 50",
            Theme::ok(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  One JSON object per line, for example:",
            Theme::dim(),
        )),
        Line::from(Span::styled(
            "    {\"model\":\"gpt-4o-mini\",\"input_tokens\":512,\"output_tokens\":256,\"latency_ms\":340}",
            Theme::normal(),
        )),
        Line::from(Span::styled(
            "  Lines your app appends while the dashboard is open show up live.",
            Theme::dim(),
        )),
    ];
    let paragraph = Paragraph::new(lines).block(
        Block::default()
            .title(" Recent Requests ")
            .borders(Borders::ALL)
            .border_style(Theme::border()),
    );
    frame.render_widget(paragraph, area);
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
    let mut spans = Vec::new();
    for (k, what) in [
        ("q", "quit"),
        ("d", "demo data"),
        ("r", "reset"),
        ("e", "export"),
        ("x", "explorer"),
        ("j/k", "scroll"),
    ] {
        spans.push(Span::styled(format!(" {k} "), Theme::header()));
        spans.push(Span::styled(format!("{what}  "), Theme::dim()));
    }
    spans.push(Span::styled(
        "  your data: llm-dash --log-file requests.ndjson",
        Theme::dim(),
    ));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_model_chart(frame: &mut Frame, area: ratatui::layout::Rect, ledger: &CostLedger) {
    use ratatui::widgets::{Bar, BarGroup};

    let block = Block::default()
        .title(" Cost by Model ")
        .borders(Borders::ALL)
        .border_style(Theme::border());

    let mut rows: Vec<(String, f64)> = ledger
        .by_model()
        .values()
        .map(|s| (s.model.clone(), s.total_cost_usd))
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    // One bar per text row inside the border.
    rows.truncate(area.height.saturating_sub(2).max(1) as usize);

    if rows.is_empty() {
        let empty = Paragraph::new(Line::from(Span::styled(
            " One bar per model appears here as requests come in.",
            Theme::dim(),
        )))
        .block(block);
        frame.render_widget(empty, area);
        return;
    }

    let label_w = rows
        .iter()
        .map(|(m, _)| m.chars().count())
        .max()
        .unwrap_or(0)
        .min(24);
    let bars: Vec<Bar> = rows
        .iter()
        .map(|(model, usd)| {
            let label: String = model.chars().take(label_w).collect();
            Bar::default()
                // micro-dollars keep small costs visible as integer bar lengths
                .value((usd * 1_000_000.0).round() as u64)
                .label(Line::from(vec![
                    Span::styled(format!("{label:<label_w$} "), Theme::normal()),
                    Span::styled(
                        format!("{:>9} ", format!("${usd:.4}")),
                        Theme::normal().add_modifier(Modifier::BOLD),
                    ),
                ]))
                .text_value(String::new())
                .style(Theme::ok())
        })
        .collect();

    let chart = BarChart::default()
        .block(block)
        .direction(Direction::Horizontal)
        .data(BarGroup::default().bars(&bars))
        .bar_width(1)
        .bar_gap(0)
        .label_style(Theme::normal());
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

/// Render the anomaly panel showing the last 10 detected cost anomalies.
///
/// Each anomaly is color-coded by severity:
/// - Low (2–3×): yellow
/// - Medium (3–5×): red
/// - High (>5×): bold red
fn render_anomalies(frame: &mut Frame, area: ratatui::layout::Rect, anomalies: &[CostAnomaly]) {
    use crate::cost::anomaly::AnomalySeverity;

    let lines: Vec<Line> = if anomalies.is_empty() {
        vec![Line::from(Span::styled(
            "  No anomalies detected",
            Theme::dim(),
        ))]
    } else {
        anomalies
            .iter()
            .rev()
            .take(4) // show at most 4 in the available height
            .map(|a| {
                let style = match a.severity {
                    AnomalySeverity::Low => Theme::warn(),
                    AnomalySeverity::Medium => Theme::danger(),
                    AnomalySeverity::High => {
                        Theme::danger().add_modifier(ratatui::style::Modifier::BOLD)
                    }
                };
                Line::from(vec![
                    Span::styled(
                        format!("  {} ", a.detected_at.format("%H:%M:%S")),
                        Theme::dim(),
                    ),
                    Span::styled(
                        format!("{:<18}", a.model.chars().take(18).collect::<String>()),
                        Theme::normal(),
                    ),
                    Span::styled(
                        format!("{} ", a.severity),
                        style,
                    ),
                    Span::styled(
                        format!("actual=${:.6} expected=${:.6} ({})", a.actual, a.expected, a.ratio_str()),
                        style,
                    ),
                ])
            })
            .collect()
    };

    let paragraph = Paragraph::new(lines).block(
        Block::default()
            .title(format!(
                " Cost Anomalies ({} detected) ",
                anomalies.len()
            ))
            .borders(Borders::ALL)
            .border_style(if anomalies.is_empty() {
                Theme::border()
            } else {
                Theme::danger()
            }),
    );
    frame.render_widget(paragraph, area);
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
            Line::from(Span::styled(
                if ledger.len() < 2 {
                    "(needs 2+ requests)"
                } else {
                    "(needs more time span)"
                },
                Theme::dim(),
            )),
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
        " 7-Day Spend Trend ({} to {}) ",
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
    let cache_read_cost: f64 = records.iter().map(|r| r.cache.cache_read_cost_usd).sum::<f64>() + 0.0;
    let cache_write_cost: f64 = records.iter().map(|r| r.cache.cache_write_cost_usd).sum::<f64>() + 0.0;

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

/// Render the SAVINGS OPPORTUNITIES panel using the model recommendation engine.
fn render_savings_opportunities(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    ledger: &CostLedger,
) {
    let recommender = ModelRecommender::new(ledger);
    let suggestions = recommender.suggest();

    let lines: Vec<Line> = if suggestions.is_empty() {
        vec![Line::from(Span::styled(
            "No savings identified yet.",
            Theme::dim(),
        ))]
    } else {
        // Show top 3 suggestions.
        // Two lines per suggestion so it fits the narrow left column.
        suggestions
            .iter()
            .take(3)
            .flat_map(|s| {
                [
                    Line::from(vec![
                        Span::styled(s.current_model.clone(), Theme::normal()),
                        Span::styled(" -> ", Theme::dim()),
                        Span::styled(s.suggested_model.clone(), Theme::ok()),
                    ]),
                    Line::from(Span::styled(
                        format!(
                            "  save {:.0}%, ${:.2}/mo",
                            s.saving_pct, s.projected_monthly_saving_usd
                        ),
                        Theme::ok(),
                    )),
                ]
            })
            .collect()
    };

    let paragraph = Paragraph::new(lines).block(
        Block::default()
            .title(" Savings Opportunities ")
            .borders(Borders::ALL)
            .border_style(Theme::border()),
    );
    frame.render_widget(paragraph, area);
}
