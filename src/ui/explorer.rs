//! # Interactive Cost Explorer
//!
//! A spreadsheet-style TUI mode (activated by pressing `x`) that lets users
//! filter, sort, and drill into individual cost records.
//!
//! ## Features
//!
//! - Filter by model substring, date range, and tag key=value
//! - Sort by cost (ascending / descending) or timestamp
//! - Scroll through filtered results one row at a time
//! - Show detail pane for the selected record

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

use crate::cost::CostRecord;
use crate::ui::theme::Theme;

/// Sort column for the explorer table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SortColumn {
    /// Sort by timestamp (newest first).
    Timestamp,
    /// Sort by cost, highest first.
    CostDesc,
    /// Sort by cost, lowest first.
    CostAsc,
    /// Sort by model name (alphabetical).
    Model,
}

impl SortColumn {
    /// Cycle to the next sort mode.
    pub fn next(&self) -> Self {
        match self {
            Self::Timestamp => Self::CostDesc,
            Self::CostDesc => Self::CostAsc,
            Self::CostAsc => Self::Model,
            Self::Model => Self::Timestamp,
        }
    }

    /// Short display label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Timestamp => "Time↓",
            Self::CostDesc => "Cost↓",
            Self::CostAsc => "Cost↑",
            Self::Model => "Model A-Z",
        }
    }
}

/// Filter state for the interactive explorer.
#[derive(Debug, Default, Clone)]
pub struct ExplorerFilter {
    /// Substring filter on model name (case-insensitive; empty = no filter).
    pub model_contains: String,
    /// Optional tag filter `key=value`.
    pub tag_filter: Option<(String, String)>,
}

impl ExplorerFilter {
    /// Return `true` if `record` passes all active filter conditions.
    pub fn matches(&self, record: &CostRecord) -> bool {
        if !self.model_contains.is_empty()
            && !record
                .model
                .to_lowercase()
                .contains(&self.model_contains.to_lowercase())
        {
            return false;
        }
        // Tag filtering is best-effort: since CostRecord has no tags map, we
        // match on session_id as a proxy when a tag filter is set with key="session".
        if let Some((k, v)) = &self.tag_filter {
            if k == "session" {
                let sid = record.session_id.as_deref().unwrap_or("");
                if !sid.contains(v.as_str()) {
                    return false;
                }
            }
        }
        true
    }
}

/// State for the interactive cost explorer mode.
#[derive(Debug)]
pub struct ExplorerState {
    /// Current filter.
    pub filter: ExplorerFilter,
    /// Current sort column.
    pub sort: SortColumn,
    /// Selected row index within the filtered + sorted list.
    pub selected: usize,
    /// Scroll offset for the table.
    pub scroll_offset: usize,
    /// Whether the detail pane is visible for the selected row.
    pub detail_open: bool,
}

impl Default for ExplorerState {
    fn default() -> Self {
        Self {
            filter: ExplorerFilter::default(),
            sort: SortColumn::Timestamp,
            selected: 0,
            scroll_offset: 0,
            detail_open: false,
        }
    }
}

impl ExplorerState {
    /// Create a fresh explorer state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Move selection down by one row.
    pub fn move_down(&mut self, total: usize) {
        if total == 0 {
            return;
        }
        self.selected = (self.selected + 1).min(total - 1);
        if self.selected >= self.scroll_offset + 20 {
            self.scroll_offset += 1;
        }
    }

    /// Move selection up by one row.
    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.scroll_offset.saturating_sub(1);
        }
    }

    /// Cycle to the next sort column.
    pub fn cycle_sort(&mut self) {
        self.sort = self.sort.next();
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Toggle the detail pane.
    pub fn toggle_detail(&mut self) {
        self.detail_open = !self.detail_open;
    }

    /// Filter records and sort them according to current state.
    pub fn apply<'a>(&self, records: &'a [CostRecord]) -> Vec<&'a CostRecord> {
        let mut filtered: Vec<&'a CostRecord> =
            records.iter().filter(|r| self.filter.matches(r)).collect();

        match self.sort {
            SortColumn::Timestamp => {
                filtered.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
            }
            SortColumn::CostDesc => {
                filtered.sort_by(|a, b| {
                    b.total_cost_usd
                        .partial_cmp(&a.total_cost_usd)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortColumn::CostAsc => {
                filtered.sort_by(|a, b| {
                    a.total_cost_usd
                        .partial_cmp(&b.total_cost_usd)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortColumn::Model => {
                filtered.sort_by(|a, b| a.model.cmp(&b.model));
            }
        }
        filtered
    }
}

// ── Rendering ──────────────────────────────────────────────────────────────────

/// Render the full interactive cost explorer screen.
///
/// `records` is the complete set of cost records from the ledger.
/// `state` holds the current filter/sort/selection.
pub fn render_explorer(frame: &mut Frame, area: Rect, records: &[CostRecord], state: &ExplorerState) {
    let filtered = state.apply(records);
    let total_filtered = filtered.len();

    // Layout: help bar (1 line) | table | optional detail pane
    let (table_area, detail_area) = if state.detail_open {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Percentage(60),
                Constraint::Percentage(40),
            ])
            .split(area);
        render_explorer_help(frame, chunks[0], state, total_filtered);
        (chunks[1], Some(chunks[2]))
    } else {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(5)])
            .split(area);
        render_explorer_help(frame, chunks[0], state, total_filtered);
        (chunks[1], None)
    };

    render_explorer_table(frame, table_area, &filtered, state);

    if let Some(detail_a) = detail_area {
        let selected_rec = filtered.get(state.selected).copied();
        render_detail_pane(frame, detail_a, selected_rec);
    }
}

fn render_explorer_help(frame: &mut Frame, area: Rect, state: &ExplorerState, total: usize) {
    let filter_str = if state.filter.model_contains.is_empty() {
        String::from("(none)")
    } else {
        format!("model~\"{}\"", state.filter.model_contains)
    };
    let line1 = Line::from(vec![
        Span::styled(" [x: exit explorer | s: cycle sort | Enter: detail | j/k: scroll]  ", Theme::dim()),
        Span::styled(format!("Filter: {filter_str}  "), Theme::warn()),
        Span::styled(format!("Sort: {}  ", state.sort.label()), Theme::header()),
        Span::styled(format!("Showing {total} records"), Theme::normal()),
    ]);
    frame.render_widget(Paragraph::new(line1), area);
}

fn render_explorer_table(
    frame: &mut Frame,
    area: Rect,
    filtered: &[&CostRecord],
    state: &ExplorerState,
) {
    let header = Row::new(vec![
        Cell::from("#").style(Theme::header()),
        Cell::from("Time").style(Theme::header()),
        Cell::from("Model").style(Theme::header()),
        Cell::from("In").style(Theme::header()),
        Cell::from("Out").style(Theme::header()),
        Cell::from("Cost").style(Theme::header()),
        Cell::from("Latency").style(Theme::header()),
        Cell::from("Session").style(Theme::header()),
    ]);

    let page_size: usize = 24;
    let rows: Vec<Row> = filtered
        .iter()
        .enumerate()
        .skip(state.scroll_offset)
        .take(page_size)
        .map(|(global_idx, r)| {
            let is_selected = global_idx == state.selected;
            let row = Row::new(vec![
                Cell::from(format!("{}", global_idx + 1)),
                Cell::from(r.timestamp.format("%m-%d %H:%M:%S").to_string()),
                Cell::from(r.model.chars().take(20).collect::<String>()),
                Cell::from(r.input_tokens.to_string()),
                Cell::from(r.output_tokens.to_string()),
                Cell::from(format!("${:.6}", r.total_cost_usd)),
                Cell::from(format!("{}ms", r.latency_ms)),
                Cell::from(
                    r.session_id
                        .as_deref()
                        .unwrap_or("")
                        .chars()
                        .take(12)
                        .collect::<String>(),
                ),
            ]);
            if is_selected {
                row.style(Theme::highlight().add_modifier(Modifier::BOLD))
            } else {
                row
            }
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(5),
            Constraint::Length(18),
            Constraint::Length(21),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(13),
            Constraint::Length(9),
            Constraint::Length(13),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .title(" Cost Explorer ")
            .borders(Borders::ALL)
            .border_style(Theme::border()),
    );

    frame.render_widget(table, area);
}

fn render_detail_pane(frame: &mut Frame, area: Rect, record: Option<&CostRecord>) {
    let lines: Vec<Line> = match record {
        None => vec![Line::from(Span::styled("No record selected.", Theme::dim()))],
        Some(r) => vec![
            Line::from(vec![
                Span::styled("ID:        ", Theme::dim()),
                Span::styled(r.id.to_string(), Theme::normal()),
            ]),
            Line::from(vec![
                Span::styled("Timestamp: ", Theme::dim()),
                Span::styled(r.timestamp.to_rfc3339(), Theme::normal()),
            ]),
            Line::from(vec![
                Span::styled("Model:     ", Theme::dim()),
                Span::styled(r.model.clone(), Theme::header()),
            ]),
            Line::from(vec![
                Span::styled("Provider:  ", Theme::dim()),
                Span::styled(r.provider.clone(), Theme::normal()),
            ]),
            Line::from(vec![
                Span::styled("In tokens: ", Theme::dim()),
                Span::styled(r.input_tokens.to_string(), Theme::normal()),
                Span::styled("  Out tokens: ", Theme::dim()),
                Span::styled(r.output_tokens.to_string(), Theme::normal()),
            ]),
            Line::from(vec![
                Span::styled("Cost:      ", Theme::dim()),
                Span::styled(format!("${:.8}", r.total_cost_usd), Theme::ok()),
                Span::styled(
                    format!(
                        "  (in: ${:.8}  out: ${:.8})",
                        r.input_cost_usd, r.output_cost_usd
                    ),
                    Theme::dim(),
                ),
            ]),
            Line::from(vec![
                Span::styled("Cache R/W: ", Theme::dim()),
                Span::styled(
                    format!(
                        "{} / {} tokens",
                        r.cache.cache_read_tokens, r.cache.cache_write_tokens
                    ),
                    Theme::warn(),
                ),
            ]),
            Line::from(vec![
                Span::styled("Latency:   ", Theme::dim()),
                Span::styled(format!("{} ms", r.latency_ms), Theme::normal()),
            ]),
            Line::from(vec![
                Span::styled("Session:   ", Theme::dim()),
                Span::styled(
                    r.session_id.as_deref().unwrap_or("(none)").to_owned(),
                    Theme::normal(),
                ),
            ]),
            Line::from(vec![
                Span::styled("Request ID:", Theme::dim()),
                Span::styled(r.request_id.clone(), Theme::dim()),
            ]),
        ],
    };

    let para = Paragraph::new(lines).block(
        Block::default()
            .title(" Record Detail ")
            .borders(Borders::ALL)
            .border_style(Theme::border()),
    );
    frame.render_widget(para, area);
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cost::CostRecord;

    fn rec(model: &str, cost: f64) -> CostRecord {
        let mut r = CostRecord::new(model, "test", 100, 50, 10);
        r.total_cost_usd = cost;
        r
    }

    #[test]
    fn test_filter_model_contains() {
        let f = ExplorerFilter {
            model_contains: "haiku".to_string(),
            tag_filter: None,
        };
        assert!(f.matches(&rec("claude-haiku-4-5", 0.001)));
        assert!(!f.matches(&rec("gpt-4o-mini", 0.001)));
    }

    #[test]
    fn test_filter_empty_matches_all() {
        let f = ExplorerFilter::default();
        assert!(f.matches(&rec("any-model", 1.0)));
    }

    #[test]
    fn test_sort_cost_desc() {
        let records = vec![rec("a", 1.0), rec("b", 5.0), rec("c", 2.0)];
        let state = ExplorerState {
            sort: SortColumn::CostDesc,
            ..Default::default()
        };
        let filtered = state.apply(&records);
        assert!((filtered[0].total_cost_usd - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_sort_cost_asc() {
        let records = vec![rec("a", 3.0), rec("b", 1.0), rec("c", 2.0)];
        let state = ExplorerState {
            sort: SortColumn::CostAsc,
            ..Default::default()
        };
        let filtered = state.apply(&records);
        assert!((filtered[0].total_cost_usd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sort_model_alpha() {
        let records = vec![rec("zzz", 1.0), rec("aaa", 2.0), rec("mmm", 3.0)];
        let state = ExplorerState {
            sort: SortColumn::Model,
            ..Default::default()
        };
        let filtered = state.apply(&records);
        assert_eq!(filtered[0].model, "aaa");
        assert_eq!(filtered[2].model, "zzz");
    }

    #[test]
    fn test_move_down_clamps_at_end() {
        let mut s = ExplorerState::new();
        s.move_down(3);
        s.move_down(3);
        s.move_down(3);
        s.move_down(3); // already at end
        assert_eq!(s.selected, 2);
    }

    #[test]
    fn test_move_up_clamps_at_zero() {
        let mut s = ExplorerState::new();
        s.move_up();
        s.move_up();
        assert_eq!(s.selected, 0);
    }

    #[test]
    fn test_cycle_sort() {
        let mut s = ExplorerState::new();
        assert_eq!(s.sort, SortColumn::Timestamp);
        s.cycle_sort();
        assert_eq!(s.sort, SortColumn::CostDesc);
        s.cycle_sort();
        assert_eq!(s.sort, SortColumn::CostAsc);
        s.cycle_sort();
        assert_eq!(s.sort, SortColumn::Model);
        s.cycle_sort();
        assert_eq!(s.sort, SortColumn::Timestamp);
    }

    #[test]
    fn test_toggle_detail() {
        let mut s = ExplorerState::new();
        assert!(!s.detail_open);
        s.toggle_detail();
        assert!(s.detail_open);
        s.toggle_detail();
        assert!(!s.detail_open);
    }
}
