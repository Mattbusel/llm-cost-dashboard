use ratatui::style::{Color, Modifier, Style};
use std::sync::OnceLock;

/// `true` when the `NO_COLOR` environment variable is set to a non-empty
/// value (see <https://no-color.org>). Checked once per process.
pub fn no_color() -> bool {
    static NO_COLOR: OnceLock<bool> = OnceLock::new();
    *NO_COLOR.get_or_init(|| std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()))
}

/// Foreground style, or a plain style when colour is disabled.
fn fg(color: Color) -> Style {
    if no_color() {
        Style::default()
    } else {
        Style::default().fg(color)
    }
}

/// Centralised colour and style palette for the dashboard.
///
/// All widgets should source their styles from this struct so that the visual
/// theme can be changed in one place.
pub struct Theme;

impl Theme {
    /// Bold cyan style used for the top title bar.
    pub fn title() -> Style {
        fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    }

    /// Bold yellow style used for table column headers.
    pub fn header() -> Style {
        fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    }

    /// Green style indicating a healthy / within-budget state.
    pub fn ok() -> Style {
        fg(Color::Green)
    }

    /// Yellow style indicating a warning state (e.g. alert threshold crossed).
    pub fn warn() -> Style {
        fg(Color::Yellow)
    }

    /// Red style indicating an error or over-budget state.
    pub fn danger() -> Style {
        fg(Color::Red)
    }

    /// Default terminal foreground for ordinary body text (readable on both
    /// dark and light terminal themes).
    pub fn normal() -> Style {
        fg(Color::Reset)
    }

    /// Dark-grey style used for labels and secondary information.
    pub fn dim() -> Style {
        fg(Color::DarkGray)
    }

    /// Cyan-on-black style used to highlight a selected table row.
    pub fn highlight() -> Style {
        if no_color() {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default().fg(Color::Black).bg(Color::Cyan)
        }
    }

    /// Dark-grey style for widget borders.
    pub fn border() -> Style {
        fg(Color::DarkGray)
    }

    /// Choose `ok`, `warn`, or `danger` based on the fraction of budget consumed.
    ///
    /// - `pct < 0.8` returns [`Theme::ok`]
    /// - `0.8 <= pct < 1.0` returns [`Theme::warn`]
    /// - `pct >= 1.0` returns [`Theme::danger`]
    pub fn budget_style(pct: f64) -> Style {
        if pct >= 1.0 {
            Self::danger()
        } else if pct >= 0.8 {
            Self::warn()
        } else {
            Self::ok()
        }
    }
}
