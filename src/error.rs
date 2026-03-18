//! Unified error type for the LLM cost dashboard.
//!
//! All fallible operations in this crate return [`DashboardError`].  The TUI
//! event loop catches every error at the boundary and degrades gracefully
//! rather than panicking.

/// The single error type used throughout the dashboard.
///
/// Each variant corresponds to one failure domain.  Callers should match on
/// the variant they care about and treat the rest as opaque display strings.
#[derive(Debug, thiserror::Error)]
pub enum DashboardError {
    /// A cost-ledger invariant was violated (e.g. negative cost).
    #[error("Cost ledger error: {0}")]
    Ledger(String),

    /// The hard monthly budget limit has been exceeded.
    #[error("Budget exceeded: spent ${spent:.4} of ${limit:.4} limit")]
    BudgetExceeded {
        /// Amount spent so far in USD.
        spent: f64,
        /// Configured hard limit in USD.
        limit: f64,
    },

    /// Spend has crossed the soft alert threshold but not the hard limit.
    #[error("Budget alert: {pct:.1}% of ${limit:.4} consumed")]
    BudgetAlert {
        /// Percentage of the limit consumed (0-100).
        pct: f64,
        /// Configured hard limit in USD.
        limit: f64,
    },

    /// A model name was not found in the pricing table; fallback pricing was used.
    #[error("Unknown model '{0}' - using fallback pricing")]
    UnknownModel(String),

    /// A log line could not be parsed as a valid [`crate::log::IncomingRecord`].
    ///
    /// The dashboard degrades gracefully on this error: the malformed line is
    /// skipped and an error is surfaced in the UI error pane rather than
    /// crashing the process.
    #[error("Log parse error: {0}")]
    LogParseError(String),

    /// A wrapped [`std::io::Error`] (file I/O, terminal setup, etc.).
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// A pricing value is invalid (e.g. negative, NaN, or infinite).
    #[error("Invalid pricing: {0}")]
    InvalidPricing(String),

    /// A JSON serialization or deserialization failure.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// A terminal / ratatui operation failed.
    #[error("Terminal error: {0}")]
    Terminal(String),
}

// Keep legacy aliases so existing call-sites that use LogParse / Io / Json
// do not need to change.  These are not public variants; callers should use
// the canonical names above.
impl DashboardError {
    /// Construct a [`DashboardError::LogParseError`] (legacy helper).
    #[allow(dead_code)]
    pub(crate) fn log_parse(msg: impl Into<String>) -> Self {
        Self::LogParseError(msg.into())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_ledger_error_display() {
        let e = DashboardError::Ledger("negative cost".into());
        assert!(e.to_string().contains("negative cost"));
    }

    #[test]
    fn test_budget_exceeded_display() {
        let e = DashboardError::BudgetExceeded {
            spent: 12.5,
            limit: 10.0,
        };
        assert!(e.to_string().contains("12.5000"));
        assert!(e.to_string().contains("10.0000"));
    }

    #[test]
    fn test_budget_alert_display() {
        let e = DashboardError::BudgetAlert {
            pct: 85.0,
            limit: 10.0,
        };
        assert!(e.to_string().contains("85.0"));
    }

    #[test]
    fn test_unknown_model_display() {
        let e = DashboardError::UnknownModel("my-model".into());
        assert!(e.to_string().contains("my-model"));
    }

    #[test]
    fn test_log_parse_error_display() {
        let e = DashboardError::LogParseError("bad JSON".into());
        assert!(e.to_string().contains("bad JSON"));
    }

    #[test]
    fn test_invalid_pricing_display() {
        let e = DashboardError::InvalidPricing("negative rate".into());
        assert!(e.to_string().contains("negative rate"));
    }

    #[test]
    fn test_terminal_error_display() {
        let e = DashboardError::Terminal("failed".into());
        assert!(e.to_string().contains("failed"));
    }

    #[test]
    fn test_io_error_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let e: DashboardError = io_err.into();
        assert!(e.to_string().contains("IO error"));
    }

    #[test]
    fn test_serialization_error_from() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let e: DashboardError = json_err.into();
        assert!(e.to_string().contains("Serialization"));
    }

    #[test]
    fn test_log_parse_helper() {
        let e = DashboardError::log_parse("oops");
        assert!(matches!(e, DashboardError::LogParseError(_)));
    }
}
