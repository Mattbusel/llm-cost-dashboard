//! Turn log lines into priced [`CostRecord`]s without the dashboard.
//!
//! [`Ingester`] is the library path the `llm-dash` binary uses: it parses
//! each line (plain NDJSON, saved OpenAI or Anthropic responses, or Claude
//! Code session logs), skips repeats and non-billable lines, prices the
//! request including prompt-cache tokens, and keeps the result in a
//! [`CostLedger`].
//!
//! ```
//! use llm_cost_dashboard::ingest::Ingester;
//!
//! let mut ing = Ingester::new();
//! ing.ingest_line(r#"{"model":"gpt-4o-mini","input_tokens":1000,"output_tokens":200}"#)?;
//! ing.ingest_line(r#"{"model":"claude-haiku-4-5","usage":{"input_tokens":900,"output_tokens":120}}"#)?;
//! assert_eq!(ing.ledger().len(), 2);
//! println!("spent ${:.6}", ing.ledger().total_usd());
//! # Ok::<(), llm_cost_dashboard::DashboardError>(())
//! ```

use std::io::BufRead;

use crate::cost::{CostLedger, CostRecord};
use crate::error::DashboardError;
use crate::log::{CacheTokens, LogEntry, RequestLog};

/// Build the priced record for one parsed log entry.
pub fn record_for(entry: &LogEntry, cache: CacheTokens) -> CostRecord {
    let mut rec = CostRecord::new(
        &entry.model,
        &entry.provider,
        entry.input_tokens,
        entry.output_tokens,
        entry.latency_ms,
    );
    if !cache.is_empty() {
        rec = rec.with_cache_tiers(cache.read, cache.write_5m, cache.write_1h);
    }
    rec.timestamp = entry.timestamp;
    rec
}

/// Counts from [`Ingester::ingest_reader`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct IngestStats {
    /// Lines that became a priced record.
    pub recorded: usize,
    /// Valid lines that were not new requests (repeats, user turns).
    pub skipped: usize,
    /// Lines that could not be parsed (blank lines are not counted).
    pub rejected: usize,
}

/// Parses log lines into a [`CostLedger`]. See the [module docs](self).
#[derive(Debug, Default)]
pub struct Ingester {
    log: RequestLog,
    ledger: CostLedger,
}

impl Ingester {
    /// An empty ingester.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse one line. `Ok(true)` when it added a record, `Ok(false)` when
    /// it was valid but skipped (a repeated request id, or a Claude Code
    /// line without usage).
    ///
    /// # Errors
    ///
    /// [`DashboardError::LogParseError`] for malformed lines; nothing is
    /// recorded and the ingester can keep going.
    pub fn ingest_line(&mut self, line: &str) -> Result<bool, DashboardError> {
        let Some(cache) = self.log.ingest_line_detailed(line)? else {
            return Ok(false);
        };
        let Some(entry) = self.log.all().last() else {
            return Ok(false);
        };
        let rec = record_for(entry, cache);
        self.ledger.add(rec)?;
        Ok(true)
    }

    /// Read every line from `reader`; malformed lines are counted, not fatal.
    ///
    /// # Errors
    ///
    /// Only I/O errors from the reader.
    pub fn ingest_reader(&mut self, reader: impl BufRead) -> Result<IngestStats, DashboardError> {
        let mut stats = IngestStats::default();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match self.ingest_line(&line) {
                Ok(true) => stats.recorded += 1,
                Ok(false) => stats.skipped += 1,
                Err(_) => stats.rejected += 1,
            }
        }
        Ok(stats)
    }

    /// The priced records so far.
    pub fn ledger(&self) -> &CostLedger {
        &self.ledger
    }

    /// The parsed log entries so far.
    pub fn log(&self) -> &RequestLog {
        &self.log
    }

    /// Take the ledger out, leaving the ingester's duplicate tracking intact.
    pub fn into_ledger(self) -> CostLedger {
        self.ledger
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_counts_and_cache_pricing() {
        let text = concat!(
            r#"{"model":"claude-haiku-4-5","input_tokens":1000000,"output_tokens":0}"#, "\n",
            "\n",
            "not json\n",
            r#"{"sessionId":"s","requestId":"r","message":{"id":"m","model":"claude-haiku-4-5","usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":1000000,"cache_creation_input_tokens":1000000,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":1000000}}}}"#, "\n",
            r#"{"sessionId":"s","requestId":"r","message":{"id":"m","model":"claude-haiku-4-5","usage":{"input_tokens":0,"output_tokens":0}}}"#, "\n",
        );
        let mut ing = Ingester::new();
        let stats = ing.ingest_reader(text.as_bytes()).unwrap();
        assert_eq!(stats, IngestStats { recorded: 2, skipped: 1, rejected: 1 });
        let recs = ing.ledger().records();
        assert!((recs[0].total_cost_usd - 1.0).abs() < 1e-9);
        // 1M cache reads at $0.10 plus 1M one-hour cache writes at $2.00.
        assert!((recs[1].total_cost_usd - 2.10).abs() < 1e-9, "{}", recs[1].total_cost_usd);
    }
}
