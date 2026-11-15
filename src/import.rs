//! Import spend from files other tools already write.
//!
//! - [`claude_code`]: Claude Code session transcripts
//!   (`~/.claude/projects/<project>/<session>.jsonl`). Every assistant turn
//!   carries a `message.usage` block with input, output, prompt-cache read
//!   and 5-minute / 1-hour cache-write token counts; each request is priced
//!   once (Claude Code writes one line per content block, and resumed
//!   sessions repeat earlier turns in a new file).
//! - [`openai_usage_csv`]: a CSV of OpenAI usage buckets, as exported from
//!   the usage dashboard or written from the Usage API
//!   (`/v1/organization/usage/completions`). Each row is priced with its
//!   cached-input and Batch API split.
//!
//! Both return an [`ImportReport`] holding a [`CostLedger`] of priced
//! records, plus counts of what was read, skipped and rejected.
//!
//! ```no_run
//! use llm_cost_dashboard::import;
//!
//! let dir = import::claude_code_projects_dir().expect("no home directory");
//! let report = import::claude_code(&dir)?;
//! println!("{} requests, ${:.2}", report.ledger.len(), report.ledger.total_usd());
//! # Ok::<(), llm_cost_dashboard::DashboardError>(())
//! ```

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use crate::cost::pricing::{cost_usage_bucket, Usage};
use crate::cost::{CostLedger, CostRecord};
use crate::error::DashboardError;
use crate::ingest::{record_for_usage, IngestStats};
use crate::log::{parse_timestamp, RequestLog};

/// What an import read and priced.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct ImportReport {
    /// The priced records.
    pub ledger: CostLedger,
    /// Totals over every file or row.
    pub stats: IngestStats,
    /// Per-file counts (one entry per transcript for [`claude_code`], none
    /// for [`openai_usage_csv`]).
    pub files: Vec<FileReport>,
    /// API requests the records stand for. Equal to the number of records
    /// for Claude Code; for OpenAI usage buckets, the sum of
    /// `num_model_requests` (a row without that column counts as one).
    pub requests: u64,
}

/// Counts for one imported file.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileReport {
    /// The file.
    pub path: PathBuf,
    /// Lines recorded, skipped and rejected in it.
    pub stats: IngestStats,
    /// USD of the records it produced.
    pub cost_usd: f64,
}

/// Claude Code's transcript directory: `$CLAUDE_CONFIG_DIR/projects` when
/// that variable is set, otherwise `~/.claude/projects` (`%USERPROFILE%` on
/// Windows). `None` when no home directory is known. The directory may not
/// exist.
pub fn claude_code_projects_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(dir).join("projects"));
    }
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|h| !h.is_empty()))
        .map(|home| PathBuf::from(home).join(".claude").join("projects"))
}

/// Every `.jsonl` file under `dir`, recursively (subagent transcripts live
/// in subdirectories), sorted by path. Unreadable directories are skipped.
pub fn claude_code_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() => walk(&p, out),
                Ok(t) if t.is_file() && p.extension().is_some_and(|x| x == "jsonl") => out.push(p),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, &mut out);
    out.sort();
    out
}

/// Price every request in the Claude Code transcripts under `dir` (see
/// [`claude_code_projects_dir`]).
///
/// Each record's `session_id` is the transcript's file name without
/// `.jsonl` (the Claude Code session id), its provider is `anthropic`, and
/// its timestamp is the line's `timestamp`. A request that appears in
/// several files (resumed sessions) is counted once, in the first file in
/// path order. Lines without usage (user turns, summaries, `<synthetic>`
/// replies, workflow journal entries) are skipped; lines that are not JSON
/// or not valid UTF-8 (a transcript cut off mid-write), or carry usage that
/// cannot be read, are counted as rejected and the import goes on.
///
/// Prices are API list prices. On a Claude subscription the same traffic
/// is not billed per token; this is what it would cost on the API.
///
/// # Errors
///
/// [`DashboardError::IoError`] when `dir` is not a readable directory or a
/// transcript cannot be opened or read (an I/O failure, not bad content).
pub fn claude_code(dir: &Path) -> Result<ImportReport, DashboardError> {
    if !dir.is_dir() {
        return Err(DashboardError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{} is not a directory", dir.display()),
        )));
    }
    let mut report = ImportReport::default();
    let mut log = RequestLog::new();
    for path in claude_code_files(dir) {
        let file = std::fs::File::open(&path)?;
        let session = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let fr = claude_code_reader(BufReader::new(file), &session, &mut log, &mut report.ledger)?;
        report.stats.recorded += fr.stats.recorded;
        report.stats.skipped += fr.stats.skipped;
        report.stats.rejected += fr.stats.rejected;
        report.files.push(FileReport { path, ..fr });
    }
    report.requests = report.ledger.len() as u64;
    Ok(report)
}

/// Read one transcript into `ledger`, tagging records with `session`.
/// `log` carries the request ids already seen, across files.
fn claude_code_reader(
    mut reader: impl BufRead,
    session: &str,
    log: &mut RequestLog,
    ledger: &mut CostLedger,
) -> Result<FileReport, DashboardError> {
    let mut stats = IngestStats::default();
    let mut cost = 0.0;
    let mut buf = Vec::new();
    loop {
        buf.clear();
        if reader.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        // Read bytes, not `lines()`: a transcript cut off inside a multibyte
        // character (Claude Code killed mid-write, or a session still being
        // written) must cost one rejected line, not the whole import.
        let Ok(line) = std::str::from_utf8(&buf) else {
            stats.rejected += 1;
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        match log.ingest_line_usage(line) {
            Ok(Some(usage)) => {
                let Some(entry) = log.all().last() else {
                    continue;
                };
                let mut rec = record_for_usage(entry, &usage);
                if !session.is_empty() {
                    rec = rec.with_session(session);
                }
                cost += rec.total_cost_usd;
                ledger.add(rec)?;
                stats.recorded += 1;
            }
            Ok(None) => stats.skipped += 1,
            Err(_) if is_bookkeeping(line) => stats.skipped += 1,
            Err(_) => stats.rejected += 1,
        }
    }
    Ok(FileReport {
        path: PathBuf::new(),
        stats,
        cost_usd: cost,
    })
}

/// A JSON object with no usage anywhere, such as the workflow journals
/// Claude Code writes under `subagents/workflows/` (`{"type":"started",...}`):
/// nothing to price, so it is skipped rather than rejected.
fn is_bookkeeping(line: &str) -> bool {
    match serde_json::from_str::<serde_json::Value>(line.trim()) {
        Ok(serde_json::Value::Object(o)) => {
            !o.contains_key("usage") && o.get("message").and_then(|m| m.get("usage")).is_none()
        }
        _ => false,
    }
}

/// Column names read from an OpenAI usage CSV, first match wins. Matching
/// ignores case and surrounding spaces.
const MODEL_COLS: &[&str] = &["model"];
const INPUT_COLS: &[&str] = &["input_tokens", "prompt_tokens"];
const OUTPUT_COLS: &[&str] = &["output_tokens", "completion_tokens"];
const CACHED_COLS: &[&str] = &["input_cached_tokens", "cached_tokens"];
const REQUESTS_COLS: &[&str] = &["num_model_requests", "requests"];
const BATCH_COLS: &[&str] = &["batch"];
const TIER_COLS: &[&str] = &["service_tier"];
const TIME_COLS: &[&str] = &[
    "start_time",
    "start_time_iso",
    "start_datetime",
    "timestamp",
    "date",
];

/// Price an OpenAI usage CSV: one row per usage bucket, with the field
/// names of the Usage API's completions result (`model`, `input_tokens`,
/// `output_tokens`, `input_cached_tokens`, `num_model_requests`, `batch`)
/// and the bucket's `start_time`.
///
/// - `input_tokens` counts cached tokens too (as the Usage API reports it);
///   `input_cached_tokens` of them are billed at the cached-input price and
///   the rest at the input price.
/// - `batch` (`true` / `false`, `1` / `0`) bills the row at Batch API
///   prices.
/// - `service_tier`: `flex` rows are billed at Batch API prices too (OpenAI
///   prices Flex processing at Batch API rates). Every other tier
///   (`default`, `auto`, `scale`, `priority`, empty) is billed at standard
///   prices, so **`priority` rows are under-priced**: the price table has
///   no Priority processing rates.
/// - `num_model_requests` is the bucket's request count. A row whose cell
///   is empty, or an export without the column, counts as one request.
/// - `start_time` may be Unix seconds or an ISO 8601 / RFC 3339 time;
///   `start_time_iso`, `start_datetime`, `timestamp` and `date` are also
///   read. Rows without a time are dated now.
/// - `prompt_tokens` / `completion_tokens` / `cached_tokens` /
///   `requests` are accepted in place of the Usage API names.
/// - Other columns (`project_id`, `api_key_id`, audio token counts, ...)
///   are ignored, so audio tokens are not priced separately.
///
/// A bucket sums many requests, so the long-context tier (GPT-5.4 and later
/// over 272,000 prompt tokens) is chosen from the average request; see
/// [`crate::cost::pricing::cost_usage_bucket`].
///
/// Each row becomes one [`CostRecord`] holding the bucket's totals, with
/// provider `openai`, latency 0 and request id `openai-usage:<row>`.
/// [`ImportReport::requests`] sums the request counts. Rows with no model
/// (an export not grouped by model) or unreadable numbers are rejected;
/// rows with no tokens are skipped.
///
/// ```
/// use llm_cost_dashboard::import::openai_usage_csv;
///
/// let csv = "start_time,model,input_tokens,input_cached_tokens,output_tokens,num_model_requests,batch\n\
///            1790000000,gpt-4o,10000,8000,1000,4,false\n";
/// let r = openai_usage_csv(csv.as_bytes())?;
/// // 2,000 x $2.50 + 8,000 x $1.25 + 1,000 x $10, per million tokens.
/// assert!((r.ledger.total_usd() - 0.025).abs() < 1e-12);
/// assert_eq!(r.requests, 4);
/// # Ok::<(), llm_cost_dashboard::DashboardError>(())
/// ```
///
/// # Errors
///
/// [`DashboardError::LogParseError`] when the header has no model, input or
/// output column, or the CSV itself is malformed (for example a row with a
/// different number of fields); [`DashboardError::IoError`] on read errors.
pub fn openai_usage_csv(reader: impl Read) -> Result<ImportReport, DashboardError> {
    let mut rdr = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(reader);
    let csv_err = |e: csv::Error| DashboardError::LogParseError(format!("OpenAI usage CSV: {e}"));
    let headers = rdr.headers().map_err(csv_err)?.clone();
    let index: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            (
                h.trim().trim_start_matches('\u{feff}').to_ascii_lowercase(),
                i,
            )
        })
        .collect();
    let col = |names: &[&str]| names.iter().find_map(|n| index.get(*n).copied());
    let need = |names: &[&str]| {
        col(names).ok_or_else(|| {
            DashboardError::LogParseError(format!(
                "OpenAI usage CSV has no {} column (header: {})",
                names[0],
                headers.iter().collect::<Vec<_>>().join(",")
            ))
        })
    };
    let (model_i, input_i, output_i) = (need(MODEL_COLS)?, need(INPUT_COLS)?, need(OUTPUT_COLS)?);
    let (cached_i, requests_i, batch_i, tier_i, time_i) = (
        col(CACHED_COLS),
        col(REQUESTS_COLS),
        col(BATCH_COLS),
        col(TIER_COLS),
        col(TIME_COLS),
    );

    let mut report = ImportReport::default();
    for (n, row) in rdr.records().enumerate() {
        let row = row.map_err(csv_err)?;
        let field = |i: Option<usize>| i.and_then(|i| row.get(i)).unwrap_or("");
        let num = |i: Option<usize>| -> Option<u64> {
            let f = field(i);
            if f.is_empty() {
                return Some(0);
            }
            f.parse::<u64>().ok().or_else(|| {
                // Spreadsheets sometimes write integers as "1234.0".
                f.parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0 && v.fract() == 0.0)
                    .map(|v| v as u64)
            })
        };
        let model = field(Some(model_i));
        let (Some(input), Some(output), Some(cached), Some(requests)) = (
            num(Some(input_i)),
            num(Some(output_i)),
            num(cached_i),
            // No column, or an empty cell: one request.
            match field(requests_i) {
                "" => Some(1),
                _ => num(requests_i),
            },
        ) else {
            report.stats.rejected += 1;
            continue;
        };
        let batch = match field(batch_i).to_ascii_lowercase().as_str() {
            "" | "false" | "0" | "no" => false,
            "true" | "1" | "yes" => true,
            _ => {
                report.stats.rejected += 1;
                continue;
            }
        };
        // Flex processing is billed at Batch API rates.
        let batch = batch || field(tier_i).eq_ignore_ascii_case("flex");
        if model.is_empty() {
            report.stats.rejected += 1;
            continue;
        }
        if input == 0 && output == 0 {
            report.stats.skipped += 1;
            continue;
        }
        let timestamp = match field(time_i) {
            "" => None,
            t => match parse_timestamp(&serde_json::Value::String(t.to_string())) {
                Some(ts) => Some(ts),
                None => {
                    report.stats.rejected += 1;
                    continue;
                }
            },
        };
        let cached = cached.min(input);
        let usage = Usage {
            input: input - cached,
            cache_read: cached,
            output,
            batch,
            ..Usage::default()
        };
        // A bucket sums many requests; pick the long-context tier from the
        // average request, not the bucket total.
        let cost = cost_usage_bucket(model, &usage, requests);
        let mut rec = CostRecord::from_cost(model.to_string(), "openai".into(), &usage, cost, 0);
        if let Some(ts) = timestamp {
            rec.timestamp = ts;
        }
        rec.request_id = format!("openai-usage:{}", n + 1);
        report.ledger.add(rec)?;
        report.stats.recorded += 1;
        report.requests += requests;
    }
    Ok(report)
}

/// [`openai_usage_csv`] on a file.
///
/// # Errors
///
/// As [`openai_usage_csv`], plus [`DashboardError::IoError`] when the file
/// cannot be opened.
pub fn openai_usage_csv_file(path: &Path) -> Result<ImportReport, DashboardError> {
    openai_usage_csv(BufReader::new(std::fs::File::open(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_requires_core_columns() {
        let err = openai_usage_csv("model,input_tokens\ngpt-4o,1\n".as_bytes()).unwrap_err();
        assert!(err.to_string().contains("output_tokens"), "{err}");
    }

    #[test]
    fn csv_rows_rejected_and_skipped() {
        let csv = "model,input_tokens,output_tokens,batch,start_time\n\
                   gpt-4o,abc,1,false,\n\
                   ,10,1,false,\n\
                   gpt-4o,10,1,maybe,\n\
                   gpt-4o,10,1,false,not-a-time\n\
                   gpt-4o,0,0,false,\n\
                   gpt-4o,1000.0,0,TRUE,2026-11-01T00:00:00Z\n";
        let r = openai_usage_csv(csv.as_bytes()).unwrap();
        assert_eq!(
            r.stats,
            IngestStats {
                recorded: 1,
                skipped: 1,
                rejected: 4
            }
        );
        let rec = &r.ledger.records()[0];
        // Batch: $1.25 per million input tokens.
        assert!((rec.total_cost_usd - 1000.0 * 1.25 / 1e6).abs() < 1e-15);
        assert_eq!(rec.timestamp.to_rfc3339(), "2026-11-01T00:00:00+00:00");
        assert_eq!(r.requests, 1);
    }

    #[test]
    fn csv_empty_request_count_is_one_request() {
        // 600,000 prompt tokens in one request is over gpt-5.4's 272,000
        // long-context threshold: $5 per million instead of $2.50.
        let csv = "model,input_tokens,output_tokens,num_model_requests\n\
                   gpt-5.4,600000,0,\n\
                   gpt-5.4,600000,0,3\n";
        let r = openai_usage_csv(csv.as_bytes()).unwrap();
        assert_eq!(r.requests, 1 + 3);
        let costs: Vec<f64> = r
            .ledger
            .records()
            .iter()
            .map(|c| c.total_cost_usd)
            .collect();
        // The second row is 200,000 per request: standard prices.
        assert!((costs[0] - 3.0).abs() < 1e-12, "{costs:?}");
        assert!((costs[1] - 1.5).abs() < 1e-12, "{costs:?}");
    }

    #[test]
    fn csv_flex_tier_bills_at_batch_rates() {
        let csv = "model,input_tokens,output_tokens,service_tier,batch\n\
                   gpt-4o,1000000,0,flex,false\n\
                   gpt-4o,1000000,0,FLEX,\n\
                   gpt-4o,1000000,0,default,false\n\
                   gpt-4o,1000000,0,priority,false\n";
        let r = openai_usage_csv(csv.as_bytes()).unwrap();
        let costs: Vec<f64> = r
            .ledger
            .records()
            .iter()
            .map(|c| c.total_cost_usd)
            .collect();
        // gpt-4o: $2.50 per million input, $1.25 through the Batch API.
        // Priority has no rate in the table and bills at standard.
        for (got, want) in costs.iter().zip([1.25, 1.25, 2.5, 2.5]) {
            assert!((got - want).abs() < 1e-12, "{costs:?}");
        }
    }

    #[test]
    fn transcript_lines_that_are_not_utf8_are_rejected() {
        let mut data = br#"{"type":"assistant","requestId":"r1","message":{"id":"m1","model":"claude-opus-4-5-20251101","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1000,"output_tokens":2000}},"timestamp":"2026-11-12T08:00:00Z"}"#.to_vec();
        // A second line cut off inside the two bytes of "é".
        data.extend_from_slice(b"\n{\"type\":\"assistant\",\"message\":{\"content\":\"caf\xc3");
        let mut ledger = CostLedger::new();
        let fr = claude_code_reader(&data[..], "s", &mut RequestLog::new(), &mut ledger).unwrap();
        assert_eq!(
            fr.stats,
            IngestStats {
                recorded: 1,
                skipped: 0,
                rejected: 1
            }
        );
        assert!((fr.cost_usd - 0.055).abs() < 1e-12, "{}", fr.cost_usd);
    }

    #[test]
    fn cached_tokens_never_exceed_input() {
        let csv = "model,input_tokens,output_tokens,input_cached_tokens\ngpt-4o,100,0,500\n";
        let r = openai_usage_csv(csv.as_bytes()).unwrap();
        let rec = &r.ledger.records()[0];
        assert_eq!((rec.input_tokens, rec.cache.cache_read_tokens), (0, 100));
    }

    #[test]
    fn claude_code_missing_dir_is_an_error() {
        assert!(claude_code(Path::new("definitely/not/a/dir")).is_err());
    }
}
