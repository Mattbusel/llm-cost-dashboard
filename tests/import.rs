//! Importers against fixture logs: Claude Code transcripts and an OpenAI
//! usage CSV export. Expected dollars are worked out by hand from the
//! `llm-prices` table (USD per million tokens) in each test.

use std::path::{Path, PathBuf};

use llm_cost_dashboard::cost::pricing::{
    cost_usage, llm_prices_row, price_of, price_source, PriceSource, Usage,
};
use llm_cost_dashboard::import;
use llm_cost_dashboard::ingest::{IngestStats, Ingester};

fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(rel)
}

/// (recorded, skipped, rejected)
fn counts(s: &IngestStats) -> (usize, usize, usize) {
    (s.recorded, s.skipped, s.rejected)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn claude_code_transcripts() {
    let report = import::claude_code(&fixture("claude_code/projects")).unwrap();

    // Four files: three transcripts (a subagent one in a subdirectory) and
    // a workflow journal.
    assert_eq!(report.files.len(), 4);
    // 4 billable requests. Skipped: a summary, user turns and tool results,
    // the second content block of msg_011, msg_011 again in the resumed
    // session, a <synthetic> reply and two workflow journal entries.
    // Rejected: one truncated line.
    assert_eq!(counts(&report.stats), (4, 9, 1));
    assert_eq!(report.requests, 4);

    let recs = report.ledger.records();
    let by_model = |m: &str| recs.iter().find(|r| r.model == m).unwrap();

    // Claude Haiku 5.5 subagent call: 110,000-token prompt is over the
    // 100,000 threshold, so all of it bills at long-context prices:
    // 20,000 x $0.50 + 90,000 x $0.05 (cache read) + 1,000 x $2.50.
    let haiku55 = by_model("claude-haiku-5-5");
    assert!(
        close(haiku55.total_cost_usd, 0.017),
        "{}",
        haiku55.total_cost_usd
    );
    assert_eq!(haiku55.session_id.as_deref(), Some("agent-a1"));

    // Sonnet 4.5, 1-hour cache write: 4 x $3 + 12,000 x $6 + 250 x $15.
    let sonnet = by_model("claude-sonnet-4-5-20250929");
    assert!(
        close(sonnet.total_cost_usd, 0.075_762),
        "{}",
        sonnet.total_cost_usd
    );
    assert_eq!(sonnet.cache.cache_write_tokens, 12_000);
    assert!(close(sonnet.cache.cache_write_cost_usd, 0.072));
    assert_eq!(sonnet.session_id.as_deref(), Some("0a1b2c3d-session-one"));
    assert_eq!(sonnet.provider, "anthropic");
    assert_eq!(sonnet.timestamp.to_rfc3339(), "2026-11-10T09:00:05+00:00");

    // Haiku 4.5, cache writes without a 5m/1h split count as 5-minute:
    // 10 x $1 + 12,000 x $0.10 + 500 x $1.25 + 80 x $5.
    let haiku45 = by_model("claude-haiku-4-5-20251001");
    assert!(
        close(haiku45.total_cost_usd, 0.002_235),
        "{}",
        haiku45.total_cost_usd
    );
    assert!(close(haiku45.cache.cache_read_cost_usd, 0.0012));

    // Opus 4.5 in the resumed session: 1,000 x $5 + 50,000 x $0.50 + 2,000 x $25.
    let opus = by_model("claude-opus-4-5-20251101");
    assert!(close(opus.total_cost_usd, 0.08), "{}", opus.total_cost_usd);
    assert_eq!(opus.session_id.as_deref(), Some("9f8e7d6c-session-two"));

    assert!(
        close(report.ledger.total_usd(), 0.174_997),
        "{}",
        report.ledger.total_usd()
    );

    let one = report
        .files
        .iter()
        .find(|f| f.path.ends_with("0a1b2c3d-session-one.jsonl"))
        .unwrap();
    assert_eq!(counts(&one.stats), (2, 5, 1));
    assert!(close(one.cost_usd, 0.075_762 + 0.002_235));
}

/// A transcript with a line that is not UTF-8 and a last line cut off
/// inside a multibyte character (Claude Code killed mid-write, or a session
/// still being written): both lines are rejected and the rest is priced.
#[test]
fn claude_code_transcript_cut_off_mid_character() {
    let report = import::claude_code(&fixture("claude_code_truncated/projects")).unwrap();
    assert_eq!(report.files.len(), 1);
    assert_eq!(counts(&report.stats), (2, 0, 2));
    assert_eq!(report.requests, 2);
    // Opus 4.5: 1,000 x $5 + 2,000 x $25. Haiku 4.5: 1,000 x $1 + 100 x $5.
    assert!(
        close(report.ledger.total_usd(), 0.055 + 0.0015),
        "{}",
        report.ledger.total_usd()
    );
    let models: Vec<_> = report
        .ledger
        .records()
        .iter()
        .map(|r| r.model.as_str())
        .collect();
    assert_eq!(
        models,
        ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"]
    );
}

#[test]
fn claude_code_files_are_found_recursively_and_sorted() {
    let files = import::claude_code_files(&fixture("claude_code/projects"));
    let names: Vec<_> = files
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        [
            "agent-a1.jsonl",
            "journal.jsonl",
            "0a1b2c3d-session-one.jsonl",
            "9f8e7d6c-session-two.jsonl"
        ]
    );
}

#[test]
fn openai_usage_csv_export() {
    let report = import::openai_usage_csv_file(&fixture("openai_usage.csv")).unwrap();
    // One row without a model (export not grouped by model) is rejected;
    // one all-zero row is skipped.
    assert_eq!(counts(&report.stats), (3, 1, 1));
    assert_eq!(report.requests, 120 + 1000 + 50);

    let recs = report.ledger.records();

    // gpt-4o, 400,000 of 600,000 input tokens cached:
    // 200,000 x $2.50 + 400,000 x $1.25 + 90,000 x $10.
    assert_eq!(recs[0].model, "gpt-4o-2024-08-06");
    assert_eq!(recs[0].provider, "openai");
    assert_eq!(
        (recs[0].input_tokens, recs[0].cache.cache_read_tokens),
        (200_000, 400_000)
    );
    assert!(
        close(recs[0].total_cost_usd, 1.90),
        "{}",
        recs[0].total_cost_usd
    );
    assert_eq!(recs[0].timestamp.to_rfc3339(), "2026-11-05T00:00:00+00:00");
    assert_eq!(recs[0].request_id, "openai-usage:1");

    // gpt-4o-mini through the Batch API: 2M x $0.075 + 500,000 x $0.30.
    assert!(
        close(recs[1].total_cost_usd, 0.30),
        "{}",
        recs[1].total_cost_usd
    );

    // gpt-5.4: 6M prompt tokens over 50 requests is 120,000 per request,
    // under the 272,000 long-context threshold, so standard prices:
    // 4M x $2.50 + 1M x $0.25 + 100,000 x $15.
    assert!(
        close(recs[2].total_cost_usd, 11.75),
        "{}",
        recs[2].total_cost_usd
    );

    assert!(
        close(report.ledger.total_usd(), 13.95),
        "{}",
        report.ledger.total_usd()
    );
}

#[test]
fn openai_csv_bad_header_is_an_error() {
    let err = import::openai_usage_csv("model,tokens\ngpt-4o,5\n".as_bytes()).unwrap_err();
    assert!(err.to_string().contains("input_tokens"), "{err}");
}

/// The cache-read prices that were stale before 1.4.0 now come from
/// `llm-prices`, together with its source page and check date.
#[test]
fn claude_cache_prices_come_from_llm_prices() {
    for (model, read) in [
        ("claude-sonnet-5-5", 0.10),
        ("claude-haiku-5-5", 0.01),
        ("claude-sonnet-5", 0.20),
    ] {
        assert_eq!(price_source(model), PriceSource::LlmPrices, "{model}");
        let p = price_of(model).unwrap();
        assert!(
            close(p.cache_read_rate(), read),
            "{model}: {}",
            p.cache_read_rate()
        );
        let row = llm_prices_row(model).unwrap();
        assert!(row.source_url.starts_with("https://platform.claude.com/"));
        assert_eq!(row.cache_read, Some(read));
    }
    // A model llm-prices does not list still has its built-in price.
    assert_eq!(price_source("mistral-large-2411"), PriceSource::BuiltIn);
    assert_eq!(price_source("no-such-model"), PriceSource::Fallback);
    // gpt-4o-2024-05-13 has its own (higher) price; the built-in table
    // used to fall back to undated gpt-4o.
    assert_eq!(
        price_of("gpt-4o-2024-05-13").map(|p| p.input_per_1m),
        Some(5.0)
    );
}

#[test]
fn batch_flag_on_log_lines() {
    let mut ing = Ingester::new();
    ing.ingest_line(r#"{"model":"claude-sonnet-4-6","input_tokens":1000000,"output_tokens":1000000,"batch":true}"#)
        .unwrap();
    ing.ingest_line(
        r#"{"model":"claude-sonnet-4-6","input_tokens":1000000,"output_tokens":1000000}"#,
    )
    .unwrap();
    let recs = ing.ledger().records();
    // Batch API: $1.50 / $7.50 instead of $3 / $15.
    assert!(
        close(recs[0].total_cost_usd, 9.0),
        "{}",
        recs[0].total_cost_usd
    );
    assert!(
        close(recs[1].total_cost_usd, 18.0),
        "{}",
        recs[1].total_cost_usd
    );
}

#[test]
fn batch_without_listed_batch_price_bills_standard() {
    // gpt-3.5-turbo lists no batch price in llm-prices.
    let u = Usage {
        input: 1_000_000,
        output: 0,
        batch: true,
        ..Usage::default()
    };
    let c = cost_usage("gpt-3.5-turbo", &u);
    assert!(!c.batch_priced);
    assert!(close(c.total, 0.5));
    // Built-in row with no batch price: same rule.
    let c = cost_usage("mistral-nemo", &u);
    assert!(!c.batch_priced && close(c.total, 0.15));
}

#[test]
fn openai_cache_writes_bill_as_input() {
    // OpenAI lists no 5-minute cache-write price; writes cost the input rate.
    let u = Usage {
        cache_write_5m: 1_000_000,
        ..Usage::default()
    };
    assert!(close(cost_usage("gpt-4o", &u).total, 2.5));
}
