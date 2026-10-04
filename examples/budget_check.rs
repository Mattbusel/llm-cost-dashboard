//! A CI-style budget gate: price an NDJSON log of LLM calls, forecast the
//! next 30 days, and exit with status 1 when the forecast is over budget.
//!
//! ```text
//! cargo run --example budget_check -- calls.ndjson 50
//! ```
//!
//! Without arguments it uses a small built-in log so you can see the output.

use llm_cost_dashboard::forecast::CostForecaster;
use llm_cost_dashboard::ingest::Ingester;

const SAMPLE: &str = r#"{"model":"gpt-4o-mini","input_tokens":12000,"output_tokens":900,"timestamp":"2026-09-27T09:00:00Z"}
{"model":"claude-haiku-4-5","usage":{"input_tokens":20000,"output_tokens":1500},"timestamp":"2026-09-27T15:00:00Z"}
{"model":"gpt-4o","usage":{"prompt_tokens":8000,"completion_tokens":700,"prompt_tokens_details":{"cached_tokens":6000}},"created":1790553600}
{"model":"gpt-4o-mini","input_tokens":15000,"output_tokens":1200,"timestamp":"2026-09-29T10:00:00Z"}
{"model":"claude-haiku-4-5","usage":{"input_tokens":25000,"output_tokens":2100},"timestamp":"2026-09-30T11:00:00Z"}"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let text = match args.first() {
        Some(path) => std::fs::read_to_string(path)?,
        None => SAMPLE.to_string(),
    };
    let budget: f64 = args.get(1).map(|b| b.parse()).transpose()?.unwrap_or(10.0);

    let mut ing = Ingester::new();
    let stats = ing.ingest_reader(text.as_bytes())?;
    let ledger = ing.ledger();
    println!(
        "{} calls priced, {} skipped, {} unreadable: ${:.4} so far",
        stats.recorded,
        stats.skipped,
        stats.rejected,
        ledger.total_usd()
    );

    let mut records = ledger.records().to_vec();
    records.sort_by_key(|r| r.timestamp);
    let mut f = CostForecaster::new();
    let mut cumulative = 0.0;
    for r in &records {
        cumulative += r.total_cost_usd;
        f.record(r.timestamp.timestamp_millis() as f64 / 1000.0, cumulative);
    }
    match f.forecast(Some(budget)) {
        Some(fc) => {
            println!("Forecast next 30 days: ${:.2} (budget ${budget:.2})", fc.next_month_usd);
            if fc.next_month_usd > budget {
                println!("OVER BUDGET");
                std::process::exit(1);
            }
            println!("within budget");
        }
        None => println!("Not enough distinct timestamps to forecast (need 3)."),
    }
    Ok(())
}
