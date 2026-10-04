//! The README's 10-line example. `cargo run --example quickstart`

use llm_cost_dashboard::ingest::Ingester;

fn main() -> Result<(), llm_cost_dashboard::DashboardError> {
    let mut costs = Ingester::new();
    // Any mix of your own log lines and saved OpenAI / Anthropic responses.
    costs.ingest_line(r#"{"model":"gpt-4o-mini","input_tokens":5200,"output_tokens":400}"#)?;
    costs.ingest_line(r#"{"model":"claude-sonnet-4-6","usage":{"input_tokens":900,"output_tokens":350,"cache_read_input_tokens":40000}}"#)?;
    for (model, s) in costs.ledger().by_model() {
        println!("{model}: ${:.6} over {} call(s)", s.total_cost_usd, s.request_count);
    }
    println!("total: ${:.6}", costs.ledger().total_usd());
    Ok(())
}
