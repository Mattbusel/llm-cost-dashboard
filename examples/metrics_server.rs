//! Expose LLM spend to Prometheus or Grafana: record calls into a shared
//! ledger from your application and serve `/metrics` plus the JSON and CSV
//! endpoints on 127.0.0.1:9898.
//!
//! ```text
//! cargo run --example metrics_server --features server
//! curl localhost:9898/metrics
//! ```
//!
//! Stops by itself after 60 seconds so it can run in scripts; drop the
//! timeout in your own code.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use llm_cost_dashboard::{CostLedger, CostRecord};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ledger = Arc::new(Mutex::new(CostLedger::new()));

    // Your application records each call as it completes.
    {
        let mut l = ledger.lock().map_err(|e| e.to_string())?;
        l.add(CostRecord::new("gpt-4o-mini", "openai", 1200, 300, 410))?;
        l.add(CostRecord::new("claude-sonnet-4-6", "anthropic", 5000, 800, 1900).with_cache(40_000, 0))?;
    }

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 9898));
    println!("serving http://{addr}/metrics for 60 s");
    let server = llm_cost_dashboard::api::serve_on(Arc::clone(&ledger), addr);
    match tokio::time::timeout(Duration::from_secs(60), server).await {
        Ok(result) => result?,
        Err(_) => println!("done"),
    }
    Ok(())
}
