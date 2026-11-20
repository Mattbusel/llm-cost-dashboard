//! What did Claude Code cost? Prices every session log under
//! `~/.claude/projects` (or a directory you pass) at API list prices,
//! including prompt-cache reads and writes, and prints a per-model and
//! per-day summary.
//!
//! ```text
//! cargo run --example claude_code_spend
//! cargo run --example claude_code_spend -- /path/to/.claude/projects
//! ```
//!
//! Subscription plans are not billed per token; this shows what the same
//! traffic would cost on the API.

use std::collections::BTreeMap;
use std::path::PathBuf;

use llm_cost_dashboard::cost::pricing::{price_source, PriceSource};
use llm_cost_dashboard::import;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = match std::env::args_os().nth(1) {
        Some(d) => PathBuf::from(d),
        None => import::claude_code_projects_dir()
            .ok_or("no home directory; pass the projects directory")?,
    };
    if import::claude_code_files(&dir).is_empty() {
        println!("No .jsonl session logs under {}", dir.display());
        return Ok(());
    }

    // One import over every file, so a request repeated across files
    // (resumed sessions) is counted once.
    let report = import::claude_code(&dir)?;
    let (files, rejected) = (report.files.len(), report.stats.rejected);
    let ledger = &report.ledger;
    println!(
        "{} requests in {} session files, ${:.2} at API list prices ({} unreadable lines)\n",
        ledger.len(),
        files,
        ledger.total_usd(),
        rejected
    );
    let mut models: Vec<_> = ledger.by_model().into_values().collect();
    models.sort_by(|a, b| b.total_cost_usd.total_cmp(&a.total_cost_usd));
    for m in &models {
        let known = price_source(&m.model) != PriceSource::Fallback;
        println!(
            "  {:<32} {:>7} requests  ${:>10.2}{}",
            m.model,
            m.request_count,
            m.total_cost_usd,
            if known {
                ""
            } else {
                "  (not in the price table: guessed)"
            }
        );
    }
    let mut days: BTreeMap<String, f64> = BTreeMap::new();
    for r in ledger.records() {
        *days
            .entry(r.timestamp.format("%Y-%m-%d").to_string())
            .or_default() += r.total_cost_usd;
    }
    println!("\nLast 7 days:");
    for (day, usd) in days
        .iter()
        .rev()
        .take(7)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        println!("  {day}  ${usd:>9.2}");
    }
    Ok(())
}
