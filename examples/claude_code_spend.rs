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
use std::path::{Path, PathBuf};

use llm_cost_dashboard::ingest::Ingester;

fn jsonl_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            jsonl_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "jsonl") {
            out.push(p);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = match std::env::args_os().nth(1) {
        Some(d) => PathBuf::from(d),
        None => {
            let home = std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))
                .ok_or("no home directory; pass the projects directory")?;
            PathBuf::from(home).join(".claude").join("projects")
        }
    };
    let mut files = Vec::new();
    jsonl_files(&dir, &mut files);
    if files.is_empty() {
        println!("No .jsonl session logs under {}", dir.display());
        return Ok(());
    }

    // One ingester for every file, so a request repeated across files
    // (resumed sessions) is counted once.
    let mut ing = Ingester::new();
    let mut rejected = 0;
    for f in &files {
        let stats = ing.ingest_reader(std::io::BufReader::new(std::fs::File::open(f)?))?;
        rejected += stats.rejected;
    }

    let ledger = ing.ledger();
    println!(
        "{} requests in {} session files, ${:.2} at API list prices ({} unreadable lines)\n",
        ledger.len(),
        files.len(),
        ledger.total_usd(),
        rejected
    );
    let mut models: Vec<_> = ledger.by_model().into_values().collect();
    models.sort_by(|a, b| b.total_cost_usd.total_cmp(&a.total_cost_usd));
    for m in &models {
        let known = llm_cost_dashboard::cost::pricing::lookup_known(&m.model).is_some();
        println!(
            "  {:<32} {:>7} requests  ${:>10.2}{}",
            m.model,
            m.request_count,
            m.total_cost_usd,
            if known { "" } else { "  (not in the price table: guessed)" }
        );
    }
    let mut days: BTreeMap<String, f64> = BTreeMap::new();
    for r in ledger.records() {
        *days.entry(r.timestamp.format("%Y-%m-%d").to_string()).or_default() += r.total_cost_usd;
    }
    println!("\nLast 7 days:");
    for (day, usd) in days.iter().rev().take(7).collect::<Vec<_>>().into_iter().rev() {
        println!("  {day}  ${usd:>9.2}");
    }
    Ok(())
}
