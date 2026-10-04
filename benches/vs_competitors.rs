//! Same inputs through this crate and llm-cost-cap 0.1 (the other
//! maintained Rust crate that prices LLM calls from a built-in table).
//!
//! Run: `cargo bench --bench vs_competitors`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llm_cost_dashboard::cost::pricing::compute_cost;
use llm_cost_dashboard::ingest::Ingester;

/// Models both crates price.
const MODELS: [&str; 5] = ["gpt-5", "gpt-5-mini", "gemini-2.5-pro", "claude-sonnet-4-6", "claude-sonnet-4-5"];

fn price_one_call(c: &mut Criterion) {
    let mut g = c.benchmark_group("price_one_call");
    g.bench_function("llm-cost-dashboard compute_cost", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = i.wrapping_add(1);
            compute_cost(black_box(MODELS[i % 5]), black_box(12_000), black_box(800))
        })
    });
    let cap = llm_cost_cap::CostCap::new(1_000.0);
    g.bench_function("llm-cost-cap CostCap::estimate", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = i.wrapping_add(1);
            cap.estimate(black_box(MODELS[i % 5]), black_box(12_000), black_box(800))
                .map(|e| e.total_usd)
        })
    });
    g.finish();
}

fn ingest_lines(c: &mut Criterion) {
    // 10 000 log lines in three shapes: flat, OpenAI response, Claude Code.
    let mut lines = Vec::new();
    for i in 0..10_000u32 {
        lines.push(match i % 3 {
            0 => format!(r#"{{"model":"gpt-4o-mini","input_tokens":{},"output_tokens":120,"latency_ms":300,"timestamp":{}}}"#, 500 + i, 1_790_000_000 + i),
            1 => format!(r#"{{"model":"gpt-4o","created":{},"usage":{{"prompt_tokens":{},"completion_tokens":50,"prompt_tokens_details":{{"cached_tokens":100}}}}}}"#, 1_790_000_000 + i, 900 + i),
            _ => format!(r#"{{"sessionId":"s","requestId":"r{i}","type":"assistant","timestamp":"2026-10-02T17:56:17Z","message":{{"id":"m{i}","model":"claude-sonnet-4-6","usage":{{"input_tokens":3,"output_tokens":400,"cache_read_input_tokens":30000,"cache_creation_input_tokens":2000}}}}}}"#),
        });
    }
    let text = lines.join("\n");
    let mut g = c.benchmark_group("ingest");
    g.throughput(criterion::Throughput::Elements(10_000));
    g.sample_size(20);
    g.bench_function("llm-cost-dashboard Ingester 10k lines", |b| {
        b.iter(|| {
            let mut ing = Ingester::new();
            ing.ingest_reader(black_box(text.as_bytes())).map(|s| s.recorded)
        })
    });
    g.finish();
}

criterion_group!(benches, price_one_call, ingest_lines);
criterion_main!(benches);
