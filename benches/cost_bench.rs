//! Criterion benchmarks for the hot path: pricing lookup and ledger aggregation.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llm_cost_dashboard::cost::{pricing, CostLedger, CostRecord};

/// Benchmark the pricing table lookup (tight inner loop in every request).
fn bench_pricing_lookup(c: &mut Criterion) {
    c.bench_function("pricing::lookup known", |b| {
        b.iter(|| pricing::lookup(black_box("claude-sonnet-4-6")))
    });

    c.bench_function("pricing::lookup unknown (fallback)", |b| {
        b.iter(|| pricing::lookup(black_box("unknown-model-xyz")))
    });
}

/// Benchmark ingesting 1 000 records and then computing by-model aggregation.
fn bench_ledger_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ledger");

    group.bench_function("add 1000 records", |b| {
        b.iter(|| {
            let mut ledger = CostLedger::new();
            for i in 0u64..1_000 {
                let _ = ledger.add(CostRecord::new(
                    "gpt-4o-mini",
                    "openai",
                    black_box(i * 10),
                    black_box(i * 5),
                    black_box(i),
                ));
            }
            ledger
        })
    });

    group.bench_function("by_model on 1000 records", |b| {
        let mut ledger = CostLedger::new();
        for i in 0u64..1_000 {
            let model = if i % 2 == 0 {
                "gpt-4o-mini"
            } else {
                "claude-sonnet-4-6"
            };
            let _ = ledger.add(CostRecord::new(model, "openai", i * 10, i * 5, i));
        }
        b.iter(|| ledger.by_model())
    });

    group.bench_function("sparkline_data 60", |b| {
        let mut ledger = CostLedger::new();
        for i in 0u64..200 {
            let _ = ledger.add(CostRecord::new("gpt-4o-mini", "openai", i, i / 2, i));
        }
        b.iter(|| ledger.sparkline_data(black_box(60)))
    });

    group.finish();
}

criterion_group!(benches, bench_pricing_lookup, bench_ledger_aggregation);
criterion_main!(benches);
