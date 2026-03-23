//! # Cost Allocation
//!
//! Allocates shared LLM model costs across teams or projects using
//! configurable allocation strategies.
//!
//! ## Quick Start
//!
//! ```rust
//! use llm_cost_dashboard::cost_allocation::{
//!     AllocationMethod, CostAllocator, CostPool, ConsumerUsage,
//! };
//!
//! let pool = CostPool {
//!     pool_id: "prod-pool".to_string(),
//!     total_cost_usd: 100.0,
//!     period_start: 1_700_000_000,
//!     period_end: 1_700_086_400,
//! };
//!
//! let consumers = vec![
//!     ConsumerUsage { consumer_id: "team-a".to_string(), tokens_used: 1000, requests: 10, priority_weight: 1.0 },
//!     ConsumerUsage { consumer_id: "team-b".to_string(), tokens_used: 3000, requests: 30, priority_weight: 1.0 },
//! ];
//!
//! let allocator = CostAllocator::new();
//! let results = allocator.allocate(&pool, &consumers, &AllocationMethod::ProportionalUsage);
//! assert!(allocator.validate_allocation(&results, &pool));
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AllocationMethod
// ---------------------------------------------------------------------------

/// Strategy used to divide a shared cost pool among consumers.
#[derive(Debug, Clone)]
pub enum AllocationMethod {
    /// Each consumer's share is proportional to their `tokens_used`.
    ProportionalUsage,
    /// Cost is divided equally among all consumers regardless of usage.
    EqualSplit,
    /// Cost is divided according to the explicit weight map
    /// `consumer_id -> weight`.  Consumers absent from the map receive
    /// weight `0.0` and are allocated no cost.
    WeightedByPriority(HashMap<String, f64>),
    /// Activity-based costing: share is proportional to `requests` (not tokens).
    ActivityBased,
}

// ---------------------------------------------------------------------------
// CostPool
// ---------------------------------------------------------------------------

/// A shared pool of LLM cost to be allocated.
#[derive(Debug, Clone)]
pub struct CostPool {
    /// Unique identifier for this cost pool (e.g. `"prod-gpt4-pool"`).
    pub pool_id: String,
    /// Total shared cost in USD for the period.
    pub total_cost_usd: f64,
    /// Unix timestamp (seconds) of period start.
    pub period_start: u64,
    /// Unix timestamp (seconds) of period end.
    pub period_end: u64,
}

// ---------------------------------------------------------------------------
// ConsumerUsage
// ---------------------------------------------------------------------------

/// Usage record for a single consumer (team / project / service).
#[derive(Debug, Clone)]
pub struct ConsumerUsage {
    /// Unique identifier for this consumer.
    pub consumer_id: String,
    /// Total tokens consumed in the period.
    pub tokens_used: u64,
    /// Total API requests made in the period.
    pub requests: u64,
    /// Relative priority weight (used by [`AllocationMethod::WeightedByPriority`]
    /// as a default when the consumer is absent from the weight map).
    pub priority_weight: f64,
}

// ---------------------------------------------------------------------------
// AllocationResult
// ---------------------------------------------------------------------------

/// The cost allocation outcome for a single consumer.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Consumer identifier (mirrors [`ConsumerUsage::consumer_id`]).
    pub consumer_id: String,
    /// USD amount allocated to this consumer.
    pub allocated_cost_usd: f64,
    /// Fraction of the total pool allocated (in `[0.0, 1.0]`).
    pub fraction: f64,
    /// Token count from the original usage record.
    pub tokens_used: u64,
}

// ---------------------------------------------------------------------------
// CostAllocator
// ---------------------------------------------------------------------------

/// Allocates cost pools across consumer groups.
pub struct CostAllocator;

impl Default for CostAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostAllocator {
    /// Create a new allocator.
    pub fn new() -> Self {
        Self
    }

    /// Allocate `pool` costs across `consumers` using `method`.
    ///
    /// Returns one [`AllocationResult`] per consumer in the same order.
    /// When `consumers` is empty the result is also empty.
    pub fn allocate(
        &self,
        pool: &CostPool,
        consumers: &[ConsumerUsage],
        method: &AllocationMethod,
    ) -> Vec<AllocationResult> {
        if consumers.is_empty() {
            return Vec::new();
        }

        let fractions = self.compute_fractions(consumers, method);

        consumers
            .iter()
            .zip(fractions.iter())
            .map(|(c, &frac)| AllocationResult {
                consumer_id: c.consumer_id.clone(),
                allocated_cost_usd: pool.total_cost_usd * frac,
                fraction: frac,
                tokens_used: c.tokens_used,
            })
            .collect()
    }

    /// Produce a formatted chargeback report table.
    ///
    /// Columns: Consumer | Tokens | Fraction | Allocated ($)
    pub fn chargeback_report(&self, results: &[AllocationResult]) -> String {
        if results.is_empty() {
            return "No allocation results to report.\n".to_string();
        }

        let mut out = String::new();
        let header = format!(
            "{:<24} {:>12} {:>10} {:>14}\n",
            "Consumer", "Tokens", "Fraction", "Allocated ($)"
        );
        let separator = "-".repeat(header.len() - 1) + "\n";

        out.push_str(&separator);
        out.push_str(&header);
        out.push_str(&separator);

        for r in results {
            out.push_str(&format!(
                "{:<24} {:>12} {:>9.2}% {:>13.4}\n",
                r.consumer_id,
                r.tokens_used,
                r.fraction * 100.0,
                r.allocated_cost_usd,
            ));
        }

        out.push_str(&separator);

        // Totals row
        let total_cost: f64 = results.iter().map(|r| r.allocated_cost_usd).sum();
        let total_tokens: u64 = results.iter().map(|r| r.tokens_used).sum();
        out.push_str(&format!(
            "{:<24} {:>12} {:>9.2}% {:>13.4}\n",
            "TOTAL",
            total_tokens,
            100.0_f64,
            total_cost,
        ));
        out.push_str(&separator);

        out
    }

    /// Returns `true` when the allocations sum to the pool total within a
    /// floating-point epsilon of `0.01` USD.
    pub fn validate_allocation(
        &self,
        results: &[AllocationResult],
        pool: &CostPool,
    ) -> bool {
        if results.is_empty() {
            return pool.total_cost_usd.abs() < 0.01;
        }
        let total: f64 = results.iter().map(|r| r.allocated_cost_usd).sum();
        (total - pool.total_cost_usd).abs() < 0.01
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Compute a normalised fraction for each consumer under `method`.
    fn compute_fractions(
        &self,
        consumers: &[ConsumerUsage],
        method: &AllocationMethod,
    ) -> Vec<f64> {
        match method {
            AllocationMethod::ProportionalUsage => {
                let total_tokens: u64 = consumers.iter().map(|c| c.tokens_used).sum();
                if total_tokens == 0 {
                    // Fall back to equal split when no tokens consumed.
                    return equal_fractions(consumers.len());
                }
                consumers
                    .iter()
                    .map(|c| c.tokens_used as f64 / total_tokens as f64)
                    .collect()
            }

            AllocationMethod::EqualSplit => equal_fractions(consumers.len()),

            AllocationMethod::WeightedByPriority(weights) => {
                let raw: Vec<f64> = consumers
                    .iter()
                    .map(|c| *weights.get(&c.consumer_id).unwrap_or(&0.0))
                    .collect();
                normalise(&raw)
            }

            AllocationMethod::ActivityBased => {
                let total_requests: u64 = consumers.iter().map(|c| c.requests).sum();
                if total_requests == 0 {
                    return equal_fractions(consumers.len());
                }
                consumers
                    .iter()
                    .map(|c| c.requests as f64 / total_requests as f64)
                    .collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Private utilities
// ---------------------------------------------------------------------------

/// Equal fractions for `n` consumers.
fn equal_fractions(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    vec![1.0 / n as f64; n]
}

/// Normalise a raw weight vector so it sums to `1.0`.
/// If the sum is zero every consumer gets an equal share.
fn normalise(raw: &[f64]) -> Vec<f64> {
    let sum: f64 = raw.iter().sum();
    if sum < f64::EPSILON {
        return equal_fractions(raw.len());
    }
    raw.iter().map(|w| w / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(total: f64) -> CostPool {
        CostPool {
            pool_id: "test-pool".to_string(),
            total_cost_usd: total,
            period_start: 0,
            period_end: 86_400,
        }
    }

    fn consumers_tokens(pairs: &[(&str, u64)]) -> Vec<ConsumerUsage> {
        pairs
            .iter()
            .map(|(id, tokens)| ConsumerUsage {
                consumer_id: id.to_string(),
                tokens_used: *tokens,
                requests: *tokens / 10,
                priority_weight: 1.0,
            })
            .collect()
    }

    fn allocator() -> CostAllocator {
        CostAllocator::new()
    }

    // ── ProportionalUsage ─────────────────────────────────────────────────

    #[test]
    fn proportional_usage_correct_fractions() {
        let p = pool(100.0);
        let c = consumers_tokens(&[("a", 1000), ("b", 3000)]);
        let r = allocator().allocate(&p, &c, &AllocationMethod::ProportionalUsage);
        assert_eq!(r.len(), 2);
        assert!((r[0].allocated_cost_usd - 25.0).abs() < 0.001);
        assert!((r[1].allocated_cost_usd - 75.0).abs() < 0.001);
    }

    #[test]
    fn proportional_usage_zero_tokens_falls_back_to_equal() {
        let p = pool(100.0);
        let c = consumers_tokens(&[("a", 0), ("b", 0)]);
        let r = allocator().allocate(&p, &c, &AllocationMethod::ProportionalUsage);
        assert!((r[0].allocated_cost_usd - 50.0).abs() < 0.001);
        assert!((r[1].allocated_cost_usd - 50.0).abs() < 0.001);
    }

    // ── EqualSplit ────────────────────────────────────────────────────────

    #[test]
    fn equal_split_divides_evenly() {
        let p = pool(90.0);
        let c = consumers_tokens(&[("a", 100), ("b", 200), ("c", 300)]);
        let r = allocator().allocate(&p, &c, &AllocationMethod::EqualSplit);
        for result in &r {
            assert!((result.allocated_cost_usd - 30.0).abs() < 0.001);
        }
    }

    // ── WeightedByPriority ────────────────────────────────────────────────

    #[test]
    fn weighted_by_priority_respects_weights() {
        let p = pool(100.0);
        let c = consumers_tokens(&[("alpha", 100), ("beta", 100)]);
        let mut weights = HashMap::new();
        weights.insert("alpha".to_string(), 3.0);
        weights.insert("beta".to_string(), 1.0);
        let r = allocator().allocate(&p, &c, &AllocationMethod::WeightedByPriority(weights));
        assert!((r[0].allocated_cost_usd - 75.0).abs() < 0.001);
        assert!((r[1].allocated_cost_usd - 25.0).abs() < 0.001);
    }

    #[test]
    fn weighted_by_priority_missing_consumer_gets_zero() {
        let p = pool(100.0);
        let c = consumers_tokens(&[("alpha", 100), ("gamma", 100)]);
        let mut weights = HashMap::new();
        weights.insert("alpha".to_string(), 1.0);
        // gamma not in weights → 0.0
        let r = allocator().allocate(&p, &c, &AllocationMethod::WeightedByPriority(weights));
        assert!((r[0].allocated_cost_usd - 100.0).abs() < 0.001);
        assert!((r[1].allocated_cost_usd - 0.0).abs() < 0.001);
    }

    #[test]
    fn weighted_all_zero_falls_back_to_equal() {
        let p = pool(100.0);
        let c = consumers_tokens(&[("a", 50), ("b", 50)]);
        let r = allocator().allocate(&p, &c, &AllocationMethod::WeightedByPriority(HashMap::new()));
        assert!((r[0].allocated_cost_usd - 50.0).abs() < 0.001);
        assert!((r[1].allocated_cost_usd - 50.0).abs() < 0.001);
    }

    // ── ActivityBased ─────────────────────────────────────────────────────

    #[test]
    fn activity_based_uses_requests() {
        let p = pool(100.0);
        let consumers = vec![
            ConsumerUsage { consumer_id: "a".to_string(), tokens_used: 9999, requests: 10, priority_weight: 1.0 },
            ConsumerUsage { consumer_id: "b".to_string(), tokens_used: 1,    requests: 40, priority_weight: 1.0 },
        ];
        let r = allocator().allocate(&p, &consumers, &AllocationMethod::ActivityBased);
        assert!((r[0].allocated_cost_usd - 20.0).abs() < 0.001);
        assert!((r[1].allocated_cost_usd - 80.0).abs() < 0.001);
    }

    // ── Empty consumers ───────────────────────────────────────────────────

    #[test]
    fn empty_consumers_returns_empty() {
        let p = pool(100.0);
        let r = allocator().allocate(&p, &[], &AllocationMethod::EqualSplit);
        assert!(r.is_empty());
    }

    // ── validate_allocation ───────────────────────────────────────────────

    #[test]
    fn validate_allocation_passes_for_correct_sum() {
        let p = pool(100.0);
        let c = consumers_tokens(&[("a", 500), ("b", 500)]);
        let r = allocator().allocate(&p, &c, &AllocationMethod::ProportionalUsage);
        assert!(allocator().validate_allocation(&r, &p));
    }

    #[test]
    fn validate_allocation_fails_for_wrong_sum() {
        let p = pool(100.0);
        let bad = vec![AllocationResult {
            consumer_id: "x".to_string(),
            allocated_cost_usd: 50.0,  // Only half the pool
            fraction: 0.5,
            tokens_used: 100,
        }];
        assert!(!allocator().validate_allocation(&bad, &p));
    }

    #[test]
    fn validate_allocation_empty_results_passes_for_zero_pool() {
        let p = pool(0.0);
        assert!(allocator().validate_allocation(&[], &p));
    }

    // ── chargeback_report ─────────────────────────────────────────────────

    #[test]
    fn chargeback_report_contains_consumer_ids() {
        let p = pool(100.0);
        let c = consumers_tokens(&[("team-alpha", 600), ("team-beta", 400)]);
        let r = allocator().allocate(&p, &c, &AllocationMethod::ProportionalUsage);
        let report = allocator().chargeback_report(&r);
        assert!(report.contains("team-alpha"));
        assert!(report.contains("team-beta"));
        assert!(report.contains("TOTAL"));
    }

    #[test]
    fn chargeback_report_empty_results() {
        let report = allocator().chargeback_report(&[]);
        assert!(report.contains("No allocation results"));
    }

    // ── AllocationResult fields ───────────────────────────────────────────

    #[test]
    fn allocation_result_fraction_sums_to_one() {
        let p = pool(200.0);
        let c = consumers_tokens(&[("x", 100), ("y", 200), ("z", 300)]);
        let r = allocator().allocate(&p, &c, &AllocationMethod::ProportionalUsage);
        let total_frac: f64 = r.iter().map(|a| a.fraction).sum();
        assert!((total_frac - 1.0).abs() < 1e-9);
    }

    #[test]
    fn allocation_result_preserves_token_counts() {
        let p = pool(100.0);
        let c = vec![ConsumerUsage {
            consumer_id: "only".to_string(),
            tokens_used: 42,
            requests: 5,
            priority_weight: 1.0,
        }];
        let r = allocator().allocate(&p, &c, &AllocationMethod::EqualSplit);
        assert_eq!(r[0].tokens_used, 42);
    }

    // ── Default impl ──────────────────────────────────────────────────────

    #[test]
    fn default_impl_works() {
        let _ = CostAllocator::default();
    }
}
