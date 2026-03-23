//! # Usage Report
//!
//! Weekly / monthly usage summaries with entity and model breakdowns,
//! trend data, and multiple render formats (text + CSV).

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// UsageRecord
// ---------------------------------------------------------------------------

/// A single recorded LLM request.
#[derive(Debug, Clone)]
pub struct UsageRecord {
    /// The entity (user, team, project) that made the request.
    pub entity: String,
    /// The model that served the request.
    pub model: String,
    /// Input tokens consumed.
    pub tokens_in: u64,
    /// Output tokens generated.
    pub tokens_out: u64,
    /// Total cost in USD.
    pub cost_usd: f64,
    /// Wall-clock time of the request.
    pub timestamp: std::time::Instant,
}

// ---------------------------------------------------------------------------
// EntitySummary
// ---------------------------------------------------------------------------

/// Aggregated statistics for one entity over the report period.
#[derive(Debug, Clone)]
pub struct EntitySummary {
    /// Entity identifier.
    pub entity: String,
    /// Total spend in USD.
    pub total_cost: f64,
    /// Total tokens (in + out).
    pub total_tokens: u64,
    /// Number of requests.
    pub request_count: u64,
    /// Model used most often by this entity.
    pub top_model: String,
    /// `total_cost / request_count`.
    pub avg_cost_per_request: f64,
}

// ---------------------------------------------------------------------------
// ModelSummary
// ---------------------------------------------------------------------------

/// Aggregated statistics for one model over the report period.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Model identifier.
    pub model: String,
    /// Total spend in USD.
    pub total_cost: f64,
    /// Number of requests.
    pub request_count: u64,
    /// Average tokens per request.
    pub avg_tokens_per_req: f64,
}

// ---------------------------------------------------------------------------
// UsageReport
// ---------------------------------------------------------------------------

/// Full summary report for a billing/usage period.
#[derive(Debug, Clone)]
pub struct UsageReport {
    /// Human-readable label for this period (e.g. "2026-W12").
    pub period_label: String,
    /// Total cost across all entities and models.
    pub total_cost: f64,
    /// Total number of requests.
    pub total_requests: u64,
    /// Per-entity breakdown, sorted by total cost descending.
    pub by_entity: Vec<EntitySummary>,
    /// Per-model breakdown, sorted by total cost descending.
    pub by_model: Vec<ModelSummary>,
    /// Top-N consuming entities (entity_id, cost).
    pub top_consumers: Vec<(String, f64)>,
    /// Daily cost trend (day_label, cost).
    pub cost_trend: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// UsageReportBuilder
// ---------------------------------------------------------------------------

/// Accumulates [`UsageRecord`]s and can produce [`UsageReport`]s.
#[derive(Debug, Default)]
pub struct UsageReportBuilder {
    records: Vec<UsageRecord>,
}

impl UsageReportBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one record to the builder.
    pub fn add_record(&mut self, r: UsageRecord) {
        self.records.push(r);
    }

    /// Build a full [`UsageReport`] by aggregating all stored records.
    pub fn build_report(&self, period_label: &str) -> UsageReport {
        let total_cost: f64 = self.records.iter().map(|r| r.cost_usd).sum();
        let total_requests = self.records.len() as u64;

        let by_entity = self.aggregate_by_entity();
        let by_model = self.aggregate_by_model();
        let top_consumers = self.top_n_consumers(10);
        let cost_trend = self.build_daily_breakdown();

        UsageReport {
            period_label: period_label.to_string(),
            total_cost,
            total_requests,
            by_entity,
            by_model,
            top_consumers,
            cost_trend,
        }
    }

    /// Aggregate costs into a daily breakdown.
    ///
    /// Because [`std::time::Instant`] has no calendar semantics, this
    /// implementation groups records by position-index day (i.e., each
    /// chunk of records is assigned a sequential day label).  In a real
    /// system you would use `SystemTime` or a chrono timestamp.
    pub fn build_daily_breakdown(&self) -> Vec<(String, f64)> {
        if self.records.is_empty() {
            return Vec::new();
        }
        // Simple approach: divide records into groups of ≈24 for "days".
        // Each group is labelled "Day N".
        let chunk_size = (self.records.len() / 7).max(1);
        let mut days: Vec<(String, f64)> = Vec::new();
        for (day_idx, chunk) in self.records.chunks(chunk_size).enumerate() {
            let label = format!("Day-{:02}", day_idx + 1);
            let cost: f64 = chunk.iter().map(|r| r.cost_usd).sum();
            if let Some(last) = days.last_mut() {
                if last.0 == label {
                    last.1 += cost;
                    continue;
                }
            }
            days.push((label, cost));
        }
        days
    }

    /// Return the `n` entities with the highest total spend, sorted descending.
    pub fn top_n_consumers(&self, n: usize) -> Vec<(String, f64)> {
        let mut totals: HashMap<String, f64> = HashMap::new();
        for r in &self.records {
            *totals.entry(r.entity.clone()).or_insert(0.0) += r.cost_usd;
        }
        let mut sorted: Vec<(String, f64)> = totals.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Render a human-readable ASCII report.
    pub fn render_text(&self, report: &UsageReport) -> String {
        let mut out = String::new();
        out.push_str(&format!("=== Usage Report: {} ===\n\n", report.period_label));
        out.push_str(&format!("Total Cost:     ${:.4}\n", report.total_cost));
        out.push_str(&format!("Total Requests: {}\n\n", report.total_requests));

        out.push_str("--- By Entity ---\n");
        for e in &report.by_entity {
            out.push_str(&format!(
                "  {:20}  ${:.4}  ({} reqs, top model: {})\n",
                e.entity, e.total_cost, e.request_count, e.top_model
            ));
        }

        out.push_str("\n--- By Model ---\n");
        for m in &report.by_model {
            out.push_str(&format!(
                "  {:20}  ${:.4}  ({} reqs, avg {:.1} tokens/req)\n",
                m.model, m.total_cost, m.request_count, m.avg_tokens_per_req
            ));
        }

        out.push_str("\n--- Cost Trend ---\n");
        for (day, cost) in &report.cost_trend {
            let bar_len = ((cost / report.total_cost.max(f64::EPSILON)) * 40.0) as usize;
            out.push_str(&format!("  {} |{}| ${:.4}\n", day, "#".repeat(bar_len), cost));
        }

        out
    }

    /// Render a CSV representation of the report (entity breakdown).
    pub fn render_csv(&self, report: &UsageReport) -> String {
        let mut out = String::new();
        out.push_str("entity,total_cost_usd,total_tokens,request_count,top_model,avg_cost_per_request\n");
        for e in &report.by_entity {
            out.push_str(&format!(
                "{},{:.6},{},{},{},{:.6}\n",
                e.entity, e.total_cost, e.total_tokens, e.request_count,
                e.top_model, e.avg_cost_per_request
            ));
        }
        out
    }

    // ------------------------------------------------------------------
    // Private aggregation helpers
    // ------------------------------------------------------------------

    fn aggregate_by_entity(&self) -> Vec<EntitySummary> {
        // entity → (cost, tokens, count, model_counts)
        let mut map: HashMap<String, (f64, u64, u64, HashMap<String, u64>)> = HashMap::new();
        for r in &self.records {
            let entry = map.entry(r.entity.clone()).or_insert_with(|| (0.0, 0, 0, HashMap::new()));
            entry.0 += r.cost_usd;
            entry.1 += r.tokens_in + r.tokens_out;
            entry.2 += 1;
            *entry.3.entry(r.model.clone()).or_insert(0) += 1;
        }

        let mut summaries: Vec<EntitySummary> = map
            .into_iter()
            .map(|(entity, (cost, tokens, count, model_counts))| {
                let top_model = model_counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(m, _)| m)
                    .unwrap_or_default();
                EntitySummary {
                    entity,
                    total_cost: cost,
                    total_tokens: tokens,
                    request_count: count,
                    top_model,
                    avg_cost_per_request: if count > 0 { cost / count as f64 } else { 0.0 },
                }
            })
            .collect();

        summaries.sort_by(|a, b| b.total_cost.partial_cmp(&a.total_cost).unwrap_or(std::cmp::Ordering::Equal));
        summaries
    }

    fn aggregate_by_model(&self) -> Vec<ModelSummary> {
        // model → (cost, count, total_tokens)
        let mut map: HashMap<String, (f64, u64, u64)> = HashMap::new();
        for r in &self.records {
            let entry = map.entry(r.model.clone()).or_insert((0.0, 0, 0));
            entry.0 += r.cost_usd;
            entry.1 += 1;
            entry.2 += r.tokens_in + r.tokens_out;
        }

        let mut summaries: Vec<ModelSummary> = map
            .into_iter()
            .map(|(model, (cost, count, tokens))| ModelSummary {
                model,
                total_cost: cost,
                request_count: count,
                avg_tokens_per_req: if count > 0 { tokens as f64 / count as f64 } else { 0.0 },
            })
            .collect();

        summaries.sort_by(|a, b| b.total_cost.partial_cmp(&a.total_cost).unwrap_or(std::cmp::Ordering::Equal));
        summaries
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(entity: &str, model: &str, cost: f64) -> UsageRecord {
        UsageRecord {
            entity: entity.to_string(),
            model: model.to_string(),
            tokens_in: 100,
            tokens_out: 200,
            cost_usd: cost,
            timestamp: std::time::Instant::now(),
        }
    }

    fn builder() -> UsageReportBuilder {
        let mut b = UsageReportBuilder::new();
        b.add_record(sample_record("alice", "gpt-4", 0.50));
        b.add_record(sample_record("bob", "gpt-3.5", 0.10));
        b.add_record(sample_record("alice", "gpt-4", 0.30));
        b.add_record(sample_record("carol", "claude-3", 0.80));
        b
    }

    #[test]
    fn total_cost_correct() {
        let b = builder();
        let report = b.build_report("2026-W12");
        assert!((report.total_cost - 1.70).abs() < 0.001);
        assert_eq!(report.total_requests, 4);
    }

    #[test]
    fn entity_breakdown_sorted_descending() {
        let b = builder();
        let report = b.build_report("test");
        // carol (0.80) > alice (0.80) or alice (0.80) > carol (0.80)?
        // alice total = 0.80, carol = 0.80 — order may vary but should be sorted
        let costs: Vec<f64> = report.by_entity.iter().map(|e| e.total_cost).collect();
        for i in 0..costs.len().saturating_sub(1) {
            assert!(costs[i] >= costs[i + 1]);
        }
    }

    #[test]
    fn top_n_consumers_limits_count() {
        let b = builder();
        let top = b.top_n_consumers(2);
        assert!(top.len() <= 2);
    }

    #[test]
    fn render_text_contains_period_label() {
        let b = builder();
        let report = b.build_report("Q1-2026");
        let text = b.render_text(&report);
        assert!(text.contains("Q1-2026"));
    }

    #[test]
    fn render_csv_has_header() {
        let b = builder();
        let report = b.build_report("test");
        let csv = b.render_csv(&report);
        assert!(csv.starts_with("entity,"));
    }

    #[test]
    fn daily_breakdown_non_empty() {
        let b = builder();
        let days = b.build_daily_breakdown();
        assert!(!days.is_empty());
    }
}
