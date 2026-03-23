//! Fine-grained cost attribution — record, group, filter, and export LLM spend
//! across multiple organisational dimensions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Global record ID counter
// ---------------------------------------------------------------------------

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Attribution dimensions
// ---------------------------------------------------------------------------

/// A named dimension along which costs can be attributed.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributionDimension {
    /// Individual user identifier.
    User(String),
    /// Team identifier.
    Team(String),
    /// Product feature identifier.
    Feature(String),
    /// API endpoint path.
    Endpoint(String),
    /// Model name/identifier.
    Model(String),
    /// Deployment environment (production, staging, dev …).
    Environment(String),
}

impl AttributionDimension {
    /// Return the dimension type label (e.g. `"user"`, `"team"`, …).
    pub fn type_label(&self) -> &'static str {
        match self {
            Self::User(_) => "user",
            Self::Team(_) => "team",
            Self::Feature(_) => "feature",
            Self::Endpoint(_) => "endpoint",
            Self::Model(_) => "model",
            Self::Environment(_) => "environment",
        }
    }

    /// Return the dimension value string.
    pub fn value(&self) -> &str {
        match self {
            Self::User(v)
            | Self::Team(v)
            | Self::Feature(v)
            | Self::Endpoint(v)
            | Self::Model(v)
            | Self::Environment(v) => v.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cost record
// ---------------------------------------------------------------------------

/// A single attributed cost record.
#[derive(Debug, Clone)]
pub struct CostRecord {
    /// Unique record identifier.
    pub id: u64,
    /// Cost in US dollars.
    pub cost_usd: f64,
    /// Number of tokens consumed.
    pub tokens_used: usize,
    /// Model that processed the request.
    pub model: String,
    /// Attribution dimensions attached to this record.
    pub dimensions: Vec<AttributionDimension>,
    /// Wall-clock time of the request (Unix seconds).
    pub timestamp_unix: u64,
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

/// Filter criteria for attribution queries.
#[derive(Debug, Clone, Default)]
pub struct AttributionFilter {
    /// Earliest timestamp to include (inclusive).
    pub start_time: Option<u64>,
    /// Latest timestamp to include (inclusive).
    pub end_time: Option<u64>,
    /// Restrict to records using one of these models (empty = all).
    pub models: Vec<String>,
    /// Minimum cost threshold.
    pub min_cost: Option<f64>,
}

impl AttributionFilter {
    /// Returns `true` if the record passes all filter criteria.
    pub fn matches(&self, record: &CostRecord) -> bool {
        if let Some(start) = self.start_time {
            if record.timestamp_unix < start {
                return false;
            }
        }
        if let Some(end) = self.end_time {
            if record.timestamp_unix > end {
                return false;
            }
        }
        if !self.models.is_empty() && !self.models.contains(&record.model) {
            return false;
        }
        if let Some(min) = self.min_cost {
            if record.cost_usd < min {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Attribution summary
// ---------------------------------------------------------------------------

/// Aggregated cost attribution for a single dimension value.
#[derive(Debug, Clone)]
pub struct AttributionSummary {
    /// Dimension label (e.g. `"team:engineering"` or `"user:alice"`).
    pub dimension_key: String,
    /// Total cost in USD.
    pub total_cost: f64,
    /// Number of records contributing to this summary.
    pub record_count: usize,
    /// Average cost per record.
    pub avg_cost: f64,
    /// Total tokens consumed.
    pub total_tokens: usize,
    /// Percentage of overall spend (0–100).
    pub pct_of_total: f64,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Thread-safe engine for recording and querying attributed LLM costs.
pub struct CostAttributionEngine {
    records: Mutex<Vec<CostRecord>>,
}

impl CostAttributionEngine {
    /// Create a new, empty engine.
    pub fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
        }
    }

    /// Append a cost record and return its assigned ID.
    pub fn record(
        &self,
        cost: f64,
        tokens: usize,
        model: &str,
        dims: Vec<AttributionDimension>,
    ) -> u64 {
        let id = next_id();
        let timestamp_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let rec = CostRecord {
            id,
            cost_usd: cost,
            tokens_used: tokens,
            model: model.to_string(),
            dimensions: dims,
            timestamp_unix,
        };

        if let Ok(mut records) = self.records.lock() {
            records.push(rec);
        }
        id
    }

    /// Group records by the value of a named dimension type and return summaries.
    ///
    /// `dimension_type` should be one of: `"user"`, `"team"`, `"feature"`,
    /// `"endpoint"`, `"model"`, `"environment"`.
    pub fn attribute_by(
        &self,
        dimension_type: &str,
        filter: &AttributionFilter,
    ) -> Vec<AttributionSummary> {
        let records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        let filtered: Vec<&CostRecord> = records.iter().filter(|r| filter.matches(r)).collect();
        let grand_total: f64 = filtered.iter().map(|r| r.cost_usd).sum();

        let mut groups: std::collections::HashMap<String, (f64, usize, usize)> =
            std::collections::HashMap::new();

        for record in &filtered {
            for dim in &record.dimensions {
                if dim.type_label() == dimension_type {
                    let key = format!("{}:{}", dimension_type, dim.value());
                    let entry = groups.entry(key).or_insert((0.0, 0, 0));
                    entry.0 += record.cost_usd;
                    entry.1 += 1;
                    entry.2 += record.tokens_used;
                }
            }
        }

        let mut summaries: Vec<AttributionSummary> = groups
            .into_iter()
            .map(|(key, (total_cost, record_count, total_tokens))| {
                let avg_cost = if record_count > 0 {
                    total_cost / record_count as f64
                } else {
                    0.0
                };
                let pct_of_total = if grand_total > f64::EPSILON {
                    total_cost / grand_total * 100.0
                } else {
                    0.0
                };
                AttributionSummary {
                    dimension_key: key,
                    total_cost,
                    record_count,
                    avg_cost,
                    total_tokens,
                    pct_of_total,
                }
            })
            .collect();

        summaries.sort_by(|a, b| b.total_cost.partial_cmp(&a.total_cost).unwrap_or(std::cmp::Ordering::Equal));
        summaries
    }

    /// Return the top-`n` consumers along `dimension_type`.
    pub fn top_consumers(
        &self,
        dimension_type: &str,
        n: usize,
        filter: &AttributionFilter,
    ) -> Vec<AttributionSummary> {
        let mut all = self.attribute_by(dimension_type, filter);
        all.truncate(n);
        all
    }

    /// Summarise costs grouped by model.
    pub fn cost_by_model(&self, filter: &AttributionFilter) -> Vec<AttributionSummary> {
        self.attribute_by("model", filter)
    }

    /// Return hourly cost breakdown for records matching a specific dimension value.
    ///
    /// Returns a list of `(hour_start_unix, cost_usd)` pairs sorted by time.
    pub fn hourly_breakdown(&self, dimension: &AttributionDimension) -> Vec<(u64, f64)> {
        let records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        let mut buckets: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();

        for record in records.iter() {
            let matches = record.dimensions.iter().any(|d| {
                d.type_label() == dimension.type_label() && d.value() == dimension.value()
            });
            if matches {
                let hour = (record.timestamp_unix / 3600) * 3600;
                *buckets.entry(hour).or_insert(0.0) += record.cost_usd;
            }
        }

        let mut result: Vec<(u64, f64)> = buckets.into_iter().collect();
        result.sort_by_key(|(ts, _)| *ts);
        result
    }

    /// Sum costs for all records matching the filter.
    pub fn total_cost(&self, filter: &AttributionFilter) -> f64 {
        let records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return 0.0,
        };
        records
            .iter()
            .filter(|r| filter.matches(r))
            .map(|r| r.cost_usd)
            .sum()
    }

    /// Export matching records as a CSV string.
    ///
    /// Columns: `id,cost_usd,tokens_used,model,dimensions,timestamp_unix`
    pub fn export_csv(&self, filter: &AttributionFilter) -> String {
        let records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return String::new(),
        };

        let mut out = String::from("id,cost_usd,tokens_used,model,dimensions,timestamp_unix\n");
        for record in records.iter().filter(|r| filter.matches(r)) {
            let dims: Vec<String> = record
                .dimensions
                .iter()
                .map(|d| format!("{}={}", d.type_label(), d.value()))
                .collect();
            let dims_str = dims.join(";");
            out.push_str(&format!(
                "{},{:.6},{},{},\"{}\",{}\n",
                record.id,
                record.cost_usd,
                record.tokens_used,
                record.model,
                dims_str,
                record.timestamp_unix,
            ));
        }
        out
    }
}

impl Default for CostAttributionEngine {
    fn default() -> Self {
        Self::new()
    }
}
