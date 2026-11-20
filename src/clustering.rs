//! Semantic cost clustering by prompt similarity.
//!
//! Groups LLM requests by semantic category using keyword-based classification
//! (no external API required). Each request is assigned to a cluster based on
//! which category keywords appear in the prompt text. Cost totals are
//! accumulated per cluster so the operator can see which use-cases are most
//! expensive.
//!
//! ## Built-in Clusters
//!
//! | Cluster           | Example keywords                             |
//! |-------------------|----------------------------------------------|
//! | `code_generation` | `function`, `implement`, `class`, `refactor` |
//! | `summarization`   | `summarize`, `summary`, `tldr`, `brief`      |
//! | `question_answer` | `what is`, `explain`, `how does`, `define`   |
//! | `translation`     | `translate`, `french`, `spanish`, `language` |
//! | `creative`        | `story`, `poem`, `creative`, `write a`       |
//! | `data_analysis`   | `analyze`, `chart`, `statistics`, `dataset`  |
//! | `other`           | catch-all for unclassified requests          |
//!
//! Custom clusters can be added via [`CostClusterer::add_cluster`].
//!
//! ## Usage
//!
//! ```
//! use llm_cost_dashboard::clustering::{CostClusterer, ClusterConfig};
//!
//! let mut clusterer = CostClusterer::new(ClusterConfig::default());
//! clusterer.observe("Please implement a Rust function", 0.012);
//! clusterer.observe("Summarize the article in 3 bullet points", 0.003);
//!
//! let report = clusterer.report();
//! for c in &report.clusters {
//!     println!("{}: ${:.4} ({} requests)", c.name, c.total_cost_usd, c.request_count);
//! }
//! ```

use std::collections::HashMap;

/// Configuration for the cost clusterer.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Whether keyword matching is case-insensitive.
    pub case_insensitive: bool,
    /// Minimum keyword matches required. `0` = first matching cluster wins.
    pub min_keyword_matches: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            case_insensitive: true,
            min_keyword_matches: 0,
        }
    }
}

/// A cluster definition with a name, label, and keywords.
#[derive(Debug, Clone)]
pub struct ClusterDef {
    /// Unique cluster identifier.
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// Keywords/phrases that indicate this cluster.
    pub keywords: Vec<String>,
    /// Match priority (lower = matched first).
    pub priority: u32,
}

impl ClusterDef {
    fn match_count(&self, text: &str) -> usize {
        self.keywords
            .iter()
            .filter(|kw| text.contains(kw.as_str()))
            .count()
    }
}

/// Per-cluster cost summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterSummary {
    /// Cluster identifier.
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// Total cost accumulated in this cluster (USD).
    pub total_cost_usd: f64,
    /// Number of requests in this cluster.
    pub request_count: u64,
    /// Average cost per request (USD).
    pub avg_cost_usd: f64,
    /// Fraction of total cost (0–1).
    pub cost_fraction: f64,
    /// Estimated daily cost based on observation window (USD).
    pub daily_cost_usd: f64,
}

/// Full cost clustering report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterReport {
    /// Cluster summaries sorted by total cost descending.
    pub clusters: Vec<ClusterSummary>,
    /// Grand total cost across all clusters (USD).
    pub total_cost_usd: f64,
    /// Total request count.
    pub total_requests: u64,
    /// Name of the most expensive cluster.
    pub top_cluster: String,
    /// Observation window in days used for daily cost estimation.
    pub observation_days: f64,
}

#[derive(Debug, Default)]
struct ClusterAccum {
    total_cost: f64,
    count: u64,
}

/// Keyword-based semantic cost clusterer.
pub struct CostClusterer {
    cfg: ClusterConfig,
    clusters: Vec<ClusterDef>,
    accum: HashMap<String, ClusterAccum>,
    observation_secs: f64,
}

impl CostClusterer {
    /// Create a new clusterer with built-in cluster definitions.
    pub fn new(cfg: ClusterConfig) -> Self {
        let mut s = Self {
            cfg,
            clusters: Vec::new(),
            accum: HashMap::new(),
            observation_secs: 86_400.0,
        };
        s.register_defaults();
        s
    }

    /// Add a custom cluster definition.
    pub fn add_cluster(&mut self, def: ClusterDef) {
        self.clusters.push(def);
        self.clusters.sort_by_key(|c| c.priority);
    }

    /// Set observation window in seconds (for daily cost estimation).
    pub fn set_observation_secs(&mut self, secs: f64) {
        self.observation_secs = secs.max(1.0);
    }

    /// Record a prompt observation with its cost.
    pub fn observe(&mut self, prompt: &str, cost_usd: f64) {
        let name = self.classify(prompt).to_string();
        let entry = self.accum.entry(name).or_default();
        entry.total_cost += cost_usd;
        entry.count += 1;
    }

    /// Classify a prompt into a cluster name.
    pub fn classify<'a>(&'a self, prompt: &str) -> &'a str {
        let text = if self.cfg.case_insensitive {
            prompt.to_lowercase()
        } else {
            prompt.to_string()
        };
        let mut best_name = "other";
        let mut best_count = 0usize;
        for cluster in &self.clusters {
            let count = cluster.match_count(&text);
            if count > 0 && count > best_count {
                best_count = count;
                best_name = &cluster.name;
            }
            if self.cfg.min_keyword_matches == 0 && count > 0 && best_count == count {
                break;
            }
        }
        best_name
    }

    /// Generate the full clustering report.
    pub fn report(&self) -> ClusterReport {
        let grand_total: f64 = self.accum.values().map(|a| a.total_cost).sum();
        let grand_count: u64 = self.accum.values().map(|a| a.count).sum();
        let days = self.observation_secs / 86_400.0;

        let mut summaries: Vec<ClusterSummary> = self
            .accum
            .iter()
            .map(|(name, acc)| {
                let label = self
                    .clusters
                    .iter()
                    .find(|c| &c.name == name)
                    .map(|c| c.label.as_str())
                    .unwrap_or(name)
                    .to_string();
                ClusterSummary {
                    name: name.clone(),
                    label,
                    total_cost_usd: acc.total_cost,
                    request_count: acc.count,
                    avg_cost_usd: if acc.count > 0 {
                        acc.total_cost / acc.count as f64
                    } else {
                        0.0
                    },
                    cost_fraction: if grand_total > 0.0 {
                        acc.total_cost / grand_total
                    } else {
                        0.0
                    },
                    daily_cost_usd: acc.total_cost / days,
                }
            })
            .collect();

        summaries.sort_by(|a, b| {
            b.total_cost_usd
                .partial_cmp(&a.total_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_cluster = summaries
            .first()
            .map(|c| c.name.clone())
            .unwrap_or_default();

        ClusterReport {
            clusters: summaries,
            total_cost_usd: grand_total,
            total_requests: grand_count,
            top_cluster,
            observation_days: days,
        }
    }

    /// Reset all accumulated data.
    pub fn reset(&mut self) {
        self.accum.clear();
    }

    fn register_defaults(&mut self) {
        let defs: &[(&str, &str, u32, &[&str])] = &[
            (
                "code_generation",
                "Code Generation",
                10,
                &[
                    "function",
                    "implement",
                    "class",
                    "method",
                    "refactor",
                    "debug",
                    "compile",
                    "syntax",
                    "algorithm",
                    "code",
                    "program",
                    "fn ",
                    "struct",
                    "fix this",
                ],
            ),
            (
                "summarization",
                "Summarization",
                20,
                &[
                    "summarize",
                    "summary",
                    "tldr",
                    "brief",
                    "overview",
                    "condense",
                    "key points",
                    "in short",
                    "bullet point",
                    "abstract",
                ],
            ),
            (
                "question_answer",
                "Question & Answer",
                30,
                &[
                    "what is", "explain", "how does", "define", "why does", "tell me", "what are",
                    "how to", "when did",
                ],
            ),
            (
                "translation",
                "Translation",
                40,
                &[
                    "translate",
                    "french",
                    "spanish",
                    "german",
                    "japanese",
                    "chinese",
                    "language",
                    "portuguese",
                    "italian",
                    "arabic",
                    "russian",
                ],
            ),
            (
                "creative",
                "Creative Writing",
                50,
                &[
                    "story",
                    "poem",
                    "creative",
                    "write a",
                    "imagine",
                    "fiction",
                    "character",
                    "plot",
                    "narrative",
                    "compose",
                    "song",
                ],
            ),
            (
                "data_analysis",
                "Data Analysis",
                60,
                &[
                    "analyze",
                    "analysis",
                    "chart",
                    "statistics",
                    "dataset",
                    "trends",
                    "regression",
                    "correlation",
                    "visualize",
                    "aggregate",
                    "sql",
                ],
            ),
        ];
        for (name, label, priority, kws) in defs {
            self.clusters.push(ClusterDef {
                name: (*name).to_string(),
                label: (*label).to_string(),
                keywords: kws.iter().map(|s| s.to_string()).collect(),
                priority: *priority,
            });
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn clusterer() -> CostClusterer {
        CostClusterer::new(ClusterConfig::default())
    }

    #[test]
    fn classifies_code_generation() {
        assert_eq!(
            clusterer().classify("Please implement a function to sort a list"),
            "code_generation"
        );
    }

    #[test]
    fn classifies_summarization() {
        assert_eq!(
            clusterer().classify("Summarize the following article in bullet points"),
            "summarization"
        );
    }

    #[test]
    fn classifies_translation() {
        assert_eq!(
            clusterer().classify("Translate the following text to French"),
            "translation"
        );
    }

    #[test]
    fn classifies_creative() {
        assert_eq!(
            clusterer().classify("Write a story about a robot"),
            "creative"
        );
    }

    #[test]
    fn unknown_prompt_is_other() {
        assert_eq!(clusterer().classify("zzzzzzzzzzz"), "other");
    }

    #[test]
    fn observe_accumulates_cost() {
        let mut c = clusterer();
        c.observe("implement a sorting algorithm", 0.01);
        c.observe("implement a search function", 0.02);
        let report = c.report();
        let code = report
            .clusters
            .iter()
            .find(|cl| cl.name == "code_generation")
            .unwrap();
        assert!((code.total_cost_usd - 0.03).abs() < 1e-9);
        assert_eq!(code.request_count, 2);
    }

    #[test]
    fn report_grand_total_correct() {
        let mut c = clusterer();
        c.observe("implement a function", 0.01);
        c.observe("summarize this", 0.005);
        let report = c.report();
        assert!((report.total_cost_usd - 0.015).abs() < 1e-9);
    }

    #[test]
    fn report_sorted_descending() {
        let mut c = clusterer();
        c.observe("summarize this", 0.001);
        c.observe("implement a complex function", 0.05);
        let report = c.report();
        let costs: Vec<f64> = report.clusters.iter().map(|cl| cl.total_cost_usd).collect();
        for i in 1..costs.len() {
            assert!(costs[i - 1] >= costs[i]);
        }
    }

    #[test]
    fn cost_fraction_sums_to_one() {
        let mut c = clusterer();
        c.observe("implement a function", 0.01);
        c.observe("summarize this", 0.02);
        c.observe("translate to french", 0.03);
        let report = c.report();
        let total_frac: f64 = report.clusters.iter().map(|cl| cl.cost_fraction).sum();
        assert!((total_frac - 1.0).abs() < 1e-9);
    }

    #[test]
    fn custom_cluster_takes_effect() {
        let mut c = clusterer();
        c.add_cluster(ClusterDef {
            name: "legal".to_string(),
            label: "Legal Review".to_string(),
            keywords: vec![
                "contract".to_string(),
                "clause".to_string(),
                "liability".to_string(),
            ],
            priority: 5,
        });
        assert_eq!(
            c.classify("Please review this contract for liability clauses"),
            "legal"
        );
    }

    #[test]
    fn reset_clears_data() {
        let mut c = clusterer();
        c.observe("implement a function", 0.01);
        c.reset();
        assert_eq!(c.report().total_requests, 0);
    }

    #[test]
    fn daily_cost_estimation() {
        let mut c = clusterer();
        c.set_observation_secs(3600.0); // 1 hour = 1/24 day
        c.observe("implement a function", 1.0);
        let report = c.report();
        let code = report
            .clusters
            .iter()
            .find(|cl| cl.name == "code_generation")
            .unwrap();
        assert!((code.daily_cost_usd - 24.0).abs() < 0.01);
    }
}
