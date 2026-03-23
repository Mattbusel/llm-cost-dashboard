//! # Model Benchmark Score Tracker
//!
//! Track benchmark scores (MMLU, HumanEval, GSM8K, etc.) for LLM models over
//! time, compute trends via OLS linear regression, rank models per suite, and
//! compare models across common suites. Also exposes a cost-adjusted efficiency
//! metric and a Markdown table renderer.

use std::collections::HashMap;

/// Well-known benchmark suites plus a custom slot for ad-hoc evaluations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BenchmarkSuite {
    /// Massive Multitask Language Understanding.
    MMLU,
    /// Human-written programming puzzles (pass@1).
    HumanEval,
    /// Grade School Math 8K.
    GSM8K,
    /// HellaSwag commonsense NLI.
    HellaSwag,
    /// AI2 Reasoning Challenge.
    ARC,
    /// Any custom benchmark identified by name.
    Custom(String),
}

impl BenchmarkSuite {
    /// Return a stable string label for the suite (used in Markdown tables,
    /// `best_model_per_suite`, etc.).
    pub fn name(&self) -> String {
        match self {
            BenchmarkSuite::MMLU => "MMLU".to_string(),
            BenchmarkSuite::HumanEval => "HumanEval".to_string(),
            BenchmarkSuite::GSM8K => "GSM8K".to_string(),
            BenchmarkSuite::HellaSwag => "HellaSwag".to_string(),
            BenchmarkSuite::ARC => "ARC".to_string(),
            BenchmarkSuite::Custom(s) => s.clone(),
        }
    }
}

/// A single benchmark evaluation result.
#[derive(Debug, Clone)]
pub struct BenchmarkScore {
    /// Model identifier (e.g. "gpt-4o", "claude-3-5-sonnet").
    pub model_id: String,
    /// The benchmark suite this score was measured on.
    pub suite: BenchmarkSuite,
    /// Score from 0 to 100.
    pub score: f64,
    /// Unix timestamp of the evaluation date.
    pub date: u64,
    /// Number of samples / questions evaluated.
    pub samples_evaluated: u64,
    /// Optional notes about the evaluation run.
    pub notes: String,
}

/// Tracks benchmark scores over time and provides analysis methods.
#[derive(Debug, Default)]
pub struct BenchmarkTracker {
    scores: Vec<BenchmarkScore>,
}

impl BenchmarkTracker {
    /// Create an empty tracker.
    pub fn new() -> Self { Self::default() }

    /// Add a benchmark score.
    pub fn add_score(&mut self, score: BenchmarkScore) {
        self.scores.push(score);
    }

    /// Return the most recent score for a (model, suite) pair.
    pub fn latest_score(&self, model_id: &str, suite: &BenchmarkSuite) -> Option<&BenchmarkScore> {
        self.scores.iter()
            .filter(|s| s.model_id == model_id && &s.suite == suite)
            .max_by_key(|s| s.date)
    }

    /// Return all scores for a (model, suite) pair, sorted ascending by date.
    pub fn score_history(&self, model_id: &str, suite: &BenchmarkSuite) -> Vec<&BenchmarkScore> {
        let mut history: Vec<&BenchmarkScore> = self.scores.iter()
            .filter(|s| s.model_id == model_id && &s.suite == suite)
            .collect();
        history.sort_by_key(|s| s.date);
        history
    }

    /// Return `(model_id, latest_score)` pairs for all models that have been
    /// evaluated on `suite`, sorted by score descending.
    pub fn model_rankings(&self, suite: &BenchmarkSuite) -> Vec<(&str, f64)> {
        // Collect latest score per model for the suite.
        let mut best: HashMap<&str, f64> = HashMap::new();
        for s in self.scores.iter().filter(|s| &s.suite == suite) {
            let entry = best.entry(s.model_id.as_str()).or_insert(f64::NEG_INFINITY);
            // Use the score from the latest date.
            if let Some(latest) = self.latest_score(&s.model_id, suite) {
                *entry = latest.score;
            }
        }
        let mut rankings: Vec<(&str, f64)> = best.into_iter().collect();
        rankings.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        rankings
    }

    /// Compute OLS slope of score over time for a (model, suite) pair.
    ///
    /// Positive = improving, negative = degrading, `None` if fewer than 2 data
    /// points.
    pub fn score_trend(&self, model_id: &str, suite: &BenchmarkSuite) -> Option<f64> {
        let history = self.score_history(model_id, suite);
        if history.len() < 2 { return None; }

        let n = history.len() as f64;
        let xs: Vec<f64> = history.iter().map(|s| s.date as f64).collect();
        let ys: Vec<f64> = history.iter().map(|s| s.score).collect();

        let x_mean = xs.iter().sum::<f64>() / n;
        let y_mean = ys.iter().sum::<f64>() / n;

        let numerator: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - x_mean) * (y - y_mean)).sum();
        let denominator: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();

        if denominator.abs() < f64::EPSILON { return Some(0.0); }
        Some(numerator / denominator)
    }

    /// Compare two models across all suites where both have at least one score.
    ///
    /// Returns `Vec<(suite, score_a, score_b)>` for common suites.
    pub fn compare_models(&self, model_a: &str, model_b: &str) -> Vec<(BenchmarkSuite, f64, f64)> {
        // Gather latest scores per suite for each model.
        let suites_a: HashMap<String, f64> = self.suites_for_model(model_a);
        let suites_b: HashMap<String, f64> = self.suites_for_model(model_b);

        let mut result = Vec::new();
        for (suite_name, score_a) in &suites_a {
            if let Some(&score_b) = suites_b.get(suite_name) {
                let suite = self.suite_from_name(suite_name);
                result.push((suite, *score_a, score_b));
            }
        }
        // Sort for deterministic output.
        result.sort_by(|(sa, _, _), (sb, _, _)| sa.name().cmp(&sb.name()));
        result
    }

    /// Return a map of `suite_name -> best_model_id` across all tracked scores.
    pub fn best_model_per_suite(&self) -> HashMap<String, String> {
        let mut best: HashMap<String, (String, f64)> = HashMap::new();
        for score in &self.scores {
            let suite_name = score.suite.name();
            let entry = best.entry(suite_name).or_insert_with(|| (score.model_id.clone(), f64::NEG_INFINITY));
            if score.score > entry.1 {
                *entry = (score.model_id.clone(), score.score);
            }
        }
        best.into_iter().map(|(suite, (model, _))| (suite, model)).collect()
    }

    /// Cost-adjusted efficiency metric: `score / (cost_per_1k * 100)`.
    ///
    /// Higher is better — rewards high scores at low cost.
    /// Returns `None` if no score exists for the (model, suite) pair.
    pub fn cost_adjusted_score(
        &self,
        model_id: &str,
        suite: &BenchmarkSuite,
        cost_per_1k: f64,
    ) -> Option<f64> {
        let score = self.latest_score(model_id, suite)?.score;
        if cost_per_1k <= 0.0 { return None; }
        Some(score / (cost_per_1k * 100.0))
    }

    /// Render a Markdown table of model rankings for a given suite.
    ///
    /// Columns: Model | Score | Trend
    pub fn to_markdown_table(&self, suite: &BenchmarkSuite) -> String {
        let rankings = self.model_rankings(suite);
        if rankings.is_empty() {
            return format!("_No benchmark data for {}_\n", suite.name());
        }

        let mut lines = vec![
            format!("## {} Benchmark Rankings", suite.name()),
            String::new(),
            "| Model | Score | Trend |".to_string(),
            "|-------|-------|-------|".to_string(),
        ];

        for (model_id, score) in &rankings {
            let trend_str = match self.score_trend(model_id, suite) {
                None => "N/A".to_string(),
                Some(t) if t > 0.0 => format!("+{:.4}", t),
                Some(t) => format!("{:.4}", t),
            };
            lines.push(format!("| {} | {:.1} | {} |", model_id, score, trend_str));
        }
        lines.push(String::new());
        lines.join("\n")
    }

    // --- Private helpers ---

    fn suites_for_model(&self, model_id: &str) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        // Collect all suites that model has scores for, then take latest.
        let suites: Vec<String> = self.scores.iter()
            .filter(|s| s.model_id == model_id)
            .map(|s| s.suite.name())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        for suite_name in suites {
            let suite = self.suite_from_name(&suite_name);
            if let Some(latest) = self.latest_score(model_id, &suite) {
                map.insert(suite_name, latest.score);
            }
        }
        map
    }

    fn suite_from_name(&self, name: &str) -> BenchmarkSuite {
        match name {
            "MMLU" => BenchmarkSuite::MMLU,
            "HumanEval" => BenchmarkSuite::HumanEval,
            "GSM8K" => BenchmarkSuite::GSM8K,
            "HellaSwag" => BenchmarkSuite::HellaSwag,
            "ARC" => BenchmarkSuite::ARC,
            other => BenchmarkSuite::Custom(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_score(model: &str, suite: BenchmarkSuite, score: f64, date: u64) -> BenchmarkScore {
        BenchmarkScore {
            model_id: model.to_string(),
            suite,
            score,
            date,
            samples_evaluated: 100,
            notes: String::new(),
        }
    }

    #[test]
    fn rankings_sorted_descending() {
        let mut tracker = BenchmarkTracker::new();
        tracker.add_score(make_score("model-a", BenchmarkSuite::MMLU, 70.0, 1000));
        tracker.add_score(make_score("model-b", BenchmarkSuite::MMLU, 90.0, 1000));
        tracker.add_score(make_score("model-c", BenchmarkSuite::MMLU, 80.0, 1000));

        let rankings = tracker.model_rankings(&BenchmarkSuite::MMLU);
        assert_eq!(rankings.len(), 3);
        // First should be highest.
        assert_eq!(rankings[0].0, "model-b");
        assert!((rankings[0].1 - 90.0).abs() < 0.001);
        // Descending order.
        assert!(rankings[0].1 >= rankings[1].1);
        assert!(rankings[1].1 >= rankings[2].1);
    }

    #[test]
    fn trend_calculation_on_known_data() {
        let mut tracker = BenchmarkTracker::new();
        // Scores: 60 at t=100, 70 at t=200, 80 at t=300 → slope = 0.2 per unit time.
        tracker.add_score(make_score("model-a", BenchmarkSuite::HumanEval, 60.0, 100));
        tracker.add_score(make_score("model-a", BenchmarkSuite::HumanEval, 70.0, 200));
        tracker.add_score(make_score("model-a", BenchmarkSuite::HumanEval, 80.0, 300));

        let trend = tracker.score_trend("model-a", &BenchmarkSuite::HumanEval).unwrap();
        // OLS slope should be 0.2 (10 point increase per 50 time units).
        assert!((trend - 0.2).abs() < 0.001, "expected slope ~0.2, got {trend}");
        assert!(trend > 0.0, "improving model should have positive trend");
    }

    #[test]
    fn trend_none_with_single_sample() {
        let mut tracker = BenchmarkTracker::new();
        tracker.add_score(make_score("model-a", BenchmarkSuite::GSM8K, 75.0, 1000));
        assert!(tracker.score_trend("model-a", &BenchmarkSuite::GSM8K).is_none());
    }

    #[test]
    fn compare_models_finds_common_suites() {
        let mut tracker = BenchmarkTracker::new();
        tracker.add_score(make_score("gpt-4", BenchmarkSuite::MMLU, 86.4, 1000));
        tracker.add_score(make_score("gpt-4", BenchmarkSuite::GSM8K, 92.0, 1000));
        tracker.add_score(make_score("gpt-4", BenchmarkSuite::HumanEval, 80.1, 1000));
        tracker.add_score(make_score("claude-3", BenchmarkSuite::MMLU, 88.7, 1000));
        tracker.add_score(make_score("claude-3", BenchmarkSuite::GSM8K, 89.5, 1000));
        // claude-3 has no HumanEval score.

        let comparison = tracker.compare_models("gpt-4", "claude-3");
        // Should find MMLU and GSM8K (2 common suites), not HumanEval.
        assert_eq!(comparison.len(), 2, "expected 2 common suites, got {}", comparison.len());
        let suite_names: Vec<String> = comparison.iter().map(|(s, _, _)| s.name()).collect();
        assert!(suite_names.contains(&"MMLU".to_string()));
        assert!(suite_names.contains(&"GSM8K".to_string()));
        assert!(!suite_names.contains(&"HumanEval".to_string()));
    }

    #[test]
    fn best_model_per_suite() {
        let mut tracker = BenchmarkTracker::new();
        tracker.add_score(make_score("model-a", BenchmarkSuite::MMLU, 85.0, 1000));
        tracker.add_score(make_score("model-b", BenchmarkSuite::MMLU, 90.0, 1000));
        tracker.add_score(make_score("model-a", BenchmarkSuite::HumanEval, 75.0, 1000));
        tracker.add_score(make_score("model-c", BenchmarkSuite::HumanEval, 70.0, 1000));

        let best = tracker.best_model_per_suite();
        assert_eq!(best.get("MMLU").map(|s| s.as_str()), Some("model-b"));
        assert_eq!(best.get("HumanEval").map(|s| s.as_str()), Some("model-a"));
    }

    #[test]
    fn latest_score_returns_most_recent() {
        let mut tracker = BenchmarkTracker::new();
        tracker.add_score(make_score("model-x", BenchmarkSuite::ARC, 60.0, 100));
        tracker.add_score(make_score("model-x", BenchmarkSuite::ARC, 75.0, 300));
        tracker.add_score(make_score("model-x", BenchmarkSuite::ARC, 65.0, 200));

        let latest = tracker.latest_score("model-x", &BenchmarkSuite::ARC).unwrap();
        assert!((latest.score - 75.0).abs() < 0.001, "latest should be the score at t=300");
    }

    #[test]
    fn cost_adjusted_score_computation() {
        let mut tracker = BenchmarkTracker::new();
        tracker.add_score(make_score("model-a", BenchmarkSuite::MMLU, 80.0, 1000));

        // score / (cost * 100) = 80 / (0.02 * 100) = 80 / 2 = 40.
        let adj = tracker.cost_adjusted_score("model-a", &BenchmarkSuite::MMLU, 0.02).unwrap();
        assert!((adj - 40.0).abs() < 0.001, "expected 40.0, got {adj}");
    }

    #[test]
    fn markdown_table_contains_model_names() {
        let mut tracker = BenchmarkTracker::new();
        tracker.add_score(make_score("alpha", BenchmarkSuite::HellaSwag, 88.0, 1000));
        tracker.add_score(make_score("beta", BenchmarkSuite::HellaSwag, 92.0, 1000));

        let table = tracker.to_markdown_table(&BenchmarkSuite::HellaSwag);
        assert!(table.contains("alpha"), "table should contain 'alpha'");
        assert!(table.contains("beta"), "table should contain 'beta'");
        assert!(table.contains("| Model |"), "table should have Model column");
        assert!(table.contains("92.0"), "table should contain beta's score");
    }

    #[test]
    fn score_history_sorted_ascending() {
        let mut tracker = BenchmarkTracker::new();
        tracker.add_score(make_score("m", BenchmarkSuite::MMLU, 70.0, 300));
        tracker.add_score(make_score("m", BenchmarkSuite::MMLU, 60.0, 100));
        tracker.add_score(make_score("m", BenchmarkSuite::MMLU, 65.0, 200));

        let history = tracker.score_history("m", &BenchmarkSuite::MMLU);
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].date, 100);
        assert_eq!(history[1].date, 200);
        assert_eq!(history[2].date, 300);
    }
}
