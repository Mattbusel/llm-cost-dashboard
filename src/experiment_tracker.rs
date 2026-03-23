//! Track LLM experiments and hyperparameter sweeps with Pearson correlation sensitivity.

use std::collections::HashMap;

/// A single hyperparameter value — float, integer, boolean, or string.
#[derive(Debug, Clone, PartialEq)]
pub enum HpValue {
    /// Continuous floating-point value.
    Float(f64),
    /// Discrete integer value.
    Int(i64),
    /// Boolean flag.
    Bool(bool),
    /// Arbitrary string (e.g. optimizer name, activation function).
    Str(String),
}

impl HpValue {
    /// Return the numeric representation used for Pearson correlation (floats
    /// and ints map directly; booleans → 0.0/1.0; strings → `None`).
    fn as_f64(&self) -> Option<f64> {
        match self {
            HpValue::Float(v) => Some(*v),
            HpValue::Int(v) => Some(*v as f64),
            HpValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            HpValue::Str(_) => None,
        }
    }
}

/// A named hyperparameter.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperParam {
    /// Parameter name (e.g. `"learning_rate"`).
    pub name: String,
    /// Parameter value.
    pub value: HpValue,
}

/// Lifecycle state of an experiment run.
#[derive(Debug, Clone, PartialEq)]
pub enum RunStatus {
    /// Run is currently executing.
    Running,
    /// Run finished successfully.
    Completed,
    /// Run encountered a fatal error.
    Failed(String),
    /// Run was cancelled before completion.
    Cancelled,
}

/// A single execution of an experiment with a specific hyperparameter set.
#[derive(Debug, Clone)]
pub struct ExperimentRun {
    /// Unique run identifier.
    pub run_id: String,
    /// Parent experiment identifier.
    pub experiment_id: String,
    /// Hyperparameters used for this run.
    pub hyperparams: Vec<HyperParam>,
    /// Logged metrics (name → value).
    pub metrics: HashMap<String, f64>,
    /// Current lifecycle status.
    pub status: RunStatus,
    /// Wall-clock start time (ms since epoch).
    pub started_at: u64,
    /// Wall-clock end time, once the run is no longer `Running`.
    pub ended_at: Option<u64>,
    /// Total cost incurred by this run in USD.
    pub cost_usd: f64,
}

/// A collection of related runs sharing an objective.
#[derive(Debug, Clone)]
pub struct Experiment {
    /// Unique experiment identifier.
    pub id: String,
    /// Short human-readable name.
    pub name: String,
    /// Longer description of purpose and scope.
    pub description: String,
    /// All runs belonging to this experiment.
    pub runs: Vec<ExperimentRun>,
    /// Wall-clock creation time (ms since epoch).
    pub created_at: u64,
}

/// Manages experiments and their associated runs.
#[derive(Debug, Default)]
pub struct ExperimentTracker {
    experiments: HashMap<String, Experiment>,
    /// Map from `run_id` to `experiment_id` for fast run lookup.
    run_index: HashMap<String, String>,
    next_exp_id: u64,
    next_run_id: u64,
}

impl ExperimentTracker {
    /// Create a new, empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new experiment.  Returns the generated `experiment_id`.
    pub fn create_experiment(&mut self, name: &str, description: &str) -> String {
        let id = format!("exp-{}", self.next_exp_id);
        self.next_exp_id += 1;
        self.experiments.insert(
            id.clone(),
            Experiment {
                id: id.clone(),
                name: name.to_string(),
                description: description.to_string(),
                runs: Vec::new(),
                created_at: 0,
            },
        );
        id
    }

    /// Start a new run within `experiment_id`.  Returns the generated `run_id`.
    ///
    /// Returns an empty string if the experiment does not exist.
    pub fn start_run(&mut self, experiment_id: &str, hyperparams: Vec<HyperParam>) -> String {
        if !self.experiments.contains_key(experiment_id) {
            return String::new();
        }
        let run_id = format!("run-{}", self.next_run_id);
        self.next_run_id += 1;
        let run = ExperimentRun {
            run_id: run_id.clone(),
            experiment_id: experiment_id.to_string(),
            hyperparams,
            metrics: HashMap::new(),
            status: RunStatus::Running,
            started_at: 0,
            ended_at: None,
            cost_usd: 0.0,
        };
        self.run_index
            .insert(run_id.clone(), experiment_id.to_string());
        self.experiments
            .get_mut(experiment_id)
            .unwrap()
            .runs
            .push(run);
        run_id
    }

    /// Log (or overwrite) a named metric for `run_id`.
    ///
    /// Returns `true` if the run was found and the metric was recorded.
    pub fn log_metric(&mut self, run_id: &str, name: &str, value: f64) -> bool {
        if let Some(exp_id) = self.run_index.get(run_id).cloned() {
            if let Some(exp) = self.experiments.get_mut(&exp_id) {
                if let Some(run) = exp.runs.iter_mut().find(|r| r.run_id == run_id) {
                    run.metrics.insert(name.to_string(), value);
                    return true;
                }
            }
        }
        false
    }

    /// Mark `run_id` as completed with the given cost.
    pub fn complete_run(&mut self, run_id: &str, cost_usd: f64) {
        self.update_run(run_id, |run| {
            run.status = RunStatus::Completed;
            run.cost_usd = cost_usd;
            run.ended_at = Some(0);
        });
    }

    /// Mark `run_id` as failed with a human-readable reason.
    pub fn fail_run(&mut self, run_id: &str, reason: &str) {
        let reason = reason.to_string();
        self.update_run(run_id, move |run| {
            run.status = RunStatus::Failed(reason.clone());
            run.ended_at = Some(0);
        });
    }

    /// Return the run in `experiment_id` with the best value of `metric`.
    ///
    /// Only completed runs with the metric logged are considered.
    /// Set `maximize` to `true` to find the highest value, `false` for the lowest.
    pub fn best_run(
        &self,
        experiment_id: &str,
        metric: &str,
        maximize: bool,
    ) -> Option<&ExperimentRun> {
        let exp = self.experiments.get(experiment_id)?;
        exp.runs
            .iter()
            .filter(|r| r.status == RunStatus::Completed)
            .filter_map(|r| r.metrics.get(metric).map(|&v| (v, r)))
            .reduce(|(best_v, best_r), (v, r)| {
                let is_better = if maximize { v > best_v } else { v < best_v };
                if is_better { (v, r) } else { (best_v, best_r) }
            })
            .map(|(_, r)| r)
    }

    /// Compute the Pearson correlation between each numeric hyperparameter and
    /// `metric` across completed runs in `experiment_id`.
    ///
    /// Returns a `Vec<(hp_name, correlation)>` sorted by absolute correlation
    /// descending.  String hyperparameters and hyperparameters with zero variance
    /// are omitted.
    pub fn hyperparameter_sensitivity(
        &self,
        experiment_id: &str,
        metric: &str,
    ) -> Vec<(String, f64)> {
        let exp = match self.experiments.get(experiment_id) {
            Some(e) => e,
            None => return Vec::new(),
        };

        // Collect completed runs that have the target metric.
        let eligible: Vec<&ExperimentRun> = exp
            .runs
            .iter()
            .filter(|r| r.status == RunStatus::Completed && r.metrics.contains_key(metric))
            .collect();

        if eligible.len() < 2 {
            return Vec::new();
        }

        // Collect the metric values.
        let y: Vec<f64> = eligible
            .iter()
            .map(|r| *r.metrics.get(metric).unwrap())
            .collect();

        // Gather all unique HP names.
        let mut hp_names: Vec<String> = eligible
            .iter()
            .flat_map(|r| r.hyperparams.iter().map(|hp| hp.name.clone()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        hp_names.sort();

        let mut results: Vec<(String, f64)> = Vec::new();

        for hp_name in &hp_names {
            // Extract numeric HP values aligned with y.
            let x: Vec<f64> = eligible
                .iter()
                .filter_map(|r| {
                    r.hyperparams
                        .iter()
                        .find(|hp| &hp.name == hp_name)
                        .and_then(|hp| hp.value.as_f64())
                })
                .collect();

            if x.len() < 2 {
                continue;
            }

            // Only use runs where we have both x and y (take the first `x.len()` y values
            // assuming alignment — full alignment is guaranteed since we filter_map).
            let n = x.len().min(y.len());
            if n < 2 {
                continue;
            }
            let x = &x[..n];
            let y_slice = &y[..n];

            if let Some(r) = pearson(x, y_slice) {
                results.push((hp_name.clone(), r));
            }
        }

        results.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    fn update_run<F: FnOnce(&mut ExperimentRun)>(&mut self, run_id: &str, f: F) {
        if let Some(exp_id) = self.run_index.get(run_id).cloned() {
            if let Some(exp) = self.experiments.get_mut(&exp_id) {
                if let Some(run) = exp.runs.iter_mut().find(|r| r.run_id == run_id) {
                    f(run);
                }
            }
        }
    }
}

/// Pearson product-moment correlation coefficient.  Returns `None` if either
/// series has zero variance.
fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
    let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

    if var_x == 0.0 || var_y == 0.0 {
        return None;
    }

    Some(cov / (var_x.sqrt() * var_y.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hp_float(name: &str, v: f64) -> HyperParam {
        HyperParam { name: name.to_string(), value: HpValue::Float(v) }
    }

    fn hp_int(name: &str, v: i64) -> HyperParam {
        HyperParam { name: name.to_string(), value: HpValue::Int(v) }
    }

    fn hp_bool(name: &str, v: bool) -> HyperParam {
        HyperParam { name: name.to_string(), value: HpValue::Bool(v) }
    }

    fn hp_str(name: &str, v: &str) -> HyperParam {
        HyperParam { name: name.to_string(), value: HpValue::Str(v.to_string()) }
    }

    #[test]
    fn create_experiment_returns_id() {
        let mut tracker = ExperimentTracker::new();
        let id = tracker.create_experiment("test", "desc");
        assert!(!id.is_empty());
    }

    #[test]
    fn start_run_unknown_experiment() {
        let mut tracker = ExperimentTracker::new();
        let run_id = tracker.start_run("nope", vec![]);
        assert!(run_id.is_empty());
    }

    #[test]
    fn log_metric_and_complete() {
        let mut tracker = ExperimentTracker::new();
        let exp_id = tracker.create_experiment("e", "d");
        let run_id = tracker.start_run(&exp_id, vec![hp_float("lr", 0.01)]);
        assert!(tracker.log_metric(&run_id, "loss", 0.5));
        tracker.complete_run(&run_id, 0.05);
        let best = tracker.best_run(&exp_id, "loss", false).unwrap();
        assert_eq!(best.run_id, run_id);
        assert!((best.cost_usd - 0.05).abs() < 1e-9);
    }

    #[test]
    fn fail_run() {
        let mut tracker = ExperimentTracker::new();
        let exp_id = tracker.create_experiment("e", "d");
        let run_id = tracker.start_run(&exp_id, vec![]);
        tracker.fail_run(&run_id, "OOM");
        let exp = tracker.experiments.get(&exp_id).unwrap();
        let run = exp.runs.iter().find(|r| r.run_id == run_id).unwrap();
        assert_eq!(run.status, RunStatus::Failed("OOM".to_string()));
    }

    #[test]
    fn best_run_maximize() {
        let mut tracker = ExperimentTracker::new();
        let exp_id = tracker.create_experiment("e", "d");
        for (lr, acc) in [(0.1, 0.7), (0.01, 0.9), (0.001, 0.8)] {
            let run_id = tracker.start_run(&exp_id, vec![hp_float("lr", lr)]);
            tracker.log_metric(&run_id, "accuracy", acc);
            tracker.complete_run(&run_id, 0.0);
        }
        let best = tracker.best_run(&exp_id, "accuracy", true).unwrap();
        assert!((best.metrics["accuracy"] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn best_run_minimize() {
        let mut tracker = ExperimentTracker::new();
        let exp_id = tracker.create_experiment("e", "d");
        for (lr, loss) in [(0.1, 0.5), (0.01, 0.2), (0.001, 0.8)] {
            let run_id = tracker.start_run(&exp_id, vec![hp_float("lr", lr)]);
            tracker.log_metric(&run_id, "loss", loss);
            tracker.complete_run(&run_id, 0.0);
        }
        let best = tracker.best_run(&exp_id, "loss", false).unwrap();
        assert!((best.metrics["loss"] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn best_run_excludes_running() {
        let mut tracker = ExperimentTracker::new();
        let exp_id = tracker.create_experiment("e", "d");
        let run_id = tracker.start_run(&exp_id, vec![]);
        tracker.log_metric(&run_id, "acc", 0.99);
        // NOT completed → should not appear as best.
        assert!(tracker.best_run(&exp_id, "acc", true).is_none());
    }

    #[test]
    fn hyperparameter_sensitivity_positive_correlation() {
        let mut tracker = ExperimentTracker::new();
        let exp_id = tracker.create_experiment("sweep", "lr sweep");
        // higher lr → higher metric (perfect positive correlation)
        for (lr, acc) in [(0.1, 0.5), (0.2, 0.6), (0.3, 0.7), (0.4, 0.8)] {
            let run_id = tracker.start_run(&exp_id, vec![hp_float("lr", lr)]);
            tracker.log_metric(&run_id, "acc", acc);
            tracker.complete_run(&run_id, 0.0);
        }
        let sensitivity = tracker.hyperparameter_sensitivity(&exp_id, "acc");
        assert!(!sensitivity.is_empty());
        let (name, corr) = &sensitivity[0];
        assert_eq!(name, "lr");
        assert!((corr - 1.0).abs() < 1e-9, "expected ~1.0, got {}", corr);
    }

    #[test]
    fn hyperparameter_sensitivity_skips_string_hp() {
        let mut tracker = ExperimentTracker::new();
        let exp_id = tracker.create_experiment("e", "d");
        for (opt, acc) in [("adam", 0.8), ("sgd", 0.7)] {
            let run_id = tracker.start_run(&exp_id, vec![hp_str("optimizer", opt)]);
            tracker.log_metric(&run_id, "acc", acc);
            tracker.complete_run(&run_id, 0.0);
        }
        let sensitivity = tracker.hyperparameter_sensitivity(&exp_id, "acc");
        // String HP has no numeric representation → should be omitted.
        assert!(sensitivity.is_empty());
    }

    #[test]
    fn hp_value_as_f64() {
        assert_eq!(HpValue::Float(3.14).as_f64(), Some(3.14));
        assert_eq!(HpValue::Int(-7).as_f64(), Some(-7.0));
        assert_eq!(HpValue::Bool(true).as_f64(), Some(1.0));
        assert_eq!(HpValue::Bool(false).as_f64(), Some(0.0));
        assert_eq!(HpValue::Str("x".to_string()).as_f64(), None);
    }

    #[test]
    fn pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let r = pearson(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_zero_variance_returns_none() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(pearson(&x, &y).is_none());
    }
}
