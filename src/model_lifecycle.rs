//! Model version lifecycle tracking, deprecation dates, and migration planning.
//!
//! Provides [`ModelLifecycleManager`] to register model versions, detect
//! deprecated or at-risk models, and generate [`MigrationPlan`]s.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ModelStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a model version.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    /// The model is fully supported and recommended for production use.
    Active,
    /// The model is deprecated and will be removed at `sunset_at`.
    Deprecated {
        /// Unix timestamp (seconds) when the model was deprecated.
        deprecated_at: u64,
        /// Unix timestamp (seconds) when the model will be removed (sunset).
        sunset_at: u64,
    },
    /// The model has been removed and is no longer available.
    Sunset,
    /// The model is available for early testing but not production-ready.
    Preview,
    /// The model is in public beta — stable but may still change.
    Beta,
}

// ---------------------------------------------------------------------------
// ModelVersion
// ---------------------------------------------------------------------------

/// Full metadata for a specific model version.
#[derive(Debug, Clone)]
pub struct ModelVersion {
    /// Primary model identifier (e.g. `"gpt-4"`).
    pub model_id: String,
    /// Specific version string (e.g. `"gpt-4-0613"`).
    pub version: String,
    /// Unix timestamp (seconds) of the public release date.
    pub release_date: u64,
    /// Current lifecycle status.
    pub status: ModelStatus,
    /// `model_id` of the recommended successor, if any.
    pub successor_id: Option<String>,
    /// List of capabilities this model supports (e.g. `["vision", "function_calling"]`).
    pub capabilities: Vec<String>,
    /// Maximum context window in tokens.
    pub context_window: u64,
}

// ---------------------------------------------------------------------------
// MigrationEffort
// ---------------------------------------------------------------------------

/// Estimated effort level required to migrate from one model to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationEffort {
    /// Drop-in replacement; no code changes needed.
    Trivial,
    /// Minor prompt adjustments; less than a day of work.
    Minor,
    /// Moderate refactoring; a few days of work.
    Moderate,
    /// Significant rework; may require new evaluation cycles.
    Major,
    /// Complete rewrite; the new model is architecturally different.
    Complete,
}

impl MigrationEffort {
    /// Rough estimate of engineering days required.
    pub fn estimated_days(&self) -> u32 {
        match self {
            MigrationEffort::Trivial => 0,
            MigrationEffort::Minor => 1,
            MigrationEffort::Moderate => 5,
            MigrationEffort::Major => 20,
            MigrationEffort::Complete => 60,
        }
    }
}

// ---------------------------------------------------------------------------
// MigrationPlan
// ---------------------------------------------------------------------------

/// A plan to migrate usage from one model to another.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Source model id.
    pub from_model: String,
    /// Target model id.
    pub to_model: String,
    /// List of human-readable breaking change descriptions.
    pub breaking_changes: Vec<String>,
    /// Estimated engineering effort.
    pub estimated_effort: MigrationEffort,
    /// Unix timestamp (seconds) by which migration should be complete (usually
    /// the source model's sunset date).
    pub deadline: u64,
}

// ---------------------------------------------------------------------------
// ModelLifecycleManager
// ---------------------------------------------------------------------------

/// Registry for model lifecycle metadata with migration planning support.
///
/// # Example
/// ```
/// use llm_cost_dashboard::model_lifecycle::{ModelLifecycleManager, ModelVersion, ModelStatus};
///
/// let mut mgr = ModelLifecycleManager::default();
/// mgr.register(ModelVersion {
///     model_id: "gpt-4".to_string(),
///     version: "gpt-4-0613".to_string(),
///     release_date: 1_686_787_200,
///     status: ModelStatus::Active,
///     successor_id: None,
///     capabilities: vec!["chat".to_string()],
///     context_window: 8192,
/// });
/// assert!(matches!(mgr.current_status("gpt-4"), Some(ModelStatus::Active)));
/// ```
#[derive(Debug, Default)]
pub struct ModelLifecycleManager {
    /// Keyed by `model_id`.
    models: HashMap<String, ModelVersion>,
}

impl ModelLifecycleManager {
    /// Register a new model version. Replaces any existing entry with the same
    /// `model_id`.
    pub fn register(&mut self, version: ModelVersion) {
        self.models.insert(version.model_id.clone(), version);
    }

    /// Return the current status of a model, or `None` if unknown.
    pub fn current_status(&self, model_id: &str) -> Option<&ModelStatus> {
        self.models.get(model_id).map(|v| &v.status)
    }

    /// Return all deprecated models sorted by `sunset_at` ascending.
    pub fn deprecated_models(&self) -> Vec<&ModelVersion> {
        let mut result: Vec<&ModelVersion> = self
            .models
            .values()
            .filter(|v| matches!(v.status, ModelStatus::Deprecated { .. }))
            .collect();
        result.sort_by_key(|v| match &v.status {
            ModelStatus::Deprecated { sunset_at, .. } => *sunset_at,
            _ => u64::MAX,
        });
        result
    }

    /// Return models that will be sunset within `within_days` from `now`
    /// (provided as a Unix timestamp in seconds).
    pub fn at_risk_models(&self, within_days: u32, now: u64) -> Vec<&ModelVersion> {
        let window_secs = within_days as u64 * 86_400;
        let mut result: Vec<&ModelVersion> = self
            .models
            .values()
            .filter(|v| {
                if let ModelStatus::Deprecated { sunset_at, .. } = &v.status {
                    *sunset_at > now && *sunset_at <= now + window_secs
                } else {
                    false
                }
            })
            .collect();
        result.sort_by_key(|v| match &v.status {
            ModelStatus::Deprecated { sunset_at, .. } => *sunset_at,
            _ => u64::MAX,
        });
        result
    }

    /// Create a migration plan from `from` model to `to` model.
    ///
    /// Breaking changes are inferred by comparing capabilities:
    /// - capabilities present in `from` but missing in `to` → feature
    ///   removal.
    /// - context window shrinkage → context window reduction.
    ///
    /// Returns `None` if either model is not registered.
    pub fn create_migration_plan(&self, from: &str, to: &str) -> Option<MigrationPlan> {
        let src = self.models.get(from)?;
        let dst = self.models.get(to)?;

        let mut breaking_changes = Vec::new();

        // Capabilities removed.
        for cap in &src.capabilities {
            if !dst.capabilities.contains(cap) {
                breaking_changes.push(format!(
                    "Capability '{cap}' supported by '{from}' is not available in '{to}'"
                ));
            }
        }

        // Context window reduction.
        if dst.context_window < src.context_window {
            breaking_changes.push(format!(
                "Context window reduced from {} to {} tokens",
                src.context_window, dst.context_window
            ));
        }

        let estimated_effort = match breaking_changes.len() {
            0 => MigrationEffort::Trivial,
            1 => MigrationEffort::Minor,
            2 => MigrationEffort::Moderate,
            3..=4 => MigrationEffort::Major,
            _ => MigrationEffort::Complete,
        };

        let deadline = match &src.status {
            ModelStatus::Deprecated { sunset_at, .. } => *sunset_at,
            _ => u64::MAX,
        };

        Some(MigrationPlan {
            from_model: from.to_string(),
            to_model: to.to_string(),
            breaking_changes,
            estimated_effort,
            deadline,
        })
    }

    /// Return the chain of successor model ids from `model_id` to the latest
    /// active model (inclusive of `model_id`).
    ///
    /// Stops when a model has no successor or is `Active`. Returns an empty
    /// `Vec` if the model is not registered. Guards against cycles (max 32
    /// hops).
    pub fn migration_path(&self, model_id: &str) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = model_id.to_string();
        let mut visited = std::collections::HashSet::new();

        for _ in 0..32 {
            if visited.contains(&current) {
                break; // cycle guard
            }
            visited.insert(current.clone());

            let Some(mv) = self.models.get(&current) else {
                break;
            };
            path.push(current.clone());

            if matches!(mv.status, ModelStatus::Active) {
                break;
            }

            match &mv.successor_id {
                Some(next) => current = next.clone(),
                None => break,
            }
        }

        path
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> ModelLifecycleManager {
        let mut mgr = ModelLifecycleManager::default();

        mgr.register(ModelVersion {
            model_id: "gpt-3".to_string(),
            version: "gpt-3-davinci".to_string(),
            release_date: 1_590_000_000,
            status: ModelStatus::Deprecated {
                deprecated_at: 1_700_000_000,
                sunset_at: 1_720_000_000,
            },
            successor_id: Some("gpt-4".to_string()),
            capabilities: vec!["chat".to_string(), "completion".to_string()],
            context_window: 4096,
        });

        mgr.register(ModelVersion {
            model_id: "gpt-4".to_string(),
            version: "gpt-4-0613".to_string(),
            release_date: 1_686_787_200,
            status: ModelStatus::Active,
            successor_id: None,
            capabilities: vec![
                "chat".to_string(),
                "function_calling".to_string(),
                "vision".to_string(),
            ],
            context_window: 8192,
        });

        mgr.register(ModelVersion {
            model_id: "claude-v1".to_string(),
            version: "claude-v1.3".to_string(),
            release_date: 1_680_000_000,
            status: ModelStatus::Deprecated {
                deprecated_at: 1_710_000_000,
                sunset_at: 1_730_000_000,
            },
            successor_id: Some("claude-v2".to_string()),
            capabilities: vec!["chat".to_string(), "code".to_string()],
            context_window: 100_000,
        });

        mgr.register(ModelVersion {
            model_id: "claude-v2".to_string(),
            version: "claude-v2.1".to_string(),
            release_date: 1_695_000_000,
            status: ModelStatus::Active,
            successor_id: None,
            capabilities: vec![
                "chat".to_string(),
                "code".to_string(),
                "vision".to_string(),
            ],
            context_window: 200_000,
        });

        mgr
    }

    #[test]
    fn test_register_and_status() {
        let mgr = make_manager();
        assert!(matches!(mgr.current_status("gpt-4"), Some(ModelStatus::Active)));
        assert!(matches!(
            mgr.current_status("gpt-3"),
            Some(ModelStatus::Deprecated { .. })
        ));
        assert!(mgr.current_status("unknown").is_none());
    }

    #[test]
    fn test_deprecated_models_sorted() {
        let mgr = make_manager();
        let deprecated = mgr.deprecated_models();
        assert_eq!(deprecated.len(), 2);
        // gpt-3 sunset_at=1_720_000_000 < claude-v1 sunset_at=1_730_000_000
        assert_eq!(deprecated[0].model_id, "gpt-3");
        assert_eq!(deprecated[1].model_id, "claude-v1");
    }

    #[test]
    fn test_at_risk_models() {
        let mgr = make_manager();
        // now = 1_719_000_000; within 30 days = 2_592_000 s => up to 1_721_592_000
        // gpt-3 sunset_at = 1_720_000_000 → within range
        // claude-v1 sunset_at = 1_730_000_000 → outside range
        let now = 1_719_000_000_u64;
        let at_risk = mgr.at_risk_models(30, now);
        assert_eq!(at_risk.len(), 1);
        assert_eq!(at_risk[0].model_id, "gpt-3");
    }

    #[test]
    fn test_at_risk_none_when_already_past() {
        let mgr = make_manager();
        // now is after both sunset dates
        let now = 1_800_000_000_u64;
        let at_risk = mgr.at_risk_models(90, now);
        assert!(at_risk.is_empty());
    }

    #[test]
    fn test_create_migration_plan_trivial() {
        let mgr = make_manager();
        // gpt-4 has a superset of gpt-3's capabilities + larger context.
        let plan = mgr.create_migration_plan("gpt-3", "gpt-4").unwrap();
        assert_eq!(plan.from_model, "gpt-3");
        assert_eq!(plan.to_model, "gpt-4");
        // chat and completion are in gpt-3; gpt-4 lacks "completion"
        assert!(plan
            .breaking_changes
            .iter()
            .any(|c| c.contains("completion")));
        assert_eq!(plan.deadline, 1_720_000_000);
    }

    #[test]
    fn test_create_migration_plan_context_window_reduction() {
        let mut mgr = make_manager();
        // Add a model with smaller context than claude-v1.
        mgr.register(ModelVersion {
            model_id: "small-model".to_string(),
            version: "small-1".to_string(),
            release_date: 1_700_000_000,
            status: ModelStatus::Active,
            successor_id: None,
            capabilities: vec!["chat".to_string(), "code".to_string()],
            context_window: 4096,
        });
        let plan = mgr.create_migration_plan("claude-v1", "small-model").unwrap();
        assert!(plan
            .breaking_changes
            .iter()
            .any(|c| c.contains("Context window reduced")));
    }

    #[test]
    fn test_create_migration_plan_unknown_model() {
        let mgr = make_manager();
        assert!(mgr.create_migration_plan("gpt-4", "unknown").is_none());
        assert!(mgr.create_migration_plan("unknown", "gpt-4").is_none());
    }

    #[test]
    fn test_migration_path_single_hop() {
        let mgr = make_manager();
        let path = mgr.migration_path("gpt-3");
        assert_eq!(path, vec!["gpt-3".to_string(), "gpt-4".to_string()]);
    }

    #[test]
    fn test_migration_path_already_active() {
        let mgr = make_manager();
        let path = mgr.migration_path("gpt-4");
        assert_eq!(path, vec!["gpt-4".to_string()]);
    }

    #[test]
    fn test_migration_path_unknown() {
        let mgr = make_manager();
        let path = mgr.migration_path("nonexistent");
        assert!(path.is_empty());
    }

    #[test]
    fn test_migration_effort_days() {
        assert_eq!(MigrationEffort::Trivial.estimated_days(), 0);
        assert_eq!(MigrationEffort::Minor.estimated_days(), 1);
        assert_eq!(MigrationEffort::Moderate.estimated_days(), 5);
        assert_eq!(MigrationEffort::Major.estimated_days(), 20);
        assert_eq!(MigrationEffort::Complete.estimated_days(), 60);
    }

    #[test]
    fn test_deprecated_list_empty_when_none() {
        let mgr = ModelLifecycleManager::default();
        assert!(mgr.deprecated_models().is_empty());
    }
}
