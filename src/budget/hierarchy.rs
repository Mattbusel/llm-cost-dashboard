//! # Org Hierarchy Budget Tree
//!
//! Models a three-level spend hierarchy: **org → team → project**.
//! Each node has its own [`BudgetEnvelope`].  Spend recorded at the leaf
//! (project) level bubbles up automatically through team and org.
//!
//! ## Design
//!
//! - Adding a project spend also charges the parent team and the org.
//! - Any node can trigger an alert if its own threshold is breached.
//! - Querying an org gives roll-up totals and per-team breakdowns.
//! - The tree is fully in-memory; persistence is the caller's responsibility.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::budget::hierarchy::{OrgTree, TeamConfig, ProjectConfig};
//!
//! let mut tree = OrgTree::new("AcmeCorp", 500.0, 0.80);
//!
//! tree.add_team(TeamConfig { name: "search".into(), limit_usd: 150.0, alert_threshold: 0.80 });
//! tree.add_team(TeamConfig { name: "chat".into(),   limit_usd: 300.0, alert_threshold: 0.75 });
//!
//! tree.add_project(ProjectConfig {
//!     team: "search".into(),
//!     name: "prod-index".into(),
//!     limit_usd: 100.0,
//!     alert_threshold: 0.90,
//! });
//!
//! // Spend $20 on search/prod-index — bubbles up to team and org.
//! let alerts = tree.spend("search", "prod-index", 20.0).unwrap();
//! assert!(alerts.is_empty()); // within all thresholds
//!
//! let summary = tree.summary();
//! assert!((summary.org_spent_usd - 20.0).abs() < 1e-9);
//! ```

use std::collections::HashMap;

use crate::budget::BudgetEnvelope;
use crate::error::DashboardError;

// ── Config types ────────────────────────────────────────────────────────────

/// Configuration for a team node.
#[derive(Debug, Clone)]
pub struct TeamConfig {
    /// Unique team name.
    pub name: String,
    /// Hard spend limit in USD.
    pub limit_usd: f64,
    /// Soft alert threshold (fraction of limit, `[0.0, 1.0]`).
    pub alert_threshold: f64,
}

/// Configuration for a project node.
#[derive(Debug, Clone)]
pub struct ProjectConfig {
    /// Parent team name (must already be added to the tree).
    pub team: String,
    /// Unique project name within the team.
    pub name: String,
    /// Hard spend limit in USD.
    pub limit_usd: f64,
    /// Soft alert threshold (fraction of limit, `[0.0, 1.0]`).
    pub alert_threshold: f64,
}

// ── Alert ───────────────────────────────────────────────────────────────────

/// A budget threshold alert emitted when a node's soft threshold is breached.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetAlert {
    /// The path of the alerted node (e.g. `"AcmeCorp/search/prod-index"`).
    pub path: String,
    /// Fraction of the limit consumed (0.0–1.0).
    pub fill: f64,
    /// Total spent so far (USD).
    pub spent_usd: f64,
    /// The hard limit for this node (USD).
    pub limit_usd: f64,
    /// Whether the hard limit has been breached.
    pub is_over_limit: bool,
}

// ── Tree ────────────────────────────────────────────────────────────────────

/// A project node (leaf).
#[derive(Debug)]
struct ProjectNode {
    name: String,
    envelope: BudgetEnvelope,
}

/// A team node with child projects.
#[derive(Debug)]
struct TeamNode {
    name: String,
    envelope: BudgetEnvelope,
    projects: HashMap<String, ProjectNode>,
}

/// Roll-up summary of the entire org tree.
#[derive(Debug, Clone)]
pub struct OrgSummary {
    /// Org name.
    pub org_name: String,
    /// Total spent by the org (USD).
    pub org_spent_usd: f64,
    /// Org hard limit (USD).
    pub org_limit_usd: f64,
    /// Fraction of org limit consumed.
    pub org_fill: f64,
    /// Per-team summaries.
    pub teams: Vec<TeamSummary>,
}

/// Roll-up summary for a single team.
#[derive(Debug, Clone)]
pub struct TeamSummary {
    /// Team name.
    pub name: String,
    /// Total spent by the team (USD).
    pub spent_usd: f64,
    /// Team hard limit (USD).
    pub limit_usd: f64,
    /// Fraction of team limit consumed.
    pub fill: f64,
    /// Per-project summaries.
    pub projects: Vec<ProjectSummary>,
}

/// Summary for a single project.
#[derive(Debug, Clone)]
pub struct ProjectSummary {
    /// Project name.
    pub name: String,
    /// Total spent by the project (USD).
    pub spent_usd: f64,
    /// Project hard limit (USD).
    pub limit_usd: f64,
    /// Fraction of project limit consumed.
    pub fill: f64,
}

// ── OrgTree ─────────────────────────────────────────────────────────────────

/// Three-level org → team → project budget hierarchy.
///
/// # Panics
///
/// No method panics.
#[derive(Debug)]
pub struct OrgTree {
    org_name: String,
    org_envelope: BudgetEnvelope,
    teams: HashMap<String, TeamNode>,
}

impl OrgTree {
    /// Create a new tree with only an org-level envelope.
    ///
    /// Teams and projects must be added with [`add_team`] and [`add_project`].
    ///
    /// # Arguments
    ///
    /// * `org_name` — Display name for the organisation.
    /// * `limit_usd` — Hard spend limit for the entire org.
    /// * `alert_threshold` — Soft alert fraction (0.0–1.0).
    pub fn new(org_name: impl Into<String>, limit_usd: f64, alert_threshold: f64) -> Self {
        let name = org_name.into();
        Self {
            org_envelope: BudgetEnvelope::new(name.clone(), limit_usd, alert_threshold),
            org_name: name,
            teams: HashMap::new(),
        }
    }

    /// Register a team.
    ///
    /// If a team with the same name already exists its configuration is
    /// replaced (existing spend is preserved).
    pub fn add_team(&mut self, cfg: TeamConfig) {
        let node = self
            .teams
            .entry(cfg.name.clone())
            .or_insert_with(|| TeamNode {
                name: cfg.name.clone(),
                envelope: BudgetEnvelope::new(&cfg.name, cfg.limit_usd, cfg.alert_threshold),
                projects: HashMap::new(),
            });
        // Update limits in case of re-registration
        node.envelope.limit_usd = cfg.limit_usd;
        node.envelope.alert_threshold = cfg.alert_threshold.clamp(0.0, 1.0);
    }

    /// Register a project under an existing team.
    ///
    /// Returns `DashboardError::Ledger` if the team does not exist.
    pub fn add_project(&mut self, cfg: ProjectConfig) -> Result<(), DashboardError> {
        let team = self
            .teams
            .get_mut(&cfg.team)
            .ok_or_else(|| DashboardError::Ledger(format!("unknown team: {}", cfg.team)))?;
        let project = team
            .projects
            .entry(cfg.name.clone())
            .or_insert_with(|| ProjectNode {
                name: cfg.name.clone(),
                envelope: BudgetEnvelope::new(&cfg.name, cfg.limit_usd, cfg.alert_threshold),
            });
        project.envelope.limit_usd = cfg.limit_usd;
        project.envelope.alert_threshold = cfg.alert_threshold.clamp(0.0, 1.0);
        Ok(())
    }

    /// Record a spend of `amount_usd` against a specific project.
    ///
    /// The spend bubbles up through the team and org envelopes automatically.
    /// Returns a list of [`BudgetAlert`]s for any node whose soft threshold
    /// has just been crossed.  Hard-limit breaches are returned as an
    /// `Err(DashboardError::BudgetExceeded)` from the **first** node that
    /// would be exceeded (project, then team, then org — in that order).
    ///
    /// On a hard-limit error, no spend is recorded at any level.
    ///
    /// # Arguments
    ///
    /// * `team` — Team name.
    /// * `project` — Project name within the team.
    /// * `amount_usd` — Spend to record (must be ≥ 0).
    pub fn spend(
        &mut self,
        team: &str,
        project: &str,
        amount_usd: f64,
    ) -> Result<Vec<BudgetAlert>, DashboardError> {
        if amount_usd < 0.0 {
            return Err(DashboardError::Ledger("negative spend amount".into()));
        }

        // Snapshot state before changes so we can roll back on error
        let org_spent_before = self.org_envelope.spent_usd;

        let team_node = self
            .teams
            .get_mut(team)
            .ok_or_else(|| DashboardError::Ledger(format!("unknown team: {}", team)))?;

        let team_spent_before = team_node.envelope.spent_usd;

        let project_node = team_node
            .projects
            .get_mut(project)
            .ok_or_else(|| {
                DashboardError::Ledger(format!("unknown project: {}/{}", team, project))
            })?;

        // --- Dry-run hard-limit checks before mutating ---
        // Project
        if project_node.envelope.spent_usd + amount_usd > project_node.envelope.limit_usd {
            return Err(DashboardError::BudgetExceeded {
                spent: project_node.envelope.spent_usd + amount_usd,
                limit: project_node.envelope.limit_usd,
            });
        }
        // Team
        if team_node.envelope.spent_usd + amount_usd > team_node.envelope.limit_usd {
            return Err(DashboardError::BudgetExceeded {
                spent: team_node.envelope.spent_usd + amount_usd,
                limit: team_node.envelope.limit_usd,
            });
        }
        // Org
        if self.org_envelope.spent_usd + amount_usd > self.org_envelope.limit_usd {
            return Err(DashboardError::BudgetExceeded {
                spent: self.org_envelope.spent_usd + amount_usd,
                limit: self.org_envelope.limit_usd,
            });
        }

        // --- Record spend, collect soft alerts ---
        let mut alerts: Vec<BudgetAlert> = Vec::new();

        let proj_threshold_before =
            project_node.envelope.spent_usd / project_node.envelope.limit_usd;
        project_node.envelope.spent_usd += amount_usd;
        let proj_fill = project_node.envelope.spent_usd / project_node.envelope.limit_usd;
        if proj_fill >= project_node.envelope.alert_threshold
            && proj_threshold_before < project_node.envelope.alert_threshold
        {
            alerts.push(BudgetAlert {
                path: format!("{}/{}/{}", self.org_name, team, project),
                fill: proj_fill,
                spent_usd: project_node.envelope.spent_usd,
                limit_usd: project_node.envelope.limit_usd,
                is_over_limit: false,
            });
        }

        let team_threshold_before = team_spent_before / team_node.envelope.limit_usd;
        team_node.envelope.spent_usd += amount_usd;
        let team_fill = team_node.envelope.spent_usd / team_node.envelope.limit_usd;
        if team_fill >= team_node.envelope.alert_threshold
            && team_threshold_before < team_node.envelope.alert_threshold
        {
            alerts.push(BudgetAlert {
                path: format!("{}/{}", self.org_name, team),
                fill: team_fill,
                spent_usd: team_node.envelope.spent_usd,
                limit_usd: team_node.envelope.limit_usd,
                is_over_limit: false,
            });
        }

        let org_threshold_before = org_spent_before / self.org_envelope.limit_usd;
        self.org_envelope.spent_usd += amount_usd;
        let org_fill = self.org_envelope.spent_usd / self.org_envelope.limit_usd;
        if org_fill >= self.org_envelope.alert_threshold
            && org_threshold_before < self.org_envelope.alert_threshold
        {
            alerts.push(BudgetAlert {
                path: self.org_name.clone(),
                fill: org_fill,
                spent_usd: self.org_envelope.spent_usd,
                limit_usd: self.org_envelope.limit_usd,
                is_over_limit: false,
            });
        }

        Ok(alerts)
    }

    /// Return a roll-up summary of the entire tree.
    pub fn summary(&self) -> OrgSummary {
        let org_fill = if self.org_envelope.limit_usd > 0.0 {
            self.org_envelope.spent_usd / self.org_envelope.limit_usd
        } else {
            0.0
        };

        let mut teams: Vec<TeamSummary> = self
            .teams
            .values()
            .map(|t| {
                let team_fill = if t.envelope.limit_usd > 0.0 {
                    t.envelope.spent_usd / t.envelope.limit_usd
                } else {
                    0.0
                };
                let projects: Vec<ProjectSummary> = t
                    .projects
                    .values()
                    .map(|p| {
                        let proj_fill = if p.envelope.limit_usd > 0.0 {
                            p.envelope.spent_usd / p.envelope.limit_usd
                        } else {
                            0.0
                        };
                        ProjectSummary {
                            name: p.name.clone(),
                            spent_usd: p.envelope.spent_usd,
                            limit_usd: p.envelope.limit_usd,
                            fill: proj_fill,
                        }
                    })
                    .collect();
                TeamSummary {
                    name: t.name.clone(),
                    spent_usd: t.envelope.spent_usd,
                    limit_usd: t.envelope.limit_usd,
                    fill: team_fill,
                    projects,
                }
            })
            .collect();
        teams.sort_by(|a, b| b.fill.partial_cmp(&a.fill).unwrap_or(std::cmp::Ordering::Equal));

        OrgSummary {
            org_name: self.org_name.clone(),
            org_spent_usd: self.org_envelope.spent_usd,
            org_limit_usd: self.org_envelope.limit_usd,
            org_fill,
            teams,
        }
    }

    /// Return all teams that have exceeded `fill_fraction` of their limits,
    /// sorted by fill fraction descending.
    pub fn teams_over_threshold(&self, fill_fraction: f64) -> Vec<TeamSummary> {
        self.summary()
            .teams
            .into_iter()
            .filter(|t| t.fill >= fill_fraction)
            .collect()
    }

    /// Reset all spend counters to zero across the entire tree.
    ///
    /// Used for period resets (e.g. monthly billing cycle rollover).
    pub fn reset_all(&mut self) {
        self.org_envelope.spent_usd = 0.0;
        for team in self.teams.values_mut() {
            team.envelope.spent_usd = 0.0;
            for project in team.projects.values_mut() {
                project.envelope.spent_usd = 0.0;
            }
        }
    }

    /// Org name.
    pub fn org_name(&self) -> &str {
        &self.org_name
    }

    /// Org total spent.
    pub fn org_spent_usd(&self) -> f64 {
        self.org_envelope.spent_usd
    }

    /// Org hard limit.
    pub fn org_limit_usd(&self) -> f64 {
        self.org_envelope.limit_usd
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn example_tree() -> OrgTree {
        let mut tree = OrgTree::new("ACME", 1000.0, 0.80);
        tree.add_team(TeamConfig {
            name: "eng".into(),
            limit_usd: 500.0,
            alert_threshold: 0.80,
        });
        tree.add_team(TeamConfig {
            name: "data".into(),
            limit_usd: 400.0,
            alert_threshold: 0.75,
        });
        tree.add_project(ProjectConfig {
            team: "eng".into(),
            name: "prod".into(),
            limit_usd: 300.0,
            alert_threshold: 0.90,
        })
        .unwrap();
        tree.add_project(ProjectConfig {
            team: "eng".into(),
            name: "staging".into(),
            limit_usd: 150.0,
            alert_threshold: 0.90,
        })
        .unwrap();
        tree.add_project(ProjectConfig {
            team: "data".into(),
            name: "analytics".into(),
            limit_usd: 200.0,
            alert_threshold: 0.75,
        })
        .unwrap();
        tree
    }

    #[test]
    fn spend_bubbles_up_to_org() {
        let mut tree = example_tree();
        let alerts = tree.spend("eng", "prod", 50.0).unwrap();
        assert!(alerts.is_empty());

        let summary = tree.summary();
        assert!((summary.org_spent_usd - 50.0).abs() < 1e-9);
        let eng = summary.teams.iter().find(|t| t.name == "eng").unwrap();
        assert!((eng.spent_usd - 50.0).abs() < 1e-9);
    }

    #[test]
    fn project_hard_limit_blocks_spend() {
        let mut tree = example_tree();
        // prod limit is $300 — spending $350 should fail
        let result = tree.spend("eng", "prod", 350.0);
        assert!(result.is_err());
        // Nothing should have been charged
        assert!(tree.org_spent_usd().abs() < 1e-9);
    }

    #[test]
    fn team_hard_limit_blocks_spend() {
        let mut tree = example_tree();
        // eng limit is $500
        let result = tree.spend("eng", "prod", 501.0);
        assert!(result.is_err());
    }

    #[test]
    fn org_hard_limit_blocks_spend() {
        let mut tree = example_tree();
        // org limit is $1000
        let result = tree.spend("eng", "prod", 1001.0);
        assert!(result.is_err());
    }

    #[test]
    fn soft_alert_fires_when_threshold_crossed() {
        // Org limit is 100 so that spending 55 (55%) also crosses the org
        // 50% soft threshold, giving alerts at all three levels.
        let mut tree = OrgTree::new("ORG", 100.0, 0.50);
        tree.add_team(TeamConfig {
            name: "team1".into(),
            limit_usd: 100.0,
            alert_threshold: 0.50,
        });
        tree.add_project(ProjectConfig {
            team: "team1".into(),
            name: "proj1".into(),
            limit_usd: 100.0,
            alert_threshold: 0.50,
        })
        .unwrap();

        // Spend 40 — below 50% threshold
        let alerts = tree.spend("team1", "proj1", 40.0).unwrap();
        assert!(alerts.is_empty());

        // Spend 15 more → 55% total — threshold crossed on project AND team AND org
        let alerts = tree.spend("team1", "proj1", 15.0).unwrap();
        assert_eq!(alerts.len(), 3, "project, team, and org should alert");
    }

    #[test]
    fn soft_alert_does_not_fire_twice() {
        let mut tree = OrgTree::new("ORG", 1000.0, 0.50);
        tree.add_team(TeamConfig {
            name: "t".into(),
            limit_usd: 100.0,
            alert_threshold: 0.50,
        });
        tree.add_project(ProjectConfig {
            team: "t".into(),
            name: "p".into(),
            limit_usd: 100.0,
            alert_threshold: 0.50,
        })
        .unwrap();

        tree.spend("t", "p", 60.0).unwrap(); // crosses threshold
        let alerts = tree.spend("t", "p", 5.0).unwrap(); // still above, but no re-trigger
        assert!(alerts.is_empty());
    }

    #[test]
    fn unknown_team_returns_error() {
        let mut tree = example_tree();
        assert!(tree.spend("nonexistent", "prod", 10.0).is_err());
    }

    #[test]
    fn unknown_project_returns_error() {
        let mut tree = example_tree();
        assert!(tree.spend("eng", "nonexistent", 10.0).is_err());
    }

    #[test]
    fn reset_all_clears_all_spend() {
        let mut tree = example_tree();
        tree.spend("eng", "prod", 100.0).unwrap();
        tree.spend("data", "analytics", 50.0).unwrap();
        tree.reset_all();

        let summary = tree.summary();
        assert!(summary.org_spent_usd.abs() < 1e-9);
        for team in &summary.teams {
            assert!(team.spent_usd.abs() < 1e-9);
            for project in &team.projects {
                assert!(project.spent_usd.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn summary_teams_sorted_by_fill_descending() {
        let mut tree = example_tree();
        tree.spend("data", "analytics", 180.0).unwrap(); // data: 180/400 = 45%
        tree.spend("eng", "prod", 50.0).unwrap();         // eng:  50/500 = 10%

        let summary = tree.summary();
        assert_eq!(summary.teams[0].name, "data"); // data has higher fill
    }

    #[test]
    fn teams_over_threshold_filters_correctly() {
        let mut tree = example_tree();
        // eng team limit=500; prod limit=300, staging limit=150.
        // Spend 250 on prod + 150 on staging = 400 total → 400/500 = 80%.
        tree.spend("eng", "prod", 250.0).unwrap();
        tree.spend("eng", "staging", 150.0).unwrap();
        tree.spend("data", "analytics", 100.0).unwrap(); // 100/400 = 25%

        let over = tree.teams_over_threshold(0.70);
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].name, "eng");
    }

    #[test]
    fn negative_spend_returns_error() {
        let mut tree = example_tree();
        assert!(tree.spend("eng", "prod", -1.0).is_err());
    }

    #[test]
    fn multiple_projects_per_team_independent() {
        let mut tree = example_tree();
        tree.spend("eng", "prod", 100.0).unwrap();
        tree.spend("eng", "staging", 50.0).unwrap();

        let summary = tree.summary();
        let eng = summary.teams.iter().find(|t| t.name == "eng").unwrap();
        assert!((eng.spent_usd - 150.0).abs() < 1e-9); // team total
    }
}
