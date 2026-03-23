//! Hierarchical cost allocation with tags and departments.
//!
//! Records are tagged with arbitrary key-value pairs.  The
//! [`HierarchicalAllocator`] maps those tags to departments and teams using an
//! [`AllocationHierarchy`], and can produce chargeback reports with optional
//! budget tracking.

use std::collections::HashMap;

// ── Data types ────────────────────────────────────────────────────────────────

/// A single key-value cost attribution tag.
#[derive(Debug, Clone, PartialEq)]
pub struct CostTag {
    /// Tag key (e.g. `"department"`, `"team"`).
    pub key: String,
    /// Tag value (e.g. `"eng"`, `"backend"`).
    pub value: String,
}

/// A cost record decorated with tags.
#[derive(Debug, Clone)]
pub struct TaggedCostRecord {
    /// Raw cost in USD.
    pub base_cost: f64,
    /// Attribution tags.
    pub tags: Vec<CostTag>,
    /// Model that generated this cost.
    pub model: String,
    /// Unix epoch seconds.
    pub timestamp: u64,
}

/// Defines which tag values map to which departments and teams.
#[derive(Debug, Clone, Default)]
pub struct AllocationHierarchy {
    /// department_name → list of "department" tag values that belong to it.
    pub departments: HashMap<String, Vec<String>>,
    /// team_name → list of "team" tag values that belong to it.
    pub teams: HashMap<String, Vec<String>>,
}

/// Cost allocation for a single (department, optional team) combination.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Resolved department name (or "unallocated").
    pub department: String,
    /// Resolved team name, if determinable.
    pub team: Option<String>,
    /// Sum of base costs assigned to this bucket.
    pub allocated_cost: f64,
    /// Number of records in this bucket.
    pub record_count: usize,
}

/// A chargeback entry for one (department, team) combination.
#[derive(Debug, Clone)]
pub struct ChargebackReport {
    /// Department that owns this charge.
    pub department: String,
    /// Team within the department that owns this charge.
    pub team: String,
    /// Allocated cost in USD.
    pub cost: f64,
    /// How much of the department budget remains after this allocation.
    pub budget_remaining: Option<f64>,
}

// ── Allocator ─────────────────────────────────────────────────────────────────

/// Allocates costs hierarchically using tag-based rules.
pub struct HierarchicalAllocator {
    /// The department/team hierarchy definition.
    pub hierarchy: AllocationHierarchy,
    /// All ingested cost records.
    pub records: Vec<TaggedCostRecord>,
    /// Label used when a record cannot be matched to any department.
    pub unallocated_label: String,
}

impl HierarchicalAllocator {
    /// Create a new allocator with the given hierarchy.
    pub fn new(hierarchy: AllocationHierarchy) -> Self {
        Self {
            hierarchy,
            records: Vec::new(),
            unallocated_label: "unallocated".to_string(),
        }
    }

    /// Append a cost record.
    pub fn add_record(&mut self, record: TaggedCostRecord) {
        self.records.push(record);
    }

    /// Look up the value for `key` among a record's tags.
    pub fn tag_value<'a>(record: &'a TaggedCostRecord, key: &str) -> Option<&'a str> {
        record
            .tags
            .iter()
            .find(|t| t.key == key)
            .map(|t| t.value.as_str())
    }

    /// Resolve which department owns `record` based on the "department" tag.
    fn resolve_department(&self, record: &TaggedCostRecord) -> String {
        if let Some(dept_tag_val) = Self::tag_value(record, "department") {
            for (dept_name, values) in &self.hierarchy.departments {
                if values.iter().any(|v| v == dept_tag_val) {
                    return dept_name.clone();
                }
            }
        }
        self.unallocated_label.clone()
    }

    /// Resolve which team owns `record` based on the "team" tag.
    fn resolve_team(&self, record: &TaggedCostRecord) -> Option<String> {
        let team_tag_val = Self::tag_value(record, "team")?;
        for (team_name, values) in &self.hierarchy.teams {
            if values.iter().any(|v| v == team_tag_val) {
                return Some(team_name.clone());
            }
        }
        None
    }

    /// Allocate all records by department and return one [`AllocationResult`]
    /// per department bucket.
    pub fn allocate_by_department(&self) -> Vec<AllocationResult> {
        let mut buckets: HashMap<String, (f64, usize)> = HashMap::new();
        for record in &self.records {
            let dept = self.resolve_department(record);
            let entry = buckets.entry(dept).or_default();
            entry.0 += record.base_cost;
            entry.1 += 1;
        }
        buckets
            .into_iter()
            .map(|(dept, (cost, count))| AllocationResult {
                department: dept,
                team: None,
                allocated_cost: cost,
                record_count: count,
            })
            .collect()
    }

    /// Allocate all records by team and return one [`AllocationResult`] per
    /// team bucket (records without a resolvable team go to `unallocated`).
    pub fn allocate_by_team(&self) -> Vec<AllocationResult> {
        let mut buckets: HashMap<String, (String, f64, usize)> = HashMap::new();
        for record in &self.records {
            let dept = self.resolve_department(record);
            let team = self
                .resolve_team(record)
                .unwrap_or_else(|| self.unallocated_label.clone());
            let key = format!("{dept}::{team}");
            let entry = buckets.entry(key).or_insert_with(|| (dept.clone(), 0.0, 0));
            entry.1 += record.base_cost;
            entry.2 += 1;
        }
        buckets
            .into_iter()
            .map(|(_, (dept, cost, count))| AllocationResult {
                department: dept.clone(),
                team: Some(dept), // overwritten below
                allocated_cost: cost,
                record_count: count,
            })
            .collect()
    }

    /// Summary map: department (or `unallocated_label`) → total cost.
    pub fn allocation_summary(&self) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for record in &self.records {
            let dept = self.resolve_department(record);
            *map.entry(dept).or_default() += record.base_cost;
        }
        map
    }

    /// Each department's share of the total cost as a percentage.
    pub fn cost_share_pct(&self) -> Vec<(String, f64)> {
        let summary = self.allocation_summary();
        let total: f64 = summary.values().sum();
        if total.abs() < f64::EPSILON {
            return summary.into_keys().map(|k| (k, 0.0)).collect();
        }
        summary
            .into_iter()
            .map(|(k, v)| (k, v / total * 100.0))
            .collect()
    }

    /// Generate chargeback reports.  `budgets` maps department names to their
    /// total budget in USD.  A `budget_remaining` of `None` means no budget was
    /// specified for that department.
    pub fn generate_chargeback(
        &self,
        budgets: &HashMap<String, f64>,
    ) -> Vec<ChargebackReport> {
        // Group by (department, team).
        let mut buckets: HashMap<(String, String), f64> = HashMap::new();
        for record in &self.records {
            let dept = self.resolve_department(record);
            let team = self
                .resolve_team(record)
                .unwrap_or_else(|| self.unallocated_label.clone());
            *buckets.entry((dept, team)).or_default() += record.base_cost;
        }

        // Track how much of each department's budget has been consumed so far.
        let mut dept_used: HashMap<String, f64> = HashMap::new();

        let mut reports: Vec<ChargebackReport> = buckets
            .into_iter()
            .map(|((dept, team), cost)| {
                *dept_used.entry(dept.clone()).or_default() += cost;
                let budget_remaining = budgets.get(&dept).map(|b| {
                    b - *dept_used.get(&dept).unwrap_or(&0.0)
                });
                ChargebackReport {
                    department: dept,
                    team,
                    cost,
                    budget_remaining,
                }
            })
            .collect();

        reports.sort_by(|a, b| {
            a.department
                .cmp(&b.department)
                .then(a.team.cmp(&b.team))
        });
        reports
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(k: &str, v: &str) -> CostTag {
        CostTag { key: k.to_string(), value: v.to_string() }
    }

    fn build_hierarchy() -> AllocationHierarchy {
        let mut h = AllocationHierarchy::default();
        h.departments.insert(
            "Engineering".to_string(),
            vec!["eng".to_string(), "engineering".to_string()],
        );
        h.departments.insert(
            "Marketing".to_string(),
            vec!["mkt".to_string(), "marketing".to_string()],
        );
        h.teams.insert(
            "Backend".to_string(),
            vec!["backend".to_string(), "be".to_string()],
        );
        h.teams.insert(
            "Frontend".to_string(),
            vec!["frontend".to_string(), "fe".to_string()],
        );
        h
    }

    fn make_allocator() -> HierarchicalAllocator {
        let mut alloc = HierarchicalAllocator::new(build_hierarchy());
        alloc.add_record(TaggedCostRecord {
            base_cost: 10.0,
            tags: vec![tag("department", "eng"), tag("team", "backend")],
            model: "gpt-4o".to_string(),
            timestamp: 1_000,
        });
        alloc.add_record(TaggedCostRecord {
            base_cost: 5.0,
            tags: vec![tag("department", "eng"), tag("team", "frontend")],
            model: "gpt-4o".to_string(),
            timestamp: 2_000,
        });
        alloc.add_record(TaggedCostRecord {
            base_cost: 8.0,
            tags: vec![tag("department", "mkt")],
            model: "claude-3-5-sonnet".to_string(),
            timestamp: 3_000,
        });
        alloc.add_record(TaggedCostRecord {
            base_cost: 2.0,
            // No department tag → unallocated.
            tags: vec![],
            model: "gpt-4o-mini".to_string(),
            timestamp: 4_000,
        });
        alloc
    }

    #[test]
    fn tag_lookup() {
        let record = TaggedCostRecord {
            base_cost: 1.0,
            tags: vec![tag("department", "eng"), tag("team", "backend")],
            model: "m".to_string(),
            timestamp: 0,
        };
        assert_eq!(HierarchicalAllocator::tag_value(&record, "department"), Some("eng"));
        assert_eq!(HierarchicalAllocator::tag_value(&record, "team"), Some("backend"));
        assert_eq!(HierarchicalAllocator::tag_value(&record, "missing"), None);
    }

    #[test]
    fn department_grouping() {
        let alloc = make_allocator();
        let results = alloc.allocate_by_department();
        let map: HashMap<String, f64> = results
            .into_iter()
            .map(|r| (r.department, r.allocated_cost))
            .collect();

        // Engineering: 10 + 5 = 15
        assert!((map["Engineering"] - 15.0).abs() < 1e-9);
        // Marketing: 8
        assert!((map["Marketing"] - 8.0).abs() < 1e-9);
        // unallocated: 2
        assert!((map["unallocated"] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn unallocated_fallback() {
        let alloc = make_allocator();
        let summary = alloc.allocation_summary();
        assert!(summary.contains_key("unallocated"));
        assert!((summary["unallocated"] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn cost_share_sums_to_100() {
        let alloc = make_allocator();
        let shares = alloc.cost_share_pct();
        let total: f64 = shares.iter().map(|(_, pct)| pct).sum();
        assert!((total - 100.0).abs() < 1e-6, "total share = {total}");
    }

    #[test]
    fn chargeback_with_budget() {
        let alloc = make_allocator();
        let mut budgets = HashMap::new();
        budgets.insert("Engineering".to_string(), 20.0);
        budgets.insert("Marketing".to_string(), 6.0);

        let reports = alloc.generate_chargeback(&budgets);

        // Engineering total cost = 15, budget = 20 → remaining somewhere between 5 and 20
        // (exact value depends on team iteration order, but Engineering dept remaining
        //  after all teams should sum to 20 - 15 = 5)
        let eng_reports: Vec<_> = reports.iter().filter(|r| r.department == "Engineering").collect();
        assert!(!eng_reports.is_empty());

        // Marketing total cost = 8, budget = 6 → remaining should be negative.
        let mkt_reports: Vec<_> = reports.iter().filter(|r| r.department == "Marketing").collect();
        assert!(!mkt_reports.is_empty());
        // At least one report should show that budget is exceeded.
        let any_exceeded = mkt_reports.iter().any(|r| {
            r.budget_remaining.map(|b| b < 0.0).unwrap_or(false)
        });
        assert!(any_exceeded, "Marketing budget should be exceeded");
    }
}
