//! # Chargeback / Cost Allocation
//!
//! Maps raw LLM cost records onto organisational departments and projects via a
//! flexible rule engine, then produces split-cost reports suitable for
//! internal chargeback or showback workflows.
//!
//! ## Flow
//!
//! ```text
//! CostAllocation  →  ChargebackEngine::allocate  →  Vec<ChargebackEntry>
//!                                                         │
//!                                          summary_report / to_csv
//! ```
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::chargeback::{
//!     ChargebackEngine, ChargebackRule, CostAllocation, RuleMatcher,
//! };
//! use std::collections::HashMap;
//!
//! let mut engine = ChargebackEngine::new();
//! let mut pcts = HashMap::new();
//! pcts.insert("engineering".to_string(), 1.0);
//! engine.add_rule(ChargebackRule {
//!     id: "r1".to_string(),
//!     name: "catch-all".to_string(),
//!     matcher: RuleMatcher { tenant_pattern: Some("*".to_string()), model_pattern: None, min_cost: None },
//!     allocation_pcts: pcts,
//! }).unwrap();
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── CostAllocation ────────────────────────────────────────────────────────────

/// A single raw cost record that the chargeback engine will process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAllocation {
    /// Tenant / customer identifier.
    pub tenant: String,
    /// Department that generated the spend.
    pub department: String,
    /// Project that generated the spend.
    pub project: String,
    /// Model used for inference.
    pub model: String,
    /// Start of the billing period (Unix timestamp, seconds).
    pub period_start: u64,
    /// End of the billing period (Unix timestamp, seconds).
    pub period_end: u64,
    /// Total tokens consumed.
    pub tokens: u64,
    /// Total cost in USD.
    pub cost: f64,
    /// Fraction of overall spend allocated to this record (0.0–1.0).
    pub allocation_pct: f64,
}

// ── RuleMatcher ───────────────────────────────────────────────────────────────

/// Predicate that determines whether a [`ChargebackRule`] applies to a given record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleMatcher {
    /// Glob-style pattern matched against `CostAllocation::tenant`.
    /// `None` means match any tenant.  Use `"*"` as a catch-all wildcard.
    pub tenant_pattern: Option<String>,
    /// Glob-style pattern matched against `CostAllocation::model`.
    /// `None` means match any model.
    pub model_pattern: Option<String>,
    /// Minimum cost in USD; records below this threshold are not matched.
    pub min_cost: Option<f64>,
}

impl RuleMatcher {
    /// Return `true` if this matcher applies to the given record attributes.
    ///
    /// Pattern matching supports `'*'` as a wildcard matching any substring.
    pub fn matches(&self, tenant: &str, model: &str, cost: f64) -> bool {
        if let Some(ref pat) = self.tenant_pattern {
            if !glob_match(pat, tenant) {
                return false;
            }
        }
        if let Some(ref pat) = self.model_pattern {
            if !glob_match(pat, model) {
                return false;
            }
        }
        if let Some(min) = self.min_cost {
            if cost < min {
                return false;
            }
        }
        true
    }
}

/// Simple glob match supporting `'*'` as a multi-character wildcard.
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    // Split on '*' and check that the parts appear in order.
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.is_empty() {
        return true;
    }
    let mut remaining = text;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            // First segment must match at the start.
            if !remaining.starts_with(part) {
                return false;
            }
            remaining = &remaining[part.len()..];
        } else {
            match remaining.find(part) {
                Some(pos) => remaining = &remaining[pos + part.len()..],
                None => return false,
            }
        }
    }
    // If the pattern does not end with '*', the last part must be a suffix match.
    if !pattern.ends_with('*') {
        let last = parts.last().unwrap_or(&"");
        if !last.is_empty() && !text.ends_with(last) {
            return false;
        }
    }
    true
}

// ── ChargebackRule ────────────────────────────────────────────────────────────

/// A named rule that splits a matched cost record across departments.
#[derive(Debug, Clone)]
pub struct ChargebackRule {
    /// Unique rule identifier.
    pub id: String,
    /// Human-readable rule name.
    pub name: String,
    /// Predicate that selects which cost records this rule applies to.
    pub matcher: RuleMatcher,
    /// Department → fraction mapping.  Values must sum to exactly 1.0 (±1e-6).
    pub allocation_pcts: HashMap<String, f64>,
}

// ── ChargebackEntry ───────────────────────────────────────────────────────────

/// The result of applying a rule to a cost record — one entry per department.
#[derive(Debug, Clone)]
pub struct ChargebackEntry {
    /// Original cost record.
    pub original: CostAllocation,
    /// Rule that produced this entry.
    pub rule_id: String,
    /// Department receiving the allocated cost.
    pub department: String,
    /// Fraction of `original.cost` allocated to `department`.
    pub allocated_cost: f64,
}

// ── ChargebackReport ──────────────────────────────────────────────────────────

/// Aggregated cost summary produced by [`ChargebackEngine::summary_report`].
#[derive(Debug, Default)]
pub struct ChargebackReport {
    /// Total allocated cost per department.
    pub by_department: HashMap<String, f64>,
    /// Total allocated cost per project.
    pub by_project: HashMap<String, f64>,
    /// Total allocated cost per model.
    pub by_model: HashMap<String, f64>,
    /// Sum of all allocated costs.
    pub total_allocated: f64,
    /// Total cost that had no matching rule.
    pub unallocated: f64,
}

// ── ChargebackError ───────────────────────────────────────────────────────────

/// Errors produced by the chargeback engine.
#[derive(Debug, thiserror::Error)]
pub enum ChargebackError {
    /// The `allocation_pcts` values in a rule do not sum to 1.0.
    #[error("invalid allocation: percentages sum to {sum:.4} (must be 1.0)")]
    InvalidAllocation {
        /// Actual sum of the allocation percentages.
        sum: f64,
    },
    /// A rule with the same ID has already been registered.
    #[error("duplicate rule id: {0}")]
    DuplicateRule(String),
    /// No rule matched a cost record (informational; not raised by default).
    #[error("no matching rule")]
    NoMatchingRule,
}

// ── ChargebackEngine ──────────────────────────────────────────────────────────

/// Rule engine that allocates cost records to departments.
#[derive(Default)]
pub struct ChargebackEngine {
    rules: Vec<ChargebackRule>,
}

impl ChargebackEngine {
    /// Create an empty engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new rule.
    ///
    /// Returns [`ChargebackError::InvalidAllocation`] if `allocation_pcts` does
    /// not sum to 1.0 (±1e-6), or [`ChargebackError::DuplicateRule`] if a rule
    /// with the same ID was already added.
    pub fn add_rule(&mut self, rule: ChargebackRule) -> Result<(), ChargebackError> {
        // Validate uniqueness.
        if self.rules.iter().any(|r| r.id == rule.id) {
            return Err(ChargebackError::DuplicateRule(rule.id.clone()));
        }

        // Validate that percentages sum to 1.0.
        let sum: f64 = rule.allocation_pcts.values().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(ChargebackError::InvalidAllocation { sum });
        }

        self.rules.push(rule);
        Ok(())
    }

    /// Apply matching rules to all records and return one [`ChargebackEntry`]
    /// per (record, department) pair.
    ///
    /// Rules are evaluated in insertion order; the **first** matching rule wins.
    /// Records with no matching rule are silently skipped (they contribute to
    /// `unallocated` in the summary report).
    pub fn allocate(&self, usage: &[CostAllocation]) -> Vec<ChargebackEntry> {
        let mut entries = Vec::new();
        for record in usage {
            if let Some(rule) = self
                .rules
                .iter()
                .find(|r| r.matcher.matches(&record.tenant, &record.model, record.cost))
            {
                for (dept, pct) in &rule.allocation_pcts {
                    entries.push(ChargebackEntry {
                        original: record.clone(),
                        rule_id: rule.id.clone(),
                        department: dept.clone(),
                        allocated_cost: record.cost * pct,
                    });
                }
            }
        }
        entries
    }

    /// Aggregate chargeback entries into a report.
    ///
    /// `all_usage` is the original (pre-allocation) slice; it is used to
    /// compute the `unallocated` total.
    pub fn summary_report(&self, entries: &[ChargebackEntry]) -> ChargebackReport {
        let mut report = ChargebackReport::default();
        for entry in entries {
            *report
                .by_department
                .entry(entry.department.clone())
                .or_insert(0.0) += entry.allocated_cost;
            *report
                .by_project
                .entry(entry.original.project.clone())
                .or_insert(0.0) += entry.allocated_cost;
            *report
                .by_model
                .entry(entry.original.model.clone())
                .or_insert(0.0) += entry.allocated_cost;
            report.total_allocated += entry.allocated_cost;
        }
        report
    }

    /// Serialise entries to CSV with a header row.
    ///
    /// Columns: `tenant,department,project,model,period_start,period_end,tokens,original_cost,rule_id,allocated_department,allocated_cost`
    pub fn to_csv(&self, entries: &[ChargebackEntry]) -> String {
        let mut out = String::from(
            "tenant,department,project,model,period_start,period_end,tokens,\
             original_cost,rule_id,allocated_department,allocated_cost\n",
        );
        for e in entries {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{:.6},{},{},{:.6}\n",
                e.original.tenant,
                e.original.department,
                e.original.project,
                e.original.model,
                e.original.period_start,
                e.original.period_end,
                e.original.tokens,
                e.original.cost,
                e.rule_id,
                e.department,
                e.allocated_cost,
            ));
        }
        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(tenant: &str, model: &str, cost: f64) -> CostAllocation {
        CostAllocation {
            tenant: tenant.to_string(),
            department: "unknown".to_string(),
            project: "proj-a".to_string(),
            model: model.to_string(),
            period_start: 1_700_000_000,
            period_end: 1_700_086_400,
            tokens: 1000,
            cost,
            allocation_pct: 1.0,
        }
    }

    fn two_dept_rule(id: &str, tenant_pattern: &str) -> ChargebackRule {
        let mut pcts = HashMap::new();
        pcts.insert("eng".to_string(), 0.6);
        pcts.insert("ops".to_string(), 0.4);
        ChargebackRule {
            id: id.to_string(),
            name: "split".to_string(),
            matcher: RuleMatcher {
                tenant_pattern: Some(tenant_pattern.to_string()),
                model_pattern: None,
                min_cost: None,
            },
            allocation_pcts: pcts,
        }
    }

    #[test]
    fn rule_matching_wildcard() {
        let matcher = RuleMatcher {
            tenant_pattern: Some("acme-*".to_string()),
            model_pattern: None,
            min_cost: None,
        };
        assert!(matcher.matches("acme-corp", "gpt-4", 1.0));
        assert!(!matcher.matches("other-corp", "gpt-4", 1.0));
    }

    #[test]
    fn invalid_allocation_rejected() {
        let mut engine = ChargebackEngine::new();
        let mut pcts = HashMap::new();
        pcts.insert("eng".to_string(), 0.5); // sum = 0.5, not 1.0
        let rule = ChargebackRule {
            id: "bad".to_string(),
            name: "bad".to_string(),
            matcher: RuleMatcher { tenant_pattern: None, model_pattern: None, min_cost: None },
            allocation_pcts: pcts,
        };
        assert!(matches!(
            engine.add_rule(rule),
            Err(ChargebackError::InvalidAllocation { .. })
        ));
    }

    #[test]
    fn duplicate_rule_rejected() {
        let mut engine = ChargebackEngine::new();
        engine.add_rule(two_dept_rule("r1", "*")).unwrap();
        assert!(matches!(
            engine.add_rule(two_dept_rule("r1", "*")),
            Err(ChargebackError::DuplicateRule(_))
        ));
    }

    #[test]
    fn split_cost_sums_to_original() {
        let mut engine = ChargebackEngine::new();
        engine.add_rule(two_dept_rule("r1", "*")).unwrap();
        let records = vec![sample_record("tenant-x", "claude-3", 10.0)];
        let entries = engine.allocate(&records);
        let total: f64 = entries.iter().map(|e| e.allocated_cost).sum();
        assert!((total - 10.0).abs() < 1e-9, "total {total} != 10.0");
    }

    #[test]
    fn csv_has_header_row() {
        let mut engine = ChargebackEngine::new();
        engine.add_rule(two_dept_rule("r1", "*")).unwrap();
        let records = vec![sample_record("t", "m", 1.0)];
        let entries = engine.allocate(&records);
        let csv = engine.to_csv(&entries);
        let first_line = csv.lines().next().unwrap_or("");
        assert!(first_line.starts_with("tenant,"), "header missing: {first_line}");
    }

    #[test]
    fn summary_report_by_department() {
        let mut engine = ChargebackEngine::new();
        engine.add_rule(two_dept_rule("r1", "*")).unwrap();
        let records = vec![
            sample_record("t1", "m1", 10.0),
            sample_record("t2", "m2", 20.0),
        ];
        let entries = engine.allocate(&records);
        let report = engine.summary_report(&entries);
        let eng = report.by_department.get("eng").copied().unwrap_or(0.0);
        let ops = report.by_department.get("ops").copied().unwrap_or(0.0);
        assert!((eng - 18.0).abs() < 1e-9, "eng={eng}");
        assert!((ops - 12.0).abs() < 1e-9, "ops={ops}");
        assert!((report.total_allocated - 30.0).abs() < 1e-9);
    }

    #[test]
    fn model_pattern_matching() {
        let mut engine = ChargebackEngine::new();
        let mut pcts = HashMap::new();
        pcts.insert("team-a".to_string(), 1.0);
        let rule = ChargebackRule {
            id: "gpt-only".to_string(),
            name: "gpt models".to_string(),
            matcher: RuleMatcher {
                tenant_pattern: None,
                model_pattern: Some("gpt-*".to_string()),
                min_cost: None,
            },
            allocation_pcts: pcts,
        };
        engine.add_rule(rule).unwrap();
        let records = vec![
            sample_record("t", "gpt-4o", 5.0),
            sample_record("t", "claude-3", 5.0),
        ];
        let entries = engine.allocate(&records);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].original.model, "gpt-4o");
    }
}
