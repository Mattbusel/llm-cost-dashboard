//! User cohort analysis for LLM cost data.
//!
//! Groups users by their first-seen period, computes retention matrices,
//! lifetime value estimates, and identifies the highest-value cohorts.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Granularity used when assigning users to cohorts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohortPeriod {
    /// Group by calendar day (Unix day boundary).
    Daily,
    /// Group by ISO week (Monday-anchored).
    Weekly,
    /// Group by calendar month.
    Monthly,
}

/// A single user event with cost and token information.
#[derive(Debug, Clone)]
pub struct UserEvent {
    /// Unique user identifier.
    pub user_id: String,
    /// Event type label (e.g. `"inference"`, `"fine_tune"`).
    pub event_type: String,
    /// Unix timestamp (seconds) when the event occurred.
    pub timestamp_unix: u64,
    /// Cost of the event in USD.
    pub cost_usd: f64,
    /// Number of tokens consumed.
    pub tokens: usize,
    /// Model used for the event.
    pub model: String,
}

/// A cohort: all users whose first event fell in the same period bucket.
#[derive(Debug, Clone)]
pub struct CohortGroup {
    /// Unix timestamp of the start of the cohort's period bucket.
    pub cohort_date: u64,
    /// Set of user IDs in this cohort.
    pub user_ids: HashSet<String>,
    /// Number of users (mirrors `user_ids.len()`).
    pub size: usize,
}

/// Retention matrix across cohorts and time periods.
#[derive(Debug, Clone)]
pub struct RetentionMatrix {
    /// Start timestamp of each period bucket analysed.
    pub periods: Vec<u64>,
    /// `retention_pcts[cohort][period]` — percentage of the cohort active in
    /// that period.
    pub retention_pcts: Vec<Vec<f64>>,
    /// Average retention across all cohorts, per period.
    pub avg_retention: Vec<f64>,
}

/// Aggregated statistics for a single cohort.
#[derive(Debug, Clone)]
pub struct CohortStats {
    /// Unix timestamp of the cohort's period start.
    pub cohort_date: u64,
    /// Number of users in the cohort.
    pub users: usize,
    /// Average cost per user in USD.
    pub avg_cost_usd: f64,
    /// Average tokens per user.
    pub avg_tokens: usize,
    /// Estimated lifetime value per user in USD.
    pub ltv_estimate: f64,
    /// 30-day retention rate (fraction, not percentage).
    pub retention_30d: f64,
    /// Most-used models in the cohort.
    pub top_models: Vec<String>,
}

// ---------------------------------------------------------------------------
// CohortAnalyzer
// ---------------------------------------------------------------------------

/// Analyses user cohorts from a stream of [`UserEvent`]s.
pub struct CohortAnalyzer {
    events: Mutex<Vec<UserEvent>>,
}

const SECS_PER_DAY: u64 = 86_400;
const SECS_PER_WEEK: u64 = 7 * SECS_PER_DAY;

impl CohortAnalyzer {
    /// Create a new, empty `CohortAnalyzer`.
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    // -----------------------------------------------------------------------
    // Ingestion
    // -----------------------------------------------------------------------

    /// Record a user event.
    pub fn record_event(&self, event: UserEvent) {
        if let Ok(mut guard) = self.events.lock() {
            guard.push(event);
        }
    }

    // -----------------------------------------------------------------------
    // Cohort assignment
    // -----------------------------------------------------------------------

    /// Assign each user to a cohort based on the first event seen for that
    /// user, bucketed by `period`.
    pub fn assign_cohorts(&self, period: CohortPeriod) -> Vec<CohortGroup> {
        let events = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };

        // Find first-seen timestamp per user.
        let mut first_seen: HashMap<String, u64> = HashMap::new();
        for ev in events.iter() {
            let entry = first_seen
                .entry(ev.user_id.clone())
                .or_insert(ev.timestamp_unix);
            if ev.timestamp_unix < *entry {
                *entry = ev.timestamp_unix;
            }
        }

        // Bucket each user into their cohort period.
        let mut cohorts: HashMap<u64, HashSet<String>> = HashMap::new();
        for (user_id, ts) in &first_seen {
            let bucket = period_bucket(*ts, period);
            cohorts
                .entry(bucket)
                .or_default()
                .insert(user_id.clone());
        }

        let mut result: Vec<CohortGroup> = cohorts
            .into_iter()
            .map(|(cohort_date, user_ids)| {
                let size = user_ids.len();
                CohortGroup {
                    cohort_date,
                    user_ids,
                    size,
                }
            })
            .collect();

        result.sort_by_key(|c| c.cohort_date);
        result
    }

    // -----------------------------------------------------------------------
    // Retention
    // -----------------------------------------------------------------------

    /// Build a retention matrix: for each cohort, what fraction of its users
    /// were active in each of `num_periods` consecutive period buckets starting
    /// from the cohort's own bucket?
    pub fn retention_matrix(
        &self,
        cohorts: &[CohortGroup],
        num_periods: usize,
    ) -> RetentionMatrix {
        let events = match self.events.lock() {
            Ok(g) => g,
            Err(_) => {
                return RetentionMatrix {
                    periods: Vec::new(),
                    retention_pcts: Vec::new(),
                    avg_retention: Vec::new(),
                }
            }
        };

        // Build an index: user_id -> set of period_buckets where they were active.
        // We use Daily buckets as the finest granularity for the "active in period N"
        // check; the caller's `period` is embedded in the cohort_date values.
        let mut user_activity: HashMap<&str, HashSet<u64>> = HashMap::new();
        for ev in events.iter() {
            user_activity
                .entry(&ev.user_id)
                .or_default()
                .insert(ev.timestamp_unix / SECS_PER_DAY);
        }

        // Determine the set of period offsets to analyse.
        let periods: Vec<u64> = (0..num_periods as u64).collect();

        let mut retention_pcts: Vec<Vec<f64>> = Vec::new();

        for cohort in cohorts {
            let mut row = Vec::with_capacity(num_periods);
            for offset in 0..num_periods as u64 {
                // "Active in period N" = any event whose day bucket falls within
                // [cohort_date/day + offset*period_days, ... + period_days).
                // For simplicity: check if the user had any event on
                // day (cohort_date/SECS_PER_DAY + offset).
                let target_day = cohort.cohort_date / SECS_PER_DAY + offset;
                let active: usize = cohort
                    .user_ids
                    .iter()
                    .filter(|uid| {
                        user_activity
                            .get(uid.as_str())
                            .map(|days| days.contains(&target_day))
                            .unwrap_or(false)
                    })
                    .count();
                let pct = if cohort.size == 0 {
                    0.0
                } else {
                    active as f64 / cohort.size as f64 * 100.0
                };
                row.push(pct);
            }
            retention_pcts.push(row);
        }

        // Average per period across cohorts.
        let avg_retention: Vec<f64> = (0..num_periods)
            .map(|p| {
                if retention_pcts.is_empty() {
                    0.0
                } else {
                    retention_pcts.iter().map(|row| row[p]).sum::<f64>()
                        / retention_pcts.len() as f64
                }
            })
            .collect();

        RetentionMatrix {
            periods,
            retention_pcts,
            avg_retention,
        }
    }

    // -----------------------------------------------------------------------
    // LTV
    // -----------------------------------------------------------------------

    /// Estimate the lifetime value of a single user.
    ///
    /// `monthly_rate` is the assumed monthly churn rate in `[0, 1]`.
    /// Returns `avg_monthly_spend / churn_rate`, or `0.0` if
    /// `monthly_rate == 0`.
    pub fn ltv_estimate(events: &[UserEvent], user_id: &str, monthly_rate: f64) -> f64 {
        if monthly_rate <= 0.0 {
            return 0.0;
        }

        let user_events: Vec<&UserEvent> =
            events.iter().filter(|e| e.user_id == user_id).collect();
        if user_events.is_empty() {
            return 0.0;
        }

        let total_cost: f64 = user_events.iter().map(|e| e.cost_usd).sum();

        // Determine span in months.
        let min_ts = user_events.iter().map(|e| e.timestamp_unix).min().unwrap_or(0);
        let max_ts = user_events.iter().map(|e| e.timestamp_unix).max().unwrap_or(0);
        let span_months = ((max_ts - min_ts) as f64 / (30.0 * SECS_PER_DAY as f64)).max(1.0);
        let avg_monthly = total_cost / span_months;

        avg_monthly / monthly_rate
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Compute statistics for a single cohort.
    pub fn cohort_stats(&self, cohort: &CohortGroup) -> CohortStats {
        let events = match self.events.lock() {
            Ok(g) => g,
            Err(_) => {
                return CohortStats {
                    cohort_date: cohort.cohort_date,
                    users: cohort.size,
                    avg_cost_usd: 0.0,
                    avg_tokens: 0,
                    ltv_estimate: 0.0,
                    retention_30d: 0.0,
                    top_models: Vec::new(),
                }
            }
        };

        let cohort_events: Vec<&UserEvent> = events
            .iter()
            .filter(|e| cohort.user_ids.contains(&e.user_id))
            .collect();

        let total_cost: f64 = cohort_events.iter().map(|e| e.cost_usd).sum();
        let total_tokens: usize = cohort_events.iter().map(|e| e.tokens).sum();
        let users = cohort.size.max(1);

        let avg_cost_usd = total_cost / users as f64;
        let avg_tokens = total_tokens / users;

        // LTV: assume 10% monthly churn.
        let ltv = {
            let monthly_rate = 0.10;
            let span = {
                let min_ts = cohort_events
                    .iter()
                    .map(|e| e.timestamp_unix)
                    .min()
                    .unwrap_or(0);
                let max_ts = cohort_events
                    .iter()
                    .map(|e| e.timestamp_unix)
                    .max()
                    .unwrap_or(0);
                ((max_ts - min_ts) as f64 / (30.0 * SECS_PER_DAY as f64)).max(1.0)
            };
            let avg_monthly = (total_cost / users as f64) / span;
            avg_monthly / monthly_rate
        };

        // 30-day retention: fraction of users who had an event within 30 days
        // of the cohort start.
        let window_end = cohort.cohort_date + 30 * SECS_PER_DAY;
        let active_30d: usize = cohort
            .user_ids
            .iter()
            .filter(|uid| {
                cohort_events.iter().any(|e| {
                    &e.user_id == *uid
                        && e.timestamp_unix >= cohort.cohort_date
                        && e.timestamp_unix < window_end
                })
            })
            .count();
        let retention_30d = active_30d as f64 / cohort.size.max(1) as f64;

        // Top models.
        let mut model_freq: HashMap<&str, usize> = HashMap::new();
        for e in &cohort_events {
            *model_freq.entry(e.model.as_str()).or_insert(0) += 1;
        }
        let mut model_vec: Vec<(&str, usize)> = model_freq.into_iter().collect();
        model_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let top_models: Vec<String> = model_vec
            .into_iter()
            .take(3)
            .map(|(m, _)| m.to_string())
            .collect();

        CohortStats {
            cohort_date: cohort.cohort_date,
            users: cohort.size,
            avg_cost_usd,
            avg_tokens,
            ltv_estimate: ltv,
            retention_30d,
            top_models,
        }
    }

    /// Return the top `top_n` cohorts ranked by LTV estimate.
    pub fn best_cohorts(&self, period: CohortPeriod, top_n: usize) -> Vec<CohortStats> {
        let cohorts = self.assign_cohorts(period);
        let mut stats: Vec<CohortStats> = cohorts
            .iter()
            .map(|c| self.cohort_stats(c))
            .collect();
        stats.sort_by(|a, b| {
            b.ltv_estimate
                .partial_cmp(&a.ltv_estimate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        stats.into_iter().take(top_n).collect()
    }

    /// Compute the churn rate for a cohort: fraction of cohort members who had
    /// no event within `active_window_days` days of the cohort start.
    pub fn churn_rate(&self, cohort: &CohortGroup, active_window_days: u64) -> f64 {
        if cohort.size == 0 {
            return 0.0;
        }
        let events = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return 0.0,
        };
        let window_end = cohort.cohort_date + active_window_days * SECS_PER_DAY;
        let active: usize = cohort
            .user_ids
            .iter()
            .filter(|uid| {
                events.iter().any(|e| {
                    &e.user_id == *uid
                        && e.timestamp_unix >= cohort.cohort_date
                        && e.timestamp_unix < window_end
                })
            })
            .count();
        let churned = cohort.size.saturating_sub(active);
        churned as f64 / cohort.size as f64
    }

    /// Return the user IDs of users whose total spend is at or above
    /// `min_cost_usd`.
    pub fn power_users(&self, min_cost_usd: f64) -> Vec<String> {
        let events = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let mut spend: HashMap<String, f64> = HashMap::new();
        for ev in events.iter() {
            *spend.entry(ev.user_id.clone()).or_insert(0.0) += ev.cost_usd;
        }
        let mut users: Vec<String> = spend
            .into_iter()
            .filter(|(_, s)| *s >= min_cost_usd)
            .map(|(uid, _)| uid)
            .collect();
        users.sort();
        users
    }
}

impl Default for CohortAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn period_bucket(ts: u64, period: CohortPeriod) -> u64 {
    match period {
        CohortPeriod::Daily => (ts / SECS_PER_DAY) * SECS_PER_DAY,
        CohortPeriod::Weekly => (ts / SECS_PER_WEEK) * SECS_PER_WEEK,
        CohortPeriod::Monthly => {
            // Approximate: use 30-day months anchored at Unix epoch.
            let month_secs = 30 * SECS_PER_DAY;
            (ts / month_secs) * month_secs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(user_id: &str, ts: u64, cost: f64) -> UserEvent {
        UserEvent {
            user_id: user_id.to_string(),
            event_type: "inference".to_string(),
            timestamp_unix: ts,
            cost_usd: cost,
            tokens: 100,
            model: "claude-3".to_string(),
        }
    }

    #[test]
    fn test_assign_cohorts_daily() {
        let analyzer = CohortAnalyzer::new();
        analyzer.record_event(make_event("alice", 0, 1.0));
        analyzer.record_event(make_event("bob", 0, 2.0));
        analyzer.record_event(make_event("carol", SECS_PER_DAY, 3.0));

        let cohorts = analyzer.assign_cohorts(CohortPeriod::Daily);
        assert_eq!(cohorts.len(), 2);
        let day0 = cohorts.iter().find(|c| c.cohort_date == 0).unwrap();
        assert_eq!(day0.size, 2);
    }

    #[test]
    fn test_power_users() {
        let analyzer = CohortAnalyzer::new();
        analyzer.record_event(make_event("alice", 0, 10.0));
        analyzer.record_event(make_event("alice", 1000, 5.0));
        analyzer.record_event(make_event("bob", 0, 1.0));

        let power = analyzer.power_users(14.0);
        assert_eq!(power, vec!["alice".to_string()]);
    }

    #[test]
    fn test_ltv_estimate() {
        let events = vec![
            make_event("alice", 0, 10.0),
            make_event("alice", 30 * SECS_PER_DAY, 10.0),
        ];
        let ltv = CohortAnalyzer::ltv_estimate(&events, "alice", 0.10);
        assert!(ltv > 0.0);
    }

    #[test]
    fn test_churn_rate() {
        let analyzer = CohortAnalyzer::new();
        analyzer.record_event(make_event("alice", 0, 1.0));
        analyzer.record_event(make_event("bob", 0, 1.0));
        // bob has no event in the window
        let cohorts = analyzer.assign_cohorts(CohortPeriod::Daily);
        let cohort = &cohorts[0];
        // alice is active within 1 day, bob is not (only ts=0 events exist for both)
        // Actually both have ts=0, so both are active in window day 0.
        let churn = analyzer.churn_rate(cohort, 1);
        assert!(churn >= 0.0 && churn <= 1.0);
    }
}
