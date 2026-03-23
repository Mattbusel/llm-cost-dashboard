//! Resource capacity planning with growth models and forecasting.
//!
//! Projects future resource usage using configurable growth models and
//! generates human-readable scaling recommendations.

use std::collections::HashMap;
use std::fmt;

// ── ResourceType ──────────────────────────────────────────────────────────────

/// The type of resource being tracked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// LLM token consumption.
    Tokens,
    /// API request count.
    Requests,
    /// Compute units (normalised CPU/GPU hours).
    ComputeUnits,
    /// Storage in gigabytes.
    StorageGB,
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ResourceType::Tokens => "Tokens",
            ResourceType::Requests => "Requests",
            ResourceType::ComputeUnits => "ComputeUnits",
            ResourceType::StorageGB => "StorageGB",
        };
        write!(f, "{s}")
    }
}

// ── CapacityMetric ────────────────────────────────────────────────────────────

/// A snapshot of resource usage at a point in time.
#[derive(Debug, Clone)]
pub struct CapacityMetric {
    /// Which resource this metric describes.
    pub resource: ResourceType,
    /// Current consumption level.
    pub current_usage: f64,
    /// Maximum allowed or provisioned capacity.
    pub capacity: f64,
    /// Utilisation as a percentage (0-100).
    pub utilization_pct: f64,
    /// Unix timestamp of this snapshot.
    pub timestamp_unix: u64,
}

impl CapacityMetric {
    /// Create a new metric, computing `utilization_pct` automatically.
    pub fn new(resource: ResourceType, current_usage: f64, capacity: f64, timestamp_unix: u64) -> Self {
        let utilization_pct = if capacity > 0.0 {
            (current_usage / capacity * 100.0).clamp(0.0, 200.0)
        } else {
            0.0
        };
        Self { resource, current_usage, capacity, utilization_pct, timestamp_unix }
    }

    /// Remaining capacity before the limit is reached.
    pub fn headroom(&self) -> f64 {
        (self.capacity - self.current_usage).max(0.0)
    }

    /// Whether utilisation exceeds 80%.
    pub fn is_constrained(&self) -> bool {
        self.utilization_pct > 80.0
    }
}

// ── GrowthModel ───────────────────────────────────────────────────────────────

/// A mathematical model that predicts how usage grows over time.
#[derive(Debug, Clone)]
pub enum GrowthModel {
    /// Usage grows by a fixed absolute amount each day.
    Linear {
        /// Daily increase in usage units.
        daily_rate: f64,
    },
    /// Usage grows by a fixed percentage each day.
    Exponential {
        /// Daily multiplicative rate (e.g. 0.03 = 3% per day).
        daily_rate: f64,
    },
    /// Usage oscillates seasonally around a base level.
    Seasonal {
        /// Mean usage level.
        base: f64,
        /// Half-amplitude of the seasonal swing.
        amplitude: f64,
        /// Period of the seasonal cycle in days.
        period_days: f64,
    },
    /// Usage jumps by a multiplier at a specific day offset.
    StepFunction {
        /// Day (relative to forecast start) at which the jump occurs.
        step_at_day: u32,
        /// Multiplier applied to usage at and after `step_at_day`.
        multiplier: f64,
    },
}

impl GrowthModel {
    /// Project usage `days` days into the future from `current`.
    pub fn project(&self, current: f64, days: u32) -> f64 {
        match self {
            GrowthModel::Linear { daily_rate } => current + *daily_rate * days as f64,
            GrowthModel::Exponential { daily_rate } => current * (1.0 + daily_rate).powi(days as i32),
            GrowthModel::Seasonal { base, amplitude, period_days } => {
                let angle = 2.0 * std::f64::consts::PI * days as f64 / period_days;
                base + amplitude * angle.sin()
            }
            GrowthModel::StepFunction { step_at_day, multiplier } => {
                if days >= *step_at_day { current * multiplier } else { current }
            }
        }
    }
}

// ── CapacityForecast ──────────────────────────────────────────────────────────

/// The result of a capacity forecast for one resource.
#[derive(Debug, Clone)]
pub struct CapacityForecast {
    /// The resource that was forecasted.
    pub resource: ResourceType,
    /// Number of days until the resource reaches capacity, if ever within the window.
    pub days_to_capacity: Option<u32>,
    /// Projected daily usage values over the forecast window.
    pub forecast_values: Vec<f64>,
    /// Recommended provisioned capacity (includes safety margin).
    pub recommended_capacity: f64,
    /// Model confidence in [0.0, 1.0].
    pub confidence: f64,
}

// ── CapacityPlanner ───────────────────────────────────────────────────────────

/// Plans future resource capacity based on current metrics and growth models.
#[derive(Debug, Default)]
pub struct CapacityPlanner {
    metrics: Vec<CapacityMetric>,
    models: HashMap<ResourceType, GrowthModel>,
}

impl CapacityPlanner {
    /// Create a new empty planner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new capacity metric snapshot.
    pub fn add_metric(&mut self, metric: CapacityMetric) {
        self.metrics.push(metric);
    }

    /// Register a growth model for a specific resource type.
    pub fn set_growth_model(&mut self, resource: ResourceType, model: GrowthModel) {
        self.models.insert(resource, model);
    }

    /// Forecast usage for a resource `days` days into the future.
    pub fn forecast(&self, resource: ResourceType, days: u32) -> CapacityForecast {
        // Find the most recent metric for this resource.
        let latest = self.metrics.iter()
            .filter(|m| m.resource == resource)
            .max_by_key(|m| m.timestamp_unix);

        let (current_usage, capacity) = match latest {
            Some(m) => (m.current_usage, m.capacity),
            None => (0.0, f64::MAX),
        };

        let model = self.models.get(&resource).cloned()
            .unwrap_or(GrowthModel::Linear { daily_rate: 0.0 });

        let forecast_values: Vec<f64> = (1..=days)
            .map(|d| model.project(current_usage, d))
            .collect();

        let days_to_capacity = Self::days_until_capacity(current_usage, capacity, &model);
        let recommended = Self::recommended_capacity(&forecast_values, 0.20);

        let confidence = match &model {
            GrowthModel::Linear { .. } => 0.75,
            GrowthModel::Exponential { .. } => 0.65,
            GrowthModel::Seasonal { .. } => 0.70,
            GrowthModel::StepFunction { .. } => 0.55,
        };

        CapacityForecast {
            resource,
            days_to_capacity,
            forecast_values,
            recommended_capacity: recommended,
            confidence,
        }
    }

    /// Calculate how many days until usage reaches `capacity` given a growth model.
    pub fn days_until_capacity(current: f64, capacity: f64, model: &GrowthModel) -> Option<u32> {
        if current >= capacity {
            return Some(0);
        }
        // Simulate day by day up to 3650 days (10 years).
        for day in 1u32..=3650 {
            let projected = model.project(current, day);
            if projected >= capacity {
                return Some(day);
            }
        }
        None
    }

    /// Compute recommended capacity: peak forecast value * (1 + safety_margin_pct/100).
    pub fn recommended_capacity(forecast: &[f64], safety_margin_pct: f64) -> f64 {
        let max = forecast.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max == f64::NEG_INFINITY { return 0.0; }
        max * (1.0 + safety_margin_pct)
    }

    /// Forecast all resources that have metrics.
    pub fn all_forecasts(&self, days: u32) -> Vec<CapacityForecast> {
        let resources: Vec<ResourceType> = self.metrics.iter()
            .map(|m| m.resource.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        resources.into_iter().map(|r| self.forecast(r, days)).collect()
    }

    /// Generate human-readable scaling recommendations for constrained resources.
    pub fn scaling_recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();

        // Group latest metric per resource.
        let mut latest_per_resource: HashMap<String, &CapacityMetric> = HashMap::new();
        for m in &self.metrics {
            let key = m.resource.to_string();
            match latest_per_resource.get(&key) {
                Some(existing) if existing.timestamp_unix >= m.timestamp_unix => {}
                _ => { latest_per_resource.insert(key, m); }
            }
        }

        for (_, metric) in &latest_per_resource {
            if metric.is_constrained() {
                let forecast = self.forecast(metric.resource.clone(), 30);
                let msg = match forecast.days_to_capacity {
                    Some(0) => format!(
                        "CRITICAL: {} is already over capacity ({:.1}% utilisation). Scale immediately.",
                        metric.resource, metric.utilization_pct
                    ),
                    Some(d) if d <= 7 => format!(
                        "URGENT: {} will reach capacity in ~{d} days ({:.1}% now). Recommended capacity: {:.0}.",
                        metric.resource, metric.utilization_pct, forecast.recommended_capacity
                    ),
                    Some(d) => format!(
                        "WARNING: {} will reach capacity in ~{d} days. Plan to provision {:.0} units.",
                        metric.resource, forecast.recommended_capacity
                    ),
                    None => format!(
                        "INFO: {} is at {:.1}% utilisation but growth model does not predict breach within 10 years.",
                        metric.resource, metric.utilization_pct
                    ),
                };
                recs.push(msg);
            }
        }

        if recs.is_empty() {
            recs.push("All resources are within acceptable utilisation limits.".to_string());
        }

        recs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_headroom() {
        let m = CapacityMetric::new(ResourceType::Tokens, 800.0, 1000.0, 0);
        assert_eq!(m.headroom(), 200.0);
        assert!(!m.is_constrained());
    }

    #[test]
    fn test_metric_constrained() {
        let m = CapacityMetric::new(ResourceType::Requests, 850.0, 1000.0, 0);
        assert!(m.is_constrained());
    }

    #[test]
    fn test_linear_growth() {
        let model = GrowthModel::Linear { daily_rate: 10.0 };
        assert_eq!(model.project(100.0, 5), 150.0);
    }

    #[test]
    fn test_exponential_growth() {
        let model = GrowthModel::Exponential { daily_rate: 0.10 };
        let projected = model.project(100.0, 1);
        assert!((projected - 110.0).abs() < 0.01);
    }

    #[test]
    fn test_days_until_capacity_linear() {
        let model = GrowthModel::Linear { daily_rate: 10.0 };
        let days = CapacityPlanner::days_until_capacity(80.0, 100.0, &model);
        assert_eq!(days, Some(2));
    }

    #[test]
    fn test_recommended_capacity() {
        let forecast = vec![100.0, 110.0, 120.0, 130.0];
        let rec = CapacityPlanner::recommended_capacity(&forecast, 0.20);
        assert!((rec - 156.0).abs() < 0.01);
    }

    #[test]
    fn test_planner_forecast() {
        let mut planner = CapacityPlanner::new();
        planner.add_metric(CapacityMetric::new(ResourceType::Tokens, 500.0, 1000.0, 1000));
        planner.set_growth_model(ResourceType::Tokens, GrowthModel::Linear { daily_rate: 50.0 });
        let forecast = planner.forecast(ResourceType::Tokens, 10);
        assert_eq!(forecast.forecast_values.len(), 10);
        assert!(forecast.days_to_capacity.is_some());
    }

    #[test]
    fn test_scaling_recommendations() {
        let mut planner = CapacityPlanner::new();
        planner.add_metric(CapacityMetric::new(ResourceType::ComputeUnits, 900.0, 1000.0, 1000));
        planner.set_growth_model(ResourceType::ComputeUnits, GrowthModel::Linear { daily_rate: 20.0 });
        let recs = planner.scaling_recommendations();
        assert!(!recs.is_empty());
        assert!(recs[0].contains("ComputeUnits") || recs[0].contains("limits"));
    }
}
