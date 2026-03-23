//! Carbon footprint estimator for LLM requests.
//!
//! Estimates the CO2-equivalent emissions (gCO2e) of each LLM API call based
//! on:
//!
//! 1. **Model energy cost**: larger/more capable models consume more GPU
//!    energy per token. Energy coefficients are expressed in Wh per 1000
//!    tokens (Wh/kTok).
//! 2. **Data center carbon intensity**: the carbon intensity of the
//!    electricity consumed varies by region (gCO2e/kWh).
//! 3. **Token count**: total tokens (input + output) processed.
//!
//! ## Emission Calculation
//!
//! ```text
//! energy_Wh = (total_tokens / 1000) * energy_per_ktok_Wh
//! co2e_g    = energy_Wh * carbon_intensity_gCO2e_per_kWh / 1000
//! ```
//!
//! ## Data Center Regions
//!
//! | Region         | Carbon intensity (gCO2e/kWh) |
//! |----------------|------------------------------|
//! | `us-east`      | 350                          |
//! | `us-west`      | 130                          |
//! | `eu-west`      | 200                          |
//! | `eu-north`     | 30  (Nordics: hydro/wind)    |
//! | `asia-pacific` | 600                          |
//! | `global-avg`   | 450                          |
//!
//! ## Model Energy Coefficients
//!
//! These are illustrative estimates based on published research. Real values
//! require direct measurement.
//!
//! | Model family       | Wh/kTok |
//! |--------------------|---------|
//! | GPT-4 / Claude-3 Opus (large)      | 3.0    |
//! | GPT-4o / Claude Sonnet (medium)    | 1.2    |
//! | GPT-4o-mini / Claude Haiku (small) | 0.4    |
//! | GPT-3.5 / small models             | 0.2    |
//!
//! ## Usage
//!
//! ```
//! use llm_cost_dashboard::carbon::{CarbonEstimator, CarbonConfig, Region};
//!
//! let cfg = CarbonConfig::default();
//! let est = CarbonEstimator::new(cfg);
//!
//! let emission = est.estimate("gpt-4o", 500, 200, Region::UsEast);
//! println!("CO2e: {:.3} gCO2e", emission.co2e_grams);
//! println!("Energy: {:.4} Wh", emission.energy_wh);
//!
//! let monthly = est.monthly_footprint_g(1000, "gpt-4o", 350, 150, Region::UsWest);
//! println!("Monthly: {:.1} gCO2e", monthly);
//! ```

use std::collections::HashMap;

/// A data center geographic region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Region {
    /// US East Coast (Virginia, Ohio). ~350 gCO2e/kWh.
    UsEast,
    /// US West Coast (Oregon, California). ~130 gCO2e/kWh (more renewables).
    UsWest,
    /// Western Europe. ~200 gCO2e/kWh.
    EuWest,
    /// Northern Europe (Nordics). ~30 gCO2e/kWh (hydro/wind-heavy).
    EuNorth,
    /// Asia-Pacific. ~600 gCO2e/kWh (coal-heavy grids).
    AsiaPacific,
    /// Global average. ~450 gCO2e/kWh.
    GlobalAverage,
}

impl Region {
    /// Carbon intensity of the electricity grid in this region (gCO2e/kWh).
    pub fn carbon_intensity_g_per_kwh(self) -> f64 {
        match self {
            Region::UsEast       => 350.0,
            Region::UsWest       => 130.0,
            Region::EuWest       => 200.0,
            Region::EuNorth      => 30.0,
            Region::AsiaPacific  => 600.0,
            Region::GlobalAverage=> 450.0,
        }
    }

    /// Display name.
    pub fn display_name(self) -> &'static str {
        match self {
            Region::UsEast        => "US East",
            Region::UsWest        => "US West",
            Region::EuWest        => "EU West",
            Region::EuNorth       => "EU North (Nordics)",
            Region::AsiaPacific   => "Asia-Pacific",
            Region::GlobalAverage => "Global Average",
        }
    }

    /// All regions sorted by ascending carbon intensity.
    pub fn all_by_intensity() -> Vec<Region> {
        let mut regions = vec![
            Region::EuNorth, Region::UsWest, Region::EuWest,
            Region::UsEast, Region::GlobalAverage, Region::AsiaPacific,
        ];
        regions.sort_by(|a, b| a.carbon_intensity_g_per_kwh().partial_cmp(&b.carbon_intensity_g_per_kwh()).unwrap_or(std::cmp::Ordering::Equal));
        regions
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

/// Configuration for the carbon estimator.
#[derive(Debug, Clone)]
pub struct CarbonConfig {
    /// Model energy coefficients: model substring → Wh per 1000 tokens.
    /// Checked in insertion order; first match wins.
    pub model_energy: Vec<(String, f64)>,
    /// Default energy coefficient when no model matches (Wh/kTok).
    pub default_energy_wh_per_ktok: f64,
    /// Power Usage Effectiveness (PUE) multiplier for data centre overhead.
    /// Typical value: 1.1–1.5.
    pub pue: f64,
}

impl Default for CarbonConfig {
    fn default() -> Self {
        Self {
            model_energy: vec![
                // Small (must come before medium/large entries that are substrings).
                ("gpt-4o-mini".to_string(),  0.4),
                ("claude-3-haiku".to_string(), 0.4),
                ("claude-haiku".to_string(), 0.4),
                ("gpt-3.5".to_string(),      0.2),
                ("gemini-flash".to_string(), 0.3),
                ("mistral-7b".to_string(),   0.25),
                // Medium.
                ("gpt-4o".to_string(),       1.2),
                ("claude-sonnet".to_string(),1.2),
                ("gpt-4-turbo".to_string(),  1.5),
                // Large / frontier.
                ("gpt-4".to_string(),        3.0),
                ("claude-3-opus".to_string(),3.0),
                ("claude-opus".to_string(),  3.0),
                ("o1".to_string(),           3.5),
                ("o3".to_string(),           3.5),
            ],
            default_energy_wh_per_ktok: 1.0,
            pue: 1.2,
        }
    }
}

/// The estimated carbon footprint of a single LLM request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CarbonEmission {
    /// Model name.
    pub model: String,
    /// Total tokens processed (input + output).
    pub total_tokens: u64,
    /// Region where inference ran.
    pub region: Region,
    /// Estimated energy consumed (Wh), including PUE overhead.
    pub energy_wh: f64,
    /// Estimated CO2-equivalent emissions (grams).
    pub co2e_grams: f64,
    /// Energy coefficient used (Wh/kTok).
    pub energy_coeff_wh_per_ktok: f64,
    /// Carbon intensity used (gCO2e/kWh).
    pub carbon_intensity: f64,
}

/// A lower-carbon model alternative suggestion.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlternativeSuggestion {
    /// Suggested model.
    pub model: String,
    /// Estimated energy coefficient.
    pub energy_wh_per_ktok: f64,
    /// Potential CO2e saving per request (grams), vs the current model.
    pub co2e_saving_grams: f64,
    /// Estimated cost saving per request (if `cost_per_ktok` is provided).
    pub cost_saving_usd: Option<f64>,
}

/// Running carbon footprint accumulator.
#[derive(Debug, Default)]
struct ModelAccum {
    total_co2e_grams: f64,
    total_tokens: u64,
    request_count: u64,
}

/// Per-model carbon summary included in monthly reports.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelCarbonSummary {
    /// Model name.
    pub model: String,
    /// Total CO2e (grams).
    pub total_co2e_grams: f64,
    /// Total tokens processed.
    pub total_tokens: u64,
    /// Request count.
    pub request_count: u64,
    /// Average CO2e per request (grams).
    pub avg_co2e_per_request: f64,
}

/// Carbon footprint monthly report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CarbonReport {
    /// Per-model summaries sorted by total CO2e descending.
    pub models: Vec<ModelCarbonSummary>,
    /// Grand total CO2e (grams).
    pub total_co2e_grams: f64,
    /// Equivalent in kg CO2e.
    pub total_co2e_kg: f64,
    /// Lowest-carbon region that could be used instead.
    pub suggested_region: Region,
    /// Potential saving if workload moved to `suggested_region` (grams CO2e).
    pub potential_saving_grams: f64,
}

/// LLM carbon footprint estimator.
pub struct CarbonEstimator {
    cfg: CarbonConfig,
    accum: HashMap<String, ModelAccum>,
    /// Default region for accumulated observations.
    pub default_region: Region,
}

impl CarbonEstimator {
    /// Create a new estimator.
    pub fn new(cfg: CarbonConfig) -> Self {
        Self { cfg, accum: HashMap::new(), default_region: Region::UsEast }
    }

    /// Look up the energy coefficient for a model (Wh per 1000 tokens).
    pub fn energy_coeff(&self, model: &str) -> f64 {
        let model_lower = model.to_lowercase();
        for (key, coeff) in &self.cfg.model_energy {
            if model_lower.contains(key.to_lowercase().as_str()) {
                return *coeff;
            }
        }
        self.cfg.default_energy_wh_per_ktok
    }

    /// Estimate the carbon footprint of a single request.
    ///
    /// - `model`: model identifier string
    /// - `input_tokens`: tokens in the prompt
    /// - `output_tokens`: tokens in the completion
    /// - `region`: data center region
    pub fn estimate(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        region: Region,
    ) -> CarbonEmission {
        let total_tokens = input_tokens + output_tokens;
        let coeff = self.energy_coeff(model);
        let intensity = region.carbon_intensity_g_per_kwh();

        // Energy in Wh (including PUE overhead).
        let energy_wh = (total_tokens as f64 / 1000.0) * coeff * self.cfg.pue;

        // CO2e: energy_Wh * intensity_gCO2e_per_kWh / 1000 (Wh → kWh).
        let co2e_grams = energy_wh * intensity / 1000.0;

        CarbonEmission {
            model: model.to_string(),
            total_tokens,
            region,
            energy_wh,
            co2e_grams,
            energy_coeff_wh_per_ktok: coeff,
            carbon_intensity: intensity,
        }
    }

    /// Record an observation and accumulate into the running total.
    pub fn observe(
        &mut self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        region: Region,
    ) -> CarbonEmission {
        let emission = self.estimate(model, input_tokens, output_tokens, region);
        let acc = self.accum.entry(model.to_string()).or_default();
        acc.total_co2e_grams += emission.co2e_grams;
        acc.total_tokens += emission.total_tokens;
        acc.request_count += 1;
        emission
    }

    /// Estimate the total monthly CO2e (grams) for a given request volume.
    ///
    /// `requests_per_day` × 30 days.
    pub fn monthly_footprint_g(
        &self,
        requests_per_day: u64,
        model: &str,
        avg_input_tokens: u64,
        avg_output_tokens: u64,
        region: Region,
    ) -> f64 {
        let per_request = self.estimate(model, avg_input_tokens, avg_output_tokens, region).co2e_grams;
        per_request * requests_per_day as f64 * 30.0
    }

    /// Suggest lower-carbon model alternatives.
    ///
    /// Returns models with lower energy coefficients than `current_model`,
    /// sorted by ascending energy cost. `cost_per_ktok_usd` optionally
    /// provides the current model's cost for estimating savings.
    pub fn suggest_alternatives(
        &self,
        current_model: &str,
        total_tokens: u64,
        region: Region,
        cost_per_ktok_usd: Option<f64>,
    ) -> Vec<AlternativeSuggestion> {
        let current_coeff = self.energy_coeff(current_model);
        let current_intensity = region.carbon_intensity_g_per_kwh();
        let current_co2e = (total_tokens as f64 / 1000.0) * current_coeff * self.cfg.pue
            * current_intensity / 1000.0;

        let mut suggestions: Vec<AlternativeSuggestion> = self.cfg.model_energy.iter()
            .filter(|(name, coeff)| *coeff < current_coeff && name != current_model)
            .map(|(name, coeff)| {
                let alt_co2e = (total_tokens as f64 / 1000.0) * coeff * self.cfg.pue
                    * current_intensity / 1000.0;
                let cost_saving = cost_per_ktok_usd.map(|cpm| {
                    (current_coeff - coeff) / 1000.0 * total_tokens as f64 * cpm
                });
                AlternativeSuggestion {
                    model: name.clone(),
                    energy_wh_per_ktok: *coeff,
                    co2e_saving_grams: current_co2e - alt_co2e,
                    cost_saving_usd: cost_saving,
                }
            })
            .collect();

        suggestions.sort_by(|a, b| a.energy_wh_per_ktok.partial_cmp(&b.energy_wh_per_ktok).unwrap_or(std::cmp::Ordering::Equal));
        suggestions
    }

    /// Generate a carbon footprint report across all observed models.
    pub fn report(&self, current_region: Region) -> CarbonReport {
        let mut models: Vec<ModelCarbonSummary> = self.accum.iter().map(|(model, acc)| {
            ModelCarbonSummary {
                model: model.clone(),
                total_co2e_grams: acc.total_co2e_grams,
                total_tokens: acc.total_tokens,
                request_count: acc.request_count,
                avg_co2e_per_request: if acc.request_count > 0 {
                    acc.total_co2e_grams / acc.request_count as f64
                } else { 0.0 },
            }
        }).collect();
        models.sort_by(|a, b| b.total_co2e_grams.partial_cmp(&a.total_co2e_grams).unwrap_or(std::cmp::Ordering::Equal));

        let total_co2e_grams: f64 = models.iter().map(|m| m.total_co2e_grams).sum();

        // Suggest the lowest-carbon region.
        let suggested_region = Region::EuNorth;
        let intensity_ratio = suggested_region.carbon_intensity_g_per_kwh()
            / current_region.carbon_intensity_g_per_kwh().max(1.0);
        let potential_saving_grams = total_co2e_grams * (1.0 - intensity_ratio).max(0.0);

        CarbonReport {
            models,
            total_co2e_grams,
            total_co2e_kg: total_co2e_grams / 1000.0,
            suggested_region,
            potential_saving_grams,
        }
    }

    /// Reset accumulated data.
    pub fn reset(&mut self) {
        self.accum.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn est() -> CarbonEstimator {
        CarbonEstimator::new(CarbonConfig::default())
    }

    #[test]
    fn estimate_is_positive_and_finite() {
        let e = est().estimate("gpt-4o", 500, 200, Region::UsEast);
        assert!(e.co2e_grams > 0.0 && e.co2e_grams.is_finite());
        assert!(e.energy_wh > 0.0 && e.energy_wh.is_finite());
    }

    #[test]
    fn larger_model_emits_more() {
        let est = est();
        let large = est.estimate("gpt-4", 500, 200, Region::UsEast);
        let small = est.estimate("gpt-4o-mini", 500, 200, Region::UsEast);
        assert!(large.co2e_grams > small.co2e_grams);
    }

    #[test]
    fn greener_region_emits_less() {
        let est = est();
        let asia = est.estimate("gpt-4o", 500, 200, Region::AsiaPacific);
        let nord = est.estimate("gpt-4o", 500, 200, Region::EuNorth);
        assert!(asia.co2e_grams > nord.co2e_grams);
    }

    #[test]
    fn token_count_linearity() {
        let est = est();
        let e1 = est.estimate("gpt-4o-mini", 1000, 0, Region::UsEast);
        let e2 = est.estimate("gpt-4o-mini", 2000, 0, Region::UsEast);
        assert!((e2.co2e_grams / e1.co2e_grams - 2.0).abs() < 1e-6);
    }

    #[test]
    fn observe_accumulates() {
        let mut est = est();
        est.observe("gpt-4o", 500, 200, Region::UsEast);
        est.observe("gpt-4o", 300, 100, Region::UsEast);
        let report = est.report(Region::UsEast);
        let gpt4o = report.models.iter().find(|m| m.model == "gpt-4o").unwrap();
        assert_eq!(gpt4o.request_count, 2);
        assert_eq!(gpt4o.total_tokens, (500 + 200 + 300 + 100));
    }

    #[test]
    fn monthly_footprint_scales_with_requests() {
        let est = est();
        let m1 = est.monthly_footprint_g(100, "gpt-4o-mini", 300, 100, Region::UsEast);
        let m2 = est.monthly_footprint_g(200, "gpt-4o-mini", 300, 100, Region::UsEast);
        assert!((m2 / m1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn suggest_alternatives_returns_cheaper_models() {
        let alternatives = est().suggest_alternatives("gpt-4", 1000, Region::UsEast, None);
        // All alternatives should have lower energy coefficient than gpt-4 (3.0 Wh/kTok).
        for alt in &alternatives {
            assert!(alt.energy_wh_per_ktok < 3.0,
                "alternative {} has higher energy than gpt-4", alt.model);
        }
    }

    #[test]
    fn alternatives_sorted_ascending() {
        let alts = est().suggest_alternatives("gpt-4", 1000, Region::UsEast, None);
        for i in 1..alts.len() {
            assert!(alts[i - 1].energy_wh_per_ktok <= alts[i].energy_wh_per_ktok);
        }
    }

    #[test]
    fn co2e_saving_is_positive_for_alternatives() {
        let alts = est().suggest_alternatives("gpt-4", 1000, Region::UsEast, None);
        for alt in alts {
            assert!(alt.co2e_saving_grams >= 0.0);
        }
    }

    #[test]
    fn report_total_co2e_matches_sum() {
        let mut est = est();
        est.observe("gpt-4o", 500, 200, Region::UsEast);
        est.observe("gpt-4o-mini", 400, 100, Region::UsWest);
        let report = est.report(Region::UsEast);
        let sum: f64 = report.models.iter().map(|m| m.total_co2e_grams).sum();
        assert!((report.total_co2e_grams - sum).abs() < 1e-9);
    }

    #[test]
    fn unknown_model_uses_default_coefficient() {
        let est = est();
        let coeff = est.energy_coeff("my-unknown-model-123");
        assert!((coeff - est.cfg.default_energy_wh_per_ktok).abs() < 1e-9);
    }

    #[test]
    fn region_by_intensity_sorted() {
        let regions = Region::all_by_intensity();
        for i in 1..regions.len() {
            assert!(regions[i - 1].carbon_intensity_g_per_kwh()
                <= regions[i].carbon_intensity_g_per_kwh());
        }
    }

    #[test]
    fn reset_clears_accumulator() {
        let mut est = est();
        est.observe("gpt-4o", 500, 200, Region::UsEast);
        est.reset();
        let report = est.report(Region::UsEast);
        assert_eq!(report.models.len(), 0);
        assert!(report.total_co2e_grams.abs() < 1e-9);
    }
}
