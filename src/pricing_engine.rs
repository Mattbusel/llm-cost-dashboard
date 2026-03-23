//! Dynamic pricing engine with versioned price history and tier discounts.
//!
//! [`PricingEngine`] stores one [`ModelPricing`] record per model and supports
//! point-in-time price lookup, tier-discounted cost computation, and batch
//! cost estimation.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// PricingTier
// ---------------------------------------------------------------------------

/// Customer pricing tier that determines the discount applied to list prices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PricingTier {
    /// No discount.
    Free,
    /// 5 % discount.
    Starter,
    /// 15 % discount.
    Professional,
    /// 30 % discount.
    Enterprise,
}

impl PricingTier {
    /// Returns the percentage discount for this tier (0–100).
    pub fn discount_pct(&self) -> f64 {
        match self {
            PricingTier::Free => 0.0,
            PricingTier::Starter => 5.0,
            PricingTier::Professional => 15.0,
            PricingTier::Enterprise => 30.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PriceVersion
// ---------------------------------------------------------------------------

/// A single versioned price point for a model.
#[derive(Debug, Clone)]
pub struct PriceVersion {
    /// Monotonically increasing version number.
    pub version: u32,
    /// Unix epoch second from which this price is effective.
    pub effective_from_epoch: u64,
    /// List price per 1,000 input tokens (USD).
    pub input_price_per_1k: f64,
    /// List price per 1,000 output tokens (USD).
    pub output_price_per_1k: f64,
    /// Human-readable notes describing the change.
    pub notes: String,
}

// ---------------------------------------------------------------------------
// ModelPricing
// ---------------------------------------------------------------------------

/// All price versions for a single model.
#[derive(Debug, Clone)]
pub struct ModelPricing {
    /// Canonical model identifier.
    pub model_id: String,
    /// Chronological list of price versions (ascending by `effective_from_epoch`).
    pub versions: Vec<PriceVersion>,
    /// Version number of the currently active price.
    pub current_version: u32,
}

// ---------------------------------------------------------------------------
// PricingError
// ---------------------------------------------------------------------------

/// Errors from [`PricingEngine`] operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PricingError {
    /// No pricing is registered for the given model id.
    ModelNotFound(String),
    /// No price version was effective at the requested epoch.
    NoVersionAtEpoch,
    /// A price value was negative or otherwise invalid.
    InvalidPrice,
}

impl fmt::Display for PricingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PricingError::ModelNotFound(id) => write!(f, "model not found: {}", id),
            PricingError::NoVersionAtEpoch => {
                write!(f, "no price version was effective at the given epoch")
            }
            PricingError::InvalidPrice => write!(f, "price values must be non-negative"),
        }
    }
}

// ---------------------------------------------------------------------------
// PricingEngine
// ---------------------------------------------------------------------------

/// Dynamic pricing engine that manages per-model versioned pricing and applies
/// tier-based discounts.
pub struct PricingEngine {
    /// All registered models, keyed by model id.
    pub models: HashMap<String, ModelPricing>,
    /// Tier used when applying discounts.
    pub default_tier: PricingTier,
}

impl PricingEngine {
    /// Creates a new engine with the supplied default tier.
    pub fn new(tier: PricingTier) -> Self {
        PricingEngine {
            models: HashMap::new(),
            default_tier: tier,
        }
    }

    /// Registers a model with an initial price.  If the model already exists
    /// the call is a no-op (use [`update_price`] to revise).
    pub fn register_model(&mut self, model_id: String, input_per_1k: f64, output_per_1k: f64) {
        if self.models.contains_key(&model_id) {
            return;
        }
        let version = PriceVersion {
            version: 1,
            effective_from_epoch: 0,
            input_price_per_1k: input_per_1k,
            output_price_per_1k: output_per_1k,
            notes: "initial".to_string(),
        };
        self.models.insert(
            model_id.clone(),
            ModelPricing {
                model_id,
                versions: vec![version],
                current_version: 1,
            },
        );
    }

    /// Adds a new price version to an existing model.
    pub fn update_price(
        &mut self,
        model_id: &str,
        input_per_1k: f64,
        output_per_1k: f64,
        epoch: u64,
        notes: String,
    ) -> Result<(), PricingError> {
        if input_per_1k < 0.0 || output_per_1k < 0.0 {
            return Err(PricingError::InvalidPrice);
        }
        let pricing = self
            .models
            .get_mut(model_id)
            .ok_or_else(|| PricingError::ModelNotFound(model_id.to_string()))?;

        let next_version = pricing.current_version + 1;
        pricing.versions.push(PriceVersion {
            version: next_version,
            effective_from_epoch: epoch,
            input_price_per_1k: input_per_1k,
            output_price_per_1k: output_per_1k,
            notes,
        });
        pricing.current_version = next_version;
        Ok(())
    }

    /// Returns the price version that was effective at `epoch` for the given
    /// model, i.e. the latest version whose `effective_from_epoch <= epoch`.
    pub fn price_at(&self, model_id: &str, epoch: u64) -> Option<&PriceVersion> {
        let pricing = self.models.get(model_id)?;
        pricing
            .versions
            .iter()
            .filter(|v| v.effective_from_epoch <= epoch)
            .max_by_key(|v| v.effective_from_epoch)
    }

    /// Computes the cost (USD) for `input_tokens` + `output_tokens` using the
    /// model's current price, then applies the engine's tier discount.
    pub fn compute_cost(
        &self,
        model_id: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<f64, PricingError> {
        let pricing = self
            .models
            .get(model_id)
            .ok_or_else(|| PricingError::ModelNotFound(model_id.to_string()))?;

        // Use the highest-versioned price (current)
        let version = pricing
            .versions
            .iter()
            .max_by_key(|v| v.version)
            .ok_or(PricingError::NoVersionAtEpoch)?;

        let raw = (input_tokens as f64 / 1_000.0) * version.input_price_per_1k
            + (output_tokens as f64 / 1_000.0) * version.output_price_per_1k;

        let discount = self.default_tier.discount_pct() / 100.0;
        Ok(raw * (1.0 - discount))
    }

    /// Computes costs for a batch of `(model_id, input_tokens, output_tokens)` tuples.
    pub fn batch_cost(&self, requests: &[(String, u64, u64)]) -> Vec<Result<f64, PricingError>> {
        requests
            .iter()
            .map(|(model_id, input, output)| self.compute_cost(model_id, *input, *output))
            .collect()
    }

    /// Returns the full price history for a model.
    pub fn price_history(&self, model_id: &str) -> Option<&[PriceVersion]> {
        self.models.get(model_id).map(|p| p.versions.as_slice())
    }

    /// Returns the id of the model with the lowest cost for the given token
    /// volumes, or `None` if no models are registered.
    pub fn cheapest_model(&self, input_tokens: u64, output_tokens: u64) -> Option<String> {
        self.models
            .keys()
            .filter_map(|id| {
                self.compute_cost(id, input_tokens, output_tokens)
                    .ok()
                    .map(|cost| (id.clone(), cost))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> PricingEngine {
        let mut engine = PricingEngine::new(PricingTier::Professional);
        engine.register_model("gpt-4".to_string(), 30.0, 60.0);
        engine.register_model("gpt-3.5".to_string(), 2.0, 2.0);
        engine
    }

    #[test]
    fn test_basic_cost_computation() {
        let engine = make_engine();
        // gpt-4: 1000 input @ $30/1k + 1000 output @ $60/1k = $90 list, 15% off = $76.50
        let cost = engine.compute_cost("gpt-4", 1_000, 1_000).unwrap();
        let expected = 90.0 * 0.85;
        assert!((cost - expected).abs() < 1e-9, "cost={} expected={}", cost, expected);
    }

    #[test]
    fn test_tier_discount_applied() {
        let mut engine_free = PricingEngine::new(PricingTier::Free);
        engine_free.register_model("m".to_string(), 10.0, 10.0);
        let cost_free = engine_free.compute_cost("m", 1_000, 1_000).unwrap();

        let mut engine_ent = PricingEngine::new(PricingTier::Enterprise);
        engine_ent.register_model("m".to_string(), 10.0, 10.0);
        let cost_ent = engine_ent.compute_cost("m", 1_000, 1_000).unwrap();

        assert!(cost_free > cost_ent);
        assert!((cost_free - 20.0).abs() < 1e-9);
        assert!((cost_ent - 14.0).abs() < 1e-9);
    }

    #[test]
    fn test_price_at_returns_correct_version() {
        let mut engine = PricingEngine::new(PricingTier::Free);
        engine.register_model("m".to_string(), 10.0, 10.0);
        engine
            .update_price("m", 20.0, 20.0, 1_000, "v2".to_string())
            .unwrap();
        engine
            .update_price("m", 30.0, 30.0, 2_000, "v3".to_string())
            .unwrap();

        let v = engine.price_at("m", 500).unwrap();
        assert!((v.input_price_per_1k - 10.0).abs() < 1e-9);

        let v2 = engine.price_at("m", 1_500).unwrap();
        assert!((v2.input_price_per_1k - 20.0).abs() < 1e-9);

        let v3 = engine.price_at("m", 9_999).unwrap();
        assert!((v3.input_price_per_1k - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_cheapest_model() {
        let engine = make_engine();
        let cheapest = engine.cheapest_model(1_000, 1_000).unwrap();
        assert_eq!(cheapest, "gpt-3.5");
    }

    #[test]
    fn test_update_adds_version() {
        let mut engine = PricingEngine::new(PricingTier::Free);
        engine.register_model("m".to_string(), 1.0, 1.0);
        engine
            .update_price("m", 2.0, 2.0, 100, "new pricing".to_string())
            .unwrap();
        let history = engine.price_history("m").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].version, 2);
        assert!((history[1].input_price_per_1k - 2.0).abs() < 1e-9);
    }
}
