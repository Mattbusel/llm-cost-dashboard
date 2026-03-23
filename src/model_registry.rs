//! # Model Registry
//!
//! Registry of LLM models with rich metadata, capability search, and cost queries.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::model_registry::{ModelRegistry, ModelCapability};
//!
//! let registry = ModelRegistry::default_registry();
//! let models = registry.search_by_capability(&ModelCapability::FunctionCalling);
//! assert!(!models.is_empty());
//!
//! let cheapest = registry.cheapest_with_capability(&ModelCapability::TextGeneration);
//! assert!(cheapest.is_some());
//! ```

// ── ModelCapability ───────────────────────────────────────────────────────────

/// Capabilities a model may support.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelCapability {
    /// Generates free-form text responses.
    TextGeneration,
    /// Assists with code generation and completion.
    CodeCompletion,
    /// Produces dense vector representations of text.
    Embeddings,
    /// Accepts images as part of the prompt context.
    ImageUnderstanding,
    /// Can call structured functions / tools provided in the prompt.
    FunctionCalling,
    /// Reliably outputs valid JSON when requested.
    JsonMode,
    /// Supports streaming token-by-token output.
    Streaming,
    /// Supports fine-tuning on custom datasets.
    FineTuning,
}

impl ModelCapability {
    /// Human-readable description of the capability.
    pub fn description(&self) -> &str {
        match self {
            Self::TextGeneration => "Generates free-form natural language text",
            Self::CodeCompletion => "Assists with code generation and completion",
            Self::Embeddings => "Produces dense vector embeddings for text",
            Self::ImageUnderstanding => "Accepts and reasons about image inputs",
            Self::FunctionCalling => "Calls structured functions / tools from prompts",
            Self::JsonMode => "Guarantees structurally valid JSON output",
            Self::Streaming => "Streams tokens to the client incrementally",
            Self::FineTuning => "Supports fine-tuning on custom datasets",
        }
    }
}

// ── ModelInfo ─────────────────────────────────────────────────────────────────

/// Metadata for a single LLM model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Canonical model identifier (e.g. `"gpt-4o"`).
    pub model_id: String,
    /// Provider name (e.g. `"OpenAI"`, `"Anthropic"`, `"Google"`).
    pub provider: String,
    /// Approximate release date (Unix epoch seconds).
    pub release_date: u64,
    /// Input cost in USD per 1 000 tokens.
    pub input_cost_per_1k: f64,
    /// Output cost in USD per 1 000 tokens.
    pub output_cost_per_1k: f64,
    /// Maximum context window in tokens.
    pub context_window: u64,
    /// Maximum output tokens per request.
    pub max_output_tokens: u64,
    /// Set of supported capabilities.
    pub capabilities: Vec<ModelCapability>,
    /// Whether this model is no longer recommended for new use.
    pub is_deprecated: bool,
    /// Model ID of the recommended replacement, if deprecated.
    pub successor_id: Option<String>,
}

impl ModelInfo {
    /// Combined cost per 1 000 tokens (input + output), used for cheapest-model queries.
    pub fn combined_cost_per_1k(&self) -> f64 {
        self.input_cost_per_1k + self.output_cost_per_1k
    }

    /// Returns `true` if this model has the given capability.
    pub fn has_capability(&self, cap: &ModelCapability) -> bool {
        self.capabilities.contains(cap)
    }
}

// ── ModelRegistry ─────────────────────────────────────────────────────────────

/// Registry of LLM models with metadata lookup and capability search.
#[derive(Default)]
pub struct ModelRegistry {
    models: Vec<ModelInfo>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Register a model.  Replaces any existing entry with the same `model_id`.
    pub fn register(&mut self, info: ModelInfo) {
        if let Some(pos) = self.models.iter().position(|m| m.model_id == info.model_id) {
            self.models[pos] = info;
        } else {
            self.models.push(info);
        }
    }

    /// Look up a model by exact `model_id`.
    pub fn get(&self, model_id: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.model_id == model_id)
    }

    /// Return all models that support `cap`.
    pub fn search_by_capability(&self, cap: &ModelCapability) -> Vec<&ModelInfo> {
        self.models.iter().filter(|m| m.has_capability(cap)).collect()
    }

    /// Return the non-deprecated model with the lowest combined cost that supports `cap`.
    pub fn cheapest_with_capability(&self, cap: &ModelCapability) -> Option<&ModelInfo> {
        self.models
            .iter()
            .filter(|m| m.has_capability(cap) && !m.is_deprecated)
            .min_by(|a, b| {
                a.combined_cost_per_1k()
                    .partial_cmp(&b.combined_cost_per_1k())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Return the non-deprecated model with the largest context window.
    pub fn largest_context(&self) -> Option<&ModelInfo> {
        self.models
            .iter()
            .filter(|m| !m.is_deprecated)
            .max_by_key(|m| m.context_window)
    }

    /// Return all deprecated models.
    pub fn deprecated_models(&self) -> Vec<&ModelInfo> {
        self.models.iter().filter(|m| m.is_deprecated).collect()
    }

    /// Return unique provider names (sorted).
    pub fn providers(&self) -> Vec<String> {
        let mut providers: Vec<String> = self
            .models
            .iter()
            .map(|m| m.provider.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        providers.sort();
        providers
    }

    /// Return all models from a given provider.
    pub fn models_by_provider(&self, provider: &str) -> Vec<&ModelInfo> {
        self.models.iter().filter(|m| m.provider == provider).collect()
    }

    /// Pre-populated registry with ~8 well-known models.
    pub fn default_registry() -> ModelRegistry {
        let mut r = ModelRegistry::new();

        // OpenAI
        r.register(ModelInfo {
            model_id: "gpt-4o".to_string(),
            provider: "OpenAI".to_string(),
            release_date: 1_715_126_400, // 2024-05-08
            input_cost_per_1k: 0.005,
            output_cost_per_1k: 0.015,
            context_window: 128_000,
            max_output_tokens: 4_096,
            capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::CodeCompletion,
                ModelCapability::ImageUnderstanding,
                ModelCapability::FunctionCalling,
                ModelCapability::JsonMode,
                ModelCapability::Streaming,
            ],
            is_deprecated: false,
            successor_id: None,
        });

        r.register(ModelInfo {
            model_id: "gpt-4o-mini".to_string(),
            provider: "OpenAI".to_string(),
            release_date: 1_721_692_800, // 2024-07-23
            input_cost_per_1k: 0.000150,
            output_cost_per_1k: 0.000600,
            context_window: 128_000,
            max_output_tokens: 16_384,
            capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::CodeCompletion,
                ModelCapability::FunctionCalling,
                ModelCapability::JsonMode,
                ModelCapability::Streaming,
            ],
            is_deprecated: false,
            successor_id: None,
        });

        r.register(ModelInfo {
            model_id: "text-embedding-3-large".to_string(),
            provider: "OpenAI".to_string(),
            release_date: 1_706_140_800, // 2024-01-25
            input_cost_per_1k: 0.000130,
            output_cost_per_1k: 0.0,
            context_window: 8_191,
            max_output_tokens: 0,
            capabilities: vec![ModelCapability::Embeddings],
            is_deprecated: false,
            successor_id: None,
        });

        r.register(ModelInfo {
            model_id: "gpt-3.5-turbo".to_string(),
            provider: "OpenAI".to_string(),
            release_date: 1_669_766_400, // 2022-11-30
            input_cost_per_1k: 0.0005,
            output_cost_per_1k: 0.0015,
            context_window: 16_385,
            max_output_tokens: 4_096,
            capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::FunctionCalling,
                ModelCapability::Streaming,
                ModelCapability::FineTuning,
            ],
            is_deprecated: true,
            successor_id: Some("gpt-4o-mini".to_string()),
        });

        // Anthropic
        r.register(ModelInfo {
            model_id: "claude-3-5-sonnet-20241022".to_string(),
            provider: "Anthropic".to_string(),
            release_date: 1_729_555_200, // 2024-10-22
            input_cost_per_1k: 0.003,
            output_cost_per_1k: 0.015,
            context_window: 200_000,
            max_output_tokens: 8_096,
            capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::CodeCompletion,
                ModelCapability::ImageUnderstanding,
                ModelCapability::FunctionCalling,
                ModelCapability::JsonMode,
                ModelCapability::Streaming,
            ],
            is_deprecated: false,
            successor_id: None,
        });

        r.register(ModelInfo {
            model_id: "claude-3-haiku-20240307".to_string(),
            provider: "Anthropic".to_string(),
            release_date: 1_709_769_600, // 2024-03-07
            input_cost_per_1k: 0.00025,
            output_cost_per_1k: 0.00125,
            context_window: 200_000,
            max_output_tokens: 4_096,
            capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::CodeCompletion,
                ModelCapability::FunctionCalling,
                ModelCapability::Streaming,
            ],
            is_deprecated: false,
            successor_id: None,
        });

        // Google
        r.register(ModelInfo {
            model_id: "gemini-1.5-pro".to_string(),
            provider: "Google".to_string(),
            release_date: 1_709_251_200, // 2024-03-01
            input_cost_per_1k: 0.00125,
            output_cost_per_1k: 0.005,
            context_window: 1_000_000,
            max_output_tokens: 8_192,
            capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::CodeCompletion,
                ModelCapability::ImageUnderstanding,
                ModelCapability::FunctionCalling,
                ModelCapability::JsonMode,
                ModelCapability::Streaming,
            ],
            is_deprecated: false,
            successor_id: None,
        });

        r.register(ModelInfo {
            model_id: "gemini-1.5-flash".to_string(),
            provider: "Google".to_string(),
            release_date: 1_715_558_400, // 2024-05-13
            input_cost_per_1k: 0.000075,
            output_cost_per_1k: 0.000300,
            context_window: 1_000_000,
            max_output_tokens: 8_192,
            capabilities: vec![
                ModelCapability::TextGeneration,
                ModelCapability::CodeCompletion,
                ModelCapability::ImageUnderstanding,
                ModelCapability::FunctionCalling,
                ModelCapability::Streaming,
            ],
            is_deprecated: false,
            successor_id: None,
        });

        r
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> ModelRegistry {
        ModelRegistry::default_registry()
    }

    #[test]
    fn test_get_known_model() {
        let r = registry();
        assert!(r.get("gpt-4o").is_some());
        assert!(r.get("nonexistent").is_none());
    }

    #[test]
    fn test_search_by_capability_text_generation() {
        let r = registry();
        let models = r.search_by_capability(&ModelCapability::TextGeneration);
        // All except the embedding model should appear.
        assert!(models.len() >= 5);
    }

    #[test]
    fn test_search_by_capability_embeddings() {
        let r = registry();
        let models = r.search_by_capability(&ModelCapability::Embeddings);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model_id, "text-embedding-3-large");
    }

    #[test]
    fn test_cheapest_with_capability() {
        let r = registry();
        let cheapest = r.cheapest_with_capability(&ModelCapability::TextGeneration);
        assert!(cheapest.is_some());
        // gemini-1.5-flash has input=0.000075 + output=0.000300 = 0.000375
        let m = cheapest.unwrap();
        // Just verify it exists and is not deprecated
        assert!(!m.is_deprecated);
    }

    #[test]
    fn test_largest_context() {
        let r = registry();
        let m = r.largest_context().unwrap();
        // Gemini 1.5 Pro/Flash have 1M context
        assert_eq!(m.context_window, 1_000_000);
    }

    #[test]
    fn test_deprecated_models() {
        let r = registry();
        let deprecated = r.deprecated_models();
        assert!(!deprecated.is_empty());
        assert!(deprecated.iter().all(|m| m.is_deprecated));
        assert!(deprecated.iter().any(|m| m.model_id == "gpt-3.5-turbo"));
    }

    #[test]
    fn test_providers() {
        let r = registry();
        let providers = r.providers();
        assert!(providers.contains(&"OpenAI".to_string()));
        assert!(providers.contains(&"Anthropic".to_string()));
        assert!(providers.contains(&"Google".to_string()));
    }

    #[test]
    fn test_models_by_provider() {
        let r = registry();
        let openai = r.models_by_provider("OpenAI");
        assert!(openai.len() >= 3);
        assert!(openai.iter().all(|m| m.provider == "OpenAI"));
    }

    #[test]
    fn test_register_replaces_existing() {
        let mut r = ModelRegistry::new();
        r.register(ModelInfo {
            model_id: "test-model".to_string(),
            provider: "TestCo".to_string(),
            release_date: 0,
            input_cost_per_1k: 1.0,
            output_cost_per_1k: 1.0,
            context_window: 1000,
            max_output_tokens: 100,
            capabilities: vec![],
            is_deprecated: false,
            successor_id: None,
        });
        r.register(ModelInfo {
            model_id: "test-model".to_string(),
            provider: "TestCo".to_string(),
            release_date: 0,
            input_cost_per_1k: 0.5,
            output_cost_per_1k: 0.5,
            context_window: 2000,
            max_output_tokens: 200,
            capabilities: vec![],
            is_deprecated: false,
            successor_id: None,
        });
        assert_eq!(r.models_by_provider("TestCo").len(), 1);
        assert_eq!(r.get("test-model").unwrap().context_window, 2000);
    }

    #[test]
    fn test_capability_description() {
        assert!(!ModelCapability::TextGeneration.description().is_empty());
        assert!(!ModelCapability::Embeddings.description().is_empty());
        assert!(!ModelCapability::FunctionCalling.description().is_empty());
    }

    #[test]
    fn test_function_calling_models() {
        let r = registry();
        let fc = r.search_by_capability(&ModelCapability::FunctionCalling);
        assert!(fc.len() >= 4);
    }

    #[test]
    fn test_image_understanding_models() {
        let r = registry();
        let img = r.search_by_capability(&ModelCapability::ImageUnderstanding);
        // gpt-4o, claude-3-5-sonnet, gemini-1.5-pro, gemini-1.5-flash
        assert!(img.len() >= 3);
    }

    #[test]
    fn test_combined_cost_per_1k() {
        let r = registry();
        let m = r.get("gpt-4o").unwrap();
        assert!((m.combined_cost_per_1k() - 0.020).abs() < 1e-9);
    }
}
