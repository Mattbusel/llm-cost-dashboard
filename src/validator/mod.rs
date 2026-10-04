//! # API Key Validator
//!
//! Validates API keys for Anthropic, OpenAI, and Google (Gemini) by making a
//! lightweight authenticated request to each provider's model-list endpoint.
//!
//! Requires the `webhooks` crate feature (enabled by default) which pulls in
//! `reqwest`.  When the feature is disabled, every `validate` call immediately
//! returns an error result.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use llm_cost_dashboard::validator::AnthropicValidator;
//!
//! # async fn run() {
//! let v = AnthropicValidator::new();
//! let result = v.validate("sk-ant-...").await;
//! println!("valid: {}", result.is_valid);
//! # }
//! ```

/// Result of validating a single API key.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the key is valid and accepted by the provider.
    pub is_valid: bool,
    /// Provider-reported account tier or plan name, if available.
    pub tier: Option<String>,
    /// Remaining quota reported by the provider, if available.
    pub remaining_quota: Option<u64>,
    /// Human-readable error message when `is_valid` is `false`.
    pub error_message: Option<String>,
}

impl ValidationResult {
    #[cfg_attr(not(feature = "webhooks"), allow(dead_code))]
    fn ok(tier: Option<String>) -> Self {
        Self { is_valid: true, tier, remaining_quota: None, error_message: None }
    }

    fn err(msg: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            tier: None,
            remaining_quota: None,
            error_message: Some(msg.into()),
        }
    }
}

// ── Anthropic ────────────────────────────────────────────────────────────────

/// Validates Anthropic API keys via `GET https://api.anthropic.com/v1/models`.
#[derive(Debug, Default)]
pub struct AnthropicValidator;

impl AnthropicValidator {
    /// Create a new Anthropic validator.
    pub fn new() -> Self {
        Self
    }

    /// Validate `key` by making a live request to the Anthropic API.
    pub async fn validate(&self, key: &str) -> ValidationResult {
        #[cfg(feature = "webhooks")]
        {
            use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
            let client = match reqwest::Client::builder().build() {
                Ok(c) => c,
                Err(e) => return ValidationResult::err(format!("HTTP client error: {e}")),
            };
            let mut headers = HeaderMap::new();
            let key_val = match HeaderValue::from_str(key) {
                Ok(v) => v,
                Err(_) => return ValidationResult::err("invalid API key format"),
            };
            headers.insert("x-api-key", key_val);
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            match client.get("https://api.anthropic.com/v1/models").headers(headers).send().await {
                Ok(r) if r.status().is_success() => ValidationResult::ok(Some("anthropic".into())),
                Ok(r) if r.status().as_u16() == 401 => ValidationResult::err("invalid API key (401)"),
                Ok(r) if r.status().as_u16() == 403 => ValidationResult::err("forbidden (403)"),
                Ok(r) => ValidationResult::err(format!("unexpected status {}", r.status())),
                Err(e) => ValidationResult::err(format!("request failed: {e}")),
            }
        }
        #[cfg(not(feature = "webhooks"))]
        {
            let _ = key;
            ValidationResult::err("HTTP support not compiled in (enable 'webhooks' feature)")
        }
    }

    /// Provider name label.
    pub fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

// ── OpenAI ───────────────────────────────────────────────────────────────────

/// Validates OpenAI API keys via `GET https://api.openai.com/v1/models`.
#[derive(Debug, Default)]
pub struct OpenAiValidator;

impl OpenAiValidator {
    /// Create a new OpenAI validator.
    pub fn new() -> Self {
        Self
    }

    /// Validate `key` by making a live request to the OpenAI API.
    pub async fn validate(&self, key: &str) -> ValidationResult {
        #[cfg(feature = "webhooks")]
        {
            use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
            let client = match reqwest::Client::builder().build() {
                Ok(c) => c,
                Err(e) => return ValidationResult::err(format!("HTTP client error: {e}")),
            };
            let bearer = format!("Bearer {key}");
            let auth_val = match HeaderValue::from_str(&bearer) {
                Ok(v) => v,
                Err(_) => return ValidationResult::err("invalid API key format"),
            };
            let mut headers = HeaderMap::new();
            headers.insert(AUTHORIZATION, auth_val);
            match client.get("https://api.openai.com/v1/models").headers(headers).send().await {
                Ok(r) if r.status().is_success() => ValidationResult::ok(Some("openai".into())),
                Ok(r) if r.status().as_u16() == 401 => ValidationResult::err("invalid API key (401)"),
                Ok(r) if r.status().as_u16() == 403 => ValidationResult::err("forbidden (403)"),
                Ok(r) => ValidationResult::err(format!("unexpected status {}", r.status())),
                Err(e) => ValidationResult::err(format!("request failed: {e}")),
            }
        }
        #[cfg(not(feature = "webhooks"))]
        {
            let _ = key;
            ValidationResult::err("HTTP support not compiled in (enable 'webhooks' feature)")
        }
    }

    /// Provider name label.
    pub fn provider_name(&self) -> &'static str {
        "openai"
    }
}

// ── Google ───────────────────────────────────────────────────────────────────

/// Validates Google (Gemini) API keys via the Generative Language API.
#[derive(Debug, Default)]
pub struct GoogleValidator;

impl GoogleValidator {
    /// Create a new Google validator.
    pub fn new() -> Self {
        Self
    }

    /// Validate `key` by making a live request to the Google Generative Language API.
    pub async fn validate(&self, key: &str) -> ValidationResult {
        #[cfg(feature = "webhooks")]
        {
            let client = match reqwest::Client::builder().build() {
                Ok(c) => c,
                Err(e) => return ValidationResult::err(format!("HTTP client error: {e}")),
            };
            let url = format!("https://generativelanguage.googleapis.com/v1/models?key={key}");
            match client.get(&url).send().await {
                Ok(r) if r.status().is_success() => ValidationResult::ok(Some("google".into())),
                Ok(r) if r.status().as_u16() == 400 => {
                    ValidationResult::err("invalid API key or bad request (400)")
                }
                Ok(r) if r.status().as_u16() == 403 => ValidationResult::err("forbidden (403)"),
                Ok(r) => ValidationResult::err(format!("unexpected status {}", r.status())),
                Err(e) => ValidationResult::err(format!("request failed: {e}")),
            }
        }
        #[cfg(not(feature = "webhooks"))]
        {
            let _ = key;
            ValidationResult::err("HTTP support not compiled in (enable 'webhooks' feature)")
        }
    }

    /// Provider name label.
    pub fn provider_name(&self) -> &'static str {
        "google"
    }
}

// ── MultiValidator ───────────────────────────────────────────────────────────

/// A (provider, key) pair to validate.
#[derive(Debug, Clone)]
pub struct KeyConfig {
    /// Provider name: `"anthropic"`, `"openai"`, `"google"`, or `"gemini"`.
    pub provider: String,
    /// The API key string.
    pub key: String,
}

/// Outcome of validating a single configured key.
#[derive(Debug, Clone)]
pub struct MultiValidationResult {
    /// Provider name.
    pub provider: String,
    /// Validation result for this key.
    pub result: ValidationResult,
}

impl MultiValidationResult {
    /// Short display string showing provider and pass/fail.
    pub fn status_str(&self) -> String {
        if self.result.is_valid {
            format!("{}: OK", self.provider)
        } else {
            let msg = self.result.error_message.as_deref().unwrap_or("unknown error");
            format!("{}: INVALID ({})", self.provider, msg)
        }
    }
}

/// Validates all configured API keys and aggregates results.
///
/// Keys are validated sequentially.  To run validations in parallel, spawn
/// this into its own Tokio task.
pub struct MultiValidator {
    configs: Vec<KeyConfig>,
}

impl MultiValidator {
    /// Create a new multi-validator from the given key configurations.
    pub fn new(configs: Vec<KeyConfig>) -> Self {
        Self { configs }
    }

    /// Validate every configured key and return one result per key.
    pub async fn validate_all(&self) -> Vec<MultiValidationResult> {
        let mut results = Vec::with_capacity(self.configs.len());
        for cfg in &self.configs {
            let result = match cfg.provider.to_lowercase().as_str() {
                "anthropic" => AnthropicValidator::new().validate(&cfg.key).await,
                "openai" => OpenAiValidator::new().validate(&cfg.key).await,
                "google" | "gemini" => GoogleValidator::new().validate(&cfg.key).await,
                other => ValidationResult::err(format!("unknown provider '{other}'")),
            };
            results.push(MultiValidationResult { provider: cfg.provider.clone(), result });
        }
        results
    }

    /// Returns `true` if every configured key validated successfully.
    pub async fn all_valid(&self) -> bool {
        self.validate_all().await.iter().all(|r| r.result.is_valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_ok() {
        let r = ValidationResult::ok(Some("free".into()));
        assert!(r.is_valid);
        assert_eq!(r.tier.as_deref(), Some("free"));
        assert!(r.error_message.is_none());
    }

    #[test]
    fn test_validation_result_err() {
        let r = ValidationResult::err("bad key");
        assert!(!r.is_valid);
        assert_eq!(r.error_message.as_deref(), Some("bad key"));
        assert!(r.tier.is_none());
    }

    #[test]
    fn test_multi_result_status_str_valid() {
        let r = MultiValidationResult {
            provider: "openai".into(),
            result: ValidationResult::ok(None),
        };
        assert!(r.status_str().contains("OK"));
        assert!(r.status_str().contains("openai"));
    }

    #[test]
    fn test_multi_result_status_str_invalid() {
        let r = MultiValidationResult {
            provider: "anthropic".into(),
            result: ValidationResult::err("401"),
        };
        assert!(r.status_str().contains("INVALID"));
        assert!(r.status_str().contains("anthropic"));
    }

    #[test]
    fn test_multi_validator_new() {
        let configs = vec![KeyConfig { provider: "openai".into(), key: "sk-test".into() }];
        let v = MultiValidator::new(configs);
        assert_eq!(v.configs.len(), 1);
    }

    #[tokio::test]
    async fn test_multi_validator_unknown_provider() {
        let configs = vec![KeyConfig { provider: "unknown".into(), key: "key".into() }];
        let v = MultiValidator::new(configs);
        let results = v.validate_all().await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].result.is_valid);
        assert!(results[0]
            .result
            .error_message
            .as_deref()
            .unwrap_or("")
            .contains("unknown provider"));
    }

    #[test]
    fn test_anthropic_provider_name() {
        assert_eq!(AnthropicValidator::new().provider_name(), "anthropic");
    }

    #[test]
    fn test_openai_provider_name() {
        assert_eq!(OpenAiValidator::new().provider_name(), "openai");
    }

    #[test]
    fn test_google_provider_name() {
        assert_eq!(GoogleValidator::new().provider_name(), "google");
    }
}
