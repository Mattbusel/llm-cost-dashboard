//! Per-model token pricing table (USD per 1 million tokens).
//!
//! All prices are stored as `f64` values representing US dollars per 1,000,000
//! tokens.  Use [`compute_cost`] for convenience, [`lookup_known`] when you
//! need to know whether a model is priced at all, or [`lookup`] for the raw
//! rates with the fallback applied.
//!
//! The built-in table can be extended or overridden at run time with
//! [`load_prices_json`] (LiteLLM's `model_prices_and_context_window.json`
//! format, or a simple `{"model": {"input_usd_per_1m": .., "output_usd_per_1m": ..}}`
//! map) or [`set_price`]. Overrides apply process-wide, to every
//! [`crate::CostRecord`] created afterwards.
//!
//! Dated model ids (`claude-sonnet-4-5-20250929`, `gpt-4o-2024-08-06`) fall
//! back to the undated entry when the dated one is not listed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

use crate::error::DashboardError;

/// Statically known per-model pricing entries.
///
/// Each tuple is `(model_id, input_usd_per_1m, output_usd_per_1m)`.
/// Model IDs are matched case-insensitively by [`lookup`].
///
/// Last updated: 2026-10-03
pub const PRICING: &[(&str, f64, f64)] = &[
    // ── Anthropic / Claude 5 family (Anthropic list prices, 2026-09) ────────
    ("claude-fable-5-1", 10.00, 50.00),
    ("claude-fable-5", 10.00, 50.00),
    ("claude-opus-5-5", 4.00, 20.00),
    ("claude-opus-5", 5.00, 25.00),
    ("claude-sonnet-5-5", 2.00, 10.00),
    ("claude-sonnet-5", 2.00, 10.00),
    // ── Anthropic — Claude 4 family ─────────────────────────────────────────
    ("claude-opus-4-8", 5.00, 25.00),
    ("claude-opus-4-7", 5.00, 25.00),
    ("claude-opus-4-6", 5.00, 25.00),
    ("claude-opus-4-5", 5.00, 25.00),
    ("claude-opus-4-1", 15.00, 75.00),
    ("claude-opus-4", 15.00, 75.00),
    ("claude-sonnet-4-5", 3.00, 15.00),
    ("claude-sonnet-4", 3.00, 15.00),
    ("claude-sonnet-4-6", 3.00, 15.00),
    ("claude-haiku-4-5", 1.00, 5.00),
    // ── Anthropic — Claude 3.5 family ───────────────────────────────────────
    ("claude-3-5-sonnet-20241022", 3.00, 15.00),
    ("claude-3-5-haiku-20241022", 0.80, 4.00),
    ("claude-3-5-sonnet-20240620", 3.00, 15.00),
    // ── Anthropic — Claude 3 family ─────────────────────────────────────────
    ("claude-3-opus-20240229", 15.00, 75.00),
    ("claude-3-sonnet-20240229", 3.00, 15.00),
    ("claude-3-haiku-20240307", 0.25, 1.25),
    // ── OpenAI / GPT-5 and GPT-4.1 families ────────────────────────────────
    ("gpt-5", 1.25, 10.00),
    ("gpt-5-mini", 0.25, 2.00),
    ("gpt-5-nano", 0.05, 0.40),
    ("gpt-4.1", 2.00, 8.00),
    ("gpt-4.1-mini", 0.40, 1.60),
    ("gpt-4.1-nano", 0.10, 0.40),
    // ── OpenAI — GPT-4o family ──────────────────────────────────────────────
    ("gpt-4o", 2.50, 10.00),
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4-turbo", 10.00, 30.00),
    ("gpt-4.5-preview", 75.00, 150.00),
    ("chatgpt-4o-latest", 5.00, 15.00),
    // ── OpenAI — o-series reasoning models ──────────────────────────────────
    ("o1", 15.00, 60.00),
    ("o1-preview", 15.00, 60.00),
    ("o1-mini", 1.10, 4.40),
    ("o3", 2.00, 8.00),
    ("o3-mini", 1.10, 4.40),
    ("o4-mini", 1.10, 4.40),
    // ── OpenAI — Legacy ─────────────────────────────────────────────────────
    ("gpt-4", 30.00, 60.00),
    ("gpt-3.5-turbo", 0.50, 1.50),
    ("gpt-3.5-turbo-instruct", 1.50, 2.00),
    // ── Google — Gemini 2 family ─────────────────────────────────────────────
    ("gemini-2.5-pro", 1.25, 10.00),
    ("gemini-2.5-flash", 0.30, 2.50),
    ("gemini-2.5-flash-lite", 0.10, 0.40),
    ("gemini-2.0-flash", 0.10, 0.40),
    ("gemini-2.0-flash-lite", 0.075, 0.30),
    ("gemini-2.0-flash-thinking", 0.15, 0.60),
    // ── Google — Gemini 1.5 family ───────────────────────────────────────────
    ("gemini-1.5-pro", 1.25, 5.00),
    ("gemini-1.5-flash", 0.075, 0.30),
    ("gemini-1.5-flash-8b", 0.0375, 0.15),
    // ── DeepSeek ─────────────────────────────────────────────────────────────
    ("deepseek-r1", 0.55, 2.19),
    ("deepseek-v3", 0.27, 1.10),
    ("deepseek-v2-5", 0.14, 0.28),
    ("deepseek-chat", 0.28, 0.42),
    ("deepseek-coder", 0.14, 0.28),
    ("deepseek-r1-distill-llama-70b", 0.55, 2.19),
    ("deepseek-r1-distill-qwen-32b", 0.55, 2.19),
    // ── Mistral ──────────────────────────────────────────────────────────────
    ("mistral-large-2411", 2.00, 6.00),
    ("mistral-large-2407", 3.00, 9.00),
    ("mistral-small-2501", 0.10, 0.30),
    ("mistral-small-2402", 1.00, 3.00),
    ("mistral-nemo", 0.15, 0.15),
    ("codestral-2501", 0.30, 0.90),
    ("pixtral-large-2411", 2.00, 6.00),
    ("pixtral-12b-2409", 0.15, 0.15),
    ("ministral-8b-2410", 0.10, 0.10),
    ("ministral-3b-2410", 0.04, 0.04),
    // ── Meta / Llama (via Together AI / Groq) ────────────────────────────────
    ("meta-llama/llama-3.1-405b-instruct-turbo", 5.00, 5.00),
    ("meta-llama/llama-3.1-70b-instruct-turbo", 0.88, 0.88),
    ("meta-llama/llama-3.1-8b-instruct-turbo", 0.18, 0.18),
    ("meta-llama/llama-3.3-70b-instruct-turbo", 0.88, 0.88),
    ("meta-llama/llama-3.2-90b-vision-instruct-turbo", 1.20, 1.20),
    ("meta-llama/llama-3.2-11b-vision-instruct-turbo", 0.18, 0.18),
    ("llama-3.3-70b-versatile", 0.59, 0.79),    // Groq
    ("llama-3.1-70b-versatile", 0.59, 0.79),    // Groq
    ("llama-3.1-8b-instant", 0.05, 0.08),        // Groq
    // ── xAI Grok ─────────────────────────────────────────────────────────────
    ("grok-3", 3.00, 15.00),
    ("grok-3-mini", 0.30, 0.50),
    ("grok-2-1212", 2.00, 10.00),
    ("grok-2-vision-1212", 2.00, 10.00),
    ("grok-beta", 5.00, 15.00),
    // ── Cohere ───────────────────────────────────────────────────────────────
    ("command-r-plus-08-2024", 2.50, 10.00),
    ("command-r-08-2024", 0.15, 0.60),
    ("command-a-03-2025", 2.50, 10.00),
    ("command-r7b-12-2024", 0.0375, 0.15),
    // ── Perplexity ───────────────────────────────────────────────────────────
    ("sonar-pro", 3.00, 15.00),
    ("sonar", 1.00, 1.00),
    ("sonar-reasoning-pro", 2.00, 8.00),
    ("sonar-reasoning", 1.00, 5.00),
    // ── Amazon (Bedrock) ─────────────────────────────────────────────────────
    ("amazon.nova-pro-v1:0", 0.80, 3.20),
    ("amazon.nova-lite-v1:0", 0.06, 0.24),
    ("amazon.nova-micro-v1:0", 0.035, 0.14),
    ("amazon.titan-text-express-v1", 0.20, 0.60),
    ("amazon.titan-text-lite-v1", 0.30, 0.40),
    // ── Alibaba Qwen ─────────────────────────────────────────────────────────
    ("qwen-max", 1.60, 6.40),
    ("qwen-plus", 0.40, 1.20),
    ("qwen-turbo", 0.05, 0.20),
    ("qwen2.5-72b-instruct", 0.90, 0.90),
    ("qwen2.5-7b-instruct", 0.10, 0.10),
    // ── Writer ───────────────────────────────────────────────────────────────
    ("palmyra-x-004", 5.00, 15.00),
    ("palmyra-x-003-instruct", 1.50, 2.00),
    // ── AI21 Labs ────────────────────────────────────────────────────────────
    ("jamba-1.5-large", 2.00, 8.00),
    ("jamba-1.5-mini", 0.20, 0.40),
];

/// Fallback pricing used when the model is not found in [`PRICING`].
///
/// This is a mid-range estimate to avoid wildly incorrect costs for unknown
/// models.  The [`crate::error::DashboardError::UnknownModel`] variant is
/// available for callers that wish to surface the absence explicitly.
pub const FALLBACK_PRICING: (f64, f64) = (5.00, 15.00);

/// One model's rates, in USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ModelPrice {
    /// Input (prompt) tokens.
    pub input_per_1m: f64,
    /// Output (completion) tokens.
    pub output_per_1m: f64,
    /// Prompt-cache reads; `None` means 10% of the input rate (Anthropic's rate).
    pub cache_read_per_1m: Option<f64>,
    /// Prompt-cache writes (5-minute cache); `None` means 125% of the input rate.
    pub cache_write_per_1m: Option<f64>,
}

impl ModelPrice {
    /// Input and output rates in USD per million tokens, default cache rates.
    pub fn new(input_per_1m: f64, output_per_1m: f64) -> Self {
        Self {
            input_per_1m,
            output_per_1m,
            cache_read_per_1m: None,
            cache_write_per_1m: None,
        }
    }

    /// Set explicit cache read and write rates (USD per million tokens).
    pub fn with_cache_rates(mut self, read_per_1m: f64, write_per_1m: f64) -> Self {
        self.cache_read_per_1m = Some(read_per_1m);
        self.cache_write_per_1m = Some(write_per_1m);
        self
    }

    /// Cache-read rate, defaulting to 10% of input.
    pub fn cache_read_rate(&self) -> f64 {
        self.cache_read_per_1m.unwrap_or(self.input_per_1m * 0.10)
    }

    /// 5-minute cache-write rate, defaulting to 125% of input.
    pub fn cache_write_rate(&self) -> f64 {
        self.cache_write_per_1m.unwrap_or(self.input_per_1m * 1.25)
    }

    /// 1-hour cache-write rate: 200% of input (Anthropic's rate).
    pub fn cache_write_1h_rate(&self) -> f64 {
        self.input_per_1m * 2.0
    }
}

/// Published cached-input (prompt-cache read) rates, USD per million tokens,
/// for models whose rate is not the default 10% of input. OpenAI bills
/// cached input at 50% (GPT-4o, o-series), 25% (GPT-4.1) or 10% (GPT-5).
pub const CACHE_READ_RATES: &[(&str, f64)] = &[
    ("claude-fable-5-1", 0.25),
    ("claude-opus-5-5", 0.20),
    ("claude-sonnet-5-5", 0.20),
    ("gpt-4o", 1.25),
    ("gpt-4o-mini", 0.075),
    ("gpt-4.1", 0.50),
    ("gpt-4.1-mini", 0.10),
    ("gpt-4.1-nano", 0.025),
    ("gpt-5", 0.125),
    ("gpt-5-mini", 0.025),
    ("gpt-5-nano", 0.005),
    ("o3", 0.50),
    ("o4-mini", 0.275),
    ("o3-mini", 0.55),
    ("o1", 7.50),
];

fn overrides() -> &'static RwLock<HashMap<String, ModelPrice>> {
    static O: OnceLock<RwLock<HashMap<String, ModelPrice>>> = OnceLock::new();
    O.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Add or replace one model's price for this process.
///
/// ```
/// use llm_cost_dashboard::cost::pricing::{compute_cost, set_price, ModelPrice};
///
/// set_price("my-finetune", ModelPrice::new(0.5, 1.5));
/// assert!((compute_cost("my-finetune", 1_000_000, 0) - 0.5).abs() < 1e-12);
/// ```
pub fn set_price(model: &str, price: ModelPrice) {
    if let Ok(mut m) = overrides().write() {
        m.insert(model.to_ascii_lowercase(), price);
        HAS_OVERRIDES.store(true, Ordering::Release);
    }
}

/// Remove every price added with [`set_price`] or [`load_prices_json`].
pub fn clear_price_overrides() {
    if let Ok(mut m) = overrides().write() {
        m.clear();
    }
}

/// Load prices from JSON and add them as overrides. Returns how many models
/// were loaded.
///
/// Two formats are accepted, and may be mixed in one file:
///
/// - LiteLLM's `model_prices_and_context_window.json`: per-token USD costs
///   in `input_cost_per_token` / `output_cost_per_token`, plus optional
///   `cache_read_input_token_cost` / `cache_creation_input_token_cost`.
///   Entries without both input and output costs (such as `sample_spec`,
///   image or embedding models priced differently) are skipped.
/// - A plain map: `{"my-model": {"input_usd_per_1m": 0.5, "output_usd_per_1m": 1.5}}`.
///
/// # Errors
///
/// [`DashboardError::SerializationError`] when the text is not JSON, and
/// [`DashboardError::LogParseError`] when it is JSON but not an object.
pub fn load_prices_json(json: &str) -> Result<usize, DashboardError> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let Some(obj) = v.as_object() else {
        return Err(DashboardError::LogParseError(
            "price file must be a JSON object keyed by model name".into(),
        ));
    };
    let num = |e: &serde_json::Value, k: &str| e.get(k).and_then(serde_json::Value::as_f64);
    let mut loaded = 0;
    for (model, e) in obj {
        let per_token = num(e, "input_cost_per_token").zip(num(e, "output_cost_per_token"));
        let per_million = num(e, "input_usd_per_1m").zip(num(e, "output_usd_per_1m"));
        let mut price = match (per_token, per_million) {
            (Some((i, o)), _) => ModelPrice::new(i * 1e6, o * 1e6),
            (None, Some((i, o))) => ModelPrice::new(i, o),
            (None, None) => continue,
        };
        if !(price.input_per_1m.is_finite()
            && price.output_per_1m.is_finite()
            && price.input_per_1m >= 0.0
            && price.output_per_1m >= 0.0)
        {
            continue;
        }
        price.cache_read_per_1m = num(e, "cache_read_input_token_cost").map(|c| c * 1e6);
        price.cache_write_per_1m = num(e, "cache_creation_input_token_cost").map(|c| c * 1e6);
        set_price(model, price);
        loaded += 1;
    }
    Ok(loaded)
}

/// Strip a trailing `-YYYYMMDD` or `-YYYY-MM-DD` date from a model id.
fn undated(model: &str) -> Option<&str> {
    let b = model.as_bytes();
    let is_digits = |s: &[u8]| s.iter().all(u8::is_ascii_digit);
    if b.len() > 9 && b[b.len() - 9] == b'-' && is_digits(&b[b.len() - 8..]) {
        return Some(&model[..model.len() - 9]);
    }
    if b.len() > 11
        && b[b.len() - 11] == b'-'
        && is_digits(&b[b.len() - 10..b.len() - 6])
        && b[b.len() - 6] == b'-'
        && is_digits(&b[b.len() - 5..b.len() - 3])
        && b[b.len() - 3] == b'-'
        && is_digits(&b[b.len() - 2..])
    {
        return Some(&model[..model.len() - 11]);
    }
    None
}

/// The built-in table keyed by lower-case model id, built on first use.
fn builtin() -> &'static HashMap<String, ModelPrice> {
    static T: OnceLock<HashMap<String, ModelPrice>> = OnceLock::new();
    T.get_or_init(|| {
        PRICING
            .iter()
            .map(|(name, i, o)| {
                let mut p = ModelPrice::new(*i, *o);
                p.cache_read_per_1m = CACHE_READ_RATES
                    .iter()
                    .find(|(m, _)| m == name)
                    .map(|(_, r)| *r);
                (name.to_ascii_lowercase(), p)
            })
            .collect()
    })
}

/// Set once any override exists, so the common path skips the lock.
static HAS_OVERRIDES: AtomicBool = AtomicBool::new(false);

fn find(model: &str) -> Option<ModelPrice> {
    let lower;
    let key: &str = if model.bytes().any(|b| b.is_ascii_uppercase()) {
        lower = model.to_ascii_lowercase();
        &lower
    } else {
        model
    };
    if HAS_OVERRIDES.load(Ordering::Acquire) {
        if let Ok(m) = overrides().read() {
            if let Some(p) = m.get(key) {
                return Some(*p);
            }
        }
    }
    builtin().get(key).copied()
}

/// The price for `model` if it is known: run-time overrides first, then the
/// built-in [`PRICING`] table, then the same id without a trailing date.
/// Case-insensitive. `None` for unknown models (no fallback).
pub fn price_of(model: &str) -> Option<ModelPrice> {
    let model = model.trim();
    find(model).or_else(|| undated(model).and_then(find))
}

/// Like [`lookup`], but `None` for a model that is not priced instead of
/// [`FALLBACK_PRICING`].
pub fn lookup_known(model: &str) -> Option<(f64, f64)> {
    price_of(model).map(|p| (p.input_per_1m, p.output_per_1m))
}

/// Look up pricing for `model`.
///
/// Same search as [`lookup_known`]. If the model is not found, returns
/// [`FALLBACK_PRICING`].
///
/// Returns `(input_usd_per_1m_tokens, output_usd_per_1m_tokens)`.
pub fn lookup(model: &str) -> (f64, f64) {
    lookup_known(model).unwrap_or(FALLBACK_PRICING)
}

/// Compute the total cost in USD for the given token counts.
///
/// Uses [`lookup`] internally; unknown models fall back to [`FALLBACK_PRICING`].
///
/// # Examples
///
/// ```
/// use llm_cost_dashboard::cost::pricing::compute_cost;
///
/// let cost = compute_cost("claude-sonnet-4-6", 1_000_000, 0);
/// assert!((cost - 3.00).abs() < 1e-9);
/// ```
pub fn compute_cost(model: &str, input_tokens: u64, output_tokens: u64) -> f64 {
    let (input_rate, output_rate) = lookup(model);
    (input_tokens as f64 * input_rate + output_tokens as f64 * output_rate) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_known_model() {
        let (i, o) = lookup("claude-sonnet-4-6");
        assert!((i - 3.00).abs() < f64::EPSILON);
        assert!((o - 15.00).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lookup_case_insensitive() {
        let (i1, _) = lookup("GPT-4O");
        let (i2, _) = lookup("gpt-4o");
        assert!((i1 - i2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lookup_unknown_uses_fallback() {
        let (i, o) = lookup("unknown-model-xyz");
        assert_eq!((i, o), FALLBACK_PRICING);
    }

    #[test]
    fn test_cache_read_rates() {
        assert!((price_of("gpt-4o").unwrap().cache_read_rate() - 1.25).abs() < 1e-12);
        assert!((price_of("claude-opus-5-5").unwrap().cache_read_rate() - 0.20).abs() < 1e-12);
        // Default: 10% of input.
        assert!((price_of("claude-sonnet-4-6").unwrap().cache_read_rate() - 0.30).abs() < 1e-12);
        for (m, _) in CACHE_READ_RATES {
            assert!(lookup_known(m).is_some(), "{m} has a cache rate but no price");
        }
    }

    #[test]
    fn test_dated_ids_fall_back_to_undated_entry() {
        assert_eq!(lookup_known("claude-sonnet-4-5-20250929"), Some((3.0, 15.0)));
        assert_eq!(lookup_known("gpt-4o-2024-08-06"), Some((2.5, 10.0)));
        assert_eq!(lookup_known("unknown-model-20250101"), None);
        assert_eq!(undated("gpt-4o"), None);
        assert_eq!(undated("x-2024-1-06"), None);
    }

    #[test]
    fn test_load_litellm_and_plain_formats() {
        let json = r#"{
            "sample_spec": {"input_cost_per_token": "varies"},
            "zz-test-litellm-model": {"input_cost_per_token": 3e-06, "output_cost_per_token": 1.5e-05,
                "cache_read_input_token_cost": 3e-07, "litellm_provider": "anthropic", "mode": "chat"},
            "zz-test-plain-model": {"input_usd_per_1m": 0.5, "output_usd_per_1m": 1.5},
            "zz-test-embedding": {"input_cost_per_token": 1e-07},
            "zz-test-negative": {"input_usd_per_1m": -1, "output_usd_per_1m": 1}
        }"#;
        assert_eq!(load_prices_json(json).unwrap(), 2);
        let p = price_of("ZZ-TEST-LITELLM-MODEL").unwrap();
        assert!((p.input_per_1m - 3.0).abs() < 1e-9);
        assert!((p.cache_read_rate() - 0.3).abs() < 1e-9);
        assert!((p.cache_write_rate() - 3.75).abs() < 1e-9);
        assert_eq!(lookup_known("zz-test-plain-model"), Some((0.5, 1.5)));
        assert_eq!(lookup_known("zz-test-embedding"), None);
        assert!(load_prices_json("[1,2]").is_err());
        assert!(load_prices_json("not json").is_err());
    }

    #[test]
    fn test_compute_cost_zero_tokens() {
        assert_eq!(compute_cost("claude-sonnet-4-6", 0, 0), 0.0);
    }

    #[test]
    fn test_compute_cost_one_million_input() {
        let cost = compute_cost("claude-sonnet-4-6", 1_000_000, 0);
        assert!((cost - 3.00).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_one_million_output() {
        let cost = compute_cost("claude-sonnet-4-6", 0, 1_000_000);
        assert!((cost - 15.00).abs() < 1e-9);
    }

    #[test]
    fn test_all_models_have_positive_rates() {
        for (model, i, o) in PRICING {
            assert!(*i > 0.0, "model {model} has zero input rate");
            assert!(*o > 0.0, "model {model} has zero output rate");
        }
    }

    // --- per-model exact pricing tests ---

    #[test]
    fn test_claude_opus_pricing() {
        let (i, o) = lookup("claude-opus-4-6");
        assert!((i - 5.00).abs() < 1e-9);
        assert!((o - 25.00).abs() < 1e-9);
    }

    #[test]
    fn test_claude_haiku_pricing() {
        let (i, o) = lookup("claude-haiku-4-5");
        assert!((i - 1.00).abs() < 1e-9);
        assert!((o - 5.00).abs() < 1e-9);
    }

    #[test]
    fn test_gpt4o_pricing() {
        let (i, o) = lookup("gpt-4o");
        assert!((i - 2.50).abs() < 1e-9);
        assert!((o - 10.00).abs() < 1e-9);
    }

    #[test]
    fn test_gpt4o_mini_pricing() {
        let (i, o) = lookup("gpt-4o-mini");
        assert!((i - 0.15).abs() < 1e-9);
        assert!((o - 0.60).abs() < 1e-9);
    }

    #[test]
    fn test_gpt4_turbo_pricing() {
        let (i, o) = lookup("gpt-4-turbo");
        assert!((i - 10.00).abs() < 1e-9);
        assert!((o - 30.00).abs() < 1e-9);
    }

    #[test]
    fn test_o1_pricing() {
        let (i, o) = lookup("o1");
        assert!((i - 15.00).abs() < 1e-9);
        assert!((o - 60.00).abs() < 1e-9);
    }

    #[test]
    fn test_o3_mini_pricing() {
        let (i, o) = lookup("o3-mini");
        assert!((i - 1.10).abs() < 1e-9);
        assert!((o - 4.40).abs() < 1e-9);
    }

    #[test]
    fn test_gemini_15_pro_pricing() {
        let (i, o) = lookup("gemini-1.5-pro");
        assert!((i - 1.25).abs() < 1e-9);
        assert!((o - 5.00).abs() < 1e-9);
    }

    #[test]
    fn test_gemini_15_flash_pricing() {
        let (i, o) = lookup("gemini-1.5-flash");
        assert!((i - 0.075).abs() < 1e-9);
        assert!((o - 0.30).abs() < 1e-9);
    }

    #[test]
    fn test_gemini_20_flash_pricing() {
        let (i, o) = lookup("gemini-2.0-flash");
        assert!((i - 0.10).abs() < 1e-9);
        assert!((o - 0.40).abs() < 1e-9);
    }

    // --- edge cases ---

    #[test]
    fn test_compute_cost_max_u64_does_not_panic() {
        // u64::MAX tokens should produce a very large but finite cost, not panic.
        let cost = compute_cost("gpt-4o-mini", u64::MAX, 0);
        assert!(cost.is_finite() || cost.is_infinite()); // either is acceptable, just no panic
    }

    #[test]
    fn test_compute_cost_fractional_result() {
        // 1 token at $0.15/1M = $0.00000015
        let cost = compute_cost("gpt-4o-mini", 1, 0);
        assert!((cost - 0.15 / 1_000_000.0).abs() < 1e-15);
    }

    #[test]
    fn test_all_pricing_table_entries_lookable() {
        for (model, expected_i, expected_o) in PRICING {
            let (i, o) = lookup(model);
            assert!((i - expected_i).abs() < 1e-9, "input mismatch for {model}");
            assert!((o - expected_o).abs() < 1e-9, "output mismatch for {model}");
        }
    }

    // ── Property tests (proptest) ─────────────────────────────────────────────

    proptest::proptest! {
        /// Cost is always non-negative for any non-negative token counts.
        #[test]
        fn prop_cost_non_negative(
            input in 0u64..10_000_000u64,
            output in 0u64..10_000_000u64,
            idx in 0usize..PRICING.len(),
        ) {
            let cost = compute_cost(PRICING[idx].0, input, output);
            proptest::prop_assert!(cost >= 0.0, "cost was negative: {cost}");
        }

        /// Cost scales linearly with input token count: doubling tokens doubles cost.
        #[test]
        fn prop_cost_linear_with_input(
            tokens in 1u64..1_000_000u64,
            idx in 0usize..PRICING.len(),
        ) {
            let m = PRICING[idx].0;
            let c1 = compute_cost(m, tokens, 0);
            let c2 = compute_cost(m, tokens * 2, 0);
            proptest::prop_assert!((c2 / c1 - 2.0).abs() < 1e-9);
        }

        /// Cost scales linearly with output token count.
        #[test]
        fn prop_cost_linear_with_output(
            tokens in 1u64..1_000_000u64,
            idx in 0usize..PRICING.len(),
        ) {
            let m = PRICING[idx].0;
            let c1 = compute_cost(m, 0, tokens);
            let c2 = compute_cost(m, 0, tokens * 2);
            proptest::prop_assert!((c2 / c1 - 2.0).abs() < 1e-9);
        }

        /// Per-model rates are consistent: looking up the same model twice
        /// gives identical rates.
        #[test]
        fn prop_lookup_deterministic(idx in 0usize..PRICING.len()) {
            let m = PRICING[idx].0;
            let (i1, o1) = lookup(m);
            let (i2, o2) = lookup(m);
            proptest::prop_assert_eq!(i1, i2);
            proptest::prop_assert_eq!(o1, o2);
        }
    }
}
