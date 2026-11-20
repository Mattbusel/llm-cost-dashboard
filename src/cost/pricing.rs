//! Per-model token prices (USD per 1 million tokens) and request costing.
//!
//! All prices are stored as `f64` values representing US dollars per 1,000,000
//! tokens.  Use [`compute_cost`] for convenience, [`cost_usage`] for a whole
//! response (prompt-cache reads and writes, Batch API, long-context tiers),
//! [`lookup_known`] when you need to know whether a model is priced at all,
//! or [`lookup`] for the raw rates with the fallback applied.
//!
//! # Where prices come from
//!
//! A model id is resolved in this order (see [`price_source`]):
//!
//! 1. A price set at run time with [`set_price`] or [`load_prices_json`]
//!    (LiteLLM's `model_prices_and_context_window.json` format, or a simple
//!    `{"model": {"input_usd_per_1m": .., "output_usd_per_1m": ..}}` map).
//! 2. The [`llm_prices`] crate: dated, source-cited Anthropic, OpenAI, Google
//!    Gemini and DeepSeek rows with cache-read, cache-write (5-minute and
//!    1-hour), Batch API and long-context prices. Updating that dependency
//!    updates these prices; [`LLM_PRICES_TABLE_DATE`] says how old they are.
//!    It also resolves provider prefixes (`anthropic/`, `models/`), Bedrock
//!    ids (`us.anthropic.claude-...-v1:0`) and Vertex `@date` ids.
//! 3. The built-in [`PRICING`] table, which still covers the models
//!    `llm-prices` does not list (Mistral, Llama hosts, Grok, Cohere,
//!    Perplexity, Bedrock Nova, Qwen and older OpenAI and Gemini ids).
//! 4. [`FALLBACK_PRICING`] when nothing matches.
//!
//! Overrides apply process-wide, to every [`crate::CostRecord`] created
//! afterwards. Dated model ids (`claude-sonnet-4-5-20250929`,
//! `gpt-4o-2024-08-06`) fall back to the undated entry when the dated one is
//! not listed.
//!
//! ```
//! use llm_cost_dashboard::cost::pricing::{cost_usage, price_source, PriceSource, Usage};
//!
//! // A Claude Code turn: 2 fresh input tokens, 30,000 read from the prompt
//! // cache, 1,000 written to the 1-hour cache, 360 output tokens.
//! let u = Usage { input: 2, cache_read: 30_000, cache_write_1h: 1_000, output: 360, ..Usage::default() };
//! let c = cost_usage("claude-sonnet-4-5-20250929", &u);
//! assert_eq!(c.source, PriceSource::LlmPrices);
//! // 2 x $3 + 30,000 x $0.30 + 1,000 x $6 + 360 x $15, per million tokens.
//! assert!((c.total - 0.020_406).abs() < 1e-12);
//!
//! // The same request through the Batch API costs half.
//! let b = cost_usage("claude-sonnet-4-5-20250929", &Usage { batch: true, ..u });
//! assert!(b.batch_priced);
//! assert!((b.total - c.total / 2.0).abs() < 1e-12);
//!
//! assert_eq!(price_source("mistral-nemo"), PriceSource::BuiltIn);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

use crate::error::DashboardError;

/// Token counts of one request, the input to [`cost_usage`] and
/// [`crate::CostRecord::from_usage`]. Re-exported from `llm-prices`; see
/// its docs for how Anthropic, OpenAI and Gemini usage objects map onto it.
///
/// This re-export (and [`llm_prices_row`]) makes `llm-prices` 1.x part of
/// this crate's public API: moving to an `llm-prices` release that changes
/// `Usage` or `Price` incompatibly (2.0 plans new `Usage` fields) will be a
/// major release of this crate. Build a `Usage` with `..Usage::default()`
/// for the fields you do not set.
pub use llm_prices::Usage;

/// Date the bundled `llm-prices` table was generated (ISO 8601). Prices from
/// that table are only as current as this date; update the `llm-prices`
/// dependency to refresh them.
pub const LLM_PRICES_TABLE_DATE: &str = llm_prices::GENERATED;

/// Statically known per-model pricing entries.
///
/// Each tuple is `(model_id, input_usd_per_1m, output_usd_per_1m)`.
/// Model IDs are matched case-insensitively by [`lookup`].
///
/// Claude rows checked against <https://platform.claude.com/docs/en/about-claude/pricing>
/// on 2026-10-17 (see `claude_prices_match_anthropic_pricing_page`).
///
/// Last updated: 2026-10-17
pub const PRICING: &[(&str, f64, f64)] = &[
    // ── Anthropic / Claude 5 family (Anthropic list prices, 2026-10-17) ─────
    ("claude-fable-5-1", 10.00, 50.00),
    ("claude-fable-5", 10.00, 50.00),
    ("claude-opus-5-5", 4.00, 20.00),
    ("claude-opus-5", 5.00, 25.00),
    ("claude-sonnet-5-5", 2.00, 10.00),
    ("claude-sonnet-5", 2.00, 10.00),
    // Prompts up to 100,000 tokens; longer prompts: see LONG_PROMPT_RATES.
    ("claude-haiku-5-5", 0.10, 0.50),
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
    ("llama-3.3-70b-versatile", 0.59, 0.79), // Groq
    ("llama-3.1-70b-versatile", 0.59, 0.79), // Groq
    ("llama-3.1-8b-instant", 0.05, 0.08),    // Groq
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

/// Fallback pricing used when the model is not found in `llm-prices`, in
/// [`PRICING`] or in the run-time overrides.
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
    /// Prompt-cache writes (1-hour cache); `None` means 200% of the input rate.
    pub cache_write_1h_per_1m: Option<f64>,
    /// Batch API input rate; `None` means the model lists no batch price, and
    /// batch requests are billed at the standard rates.
    pub batch_input_per_1m: Option<f64>,
    /// Batch API output rate; see [`ModelPrice::batch_input_per_1m`].
    pub batch_output_per_1m: Option<f64>,
}

impl ModelPrice {
    /// Input and output rates in USD per million tokens, default cache rates.
    pub fn new(input_per_1m: f64, output_per_1m: f64) -> Self {
        Self {
            input_per_1m,
            output_per_1m,
            cache_read_per_1m: None,
            cache_write_per_1m: None,
            cache_write_1h_per_1m: None,
            batch_input_per_1m: None,
            batch_output_per_1m: None,
        }
    }

    /// Set Batch API input and output rates (USD per million tokens).
    pub fn with_batch_rates(mut self, input_per_1m: f64, output_per_1m: f64) -> Self {
        self.batch_input_per_1m = Some(input_per_1m);
        self.batch_output_per_1m = Some(output_per_1m);
        self
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

    /// 1-hour cache-write rate, defaulting to 200% of input (Anthropic's rate).
    pub fn cache_write_1h_rate(&self) -> f64 {
        self.cache_write_1h_per_1m
            .unwrap_or(self.input_per_1m * 2.0)
    }

    /// The rates of one `llm-prices` row, or of its long-context tier when
    /// `long` is set and the row has one. `None` when the row lists no input
    /// price. A row without an output price (embedding models) gets 0.
    fn from_table(p: &llm_prices::Price, long: bool) -> Option<Self> {
        let (input, output, read, writes, bi, bo) = match p.long_context {
            Some(lc) if long => (
                lc.input,
                lc.output,
                lc.cache_read,
                lc.cache_write,
                lc.batch_input,
                lc.batch_output,
            ),
            _ => (
                p.input,
                p.output,
                p.cache_read,
                p.cache_write,
                p.batch_input,
                p.batch_output,
            ),
        };
        let input = input?;
        let write = |ttl: &str| writes.iter().find(|(k, _)| *k == ttl).map(|(_, v)| *v);
        let batch = bi.zip(bo);
        Some(Self {
            input_per_1m: input,
            output_per_1m: output.unwrap_or(0.0),
            // No listed cached-input price: cache reads bill as input.
            cache_read_per_1m: Some(read.unwrap_or(input)),
            // No listed 5m / 1h write price (OpenAI, Gemini, DeepSeek):
            // writes bill as ordinary input.
            cache_write_per_1m: Some(write("5m").unwrap_or(input)),
            cache_write_1h_per_1m: Some(write("1h").unwrap_or(input)),
            batch_input_per_1m: batch.map(|(i, _)| i),
            batch_output_per_1m: batch.map(|(_, o)| o),
        })
    }
}

/// Where the price used for a model came from. See the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PriceSource {
    /// Set at run time with [`set_price`] or [`load_prices_json`].
    Override,
    /// The `llm-prices` crate table.
    LlmPrices,
    /// The built-in [`PRICING`] table.
    BuiltIn,
    /// Not found anywhere: [`FALLBACK_PRICING`].
    Fallback,
}

/// Cost in USD of one request, split by token kind. Returned by [`cost_usage`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct UsageCost {
    /// Uncached input tokens.
    pub input: f64,
    /// Output tokens.
    pub output: f64,
    /// Tokens read from the prompt cache.
    pub cache_read: f64,
    /// Tokens written to the prompt cache, 5-minute and 1-hour together.
    pub cache_write: f64,
    /// Sum of the other cost fields.
    pub total: f64,
    /// The prompt was over the model's long-context threshold, so the whole
    /// request was billed at the long-context prices.
    pub long_context: bool,
    /// Batch API prices were applied. `false` for a batch request on a model
    /// that lists no batch price (it is billed at the standard prices).
    pub batch_priced: bool,
    /// Where the prices came from.
    pub source: PriceSource,
}

/// The price entry a model id resolves to.
enum Found {
    Custom(ModelPrice, PriceSource),
    Table(&'static llm_prices::Price),
}

fn find_override(model: &str) -> Option<ModelPrice> {
    if !HAS_OVERRIDES.load(Ordering::Acquire) {
        return None;
    }
    let key = model.to_ascii_lowercase();
    overrides().read().ok().and_then(|m| m.get(&key).copied())
}

fn find_table(model: &str) -> Option<&'static llm_prices::Price> {
    llm_prices::lookup(model).filter(|p| p.input.is_some())
}

fn find_builtin(model: &str) -> Option<ModelPrice> {
    let table = builtin();
    if model.bytes().any(|b| b.is_ascii_uppercase()) {
        table.get(&model.to_ascii_lowercase()).copied()
    } else {
        table.get(model).copied()
    }
}

/// Resolve a model id: overrides (exact, then undated), then `llm-prices`,
/// then the built-in table (each exact, then undated).
fn resolve(model: &str) -> Option<Found> {
    let model = model.trim();
    let undated_id = undated(model);
    if let Some(p) = find_override(model).or_else(|| undated_id.and_then(find_override)) {
        return Some(Found::Custom(p, PriceSource::Override));
    }
    let table_or_builtin = |m: &str| {
        find_table(m)
            .map(Found::Table)
            .or_else(|| find_builtin(m).map(|p| Found::Custom(p, PriceSource::BuiltIn)))
    };
    table_or_builtin(model).or_else(|| undated_id.and_then(table_or_builtin))
}

/// Where the price for `model` comes from.
pub fn price_source(model: &str) -> PriceSource {
    match resolve(model) {
        Some(Found::Table(_)) => PriceSource::LlmPrices,
        Some(Found::Custom(_, s)) => s,
        None => PriceSource::Fallback,
    }
}

/// The `llm-prices` row `model` is priced from, if it is (a run-time
/// override takes precedence and makes this `None`). The row carries the
/// provider's pricing page URL and the date it was checked.
pub fn llm_prices_row(model: &str) -> Option<&'static llm_prices::Price> {
    match resolve(model) {
        Some(Found::Table(p)) => Some(p),
        _ => None,
    }
}

/// Cost in USD of one request from its token counts.
///
/// Rules (the same as [`llm_prices::Price::cost_usage`], which prices every
/// model the `llm-prices` table lists):
///
/// - `usage.input` is the **uncached** input only. Cache reads bill at the
///   model's cache-read price, cache writes at its 5-minute or 1-hour price.
///   OpenAI, Gemini and DeepSeek list no separate 5-minute / 1-hour write
///   price; writes reported on those models bill as ordinary input.
/// - **Long context**: when the whole prompt (input plus cache reads and
///   writes) is over the model's threshold, every token bills at the
///   long-context prices.
/// - **Batch**: with `usage.batch`, input and output bill at the Batch API
///   prices and cache tokens get the same discount. A model with no listed
///   batch price is billed at standard prices and
///   [`UsageCost::batch_priced`] is `false`.
/// - Unknown models use [`FALLBACK_PRICING`] with the default cache rates.
///
/// ```
/// use llm_cost_dashboard::cost::pricing::{cost_usage, Usage};
///
/// // OpenAI reports prompt_tokens including cached tokens; pass the split.
/// let u = Usage { input: 1_000, cache_read: 9_000, output: 500, ..Usage::default() };
/// let c = cost_usage("gpt-4o", &u);
/// // 1,000 x $2.50 + 9,000 x $1.25 + 500 x $10, per million tokens.
/// assert!((c.total - 0.018_75).abs() < 1e-12);
/// ```
pub fn cost_usage(model: &str, usage: &Usage) -> UsageCost {
    match resolve(model) {
        Some(Found::Table(p)) => table_cost(p, usage).unwrap_or_else(|| {
            let long = p
                .long_context
                .is_some_and(|lc| usage.prompt_tokens() > lc.threshold_tokens);
            let price = ModelPrice::from_table(p, long)
                .or_else(|| ModelPrice::from_table(p, false))
                .unwrap_or(ModelPrice::new(FALLBACK_PRICING.0, FALLBACK_PRICING.1));
            UsageCost {
                long_context: long,
                ..rates_cost(price, usage, PriceSource::LlmPrices)
            }
        }),
        Some(Found::Custom(p, PriceSource::BuiltIn)) => {
            let price = builtin_for_prompt(model, p, usage.prompt_tokens());
            rates_cost(price, usage, PriceSource::BuiltIn)
        }
        Some(Found::Custom(p, source)) => rates_cost(p, usage, source),
        None => rates_cost(
            ModelPrice::new(FALLBACK_PRICING.0, FALLBACK_PRICING.1),
            usage,
            PriceSource::Fallback,
        ),
    }
}

/// Cost in USD of `requests` requests whose token counts add up to `usage`,
/// such as one bucket of a provider's usage export.
///
/// Same as [`cost_usage`], except that the long-context tier is chosen from
/// the **average** prompt per request, not the bucket's total: 1,000
/// requests of 1,000 tokens each are 1,000,000 prompt tokens, but none of
/// them is a long-context request. Requests in one bucket are assumed to
/// fall on the same side of the threshold. `requests` of 0 or 1 is
/// [`cost_usage`].
///
/// ```
/// use llm_cost_dashboard::cost::pricing::{cost_usage, cost_usage_bucket, Usage};
///
/// let bucket = Usage { input: 1_000_000, output: 10_000, ..Usage::default() };
/// // As one request, 1M tokens is over GPT-5.4's 272,000-token threshold.
/// assert!(cost_usage("gpt-5.4", &bucket).long_context);
/// // As 1,000 requests of 1,000 tokens, it is not.
/// let c = cost_usage_bucket("gpt-5.4", &bucket, 1_000);
/// assert!(!c.long_context);
/// // 1M x $2.50 + 10,000 x $15, per million tokens.
/// assert!((c.total - 2.65).abs() < 1e-9);
/// ```
pub fn cost_usage_bucket(model: &str, usage: &Usage, requests: u64) -> UsageCost {
    if requests <= 1 {
        return cost_usage(model, usage);
    }
    let avg_prompt = usage.prompt_tokens().div_ceil(requests);
    match resolve(model) {
        Some(Found::Table(p)) => {
            let long = p
                .long_context
                .is_some_and(|lc| avg_prompt > lc.threshold_tokens);
            let full = cost_usage(model, usage);
            if full.long_context == long {
                return full;
            }
            match ModelPrice::from_table(p, long) {
                Some(price) => UsageCost {
                    long_context: long,
                    ..rates_cost(price, usage, PriceSource::LlmPrices)
                },
                None => full,
            }
        }
        Some(Found::Custom(p, PriceSource::BuiltIn)) => rates_cost(
            builtin_for_prompt(model, p, avg_prompt),
            usage,
            PriceSource::BuiltIn,
        ),
        _ => cost_usage(model, usage),
    }
}

/// Price a request with `llm-prices`. Cache writes on a row that lists no
/// price for their lifetime are billed as input, and a batch request on a
/// row with no batch price is billed at standard prices. `None` when the row
/// still cannot price it (no output price).
fn table_cost(p: &llm_prices::Price, usage: &Usage) -> Option<UsageCost> {
    let mut u = *usage;
    if u.cache_write_5m > 0 && p.cache_write("5m").is_none() {
        u.input = u.input.saturating_add(u.cache_write_5m);
        u.cache_write_5m = 0;
    }
    if u.cache_write_1h > 0 && p.cache_write("1h").is_none() {
        u.input = u.input.saturating_add(u.cache_write_1h);
        u.cache_write_1h = 0;
    }
    let (c, batch_priced) = match p.cost_usage(&u) {
        Some(c) => (c, u.batch),
        None if u.batch => (p.cost_usage(&Usage { batch: false, ..u })?, false),
        None => return None,
    };
    Some(UsageCost {
        input: c.input,
        output: c.output,
        cache_read: c.cache_read,
        cache_write: c.cache_write,
        total: c.total,
        long_context: c.long_context,
        batch_priced,
        source: PriceSource::LlmPrices,
    })
}

/// Price a request from plain rates (overrides, built-in rows, fallback).
fn rates_cost(price: ModelPrice, usage: &Usage, source: PriceSource) -> UsageCost {
    let batch = if usage.batch {
        price.batch_input_per_1m.zip(price.batch_output_per_1m)
    } else {
        None
    };
    let (in_rate, out_rate, discount) = match batch {
        Some((bi, bo)) => {
            let d = if price.input_per_1m > 0.0 {
                bi / price.input_per_1m
            } else {
                1.0
            };
            (bi, bo, d)
        }
        None => (price.input_per_1m, price.output_per_1m, 1.0),
    };
    const M: f64 = 1_000_000.0;
    let input = usage.input as f64 * in_rate / M;
    let output = usage.output as f64 * out_rate / M;
    let cache_read = usage.cache_read as f64 * price.cache_read_rate() * discount / M;
    let cache_write = (usage.cache_write_5m as f64 * price.cache_write_rate()
        + usage.cache_write_1h as f64 * price.cache_write_1h_rate())
        * discount
        / M;
    UsageCost {
        input,
        output,
        cache_read,
        cache_write,
        total: input + output + cache_read + cache_write,
        long_context: false,
        batch_priced: batch.is_some(),
        source,
    }
}

/// A built-in row, switched to its [`LONG_PROMPT_RATES`] entry when the
/// prompt is over the threshold.
fn builtin_for_prompt(model: &str, base: ModelPrice, prompt_tokens: u64) -> ModelPrice {
    let m = model.trim().to_ascii_lowercase();
    let key = undated(&m).unwrap_or(&m);
    match LONG_PROMPT_RATES.iter().find(|(id, ..)| *id == key) {
        Some(&(_, threshold, i, o)) if prompt_tokens > threshold => ModelPrice::new(i, o),
        _ => base,
    }
}

/// Published cached-input (prompt-cache read) rates, USD per million tokens,
/// for models whose rate is not the default 10% of input. OpenAI bills
/// cached input at 50% (GPT-4o, o-series), 25% (GPT-4.1) or 10% (GPT-5).
pub const CACHE_READ_RATES: &[(&str, f64)] = &[
    ("claude-fable-5-1", 0.25),
    ("claude-opus-5-5", 0.20),
    ("claude-sonnet-5-5", 0.10),
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

/// Built-in models whose whole request is billed at higher rates once its
/// prompt is longer than a threshold. Models priced from `llm-prices` use
/// that table's long-context tiers instead (Claude Haiku 5.5 among them), so
/// this list only applies to [`PRICING`] rows: `(model, threshold_tokens, input_usd_per_1m,
/// output_usd_per_1m)`. The prompt length counts every input token,
/// including cache reads and writes, and cache rates scale from the higher
/// input rate. Claude Haiku 5.5 pays $0.50 / $2.50 for prompts over 100,000
/// tokens (Anthropic pricing page, long context pricing).
pub const LONG_PROMPT_RATES: &[(&str, u64, f64, f64)] =
    &[("claude-haiku-5-5", 100_000, 0.50, 2.50)];

/// The price for one request of `model` whose prompt (input plus cache
/// read and write tokens) is `prompt_tokens` long. Same as [`price_of`],
/// except that a model with a long-context tier (in `llm-prices`, or in
/// [`LONG_PROMPT_RATES`] for built-in rows) gets its higher rates when the
/// prompt is over the threshold. A price set with [`set_price`] or
/// [`load_prices_json`] is returned unchanged.
///
/// ```
/// use llm_cost_dashboard::cost::pricing::price_for_prompt;
///
/// let short = price_for_prompt("claude-haiku-5-5", 100_000).unwrap();
/// let long = price_for_prompt("claude-haiku-5-5", 100_001).unwrap();
/// assert!((short.input_per_1m - 0.10).abs() < 1e-12);
/// assert!((long.input_per_1m - 0.50).abs() < 1e-12);
/// ```
pub fn price_for_prompt(model: &str, prompt_tokens: u64) -> Option<ModelPrice> {
    match resolve(model)? {
        Found::Custom(p, PriceSource::BuiltIn) => Some(builtin_for_prompt(model, p, prompt_tokens)),
        Found::Custom(p, _) => Some(p),
        Found::Table(p) => {
            let long = p
                .long_context
                .is_some_and(|lc| prompt_tokens > lc.threshold_tokens);
            ModelPrice::from_table(p, long).or_else(|| ModelPrice::from_table(p, false))
        }
    }
}

fn overrides() -> &'static RwLock<HashMap<String, ModelPrice>> {
    static O: OnceLock<RwLock<HashMap<String, ModelPrice>>> = OnceLock::new();
    O.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Add or replace one model's price for this process.
///
/// Overrides come first in the lookup, ahead of `llm-prices` and the
/// built-in table. An override for an undated id also covers its dated
/// ids that have no override of their own: `set_price("gpt-4o", ..)` prices
/// `gpt-4o-2024-08-06` too.
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
///   `cache_read_input_token_cost` / `cache_creation_input_token_cost`,
///   `cache_creation_input_token_cost_above_1hr` (1-hour cache writes) and
///   `input_cost_per_token_batches` / `output_cost_per_token_batches`
///   (Batch API).
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
        price.cache_write_1h_per_1m =
            num(e, "cache_creation_input_token_cost_above_1hr").map(|c| c * 1e6);
        if let Some((bi, bo)) =
            num(e, "input_cost_per_token_batches").zip(num(e, "output_cost_per_token_batches"))
        {
            price = price.with_batch_rates(bi * 1e6, bo * 1e6);
        }
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

/// The price for `model` if it is known: run-time overrides first, then the
/// `llm-prices` table, then the built-in [`PRICING`] table, each also tried
/// with the id's trailing date removed. Case-insensitive. `None` for unknown
/// models (no fallback). Models priced from `llm-prices` come with their
/// cache-read, 5-minute and 1-hour cache-write and batch rates filled in.
pub fn price_of(model: &str) -> Option<ModelPrice> {
    match resolve(model)? {
        Found::Custom(p, _) => Some(p),
        Found::Table(p) => ModelPrice::from_table(p, false),
    }
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
        assert!((price_of("claude-sonnet-5-5").unwrap().cache_read_rate() - 0.10).abs() < 1e-12);
        // Default: 10% of input.
        assert!((price_of("claude-sonnet-4-6").unwrap().cache_read_rate() - 0.30).abs() < 1e-12);
        for (m, _) in CACHE_READ_RATES {
            assert!(
                lookup_known(m).is_some(),
                "{m} has a cache rate but no price"
            );
        }
    }

    /// Every Claude row in Anthropic's pricing table, USD per million tokens:
    /// (model id, base input, 5m cache write, 1h cache write, cache hit, output).
    /// Source: https://platform.claude.com/docs/en/about-claude/pricing
    /// ("Model pricing" table), checked 2026-10-17. If Anthropic changes a
    /// price, update this table from the page and the code to match.
    /// Not covered: Claude Mythos 5.1 / 5 (limited availability, no public
    /// model id on the models overview page) and the retired Claude 3 rows
    /// that the page no longer lists.
    const ANTHROPIC_PAGE_2026_10_17: &[(&str, f64, f64, f64, f64, f64)] = &[
        ("claude-fable-5-1", 10.00, 12.50, 20.00, 0.25, 50.00),
        ("claude-fable-5", 10.00, 12.50, 20.00, 1.00, 50.00),
        ("claude-opus-5-5", 4.00, 5.00, 8.00, 0.20, 20.00),
        ("claude-opus-5", 5.00, 6.25, 10.00, 0.50, 25.00),
        ("claude-opus-4-8", 5.00, 6.25, 10.00, 0.50, 25.00),
        ("claude-opus-4-7", 5.00, 6.25, 10.00, 0.50, 25.00),
        ("claude-opus-4-6", 5.00, 6.25, 10.00, 0.50, 25.00),
        ("claude-opus-4-5", 5.00, 6.25, 10.00, 0.50, 25.00),
        ("claude-opus-4-1", 15.00, 18.75, 30.00, 1.50, 75.00),
        ("claude-opus-4", 15.00, 18.75, 30.00, 1.50, 75.00),
        ("claude-sonnet-5-5", 2.00, 2.50, 4.00, 0.10, 10.00),
        ("claude-sonnet-5", 2.00, 2.50, 4.00, 0.20, 10.00),
        ("claude-sonnet-4-6", 3.00, 3.75, 6.00, 0.30, 15.00),
        ("claude-sonnet-4-5", 3.00, 3.75, 6.00, 0.30, 15.00),
        ("claude-sonnet-4", 3.00, 3.75, 6.00, 0.30, 15.00),
        // Haiku 5.5, prompts up to 100,000 tokens.
        ("claude-haiku-5-5", 0.10, 0.125, 0.20, 0.01, 0.50),
        ("claude-haiku-4-5", 1.00, 1.25, 2.00, 0.10, 5.00),
        ("claude-3-5-haiku-20241022", 0.80, 1.00, 1.60, 0.08, 4.00),
    ];

    #[test]
    fn claude_prices_match_anthropic_pricing_page() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        for &(id, input, w5m, w1h, read, output) in ANTHROPIC_PAGE_2026_10_17 {
            let p = price_of(id).unwrap_or_else(|| panic!("{id} is missing from PRICING"));
            assert!(
                close(p.input_per_1m, input),
                "{id} input {} != {input}",
                p.input_per_1m
            );
            assert!(
                close(p.output_per_1m, output),
                "{id} output {} != {output}",
                p.output_per_1m
            );
            assert!(
                close(p.cache_write_rate(), w5m),
                "{id} 5m write {} != {w5m}",
                p.cache_write_rate()
            );
            assert!(
                close(p.cache_write_1h_rate(), w1h),
                "{id} 1h write {} != {w1h}",
                p.cache_write_1h_rate()
            );
            assert!(
                close(p.cache_read_rate(), read),
                "{id} cache read {} != {read}",
                p.cache_read_rate()
            );
        }
        // Haiku 5.5, prompts over 100,000 tokens (same page, same date):
        // $0.50 input, $0.625 5m write, $1 1h write, $0.05 cache hit, $2.50 output.
        let long = price_for_prompt("claude-haiku-5-5", 100_001).unwrap();
        assert!(close(long.input_per_1m, 0.50));
        assert!(close(long.cache_write_rate(), 0.625));
        assert!(close(long.cache_write_1h_rate(), 1.00));
        assert!(close(long.cache_read_rate(), 0.05));
        assert!(close(long.output_per_1m, 2.50));
        let short = price_for_prompt("claude-haiku-5-5", 100_000).unwrap();
        assert!(close(short.input_per_1m, 0.10));
        // Other models are not tiered.
        let s = price_for_prompt("claude-sonnet-5-5", 900_000).unwrap();
        assert!(close(s.input_per_1m, 2.00) && close(s.cache_read_rate(), 0.10));
    }

    #[test]
    fn test_dated_ids_fall_back_to_undated_entry() {
        assert_eq!(
            lookup_known("claude-sonnet-4-5-20250929"),
            Some((3.0, 15.0))
        );
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
    fn test_overrides_win_over_llm_prices_even_undated() {
        // llm-prices resolves this id (prefix stripped)...
        assert_eq!(
            price_source("eu.anthropic.claude-opus-4-20250514"),
            PriceSource::LlmPrices
        );
        // ...but an override of its undated form wins, as before 1.4.0.
        // (An id no other test prices, since overrides are process-wide.)
        set_price("eu.anthropic.claude-opus-4", ModelPrice::new(1.0, 2.0));
        assert_eq!(
            price_source("eu.anthropic.claude-opus-4-20250514"),
            PriceSource::Override
        );
        assert_eq!(
            lookup_known("eu.anthropic.claude-opus-4-20250514"),
            Some((1.0, 2.0))
        );
        assert_eq!(
            price_source("claude-opus-4-20250514"),
            PriceSource::LlmPrices
        );
    }

    #[test]
    fn test_llm_prices_ids_and_rates() {
        // Bedrock ids and provider prefixes resolve through llm-prices.
        assert_eq!(
            lookup_known("us.anthropic.claude-sonnet-4-20250514-v1:0"),
            Some((3.0, 15.0))
        );
        assert_eq!(lookup_known("models/gemini-2.5-pro"), Some((1.25, 10.0)));
        let p = price_of("claude-opus-5-5").unwrap();
        assert_eq!(p.cache_write_per_1m, Some(5.0));
        assert_eq!(p.cache_write_1h_per_1m, Some(8.0));
        assert_eq!(
            (p.batch_input_per_1m, p.batch_output_per_1m),
            (Some(2.0), Some(10.0))
        );
        // Gemini 2.5 Pro long-context tier (over 200,000 prompt tokens).
        assert_eq!(
            price_for_prompt("gemini-2.5-pro", 200_001).map(|p| p.input_per_1m),
            Some(2.5)
        );
        assert_eq!(
            price_for_prompt("gemini-2.5-pro", 200_000).map(|p| p.input_per_1m),
            Some(1.25)
        );
        // Embedding rows have no output price: priced as input only.
        let e = cost_usage(
            "text-embedding-3-small",
            &Usage {
                input: 1_000_000,
                ..Usage::default()
            },
        );
        assert!((e.total - 0.02).abs() < 1e-12);
        assert_eq!(e.source, PriceSource::LlmPrices);
    }

    #[test]
    fn test_litellm_batch_and_1h_fields() {
        let json = r#"{"zz-test-batch-model": {"input_cost_per_token": 2e-06, "output_cost_per_token": 8e-06,
            "input_cost_per_token_batches": 1e-06, "output_cost_per_token_batches": 4e-06,
            "cache_creation_input_token_cost_above_1hr": 5e-06}}"#;
        assert_eq!(load_prices_json(json).unwrap(), 1);
        let u = Usage {
            input: 1_000_000,
            output: 1_000_000,
            cache_write_1h: 1_000_000,
            batch: true,
            ..Usage::default()
        };
        let c = cost_usage("zz-test-batch-model", &u);
        assert!(c.batch_priced);
        // $1 + $4, and the 1h write ($5) at the batch discount (50%).
        assert!((c.total - 7.5).abs() < 1e-9, "{}", c.total);
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
