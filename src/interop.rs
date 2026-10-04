//! Price responses from the Rust LLM clients you already use.
//!
//! Each adapter is behind its own optional feature, off by default:
//!
//! | Feature | Adds |
//! |---|---|
//! | `async-openai` | [`async_openai`]: cost of an `async_openai` chat completion response or `CompletionUsage` |
//!
//! Any other client works through JSON: serialize its response object and
//! feed it to [`crate::ingest::Ingester::ingest_line`], which understands
//! OpenAI-style (`usage.prompt_tokens`) and Anthropic-style
//! (`usage.input_tokens`, cache fields) shapes.

/// Adapter for [async-openai](https://crates.io/crates/async-openai)
/// (types only; no HTTP client is pulled in).
///
/// ```
/// use async_openai::types::chat::CompletionUsage;
/// use llm_cost_dashboard::interop::async_openai::record_from_usage;
///
/// let usage = CompletionUsage {
///     prompt_tokens: 1_000_000,
///     completion_tokens: 0,
///     total_tokens: 1_000_000,
///     prompt_tokens_details: None,
///     completion_tokens_details: None,
/// };
/// let rec = record_from_usage("gpt-4o", &usage);
/// assert!((rec.total_cost_usd - 2.50).abs() < 1e-9);
/// ```
#[cfg(feature = "async-openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "async-openai")))]
pub mod async_openai {
    use ::async_openai::types::chat::{CompletionUsage, CreateChatCompletionResponse};

    use crate::cost::CostRecord;

    /// A priced record for one chat completion's usage. Cached prompt tokens
    /// (`prompt_tokens_details.cached_tokens`, which OpenAI also counts in
    /// `prompt_tokens`) are priced at the model's cached-input rate.
    pub fn record_from_usage(model: &str, usage: &CompletionUsage) -> CostRecord {
        let cached = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0)
            .min(usage.prompt_tokens);
        let rec = CostRecord::new(
            model,
            "openai",
            u64::from(usage.prompt_tokens - cached),
            u64::from(usage.completion_tokens),
            0,
        );
        if cached > 0 {
            rec.with_cache(u64::from(cached), 0)
        } else {
            rec
        }
    }

    /// A priced record for a whole response, dated by its `created` field.
    /// `None` when the response carries no usage.
    pub fn record_from_response(resp: &CreateChatCompletionResponse) -> Option<CostRecord> {
        let usage = resp.usage.as_ref()?;
        let mut rec = record_from_usage(&resp.model, usage);
        if let Some(ts) = chrono::DateTime::from_timestamp(i64::from(resp.created), 0) {
            rec.timestamp = ts;
        }
        Some(rec)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn response_with_cached_tokens() {
            // A real-shaped Chat Completions response, deserialized by async-openai itself.
            let json = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1790000000,
                "model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop",
                "message":{"role":"assistant","content":"hi"}}],
                "usage":{"prompt_tokens":1000,"completion_tokens":100,"total_tokens":1100,
                "prompt_tokens_details":{"cached_tokens":600}}}"#;
            let resp: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();
            let rec = record_from_response(&resp).unwrap();
            assert_eq!(rec.input_tokens, 400);
            assert_eq!(rec.cache.cache_read_tokens, 600);
            // 400 x $2.50 + 600 x $1.25 + 100 x $10, per million.
            let expected = (400.0 * 2.5 + 600.0 * 1.25 + 100.0 * 10.0) / 1e6;
            assert!((rec.total_cost_usd - expected).abs() < 1e-12, "{}", rec.total_cost_usd);
            assert_eq!(rec.timestamp.timestamp(), 1_790_000_000);
            // The same response as a log line gives the same cost.
            let mut ing = crate::ingest::Ingester::new();
            ing.ingest_line(&json.replace('\n', " ")).unwrap();
            assert!((ing.ledger().records()[0].total_cost_usd - expected).abs() < 1e-12);
        }
    }
}
