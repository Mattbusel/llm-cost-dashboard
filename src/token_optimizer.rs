//! Prompt token optimization via redundancy removal.

/// Optimization strategies available.
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Remove duplicate sentences, keeping the first occurrence.
    RemoveDuplicateSentences,
    /// Truncate example lines to at most this many examples.
    TruncateExamples(usize),
    /// Strip common filler words from the text.
    RemoveFillerWords,
    /// Collapse multiple spaces and newlines into single whitespace.
    CompressWhitespace,
    /// Abbreviate large numbers: 1000 -> 1K, 1000000 -> 1M.
    AbbreviateNumbers,
    /// Apply all available strategies.
    All,
}

/// Result of an optimization pass.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Token count before optimization.
    pub original_tokens: usize,
    /// Token count after optimization.
    pub optimized_tokens: usize,
    /// Percentage of tokens saved.
    pub savings_pct: f64,
    /// Names of strategies that were applied.
    pub strategies_applied: Vec<String>,
}

/// Prompt token optimizer.
pub struct TokenOptimizer {
    /// Strategies to apply in order.
    pub strategies: Vec<OptimizationStrategy>,
    /// Filler words to remove when `RemoveFillerWords` is applied.
    pub filler_words: Vec<String>,
}

impl TokenOptimizer {
    /// Create a new `TokenOptimizer` with the given strategies.
    pub fn new(strategies: Vec<OptimizationStrategy>) -> Self {
        Self {
            strategies,
            filler_words: vec![
                "basically".to_string(),
                "actually".to_string(),
                "literally".to_string(),
                "just".to_string(),
                "very".to_string(),
                "really".to_string(),
                "quite".to_string(),
                "simply".to_string(),
            ],
        }
    }

    /// Estimate token count: words * 1.3 + 1.
    pub fn estimate_tokens(text: &str) -> usize {
        let words = text.split_whitespace().count();
        ((words as f64 * 1.3) + 1.0) as usize
    }

    /// Remove duplicate sentences, keeping the first occurrence.
    /// Sentences are split on [.!?]+ followed by whitespace.
    pub fn remove_duplicate_sentences(text: &str) -> String {
        let mut seen = std::collections::HashSet::new();
        let mut result = String::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find next sentence boundary
            let boundary = remaining
                .char_indices()
                .find(|(_, c)| matches!(c, '.' | '!' | '?'))
                .map(|(i, c)| i + c.len_utf8());

            let (sentence, rest) = if let Some(end) = boundary {
                // Consume trailing punctuation and whitespace
                let mut cut = end;
                let bytes = remaining.as_bytes();
                while cut < bytes.len() && matches!(bytes[cut], b'.' | b'!' | b'?') {
                    cut += 1;
                }
                let sep_end = {
                    let mut s = cut;
                    while s < bytes.len() && bytes[s] == b' ' {
                        s += 1;
                    }
                    s
                };
                (&remaining[..cut], &remaining[sep_end..])
            } else {
                (remaining, "")
            };

            let key = sentence.trim().to_lowercase();
            if !key.is_empty() && !seen.contains(&key) {
                seen.insert(key);
                if !result.is_empty() && !result.ends_with(' ') {
                    result.push(' ');
                }
                result.push_str(sentence.trim());
            }
            remaining = rest;
        }
        result
    }

    /// Keep only the first `max_examples` lines that start with "Example", "-", or "*".
    pub fn truncate_examples(text: &str, max_examples: usize) -> String {
        let mut example_count = 0;
        let mut lines = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Example") || trimmed.starts_with('-') || trimmed.starts_with('*') {
                if example_count < max_examples {
                    example_count += 1;
                    lines.push(line);
                }
                // Skip examples beyond max
            } else {
                lines.push(line);
            }
        }
        lines.join("\n")
    }

    /// Remove filler words from the text (whole-word matching).
    pub fn remove_filler_words(text: &str, fillers: &[String]) -> String {
        let mut result = text.to_string();
        for filler in fillers {
            // Replace whole-word occurrences (case-insensitive)
            let pattern_lower = filler.to_lowercase();
            let mut new_result = String::new();
            let mut remaining = result.as_str();
            while !remaining.is_empty() {
                let lower_rem = remaining.to_lowercase();
                if let Some(pos) = lower_rem.find(pattern_lower.as_str()) {
                    // Check word boundaries
                    let before_ok = pos == 0 || !remaining.as_bytes()[pos - 1].is_ascii_alphanumeric();
                    let after_pos = pos + filler.len();
                    let after_ok = after_pos >= remaining.len() || !remaining.as_bytes()[after_pos].is_ascii_alphanumeric();
                    if before_ok && after_ok {
                        new_result.push_str(&remaining[..pos]);
                        // Skip trailing space if present
                        let skip_end = if after_pos < remaining.len() && remaining.as_bytes()[after_pos] == b' ' {
                            after_pos + 1
                        } else {
                            after_pos
                        };
                        remaining = &remaining[skip_end..];
                    } else {
                        new_result.push_str(&remaining[..pos + 1]);
                        remaining = &remaining[pos + 1..];
                    }
                } else {
                    new_result.push_str(remaining);
                    remaining = "";
                }
            }
            result = new_result;
        }
        result
    }

    /// Collapse multiple spaces and newlines into single whitespace.
    pub fn compress_whitespace(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_ws = false;
        let mut prev_nl = false;
        for c in text.chars() {
            if c == '\n' {
                if !prev_nl {
                    result.push('\n');
                    prev_nl = true;
                }
                prev_ws = true;
            } else if c.is_whitespace() {
                if !prev_ws {
                    result.push(' ');
                }
                prev_ws = true;
                prev_nl = false;
            } else {
                result.push(c);
                prev_ws = false;
                prev_nl = false;
            }
        }
        result
    }

    /// Abbreviate large numbers: 1000 -> 1K, 1000000 -> 1M.
    pub fn abbreviate_numbers(text: &str) -> String {
        let mut result = String::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            // Find a run of digits
            if let Some(start) = remaining.bytes().position(|b| b.is_ascii_digit()) {
                result.push_str(&remaining[..start]);
                let rest = &remaining[start..];
                let end = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
                let num_str = &rest[..end];
                if let Ok(n) = num_str.parse::<u64>() {
                    if n >= 1_000_000 {
                        let m = n / 1_000_000;
                        result.push_str(&format!("{}M", m));
                    } else if n >= 1_000 {
                        let k = n / 1_000;
                        result.push_str(&format!("{}K", k));
                    } else {
                        result.push_str(num_str);
                    }
                } else {
                    result.push_str(num_str);
                }
                remaining = &rest[end..];
            } else {
                result.push_str(remaining);
                break;
            }
        }
        result
    }

    /// Apply each configured strategy in order.
    pub fn optimize(&self, text: &str) -> (String, OptimizationResult) {
        let original_tokens = Self::estimate_tokens(text);
        let mut current = text.to_string();
        let mut strategies_applied = Vec::new();

        for strategy in &self.strategies {
            match strategy {
                OptimizationStrategy::RemoveDuplicateSentences => {
                    current = Self::remove_duplicate_sentences(&current);
                    strategies_applied.push("RemoveDuplicateSentences".to_string());
                }
                OptimizationStrategy::TruncateExamples(max) => {
                    current = Self::truncate_examples(&current, *max);
                    strategies_applied.push(format!("TruncateExamples({})", max));
                }
                OptimizationStrategy::RemoveFillerWords => {
                    current = Self::remove_filler_words(&current, &self.filler_words);
                    strategies_applied.push("RemoveFillerWords".to_string());
                }
                OptimizationStrategy::CompressWhitespace => {
                    current = Self::compress_whitespace(&current);
                    strategies_applied.push("CompressWhitespace".to_string());
                }
                OptimizationStrategy::AbbreviateNumbers => {
                    current = Self::abbreviate_numbers(&current);
                    strategies_applied.push("AbbreviateNumbers".to_string());
                }
                OptimizationStrategy::All => {
                    current = Self::remove_duplicate_sentences(&current);
                    current = Self::remove_filler_words(&current, &self.filler_words);
                    current = Self::compress_whitespace(&current);
                    current = Self::abbreviate_numbers(&current);
                    strategies_applied.push("All".to_string());
                }
            }
        }

        let optimized_tokens = Self::estimate_tokens(&current);
        let savings_pct = if original_tokens > 0 {
            (1.0 - optimized_tokens as f64 / original_tokens as f64) * 100.0
        } else {
            0.0
        };

        (
            current,
            OptimizationResult {
                original_tokens,
                optimized_tokens,
                savings_pct,
                strategies_applied,
            },
        )
    }

    /// Apply all strategies.
    pub fn optimize_all(text: &str) -> (String, OptimizationResult) {
        let optimizer = TokenOptimizer::new(vec![OptimizationStrategy::All]);
        optimizer.optimize(text)
    }

    /// Apply optimization to a batch of prompts.
    pub fn batch_optimize(&self, prompts: &[String]) -> Vec<(String, OptimizationResult)> {
        prompts.iter().map(|p| self.optimize(p)).collect()
    }
}

/// Aggregated savings report across a batch.
#[derive(Debug, Clone)]
pub struct SavingsReport {
    /// Total tokens across all original prompts.
    pub total_original_tokens: usize,
    /// Total tokens across all optimized prompts.
    pub total_optimized_tokens: usize,
    /// Overall percentage saved.
    pub total_savings_pct: f64,
    /// Per-prompt optimization results.
    pub per_prompt: Vec<OptimizationResult>,
}

/// Build a `SavingsReport` from batch results.
pub fn batch_report(results: &[(String, OptimizationResult)]) -> SavingsReport {
    let per_prompt: Vec<OptimizationResult> = results.iter().map(|(_, r)| r.clone()).collect();
    let total_original_tokens: usize = per_prompt.iter().map(|r| r.original_tokens).sum();
    let total_optimized_tokens: usize = per_prompt.iter().map(|r| r.optimized_tokens).sum();
    let total_savings_pct = if total_original_tokens > 0 {
        (1.0 - total_optimized_tokens as f64 / total_original_tokens as f64) * 100.0
    } else {
        0.0
    };
    SavingsReport {
        total_original_tokens,
        total_optimized_tokens,
        total_savings_pct,
        per_prompt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duplicate_sentences_removed() {
        let text = "Hello world. Hello world. This is unique.";
        let result = TokenOptimizer::remove_duplicate_sentences(text);
        // Should appear only once
        let count = result.matches("Hello world").count();
        assert_eq!(count, 1, "Duplicate sentence should be removed, got: {}", result);
    }

    #[test]
    fn test_filler_words_stripped() {
        let fillers = vec!["basically".to_string(), "just".to_string()];
        let text = "It is basically just a test.";
        let result = TokenOptimizer::remove_filler_words(text, &fillers);
        assert!(!result.contains("basically"), "filler 'basically' should be removed");
        assert!(!result.contains(" just "), "filler 'just' should be removed");
    }

    #[test]
    fn test_whitespace_compressed() {
        let text = "Hello    world\n\n\nfoo";
        let result = TokenOptimizer::compress_whitespace(text);
        assert!(!result.contains("    "), "multiple spaces should be collapsed");
        assert!(!result.contains("\n\n"), "multiple newlines should be collapsed");
    }

    #[test]
    fn test_numbers_abbreviated() {
        let text = "We have 1000 items and 2000000 users.";
        let result = TokenOptimizer::abbreviate_numbers(text);
        assert!(result.contains("1K"), "1000 should become 1K, got: {}", result);
        assert!(result.contains("2M"), "2000000 should become 2M, got: {}", result);
    }

    #[test]
    fn test_savings_pct_correct() {
        // A text with duplicate sentences will shrink
        let text = "Hello world. Hello world. Hello world. Hello world. Extra sentence here.";
        let (_, result) = TokenOptimizer::optimize_all(text);
        assert!(result.savings_pct >= 0.0, "savings_pct must be non-negative");
        assert!(result.optimized_tokens <= result.original_tokens, "optimization should not increase tokens");
        // Verify formula
        let expected = (1.0 - result.optimized_tokens as f64 / result.original_tokens as f64) * 100.0;
        assert!((result.savings_pct - expected).abs() < 1e-6);
    }
}
