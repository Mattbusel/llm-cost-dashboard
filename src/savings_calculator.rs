//! Cost savings opportunity identification for LLM workloads.
//!
//! Analyses cost metrics and produces prioritised, actionable savings
//! recommendations across six categories.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// SavingsCategory
// ---------------------------------------------------------------------------

/// Category of a cost savings opportunity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SavingsCategory {
    /// Switch some requests to a cheaper model.
    ModelDowngrade,
    /// Cache repeated prompts to avoid redundant inference.
    Caching,
    /// Group small requests into larger batches.
    Batching,
    /// Compress prompts to reduce token counts.
    Compression,
    /// Shorten or restructure prompts.
    PromptOptimization,
    /// Remove spending during idle periods.
    IdleElimination,
}

impl fmt::Display for SavingsCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SavingsCategory::ModelDowngrade => write!(f, "Model Downgrade"),
            SavingsCategory::Caching => write!(f, "Caching"),
            SavingsCategory::Batching => write!(f, "Batching"),
            SavingsCategory::Compression => write!(f, "Compression"),
            SavingsCategory::PromptOptimization => write!(f, "Prompt Optimization"),
            SavingsCategory::IdleElimination => write!(f, "Idle Elimination"),
        }
    }
}

// ---------------------------------------------------------------------------
// SavingsOpportunity
// ---------------------------------------------------------------------------

/// A single identified savings opportunity.
#[derive(Debug, Clone)]
pub struct SavingsOpportunity {
    /// Which category this opportunity belongs to.
    pub category: SavingsCategory,
    /// Estimated monthly savings in USD.
    pub potential_savings_usd: f64,
    /// Human-readable description of the opportunity.
    pub description: String,
    /// Confidence in the estimate, in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Concrete action the user should take to realise the saving.
    pub action_required: String,
}

// ---------------------------------------------------------------------------
// ModelSubstitution / BatchingOpportunity (helper structs)
// ---------------------------------------------------------------------------

/// Details of a model substitution opportunity.
#[derive(Debug, Clone)]
pub struct ModelSubstitution {
    /// Model to move away from.
    pub from_model: String,
    /// Cheaper model to use instead.
    pub to_model: String,
    /// Cost ratio of `to_model` relative to `from_model` (< 1.0 = cheaper).
    pub cost_ratio: f64,
    /// Expected quality impact (negative = degradation, 0 = neutral).
    pub quality_delta: f64,
    /// Fraction of requests eligible for substitution.
    pub eligible_request_pct: f64,
}

/// Details of a batching efficiency opportunity.
#[derive(Debug, Clone)]
pub struct BatchingOpportunity {
    /// Current average number of requests per batch.
    pub avg_batch_size: f64,
    /// Recommended batch size for maximum efficiency.
    pub optimal_batch_size: usize,
    /// Current requests per second.
    pub current_rps: f64,
    /// Estimated cost reduction percentage from better batching.
    pub estimated_savings_pct: f64,
}

// ---------------------------------------------------------------------------
// SavingsMetrics
// ---------------------------------------------------------------------------

/// Aggregate metrics required by [`SavingsCalculator::analyze_all`].
#[derive(Debug, Clone)]
pub struct SavingsMetrics {
    /// Average number of tokens per prompt.
    pub avg_prompt_tokens: f64,
    /// Current cache hit rate in `[0.0, 1.0]`.
    pub cache_hit_rate: f64,
    /// Average requests per batch.
    pub avg_batch_size: f64,
    /// Fraction of cost attributable to each model.
    pub model_distribution: HashMap<String, f64>,
    /// Fraction of hours in which the system is idle.
    pub idle_hours_pct: f64,
}

// ---------------------------------------------------------------------------
// SavingsCalculator
// ---------------------------------------------------------------------------

/// Computes cost savings opportunities from usage metrics.
pub struct SavingsCalculator;

impl SavingsCalculator {
    /// Create a new `SavingsCalculator`.
    pub fn new() -> Self {
        Self
    }

    // -----------------------------------------------------------------------
    // Individual opportunity calculators
    // -----------------------------------------------------------------------

    /// Estimate savings from substituting the current model mix with a cheaper
    /// alternative for `eligible_pct` of requests.
    ///
    /// `requests` is a slice of `(model, cost_usd, tokens)` tuples.
    /// `cheaper_model_ratio` is the cost ratio of the cheaper model (e.g.
    /// `0.3` means the alternative costs 30% of the current model).
    pub fn model_substitution_savings(
        requests: &[(String, f64, usize)],
        cheaper_model_ratio: f64,
        eligible_pct: f64,
    ) -> SavingsOpportunity {
        let total_cost: f64 = requests.iter().map(|(_, c, _)| c).sum();
        let eligible_cost = total_cost * eligible_pct.clamp(0.0, 1.0);
        let savings = eligible_cost * (1.0 - cheaper_model_ratio.clamp(0.0, 1.0));

        SavingsOpportunity {
            category: SavingsCategory::ModelDowngrade,
            potential_savings_usd: savings,
            description: format!(
                "Route {:.0}% of requests to a model costing {:.0}% of the current price",
                eligible_pct * 100.0,
                cheaper_model_ratio * 100.0
            ),
            confidence: 0.75,
            action_required: "Identify low-complexity requests and configure a cheaper model \
                              fallback in your routing rules."
                .to_string(),
        }
    }

    /// Estimate savings from caching repeated prompt responses.
    ///
    /// `total_cost` — total monthly spend.
    /// `cache_hit_rate` — fraction of requests that would be served from cache.
    /// `cacheable_pct` — fraction of requests that are cacheable.
    pub fn caching_savings(
        total_cost: f64,
        cache_hit_rate: f64,
        cacheable_pct: f64,
    ) -> SavingsOpportunity {
        let effective_hit_rate = cache_hit_rate.clamp(0.0, 1.0) * cacheable_pct.clamp(0.0, 1.0);
        let savings = total_cost * effective_hit_rate;

        SavingsOpportunity {
            category: SavingsCategory::Caching,
            potential_savings_usd: savings,
            description: format!(
                "A {:.0}% cache hit rate on {:.0}% cacheable traffic eliminates ${:.2}/month",
                cache_hit_rate * 100.0,
                cacheable_pct * 100.0,
                savings
            ),
            confidence: 0.80,
            action_required:
                "Enable semantic response caching with a TTL matched to your staleness tolerance."
                    .to_string(),
        }
    }

    /// Estimate savings from increasing request batch sizes.
    ///
    /// Larger batches amortise per-request overhead; savings are proportional
    /// to the ratio of batch size improvement.
    pub fn batching_savings(
        current_batch_size: f64,
        optimal_batch_size: usize,
        total_cost: f64,
    ) -> SavingsOpportunity {
        let optimal = optimal_batch_size as f64;
        let improvement = if current_batch_size <= 0.0 || optimal <= current_batch_size {
            0.0
        } else {
            1.0 - (current_batch_size / optimal)
        };
        let savings = total_cost * improvement * 0.5; // conservative: 50% of theoretical

        SavingsOpportunity {
            category: SavingsCategory::Batching,
            potential_savings_usd: savings,
            description: format!(
                "Increasing average batch size from {:.1} to {} could save {:.0}% of costs",
                current_batch_size,
                optimal_batch_size,
                improvement * 50.0
            ),
            confidence: 0.65,
            action_required: format!(
                "Configure request batching with a target batch size of {} and a \
                 max-wait window of 50 ms.",
                optimal_batch_size
            ),
        }
    }

    /// Estimate savings from compressing prompts.
    ///
    /// `avg_prompt_tokens` — average tokens per prompt before compression.
    /// `compression_ratio` — fraction of tokens remaining after compression
    ///   (e.g. `0.7` = 30% reduction).
    /// `cost_per_token` — cost in USD per input token.
    /// `num_requests` — number of requests per month.
    pub fn compression_savings(
        avg_prompt_tokens: f64,
        compression_ratio: f64,
        cost_per_token: f64,
        num_requests: u64,
    ) -> SavingsOpportunity {
        let tokens_saved_per_request = avg_prompt_tokens * (1.0 - compression_ratio.clamp(0.0, 1.0));
        let savings = tokens_saved_per_request * cost_per_token * num_requests as f64;

        SavingsOpportunity {
            category: SavingsCategory::Compression,
            potential_savings_usd: savings,
            description: format!(
                "Compressing prompts by {:.0}% saves {:.0} tokens per request across {} requests",
                (1.0 - compression_ratio) * 100.0,
                tokens_saved_per_request,
                num_requests
            ),
            confidence: 0.70,
            action_required: "Apply prompt summarisation or retrieval-augmented compression to \
                              reduce average prompt token count."
                .to_string(),
        }
    }

    /// Estimate savings from eliminating spend during idle hours.
    pub fn idle_elimination_savings(
        total_cost: f64,
        idle_hours_pct: f64,
    ) -> SavingsOpportunity {
        let savings = total_cost * idle_hours_pct.clamp(0.0, 1.0) * 0.90; // 90% recoverable

        SavingsOpportunity {
            category: SavingsCategory::IdleElimination,
            potential_savings_usd: savings,
            description: format!(
                "System is idle {:.0}% of the time; eliminating idle spend could save ${:.2}/month",
                idle_hours_pct * 100.0,
                savings
            ),
            confidence: 0.85,
            action_required:
                "Schedule workloads during peak hours and implement auto-scaling to zero during \
                 idle windows."
                    .to_string(),
        }
    }

    /// Run all analyses and return a full list of opportunities.
    pub fn analyze_all(
        &self,
        total_monthly_cost: f64,
        metrics: &SavingsMetrics,
    ) -> Vec<SavingsOpportunity> {
        let mut opportunities = Vec::new();

        // Model substitution: assume 40% eligible, cheaper model at 30% cost.
        let requests: Vec<(String, f64, usize)> = metrics
            .model_distribution
            .iter()
            .map(|(model, frac)| {
                (
                    model.clone(),
                    total_monthly_cost * frac,
                    (metrics.avg_prompt_tokens * frac) as usize,
                )
            })
            .collect();
        opportunities.push(Self::model_substitution_savings(&requests, 0.30, 0.40));

        // Caching.
        opportunities.push(Self::caching_savings(
            total_monthly_cost,
            metrics.cache_hit_rate,
            0.60,
        ));

        // Batching (optimal = 32 requests per batch).
        opportunities.push(Self::batching_savings(
            metrics.avg_batch_size,
            32,
            total_monthly_cost,
        ));

        // Compression (assume $0.000003 per token, 1M requests/month).
        opportunities.push(Self::compression_savings(
            metrics.avg_prompt_tokens,
            0.70,
            0.000_003,
            1_000_000,
        ));

        // Prompt optimisation: simple heuristic.
        let prompt_opt_savings = total_monthly_cost * 0.10;
        opportunities.push(SavingsOpportunity {
            category: SavingsCategory::PromptOptimization,
            potential_savings_usd: prompt_opt_savings,
            description: "Removing redundant instructions and system-prompt boilerplate could \
                          reduce tokens by ~10%."
                .to_string(),
            confidence: 0.60,
            action_required: "Audit system prompts for repetition and compress via few-shot \
                              distillation."
                .to_string(),
        });

        // Idle elimination.
        opportunities.push(Self::idle_elimination_savings(
            total_monthly_cost,
            metrics.idle_hours_pct,
        ));

        opportunities
    }

    /// Sum potential savings weighted by confidence.
    pub fn total_potential_savings(opportunities: &[SavingsOpportunity]) -> f64 {
        opportunities
            .iter()
            .map(|o| o.potential_savings_usd * o.confidence)
            .sum()
    }

    /// Return opportunities sorted descending by `potential_savings_usd *
    /// confidence`.
    pub fn prioritized_recommendations<'a>(
        opportunities: &'a [SavingsOpportunity],
    ) -> Vec<&'a SavingsOpportunity> {
        let mut refs: Vec<&SavingsOpportunity> = opportunities.iter().collect();
        refs.sort_by(|a, b| {
            let score_a = a.potential_savings_usd * a.confidence;
            let score_b = b.potential_savings_usd * b.confidence;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        refs
    }
}

impl Default for SavingsCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caching_savings() {
        let opp = SavingsCalculator::caching_savings(1000.0, 0.30, 0.80);
        assert!((opp.potential_savings_usd - 240.0).abs() < 1e-6);
        assert_eq!(opp.category, SavingsCategory::Caching);
    }

    #[test]
    fn test_idle_elimination() {
        let opp = SavingsCalculator::idle_elimination_savings(1000.0, 0.20);
        assert!((opp.potential_savings_usd - 180.0).abs() < 1e-6);
    }

    #[test]
    fn test_model_substitution() {
        let requests = vec![
            ("gpt-4".to_string(), 500.0, 1000),
            ("gpt-4".to_string(), 500.0, 1000),
        ];
        let opp = SavingsCalculator::model_substitution_savings(&requests, 0.30, 0.40);
        // eligible_cost = 1000 * 0.40 = 400; savings = 400 * 0.70 = 280
        assert!((opp.potential_savings_usd - 280.0).abs() < 1e-6);
    }

    #[test]
    fn test_batching_no_improvement_when_already_optimal() {
        let opp = SavingsCalculator::batching_savings(32.0, 32, 1000.0);
        assert_eq!(opp.potential_savings_usd, 0.0);
    }

    #[test]
    fn test_total_potential_savings() {
        let opps = vec![
            SavingsOpportunity {
                category: SavingsCategory::Caching,
                potential_savings_usd: 100.0,
                description: String::new(),
                confidence: 0.8,
                action_required: String::new(),
            },
            SavingsOpportunity {
                category: SavingsCategory::Batching,
                potential_savings_usd: 50.0,
                description: String::new(),
                confidence: 0.6,
                action_required: String::new(),
            },
        ];
        let total = SavingsCalculator::total_potential_savings(&opps);
        assert!((total - 110.0).abs() < 1e-9);
    }

    #[test]
    fn test_prioritized_recommendations() {
        let opps = vec![
            SavingsOpportunity {
                category: SavingsCategory::Caching,
                potential_savings_usd: 100.0,
                description: String::new(),
                confidence: 0.5,
                action_required: String::new(),
            },
            SavingsOpportunity {
                category: SavingsCategory::ModelDowngrade,
                potential_savings_usd: 200.0,
                description: String::new(),
                confidence: 0.9,
                action_required: String::new(),
            },
        ];
        let ranked = SavingsCalculator::prioritized_recommendations(&opps);
        assert_eq!(ranked[0].category, SavingsCategory::ModelDowngrade);
    }

    #[test]
    fn test_savings_category_display() {
        assert_eq!(SavingsCategory::ModelDowngrade.to_string(), "Model Downgrade");
        assert_eq!(SavingsCategory::IdleElimination.to_string(), "Idle Elimination");
    }

    #[test]
    fn test_analyze_all() {
        let calc = SavingsCalculator::new();
        let metrics = SavingsMetrics {
            avg_prompt_tokens: 500.0,
            cache_hit_rate: 0.25,
            avg_batch_size: 4.0,
            model_distribution: {
                let mut m = HashMap::new();
                m.insert("gpt-4".to_string(), 1.0);
                m
            },
            idle_hours_pct: 0.15,
        };
        let opps = calc.analyze_all(1000.0, &metrics);
        assert_eq!(opps.len(), 6);
    }
}
