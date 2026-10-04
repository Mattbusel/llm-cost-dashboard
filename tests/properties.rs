//! Property tests (proptest) for the parts that take untrusted input or do
//! arithmetic: log-line parsing, price files, and the forecaster.

use llm_cost_dashboard::cost::pricing::load_prices_json;
use llm_cost_dashboard::forecast::CostForecaster;
use llm_cost_dashboard::log::{parse_timestamp, RequestLog};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Any text at all: parsing returns Ok or Err, never panics.
    #[test]
    fn ingest_never_panics_on_arbitrary_text(line in ".{0,400}") {
        let mut log = RequestLog::new();
        let _ = log.ingest_line(&line);
    }

    /// JSON objects built from the field names the parser looks at, with
    /// values of any JSON type: never panics, and a line it accepts has the
    /// token counts it was given.
    #[test]
    fn ingest_handles_hostile_json(
        model in prop_oneof![Just(serde_json::json!("gpt-4o")), any::<i64>().prop_map(serde_json::Value::from), Just(serde_json::Value::Null)],
        input in prop_oneof![any::<u64>().prop_map(serde_json::Value::from), any::<f64>().prop_map(|f| serde_json::json!(f)), Just(serde_json::json!("12"))],
        output in any::<u64>(),
        ts in prop_oneof![Just(serde_json::json!("2026-10-03T00:00:00Z")), any::<f64>().prop_map(|f| serde_json::json!(f)), Just(serde_json::json!("garbage")), Just(serde_json::json!([1]))],
        nested in any::<bool>(),
        claude_code in any::<bool>(),
    ) {
        let mut v = if nested {
            serde_json::json!({"model": model, "usage": {"prompt_tokens": input, "completion_tokens": output}, "timestamp": ts})
        } else {
            serde_json::json!({"model": model, "input_tokens": input, "output_tokens": output, "timestamp": ts})
        };
        if claude_code {
            v["sessionId"] = serde_json::json!("s");
        }
        let mut log = RequestLog::new();
        if log.ingest_line(&v.to_string()).is_ok() && !log.is_empty() {
            prop_assert_eq!(log.all()[0].output_tokens, output);
        }
    }

    /// Timestamps: any number or string, never panics.
    #[test]
    fn parse_timestamp_never_panics(n in any::<f64>(), s in ".{0,40}") {
        let _ = parse_timestamp(&serde_json::json!(n));
        let _ = parse_timestamp(&serde_json::json!(s));
    }

    /// Price files: arbitrary objects never panic, and a loaded price is never
    /// negative or NaN.
    #[test]
    fn price_file_never_loads_bad_numbers(
        name in "zzprop-[a-z]{1,8}",
        i in any::<f64>(),
        o in any::<f64>(),
    ) {
        let json = serde_json::json!({ &name: {"input_usd_per_1m": i, "output_usd_per_1m": o} }).to_string();
        if let Ok(1) = load_prices_json(&json) {
            let (pi, po) = llm_cost_dashboard::cost::pricing::lookup_known(&name).unwrap();
            prop_assert!(pi.is_finite() && pi >= 0.0 && po.is_finite() && po >= 0.0);
        }
    }

    /// Forecast of a non-decreasing cumulative series: every horizon is
    /// finite, non-negative, and longer horizons never project less.
    #[test]
    fn forecast_is_finite_monotone_and_non_negative(
        gaps in prop::collection::vec(1.0f64..200_000.0, 3..80),
        costs in prop::collection::vec(0.0f64..5.0, 3..80),
    ) {
        let mut f = CostForecaster::new();
        let mut t = 1_790_000_000.0;
        let mut cum = 0.0;
        for (g, c) in gaps.iter().zip(costs.iter()) {
            t += g;
            cum += c;
            f.record(t, cum);
        }
        if let Some(r) = f.forecast(Some(100.0)) {
            for v in [r.next_hour_usd, r.next_day_usd, r.next_week_usd, r.next_month_usd] {
                prop_assert!(v.is_finite() && v >= 0.0, "bad value {}", v);
            }
            prop_assert!(r.next_day_usd + 1e-9 >= r.next_hour_usd);
            prop_assert!(r.next_week_usd + 1e-9 >= r.next_day_usd);
            prop_assert!(r.next_month_usd + 1e-9 >= r.next_week_usd);
            prop_assert!(r.confidence_interval.0 <= r.confidence_interval.1);
        }
    }
}
