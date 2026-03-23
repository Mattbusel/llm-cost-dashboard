//! Immutable audit trail for cost events.
//!
//! [`AuditLogger`] records append-only [`AuditEvent`] entries with FNV-1a
//! checksums for tamper detection.  All public query methods return cloned
//! snapshots so callers are not coupled to the internal lock.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// AuditEventKind
// ---------------------------------------------------------------------------

/// Discriminant for the kind of activity captured in an [`AuditEvent`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AuditEventKind {
    /// A cost record was appended to the ledger.
    CostRecorded,
    /// Spend exceeded a configured budget threshold.
    BudgetExceeded,
    /// The model associated with a request was changed.
    ModelChanged,
    /// A model's price was updated.
    PriceUpdated,
    /// An alert was triggered and dispatched.
    AlertTriggered,
    /// A cost report was exported.
    ExportGenerated,
}

impl AuditEventKind {
    fn as_str(&self) -> &'static str {
        match self {
            AuditEventKind::CostRecorded => "CostRecorded",
            AuditEventKind::BudgetExceeded => "BudgetExceeded",
            AuditEventKind::ModelChanged => "ModelChanged",
            AuditEventKind::PriceUpdated => "PriceUpdated",
            AuditEventKind::AlertTriggered => "AlertTriggered",
            AuditEventKind::ExportGenerated => "ExportGenerated",
        }
    }
}

impl fmt::Display for AuditEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// AuditEvent
// ---------------------------------------------------------------------------

/// A single immutable audit record.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Unique, monotonically increasing event identifier.
    pub id: u64,
    /// Unix-epoch milliseconds when the event was recorded.
    pub timestamp_ms: u64,
    /// The kind of event.
    pub kind: AuditEventKind,
    /// Identity of the actor that caused the event (user, service name, etc.).
    pub actor: String,
    /// The resource (model id, session id, budget name, …) the event concerns.
    pub resource: String,
    /// Arbitrary key/value metadata.
    pub details: HashMap<String, String>,
    /// FNV-1a checksum over `id + timestamp_ms + actor + resource`.
    pub checksum: u64,
}

// ---------------------------------------------------------------------------
// Checksum
// ---------------------------------------------------------------------------

/// Computes the FNV-1a 64-bit checksum for an event.
///
/// The hash input is the concatenation (as bytes) of:
/// `id` (little-endian u64) + `timestamp_ms` (little-endian u64) +
/// `actor` (UTF-8) + `resource` (UTF-8).
pub fn checksum(event: &AuditEvent) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let mut hash = FNV_OFFSET;

    let hash_bytes = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(FNV_PRIME);
        }
    };

    hash_bytes(&mut hash, &event.id.to_le_bytes());
    hash_bytes(&mut hash, &event.timestamp_ms.to_le_bytes());
    hash_bytes(&mut hash, event.actor.as_bytes());
    hash_bytes(&mut hash, event.resource.as_bytes());

    hash
}

// ---------------------------------------------------------------------------
// AuditSummary
// ---------------------------------------------------------------------------

/// Aggregate statistics over all events in the logger.
#[derive(Debug, Clone)]
pub struct AuditSummary {
    /// Total number of events recorded.
    pub total_events: usize,
    /// Count of events grouped by their kind name.
    pub events_by_type: HashMap<String, usize>,
    /// Timestamp of the earliest event, if any.
    pub first_event_ms: Option<u64>,
    /// Timestamp of the most recent event, if any.
    pub last_event_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// AuditLogger
// ---------------------------------------------------------------------------

/// Thread-safe, append-only audit log with FNV-1a integrity verification.
pub struct AuditLogger {
    /// Interior-mutable event store.
    events: Mutex<Vec<AuditEvent>>,
    /// Monotonically increasing event id counter.
    next_id: AtomicU64,
}

impl AuditLogger {
    /// Creates an empty audit logger.
    pub fn new() -> Self {
        AuditLogger {
            events: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Records a new audit event and returns the assigned event id.
    ///
    /// The `timestamp_ms` is set to 0 in this implementation; callers that
    /// require wall-clock timestamps should pass the value via `details`.
    /// For a production system the timestamp would be injected as a
    /// dependency (clock abstraction), but for testability we keep it simple.
    pub fn log(
        &self,
        kind: AuditEventKind,
        actor: String,
        resource: String,
        details: HashMap<String, String>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let timestamp_ms = details
            .get("timestamp_ms")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0u64);

        // Build a partial event to compute the checksum
        let mut event = AuditEvent {
            id,
            timestamp_ms,
            kind,
            actor,
            resource,
            details,
            checksum: 0,
        };
        event.checksum = checksum(&event);

        let mut guard = self.events.lock().expect("audit lock poisoned");
        guard.push(event);
        id
    }

    /// Returns `true` iff every stored event's checksum matches its recomputed
    /// value.  A mismatch indicates tampering.
    pub fn verify_integrity(&self) -> bool {
        let guard = self.events.lock().expect("audit lock poisoned");
        for event in guard.iter() {
            if checksum(event) != event.checksum {
                return false;
            }
        }
        true
    }

    /// Returns clones of all events whose `resource` field equals `resource`.
    pub fn events_for_resource(&self, resource: &str) -> Vec<AuditEvent> {
        let guard = self.events.lock().expect("audit lock poisoned");
        guard
            .iter()
            .filter(|e| e.resource == resource)
            .cloned()
            .collect()
    }

    /// Returns clones of all events with `timestamp_ms >= timestamp_ms`.
    pub fn events_since(&self, timestamp_ms: u64) -> Vec<AuditEvent> {
        let guard = self.events.lock().expect("audit lock poisoned");
        guard
            .iter()
            .filter(|e| e.timestamp_ms >= timestamp_ms)
            .cloned()
            .collect()
    }

    /// Returns clones of all events whose kind discriminant matches `kind`.
    pub fn events_by_kind(&self, kind: &AuditEventKind) -> Vec<AuditEvent> {
        let target = kind.as_str();
        let guard = self.events.lock().expect("audit lock poisoned");
        guard
            .iter()
            .filter(|e| e.kind.as_str() == target)
            .cloned()
            .collect()
    }

    /// Serialises all events to CSV with header
    /// `id,timestamp,kind,actor,resource,checksum`.
    pub fn export_csv(&self) -> String {
        let guard = self.events.lock().expect("audit lock poisoned");
        let mut out = String::from("id,timestamp,kind,actor,resource,checksum\n");
        for e in guard.iter() {
            out.push_str(&format!(
                "{},{},{},{},{},{}\n",
                e.id,
                e.timestamp_ms,
                e.kind.as_str(),
                e.actor,
                e.resource,
                e.checksum
            ));
        }
        out
    }

    /// Computes a summary of all recorded events.
    pub fn summary(&self) -> AuditSummary {
        let guard = self.events.lock().expect("audit lock poisoned");
        let mut events_by_type: HashMap<String, usize> = HashMap::new();
        let mut first_event_ms: Option<u64> = None;
        let mut last_event_ms: Option<u64> = None;

        for e in guard.iter() {
            *events_by_type.entry(e.kind.as_str().to_string()).or_insert(0) += 1;
            first_event_ms = Some(match first_event_ms {
                None => e.timestamp_ms,
                Some(prev) => prev.min(e.timestamp_ms),
            });
            last_event_ms = Some(match last_event_ms {
                None => e.timestamp_ms,
                Some(prev) => prev.max(e.timestamp_ms),
            });
        }

        AuditSummary {
            total_events: guard.len(),
            events_by_type,
            first_event_ms,
            last_event_ms,
        }
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_details(ts: u64) -> HashMap<String, String> {
        let mut d = HashMap::new();
        d.insert("timestamp_ms".to_string(), ts.to_string());
        d
    }

    #[test]
    fn test_log_and_retrieve() {
        let logger = AuditLogger::new();
        let id = logger.log(
            AuditEventKind::CostRecorded,
            "svc".to_string(),
            "gpt-4".to_string(),
            make_details(1000),
        );
        assert_eq!(id, 1);
        let events = logger.events_for_resource("gpt-4");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, 1);
    }

    #[test]
    fn test_integrity_check_passes() {
        let logger = AuditLogger::new();
        logger.log(
            AuditEventKind::PriceUpdated,
            "admin".to_string(),
            "gpt-3.5".to_string(),
            make_details(500),
        );
        assert!(logger.verify_integrity());
    }

    #[test]
    fn test_tamper_detection() {
        let logger = AuditLogger::new();
        logger.log(
            AuditEventKind::BudgetExceeded,
            "system".to_string(),
            "budget-a".to_string(),
            make_details(0),
        );
        // Tamper: modify the actor directly
        {
            let mut guard = logger.events.lock().unwrap();
            guard[0].actor = "attacker".to_string();
        }
        assert!(!logger.verify_integrity());
    }

    #[test]
    fn test_csv_export_has_header() {
        let logger = AuditLogger::new();
        logger.log(
            AuditEventKind::ExportGenerated,
            "user".to_string(),
            "report.csv".to_string(),
            HashMap::new(),
        );
        let csv = logger.export_csv();
        assert!(csv.starts_with("id,timestamp,kind,actor,resource,checksum\n"));
        let lines: Vec<&str> = csv.trim_end().lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 data row
    }

    #[test]
    fn test_resource_filter() {
        let logger = AuditLogger::new();
        logger.log(
            AuditEventKind::CostRecorded,
            "svc".to_string(),
            "model-a".to_string(),
            HashMap::new(),
        );
        logger.log(
            AuditEventKind::CostRecorded,
            "svc".to_string(),
            "model-b".to_string(),
            HashMap::new(),
        );
        let filtered = logger.events_for_resource("model-a");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].resource, "model-a");
    }

    #[test]
    fn test_events_since() {
        let logger = AuditLogger::new();
        logger.log(
            AuditEventKind::CostRecorded,
            "svc".to_string(),
            "m".to_string(),
            make_details(100),
        );
        logger.log(
            AuditEventKind::CostRecorded,
            "svc".to_string(),
            "m".to_string(),
            make_details(200),
        );
        let recent = logger.events_since(150);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].timestamp_ms, 200);
    }
}
