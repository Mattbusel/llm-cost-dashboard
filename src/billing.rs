//! # Billing
//!
//! Invoice generation with line items, fluent builder API, and plain-text
//! invoice rendering.
//!
//! ## Example
//!
//! ```rust
//! use llm_cost_dashboard::billing::{InvoiceBuilder, InvoiceStore};
//!
//! let invoice = InvoiceBuilder::new("INV-001")
//!     .for_tenant("Acme Corp")
//!     .period(1_700_000_000, 1_702_678_400)
//!     .add_model_usage("gpt-4", 500_000, 30.0)
//!     .tax_rate(0.10)
//!     .build();
//!
//! let text = invoice.to_text();
//! assert!(text.contains("Acme Corp"));
//!
//! let mut store = InvoiceStore::default();
//! store.add(invoice);
//! ```

use std::collections::HashMap;

// ── LineItem ──────────────────────────────────────────────────────────────────

/// A single line on an invoice.
#[derive(Debug, Clone)]
pub struct LineItem {
    /// Human-readable description.
    pub description: String,
    /// Quantity of the unit consumed (e.g. number of tokens in thousands).
    pub quantity: f64,
    /// Unit label (e.g. "1k tokens", "requests").
    pub unit: String,
    /// Price per unit in USD.
    pub unit_price: f64,
    /// Pre-computed amount = `quantity * unit_price`.
    pub amount: f64,
}

impl LineItem {
    /// Create a new line item.  `amount` is computed automatically.
    pub fn new(
        description: impl Into<String>,
        quantity: f64,
        unit: impl Into<String>,
        unit_price: f64,
    ) -> Self {
        Self {
            description: description.into(),
            quantity,
            unit: unit.into(),
            unit_price,
            amount: quantity * unit_price,
        }
    }
}

// ── InvoiceStatus ─────────────────────────────────────────────────────────────

/// Lifecycle state of an invoice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvoiceStatus {
    /// Created but not yet sent to the tenant.
    Draft,
    /// Sent to the tenant and awaiting payment.
    Sent,
    /// Fully paid.
    Paid,
    /// Past the due date and unpaid.
    Overdue,
}

impl std::fmt::Display for InvoiceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvoiceStatus::Draft => write!(f, "DRAFT"),
            InvoiceStatus::Sent => write!(f, "SENT"),
            InvoiceStatus::Paid => write!(f, "PAID"),
            InvoiceStatus::Overdue => write!(f, "OVERDUE"),
        }
    }
}

// ── Invoice ───────────────────────────────────────────────────────────────────

/// A complete invoice for a billing period.
#[derive(Debug, Clone)]
pub struct Invoice {
    /// Unique invoice identifier.
    pub id: String,
    /// Tenant / customer name.
    pub tenant: String,
    /// Billing period start (Unix timestamp seconds).
    pub period_start: u64,
    /// Billing period end (Unix timestamp seconds).
    pub period_end: u64,
    /// Individual line items.
    pub line_items: Vec<LineItem>,
    /// Sum of all line item amounts before tax.
    pub subtotal: f64,
    /// Tax rate as a decimal (e.g. 0.10 for 10%).
    pub tax_rate: f64,
    /// Computed tax amount = `subtotal * tax_rate`.
    pub tax_amount: f64,
    /// Grand total = `subtotal + tax_amount`.
    pub total: f64,
    /// Invoice lifecycle status.
    pub status: InvoiceStatus,
    /// When the invoice was generated (Unix timestamp seconds).
    pub issued_at: u64,
    /// When payment is due (Unix timestamp seconds).
    pub due_at: u64,
}

impl Invoice {
    /// Create a new invoice with no line items.
    pub fn new(
        id: impl Into<String>,
        tenant: impl Into<String>,
        period_start: u64,
        period_end: u64,
        tax_rate: f64,
    ) -> Self {
        Self {
            id: id.into(),
            tenant: tenant.into(),
            period_start,
            period_end,
            line_items: Vec::new(),
            subtotal: 0.0,
            tax_rate,
            tax_amount: 0.0,
            total: 0.0,
            status: InvoiceStatus::Draft,
            issued_at: 0,
            due_at: 0,
        }
    }

    /// Add a line item and recompute `subtotal`, `tax_amount`, and `total`.
    pub fn add_line_item(&mut self, item: LineItem) {
        self.subtotal += item.amount;
        self.line_items.push(item);
        self.tax_amount = self.subtotal * self.tax_rate;
        self.total = self.subtotal + self.tax_amount;
    }

    /// Recompute all totals from the current line items (useful after mutation).
    fn recompute(&mut self) {
        self.subtotal = self.line_items.iter().map(|l| l.amount).sum();
        self.tax_amount = self.subtotal * self.tax_rate;
        self.total = self.subtotal + self.tax_amount;
    }

    /// Render the invoice as a plain-text ASCII document.
    pub fn to_text(&self) -> String {
        let width = 70usize;
        let divider = "─".repeat(width);
        let mut out = String::new();

        out.push_str(&format!("{}\n", divider));
        out.push_str(&format!(
            "{:^width$}\n",
            "INVOICE",
            width = width
        ));
        out.push_str(&format!("{}\n", divider));
        out.push_str(&format!("Invoice #: {}\n", self.id));
        out.push_str(&format!("Tenant   : {}\n", self.tenant));
        out.push_str(&format!(
            "Period   : {} – {}\n",
            self.period_start, self.period_end
        ));
        out.push_str(&format!(
            "Issued   : {}     Due: {}\n",
            self.issued_at, self.due_at
        ));
        out.push_str(&format!("Status   : {}\n", self.status));
        out.push_str(&format!("{}\n", divider));

        // Column headers.
        out.push_str(&format!(
            "{:<32} {:>8} {:<10} {:>10} {:>10}\n",
            "Description", "Qty", "Unit", "Unit Price", "Amount"
        ));
        out.push_str(&format!("{}\n", "─".repeat(width)));

        for item in &self.line_items {
            out.push_str(&format!(
                "{:<32} {:>8.2} {:<10} {:>10.4} {:>10.2}\n",
                truncate(&item.description, 32),
                item.quantity,
                truncate(&item.unit, 10),
                item.unit_price,
                item.amount
            ));
        }

        out.push_str(&format!("{}\n", divider));
        out.push_str(&format!("{:>62} {:>8.2}\n", "Subtotal:", self.subtotal));
        out.push_str(&format!(
            "{:>62} {:>8.2}\n",
            format!("Tax ({:.1}%):", self.tax_rate * 100.0),
            self.tax_amount
        ));
        out.push_str(&format!("{:>62} {:>8.2}\n", "TOTAL:", self.total));
        out.push_str(&format!("{}\n", divider));
        out
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

// ── InvoiceBuilder ────────────────────────────────────────────────────────────

/// Fluent builder for [`Invoice`].
pub struct InvoiceBuilder {
    id: String,
    tenant: String,
    period_start: u64,
    period_end: u64,
    tax_rate: f64,
    line_items: Vec<LineItem>,
    issued_at: u64,
    due_at: u64,
}

impl InvoiceBuilder {
    /// Start building an invoice with the given identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tenant: String::new(),
            period_start: 0,
            period_end: 0,
            tax_rate: 0.0,
            line_items: Vec::new(),
            issued_at: 0,
            due_at: 0,
        }
    }

    /// Set the tenant name.
    pub fn for_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = tenant.into();
        self
    }

    /// Set the billing period (Unix timestamp seconds).
    pub fn period(mut self, start: u64, end: u64) -> Self {
        self.period_start = start;
        self.period_end = end;
        self
    }

    /// Add a model usage line item.
    ///
    /// `tokens` is the raw token count; `cost_per_1k` is the cost per 1 000
    /// tokens.  The line item quantity is expressed in thousands of tokens.
    pub fn add_model_usage(
        mut self,
        model: &str,
        tokens: u64,
        cost_per_1k: f64,
    ) -> Self {
        let qty = tokens as f64 / 1000.0;
        let item = LineItem::new(
            format!("Model: {}", model),
            qty,
            "1k tokens".to_string(),
            cost_per_1k,
        );
        self.line_items.push(item);
        self
    }

    /// Set the tax rate (e.g. 0.10 for 10%).
    pub fn tax_rate(mut self, pct: f64) -> Self {
        self.tax_rate = pct;
        self
    }

    /// Set the `issued_at` timestamp.
    pub fn issued_at(mut self, ts: u64) -> Self {
        self.issued_at = ts;
        self
    }

    /// Set the `due_at` timestamp.
    pub fn due_at(mut self, ts: u64) -> Self {
        self.due_at = ts;
        self
    }

    /// Consume the builder and produce an [`Invoice`].
    pub fn build(self) -> Invoice {
        let mut invoice = Invoice::new(
            self.id,
            self.tenant,
            self.period_start,
            self.period_end,
            self.tax_rate,
        );
        invoice.issued_at = self.issued_at;
        invoice.due_at = self.due_at;
        for item in self.line_items {
            invoice.add_line_item(item);
        }
        invoice.recompute();
        invoice
    }
}

// ── InvoiceStore ──────────────────────────────────────────────────────────────

/// In-memory store for [`Invoice`] records.
#[derive(Default)]
pub struct InvoiceStore {
    invoices: HashMap<String, Invoice>,
}

impl InvoiceStore {
    /// Insert or replace an invoice.
    pub fn add(&mut self, invoice: Invoice) {
        self.invoices.insert(invoice.id.clone(), invoice);
    }

    /// Retrieve an invoice by ID.
    pub fn get(&self, id: &str) -> Option<&Invoice> {
        self.invoices.get(id)
    }

    /// Return all invoices for a given tenant (by name).
    pub fn list_for_tenant(&self, tenant: &str) -> Vec<&Invoice> {
        self.invoices
            .values()
            .filter(|inv| inv.tenant == tenant)
            .collect()
    }

    /// Mark an invoice as paid.  Returns `true` if the invoice existed.
    pub fn mark_paid(&mut self, id: &str) -> bool {
        if let Some(inv) = self.invoices.get_mut(id) {
            inv.status = InvoiceStatus::Paid;
            true
        } else {
            false
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_item_amount() {
        let item = LineItem::new("GPT-4 tokens", 100.0, "1k tokens", 0.03);
        // 100 * 0.03 = 3.0
        assert!((item.amount - 3.0).abs() < 1e-9, "amount={}", item.amount);
    }

    #[test]
    fn test_invoice_add_line_item_totals() {
        let mut inv = Invoice::new("INV-1", "Acme", 0, 0, 0.10);
        inv.add_line_item(LineItem::new("Item A", 10.0, "units", 5.0)); // 50
        inv.add_line_item(LineItem::new("Item B", 2.0, "units", 25.0)); // 50
        // subtotal = 100, tax = 10, total = 110
        assert!((inv.subtotal - 100.0).abs() < 1e-9, "subtotal={}", inv.subtotal);
        assert!((inv.tax_amount - 10.0).abs() < 1e-9, "tax={}", inv.tax_amount);
        assert!((inv.total - 110.0).abs() < 1e-9, "total={}", inv.total);
    }

    #[test]
    fn test_to_text_contains_tenant() {
        let inv = InvoiceBuilder::new("INV-42")
            .for_tenant("BigCo LLC")
            .period(1_700_000_000, 1_702_678_400)
            .add_model_usage("claude-3", 200_000, 15.0)
            .tax_rate(0.08)
            .build();
        let text = inv.to_text();
        assert!(text.contains("BigCo LLC"), "tenant name missing from invoice");
        assert!(text.contains("INV-42"), "invoice id missing");
        assert!(text.contains("TOTAL:"), "total line missing");
    }

    #[test]
    fn test_builder_roundtrip() {
        let inv = InvoiceBuilder::new("INV-99")
            .for_tenant("Startup")
            .period(0, 86_400)
            .add_model_usage("gpt-4", 500_000, 30.0)
            .tax_rate(0.0)
            .build();

        // 500 (qty in k) * 30.0 = 15000
        assert!((inv.subtotal - 15_000.0).abs() < 1e-6, "subtotal={}", inv.subtotal);
        assert_eq!(inv.tenant, "Startup");
        assert_eq!(inv.line_items.len(), 1);
    }

    #[test]
    fn test_store_mark_paid() {
        let mut store = InvoiceStore::default();
        let inv = InvoiceBuilder::new("INV-7")
            .for_tenant("Corp")
            .build();
        store.add(inv);
        assert!(store.mark_paid("INV-7"));
        assert_eq!(store.get("INV-7").unwrap().status, InvoiceStatus::Paid);
        assert!(!store.mark_paid("INV-NOPE"));
    }

    #[test]
    fn test_list_for_tenant() {
        let mut store = InvoiceStore::default();
        store.add(InvoiceBuilder::new("A").for_tenant("X").build());
        store.add(InvoiceBuilder::new("B").for_tenant("X").build());
        store.add(InvoiceBuilder::new("C").for_tenant("Y").build());
        assert_eq!(store.list_for_tenant("X").len(), 2);
        assert_eq!(store.list_for_tenant("Y").len(), 1);
        assert_eq!(store.list_for_tenant("Z").len(), 0);
    }

    #[test]
    fn test_zero_tax() {
        let mut inv = Invoice::new("I", "T", 0, 0, 0.0);
        inv.add_line_item(LineItem::new("x", 1.0, "u", 50.0));
        assert!((inv.tax_amount).abs() < 1e-9);
        assert!((inv.total - inv.subtotal).abs() < 1e-9);
    }
}
