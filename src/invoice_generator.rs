//! # Invoice Generator
//!
//! Generates detailed, line-item invoices from raw LLM usage records.
//!
//! ## Example
//!
//! ```rust
//! use std::collections::HashMap;
//! use llm_cost_dashboard::invoice_generator::{InvoiceGenerator, UsageRecord};
//!
//! let mut prices = HashMap::new();
//! prices.insert("gpt-4o".to_string(), 0.005); // USD per 1k tokens (combined)
//!
//! let records = vec![
//!     UsageRecord {
//!         model_id: "gpt-4o".to_string(),
//!         input_tokens: 1000,
//!         output_tokens: 500,
//!         timestamp: 1_700_000_100,
//!         customer_id: "acme".to_string(),
//!     },
//! ];
//!
//! let invoice = InvoiceGenerator::generate(
//!     "acme", &records, &prices,
//!     1_700_000_000, 1_700_086_400, 0.1,
//! );
//! assert!(invoice.total_usd > 0.0);
//! ```

use std::collections::HashMap;

// ── UsageRecord ───────────────────────────────────────────────────────────────

/// A single LLM API call recorded for billing purposes.
#[derive(Debug, Clone)]
pub struct UsageRecord {
    /// Model identifier (must match a key in the price map).
    pub model_id: String,
    /// Number of input tokens consumed.
    pub input_tokens: u64,
    /// Number of output tokens generated.
    pub output_tokens: u64,
    /// Unix-epoch seconds when the call occurred.
    pub timestamp: u64,
    /// Customer / tenant identifier.
    pub customer_id: String,
}

// ── InvoiceStatus ─────────────────────────────────────────────────────────────

/// Lifecycle status of an [`Invoice`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvoiceStatus {
    /// Created but not yet sent to the customer.
    Draft,
    /// Sent to the customer; payment not yet received.
    Issued,
    /// Payment received in full.
    Paid,
    /// Payment is past due.
    Overdue,
    /// Invoice has been cancelled.
    Void,
}

impl std::fmt::Display for InvoiceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Draft => "Draft",
            Self::Issued => "Issued",
            Self::Paid => "Paid",
            Self::Overdue => "Overdue",
            Self::Void => "Void",
        };
        write!(f, "{}", s)
    }
}

// ── InvoiceLineItem ───────────────────────────────────────────────────────────

/// One line on an invoice corresponding to a model's usage in a period.
#[derive(Debug, Clone)]
pub struct InvoiceLineItem {
    /// Human-readable description (e.g. "GPT-4o usage — 2024-11").
    pub description: String,
    /// Model identifier.
    pub model_id: String,
    /// Total input tokens for this line.
    pub input_tokens: u64,
    /// Total output tokens for this line.
    pub output_tokens: u64,
    /// Cost per 1 000 tokens (combined input+output for simplicity).
    pub unit_cost_usd: f64,
    /// Computed total: `(input_tokens + output_tokens) / 1000 * unit_cost_usd`.
    pub total_usd: f64,
    /// Billing period label (e.g. "2024-11-01 – 2024-11-30").
    pub period: String,
}

// ── Invoice ───────────────────────────────────────────────────────────────────

/// A fully computed invoice ready for issuance.
#[derive(Debug, Clone)]
pub struct Invoice {
    /// Unique invoice identifier.
    pub invoice_id: String,
    /// Customer / tenant identifier.
    pub customer_id: String,
    /// Billing period start (Unix epoch seconds).
    pub period_start: u64,
    /// Billing period end (Unix epoch seconds).
    pub period_end: u64,
    /// Individual line items (one per model, plus optional discount).
    pub line_items: Vec<InvoiceLineItem>,
    /// Sum of line item totals before tax and discounts.
    pub subtotal_usd: f64,
    /// Tax rate fraction (e.g. 0.10 for 10 %).
    pub tax_rate: f64,
    /// Computed tax: `subtotal_usd * tax_rate`.
    pub tax_usd: f64,
    /// Final amount due: `subtotal_usd + tax_usd`.
    pub total_usd: f64,
    /// When this invoice was created (Unix epoch seconds).
    pub issued_at: u64,
    /// Payment due date (Unix epoch seconds).
    pub due_at: u64,
    /// Current lifecycle status.
    pub status: InvoiceStatus,
}

// ── InvoiceGenerator ─────────────────────────────────────────────────────────

/// Factory for creating and formatting invoices.
pub struct InvoiceGenerator;

impl InvoiceGenerator {
    /// Generate an invoice for `customer_id` from `usage_records`.
    ///
    /// Only records whose `timestamp` falls in `[period_start, period_end]` and
    /// whose `customer_id` matches are included.
    ///
    /// `prices` maps `model_id` to USD-per-1k-tokens (combined input+output).
    /// Records for unknown models are grouped under the model_id with `unit_cost_usd = 0.0`.
    pub fn generate(
        customer_id: &str,
        usage_records: &[UsageRecord],
        prices: &HashMap<String, f64>,
        period_start: u64,
        period_end: u64,
        tax_rate: f64,
    ) -> Invoice {
        // Filter records for this customer and period.
        let relevant: Vec<&UsageRecord> = usage_records
            .iter()
            .filter(|r| {
                r.customer_id == customer_id
                    && r.timestamp >= period_start
                    && r.timestamp <= period_end
            })
            .collect();

        // Aggregate by model_id.
        let mut by_model: HashMap<&str, (u64, u64)> = HashMap::new();
        for r in &relevant {
            let entry = by_model.entry(r.model_id.as_str()).or_insert((0, 0));
            entry.0 += r.input_tokens;
            entry.1 += r.output_tokens;
        }

        let period_label = format!(
            "{} – {}",
            epoch_to_date(period_start),
            epoch_to_date(period_end)
        );

        let mut line_items: Vec<InvoiceLineItem> = by_model
            .iter()
            .map(|(model_id, (inp, out))| {
                let unit_cost = prices.get(*model_id).copied().unwrap_or(0.0);
                let total_tokens = inp + out;
                let total_usd = (total_tokens as f64 / 1000.0) * unit_cost;
                InvoiceLineItem {
                    description: format!("{} usage — {}", model_id, period_label),
                    model_id: model_id.to_string(),
                    input_tokens: *inp,
                    output_tokens: *out,
                    unit_cost_usd: unit_cost,
                    total_usd,
                    period: period_label.clone(),
                }
            })
            .collect();

        // Sort by model_id for deterministic output.
        line_items.sort_by(|a, b| a.model_id.cmp(&b.model_id));

        let subtotal_usd: f64 = line_items.iter().map(|li| li.total_usd).sum();
        let tax_usd = subtotal_usd * tax_rate;
        let total_usd = subtotal_usd + tax_usd;

        // Invoice ID: customer + period start for uniqueness.
        let invoice_id = format!("INV-{}-{}", customer_id, period_start);

        Invoice {
            invoice_id,
            customer_id: customer_id.to_string(),
            period_start,
            period_end,
            line_items,
            subtotal_usd,
            tax_rate,
            tax_usd,
            total_usd,
            issued_at: period_end + 1,
            due_at: period_end + 30 * 86_400, // net-30
            status: InvoiceStatus::Draft,
        }
    }

    /// Render the invoice as a human-readable text document.
    pub fn to_text(invoice: &Invoice) -> String {
        let mut out = String::new();
        out.push_str("=======================================================\n");
        out.push_str(&format!("INVOICE  {}\n", invoice.invoice_id));
        out.push_str("=======================================================\n");
        out.push_str(&format!("Customer : {}\n", invoice.customer_id));
        out.push_str(&format!(
            "Period   : {} – {}\n",
            epoch_to_date(invoice.period_start),
            epoch_to_date(invoice.period_end)
        ));
        out.push_str(&format!("Issued   : {}\n", epoch_to_date(invoice.issued_at)));
        out.push_str(&format!("Due      : {}\n", epoch_to_date(invoice.due_at)));
        out.push_str(&format!("Status   : {}\n", invoice.status));
        out.push_str("-------------------------------------------------------\n");
        out.push_str(&format!(
            "{:<30} {:>10} {:>10} {:>12}\n",
            "Description", "Tokens", "Unit/1k", "Total"
        ));
        out.push_str("-------------------------------------------------------\n");
        for li in &invoice.line_items {
            let total_tok = li.input_tokens + li.output_tokens;
            out.push_str(&format!(
                "{:<30} {:>10} {:>10.5} {:>12.6}\n",
                li.description, total_tok, li.unit_cost_usd, li.total_usd
            ));
        }
        out.push_str("-------------------------------------------------------\n");
        out.push_str(&format!("{:>54.6} (subtotal)\n", invoice.subtotal_usd));
        out.push_str(&format!(
            "{:>54.6} (tax {:.0}%)\n",
            invoice.tax_usd,
            invoice.tax_rate * 100.0
        ));
        out.push_str("=======================================================\n");
        out.push_str(&format!("TOTAL DUE: ${:.6}\n", invoice.total_usd));
        out.push_str("=======================================================\n");
        out
    }

    /// Render line items as CSV.
    pub fn to_csv(invoice: &Invoice) -> String {
        let mut out = String::new();
        out.push_str(
            "invoice_id,customer_id,model_id,input_tokens,output_tokens,unit_cost_usd,total_usd,period\n",
        );
        for li in &invoice.line_items {
            out.push_str(&format!(
                "{},{},{},{},{},{:.6},{:.6},{}\n",
                invoice.invoice_id,
                invoice.customer_id,
                li.model_id,
                li.input_tokens,
                li.output_tokens,
                li.unit_cost_usd,
                li.total_usd,
                li.period,
            ));
        }
        out
    }

    /// Apply a percentage discount, adding a negative line item and recomputing totals.
    ///
    /// `discount_pct` is 0.0 – 100.0 (e.g. `10.0` for 10 %).
    pub fn apply_discount(invoice: &mut Invoice, discount_pct: f64) {
        let discount_amount = -(invoice.subtotal_usd * discount_pct / 100.0);
        invoice.line_items.push(InvoiceLineItem {
            description: format!("Discount ({:.0}%)", discount_pct),
            model_id: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            unit_cost_usd: 0.0,
            total_usd: discount_amount,
            period: String::new(),
        });
        invoice.subtotal_usd += discount_amount;
        invoice.tax_usd = invoice.subtotal_usd * invoice.tax_rate;
        invoice.total_usd = invoice.subtotal_usd + invoice.tax_usd;
    }

    /// Transition the invoice status to [`InvoiceStatus::Paid`].
    pub fn mark_paid(invoice: &mut Invoice, _paid_at: u64) {
        invoice.status = InvoiceStatus::Paid;
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Very lightweight epoch → "YYYY-MM-DD" formatter (no chrono dependency).
fn epoch_to_date(epoch_secs: u64) -> String {
    // Use a simplified calculation good for dates from ~1970 to 2100.
    let days_since_epoch = epoch_secs / 86_400;
    let (y, m, d) = days_to_ymd(days_since_epoch);
    format!("{:04}-{:02}-{:02}", y, m, d)
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Gregorian calendar computation.
    let mut year = 1970u64;
    loop {
        let leap = is_leap(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut month = 1u64;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

// ── New billing types (Round 29) ──────────────────────────────────────────────

use std::sync::Mutex;

/// A single line item on a new-style invoice.
#[derive(Debug, Clone)]
pub struct LineItem {
    /// Human-readable description.
    pub description: String,
    /// Quantity (e.g. number of 1k-token blocks).
    pub quantity: f64,
    /// Price per unit in USD.
    pub unit_price_usd: f64,
    /// Model this line item is billed for.
    pub model: String,
    /// Billing period start (Unix epoch seconds).
    pub period_start: u64,
    /// Billing period end (Unix epoch seconds).
    pub period_end: u64,
}

impl LineItem {
    /// Compute the total cost for this line item.
    pub fn total(&self) -> f64 {
        self.quantity * self.unit_price_usd
    }
}

/// A tax rate definition.
#[derive(Debug, Clone)]
pub struct TaxRate {
    /// Tax name (e.g. "VAT").
    pub name: String,
    /// Tax percentage (e.g. 20.0 for 20%).
    pub rate_pct: f64,
    /// Jurisdiction (e.g. "EU", "US-CA").
    pub jurisdiction: String,
}

/// Status of a new-style invoice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NewInvoiceStatus {
    /// Not yet sent.
    Draft,
    /// Sent to customer.
    Issued,
    /// Payment received.
    Paid,
    /// Payment overdue.
    Overdue,
    /// Invoice cancelled.
    Voided,
}

impl std::fmt::Display for NewInvoiceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Draft => "Draft",
            Self::Issued => "Issued",
            Self::Paid => "Paid",
            Self::Overdue => "Overdue",
            Self::Voided => "Voided",
        };
        write!(f, "{}", s)
    }
}

/// A rich invoice with builder-pattern support.
#[derive(Debug, Clone)]
pub struct NewInvoice {
    /// Unique invoice identifier.
    pub invoice_id: String,
    /// Customer identifier.
    pub customer_id: String,
    /// Customer display name.
    pub customer_name: String,
    /// When the invoice was issued (Unix epoch seconds).
    pub issued_at_unix: u64,
    /// Payment due date (Unix epoch seconds).
    pub due_at_unix: u64,
    /// Line items.
    pub line_items: Vec<LineItem>,
    /// Sum of line item totals.
    pub subtotal: f64,
    /// Tax amount.
    pub tax_amount: f64,
    /// Total amount due.
    pub total_usd: f64,
    /// Currency code (e.g. "USD").
    pub currency: String,
    /// Current status.
    pub status: NewInvoiceStatus,
    /// Free-form notes.
    pub notes: String,
}

/// Builder for [`NewInvoice`].
pub struct InvoiceBuilder {
    customer_id: String,
    customer_name: String,
    line_items: Vec<LineItem>,
    tax: Option<TaxRate>,
    discount_pct: f64,
    due_days: u32,
    notes: String,
}

impl InvoiceBuilder {
    /// Start building an invoice for a customer.
    pub fn new(customer_id: &str, customer_name: &str) -> Self {
        Self {
            customer_id: customer_id.to_string(),
            customer_name: customer_name.to_string(),
            line_items: Vec::new(),
            tax: None,
            discount_pct: 0.0,
            due_days: 30,
            notes: String::new(),
        }
    }

    /// Add a line item.
    pub fn add_line_item(&mut self, item: LineItem) -> &mut Self {
        self.line_items.push(item);
        self
    }

    /// Apply a tax rate.
    pub fn apply_tax(&mut self, tax: TaxRate) -> &mut Self {
        self.tax = Some(tax);
        self
    }

    /// Apply a percentage discount (0.0–100.0).
    pub fn apply_discount(&mut self, discount_pct: f64) -> &mut Self {
        self.discount_pct = discount_pct;
        self
    }

    /// Set the payment due window in days from issuance.
    pub fn set_due_days(&mut self, days: u32) -> &mut Self {
        self.due_days = days;
        self
    }

    /// Finalise and produce a [`NewInvoice`].
    pub fn build(self) -> NewInvoice {
        let subtotal_raw: f64 = self.line_items.iter().map(|li| li.total()).sum();
        let discount_amount = subtotal_raw * self.discount_pct / 100.0;
        let subtotal = subtotal_raw - discount_amount;
        let tax_amount = self
            .tax
            .as_ref()
            .map(|t| subtotal * t.rate_pct / 100.0)
            .unwrap_or(0.0);
        let total_usd = subtotal + tax_amount;
        // Use a simple counter as a proxy for "now".
        let issued_at_unix = 0u64;
        let due_at_unix = issued_at_unix + self.due_days as u64 * 86_400;
        let invoice_id = format!("INV-{}-{}", self.customer_id, issued_at_unix);
        NewInvoice {
            invoice_id,
            customer_id: self.customer_id,
            customer_name: self.customer_name,
            issued_at_unix,
            due_at_unix,
            line_items: self.line_items,
            subtotal,
            tax_amount,
            total_usd,
            currency: "USD".to_string(),
            status: NewInvoiceStatus::Draft,
            notes: self.notes,
        }
    }
}

/// Summary of revenue across a time range.
#[derive(Debug, Clone)]
pub struct RevenueSummary {
    /// Total revenue in USD.
    pub total_revenue: f64,
    /// Number of invoices in the range.
    pub invoice_count: usize,
    /// Number of paid invoices.
    pub paid_count: usize,
    /// Number of overdue invoices.
    pub overdue_count: usize,
    /// Average invoice value in USD.
    pub avg_invoice_usd: f64,
    /// Top customers by revenue `(customer_id, total_usd)`.
    pub top_customers: Vec<(String, f64)>,
}

/// Instance-based invoice generator with persistent storage.
pub struct NewInvoiceGenerator {
    store: Mutex<Vec<NewInvoice>>,
}

impl Default for NewInvoiceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl NewInvoiceGenerator {
    /// Create a new generator.
    pub fn new() -> Self {
        Self {
            store: Mutex::new(Vec::new()),
        }
    }

    /// Generate an invoice from raw usage records `(model, cost, tokens)`.
    pub fn generate_from_usage(
        &self,
        customer_id: &str,
        customer_name: &str,
        usage_records: &[(String, f64, usize)],
        period_days: u32,
    ) -> NewInvoice {
        let mut builder = InvoiceBuilder::new(customer_id, customer_name);
        let period_end = period_days as u64 * 86_400;
        for (model, cost, tokens) in usage_records {
            let quantity = *tokens as f64 / 1000.0;
            let unit_price = if quantity > 0.0 { cost / quantity } else { 0.0 };
            builder.add_line_item(LineItem {
                description: format!("{} usage ({} tokens)", model, tokens),
                quantity,
                unit_price_usd: unit_price,
                model: model.clone(),
                period_start: 0,
                period_end,
            });
        }
        builder.build()
    }

    /// Render a plain-text invoice.
    pub fn render_text(&self, invoice: &NewInvoice) -> String {
        let mut out = String::new();
        out.push_str("=======================================================\n");
        out.push_str(&format!("INVOICE  {}\n", invoice.invoice_id));
        out.push_str("=======================================================\n");
        out.push_str(&format!("Customer : {} ({})\n", invoice.customer_name, invoice.customer_id));
        out.push_str(&format!("Currency : {}\n", invoice.currency));
        out.push_str(&format!("Status   : {}\n", invoice.status));
        if !invoice.notes.is_empty() {
            out.push_str(&format!("Notes    : {}\n", invoice.notes));
        }
        out.push_str("-------------------------------------------------------\n");
        out.push_str(&format!("{:<30} {:>10} {:>12}\n", "Description", "Qty", "Total"));
        out.push_str("-------------------------------------------------------\n");
        for li in &invoice.line_items {
            out.push_str(&format!(
                "{:<30} {:>10.2} {:>12.6}\n",
                li.description,
                li.quantity,
                li.total()
            ));
        }
        out.push_str("-------------------------------------------------------\n");
        out.push_str(&format!("{:>54.6} (subtotal)\n", invoice.subtotal));
        out.push_str(&format!("{:>54.6} (tax)\n", invoice.tax_amount));
        out.push_str("=======================================================\n");
        out.push_str(&format!("TOTAL DUE: ${:.6}\n", invoice.total_usd));
        out.push_str("=======================================================\n");
        out
    }

    /// Render an invoice's line items as CSV.
    pub fn render_csv(&self, invoice: &NewInvoice) -> String {
        let mut out = String::new();
        out.push_str("invoice_id,customer_id,model,description,quantity,unit_price_usd,total_usd\n");
        for li in &invoice.line_items {
            out.push_str(&format!(
                "{},{},{},{},{:.4},{:.6},{:.6}\n",
                invoice.invoice_id,
                invoice.customer_id,
                li.model,
                li.description,
                li.quantity,
                li.unit_price_usd,
                li.total(),
            ));
        }
        out
    }

    /// Persist an invoice in the internal store.
    pub fn store(&self, invoice: NewInvoice) {
        if let Ok(mut guard) = self.store.lock() {
            guard.push(invoice);
        }
    }

    /// Retrieve an invoice by ID.
    pub fn get_invoice(&self, invoice_id: &str) -> Option<NewInvoice> {
        self.store
            .lock()
            .ok()?
            .iter()
            .find(|inv| inv.invoice_id == invoice_id)
            .cloned()
    }

    /// Return all invoices that are overdue (status == Overdue and due_at_unix < current_time).
    pub fn overdue_invoices(&self, current_time: u64) -> Vec<NewInvoice> {
        match self.store.lock() {
            Ok(guard) => guard
                .iter()
                .filter(|inv| {
                    inv.status == NewInvoiceStatus::Overdue && inv.due_at_unix < current_time
                })
                .cloned()
                .collect(),
            Err(_) => vec![],
        }
    }

    /// Summarise revenue for invoices issued within `[from, to]` Unix epoch seconds.
    pub fn revenue_summary(&self, from: u64, to: u64) -> RevenueSummary {
        let guard = match self.store.lock() {
            Ok(g) => g,
            Err(_) => {
                return RevenueSummary {
                    total_revenue: 0.0,
                    invoice_count: 0,
                    paid_count: 0,
                    overdue_count: 0,
                    avg_invoice_usd: 0.0,
                    top_customers: vec![],
                }
            }
        };
        let in_range: Vec<&NewInvoice> = guard
            .iter()
            .filter(|inv| inv.issued_at_unix >= from && inv.issued_at_unix <= to)
            .collect();

        let invoice_count = in_range.len();
        let total_revenue: f64 = in_range.iter().map(|inv| inv.total_usd).sum();
        let paid_count = in_range
            .iter()
            .filter(|inv| inv.status == NewInvoiceStatus::Paid)
            .count();
        let overdue_count = in_range
            .iter()
            .filter(|inv| inv.status == NewInvoiceStatus::Overdue)
            .count();
        let avg_invoice_usd = if invoice_count > 0 {
            total_revenue / invoice_count as f64
        } else {
            0.0
        };

        // Aggregate by customer.
        let mut customer_totals: HashMap<String, f64> = HashMap::new();
        for inv in &in_range {
            *customer_totals.entry(inv.customer_id.clone()).or_insert(0.0) += inv.total_usd;
        }
        let mut top_customers: Vec<(String, f64)> = customer_totals.into_iter().collect();
        top_customers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_customers.truncate(10);

        RevenueSummary {
            total_revenue,
            invoice_count,
            paid_count,
            overdue_count,
            avg_invoice_usd,
            top_customers,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prices() -> HashMap<String, f64> {
        let mut p = HashMap::new();
        p.insert("gpt-4o".to_string(), 0.005);
        p.insert("claude-3-5-sonnet".to_string(), 0.003);
        p
    }

    fn make_records(customer_id: &str) -> Vec<UsageRecord> {
        vec![
            UsageRecord {
                model_id: "gpt-4o".to_string(),
                input_tokens: 2000,
                output_tokens: 1000,
                timestamp: 1_700_050_000,
                customer_id: customer_id.to_string(),
            },
            UsageRecord {
                model_id: "claude-3-5-sonnet".to_string(),
                input_tokens: 500,
                output_tokens: 500,
                timestamp: 1_700_060_000,
                customer_id: customer_id.to_string(),
            },
        ]
    }

    #[test]
    fn test_generate_basic() {
        let records = make_records("acme");
        let invoice = InvoiceGenerator::generate(
            "acme",
            &records,
            &make_prices(),
            1_700_000_000,
            1_700_086_400,
            0.0,
        );
        assert_eq!(invoice.customer_id, "acme");
        assert_eq!(invoice.line_items.len(), 2);
        // gpt-4o: (2000+1000)/1000 * 0.005 = 0.015
        let gpt_line = invoice.line_items.iter().find(|l| l.model_id == "gpt-4o").unwrap();
        assert!((gpt_line.total_usd - 0.015).abs() < 1e-9);
    }

    #[test]
    fn test_tax_calculation() {
        let records = make_records("acme");
        let invoice = InvoiceGenerator::generate(
            "acme",
            &records,
            &make_prices(),
            1_700_000_000,
            1_700_086_400,
            0.10,
        );
        assert!((invoice.tax_usd - invoice.subtotal_usd * 0.10).abs() < 1e-9);
        assert!((invoice.total_usd - invoice.subtotal_usd - invoice.tax_usd).abs() < 1e-9);
    }

    #[test]
    fn test_filter_by_customer() {
        let mut records = make_records("acme");
        records.push(UsageRecord {
            model_id: "gpt-4o".to_string(),
            input_tokens: 999,
            output_tokens: 999,
            timestamp: 1_700_050_001,
            customer_id: "other".to_string(),
        });
        let invoice = InvoiceGenerator::generate(
            "acme",
            &records,
            &make_prices(),
            1_700_000_000,
            1_700_086_400,
            0.0,
        );
        // Only acme records should appear.
        assert_eq!(invoice.line_items.len(), 2);
    }

    #[test]
    fn test_filter_by_period() {
        let mut records = make_records("acme");
        // Record outside the period
        records.push(UsageRecord {
            model_id: "gpt-4o".to_string(),
            input_tokens: 5000,
            output_tokens: 5000,
            timestamp: 1_800_000_000, // far future
            customer_id: "acme".to_string(),
        });
        let invoice = InvoiceGenerator::generate(
            "acme",
            &records,
            &make_prices(),
            1_700_000_000,
            1_700_086_400,
            0.0,
        );
        // gpt-4o total should reflect only the in-period record
        let gpt = invoice.line_items.iter().find(|l| l.model_id == "gpt-4o").unwrap();
        assert_eq!(gpt.input_tokens, 2000);
    }

    #[test]
    fn test_to_text_contains_invoice_id() {
        let records = make_records("acme");
        let invoice = InvoiceGenerator::generate(
            "acme",
            &records,
            &make_prices(),
            1_700_000_000,
            1_700_086_400,
            0.0,
        );
        let text = InvoiceGenerator::to_text(&invoice);
        assert!(text.contains(&invoice.invoice_id));
        assert!(text.contains("acme"));
    }

    #[test]
    fn test_to_csv_header() {
        let records = make_records("acme");
        let invoice = InvoiceGenerator::generate(
            "acme",
            &records,
            &make_prices(),
            1_700_000_000,
            1_700_086_400,
            0.0,
        );
        let csv = InvoiceGenerator::to_csv(&invoice);
        assert!(csv.starts_with("invoice_id,customer_id,model_id"));
        // Two data rows
        assert_eq!(csv.lines().count(), 3); // header + 2 rows
    }

    #[test]
    fn test_apply_discount() {
        let records = make_records("acme");
        let mut invoice = InvoiceGenerator::generate(
            "acme",
            &records,
            &make_prices(),
            1_700_000_000,
            1_700_086_400,
            0.0,
        );
        let original_subtotal = invoice.subtotal_usd;
        InvoiceGenerator::apply_discount(&mut invoice, 10.0);
        // subtotal should be 90 % of original
        assert!((invoice.subtotal_usd - original_subtotal * 0.9).abs() < 1e-9);
        // A discount line item should have been added
        assert!(invoice.line_items.iter().any(|l| l.description.contains("Discount")));
    }

    #[test]
    fn test_mark_paid() {
        let records = make_records("acme");
        let mut invoice = InvoiceGenerator::generate(
            "acme",
            &records,
            &make_prices(),
            1_700_000_000,
            1_700_086_400,
            0.0,
        );
        assert_ne!(invoice.status, InvoiceStatus::Paid);
        InvoiceGenerator::mark_paid(&mut invoice, 1_700_090_000);
        assert_eq!(invoice.status, InvoiceStatus::Paid);
    }

    #[test]
    fn test_status_display() {
        assert_eq!(InvoiceStatus::Draft.to_string(), "Draft");
        assert_eq!(InvoiceStatus::Issued.to_string(), "Issued");
        assert_eq!(InvoiceStatus::Paid.to_string(), "Paid");
        assert_eq!(InvoiceStatus::Overdue.to_string(), "Overdue");
        assert_eq!(InvoiceStatus::Void.to_string(), "Void");
    }
}
