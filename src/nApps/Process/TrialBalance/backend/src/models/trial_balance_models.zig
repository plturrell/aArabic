//! ============================================================================
//! Trial Balance Data Models
//! SAP S/4HANA ACDOCA-compliant data structures with ODPS support
//! ============================================================================
//!
//! [CODE:file=trial_balance_models.zig]
//! [CODE:module=models]
//! [CODE:language=zig]
//!
//! [ODPS:product=acdoca-journal-entries,trial-balance-aggregated,variances,exchange-rates,account-master]
//!
//! This module defines the core data structures aligned with ODPS field mappings:
//! - JournalEntry → acdoca-journal-entries.odps.yaml
//! - TrialBalanceEntry → trial-balance-aggregated.odps.yaml
//! - VarianceEntry → variances.odps.yaml
//! - ExchangeRate → exchange-rates.odps.yaml
//! - AccountMaster → account-master.odps.yaml
//!
//! [RELATION:used_by=CODE:balance_engine.zig]
//! [RELATION:used_by=CODE:fx_converter.zig]
//! [RELATION:used_by=CODE:acdoca_table.zig]
//! [RELATION:used_by=CODE:trial_balance.zig]
//!
//! Field mappings documented in ODPS_ZIG_FIELD_MAPPING.md

const std = @import("std");
const Allocator = std.mem.Allocator;

/// SAP S/4HANA ACDOCA Universal Journal Entry
/// Full field set for complete data lineage and IFRS compliance
pub const JournalEntry = struct {
    // Primary Keys
    rldnr: []const u8, // Ledger (e.g., "0L" = Leading ledger)
    rbukrs: []const u8, // Company code (e.g., "HKG")
    gjahr: u16, // Fiscal year (e.g., 2025)
    belnr: []const u8, // Document number (10 chars)
    docln: u32, // Line item number within document

    // Account Information
    racct: []const u8, // G/L Account (required)
    rcntr: ?[]const u8, // Cost center (optional)
    prctr: ?[]const u8, // Profit center (optional)

    // Amount Fields (in different currencies)
    hsl: f64, // Amount in local currency (HKD)
    ksl: f64, // Amount in group currency (USD)
    wsl: f64, // Amount in transaction currency

    // Currency & Exchange Rate
    rhcur: []const u8, // Local currency code (e.g., "HKD")
    rkcur: []const u8, // Group currency code (e.g., "USD")
    rwcur: []const u8, // Transaction currency code
    ukurs: f64, // Exchange rate (transaction to local)

    // Debit/Credit Indicator
    drcrk: []const u8, // "S"=Debit (Soll), "H"=Credit (Haben)

    // Posting Period Information
    poper: u8, // Posting period (1-12)
    budat: []const u8, // Posting date (YYYY-MM-DD)
    bldat: []const u8, // Document date (YYYY-MM-DD)

    // Additional Dimensions (IFRS/Segment Reporting)
    segment: ?[]const u8, // Segment for segment reporting
    scntr: ?[]const u8, // Sender cost center
    pprctr: ?[]const u8, // Partner profit center

    // Document Information
    bktxt: ?[]const u8, // Document header text
    sgtxt: ?[]const u8, // Line item text
    xblnr: ?[]const u8, // Reference document number

    // Data Quality & Lineage (Open Data Product Spec)
    data_hash: [64]u8, // SHA-256 hash (hex string) for lineage tracking
    source_system: []const u8, // "SAP_S4HANA"
    extraction_ts: i64, // Unix timestamp of extraction
    data_quality: []const u8, // "verified", "unverified", "suspect"

    pub fn init(_: Allocator) JournalEntry {
        return .{
            .rldnr = "",
            .rbukrs = "",
            .gjahr = 2025,
            .belnr = "",
            .docln = 0,
            .racct = "",
            .rcntr = null,
            .prctr = null,
            .hsl = 0.0,
            .ksl = 0.0,
            .wsl = 0.0,
            .rhcur = "HKD",
            .rkcur = "USD",
            .rwcur = "HKD",
            .ukurs = 1.0,
            .drcrk = "S",
            .poper = 11,
            .budat = "",
            .bldat = "",
            .segment = null,
            .scntr = null,
            .pprctr = null,
            .bktxt = null,
            .sgtxt = null,
            .xblnr = null,
            .data_hash = [_]u8{0} ** 64,
            .source_system = "SAP_S4HANA",
            .extraction_ts = std.time.timestamp(),
            .data_quality = "unverified",
        };
    }

    /// Calculate SHA-256 hash for data lineage tracking
    pub fn calculateHash(self: *const JournalEntry, _: Allocator) ![64]u8 {
        var hash_buffer: [32]u8 = undefined;
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});

        // Hash key fields to create unique identifier
        hasher.update(self.rldnr);
        hasher.update(self.rbukrs);
        hasher.update(&std.mem.toBytes(self.gjahr));
        hasher.update(self.belnr);
        hasher.update(&std.mem.toBytes(self.docln));
        hasher.update(self.racct);
        hasher.update(&std.mem.toBytes(self.hsl));
        hasher.update(self.drcrk);

        hasher.final(&hash_buffer);

        // Convert to hex string
        var hex_hash: [64]u8 = undefined;
        _ = try std.fmt.bufPrint(&hex_hash, "{x}", .{std.fmt.fmtSliceHexLower(&hash_buffer)});

        return hex_hash;
    }

    /// Validate required fields are present
    pub fn validate(self: *const JournalEntry) bool {
        return self.racct.len > 0 and
            self.rbukrs.len > 0 and
            (std.mem.eql(u8, self.drcrk, "S") or std.mem.eql(u8, self.drcrk, "H")) and
            self.poper >= 1 and self.poper <= 12;
    }
};

/// Exchange Rate for currency conversion
pub const ExchangeRate = struct {
    // Rate identification
    rate_type: []const u8, // "M" = Standard rate, "P" = Average rate
    from_curr: []const u8, // Source currency (e.g., "HKD")
    to_curr: []const u8, // Target currency (e.g., "USD")
    valid_from: []const u8, // Valid from date (YYYY-MM-DD)

    // Rate data
    exchange_rate: f64, // Exchange rate value
    ratio_from: f64, // From ratio (usually 1.0)
    ratio_to: f64, // To ratio (e.g., 100.0 for JPY)

    // Metadata
    source: []const u8, // "ECB", "FED", "MANUAL", "SAP"
    last_updated: i64, // Unix timestamp

    pub fn init() ExchangeRate {
        return .{
            .rate_type = "M",
            .from_curr = "",
            .to_curr = "",
            .valid_from = "",
            .exchange_rate = 1.0,
            .ratio_from = 1.0,
            .ratio_to = 1.0,
            .source = "SAP",
            .last_updated = std.time.timestamp(),
        };
    }

    /// Convert amount using this exchange rate
    pub fn convert(self: *const ExchangeRate, amount: f64) f64 {
        return (amount / self.ratio_from) * self.exchange_rate * self.ratio_to;
    }
};

/// Main trial balance entry representing a single account line item
pub const TrialBalanceEntry = struct {
    // Identification fields
    account_code: []const u8, // Account code (e.g., "1000", "2000")
    account_name: []const u8, // Account name (e.g., "Cash", "Accounts Payable")

    // Classification fields
    account_type: []const u8, // Asset, Liability, Equity, Revenue, Expense
    ifrs_category: []const u8, // IFRS classification
    gcoa_code: ?[]const u8, // Group Chart of Accounts code (optional)

    // Financial data
    opening_balance: f64, // Opening balance
    debit_amount: f64, // Total debits for period
    credit_amount: f64, // Total credits for period
    closing_balance: f64, // Calculated closing balance

    // Period information
    business_unit: []const u8, // Business unit code (e.g., "HKG", "SGP")
    fiscal_period: []const u8, // Period (e.g., "2025-11", "2025-12")
    fiscal_year: u16, // Fiscal year
    period_month: u8, // Month number (1-12)

    // Data quality and metadata
    currency_code: []const u8, // Currency (e.g., "HKD", "USD")
    data_quality: []const u8, // verified, unverified, suspect
    last_updated: i64, // Unix timestamp
    source_system: []const u8, // Source system (e.g., "SAP S/4HANA")

    // Calculated/derived fields
    net_movement: f64, // Debit - Credit
    balance_type: []const u8, // "DR" or "CR"

    pub fn init(
        account_code: []const u8,
        account_name: []const u8,
        account_type: []const u8,
        business_unit: []const u8,
        fiscal_period: []const u8,
    ) TrialBalanceEntry {
        return .{
            .account_code = account_code,
            .account_name = account_name,
            .account_type = account_type,
            .ifrs_category = "",
            .gcoa_code = null,
            .opening_balance = 0.0,
            .debit_amount = 0.0,
            .credit_amount = 0.0,
            .closing_balance = 0.0,
            .business_unit = business_unit,
            .fiscal_period = fiscal_period,
            .fiscal_year = 2025,
            .period_month = 11,
            .currency_code = "HKD",
            .data_quality = "unverified",
            .last_updated = std.time.timestamp(),
            .source_system = "SAP S/4HANA",
            .net_movement = 0.0,
            .balance_type = "DR",
        };
    }

    /// Calculate closing balance: Opening + Debits - Credits
    pub fn calculateClosingBalance(self: *TrialBalanceEntry) void {
        self.closing_balance = self.opening_balance + self.debit_amount - self.credit_amount;
        self.net_movement = self.debit_amount - self.credit_amount;

        // Determine balance type
        if (self.closing_balance >= 0) {
            self.balance_type = "DR";
        } else {
            self.balance_type = "CR";
        }
    }
};

/// Variance entry representing period-over-period changes
pub const VarianceEntry = struct {
    // Identification
    account_code: []const u8,
    account_name: []const u8,
    account_type: []const u8,

    // Comparison data
    current_period: []const u8, // e.g., "2025-11"
    previous_period: []const u8, // e.g., "2025-10"
    current_amount: f64, // Current period balance
    previous_amount: f64, // Previous period balance

    // Variance calculations
    variance_amount: f64, // Absolute variance
    variance_percent: f64, // Percentage variance
    is_significant: bool, // Exceeds threshold (10% or $3M/$100M)

    // Context
    business_unit: []const u8,
    currency_code: []const u8,

    // AI commentary
    commentary: ?[]const u8, // AI-generated explanation
    commentary_generated_at: ?i64, // Timestamp of commentary generation
    requires_explanation: bool, // Flags if variance needs attention

    pub fn init(
        account_code: []const u8,
        account_name: []const u8,
        current_period: []const u8,
        previous_period: []const u8,
    ) VarianceEntry {
        return .{
            .account_code = account_code,
            .account_name = account_name,
            .account_type = "",
            .current_period = current_period,
            .previous_period = previous_period,
            .current_amount = 0.0,
            .previous_amount = 0.0,
            .variance_amount = 0.0,
            .variance_percent = 0.0,
            .is_significant = false,
            .business_unit = "HKG",
            .currency_code = "HKD",
            .commentary = null,
            .commentary_generated_at = null,
            .requires_explanation = false,
        };
    }

    /// Calculate variance and determine significance
    pub fn calculateVariance(self: *VarianceEntry, bs_threshold_amount: f64, pl_threshold_amount: f64) void {
        self.variance_amount = self.current_amount - self.previous_amount;

        if (self.previous_amount != 0.0) {
            self.variance_percent = (self.variance_amount / @abs(self.previous_amount)) * 100.0;
        } else {
            self.variance_percent = if (self.current_amount != 0.0) 100.0 else 0.0;
        }

        // Determine significance based on thresholds
        const abs_variance = @abs(self.variance_amount);
        const abs_percent = @abs(self.variance_percent);

        // BS threshold: $100M or 10%
        // PL threshold: $3M or 10%
        const amount_threshold = if (std.mem.eql(u8, self.account_type, "Asset") or
            std.mem.eql(u8, self.account_type, "Liability") or
            std.mem.eql(u8, self.account_type, "Equity"))
            bs_threshold_amount
        else
            pl_threshold_amount;

        self.is_significant = (abs_variance >= amount_threshold) or (abs_percent >= 10.0);
        self.requires_explanation = self.is_significant;
    }
};

/// Account master data for names and classifications
pub const AccountMaster = struct {
    account_code: []const u8,
    account_name: []const u8,
    account_type: []const u8, // Asset, Liability, Equity, Revenue, Expense
    ifrs_category: []const u8,
    parent_account: ?[]const u8, // For hierarchical accounts
    is_active: bool,

    pub fn init(account_code: []const u8, account_name: []const u8, account_type: []const u8) AccountMaster {
        return .{
            .account_code = account_code,
            .account_name = account_name,
            .account_type = account_type,
            .ifrs_category = "",
            .parent_account = null,
            .is_active = true,
        };
    }
};

/// Checklist item for workflow tracking
pub const ChecklistItem = struct {
    id: []const u8, // Unique identifier
    stage_id: []const u8, // Workflow stage ID (S01-S13)
    title: []const u8, // Item title
    description: []const u8, // Detailed description
    status: []const u8, // Pending, InProgress, Complete
    assigned_to: ?[]const u8, // Assigned user/role
    due_date: ?i64, // Unix timestamp
    completed_at: ?i64, // Completion timestamp
    notes: ?[]const u8, // Additional notes

    pub fn init(id: []const u8, stage_id: []const u8, title: []const u8) ChecklistItem {
        return .{
            .id = id,
            .stage_id = stage_id,
            .title = title,
            .description = "",
            .status = "Pending",
            .assigned_to = null,
            .due_date = null,
            .completed_at = null,
            .notes = null,
        };
    }

    pub fn markComplete(self: *ChecklistItem) void {
        self.status = "Complete";
        self.completed_at = std.time.timestamp();
    }
};

/// Dataset metadata for lineage and quality tracking (Open Data Product Spec)
pub const DatasetMetadata = struct {
    dataset_id: []const u8, // Unique dataset identifier
    dataset_name: []const u8, // Human-readable name
    business_unit: []const u8,
    fiscal_period: []const u8,

    // Source information
    source_system: []const u8, // e.g., "SAP S/4HANA"
    extraction_timestamp: i64, // When data was extracted
    data_hash: []const u8, // SHA-256 hash for verification

    // Quality metrics
    total_records: usize,
    valid_records: usize,
    invalid_records: usize,
    data_quality_score: f64, // 0.0 to 100.0

    // Lineage
    parent_dataset_id: ?[]const u8, // Previous transformation
    transformation_type: []const u8, // extract, validate, aggregate, etc.
    transformation_timestamp: i64,

    // Status
    status: []const u8, // draft, validated, published, archived

    pub fn init(dataset_id: []const u8, dataset_name: []const u8, business_unit: []const u8) DatasetMetadata {
        return .{
            .dataset_id = dataset_id,
            .dataset_name = dataset_name,
            .business_unit = business_unit,
            .fiscal_period = "2025-11",
            .source_system = "SAP S/4HANA",
            .extraction_timestamp = std.time.timestamp(),
            .data_hash = "",
            .total_records = 0,
            .valid_records = 0,
            .invalid_records = 0,
            .data_quality_score = 0.0,
            .parent_dataset_id = null,
            .transformation_type = "extract",
            .transformation_timestamp = std.time.timestamp(),
            .status = "draft",
        };
    }

    pub fn calculateQualityScore(self: *DatasetMetadata) void {
        if (self.total_records > 0) {
            self.data_quality_score = (@as(f64, @floatFromInt(self.valid_records)) /
                @as(f64, @floatFromInt(self.total_records))) * 100.0;
        } else {
            self.data_quality_score = 0.0;
        }
    }
};