//! ============================================================================
//! Data Quality Framework
//! IFRS-compliant validation rules and quality scoring for Trial Balance data
//! ============================================================================
//!
//! [CODE:file=data_quality.zig]
//! [CODE:module=models]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-aggregated,acdoca-journal-entries,exchange-rates]
//! [ODPS:rules=TB001,TB002,TB003,FX005,FX006]
//!
//! This module implements IFRS validation rules:
//! - R001: RACCT mandatory → supports TB003
//! - R002: DRCRK validity → supports TB001/TB002
//! - R003: POPER range → supports TB004
//! - R008: Currency code → supports FX005/FX006
//!
//! [RELATION:called_by=CODE:acdoca_table.zig]
//! [RELATION:called_by=CODE:balance_engine.zig]
//! [RELATION:called_by=CODE:fx_converter.zig]
//! [RELATION:called_by=CODE:odps_quality_service.zig]
//!
//! Quality scores are propagated to ODPS dataQualityScore fields.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Data quality classification
pub const DataQuality = enum {
    verified, // Passed all validation checks
    unverified, // Not yet validated
    suspect, // Failed one or more validation checks
    missing_required, // Missing mandatory fields
    invalid_format, // Format/type errors

    pub fn toString(self: DataQuality) []const u8 {
        return switch (self) {
            .verified => "verified",
            .unverified => "unverified",
            .suspect => "suspect",
            .missing_required => "missing_required",
            .invalid_format => "invalid_format",
        };
    }

    pub fn fromString(s: []const u8) DataQuality {
        if (std.mem.eql(u8, s, "verified")) return .verified;
        if (std.mem.eql(u8, s, "suspect")) return .suspect;
        if (std.mem.eql(u8, s, "missing_required")) return .missing_required;
        if (std.mem.eql(u8, s, "invalid_format")) return .invalid_format;
        return .unverified;
    }
};

/// Validation rule severity levels
pub const Severity = enum {
    @"error", // Critical - data cannot be used
    warning, // Important - should be reviewed
    info, // Informational - for awareness

    pub fn toString(self: Severity) []const u8 {
        return switch (self) {
            .@"error" => "error",
            .warning => "warning",
            .info => "info",
        };
    }
};

/// Individual validation rule violation
pub const RuleViolation = struct {
    rule_id: []const u8,
    rule_name: []const u8,
    field: []const u8,
    message: []const u8,
    severity: Severity,
    current_value: ?[]const u8,

    pub fn init(
        rule_id: []const u8,
        rule_name: []const u8,
        field: []const u8,
        message: []const u8,
        severity: Severity,
    ) RuleViolation {
        return .{
            .rule_id = rule_id,
            .rule_name = rule_name,
            .field = field,
            .message = message,
            .severity = severity,
            .current_value = null,
        };
    }
};

/// Validation result containing all violations and quality score
pub const ValidationResult = struct {
    allocator: Allocator,
    is_valid: bool,
    violations: std.ArrayList(RuleViolation),
    quality_score: f64, // 0.0 to 100.0
    error_count: usize,
    warning_count: usize,
    info_count: usize,

    pub fn init(allocator: Allocator) ValidationResult {
        return .{
            .allocator = allocator,
            .is_valid = true,
            .violations = std.ArrayList(RuleViolation).init(allocator),
            .quality_score = 100.0,
            .error_count = 0,
            .warning_count = 0,
            .info_count = 0,
        };
    }

    pub fn deinit(self: *ValidationResult) void {
        self.violations.deinit();
    }

    pub fn addViolation(self: *ValidationResult, violation: RuleViolation) !void {
        try self.violations.append(violation);

        // Update counts
        switch (violation.severity) {
            .@"error" => self.error_count += 1,
            .warning => self.warning_count += 1,
            .info => self.info_count += 1,
        }

        // Mark as invalid if any errors
        if (violation.severity == .@"error") {
            self.is_valid = false;
        }
    }

    pub fn calculateQualityScore(self: *ValidationResult) void {
        // Quality score calculation:
        // - Errors: -20 points each
        // - Warnings: -5 points each
        // - Info: -1 point each
        // Minimum score: 0.0

        const error_penalty = @as(f64, @floatFromInt(self.error_count)) * 20.0;
        const warning_penalty = @as(f64, @floatFromInt(self.warning_count)) * 5.0;
        const info_penalty = @as(f64, @floatFromInt(self.info_count)) * 1.0;

        self.quality_score = @max(0.0, 100.0 - error_penalty - warning_penalty - info_penalty);
    }
};

/// IFRS Validation Rules for Journal Entries
pub const IFRSValidationRules = struct {
    /// R001: RACCT (G/L Account) is mandatory
    pub fn validateRACCT(entry: anytype, result: *ValidationResult) !void {
        if (entry.racct.len == 0) {
            try result.addViolation(RuleViolation.init(
                "R001",
                "RACCT Mandatory",
                "racct",
                "G/L Account (RACCT) is mandatory per IFRS requirements",
                .@"error",
            ));
        }
    }

    /// R002: DRCRK must be 'S' (Debit) or 'H' (Credit)
    pub fn validateDRCRK(entry: anytype, result: *ValidationResult) !void {
        if (!std.mem.eql(u8, entry.drcrk, "S") and !std.mem.eql(u8, entry.drcrk, "H")) {
            try result.addViolation(RuleViolation.init(
                "R002",
                "DRCRK Invalid",
                "drcrk",
                "Debit/Credit indicator must be 'S' (Soll/Debit) or 'H' (Haben/Credit)",
                .@"error",
            ));
        }
    }

    /// R003: POPER (Posting Period) must be between 1 and 12
    pub fn validatePOPER(entry: anytype, result: *ValidationResult) !void {
        if (entry.poper < 1 or entry.poper > 12) {
            try result.addViolation(RuleViolation.init(
                "R003",
                "POPER Range",
                "poper",
                "Posting period must be between 1 and 12",
                .@"error",
            ));
        }
    }

    /// R004: HSL (Amount) should not be zero (warning)
    pub fn validateHSL(entry: anytype, result: *ValidationResult) !void {
        if (entry.hsl == 0.0) {
            try result.addViolation(RuleViolation.init(
                "R004",
                "Zero Amount",
                "hsl",
                "Amount (HSL) is zero - review if this is intentional",
                .warning,
            ));
        }
    }

    /// R005: Company Code (RBUKRS) is mandatory
    pub fn validateRBUKRS(entry: anytype, result: *ValidationResult) !void {
        if (entry.rbukrs.len == 0) {
            try result.addViolation(RuleViolation.init(
                "R005",
                "Company Code Mandatory",
                "rbukrs",
                "Company code (RBUKRS) is mandatory",
                .@"error",
            ));
        }
    }

    /// R006: Fiscal Year (GJAHR) must be reasonable (2000-2099)
    pub fn validateGJAHR(entry: anytype, result: *ValidationResult) !void {
        if (entry.gjahr < 2000 or entry.gjahr > 2099) {
            try result.addViolation(RuleViolation.init(
                "R006",
                "Fiscal Year Range",
                "gjahr",
                "Fiscal year must be between 2000 and 2099",
                .@"error",
            ));
        }
    }

    /// R007: Document Number (BELNR) is mandatory
    pub fn validateBELNR(entry: anytype, result: *ValidationResult) !void {
        if (entry.belnr.len == 0) {
            try result.addViolation(RuleViolation.init(
                "R007",
                "Document Number Mandatory",
                "belnr",
                "Document number (BELNR) is mandatory for audit trail",
                .@"error",
            ));
        }
    }

    /// R008: Currency Code (RHCUR) is mandatory
    pub fn validateRHCUR(entry: anytype, result: *ValidationResult) !void {
        if (entry.rhcur.len == 0) {
            try result.addViolation(RuleViolation.init(
                "R008",
                "Currency Code Mandatory",
                "rhcur",
                "Local currency code (RHCUR) is mandatory",
                .@"error",
            ));
        } else if (entry.rhcur.len != 3) {
            try result.addViolation(RuleViolation.init(
                "R008",
                "Currency Code Format",
                "rhcur",
                "Currency code must be 3 characters (ISO 4217)",
                .warning,
            ));
        }
    }

    /// R009: Exchange Rate (UKURS) must be positive
    pub fn validateUKURS(entry: anytype, result: *ValidationResult) !void {
        if (entry.ukurs <= 0.0) {
            try result.addViolation(RuleViolation.init(
                "R009",
                "Exchange Rate Positive",
                "ukurs",
                "Exchange rate (UKURS) must be positive",
                .@"error",
            ));
        }
    }

    /// R010: Posting Date (BUDAT) format check (info)
    pub fn validateBUDAT(entry: anytype, result: *ValidationResult) !void {
        if (entry.budat.len > 0 and entry.budat.len != 10) {
            try result.addViolation(RuleViolation.init(
                "R010",
                "Date Format",
                "budat",
                "Posting date should be in YYYY-MM-DD format",
                .info,
            ));
        }
    }
};

/// Validate a journal entry against all IFRS rules
pub fn validateJournalEntry(entry: anytype, allocator: Allocator) !ValidationResult {
    var result = ValidationResult.init(allocator);

    // Run all validation rules
    try IFRSValidationRules.validateRACCT(entry, &result);
    try IFRSValidationRules.validateDRCRK(entry, &result);
    try IFRSValidationRules.validatePOPER(entry, &result);
    try IFRSValidationRules.validateHSL(entry, &result);
    try IFRSValidationRules.validateRBUKRS(entry, &result);
    try IFRSValidationRules.validateGJAHR(entry, &result);
    try IFRSValidationRules.validateBELNR(entry, &result);
    try IFRSValidationRules.validateRHCUR(entry, &result);
    try IFRSValidationRules.validateUKURS(entry, &result);
    try IFRSValidationRules.validateBUDAT(entry, &result);

    // Calculate quality score
    result.calculateQualityScore();

    return result;
}

/// Validate exchange rate entry
pub fn validateExchangeRate(rate: anytype, allocator: Allocator) !ValidationResult {
    var result = ValidationResult.init(allocator);

    // Check from currency
    if (rate.from_curr.len == 0) {
        try result.addViolation(RuleViolation.init(
            "X001",
            "From Currency Mandatory",
            "from_curr",
            "Source currency is mandatory",
            .@"error",
        ));
    }

    // Check to currency
    if (rate.to_curr.len == 0) {
        try result.addViolation(RuleViolation.init(
            "X002",
            "To Currency Mandatory",
            "to_curr",
            "Target currency is mandatory",
            .@"error",
        ));
    }

    // Check rate is positive
    if (rate.exchange_rate <= 0.0) {
        try result.addViolation(RuleViolation.init(
            "X003",
            "Rate Positive",
            "exchange_rate",
            "Exchange rate must be positive",
            .@"error",
        ));
    }

    // Check ratios are positive
    if (rate.ratio_from <= 0.0 or rate.ratio_to <= 0.0) {
        try result.addViolation(RuleViolation.init(
            "X004",
            "Ratio Positive",
            "ratio_from/ratio_to",
            "Currency ratios must be positive",
            .@"error",
        ));
    }

    result.calculateQualityScore();
    return result;
}

/// Calculate overall quality score for a dataset
pub fn calculateDatasetQuality(
    total_records: usize,
    valid_records: usize,
    suspect_records: usize,
    missing_required: usize,
) f64 {
    if (total_records == 0) return 0.0;

    const valid_pct = (@as(f64, @floatFromInt(valid_records)) / @as(f64, @floatFromInt(total_records))) * 100.0;
    const suspect_penalty = (@as(f64, @floatFromInt(suspect_records)) / @as(f64, @floatFromInt(total_records))) * 20.0;
    const missing_penalty = (@as(f64, @floatFromInt(missing_required)) / @as(f64, @floatFromInt(total_records))) * 30.0;

    return @max(0.0, valid_pct - suspect_penalty - missing_penalty);
}

// Tests
test "data quality enum conversions" {
    const testing = std.testing;

    try testing.expectEqualStrings("verified", DataQuality.verified.toString());
    try testing.expectEqualStrings("suspect", DataQuality.suspect.toString());

    try testing.expect(DataQuality.fromString("verified") == .verified);
    try testing.expect(DataQuality.fromString("invalid_format") == .invalid_format);
    try testing.expect(DataQuality.fromString("unknown") == .unverified);
}

test "validation result quality scoring" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var result = ValidationResult.init(allocator);
    defer result.deinit();

    // Add violations
    try result.addViolation(RuleViolation.init("R001", "Test", "field1", "Error", .@"error"));
    try result.addViolation(RuleViolation.init("R002", "Test", "field2", "Warning", .warning));
    try result.addViolation(RuleViolation.init("R003", "Test", "field3", "Info", .info));

    result.calculateQualityScore();

    // Score should be 100 - 20 - 5 - 1 = 74
    try testing.expectEqual(@as(usize, 1), result.error_count);
    try testing.expectEqual(@as(usize, 1), result.warning_count);
    try testing.expectEqual(@as(usize, 1), result.info_count);
    try testing.expectEqual(@as(f64, 74.0), result.quality_score);
    try testing.expect(!result.is_valid); // Errors make it invalid
}

test "dataset quality calculation" {
    const testing = std.testing;

    const score1 = calculateDatasetQuality(100, 95, 5, 0);
    try testing.expect(score1 > 90.0); // High quality

    const score2 = calculateDatasetQuality(100, 50, 30, 20);
    try testing.expect(score2 < 50.0); // Low quality

    const score3 = calculateDatasetQuality(0, 0, 0, 0);
    try testing.expectEqual(@as(f64, 0.0), score3); // No data
}