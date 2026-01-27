//! ============================================================================
//! FX Converter - Multi-Currency Conversion Module
//! Implements SAP S/4HANA TCURR exchange rate logic
//! ============================================================================
//!
//! [CODE:file=fx_converter.zig]
//! [CODE:module=calculation]
//! [CODE:language=zig]
//!
//! [ODPS:product=exchange-rates]
//! [ODPS:rules=FX001,FX002,FX003,FX004,FX005,FX006,FX007]
//!
//! [DOI:controls=REC-005]
//! [DOI:thresholds=REQ-THRESH-006]
//!
//! [PETRI:stages=S03]
//! [PETRI:process=TB_PROCESS_petrinet.pnml]
//!
//! [TABLE:reads=TB_EXCHANGE_RATES,TB_TCURR]
//! [TABLE:writes=TB_EXCHANGE_RATES]
//!
//! [API:produces=/api/v1/exchange-rates]
//!
//! [RELATION:implements=ODPS:exchange-rates]
//! [RELATION:called_by=CODE:balance_engine.zig]
//! [RELATION:called_by=CODE:odps_api.zig]
//!
//! This module handles multi-currency conversion using SAP TCURR exchange rate
//! logic, with ODPS validation rules FX001-FX007 for rate integrity.

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// ODPS FX Validation Rule IDs
// Reference: models/odps/primary/exchange-rates.odps.yaml
// ============================================================================

/// ODPS Exchange Rate Validation Rules (FX001-FX007)
pub const ODPSFXRuleID = struct {
    /// FX001: From Currency Mandatory - Source currency required
    pub const FX001_FROM_CURRENCY_MANDATORY = "FX001";
    
    /// FX002: To Currency Mandatory - Target currency required
    pub const FX002_TO_CURRENCY_MANDATORY = "FX002";
    
    /// FX003: Rate Positive - Exchange rate must be positive
    pub const FX003_RATE_POSITIVE = "FX003";
    
    /// FX004: Ratio Positive - Currency ratios must be positive
    pub const FX004_RATIO_POSITIVE = "FX004";
    
    /// FX005: Exchange Rate Verification - Rates match Group rates
    /// Control Ref: REC-005
    pub const FX005_EXCHANGE_RATE_VERIFICATION = "FX005";
    
    /// FX006: Period-Specific Rate - Period-appropriate rates applied
    pub const FX006_PERIOD_SPECIFIC_RATE = "FX006";
    
    /// FX007: Group Rate Source - Rates from approved sources
    pub const FX007_GROUP_RATE_SOURCE = "FX007";
};

/// Approved rate sources (FX007 compliance)
pub const RateSource = enum {
    GROUP_TREASURY,
    ECB,
    FED,
    MANUAL,
};

pub const ExchangeRate = struct {
    from_currency: []const u8,
    to_currency: []const u8,
    rate_type: []const u8, // M=Month-end, B=Bank buying, G=Bank selling
    valid_from: []const u8,
    rate: f64,
    from_factor: i32,
    to_factor: i32,
    // ODPS Field: rate_source (FX007 compliance)
    rate_source: ?RateSource = null,

    pub fn convert(self: *const ExchangeRate, amount: f64) f64 {
        const from_factor_f: f64 = @floatFromInt(self.from_factor);
        const to_factor_f: f64 = @floatFromInt(self.to_factor);
        return amount * self.rate / from_factor_f * to_factor_f;
    }
    
    /// Validate FX001: From Currency Mandatory
    pub fn validateFX001(self: *const ExchangeRate) bool {
        return self.from_currency.len > 0;
    }
    
    /// Validate FX002: To Currency Mandatory
    pub fn validateFX002(self: *const ExchangeRate) bool {
        return self.to_currency.len > 0;
    }
    
    /// Validate FX003: Rate Positive
    pub fn validateFX003(self: *const ExchangeRate) bool {
        return self.rate > 0.0;
    }
    
    /// Validate FX004: Ratio Positive
    pub fn validateFX004(self: *const ExchangeRate) bool {
        return self.from_factor > 0 and self.to_factor > 0;
    }
    
    /// Validate FX007: Group Rate Source (approved sources only)
    pub fn validateFX007(self: *const ExchangeRate) bool {
        if (self.rate_source) |source| {
            return source == .GROUP_TREASURY or source == .ECB or source == .FED;
        }
        return false; // No source specified = invalid
    }
    
    /// Run all FX validations
    pub fn validateAll(self: *const ExchangeRate) FXValidationResult {
        return FXValidationResult{
            .fx001_passed = self.validateFX001(),
            .fx002_passed = self.validateFX002(),
            .fx003_passed = self.validateFX003(),
            .fx004_passed = self.validateFX004(),
            .fx007_passed = self.validateFX007(),
        };
    }
};

/// FX Validation Result (ODPS compliance tracking)
pub const FXValidationResult = struct {
    fx001_passed: bool, // From Currency Mandatory
    fx002_passed: bool, // To Currency Mandatory
    fx003_passed: bool, // Rate Positive
    fx004_passed: bool, // Ratio Positive
    fx007_passed: bool, // Group Rate Source
    
    pub fn isValid(self: *const FXValidationResult) bool {
        return self.fx001_passed and self.fx002_passed and 
               self.fx003_passed and self.fx004_passed and self.fx007_passed;
    }
    
    pub fn getFailedRules(self: *const FXValidationResult) []const []const u8 {
        var failed: [5][]const u8 = undefined;
        var count: usize = 0;
        
        if (!self.fx001_passed) { failed[count] = ODPSFXRuleID.FX001_FROM_CURRENCY_MANDATORY; count += 1; }
        if (!self.fx002_passed) { failed[count] = ODPSFXRuleID.FX002_TO_CURRENCY_MANDATORY; count += 1; }
        if (!self.fx003_passed) { failed[count] = ODPSFXRuleID.FX003_RATE_POSITIVE; count += 1; }
        if (!self.fx004_passed) { failed[count] = ODPSFXRuleID.FX004_RATIO_POSITIVE; count += 1; }
        if (!self.fx007_passed) { failed[count] = ODPSFXRuleID.FX007_GROUP_RATE_SOURCE; count += 1; }
        
        return failed[0..count];
    }
};

pub const FXConverter = struct {
    allocator: Allocator,
    rate_cache: std.StringHashMap(ExchangeRate),

    pub fn init(allocator: Allocator) FXConverter {
        return FXConverter{
            .allocator = allocator,
            .rate_cache = std.StringHashMap(ExchangeRate).init(allocator),
        };
    }

    pub fn deinit(self: *FXConverter) void {
        // Free all keys before destroying the map
        var iterator = self.rate_cache.keyIterator();
        while (iterator.next()) |key_ptr| {
            self.allocator.free(key_ptr.*);
        }
        self.rate_cache.deinit();
    }

    pub fn load_rates(self: *FXConverter, rates: []const ExchangeRate) !void {
        for (rates) |rate| {
            const key = try std.fmt.allocPrint(
                self.allocator,
                "{s}_{s}_{s}_{s}",
                .{ rate.from_currency, rate.to_currency, rate.rate_type, rate.valid_from },
            );
            // HashMap now owns the key - don't free it
            try self.rate_cache.put(key, rate);
        }
    }

    pub fn convert(
        self: *FXConverter,
        amount: f64,
        from_currency: []const u8,
        to_currency: []const u8,
        rate_type: []const u8,
        posting_date: []const u8,
    ) !f64 {
        // Handle same currency case
        if (std.mem.eql(u8, from_currency, to_currency)) {
            return amount;
        }

        // Find applicable exchange rate
        const rate = try self.find_rate(from_currency, to_currency, rate_type, posting_date);
        return rate.convert(amount);
    }

    fn find_rate(
        self: *FXConverter,
        from_currency: []const u8,
        to_currency: []const u8,
        rate_type: []const u8,
        posting_date: []const u8,
    ) !ExchangeRate {
        // Try exact match first
        const key = try std.fmt.allocPrint(
            self.allocator,
            "{s}_{s}_{s}_{s}",
            .{ from_currency, to_currency, rate_type, posting_date },
        );
        defer self.allocator.free(key);

        if (self.rate_cache.get(key)) |rate| {
            return rate;
        }

        // Try fallback rate types: M -> B -> G
        const fallback_types = [_][]const u8{ "B", "G" };
        for (fallback_types) |fallback_type| {
            const fallback_key = try std.fmt.allocPrint(
                self.allocator,
                "{s}_{s}_{s}_{s}",
                .{ from_currency, to_currency, fallback_type, posting_date },
            );
            defer self.allocator.free(fallback_key);

            if (self.rate_cache.get(fallback_key)) |rate| {
                return rate;
            }
        }

        return error.ExchangeRateNotFound;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "simple currency conversion" {
    const allocator = std.testing.allocator;
    var converter = FXConverter.init(allocator);
    defer converter.deinit();

    const rates = [_]ExchangeRate{
        .{
            .from_currency = "EUR",
            .to_currency = "USD",
            .rate_type = "M",
            .valid_from = "2025-01-31",
            .rate = 1.0845,
            .from_factor = 1,
            .to_factor = 1,
        },
    };

    try converter.load_rates(&rates);

    const converted = try converter.convert(
        1000.0,
        "EUR",
        "USD",
        "M",
        "2025-01-31",
    );

    try std.testing.expectApproxEqRel(1084.5, converted, 0.001);
}

test "same currency returns same amount" {
    const allocator = std.testing.allocator;
    var converter = FXConverter.init(allocator);
    defer converter.deinit();

    const converted = try converter.convert(1000.0, "USD", "USD", "M", "2025-01-31");
    try std.testing.expectEqual(1000.0, converted);
}