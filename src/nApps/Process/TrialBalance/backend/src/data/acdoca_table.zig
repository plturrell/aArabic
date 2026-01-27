//! ============================================================================
//! ACDOCA Table - SAP S/4HANA Universal Journal with Lineage Tracking
//! Mock ACDOCA source table for journal entry data
//! ============================================================================
//!
//! [CODE:file=acdoca_table.zig]
//! [CODE:module=data]
//! [CODE:language=zig]
//!
//! [ODPS:product=acdoca-journal-entries]
//! Note: This module provides the primary source data for all Trial Balance products.
//!
//! [PETRI:stages=S01,S02]
//!
//! [TABLE:provides=TB_JOURNAL_ENTRIES]
//! [TABLE:indices=BY_ACCOUNT,BY_PERIOD,BY_COMPANY]
//!
//! [RELATION:feeds=CODE:balance_engine.zig]
//! [RELATION:feeds=CODE:fx_converter.zig]
//! [RELATION:tracked_by=ODPS:data-lineage]
//! [RELATION:calls=CODE:data_quality.zig]
//! [RELATION:calls=CODE:trial_balance_models.zig]
//!
//! ACDOCA is the source for TB001/TB002 balance validation.
//! All transformations maintain hash-based lineage for ODPS compliance.

const std = @import("std");
const models = @import("trial_balance_models");
const data_quality = @import("data_quality");
const Allocator = std.mem.Allocator;

/// Data lineage entry for tracking transformations
pub const DataLineage = struct {
    lineage_id: [36]u8, // UUID string
    source_dataset_id: []const u8,
    target_dataset_id: []const u8,
    source_hash: [64]u8, // SHA-256 hash (hex string)
    target_hash: [64]u8, // SHA-256 hash (hex string)
    transformation: []const u8, // extract, validate, aggregate, convert, calculate
    transformation_timestamp: i64,
    quality_score: f64, // 0.0 to 100.0
    record_count: usize,

    pub fn init(allocator: Allocator, transformation_type: []const u8) !DataLineage {
        return .{
            .lineage_id = try generateUUID(),
            .source_dataset_id = try allocator.dupe(u8, ""),
            .target_dataset_id = try allocator.dupe(u8, ""),
            .source_hash = [_]u8{0} ** 64,
            .target_hash = [_]u8{0} ** 64,
            .transformation = try allocator.dupe(u8, transformation_type),
            .transformation_timestamp = std.time.timestamp(),
            .quality_score = 0.0,
            .record_count = 0,
        };
    }
};

/// Quality summary for the entire dataset
pub const QualitySummary = struct {
    total_entries: usize,
    verified_entries: usize,
    suspect_entries: usize,
    missing_required: usize,
    invalid_format: usize,
    overall_score: f64, // 0.0 to 100.0

    pub fn init() QualitySummary {
        return .{
            .total_entries = 0,
            .verified_entries = 0,
            .suspect_entries = 0,
            .missing_required = 0,
            .invalid_format = 0,
            .overall_score = 0.0,
        };
    }

    pub fn updateScore(self: *QualitySummary) void {
        self.overall_score = data_quality.calculateDatasetQuality(
            self.total_entries,
            self.verified_entries,
            self.suspect_entries,
            self.missing_required,
        );
    }
};

/// Mock ACDOCA Table - SAP S/4HANA Universal Journal
pub const ACDOCATable = struct {
    allocator: Allocator,
    entries: std.ArrayList(models.JournalEntry),
    lineage: std.ArrayList(DataLineage),
    quality_summary: QualitySummary,

    // Indices for fast lookups
    by_account: std.StringHashMap(std.ArrayList(usize)),
    by_period: std.AutoHashMap(u8, std.ArrayList(usize)),
    by_company: std.StringHashMap(std.ArrayList(usize)),

    pub fn init(allocator: Allocator) ACDOCATable {
        return .{
            .allocator = allocator,
            .entries = std.ArrayList(models.JournalEntry).init(allocator),
            .lineage = std.ArrayList(DataLineage).init(allocator),
            .quality_summary = QualitySummary.init(),
            .by_account = std.StringHashMap(std.ArrayList(usize)).init(allocator),
            .by_period = std.AutoHashMap(u8, std.ArrayList(usize)).init(allocator),
            .by_company = std.StringHashMap(std.ArrayList(usize)).init(allocator),
        };
    }

    pub fn deinit(self: *ACDOCATable) void {
        // Free entries
        for (self.entries.items) |entry| {
            self.allocator.free(entry.rldnr);
            self.allocator.free(entry.rbukrs);
            self.allocator.free(entry.belnr);
            self.allocator.free(entry.racct);
            if (entry.rcntr) |rcntr| self.allocator.free(rcntr);
            if (entry.prctr) |prctr| self.allocator.free(prctr);
            self.allocator.free(entry.rhcur);
            self.allocator.free(entry.rkcur);
            self.allocator.free(entry.rwcur);
            self.allocator.free(entry.drcrk);
            self.allocator.free(entry.budat);
            self.allocator.free(entry.bldat);
            if (entry.segment) |segment| self.allocator.free(segment);
            if (entry.scntr) |scntr| self.allocator.free(scntr);
            if (entry.pprctr) |pprctr| self.allocator.free(pprctr);
            if (entry.bktxt) |bktxt| self.allocator.free(bktxt);
            if (entry.sgtxt) |sgtxt| self.allocator.free(sgtxt);
            if (entry.xblnr) |xblnr| self.allocator.free(xblnr);
            self.allocator.free(entry.source_system);
            self.allocator.free(entry.data_quality);
        }
        self.entries.deinit();

        // Free lineage
        for (self.lineage.items) |lineage_entry| {
            self.allocator.free(lineage_entry.source_dataset_id);
            self.allocator.free(lineage_entry.target_dataset_id);
            self.allocator.free(lineage_entry.transformation);
        }
        self.lineage.deinit();

        // Free indices
        var account_iter = self.by_account.valueIterator();
        while (account_iter.next()) |list| {
            list.deinit();
        }
        self.by_account.deinit();

        var period_iter = self.by_period.valueIterator();
        while (period_iter.next()) |list| {
            list.deinit();
        }
        self.by_period.deinit();

        var company_iter = self.by_company.valueIterator();
        while (company_iter.next()) |list| {
            list.deinit();
        }
        self.by_company.deinit();
    }

    /// Add a journal entry with validation and lineage tracking
    pub fn addEntry(self: *ACDOCATable, entry: models.JournalEntry) !void {
        // Validate entry
        var validation = try data_quality.validateJournalEntry(&entry, self.allocator);
        defer validation.deinit();

        // Update quality summary
        self.quality_summary.total_entries += 1;
        if (validation.is_valid) {
            self.quality_summary.verified_entries += 1;
        } else {
            if (validation.error_count > 0) {
                self.quality_summary.suspect_entries += 1;
            }
        }

        // Calculate data hash for lineage
        const hash = try entry.calculateHash(self.allocator);
        var entry_with_hash = entry;
        entry_with_hash.data_hash = hash;

        // Update data quality based on validation
        entry_with_hash.data_quality = if (validation.is_valid)
            try self.allocator.dupe(u8, "verified")
        else if (validation.error_count > 0)
            try self.allocator.dupe(u8, "suspect")
        else
            try self.allocator.dupe(u8, "unverified");

        // Add to entries
        const entry_index = self.entries.items.len;
        try self.entries.append(entry_with_hash);

        // Update indices
        try self.updateIndices(entry_index, &entry_with_hash);

        // Track lineage
        var lineage_entry = try DataLineage.init(self.allocator, "extract");
        lineage_entry.source_dataset_id = try self.allocator.dupe(u8, "ACDOCA_RAW");
        lineage_entry.target_dataset_id = try self.allocator.dupe(u8, "ACDOCA_TABLE");
        lineage_entry.source_hash = hash;
        lineage_entry.target_hash = hash;
        lineage_entry.quality_score = validation.quality_score;
        lineage_entry.record_count = 1;

        try self.lineage.append(lineage_entry);

        // Update overall quality score
        self.quality_summary.updateScore();
    }

    /// Update indices for fast lookups
    fn updateIndices(self: *ACDOCATable, entry_index: usize, entry: *const models.JournalEntry) !void {
        // Index by account
        const account_key = entry.racct;
        var account_result = try self.by_account.getOrPut(account_key);
        if (!account_result.found_existing) {
            account_result.value_ptr.* = std.ArrayList(usize).init(self.allocator);
        }
        try account_result.value_ptr.append(entry_index);

        // Index by period
        var period_result = try self.by_period.getOrPut(entry.poper);
        if (!period_result.found_existing) {
            period_result.value_ptr.* = std.ArrayList(usize).init(self.allocator);
        }
        try period_result.value_ptr.append(entry_index);

        // Index by company
        const company_key = entry.rbukrs;
        var company_result = try self.by_company.getOrPut(company_key);
        if (!company_result.found_existing) {
            company_result.value_ptr.* = std.ArrayList(usize).init(self.allocator);
        }
        try company_result.value_ptr.append(entry_index);
    }

    /// Get entries by posting period
    pub fn getEntriesByPeriod(self: *ACDOCATable, period: u8) !std.ArrayList(models.JournalEntry) {
        var result = std.ArrayList(models.JournalEntry).init(self.allocator);

        if (self.by_period.get(period)) |indices| {
            for (indices.items) |index| {
                try result.append(self.entries.items[index]);
            }
        }

        return result;
    }

    /// Get entries by G/L account
    pub fn getEntriesByAccount(self: *ACDOCATable, racct: []const u8) !std.ArrayList(models.JournalEntry) {
        var result = std.ArrayList(models.JournalEntry).init(self.allocator);

        if (self.by_account.get(racct)) |indices| {
            for (indices.items) |index| {
                try result.append(self.entries.items[index]);
            }
        }

        return result;
    }

    /// Get entries by company code
    pub fn getEntriesByCompany(self: *ACDOCATable, rbukrs: []const u8) !std.ArrayList(models.JournalEntry) {
        var result = std.ArrayList(models.JournalEntry).init(self.allocator);

        if (self.by_company.get(rbukrs)) |indices| {
            for (indices.items) |index| {
                try result.append(self.entries.items[index]);
            }
        }

        return result;
    }

    /// Calculate quality score for current dataset
    pub fn getQualityScore(self: *const ACDOCATable) f64 {
        return self.quality_summary.overall_score;
    }

    /// Get lineage chain for data traceability
    pub fn getLineageChain(self: *const ACDOCATable) []const DataLineage {
        return self.lineage.items;
    }

    /// Verify debit = credit balance
    pub fn verifyBalance(self: *const ACDOCATable) !bool {
        var total_debit: f64 = 0.0;
        var total_credit: f64 = 0.0;

        for (self.entries.items) |entry| {
            if (std.mem.eql(u8, entry.drcrk, "S")) {
                // Debit
                total_debit += entry.hsl;
            } else if (std.mem.eql(u8, entry.drcrk, "H")) {
                // Credit
                total_credit += entry.hsl;
            }
        }

        // Allow for small rounding differences (< 0.01)
        const diff = @abs(total_debit - total_credit);
        return diff < 0.01;
    }

    /// Get statistics about the ACDOCA table
    pub fn getStatistics(self: *const ACDOCATable) Statistics {
        return .{
            .total_entries = self.entries.items.len,
            .verified_entries = self.quality_summary.verified_entries,
            .suspect_entries = self.quality_summary.suspect_entries,
            .quality_score = self.quality_summary.overall_score,
            .unique_accounts = self.by_account.count(),
            .unique_periods = self.by_period.count(),
            .unique_companies = self.by_company.count(),
        };
    }
};

/// Statistics about ACDOCA table
pub const Statistics = struct {
    total_entries: usize,
    verified_entries: usize,
    suspect_entries: usize,
    quality_score: f64,
    unique_accounts: usize,
    unique_periods: usize,
    unique_companies: usize,
};

/// Generate a simple UUID (v4-like)
fn generateUUID() ![36]u8 {
    var uuid: [36]u8 = undefined;
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = prng.random();

    // Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    const hex = "0123456789abcdef";
    for (0..36) |i| {
        if (i == 8 or i == 13 or i == 18 or i == 23) {
            uuid[i] = '-';
        } else if (i == 14) {
            uuid[i] = '4'; // Version 4
        } else if (i == 19) {
            uuid[i] = hex[8 + random.int(u8) % 4]; // 8, 9, a, or b
        } else {
            uuid[i] = hex[random.int(u8) % 16];
        }
    }

    return uuid;
}

// Tests
test "ACDOCA table initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var table = ACDOCATable.init(allocator);
    defer table.deinit();

    try testing.expectEqual(@as(usize, 0), table.entries.items.len);
    try testing.expectEqual(@as(f64, 0.0), table.quality_summary.overall_score);
}

test "ACDOCA table add entry" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var table = ACDOCATable.init(allocator);
    defer table.deinit();

    var entry = models.JournalEntry.init(allocator);
    entry.rldnr = try allocator.dupe(u8, "0L");
    entry.rbukrs = try allocator.dupe(u8, "HKG");
    entry.gjahr = 2025;
    entry.belnr = try allocator.dupe(u8, "1000000001");
    entry.docln = 1;
    entry.racct = try allocator.dupe(u8, "1000000");
    entry.hsl = 1000.0;
    entry.rhcur = try allocator.dupe(u8, "HKD");
    entry.rkcur = try allocator.dupe(u8, "USD");
    entry.rwcur = try allocator.dupe(u8, "HKD");
    entry.drcrk = try allocator.dupe(u8, "S");
    entry.poper = 11;
    entry.budat = try allocator.dupe(u8, "2025-11-01");
    entry.bldat = try allocator.dupe(u8, "2025-11-01");
    entry.source_system = try allocator.dupe(u8, "SAP_S4HANA");
    entry.data_quality = try allocator.dupe(u8, "unverified");

    try table.addEntry(entry);

    try testing.expectEqual(@as(usize, 1), table.entries.items.len);
    try testing.expect(table.quality_summary.total_entries > 0);
}

test "ACDOCA table UUID generation" {
    const uuid = try generateUUID();

    // Check format
    std.testing.expect(uuid[8] == '-') catch unreachable;
    std.testing.expect(uuid[13] == '-') catch unreachable;
    std.testing.expect(uuid[18] == '-') catch unreachable;
    std.testing.expect(uuid[23] == '-') catch unreachable;
    std.testing.expect(uuid[14] == '4') catch unreachable; // Version 4
}