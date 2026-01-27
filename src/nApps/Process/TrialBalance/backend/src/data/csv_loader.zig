//! ============================================================================
//! CSV Loader for Trial Balance Data
//! Load ACDOCA journal entries, exchange rates, trial balance entries, etc.
//! ============================================================================
//!
//! [CODE:file=csv_loader.zig]
//! [CODE:module=data]
//! [CODE:language=zig]
//!
//! [ODPS:product=acdoca-journal-entries,exchange-rates,trial-balance-aggregated,variances]
//!
//! [PETRI:stages=S01]
//! [PETRI:transitions=T_EXTRACT]
//!
//! [TABLE:produces=TB_JOURNAL_ENTRIES,TB_EXCHANGE_RATES,TB_TRIAL_BALANCE]
//!
//! [RELATION:feeds=CODE:acdoca_table.zig]
//! [RELATION:feeds=CODE:balance_engine.zig]
//! [RELATION:calls=CODE:trial_balance_models.zig]
//!
//! Loads raw CSV data from BusDocs/sample-data and converts to ODPS-compliant structures.
//! Data lineage is established from file load through hash calculation.

const std = @import("std");
const models = @import("trial_balance_models");
const data_quality = @import("data_quality");
const Allocator = std.mem.Allocator;

/// Base path for all CSV files (relative to backend directory)
const BASE_DATA_PATH = "../BusDocs/sample-data/extracted/";

/// CSV parsing errors
pub const CsvError = error{
    FileNotFound,
    InvalidFormat,
    ParseError,
    OutOfMemory,
};

// ============================================================================
// NEW: RAW ACDOCA JOURNAL ENTRY LOADERS (Day 1 - Phase 4)
// ============================================================================

/// Load TB October raw journal entries
pub fn loadTBRawOct(allocator: Allocator) !std.ArrayList(models.JournalEntry) {
    const file_path = BASE_DATA_PATH ++ "HKG_TB review Nov'25(RAW TB OCT'25).csv";
    return loadRawJournalEntries(allocator, file_path, 10); // POPER=10 (October)
}

/// Load TB November raw journal entries
pub fn loadTBRawNov(allocator: Allocator) !std.ArrayList(models.JournalEntry) {
    const file_path = BASE_DATA_PATH ++ "HKG_TB review Nov'25(RAW TB NOV'25).csv";
    return loadRawJournalEntries(allocator, file_path, 11); // POPER=11 (November)
}

/// Load PL October raw journal entries
pub fn loadPLRawOct(allocator: Allocator) !std.ArrayList(models.JournalEntry) {
    const file_path = BASE_DATA_PATH ++ "HKG_PL review Nov'25(Raw TB oct'25).csv";
    return loadRawJournalEntries(allocator, file_path, 10); // POPER=10 (October)
}

/// Load PL November raw journal entries
pub fn loadPLRawNov(allocator: Allocator) !std.ArrayList(models.JournalEntry) {
    const file_path = BASE_DATA_PATH ++ "HKG_PL review Nov'25(Raw TB Nov'25).csv";
    return loadRawJournalEntries(allocator, file_path, 11); // POPER=11 (November)
}

/// Generic function to load raw ACDOCA journal entries from CSV
fn loadRawJournalEntries(
    allocator: Allocator,
    file_path: []const u8,
    period: u8,
) !std.ArrayList(models.JournalEntry) {
    var entries = std.ArrayList(models.JournalEntry).init(allocator);
    errdefer entries.deinit();

    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024); // 10MB max
    defer allocator.free(file_content);

    var line_iter = std.mem.splitScalar(u8, file_content, '\n');
    var line_num: usize = 0;

    // Skip header rows (lines 1-9)
    while (line_num < 9) : (line_num += 1) {
        _ = line_iter.next();
    }

    // Read data rows
    while (line_iter.next()) |line| {
        line_num += 1;
        if (line.len == 0) continue;

        const entry = try parseJournalEntryLine(allocator, line, period) orelse continue;
        try entries.append(entry);
    }

    return entries;
}

/// Parse a single ACDOCA journal entry CSV line
fn parseJournalEntryLine(
    allocator: Allocator,
    line: []const u8,
    period: u8,
) !?models.JournalEntry {
    var fields = std.ArrayList([]const u8).init(allocator);
    defer fields.deinit();

    // Parse CSV fields
    var field_start: usize = 0;
    var in_quotes = false;

    for (line, 0..) |char, i| {
        if (char == '"') {
            in_quotes = !in_quotes;
        } else if (char == ',' and !in_quotes) {
            const field = std.mem.trim(u8, line[field_start..i], " \t\"");
            try fields.append(field);
            field_start = i + 1;
        }
    }
    const last_field = std.mem.trim(u8, line[field_start..], " \t\"\r\n");
    try fields.append(last_field);

    // Need at least 15 fields to be valid
    if (fields.items.len < 15) return null;

    // Create journal entry (map CSV columns to ACDOCA fields)
    var entry = models.JournalEntry.init(allocator);
    
    // Key fields
    entry.rldnr = try allocator.dupe(u8, "0L"); // Leading ledger (default)
    entry.rbukrs = try allocator.dupe(u8, if (fields.items.len > 2) fields.items[2] else "HKG"); // Business unit as company code
    entry.gjahr = 2025; // Fiscal year
    entry.belnr = try allocator.dupe(u8, if (line_num < 10000000) try std.fmt.allocPrint(allocator, "{d:0>10}", .{line_num}) else "9999999999"); // Generate doc number
    entry.docln = @intCast(line_num % 999999); // Line item number
    
    // Account
    entry.racct = try allocator.dupe(u8, if (fields.items.len > 3) fields.items[3] else ""); // Account code
    
    // Amounts
    if (fields.items.len > 14 and fields.items[14].len > 0) {
        const amt_str = try cleanAmountString(allocator, fields.items[14]);
        defer allocator.free(amt_str);
        entry.hsl = std.fmt.parseFloat(f64, amt_str) catch 0.0;
    }
    entry.ksl = entry.hsl; // Same as local (simplified)
    entry.wsl = entry.hsl; // Same as local (simplified)
    
    // Currency
    entry.rhcur = try allocator.dupe(u8, if (fields.items.len > 13 and fields.items[13].len > 0) fields.items[13] else "HKD");
    entry.rkcur = try allocator.dupe(u8, "USD"); // Group currency
    entry.rwcur = entry.rhcur; // Transaction currency same as local
    
    // Exchange rate
    if (fields.items.len > 23 and fields.items[23].len > 0) {
        const rate_str = try cleanAmountString(allocator, fields.items[23]);
        defer allocator.free(rate_str);
        entry.ukurs = std.fmt.parseFloat(f64, rate_str) catch 1.0;
    } else {
        entry.ukurs = 1.0;
    }
    
    // Debit/Credit indicator (determine from amount sign)
    entry.drcrk = try allocator.dupe(u8, if (entry.hsl >= 0) "S" else "H");
    
    // Posting period
    entry.poper = period;
    
    // Dates
    entry.budat = try allocator.dupe(u8, try std.fmt.allocPrint(allocator, "2025-{d:0>2}-01", .{period}));
    entry.bldat = entry.budat;
    
    // Data quality
    entry.data_quality = try allocator.dupe(u8, "unverified");
    entry.source_system = try allocator.dupe(u8, "SAP_S4HANA");
    entry.extraction_ts = std.time.timestamp();
    
    // Calculate hash for lineage (will be done later when we have complete data)
    entry.data_hash = [_]u8{0} ** 64;
    
    return entry;
}

// ============================================================================
// NEW: EXCHANGE RATE LOADERS (Day 1 - Phase 4)
// ============================================================================

/// Load TB exchange rates
pub fn loadTBRates(allocator: Allocator) !std.ArrayList(models.ExchangeRate) {
    const file_path = BASE_DATA_PATH ++ "HKG_TB review Nov'25(Rates).csv";
    return loadExchangeRates(allocator, file_path);
}

/// Load PL exchange rates
pub fn loadPLRates(allocator: Allocator) !std.ArrayList(models.ExchangeRate) {
    const file_path = BASE_DATA_PATH ++ "HKG_PL review Nov'25(Rates).csv";
    return loadExchangeRates(allocator, file_path);
}

/// Generic function to load exchange rates from CSV
fn loadExchangeRates(
    allocator: Allocator,
    file_path: []const u8,
) !std.ArrayList(models.ExchangeRate) {
    var rates = std.ArrayList(models.ExchangeRate).init(allocator);
    errdefer rates.deinit();

    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(file_content);

    var line_iter = std.mem.splitScalar(u8, file_content, '\n');

    // Skip header row
    _ = line_iter.next();

    // Read data rows
    while (line_iter.next()) |line| {
        if (line.len == 0) continue;

        const rate = try parseExchangeRateLine(allocator, line) orelse continue;
        try rates.append(rate);
    }

    return rates;
}

/// Parse a single exchange rate CSV line
fn parseExchangeRateLine(
    allocator: Allocator,
    line: []const u8,
) !?models.ExchangeRate {
    var fields = std.ArrayList([]const u8).init(allocator);
    defer fields.deinit();

    var field_start: usize = 0;
    var in_quotes = false;

    for (line, 0..) |char, i| {
        if (char == '"') {
            in_quotes = !in_quotes;
        } else if (char == ',' and !in_quotes) {
            const field = std.mem.trim(u8, line[field_start..i], " \t\"");
            try fields.append(field);
            field_start = i + 1;
        }
    }
    const last_field = std.mem.trim(u8, line[field_start..], " \t\"\r\n");
    try fields.append(last_field);

    // Expected columns: FromCurrency, ToCurrency, RateType, Rate, ValidFrom, etc.
    if (fields.items.len < 4) return null;

    var rate = models.ExchangeRate.init();
    rate.from_curr = try allocator.dupe(u8, fields.items[0]);
    rate.to_curr = try allocator.dupe(u8, fields.items[1]);
    rate.rate_type = try allocator.dupe(u8, if (fields.items.len > 2 and fields.items[2].len > 0) fields.items[2] else "M");

    // Parse rate
    if (fields.items.len > 3 and fields.items[3].len > 0) {
        const rate_str = try cleanAmountString(allocator, fields.items[3]);
        defer allocator.free(rate_str);
        rate.exchange_rate = std.fmt.parseFloat(f64, rate_str) catch 1.0;
    }

    // Valid from date
    if (fields.items.len > 4 and fields.items[4].len > 0) {
        rate.valid_from = try allocator.dupe(u8, fields.items[4]);
    } else {
        rate.valid_from = try allocator.dupe(u8, "2025-01-01");
    }

    rate.source = try allocator.dupe(u8, "SAP");
    rate.last_updated = std.time.timestamp();

    return rate;
}

// ============================================================================
// REFACTORED: EXISTING LOADERS WITH UPDATED BASE PATH (Day 1 - Phase 4)
// ============================================================================

/// Load trial balance data from MTD-TB CSV file
/// REFACTORED: Updated to use BASE_DATA_PATH
pub fn loadTrialBalanceData(allocator: Allocator) !std.ArrayList(models.TrialBalanceEntry) {
    const file_path = BASE_DATA_PATH ++ "HKG_PL review Nov'25(MTD-TB).csv";
    return loadTrialBalanceDataFromPath(allocator, file_path);
}

/// Load trial balance data from specific path (for testing)
pub fn loadTrialBalanceDataFromPath(
    allocator: Allocator,
    file_path: []const u8,
) !std.ArrayList(models.TrialBalanceEntry) {
    var entries = std.ArrayList(models.TrialBalanceEntry).init(allocator);
    errdefer entries.deinit();

    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(file_content);

    var line_iter = std.mem.splitScalar(u8, file_content, '\n');
    var line_num: usize = 0;

    // Skip header rows (lines 1-9)
    while (line_num < 9) : (line_num += 1) {
        _ = line_iter.next();
    }

    // Read data rows
    while (line_iter.next()) |line| {
        line_num += 1;
        if (line.len == 0) continue;

        const entry = try parseTrialBalanceLine(allocator, line) orelse continue;
        try entries.append(entry);
    }

    return entries;
}

/// Parse a single trial balance CSV line
fn parseTrialBalanceLine(
    allocator: Allocator,
    line: []const u8,
) !?models.TrialBalanceEntry {
    var fields = std.ArrayList([]const u8).init(allocator);
    defer fields.deinit();

    var field_start: usize = 0;
    var in_quotes = false;

    for (line, 0..) |char, i| {
        if (char == '"') {
            in_quotes = !in_quotes;
        } else if (char == ',' and !in_quotes) {
            const field = std.mem.trim(u8, line[field_start..i], " \t\"");
            try fields.append(field);
            field_start = i + 1;
        }
    }
    const last_field = std.mem.trim(u8, line[field_start..], " \t\"\r\n");
    try fields.append(last_field);

    if (fields.items.len < 15) return null;

    var entry = models.TrialBalanceEntry.init(
        try allocator.dupe(u8, fields.items[3]), // Account code
        try allocator.dupe(u8, fields.items[4]), // Account name
        try allocator.dupe(u8, if (fields.items.len > 22) fields.items[22] else "Asset"), // Account type
        try allocator.dupe(u8, fields.items[2]), // Business unit
        try allocator.dupe(u8, fields.items[0]), // Fiscal period
    );

    // Parse Base Amt (field 14)
    if (fields.items[14].len > 0) {
        const amt_str = try cleanAmountString(allocator, fields.items[14]);
        defer allocator.free(amt_str);
        entry.closing_balance = std.fmt.parseFloat(f64, amt_str) catch 0.0;
    }

    // Set IFRS category
    if (fields.items.len > 18 and fields.items[18].len > 0) {
        entry.ifrs_category = try allocator.dupe(u8, fields.items[18]);
    }

    // Set currency
    if (fields.items.len > 13 and fields.items[13].len > 0) {
        entry.currency_code = try allocator.dupe(u8, fields.items[13]);
    }

    entry.data_quality = try allocator.dupe(u8, "unverified");
    entry.source_system = try allocator.dupe(u8, "SAP S/4HANA");
    entry.last_updated = std.time.timestamp();

    return entry;
}

/// Load variance data from PL Variance CSV file
/// REFACTORED: Updated to use BASE_DATA_PATH
pub fn loadVarianceData(allocator: Allocator) !std.ArrayList(models.VarianceEntry) {
    const file_path = BASE_DATA_PATH ++ "HKG_PL review Nov'25(PL Variance).csv";
    return loadVarianceDataFromPath(allocator, file_path);
}

/// Load variance data from specific path (for testing)
pub fn loadVarianceDataFromPath(
    allocator: Allocator,
    file_path: []const u8,
) !std.ArrayList(models.VarianceEntry) {
    var variances = std.ArrayList(models.VarianceEntry).init(allocator);
    errdefer variances.deinit();

    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(file_content);

    var line_iter = std.mem.splitScalar(u8, file_content, '\n');
    var line_num: usize = 0;

    // Skip header rows
    while (line_num < 9) : (line_num += 1) {
        _ = line_iter.next();
    }

    // Read data rows
    while (line_iter.next()) |line| {
        line_num += 1;
        if (line.len == 0) continue;

        const variance = try parseVarianceLine(allocator, line) orelse continue;
        try variances.append(variance);
    }

    return variances;
}

/// Parse a variance CSV line
fn parseVarianceLine(
    allocator: Allocator,
    line: []const u8,
) !?models.VarianceEntry {
    var fields = std.ArrayList([]const u8).init(allocator);
    defer fields.deinit();

    var field_start: usize = 0;
    var in_quotes = false;

    for (line, 0..) |char, i| {
        if (char == '"') {
            in_quotes = !in_quotes;
        } else if (char == ',' and !in_quotes) {
            const field = std.mem.trim(u8, line[field_start..i], " \t\"");
            try fields.append(field);
            field_start = i + 1;
        }
    }
    const last_field = std.mem.trim(u8, line[field_start..], " \t\"\r\n");
    try fields.append(last_field);

    if (fields.items.len < 10) return null;

    var variance = models.VarianceEntry.init(
        try allocator.dupe(u8, fields.items[3]), // Account code
        try allocator.dupe(u8, fields.items[4]), // Account name
        try allocator.dupe(u8, "2025-11"), // Current period
        try allocator.dupe(u8, "2025-10"), // Previous period
    );

    // Set account type
    if (fields.items.len > 22) {
        variance.account_type = try allocator.dupe(u8, fields.items[22]);
    }

    return variance;
}

/// Load checklist data from Checklist CSV file
/// REFACTORED: Updated to use BASE_DATA_PATH
pub fn loadChecklistData(allocator: Allocator) !std.ArrayList(models.ChecklistItem) {
    const file_path = BASE_DATA_PATH ++ "HKG_PL review Nov'25(Checklist).csv";
    return loadChecklistDataFromPath(allocator, file_path);
}

/// Load checklist data from specific path (for testing)
pub fn loadChecklistDataFromPath(
    allocator: Allocator,
    file_path: []const u8,
) !std.ArrayList(models.ChecklistItem) {
    var items = std.ArrayList(models.ChecklistItem).init(allocator);
    errdefer items.deinit();

    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(file_content);

    var line_iter = std.mem.splitScalar(u8, file_content, '\n');
    var line_num: usize = 0;

    // Skip header row
    _ = line_iter.next();

    // Read data rows
    while (line_iter.next()) |line| {
        line_num += 1;
        if (line.len == 0) continue;

        const item = try parseChecklistLine(allocator, line, line_num) orelse continue;
        try items.append(item);
    }

    return items;
}

/// Parse a checklist CSV line
fn parseChecklistLine(
    allocator: Allocator,
    line: []const u8,
    line_num: usize,
) !?models.ChecklistItem {
    var fields = std.ArrayList([]const u8).init(allocator);
    defer fields.deinit();

    var field_start: usize = 0;
    var in_quotes = false;

    for (line, 0..) |char, i| {
        if (char == '"') {
            in_quotes = !in_quotes;
        } else if (char == ',' and !in_quotes) {
            const field = std.mem.trim(u8, line[field_start..i], " \t\"");
            try fields.append(field);
            field_start = i + 1;
        }
    }
    const last_field = std.mem.trim(u8, line[field_start..], " \t\"\r\n");
    try fields.append(last_field);

    if (fields.items.len < 2) return null;

    const id_buf = try std.fmt.allocPrint(allocator, "CHK{d:0>3}", .{line_num});
    const stage_id = try allocator.dupe(u8, if (fields.items[0].len > 0) fields.items[0] else "S01");
    const title = try allocator.dupe(u8, if (fields.items[1].len > 0) fields.items[1] else "Unnamed Item");

    var item = models.ChecklistItem.init(id_buf, stage_id, title);

    if (fields.items.len > 2 and fields.items[2].len > 0) {
        item.description = try allocator.dupe(u8, fields.items[2]);
    }

    if (fields.items.len > 3 and fields.items[3].len > 0) {
        item.status = try allocator.dupe(u8, fields.items[3]);
    }

    return item;
}

/// Load account names from Names CSV file
/// REFACTORED: Updated to use BASE_DATA_PATH
pub fn loadAccountNames(allocator: Allocator) !std.ArrayList(models.AccountMaster) {
    const file_path = BASE_DATA_PATH ++ "HKG_PL review Nov'25(Names).csv";
    return loadAccountNamesFromPath(allocator, file_path);
}

/// Load account names from specific path (for testing)
pub fn loadAccountNamesFromPath(
    allocator: Allocator,
    file_path: []const u8,
) !std.ArrayList(models.AccountMaster) {
    var accounts = std.ArrayList(models.AccountMaster).init(allocator);
    errdefer accounts.deinit();

    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(file_content);

    var line_iter = std.mem.splitScalar(u8, file_content, '\n');

    // Skip header row
    _ = line_iter.next();

    // Read data rows
    while (line_iter.next()) |line| {
        if (line.len == 0) continue;

        const account = try parseAccountNameLine(allocator, line) orelse continue;
        try accounts.append(account);
    }

    return accounts;
}

/// Parse an account name CSV line
fn parseAccountNameLine(
    allocator: Allocator,
    line: []const u8,
) !?models.AccountMaster {
    var fields = std.ArrayList([]const u8).init(allocator);
    defer fields.deinit();

    var field_start: usize = 0;
    var in_quotes = false;

    for (line, 0..) |char, i| {
        if (char == '"') {
            in_quotes = !in_quotes;
        } else if (char == ',' and !in_quotes) {
            const field = std.mem.trim(u8, line[field_start..i], " \t\"");
            try fields.append(field);
            field_start = i + 1;
        }
    }
    const last_field = std.mem.trim(u8, line[field_start..], " \t\"\r\n");
    try fields.append(last_field);

    if (fields.items.len < 3) return null;

    var account = models.AccountMaster.init(
        try allocator.dupe(u8, fields.items[0]), // Account code
        try allocator.dupe(u8, fields.items[1]), // Account name
        try allocator.dupe(u8, if (fields.items[2].len > 0) fields.items[2] else "Asset"), // Account type
    );

    if (fields.items.len > 3 and fields.items[3].len > 0) {
        account.ifrs_category = try allocator.dupe(u8, fields.items[3]);
    }

    return account;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Clean amount string by removing commas, spaces, and handling parentheses for negatives
fn cleanAmountString(allocator: Allocator, amount_str: []const u8) ![]u8 {
    var cleaned = std.ArrayList(u8).init(allocator);
    defer cleaned.deinit();

    var is_negative = false;

    for (amount_str) |char| {
        switch (char) {
            '0'...'9', '.', '-', '+' => try cleaned.append(char),
            '(' => is_negative = true,
            ')', ',', ' ', '\t' => {}, // Skip these
            else => {}, // Skip other characters
        }
    }

    // If negative due to parentheses, prepend minus sign
    if (is_negative and cleaned.items.len > 0 and cleaned.items[0] != '-') {
        try cleaned.insert(0, '-');
    }

    return cleaned.toOwnedSlice();
}

/// Count records in CSV file (for validation)
pub fn countRecords(file_path: []const u8, skip_lines: usize) !usize {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(file_content);

    var line_iter = std.mem.splitScalar(u8, file_content, '\n');
    var count: usize = 0;
    var line_num: usize = 0;

    // Skip header lines
    while (line_num < skip_lines) : (line_num += 1) {
        _ = line_iter.next();
    }

    // Count data lines
    while (line_iter.next()) |line| {
        if (line.len > 0) count += 1;
    }

    return count;
}

// ============================================================================
// TESTS
// ============================================================================

test "clean amount string" {
    const testing = std.testing;
    const allocator = testing.allocator;

    {
        const cleaned = try cleanAmountString(allocator, " 1,234,567.89 ");
        defer allocator.free(cleaned);
        try testing.expectEqualStrings("1234567.89", cleaned);
    }

    {
        const cleaned = try cleanAmountString(allocator, " (1,234.56) ");
        defer allocator.free(cleaned);
        try testing.expectEqualStrings("-1234.56", cleaned);
    }
}