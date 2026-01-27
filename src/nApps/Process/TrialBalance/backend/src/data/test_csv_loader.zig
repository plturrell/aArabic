//! ============================================================================
//! Test CSV Loader with Real Hong Kong Sample Data
//! Tests loading HKG_PL review Nov'25 data files
//! ============================================================================
//!
//! [CODE:file=test_csv_loader.zig]
//! [CODE:module=data]
//! [CODE:language=zig]
//!
//! [RELATION:tests=CODE:csv_loader.zig]
//! [RELATION:tests=CODE:trial_balance_models.zig]
//!
//! Note: Test code - validates CSV loading functionality with real HKG data.

const std = @import("std");
const csv_loader = @import("csv_loader");
const models = @import("trial_balance_models");

test "load HKG MTD-TB CSV file" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    // Path to the HKG sample data
    const csv_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(MTD-TB).csv";
    
    // Try to load the data
    var entries = csv_loader.loadTrialBalanceData(allocator, csv_path) catch |err| {
        std.debug.print("Failed to load CSV: {}\n", .{err});
        return err;
    };
    defer entries.deinit(allocator);
    defer {
        for (entries.items) |entry| {
            allocator.free(entry.account_code);
            allocator.free(entry.account_name);
            allocator.free(entry.account_type);
            allocator.free(entry.business_unit);
            allocator.free(entry.fiscal_period);
            if (entry.ifrs_category.len > 0) allocator.free(entry.ifrs_category);
            if (entry.currency_code.len > 0) allocator.free(entry.currency_code);
        }
    }
    
    // Verify we loaded records
    std.debug.print("\n=== HKG MTD-TB Load Test ===\n", .{});
    std.debug.print("Records loaded: {d}\n", .{entries.items.len});
    
    // Should have loaded records
    try testing.expect(entries.items.len > 0);
    
    // Check first entry structure
    if (entries.items.len > 0) {
        const first = entries.items[0];
        std.debug.print("\nFirst Entry:\n", .{});
        std.debug.print("  Account Code: {s}\n", .{first.account_code});
        std.debug.print("  Account Name: {s}\n", .{first.account_name});
        std.debug.print("  Account Type: {s}\n", .{first.account_type});
        std.debug.print("  Business Unit: {s}\n", .{first.business_unit});
        std.debug.print("  Fiscal Period: {s}\n", .{first.fiscal_period});
        std.debug.print("  Currency: {s}\n", .{first.currency_code});
        std.debug.print("  Closing Balance: {d:.2}\n", .{first.closing_balance});
        std.debug.print("  IFRS Category: {s}\n", .{first.ifrs_category});
        
        // Verify field types are correct
        try testing.expect(first.account_code.len > 0);
        try testing.expect(first.account_name.len > 0);
        try testing.expect(first.business_unit.len > 0);
        try testing.expect(first.fiscal_period.len > 0);
    }
    
    // Show sample of entries
    std.debug.print("\nSample Entries (first 5):\n", .{});
    const sample_count = @min(5, entries.items.len);
    for (entries.items[0..sample_count], 0..) |entry, i| {
        std.debug.print("  {d}. {s} | {s} | {s} | {d:.2}\n", .{
            i + 1,
            entry.account_code,
            entry.account_name,
            entry.account_type,
            entry.closing_balance,
        });
    }
}

test "count records in HKG MTD-TB CSV" {
    const testing = std.testing;
    
    const csv_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(MTD-TB).csv";
    
    const count = csv_loader.countRecords(csv_path, 9) catch |err| {
        std.debug.print("Failed to count records: {}\n", .{err});
        return err;
    };
    
    std.debug.print("\n=== Record Count Test ===\n", .{});
    std.debug.print("Total data records: {d}\n", .{count});
    
    try testing.expect(count > 0);
}

test "load HKG variance CSV file" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const csv_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(PL Variance).csv";
    
    var variances = csv_loader.loadVarianceData(allocator, csv_path) catch |err| {
        std.debug.print("Failed to load variance CSV: {}\n", .{err});
        return err;
    };
    defer variances.deinit(allocator);
    defer {
        for (variances.items) |variance| {
            allocator.free(variance.account_code);
            allocator.free(variance.account_name);
            if (variance.account_type.len > 0) allocator.free(variance.account_type);
        }
    }
    
    std.debug.print("\n=== HKG Variance Load Test ===\n", .{});
    std.debug.print("Variance records loaded: {d}\n", .{variances.items.len});
    
    try testing.expect(variances.items.len > 0);
    
    if (variances.items.len > 0) {
        const first = variances.items[0];
        std.debug.print("\nFirst Variance:\n", .{});
        std.debug.print("  Account: {s} - {s}\n", .{ first.account_code, first.account_name });
        std.debug.print("  Current Period: {s}\n", .{first.current_period});
        std.debug.print("  Previous Period: {s}\n", .{first.previous_period});
    }
}

test "load HKG checklist CSV file" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const csv_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(Checklist).csv";
    
    var items = csv_loader.loadChecklistData(allocator, csv_path) catch |err| {
        std.debug.print("Failed to load checklist CSV: {}\n", .{err});
        return err;
    };
    defer items.deinit(allocator);
    defer {
        for (items.items) |item| {
            allocator.free(item.id);
            allocator.free(item.stage_id);
            allocator.free(item.title);
            if (item.description.len > 0) allocator.free(item.description);
        }
    }
    
    std.debug.print("\n=== HKG Checklist Load Test ===\n", .{});
    std.debug.print("Checklist items loaded: {d}\n", .{items.items.len});
    
    try testing.expect(items.items.len > 0);
    
    if (items.items.len > 0) {
        const first = items.items[0];
        std.debug.print("\nFirst Checklist Item:\n", .{});
        std.debug.print("  ID: {s}\n", .{first.id});
        std.debug.print("  Stage: {s}\n", .{first.stage_id});
        std.debug.print("  Title: {s}\n", .{first.title});
        std.debug.print("  Status: {s}\n", .{first.status});
    }
}

test "load HKG account names CSV file" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const csv_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(Names).csv";
    
    var accounts = csv_loader.loadAccountNames(allocator, csv_path) catch |err| {
        std.debug.print("Failed to load names CSV: {}\n", .{err});
        return err;
    };
    defer accounts.deinit(allocator);
    defer {
        for (accounts.items) |account| {
            allocator.free(account.account_code);
            allocator.free(account.account_name);
            allocator.free(account.account_type);
            if (account.ifrs_category.len > 0) allocator.free(account.ifrs_category);
        }
    }
    
    std.debug.print("\n=== HKG Account Names Load Test ===\n", .{});
    std.debug.print("Account names loaded: {d}\n", .{accounts.items.len});
    
    try testing.expect(accounts.items.len > 0);
    
    if (accounts.items.len > 0) {
        const first = accounts.items[0];
        std.debug.print("\nFirst Account:\n", .{});
        std.debug.print("  Code: {s}\n", .{first.account_code});
        std.debug.print("  Name: {s}\n", .{first.account_name});
        std.debug.print("  Type: {s}\n", .{first.account_type});
    }
}

test "validate data types from HKG data" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const csv_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(MTD-TB).csv";
    
    var entries = csv_loader.loadTrialBalanceData(allocator, csv_path) catch |err| {
        std.debug.print("Failed to load CSV for validation: {}\n", .{err});
        return err;
    };
    defer entries.deinit(allocator);
    defer {
        for (entries.items) |entry| {
            allocator.free(entry.account_code);
            allocator.free(entry.account_name);
            allocator.free(entry.account_type);
            allocator.free(entry.business_unit);
            allocator.free(entry.fiscal_period);
            if (entry.ifrs_category.len > 0) allocator.free(entry.ifrs_category);
            if (entry.currency_code.len > 0) allocator.free(entry.currency_code);
        }
    }
    
    std.debug.print("\n=== Data Type Validation Test ===\n", .{});
    
    var valid_count: usize = 0;
    var invalid_count: usize = 0;
    
    for (entries.items) |entry| {
        // Validate required fields
        const is_valid = entry.account_code.len > 0 and
            entry.account_name.len > 0 and
            entry.business_unit.len > 0 and
            entry.fiscal_period.len > 0;
        
        if (is_valid) {
            valid_count += 1;
        } else {
            invalid_count += 1;
        }
    }
    
    std.debug.print("Valid entries: {d}\n", .{valid_count});
    std.debug.print("Invalid entries: {d}\n", .{invalid_count});
    
    // Calculate data quality percentage
    const total = valid_count + invalid_count;
    const quality_pct = if (total > 0) (@as(f64, @floatFromInt(valid_count)) / @as(f64, @floatFromInt(total))) * 100.0 else 0.0;
    std.debug.print("Data quality: {d:.2}%\n", .{quality_pct});
    
    // Should have mostly valid data
    try testing.expect(quality_pct > 90.0);
}

test "integration test - load all HKG files" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    std.debug.print("\n=== Integration Test: Load All HKG Files ===\n", .{});
    
    // Load all files
    const tb_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(MTD-TB).csv";
    const var_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(PL Variance).csv";
    const check_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(Checklist).csv";
    const names_path = "BusDocs/sample-data/extracted/HKG_PL review Nov'25(Names).csv";
    
    var tb_entries = csv_loader.loadTrialBalanceData(allocator, tb_path) catch |err| {
        std.debug.print("Failed to load TB: {}\n", .{err});
        return err;
    };
    defer tb_entries.deinit(allocator);
    defer {
        for (tb_entries.items) |entry| {
            allocator.free(entry.account_code);
            allocator.free(entry.account_name);
            allocator.free(entry.account_type);
            allocator.free(entry.business_unit);
            allocator.free(entry.fiscal_period);
            if (entry.ifrs_category.len > 0) allocator.free(entry.ifrs_category);
            if (entry.currency_code.len > 0) allocator.free(entry.currency_code);
        }
    }
    
    var variances = csv_loader.loadVarianceData(allocator, var_path) catch |err| {
        std.debug.print("Failed to load variances: {}\n", .{err});
        return err;
    };
    defer variances.deinit(allocator);
    defer {
        for (variances.items) |v| {
            allocator.free(v.account_code);
            allocator.free(v.account_name);
            if (v.account_type.len > 0) allocator.free(v.account_type);
        }
    }
    
    var checklist = csv_loader.loadChecklistData(allocator, check_path) catch |err| {
        std.debug.print("Failed to load checklist: {}\n", .{err});
        return err;
    };
    defer checklist.deinit(allocator);
    defer {
        for (checklist.items) |item| {
            allocator.free(item.id);
            allocator.free(item.stage_id);
            allocator.free(item.title);
            if (item.description.len > 0) allocator.free(item.description);
        }
    }
    
    var accounts = csv_loader.loadAccountNames(allocator, names_path) catch |err| {
        std.debug.print("Failed to load account names: {}\n", .{err});
        return err;
    };
    defer accounts.deinit(allocator);
    defer {
        for (accounts.items) |acc| {
            allocator.free(acc.account_code);
            allocator.free(acc.account_name);
            allocator.free(acc.account_type);
            if (acc.ifrs_category.len > 0) allocator.free(acc.ifrs_category);
        }
    }
    
    // Print summary
    std.debug.print("\nSummary:\n", .{});
    std.debug.print("  Trial Balance Entries: {d}\n", .{tb_entries.items.len});
    std.debug.print("  Variance Entries: {d}\n", .{variances.items.len});
    std.debug.print("  Checklist Items: {d}\n", .{checklist.items.len});
    std.debug.print("  Account Names: {d}\n", .{accounts.items.len});
    
    // All should have loaded successfully
    try testing.expect(tb_entries.items.len > 0);
    try testing.expect(variances.items.len > 0);
    try testing.expect(checklist.items.len > 0);
    try testing.expect(accounts.items.len > 0);
    
    std.debug.print("\n✅ All HKG files loaded successfully!\n", .{});
}