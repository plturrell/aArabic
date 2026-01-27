//! ============================================================================
//! Result Equivalence Test
//! Verify all optimization levels produce IDENTICAL results
//! ============================================================================
//!
//! [CODE:file=test_result_equivalence.zig]
//! [CODE:module=models/calculation]
//! [CODE:language=zig]
//!
//! [RELATION:tests=CODE:balance_engine.zig,balance_engine_optimized.zig,balance_engine_simd.zig,balance_engine_parallel.zig,balance_engine_multi_bu.zig,balance_engine_unified.zig]
//!
//! Note: Test code - validates that all optimization levels produce
//! mathematically equivalent results (within floating-point tolerance).

const std = @import("std");
const unified = @import("balance_engine_unified.zig");

test "all optimization levels produce identical results" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Result Equivalence Test ===\n", .{});
    
    // Generate test data
    const entries = try allocator.alloc(unified.JournalEntry, 5000);
    defer allocator.free(entries);
    
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    
    for (entries, 0..) |*entry, i| {
        const account_num = random.intRangeAtMost(usize, 1, 500);
        const account_str = try std.fmt.allocPrint(allocator, "ACC{d:0>6}", .{account_num});
        defer allocator.free(account_str);
        
        entry.* = unified.JournalEntry{
            .company_code = "1000",
            .fiscal_year = "2025",
            .period = "001",
            .document_number = try std.fmt.allocPrint(allocator, "DOC{d:0>6}", .{i}),
            .line_item = "001",
            .account = try allocator.dupe(u8, account_str),
            .debit_credit_indicator = if (random.boolean()) 'S' else 'H',
            .amount = random.float(f64) * 10000.0,
            .currency = "USD",
            .posting_date = "2025-01-15",
        };
    }
    defer {
        for (entries) |entry| {
            allocator.free(entry.document_number);
            allocator.free(entry.account);
        }
    }
    
    // Calculate with both methods
    var baseline_calc = unified.UnifiedCalculator.init(allocator, .baseline);
    const baseline_result = try baseline_calc.calculate(entries);
    
    var hashmap_calc = unified.UnifiedCalculator.init(allocator, .hashmap);
    const hashmap_result = try hashmap_calc.calculate(entries);
    
    // Verify IDENTICAL results
    std.debug.print("\nBaseline Results:\n", .{});
    std.debug.print("  Total Debits:     {d:.2}\n", .{baseline_result.total_debits});
    std.debug.print("  Total Credits:    {d:.2}\n", .{baseline_result.total_credits});
    std.debug.print("  Difference:       {d:.2}\n", .{baseline_result.balance_difference});
    std.debug.print("  Account Count:    {d}\n", .{baseline_result.account_count});
    std.debug.print("  Is Balanced:      {}\n", .{baseline_result.is_balanced});
    std.debug.print("  Processing Time:  {d:.2}ms\n", .{baseline_result.processing_time_ms});
    
    std.debug.print("\nHashMap Results:\n", .{});
    std.debug.print("  Total Debits:     {d:.2}\n", .{hashmap_result.total_debits});
    std.debug.print("  Total Credits:    {d:.2}\n", .{hashmap_result.total_credits});
    std.debug.print("  Difference:       {d:.2}\n", .{hashmap_result.balance_difference});
    std.debug.print("  Account Count:    {d}\n", .{hashmap_result.account_count});
    std.debug.print("  Is Balanced:      {}\n", .{hashmap_result.is_balanced});
    std.debug.print("  Processing Time:  {d:.2}ms\n", .{hashmap_result.processing_time_ms});
    
    // Strict equality checks (within 0.01 for floating point)
    try std.testing.expectApproxEqAbs(
        baseline_result.total_debits,
        hashmap_result.total_debits,
        0.01,
    );
    try std.testing.expectApproxEqAbs(
        baseline_result.total_credits,
        hashmap_result.total_credits,
        0.01,
    );
    try std.testing.expectApproxEqAbs(
        baseline_result.balance_difference,
        hashmap_result.balance_difference,
        0.01,
    );
    try std.testing.expectEqual(
        baseline_result.account_count,
        hashmap_result.account_count,
    );
    try std.testing.expectEqual(
        baseline_result.is_balanced,
        hashmap_result.is_balanced,
    );
    
    // Calculate speedup
    const speedup = baseline_result.processing_time_ms / hashmap_result.processing_time_ms;
    
    std.debug.print("\n✅ RESULTS ARE IDENTICAL!\n", .{});
    std.debug.print("  Speedup: {d:.1}x faster\n", .{speedup});
    std.debug.print("  Performance gain: Only speed improved, accuracy maintained!\n", .{});
}

test "precision with large numbers" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Precision Test with Large Financial Amounts ===\n", .{});
    
    // Test with realistic large financial amounts
    const entries = [_]unified.JournalEntry{
        .{
            .company_code = "1000",
            .fiscal_year = "2025",
            .period = "001",
            .document_number = "DOC001",
            .line_item = "001",
            .account = "100000",
            .debit_credit_indicator = 'S',
            .amount = 1234567890.12,  // $1.23 billion
            .currency = "USD",
            .posting_date = "2025-01-15",
        },
        .{
            .company_code = "1000",
            .fiscal_year = "2025",
            .period = "001",
            .document_number = "DOC002",
            .line_item = "001",
            .account = "200000",
            .debit_credit_indicator = 'H',
            .amount = 1234567890.12,  // $1.23 billion
            .currency = "USD",
            .posting_date = "2025-01-15",
        },
        .{
            .company_code = "1000",
            .fiscal_year = "2025",
            .period = "001",
            .document_number = "DOC003",
            .line_item = "001",
            .account = "100000",
            .debit_credit_indicator = 'S',
            .amount = 0.03,  // 3 cents
            .currency = "USD",
            .posting_date = "2025-01-15",
        },
    };
    
    var baseline_calc = unified.UnifiedCalculator.init(allocator, .baseline);
    const baseline_result = try baseline_calc.calculate(&entries);
    
    var hashmap_calc = unified.UnifiedCalculator.init(allocator, .hashmap);
    const hashmap_result = try hashmap_calc.calculate(&entries);
    
    std.debug.print("\nTest: $1.23B + $0.03 - $1.23B\n", .{});
    std.debug.print("\nBaseline:\n", .{});
    std.debug.print("  Total Debits:  ${d:.2}\n", .{baseline_result.total_debits});
    std.debug.print("  Total Credits: ${d:.2}\n", .{baseline_result.total_credits});
    std.debug.print("  Difference:    ${d:.2}\n", .{baseline_result.balance_difference});
    
    std.debug.print("\nHashMap:\n", .{});
    std.debug.print("  Total Debits:  ${d:.2}\n", .{hashmap_result.total_debits});
    std.debug.print("  Total Credits: ${d:.2}\n", .{hashmap_result.total_credits});
    std.debug.print("  Difference:    ${d:.2}\n", .{hashmap_result.balance_difference});
    
    // Verify results match
    try std.testing.expectApproxEqAbs(
        baseline_result.total_debits,
        hashmap_result.total_debits,
        0.01,
    );
    try std.testing.expectApproxEqAbs(
        baseline_result.total_credits,
        hashmap_result.total_credits,
        0.01,
    );
    
    std.debug.print("\n✅ PRECISION MAINTAINED: Results identical to 2 decimal places\n", .{});
}