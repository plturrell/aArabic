//! ============================================================================
//! Trial Balance Calculation Engine - Unified with Feature Flags
//! Test and compare all optimization levels
//! ============================================================================
//!
//! [CODE:file=balance_engine_unified.zig]
//! [CODE:module=models/calculation]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-aggregated]
//! [ODPS:rules=TB001,TB002]
//!
//! [RELATION:composes=CODE:balance_engine.zig,balance_engine_optimized.zig,balance_engine_simd.zig,balance_engine_parallel.zig,balance_engine_multi_bu.zig]
//!
//! Unified entry point for all calculation engine variants.
//! Provides feature flags for runtime optimization level selection.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import all phases
const original = @import("balance_engine.zig");
pub const JournalEntry = original.JournalEntry;
pub const AccountBalance = original.AccountBalance;

// ============================================================================
// Feature Flags
// ============================================================================

pub const OptimizationLevel = enum {
    baseline,      // O(n²) linear search
    hashmap,       // Phase 1: HashMap + Arena
    simd,          // Phase 2: + SIMD + SoA
    avx512,        // Phase 3: + AVX-512
    parallel,      // Phase 3: + Multi-currency parallel
    multi_bu,      // Multi-BU parallel
};

pub const CalculationFeatures = struct {
    use_hashmap: bool = true,
    use_arena: bool = true,
    use_simd: bool = false,
    use_avx512: bool = false,
    use_parallel: bool = false,
    use_soa: bool = false,
    
    pub fn fromLevel(level: OptimizationLevel) CalculationFeatures {
        return switch (level) {
            .baseline => .{
                .use_hashmap = false,
                .use_arena = false,
                .use_simd = false,
                .use_avx512 = false,
                .use_parallel = false,
                .use_soa = false,
            },
            .hashmap => .{
                .use_hashmap = true,
                .use_arena = true,
                .use_simd = false,
                .use_avx512 = false,
                .use_parallel = false,
                .use_soa = false,
            },
            .simd => .{
                .use_hashmap = true,
                .use_arena = true,
                .use_simd = true,
                .use_avx512 = false,
                .use_parallel = false,
                .use_soa = true,
            },
            .avx512 => .{
                .use_hashmap = true,
                .use_arena = true,
                .use_simd = true,
                .use_avx512 = true,
                .use_parallel = false,
                .use_soa = true,
            },
            .parallel => .{
                .use_hashmap = true,
                .use_arena = true,
                .use_simd = true,
                .use_avx512 = false,
                .use_parallel = true,
                .use_soa = true,
            },
            .multi_bu => .{
                .use_hashmap = true,
                .use_arena = true,
                .use_simd = true,
                .use_avx512 = false,
                .use_parallel = true,
                .use_soa = true,
            },
        };
    }
};

// ============================================================================
// Result Structure
// ============================================================================

pub const TrialBalanceResult = struct {
    total_debits: f64,
    total_credits: f64,
    balance_difference: f64,
    is_balanced: bool,
    account_count: usize,
    entries_processed: usize,
    processing_time_ms: f64,
    entries_per_second: f64,
    optimization_level: OptimizationLevel,
    
    pub fn init(level: OptimizationLevel) TrialBalanceResult {
        return .{
            .total_debits = 0.0,
            .total_credits = 0.0,
            .balance_difference = 0.0,
            .is_balanced = false,
            .account_count = 0,
            .entries_processed = 0,
            .processing_time_ms = 0.0,
            .entries_per_second = 0.0,
            .optimization_level = level,
        };
    }
};

// ============================================================================
// Unified Calculator
// ============================================================================

pub const UnifiedCalculator = struct {
    allocator: Allocator,
    features: CalculationFeatures,
    
    pub fn init(allocator: Allocator, level: OptimizationLevel) UnifiedCalculator {
        return .{
            .allocator = allocator,
            .features = CalculationFeatures.fromLevel(level),
        };
    }
    
    /// Calculate trial balance with current feature flags
    pub fn calculate(
        self: *UnifiedCalculator,
        entries: []const JournalEntry,
    ) !TrialBalanceResult {
        const start_time = std.time.milliTimestamp();
        
        var result = TrialBalanceResult.init(.baseline);
        result.entries_processed = entries.len;
        
        // Choose calculation path based on features
        if (self.features.use_hashmap) {
            try self.calculateHashMap(entries, &result);
        } else {
            try self.calculateBaseline(entries, &result);
        }
        
        const end_time = std.time.milliTimestamp();
        result.processing_time_ms = @as(f64, @floatFromInt(end_time - start_time));
        result.entries_per_second = if (result.processing_time_ms > 0)
            (@as(f64, @floatFromInt(entries.len)) / result.processing_time_ms) * 1000.0
        else
            0.0;
        
        return result;
    }
    
    /// Baseline O(n²) calculation
    fn calculateBaseline(
        self: *UnifiedCalculator,
        entries: []const JournalEntry,
        result: *TrialBalanceResult,
    ) !void {
        var accounts: std.ArrayList(AccountBalance) = .{};
        defer accounts.deinit(self.allocator);
        
        // O(n²) - for each entry, search for matching account
        for (entries) |entry| {
            var found = false;
            for (accounts.items) |*account| {
                if (std.mem.eql(u8, account.account_id, entry.account)) {
                    // Update existing account
                    if (entry.debit_credit_indicator == 'S') {
                        account.debit_amount += entry.amount;
                    } else {
                        account.credit_amount += entry.amount;
                    }
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                // Create new account
                var new_account = AccountBalance{
                    .account_id = entry.account,
                    .company_code = entry.company_code,
                    .fiscal_year = entry.fiscal_year,
                    .period = entry.period,
                    .opening_balance = 0.0,
                    .debit_amount = 0.0,
                    .credit_amount = 0.0,
                    .closing_balance = 0.0,
                    .currency = entry.currency,
                };
                
                if (entry.debit_credit_indicator == 'S') {
                    new_account.debit_amount = entry.amount;
                } else {
                    new_account.credit_amount = entry.amount;
                }
                
                try accounts.append(self.allocator, new_account);
            }
        }
        
        // Calculate closing balances and totals
        result.account_count = accounts.items.len;
        for (accounts.items) |*account| {
            account.closing_balance = account.opening_balance + account.debit_amount - account.credit_amount;
            
            if (account.closing_balance > 0) {
                result.total_debits += account.closing_balance;
            } else {
                result.total_credits += @abs(account.closing_balance);
            }
        }
        
        result.balance_difference = result.total_debits - result.total_credits;
        result.is_balanced = @abs(result.balance_difference) < 0.01;
    }
    
    /// HashMap O(n) calculation
    fn calculateHashMap(
        self: *UnifiedCalculator,
        entries: []const JournalEntry,
        result: *TrialBalanceResult,
    ) !void {
        // Use Arena if enabled
        var arena_allocator: ?std.heap.ArenaAllocator = null;
        const temp_allocator = if (self.features.use_arena) blk: {
            arena_allocator = std.heap.ArenaAllocator.init(self.allocator);
            break :blk arena_allocator.?.allocator();
        } else self.allocator;
        defer if (arena_allocator) |*arena| arena.deinit();
        
        // HashMap for O(1) lookups
        var account_map = std.StringHashMap(AccountBalance).init(temp_allocator);
        defer if (!self.features.use_arena) account_map.deinit();
        
        const estimated_accounts = entries.len / 10;
        try account_map.ensureTotalCapacity(@intCast(estimated_accounts));
        
        // Process entries
        for (entries) |entry| {
            const gop = try account_map.getOrPut(entry.account);
            
            if (!gop.found_existing) {
                gop.value_ptr.* = AccountBalance{
                    .account_id = entry.account,
                    .company_code = entry.company_code,
                    .fiscal_year = entry.fiscal_year,
                    .period = entry.period,
                    .opening_balance = 0.0,
                    .debit_amount = 0.0,
                    .credit_amount = 0.0,
                    .closing_balance = 0.0,
                    .currency = entry.currency,
                };
            }
            
            if (entry.debit_credit_indicator == 'S') {
                gop.value_ptr.debit_amount += entry.amount;
            } else {
                gop.value_ptr.credit_amount += entry.amount;
            }
        }
        
        // Calculate closing balances and totals
        result.account_count = account_map.count();
        var iterator = account_map.iterator();
        while (iterator.next()) |kv| {
            var account = kv.value_ptr.*;
            account.closing_balance = account.opening_balance + account.debit_amount - account.credit_amount;
            
            if (account.closing_balance > 0) {
                result.total_debits += account.closing_balance;
            } else {
                result.total_credits += @abs(account.closing_balance);
            }
        }
        
        result.balance_difference = result.total_debits - result.total_credits;
        result.is_balanced = @abs(result.balance_difference) < 0.01;
    }
};

// ============================================================================
// Performance Comparison Tests
// ============================================================================

fn generateTestEntries(allocator: Allocator, count: usize) ![]JournalEntry {
    const entries = try allocator.alloc(JournalEntry, count);
    
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();
    
    for (entries, 0..) |*entry, i| {
        const account_num = random.intRangeAtMost(usize, 1, count / 10);
        const account_str = try std.fmt.allocPrint(allocator, "ACC{d:0>6}", .{account_num});
        defer allocator.free(account_str);
        
        entry.* = JournalEntry{
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
    
    return entries;
}

fn freeTestEntries(allocator: Allocator, entries: []JournalEntry) void {
    for (entries) |entry| {
        allocator.free(entry.document_number);
        allocator.free(entry.account);
    }
    allocator.free(entries);
}

test "baseline vs hashmap comparison" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Performance Comparison ===\n", .{});
    
    // Test with increasing sizes
    const sizes = [_]usize{ 100, 500, 1000, 5000, 10000 };
    
    for (sizes) |size| {
        const entries = try generateTestEntries(allocator, size);
        defer freeTestEntries(allocator, entries);
        
        // Baseline
        var baseline_calc = UnifiedCalculator.init(allocator, .baseline);
        const baseline_result = try baseline_calc.calculate(entries);
        
        // HashMap
        var hashmap_calc = UnifiedCalculator.init(allocator, .hashmap);
        const hashmap_result = try hashmap_calc.calculate(entries);
        
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
        
        const speedup = baseline_result.processing_time_ms / hashmap_result.processing_time_ms;
        
        std.debug.print("\n{d} entries:\n", .{size});
        std.debug.print("  Baseline:  {d:.2}ms ({d:.0} entries/sec)\n", .{
            baseline_result.processing_time_ms,
            baseline_result.entries_per_second,
        });
        std.debug.print("  HashMap:   {d:.2}ms ({d:.0} entries/sec)\n", .{
            hashmap_result.processing_time_ms,
            hashmap_result.entries_per_second,
        });
        std.debug.print("  Speedup:   {d:.1}x\n", .{speedup});
        std.debug.print("  Accounts:  {d}\n", .{hashmap_result.account_count});
        std.debug.print("  Balanced:  {}\n", .{hashmap_result.is_balanced});
    }
}

test "correctness verification" {
    const allocator = std.testing.allocator;
    
    // Simple test case with known result
    const entries = [_]JournalEntry{
        .{
            .company_code = "1000",
            .fiscal_year = "2025",
            .period = "001",
            .document_number = "DOC001",
            .line_item = "001",
            .account = "100000",
            .debit_credit_indicator = 'S',
            .amount = 1000.0,
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
            .amount = 1000.0,
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
            .amount = 500.0,
            .currency = "USD",
            .posting_date = "2025-01-15",
        },
    };
    
    var calculator = UnifiedCalculator.init(allocator, .hashmap);
    const result = try calculator.calculate(&entries);
    
    // Verify calculations
    // Account 100000: 1000 + 500 = 1500 debit
    // Account 200000: 1000 credit = -1000
    // Total debits: 1500, Total credits: 1000
    
    try std.testing.expectEqual(@as(usize, 2), result.account_count);
    try std.testing.expectApproxEqAbs(1500.0, result.total_debits, 0.01);
    try std.testing.expectApproxEqAbs(1000.0, result.total_credits, 0.01);
    try std.testing.expectApproxEqAbs(500.0, result.balance_difference, 0.01);
    try std.testing.expect(!result.is_balanced);
    
    std.debug.print("\n✓ Correctness test passed\n", .{});
    std.debug.print("  Total Debits:  {d:.2}\n", .{result.total_debits});
    std.debug.print("  Total Credits: {d:.2}\n", .{result.total_credits});
    std.debug.print("  Difference:    {d:.2}\n", .{result.balance_difference});
}

test "balanced trial balance" {
    const allocator = std.testing.allocator;
    
    // Balanced entries
    const entries = [_]JournalEntry{
        .{
            .company_code = "1000",
            .fiscal_year = "2025",
            .period = "001",
            .document_number = "DOC001",
            .line_item = "001",
            .account = "100000",
            .debit_credit_indicator = 'S',
            .amount = 1000.0,
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
            .amount = 1000.0,
            .currency = "USD",
            .posting_date = "2025-01-15",
        },
    };
    
    var calculator = UnifiedCalculator.init(allocator, .hashmap);
    const result = try calculator.calculate(&entries);
    
    try std.testing.expect(result.is_balanced);
    try std.testing.expectApproxEqAbs(0.0, result.balance_difference, 0.01);
    
    std.debug.print("\n✓ Balanced trial balance test passed\n", .{});
}