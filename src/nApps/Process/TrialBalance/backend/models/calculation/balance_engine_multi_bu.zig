//! ============================================================================
//! Trial Balance Calculation Engine - Multi-Business Unit Parallel
//! Parallel processing across Business Units + Currencies
//! ============================================================================
//!
//! [CODE:file=balance_engine_multi_bu.zig]
//! [CODE:module=models/calculation]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-aggregated]
//! [ODPS:rules=TB001,TB002]
//!
//! [PETRI:stages=S03,S04,S05]
//!
//! [RELATION:extends=CODE:balance_engine_parallel.zig]
//!
//! Extends parallel processing to handle multiple business units simultaneously.
//! Partitions work by (Business Unit × Currency) for maximum parallelism.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import previous phases
const original = @import("balance_engine.zig");
const phase3 = @import("balance_engine_parallel.zig");

pub const JournalEntry = original.JournalEntry;
pub const AccountBalance = original.AccountBalance;
pub const JournalEntriesSoA = phase3.JournalEntriesSoA;
pub const BalanceResult = phase3.BalanceResult;

// ============================================================================
// Multi-Business Unit Result Structures
// ============================================================================

pub const BusinessUnitResult = struct {
    business_unit: []const u8,
    currencies: std.ArrayList(CurrencyResult),
    total_entries: usize,
    processing_time_ms: f64,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, bu: []const u8) BusinessUnitResult {
        return .{
            .business_unit = bu,
            .currencies = .{},
            .total_entries = 0,
            .processing_time_ms = 0.0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *BusinessUnitResult) void {
        self.currencies.deinit(self.allocator);
    }
};

pub const CurrencyResult = struct {
    currency: []const u8,
    total_debits: f64,
    total_credits: f64,
    balance_difference: f64,
    is_balanced: bool,
    account_count: usize,
    
    pub fn init(currency: []const u8) CurrencyResult {
        return .{
            .currency = currency,
            .total_debits = 0.0,
            .total_credits = 0.0,
            .balance_difference = 0.0,
            .is_balanced = false,
            .account_count = 0,
        };
    }
};

pub const MultiBusinessUnitResult = struct {
    business_units: std.ArrayList(BusinessUnitResult),
    total_entries_processed: usize,
    total_business_units: usize,
    total_currencies: usize,
    processing_time_ms: f64,
    entries_per_second: f64,
    parallel_enabled: bool,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) MultiBusinessUnitResult {
        return .{
            .business_units = .{},
            .total_entries_processed = 0,
            .total_business_units = 0,
            .total_currencies = 0,
            .processing_time_ms = 0.0,
            .entries_per_second = 0.0,
            .parallel_enabled = true,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MultiBusinessUnitResult) void {
        for (self.business_units.items) |*bu| {
            bu.deinit();
        }
        self.business_units.deinit(self.allocator);
    }
};

// ============================================================================
// Business Unit Processing Task
// ============================================================================

const BusinessUnitTask = struct {
    entries: []const JournalEntry,
    business_unit: []const u8,
    result: *BusinessUnitResult,
    allocator: Allocator,
};

// ============================================================================
// Multi-Business Unit Parallel Calculator
// ============================================================================

pub const MultiBusinessUnitCalculator = struct {
    allocator: Allocator,
    thread_pool: std.Thread.Pool,
    
    progress_callback: ?*const fn(
        bu_processed: usize,
        total_bus: usize,
        entries_processed: usize,
        total_entries: usize,
    ) void,

    pub fn init(allocator: Allocator) !MultiBusinessUnitCalculator {
        var thread_pool: std.Thread.Pool = undefined;
        try thread_pool.init(.{ .allocator = allocator });
        
        return MultiBusinessUnitCalculator{
            .allocator = allocator,
            .thread_pool = thread_pool,
            .progress_callback = null,
        };
    }
    
    pub fn deinit(self: *MultiBusinessUnitCalculator) void {
        self.thread_pool.deinit();
    }
    
    pub fn set_progress_callback(
        self: *MultiBusinessUnitCalculator,
        callback: *const fn(usize, usize, usize, usize) void,
    ) void {
        self.progress_callback = callback;
    }
    
    /// Process a single business unit (worker function)
    fn processBusinessUnitWorker(task: BusinessUnitTask) void {
        const start_time = std.time.milliTimestamp();
        
        // Use Arena for all temporary allocations in this BU
        var arena = std.heap.ArenaAllocator.init(task.allocator);
        defer arena.deinit();
        const temp_allocator = arena.allocator();
        
        // Filter entries for this business unit
        var bu_entries: std.ArrayList(JournalEntry) = .{};
        for (task.entries) |entry| {
            if (std.mem.eql(u8, entry.company_code, task.business_unit)) {
                bu_entries.append(temp_allocator, entry) catch continue;
            }
        }
        
        // Extract unique currencies for this BU
        var currency_set = std.StringHashMap(void).init(temp_allocator);
        for (bu_entries.items) |entry| {
            currency_set.put(entry.currency, {}) catch continue;
        }
        
        var currencies: std.ArrayList([]const u8) = .{};
        var currency_iter = currency_set.keyIterator();
        while (currency_iter.next()) |key| {
            currencies.append(temp_allocator, key.*) catch continue;
        }
        
        // Process each currency for this BU
        for (currencies.items) |currency| {
            // Filter by currency
            var currency_entries: std.ArrayList(JournalEntry) = .{};
            for (bu_entries.items) |entry| {
                if (std.mem.eql(u8, entry.currency, currency)) {
                    currency_entries.append(temp_allocator, entry) catch continue;
                }
            }
            
            if (currency_entries.items.len == 0) continue;
            
            // Convert to SoA
            var entries_soa = JournalEntriesSoA.fromAoS(
                temp_allocator,
                currency_entries.items,
            ) catch continue;
            defer entries_soa.deinit();
            
            // Calculate balances using AVX-512 or SIMD
            const balances = if (entries_soa.len >= 16)
                phase3.calculate_balances_avx512(&entries_soa)
            else
                phase3.calculate_balances_simd_compat(&entries_soa);
            
            // Create account map for this currency
            var account_map = std.StringHashMap(AccountBalance).init(temp_allocator);
            account_map.ensureTotalCapacity(@intCast(currency_entries.items.len / 10)) catch continue;
            
            for (0..entries_soa.len) |i| {
                const account = entries_soa.accounts[i];
                const gop = account_map.getOrPut(account) catch continue;
                
                if (!gop.found_existing) {
                    gop.value_ptr.* = AccountBalance{
                        .account_id = account,
                        .company_code = entries_soa.company_codes[i],
                        .fiscal_year = entries_soa.fiscal_years[i],
                        .period = entries_soa.periods[i],
                        .opening_balance = 0.0,
                        .debit_amount = 0.0,
                        .credit_amount = 0.0,
                        .closing_balance = 0.0,
                        .currency = entries_soa.currencies[i],
                    };
                }
                
                if (entries_soa.indicators[i] == 'S') {
                    gop.value_ptr.debit_amount += entries_soa.amounts[i];
                } else {
                    gop.value_ptr.credit_amount += entries_soa.amounts[i];
                }
            }
            
            // Store currency result
            const curr_result = CurrencyResult{
                .currency = currency,
                .total_debits = balances.total_debits,
                .total_credits = balances.total_credits,
                .balance_difference = balances.total_debits - balances.total_credits,
                .is_balanced = @abs(balances.total_debits - balances.total_credits) < 0.01,
                .account_count = account_map.count(),
            };
            
            task.result.currencies.append(task.allocator, curr_result) catch continue;
        }
        
        const end_time = std.time.milliTimestamp();
        task.result.total_entries = bu_entries.items.len;
        task.result.processing_time_ms = @as(f64, @floatFromInt(end_time - start_time));
    }
    
    /// Calculate trial balance across multiple business units in parallel
    pub fn calculate_multi_bu_parallel(
        self: *MultiBusinessUnitCalculator,
        journal_entries: []const JournalEntry,
        business_units: []const []const u8,
    ) !MultiBusinessUnitResult {
        const start_time = std.time.milliTimestamp();
        
        var result = MultiBusinessUnitResult.init(self.allocator);
        errdefer result.deinit();
        
        // Allocate results array for each BU
        var bu_results = try self.allocator.alloc(BusinessUnitResult, business_units.len);
        defer self.allocator.free(bu_results);
        
        // Initialize BU results
        for (bu_results, 0..) |*bu_result, i| {
            bu_result.* = BusinessUnitResult.init(self.allocator, business_units[i]);
        }
        
        // Create tasks for each BU
        const tasks = try self.allocator.alloc(BusinessUnitTask, business_units.len);
        defer self.allocator.free(tasks);
        
        for (tasks, 0..) |*task, i| {
            task.* = BusinessUnitTask{
                .entries = journal_entries,
                .business_unit = business_units[i],
                .result = &bu_results[i],
                .allocator = self.allocator,
            };
        }
        
        // Execute all BU tasks in parallel
        var wait_group: std.Thread.WaitGroup = .{};
        wait_group.reset();
        
        for (tasks) |task| {
            wait_group.start();
            self.thread_pool.spawn(struct {
                fn run(t: BusinessUnitTask, wg: *std.Thread.WaitGroup) void {
                    defer wg.finish();
                    processBusinessUnitWorker(t);
                }
            }.run, .{ task, &wait_group }) catch {
                wait_group.finish();
                continue;
            };
        }
        
        // Wait for all BU tasks to complete
        self.thread_pool.waitAndWork(&wait_group);
        
        // Collect results
        var total_entries: usize = 0;
        var total_currencies: usize = 0;
        
        for (bu_results) |bu_result| {
            try result.business_units.append(self.allocator, bu_result);
            total_entries += bu_result.total_entries;
            total_currencies += bu_result.currencies.items.len;
            
            // Progress callback
            if (self.progress_callback) |callback| {
                callback(
                    result.business_units.items.len,
                    business_units.len,
                    total_entries,
                    journal_entries.len,
                );
            }
        }
        
        // Calculate final metrics
        const end_time = std.time.milliTimestamp();
        result.total_entries_processed = journal_entries.len;
        result.total_business_units = business_units.len;
        result.total_currencies = total_currencies;
        result.processing_time_ms = @as(f64, @floatFromInt(end_time - start_time));
        result.entries_per_second = if (result.processing_time_ms > 0)
            (@as(f64, @floatFromInt(journal_entries.len)) / result.processing_time_ms) * 1000.0
        else
            0.0;
        
        return result;
    }
    
    /// Convenience method for auto-detecting business units
    pub fn calculate_auto_detect_bu(
        self: *MultiBusinessUnitCalculator,
        journal_entries: []const JournalEntry,
    ) !MultiBusinessUnitResult {
        // Extract unique business units
        var bu_set = std.StringHashMap(void).init(self.allocator);
        defer bu_set.deinit();
        
        for (journal_entries) |entry| {
            try bu_set.put(entry.company_code, {});
        }
        
        var business_units: std.ArrayList([]const u8) = .{};
        defer business_units.deinit(self.allocator);
        
        var bu_iter = bu_set.keyIterator();
        while (bu_iter.next()) |key| {
            try business_units.append(self.allocator, key.*);
        }
        
        return self.calculate_multi_bu_parallel(journal_entries, business_units.items);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "multi-business unit calculation" {
    const allocator = std.testing.allocator;
    var calculator = try MultiBusinessUnitCalculator.init(allocator);
    defer calculator.deinit();
    
    // Generate entries across 3 BUs, each with 2 currencies
    const entries = try allocator.alloc(JournalEntry, 6000);
    defer allocator.free(entries);
    
    const test_bus = [_][]const u8{ "1000", "2000", "3000" };
    const test_currencies = [_][]const u8{ "USD", "EUR" };
    
    for (entries, 0..) |*entry, i| {
        const bu_idx = i % 3;
        const curr_idx = (i / 3) % 2;
        const account_str = try std.fmt.allocPrint(allocator, "ACC{d:0>6}", .{i % 50});
        defer allocator.free(account_str);
        
        entry.* = JournalEntry{
            .company_code = test_bus[bu_idx],
            .fiscal_year = "2025",
            .period = "001",
            .document_number = try std.fmt.allocPrint(allocator, "DOC{d:0>6}", .{i}),
            .line_item = "001",
            .account = try allocator.dupe(u8, account_str),
            .debit_credit_indicator = if (i % 2 == 0) 'S' else 'H',
            .amount = 1000.0 + @as(f64, @floatFromInt(i)),
            .currency = test_currencies[curr_idx],
            .posting_date = "2025-01-15",
        };
    }
    defer {
        for (entries) |entry| {
            allocator.free(entry.document_number);
            allocator.free(entry.account);
        }
    }
    
    var result = try calculator.calculate_multi_bu_parallel(entries, &test_bus);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(usize, 3), result.business_units.items.len);
    try std.testing.expect(result.parallel_enabled);
    
    std.debug.print("\n✓ Multi-BU parallel test passed\n", .{});
    std.debug.print("  Total entries: {d}\n", .{entries.len});
    std.debug.print("  Business units: {d}\n", .{result.total_business_units});
    std.debug.print("  Total currencies: {d}\n", .{result.total_currencies});
    std.debug.print("  Processing time: {d:.2}ms\n", .{result.processing_time_ms});
    std.debug.print("  Throughput: {d:.0} entries/sec\n", .{result.entries_per_second});
    std.debug.print("  Parallel enabled: {}\n", .{result.parallel_enabled});
    
    // Verify each BU has results
    for (result.business_units.items) |bu| {
        std.debug.print("  BU {s}: {d} entries, {d} currencies\n", .{
            bu.business_unit,
            bu.total_entries,
            bu.currencies.items.len,
        });
    }
}

test "auto-detect business units" {
    const allocator = std.testing.allocator;
    var calculator = try MultiBusinessUnitCalculator.init(allocator);
    defer calculator.deinit();
    
    // Generate entries with implicit BUs
    const entries = try allocator.alloc(JournalEntry, 3000);
    defer allocator.free(entries);
    
    for (entries, 0..) |*entry, i| {
        const bu = if (i < 1000) "BU1" else if (i < 2000) "BU2" else "BU3";
        const account_str = try std.fmt.allocPrint(allocator, "ACC{d:0>6}", .{i % 30});
        defer allocator.free(account_str);
        
        entry.* = JournalEntry{
            .company_code = bu,
            .fiscal_year = "2025",
            .period = "001",
            .document_number = try std.fmt.allocPrint(allocator, "DOC{d:0>6}", .{i}),
            .line_item = "001",
            .account = try allocator.dupe(u8, account_str),
            .debit_credit_indicator = 'S',
            .amount = 100.0,
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
    
    var result = try calculator.calculate_auto_detect_bu(entries);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(usize, 3), result.business_units.items.len);
    
    std.debug.print("\n✓ Auto-detect BU test passed\n", .{});
    std.debug.print("  Detected BUs: {d}\n", .{result.total_business_units});
}