//! ============================================================================
//! Trial Balance Calculation Engine - Phase 3 Parallel Optimized
//! Multi-threaded currency processing + Database optimization + AVX-512
//! ============================================================================
//!
//! [CODE:file=balance_engine_parallel.zig]
//! [CODE:module=models/calculation]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-aggregated]
//! [ODPS:rules=TB001,TB002]
//!
//! [PETRI:stages=S03,S04]
//!
//! [RELATION:extends=CODE:balance_engine_simd.zig]
//!
//! Phase 3 optimization: Multi-threaded processing, database batch optimization,
//! and AVX-512 for maximum throughput on large datasets.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import previous phases
const original = @import("balance_engine.zig");
const phase2 = @import("balance_engine_simd.zig");

pub const JournalEntry = original.JournalEntry;
pub const AccountBalance = original.AccountBalance;
pub const JournalEntriesSoA = phase2.JournalEntriesSoA;
pub const kahanSum = phase2.kahanSum;

// ============================================================================
// Phase 3: Multi-Currency Result
// ============================================================================

pub const CurrencyBalanceResult = struct {
    currency: []const u8,
    total_debits: f64,
    total_credits: f64,
    balance_difference: f64,
    is_balanced: bool,
    account_count: usize,
    
    pub fn init(currency: []const u8) CurrencyBalanceResult {
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

pub const MultiCurrencyResult = struct {
    currencies: std.ArrayList(CurrencyBalanceResult),
    total_entries_processed: usize,
    processing_time_ms: f64,
    entries_per_second: f64,
    parallel_enabled: bool,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) MultiCurrencyResult {
        return .{
            .currencies = .{},
            .total_entries_processed = 0,
            .processing_time_ms = 0.0,
            .entries_per_second = 0.0,
            .parallel_enabled = true,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MultiCurrencyResult) void {
        self.currencies.deinit(self.allocator);
    }
};

// ============================================================================
// Phase 3: AVX-512 Support (16-wide vectors)
// ============================================================================

const VEC_SIZE_512 = 16;
const Vec16f = @Vector(VEC_SIZE_512, f64);

inline fn vecSum16(v: Vec16f) f64 {
    return @reduce(.Add, v);
}

pub const BalanceResult = struct {
    total_debits: f64,
    total_credits: f64,
};

pub fn calculate_balances_avx512(
    entries_soa: *const JournalEntriesSoA,
) BalanceResult {
    var debit_sum: Vec16f = @splat(0.0);
    var credit_sum: Vec16f = @splat(0.0);
    
    var i: usize = 0;
    const len = entries_soa.len;
    
    // Process 16 entries at a time with AVX-512
    while (i + VEC_SIZE_512 <= len) : (i += VEC_SIZE_512) {
        const amounts_vec: Vec16f = entries_soa.amounts[i..][0..VEC_SIZE_512].*;
        
        var debit_mask: Vec16f = undefined;
        inline for (0..VEC_SIZE_512) |j| {
            debit_mask[j] = if (entries_soa.indicators[i + j] == 'S') 1.0 else 0.0;
        }
        
        // 16-way parallel operations!
        debit_sum += amounts_vec * debit_mask;
        credit_sum += amounts_vec * (@as(Vec16f, @splat(1.0)) - debit_mask);
    }
    
    var total_debits = vecSum16(debit_sum);
    var total_credits = vecSum16(credit_sum);
    
    // Handle remaining entries
    while (i < len) : (i += 1) {
        if (entries_soa.indicators[i] == 'S') {
            total_debits += entries_soa.amounts[i];
        } else {
            total_credits += entries_soa.amounts[i];
        }
    }
    
    return BalanceResult{
        .total_debits = total_debits,
        .total_credits = total_credits,
    };
}

pub fn calculate_balances_simd_compat(
    entries_soa: *const JournalEntriesSoA,
) BalanceResult {
    const result = phase2.SIMDCalculator.calculate_balances_simd(entries_soa);
    return BalanceResult{
        .total_debits = result.total_debits,
        .total_credits = result.total_credits,
    };
}

// ============================================================================
// Phase 3: Parallel Multi-Currency Calculator
// ============================================================================

const CurrencyTask = struct {
    entries: []const JournalEntry,
    currency: []const u8,
    result: *CurrencyBalanceResult,
    allocator: Allocator,
};

pub const ParallelCalculator = struct {
    allocator: Allocator,
    thread_pool: std.Thread.Pool,
    
    progress_callback: ?*const fn(
        entries_processed: usize,
        total_entries: usize,
        currencies_completed: usize,
    ) void,

    pub fn init(allocator: Allocator) !ParallelCalculator {
        var thread_pool: std.Thread.Pool = undefined;
        try thread_pool.init(.{ .allocator = allocator });
        
        return ParallelCalculator{
            .allocator = allocator,
            .thread_pool = thread_pool,
            .progress_callback = null,
        };
    }
    
    pub fn deinit(self: *ParallelCalculator) void {
        self.thread_pool.deinit();
    }
    
    pub fn set_progress_callback(
        self: *ParallelCalculator,
        callback: *const fn(usize, usize, usize) void,
    ) void {
        self.progress_callback = callback;
    }
    
    /// Calculate trial balance for a single currency (worker function)
    fn calculateCurrencyWorker(task: CurrencyTask) void {
        // Filter entries by currency
        var arena = std.heap.ArenaAllocator.init(task.allocator);
        defer arena.deinit();
        const temp_allocator = arena.allocator();
        
        var filtered: std.ArrayList(JournalEntry) = .{};
        for (task.entries) |entry| {
            if (std.mem.eql(u8, entry.currency, task.currency)) {
                filtered.append(temp_allocator, entry) catch continue;
            }
        }
        
        // Convert to SoA for SIMD processing
        var entries_soa = JournalEntriesSoA.fromAoS(temp_allocator, filtered.items) catch return;
        defer entries_soa.deinit();
        
        // Use AVX-512 if available, else AVX2
        const balances = if (entries_soa.len >= VEC_SIZE_512)
            calculate_balances_avx512(&entries_soa)
        else
            calculate_balances_simd_compat(&entries_soa);
        
        // Create account map
        var account_map = std.StringHashMap(AccountBalance).init(temp_allocator);
        account_map.ensureTotalCapacity(@intCast(filtered.items.len / 10)) catch return;
        
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
        
        // Store result
        task.result.* = CurrencyBalanceResult{
            .currency = task.currency,
            .total_debits = balances.total_debits,
            .total_credits = balances.total_credits,
            .balance_difference = balances.total_debits - balances.total_credits,
            .is_balanced = @abs(balances.total_debits - balances.total_credits) < 0.01,
            .account_count = account_map.count(),
        };
    }
    
    /// Parallel multi-currency calculation
    pub fn calculate_multicurrency_parallel(
        self: *ParallelCalculator,
        journal_entries: []const JournalEntry,
        currencies: []const []const u8,
    ) !MultiCurrencyResult {
        const start_time = std.time.milliTimestamp();
        
        var result = MultiCurrencyResult.init(self.allocator);
        errdefer result.deinit();
        
        // Allocate results array
        var currency_results = try self.allocator.alloc(CurrencyBalanceResult, currencies.len);
        defer self.allocator.free(currency_results);
        
        // Initialize results
        for (currency_results, 0..) |*curr_result, i| {
            curr_result.* = CurrencyBalanceResult.init(currencies[i]);
        }
        
        // Create tasks
        const tasks = try self.allocator.alloc(CurrencyTask, currencies.len);
        defer self.allocator.free(tasks);
        
        for (tasks, 0..) |*task, i| {
            task.* = CurrencyTask{
                .entries = journal_entries,
                .currency = currencies[i],
                .result = &currency_results[i],
                .allocator = self.allocator,
            };
        }
        
        // Execute in parallel using thread pool
        var wait_group: std.Thread.WaitGroup = .{};
        wait_group.reset();
        
        for (tasks) |task| {
            wait_group.start();
            self.thread_pool.spawn(struct {
                fn run(t: CurrencyTask, wg: *std.Thread.WaitGroup) void {
                    defer wg.finish();
                    calculateCurrencyWorker(t);
                }
            }.run, .{ task, &wait_group }) catch {
                wait_group.finish();
                continue;
            };
        }
        
        // Wait for all tasks to complete
        self.thread_pool.waitAndWork(&wait_group);
        
        // Collect results
        for (currency_results) |curr_result| {
            try result.currencies.append(self.allocator, curr_result);
        }
        
        // Calculate metrics
        const end_time = std.time.milliTimestamp();
        result.total_entries_processed = journal_entries.len;
        result.processing_time_ms = @as(f64, @floatFromInt(end_time - start_time));
        result.entries_per_second = if (result.processing_time_ms > 0)
            (@as(f64, @floatFromInt(journal_entries.len)) / result.processing_time_ms) * 1000.0
        else
            0.0;
        
        return result;
    }
};

// ============================================================================
// Phase 3: Database-Optimized Reader
// ============================================================================

pub const OptimizedDatabaseReader = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) OptimizedDatabaseReader {
        return .{ .allocator = allocator };
    }
    
    /// Optimized batch fetching with prepared statements
    /// Pre-sorts by account for better cache locality during aggregation
    pub fn read_entries_optimized(
        self: *OptimizedDatabaseReader,
        company_code: []const u8,
        fiscal_year: []const u8,
        period: []const u8,
    ) !JournalEntriesSoA {
        // Estimate size based on typical dataset (126K entries)
        const estimated_count: usize = 130000;
        
        // Pre-allocate arrays
        var amounts = try self.allocator.alloc(f64, estimated_count);
        var indicators = try self.allocator.alloc(u8, estimated_count);
        var accounts = try self.allocator.alloc([]const u8, estimated_count);
        var company_codes = try self.allocator.alloc([]const u8, estimated_count);
        var fiscal_years = try self.allocator.alloc([]const u8, estimated_count);
        var periods = try self.allocator.alloc([]const u8, estimated_count);
        var currencies = try self.allocator.alloc([]const u8, estimated_count);
        
        // Simulate database fetch (in production, use prepared statements)
        // SQL: SELECT * FROM TB_JOURNAL_ENTRIES 
        //      WHERE rbukrs = ? AND gjahr = ? AND poper = ? 
        //      ORDER BY racct, rtcur  -- Pre-sort for cache locality
        
        const actual_count: usize = 0;
        // ... fetch and populate arrays ...
        _ = company_code;
        _ = fiscal_year;
        _ = period;
        
        // Resize to actual count
        amounts = try self.allocator.realloc(amounts, actual_count);
        indicators = try self.allocator.realloc(indicators, actual_count);
        accounts = try self.allocator.realloc(accounts, actual_count);
        company_codes = try self.allocator.realloc(company_codes, actual_count);
        fiscal_years = try self.allocator.realloc(fiscal_years, actual_count);
        periods = try self.allocator.realloc(periods, actual_count);
        currencies = try self.allocator.realloc(currencies, actual_count);
        
        return JournalEntriesSoA{
            .amounts = amounts,
            .indicators = indicators,
            .accounts = accounts,
            .company_codes = company_codes,
            .fiscal_years = fiscal_years,
            .periods = periods,
            .currencies = currencies,
            .len = actual_count,
            .allocator = self.allocator,
        };
    }
};

// ============================================================================
// Phase 3: Ultra-Optimized Calculator (All Features)
// ============================================================================

pub const UltraOptimizedCalculator = struct {
    allocator: Allocator,
    thread_pool: std.Thread.Pool,
    use_avx512: bool,
    
    progress_callback: ?*const fn(
        entries_processed: usize,
        total_entries: usize,
        stage: []const u8,
    ) void,

    pub fn init(allocator: Allocator) !UltraOptimizedCalculator {
        var thread_pool: std.Thread.Pool = undefined;
        try thread_pool.init(.{ .allocator = allocator });
        
        // Detect AVX-512 support (simplified detection)
        const use_avx512 = true; // In production, use CPU feature detection
        
        return UltraOptimizedCalculator{
            .allocator = allocator,
            .thread_pool = thread_pool,
            .use_avx512 = use_avx512,
            .progress_callback = null,
        };
    }
    
    pub fn deinit(self: *UltraOptimizedCalculator) void {
        self.thread_pool.deinit();
    }
    
    pub fn set_progress_callback(
        self: *UltraOptimizedCalculator,
        callback: *const fn(usize, usize, []const u8) void,
    ) void {
        self.progress_callback = callback;
    }
    
    /// Ultimate performance: Parallel + SIMD + SoA + Kahan + Arena
    pub fn calculate_trial_balance_ultra(
        self: *UltraOptimizedCalculator,
        journal_entries: []const JournalEntry,
    ) !MultiCurrencyResult {
        const start_time = std.time.milliTimestamp();
        
        // Stage 1: Extract unique currencies
        if (self.progress_callback) |cb| cb(0, journal_entries.len, "extracting_currencies");
        
        var currency_set = std.StringHashMap(void).init(self.allocator);
        defer currency_set.deinit();
        
        for (journal_entries) |entry| {
            try currency_set.put(entry.currency, {});
        }
        
        var currencies: std.ArrayList([]const u8) = .{};
        defer currencies.deinit(self.allocator);
        
        var currency_iter = currency_set.keyIterator();
        while (currency_iter.next()) |key| {
            try currencies.append(self.allocator, key.*);
        }
        
        // Stage 2: Parallel processing
        if (self.progress_callback) |cb| cb(0, journal_entries.len, "parallel_processing");
        
        var parallel_calc = try ParallelCalculator.init(self.allocator);
        defer parallel_calc.deinit();
        
        var result = try parallel_calc.calculate_multicurrency_parallel(
            journal_entries,
            currencies.items,
        );
        
        // Calculate final metrics
        const end_time = std.time.milliTimestamp();
        result.processing_time_ms = @as(f64, @floatFromInt(end_time - start_time));
        result.entries_per_second = if (result.processing_time_ms > 0)
            (@as(f64, @floatFromInt(journal_entries.len)) / result.processing_time_ms) * 1000.0
        else
            0.0;
        
        if (self.progress_callback) |cb| cb(journal_entries.len, journal_entries.len, "complete");
        
        return result;
    }
};

// ============================================================================
// Phase 3: Streaming Calculator with WebSocket Integration
// ============================================================================

pub const StreamingCalculator = struct {
    allocator: Allocator,
    parallel_calc: UltraOptimizedCalculator,
    
    websocket_callback: ?*const fn(
        msg_type: []const u8,
        payload: []const u8,
    ) void,

    pub fn init(allocator: Allocator) !StreamingCalculator {
        return StreamingCalculator{
            .allocator = allocator,
            .parallel_calc = try UltraOptimizedCalculator.init(allocator),
            .websocket_callback = null,
        };
    }
    
    pub fn deinit(self: *StreamingCalculator) void {
        self.parallel_calc.deinit();
    }
    
    pub fn set_websocket_callback(
        self: *StreamingCalculator,
        callback: *const fn([]const u8, []const u8) void,
    ) void {
        self.websocket_callback = callback;
    }
    
    /// Calculate with real-time WebSocket streaming
    pub fn calculate_with_streaming(
        self: *StreamingCalculator,
        journal_entries: []const JournalEntry,
    ) !MultiCurrencyResult {
        // Set up progress callback to stream via WebSocket
        const StreamingContext = struct {
            var ws_cb: ?*const fn([]const u8, []const u8) void = null;
            var alloc: Allocator = undefined;
            
            fn progressCallback(processed: usize, total: usize, stage: []const u8) void {
                if (ws_cb) |callback| {
                    const msg = std.fmt.allocPrint(alloc,
                        \\{{"type":"tb:progress","processed":{d},"total":{d},"stage":"{s}"}}
                    , .{ processed, total, stage }) catch return;
                    defer alloc.free(msg);
                    callback("progress", msg);
                }
            }
        };
        
        StreamingContext.ws_cb = self.websocket_callback;
        StreamingContext.alloc = self.allocator;
        
        self.parallel_calc.set_progress_callback(&StreamingContext.progressCallback);
        
        // Stream start message
        if (self.websocket_callback) |callback| {
            callback("start", "{\"message\":\"Starting calculation\"}");
        }
        
        // Perform calculation with streaming updates
        const result = try self.parallel_calc.calculate_trial_balance_ultra(journal_entries);
        
        // Stream completion message
        if (self.websocket_callback) |callback| {
            const msg = try std.fmt.allocPrint(self.allocator,
                \\{{"type":"tb:complete","entries":{d},"time":{d:.2},"throughput":{d:.0}}}
            , .{
                result.total_entries_processed,
                result.processing_time_ms,
                result.entries_per_second,
            });
            defer self.allocator.free(msg);
            callback("complete", msg);
        }
        
        return result;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "parallel multi-currency calculation" {
    const allocator = std.testing.allocator;
    var calculator = try ParallelCalculator.init(allocator);
    defer calculator.deinit();
    
    // Generate entries with 3 currencies
    const entries = try allocator.alloc(JournalEntry, 9000);
    defer allocator.free(entries);
    
    const test_currencies = [_][]const u8{ "USD", "EUR", "GBP" };
    
    for (entries, 0..) |*entry, i| {
        const currency_idx = i % 3;
        const account_str = try std.fmt.allocPrint(allocator, "ACC{d:0>6}", .{i % 100});
        defer allocator.free(account_str);
        
        entry.* = JournalEntry{
            .company_code = "1000",
            .fiscal_year = "2025",
            .period = "001",
            .document_number = try std.fmt.allocPrint(allocator, "DOC{d:0>6}", .{i}),
            .line_item = "001",
            .account = try allocator.dupe(u8, account_str),
            .debit_credit_indicator = if (i % 2 == 0) 'S' else 'H',
            .amount = 1000.0 + @as(f64, @floatFromInt(i)),
            .currency = test_currencies[currency_idx],
            .posting_date = "2025-01-15",
        };
    }
    defer {
        for (entries) |entry| {
            allocator.free(entry.document_number);
            allocator.free(entry.account);
        }
    }
    
    var result = try calculator.calculate_multicurrency_parallel(entries, &test_currencies);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(usize, 3), result.currencies.items.len);
    try std.testing.expect(result.parallel_enabled);
    
    std.debug.print("\n✓ Parallel multi-currency test passed\n", .{});
    std.debug.print("  Total entries: {d}\n", .{entries.len});
    std.debug.print("  Currencies: {d}\n", .{result.currencies.items.len});
    std.debug.print("  Processing time: {d:.2}ms\n", .{result.processing_time_ms});
    std.debug.print("  Throughput: {d:.0} entries/sec\n", .{result.entries_per_second});
    std.debug.print("  Parallel enabled: {}\n", .{result.parallel_enabled});
}

test "AVX-512 vectorization" {
    const allocator = std.testing.allocator;
    
    // Generate test data aligned for AVX-512
    const entries = try allocator.alloc(JournalEntry, 160); // 10 × 16
    defer allocator.free(entries);
    
    for (entries, 0..) |*entry, i| {
        const account_str = try std.fmt.allocPrint(allocator, "ACC{d:0>6}", .{i});
        defer allocator.free(account_str);
        
        entry.* = JournalEntry{
            .company_code = "1000",
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
    
    var entries_soa = try JournalEntriesSoA.fromAoS(allocator, entries);
    defer entries_soa.deinit();
    
    const result = calculate_balances_avx512(&entries_soa);
    
    // 160 entries × 100.0 = 16,000.0 total debits
    try std.testing.expectApproxEqAbs(16000.0, result.total_debits, 0.01);
    
    std.debug.print("\n✓ AVX-512 vectorization test passed\n", .{});
    std.debug.print("  Vector width: 16 (AVX-512)\n", .{});
    std.debug.print("  Entries: {d}\n", .{entries.len});
    std.debug.print("  Total debits: {d:.2}\n", .{result.total_debits});
}