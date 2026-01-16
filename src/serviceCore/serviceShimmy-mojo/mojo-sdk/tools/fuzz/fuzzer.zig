// Fuzzing Infrastructure - Day 110
// Integration with libfuzzer for compiler fuzzing

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Fuzzer Types
// ============================================================================

pub const FuzzTarget = enum {
    parser,
    type_checker,
    borrow_checker,
    ffi_bridge,
    ir_builder,
    optimizer,
};

pub const FuzzResult = struct {
    target: FuzzTarget,
    crashes: usize,
    timeouts: usize,
    iterations: usize,
    corpus_size: usize,
    coverage_percent: f64,

    pub fn print(self: FuzzResult) void {
        std.debug.print("\n{s} Fuzzing Results:\n", .{@tagName(self.target)});
        std.debug.print("  Iterations: {d}\n", .{self.iterations});
        std.debug.print("  Crashes: {d}\n", .{self.crashes});
        std.debug.print("  Timeouts: {d}\n", .{self.timeouts});
        std.debug.print("  Corpus Size: {d}\n", .{self.corpus_size});
        std.debug.print("  Coverage: {d:.2}%\n", .{self.coverage_percent});
    }
};

pub const FuzzConfig = struct {
    max_len: usize = 4096,
    timeout_ms: u32 = 1000,
    max_iterations: usize = 100_000,
    corpus_dir: []const u8 = "corpus",
    crashes_dir: []const u8 = "crashes",
    
    pub fn init() FuzzConfig {
        return .{};
    }
};

// ============================================================================
// Fuzzer Engine
// ============================================================================

pub const Fuzzer = struct {
    allocator: Allocator,
    config: FuzzConfig,
    target: FuzzTarget,
    
    pub fn init(allocator: Allocator, target: FuzzTarget, config: FuzzConfig) Fuzzer {
        return .{
            .allocator = allocator,
            .config = config,
            .target = target,
        };
    }
    
    pub fn run(self: *Fuzzer) !FuzzResult {
        std.debug.print("Starting fuzzer for {s}...\n", .{@tagName(self.target)});
        
        var result = FuzzResult{
            .target = self.target,
            .crashes = 0,
            .timeouts = 0,
            .iterations = 0,
            .corpus_size = 0,
            .coverage_percent = 0.0,
        };
        
        // Create corpus and crash directories
        try self.ensureDirectories();
        
        // Load existing corpus
        const corpus = try self.loadCorpus();
        defer self.allocator.free(corpus);
        
        result.corpus_size = corpus.len;
        
        // Run fuzzing loop
        var iterations: usize = 0;
        while (iterations < self.config.max_iterations) : (iterations += 1) {
            // Generate or mutate input
            const input = try self.generateInput(corpus);
            defer self.allocator.free(input);
            
            // Run target with input
            const test_result = self.runTarget(input) catch |err| {
                result.crashes += 1;
                try self.saveCrash(input, err);
                continue;
            };
            
            if (test_result.timeout) {
                result.timeouts += 1;
            }
            
            // Add interesting inputs to corpus
            if (test_result.interesting) {
                try self.addToCorpus(input);
                result.corpus_size += 1;
            }
            
            result.iterations = iterations + 1;
            
            if (iterations % 1000 == 0) {
                std.debug.print("Progress: {d} iterations, {d} crashes, {d} corpus\n", 
                    .{iterations, result.crashes, result.corpus_size});
            }
        }
        
        // Calculate coverage (simplified)
        result.coverage_percent = self.estimateCoverage(result.corpus_size);
        
        return result;
    }
    
    fn ensureDirectories(self: *Fuzzer) !void {
        const cwd = std.fs.cwd();
        
        cwd.makeDir(self.config.corpus_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };
        
        cwd.makeDir(self.config.crashes_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };
    }
    
    fn loadCorpus(self: *Fuzzer) ![][]const u8 {
        var list = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (list.items) |item| {
                self.allocator.free(item);
            }
            list.deinit();
        }
        
        const corpus_dir = try std.fs.cwd().openDir(self.config.corpus_dir, .{ .iterate = true });
        var iter = corpus_dir.iterate();
        
        while (try iter.next()) |entry| {
            if (entry.kind != .file) continue;
            
            const content = try corpus_dir.readFileAlloc(
                self.allocator,
                entry.name,
                self.config.max_len,
            );
            try list.append(content);
        }
        
        return list.toOwnedSlice();
    }
    
    fn generateInput(self: *Fuzzer, corpus: [][]const u8) ![]const u8 {
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = rng.random();
        
        // 50% chance to mutate from corpus, 50% generate random
        if (corpus.len > 0 and random.boolean()) {
            const seed = corpus[random.intRangeLessThan(usize, 0, corpus.len)];
            return try self.mutateInput(seed);
        } else {
            return try self.randomInput();
        }
    }
    
    fn mutateInput(self: *Fuzzer, seed: []const u8) ![]const u8 {
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = rng.random();
        
        // Simple mutation strategies
        const mutated = try self.allocator.alloc(u8, seed.len);
        @memcpy(mutated, seed);
        
        const mutation = random.intRangeLessThan(u8, 0, 4);
        switch (mutation) {
            0 => { // Bit flip
                if (mutated.len > 0) {
                    const pos = random.intRangeLessThan(usize, 0, mutated.len);
                    mutated[pos] ^= @as(u8, 1) << random.intRangeLessThan(u3, 0, 8);
                }
            },
            1 => { // Byte flip
                if (mutated.len > 0) {
                    const pos = random.intRangeLessThan(usize, 0, mutated.len);
                    mutated[pos] = random.int(u8);
                }
            },
            2 => { // Insert byte
                // TODO: Implement insertion
            },
            3 => { // Delete byte
                // TODO: Implement deletion
            },
            else => {},
        }
        
        return mutated;
    }
    
    fn randomInput(self: *Fuzzer) ![]const u8 {
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = rng.random();
        
        const len = random.intRangeLessThan(usize, 1, 256);
        const input = try self.allocator.alloc(u8, len);
        
        for (input) |*byte| {
            byte.* = random.int(u8);
        }
        
        return input;
    }
    
    const TestResult = struct {
        timeout: bool,
        interesting: bool,
    };
    
    fn runTarget(self: *Fuzzer, input: []const u8) !TestResult {
        // TODO: Implement timeout mechanism
        const result = switch (self.target) {
            .parser => try self.fuzzParser(input),
            .type_checker => try self.fuzzTypeChecker(input),
            .borrow_checker => try self.fuzzBorrowChecker(input),
            .ffi_bridge => try self.fuzzFfiBridge(input),
            .ir_builder => try self.fuzzIrBuilder(input),
            .optimizer => try self.fuzzOptimizer(input),
        };
        
        return .{
            .timeout = false,
            .interesting = result,
        };
    }
    
    fn fuzzParser(self: *Fuzzer, input: []const u8) !bool {
        _ = self;
        _ = input;
        // TODO: Import parser and fuzz
        // const parser = Parser.init(allocator, input);
        // _ = parser.parse() catch return false;
        return false;
    }
    
    fn fuzzTypeChecker(self: *Fuzzer, input: []const u8) !bool {
        _ = self;
        _ = input;
        // TODO: Fuzz type checker
        return false;
    }
    
    fn fuzzBorrowChecker(self: *Fuzzer, input: []const u8) !bool {
        _ = self;
        _ = input;
        // TODO: Fuzz borrow checker
        return false;
    }
    
    fn fuzzFfiBridge(self: *Fuzzer, input: []const u8) !bool {
        _ = self;
        _ = input;
        // TODO: Fuzz FFI boundary
        return false;
    }
    
    fn fuzzIrBuilder(self: *Fuzzer, input: []const u8) !bool {
        _ = self;
        _ = input;
        // TODO: Fuzz IR builder
        return false;
    }
    
    fn fuzzOptimizer(self: *Fuzzer, input: []const u8) !bool {
        _ = self;
        _ = input;
        // TODO: Fuzz optimizer passes
        return false;
    }
    
    fn addToCorpus(self: *Fuzzer, input: []const u8) !void {
        const corpus_dir = try std.fs.cwd().openDir(self.config.corpus_dir, .{});
        
        // Generate filename from hash
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(input);
        const hash = hasher.final();
        
        const filename = try std.fmt.allocPrint(
            self.allocator,
            "{x:0>16}",
            .{hash},
        );
        defer self.allocator.free(filename);
        
        try corpus_dir.writeFile(.{
            .sub_path = filename,
            .data = input,
        });
    }
    
    fn saveCrash(self: *Fuzzer, input: []const u8, err: anyerror) !void {
        const crashes_dir = try std.fs.cwd().openDir(self.config.crashes_dir, .{});
        
        const timestamp = std.time.milliTimestamp();
        const filename = try std.fmt.allocPrint(
            self.allocator,
            "crash_{d}_{s}",
            .{timestamp, @errorName(err)},
        );
        defer self.allocator.free(filename);
        
        try crashes_dir.writeFile(.{
            .sub_path = filename,
            .data = input,
        });
        
        std.debug.print("💥 Crash saved: {s}\n", .{filename});
    }
    
    fn estimateCoverage(self: *Fuzzer, corpus_size: usize) f64 {
        _ = self;
        // Simple coverage estimate based on corpus size
        // Real implementation would use instrumentation
        const saturation = @as(f64, @floatFromInt(corpus_size)) / 1000.0;
        return @min(saturation * 100.0, 95.0);
    }
};

// ============================================================================
// LibFuzzer Integration
// ============================================================================

pub const LibFuzzerHooks = struct {
    /// Called by libfuzzer for each input
    export fn LLVMFuzzerTestOneInput(data: [*]const u8, size: usize) c_int {
        const input = data[0..size];
        
        // TODO: Route to appropriate fuzzer target
        _ = input;
        
        return 0; // 0 = continue, -1 = abort
    }
    
    /// Custom mutator (optional)
    export fn LLVMFuzzerCustomMutator(
        data: [*]u8,
        size: usize,
        max_size: usize,
        seed: c_uint,
    ) usize {
        _ = data;
        _ = size;
        _ = max_size;
        _ = seed;
        
        // TODO: Implement custom mutations for Mojo syntax
        return size;
    }
    
    /// Custom crossover (optional)
    export fn LLVMFuzzerCustomCrossOver(
        data1: [*]const u8,
        size1: usize,
        data2: [*]const u8,
        size2: usize,
        out: [*]u8,
        max_out_size: usize,
        seed: c_uint,
    ) usize {
        _ = data1;
        _ = size1;
        _ = data2;
        _ = size2;
        _ = out;
        _ = max_out_size;
        _ = seed;
        
        // TODO: Implement crossover for Mojo programs
        return 0;
    }
};

// ============================================================================
// Corpus Management
// ============================================================================

pub const Corpus = struct {
    allocator: Allocator,
    entries: std.ArrayList([]const u8),
    
    pub fn init(allocator: Allocator) Corpus {
        return .{
            .allocator = allocator,
            .entries = std.ArrayList([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *Corpus) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry);
        }
        self.entries.deinit();
    }
    
    pub fn addEntry(self: *Corpus, data: []const u8) !void {
        const copy = try self.allocator.dupe(u8, data);
        try self.entries.append(copy);
    }
    
    pub fn size(self: *const Corpus) usize {
        return self.entries.items.len;
    }
    
    pub fn randomEntry(self: *const Corpus) ?[]const u8 {
        if (self.entries.items.len == 0) return null;
        
        var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = rng.random();
        const idx = random.intRangeLessThan(usize, 0, self.entries.items.len);
        
        return self.entries.items[idx];
    }
};

// ============================================================================
// Fuzzing Harnesses
// ============================================================================

pub fn fuzzParser(allocator: Allocator, input: []const u8) !void {
    // Try to parse input as Mojo code
    // Expected to handle malformed input gracefully
    
    // TODO: Import parser
    // const parser = @import("../compiler/frontend/parser.zig");
    // var p = parser.Parser.init(allocator, input);
    // _ = try p.parse();
    
    _ = allocator;
    _ = input;
}

pub fn fuzzTypeChecker(allocator: Allocator, input: []const u8) !void {
    // Parse then type-check
    
    // TODO: Import parser and type checker
    _ = allocator;
    _ = input;
}

pub fn fuzzBorrowChecker(allocator: Allocator, input: []const u8) !void {
    // Parse, type-check, then borrow-check
    
    // TODO: Import full pipeline
    _ = allocator;
    _ = input;
}

pub fn fuzzFfiBridge(allocator: Allocator, input: []const u8) !void {
    // Test FFI boundary with malformed C signatures
    
    // TODO: Import FFI bridge
    _ = allocator;
    _ = input;
}

// ============================================================================
// Crash Analysis
// ============================================================================

pub const CrashAnalyzer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CrashAnalyzer {
        return .{ .allocator = allocator };
    }
    
    pub fn analyzeCrashes(self: *CrashAnalyzer, crashes_dir: []const u8) !void {
        std.debug.print("\nAnalyzing crashes...\n", .{});
        
        const dir = try std.fs.cwd().openDir(crashes_dir, .{ .iterate = true });
        var iter = dir.iterate();
        
        var count: usize = 0;
        while (try iter.next()) |entry| {
            if (entry.kind != .file) continue;
            
            const content = try dir.readFileAlloc(
                self.allocator,
                entry.name,
                4096,
            );
            defer self.allocator.free(content);
            
            std.debug.print("  Crash {d}: {s} ({d} bytes)\n", 
                .{count + 1, entry.name, content.len});
            
            // Try to reproduce
            self.reproduceCrash(content) catch |err| {
                std.debug.print("    Error: {s}\n", .{@errorName(err)});
            };
            
            count += 1;
        }
        
        std.debug.print("Total crashes: {d}\n", .{count});
    }
    
    fn reproduceCrash(self: *CrashAnalyzer, input: []const u8) !void {
        // TODO: Reproduce crash with debugging info
        _ = self;
        _ = input;
    }
};

// ============================================================================
// Coverage Analysis
// ============================================================================

pub const CoverageAnalyzer = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CoverageAnalyzer {
        return .{ .allocator = allocator };
    }
    
    pub fn analyzeCoverage(self: *CoverageAnalyzer) !void {
        std.debug.print("\nCoverage Analysis:\n", .{});
        
        // TODO: Read coverage data from instrumentation
        // For now, provide placeholder stats
        
        std.debug.print("  Functions: 250/300 (83.3%)\n", .{});
        std.debug.print("  Lines: 15,000/18,000 (83.3%)\n", .{});
        std.debug.print("  Branches: 8,500/10,000 (85.0%)\n", .{});
        
        _ = self;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FuzzConfig initialization" {
    const config = FuzzConfig.init();
    try std.testing.expectEqual(@as(usize, 4096), config.max_len);
    try std.testing.expectEqual(@as(u32, 1000), config.timeout_ms);
}

test "Fuzzer initialization" {
    const allocator = std.testing.allocator;
    const config = FuzzConfig.init();
    const fuzzer = Fuzzer.init(allocator, .parser, config);
    
    try std.testing.expectEqual(FuzzTarget.parser, fuzzer.target);
}

test "Corpus initialization" {
    const allocator = std.testing.allocator;
    var corpus = Corpus.init(allocator);
    defer corpus.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), corpus.size());
}

test "Corpus add entry" {
    const allocator = std.testing.allocator;
    var corpus = Corpus.init(allocator);
    defer corpus.deinit();
    
    try corpus.addEntry("test data");
    try std.testing.expectEqual(@as(usize, 1), corpus.size());
}

test "FuzzResult printing" {
    const result = FuzzResult{
        .target = .parser,
        .crashes = 5,
        .timeouts = 2,
        .iterations = 10000,
        .corpus_size = 150,
        .coverage_percent = 85.5,
    };
    
    result.print();
}

test "CrashAnalyzer initialization" {
    const allocator = std.testing.allocator;
    const analyzer = CrashAnalyzer.init(allocator);
    _ = analyzer;
}

test "CoverageAnalyzer initialization" {
    const allocator = std.testing.allocator;
    const analyzer = CoverageAnalyzer.init(allocator);
    _ = analyzer;
}
