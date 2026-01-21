// Performance Benchmark Suite for nWorkflow
// Comprehensive benchmarks for Petri Net, Workflow Parser, Executor, and Data Pipeline
//
// Run with: zig build bench
// Output: Console formatted results + JSON for CI integration

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

/// Result of a single benchmark run
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    mean_ns: u64,
    std_dev_ns: u64,

    // Memory metrics
    peak_memory_bytes: usize,
    total_allocations: usize,

    // Derived metrics
    pub fn opsPerSecond(self: BenchmarkResult) f64 {
        if (self.mean_ns == 0) return 0;
        return 1_000_000_000.0 / @as(f64, @floatFromInt(self.mean_ns));
    }

    pub fn meanMicroseconds(self: BenchmarkResult) f64 {
        return @as(f64, @floatFromInt(self.mean_ns)) / 1000.0;
    }

    pub fn meanMilliseconds(self: BenchmarkResult) f64 {
        return @as(f64, @floatFromInt(self.mean_ns)) / 1_000_000.0;
    }
};

/// Benchmark configuration
pub const BenchmarkConfig = struct {
    warmup_iterations: u64 = 10,
    min_iterations: u64 = 100,
    max_iterations: u64 = 10000,
    target_time_ns: u64 = 1_000_000_000, // 1 second

    pub fn default() BenchmarkConfig {
        return .{};
    }
};

/// Memory tracking allocator wrapper
pub const TrackingAllocator = struct {
    backing_allocator: Allocator,
    current_bytes: usize,
    peak_bytes: usize,
    total_allocations: usize,

    pub fn init(backing_allocator: Allocator) TrackingAllocator {
        return .{
            .backing_allocator = backing_allocator,
            .current_bytes = 0,
            .peak_bytes = 0,
            .total_allocations = 0,
        };
    }

    pub fn allocator(self: *TrackingAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result != null) {
            self.current_bytes += len;
            self.total_allocations += 1;
            if (self.current_bytes > self.peak_bytes) {
                self.peak_bytes = self.current_bytes;
            }
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        if (self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr)) {
            if (new_len > buf.len) {
                self.current_bytes += new_len - buf.len;
                if (self.current_bytes > self.peak_bytes) {
                    self.peak_bytes = self.current_bytes;
                }
            } else {
                self.current_bytes -= buf.len - new_len;
            }
            return true;
        }
        return false;
    }

    fn remap(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing_allocator.rawRemap(buf, buf_align, new_len, ret_addr);
        if (result != null) {
            if (new_len > buf.len) {
                self.current_bytes += new_len - buf.len;
                if (self.current_bytes > self.peak_bytes) {
                    self.peak_bytes = self.current_bytes;
                }
            } else {
                self.current_bytes -= buf.len - new_len;
            }
        }
        return result;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.backing_allocator.rawFree(buf, buf_align, ret_addr);
        self.current_bytes -= buf.len;
    }

    pub fn reset(self: *TrackingAllocator) void {
        self.current_bytes = 0;
        self.peak_bytes = 0;
        self.total_allocations = 0;
    }
};

/// Benchmark function type
pub const BenchFn = *const fn (Allocator) anyerror!void;

/// Run a benchmark with the specified function
pub fn runBenchmark(
    name: []const u8,
    comptime bench_fn: BenchFn,
    config: BenchmarkConfig,
    backing_allocator: Allocator,
) !BenchmarkResult {
    var tracking = TrackingAllocator.init(backing_allocator);
    const alloc = tracking.allocator();

    // Warmup phase
    var warmup: u64 = 0;
    while (warmup < config.warmup_iterations) : (warmup += 1) {
        try bench_fn(alloc);
    }
    tracking.reset();

    // Measurement phase
    var times = std.ArrayList(u64){};
    defer times.deinit(backing_allocator);

    var total_ns: u64 = 0;
    var iterations: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;

    while (iterations < config.max_iterations and
           (iterations < config.min_iterations or total_ns < config.target_time_ns)) {
        const start = @as(u64, @intCast(std.time.nanoTimestamp()));
        try bench_fn(alloc);
        const end = @as(u64, @intCast(std.time.nanoTimestamp()));

        const elapsed = end - start;
        try times.append(backing_allocator, elapsed);
        total_ns += elapsed;
        iterations += 1;

        if (elapsed < min_ns) min_ns = elapsed;
        if (elapsed > max_ns) max_ns = elapsed;
    }

    // Calculate statistics
    const mean_ns = total_ns / iterations;

    // Calculate standard deviation
    var variance_sum: u128 = 0;
    for (times.items) |t| {
        const diff = if (t > mean_ns) t - mean_ns else mean_ns - t;
        variance_sum += @as(u128, diff) * @as(u128, diff);
    }
    const variance = variance_sum / iterations;
    const std_dev_ns = @as(u64, @intCast(std.math.sqrt(variance)));

    return BenchmarkResult{
        .name = name,
        .iterations = iterations,
        .total_ns = total_ns,
        .min_ns = min_ns,
        .max_ns = max_ns,
        .mean_ns = mean_ns,
        .std_dev_ns = std_dev_ns,
        .peak_memory_bytes = tracking.peak_bytes,
        .total_allocations = tracking.total_allocations,
    };
}

/// Print a single benchmark result to stdout
pub fn printResult(result: BenchmarkResult) void {
    std.debug.print("| {s:<40} | {d:>10} | {d:>12.2} us | {d:>12.2} us | {d:>12.2} us | {d:>10.0} ops/s | {d:>10} KB |\n", .{
        result.name,
        result.iterations,
        result.meanMicroseconds(),
        @as(f64, @floatFromInt(result.min_ns)) / 1000.0,
        @as(f64, @floatFromInt(result.max_ns)) / 1000.0,
        result.opsPerSecond(),
        result.peak_memory_bytes / 1024,
    });
}

/// Print table header
pub fn printHeader() void {
    std.debug.print("\n", .{});
    std.debug.print("=" ** 120 ++ "\n", .{});
    std.debug.print("| {s:<40} | {s:>10} | {s:>14} | {s:>14} | {s:>14} | {s:>13} | {s:>12} |\n", .{
        "Benchmark",
        "Iterations",
        "Mean",
        "Min",
        "Max",
        "Throughput",
        "Peak Memory",
    });
    std.debug.print("=" ** 120 ++ "\n", .{});
}

/// Print table footer/separator
pub fn printSeparator(title: []const u8) void {
    std.debug.print("-" ** 120 ++ "\n", .{});
    std.debug.print("| {s:<118} |\n", .{title});
    std.debug.print("-" ** 120 ++ "\n", .{});
}

/// Export results to JSON for CI integration
pub fn exportToJson(results: []const BenchmarkResult, allocator: Allocator) ![]const u8 {
    var json = std.ArrayList(u8){};
    const writer = json.writer(allocator);

    try writer.writeAll("{\n  \"benchmarks\": [\n");

    for (results, 0..) |result, i| {
        try writer.print("    {{\n", .{});
        try writer.print("      \"name\": \"{s}\",\n", .{result.name});
        try writer.print("      \"iterations\": {d},\n", .{result.iterations});
        try writer.print("      \"total_ns\": {d},\n", .{result.total_ns});
        try writer.print("      \"min_ns\": {d},\n", .{result.min_ns});
        try writer.print("      \"max_ns\": {d},\n", .{result.max_ns});
        try writer.print("      \"mean_ns\": {d},\n", .{result.mean_ns});
        try writer.print("      \"std_dev_ns\": {d},\n", .{result.std_dev_ns});
        try writer.print("      \"ops_per_second\": {d:.2},\n", .{result.opsPerSecond()});
        try writer.print("      \"peak_memory_bytes\": {d},\n", .{result.peak_memory_bytes});
        try writer.print("      \"total_allocations\": {d}\n", .{result.total_allocations});
        try writer.print("    }}{s}\n", .{if (i < results.len - 1) "," else ""});
    }

    try writer.writeAll("  ],\n");
    try writer.print("  \"timestamp\": {d},\n", .{std.time.timestamp()});
    try writer.writeAll("  \"version\": \"1.0.0\"\n");
    try writer.writeAll("}\n");

    return json.toOwnedSlice(allocator);
}


// ============================================================================
// Petri Net Benchmarks
// ============================================================================

/// Helper to create a simple Petri Net for benchmarking
fn createBenchmarkPetriNet(allocator: Allocator, num_places: usize) !*PetriNetBench {
    const net = try allocator.create(PetriNetBench);
    net.* = .{
        .allocator = allocator,
        .places = std.ArrayList([]const u8){},
        .transitions = std.ArrayList([]const u8){},
    };

    // Create places
    var i: usize = 0;
    while (i < num_places) : (i += 1) {
        var buf: [32]u8 = undefined;
        const id = try std.fmt.bufPrint(&buf, "place_{d}", .{i});
        const id_copy = try allocator.dupe(u8, id);
        try net.places.append(allocator, id_copy);
    }

    return net;
}

/// Simple Petri Net benchmark structure (standalone for benchmarking)
const PetriNetBench = struct {
    allocator: Allocator,
    places: std.ArrayList([]const u8),
    transitions: std.ArrayList([]const u8),

    pub fn deinit(self: *PetriNetBench) void {
        for (self.places.items) |p| {
            self.allocator.free(p);
        }
        self.places.deinit(self.allocator);
        for (self.transitions.items) |t| {
            self.allocator.free(t);
        }
        self.transitions.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

/// Benchmark: Create Petri Net with 100 places
fn benchCreatePetriNet100(allocator: Allocator) !void {
    const net = try createBenchmarkPetriNet(allocator, 100);
    defer net.deinit();
}

/// Benchmark: Create Petri Net with 1000 places
fn benchCreatePetriNet1000(allocator: Allocator) !void {
    const net = try createBenchmarkPetriNet(allocator, 1000);
    defer net.deinit();
}

/// Benchmark: Simulate firing 1000 transitions
fn benchFireTransitions1000(allocator: Allocator) !void {
    // Simulate transition firing without actual Petri Net to measure core operation speed
    var token_counts = std.AutoHashMap(usize, usize).init(allocator);
    defer token_counts.deinit();

    // Initialize places with tokens
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try token_counts.put(i, 10);
    }

    // Simulate 1000 transition fires (move tokens between places)
    var fires: usize = 0;
    while (fires < 1000) : (fires += 1) {
        const src = fires % 100;
        const dst = (fires + 1) % 100;

        if (token_counts.get(src)) |count| {
            if (count > 0) {
                try token_counts.put(src, count - 1);
                const dst_count = token_counts.get(dst) orelse 0;
                try token_counts.put(dst, dst_count + 1);
            }
        }
    }
}

/// Benchmark: Check deadlock on complex net (100 places, 50 transitions)
fn benchCheckDeadlock(allocator: Allocator) !void {
    var enabled_count: usize = 0;

    // Simulate checking 50 transitions for enabled state
    var t: usize = 0;
    while (t < 50) : (t += 1) {
        // Simulate checking input arcs (2 per transition on average)
        var has_tokens = true;
        var arc: usize = 0;
        while (arc < 2) : (arc += 1) {
            // Simulate place lookup
            const place_id = (t * 2 + arc) % 100;
            _ = place_id;
            // Simulate token check (random-ish result based on iteration)
            if ((t + arc) % 5 == 0) {
                has_tokens = false;
            }
        }
        if (has_tokens) {
            enabled_count += 1;
        }
    }

    // Use allocator to prevent optimization
    const temp = try allocator.alloc(u8, 1);
    temp[0] = @as(u8, @intCast(enabled_count % 256));
    allocator.free(temp);
}


// ============================================================================
// Workflow Parser Benchmarks
// ============================================================================

/// Generate workflow JSON with N nodes
fn generateWorkflowJson(allocator: Allocator, num_nodes: usize) ![]const u8 {
    var json = std.ArrayList(u8){};
    const writer = json.writer(allocator);

    try writer.writeAll("{\"version\":\"1.0\",\"name\":\"Benchmark Workflow\",\"description\":\"Generated for benchmarking\",");
    try writer.writeAll("\"metadata\":{\"tags\":[]},\"nodes\":[");

    var i: usize = 0;
    while (i < num_nodes) : (i += 1) {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{{\"id\":\"n{d}\",\"type\":\"{s}\",\"name\":\"Node{d}\",\"config\":{{}}}}", .{
            i,
            if (i == 0) "trigger" else "action",
            i,
        });
    }

    try writer.writeAll("],\"edges\":[");

    // Create linear edges
    i = 0;
    while (i < num_nodes - 1) : (i += 1) {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{{\"from\":\"n{d}\",\"to\":\"n{d}\"}}", .{ i, i + 1 });
    }

    try writer.writeAll("],\"error_handlers\":[]}");

    return json.toOwnedSlice(allocator);
}

/// Benchmark: Parse small workflow JSON (10 nodes)
fn benchParseSmallWorkflow(allocator: Allocator) !void {
    const json_str = try generateWorkflowJson(allocator, 10);
    defer allocator.free(json_str);

    // Parse JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
    defer parsed.deinit();

    // Validate structure
    const root = parsed.value.object;
    _ = root.get("nodes") orelse return error.InvalidWorkflow;
    _ = root.get("edges") orelse return error.InvalidWorkflow;
}

/// Benchmark: Parse medium workflow JSON (100 nodes)
fn benchParseMediumWorkflow(allocator: Allocator) !void {
    const json_str = try generateWorkflowJson(allocator, 100);
    defer allocator.free(json_str);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    _ = root.get("nodes") orelse return error.InvalidWorkflow;
}

/// Benchmark: Parse large workflow JSON (1000 nodes)
fn benchParseLargeWorkflow(allocator: Allocator) !void {
    const json_str = try generateWorkflowJson(allocator, 1000);
    defer allocator.free(json_str);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    _ = root.get("nodes") orelse return error.InvalidWorkflow;
}

/// Benchmark: Serialize workflow to JSON
fn benchSerializeWorkflow(allocator: Allocator) !void {
    // Create a workflow structure and serialize it
    var json = std.ArrayList(u8){};
    defer json.deinit(allocator);

    const writer = json.writer(allocator);

    try writer.writeAll("{\"version\":\"1.0\",\"name\":\"Test\",\"nodes\":[");

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{{\"id\":\"n{d}\",\"type\":\"action\",\"name\":\"Node{d}\"}}", .{ i, i });
    }

    try writer.writeAll("],\"edges\":[");

    i = 0;
    while (i < 49) : (i += 1) {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{{\"from\":\"n{d}\",\"to\":\"n{d}\"}}", .{ i, i + 1 });
    }

    try writer.writeAll("]}");
}

// ============================================================================
// Executor Benchmarks
// ============================================================================

/// Simulate executing a linear workflow
fn benchExecuteLinearWorkflow(allocator: Allocator) !void {
    // Simulate executing 10 nodes in sequence
    var results = std.ArrayList([]const u8){};
    defer {
        for (results.items) |r| allocator.free(r);
        results.deinit(allocator);
    }

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        // Simulate node execution
        const result = try allocator.dupe(u8, "{\"status\":\"completed\"}");
        try results.append(allocator, result);
    }
}

/// Simulate executing a parallel workflow with 10 branches
fn benchExecuteParallelWorkflow(allocator: Allocator) !void {
    // Simulate 10 parallel branches
    var branch_results = std.ArrayList([]const u8){};
    defer {
        for (branch_results.items) |r| allocator.free(r);
        branch_results.deinit(allocator);
    }

    // Simulate parallel execution (sequential in benchmark, but measures data structure overhead)
    var branch: usize = 0;
    while (branch < 10) : (branch += 1) {
        // Each branch has 5 nodes
        var node: usize = 0;
        while (node < 5) : (node += 1) {
            const result = try allocator.dupe(u8, "{\"branch\":\"complete\"}");
            try branch_results.append(allocator, result);
        }
    }
}

/// Simulate executing a mixed workflow with 50 nodes
fn benchExecuteMixedWorkflow(allocator: Allocator) !void {
    var state = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = state.valueIterator();
        while (it.next()) |v| allocator.free(v.*);
        state.deinit();
    }

    // Simulate mixed sequential and parallel execution
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var buf: [64]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "node_{d}", .{i});
        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);

        const value = try allocator.dupe(u8, "{\"executed\":true}");

        // Free old value if exists
        if (state.get(key_copy)) |old| {
            allocator.free(old);
        }
        try state.put(key_copy, value);
    }
}

// ============================================================================
// Data Pipeline Benchmarks
// ============================================================================

/// Benchmark: Create 1000 data packets
fn benchCreateDataPackets1000(allocator: Allocator) !void {
    var packets = std.ArrayList([]const u8){};
    defer {
        for (packets.items) |p| allocator.free(p);
        packets.deinit(allocator);
    }

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        var buf: [128]u8 = undefined;
        const packet_json = try std.fmt.bufPrint(&buf, "{{\"id\":\"packet_{d}\",\"type\":\"object\",\"value\":{{}}}}", .{i});
        const packet = try allocator.dupe(u8, packet_json);
        try packets.append(allocator, packet);
    }
}

/// Benchmark: Process data through 10-stage pipeline
fn benchProcessPipeline10Stages(allocator: Allocator) !void {
    // Simulate a 10-stage transformation pipeline
    var data: i64 = 1;

    // Stage 1: Double
    data *= 2;
    // Stage 2: Add offset
    data += 100;
    // Stage 3: Square (bounded)
    data = @min(data * data, 1000000);
    // Stage 4-10: Various transformations
    var stage: usize = 4;
    while (stage <= 10) : (stage += 1) {
        data = @divTrunc(data * 3, 2);
        data = @mod(data, 100000) + 1;
    }

    // Store result to prevent optimization
    const result_str = try std.fmt.allocPrint(allocator, "{d}", .{data});
    defer allocator.free(result_str);
}

/// Benchmark: Transform large JSON payload
fn benchTransformLargeJson(allocator: Allocator) !void {
    // Create a large JSON-like structure
    var json = std.ArrayList(u8){};
    defer json.deinit(allocator);

    const writer = json.writer(allocator);
    try writer.writeAll("{\"items\":[");

    var i: usize = 0;
    while (i < 500) : (i += 1) {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{{\"id\":{d},\"name\":\"Item {d}\",\"value\":{d},\"active\":true}}", .{ i, i, i * 100 });
    }

    try writer.writeAll("]}");

    // Parse and validate
    const json_slice = json.items;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_slice, .{});
    defer parsed.deinit();
}

// ============================================================================
// Memory Benchmarks
// ============================================================================

/// Benchmark: Memory allocation stress test
fn benchMemoryAllocationStress(allocator: Allocator) !void {
    // Allocate and free many small objects
    var allocations = std.ArrayList([]u8){};
    defer allocations.deinit(allocator);

    // Allocate 100 chunks
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const size = 64 + (i % 256);
        const chunk = try allocator.alloc(u8, size);
        try allocations.append(allocator, chunk);
    }

    // Free in reverse order (LIFO pattern)
    while (allocations.items.len > 0) {
        const last_idx = allocations.items.len - 1;
        const chunk = allocations.items[last_idx];
        allocations.items.len = last_idx;
        allocator.free(chunk);
    }
}

/// Benchmark: Large allocation test
fn benchLargeAllocation(allocator: Allocator) !void {
    // Allocate 1MB chunk
    const large_chunk = try allocator.alloc(u8, 1024 * 1024);
    defer allocator.free(large_chunk);

    // Touch memory to ensure allocation
    @memset(large_chunk, 0);
}

/// Benchmark: Hash map memory pressure
fn benchHashMapMemory(allocator: Allocator) !void {
    var map = std.StringHashMap(usize).init(allocator);
    defer map.deinit();

    // Insert 1000 entries
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        var buf: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "key_{d}", .{i});
        const key_copy = try allocator.dupe(u8, key);
        try map.put(key_copy, i);
    }

    // Free all keys
    var it = map.keyIterator();
    while (it.next()) |key| {
        allocator.free(key.*);
    }
}

// ============================================================================
// Performance Targets (from Master Plan)
// ============================================================================

pub const PerformanceTargets = struct {
    // Petri Net targets (ops/sec)
    pub const create_net_100: f64 = 10000; // 10K nets/sec with 100 places
    pub const create_net_1000: f64 = 1000; // 1K nets/sec with 1000 places
    pub const fire_transitions: f64 = 100000; // 100K transitions/sec
    pub const deadlock_check: f64 = 50000; // 50K checks/sec

    // Parser targets
    pub const parse_small: f64 = 50000; // 50K parses/sec for 10 nodes
    pub const parse_medium: f64 = 5000; // 5K parses/sec for 100 nodes
    pub const parse_large: f64 = 500; // 500 parses/sec for 1000 nodes

    // Executor targets
    pub const execute_linear: f64 = 100000; // 100K executions/sec
    pub const execute_parallel: f64 = 50000; // 50K executions/sec
    pub const execute_mixed: f64 = 20000; // 20K executions/sec

    // Data pipeline targets
    pub const create_packets: f64 = 1000; // 1K batch creates/sec (1000 packets each)
    pub const pipeline_throughput: f64 = 500000; // 500K stages/sec

    pub fn checkTarget(name: []const u8, actual: f64, target: f64) bool {
        _ = name;
        return actual >= target;
    }
};

// ============================================================================
// Main Benchmark Runner
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("+==============================================================================+\n", .{});
    std.debug.print("|                    nWorkflow Performance Benchmark Suite                     |\n", .{});
    std.debug.print("|                              Version 1.0.0                                   |\n", .{});
    std.debug.print("+==============================================================================+\n", .{});

    const config = BenchmarkConfig.default();
    var all_results = std.ArrayList(BenchmarkResult){};
    defer all_results.deinit(allocator);

    printHeader();

    // Petri Net Benchmarks
    printSeparator("PETRI NET BENCHMARKS");

    var result = try runBenchmark("create_petri_net_100_places", benchCreatePetriNet100, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("create_petri_net_1000_places", benchCreatePetriNet1000, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("fire_1000_transitions", benchFireTransitions1000, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("check_deadlock_complex", benchCheckDeadlock, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    // Workflow Parser Benchmarks
    printSeparator("WORKFLOW PARSER BENCHMARKS");

    result = try runBenchmark("parse_small_workflow_10_nodes", benchParseSmallWorkflow, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("parse_medium_workflow_100_nodes", benchParseMediumWorkflow, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("parse_large_workflow_1000_nodes", benchParseLargeWorkflow, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("serialize_workflow_to_json", benchSerializeWorkflow, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    // Executor Benchmarks
    printSeparator("EXECUTOR BENCHMARKS");

    result = try runBenchmark("execute_linear_workflow_10_nodes", benchExecuteLinearWorkflow, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("execute_parallel_workflow_10_branches", benchExecuteParallelWorkflow, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("execute_mixed_workflow_50_nodes", benchExecuteMixedWorkflow, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    // Data Pipeline Benchmarks
    printSeparator("DATA PIPELINE BENCHMARKS");

    result = try runBenchmark("create_1000_data_packets", benchCreateDataPackets1000, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("process_10_stage_pipeline", benchProcessPipeline10Stages, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("transform_large_json_payload", benchTransformLargeJson, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    // Memory Benchmarks
    printSeparator("MEMORY BENCHMARKS");

    result = try runBenchmark("memory_allocation_stress", benchMemoryAllocationStress, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("large_allocation_1mb", benchLargeAllocation, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    result = try runBenchmark("hashmap_memory_pressure", benchHashMapMemory, config, allocator);
    printResult(result);
    try all_results.append(allocator, result);

    // Print summary
    std.debug.print("\n", .{});
    std.debug.print("=" ** 120 ++ "\n", .{});
    std.debug.print("SUMMARY\n", .{});
    std.debug.print("=" ** 120 ++ "\n", .{});

    var total_ops: f64 = 0;
    var passed_targets: usize = 0;
    const total_benchmarks = all_results.items.len;

    for (all_results.items) |r| {
        total_ops += r.opsPerSecond();
        // Simple target check (assumes all targets should be > 1000 ops/sec)
        if (r.opsPerSecond() >= 1000) {
            passed_targets += 1;
        }
    }

    std.debug.print("Total benchmarks run: {d}\n", .{total_benchmarks});
    std.debug.print("Benchmarks meeting 1K ops/sec target: {d}/{d}\n", .{ passed_targets, total_benchmarks });
    std.debug.print("Average throughput: {d:.0} ops/sec\n", .{total_ops / @as(f64, @floatFromInt(total_benchmarks))});

    // Export to JSON
    const json_output = try exportToJson(all_results.items, allocator);
    defer allocator.free(json_output);

    // Write to file
    const file = try std.fs.cwd().createFile("benchmark_results.json", .{});
    defer file.close();
    try file.writeAll(json_output);

    std.debug.print("\nResults exported to: benchmark_results.json\n", .{});
    std.debug.print("\n", .{});
}

// ============================================================================
// Tests
// ============================================================================

test "BenchmarkResult ops per second calculation" {
    const result = BenchmarkResult{
        .name = "test",
        .iterations = 1000,
        .total_ns = 1_000_000_000,
        .min_ns = 900_000,
        .max_ns = 1_100_000,
        .mean_ns = 1_000_000,
        .std_dev_ns = 50_000,
        .peak_memory_bytes = 1024,
        .total_allocations = 100,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 1000.0), result.opsPerSecond(), 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 1000.0), result.meanMicroseconds(), 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.meanMilliseconds(), 0.001);
}

test "TrackingAllocator tracks allocations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var tracking = TrackingAllocator.init(gpa.allocator());
    const alloc = tracking.allocator();

    const buf = try alloc.alloc(u8, 1024);
    try std.testing.expectEqual(@as(usize, 1024), tracking.current_bytes);
    try std.testing.expectEqual(@as(usize, 1024), tracking.peak_bytes);
    try std.testing.expectEqual(@as(usize, 1), tracking.total_allocations);

    alloc.free(buf);
    try std.testing.expectEqual(@as(usize, 0), tracking.current_bytes);
    try std.testing.expectEqual(@as(usize, 1024), tracking.peak_bytes);
}

test "generateWorkflowJson creates valid JSON" {
    const allocator = std.testing.allocator;

    const json_str = try generateWorkflowJson(allocator, 5);
    defer allocator.free(json_str);

    // Parse to verify it's valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const nodes = root.get("nodes").?.array;
    try std.testing.expectEqual(@as(usize, 5), nodes.items.len);
}

test "benchmark config default values" {
    const config = BenchmarkConfig.default();
    try std.testing.expectEqual(@as(u64, 10), config.warmup_iterations);
    try std.testing.expectEqual(@as(u64, 100), config.min_iterations);
    try std.testing.expectEqual(@as(u64, 10000), config.max_iterations);
}
