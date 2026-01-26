// CPU Profiler - Statistical Sampling Profiler
// Captures call stacks at regular intervals to identify hot functions

const std = @import("std");
const builtin = @import("builtin");
const Thread = std.Thread;
const Allocator = std.mem.Allocator;

pub const CpuProfileConfig = struct {
    sample_rate_hz: u32 = 1000,
    max_stack_depth: u32 = 128,
    capture_threads: bool = true,
    persist_to_hana: bool = true,
};

pub const StackFrame = struct {
    function_name: []const u8,
    file_path: []const u8,
    line_number: u32,
    address: usize,
};

pub const Sample = struct {
    timestamp_ns: i64,
    thread_id: u32,
    stack: []StackFrame,
    allocator: Allocator,

    pub fn deinit(self: *Sample) void {
        for (self.stack) |frame| {
            self.allocator.free(frame.function_name);
            self.allocator.free(frame.file_path);
        }
        self.allocator.free(self.stack);
    }
};

pub const FunctionStats = struct {
    name: []const u8,
    file: []const u8,
    line: u32,
    samples: u64,
    percent: f64,
};

pub const CpuProfile = struct {
    samples: std.ArrayList(Sample),
    total_samples: u64,
    duration_ns: i64,
    start_time_ns: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) CpuProfile {
        return .{
            .samples = std.ArrayList(Sample){},
            .total_samples = 0,
            .duration_ns = 0,
            .start_time_ns = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CpuProfile) void {
        for (self.samples.items) |*sample| {
            sample.deinit();
        }
        self.samples.deinit();
    }

    pub fn getTopFunctions(self: *const CpuProfile, limit: usize) ![]FunctionStats {
        var function_counts = std.StringHashMap(u64).init(self.allocator);
        defer function_counts.deinit();

        var function_info = std.StringHashMap(struct { file: []const u8, line: u32 }).init(self.allocator);
        defer function_info.deinit();

        // Count samples per function
        for (self.samples.items) |sample| {
            if (sample.stack.len > 0) {
                const frame = sample.stack[0];
                const entry = try function_counts.getOrPut(frame.function_name);
                if (!entry.found_existing) {
                    entry.value_ptr.* = 0;
                    try function_info.put(frame.function_name, .{
                        .file = frame.file_path,
                        .line = frame.line_number,
                    });
                }
                entry.value_ptr.* += 1;
            }
        }

        // Convert to sorted array
        var stats_list = std.ArrayList(FunctionStats){};
        errdefer stats_list.deinit();

        var iter = function_counts.iterator();
        while (iter.next()) |entry| {
            const info = function_info.get(entry.key_ptr.*) orelse continue;
            try stats_list.append(.{
                .name = entry.key_ptr.*,
                .file = info.file,
                .line = info.line,
                .samples = entry.value_ptr.*,
                .percent = (@as(f64, @floatFromInt(entry.value_ptr.*)) / @as(f64, @floatFromInt(self.total_samples))) * 100.0,
            });
        }

        // Sort by sample count (descending)
        std.sort.pdq(FunctionStats, stats_list.items, {}, struct {
            fn lessThan(_: void, a: FunctionStats, b: FunctionStats) bool {
                return a.samples > b.samples;
            }
        }.lessThan);

        // Return top N
        const result_len = @min(limit, stats_list.items.len);
        const result = try self.allocator.alloc(FunctionStats, result_len);
        @memcpy(result, stats_list.items[0..result_len]);
        stats_list.deinit();

        return result;
    }

    pub fn toJson(self: *const CpuProfile, writer: anytype) !void {
        try writer.writeAll("{");
        try writer.print("\"total_samples\":{d},", .{self.total_samples});
        try writer.print("\"duration_seconds\":{d:.2},", .{@as(f64, @floatFromInt(self.duration_ns)) / 1_000_000_000.0});

        try writer.writeAll("\"top_functions\":[");
        const top = try self.getTopFunctions(20);
        defer self.allocator.free(top);

        for (top, 0..) |stat, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll("{");
            try writer.print("\"name\":\"{s}\",", .{stat.name});
            try writer.print("\"file\":\"{s}\",", .{stat.file});
            try writer.print("\"line\":{d},", .{stat.line});
            try writer.print("\"samples\":{d},", .{stat.samples});
            try writer.print("\"percent\":{d:.2}", .{stat.percent});
            try writer.writeAll("}");
        }
        try writer.writeAll("]");
        try writer.writeAll("}");
    }
};

pub const CpuProfiler = struct {
    config: CpuProfileConfig,
    profile: CpuProfile,
    is_running: std.atomic.Value(bool),
    sampler_thread: ?Thread,
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: CpuProfileConfig) !*CpuProfiler {
        const profiler = try allocator.create(CpuProfiler);
        profiler.* = .{
            .config = config,
            .profile = CpuProfile.init(allocator),
            .is_running = std.atomic.Value(bool).init(false),
            .sampler_thread = null,
            .allocator = allocator,
        };
        return profiler;
    }

    pub fn deinit(self: *CpuProfiler) void {
        self.stop();
        self.profile.deinit();
        self.allocator.destroy(self);
    }

    pub fn start(self: *CpuProfiler) !void {
        if (self.is_running.load(.acquire)) {
            return error.AlreadyRunning;
        }

        self.profile = CpuProfile.init(self.allocator);
        self.profile.start_time_ns = std.time.nanoTimestamp();
        self.is_running.store(true, .release);

        self.sampler_thread = try Thread.spawn(.{}, samplerThreadFn, .{self});
    }

    pub fn stop(self: *CpuProfiler) void {
        if (!self.is_running.load(.acquire)) {
            return;
        }

        self.is_running.store(false, .release);

        if (self.sampler_thread) |thread| {
            thread.join();
            self.sampler_thread = null;
        }

        self.profile.duration_ns = std.time.nanoTimestamp() - self.profile.start_time_ns;
    }

    pub fn getProfile(self: *CpuProfiler) *const CpuProfile {
        return &self.profile;
    }

    fn samplerThreadFn(self: *CpuProfiler) void {
        const interval_ns = @divFloor(1_000_000_000, self.config.sample_rate_hz);

        while (self.is_running.load(.acquire)) {
            self.captureSample() catch |err| {
                std.log.warn("Failed to capture sample: {}", .{err});
            };

            std.time.sleep(interval_ns);
        }
    }

    fn captureSample(self: *CpuProfiler) !void {
        const sample = try self.captureStackTrace();
        try self.profile.samples.append(sample);
        self.profile.total_samples += 1;
    }

    fn captureStackTrace(self: *CpuProfiler) !Sample {
        var stack_frames = try self.allocator.alloc(StackFrame, self.config.max_stack_depth);
        var frame_count: usize = 0;

        // Platform-specific backtrace capture
        if (builtin.os.tag == .linux or builtin.os.tag == .macos) {
            var addresses: [128]usize = undefined;
            const count = captureBacktrace(&addresses);

            frame_count = @min(count, self.config.max_stack_depth);
            for (0..frame_count) |i| {
                stack_frames[i] = try self.resolveAddress(addresses[i]);
            }
        } else {
            // Fallback: use debug.captureStackTrace
            var stack_trace: std.builtin.StackTrace = undefined;
            std.debug.captureStackTrace(null, &stack_trace);

            frame_count = @min(stack_trace.index, self.config.max_stack_depth);
            for (0..frame_count) |i| {
                const addr = stack_trace.instruction_addresses[i];
                stack_frames[i] = try self.resolveAddress(addr);
            }
        }

        const final_stack = try self.allocator.realloc(stack_frames, frame_count);

        return Sample{
            .timestamp_ns = std.time.nanoTimestamp(),
            .thread_id = Thread.getCurrentId(),
            .stack = final_stack,
            .allocator = self.allocator,
        };
    }

    fn resolveAddress(self: *CpuProfiler, addr: usize) !StackFrame {
        // Try to resolve symbol information
        // This is a simplified version - real implementation would use libunwind or similar
        var function_name: []const u8 = "unknown";
        var file_path: []const u8 = "unknown";
        var line_number: u32 = 0;

        if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) {
            // In debug builds, try to get debug info
            const debug_info = std.debug.getSelfDebugInfo() catch null;
            if (debug_info) |info| {
                const module = info.getModuleForAddress(addr) catch null;
                if (module) |mod| {
                    const symbol = mod.getSymbolAtAddress(self.allocator, addr) catch null;
                    if (symbol) |sym| {
                        function_name = sym.name;
                        file_path = sym.compile_unit_name orelse "unknown";
                        line_number = sym.line_info.?.line;
                    }
                }
            }
        }

        return StackFrame{
            .function_name = try self.allocator.dupe(u8, function_name),
            .file_path = try self.allocator.dupe(u8, file_path),
            .line_number = line_number,
            .address = addr,
        };
    }
};

// Platform-specific backtrace capture
fn captureBacktrace(addresses: []usize) usize {
    if (builtin.os.tag == .linux or builtin.os.tag == .macos) {
        // Use backtrace() from libc if available
        const c = @cImport({
            @cInclude("execinfo.h");
        });
        const count = c.backtrace(@ptrCast(addresses.ptr), @intCast(addresses.len));
        return @intCast(count);
    }
    return 0;
}

// Testing
test "CpuProfiler basic" {
    const allocator = std.testing.allocator;

    const config = CpuProfileConfig{
        .sample_rate_hz = 100,
        .max_stack_depth = 32,
        .capture_threads = true,
    };

    var profiler = try CpuProfiler.init(allocator, config);
    defer profiler.deinit();

    try profiler.start();
    std.time.sleep(100 * std.time.ns_per_ms); // Profile for 100ms
    profiler.stop();

    const profile = profiler.getProfile();
    try std.testing.expect(profile.total_samples > 0);
}
