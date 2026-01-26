// Memory Profiler - Heap Allocation Tracking and Leak Detection
// Monitors memory allocations to identify leaks and hotspots

const std = @import("std");
const builtin = @import("builtin");
const Thread = std.Thread;
const Allocator = std.mem.Allocator;

pub const MemoryProfileConfig = struct {
    track_allocations: bool = true,
    sample_rate: u32 = 100, // Track 1 in N allocations
    leak_detection: bool = true,
    capture_stack_traces: bool = true,
    max_tracked_allocs: usize = 100_000,
};

pub const AllocationInfo = struct {
    address: usize,
    size: usize,
    timestamp_ns: i64,
    thread_id: u32,
    stack_trace: []const u8,
    freed: bool = false,
    freed_at: i64 = 0,
    allocator: Allocator,

    pub fn deinit(self: *AllocationInfo) void {
        self.allocator.free(self.stack_trace);
    }
};

pub const MemoryStats = struct {
    total_allocated: u64,
    total_freed: u64,
    peak_usage: u64,
    current_usage: u64,
    allocation_count: u64,
    free_count: u64,
    leaked_bytes: u64,
    leaked_count: u64,
};

pub const AllocationHotspot = struct {
    stack_trace: []const u8,
    total_bytes: u64,
    count: u64,
    avg_size: f64,
};

pub const MemoryProfile = struct {
    allocations: std.AutoHashMap(usize, AllocationInfo),
    stats: MemoryStats,
    peak_usage_time_ns: i64,
    start_time_ns: i64,
    allocator: Allocator,
    mutex: Thread.Mutex,

    pub fn init(allocator: Allocator) MemoryProfile {
        return .{
            .allocations = std.AutoHashMap(usize, AllocationInfo).init(allocator),
            .stats = .{
                .total_allocated = 0,
                .total_freed = 0,
                .peak_usage = 0,
                .current_usage = 0,
                .allocation_count = 0,
                .free_count = 0,
                .leaked_bytes = 0,
                .leaked_count = 0,
            },
            .peak_usage_time_ns = 0,
            .start_time_ns = std.time.nanoTimestamp(),
            .allocator = allocator,
            .mutex = Thread.Mutex{},
        };
    }

    pub fn deinit(self: *MemoryProfile) void {
        var iter = self.allocations.valueIterator();
        while (iter.next()) |info| {
            var alloc_info = info.*;
            alloc_info.deinit();
        }
        self.allocations.deinit();
    }

    pub fn getHotspots(self: *MemoryProfile, limit: usize) ![]AllocationHotspot {
        self.mutex.lock();
        defer self.mutex.unlock();

        var trace_stats = std.StringHashMap(struct { bytes: u64, count: u64 }).init(self.allocator);
        defer trace_stats.deinit();

        // Aggregate by stack trace
        var iter = self.allocations.valueIterator();
        while (iter.next()) |info| {
            if (!info.freed) {
                const entry = try trace_stats.getOrPut(info.stack_trace);
                if (!entry.found_existing) {
                    entry.value_ptr.* = .{ .bytes = 0, .count = 0 };
                }
                entry.value_ptr.bytes += info.size;
                entry.value_ptr.count += 1;
            }
        }

        // Convert to array
        var hotspots = std.ArrayList(AllocationHotspot){};
        errdefer hotspots.deinit();

        var trace_iter = trace_stats.iterator();
        while (trace_iter.next()) |entry| {
            try hotspots.append(.{
                .stack_trace = entry.key_ptr.*,
                .total_bytes = entry.value_ptr.bytes,
                .count = entry.value_ptr.count,
                .avg_size = @as(f64, @floatFromInt(entry.value_ptr.bytes)) / @as(f64, @floatFromInt(entry.value_ptr.count)),
            });
        }

        // Sort by total bytes (descending)
        std.sort.pdq(AllocationHotspot, hotspots.items, {}, struct {
            fn lessThan(_: void, a: AllocationHotspot, b: AllocationHotspot) bool {
                return a.total_bytes > b.total_bytes;
            }
        }.lessThan);

        // Return top N
        const result_len = @min(limit, hotspots.items.len);
        const result = try self.allocator.alloc(AllocationHotspot, result_len);
        @memcpy(result, hotspots.items[0..result_len]);
        hotspots.deinit();

        return result;
    }

    pub fn detectLeaks(self: *MemoryProfile) ![]AllocationInfo {
        self.mutex.lock();
        defer self.mutex.unlock();

        var leaks = std.ArrayList(AllocationInfo){};
        errdefer leaks.deinit();

        var iter = self.allocations.valueIterator();
        while (iter.next()) |info| {
            if (!info.freed) {
                try leaks.append(info.*);
            }
        }

        self.stats.leaked_count = leaks.items.len;
        self.stats.leaked_bytes = 0;
        for (leaks.items) |leak| {
            self.stats.leaked_bytes += leak.size;
        }

        return leaks.toOwnedSlice();
    }

    pub fn toJson(self: *MemoryProfile, writer: anytype) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try writer.writeAll("{");
        try writer.print("\"total_allocated_mb\":{d:.2},", .{@as(f64, @floatFromInt(self.stats.total_allocated)) / 1_048_576.0});
        try writer.print("\"total_freed_mb\":{d:.2},", .{@as(f64, @floatFromInt(self.stats.total_freed)) / 1_048_576.0});
        try writer.print("\"peak_usage_mb\":{d:.2},", .{@as(f64, @floatFromInt(self.stats.peak_usage)) / 1_048_576.0});
        try writer.print("\"current_usage_mb\":{d:.2},", .{@as(f64, @floatFromInt(self.stats.current_usage)) / 1_048_576.0});
        try writer.print("\"allocation_count\":{d},", .{self.stats.allocation_count});
        try writer.print("\"free_count\":{d},", .{self.stats.free_count});
        try writer.print("\"leaked_mb\":{d:.2},", .{@as(f64, @floatFromInt(self.stats.leaked_bytes)) / 1_048_576.0});
        try writer.print("\"leaked_count\":{d},", .{self.stats.leaked_count});

        // Add hotspots
        try writer.writeAll("\"hotspots\":[");
        const hotspots = try self.getHotspots(10);
        defer self.allocator.free(hotspots);

        for (hotspots, 0..) |hotspot, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll("{");
            try writer.print("\"stack_trace\":\"{s}\",", .{hotspot.stack_trace});
            try writer.print("\"total_mb\":{d:.2},", .{@as(f64, @floatFromInt(hotspot.total_bytes)) / 1_048_576.0});
            try writer.print("\"count\":{d},", .{hotspot.count});
            try writer.print("\"avg_size_kb\":{d:.2}", .{hotspot.avg_size / 1024.0});
            try writer.writeAll("}");
        }
        try writer.writeAll("]");
        try writer.writeAll("}");
    }
};

pub const MemoryProfiler = struct {
    config: MemoryProfileConfig,
    profile: MemoryProfile,
    is_running: std.atomic.Value(bool),
    sample_counter: std.atomic.Value(u32),
    base_allocator: Allocator,
    allocator: Allocator,

    pub fn init(base_allocator: Allocator, config: MemoryProfileConfig) !*MemoryProfiler {
        const profiler = try base_allocator.create(MemoryProfiler);
        profiler.* = .{
            .config = config,
            .profile = MemoryProfile.init(base_allocator),
            .is_running = std.atomic.Value(bool).init(false),
            .sample_counter = std.atomic.Value(u32).init(0),
            .base_allocator = base_allocator,
            .allocator = base_allocator,
        };
        return profiler;
    }

    pub fn deinit(self: *MemoryProfiler) void {
        self.stop();
        self.profile.deinit();
        self.base_allocator.destroy(self);
    }

    pub fn start(self: *MemoryProfiler) void {
        self.is_running.store(true, .release);
    }

    pub fn stop(self: *MemoryProfiler) void {
        self.is_running.store(false, .release);
    }

    pub fn getProfile(self: *MemoryProfiler) *MemoryProfile {
        return &self.profile;
    }

    pub fn trackAllocation(self: *MemoryProfiler, address: usize, size: usize) !void {
        if (!self.is_running.load(.acquire)) return;
        if (!self.config.track_allocations) return;

        // Sample allocations
        const counter = self.sample_counter.fetchAdd(1, .monotonic);
        if (counter % self.config.sample_rate != 0) return;

        self.profile.mutex.lock();
        defer self.profile.mutex.unlock();

        // Limit tracked allocations
        if (self.profile.allocations.count() >= self.config.max_tracked_allocs) {
            return;
        }

        const stack_trace = if (self.config.capture_stack_traces)
            try self.captureStackTrace()
        else
            try self.base_allocator.dupe(u8, "");

        const info = AllocationInfo{
            .address = address,
            .size = size,
            .timestamp_ns = std.time.nanoTimestamp(),
            .thread_id = Thread.getCurrentId(),
            .stack_trace = stack_trace,
            .allocator = self.base_allocator,
        };

        try self.profile.allocations.put(address, info);

        // Update stats
        self.profile.stats.total_allocated += size;
        self.profile.stats.current_usage += size;
        self.profile.stats.allocation_count += 1;

        if (self.profile.stats.current_usage > self.profile.stats.peak_usage) {
            self.profile.stats.peak_usage = self.profile.stats.current_usage;
            self.profile.peak_usage_time_ns = std.time.nanoTimestamp();
        }
    }

    pub fn trackFree(self: *MemoryProfiler, address: usize) void {
        if (!self.is_running.load(.acquire)) return;
        if (!self.config.track_allocations) return;

        self.profile.mutex.lock();
        defer self.profile.mutex.unlock();

        if (self.profile.allocations.getPtr(address)) |info| {
            info.freed = true;
            info.freed_at = std.time.nanoTimestamp();

            self.profile.stats.total_freed += info.size;
            self.profile.stats.current_usage -= info.size;
            self.profile.stats.free_count += 1;
        }
    }

    fn captureStackTrace(self: *MemoryProfiler) ![]const u8 {
        var buffer: [4096]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buffer);
        const writer = fbs.writer();

        if (builtin.os.tag == .linux or builtin.os.tag == .macos) {
            const c = @cImport({
                @cInclude("execinfo.h");
            });

            var addresses: [32]usize = undefined;
            const count = c.backtrace(@ptrCast(&addresses), addresses.len);

            // Format addresses as hex strings
            for (0..@intCast(count)) |i| {
                try writer.print("0x{x}\n", .{addresses[i]});
            }
        } else {
            // Fallback
            try writer.writeAll("stack_trace_unavailable");
        }

        const trace_len = fbs.getPos() catch 0;
        return try self.base_allocator.dupe(u8, buffer[0..trace_len]);
    }

    // Allocator interface wrapper
    pub fn allocator(self: *MemoryProfiler) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *MemoryProfiler = @ptrCast(@alignCast(ctx));
        const result = self.base_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result) |ptr| {
            self.trackAllocation(@intFromPtr(ptr), len) catch {};
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *MemoryProfiler = @ptrCast(@alignCast(ctx));
        const old_size = buf.len;
        const success = self.base_allocator.rawResize(buf, buf_align, new_len, ret_addr);
        if (success) {
            self.trackFree(@intFromPtr(buf.ptr));
            self.trackAllocation(@intFromPtr(buf.ptr), new_len) catch {};
        }
        return success;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *MemoryProfiler = @ptrCast(@alignCast(ctx));
        self.trackFree(@intFromPtr(buf.ptr));
        self.base_allocator.rawFree(buf, buf_align, ret_addr);
    }
};

// Testing
test "MemoryProfiler basic" {
    const allocator = std.testing.allocator;

    const config = MemoryProfileConfig{
        .track_allocations = true,
        .sample_rate = 1, // Track all allocations in test
        .leak_detection = true,
        .capture_stack_traces = false, // Disable for faster test
    };

    var profiler = try MemoryProfiler.init(allocator, config);
    defer profiler.deinit();

    profiler.start();

    // Simulate some allocations
    try profiler.trackAllocation(0x1000, 1024);
    try profiler.trackAllocation(0x2000, 2048);
    profiler.trackFree(0x1000);

    const profile = profiler.getProfile();
    try std.testing.expect(profile.stats.allocation_count == 2);
    try std.testing.expect(profile.stats.free_count == 1);
    try std.testing.expect(profile.stats.current_usage == 2048);
}
