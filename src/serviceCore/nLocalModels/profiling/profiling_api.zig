// Profiling API - HTTP endpoints for performance profiling
// Integrates all profiling components with the inference server

const std = @import("std");
const Allocator = std.mem.Allocator;
const cpu_profiler = @import("cpu_profiler.zig");
const memory_profiler = @import("memory_profiler.zig");
const gpu_monitor = @import("gpu_monitor.zig");
const flamegraph = @import("flamegraph.zig");
const bottleneck_detector = @import("bottleneck_detector.zig");

pub const ProfilingSession = struct {
    id: []const u8,
    name: []const u8,
    cpu_profiler: ?*cpu_profiler.CpuProfiler,
    memory_profiler: ?*memory_profiler.MemoryProfiler,
    gpu_monitor: ?*gpu_monitor.GpuMonitor,
    start_time_ns: i64,
    allocator: Allocator,

    pub fn deinit(self: *ProfilingSession) void {
        if (self.cpu_profiler) |profiler| {
            profiler.deinit();
        }
        if (self.memory_profiler) |profiler| {
            profiler.deinit();
        }
        if (self.gpu_monitor) |monitor| {
            monitor.deinit();
        }
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.destroy(self);
    }
};

pub const ProfilingManager = struct {
    active_sessions: std.StringHashMap(*ProfilingSession),
    continuous_profiling: bool,
    continuous_session: ?*ProfilingSession,
    allocator: Allocator,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: Allocator) ProfilingManager {
        return .{
            .active_sessions = std.StringHashMap(*ProfilingSession).init(allocator),
            .continuous_profiling = false,
            .continuous_session = null,
            .allocator = allocator,
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *ProfilingManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Clean up all sessions
        var iter = self.active_sessions.valueIterator();
        while (iter.next()) |session| {
            session.*.deinit();
        }
        self.active_sessions.deinit();

        if (self.continuous_session) |session| {
            session.deinit();
        }
    }

    pub fn startSession(self: *ProfilingManager, name: []const u8, enable_cpu: bool, enable_memory: bool, enable_gpu: bool) ![]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Generate session ID
        const session_id = try self.generateSessionId();

        const session = try self.allocator.create(ProfilingSession);
        errdefer self.allocator.destroy(session);

        session.* = .{
            .id = session_id,
            .name = try self.allocator.dupe(u8, name),
            .cpu_profiler = if (enable_cpu) blk: {
                const config = cpu_profiler.CpuProfileConfig{};
                const profiler = try cpu_profiler.CpuProfiler.init(self.allocator, config);
                try profiler.start();
                break :blk profiler;
            } else null,
            .memory_profiler = if (enable_memory) blk: {
                const config = memory_profiler.MemoryProfileConfig{};
                const profiler = try memory_profiler.MemoryProfiler.init(self.allocator, config);
                profiler.start();
                break :blk profiler;
            } else null,
            .gpu_monitor = if (enable_gpu) blk: {
                const config = gpu_monitor.GpuMonitorConfig{};
                const monitor = try gpu_monitor.GpuMonitor.init(self.allocator, config);
                try monitor.start();
                break :blk monitor;
            } else null,
            .start_time_ns = std.time.nanoTimestamp(),
            .allocator = self.allocator,
        };

        try self.active_sessions.put(try self.allocator.dupe(u8, session_id), session);

        return session_id;
    }

    pub fn stopSession(self: *ProfilingManager, session_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.active_sessions.get(session_id)) |session| {
            if (session.cpu_profiler) |profiler| {
                profiler.stop();
            }
            if (session.memory_profiler) |profiler| {
                profiler.stop();
            }
            if (session.gpu_monitor) |monitor| {
                monitor.stop();
            }
        } else {
            return error.SessionNotFound;
        }
    }

    pub fn getSession(self: *ProfilingManager, session_id: []const u8) ?*ProfilingSession {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.active_sessions.get(session_id);
    }

    pub fn generateFlameGraph(self: *ProfilingManager, session_id: []const u8, writer: anytype) !void {
        const session = self.getSession(session_id) orelse return error.SessionNotFound;

        if (session.cpu_profiler) |profiler| {
            const profile = profiler.getProfile();
            const config = flamegraph.FlameGraphConfig{};
            var graph = try flamegraph.generateFlameGraph(self.allocator, profile, config);
            defer graph.deinit();

            try graph.generateHtml(writer);
        } else {
            return error.CpuProfilingNotEnabled;
        }
    }

    pub fn getBottleneckReport(self: *ProfilingManager, session_id: []const u8) !bottleneck_detector.BottleneckReport {
        const session = self.getSession(session_id) orelse return error.SessionNotFound;

        const config = bottleneck_detector.BottleneckConfig{};
        var detector = try bottleneck_detector.BottleneckDetector.init(self.allocator, config);
        defer detector.deinit();

        const cpu_profile = if (session.cpu_profiler) |p| p.getProfile() else null;
        const mem_profile = if (session.memory_profiler) |p| p.getProfile() else null;
        const gpu_profile = if (session.gpu_monitor) |m| m.getProfile() else null;

        return try detector.analyze(cpu_profile, mem_profile, gpu_profile);
    }

    pub fn toJson(self: *ProfilingManager, session_id: []const u8, writer: anytype) !void {
        const session = self.getSession(session_id) orelse return error.SessionNotFound;

        try writer.writeAll("{");
        try writer.print("\"session_id\":\"{s}\",", .{session.id});
        try writer.print("\"name\":\"{s}\",", .{session.name});
        try writer.print("\"start_time_ns\":{d},", .{session.start_time_ns});
        try writer.print("\"duration_ms\":{d:.2},", .{@as(f32, @floatFromInt(std.time.nanoTimestamp() - session.start_time_ns)) / 1_000_000.0});

        if (session.cpu_profiler) |profiler| {
            try writer.writeAll("\"cpu\":");
            try profiler.getProfile().toJson(writer);
            try writer.writeAll(",");
        }

        if (session.memory_profiler) |profiler| {
            try writer.writeAll("\"memory\":");
            try profiler.getProfile().toJson(writer);
            try writer.writeAll(",");
        }

        if (session.gpu_monitor) |monitor| {
            try writer.writeAll("\"gpu\":");
            try monitor.getProfile().toJson(writer);
        }

        try writer.writeAll("}");
    }

    fn generateSessionId(self: *ProfilingManager) ![]const u8 {
        var buffer: [36]u8 = undefined;
        const timestamp = std.time.nanoTimestamp();
        const random = @mod(@as(u64, @bitCast(timestamp)), 1000000);

        const id = try std.fmt.bufPrint(&buffer, "prof_{d}_{d}", .{ timestamp, random });
        return try self.allocator.dupe(u8, id);
    }
};

// HTTP Handler Functions
pub fn handleProfileStart(manager: *ProfilingManager, request_body: []const u8, writer: anytype) !void {
    // Parse JSON request
    var enable_cpu = true;
    var enable_memory = true;
    var enable_gpu = true;
    var name: []const u8 = "unnamed_profile";

    // Simple JSON parsing (in production, use a proper JSON parser)
    if (std.mem.indexOf(u8, request_body, "\"cpu\":false") != null) enable_cpu = false;
    if (std.mem.indexOf(u8, request_body, "\"memory\":false") != null) enable_memory = false;
    if (std.mem.indexOf(u8, request_body, "\"gpu\":false") != null) enable_gpu = false;

    // Extract name if provided
    if (std.mem.indexOf(u8, request_body, "\"name\":\"")) |start_idx| {
        const name_start = start_idx + 8;
        if (std.mem.indexOfPos(u8, request_body, name_start, "\"")) |end_idx| {
            name = request_body[name_start..end_idx];
        }
    }

    const session_id = try manager.startSession(name, enable_cpu, enable_memory, enable_gpu);

    try writer.print("{{\"session_id\":\"{s}\",\"status\":\"started\"}}", .{session_id});
}

pub fn handleProfileStop(manager: *ProfilingManager, session_id: []const u8, writer: anytype) !void {
    try manager.stopSession(session_id);
    try manager.toJson(session_id, writer);
}

pub fn handleProfileStatus(manager: *ProfilingManager, session_id: []const u8, writer: anytype) !void {
    try manager.toJson(session_id, writer);
}

pub fn handleProfileFlameGraph(manager: *ProfilingManager, session_id: []const u8, writer: anytype) !void {
    try manager.generateFlameGraph(session_id, writer);
}

pub fn handleProfileBottlenecks(manager: *ProfilingManager, session_id: []const u8, writer: anytype) !void {
    var report = try manager.getBottleneckReport(session_id);
    defer report.deinit();

    try report.toJson(writer);
}

// Testing
test "ProfilingManager basic" {
    const allocator = std.testing.allocator;

    var manager = ProfilingManager.init(allocator);
    defer manager.deinit();

    const session_id = try manager.startSession("test_profile", true, false, false);
    defer allocator.free(session_id);

    std.time.sleep(10 * std.time.ns_per_ms);

    try manager.stopSession(session_id);

    const session = manager.getSession(session_id);
    try std.testing.expect(session != null);
}
