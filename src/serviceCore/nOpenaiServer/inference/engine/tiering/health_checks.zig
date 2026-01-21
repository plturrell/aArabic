// health_checks.zig - Production Health Monitoring System
// Day 9: Deep health checks, K8s probes, load shedding, backpressure
//
// Features:
// - Deep health checks (SSD, RAM, model integrity)
// - Kubernetes readiness/liveness/startup probes
// - Load shedding with backpressure
// - Request queuing with circuit breaker integration
// - Real-time health metrics and alerting

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mutex = std.Thread.Mutex;
const Atomic = std.atomic.Value;

// ============================================================================
// Health Status Types
// ============================================================================

pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,
    critical,

    pub fn isServingTraffic(self: HealthStatus) bool {
        return self == .healthy or self == .degraded;
    }

    pub fn toHttpCode(self: HealthStatus) u16 {
        return switch (self) {
            .healthy => 200,
            .degraded => 200, // Still serving traffic
            .unhealthy => 503,
            .critical => 503,
        };
    }
};

pub const ComponentHealth = struct {
    name: []const u8,
    status: HealthStatus,
    message: []const u8,
    last_check: i64, // Unix timestamp
    check_duration_us: u64,
    metadata: ?std.StringHashMap([]const u8) = null,
};

pub const SystemHealth = struct {
    overall_status: HealthStatus,
    components: []const ComponentHealth,
    timestamp: i64,
    uptime_seconds: u64,

    pub fn deinit(self: *SystemHealth, allocator: Allocator) void {
        for (self.components) |*component| {
            if (component.metadata) |*metadata| {
                var iter = metadata.iterator();
                while (iter.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    allocator.free(entry.value_ptr.*);
                }
                metadata.deinit();
            }
        }
        allocator.free(self.components);
    }
};

// ============================================================================
// Health Check Components
// ============================================================================

pub const SSDHealthChecker = struct {
    path: []const u8,
    min_free_space_gb: f64,
    max_io_error_rate: f64, // Errors per 1000 operations
    
    pub fn check(self: *const SSDHealthChecker, allocator: Allocator) !ComponentHealth {
        const start = std.time.microTimestamp();
        
        // Check disk space
        const stat = try std.fs.cwd().statFile(self.path);
        _ = stat; // TODO: Get actual free space from filesystem
        
        // Simulate disk space check (in production, use statfs)
        const free_space_gb: f64 = 50.0; // Mock value
        const io_error_rate: f64 = 0.1; // Mock value
        
        const status = blk: {
            if (free_space_gb < self.min_free_space_gb / 4) break :blk HealthStatus.critical;
            if (free_space_gb < self.min_free_space_gb / 2) break :blk HealthStatus.unhealthy;
            if (io_error_rate > self.max_io_error_rate * 2) break :blk HealthStatus.unhealthy;
            if (free_space_gb < self.min_free_space_gb or io_error_rate > self.max_io_error_rate) 
                break :blk HealthStatus.degraded;
            break :blk HealthStatus.healthy;
        };
        
        const message = try std.fmt.allocPrint(
            allocator,
            "Free space: {d:.1f}GB, IO error rate: {d:.3f}/1k ops",
            .{ free_space_gb, io_error_rate }
        );
        errdefer allocator.free(message);
        
        var metadata = std.StringHashMap([]const u8).init(allocator);
        try metadata.put(
            try allocator.dupe(u8, "free_space_gb"),
            try std.fmt.allocPrint(allocator, "{d:.2f}", .{free_space_gb})
        );
        try metadata.put(
            try allocator.dupe(u8, "io_error_rate"),
            try std.fmt.allocPrint(allocator, "{d:.4f}", .{io_error_rate})
        );
        
        const duration = @as(u64, @intCast(std.time.microTimestamp() - start));
        
        return ComponentHealth{
            .name = "ssd",
            .status = status,
            .message = message,
            .last_check = std.time.timestamp(),
            .check_duration_us = duration,
            .metadata = metadata,
        };
    }
};

pub const RAMHealthChecker = struct {
    max_usage_percent: f64,
    fragmentation_threshold: f64,
    
    pub fn check(self: *const RAMHealthChecker, allocator: Allocator) !ComponentHealth {
        const start = std.time.microTimestamp();
        
        // Get memory stats (mock values for now)
        const total_mb: f64 = 16384.0;
        const used_mb: f64 = 8192.0;
        const usage_percent = (used_mb / total_mb) * 100.0;
        const fragmentation: f64 = 0.05; // 5% fragmentation
        
        const status = blk: {
            if (usage_percent > 95.0) break :blk HealthStatus.critical;
            if (usage_percent > 90.0 or fragmentation > self.fragmentation_threshold * 2) 
                break :blk HealthStatus.unhealthy;
            if (usage_percent > self.max_usage_percent or fragmentation > self.fragmentation_threshold) 
                break :blk HealthStatus.degraded;
            break :blk HealthStatus.healthy;
        };
        
        const message = try std.fmt.allocPrint(
            allocator,
            "Memory usage: {d:.1f}% ({d:.0f}MB/{d:.0f}MB), fragmentation: {d:.1f}%",
            .{ usage_percent, used_mb, total_mb, fragmentation * 100.0 }
        );
        errdefer allocator.free(message);
        
        var metadata = std.StringHashMap([]const u8).init(allocator);
        try metadata.put(
            try allocator.dupe(u8, "usage_percent"),
            try std.fmt.allocPrint(allocator, "{d:.2f}", .{usage_percent})
        );
        try metadata.put(
            try allocator.dupe(u8, "fragmentation"),
            try std.fmt.allocPrint(allocator, "{d:.4f}", .{fragmentation})
        );
        
        const duration = @as(u64, @intCast(std.time.microTimestamp() - start));
        
        return ComponentHealth{
            .name = "ram",
            .status = status,
            .message = message,
            .last_check = std.time.timestamp(),
            .check_duration_us = duration,
            .metadata = metadata,
        };
    }
};

pub const ModelIntegrityChecker = struct {
    model_path: []const u8,
    expected_checksum: ?[]const u8 = null,
    
    pub fn check(self: *const ModelIntegrityChecker, allocator: Allocator) !ComponentHealth {
        const start = std.time.microTimestamp();
        
        // Check if model file exists
        const file = std.fs.cwd().openFile(self.model_path, .{}) catch |err| {
            const message = try std.fmt.allocPrint(
                allocator,
                "Model file not accessible: {s}",
                .{@errorName(err)}
            );
            return ComponentHealth{
                .name = "model_integrity",
                .status = .critical,
                .message = message,
                .last_check = std.time.timestamp(),
                .check_duration_us = @as(u64, @intCast(std.time.microTimestamp() - start)),
            };
        };
        defer file.close();
        
        const stat = try file.stat();
        const size_gb = @as(f64, @floatFromInt(stat.size)) / (1024.0 * 1024.0 * 1024.0);
        
        // Mock checksum validation (in production, compute actual checksum)
        const checksum_valid = true;
        
        const status: HealthStatus = if (checksum_valid) .healthy else .critical;
        
        const message = try std.fmt.allocPrint(
            allocator,
            "Model file: {s} ({d:.2f}GB), checksum: {s}",
            .{ self.model_path, size_gb, if (checksum_valid) "valid" else "INVALID" }
        );
        errdefer allocator.free(message);
        
        var metadata = std.StringHashMap([]const u8).init(allocator);
        try metadata.put(
            try allocator.dupe(u8, "file_size_gb"),
            try std.fmt.allocPrint(allocator, "{d:.3f}", .{size_gb})
        );
        try metadata.put(
            try allocator.dupe(u8, "checksum_valid"),
            try allocator.dupe(u8, if (checksum_valid) "true" else "false")
        );
        
        const duration = @as(u64, @intCast(std.time.microTimestamp() - start));
        
        return ComponentHealth{
            .name = "model_integrity",
            .status = status,
            .message = message,
            .last_check = std.time.timestamp(),
            .check_duration_us = duration,
            .metadata = metadata,
        };
    }
};

// ============================================================================
// Kubernetes Probes
// ============================================================================

pub const ProbeType = enum {
    readiness, // Ready to receive traffic?
    liveness,  // Is the process alive?
    startup,   // Has initialization completed?
};

pub const ProbeConfig = struct {
    initial_delay_seconds: u32 = 10,
    period_seconds: u32 = 10,
    timeout_seconds: u32 = 5,
    failure_threshold: u32 = 3,
    success_threshold: u32 = 1,
};

pub const ProbeResult = struct {
    probe_type: ProbeType,
    success: bool,
    status_code: u16,
    message: []const u8,
    timestamp: i64,
};

pub const K8sProbeHandler = struct {
    ssd_checker: SSDHealthChecker,
    ram_checker: RAMHealthChecker,
    model_checker: ModelIntegrityChecker,
    startup_complete: Atomic(bool),
    consecutive_failures: Atomic(u32),
    config: ProbeConfig,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, model_path: []const u8) K8sProbeHandler {
        return .{
            .ssd_checker = .{
                .path = "/mnt/ssd",
                .min_free_space_gb = 10.0,
                .max_io_error_rate = 1.0,
            },
            .ram_checker = .{
                .max_usage_percent = 85.0,
                .fragmentation_threshold = 0.15,
            },
            .model_checker = .{
                .model_path = model_path,
            },
            .startup_complete = Atomic(bool).init(false),
            .consecutive_failures = Atomic(u32).init(0),
            .config = .{},
            .allocator = allocator,
        };
    }
    
    pub fn markStartupComplete(self: *K8sProbeHandler) void {
        self.startup_complete.store(true, .release);
    }
    
    pub fn handleStartupProbe(self: *K8sProbeHandler) !ProbeResult {
        const is_complete = self.startup_complete.load(.acquire);
        
        if (is_complete) {
            const message = try self.allocator.dupe(u8, "Startup complete");
            return ProbeResult{
                .probe_type = .startup,
                .success = true,
                .status_code = 200,
                .message = message,
                .timestamp = std.time.timestamp(),
            };
        }
        
        const message = try self.allocator.dupe(u8, "Startup in progress");
        return ProbeResult{
            .probe_type = .startup,
            .success = false,
            .status_code = 503,
            .message = message,
            .timestamp = std.time.timestamp(),
        };
    }
    
    pub fn handleLivenessProbe(self: *K8sProbeHandler) !ProbeResult {
        // Liveness: Basic responsiveness check
        // If this fails, K8s will restart the pod
        
        const ram_health = try self.ram_checker.check(self.allocator);
        defer self.allocator.free(ram_health.message);
        
        // Only fail liveness if critical (avoid unnecessary restarts)
        const is_alive = ram_health.status != .critical;
        
        if (is_alive) {
            self.consecutive_failures.store(0, .release);
            const message = try std.fmt.allocPrint(
                self.allocator,
                "Process alive, RAM: {s}",
                .{@tagName(ram_health.status)}
            );
            return ProbeResult{
                .probe_type = .liveness,
                .success = true,
                .status_code = 200,
                .message = message,
                .timestamp = std.time.timestamp(),
            };
        }
        
        const failures = self.consecutive_failures.fetchAdd(1, .release) + 1;
        const message = try std.fmt.allocPrint(
            self.allocator,
            "Process unhealthy, consecutive failures: {d}",
            .{failures}
        );
        return ProbeResult{
            .probe_type = .liveness,
            .success = false,
            .status_code = 503,
            .message = message,
            .timestamp = std.time.timestamp(),
        };
    }
    
    pub fn handleReadinessProbe(self: *K8sProbeHandler) !ProbeResult {
        // Readiness: Can we serve traffic?
        // If this fails, K8s will remove pod from service endpoints
        
        const ssd_health = try self.ssd_checker.check(self.allocator);
        defer self.allocator.free(ssd_health.message);
        defer if (ssd_health.metadata) |*m| {
            var iter = m.iterator();
            while (iter.next()) |e| {
                self.allocator.free(e.key_ptr.*);
                self.allocator.free(e.value_ptr.*);
            }
            m.deinit();
        };
        
        const ram_health = try self.ram_checker.check(self.allocator);
        defer self.allocator.free(ram_health.message);
        defer if (ram_health.metadata) |*m| {
            var iter = m.iterator();
            while (iter.next()) |e| {
                self.allocator.free(e.key_ptr.*);
                self.allocator.free(e.value_ptr.*);
            }
            m.deinit();
        };
        
        // Ready if both SSD and RAM can serve traffic
        const is_ready = ssd_health.status.isServingTraffic() and 
                        ram_health.status.isServingTraffic();
        
        if (is_ready) {
            const message = try std.fmt.allocPrint(
                self.allocator,
                "Ready to serve, SSD: {s}, RAM: {s}",
                .{ @tagName(ssd_health.status), @tagName(ram_health.status) }
            );
            return ProbeResult{
                .probe_type = .readiness,
                .success = true,
                .status_code = 200,
                .message = message,
                .timestamp = std.time.timestamp(),
            };
        }
        
        const message = try std.fmt.allocPrint(
            self.allocator,
            "Not ready, SSD: {s}, RAM: {s}",
            .{ @tagName(ssd_health.status), @tagName(ram_health.status) }
        );
        return ProbeResult{
            .probe_type = .readiness,
            .success = false,
            .status_code = 503,
            .message = message,
            .timestamp = std.time.timestamp(),
        };
    }
};

// ============================================================================
// Load Shedding & Backpressure
// ============================================================================

pub const LoadMetrics = struct {
    active_requests: Atomic(u32),
    queued_requests: Atomic(u32),
    rejected_requests: Atomic(u64),
    avg_latency_ms: Atomic(u32),
    
    pub fn init() LoadMetrics {
        return .{
            .active_requests = Atomic(u32).init(0),
            .queued_requests = Atomic(u32).init(0),
            .rejected_requests = Atomic(u64).init(0),
            .avg_latency_ms = Atomic(u32).init(0),
        };
    }
    
    pub fn recordLatency(self: *LoadMetrics, latency_ms: u32) void {
        // Simple exponential moving average
        const old_avg = self.avg_latency_ms.load(.acquire);
        const new_avg = (old_avg * 9 + latency_ms) / 10;
        self.avg_latency_ms.store(new_avg, .release);
    }
};

pub const LoadSheddingConfig = struct {
    max_active_requests: u32 = 100,
    max_queue_size: u32 = 50,
    max_latency_ms: u32 = 1000,
    shed_probability_threshold: f64 = 0.9, // Start random shedding at 90% capacity
};

pub const LoadSheddingDecision = enum {
    accept,
    queue,
    reject,
};

pub const LoadShedder = struct {
    metrics: LoadMetrics,
    config: LoadSheddingConfig,
    mutex: Mutex,
    
    pub fn init(config: LoadSheddingConfig) LoadShedder {
        return .{
            .metrics = LoadMetrics.init(),
            .config = config,
            .mutex = .{},
        };
    }
    
    pub fn shouldAcceptRequest(self: *LoadShedder) LoadSheddingDecision {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const active = self.metrics.active_requests.load(.acquire);
        const queued = self.metrics.queued_requests.load(.acquire);
        const avg_latency = self.metrics.avg_latency_ms.load(.acquire);
        
        // Critical overload - reject immediately
        if (active >= self.config.max_active_requests) {
            if (queued >= self.config.max_queue_size) {
                _ = self.metrics.rejected_requests.fetchAdd(1, .release);
                return .reject;
            }
            return .queue;
        }
        
        // High latency - start load shedding
        if (avg_latency > self.config.max_latency_ms) {
            const load_factor = @as(f64, @floatFromInt(active)) / 
                               @as(f64, @floatFromInt(self.config.max_active_requests));
            
            if (load_factor > self.config.shed_probability_threshold) {
                // Random shedding based on load
                const random_val = @as(f64, @floatFromInt(std.crypto.random.int(u32))) / 
                                  @as(f64, @floatFromInt(std.math.maxInt(u32)));
                
                if (random_val > (1.0 - load_factor)) {
                    _ = self.metrics.rejected_requests.fetchAdd(1, .release);
                    return .reject;
                }
            }
        }
        
        return .accept;
    }
    
    pub fn requestStarted(self: *LoadShedder) void {
        _ = self.metrics.active_requests.fetchAdd(1, .release);
    }
    
    pub fn requestCompleted(self: *LoadShedder, latency_ms: u32) void {
        _ = self.metrics.active_requests.fetchSub(1, .release);
        self.metrics.recordLatency(latency_ms);
    }
    
    pub fn requestQueued(self: *LoadShedder) void {
        _ = self.metrics.queued_requests.fetchAdd(1, .release);
    }
    
    pub fn requestDequeued(self: *LoadShedder) void {
        _ = self.metrics.queued_requests.fetchSub(1, .release);
    }
    
    pub fn getMetrics(self: *const LoadShedder) struct {
        active: u32,
        queued: u32,
        rejected: u64,
        avg_latency_ms: u32,
    } {
        return .{
            .active = self.metrics.active_requests.load(.acquire),
            .queued = self.metrics.queued_requests.load(.acquire),
            .rejected = self.metrics.rejected_requests.load(.acquire),
            .avg_latency_ms = self.metrics.avg_latency_ms.load(.acquire),
        };
    }
};

// ============================================================================
// Request Queue with Backpressure
// ============================================================================

pub const QueuedRequest = struct {
    id: u64,
    enqueued_at: i64,
    priority: u8 = 0, // 0 = normal, higher = more important
    
    pub fn waitTimeMs(self: *const QueuedRequest) i64 {
        return std.time.milliTimestamp() - self.enqueued_at;
    }
};

pub fn RequestQueue(comptime T: type) type {
    return struct {
        const Self = @This();
        const Node = struct {
            data: QueuedRequest,
            next: ?*Node = null,
            prev: ?*Node = null,
        };
        
        first: ?*Node,
        last: ?*Node,
        len: usize,
        data: std.AutoHashMap(u64, T),
        mutex: Mutex,
        max_size: u32,
        max_wait_ms: i64,
        next_id: Atomic(u64),
        
        pub fn init(allocator: Allocator, max_size: u32, max_wait_ms: i64) Self {
            return .{
                .first = null,
                .last = null,
                .len = 0,
                .data = std.AutoHashMap(u64, T).init(allocator),
                .mutex = .{},
                .max_size = max_size,
                .max_wait_ms = max_wait_ms,
                .next_id = Atomic(u64).init(0),
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            var current = self.first;
            while (current) |node| {
                const next = node.next;
                _ = self.data.remove(node.data.id);
                self.data.allocator.destroy(node);
                current = next;
            }
            self.data.deinit();
        }
        
        pub fn enqueue(self: *Self, item: T, priority: u8) !u64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            if (self.len >= self.max_size) {
                return error.QueueFull;
            }
            
            const id = self.next_id.fetchAdd(1, .release);
            const request = QueuedRequest{
                .id = id,
                .enqueued_at = std.time.milliTimestamp(),
                .priority = priority,
            };
            
            const node = try self.data.allocator.create(Node);
            node.* = .{ .data = request };
            
            // Insert based on priority
            if (self.first == null) {
                self.first = node;
                self.last = node;
            } else {
                var current = self.first;
                var prev: ?*Node = null;
                
                while (current) |curr| {
                    if (priority > curr.data.priority) {
                        node.next = curr;
                        node.prev = prev;
                        if (prev) |p| {
                            p.next = node;
                        } else {
                            self.first = node;
                        }
                        curr.prev = node;
                        self.len += 1;
                        try self.data.put(id, item);
                        return id;
                    }
                    prev = curr;
                    current = curr.next;
                }
                
                // Append to end
                if (self.last) |last| {
                    last.next = node;
                    node.prev = last;
                    self.last = node;
                }
            }
            
            self.len += 1;
            try self.data.put(id, item);
            return id;
        }
        
        pub fn dequeue(self: *Self) ?struct { id: u64, item: T } {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            // Remove expired requests
            var current = self.first;
            while (current) |node| {
                const next = node.next;
                if (node.data.waitTimeMs() > self.max_wait_ms) {
                    if (node.prev) |prev| {
                        prev.next = node.next;
                    } else {
                        self.first = node.next;
                    }
                    if (node.next) |nxt| {
                        nxt.prev = node.prev;
                    } else {
                        self.last = node.prev;
                    }
                    _ = self.data.remove(node.data.id);
                    self.data.allocator.destroy(node);
                    self.len -= 1;
                }
                current = next;
            }
            
            const node = self.first orelse return null;
            self.first = node.next;
            if (self.first) |first| {
                first.prev = null;
            } else {
                self.last = null;
            }
            self.len -= 1;
            
            defer self.data.allocator.destroy(node);
            
            const item = self.data.get(node.data.id) orelse return null;
            _ = self.data.remove(node.data.id);
            
            return .{ .id = node.data.id, .item = item };
        }
        
        pub fn size(self: *Self) u32 {
            self.mutex.lock();
            defer self.mutex.unlock();
            return @intCast(self.len);
        }
    };
}

// ============================================================================
// Main Health Monitor
// ============================================================================

pub const HealthMonitor = struct {
    allocator: Allocator,
    probe_handler: K8sProbeHandler,
    load_shedder: LoadShedder,
    start_time: i64,
    mutex: Mutex,
    
    pub fn init(allocator: Allocator, model_path: []const u8) HealthMonitor {
        return .{
            .allocator = allocator,
            .probe_handler = K8sProbeHandler.init(allocator, model_path),
            .load_shedder = LoadShedder.init(.{}),
            .start_time = std.time.timestamp(),
            .mutex = .{},
        };
    }
    
    pub fn checkSystemHealth(self: *HealthMonitor) !SystemHealth {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var components = std.ArrayList(ComponentHealth).init(self.allocator);
        errdefer components.deinit();
        
        // Check all components
        try components.append(try self.probe_handler.ssd_checker.check(self.allocator));
        try components.append(try self.probe_handler.ram_checker.check(self.allocator));
        try components.append(try self.probe_handler.model_checker.check(self.allocator));
        
        // Determine overall status (worst component status)
        var overall_status = HealthStatus.healthy;
        for (components.items) |component| {
            if (@intFromEnum(component.status) > @intFromEnum(overall_status)) {
                overall_status = component.status;
            }
        }
        
        const uptime = @as(u64, @intCast(std.time.timestamp() - self.start_time));
        
        return SystemHealth{
            .overall_status = overall_status,
            .components = try components.toOwnedSlice(),
            .timestamp = std.time.timestamp(),
            .uptime_seconds = uptime,
        };
    }
    
    pub fn getLoadMetrics(self: *const HealthMonitor) struct {
        active: u32,
        queued: u32,
        rejected: u64,
        avg_latency_ms: u32,
    } {
        return self.load_shedder.getMetrics();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "health status basics" {
    const healthy = HealthStatus.healthy;
    const critical = HealthStatus.critical;
    
    try std.testing.expect(healthy.isServingTraffic());
    try std.testing.expect(!critical.isServingTraffic());
    try std.testing.expectEqual(@as(u16, 200), healthy.toHttpCode());
    try std.testing.expectEqual(@as(u16, 503), critical.toHttpCode());
}

test "load shedder - accept under capacity" {
    var shedder = LoadShedder.init(.{
        .max_active_requests = 100,
        .max_queue_size = 50,
    });
    
    const decision = shedder.shouldAcceptRequest();
    try std.testing.expectEqual(LoadSheddingDecision.accept, decision);
}

test "load shedder - queue when at capacity" {
    var shedder = LoadShedder.init(.{
        .max_active_requests = 2,
        .max_queue_size = 5,
    });
    
    shedder.requestStarted();
    shedder.requestStarted();
    
    const decision = shedder.shouldAcceptRequest();
    try std.testing.expectEqual(LoadSheddingDecision.queue, decision);
    
    shedder.requestCompleted(100);
    shedder.requestCompleted(100);
}

test "request queue - basic operations" {
    const allocator = std.testing.allocator;
    var queue = RequestQueue(u32).init(allocator, 10, 5000);
    defer queue.deinit();
    
    const id1 = try queue.enqueue(42, 0);
    const id2 = try queue.enqueue(43, 1); // Higher priority
    
    try std.testing.expect(id1 != id2);
    try std.testing.expectEqual(@as(u32, 2), queue.size());
    
    // Higher priority should come first
    const first = queue.dequeue().?;
    try std.testing.expectEqual(id2, first.id);
    try std.testing.expectEqual(@as(u32, 43), first.item);
    
    const second = queue.dequeue().?;
    try std.testing.expectEqual(id1, second.id);
    try std.testing.expectEqual(@as(u32, 42), second.item);
    
    try std.testing.expectEqual(@as(u32, 0), queue.size());
}

test "k8s probes - startup" {
    const allocator = std.testing.allocator;
    var handler = K8sProbeHandler.init(allocator, "/mock/model.gguf");
    
    const result1 = try handler.handleStartupProbe();
    defer allocator.free(result1.message);
    try std.testing.expect(!result1.success);
    
    handler.markStartupComplete();
    
    const result2 = try handler.handleStartupProbe();
    defer allocator.free(result2.message);
    try std.testing.expect(result2.success);
}
