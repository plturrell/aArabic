// AI Core Health Check Endpoints
// Provides Kubernetes-compatible health, liveness, and readiness checks
// for SAP AI Core deployment

const std = @import("std");

/// Health status enum for Kubernetes health checks
pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,

    pub fn toString(self: HealthStatus) []const u8 {
        return switch (self) {
            .healthy => "healthy",
            .degraded => "degraded",
            .unhealthy => "unhealthy",
        };
    }

    pub fn httpStatusCode(self: HealthStatus) u16 {
        return switch (self) {
            .healthy => 200,
            .degraded => 200, // Still serving but with issues
            .unhealthy => 503,
        };
    }
};

/// Detailed health information
pub const HealthDetails = struct {
    model_loaded: bool = false,
    gpu_available: bool = false,
    memory_usage_mb: u64 = 0,
    inference_latency_ms: ?f32 = null,
    requests_served: u64 = 0,
    errors_count: u64 = 0,
};

/// Main health check response
pub const HealthCheck = struct {
    status: HealthStatus,
    message: []const u8,
    timestamp_ms: i64,
    details: ?HealthDetails,
    allocator: std.mem.Allocator,

    pub fn healthy(allocator: std.mem.Allocator, details: ?HealthDetails) HealthCheck {
        return .{
            .status = .healthy,
            .message = "Service is healthy",
            .timestamp_ms = std.time.milliTimestamp(),
            .details = details,
            .allocator = allocator,
        };
    }

    pub fn degraded(allocator: std.mem.Allocator, message: []const u8, details: ?HealthDetails) HealthCheck {
        return .{
            .status = .degraded,
            .message = message,
            .timestamp_ms = std.time.milliTimestamp(),
            .details = details,
            .allocator = allocator,
        };
    }

    pub fn unhealthy(allocator: std.mem.Allocator, message: []const u8, details: ?HealthDetails) HealthCheck {
        return .{
            .status = .unhealthy,
            .message = message,
            .timestamp_ms = std.time.milliTimestamp(),
            .details = details,
            .allocator = allocator,
        };
    }

    pub fn toJson(self: *const HealthCheck) ![]u8 {
        var buf = std.ArrayList(u8){};
        errdefer buf.deinit();

        try buf.appendSlice("{\"status\":\"");
        try buf.appendSlice(self.status.toString());
        try buf.appendSlice("\",\"message\":\"");
        try buf.appendSlice(self.message);
        try buf.appendSlice("\",\"timestamp_ms\":");
        try std.fmt.format(buf.writer(), "{d}", .{self.timestamp_ms});

        if (self.details) |d| {
            try buf.appendSlice(",\"details\":{\"model_loaded\":");
            try buf.appendSlice(if (d.model_loaded) "true" else "false");
            try buf.appendSlice(",\"gpu_available\":");
            try buf.appendSlice(if (d.gpu_available) "true" else "false");
            try std.fmt.format(buf.writer(), ",\"memory_usage_mb\":{d}", .{d.memory_usage_mb});
            if (d.inference_latency_ms) |lat| {
                try std.fmt.format(buf.writer(), ",\"inference_latency_ms\":{d:.2}", .{lat});
            }
            try std.fmt.format(buf.writer(), ",\"requests_served\":{d},\"errors_count\":{d}}}", .{ d.requests_served, d.errors_count });
        }

        try buf.appendSlice("}");
        return buf.toOwnedSlice();
    }
};

/// Simple liveness check (is the process running?)
pub const LivenessCheck = struct {
    ok: bool,
    timestamp_ms: i64,

    pub fn check() LivenessCheck {
        return .{ .ok = true, .timestamp_ms = std.time.milliTimestamp() };
    }

    pub fn toJson(self: *const LivenessCheck, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        return try std.fmt.allocPrint(allocator, "{{\"status\":\"ok\"}}", .{});
    }
};

/// Readiness check (can the service handle requests?)
pub const ReadinessCheck = struct {
    ready: bool,
    model_loaded: bool,
    gpu_available: bool,
    timestamp_ms: i64,

    pub fn check(model_loaded: bool, gpu_available: bool) ReadinessCheck {
        return .{
            .ready = model_loaded and gpu_available,
            .model_loaded = model_loaded,
            .gpu_available = gpu_available,
            .timestamp_ms = std.time.milliTimestamp(),
        };
    }

    pub fn toJson(self: *const ReadinessCheck, allocator: std.mem.Allocator) ![]u8 {
        const status = if (self.ready) "ready" else "not_ready";
        return try std.fmt.allocPrint(allocator, "{{\"status\":\"{s}\",\"model_loaded\":{},\"gpu_available\":{}}}", .{ status, self.model_loaded, self.gpu_available });
    }
};


// ============================================================================
// Tests
// ============================================================================

test "HealthStatus toString" {
    try std.testing.expectEqualStrings("healthy", HealthStatus.healthy.toString());
    try std.testing.expectEqualStrings("degraded", HealthStatus.degraded.toString());
    try std.testing.expectEqualStrings("unhealthy", HealthStatus.unhealthy.toString());
}

test "HealthCheck healthy" {
    const allocator = std.testing.allocator;
    const details = HealthDetails{ .model_loaded = true, .gpu_available = true };
    const check = HealthCheck.healthy(allocator, details);
    try std.testing.expectEqual(HealthStatus.healthy, check.status);
}

test "checkHealth model not loaded" {
    const allocator = std.testing.allocator;
    const check = checkHealth(allocator, false, true, null);
    try std.testing.expectEqual(HealthStatus.unhealthy, check.status);
}

test "checkHealth gpu unavailable" {
    const allocator = std.testing.allocator;
    const check = checkHealth(allocator, true, false, null);
    try std.testing.expectEqual(HealthStatus.degraded, check.status);
}

test "LivenessCheck" {
    const check = LivenessCheck.check();
    try std.testing.expect(check.ok);
}

test "ReadinessCheck ready" {
    const check = ReadinessCheck.check(true, true);
    try std.testing.expect(check.ready);
}

test "ReadinessCheck not ready" {
    const check = ReadinessCheck.check(true, false);
    try std.testing.expect(!check.ready);
}

