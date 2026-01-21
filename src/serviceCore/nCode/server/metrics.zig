// Metrics tracking module for nCode server
// Provides Prometheus-compatible metrics endpoint

const std = @import("std");

pub const Metrics = struct {
    // Request metrics
    total_requests: std.atomic.Value(u64),
    requests_by_endpoint: std.StringHashMap(std.atomic.Value(u64)),
    requests_by_status: std.AutoHashMap(u16, std.atomic.Value(u64)),
    
    // Performance metrics
    total_request_duration_ms: std.atomic.Value(u64),
    
    // Cache metrics
    cache_hits: std.atomic.Value(u64),
    cache_misses: std.atomic.Value(u64),
    
    // Database operation metrics
    db_operations: std.atomic.Value(u64),
    db_errors: std.atomic.Value(u64),
    
    // Index metrics
    loaded_indices: std.atomic.Value(u64),
    active_symbols: std.atomic.Value(u64),
    
    // Server state
    start_time: i64,
    
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Metrics {
        return .{
            .total_requests = std.atomic.Value(u64).init(0),
            .requests_by_endpoint = std.StringHashMap(std.atomic.Value(u64)).init(allocator),
            .requests_by_status = std.AutoHashMap(u16, std.atomic.Value(u64)).init(allocator),
            .total_request_duration_ms = std.atomic.Value(u64).init(0),
            .cache_hits = std.atomic.Value(u64).init(0),
            .cache_misses = std.atomic.Value(u64).init(0),
            .db_operations = std.atomic.Value(u64).init(0),
            .db_errors = std.atomic.Value(u64).init(0),
            .loaded_indices = std.atomic.Value(u64).init(0),
            .active_symbols = std.atomic.Value(u64).init(0),
            .start_time = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Metrics) void {
        self.requests_by_endpoint.deinit();
        self.requests_by_status.deinit();
    }

    pub fn recordRequest(self: *Metrics, endpoint: []const u8, status: u16, duration_ms: u64) void {
        _ = self.total_requests.fetchAdd(1, .monotonic);
        _ = self.total_request_duration_ms.fetchAdd(duration_ms, .monotonic);
        
        // Record by endpoint (simplified - would need proper tracking in production)
        _ = endpoint;
        
        // Record by status code
        const entry = self.requests_by_status.getOrPut(status) catch return;
        if (!entry.found_existing) {
            entry.value_ptr.* = std.atomic.Value(u64).init(0);
        }
        _ = entry.value_ptr.fetchAdd(1, .monotonic);
    }

    pub fn recordCacheHit(self: *Metrics) void {
        _ = self.cache_hits.fetchAdd(1, .monotonic);
    }

    pub fn recordCacheMiss(self: *Metrics) void {
        _ = self.cache_misses.fetchAdd(1, .monotonic);
    }

    pub fn recordDbOperation(self: *Metrics) void {
        _ = self.db_operations.fetchAdd(1, .monotonic);
    }

    pub fn recordDbError(self: *Metrics) void {
        _ = self.db_errors.fetchAdd(1, .monotonic);
    }

    pub fn recordIndexLoaded(self: *Metrics, symbol_count: u64) void {
        _ = self.loaded_indices.fetchAdd(1, .monotonic);
        _ = self.active_symbols.fetchAdd(symbol_count, .monotonic);
    }

    pub fn getUptime(self: *Metrics) i64 {
        return std.time.timestamp() - self.start_time;
    }

    pub fn formatPrometheus(self: *Metrics, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        const writer = buffer.writer();

        // Help text and types
        try writer.writeAll("# HELP ncode_requests_total Total number of HTTP requests\n");
        try writer.writeAll("# TYPE ncode_requests_total counter\n");
        try writer.print("ncode_requests_total {d}\n\n", .{self.total_requests.load(.monotonic)});

        try writer.writeAll("# HELP ncode_request_duration_ms_total Total request duration in milliseconds\n");
        try writer.writeAll("# TYPE ncode_request_duration_ms_total counter\n");
        try writer.print("ncode_request_duration_ms_total {d}\n\n", .{self.total_request_duration_ms.load(.monotonic)});

        try writer.writeAll("# HELP ncode_cache_hits_total Total cache hits\n");
        try writer.writeAll("# TYPE ncode_cache_hits_total counter\n");
        try writer.print("ncode_cache_hits_total {d}\n\n", .{self.cache_hits.load(.monotonic)});

        try writer.writeAll("# HELP ncode_cache_misses_total Total cache misses\n");
        try writer.writeAll("# TYPE ncode_cache_misses_total counter\n");
        try writer.print("ncode_cache_misses_total {d}\n\n", .{self.cache_misses.load(.monotonic)});

        try writer.writeAll("# HELP ncode_db_operations_total Total database operations\n");
        try writer.writeAll("# TYPE ncode_db_operations_total counter\n");
        try writer.print("ncode_db_operations_total {d}\n\n", .{self.db_operations.load(.monotonic)});

        try writer.writeAll("# HELP ncode_db_errors_total Total database errors\n");
        try writer.writeAll("# TYPE ncode_db_errors_total counter\n");
        try writer.print("ncode_db_errors_total {d}\n\n", .{self.db_errors.load(.monotonic)});

        try writer.writeAll("# HELP ncode_loaded_indices Total number of loaded SCIP indices\n");
        try writer.writeAll("# TYPE ncode_loaded_indices counter\n");
        try writer.print("ncode_loaded_indices {d}\n\n", .{self.loaded_indices.load(.monotonic)});

        try writer.writeAll("# HELP ncode_active_symbols Total number of active symbols\n");
        try writer.writeAll("# TYPE ncode_active_symbols gauge\n");
        try writer.print("ncode_active_symbols {d}\n\n", .{self.active_symbols.load(.monotonic)});

        try writer.writeAll("# HELP ncode_uptime_seconds Server uptime in seconds\n");
        try writer.writeAll("# TYPE ncode_uptime_seconds gauge\n");
        try writer.print("ncode_uptime_seconds {d}\n\n", .{self.getUptime()});

        // Requests by status code
        try writer.writeAll("# HELP ncode_requests_by_status_total HTTP requests by status code\n");
        try writer.writeAll("# TYPE ncode_requests_by_status_total counter\n");
        var status_iter = self.requests_by_status.iterator();
        while (status_iter.next()) |entry| {
            try writer.print("ncode_requests_by_status_total{{status=\"{d}\"}} {d}\n", 
                .{ entry.key_ptr.*, entry.value_ptr.load(.monotonic) });
        }
        try writer.writeAll("\n");

        return buffer.toOwnedSlice();
    }

    pub fn formatJson(self: *Metrics, allocator: std.mem.Allocator) ![]const u8 {
        const total_reqs = self.total_requests.load(.monotonic);
        const total_duration = self.total_request_duration_ms.load(.monotonic);
        const avg_duration = if (total_reqs > 0) total_duration / total_reqs else 0;

        const cache_total = self.cache_hits.load(.monotonic) + self.cache_misses.load(.monotonic);
        const cache_hit_rate = if (cache_total > 0) 
            (@as(f64, @floatFromInt(self.cache_hits.load(.monotonic))) / @as(f64, @floatFromInt(cache_total))) * 100.0
        else 0.0;

        return std.fmt.allocPrint(allocator,
            \\{{
            \\  "requests": {{
            \\    "total": {d},
            \\    "average_duration_ms": {d}
            \\  }},
            \\  "cache": {{
            \\    "hits": {d},
            \\    "misses": {d},
            \\    "hit_rate_percent": {d:.2}
            \\  }},
            \\  "database": {{
            \\    "operations": {d},
            \\    "errors": {d}
            \\  }},
            \\  "index": {{
            \\    "loaded_indices": {d},
            \\    "active_symbols": {d}
            \\  }},
            \\  "server": {{
            \\    "uptime_seconds": {d}
            \\  }}
            \\}}
        , .{
            total_reqs,
            avg_duration,
            self.cache_hits.load(.monotonic),
            self.cache_misses.load(.monotonic),
            cache_hit_rate,
            self.db_operations.load(.monotonic),
            self.db_errors.load(.monotonic),
            self.loaded_indices.load(.monotonic),
            self.active_symbols.load(.monotonic),
            self.getUptime(),
        });
    }
};
