// OpenTelemetry Distributed Tracing for Production LLM Server
// Day 7: Request tracing with span instrumentation
//
// Features:
// - W3C Trace Context propagation
// - Span creation and management
// - Automatic parent-child span relationships
// - Cache operation tracing
// - Integration with structured logging
// - Jaeger exporter support

const std = @import("std");
const log = @import("structured_logging.zig");

/// Trace context following W3C Trace Context specification
pub const TraceContext = struct {
    trace_id: [16]u8,     // 128-bit trace ID
    span_id: [8]u8,       // 64-bit span ID
    trace_flags: u8,      // Sampling and other flags
    
    /// Generate a new trace ID
    pub fn generateTraceId() [16]u8 {
        var trace_id: [16]u8 = undefined;
        std.crypto.random.bytes(&trace_id);
        return trace_id;
    }
    
    /// Generate a new span ID
    pub fn generateSpanId() [8]u8 {
        var span_id: [8]u8 = undefined;
        std.crypto.random.bytes(&span_id);
        return span_id;
    }
    
    /// Convert trace ID to hex string
    pub fn traceIdToHex(trace_id: [16]u8, buf: []u8) ![]u8 {
        if (buf.len < 32) return error.BufferTooSmall;
        return std.fmt.bufPrint(buf, "{x:0>32}", .{std.fmt.fmtSliceHexLower(&trace_id)}) catch buf[0..32];
    }
    
    /// Convert span ID to hex string
    pub fn spanIdToHex(span_id: [8]u8, buf: []u8) ![]u8 {
        if (buf.len < 16) return error.BufferTooSmall;
        return std.fmt.bufPrint(buf, "{x:0>16}", .{std.fmt.fmtSliceHexLower(&span_id)}) catch buf[0..16];
    }
};

/// Span kind following OpenTelemetry semantic conventions
pub const SpanKind = enum {
    internal,      // Internal operation
    server,        // Server handling request
    client,        // Client making request
    producer,      // Message producer
    consumer,      // Message consumer
};

/// Span status
pub const SpanStatus = enum {
    unset,
    ok,
    error_status,
    
    pub fn toString(self: SpanStatus) []const u8 {
        return switch (self) {
            .unset => "UNSET",
            .ok => "OK",
            .error_status => "ERROR",
        };
    }
};

/// Span event for recording significant moments
pub const SpanEvent = struct {
    name: []const u8,
    timestamp: i64,
    attributes: std.StringHashMap([]const u8),
};

/// Span represents a single operation in a trace
pub const Span = struct {
    allocator: std.mem.Allocator,
    
    // Identification
    trace_context: TraceContext,
    parent_span_id: ?[8]u8,
    
    // Metadata
    name: []const u8,
    kind: SpanKind,
    start_time: i64,
    end_time: ?i64,
    status: SpanStatus,
    
    // Attributes and events
    attributes: std.StringHashMap([]const u8),
    events: std.ArrayList(SpanEvent),
    
    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        kind: SpanKind,
        parent: ?*const Span,
    ) !*Span {
        const self = try allocator.create(Span);
        errdefer allocator.destroy(self);
        
        const trace_context = if (parent) |p|
            TraceContext{
                .trace_id = p.trace_context.trace_id,
                .span_id = TraceContext.generateSpanId(),
                .trace_flags = p.trace_context.trace_flags,
            }
        else
            TraceContext{
                .trace_id = TraceContext.generateTraceId(),
                .span_id = TraceContext.generateSpanId(),
                .trace_flags = 1, // Sampled
            };
        
        self.* = Span{
            .allocator = allocator,
            .trace_context = trace_context,
            .parent_span_id = if (parent) |p| p.trace_context.span_id else null,
            .name = name,
            .kind = kind,
            .start_time = std.time.milliTimestamp(),
            .end_time = null,
            .status = .unset,
            .attributes = std.StringHashMap([]const u8).init(allocator),
            .events = std.ArrayList(SpanEvent).init(allocator),
        };
        
        return self;
    }
    
    pub fn deinit(self: *Span) void {
        self.attributes.deinit();
        for (self.events.items) |*event| {
            event.attributes.deinit();
        }
        self.events.deinit();
        self.allocator.destroy(self);
    }
    
    /// Set a span attribute
    pub fn setAttribute(self: *Span, key: []const u8, value: []const u8) !void {
        try self.attributes.put(key, value);
    }
    
    /// Add an event to the span
    pub fn addEvent(self: *Span, name: []const u8) !void {
        try self.events.append(.{
            .name = name,
            .timestamp = std.time.milliTimestamp(),
            .attributes = std.StringHashMap([]const u8).init(self.allocator),
        });
    }
    
    /// Set span status
    pub fn setStatus(self: *Span, status: SpanStatus) void {
        self.status = status;
    }
    
    /// End the span
    pub fn end(self: *Span) void {
        if (self.end_time == null) {
            self.end_time = std.time.milliTimestamp();
            
            // Update logging context with trace info
            var trace_id_buf: [32]u8 = undefined;
            var span_id_buf: [16]u8 = undefined;
            const trace_id_hex = TraceContext.traceIdToHex(self.trace_context.trace_id, &trace_id_buf) catch "unknown";
            const span_id_hex = TraceContext.spanIdToHex(self.trace_context.span_id, &span_id_buf) catch "unknown";
            
            log.setContext(.{
                .trace_id = trace_id_hex,
                .span_id = span_id_hex,
                .operation = self.name,
            });
        }
    }
    
    /// Get span duration in milliseconds
    pub fn getDuration(self: *const Span) i64 {
        if (self.end_time) |end_time| {
            return end_time - self.start_time;
        }
        return std.time.milliTimestamp() - self.start_time;
    }
    
    /// Export span to OTLP JSON format
    pub fn toJson(self: *const Span, allocator: std.mem.Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        errdefer buffer.deinit();
        
        const writer = buffer.writer();
        
        try writer.writeAll("{");
        
        // Trace and span IDs
        var trace_id_buf: [32]u8 = undefined;
        var span_id_buf: [16]u8 = undefined;
        const trace_id_hex = try TraceContext.traceIdToHex(self.trace_context.trace_id, &trace_id_buf);
        const span_id_hex = try TraceContext.spanIdToHex(self.trace_context.span_id, &span_id_buf);
        
        try writer.print("\"traceId\":\"{s}\"", .{trace_id_hex});
        try writer.print(",\"spanId\":\"{s}\"", .{span_id_hex});
        
        // Parent span ID if exists
        if (self.parent_span_id) |parent_id| {
            const parent_hex = try TraceContext.spanIdToHex(parent_id, &span_id_buf);
            try writer.print(",\"parentSpanId\":\"{s}\"", .{parent_hex});
        }
        
        // Span metadata
        try writer.print(",\"name\":\"{s}\"", .{self.name});
        try writer.print(",\"kind\":\"{s}\"", .{@tagName(self.kind)});
        try writer.print(",\"startTimeUnixNano\":{d}", .{self.start_time * 1000000});
        
        if (self.end_time) |end_time| {
            try writer.print(",\"endTimeUnixNano\":{d}", .{end_time * 1000000});
        }
        
        // Status
        try writer.print(",\"status\":{{\"code\":\"{s}\"}}", .{self.status.toString()});
        
        // Attributes
        try writer.writeAll(",\"attributes\":[");
        var attr_it = self.attributes.iterator();
        var first_attr = true;
        while (attr_it.next()) |entry| {
            if (!first_attr) try writer.writeAll(",");
            try writer.print("{{\"key\":\"{s}\",\"value\":{{\"stringValue\":\"{s}\"}}}}", .{
                entry.key_ptr.*,
                entry.value_ptr.*,
            });
            first_attr = false;
        }
        try writer.writeAll("]");
        
        // Events
        try writer.writeAll(",\"events\":[");
        for (self.events.items, 0..) |event, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{{\"timeUnixNano\":{d},\"name\":\"{s}\"}}", .{
                event.timestamp * 1000000,
                event.name,
            });
        }
        try writer.writeAll("]");
        
        try writer.writeAll("}");
        
        return buffer.toOwnedSlice();
    }
};

/// Tracer for creating and managing spans
pub const Tracer = struct {
    allocator: std.mem.Allocator,
    service_name: []const u8,
    active_spans: std.ArrayList(*Span),
    mutex: std.Thread.Mutex,
    
    pub fn init(allocator: std.mem.Allocator, service_name: []const u8) !*Tracer {
        const self = try allocator.create(Tracer);
        errdefer allocator.destroy(self);
        
        self.* = Tracer{
            .allocator = allocator,
            .service_name = service_name,
            .active_spans = std.ArrayList(*Span).init(allocator),
            .mutex = .{},
        };
        
        return self;
    }
    
    pub fn deinit(self: *Tracer) void {
        for (self.active_spans.items) |span| {
            span.deinit();
        }
        self.active_spans.deinit();
        self.allocator.destroy(self);
    }
    
    /// Start a new span
    pub fn startSpan(
        self: *Tracer,
        name: []const u8,
        kind: SpanKind,
        parent: ?*const Span,
    ) !*Span {
        const span = try Span.init(self.allocator, name, kind, parent);
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.active_spans.append(span);
        
        // Log span start
        var trace_id_buf: [32]u8 = undefined;
        var span_id_buf: [16]u8 = undefined;
        const trace_id_hex = TraceContext.traceIdToHex(span.trace_context.trace_id, &trace_id_buf) catch "unknown";
        const span_id_hex = TraceContext.spanIdToHex(span.trace_context.span_id, &span_id_buf) catch "unknown";
        
        log.info("Span started: name={s}, trace_id={s}, span_id={s}", .{
            name,
            trace_id_hex,
            span_id_hex,
        });
        
        return span;
    }
    
    /// End and remove span from active list
    pub fn endSpan(self: *Tracer, span: *Span) void {
        span.end();
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Remove from active spans
        for (self.active_spans.items, 0..) |s, i| {
            if (s == span) {
                _ = self.active_spans.swapRemove(i);
                break;
            }
        }
        
        // Log span completion
        log.info("Span ended: name={s}, duration_ms={d}, status={s}", .{
            span.name,
            span.getDuration(),
            span.status.toString(),
        });
        
        // Export to Jaeger (placeholder - would send via HTTP)
        const json = span.toJson(self.allocator) catch {
            log.err("Failed to export span to JSON", .{});
            return;
        };
        defer self.allocator.free(json);
        
        // In production, send to Jaeger collector
        // For now, just log at debug level
        log.debug("Span JSON: {s}", .{json});
    }
};

/// Global tracer instance
var global_tracer: ?*Tracer = null;
var global_tracer_mutex: std.Thread.Mutex = .{};

/// Initialize global tracer
pub fn initGlobalTracer(allocator: std.mem.Allocator, service_name: []const u8) !void {
    global_tracer_mutex.lock();
    defer global_tracer_mutex.unlock();
    
    if (global_tracer != null) {
        return error.TracerAlreadyInitialized;
    }
    
    global_tracer = try Tracer.init(allocator, service_name);
}

/// Deinitialize global tracer
pub fn deinitGlobalTracer() void {
    global_tracer_mutex.lock();
    defer global_tracer_mutex.unlock();
    
    if (global_tracer) |tracer| {
        tracer.deinit();
        global_tracer = null;
    }
}

/// Get global tracer instance
pub fn getGlobalTracer() ?*Tracer {
    global_tracer_mutex.lock();
    defer global_tracer_mutex.unlock();
    return global_tracer;
}

// ============================================================================
// Convenience Functions for Common Tracing Patterns
// ============================================================================

/// Start a server span (for handling incoming requests)
pub fn startServerSpan(name: []const u8) !*Span {
    if (getGlobalTracer()) |tracer| {
        return try tracer.startSpan(name, .server, null);
    }
    return error.TracerNotInitialized;
}

/// Start an internal span (child of current span)
pub fn startInternalSpan(name: []const u8, parent: *const Span) !*Span {
    if (getGlobalTracer()) |tracer| {
        return try tracer.startSpan(name, .internal, parent);
    }
    return error.TracerNotInitialized;
}

/// End a span
pub fn endSpan(span: *Span) void {
    if (getGlobalTracer()) |tracer| {
        tracer.endSpan(span);
        span.deinit();
    }
}

/// Trace a cache operation
pub fn traceCacheOp(
    operation: []const u8,
    parent: *const Span,
    hit: bool,
    layer: u32,
) !*Span {
    const span = try startInternalSpan(operation, parent);
    
    var layer_buf: [32]u8 = undefined;
    const layer_str = try std.fmt.bufPrint(&layer_buf, "{d}", .{layer});
    
    try span.setAttribute("cache.hit", if (hit) "true" else "false");
    try span.setAttribute("cache.layer", layer_str);
    try span.setAttribute("component", "kv_cache");
    
    return span;
}

/// Trace an eviction operation
pub fn traceEviction(
    policy: []const u8,
    parent: *const Span,
    tokens_evicted: u32,
) !*Span {
    const span = try startInternalSpan("cache_eviction", parent);
    
    var tokens_buf: [32]u8 = undefined;
    const tokens_str = try std.fmt.bufPrint(&tokens_buf, "{d}", .{tokens_evicted});
    
    try span.setAttribute("eviction.policy", policy);
    try span.setAttribute("eviction.tokens", tokens_str);
    try span.setAttribute("component", "kv_cache");
    
    return span;
}
