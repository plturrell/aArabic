// Backend Selection Tracer
// Runtime instrumentation for backend selection and operation routing
//
// This module provides diagnostic tracing to understand:
// - Which backend is being selected (CPU/CUDA/Metal)
// - Why that backend was chosen
// - Per-operation backend routing
// - Fallback reasons if GPU is unavailable

const std = @import("std");

pub const BackendType = enum {
    CPU,
    CUDA,
    Metal,
    Unknown,

    pub fn toString(self: BackendType) []const u8 {
        return switch (self) {
            .CPU => "CPU",
            .CUDA => "CUDA (GPU)",
            .Metal => "Metal (GPU)",
            .Unknown => "Unknown",
        };
    }
};

pub const SelectionReason = enum {
    ExplicitRequest,
    GPUAvailable,
    GPUUnavailable,
    PlatformDefault,
    Fallback,
    EnvironmentVariable,
    
    pub fn toString(self: SelectionReason) []const u8 {
        return switch (self) {
            .ExplicitRequest => "Explicitly requested by user",
            .GPUAvailable => "GPU detected and available",
            .GPUUnavailable => "No GPU available",
            .PlatformDefault => "Platform default",
            .Fallback => "Fallback from failed GPU initialization",
            .EnvironmentVariable => "Set via environment variable",
        };
    }
};

pub const OperationType = enum {
    MatrixMultiplication,
    RMSNormalization,
    Attention,
    Quantization,
    Dequantization,
    MemoryTransfer,
    Other,

    pub fn toString(self: OperationType) []const u8 {
        return switch (self) {
            .MatrixMultiplication => "Matrix Multiplication",
            .RMSNormalization => "RMS Normalization",
            .Attention => "Attention",
            .Quantization => "Quantization",
            .Dequantization => "Dequantization",
            .MemoryTransfer => "Memory Transfer",
            .Other => "Other",
        };
    }
};

pub const BackendEvent = struct {
    timestamp: i64,
    event_type: EventType,
    backend: BackendType,
    operation: ?OperationType,
    reason: ?SelectionReason,
    message: []const u8,
    
    pub const EventType = enum {
        Selection,
        Operation,
        Fallback,
        Error,
    };
};

pub const BackendTracer = struct {
    allocator: std.mem.Allocator,
    enabled: bool,
    verbose: bool,
    current_backend: BackendType,
    selection_reason: SelectionReason,
    events: std.ArrayList(BackendEvent),
    operation_counts: std.AutoHashMap(OperationType, usize),
    backend_operation_counts: std.AutoHashMap(BackendType, usize),

    pub fn init(allocator: std.mem.Allocator) !*BackendTracer {
        const self = try allocator.create(BackendTracer);
        self.* = BackendTracer{
            .allocator = allocator,
            .enabled = checkEnabled(),
            .verbose = checkVerbose(),
            .current_backend = .Unknown,
            .selection_reason = .PlatformDefault,
            .events = std.ArrayList(BackendEvent).init(allocator),
            .operation_counts = std.AutoHashMap(OperationType, usize).init(allocator),
            .backend_operation_counts = std.AutoHashMap(BackendType, usize).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *BackendTracer) void {
        self.events.deinit();
        self.operation_counts.deinit();
        self.backend_operation_counts.deinit();
        self.allocator.destroy(self);
    }

    fn checkEnabled() bool {
        if (std.posix.getenv("BACKEND_TRACE")) |val| {
            return std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        }
        return false;
    }

    fn checkVerbose() bool {
        if (std.posix.getenv("BACKEND_TRACE_VERBOSE")) |val| {
            return std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        }
        return false;
    }

    pub fn logBackendSelection(
        self: *BackendTracer,
        backend: BackendType,
        reason: SelectionReason,
        details: []const u8,
    ) !void {
        self.current_backend = backend;
        self.selection_reason = reason;

        if (!self.enabled) return;

        const event = BackendEvent{
            .timestamp = std.time.nanoTimestamp(),
            .event_type = .Selection,
            .backend = backend,
            .operation = null,
            .reason = reason,
            .message = details,
        };

        try self.events.append(event);

        // Always print backend selection
        std.debug.print("\n🔧 Backend Selected: {s}\n", .{backend.toString()});
        std.debug.print("   Reason: {s}\n", .{reason.toString()});
        if (details.len > 0) {
            std.debug.print("   Details: {s}\n", .{details});
        }
        std.debug.print("\n", .{});
    }

    pub fn logOperation(
        self: *BackendTracer,
        operation: OperationType,
        backend: BackendType,
    ) !void {
        // Update counters
        const op_count = self.operation_counts.get(operation) orelse 0;
        try self.operation_counts.put(operation, op_count + 1);

        const backend_count = self.backend_operation_counts.get(backend) orelse 0;
        try self.backend_operation_counts.put(backend, backend_count + 1);

        if (!self.enabled or !self.verbose) return;

        const event = BackendEvent{
            .timestamp = std.time.nanoTimestamp(),
            .event_type = .Operation,
            .backend = backend,
            .operation = operation,
            .reason = null,
            .message = "",
        };

        try self.events.append(event);

        std.debug.print("   [{s}] {s} → {s}\n", .{
            operation.toString(),
            backend.toString(),
            if (backend == .CPU) "❌ CPU fallback" else "✓ GPU accelerated",
        });
    }

    pub fn logFallback(
        self: *BackendTracer,
        from_backend: BackendType,
        to_backend: BackendType,
        reason: []const u8,
    ) !void {
        if (!self.enabled) return;

        const event = BackendEvent{
            .timestamp = std.time.nanoTimestamp(),
            .event_type = .Fallback,
            .backend = to_backend,
            .operation = null,
            .reason = .Fallback,
            .message = reason,
        };

        try self.events.append(event);

        std.debug.print("\n⚠️  Backend Fallback: {s} → {s}\n", .{
            from_backend.toString(),
            to_backend.toString(),
        });
        std.debug.print("   Reason: {s}\n\n", .{reason});
    }

    pub fn logError(
        self: *BackendTracer,
        backend: BackendType,
        error_msg: []const u8,
    ) !void {
        if (!self.enabled) return;

        const event = BackendEvent{
            .timestamp = std.time.nanoTimestamp(),
            .event_type = .Error,
            .backend = backend,
            .operation = null,
            .reason = null,
            .message = error_msg,
        };

        try self.events.append(event);

        std.debug.print("\n❌ Backend Error [{s}]: {s}\n\n", .{
            backend.toString(),
            error_msg,
        });
    }

    pub fn printSummary(self: *BackendTracer) void {
        if (!self.enabled) return;

        std.debug.print("\n" ++ "=" ** 60 ++ "\n", .{});
        std.debug.print("  BACKEND TRACER SUMMARY\n", .{});
        std.debug.print("=" ** 60 ++ "\n\n", .{});

        std.debug.print("Selected Backend: {s}\n", .{self.current_backend.toString()});
        std.debug.print("Selection Reason: {s}\n\n", .{self.selection_reason.toString()});

        // Print operation counts by type
        std.debug.print("Operations by Type:\n", .{});
        var op_iter = self.operation_counts.iterator();
        while (op_iter.next()) |entry| {
            std.debug.print("   {s}: {d}\n", .{
                entry.key_ptr.*.toString(),
                entry.value_ptr.*,
            });
        }

        // Print operation counts by backend
        std.debug.print("\nOperations by Backend:\n", .{});
        var backend_iter = self.backend_operation_counts.iterator();
        while (backend_iter.next()) |entry| {
            std.debug.print("   {s}: {d}\n", .{
                entry.key_ptr.*.toString(),
                entry.value_ptr.*,
            });
        }

        // Calculate GPU utilization
        const cpu_ops = self.backend_operation_counts.get(.CPU) orelse 0;
        const cuda_ops = self.backend_operation_counts.get(.CUDA) orelse 0;
        const metal_ops = self.backend_operation_counts.get(.Metal) orelse 0;
        const total_ops = cpu_ops + cuda_ops + metal_ops;

        if (total_ops > 0) {
            const gpu_ops = cuda_ops + metal_ops;
            const gpu_percent = (@as(f64, @floatFromInt(gpu_ops)) / @as(f64, @floatFromInt(total_ops))) * 100.0;
            
            std.debug.print("\nGPU Utilization: {d:.1}%\n", .{gpu_percent});
            
            if (gpu_percent < 10.0) {
                std.debug.print("⚠️  WARNING: Very low GPU utilization!\n", .{});
                std.debug.print("   Most operations are running on CPU.\n", .{});
            } else if (gpu_percent < 50.0) {
                std.debug.print("⚠️  Low GPU utilization - consider optimization.\n", .{});
            } else {
                std.debug.print("✓ Good GPU utilization.\n", .{});
            }
        }

        std.debug.print("\n" ++ "=" ** 60 ++ "\n\n", .{});
    }

    pub fn getCurrentBackend(self: *BackendTracer) BackendType {
        return self.current_backend;
    }

    pub fn isGPUActive(self: *BackendTracer) bool {
        return self.current_backend == .CUDA or self.current_backend == .Metal;
    }
};

// Global singleton tracer (optional convenience)
var global_tracer: ?*BackendTracer = null;
var global_tracer_mutex = std.Thread.Mutex{};

pub fn getGlobalTracer(allocator: std.mem.Allocator) !*BackendTracer {
    global_tracer_mutex.lock();
    defer global_tracer_mutex.unlock();

    if (global_tracer == null) {
        global_tracer = try BackendTracer.init(allocator);
    }
    return global_tracer.?;
}

pub fn deinitGlobalTracer() void {
    global_tracer_mutex.lock();
    defer global_tracer_mutex.unlock();

    if (global_tracer) |tracer| {
        tracer.deinit();
        global_tracer = null;
    }
}

// Convenience functions for common patterns
pub fn traceBackendSelection(
    allocator: std.mem.Allocator,
    backend: BackendType,
    reason: SelectionReason,
    details: []const u8,
) !void {
    const tracer = try getGlobalTracer(allocator);
    try tracer.logBackendSelection(backend, reason, details);
}

pub fn traceOperation(
    allocator: std.mem.Allocator,
    operation: OperationType,
    backend: BackendType,
) !void {
    const tracer = try getGlobalTracer(allocator);
    try tracer.logOperation(operation, backend);
}

pub fn traceFallback(
    allocator: std.mem.Allocator,
    from_backend: BackendType,
    to_backend: BackendType,
    reason: []const u8,
) !void {
    const tracer = try getGlobalTracer(allocator);
    try tracer.logFallback(from_backend, to_backend, reason);
}

pub fn printTraceSummary(allocator: std.mem.Allocator) !void {
    const tracer = try getGlobalTracer(allocator);
    tracer.printSummary();
}

test "BackendTracer: basic functionality" {
    const allocator = std.testing.allocator;
    
    const tracer = try BackendTracer.init(allocator);
    defer tracer.deinit();

    try tracer.logBackendSelection(.CUDA, .GPUAvailable, "T4 GPU detected");
    try tracer.logOperation(.MatrixMultiplication, .CUDA);
    try tracer.logOperation(.RMSNormalization, .CUDA);
    
    try std.testing.expectEqual(BackendType.CUDA, tracer.getCurrentBackend());
    try std.testing.expect(tracer.isGPUActive());
}

test "BackendTracer: fallback tracking" {
    const allocator = std.testing.allocator;
    
    const tracer = try BackendTracer.init(allocator);
    defer tracer.deinit();

    try tracer.logBackendSelection(.CUDA, .GPUAvailable, "Initial GPU selection");
    try tracer.logFallback(.CUDA, .CPU, "GPU memory exhausted");
    
    try std.testing.expectEqual(BackendType.CUDA, tracer.getCurrentBackend());
}
