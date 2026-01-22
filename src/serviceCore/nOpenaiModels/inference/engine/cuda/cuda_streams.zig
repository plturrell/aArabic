// CUDA Stream Management
// Provides high-level stream creation, management, and synchronization
//
// Features:
// - CUDA stream creation and destruction
// - Stream synchronization
// - Multi-stream parallelism support
// - Stream priority management
// - Event-based timing and synchronization

const std = @import("std");
const cuda = @import("cuda_bindings");
const CudaContext = @import("cuda_context").CudaContext;

// ============================================================================
// CUDA Stream
// ============================================================================

/// CUDA stream wrapper for async operations
pub const CudaStream = struct {
    handle: *anyopaque,
    allocator: std.mem.Allocator,
    priority: i32,
    
    /// Create a new CUDA stream
    pub fn init(allocator: std.mem.Allocator) !CudaStream {
        var handle: *anyopaque = undefined;
        try cuda.checkCudaError(
            cuda.cudaStreamCreate(@ptrCast(&handle)),
            "cudaStreamCreate"
        );
        
        return CudaStream{
            .handle = handle,
            .allocator = allocator,
            .priority = 0,
        };
    }
    
    /// Create a stream with specific priority
    pub fn initWithPriority(allocator: std.mem.Allocator, priority: i32) !CudaStream {
        var handle: *anyopaque = undefined;
        
        // Create stream with priority if supported
        const result = cuda.cudaStreamCreateWithPriority(
            @ptrCast(&handle),
            cuda.cudaStreamDefault,
            priority
        );
        
        if (result == cuda.cudaSuccess) {
            return CudaStream{
                .handle = handle,
                .allocator = allocator,
                .priority = priority,
            };
        }
        
        // Fall back to regular stream if priority not supported
        try cuda.checkCudaError(
            cuda.cudaStreamCreate(@ptrCast(&handle)),
            "cudaStreamCreate (fallback)"
        );
        
        return CudaStream{
            .handle = handle,
            .allocator = allocator,
            .priority = 0,
        };
    }
    
    /// Destroy the stream
    pub fn deinit(self: *CudaStream) void {
        cuda.checkCudaError(
            cuda.cudaStreamDestroy(self.handle),
            "cudaStreamDestroy"
        ) catch |err| {
            std.debug.print("Warning: Failed to destroy stream: {}\n", .{err});
        };
    }
    
    /// Wait for all operations in this stream to complete
    pub fn synchronize(self: *CudaStream) !void {
        try cuda.checkCudaError(
            cuda.cudaStreamSynchronize(self.handle),
            "cudaStreamSynchronize"
        );
    }
    
    /// Check if stream has completed all operations
    pub fn isComplete(self: *CudaStream) !bool {
        const result = cuda.cudaStreamQuery(self.handle);
        if (result == cuda.cudaSuccess) {
            return true;
        } else if (result == cuda.cudaErrorNotReady) {
            return false;
        } else {
            try cuda.checkCudaError(result, "cudaStreamQuery");
            return false;
        }
    }
    
    /// Get the raw stream handle (for use with CUDA APIs)
    pub fn getHandle(self: *CudaStream) *anyopaque {
        return self.handle;
    }
};

// ============================================================================
// CUDA Event
// ============================================================================

/// CUDA event for timing and synchronization
pub const CudaEvent = struct {
    handle: *anyopaque,
    allocator: std.mem.Allocator,
    
    /// Create a new CUDA event
    pub fn init(allocator: std.mem.Allocator) !CudaEvent {
        var handle: *anyopaque = undefined;
        try cuda.checkCudaError(
            cuda.cudaEventCreate(@ptrCast(&handle)),
            "cudaEventCreate"
        );
        
        return CudaEvent{
            .handle = handle,
            .allocator = allocator,
        };
    }
    
    /// Create an event with specific flags
    pub fn initWithFlags(allocator: std.mem.Allocator, flags: c_uint) !CudaEvent {
        var handle: *anyopaque = undefined;
        try cuda.checkCudaError(
            cuda.cudaEventCreateWithFlags(@ptrCast(&handle), flags),
            "cudaEventCreateWithFlags"
        );
        
        return CudaEvent{
            .handle = handle,
            .allocator = allocator,
        };
    }
    
    /// Destroy the event
    pub fn deinit(self: *CudaEvent) void {
        cuda.checkCudaError(
            cuda.cudaEventDestroy(self.handle),
            "cudaEventDestroy"
        ) catch |err| {
            std.debug.print("Warning: Failed to destroy event: {}\n", .{err});
        };
    }
    
    /// Record the event in the specified stream
    pub fn record(self: *CudaEvent, stream: *CudaStream) !void {
        try cuda.checkCudaError(
            cuda.cudaEventRecord(self.handle, stream.handle),
            "cudaEventRecord"
        );
    }
    
    /// Wait for the event to complete
    pub fn synchronize(self: *CudaEvent) !void {
        try cuda.checkCudaError(
            cuda.cudaEventSynchronize(self.handle),
            "cudaEventSynchronize"
        );
    }
    
    /// Check if event has completed
    pub fn isComplete(self: *CudaEvent) !bool {
        const result = cuda.cudaEventQuery(self.handle);
        if (result == cuda.cudaSuccess) {
            return true;
        } else if (result == cuda.cudaErrorNotReady) {
            return false;
        } else {
            try cuda.checkCudaError(result, "cudaEventQuery");
            return false;
        }
    }
    
    /// Calculate elapsed time between two events (in milliseconds)
    pub fn elapsedTime(start_event: *CudaEvent, end_event: *CudaEvent) !f32 {
        var ms: f32 = 0;
        try cuda.checkCudaError(
            cuda.cudaEventElapsedTime(&ms, start_event.handle, end_event.handle),
            "cudaEventElapsedTime"
        );
        return ms;
    }
};

// ============================================================================
// Stream Pool
// ============================================================================

/// Pool of CUDA streams for efficient reuse
pub const StreamPool = struct {
    allocator: std.mem.Allocator,
    streams: std.ArrayList(CudaStream),
    available: std.ArrayList(usize),
    pool_size: usize,
    
    /// Create a stream pool with specified size
    pub fn init(allocator: std.mem.Allocator, pool_size: usize) !StreamPool {
        var streams = try std.ArrayList(CudaStream).initCapacity(allocator, pool_size);
        var available = try std.ArrayList(usize).initCapacity(allocator, pool_size);
        
        // Pre-allocate all streams
        var i: usize = 0;
        while (i < pool_size) : (i += 1) {
            const stream = try CudaStream.init(allocator);
            try streams.append(stream);
            try available.append(i);
        }
        
        return StreamPool{
            .allocator = allocator,
            .streams = streams,
            .available = available,
            .pool_size = pool_size,
        };
    }
    
    /// Destroy the stream pool
    pub fn deinit(self: *StreamPool) void {
        for (self.streams.items) |*stream| {
            stream.deinit();
        }
        self.streams.deinit();
        self.available.deinit();
    }
    
    /// Acquire a stream from the pool
    pub fn acquire(self: *StreamPool) !*CudaStream {
        if (self.available.items.len == 0) {
            return error.NoStreamsAvailable;
        }
        
        const last_idx = self.available.items.len - 1;
        const idx = self.available.items[last_idx];
        _ = self.available.swapRemove(last_idx);
        return &self.streams.items[idx];
    }
    
    /// Release a stream back to the pool
    pub fn release(self: *StreamPool, stream: *CudaStream) !void {
        // Find the stream index
        for (self.streams.items, 0..) |*s, i| {
            if (@intFromPtr(s.handle) == @intFromPtr(stream.handle)) {
                // Synchronize before releasing
                try stream.synchronize();
                try self.available.append(i);
                return;
            }
        }
        return error.StreamNotInPool;
    }
    
    /// Get pool statistics
    pub fn getStats(self: *StreamPool) struct {
        total: usize,
        available: usize,
        in_use: usize,
    } {
        return .{
            .total = self.pool_size,
            .available = self.available.items.len,
            .in_use = self.pool_size - self.available.items.len,
        };
    }
};

// ============================================================================
// Stream Synchronization Utilities
// ============================================================================

/// Wait for multiple streams to complete
pub fn synchronizeStreams(streams: []CudaStream) !void {
    for (streams) |*stream| {
        try stream.synchronize();
    }
}

/// Check if all streams have completed
pub fn allStreamsComplete(streams: []CudaStream) !bool {
    for (streams) |*stream| {
        if (!try stream.isComplete()) {
            return false;
        }
    }
    return true;
}

/// Make a stream wait for an event
pub fn streamWaitEvent(stream: *CudaStream, event: *CudaEvent) !void {
    try cuda.checkCudaError(
        cuda.cudaStreamWaitEvent(stream.handle, event.handle, 0),
        "cudaStreamWaitEvent"
    );
}

// ============================================================================
// Performance Measurement
// ============================================================================

/// Measure execution time of operations in a stream
pub const StreamTimer = struct {
    start_event: CudaEvent,
    end_event: CudaEvent,
    
    pub fn init(allocator: std.mem.Allocator) !StreamTimer {
        return StreamTimer{
            .start_event = try CudaEvent.init(allocator),
            .end_event = try CudaEvent.init(allocator),
        };
    }
    
    pub fn deinit(self: *StreamTimer) void {
        self.start_event.deinit();
        self.end_event.deinit();
    }
    
    /// Start timing
    pub fn start(self: *StreamTimer, stream: *CudaStream) !void {
        try self.start_event.record(stream);
    }
    
    /// Stop timing
    pub fn stop(self: *StreamTimer, stream: *CudaStream) !void {
        try self.end_event.record(stream);
    }
    
    /// Get elapsed time in milliseconds
    pub fn getElapsedTime(self: *StreamTimer) !f32 {
        try self.end_event.synchronize();
        return try CudaEvent.elapsedTime(&self.start_event, &self.end_event);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "cuda_streams: stream creation" {
    const allocator = std.testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    try std.testing.expect(@intFromPtr(stream.handle) != 0);
}

test "cuda_streams: stream synchronization" {
    const allocator = std.testing.allocator;
    
    var stream = CudaStream.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer stream.deinit();
    
    try stream.synchronize();
}

test "cuda_streams: event creation" {
    const allocator = std.testing.allocator;
    
    var event = CudaEvent.init(allocator) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer event.deinit();
    
    try std.testing.expect(@intFromPtr(event.handle) != 0);
}

test "cuda_streams: stream pool" {
    const allocator = std.testing.allocator;
    
    var pool = StreamPool.init(allocator, 4) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer pool.deinit();
    
    const stats = pool.getStats();
    try std.testing.expect(stats.total == 4);
    try std.testing.expect(stats.available == 4);
}
