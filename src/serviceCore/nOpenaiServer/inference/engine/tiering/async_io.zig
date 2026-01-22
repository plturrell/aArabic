// Async I/O module for SSD tiering
// Provides non-blocking read/write with io_uring (Linux) or kqueue (macOS)

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Platform-agnostic async I/O interface
// ============================================================================

pub const AsyncIOConfig = struct {
    queue_depth: u32 = 256,       // Number of pending operations
    batch_size: u32 = 16,         // Submit batch size
    timeout_ms: u32 = 5000,       // Operation timeout
};

pub const IOOperation = struct {
    op_type: OpType,
    fd: std.fs.File.Handle,
    offset: u64,
    data: []u8,
    callback: ?*const fn(*IOResult) void = null,
    user_data: ?*anyopaque = null,
};

pub const OpType = enum {
    read,
    write,
    sync,
};

pub const IOResult = struct {
    operation: IOOperation,
    bytes_transferred: usize,
    err: ?anyerror = null,
    latency_ns: i64 = 0,
};

pub const AsyncIOStats = struct {
    reads: u64 = 0,
    writes: u64 = 0,
    bytes_read: u64 = 0,
    bytes_written: u64 = 0,
    total_latency_ns: u64 = 0,
    completed: u64 = 0,
    
    pub fn avgLatencyUs(self: AsyncIOStats) f64 {
        if (self.completed == 0) return 0;
        return @as(f64, @floatFromInt(self.total_latency_ns)) / 
               @as(f64, @floatFromInt(self.completed)) / 1000.0;
    }
};

// ============================================================================
// Async I/O Engine
// ============================================================================

pub const AsyncIOEngine = struct {
    allocator: std.mem.Allocator,
    config: AsyncIOConfig,
    stats: AsyncIOStats,
    
    // Pending operations queue
    pending: std.ArrayList(PendingOp),
    
    const PendingOp = struct {
        op: IOOperation,
        start_time: i128,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: AsyncIOConfig) !*AsyncIOEngine {
        const self = try allocator.create(AsyncIOEngine);
        
        self.* = AsyncIOEngine{
            .allocator = allocator,
            .config = config,
            .stats = .{},
            .pending = .{},
        };
        
        return self;
    }
    
    pub fn deinit(self: *AsyncIOEngine) void {
        self.pending.deinit(self.allocator);
        self.allocator.destroy(self);
    }
    
    /// Submit an async read operation
    pub fn submitRead(
        self: *AsyncIOEngine,
        file: std.fs.File,
        offset: u64,
        buffer: []u8,
        callback: ?*const fn(*IOResult) void,
    ) !void {
        try self.pending.append(.{
            .op = .{
                .op_type = .read,
                .fd = file.handle,
                .offset = offset,
                .data = buffer,
                .callback = callback,
            },
            .start_time = std.time.nanoTimestamp(),
        });
        
        // Auto-submit if batch full
        if (self.pending.items.len >= self.config.batch_size) {
            try self.submit();
        }
    }
    
    /// Submit an async write operation
    pub fn submitWrite(
        self: *AsyncIOEngine,
        file: std.fs.File,
        offset: u64,
        data: []const u8,
        callback: ?*const fn(*IOResult) void,
    ) !void {
        // Need mutable copy for the operation
        const data_copy = try self.allocator.dupe(u8, data);
        
        try self.pending.append(.{
            .op = .{
                .op_type = .write,
                .fd = file.handle,
                .offset = offset,
                .data = data_copy,
                .callback = callback,
            },
            .start_time = std.time.nanoTimestamp(),
        });
        
        if (self.pending.items.len >= self.config.batch_size) {
            try self.submit();
        }
    }
    
    /// Submit pending operations
    pub fn submit(self: *AsyncIOEngine) !void {
        // For now, execute synchronously
        // In production, use io_uring on Linux or kqueue on macOS
        for (self.pending.items) |*pending_op| {
            var result = IOResult{
                .operation = pending_op.op,
                .bytes_transferred = 0,
            };
            
            const file = std.fs.File{ .handle = pending_op.op.fd };
            
            switch (pending_op.op.op_type) {
                .read => {
                    result.bytes_transferred = file.pread(pending_op.op.data, pending_op.op.offset) catch |err| blk: {
                        result.err = err;
                        break :blk 0;
                    };
                    self.stats.reads += 1;
                    self.stats.bytes_read += result.bytes_transferred;
                },
                .write => {
                    const written = file.pwrite(pending_op.op.data, pending_op.op.offset) catch |err| blk: {
                        result.err = err;
                        break :blk 0;
                    };
                    result.bytes_transferred = written;
                    self.stats.writes += 1;
                    self.stats.bytes_written += result.bytes_transferred;

                    // Free the copy we made
                    self.allocator.free(pending_op.op.data);
                },
                .sync => {
                    file.sync() catch |err| {
                        result.err = err;
                    };
                },
            }
            
            const latency: i128 = std.time.nanoTimestamp() - pending_op.start_time;
            result.latency_ns = @intCast(@min(latency, std.math.maxInt(i64)));
            self.stats.total_latency_ns +|= @intCast(@min(latency, std.math.maxInt(u64)));
            self.stats.completed += 1;
            
            if (pending_op.op.callback) |cb| {
                cb(&result);
            }
        }
        
        self.pending.clearRetainingCapacity();
    }
    
    /// Wait for all pending operations
    pub fn waitAll(self: *AsyncIOEngine) !void {
        try self.submit();
    }
    
    /// Print stats
    pub fn printStats(self: *AsyncIOEngine) void {
        std.debug.print("\n⚡ Async I/O Stats\n", .{});
        std.debug.print("   Reads: {d}, Writes: {d}\n", .{self.stats.reads, self.stats.writes});
        std.debug.print("   Bytes: {d:.1} MB read, {d:.1} MB written\n", .{
            @as(f64, @floatFromInt(self.stats.bytes_read)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.stats.bytes_written)) / (1024.0 * 1024.0),
        });
        std.debug.print("   Avg latency: {d:.1} µs\n", .{self.stats.avgLatencyUs()});
    }
};

// ============================================================================
// Prefetch helper for model loading
// ============================================================================

pub fn prefetchFile(file: std.fs.File, offset: u64, length: u64) void {
    // Use madvise/posix_fadvise to hint kernel
    const ptr = std.posix.mmap(
        null,
        length,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        offset,
    ) catch return;
    
    std.posix.madvise(ptr.ptr, length, .WILLNEED) catch {};
    std.posix.munmap(ptr);
}

