// CUDA Memory Management
// Provides high-level memory allocation, transfer, and management utilities
//
// Features:
// - Device memory allocation/deallocation
// - Pinned (page-locked) host memory
// - Memory transfer utilities (H2D, D2H, D2D)
// - Memory pool for efficient reuse
// - Allocation tracking and leak detection

const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const CudaContext = @import("cuda_context.zig").CudaContext;

// ============================================================================
// Device Memory Allocation
// ============================================================================

/// Device memory allocation wrapper
pub const DeviceMemory = struct {
    ptr: *anyopaque,
    size: usize,
    allocator: std.mem.Allocator,
    
    /// Allocate device memory
    pub fn alloc(allocator: std.mem.Allocator, size: usize) !DeviceMemory {
        var ptr: *anyopaque = undefined;
        try cuda.checkCudaError(
            cuda.cudaMalloc(@ptrCast(&ptr), size),
            "cudaMalloc"
        );
        
        return DeviceMemory{
            .ptr = ptr,
            .size = size,
            .allocator = allocator,
        };
    }
    
    /// Free device memory
    pub fn deinit(self: *DeviceMemory) void {
        cuda.checkCudaError(
            cuda.cudaFree(self.ptr),
            "cudaFree"
        ) catch |err| {
            std.debug.print("Warning: Failed to free device memory: {}\n", .{err});
        };
    }
    
    /// Zero out device memory
    pub fn zero(self: *DeviceMemory) !void {
        try cuda.checkCudaError(
            cuda.cudaMemset(self.ptr, 0, self.size),
            "cudaMemset"
        );
    }
    
    /// Copy from host to device
    pub fn copyFromHost(self: *DeviceMemory, host_data: []const u8) !void {
        if (host_data.len > self.size) {
            return error.BufferTooSmall;
        }
        
        try cuda.checkCudaError(
            cuda.cudaMemcpy(
                self.ptr,
                host_data.ptr,
                host_data.len,
                cuda.cudaMemcpyHostToDevice
            ),
            "cudaMemcpy H2D"
        );
    }
    
    /// Copy from device to host
    pub fn copyToHost(self: *DeviceMemory, host_buffer: []u8) !void {
        if (host_buffer.len < self.size) {
            return error.BufferTooSmall;
        }
        
        try cuda.checkCudaError(
            cuda.cudaMemcpy(
                host_buffer.ptr,
                self.ptr,
                self.size,
                cuda.cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy D2H"
        );
    }
    
    /// Copy from another device memory
    pub fn copyFromDevice(self: *DeviceMemory, src: *const DeviceMemory) !void {
        const copy_size = @min(self.size, src.size);
        
        try cuda.checkCudaError(
            cuda.cudaMemcpy(
                self.ptr,
                src.ptr,
                copy_size,
                cuda.cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpy D2D"
        );
    }
};

// ============================================================================
// Pinned Host Memory
// ============================================================================

/// Pinned (page-locked) host memory for faster transfers
pub const PinnedMemory = struct {
    ptr: [*]u8,
    size: usize,
    allocator: std.mem.Allocator,
    
    /// Allocate pinned host memory
    pub fn alloc(allocator: std.mem.Allocator, size: usize) !PinnedMemory {
        var ptr: *anyopaque = undefined;
        try cuda.checkCudaError(
            cuda.cudaMallocHost(@ptrCast(&ptr), size),
            "cudaMallocHost"
        );
        
        return PinnedMemory{
            .ptr = @ptrCast(ptr),
            .size = size,
            .allocator = allocator,
        };
    }
    
    /// Free pinned host memory
    pub fn deinit(self: *PinnedMemory) void {
        cuda.checkCudaError(
            cuda.cudaFreeHost(self.ptr),
            "cudaFreeHost"
        ) catch |err| {
            std.debug.print("Warning: Failed to free pinned memory: {}\n", .{err});
        };
    }
    
    /// Get as slice
    pub fn asSlice(self: *PinnedMemory) []u8 {
        return self.ptr[0..self.size];
    }
    
    /// Zero out memory
    pub fn zero(self: *PinnedMemory) void {
        @memset(self.asSlice(), 0);
    }
    
    /// Copy to device
    pub fn copyToDevice(self: *PinnedMemory, device_mem: *DeviceMemory) !void {
        try device_mem.copyFromHost(self.asSlice());
    }
    
    /// Copy from device
    pub fn copyFromDevice(self: *PinnedMemory, device_mem: *DeviceMemory) !void {
        try device_mem.copyToHost(self.asSlice());
    }
};

// ============================================================================
// Memory Pool
// ============================================================================

/// Memory pool for efficient allocation reuse
pub const MemoryPool = struct {
    allocator: std.mem.Allocator,
    free_blocks: std.ArrayList(DeviceMemory),
    allocated_blocks: std.ArrayList(DeviceMemory),
    total_allocated: usize,
    total_freed: usize,
    block_size: usize,
    
    pub fn init(allocator: std.mem.Allocator, block_size: usize) !MemoryPool {
        return MemoryPool{
            .allocator = allocator,
            .free_blocks = try std.ArrayList(DeviceMemory).initCapacity(allocator, 0),
            .allocated_blocks = try std.ArrayList(DeviceMemory).initCapacity(allocator, 0),
            .total_allocated = 0,
            .total_freed = 0,
            .block_size = block_size,
        };
    }
    
    pub fn deinit(self: *MemoryPool) void {
        // Free all blocks
        for (self.free_blocks.items) |*block| {
            block.deinit();
        }
        self.free_blocks.deinit(self.allocator);
        
        for (self.allocated_blocks.items) |*block| {
            block.deinit();
        }
        self.allocated_blocks.deinit(self.allocator);
    }
    
    /// Allocate from pool or create new block
    pub fn alloc(self: *MemoryPool) !DeviceMemory {
        // Try to reuse from free list
        if (self.free_blocks.items.len > 0) {
            const last_idx = self.free_blocks.items.len - 1;
            const block = self.free_blocks.items[last_idx];
            _ = self.free_blocks.swapRemove(last_idx);
            try self.allocated_blocks.append(self.allocator, block);
            return block;
        }
        
        // Allocate new block
        const block = try DeviceMemory.alloc(self.allocator, self.block_size);
        try self.allocated_blocks.append(self.allocator, block);
        self.total_allocated += 1;
        
        return block;
    }
    
    /// Return block to pool
    pub fn free(self: *MemoryPool, block: DeviceMemory) !void {
        // Find and remove from allocated list
        for (self.allocated_blocks.items, 0..) |item, i| {
            if (item.ptr == block.ptr) {
                _ = self.allocated_blocks.orderedRemove(i);
                try self.free_blocks.append(self.allocator, block);
                self.total_freed += 1;
                return;
            }
        }
        
        return error.BlockNotFound;
    }
    
    /// Get pool statistics
    pub fn getStats(self: *MemoryPool) struct {
        total_allocated: usize,
        total_freed: usize,
        currently_allocated: usize,
        free_blocks: usize,
        block_size: usize,
    } {
        return .{
            .total_allocated = self.total_allocated,
            .total_freed = self.total_freed,
            .currently_allocated = self.allocated_blocks.items.len,
            .free_blocks = self.free_blocks.items.len,
            .block_size = self.block_size,
        };
    }
};

// ============================================================================
// Memory Transfer Utilities
// ============================================================================

/// Copy data from host to device
pub fn copyHostToDevice(
    device_ptr: *anyopaque,
    host_data: []const u8,
) !void {
    try cuda.checkCudaError(
        cuda.cudaMemcpy(
            device_ptr,
            host_data.ptr,
            host_data.len,
            cuda.cudaMemcpyHostToDevice
        ),
        "copyHostToDevice"
    );
}

/// Copy data from device to host
pub fn copyDeviceToHost(
    host_buffer: []u8,
    device_ptr: *const anyopaque,
    size: usize,
) !void {
    if (host_buffer.len < size) {
        return error.BufferTooSmall;
    }
    
    try cuda.checkCudaError(
        cuda.cudaMemcpy(
            host_buffer.ptr,
            device_ptr,
            size,
            cuda.cudaMemcpyDeviceToHost
        ),
        "copyDeviceToHost"
    );
}

/// Copy data between device memories
pub fn copyDeviceToDevice(
    dst_ptr: *anyopaque,
    src_ptr: *const anyopaque,
    size: usize,
) !void {
    try cuda.checkCudaError(
        cuda.cudaMemcpy(
            dst_ptr,
            src_ptr,
            size,
            cuda.cudaMemcpyDeviceToDevice
        ),
        "copyDeviceToDevice"
    );
}

/// Async copy from host to device
pub fn copyHostToDeviceAsync(
    device_ptr: *anyopaque,
    host_data: []const u8,
    stream: ?*anyopaque,
) !void {
    try cuda.checkCudaError(
        cuda.cudaMemcpyAsync(
            device_ptr,
            host_data.ptr,
            host_data.len,
            cuda.cudaMemcpyHostToDevice,
            stream
        ),
        "copyHostToDeviceAsync"
    );
}

/// Async copy from device to host
pub fn copyDeviceToHostAsync(
    host_buffer: []u8,
    device_ptr: *const anyopaque,
    size: usize,
    stream: ?*anyopaque,
) !void {
    if (host_buffer.len < size) {
        return error.BufferTooSmall;
    }
    
    try cuda.checkCudaError(
        cuda.cudaMemcpyAsync(
            host_buffer.ptr,
            device_ptr,
            size,
            cuda.cudaMemcpyDeviceToHost,
            stream
        ),
        "copyDeviceToHostAsync"
    );
}

// ============================================================================
// Memory Utilities
// ============================================================================

/// Zero out device memory
pub fn zeroDeviceMemory(device_ptr: *anyopaque, size: usize) !void {
    try cuda.checkCudaError(
        cuda.cudaMemset(device_ptr, 0, size),
        "zeroDeviceMemory"
    );
}

/// Set device memory to a value
pub fn setDeviceMemory(device_ptr: *anyopaque, value: u8, size: usize) !void {
    try cuda.checkCudaError(
        cuda.cudaMemset(device_ptr, @intCast(value), size),
        "setDeviceMemory"
    );
}

// ============================================================================
// Allocation Tracker (for debugging)
// ============================================================================

pub const AllocationTracker = struct {
    allocator: std.mem.Allocator,
    allocations: std.AutoHashMap(*anyopaque, AllocationInfo),
    total_allocated: usize,
    total_freed: usize,
    peak_usage: usize,
    
    const AllocationInfo = struct {
        size: usize,
        timestamp: i64,
    };
    
    pub fn init(allocator: std.mem.Allocator) AllocationTracker {
        return AllocationTracker{
            .allocator = allocator,
            .allocations = std.AutoHashMap(*anyopaque, AllocationInfo).init(allocator),
            .total_allocated = 0,
            .total_freed = 0,
            .peak_usage = 0,
        };
    }
    
    pub fn deinit(self: *AllocationTracker) void {
        self.allocations.deinit();
    }
    
    pub fn trackAllocation(self: *AllocationTracker, ptr: *anyopaque, size: usize) !void {
        const timestamp = std.time.milliTimestamp();
        try self.allocations.put(ptr, .{
            .size = size,
            .timestamp = timestamp,
        });
        
        self.total_allocated += size;
        const current_usage = self.total_allocated - self.total_freed;
        if (current_usage > self.peak_usage) {
            self.peak_usage = current_usage;
        }
    }
    
    pub fn trackFree(self: *AllocationTracker, ptr: *anyopaque) !void {
        if (self.allocations.get(ptr)) |info| {
            self.total_freed += info.size;
            _ = self.allocations.remove(ptr);
        } else {
            return error.PointerNotTracked;
        }
    }
    
    pub fn getStats(self: *AllocationTracker) struct {
        total_allocated: usize,
        total_freed: usize,
        current_usage: usize,
        peak_usage: usize,
        active_allocations: usize,
    } {
        return .{
            .total_allocated = self.total_allocated,
            .total_freed = self.total_freed,
            .current_usage = self.total_allocated - self.total_freed,
            .peak_usage = self.peak_usage,
            .active_allocations = self.allocations.count(),
        };
    }
    
    pub fn printStats(self: *AllocationTracker) void {
        const stats = self.getStats();
        std.debug.print("\n📊 GPU Memory Allocation Stats:\n", .{});
        std.debug.print("   Total Allocated: {d} bytes ({d:.2} MB)\n", .{
            stats.total_allocated,
            @as(f32, @floatFromInt(stats.total_allocated)) / (1024.0 * 1024.0),
        });
        std.debug.print("   Total Freed: {d} bytes ({d:.2} MB)\n", .{
            stats.total_freed,
            @as(f32, @floatFromInt(stats.total_freed)) / (1024.0 * 1024.0),
        });
        std.debug.print("   Current Usage: {d} bytes ({d:.2} MB)\n", .{
            stats.current_usage,
            @as(f32, @floatFromInt(stats.current_usage)) / (1024.0 * 1024.0),
        });
        std.debug.print("   Peak Usage: {d} bytes ({d:.2} MB)\n", .{
            stats.peak_usage,
            @as(f32, @floatFromInt(stats.peak_usage)) / (1024.0 * 1024.0),
        });
        std.debug.print("   Active Allocations: {d}\n", .{stats.active_allocations});
    }
    
    pub fn checkLeaks(self: *AllocationTracker) bool {
        return self.allocations.count() > 0;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "cuda_memory: device allocation" {
    const allocator = std.testing.allocator;
    
    var mem = DeviceMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    try std.testing.expect(mem.size == 1024);
}

test "cuda_memory: pinned allocation" {
    const allocator = std.testing.allocator;
    
    var mem = PinnedMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    try std.testing.expect(mem.size == 1024);
}
