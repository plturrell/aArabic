// CUDA Manager - Unified GPU Management Interface
// Integrates all CUDA components into a single, easy-to-use interface
//
// Features:
// - Unified initialization and cleanup
// - Automatic device selection
// - Resource management
// - Performance monitoring
// - T4-specific optimizations

const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const nvidia_smi = @import("nvidia_smi.zig");
const CudaContext = @import("cuda_context.zig").CudaContext;
const cuda_memory = @import("cuda_memory.zig");
const cuda_streams = @import("cuda_streams.zig");

// Re-export common types for convenience
pub const DeviceMemory = cuda_memory.DeviceMemory;
pub const PinnedMemory = cuda_memory.PinnedMemory;
pub const MemoryPool = cuda_memory.MemoryPool;
pub const AllocationTracker = cuda_memory.AllocationTracker;
pub const CudaStream = cuda_streams.CudaStream;
pub const CudaEvent = cuda_streams.CudaEvent;
pub const StreamPool = cuda_streams.StreamPool;
pub const StreamTimer = cuda_streams.StreamTimer;

// ============================================================================
// CUDA Manager
// ============================================================================

/// Unified CUDA manager that integrates all GPU functionality
pub const CudaManager = struct {
    allocator: std.mem.Allocator,
    context: *CudaContext,
    stream_pool: ?StreamPool,
    memory_pool: ?MemoryPool,
    allocation_tracker: ?AllocationTracker,
    default_stream: ?CudaStream,
    
    /// Configuration options
    pub const Config = struct {
        device_id: i32 = 0,
        enable_stream_pool: bool = true,
        stream_pool_size: usize = 4,
        enable_memory_pool: bool = true,
        memory_pool_block_size: usize = 1024 * 1024, // 1MB default
        enable_allocation_tracking: bool = false,
        create_default_stream: bool = true,
    };
    
    /// Initialize CUDA manager with configuration
    pub fn init(allocator: std.mem.Allocator, config: Config) !*CudaManager {
        std.debug.print("\n🚀 Initializing CUDA Manager\n", .{});
        std.debug.print("   Configuration:\n", .{});
        std.debug.print("     Device ID: {d}\n", .{config.device_id});
        std.debug.print("     Stream Pool: {s} (size: {})\n", .{
            if (config.enable_stream_pool) "enabled" else "disabled",
            config.stream_pool_size,
        });
        std.debug.print("     Memory Pool: {s} (block: {} MB)\n", .{
            if (config.enable_memory_pool) "enabled" else "disabled",
            config.memory_pool_block_size / (1024 * 1024),
        });
        std.debug.print("     Allocation Tracking: {s}\n", .{
            if (config.enable_allocation_tracking) "enabled" else "disabled",
        });
        
        // Initialize CUDA context
        const context = try CudaContext.init(allocator, config.device_id);
        errdefer context.deinit();
        
        // Initialize stream pool if enabled
        var stream_pool: ?StreamPool = null;
        if (config.enable_stream_pool) {
            stream_pool = try StreamPool.init(allocator, config.stream_pool_size);
            std.debug.print("   ✓ Stream pool created\n", .{});
        }
        errdefer if (stream_pool) |*pool| pool.deinit();
        
        // Initialize memory pool if enabled
        var memory_pool: ?MemoryPool = null;
        if (config.enable_memory_pool) {
            memory_pool = try MemoryPool.init(allocator, config.memory_pool_block_size);
            std.debug.print("   ✓ Memory pool created\n", .{});
        }
        errdefer if (memory_pool) |*pool| pool.deinit();
        
        // Initialize allocation tracker if enabled
        var allocation_tracker: ?AllocationTracker = null;
        if (config.enable_allocation_tracking) {
            allocation_tracker = AllocationTracker.init(allocator);
            std.debug.print("   ✓ Allocation tracker created\n", .{});
        }
        errdefer if (allocation_tracker) |*tracker| tracker.deinit();
        
        // Create default stream if enabled
        var default_stream: ?CudaStream = null;
        if (config.create_default_stream) {
            default_stream = try CudaStream.init(allocator);
            std.debug.print("   ✓ Default stream created\n", .{});
        }
        errdefer if (default_stream) |*stream| stream.deinit();
        
        const self = try allocator.create(CudaManager);
        self.* = CudaManager{
            .allocator = allocator,
            .context = context,
            .stream_pool = stream_pool,
            .memory_pool = memory_pool,
            .allocation_tracker = allocation_tracker,
            .default_stream = default_stream,
        };
        
        std.debug.print("   ✅ CUDA Manager initialized successfully\n", .{});
        return self;
    }
    
    /// Initialize with default configuration
    pub fn initDefault(allocator: std.mem.Allocator) !*CudaManager {
        return try init(allocator, Config{});
    }
    
    /// Initialize with T4-optimized configuration
    pub fn initForT4(allocator: std.mem.Allocator) !*CudaManager {
        const config = Config{
            .device_id = 0,
            .enable_stream_pool = true,
            .stream_pool_size = 8, // T4 benefits from more streams
            .enable_memory_pool = true,
            .memory_pool_block_size = 4 * 1024 * 1024, // 4MB blocks for T4
            .enable_allocation_tracking = true,
            .create_default_stream = true,
        };
        
        return try init(allocator, config);
    }
    
    /// Clean up all resources
    pub fn deinit(self: *CudaManager) void {
        std.debug.print("\n🛑 Shutting down CUDA Manager\n", .{});
        
        if (self.default_stream) |*stream| {
            stream.deinit();
            std.debug.print("   ✓ Default stream destroyed\n", .{});
        }
        
        if (self.allocation_tracker) |*tracker| {
            if (tracker.checkLeaks()) {
                std.debug.print("   ⚠️  Memory leaks detected!\n", .{});
                tracker.printStats();
            }
            tracker.deinit();
            std.debug.print("   ✓ Allocation tracker destroyed\n", .{});
        }
        
        if (self.memory_pool) |*pool| {
            const stats = pool.getStats();
            std.debug.print("   📊 Memory Pool Stats:\n", .{});
            std.debug.print("      Total allocated: {}\n", .{stats.total_allocated});
            std.debug.print("      Total freed: {}\n", .{stats.total_freed});
            std.debug.print("      Currently allocated: {}\n", .{stats.currently_allocated});
            pool.deinit();
            std.debug.print("   ✓ Memory pool destroyed\n", .{});
        }
        
        if (self.stream_pool) |*pool| {
            const stats = pool.getStats();
            std.debug.print("   📊 Stream Pool Stats:\n", .{});
            std.debug.print("      Total streams: {}\n", .{stats.total});
            std.debug.print("      Available: {}\n", .{stats.available});
            std.debug.print("      In use: {}\n", .{stats.in_use});
            pool.deinit();
            std.debug.print("   ✓ Stream pool destroyed\n", .{});
        }
        
        self.context.deinit();
        std.debug.print("   ✓ CUDA context destroyed\n", .{});
        
        self.allocator.destroy(self);
        std.debug.print("   ✅ CUDA Manager shutdown complete\n", .{});
    }
    
    /// Get the CUDA context
    pub fn getContext(self: *CudaManager) *CudaContext {
        return self.context;
    }
    
    /// Get or create a stream
    pub fn acquireStream(self: *CudaManager) !*CudaStream {
        if (self.stream_pool) |*pool| {
            return try pool.acquire();
        } else if (self.default_stream) |*stream| {
            return stream;
        } else {
            return error.NoStreamAvailable;
        }
    }
    
    /// Release a stream back to the pool
    pub fn releaseStream(self: *CudaManager, stream: *CudaStream) !void {
        if (self.stream_pool) |*pool| {
            try pool.release(stream);
        }
        // If no pool, assume it's the default stream (no action needed)
    }
    
    /// Allocate device memory (from pool if available)
    pub fn allocDeviceMemory(self: *CudaManager, size: usize) !DeviceMemory {
        if (self.memory_pool) |*pool| {
            if (size <= pool.block_size) {
                return try pool.alloc();
            }
        }
        
        // Fallback to direct allocation
        const mem = try DeviceMemory.alloc(self.allocator, size);
        
        if (self.allocation_tracker) |*tracker| {
            try tracker.trackAllocation(mem.ptr, mem.size);
        }
        
        return mem;
    }
    
    /// Free device memory (return to pool if applicable)
    pub fn freeDeviceMemory(self: *CudaManager, mem: DeviceMemory) !void {
        if (self.memory_pool) |*pool| {
            if (mem.size == pool.block_size) {
                return try pool.free(mem);
            }
        }
        
        if (self.allocation_tracker) |*tracker| {
            try tracker.trackFree(mem.ptr);
        }
        
        var mutable_mem = mem;
        mutable_mem.deinit();
    }
    
    /// Allocate pinned host memory
    pub fn allocPinnedMemory(self: *CudaManager, size: usize) !PinnedMemory {
        const mem = try PinnedMemory.alloc(self.allocator, size);
        
        if (self.allocation_tracker) |*tracker| {
            try tracker.trackAllocation(@ptrCast(mem.ptr), mem.size);
        }
        
        return mem;
    }
    
    /// Free pinned host memory
    pub fn freePinnedMemory(self: *CudaManager, mem: PinnedMemory) !void {
        if (self.allocation_tracker) |*tracker| {
            try tracker.trackFree(@ptrCast(mem.ptr));
        }
        
        var mutable_mem = mem;
        mutable_mem.deinit();
    }
    
    /// Synchronize all operations
    pub fn synchronize(self: *CudaManager) !void {
        try self.context.synchronize();
    }
    
    /// Get memory information
    pub fn getMemoryInfo(self: *CudaManager) !struct {
        free_mb: u64,
        total_mb: u64,
        used_mb: u64,
    } {
        const info = try self.context.getMemoryInfo();
        return .{
            .free_mb = info.free_mb,
            .total_mb = info.total_mb,
            .used_mb = info.used_mb,
        };
    }
    
    /// Get allocation statistics (if tracking is enabled)
    pub fn getAllocationStats(self: *CudaManager) ?struct {
        total_allocated: usize,
        total_freed: usize,
        current_usage: usize,
        peak_usage: usize,
        active_allocations: usize,
    } {
        if (self.allocation_tracker) |*tracker| {
            const stats = tracker.getStats();
            return .{
                .total_allocated = stats.total_allocated,
                .total_freed = stats.total_freed,
                .current_usage = stats.current_usage,
                .peak_usage = stats.peak_usage,
                .active_allocations = stats.active_allocations,
            };
        }
        return null;
    }
    
    /// Print comprehensive statistics
    pub fn printStats(self: *CudaManager) void {
        std.debug.print("\n📊 CUDA Manager Statistics\n", .{});
        std.debug.print("   Device: {s}\n", .{self.context.properties.name});
        std.debug.print("   Compute: {d}.{d}\n", .{
            self.context.properties.compute_capability.major,
            self.context.properties.compute_capability.minor,
        });
        
        const mem_info = self.getMemoryInfo() catch return;
        std.debug.print("   GPU Memory:\n", .{});
        std.debug.print("     Total: {} MB\n", .{mem_info.total_mb});
        std.debug.print("     Used: {} MB\n", .{mem_info.used_mb});
        std.debug.print("     Free: {} MB\n", .{mem_info.free_mb});
        
        if (self.stream_pool) |*pool| {
            const stats = pool.getStats();
            std.debug.print("   Stream Pool:\n", .{});
            std.debug.print("     Total: {}\n", .{stats.total});
            std.debug.print("     Available: {}\n", .{stats.available});
            std.debug.print("     In use: {}\n", .{stats.in_use});
        }
        
        if (self.memory_pool) |*pool| {
            const stats = pool.getStats();
            std.debug.print("   Memory Pool:\n", .{});
            std.debug.print("     Block size: {} MB\n", .{stats.block_size / (1024 * 1024)});
            std.debug.print("     Total allocated: {}\n", .{stats.total_allocated});
            std.debug.print("     Currently allocated: {}\n", .{stats.currently_allocated});
            std.debug.print("     Free blocks: {}\n", .{stats.free_blocks});
        }
        
        if (self.getAllocationStats()) |stats| {
            std.debug.print("   Allocation Tracking:\n", .{});
            std.debug.print("     Total allocated: {} bytes\n", .{stats.total_allocated});
            std.debug.print("     Total freed: {} bytes\n", .{stats.total_freed});
            std.debug.print("     Current usage: {} bytes\n", .{stats.current_usage});
            std.debug.print("     Peak usage: {} bytes\n", .{stats.peak_usage});
            std.debug.print("     Active allocations: {}\n", .{stats.active_allocations});
        }
    }
    
    /// Check if this is a T4 GPU
    pub fn isT4(self: *CudaManager) bool {
        return self.context.properties.isT4();
    }
    
    /// Check if Tensor Cores are available
    pub fn hasTensorCores(self: *CudaManager) bool {
        return self.context.properties.hasTensorCores();
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick initialization for common use cases
pub fn initCUDA(allocator: std.mem.Allocator) !*CudaManager {
    return try CudaManager.initDefault(allocator);
}

/// Initialize with T4-specific optimizations
pub fn initCUDAForT4(allocator: std.mem.Allocator) !*CudaManager {
    return try CudaManager.initForT4(allocator);
}

/// Detect available GPUs using nvidia-smi
pub fn detectGPUs(allocator: std.mem.Allocator) ![]nvidia_smi.GPUInfo {
    return try nvidia_smi.detectGPUs(allocator);
}

/// Quick check for NVIDIA GPU availability
pub fn hasGPU() bool {
    return nvidia_smi.hasNvidiaGPU();
}

// ============================================================================
// Tests
// ============================================================================

test "cuda_manager: initialization" {
    const allocator = std.testing.allocator;
    
    var manager = CudaManager.initDefault(allocator) catch |err| {
        if (err == error.CudaError or err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer manager.deinit();
    
    try std.testing.expect(manager.context != undefined);
    std.debug.print("✓ Manager initialized\n", .{});
}

test "cuda_manager: T4 initialization" {
    const allocator = std.testing.allocator;
    
    var manager = CudaManager.initForT4(allocator) catch |err| {
        if (err == error.CudaError or err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer manager.deinit();
    
    try std.testing.expect(manager.stream_pool != null);
    try std.testing.expect(manager.memory_pool != null);
    try std.testing.expect(manager.allocation_tracker != null);
    
    std.debug.print("✓ T4-optimized manager initialized\n", .{});
}

test "cuda_manager: stream operations" {
    const allocator = std.testing.allocator;
    
    var manager = CudaManager.initDefault(allocator) catch |err| {
        if (err == error.CudaError or err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer manager.deinit();
    
    const stream = try manager.acquireStream();
    try manager.releaseStream(stream);
    
    std.debug.print("✓ Stream acquire/release working\n", .{});
}

test "cuda_manager: memory operations" {
    const allocator = std.testing.allocator;
    
    var manager = CudaManager.initDefault(allocator) catch |err| {
        if (err == error.CudaError or err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer manager.deinit();
    
    const mem = try manager.allocDeviceMemory(1024);
    try manager.freeDeviceMemory(mem);
    
    std.debug.print("✓ Memory allocation/deallocation working\n", .{});
}

test "cuda_manager: statistics" {
    const allocator = std.testing.allocator;
    
    var manager = CudaManager.initForT4(allocator) catch |err| {
        if (err == error.CudaError or err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer manager.deinit();
    
    manager.printStats();
    
    const mem_info = try manager.getMemoryInfo();
    try std.testing.expect(mem_info.total_mb > 0);
    
    std.debug.print("✓ Statistics working\n", .{});
}
