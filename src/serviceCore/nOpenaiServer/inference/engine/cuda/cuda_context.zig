// CUDA Context Management
// Handles CUDA device initialization, properties, and resource management
//
// This module provides high-level abstractions for CUDA device operations
// including device selection, property querying, and memory monitoring.

const std = @import("std");
const cuda = @import("cuda_bindings");

// ============================================================================
// Device Properties
// ============================================================================

pub const DeviceProperties = struct {
    name: []const u8,
    compute_capability: struct { 
        major: i32, 
        minor: i32 
    },
    total_memory_gb: f32,
    multiprocessor_count: i32,
    max_threads_per_block: i32,
    max_threads_per_sm: i32,
    warp_size: i32,
    memory_clock_khz: i32,
    memory_bus_width: i32,
    l2_cache_size: i32,
    shared_mem_per_block: usize,
    shared_mem_per_sm: usize,
    registers_per_block: i32,
    registers_per_sm: i32,
    
    pub fn isT4(self: DeviceProperties) bool {
        // T4 has compute capability 7.5
        if (self.compute_capability.major == 7 and 
            self.compute_capability.minor == 5) {
            return true;
        }
        
        // Also check name
        return std.mem.indexOf(u8, self.name, "Tesla T4") != null or
               std.mem.indexOf(u8, self.name, "T4") != null;
    }
    
    pub fn hasTensorCores(self: DeviceProperties) bool {
        // Tensor Cores introduced in Volta (7.0)
        return self.compute_capability.major >= 7;
    }
    
    pub fn getArchitectureName(self: DeviceProperties) []const u8 {
        return switch (self.compute_capability.major) {
            7 => switch (self.compute_capability.minor) {
                0, 2 => "Volta",
                5 => "Turing (T4)",
                else => "Volta/Turing",
            },
            8 => switch (self.compute_capability.minor) {
                0 => "Ampere (A100)",
                6 => "Ampere (A40/A10)",
                9 => "Ada Lovelace (L4/L40)",
                else => "Ampere/Ada",
            },
            9 => switch (self.compute_capability.minor) {
                0 => "Hopper (H100)",
                else => "Hopper",
            },
            else => "Unknown",
        };
    }
    
    /// Get recommended batch size based on GPU capabilities
    pub fn getRecommendedBatchSize(self: DeviceProperties) u32 {
        if (self.isT4()) {
            return 8; // T4 optimized batch size
        }
        
        // Scale based on memory
        const mem_gb = @as(u32, @intFromFloat(self.total_memory_gb));
        return if (mem_gb >= 32) 16 else if (mem_gb >= 16) 8 else 4;
    }
    
    /// Get recommended KV cache size in tokens
    pub fn getRecommendedKVCacheTokens(self: DeviceProperties) u32 {
        if (self.isT4()) {
            return 2048; // T4 optimized
        }
        
        const mem_gb = @as(u32, @intFromFloat(self.total_memory_gb));
        return if (mem_gb >= 32) 4096 else if (mem_gb >= 16) 2048 else 1024;
    }
};

// ============================================================================
// CUDA Context
// ============================================================================

pub const CudaContext = struct {
    allocator: std.mem.Allocator,
    device_id: i32,
    properties: DeviceProperties,
    initialized: bool = false,
    
    /// Initialize CUDA context for specified device
    pub fn init(allocator: std.mem.Allocator, device_id: i32) !*CudaContext {
        std.debug.print("\n🎮 Initializing CUDA Context\n", .{});
        std.debug.print("   Target Device ID: {d}\n", .{device_id});
        
        // Check device count
        var device_count: c_int = 0;
        try cuda.checkCudaError(cuda.cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        
        if (device_count == 0) {
            std.debug.print("   ❌ No CUDA devices found\n", .{});
            return error.NoGPUFound;
        }
        
        std.debug.print("   Available devices: {d}\n", .{device_count});
        
        if (device_id >= device_count) {
            std.debug.print("   ❌ Invalid device ID: {d} (max: {d})\n", .{ device_id, device_count - 1 });
            return error.InvalidDeviceId;
        }
        
        // Set active device
        try cuda.checkCudaError(cuda.cudaSetDevice(device_id), "cudaSetDevice");
        std.debug.print("   ✅ Set active device to {d}\n", .{device_id});
        
        // Get device properties
        var props: cuda.CudaDeviceProp = undefined;
        try cuda.checkCudaError(
            cuda.cudaGetDeviceProperties(&props, device_id), 
            "cudaGetDeviceProperties"
        );
        
        // Extract device name (null-terminated string)
        const name_end = std.mem.indexOfScalar(u8, &props.name, 0) orelse props.name.len;
        const name = try allocator.dupe(u8, props.name[0..name_end]);
        
        const device_props = DeviceProperties{
            .name = name,
            .compute_capability = .{
                .major = props.major,
                .minor = props.minor,
            },
            .total_memory_gb = @as(f32, @floatFromInt(props.totalGlobalMem)) / (1024.0 * 1024.0 * 1024.0),
            .multiprocessor_count = props.multiProcessorCount,
            .max_threads_per_block = props.maxThreadsPerBlock,
            .max_threads_per_sm = props.maxThreadsPerMultiProcessor,
            .warp_size = props.warpSize,
            .memory_clock_khz = props.memoryClockRate,
            .memory_bus_width = props.memoryBusWidth,
            .l2_cache_size = props.l2CacheSize,
            .shared_mem_per_block = props.sharedMemPerBlock,
            .shared_mem_per_sm = props.sharedMemPerMultiprocessor,
            .registers_per_block = props.regsPerBlock,
            .registers_per_sm = props.regsPerMultiprocessor,
        };
        
        // Print device information
        std.debug.print("\n   📊 Device Properties:\n", .{});
        std.debug.print("      Name: {s}\n", .{device_props.name});
        std.debug.print("      Architecture: {s}\n", .{device_props.getArchitectureName()});
        std.debug.print("      Compute Capability: {d}.{d}\n", .{
            device_props.compute_capability.major,
            device_props.compute_capability.minor,
        });
        std.debug.print("      Total Memory: {d:.2} GB\n", .{device_props.total_memory_gb});
        std.debug.print("      Multiprocessors: {d}\n", .{device_props.multiprocessor_count});
        std.debug.print("      Max Threads/Block: {d}\n", .{device_props.max_threads_per_block});
        std.debug.print("      Max Threads/SM: {d}\n", .{device_props.max_threads_per_sm});
        std.debug.print("      Warp Size: {d}\n", .{device_props.warp_size});
        std.debug.print("      Memory Clock: {d:.1} MHz\n", .{
            @as(f32, @floatFromInt(device_props.memory_clock_khz)) / 1000.0
        });
        std.debug.print("      Memory Bus Width: {d}-bit\n", .{device_props.memory_bus_width});
        std.debug.print("      L2 Cache: {d} KB\n", .{@divTrunc(device_props.l2_cache_size, 1024)});
        std.debug.print("      Shared Mem/Block: {d} KB\n", .{@divTrunc(device_props.shared_mem_per_block, 1024)});
        std.debug.print("      Tensor Cores: {s}\n", .{
            if (device_props.hasTensorCores()) "✅ Yes" else "❌ No"
        });
        
        if (device_props.isT4()) {
            std.debug.print("\n   ⚡ Tesla T4 Detected - Optimizations Enabled\n", .{});
            std.debug.print("      Recommended Batch Size: {d}\n", .{device_props.getRecommendedBatchSize()});
            std.debug.print("      Recommended KV Tokens: {d}\n", .{device_props.getRecommendedKVCacheTokens()});
        }
        
        const self = try allocator.create(CudaContext);
        self.* = CudaContext{
            .allocator = allocator,
            .device_id = device_id,
            .properties = device_props,
            .initialized = true,
        };
        
        std.debug.print("\n   ✅ CUDA Context Initialized Successfully\n\n", .{});
        
        return self;
    }
    
    pub fn deinit(self: *CudaContext) void {
        if (self.initialized) {
            self.allocator.free(self.properties.name);
            // Note: We don't call cudaDeviceReset() here to allow multiple contexts
        }
        self.allocator.destroy(self);
    }
    
    /// Get current memory usage information
    pub fn getMemoryInfo(self: *CudaContext) !struct { 
        free_mb: u64, 
        total_mb: u64, 
        used_mb: u64,
        utilization_percent: f32,
    } {
        _ = self;
        
        var free_bytes: usize = 0;
        var total_bytes: usize = 0;
        
        try cuda.checkCudaError(
            cuda.cudaMemGetInfo(&free_bytes, &total_bytes), 
            "cudaMemGetInfo"
        );
        
        const free_mb = free_bytes / (1024 * 1024);
        const total_mb = total_bytes / (1024 * 1024);
        const used_mb = total_mb - free_mb;
        const utilization = if (total_mb > 0)
            @as(f32, @floatFromInt(used_mb)) / @as(f32, @floatFromInt(total_mb)) * 100.0
        else
            0.0;
        
        return .{
            .free_mb = free_mb,
            .total_mb = total_mb,
            .used_mb = used_mb,
            .utilization_percent = utilization,
        };
    }
    
    /// Synchronize device (wait for all operations to complete)
    pub fn synchronize(self: *CudaContext) !void {
        _ = self;
        try cuda.checkCudaError(cuda.cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    
    /// Reset the device (clears all state)
    pub fn reset(self: *CudaContext) !void {
        _ = self;
        try cuda.checkCudaError(cuda.cudaDeviceReset(), "cudaDeviceReset");
    }
    
    /// Print current memory status
    pub fn printMemoryStatus(self: *CudaContext) !void {
        const mem = try self.getMemoryInfo();
        
        std.debug.print("\n📊 GPU Memory Status:\n", .{});
        std.debug.print("   Total: {d} MB ({d:.2} GB)\n", .{
            mem.total_mb,
            @as(f32, @floatFromInt(mem.total_mb)) / 1024.0,
        });
        std.debug.print("   Used: {d} MB ({d:.1}%)\n", .{
            mem.used_mb,
            mem.utilization_percent,
        });
        std.debug.print("   Free: {d} MB ({d:.2} GB)\n", .{
            mem.free_mb,
            @as(f32, @floatFromInt(mem.free_mb)) / 1024.0,
        });
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Initialize CUDA with device 0 (default)
pub fn initCUDA(allocator: std.mem.Allocator) !*CudaContext {
    return try CudaContext.init(allocator, 0);
}

/// Select device with most free memory
pub fn selectBestDevice(allocator: std.mem.Allocator) !*CudaContext {
    std.debug.print("\n🔍 Searching for best CUDA device...\n", .{});
    
    var device_count: c_int = 0;
    try cuda.checkCudaError(cuda.cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    
    if (device_count == 0) {
        return error.NoGPUFound;
    }
    
    var best_device: i32 = 0;
    var max_memory: usize = 0;
    
    var i: i32 = 0;
    while (i < device_count) : (i += 1) {
        var props: cuda.CudaDeviceProp = undefined;
        try cuda.checkCudaError(
            cuda.cudaGetDeviceProperties(&props, i), 
            "cudaGetDeviceProperties"
        );
        
        const name_end = std.mem.indexOfScalar(u8, &props.name, 0) orelse props.name.len;
        const name = props.name[0..name_end];
        
        std.debug.print("   Device {d}: {s} ({d:.1} GB)\n", .{
            i,
            name,
            @as(f32, @floatFromInt(props.totalGlobalMem)) / (1024.0 * 1024.0 * 1024.0),
        });
        
        if (props.totalGlobalMem > max_memory) {
            max_memory = props.totalGlobalMem;
            best_device = i;
        }
    }
    
    std.debug.print("   Selected device {d} (most memory)\n", .{best_device});
    
    return try CudaContext.init(allocator, best_device);
}

/// List all available CUDA devices
pub fn listDevices(allocator: std.mem.Allocator) !void {
    std.debug.print("\n📋 CUDA Devices:\n", .{});
    
    var device_count: c_int = 0;
    try cuda.checkCudaError(cuda.cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    
    if (device_count == 0) {
        std.debug.print("   No CUDA devices found\n", .{});
        return;
    }
    
    var i: i32 = 0;
    while (i < device_count) : (i += 1) {
        var props: cuda.CudaDeviceProp = undefined;
        try cuda.checkCudaError(
            cuda.cudaGetDeviceProperties(&props, i), 
            "cudaGetDeviceProperties"
        );
        
        const name_end = std.mem.indexOfScalar(u8, &props.name, 0) orelse props.name.len;
        const name = props.name[0..name_end];
        
        std.debug.print("\n   Device {d}: {s}\n", .{ i, name });
        std.debug.print("      Compute: {d}.{d}\n", .{ props.major, props.minor });
        std.debug.print("      Memory: {d:.2} GB\n", .{
            @as(f32, @floatFromInt(props.totalGlobalMem)) / (1024.0 * 1024.0 * 1024.0),
        });
        std.debug.print("      SMs: {d}\n", .{props.multiProcessorCount});
    }
    
    _ = allocator;
}

// ============================================================================
// Tests
// ============================================================================

test "cuda_context: initialization" {
    const ctx = CudaContext.init(std.testing.allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    try std.testing.expect(ctx.initialized);
    try std.testing.expect(ctx.properties.name.len > 0);
}

test "cuda_context: memory info" {
    const ctx = CudaContext.init(std.testing.allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    const mem = try ctx.getMemoryInfo();
    try std.testing.expect(mem.total_mb > 0);
    try std.testing.expect(mem.free_mb <= mem.total_mb);
}
