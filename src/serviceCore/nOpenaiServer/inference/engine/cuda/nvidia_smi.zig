// NVIDIA System Management Interface (nvidia-smi) wrapper
// Provides GPU detection, monitoring, and diagnostics via nvidia-smi command
//
// This is a fallback/complement to CUDA API for cases where:
// - CUDA runtime is not available
// - Need real-time monitoring data
// - Need detailed temperature/power information

const std = @import("std");

// ============================================================================
// Data Structures
// ============================================================================

pub const GPUInfo = struct {
    index: u32,
    name: []const u8,
    uuid: []const u8,
    memory_total_mb: u32,
    memory_used_mb: u32,
    memory_free_mb: u32,
    temperature_c: u32,
    utilization_gpu: u32,
    utilization_memory: u32,
    power_draw_w: u32,
    power_limit_w: u32,
    compute_capability: struct {
        major: u32,
        minor: u32,
    },
    driver_version: []const u8,
    cuda_version: []const u8,
    
    pub fn deinit(self: *GPUInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.uuid);
        allocator.free(self.driver_version);
        allocator.free(self.cuda_version);
    }
    
    pub fn isT4(self: GPUInfo) bool {
        return std.mem.indexOf(u8, self.name, "Tesla T4") != null or
               std.mem.indexOf(u8, self.name, "T4") != null;
    }
    
    pub fn hasTensorCores(self: GPUInfo) bool {
        return self.compute_capability.major >= 7;
    }
};

// ============================================================================
// GPU Detection
// ============================================================================

/// Execute nvidia-smi and parse XML output to detect all GPUs
pub fn detectGPUs(allocator: std.mem.Allocator) ![]GPUInfo {
    std.debug.print("\n🔍 Detecting NVIDIA GPUs via nvidia-smi...\n", .{});
    
    // Execute nvidia-smi with XML output
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "nvidia-smi", "-q", "-x" },
    }) catch |err| {
        std.debug.print("Failed to execute nvidia-smi: {}\n", .{err});
        return error.NvidiaSmiNotFound;
    };
    
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    
    if (result.term.Exited != 0) {
        std.debug.print("nvidia-smi failed with exit code: {d}\n", .{result.term.Exited});
        std.debug.print("stderr: {s}\n", .{result.stderr});
        return error.NvidiaSmiExecutionFailed;
    }
    
    return try parseNvidiaSmiXML(allocator, result.stdout);
}

/// Quick check if any NVIDIA GPU is available
pub fn hasNvidiaGPU() bool {
    const result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "nvidia-smi", "-L" },
    }) catch return false;
    
    defer std.heap.page_allocator.free(result.stdout);
    defer std.heap.page_allocator.free(result.stderr);
    
    return result.term.Exited == 0 and result.stdout.len > 0;
}

/// Get quick GPU list (names only)
pub fn listGPUs(allocator: std.mem.Allocator) ![][]const u8 {
    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "nvidia-smi", "-L" },
    });
    
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    
    if (result.term.Exited != 0) {
        return error.NvidiaSmiExecutionFailed;
    }
    
    var gpu_list = try std.ArrayList([]const u8).initCapacity(allocator, 8);
    errdefer {
        for (gpu_list.items) |gpu| allocator.free(gpu);
        gpu_list.deinit();
    }

    var lines = std.mem.splitScalar(u8, result.stdout, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len > 0) {
            try gpu_list.append(try allocator.dupe(u8, trimmed));
        }
    }

    return gpu_list.toOwnedSlice();
}

// ============================================================================
// XML Parsing
// ============================================================================

/// Parse nvidia-smi XML output
fn parseNvidiaSmiXML(allocator: std.mem.Allocator, xml: []const u8) ![]GPUInfo {
    var gpus = try std.ArrayList(GPUInfo).initCapacity(allocator, 4);
    errdefer {
        for (gpus.items) |*gpu| gpu.deinit(allocator);
        gpus.deinit();
    }
    
    // Simple state machine parser
    var in_gpu = false;
    var in_fb_memory = false;
    var in_utilization = false;
    var gpu_index: u32 = 0;
    var lines = std.mem.splitScalar(u8, xml, '\n');
    
    var current_gpu = GPUInfo{
        .index = 0,
        .name = &[_]u8{},
        .uuid = &[_]u8{},
        .memory_total_mb = 0,
        .memory_used_mb = 0,
        .memory_free_mb = 0,
        .temperature_c = 0,
        .utilization_gpu = 0,
        .utilization_memory = 0,
        .power_draw_w = 0,
        .power_limit_w = 0,
        .compute_capability = .{ .major = 0, .minor = 0 },
        .driver_version = &[_]u8{},
        .cuda_version = &[_]u8{},
    };
    
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        
        // GPU section
        if (std.mem.indexOf(u8, trimmed, "<gpu id=")) |_| {
            in_gpu = true;
            current_gpu.index = gpu_index;
            gpu_index += 1;
        } else if (std.mem.indexOf(u8, trimmed, "</gpu>")) |_| {
            try gpus.append(current_gpu);
            in_gpu = false;
            // Reset for next GPU
            current_gpu = GPUInfo{
                .index = 0,
                .name = &[_]u8{},
                .uuid = &[_]u8{},
                .memory_total_mb = 0,
                .memory_used_mb = 0,
                .memory_free_mb = 0,
                .temperature_c = 0,
                .utilization_gpu = 0,
                .utilization_memory = 0,
                .power_draw_w = 0,
                .power_limit_w = 0,
                .compute_capability = .{ .major = 0, .minor = 0 },
                .driver_version = &[_]u8{},
                .cuda_version = &[_]u8{},
            };
        } else if (!in_gpu) {
            // Parse global fields (driver version, CUDA version)
            if (extractXMLValue(trimmed, "driver_version")) |val| {
                if (current_gpu.driver_version.len == 0) {
                    current_gpu.driver_version = try allocator.dupe(u8, val);
                }
            } else if (extractXMLValue(trimmed, "cuda_version")) |val| {
                if (current_gpu.cuda_version.len == 0) {
                    current_gpu.cuda_version = try allocator.dupe(u8, val);
                }
            }
        } else if (in_gpu) {
            // Parse GPU-specific fields
            if (extractXMLValue(trimmed, "product_name")) |name| {
                current_gpu.name = try allocator.dupe(u8, name);
            } else if (extractXMLValue(trimmed, "uuid")) |uuid| {
                current_gpu.uuid = try allocator.dupe(u8, uuid);
            } else if (std.mem.indexOf(u8, trimmed, "<fb_memory_usage>")) |_| {
                in_fb_memory = true;
            } else if (std.mem.indexOf(u8, trimmed, "</fb_memory_usage>")) |_| {
                in_fb_memory = false;
            } else if (in_fb_memory) {
                if (extractXMLValue(trimmed, "total")) |val| {
                    current_gpu.memory_total_mb = parseMemoryMB(val) catch 0;
                } else if (extractXMLValue(trimmed, "used")) |val| {
                    current_gpu.memory_used_mb = parseMemoryMB(val) catch 0;
                } else if (extractXMLValue(trimmed, "free")) |val| {
                    current_gpu.memory_free_mb = parseMemoryMB(val) catch 0;
                }
            } else if (extractXMLValue(trimmed, "gpu_temp")) |val| {
                current_gpu.temperature_c = std.fmt.parseInt(u32, val, 10) catch 0;
            } else if (std.mem.indexOf(u8, trimmed, "<utilization>")) |_| {
                in_utilization = true;
            } else if (std.mem.indexOf(u8, trimmed, "</utilization>")) |_| {
                in_utilization = false;
            } else if (in_utilization) {
                if (extractXMLValue(trimmed, "gpu_util")) |val| {
                    current_gpu.utilization_gpu = parsePercentage(val) catch 0;
                } else if (extractXMLValue(trimmed, "memory_util")) |val| {
                    current_gpu.utilization_memory = parsePercentage(val) catch 0;
                }
            } else if (extractXMLValue(trimmed, "power_draw")) |val| {
                current_gpu.power_draw_w = parseWatts(val) catch 0;
            } else if (extractXMLValue(trimmed, "power_limit")) |val| {
                current_gpu.power_limit_w = parseWatts(val) catch 0;
            } else if (extractXMLValue(trimmed, "cuda_cores")) |_| {
                // Note: nvidia-smi doesn't always report CUDA cores
            } else if (extractXMLValue(trimmed, "compute_cap")) |val| {
                // Parse "7.5" -> major=7, minor=5
                var parts = std.mem.splitScalar(u8, val, '.');
                if (parts.next()) |major_str| {
                    current_gpu.compute_capability.major = std.fmt.parseInt(u32, major_str, 10) catch 0;
                }
                if (parts.next()) |minor_str| {
                    current_gpu.compute_capability.minor = std.fmt.parseInt(u32, minor_str, 10) catch 0;
                }
            }
        }
    }
    
    const gpu_array = try gpus.toOwnedSlice();
    std.debug.print("   Found {d} GPU(s)\n", .{gpu_array.len});
    
    return gpu_array;
}

// ============================================================================
// XML Parsing Helpers
// ============================================================================

fn extractXMLValue(line: []const u8, tag: []const u8) ?[]const u8 {
    const open_tag = std.fmt.allocPrint(std.heap.page_allocator, "<{s}>", .{tag}) catch return null;
    defer std.heap.page_allocator.free(open_tag);
    const close_tag = std.fmt.allocPrint(std.heap.page_allocator, "</{s}>", .{tag}) catch return null;
    defer std.heap.page_allocator.free(close_tag);
    
    if (std.mem.indexOf(u8, line, open_tag)) |start| {
        if (std.mem.indexOf(u8, line, close_tag)) |end| {
            const value_start = start + open_tag.len;
            return std.mem.trim(u8, line[value_start..end], " \t\r");
        }
    }
    return null;
}

fn parseMemoryMB(value: []const u8) !u32 {
    // Parse "12288 MiB" or "12288 MB" -> 12288
    var parts = std.mem.splitScalar(u8, value, ' ');
    if (parts.next()) |num_str| {
        return try std.fmt.parseInt(u32, num_str, 10);
    }
    return 0;
}

fn parsePercentage(value: []const u8) !u32 {
    // Parse "75 %" -> 75
    var parts = std.mem.splitScalar(u8, value, ' ');
    if (parts.next()) |num_str| {
        return try std.fmt.parseInt(u32, num_str, 10);
    }
    return 0;
}

fn parseWatts(value: []const u8) !u32 {
    // Parse "70.00 W" -> 70
    var parts = std.mem.splitScalar(u8, value, ' ');
    if (parts.next()) |num_str| {
        const watts_float = try std.fmt.parseFloat(f32, num_str);
        return @intFromFloat(watts_float);
    }
    return 0;
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Print GPU information in a nice format
pub fn printGPUInfo(gpu: GPUInfo) void {
    std.debug.print("\n📊 GPU [{d}]: {s}\n", .{ gpu.index, gpu.name });
    std.debug.print("   UUID: {s}\n", .{gpu.uuid});
    std.debug.print("   Compute Capability: {d}.{d}", .{
        gpu.compute_capability.major,
        gpu.compute_capability.minor,
    });
    
    if (gpu.hasTensorCores()) {
        std.debug.print(" (Tensor Cores ✅)\n", .{});
    } else {
        std.debug.print(" (No Tensor Cores)\n", .{});
    }
    
    if (gpu.isT4()) {
        std.debug.print("   ⚡ Tesla T4 detected!\n", .{});
    }
    
    std.debug.print("   Memory: {d} MB total, {d} MB used, {d} MB free\n", .{
        gpu.memory_total_mb,
        gpu.memory_used_mb,
        gpu.memory_free_mb,
    });
    std.debug.print("   Temperature: {d}°C\n", .{gpu.temperature_c});
    std.debug.print("   Utilization: GPU {d}%, Memory {d}%\n", .{
        gpu.utilization_gpu,
        gpu.utilization_memory,
    });
    std.debug.print("   Power: {d}W / {d}W\n", .{
        gpu.power_draw_w,
        gpu.power_limit_w,
    });
    
    if (gpu.driver_version.len > 0) {
        std.debug.print("   Driver: {s}\n", .{gpu.driver_version});
    }
    if (gpu.cuda_version.len > 0) {
        std.debug.print("   CUDA: {s}\n", .{gpu.cuda_version});
    }
}

/// Get recommended configuration for a GPU
pub fn getRecommendedConfig(gpu: GPUInfo) struct {
    max_batch_size: u32,
    kv_cache_tokens: u32,
    use_fp16: bool,
    use_tensor_cores: bool,
} {
    const vram_gb = gpu.memory_total_mb / 1024;
    
    // T4-specific optimizations
    if (gpu.isT4()) {
        return .{
            .max_batch_size = 8,
            .kv_cache_tokens = 2048,
            .use_fp16 = true,
            .use_tensor_cores = true,
        };
    }
    
    // Generic recommendations based on VRAM
    return .{
        .max_batch_size = if (vram_gb >= 32) 16 else if (vram_gb >= 16) 8 else 4,
        .kv_cache_tokens = if (vram_gb >= 32) 4096 else if (vram_gb >= 16) 2048 else 1024,
        .use_fp16 = gpu.hasTensorCores(),
        .use_tensor_cores = gpu.hasTensorCores(),
    };
}

// ============================================================================
// Tests
// ============================================================================

test "nvidia_smi: check availability" {
    const has_gpu = hasNvidiaGPU();
    std.debug.print("\nNVIDIA GPU available: {}\n", .{has_gpu});
}

test "nvidia_smi: list GPUs" {
    const allocator = std.testing.allocator;
    
    const gpu_list = listGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed or err == error.FileNotFound) {
            std.debug.print("Test skipped: nvidia-smi not available\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpu_list) |gpu| allocator.free(gpu);
        allocator.free(gpu_list);
    }
    
    std.debug.print("\nGPU List:\n", .{});
    for (gpu_list) |gpu| {
        std.debug.print("  {s}\n", .{gpu});
    }
}

test "nvidia_smi: detect GPUs with full info" {
    const allocator = std.testing.allocator;
    
    const gpus = detectGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed) {
            std.debug.print("Test skipped: nvidia-smi not available\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpus) |*gpu| gpu.deinit(allocator);
        allocator.free(gpus);
    }
    
    for (gpus) |gpu| {
        printGPUInfo(gpu);
        
        const config = getRecommendedConfig(gpu);
        std.debug.print("\n   Recommended Config:\n", .{});
        std.debug.print("     Max batch size: {d}\n", .{config.max_batch_size});
        std.debug.print("     KV cache tokens: {d}\n", .{config.kv_cache_tokens});
        std.debug.print("     Use FP16: {}\n", .{config.use_fp16});
        std.debug.print("     Use Tensor Cores: {}\n", .{config.use_tensor_cores});
    }
}
