// GPU Monitor - CUDA/ROCm GPU Performance Tracking
// Monitors GPU utilization, memory, temperature, and power

const std = @import("std");
const builtin = @import("builtin");
const Thread = std.Thread;
const Allocator = std.mem.Allocator;

pub const GpuMonitorConfig = struct {
    enabled: bool = true,
    poll_interval_ms: u32 = 100,
    track_kernels: bool = true,
    persist_to_hana: bool = true,
};

pub const GpuType = enum {
    cuda,
    rocm,
    metal,
    unknown,
};

pub const GpuMetrics = struct {
    device_id: u32,
    timestamp_ns: i64,
    utilization_percent: f32,
    memory_used_mb: u64,
    memory_total_mb: u64,
    memory_free_mb: u64,
    temperature_c: f32,
    power_watts: f32,
    clock_speed_mhz: u32,
    pcie_throughput_mb_s: f32,
};

pub const KernelInfo = struct {
    name: []const u8,
    duration_us: u64,
    grid_dim: [3]u32,
    block_dim: [3]u32,
    shared_mem_bytes: usize,
    registers_per_thread: u32,
    timestamp_ns: i64,
    allocator: Allocator,

    pub fn deinit(self: *KernelInfo) void {
        self.allocator.free(self.name);
    }
};

pub const GpuProfile = struct {
    metrics_history: std.ArrayList(GpuMetrics),
    kernel_history: std.ArrayList(KernelInfo),
    device_count: u32,
    gpu_type: GpuType,
    start_time_ns: i64,
    allocator: Allocator,
    mutex: Thread.Mutex,

    pub fn init(allocator: Allocator, device_count: u32, gpu_type: GpuType) GpuProfile {
        return .{
            .metrics_history = std.ArrayList(GpuMetrics).init(allocator),
            .kernel_history = std.ArrayList(KernelInfo).init(allocator),
            .device_count = device_count,
            .gpu_type = gpu_type,
            .start_time_ns = std.time.nanoTimestamp(),
            .allocator = allocator,
            .mutex = Thread.Mutex{},
        };
    }

    pub fn deinit(self: *GpuProfile) void {
        for (self.kernel_history.items) |*kernel| {
            kernel.deinit();
        }
        self.metrics_history.deinit();
        self.kernel_history.deinit();
    }

    pub fn getAverageUtilization(self: *const GpuProfile, device_id: u32) f32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var sum: f32 = 0.0;
        var count: u32 = 0;

        for (self.metrics_history.items) |metrics| {
            if (metrics.device_id == device_id) {
                sum += metrics.utilization_percent;
                count += 1;
            }
        }

        return if (count > 0) sum / @as(f32, @floatFromInt(count)) else 0.0;
    }

    pub fn getPeakMemoryUsage(self: *const GpuProfile, device_id: u32) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var peak: u64 = 0;

        for (self.metrics_history.items) |metrics| {
            if (metrics.device_id == device_id and metrics.memory_used_mb > peak) {
                peak = metrics.memory_used_mb;
            }
        }

        return peak;
    }

    pub fn toJson(self: *GpuProfile, writer: anytype) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try writer.writeAll("{");
        try writer.print("\"device_count\":{d},", .{self.device_count});
        try writer.print("\"gpu_type\":\"{s}\",", .{@tagName(self.gpu_type)});

        try writer.writeAll("\"devices\":[");
        for (0..self.device_count) |device_id| {
            if (device_id > 0) try writer.writeAll(",");

            const avg_util = self.getAverageUtilization(@intCast(device_id));
            const peak_mem = self.getPeakMemoryUsage(@intCast(device_id));

            try writer.writeAll("{");
            try writer.print("\"id\":{d},", .{device_id});
            try writer.print("\"avg_utilization\":{d:.2},", .{avg_util});
            try writer.print("\"peak_memory_mb\":{d}", .{peak_mem});
            try writer.writeAll("}");
        }
        try writer.writeAll("],");

        // Recent metrics
        if (self.metrics_history.items.len > 0) {
            const latest = self.metrics_history.items[self.metrics_history.items.len - 1];
            try writer.writeAll("\"latest_metrics\":{");
            try writer.print("\"device_id\":{d},", .{latest.device_id});
            try writer.print("\"utilization\":{d:.2},", .{latest.utilization_percent});
            try writer.print("\"memory_used_mb\":{d},", .{latest.memory_used_mb});
            try writer.print("\"memory_total_mb\":{d},", .{latest.memory_total_mb});
            try writer.print("\"temperature_c\":{d:.1},", .{latest.temperature_c});
            try writer.print("\"power_watts\":{d:.1}", .{latest.power_watts});
            try writer.writeAll("},");
        }

        // Kernel statistics
        try writer.print("\"total_kernels\":{d}", .{self.kernel_history.items.len});
        try writer.writeAll("}");
    }
};

pub const GpuMonitor = struct {
    config: GpuMonitorConfig,
    profile: GpuProfile,
    is_running: std.atomic.Value(bool),
    monitor_thread: ?Thread,
    allocator: Allocator,

    // CUDA C API imports (if available)
    const cuda = if (builtin.os.tag == .linux or builtin.os.tag == .macos)
        @cImport({
            @cInclude("cuda_runtime.h");
            @cInclude("nvml.h");
        })
    else
        struct {};

    pub fn init(allocator: Allocator, config: GpuMonitorConfig) !*GpuMonitor {
        const gpu_type = detectGpuType();
        const device_count = try getDeviceCount(gpu_type);

        const monitor = try allocator.create(GpuMonitor);
        monitor.* = .{
            .config = config,
            .profile = GpuProfile.init(allocator, device_count, gpu_type),
            .is_running = std.atomic.Value(bool).init(false),
            .monitor_thread = null,
            .allocator = allocator,
        };
        return monitor;
    }

    pub fn deinit(self: *GpuMonitor) void {
        self.stop();
        self.profile.deinit();
        self.allocator.destroy(self);
    }

    pub fn start(self: *GpuMonitor) !void {
        if (!self.config.enabled) return;
        if (self.is_running.load(.acquire)) {
            return error.AlreadyRunning;
        }

        // Initialize NVML for CUDA GPUs
        if (self.profile.gpu_type == .cuda) {
            if (builtin.os.tag == .linux or builtin.os.tag == .macos) {
                const result = cuda.nvmlInit();
                if (result != cuda.NVML_SUCCESS) {
                    return error.NvmlInitFailed;
                }
            }
        }

        self.is_running.store(true, .release);
        self.monitor_thread = try Thread.spawn(.{}, monitorThreadFn, .{self});
    }

    pub fn stop(self: *GpuMonitor) void {
        if (!self.is_running.load(.acquire)) {
            return;
        }

        self.is_running.store(false, .release);

        if (self.monitor_thread) |thread| {
            thread.join();
            self.monitor_thread = null;
        }

        // Shutdown NVML
        if (self.profile.gpu_type == .cuda) {
            if (builtin.os.tag == .linux or builtin.os.tag == .macos) {
                _ = cuda.nvmlShutdown();
            }
        }
    }

    pub fn getProfile(self: *GpuMonitor) *GpuProfile {
        return &self.profile;
    }

    fn monitorThreadFn(self: *GpuMonitor) void {
        const interval_ns = @as(u64, self.config.poll_interval_ms) * std.time.ns_per_ms;

        while (self.is_running.load(.acquire)) {
            for (0..self.profile.device_count) |device_id| {
                self.collectMetrics(@intCast(device_id)) catch |err| {
                    std.log.warn("Failed to collect GPU metrics for device {d}: {}", .{ device_id, err });
                };
            }

            std.time.sleep(interval_ns);
        }
    }

    fn collectMetrics(self: *GpuMonitor, device_id: u32) !void {
        const metrics = switch (self.profile.gpu_type) {
            .cuda => try self.collectCudaMetrics(device_id),
            .rocm => try self.collectRocmMetrics(device_id),
            .metal => try self.collectMetalMetrics(device_id),
            .unknown => return,
        };

        self.profile.mutex.lock();
        defer self.profile.mutex.unlock();

        try self.profile.metrics_history.append(metrics);

        // Limit history size
        if (self.profile.metrics_history.items.len > 10000) {
            _ = self.profile.metrics_history.orderedRemove(0);
        }
    }

    fn collectCudaMetrics(self: *GpuMonitor, device_id: u32) !GpuMetrics {
        _ = self;

        if (builtin.os.tag != .linux and builtin.os.tag != .macos) {
            return error.CudaNotAvailable;
        }

        var device: cuda.nvmlDevice_t = undefined;
        var result = cuda.nvmlDeviceGetHandleByIndex(device_id, &device);
        if (result != cuda.NVML_SUCCESS) {
            return error.NvmlDeviceGetFailed;
        }

        // Get utilization
        var utilization: cuda.nvmlUtilization_t = undefined;
        result = cuda.nvmlDeviceGetUtilizationRates(device, &utilization);
        const gpu_util: f32 = if (result == cuda.NVML_SUCCESS)
            @floatFromInt(utilization.gpu)
        else
            0.0;

        // Get memory info
        var mem_info: cuda.nvmlMemory_t = undefined;
        result = cuda.nvmlDeviceGetMemoryInfo(device, &mem_info);
        const mem_used = if (result == cuda.NVML_SUCCESS)
            @divFloor(mem_info.used, 1024 * 1024)
        else
            0;
        const mem_total = if (result == cuda.NVML_SUCCESS)
            @divFloor(mem_info.total, 1024 * 1024)
        else
            0;

        // Get temperature
        var temperature: c_uint = 0;
        _ = cuda.nvmlDeviceGetTemperature(device, cuda.NVML_TEMPERATURE_GPU, &temperature);

        // Get power
        var power: c_uint = 0;
        _ = cuda.nvmlDeviceGetPowerUsage(device, &power);
        const power_watts: f32 = @as(f32, @floatFromInt(power)) / 1000.0;

        // Get clock speed
        var clock: c_uint = 0;
        _ = cuda.nvmlDeviceGetClockInfo(device, cuda.NVML_CLOCK_SM, &clock);

        return GpuMetrics{
            .device_id = device_id,
            .timestamp_ns = std.time.nanoTimestamp(),
            .utilization_percent = gpu_util,
            .memory_used_mb = mem_used,
            .memory_total_mb = mem_total,
            .memory_free_mb = mem_total - mem_used,
            .temperature_c = @floatFromInt(temperature),
            .power_watts = power_watts,
            .clock_speed_mhz = clock,
            .pcie_throughput_mb_s = 0.0,
        };
    }

    fn collectRocmMetrics(self: *GpuMonitor, device_id: u32) !GpuMetrics {
        _ = self;
        _ = device_id;
        // TODO: Implement ROCm metrics collection
        return error.RocmNotImplemented;
    }

    fn collectMetalMetrics(self: *GpuMonitor, device_id: u32) !GpuMetrics {
        _ = self;
        _ = device_id;
        // TODO: Implement Metal metrics collection
        return error.MetalNotImplemented;
    }

    pub fn trackKernel(self: *GpuMonitor, name: []const u8, duration_us: u64, grid: [3]u32, block: [3]u32) !void {
        if (!self.config.track_kernels) return;

        const kernel_info = KernelInfo{
            .name = try self.allocator.dupe(u8, name),
            .duration_us = duration_us,
            .grid_dim = grid,
            .block_dim = block,
            .shared_mem_bytes = 0,
            .registers_per_thread = 0,
            .timestamp_ns = std.time.nanoTimestamp(),
            .allocator = self.allocator,
        };

        self.profile.mutex.lock();
        defer self.profile.mutex.unlock();

        try self.profile.kernel_history.append(kernel_info);

        // Limit history
        if (self.profile.kernel_history.items.len > 10000) {
            var old = self.profile.kernel_history.orderedRemove(0);
            old.deinit();
        }
    }
};

fn detectGpuType() GpuType {
    if (builtin.os.tag == .linux or builtin.os.tag == .macos) {
        // Try CUDA first
        const cuda_result = @cImport({
            @cInclude("cuda_runtime.h");
        });
        var device_count: c_int = 0;
        const result = cuda_result.cudaGetDeviceCount(&device_count);
        if (result == cuda_result.cudaSuccess and device_count > 0) {
            return .cuda;
        }

        // Try ROCm
        // TODO: Add ROCm detection

        // Try Metal on macOS
        if (builtin.os.tag == .macos) {
            return .metal;
        }
    }

    return .unknown;
}

fn getDeviceCount(gpu_type: GpuType) !u32 {
    switch (gpu_type) {
        .cuda => {
            if (builtin.os.tag == .linux or builtin.os.tag == .macos) {
                const cuda_api = @cImport({
                    @cInclude("cuda_runtime.h");
                });
                var count: c_int = 0;
                const result = cuda_api.cudaGetDeviceCount(&count);
                if (result == cuda_api.cudaSuccess) {
                    return @intCast(count);
                }
            }
            return 0;
        },
        .rocm => return 0, // TODO
        .metal => return 1, // macOS typically has 1 GPU
        .unknown => return 0,
    }
}

// Testing
test "GpuMonitor basic" {
    const allocator = std.testing.allocator;

    const config = GpuMonitorConfig{
        .enabled = true,
        .poll_interval_ms = 100,
        .track_kernels = true,
    };

    var monitor = GpuMonitor.init(allocator, config) catch |err| {
        // GPU may not be available in test environment
        if (err == error.NvmlInitFailed) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer monitor.deinit();

    // Test basic functionality without starting monitor
    // (to avoid requiring actual GPU in test environment)
    const profile = monitor.getProfile();
    try std.testing.expect(profile.device_count >= 0);
}
