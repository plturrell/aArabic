//! GPU Load Monitoring Module
//! Real-time GPU state monitoring for dynamic model routing
//!
//! Features:
//! - Query GPU state via nvidia-smi or CUDA API
//! - Track memory usage, temperature, utilization
//! - Health checking and load balancing
//! - Automatic GPU selection for model deployment

const std = @import("std");
const Allocator = std.mem.Allocator;

/// GPU state information
pub const GPUState = struct {
    device_id: u32,
    name: []const u8,
    total_memory_mb: usize,
    used_memory_mb: usize,
    free_memory_mb: usize,
    temperature_c: f32,
    utilization_percent: f32,
    power_usage_w: f32,
    power_limit_w: f32,
    
    pub fn deinit(self: *GPUState, allocator: Allocator) void {
        allocator.free(self.name);
    }
    
    pub fn availableMemoryMB(self: GPUState) usize {
        return self.free_memory_mb;
    }
    
    pub fn isHealthy(self: GPUState) bool {
        return self.temperature_c < 85.0 and 
               self.utilization_percent < 95.0 and
               self.power_usage_w < self.power_limit_w * 0.95;
    }
    
    pub fn isAvailable(self: GPUState, required_memory_mb: usize) bool {
        return self.isHealthy() and self.availableMemoryMB() >= required_memory_mb;
    }
    
    pub fn loadScore(self: GPUState) f32 {
        // Calculate load score (0.0 = idle, 1.0 = fully loaded)
        const mem_load = @as(f32, @floatFromInt(self.used_memory_mb)) / 
                        @as(f32, @floatFromInt(self.total_memory_mb));
        const util_load = self.utilization_percent / 100.0;
        const temp_load = self.temperature_c / 85.0; // 85°C threshold
        
        // Weighted average
        return mem_load * 0.5 + util_load * 0.3 + temp_load * 0.2;
    }
};

/// GPU Monitor - Real-time GPU state management
pub const GPUMonitor = struct {
    allocator: Allocator,
    gpu_states: std.ArrayList(GPUState),
    update_interval_ms: u32,
    last_update: i64,
    nvidia_smi_path: []const u8,
    
    pub fn init(allocator: Allocator, update_interval_ms: u32) !*GPUMonitor {
        const self = try allocator.create(GPUMonitor);
        self.* = .{
            .allocator = allocator,
            .gpu_states = std.ArrayList(GPUState).init(allocator),
            .update_interval_ms = update_interval_ms,
            .last_update = 0,
            .nvidia_smi_path = "nvidia-smi",
        };
        
        // Initial query
        try self.refresh();
        
        return self;
    }
    
    pub fn deinit(self: *GPUMonitor) void {
        for (self.gpu_states.items) |*state| {
            state.deinit(self.allocator);
        }
        self.gpu_states.deinit();
        self.allocator.destroy(self);
    }
    
    /// Refresh GPU states from nvidia-smi
    pub fn refresh(self: *GPUMonitor) !void {
        const now = std.time.milliTimestamp();
        
        // Check if update needed
        if (self.last_update > 0 and (now - self.last_update) < self.update_interval_ms) {
            return; // Use cached data
        }
        
        // Clear old states
        for (self.gpu_states.items) |*state| {
            state.deinit(self.allocator);
        }
        self.gpu_states.clearRetainingCapacity();
        
        // Query nvidia-smi
        const gpu_data = try self.queryNvidiaSMI();
        defer self.allocator.free(gpu_data);
        
        // Parse output
        try self.parseNvidiaSMIOutput(gpu_data);
        
        self.last_update = now;
    }
    
    fn queryNvidiaSMI(self: *GPUMonitor) ![]const u8 {
        // Execute nvidia-smi command
        // Format: nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
        
        var child = std.process.Child.init(
            &[_][]const u8{
                self.nvidia_smi_path,
                "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            },
            self.allocator,
        );
        
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;
        
        try child.spawn();
        
        const stdout = try child.stdout.?.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
        errdefer self.allocator.free(stdout);
        
        const stderr = try child.stderr.?.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(stderr);
        
        const term = try child.wait();
        
        if (term != .Exited or term.Exited != 0) {
            std.log.err("nvidia-smi failed: {s}", .{stderr});
            return error.NvidiaSMIFailed;
        }
        
        return stdout;
    }
    
    fn parseNvidiaSMIOutput(self: *GPUMonitor, output: []const u8) !void {
        var lines = std.mem.splitScalar(u8, output, '\n');
        
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            
            // Parse CSV line: index, name, mem_total, mem_used, mem_free, temp, util, power, power_limit
            var fields = std.mem.splitScalar(u8, line, ',');
            
            var field_values: [9][]const u8 = undefined;
            var field_idx: usize = 0;
            
            while (fields.next()) |field| : (field_idx += 1) {
                if (field_idx >= 9) break;
                field_values[field_idx] = std.mem.trim(u8, field, " ");
            }
            
            if (field_idx < 9) continue; // Incomplete line
            
            const device_id = try std.fmt.parseInt(u32, field_values[0], 10);
            const name = try self.allocator.dupe(u8, field_values[1]);
            const total_memory_mb = try std.fmt.parseInt(usize, field_values[2], 10);
            const used_memory_mb = try std.fmt.parseInt(usize, field_values[3], 10);
            const free_memory_mb = try std.fmt.parseInt(usize, field_values[4], 10);
            const temperature_c = try std.fmt.parseFloat(f32, field_values[5]);
            const utilization_percent = try std.fmt.parseFloat(f32, field_values[6]);
            const power_usage_w = try std.fmt.parseFloat(f32, field_values[7]);
            const power_limit_w = try std.fmt.parseFloat(f32, field_values[8]);
            
            try self.gpu_states.append(.{
                .device_id = device_id,
                .name = name,
                .total_memory_mb = total_memory_mb,
                .used_memory_mb = used_memory_mb,
                .free_memory_mb = free_memory_mb,
                .temperature_c = temperature_c,
                .utilization_percent = utilization_percent,
                .power_usage_w = power_usage_w,
                .power_limit_w = power_limit_w,
            });
        }
    }
    
    /// Select best GPU for a model based on available memory
    pub fn selectBestGPU(self: *GPUMonitor, required_memory_mb: usize) !GPUSelection {
        try self.refresh();
        
        var best_gpu: ?GPUSelection = null;
        var best_load: f32 = 2.0; // Start with impossible load
        
        for (self.gpu_states.items) |state| {
            if (!state.isAvailable(required_memory_mb)) continue;
            
            const load = state.loadScore();
            if (load < best_load) {
                best_load = load;
                best_gpu = .{
                    .device_id = state.device_id,
                    .available_memory_mb = state.availableMemoryMB(),
                    .load_score = load,
                };
            }
        }
        
        return best_gpu orelse error.NoGPUAvailable;
    }
    
    /// Get all healthy GPUs
    pub fn getHealthyGPUs(self: *GPUMonitor, allocator: Allocator) ![]GPUState {
        try self.refresh();
        
        var healthy = std.ArrayList(GPUState).init(allocator);
        errdefer healthy.deinit();
        
        for (self.gpu_states.items) |state| {
            if (state.isHealthy()) {
                try healthy.append(state);
            }
        }
        
        return try healthy.toOwnedSlice();
    }
    
    /// Get total available memory across all GPUs
    pub fn getTotalAvailableMemory(self: *GPUMonitor) !usize {
        try self.refresh();
        
        var total: usize = 0;
        for (self.gpu_states.items) |state| {
            if (state.isHealthy()) {
                total += state.availableMemoryMB();
            }
        }
        
        return total;
    }
    
    /// Get GPU count
    pub fn getGPUCount(self: *GPUMonitor) usize {
        return self.gpu_states.items.len;
    }
    
    /// Get specific GPU state
    pub fn getGPUState(self: *GPUMonitor, device_id: u32) ?*const GPUState {
        for (self.gpu_states.items) |*state| {
            if (state.device_id == device_id) {
                return state;
            }
        }
        return null;
    }
};

/// GPU selection result
pub const GPUSelection = struct {
    device_id: u32,
    available_memory_mb: usize,
    load_score: f32,
};

// Tests
test "GPUState - available memory" {
    const state = GPUState{
        .device_id = 0,
        .name = "Tesla T4",
        .total_memory_mb = 16384,
        .used_memory_mb = 4096,
        .free_memory_mb = 12288,
        .temperature_c = 65.0,
        .utilization_percent = 45.0,
        .power_usage_w = 50.0,
        .power_limit_w = 70.0,
    };
    
    try std.testing.expectEqual(@as(usize, 12288), state.availableMemoryMB());
}

test "GPUState - health check" {
    var state = GPUState{
        .device_id = 0,
        .name = "Tesla T4",
        .total_memory_mb = 16384,
        .used_memory_mb = 4096,
        .free_memory_mb = 12288,
        .temperature_c = 65.0,
        .utilization_percent = 45.0,
        .power_usage_w = 50.0,
        .power_limit_w = 70.0,
    };
    
    try std.testing.expect(state.isHealthy());
    
    // Test unhealthy conditions
    state.temperature_c = 90.0;
    try std.testing.expect(!state.isHealthy());
    
    state.temperature_c = 65.0;
    state.utilization_percent = 98.0;
    try std.testing.expect(!state.isHealthy());
}

test "GPUState - load score" {
    const state = GPUState{
        .device_id = 0,
        .name = "Tesla T4",
        .total_memory_mb = 16384,
        .used_memory_mb = 8192, // 50% used
        .free_memory_mb = 8192,
        .temperature_c = 42.5, // 50% of 85°C
        .utilization_percent = 50.0,
        .power_usage_w = 50.0,
        .power_limit_w = 70.0,
    };
    
    const load = state.loadScore();
    // Expected: 0.5 * 0.5 + 0.5 * 0.3 + 0.5 * 0.2 = 0.25 + 0.15 + 0.10 = 0.50
    try std.testing.expectApproxEqAbs(@as(f32, 0.50), load, 0.01);
}

test "GPUState - availability check" {
    const state = GPUState{
        .device_id = 0,
        .name = "Tesla T4",
        .total_memory_mb = 16384,
        .used_memory_mb = 4096,
        .free_memory_mb = 12288,
        .temperature_c = 65.0,
        .utilization_percent = 45.0,
        .power_usage_w = 50.0,
        .power_limit_w = 70.0,
    };
    
    try std.testing.expect(state.isAvailable(8 * 1024)); // 8GB
    try std.testing.expect(!state.isAvailable(20 * 1024)); // 20GB
}

// Note: Remaining tests require actual nvidia-smi or mock implementation
// These would be integration tests
