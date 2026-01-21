// ============================================================================
// Load Tracker - Day 37 Implementation
// ============================================================================
// Purpose: Real-time load tracking and capacity management for model router
// Week: Week 8 (Days 36-40) - Load Balancing & Distribution
// Phase: Month 3 - Advanced Features & Optimization
// ============================================================================

const std = @import("std");

// ============================================================================
// LOAD TRACKING STRUCTURES
// ============================================================================

pub const ModelLoad = struct {
    active_requests: u32,
    queue_depth: u32,
    avg_latency_ms: f32,
    utilization: f32, // 0.0-1.0
    last_updated: i64,
    
    pub fn calculateUtilization(
        active: u32,
        max_concurrent: u32,
    ) f32 {
        if (max_concurrent == 0) return 0.0;
        return @as(f32, @floatFromInt(active)) / @as(f32, @floatFromInt(max_concurrent));
    }
};

pub const Capacity = struct {
    max_concurrent: u32,
    max_queue: u32,
    target_utilization: f32,
    
    pub fn isWithinCapacity(self: *const Capacity, load: ModelLoad) bool {
        return load.active_requests < self.max_concurrent and
               load.queue_depth < self.max_queue;
    }
    
    pub fn isOverTarget(self: *const Capacity, load: ModelLoad) bool {
        return load.utilization > self.target_utilization;
    }
};

// ============================================================================
// LOAD TRACKER
// ============================================================================

pub const LoadTracker = struct {
    allocator: std.mem.Allocator,
    
    // Track current load per model
    model_loads: std.StringHashMap(ModelLoad),
    
    // Track capacity limits
    model_capacities: std.StringHashMap(Capacity),
    
    // Configuration
    stale_threshold_ms: i64,
    
    pub fn init(allocator: std.mem.Allocator) LoadTracker {
        return .{
            .allocator = allocator,
            .model_loads = std.StringHashMap(ModelLoad).init(allocator),
            .model_capacities = std.StringHashMap(Capacity).init(allocator),
            .stale_threshold_ms = 60_000, // 1 minute
        };
    }
    
    pub fn deinit(self: *LoadTracker) void {
        self.model_loads.deinit();
        self.model_capacities.deinit();
    }
    
    /// Get current load for a model
    pub fn getCurrentLoad(
        self: *LoadTracker,
        model_id: []const u8,
    ) ?ModelLoad {
        if (self.model_loads.get(model_id)) |load| {
            const now = std.time.milliTimestamp();
            if (now - load.last_updated < self.stale_threshold_ms) {
                return load;
            }
        }
        return null;
    }
    
    /// Update load for a model (increment/decrement active requests)
    pub fn updateLoad(
        self: *LoadTracker,
        model_id: []const u8,
        delta_active: i32,
    ) !void {
        const now = std.time.milliTimestamp();
        
        var load = self.model_loads.get(model_id) orelse ModelLoad{
            .active_requests = 0,
            .queue_depth = 0,
            .avg_latency_ms = 0.0,
            .utilization = 0.0,
            .last_updated = now,
        };
        
        // Update active requests
        if (delta_active > 0) {
            load.active_requests += @intCast(delta_active);
        } else {
            const abs_delta: u32 = @intCast(-delta_active);
            load.active_requests = if (load.active_requests > abs_delta)
                load.active_requests - abs_delta
            else
                0;
        }
        
        // Recalculate utilization
        if (self.model_capacities.get(model_id)) |capacity| {
            load.utilization = ModelLoad.calculateUtilization(
                load.active_requests,
                capacity.max_concurrent
            );
        }
        
        load.last_updated = now;
        
        const key_copy = try self.allocator.dupe(u8, model_id);
        try self.model_loads.put(key_copy, load);
    }
    
    /// Set capacity limits for a model
    pub fn setCapacity(
        self: *LoadTracker,
        model_id: []const u8,
        capacity: Capacity,
    ) !void {
        const key_copy = try self.allocator.dupe(u8, model_id);
        try self.model_capacities.put(key_copy, capacity);
    }
    
    /// Check if model is overloaded
    pub fn isOverloaded(
        self: *LoadTracker,
        model_id: []const u8,
    ) bool {
        const load = self.getCurrentLoad(model_id) orelse return false;
        const capacity = self.model_capacities.get(model_id) orelse return false;
        
        return !capacity.isWithinCapacity(load);
    }
    
    /// Check if model is over target utilization
    pub fn isOverTarget(
        self: *LoadTracker,
        model_id: []const u8,
    ) bool {
        const load = self.getCurrentLoad(model_id) orelse return false;
        const capacity = self.model_capacities.get(model_id) orelse return false;
        
        return capacity.isOverTarget(load);
    }
    
    /// Get utilization for a model
    pub fn getUtilization(
        self: *LoadTracker,
        model_id: []const u8,
    ) ?f32 {
        if (self.getCurrentLoad(model_id)) |load| {
            return load.utilization;
        }
        return null;
    }
    
    /// Get all model loads
    pub fn getAllLoads(self: *LoadTracker) std.StringHashMap(ModelLoad) {
        return self.model_loads;
    }
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "LoadTracker: basic load tracking" {
    const allocator = std.testing.allocator;
    
    var tracker = LoadTracker.init(allocator);
    defer tracker.deinit();
    
    // Set capacity
    try tracker.setCapacity("model-1", .{
        .max_concurrent = 10,
        .max_queue = 5,
        .target_utilization = 0.8,
    });
    
    // Update load
    try tracker.updateLoad("model-1", 3); // +3 active
    
    const load = tracker.getCurrentLoad("model-1");
    try std.testing.expect(load != null);
    try std.testing.expectEqual(@as(u32, 3), load.?.active_requests);
}

test "LoadTracker: utilization calculation" {
    const allocator = std.testing.allocator;
    
    var tracker = LoadTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.setCapacity("model-1", .{
        .max_concurrent = 10,
        .max_queue = 5,
        .target_utilization = 0.8,
    });
    
    try tracker.updateLoad("model-1", 5); // 5 active out of 10
    
    const util = tracker.getUtilization("model-1");
    try std.testing.expect(util != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), util.?, 0.01);
}

test "LoadTracker: overload detection" {
    const allocator = std.testing.allocator;
    
    var tracker = LoadTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.setCapacity("model-1", .{
        .max_concurrent = 5,
        .max_queue = 2,
        .target_utilization = 0.8,
    });
    
    try tracker.updateLoad("model-1", 6); // 6 > 5 (overloaded)
    
    try std.testing.expect(tracker.isOverloaded("model-1"));
}

test "LoadTracker: decrement load" {
    const allocator = std.testing.allocator;
    
    var tracker = LoadTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.setCapacity("model-1", .{
        .max_concurrent = 10,
        .max_queue = 5,
        .target_utilization = 0.8,
    });
    
    try tracker.updateLoad("model-1", 5); // +5
    try tracker.updateLoad("model-1", -2); // -2 = 3
    
    const load = tracker.getCurrentLoad("model-1");
    try std.testing.expect(load != null);
    try std.testing.expectEqual(@as(u32, 3), load.?.active_requests);
}

test "LoadTracker: target utilization check" {
    const allocator = std.testing.allocator;
    
    var tracker = LoadTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.setCapacity("model-1", .{
        .max_concurrent = 10,
        .max_queue = 5,
        .target_utilization = 0.7, // 70%
    });
    
    try tracker.updateLoad("model-1", 8); // 80% utilization
    
    try std.testing.expect(tracker.isOverTarget("model-1"));
}
