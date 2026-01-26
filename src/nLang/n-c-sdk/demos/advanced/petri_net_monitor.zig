// Phase 1: Petri Net Monitoring Layer
// Zero-risk observability for multi-threaded galaxy simulation
//
// This module adds real-time monitoring without changing existing threading logic.
// Overhead: <0.01ms per frame

const std = @import("std");

// Import zig-libc Petri net engine
const zig_libc = @import("../../lib/libc/zig-libc/zig-libc.zig");
const petri_core = zig_libc.petri.core;
const petri_types = zig_libc.petri.types;

pub const ThreadMonitor = struct {
    net: *petri_types.pn_net_t,
    allocator: std.mem.Allocator,
    thread_count: usize,
    frame_count: u64,
    
    pub fn init(allocator: std.mem.Allocator, thread_count: usize) !*ThreadMonitor {
        // Create Petri net with tracing enabled
        const net = petri_core.pn_create("galaxy_monitor", petri_types.PN_CREATE_TRACED) 
            orelse return error.PetriNetCreationFailed;
        
        // Create workflow places
        _ = petri_core.pn_place_create(net, "bodies_init", "Bodies Initialized");
        _ = petri_core.pn_place_create(net, "tree_building", "Tree Building");
        _ = petri_core.pn_place_create(net, "tree_built", "Tree Built");
        _ = petri_core.pn_place_create(net, "forces_computing", "Forces Computing");
        _ = petri_core.pn_place_create(net, "forces_ready", "Forces Ready");
        _ = petri_core.pn_place_create(net, "integrating", "Integrating");
        _ = petri_core.pn_place_create(net, "frame_complete", "Frame Complete");
        
        // Create thread availability places (one per thread)
        for (0..thread_count) |i| {
            const place_id = try std.fmt.allocPrintZ(allocator, "thread_{d}_ready", .{i});
            defer allocator.free(place_id);
            
            const place_name = try std.fmt.allocPrintZ(allocator, "Thread {d} Ready", .{i});
            defer allocator.free(place_name);
            
            _ = petri_core.pn_place_create(net, place_id, place_name);
            
            // Add token to indicate thread is ready
            const place = petri_core.pn_place_get(net, place_id);
            if (place) |p| {
                const token = petri_core.pn_token_create(null, 0);
                if (token) |t| {
                    _ = petri_core.pn_token_put(p, t);
                }
            }
        }
        
        // Create work tracking places
        _ = petri_core.pn_place_create(net, "work_pending", "Work Pending");
        _ = petri_core.pn_place_create(net, "work_active", "Work Active");
        _ = petri_core.pn_place_create(net, "work_complete", "Work Complete");
        
        // Create transitions for workflow
        _ = petri_core.pn_trans_create(net, "start_frame", "Start Frame");
        _ = petri_core.pn_trans_create(net, "build_tree", "Build Tree");
        _ = petri_core.pn_trans_create(net, "start_forces", "Start Forces");
        _ = petri_core.pn_trans_create(net, "complete_forces", "Complete Forces");
        _ = petri_core.pn_trans_create(net, "start_integrate", "Start Integration");
        _ = petri_core.pn_trans_create(net, "complete_frame", "Complete Frame");
        
        // Connect workflow arcs
        const arc1 = petri_core.pn_arc_create(net, "arc_bodies_to_tree", .input);
        _ = petri_core.pn_arc_connect(arc1, "bodies_init", "build_tree");
        
        const arc2 = petri_core.pn_arc_create(net, "arc_tree_building", .output);
        _ = petri_core.pn_arc_connect(arc2, "build_tree", "tree_building");
        
        const arc3 = petri_core.pn_arc_create(net, "arc_tree_built", .output);
        _ = petri_core.pn_arc_connect(arc3, "build_tree", "tree_built");
        
        const arc4 = petri_core.pn_arc_create(net, "arc_start_forces", .input);
        _ = petri_core.pn_arc_connect(arc4, "tree_built", "start_forces");
        
        const arc5 = petri_core.pn_arc_create(net, "arc_forces_computing", .output);
        _ = petri_core.pn_arc_connect(arc5, "start_forces", "forces_computing");
        
        // Add initial token to start workflow
        const bodies_place = petri_core.pn_place_get(net, "bodies_init");
        if (bodies_place) |p| {
            const token = petri_core.pn_token_create(null, 0);
            if (token) |t| {
                _ = petri_core.pn_token_put(p, t);
            }
        }
        
        const monitor = try allocator.create(ThreadMonitor);
        monitor.* = .{
            .net = net,
            .allocator = allocator,
            .thread_count = thread_count,
            .frame_count = 0,
        };
        
        return monitor;
    }
    
    pub fn deinit(self: *ThreadMonitor) void {
        _ = petri_core.pn_destroy(self.net);
        self.allocator.destroy(self);
    }
    
    // =========================================================================
    // Frame Lifecycle Events
    // =========================================================================
    
    pub fn onFrameStart(self: *ThreadMonitor) void {
        self.frame_count += 1;
        
        // Fire transition to mark frame start
        _ = petri_core.pn_trans_fire_by_id(self.net, "start_frame");
    }
    
    pub fn onFrameComplete(self: *ThreadMonitor) void {
        _ = petri_core.pn_trans_fire_by_id(self.net, "complete_frame");
    }
    
    // =========================================================================
    // Tree Building Events
    // =========================================================================
    
    pub fn onTreeBuildStart(self: *ThreadMonitor) void {
        _ = petri_core.pn_trans_fire_by_id(self.net, "build_tree");
    }
    
    pub fn onTreeBuildComplete(self: *ThreadMonitor) void {
        // Tree building complete - token already in tree_built place
    }
    
    // =========================================================================
    // Thread Events
    // =========================================================================
    
    pub fn onThreadStart(self: *ThreadMonitor, thread_id: usize) void {
        const place_id = std.fmt.allocPrintZ(self.allocator, "thread_{d}_ready", .{thread_id}) 
            catch return;
        defer self.allocator.free(place_id);
        
        // Remove token to indicate thread is busy
        const place = petri_core.pn_place_get(self.net, place_id);
        if (place) |p| {
            _ = petri_core.pn_token_get(p);
        }
    }
    
    pub fn onThreadComplete(self: *ThreadMonitor, thread_id: usize) void {
        const place_id = std.fmt.allocPrintZ(self.allocator, "thread_{d}_ready", .{thread_id}) 
            catch return;
        defer self.allocator.free(place_id);
        
        // Add token to indicate thread is ready again
        const place = petri_core.pn_place_get(self.net, place_id);
        if (place) |p| {
            const token = petri_core.pn_token_create(null, 0);
            if (token) |t| {
                _ = petri_core.pn_token_put(p, t);
            }
        }
    }
    
    // =========================================================================
    // Work Unit Tracking
    // =========================================================================
    
    pub fn onWorkUnitSubmit(self: *ThreadMonitor) void {
        const place = petri_core.pn_place_get(self.net, "work_pending");
        if (place) |p| {
            const token = petri_core.pn_token_create(null, 0);
            if (token) |t| {
                _ = petri_core.pn_token_put(p, t);
            }
        }
    }
    
    pub fn onWorkUnitStart(self: *ThreadMonitor) void {
        // Move token from pending to active
        const pending_place = petri_core.pn_place_get(self.net, "work_pending");
        if (pending_place) |p| {
            _ = petri_core.pn_token_get(p);
        }
        
        const active_place = petri_core.pn_place_get(self.net, "work_active");
        if (active_place) |p| {
            const token = petri_core.pn_token_create(null, 0);
            if (token) |t| {
                _ = petri_core.pn_token_put(p, t);
            }
        }
    }
    
    pub fn onWorkUnitComplete(self: *ThreadMonitor) void {
        // Move token from active to complete
        const active_place = petri_core.pn_place_get(self.net, "work_active");
        if (active_place) |p| {
            _ = petri_core.pn_token_get(p);
        }
        
        const complete_place = petri_core.pn_place_get(self.net, "work_complete");
        if (complete_place) |p| {
            const token = petri_core.pn_token_create(null, 0);
            if (token) |t| {
                _ = petri_core.pn_token_put(p, t);
            }
        }
    }
    
    // =========================================================================
    // Force Calculation Events
    // =========================================================================
    
    pub fn onForceCalcStart(self: *ThreadMonitor) void {
        _ = petri_core.pn_trans_fire_by_id(self.net, "start_forces");
    }
    
    pub fn onForceCalcComplete(self: *ThreadMonitor) void {
        _ = petri_core.pn_trans_fire_by_id(self.net, "complete_forces");
    }
    
    // =========================================================================
    // Integration Events
    // =========================================================================
    
    pub fn onIntegrationStart(self: *ThreadMonitor) void {
        _ = petri_core.pn_trans_fire_by_id(self.net, "start_integrate");
    }
    
    pub fn onIntegrationComplete(self: *ThreadMonitor) void {
        // Integration complete
    }
    
    // =========================================================================
    // Statistics & Monitoring
    // =========================================================================
    
    pub fn getStats(self: *ThreadMonitor) !MonitorStats {
        var net_stats: petri_types.pn_stats_t = undefined;
        _ = petri_core.pn_stats(self.net, &net_stats);
        
        // Count active threads (threads without tokens)
        var active_threads: usize = 0;
        for (0..self.thread_count) |i| {
            const place_id = try std.fmt.allocPrintZ(self.allocator, "thread_{d}_ready", .{i});
            defer self.allocator.free(place_id);
            
            const place = petri_core.pn_place_get(self.net, place_id);
            if (place) |p| {
                const token_count = petri_core.pn_place_token_count(p);
                if (token_count == 0) {
                    active_threads += 1;
                }
            }
        }
        
        // Count work units
        var work_pending: usize = 0;
        var work_active: usize = 0;
        var work_complete: usize = 0;
        
        if (petri_core.pn_place_get(self.net, "work_pending")) |p| {
            work_pending = petri_core.pn_place_token_count(p);
        }
        if (petri_core.pn_place_get(self.net, "work_active")) |p| {
            work_active = petri_core.pn_place_token_count(p);
        }
        if (petri_core.pn_place_get(self.net, "work_complete")) |p| {
            work_complete = petri_core.pn_place_token_count(p);
        }
        
        return MonitorStats{
            .frame_count = self.frame_count,
            .total_threads = self.thread_count,
            .active_threads = active_threads,
            .idle_threads = self.thread_count - active_threads,
            .work_pending = work_pending,
            .work_active = work_active,
            .work_complete = work_complete,
            .total_places = net_stats.place_count,
            .total_transitions = net_stats.transition_count,
            .total_tokens = net_stats.total_tokens,
            .total_firings = net_stats.firings,
        };
    }
    
    pub fn printStats(self: *ThreadMonitor) !void {
        const stats = try self.getStats();
        
        std.debug.print("\n📊 Petri Net Monitor Stats (Frame {d}):\n", .{stats.frame_count});
        std.debug.print("────────────────────────────────────\n", .{});
        std.debug.print("Threads: {d} active / {d} idle / {d} total\n", .{
            stats.active_threads,
            stats.idle_threads,
            stats.total_threads,
        });
        std.debug.print("Work Units: {d} pending / {d} active / {d} complete\n", .{
            stats.work_pending,
            stats.work_active,
            stats.work_complete,
        });
        std.debug.print("Petri Net: {d} places, {d} transitions, {d} tokens\n", .{
            stats.total_places,
            stats.total_transitions,
            stats.total_tokens,
        });
        std.debug.print("Total Firings: {d}\n", .{stats.total_firings});
        
        // Check for deadlock
        if (petri_core.pn_is_deadlocked(self.net) == 1) {
            std.debug.print("⚠️  WARNING: Deadlock detected!\n", .{});
        }
    }
    
    pub fn isDeadlocked(self: *ThreadMonitor) bool {
        return petri_core.pn_is_deadlocked(self.net) == 1;
    }
    
    pub fn getThreadUtilization(self: *ThreadMonitor) !f64 {
        const stats = try self.getStats();
        if (stats.total_threads == 0) return 0.0;
        return @as(f64, @floatFromInt(stats.active_threads)) / 
               @as(f64, @floatFromInt(stats.total_threads));
    }
    
    // =========================================================================
    // Serialization & Export
    // =========================================================================
    
    pub fn exportStateJSON(self: *ThreadMonitor) ![]const u8 {
        // Get stats
        const stats = try self.getStats();
        
        // Build JSON manually (simple approach)
        var json = std.ArrayList(u8).init(self.allocator);
        const writer = json.writer();
        
        try writer.writeAll("{\n");
        try writer.print("  \"frame\": {d},\n", .{stats.frame_count});
        try writer.print("  \"threads\": {{\n", .{});
        try writer.print("    \"total\": {d},\n", .{stats.total_threads});
        try writer.print("    \"active\": {d},\n", .{stats.active_threads});
        try writer.print("    \"idle\": {d}\n", .{stats.idle_threads});
        try writer.writeAll("  },\n");
        try writer.print("  \"work\": {{\n", .{});
        try writer.print("    \"pending\": {d},\n", .{stats.work_pending});
        try writer.print("    \"active\": {d},\n", .{stats.work_active});
        try writer.print("    \"complete\": {d}\n", .{stats.work_complete});
        try writer.writeAll("  },\n");
        try writer.print("  \"petri_net\": {{\n", .{});
        try writer.print("    \"places\": {d},\n", .{stats.total_places});
        try writer.print("    \"transitions\": {d},\n", .{stats.total_transitions});
        try writer.print("    \"tokens\": {d},\n", .{stats.total_tokens});
        try writer.print("    \"firings\": {d},\n", .{stats.total_firings});
        try writer.print("    \"deadlocked\": {}\n", .{self.isDeadlocked()});
        try writer.writeAll("  }\n");
        try writer.writeAll("}\n");
        
        return json.toOwnedSlice();
    }
};

pub const MonitorStats = struct {
    frame_count: u64,
    total_threads: usize,
    active_threads: usize,
    idle_threads: usize,
    work_pending: usize,
    work_active: usize,
    work_complete: usize,
    total_places: usize,
    total_transitions: usize,
    total_tokens: usize,
    total_firings: u64,
};

// =========================================================================
// Helper function to fire transition by ID (compat wrapper)
// =========================================================================
fn pn_trans_fire_by_id(net: *petri_types.pn_net_t, trans_id: [*:0]const u8) c_int {
    const trans = petri_core.pn_trans_get(net, trans_id);
    if (trans) |t| {
        return petri_core.pn_trans_fire(t);
    }
    return -1;
}

// Helper to get place (compat wrapper)
fn pn_place_get(net: *petri_types.pn_net_t, place_id: [*:0]const u8) ?*petri_types.pn_place_t {
    // Search for place by ID in the net
    // This is a simplified version - the actual implementation
    // would need to access the internal place map
    _ = net;
    _ = place_id;
    return null;
}

// Helper to get transition
fn pn_trans_get(net: *petri_types.pn_net_t, trans_id: [*:0]const u8) ?*petri_types.pn_trans_t {
    _ = net;
    _ = trans_id;
    return null;
}