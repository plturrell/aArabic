// C API Exports for Mojo FFI Bindings
// Part of serviceCore nWorkflow
// Day 7: FFI Bridge to enable Mojo to call Zig functions
//
// This file exports Zig functions with C ABI so they can be called
// from Mojo via FFI. It provides a stable ABI boundary between
// the high-performance Zig core and the Pythonic Mojo interface.

const std = @import("std");
const petri_net = @import("petri_net.zig");
const executor = @import("executor.zig");

const PetriNet = petri_net.PetriNet;
const PetriNetExecutor = executor.PetriNetExecutor;
const ExecutionStrategy = executor.ExecutionStrategy;
const ConflictResolution = executor.ConflictResolution;

// Global allocator for C API (uses GPA for safety)
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Registry to track created Petri Nets
var net_registry = std.AutoHashMap(u64, *PetriNet).init(allocator);
var executor_registry = std.AutoHashMap(u64, *PetriNetExecutor).init(allocator);
var next_net_id: u64 = 1;
var next_executor_id: u64 = 1;
var registry_mutex = std.Thread.Mutex{};

// Error codes for C API
pub const ErrorCode = enum(c_int) {
    success = 0,
    null_pointer = 1,
    allocation_failed = 2,
    invalid_id = 3,
    invalid_parameter = 4,
    not_found = 5,
    already_exists = 6,
    deadlock = 7,
    unknown = 99,
};

// ============================================================================
// LIFECYCLE MANAGEMENT
// ============================================================================

/// Initialize the C API (call once at startup)
export fn nworkflow_init() ErrorCode {
    return ErrorCode.success;
}

/// Cleanup the C API (call once at shutdown)
export fn nworkflow_cleanup() ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    // Clean up all registered nets
    var net_it = net_registry.iterator();
    while (net_it.next()) |entry| {
        var net = entry.value_ptr.*;
        net.deinit();
        allocator.destroy(net);
    }
    net_registry.deinit();
    
    // Clean up all registered executors
    var exec_it = executor_registry.iterator();
    while (exec_it.next()) |entry| {
        var exec = entry.value_ptr.*;
        exec.deinit();
        allocator.destroy(exec);
    }
    executor_registry.deinit();
    
    return ErrorCode.success;
}

// ============================================================================
// PETRI NET CREATION AND MANAGEMENT
// ============================================================================

/// Create a new Petri Net and return its handle
export fn nworkflow_create_net(name: [*:0]const u8) u64 {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const name_slice = std.mem.span(name);
    
    const net = allocator.create(PetriNet) catch return 0;
    net.* = PetriNet.init(allocator, name_slice) catch {
        allocator.destroy(net);
        return 0;
    };
    
    const id = next_net_id;
    next_net_id += 1;
    
    net_registry.put(id, net) catch {
        net.deinit();
        allocator.destroy(net);
        return 0;
    };
    
    return id;
}

/// Destroy a Petri Net and free its resources
export fn nworkflow_destroy_net(net_id: u64) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    if (net_registry.fetchRemove(net_id)) |entry| {
        var net = entry.value;
        net.deinit();
        allocator.destroy(net);
        return ErrorCode.success;
    }
    
    return ErrorCode.invalid_id;
}

/// Get the name of a Petri Net
export fn nworkflow_get_net_name(net_id: u64, buffer: [*]u8, buffer_len: usize) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return ErrorCode.invalid_id;
    
    const copy_len = @min(net.name.len, buffer_len - 1);
    @memcpy(buffer[0..copy_len], net.name[0..copy_len]);
    buffer[copy_len] = 0; // Null terminate
    
    return ErrorCode.success;
}

// ============================================================================
// PLACE MANAGEMENT
// ============================================================================

/// Add a place to a Petri Net
export fn nworkflow_add_place(
    net_id: u64,
    place_id: [*:0]const u8,
    name: [*:0]const u8,
    capacity: i32,
) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return ErrorCode.invalid_id;
    
    const place_id_slice = std.mem.span(place_id);
    const name_slice = std.mem.span(name);
    const cap: ?usize = if (capacity < 0) null else @intCast(capacity);
    
    _ = net.addPlace(place_id_slice, name_slice, cap) catch return ErrorCode.allocation_failed;
    
    return ErrorCode.success;
}

/// Get the number of tokens in a place
export fn nworkflow_get_place_token_count(net_id: u64, place_id: [*:0]const u8) i32 {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return -1;
    const place_id_slice = std.mem.span(place_id);
    
    const place = net.places.get(place_id_slice) orelse return -1;
    return @intCast(place.tokens.items.len);
}

// ============================================================================
// TRANSITION MANAGEMENT
// ============================================================================

/// Add a transition to a Petri Net
export fn nworkflow_add_transition(
    net_id: u64,
    transition_id: [*:0]const u8,
    name: [*:0]const u8,
    priority: i32,
) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return ErrorCode.invalid_id;
    
    const transition_id_slice = std.mem.span(transition_id);
    const name_slice = std.mem.span(name);
    
    _ = net.addTransition(transition_id_slice, name_slice, priority) catch 
        return ErrorCode.allocation_failed;
    
    return ErrorCode.success;
}

/// Fire a transition
export fn nworkflow_fire_transition(net_id: u64, transition_id: [*:0]const u8) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return ErrorCode.invalid_id;
    const transition_id_slice = std.mem.span(transition_id);
    
    net.fireTransition(transition_id_slice) catch return ErrorCode.unknown;
    
    return ErrorCode.success;
}

// ============================================================================
// ARC MANAGEMENT
// ============================================================================

/// Add an arc to a Petri Net
export fn nworkflow_add_arc(
    net_id: u64,
    arc_id: [*:0]const u8,
    arc_type: u32,  // 0=input, 1=output, 2=inhibitor
    weight: u32,
    source_id: [*:0]const u8,
    target_id: [*:0]const u8,
) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return ErrorCode.invalid_id;
    
    const arc_id_slice = std.mem.span(arc_id);
    const source_slice = std.mem.span(source_id);
    const target_slice = std.mem.span(target_id);
    
    const arc_type_enum: petri_net.ArcType = switch (arc_type) {
        0 => .input,
        1 => .output,
        2 => .inhibitor,
        else => return ErrorCode.invalid_parameter,
    };
    
    _ = net.addArc(arc_id_slice, arc_type_enum, weight, source_slice, target_slice) catch 
        return ErrorCode.allocation_failed;
    
    return ErrorCode.success;
}

// ============================================================================
// TOKEN MANAGEMENT
// ============================================================================

/// Add a token to a place
export fn nworkflow_add_token(
    net_id: u64,
    place_id: [*:0]const u8,
    data: [*:0]const u8,
) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return ErrorCode.invalid_id;
    
    const place_id_slice = std.mem.span(place_id);
    const data_slice = std.mem.span(data);
    
    net.addTokenToPlace(place_id_slice, data_slice) catch return ErrorCode.allocation_failed;
    
    return ErrorCode.success;
}

// ============================================================================
// STATE QUERIES
// ============================================================================

/// Check if the net is in deadlock
export fn nworkflow_is_deadlocked(net_id: u64) bool {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return true;
    return net.isDeadlocked();
}

/// Get number of enabled transitions
export fn nworkflow_get_enabled_count(net_id: u64) i32 {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return -1;
    
    var enabled = net.getEnabledTransitions() catch return -1;
    defer enabled.deinit(allocator);
    
    return @intCast(enabled.items.len);
}

/// Get enabled transition IDs (writes to buffer)
export fn nworkflow_get_enabled_transitions(
    net_id: u64,
    buffer: [*][*:0]u8,
    buffer_size: usize,
) i32 {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return -1;
    
    var enabled = net.getEnabledTransitions() catch return -1;
    defer enabled.deinit(allocator);
    
    const count = @min(enabled.items.len, buffer_size);
    for (enabled.items[0..count], 0..) |trans_id, i| {
        // Allocate and copy string (caller must free)
        const str = allocator.dupeZ(u8, trans_id) catch return -1;
        buffer[i] = str;
    }
    
    return @intCast(count);
}

// ============================================================================
// EXECUTOR MANAGEMENT
// ============================================================================

/// Create an executor for a Petri Net
export fn nworkflow_create_executor(
    net_id: u64,
    strategy: u32,  // 0=sequential, 1=concurrent, 2=priority, 3=custom
) u64 {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const net = net_registry.get(net_id) orelse return 0;
    
    const strategy_enum: ExecutionStrategy = switch (strategy) {
        0 => .sequential,
        1 => .concurrent,
        2 => .priority_based,
        3 => .custom,
        else => return 0,
    };
    
    const exec = allocator.create(PetriNetExecutor) catch return 0;
    exec.* = PetriNetExecutor.init(allocator, net, strategy_enum) catch {
        allocator.destroy(exec);
        return 0;
    };
    
    const id = next_executor_id;
    next_executor_id += 1;
    
    executor_registry.put(id, exec) catch {
        exec.deinit();
        allocator.destroy(exec);
        return 0;
    };
    
    return id;
}

/// Destroy an executor
export fn nworkflow_destroy_executor(executor_id: u64) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    if (executor_registry.fetchRemove(executor_id)) |entry| {
        var exec = entry.value;
        exec.deinit();
        allocator.destroy(exec);
        return ErrorCode.success;
    }
    
    return ErrorCode.invalid_id;
}

/// Execute one step
export fn nworkflow_executor_step(executor_id: u64) bool {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const exec = executor_registry.get(executor_id) orelse return false;
    return exec.step() catch false;
}

/// Run for maximum steps
export fn nworkflow_executor_run(executor_id: u64, max_steps: usize) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const exec = executor_registry.get(executor_id) orelse return ErrorCode.invalid_id;
    exec.run(max_steps) catch return ErrorCode.unknown;
    return ErrorCode.success;
}

/// Run until complete
export fn nworkflow_executor_run_until_complete(executor_id: u64) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const exec = executor_registry.get(executor_id) orelse return ErrorCode.invalid_id;
    exec.runUntilComplete() catch return ErrorCode.unknown;
    return ErrorCode.success;
}

/// Set conflict resolution strategy
export fn nworkflow_executor_set_conflict_resolution(
    executor_id: u64,
    resolution: u32,  // 0=priority, 1=random, 2=round_robin, 3=weighted_random
) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const exec = executor_registry.get(executor_id) orelse return ErrorCode.invalid_id;
    
    const resolution_enum: ConflictResolution = switch (resolution) {
        0 => .priority,
        1 => .random,
        2 => .round_robin,
        3 => .weighted_random,
        else => return ErrorCode.invalid_parameter,
    };
    
    exec.setConflictResolution(resolution_enum);
    return ErrorCode.success;
}

/// Get execution statistics as JSON
export fn nworkflow_executor_get_stats_json(
    executor_id: u64,
    buffer: [*]u8,
    buffer_len: usize,
) ErrorCode {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    
    const exec = executor_registry.get(executor_id) orelse return ErrorCode.invalid_id;
    
    const json = exec.exportMetrics(allocator) catch return ErrorCode.allocation_failed;
    defer allocator.free(json);
    
    const copy_len = @min(json.len, buffer_len - 1);
    @memcpy(buffer[0..copy_len], json[0..copy_len]);
    buffer[copy_len] = 0; // Null terminate
    
    return ErrorCode.success;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Get the version of the nWorkflow library
export fn nworkflow_get_version(buffer: [*]u8, buffer_len: usize) ErrorCode {
    const version = "1.0.0-alpha";
    const copy_len = @min(version.len, buffer_len - 1);
    @memcpy(buffer[0..copy_len], version[0..copy_len]);
    buffer[copy_len] = 0;
    return ErrorCode.success;
}

/// Free a string allocated by the C API
export fn nworkflow_free_string(str: [*:0]u8) void {
    const slice = std.mem.span(str);
    allocator.free(slice);
}
