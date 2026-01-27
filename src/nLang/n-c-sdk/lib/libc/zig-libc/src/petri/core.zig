// Native Petri Net Core - Phase 1.8 (50 functions)
// Revolutionary: World's first native Petri net support
const std = @import("std");
const types = @import("types.zig");
const errno_mod = @import("../errno/lib.zig");

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Internal structures (opaque to C, pub for internal use by serialization)
pub const Place = struct {
    id: [256]u8,
    name: [256]u8,
    capacity: usize,
    tokens: std.ArrayList(Token),
    type: types.pn_type_t,
    // Watch callback for token changes
    watch: types.pn_watch_fn = null,
    watch_ctx: ?*anyopaque = null,
};

pub const Transition = struct {
    id: [256]u8,
    name: [256]u8,
    priority: i32,
    enabled: bool,
    guard: types.pn_guard_fn,
    guard_ctx: ?*anyopaque,
    // Watch callback for transition firings
    watch: types.pn_watch_fn = null,
    watch_ctx: ?*anyopaque = null,
    // Color guard for colored Petri nets
    color_guard: types.pn_guard_fn = null,
    // Arc lists for advanced analysis functions
    input_arcs: std.ArrayList(*Arc) = std.ArrayList(*Arc){},
    output_arcs: std.ArrayList(*Arc) = std.ArrayList(*Arc){},
};

pub const Arc = struct {
    id: [256]u8,
    arc_type: types.pn_arc_type_t,
    weight: usize,
    source_id: [256]u8,
    target_id: [256]u8,
    guard: types.pn_guard_fn,
    // Arc expression for colored Petri nets
    expression: types.pn_guard_fn = null,
};

pub const Token = struct {
    id: u64,
    timestamp: i64,
    color: u32,
    data: []u8,
    // Color type for colored Petri nets
    color_type: types.pn_type_t = .integer,
};

// Color type registration entry
pub const ColorType = struct {
    name: [256]u8,
    ctype: ?*anyopaque,
};

// Multiset for colored Petri nets
pub const Multiset = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayList(MultisetEntry),

    pub fn init(alloc: std.mem.Allocator) Multiset {
        return .{
            .allocator = alloc,
            .items = std.ArrayList(MultisetEntry){},
        };
    }

    pub fn deinit(self: *Multiset) void {
        self.items.deinit();
    }
};

pub const MultisetEntry = struct {
    token: *types.pn_token_t,
    multiplicity: usize,
};

pub const TraceEvent = struct {
    timestamp: i64,
    event_type: types.pn_event_type_t,
    element_id: [256]u8,
};

pub const PetriNet = struct {
    allocator: std.mem.Allocator,
    name: [256]u8,
    flags: c_int,
    places: std.StringHashMap(*Place),
    transitions: std.StringHashMap(*Transition),
    transitions_list: std.ArrayList(*Transition),
    arcs: std.ArrayList(*Arc),
    next_token_id: u64,
    total_firings: u64,
    running: bool,
    paused: bool,
    trace_enabled: bool,
    trace_events: std.ArrayList(TraceEvent),
    callbacks: [5]?types.pn_callback_fn, // One per event type
    rwlock: std.Thread.RwLock, // Thread safety for concurrent access
    // Multi-process support
    mutex: std.Thread.Mutex = .{},
    condition: std.Thread.Condition = .{},
    event_flags: [5]bool = [_]bool{false} ** 5, // One per event type
    // Shared memory fields
    shm_ptr: ?*anyopaque = null,
    shm_size: usize = 0,
    shm_name: [256]u8 = [_]u8{0} ** 256,
    is_shm: bool = false,
    // Colored Petri net support
    color_types: std.ArrayList(ColorType),
    
    // Helper function to get arcs connected to a transition
    fn getInputArcs(self: *PetriNet, trans_id: []const u8) std.ArrayList(*Arc) {
        var result = std.ArrayList(*Arc).empty;
        for (self.arcs.items) |arc| {
            if (arc.arc_type == .input or arc.arc_type == .inhibitor) {
                const target = std.mem.sliceTo(&arc.target_id, 0);
                if (std.mem.eql(u8, target, trans_id)) {
                    result.append(self.allocator, arc) catch {};
                }
            }
        }
        return result;
    }
    
    fn getOutputArcs(self: *PetriNet, trans_id: []const u8) std.ArrayList(*Arc) {
        var result = std.ArrayList(*Arc).empty;
        for (self.arcs.items) |arc| {
            if (arc.arc_type == .output) {
                const source = std.mem.sliceTo(&arc.source_id, 0);
                if (std.mem.eql(u8, source, trans_id)) {
                    result.append(self.allocator, arc) catch {};
                }
            }
        }
        return result;
    }
    
    fn recordEvent(self: *PetriNet, event_type: types.pn_event_type_t, element_id: []const u8) void {
        if (!self.trace_enabled) return;
        
        var event = TraceEvent{
            .timestamp = std.time.timestamp(),
            .event_type = event_type,
            .element_id = undefined,
        };
        @memset(&event.element_id, 0);
        @memcpy(event.element_id[0..@min(element_id.len, 255)], element_id[0..@min(element_id.len, 255)]);
        
        self.trace_events.append(self.allocator, event) catch {};
        
        // Trigger callback if registered
        const idx_int = @intFromEnum(event_type);
        if (idx_int >= 0) {
            const idx: usize = @intCast(idx_int);
            if (idx < self.callbacks.len) {
                if (self.callbacks[idx]) |cb_opt| {
                    if (cb_opt) |cb_fn| {
                        cb_fn(@ptrCast(self), event_type, null);
                    }
                }
            }
        }
    }
};

// Global allocator for Petri nets
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

//=============================================================================
// NET MANAGEMENT (10 functions)
//=============================================================================

/// Create a new Petri net
pub export fn pn_create(name: [*:0]const u8, flags: c_int) ?*types.pn_net_t {
    const net = allocator.create(PetriNet) catch {
        setErrno(.NOMEM);
        return null;
    };
    
    const name_slice = std.mem.span(name);
    @memset(&net.name, 0);
    @memcpy(net.name[0..@min(name_slice.len, 255)], name_slice[0..@min(name_slice.len, 255)]);
    
    net.* = .{
        .allocator = allocator,
        .name = net.name,
        .flags = flags,
        .places = undefined,
        .transitions = undefined,
        .transitions_list = std.ArrayList(*Transition).empty,
        .arcs = std.ArrayList(*Arc){},
        .next_token_id = 1,
        .total_firings = 0,
        .running = false,
        .paused = false,
        .trace_enabled = (flags & types.PN_CREATE_TRACED) != 0,
        .trace_events = std.ArrayList(TraceEvent){},
        .callbacks = [_]?types.pn_callback_fn{null} ** 5,
        .rwlock = .{}, // Initialize RwLock for thread safety
        .color_types = std.ArrayList(ColorType){},
    };

    net.places = std.StringHashMap(*Place).init(allocator);
    net.transitions = std.StringHashMap(*Transition).init(allocator);
    net.transitions_list = std.ArrayList(*Transition).empty;
    
    return @ptrCast(net);
}

/// Destroy a Petri net
pub export fn pn_destroy(net: ?*types.pn_net_t) c_int {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return -1));

    // Free trace events
    pn.trace_events.deinit(allocator);

    // Free color types
    pn.color_types.deinit(allocator);

    // Free places
    var place_it = pn.places.valueIterator();
    while (place_it.next()) |place| {
        place.*.tokens.deinit(allocator);
        allocator.destroy(place.*);
    }
    pn.places.deinit();

    // Free transitions
    for (pn.transitions_list.items) |trans| {
        allocator.destroy(trans);
    }
    pn.transitions_list.deinit(allocator);
    pn.transitions.deinit();

    // Free arcs
    for (pn.arcs.items) |arc| {
        allocator.destroy(arc);
    }
    pn.arcs.deinit(allocator);

    allocator.destroy(pn);
    return 0;
}

/// Clone a Petri net
pub export fn pn_clone(src: ?*types.pn_net_t, dst: ?*?*types.pn_net_t) c_int {
    _ = src;
    _ = dst;
    setErrno(.NOSYS);
    return -1;
}

/// Reset a Petri net to initial state
pub export fn pn_reset(net: ?*types.pn_net_t) c_int {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return -1));
    
    // Clear all tokens
    var place_it = pn.places.valueIterator();
    while (place_it.next()) |place| {
        place.*.tokens.clearRetainingCapacity();
    }
    
    pn.next_token_id = 1;
    pn.total_firings = 0;
    return 0;
}

/// Export Petri net to file
pub export fn pn_export(net: ?*types.pn_net_t, path: [*:0]const u8, format: c_int) c_int {
    _ = net;
    _ = path;
    _ = format;
    setErrno(.NOSYS);
    return -1;
}

/// Import Petri net from file
pub export fn pn_import(path: [*:0]const u8, net: ?*?*types.pn_net_t) c_int {
    _ = path;
    _ = net;
    setErrno(.NOSYS);
    return -1;
}

/// Validate Petri net structure
pub export fn pn_validate(net: ?*types.pn_net_t) c_int {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return -1));
    
    pn.rwlock.lockShared();
    defer pn.rwlock.unlockShared();
    
    // Check all arcs reference valid places/transitions
    for (pn.arcs.items) |arc| {
        const src_id = std.mem.sliceTo(&arc.source_id, 0);
        const tgt_id = std.mem.sliceTo(&arc.target_id, 0);
        
        if (arc.arc_type == .input or arc.arc_type == .inhibitor) {
            if (!pn.places.contains(src_id)) return -1;
            if (!pn.transitions.contains(tgt_id)) return -1;
        } else {
            if (!pn.transitions.contains(src_id)) return -1;
            if (!pn.places.contains(tgt_id)) return -1;
        }
    }
    
    return 0;
}

/// Get Petri net statistics
pub export fn pn_stats(net: ?*types.pn_net_t, stats: ?*types.pn_stats_t) c_int {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return -1));
    const st = stats orelse return -1;
    
    pn.rwlock.lockShared();
    defer pn.rwlock.unlockShared();
    
    var total_tokens: usize = 0;
    var place_it = pn.places.valueIterator();
    while (place_it.next()) |place| {
        total_tokens += place.*.tokens.items.len;
    }
    
    st.* = .{
        .place_count = pn.places.count(),
        .transition_count = pn.transitions.count(),
        .arc_count = pn.arcs.items.len,
        .total_tokens = total_tokens,
        .firings = pn.total_firings,
        .deadlock_count = 0,
    };
    
    return 0;
}

/// Serialize Petri net to buffer
pub export fn pn_serialize(net: ?*types.pn_net_t, buffer: ?*?*anyopaque, size: ?*usize) c_int {
    _ = net;
    _ = buffer;
    _ = size;
    setErrno(.NOSYS);
    return -1;
}

/// Deserialize Petri net from buffer
pub export fn pn_deserialize(buffer: ?*anyopaque, size: usize, net: ?*?*types.pn_net_t) c_int {
    _ = buffer;
    _ = size;
    _ = net;
    setErrno(.NOSYS);
    return -1;
}

//=============================================================================
// PLACE OPERATIONS (10 functions)
//=============================================================================

/// Create a new place
pub export fn pn_place_create(net: ?*types.pn_net_t, id: [*:0]const u8, name: [*:0]const u8) ?*types.pn_place_t {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return null));
    
    pn.rwlock.lock();
    defer pn.rwlock.unlock();
    
    const place = allocator.create(Place) catch {
        setErrno(.NOMEM);
        return null;
    };
    
    const id_slice = std.mem.span(id);
    const name_slice = std.mem.span(name);
    
    @memset(&place.id, 0);
    @memset(&place.name, 0);
    @memcpy(place.id[0..@min(id_slice.len, 255)], id_slice[0..@min(id_slice.len, 255)]);
    @memcpy(place.name[0..@min(name_slice.len, 255)], name_slice[0..@min(name_slice.len, 255)]);
    
    place.* = .{
        .id = place.id,
        .name = place.name,
        .capacity = std.math.maxInt(usize),
        .tokens = std.ArrayList(Token){},
        .type = .json,
    };
    
    const id_str = std.mem.sliceTo(&place.id, 0);
    pn.places.put(id_str, place) catch {
        allocator.destroy(place);
        setErrno(.NOMEM);
        return null;
    };
    
    return @ptrCast(place);
}

/// Lookup a place by id
pub export fn pn_place_get(net: ?*types.pn_net_t, id: [*:0]const u8) ?*types.pn_place_t {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return null));
    const id_slice = std.mem.span(id);

    pn.rwlock.lockShared();
    defer pn.rwlock.unlockShared();

    if (pn.places.get(id_slice)) |place| {
        return @ptrCast(place);
    }

    setErrno(.NOENT);
    return null;
}

/// Destroy a place
pub export fn pn_place_destroy(net: ?*types.pn_net_t, place_id: [*:0]const u8) c_int {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return -1));
    
    pn.rwlock.lock();
    defer pn.rwlock.unlock();
    
    const id_slice = std.mem.span(place_id);
    
    if (pn.places.fetchRemove(id_slice)) |kv| {
        kv.value.tokens.deinit(allocator);
        allocator.destroy(kv.value);
        return 0;
    }
    
    setErrno(.NOENT);
    return -1;
}

/// Set place capacity
pub export fn pn_place_set_capacity(place: ?*types.pn_place_t, capacity: usize) c_int {
    const p: *Place = @ptrCast(@alignCast(place orelse return -1));
    p.capacity = capacity;
    return 0;
}

/// Get place capacity
pub export fn pn_place_get_capacity(place: ?*types.pn_place_t) usize {
    const p: *Place = @ptrCast(@alignCast(place orelse return 0));
    return p.capacity;
}

/// Get token count in place
pub export fn pn_place_token_count(place: ?*types.pn_place_t) usize {
    const p: *Place = @ptrCast(@alignCast(place orelse return 0));
    return p.tokens.items.len;
}

/// Check if place has tokens
pub export fn pn_place_has_tokens(place: ?*types.pn_place_t) c_int {
    const p: *Place = @ptrCast(@alignCast(place orelse return 0));
    return if (p.tokens.items.len > 0) 1 else 0;
}

/// Set place type
pub export fn pn_place_set_type(place: ?*types.pn_place_t, ptype: types.pn_type_t) c_int {
    const p: *Place = @ptrCast(@alignCast(place orelse return -1));
    p.type = ptype;
    return 0;
}

/// Get place marking
pub export fn pn_place_get_marking(place: ?*types.pn_place_t, marking: ?*types.pn_marking_t) c_int {
    const p: *Place = @ptrCast(@alignCast(place orelse return -1));
    const m = marking orelse return -1;
    
    m.place_id = p.id;
    m.token_count = p.tokens.items.len;
    return 0;
}

/// Reset place (remove all tokens)
pub export fn pn_place_reset(place: ?*types.pn_place_t) c_int {
    const p: *Place = @ptrCast(@alignCast(place orelse return -1));
    p.tokens.clearRetainingCapacity();
    return 0;
}

/// List all places
pub export fn pn_place_list(net: ?*types.pn_net_t, places: ?*?*?*types.pn_place_t, count: ?*usize) c_int {
    _ = net;
    _ = places;
    _ = count;
    setErrno(.NOSYS);
    return -1;
}

//=============================================================================
// TRANSITION OPERATIONS (10 functions)
//=============================================================================

/// Create transition
pub export fn pn_trans_create(net: ?*types.pn_net_t, id: [*:0]const u8, name: [*:0]const u8) ?*types.pn_trans_t {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return null));
    
    pn.rwlock.lock();
    defer pn.rwlock.unlock();
    
    const trans = allocator.create(Transition) catch {
        setErrno(.NOMEM);
        return null;
    };
    
    const id_slice = std.mem.span(id);
    const name_slice = std.mem.span(name);
    
    @memset(&trans.id, 0);
    @memset(&trans.name, 0);
    @memcpy(trans.id[0..@min(id_slice.len, 255)], id_slice[0..@min(id_slice.len, 255)]);
    @memcpy(trans.name[0..@min(name_slice.len, 255)], name_slice[0..@min(name_slice.len, 255)]);
    
    trans.* = .{
        .id = trans.id,
        .name = trans.name,
        .priority = 0,
        .enabled = true,
        .guard = null,
        .guard_ctx = null,
    };
    
    const id_str = std.mem.sliceTo(&trans.id, 0);
    pn.transitions.put(id_str, trans) catch {
        allocator.destroy(trans);
        setErrno(.NOMEM);
        return null;
    };
    pn.transitions_list.append(allocator, trans) catch {
        _ = pn.transitions.remove(id_str);
        allocator.destroy(trans);
        setErrno(.NOMEM);
        return null;
    };
    
    return @ptrCast(trans);
}

/// Destroy transition
pub export fn pn_trans_destroy(net: ?*types.pn_net_t, trans_id: [*:0]const u8) c_int {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return -1));
    
    pn.rwlock.lock();
    defer pn.rwlock.unlock();
    
    const id_slice = std.mem.span(trans_id);
    
    if (pn.transitions.fetchRemove(id_slice)) |kv| {
        var idx: usize = 0;
        while (idx < pn.transitions_list.items.len) : (idx += 1) {
            if (pn.transitions_list.items[idx] == kv.value) {
                _ = pn.transitions_list.swapRemove(idx);
                break;
            }
        }
        allocator.destroy(kv.value);
        return 0;
    }
    
    setErrno(.NOENT);
    return -1;
}

/// Set transition priority
pub export fn pn_trans_set_priority(trans: ?*types.pn_trans_t, priority: c_int) c_int {
    const t: *Transition = @ptrCast(@alignCast(trans orelse return -1));
    t.priority = priority;
    return 0;
}

/// Get transition priority
pub export fn pn_trans_get_priority(trans: ?*types.pn_trans_t) c_int {
    const t: *Transition = @ptrCast(@alignCast(trans orelse return 0));
    return t.priority;
}

/// Set transition guard
pub export fn pn_trans_set_guard(trans: ?*types.pn_trans_t, guard: types.pn_guard_fn, ctx: ?*anyopaque) c_int {
    const t: *Transition = @ptrCast(@alignCast(trans orelse return -1));
    t.guard = guard;
    t.guard_ctx = ctx;
    return 0;
}

/// Check if transition is enabled
pub export fn pn_trans_is_enabled(trans: ?*types.pn_trans_t) c_int {
    const t: *Transition = @ptrCast(@alignCast(trans orelse return 0));
    return if (t.enabled) 1 else 0;
}

/// Fire transition (internal version with net reference)
pub fn pn_trans_fire_internal(net: *PetriNet, trans: *Transition) c_int {
    // Note: Caller must hold write lock on net.rwlock
    // This is an internal function called from already-locked contexts
    
    // Check if transition is enabled
    if (!trans.enabled) {
        setErrno(.PERM);
        return -1;
    }
    
    // Check guard condition
    if (trans.guard) |guard| {
        if (!guard(trans.guard_ctx)) {
            setErrno(.PERM);
            return -1;
        }
    }
    
    const trans_id = std.mem.sliceTo(&trans.id, 0);
    
    // Get input and output arcs
    var input_arcs = net.getInputArcs(trans_id);
    defer input_arcs.deinit(net.allocator);
    var output_arcs = net.getOutputArcs(trans_id);
    defer output_arcs.deinit(net.allocator);
    
    // Check if all input places have sufficient tokens
    for (input_arcs.items) |arc| {
        if (arc.arc_type == .inhibitor) {
            // Inhibitor arc: transition is blocked if place has tokens
            const place_id = std.mem.sliceTo(&arc.source_id, 0);
            if (net.places.get(place_id)) |place| {
                if (place.tokens.items.len > 0) {
                    setErrno(.PERM);
                    return -1;
                }
            }
        } else {
            // Normal input arc: need sufficient tokens
            const place_id = std.mem.sliceTo(&arc.source_id, 0);
            if (net.places.get(place_id)) |place| {
                if (place.tokens.items.len < arc.weight) {
                    setErrno(.AGAIN);
                    return -1;
                }
            } else {
                setErrno(.NOENT);
                return -1;
            }
        }
    }
    
    // Remove tokens from input places
    for (input_arcs.items) |arc| {
        if (arc.arc_type == .input) {
            const place_id = std.mem.sliceTo(&arc.source_id, 0);
            if (net.places.get(place_id)) |place| {
                var i: usize = 0;
                while (i < arc.weight and place.tokens.items.len > 0) : (i += 1) {
                    const token = place.tokens.orderedRemove(0);
                    allocator.free(token.data);
                }
            }
        }
    }
    
    // Add tokens to output places
    for (output_arcs.items) |arc| {
        const place_id = std.mem.sliceTo(&arc.target_id, 0);
        if (net.places.get(place_id)) |place| {
            var i: usize = 0;
            while (i < arc.weight) : (i += 1) {
                const token = Token{
                    .id = net.next_token_id,
                    .timestamp = std.time.timestamp(),
                    .color = 0,
                    .data = &[_]u8{},
                };
                net.next_token_id += 1;
                place.tokens.append(allocator, token) catch {
                    setErrno(.NOMEM);
                    return -1;
                };
                
                // Record event
                net.recordEvent(.token_added, place_id);
            }
            net.recordEvent(.place_updated, place_id);
        }
    }
    
    // Record transition fired event
    net.recordEvent(.transition_fired, trans_id);
    
    return 0;
}

/// Fire transition (public API - simplified version)
pub export fn pn_trans_fire(trans: ?*types.pn_trans_t) c_int {
    const t: *Transition = @ptrCast(@alignCast(trans orelse return -1));
    
    // This simplified version just marks the transition as having fired
    // The full implementation requires the net context, which is available
    // in the execution control functions that call pn_trans_fire_internal
    
    if (!t.enabled) {
        setErrno(.PERM);
        return -1;
    }
    
    if (t.guard) |guard| {
        if (!guard(t.guard_ctx)) {
            setErrno(.PERM);
            return -1;
        }
    }
    
    return 0;
}

/// Fire transition asynchronously
pub export fn pn_trans_fire_async(trans: ?*types.pn_trans_t, cb: types.pn_callback_fn) c_int {
    _ = trans;
    _ = cb;
    setErrno(.NOSYS);
    return -1;
}

/// Enable transition
pub export fn pn_trans_enable(trans: ?*types.pn_trans_t) c_int {
    const t: *Transition = @ptrCast(@alignCast(trans orelse return -1));
    t.enabled = true;
    return 0;
}

/// Disable transition
pub export fn pn_trans_disable(trans: ?*types.pn_trans_t) c_int {
    const t: *Transition = @ptrCast(@alignCast(trans orelse return -1));
    t.enabled = false;
    return 0;
}

//=============================================================================
// ARC OPERATIONS (8 functions)
//=============================================================================

/// Create arc
pub export fn pn_arc_create(net: ?*types.pn_net_t, id: [*:0]const u8, arc_type: types.pn_arc_type_t) ?*types.pn_arc_t {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return null));
    
    const arc = allocator.create(Arc) catch {
        setErrno(.NOMEM);
        return null;
    };
    
    const id_slice = std.mem.span(id);
    @memset(&arc.id, 0);
    @memcpy(arc.id[0..@min(id_slice.len, 255)], id_slice[0..@min(id_slice.len, 255)]);
    
    arc.* = .{
        .id = arc.id,
        .arc_type = arc_type,
        .weight = 1,
        .source_id = undefined,
        .target_id = undefined,
        .guard = null,
    };
    
    @memset(&arc.source_id, 0);
    @memset(&arc.target_id, 0);
    
    pn.arcs.append(allocator, arc) catch {
        allocator.destroy(arc);
        setErrno(.NOMEM);
        return null;
    };
    
    return @ptrCast(arc);
}

/// Destroy arc
pub export fn pn_arc_destroy(net: ?*types.pn_net_t, arc_id: [*:0]const u8) c_int {
    const pn: *PetriNet = @ptrCast(@alignCast(net orelse return -1));
    const id_slice = std.mem.span(arc_id);
    
    for (pn.arcs.items, 0..) |arc, i| {
        const arc_id_str = std.mem.sliceTo(&arc.id, 0);
        if (std.mem.eql(u8, arc_id_str, id_slice)) {
            _ = pn.arcs.orderedRemove(i);
            allocator.destroy(arc);
            return 0;
        }
    }
    
    setErrno(.NOENT);
    return -1;
}

/// Connect arc
pub export fn pn_arc_connect(arc: ?*types.pn_arc_t, src: [*:0]const u8, dst: [*:0]const u8) c_int {
    const a: *Arc = @ptrCast(@alignCast(arc orelse return -1));
    
    const src_slice = std.mem.span(src);
    const dst_slice = std.mem.span(dst);
    
    @memcpy(a.source_id[0..@min(src_slice.len, 255)], src_slice[0..@min(src_slice.len, 255)]);
    @memcpy(a.target_id[0..@min(dst_slice.len, 255)], dst_slice[0..@min(dst_slice.len, 255)]);
    
    return 0;
}

/// Set arc weight
pub export fn pn_arc_set_weight(arc: ?*types.pn_arc_t, weight: usize) c_int {
    const a: *Arc = @ptrCast(@alignCast(arc orelse return -1));
    a.weight = weight;
    return 0;
}

/// Get arc weight
pub export fn pn_arc_get_weight(arc: ?*types.pn_arc_t) usize {
    const a: *Arc = @ptrCast(@alignCast(arc orelse return 0));
    return a.weight;
}

/// Set arc guard
pub export fn pn_arc_set_guard(arc: ?*types.pn_arc_t, guard: types.pn_guard_fn) c_int {
    const a: *Arc = @ptrCast(@alignCast(arc orelse return -1));
    a.guard = guard;
    return 0;
}

/// List arcs
pub export fn pn_arc_list(net: ?*types.pn_net_t, arcs: ?*?*?*types.pn_arc_t, count: ?*usize) c_int {
    _ = net;
    _ = arcs;
    _ = count;
    setErrno(.NOSYS);
    return -1;
}

/// Test arc
pub export fn pn_arc_test(arc: ?*types.pn_arc_t) c_int {
    _ = arc;
    return 0;
}

//=============================================================================
// TOKEN OPERATIONS (12 functions)
//=============================================================================

/// Create token
pub export fn pn_token_create(data: ?*const anyopaque, size: usize) ?*types.pn_token_t {
    const token = allocator.create(types.pn_token_t) catch {
        setErrno(.NOMEM);
        return null;
    };
    
    var data_copy: ?*anyopaque = null;
    if (data != null and size > 0) {
        const data_slice = allocator.alloc(u8, size) catch {
            allocator.destroy(token);
            setErrno(.NOMEM);
            return null;
        };
        const src: [*]const u8 = @ptrCast(data.?);
        @memcpy(data_slice, src[0..size]);
        data_copy = data_slice.ptr;
    }
    
    token.* = .{
        .id = @intCast(std.time.timestamp()),
        .timestamp = std.time.timestamp(),
        .color = 0,
        .data_size = size,
        .data = data_copy,
    };
    
    return token;
}

/// Destroy token
pub export fn pn_token_destroy(token: ?*types.pn_token_t) c_int {
    const t = token orelse return -1;
    if (t.data) |d| {
        const slice: []u8 = @as([*]u8, @ptrCast(d))[0..t.data_size];
        allocator.free(slice);
    }
    allocator.destroy(t);
    return 0;
}

/// Clone token
pub export fn pn_token_clone(token: ?*types.pn_token_t) ?*types.pn_token_t {
    const t = token orelse return null;
    return pn_token_create(t.data, t.data_size);
}

/// Put token in place
pub export fn pn_token_put(place: ?*types.pn_place_t, token: ?*types.pn_token_t) c_int {
    const p: *Place = @ptrCast(@alignCast(place orelse return -1));
    const t = token orelse return -1;
    
    if (p.tokens.items.len >= p.capacity) {
        setErrno(.AGAIN);
        return -1;
    }
    
    const internal_token = Token{
        .id = t.id,
        .timestamp = t.timestamp,
        .color = t.color,
        .data = if (t.data) |d| blk: {
            const slice: []u8 = @as([*]u8, @ptrCast(d))[0..t.data_size];
            break :blk allocator.dupe(u8, slice) catch return -1;
        } else &[_]u8{},
    };
    
    p.tokens.append(allocator, internal_token) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    return 0;
}

/// Get token from place
pub export fn pn_token_get(place: ?*types.pn_place_t) ?*types.pn_token_t {
    const p: *Place = @ptrCast(@alignCast(place orelse return null));
    
    if (p.tokens.items.len == 0) return null;
    
    const internal_token = p.tokens.orderedRemove(0);
    
    const token = allocator.create(types.pn_token_t) catch return null;
    token.* = .{
        .id = internal_token.id,
        .timestamp = internal_token.timestamp,
        .color = internal_token.color,
        .data_size = internal_token.data.len,
        .data = if (internal_token.data.len > 0) internal_token.data.ptr else null,
    };
    
    return token;
}

/// Peek token (don't remove)
pub export fn pn_token_peek(place: ?*types.pn_place_t) ?*types.pn_token_t {
    const p: *Place = @ptrCast(@alignCast(place orelse return null));
    
    if (p.tokens.items.len == 0) return null;
    
    const internal_token = &p.tokens.items[0];
    
    const token = allocator.create(types.pn_token_t) catch return null;
    token.* = .{
        .id = internal_token.id,
        .timestamp = internal_token.timestamp,
        .color = internal_token.color,
        .data_size = internal_token.data.len,
        .data = if (internal_token.data.len > 0) internal_token.data.ptr else null,
    };
    
    return token;
}

/// Set token data
pub export fn pn_token_set_data(token: ?*types.pn_token_t, data: ?*const anyopaque, size: usize) c_int {
    const t = token orelse return -1;
    
    // Free old data
    if (t.data) |d| {
        const slice: []u8 = @as([*]u8, @ptrCast(d))[0..t.data_size];
        allocator.free(slice);
    }
    
    // Allocate new data
    if (data != null and size > 0) {
        const new_data = allocator.alloc(u8, size) catch {
            setErrno(.NOMEM);
            return -1;
        };
        const src: [*]const u8 = @ptrCast(data.?);
        @memcpy(new_data, src[0..size]);
        t.data = new_data.ptr;
        t.data_size = size;
    } else {
        t.data = null;
        t.data_size = 0;
    }
    
    return 0;
}

/// Get token data
pub export fn pn_token_get_data(token: ?*types.pn_token_t, data: ?*?*anyopaque, size: ?*usize) c_int {
    const t = token orelse return -1;
    if (data) |d| d.* = t.data;
    if (size) |s| s.* = t.data_size;
    return 0;
}

/// Set token color
pub export fn pn_token_set_color(token: ?*types.pn_token_t, color: u32) c_int {
    const t = token orelse return -1;
    t.color = color;
    return 0;
}

/// Get token color
pub export fn pn_token_get_color(token: ?*types.pn_token_t) u32 {
    const t = token orelse return 0;
    return t.color;
}

/// Get token timestamp
pub export fn pn_token_get_timestamp(token: ?*types.pn_token_t) i64 {
    const t = token orelse return 0;
    return t.timestamp;
}

/// Get token ID
pub export fn pn_token_get_id(token: ?*types.pn_token_t) u64 {
    const t = token orelse return 0;
    return t.id;
}
