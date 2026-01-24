//! Petri Net - zig-libc Integration Wrapper
//! 
//! This module provides a compatibility layer between nAgentFlow and
//! the production-grade zig-libc Petri net implementation.
//!
//! Migration: Phase 1 - Direct replacement with enhanced features
//! - Thread safety (RwLock)
//! - 100 comprehensive functions
//! - Serialization support
//! - Production-tested

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import zig-libc Petri net implementation via module dependency
const zig_libc = @import("zig_libc");
const petri_lib = zig_libc.petri;
const petri_core = petri_lib.core;
const petri_types = petri_lib.types;

// Re-export types for compatibility
pub const pn_net_t = petri_types.pn_net_t;
pub const pn_place_t = petri_types.pn_place_t;
pub const pn_trans_t = petri_types.pn_trans_t;
pub const pn_arc_t = petri_types.pn_arc_t;
pub const pn_token_t = petri_types.pn_token_t;

/// Token represents data flowing through the Petri Net
/// Compatibility wrapper for zig-libc token
pub const Token = struct {
    id: u64,
    data: []const u8,
    timestamp: i64,
    
    pub fn init(id: u64, data: []const u8) Token {
        return Token{
            .id = id,
            .data = data,
            .timestamp = std.time.timestamp(),
        };
    }
    
    pub fn deinit(self: *Token, allocator: Allocator) void {
        allocator.free(self.data);
    }
    
    pub fn clone(self: *const Token, allocator: Allocator) !Token {
        const data_copy = try allocator.dupe(u8, self.data);
        return Token{
            .id = self.id,
            .data = data_copy,
            .timestamp = self.timestamp,
        };
    }
};

/// Place wrapper - maintains API compatibility
pub const Place = struct {
    handle: *pn_place_t,
    id: []const u8,
    name: []const u8,
    
    pub fn tokenCount(self: *const Place) usize {
        return petri_core.pn_place_token_count(self.handle);
    }
    
    pub fn hasTokens(self: *const Place) bool {
        return petri_core.pn_place_has_tokens(self.handle) == 1;
    }
};

/// ArcType for compatibility
pub const ArcType = enum {
    input,
    output,
    inhibitor,
    
    fn toZigLibc(self: ArcType) petri_types.pn_arc_type_t {
        return switch (self) {
            .input => .input,
            .output => .output,
            .inhibitor => .inhibitor,
        };
    }
};

/// Arc wrapper
pub const Arc = struct {
    handle: *pn_arc_t,
    id: []const u8,
    arc_type: ArcType,
    weight: usize,
    source_id: []const u8,
    target_id: []const u8,
};

/// TransitionGuard compatibility wrapper
pub const TransitionGuard = struct {
    expression: []const u8,
    
    pub fn init(allocator: Allocator, expression: []const u8) !TransitionGuard {
        return TransitionGuard{
            .expression = try allocator.dupe(u8, expression),
        };
    }
    
    pub fn deinit(self: *TransitionGuard, allocator: Allocator) void {
        allocator.free(self.expression);
    }
    
    pub fn evaluate(self: *const TransitionGuard, tokens: []const Token) bool {
        if (std.mem.eql(u8, self.expression, "true")) return true;
        if (std.mem.eql(u8, self.expression, "has_tokens")) return tokens.len > 0;
        return false;
    }
};

/// Transition wrapper
pub const Transition = struct {
    handle: *pn_trans_t,
    id: []const u8,
    name: []const u8,
    guard: ?TransitionGuard,
    priority: i32,
    enabled: bool,
    
    pub fn setGuard(self: *Transition, guard: TransitionGuard) void {
        self.guard = guard;
        // TODO: Set guard on zig-libc transition if supported
    }
};

/// Marking (state) wrapper
pub const Marking = struct {
    place_tokens: std.StringHashMap(usize),
    
    pub fn init(allocator: Allocator) Marking {
        return Marking{
            .place_tokens = std.StringHashMap(usize).init(allocator),
        };
    }
    
    pub fn deinit(self: *Marking) void {
        self.place_tokens.deinit();
    }
    
    pub fn set(self: *Marking, place_id: []const u8, count: usize) !void {
        try self.place_tokens.put(place_id, count);
    }
    
    pub fn get(self: *const Marking, place_id: []const u8) usize {
        return self.place_tokens.get(place_id) orelse 0;
    }
    
    pub fn equals(self: *const Marking, other: *const Marking) bool {
        if (self.place_tokens.count() != other.place_tokens.count()) return false;
        
        var it = self.place_tokens.iterator();
        while (it.next()) |entry| {
            const other_count = other.place_tokens.get(entry.key_ptr.*) orelse return false;
            if (entry.value_ptr.* != other_count) return false;
        }
        return true;
    }
};

/// Main PetriNet wrapper - provides nAgentFlow API backed by zig-libc
pub const PetriNet = struct {
    allocator: Allocator,
    handle: *pn_net_t,
    name: []const u8,
    places_map: std.StringHashMap(*Place),
    transitions_map: std.StringHashMap(*Transition),
    arcs_list: std.ArrayList(*Arc),
    next_token_id: u64,
    
    pub fn init(allocator: Allocator, name: []const u8) !PetriNet {
        // Create zig-libc Petri net (thread-safe!)
        const name_z = try allocator.dupeZ(u8, name);
        defer allocator.free(name_z);
        
        const handle = petri_core.pn_create(name_z.ptr, 0) orelse return error.PetriNetCreationFailed;
        
        return PetriNet{
            .allocator = allocator,
            .handle = handle,
            .name = try allocator.dupe(u8, name),
            .places_map = std.StringHashMap(*Place).init(allocator),
            .transitions_map = std.StringHashMap(*Transition).init(allocator),
            .arcs_list = std.ArrayList(*Arc){},
            .next_token_id = 1,
        };
    }
    
    pub fn deinit(self: *PetriNet) void {
        // Clean up wrappers
        var place_it = self.places_map.valueIterator();
        while (place_it.next()) |place_ptr| {
            self.allocator.destroy(place_ptr.*);
        }
        self.places_map.deinit();
        
        var trans_it = self.transitions_map.valueIterator();
        while (trans_it.next()) |trans_ptr| {
            if (trans_ptr.*.guard) |*g| {
                g.deinit(self.allocator);
            }
            self.allocator.destroy(trans_ptr.*);
        }
        self.transitions_map.deinit();
        
        for (self.arcs_list.items) |arc| {
            self.allocator.destroy(arc);
        }
        self.arcs_list.deinit();
        
        self.allocator.free(self.name);
        
        // Destroy zig-libc net
        _ = petri_core.pn_destroy(self.handle);
    }
    
    /// Add a place (uses zig-libc backend)
    pub fn addPlace(self: *PetriNet, id: []const u8, name: []const u8, capacity: ?usize) !*Place {
        const id_z = try self.allocator.dupeZ(u8, id);
        defer self.allocator.free(id_z);
        const name_z = try self.allocator.dupeZ(u8, name);
        defer self.allocator.free(name_z);
        
        const handle = petri_core.pn_place_create(self.handle, id_z.ptr, name_z.ptr) orelse 
            return error.PlaceCreationFailed;
        
        if (capacity) |cap| {
            _ = petri_core.pn_place_set_capacity(handle, cap);
        }
        
        const place = try self.allocator.create(Place);
        place.* = Place{
            .handle = handle,
            .id = try self.allocator.dupe(u8, id),
            .name = try self.allocator.dupe(u8, name),
        };
        
        try self.places_map.put(place.id, place);
        return place;
    }
    
    /// Add a transition (uses zig-libc backend)
    pub fn addTransition(self: *PetriNet, id: []const u8, name: []const u8, priority: i32) !*Transition {
        const id_z = try self.allocator.dupeZ(u8, id);
        defer self.allocator.free(id_z);
        const name_z = try self.allocator.dupeZ(u8, name);
        defer self.allocator.free(name_z);
        
        const handle = petri_core.pn_trans_create(self.handle, id_z.ptr, name_z.ptr) orelse
            return error.TransitionCreationFailed;
        
        _ = petri_core.pn_trans_set_priority(handle, priority);
        
        const trans = try self.allocator.create(Transition);
        trans.* = Transition{
            .handle = handle,
            .id = try self.allocator.dupe(u8, id),
            .name = try self.allocator.dupe(u8, name),
            .guard = null,
            .priority = priority,
            .enabled = true,
        };
        
        try self.transitions_map.put(trans.id, trans);
        return trans;
    }
    
    /// Add an arc (uses zig-libc backend)
    pub fn addArc(self: *PetriNet, id: []const u8, arc_type: ArcType, weight: usize, source_id: []const u8, target_id: []const u8) !*Arc {
        const id_z = try self.allocator.dupeZ(u8, id);
        defer self.allocator.free(id_z);
        
        const handle = petri_core.pn_arc_create(self.handle, id_z.ptr, arc_type.toZigLibc()) orelse
            return error.ArcCreationFailed;
        
        _ = petri_core.pn_arc_set_weight(handle, weight);
        
        const source_z = try self.allocator.dupeZ(u8, source_id);
        defer self.allocator.free(source_z);
        const target_z = try self.allocator.dupeZ(u8, target_id);
        defer self.allocator.free(target_z);
        
        _ = petri_core.pn_arc_connect(handle, source_z.ptr, target_z.ptr);
        
        const arc = try self.allocator.create(Arc);
        arc.* = Arc{
            .handle = handle,
            .id = try self.allocator.dupe(u8, id),
            .arc_type = arc_type,
            .weight = weight,
            .source_id = try self.allocator.dupe(u8, source_id),
            .target_id = try self.allocator.dupe(u8, target_id),
        };
        
        try self.arcs_list.append(arc);
        return arc;
    }
    
    /// Add token to place (uses zig-libc backend)
    pub fn addTokenToPlace(self: *PetriNet, place_id: []const u8, data: []const u8) !void {
        const place_id_z = try self.allocator.dupeZ(u8, place_id);
        defer self.allocator.free(place_id_z);
        
        const place = petri_core.pn_place_get(self.handle, place_id_z.ptr) orelse
            return error.PlaceNotFound;
        
        const token = petri_core.pn_token_create(@ptrCast(data.ptr), data.len) orelse
            return error.TokenCreationFailed;
        
        _ = petri_core.pn_token_put(place, token);
        self.next_token_id += 1;
    }
    
    /// Get current marking
    pub fn getCurrentMarking(self: *const PetriNet) !Marking {
        var marking = Marking.init(self.allocator);
        
        var it = self.places_map.iterator();
        while (it.next()) |entry| {
            const count = entry.value_ptr.*.tokenCount();
            try marking.set(entry.key_ptr.*, count);
        }
        
        return marking;
    }
    
    /// Check if transition is enabled
    pub fn isTransitionEnabled(self: *const PetriNet, transition_id: []const u8) bool {
        const trans = self.transitions_map.get(transition_id) orelse return false;
        return petri_core.pn_trans_is_enabled(trans.handle) == 1;
    }
    
    /// Get enabled transitions
    pub fn getEnabledTransitions(self: *const PetriNet) !std.ArrayList([]const u8) {
        var enabled = std.ArrayList([]const u8).init(self.allocator);
        
        var it = self.transitions_map.iterator();
        while (it.next()) |entry| {
            if (self.isTransitionEnabled(entry.key_ptr.*)) {
                try enabled.append(entry.key_ptr.*);
            }
        }
        
        return enabled;
    }
    
    /// Fire a transition (uses zig-libc backend - THREAD-SAFE!)
    pub fn fireTransition(self: *PetriNet, transition_id: []const u8) !void {
        const trans = self.transitions_map.get(transition_id) orelse
            return error.TransitionNotFound;
        
        const result = petri_core.pn_trans_fire(trans.handle);
        if (result != 0) {
            return error.TransitionNotEnabled;
        }
    }
    
    /// Check for deadlock
    pub fn isDeadlocked(self: *const PetriNet) bool {
        return petri_core.pn_is_deadlocked(self.handle) == 1;
    }
    
    /// Get statistics
    pub fn getStats(self: *const PetriNet) PetriNetStats {
        var stats: petri_types.pn_stats_t = undefined;
        _ = petri_core.pn_stats(self.handle, &stats);
        
        return PetriNetStats{
            .place_count = stats.place_count,
            .transition_count = stats.transition_count,
            .arc_count = stats.arc_count,
            .total_tokens = stats.total_tokens,
        };
    }
};

pub const PetriNetStats = struct {
    place_count: usize,
    transition_count: usize,
    arc_count: usize,
    total_tokens: usize,
};

// Re-export for compatibility
pub const places = struct {};
pub const transitions = struct {};
pub const arcs = struct {};
