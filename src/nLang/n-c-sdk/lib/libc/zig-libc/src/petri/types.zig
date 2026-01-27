// Native Petri Net Types - Phase 1.8
// Revolutionary: First language with native Petri net support
const std = @import("std");

/// Opaque handle to Petri net
pub const pn_net_t = opaque {};

/// Opaque handle to place
pub const pn_place_t = opaque {};

/// Opaque handle to transition
pub const pn_trans_t = opaque {};

/// Opaque handle to arc
pub const pn_arc_t = opaque {};

/// Token structure
pub const pn_token_t = extern struct {
    id: u64,
    timestamp: i64,
    color: u32,
    data_size: usize,
    data: ?*anyopaque,
};

/// Arc types
pub const pn_arc_type_t = enum(c_int) {
    input = 0,      // Place -> Transition
    output = 1,     // Transition -> Place  
    inhibitor = 2,  // Place -o Transition
};

/// Execution modes
pub const pn_exec_mode_t = enum(c_int) {
    single_step = 0,
    continuous = 1,
    until_deadlock = 2,
    until_condition = 3,
};

/// Event types
pub const pn_event_type_t = enum(c_int) {
    place_updated = 0,
    transition_fired = 1,
    token_added = 2,
    token_removed = 3,
    net_deadlocked = 4,
};

/// Statistics
pub const pn_stats_t = extern struct {
    place_count: usize,
    transition_count: usize,
    arc_count: usize,
    total_tokens: usize,
    firings: u64,
    deadlock_count: u64,
};

/// Marking (state snapshot)
pub const pn_marking_t = extern struct {
    place_id: [256]u8,
    token_count: usize,
};

/// Metrics
pub const pn_metrics_t = extern struct {
    avg_tokens_per_place: f64,
    max_tokens_per_place: usize,
    transitions_enabled: usize,
    throughput: f64,
    cycle_time_ms: f64,
};

/// Creation flags
pub const PN_CREATE_SHARED: c_int = 0x01;
pub const PN_CREATE_PERSISTENT: c_int = 0x02;
pub const PN_CREATE_TRACED: c_int = 0x04;
pub const PN_CREATE_VERIFIED: c_int = 0x08;

/// Export formats
pub const PN_FORMAT_JSON: c_int = 0;
pub const PN_FORMAT_PNML: c_int = 1;
pub const PN_FORMAT_DOT: c_int = 2;
pub const PN_FORMAT_BPMN: c_int = 3;

/// Type definitions
pub const pn_type_t = enum(c_int) {
    integer = 0,
    string = 1,
    boolean = 2,
    json = 3,
    binary = 4,
};

/// Callback function types
pub const pn_callback_fn = ?*const fn (?*pn_net_t, pn_event_type_t, ?*anyopaque) callconv(.c) void;
pub const pn_guard_fn = ?*const fn (?*anyopaque) callconv(.c) bool;
pub const pn_condition_fn = ?*const fn (?*pn_net_t) callconv(.c) bool;
pub const pn_watch_fn = ?*const fn (?*anyopaque, ?*anyopaque) callconv(.c) void;
