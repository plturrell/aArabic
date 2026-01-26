// Mojo Runtime Core
// Memory allocation, reference counting, and runtime initialization
//
// This module provides the fundamental runtime support for compiled Mojo programs.
// It handles memory management through reference counting with cycle detection.

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Runtime Configuration
// ============================================================================

pub const RuntimeConfig = struct {
    /// Enable reference counting (default: true)
    enable_rc: bool = true,

    /// Enable cycle detection for RC (default: true)
    enable_cycle_detection: bool = true,

    /// Initial heap size in bytes (default: 64MB)
    initial_heap_size: usize = 64 * 1024 * 1024,

    /// Maximum heap size in bytes (default: 1GB)
    max_heap_size: usize = 1024 * 1024 * 1024,

    /// Enable runtime statistics (default: false in release)
    enable_stats: bool = builtin.mode == .Debug,

    /// Enable memory poisoning for debugging (default: false in release)
    enable_poisoning: bool = builtin.mode == .Debug,
};

pub var config: RuntimeConfig = .{};

// ============================================================================
// Runtime Statistics
// ============================================================================

pub const RuntimeStats = struct {
    /// Total allocations made
    total_allocations: u64 = 0,

    /// Total deallocations made
    total_deallocations: u64 = 0,

    /// Current live allocations
    live_allocations: u64 = 0,

    /// Total bytes allocated
    total_bytes_allocated: u64 = 0,

    /// Current bytes in use
    current_bytes_in_use: u64 = 0,

    /// Peak bytes in use
    peak_bytes_in_use: u64 = 0,

    /// Reference count increments
    rc_increments: u64 = 0,

    /// Reference count decrements
    rc_decrements: u64 = 0,

    /// Objects destroyed via RC
    rc_destructions: u64 = 0,

    /// Cycles detected and collected
    cycles_collected: u64 = 0,

    pub fn recordAllocation(self: *RuntimeStats, size: usize) void {
        self.total_allocations += 1;
        self.live_allocations += 1;
        self.total_bytes_allocated += size;
        self.current_bytes_in_use += size;
        if (self.current_bytes_in_use > self.peak_bytes_in_use) {
            self.peak_bytes_in_use = self.current_bytes_in_use;
        }
    }

    pub fn recordDeallocation(self: *RuntimeStats, size: usize) void {
        self.total_deallocations += 1;
        self.live_allocations -= 1;
        self.current_bytes_in_use -= size;
    }

    pub fn print(self: *const RuntimeStats) void {
        std.debug.print("\n=== Mojo Runtime Statistics ===\n", .{});
        std.debug.print("Allocations:     {d} total, {d} live\n", .{ self.total_allocations, self.live_allocations });
        std.debug.print("Bytes:           {d} total, {d} current, {d} peak\n", .{
            self.total_bytes_allocated, self.current_bytes_in_use, self.peak_bytes_in_use
        });
        std.debug.print("Reference Count: {d} inc, {d} dec, {d} destroyed\n", .{
            self.rc_increments, self.rc_decrements, self.rc_destructions
        });
        std.debug.print("Cycles:          {d} collected\n", .{self.cycles_collected});
        std.debug.print("===============================\n\n", .{});
    }
};

pub var stats: RuntimeStats = .{};

// ============================================================================
// Reference Counted Object Header
// ============================================================================

/// Header prepended to all reference-counted allocations
pub const RcHeader = struct {
    /// Reference count (atomic for thread safety)
    ref_count: std.atomic.Value(u32),

    /// Weak reference count
    weak_count: std.atomic.Value(u32),

    /// Size of the allocation (excluding header)
    size: u32,

    /// Type ID for runtime type information
    type_id: u32,

    /// Flags for GC and debugging
    flags: Flags,

    /// Destructor function pointer
    destructor: ?*const fn (*anyopaque) void,

    pub const Flags = packed struct {
        /// Object is part of a cycle detection scan
        in_cycle_scan: bool = false,

        /// Object has been marked as reachable
        marked: bool = false,

        /// Object should not be freed (static allocation)
        pinned: bool = false,

        /// Object contains pointers to other RC objects
        has_pointers: bool = false,

        /// Reserved for future use
        _reserved: u4 = 0,
    };

    pub fn init(size: u32, type_id: u32, destructor: ?*const fn (*anyopaque) void) RcHeader {
        return .{
            .ref_count = std.atomic.Value(u32).init(1),
            .weak_count = std.atomic.Value(u32).init(0),
            .size = size,
            .type_id = type_id,
            .flags = .{},
            .destructor = destructor,
        };
    }

    /// Increment reference count, returns new count
    pub fn retain(self: *RcHeader) u32 {
        const old = self.ref_count.fetchAdd(1, .seq_cst);
        if (config.enable_stats) {
            stats.rc_increments += 1;
        }
        return old + 1;
    }

    /// Decrement reference count, returns true if object should be freed
    pub fn release(self: *RcHeader) bool {
        const old = self.ref_count.fetchSub(1, .seq_cst);
        if (config.enable_stats) {
            stats.rc_decrements += 1;
        }

        if (old == 1) {
            // Reference count reached zero
            if (config.enable_stats) {
                stats.rc_destructions += 1;
            }
            return true;
        }
        return false;
    }

    /// Get current reference count
    pub fn getCount(self: *const RcHeader) u32 {
        return self.ref_count.load(.seq_cst);
    }

    /// Check if this is the only reference
    pub fn isUnique(self: *const RcHeader) bool {
        return self.ref_count.load(.seq_cst) == 1;
    }
};

// ============================================================================
// Memory Allocator
// ============================================================================

/// The runtime allocator - wraps the system allocator with tracking
pub const RuntimeAllocator = struct {
    backing: std.mem.Allocator,

    pub fn init(backing: std.mem.Allocator) RuntimeAllocator {
        return .{ .backing = backing };
    }

    /// Allocate raw memory
    pub fn alloc(self: *RuntimeAllocator, comptime T: type, n: usize) ![]T {
        const slice = try self.backing.alloc(T, n);

        if (config.enable_stats) {
            stats.recordAllocation(n * @sizeOf(T));
        }

        if (config.enable_poisoning) {
            @memset(std.mem.sliceAsBytes(slice), 0xAA);
        }

        return slice;
    }

    /// Free raw memory
    pub fn free(self: *RuntimeAllocator, comptime T: type, slice: []T) void {
        if (config.enable_stats) {
            stats.recordDeallocation(slice.len * @sizeOf(T));
        }

        if (config.enable_poisoning) {
            @memset(std.mem.sliceAsBytes(slice), 0xDD);
        }

        self.backing.free(slice);
    }

    /// Allocate a reference-counted object
    pub fn allocRc(self: *RuntimeAllocator, comptime T: type, type_id: u32, destructor: ?*const fn (*anyopaque) void) !*T {
        const header_size = @sizeOf(RcHeader);
        const obj_size = @sizeOf(T);
        const total_size = header_size + obj_size;

        // Allocate header + object together
        const raw = try self.backing.alloc(u8, total_size);

        if (config.enable_stats) {
            stats.recordAllocation(total_size);
        }

        // Initialize header
        const header: *RcHeader = @ptrCast(@alignCast(raw.ptr));
        header.* = RcHeader.init(@intCast(obj_size), type_id, destructor);

        // Return pointer to object (after header)
        const obj_ptr: *T = @ptrCast(@alignCast(raw.ptr + header_size));

        if (config.enable_poisoning) {
            @memset(std.mem.asBytes(obj_ptr), 0xAA);
        }

        return obj_ptr;
    }

    /// Get the RC header for an object
    pub fn getHeader(ptr: anytype) *RcHeader {
        const byte_ptr: [*]u8 = @ptrCast(ptr);
        const header_ptr = byte_ptr - @sizeOf(RcHeader);
        return @ptrCast(@alignCast(header_ptr));
    }

    /// Free a reference-counted object
    pub fn freeRc(self: *RuntimeAllocator, ptr: anytype) void {
        const header = getHeader(ptr);
        const total_size = @sizeOf(RcHeader) + header.size;

        // Call destructor if present
        if (header.destructor) |dtor| {
            dtor(ptr);
        }

        if (config.enable_stats) {
            stats.recordDeallocation(total_size);
        }

        // Get pointer to start of allocation (header)
        const raw_ptr: [*]u8 = @ptrCast(header);
        const slice = raw_ptr[0..total_size];

        if (config.enable_poisoning) {
            @memset(slice, 0xDD);
        }

        self.backing.free(slice);
    }

    /// Retain (increment ref count) an RC object
    pub fn retain(ptr: anytype) void {
        const header = getHeader(ptr);
        _ = header.retain();
    }

    /// Release (decrement ref count) an RC object, free if count reaches zero
    pub fn release(self: *RuntimeAllocator, ptr: anytype) void {
        const header = getHeader(ptr);
        if (header.release()) {
            self.freeRc(ptr);
        }
    }
};

// Global runtime allocator instance
var global_allocator: ?RuntimeAllocator = null;

/// Get the global runtime allocator
pub fn getAllocator() *RuntimeAllocator {
    if (global_allocator == null) {
        @panic("Mojo runtime not initialized. Call mojo_runtime_init() first.");
    }
    return &global_allocator.?;
}

// ============================================================================
// Cycle Detector (Mark & Sweep for RC cycles)
// ============================================================================

pub const CycleDetector = struct {
    /// Objects suspected of being in cycles
    suspects: std.ArrayListUnmanaged(*RcHeader),

    /// Allocator for internal structures
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CycleDetector {
        return .{
            .suspects = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CycleDetector) void {
        self.suspects.deinit(self.allocator);
    }

    /// Add an object to the suspect list
    pub fn addSuspect(self: *CycleDetector, header: *RcHeader) !void {
        if (!header.flags.in_cycle_scan) {
            header.flags.in_cycle_scan = true;
            try self.suspects.append(self.allocator, header);
        }
    }

    /// Run cycle detection and collection
    pub fn collect(self: *CycleDetector) usize {
        if (self.suspects.items.len == 0) return 0;

        var collected: usize = 0;

        // Phase 1: Mark all suspects as potentially garbage
        for (self.suspects.items) |header| {
            header.flags.marked = false;
        }

        // Phase 2: Trace from roots (objects with external references)
        for (self.suspects.items) |header| {
            if (header.getCount() > 0 and !header.flags.marked) {
                self.markReachable(header);
            }
        }

        // Phase 3: Collect unmarked objects (they form cycles)
        var i: usize = 0;
        while (i < self.suspects.items.len) {
            const header = self.suspects.items[i];
            if (!header.flags.marked and header.getCount() == 0) {
                // This object is garbage - free it
                self.freeObject(header);
                _ = self.suspects.swapRemove(i);
                collected += 1;
            } else {
                header.flags.in_cycle_scan = false;
                i += 1;
            }
        }

        if (config.enable_stats) {
            stats.cycles_collected += collected;
        }

        return collected;
    }

    fn markReachable(self: *CycleDetector, header: *RcHeader) void {
        if (header.flags.marked) return;
        header.flags.marked = true;

        // If this object contains pointers, we'd trace them here
        // For now, this is a simplified implementation
        _ = self;
    }

    fn freeObject(self: *CycleDetector, header: *RcHeader) void {
        // Call destructor if present
        if (header.destructor) |dtor| {
            const obj_ptr: *anyopaque = @ptrFromInt(@intFromPtr(header) + @sizeOf(RcHeader));
            dtor(obj_ptr);
        }

        const total_size = @sizeOf(RcHeader) + header.size;
        const raw_ptr: [*]u8 = @ptrCast(header);

        if (config.enable_stats) {
            stats.recordDeallocation(total_size);
        }

        self.allocator.free(raw_ptr[0..total_size]);
    }
};

var cycle_detector: ?CycleDetector = null;

/// Get the cycle detector
pub fn getCycleDetector() *CycleDetector {
    if (cycle_detector == null) {
        @panic("Mojo runtime not initialized. Call mojo_runtime_init() first.");
    }
    return &cycle_detector.?;
}

// ============================================================================
// Type Registry (Runtime Type Information)
// ============================================================================

pub const TypeInfo = struct {
    /// Unique type ID
    id: u32,

    /// Type name (for debugging)
    name: []const u8,

    /// Size of the type in bytes
    size: usize,

    /// Alignment requirement
    alignment: usize,

    /// Destructor function
    destructor: ?*const fn (*anyopaque) void,

    /// Copy function (for value types)
    copy: ?*const fn (*anyopaque, *const anyopaque) void,

    /// Equality function
    equals: ?*const fn (*const anyopaque, *const anyopaque) bool,

    /// Hash function
    hash: ?*const fn (*const anyopaque) u64,
};

pub const TypeRegistry = struct {
    types: std.AutoHashMapUnmanaged(u32, TypeInfo),
    next_id: u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TypeRegistry {
        return .{
            .types = .{},
            .next_id = 1, // 0 reserved for unknown type
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TypeRegistry) void {
        self.types.deinit(self.allocator);
    }

    /// Register a new type, returns its ID
    pub fn register(self: *TypeRegistry, info: TypeInfo) !u32 {
        const id = self.next_id;
        self.next_id += 1;

        var type_info = info;
        type_info.id = id;

        try self.types.put(self.allocator, id, type_info);
        return id;
    }

    /// Get type info by ID
    pub fn get(self: *const TypeRegistry, id: u32) ?TypeInfo {
        return self.types.get(id);
    }
};

var type_registry: ?TypeRegistry = null;

/// Get the type registry
pub fn getTypeRegistry() *TypeRegistry {
    if (type_registry == null) {
        @panic("Mojo runtime not initialized. Call mojo_runtime_init() first.");
    }
    return &type_registry.?;
}

// ============================================================================
// Built-in Type IDs
// ============================================================================

pub const BuiltinTypeId = enum(u32) {
    Unknown = 0,
    Int = 1,
    Int8 = 2,
    Int16 = 3,
    Int32 = 4,
    Int64 = 5,
    UInt = 6,
    UInt8 = 7,
    UInt16 = 8,
    UInt32 = 9,
    UInt64 = 10,
    Float32 = 11,
    Float64 = 12,
    Bool = 13,
    String = 14,
    List = 15,
    Dict = 16,
    Set = 17,
    Tuple = 18,
    Optional = 19,
    Result = 20,

    // User types start at 1000
    pub const USER_TYPE_START: u32 = 1000;
};

// ============================================================================
// Runtime Initialization and Cleanup
// ============================================================================

var runtime_initialized: bool = false;

/// Initialize the Mojo runtime
pub fn init(cfg: RuntimeConfig) !void {
    if (runtime_initialized) {
        return error.AlreadyInitialized;
    }

    config = cfg;

    // Initialize global allocator
    global_allocator = RuntimeAllocator.init(std.heap.page_allocator);

    // Initialize type registry
    type_registry = TypeRegistry.init(std.heap.page_allocator);

    // Initialize cycle detector if enabled
    if (config.enable_cycle_detection) {
        cycle_detector = CycleDetector.init(std.heap.page_allocator);
    }

    // Register built-in types
    try registerBuiltinTypes();

    runtime_initialized = true;
}

/// Initialize with default configuration
pub fn initDefault() !void {
    try init(.{});
}

/// Shutdown the Mojo runtime
pub fn deinit() void {
    if (!runtime_initialized) return;

    // Print stats if enabled
    if (config.enable_stats) {
        stats.print();
    }

    // Run final cycle collection
    if (cycle_detector) |*cd| {
        _ = cd.collect();
        cd.deinit();
        cycle_detector = null;
    }

    // Cleanup type registry
    if (type_registry) |*tr| {
        tr.deinit();
        type_registry = null;
    }

    // Check for leaks
    if (config.enable_stats and stats.live_allocations > 0) {
        std.debug.print("WARNING: {d} allocations leaked!\n", .{stats.live_allocations});
    }

    global_allocator = null;
    runtime_initialized = false;
}

/// Check if runtime is initialized
pub fn isInitialized() bool {
    return runtime_initialized;
}

fn registerBuiltinTypes() !void {
    const registry = getTypeRegistry();

    // Register primitive types
    _ = try registry.register(.{
        .id = @intFromEnum(BuiltinTypeId.Int),
        .name = "Int",
        .size = @sizeOf(i64),
        .alignment = @alignOf(i64),
        .destructor = null,
        .copy = null,
        .equals = null,
        .hash = null,
    });

    _ = try registry.register(.{
        .id = @intFromEnum(BuiltinTypeId.Float64),
        .name = "Float",
        .size = @sizeOf(f64),
        .alignment = @alignOf(f64),
        .destructor = null,
        .copy = null,
        .equals = null,
        .hash = null,
    });

    _ = try registry.register(.{
        .id = @intFromEnum(BuiltinTypeId.Bool),
        .name = "Bool",
        .size = @sizeOf(bool),
        .alignment = @alignOf(bool),
        .destructor = null,
        .copy = null,
        .equals = null,
        .hash = null,
    });
}

// ============================================================================
// C ABI Exports (for linking with compiled Mojo code)
// ============================================================================

/// C-callable runtime initialization
export fn mojo_runtime_init() callconv(.c) i32 {
    initDefault() catch |err| {
        std.debug.print("Failed to initialize Mojo runtime: {}\n", .{err});
        return -1;
    };
    return 0;
}

/// C-callable runtime shutdown
export fn mojo_runtime_deinit() callconv(.c) void {
    deinit();
}

/// C-callable memory allocation
export fn mojo_alloc(size: usize) callconv(.c) ?*anyopaque {
    const allocator = getAllocator();
    const slice = allocator.backing.alloc(u8, size) catch return null;

    if (config.enable_stats) {
        stats.recordAllocation(size);
    }

    return slice.ptr;
}

/// C-callable memory deallocation
export fn mojo_free(ptr: ?*anyopaque, size: usize) callconv(.c) void {
    if (ptr == null) return;

    const allocator = getAllocator();
    const slice: []u8 = @as([*]u8, @ptrCast(ptr.?))[0..size];

    if (config.enable_stats) {
        stats.recordDeallocation(size);
    }

    allocator.backing.free(slice);
}

/// C-callable RC retain
export fn mojo_retain(ptr: ?*anyopaque) callconv(.c) void {
    if (ptr == null) return;
    RuntimeAllocator.retain(ptr.?);
}

/// C-callable RC release
export fn mojo_release(ptr: ?*anyopaque) callconv(.c) void {
    if (ptr == null) return;
    getAllocator().release(ptr.?);
}

/// C-callable get reference count
export fn mojo_get_refcount(ptr: ?*anyopaque) callconv(.c) u32 {
    if (ptr == null) return 0;
    const header = RuntimeAllocator.getHeader(ptr.?);
    return header.getCount();
}

// ============================================================================
// Tests
// ============================================================================

test "runtime initialization" {
    try initDefault();
    defer deinit();

    try std.testing.expect(isInitialized());
}

test "basic allocation" {
    try initDefault();
    defer deinit();

    var allocator = getAllocator();

    const slice = try allocator.alloc(u8, 100);
    defer allocator.free(u8, slice);

    try std.testing.expect(slice.len == 100);
}

test "reference counting" {
    try initDefault();
    defer deinit();

    var allocator = getAllocator();

    const TestStruct = struct {
        value: i32,
    };

    const obj = try allocator.allocRc(TestStruct, 0, null);
    obj.value = 42;

    // Initial ref count should be 1
    const header = RuntimeAllocator.getHeader(obj);
    try std.testing.expect(header.getCount() == 1);

    // Retain increases count
    RuntimeAllocator.retain(obj);
    try std.testing.expect(header.getCount() == 2);

    // Release decreases count
    allocator.release(obj);
    try std.testing.expect(header.getCount() == 1);

    // Final release frees object
    allocator.release(obj);
}

test "type registry" {
    try initDefault();
    defer deinit();

    const registry = getTypeRegistry();

    const id = try registry.register(.{
        .id = 0,
        .name = "TestType",
        .size = 16,
        .alignment = 8,
        .destructor = null,
        .copy = null,
        .equals = null,
        .hash = null,
    });

    const info = registry.get(id);
    try std.testing.expect(info != null);
    try std.testing.expectEqualStrings("TestType", info.?.name);
}

test "statistics tracking" {
    config.enable_stats = true;
    try initDefault();
    defer deinit();

    var allocator = getAllocator();

    const initial_allocs = stats.total_allocations;

    const slice = try allocator.alloc(u8, 50);
    try std.testing.expect(stats.total_allocations == initial_allocs + 1);

    allocator.free(u8, slice);
    try std.testing.expect(stats.total_deallocations > 0);
}
