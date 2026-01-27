//! ============================================================================
//! Extension Registry
//! Manages registration, lifecycle, and execution of backend extensions
//! ============================================================================
//!
//! [CODE:file=extension_registry.zig]
//! [CODE:module=extensions]
//! [CODE:language=zig]
//!
//! [TABLE:manages=TB_EXTENSION_REGISTRY]
//!
//! [API:serves=/api/v1/extensions/*]
//!
//! [RELATION:called_by=CODE:extension_api.zig]
//! [RELATION:called_by=CODE:main.zig]
//!
//! Note: Infrastructure code - no ODPS business rules implemented here.
//! Extensions enable runtime pluggable functionality for calculations,
//! data transformations, and API handlers.

const std = @import("std");

/// Extension types supported by the system
pub const ExtensionType = enum {
    data_source,
    calculator,
    transformer,
    api_handler,
    middleware,
};

/// Extension hook function signatures
pub const HookFn = struct {
    init: ?*const fn (allocator: std.mem.Allocator) anyerror!void,
    deinit: ?*const fn () void,
    onDataLoad: ?*const fn (allocator: std.mem.Allocator, data: []const u8) anyerror![]const u8,
    onCalculate: ?*const fn (allocator: std.mem.Allocator, input: []const u8) anyerror![]const u8,
    handleRequest: ?*const fn (allocator: std.mem.Allocator, path: []const u8, method: []const u8, body: []const u8) anyerror![]const u8,
};

/// Extension metadata and configuration
pub const Extension = struct {
    id: []const u8,
    name: []const u8,
    version: []const u8,
    ext_type: ExtensionType,
    priority: i32,
    enabled: bool,
    
    // Hook functions
    hooks: HookFn,
    
    // Extension-specific data
    user_data: ?*anyopaque,
    
    pub fn init(
        id: []const u8,
        name: []const u8,
        version: []const u8,
        ext_type: ExtensionType,
    ) Extension {
        return Extension{
            .id = id,
            .name = name,
            .version = version,
            .ext_type = ext_type,
            .priority = 0,
            .enabled = true,
            .hooks = HookFn{
                .init = null,
                .deinit = null,
                .onDataLoad = null,
                .onCalculate = null,
                .handleRequest = null,
            },
            .user_data = null,
        };
    }
};

/// Extension Registry
/// Manages registration, lifecycle, and execution of backend extensions
pub const ExtensionRegistry = struct {
    allocator: std.mem.Allocator,
    extensions: std.StringHashMap(*Extension),
    hooks_by_type: std.AutoHashMap(ExtensionType, std.ArrayList(*Extension)),
    initialized: bool,
    
    pub fn init(allocator: std.mem.Allocator) !ExtensionRegistry {
        return ExtensionRegistry{
            .allocator = allocator,
            .extensions = std.StringHashMap(*Extension).init(allocator),
            .hooks_by_type = std.AutoHashMap(ExtensionType, std.ArrayList(*Extension)).init(allocator),
            .initialized = false,
        };
    }
    
    pub fn deinit(self: *ExtensionRegistry) void {
        // Call deinit hooks for all extensions
        var iter = self.extensions.valueIterator();
        while (iter.next()) |ext| {
            if (ext.*.hooks.deinit) |deinitFn| {
                deinitFn();
            }
        }
        
        // Clean up hooks by type
        var hooks_iter = self.hooks_by_type.valueIterator();
        while (hooks_iter.next()) |list| {
            list.deinit();
        }
        self.hooks_by_type.deinit();
        
        // Clean up extensions map
        self.extensions.deinit();
    }
    
    /// Initialize all registered extensions
    pub fn initializeExtensions(self: *ExtensionRegistry) !void {
        if (self.initialized) return;
        
        std.debug.print("Initializing extensions...\n", .{});
        
        var iter = self.extensions.valueIterator();
        while (iter.next()) |ext| {
            if (!ext.*.enabled) continue;
            
            if (ext.*.hooks.init) |initFn| {
                initFn(self.allocator) catch |err| {
                    std.debug.print("Error initializing extension {s}: {}\n", .{ ext.*.id, err });
                    return err;
                };
                std.debug.print("Initialized extension: {s} v{s}\n", .{ ext.*.name, ext.*.version });
            }
        }
        
        self.initialized = true;
        std.debug.print("All extensions initialized successfully\n", .{});
    }
    
    /// Register a new extension
    pub fn register(self: *ExtensionRegistry, extension: *Extension) !void {
        // Check if already registered
        if (self.extensions.get(extension.id)) |_| {
            std.debug.print("Warning: Extension already registered: {s}\n", .{extension.id});
            return;
        }
        
        // Register in main map
        try self.extensions.put(extension.id, extension);
        
        // Register in hooks by type
        const result = try self.hooks_by_type.getOrPut(extension.ext_type);
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList(*Extension).init(self.allocator);
        }
        try result.value_ptr.append(extension);
        
        // Sort by priority (higher priority first)
        const items = result.value_ptr.items;
        std.mem.sort(*Extension, items, {}, extensionPriorityDesc);
        
        std.debug.print("Registered extension: {s} (type: {s}, priority: {})\n", 
                       .{ extension.name, @tagName(extension.ext_type), extension.priority });
    }
    
    /// Unregister an extension
    pub fn unregister(self: *ExtensionRegistry, extension_id: []const u8) !void {
        const ext = self.extensions.get(extension_id) orelse return;
        
        // Call deinit hook
        if (ext.hooks.deinit) |deinitFn| {
            deinitFn();
        }
        
        // Remove from hooks by type
        if (self.hooks_by_type.get(ext.ext_type)) |list| {
            for (list.items, 0..) |item, i| {
                if (std.mem.eql(u8, item.id, extension_id)) {
                    _ = list.swapRemove(i);
                    break;
                }
            }
        }
        
        // Remove from main map
        _ = self.extensions.remove(extension_id);
        
        std.debug.print("Unregistered extension: {s}\n", .{extension_id});
    }
    
    /// Get extension by ID
    pub fn get(self: *ExtensionRegistry, extension_id: []const u8) ?*Extension {
        return self.extensions.get(extension_id);
    }
    
    /// Get all extensions of a specific type
    pub fn getByType(self: *ExtensionRegistry, ext_type: ExtensionType) ?[]const *Extension {
        const list = self.hooks_by_type.get(ext_type) orelse return null;
        return list.items;
    }
    
    /// Execute data load hooks
    pub fn executeDataLoadHooks(
        self: *ExtensionRegistry,
        allocator: std.mem.Allocator,
        data: []const u8,
    ) ![]const u8 {
        const extensions = self.getByType(.data_source) orelse return try allocator.dupe(u8, data);
        
        var result = try allocator.dupe(u8, data);
        
        for (extensions) |ext| {
            if (!ext.enabled) continue;
            
            if (ext.hooks.onDataLoad) |hookFn| {
                const new_result = hookFn(allocator, result) catch |err| {
                    std.debug.print("Error in data load hook for {s}: {}\n", .{ ext.id, err });
                    continue;
                };
                
                allocator.free(result);
                result = new_result;
            }
        }
        
        return result;
    }
    
    /// Execute calculation hooks
    pub fn executeCalculationHooks(
        self: *ExtensionRegistry,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) ![]const u8 {
        const extensions = self.getByType(.calculator) orelse return try allocator.dupe(u8, input);
        
        var result = try allocator.dupe(u8, input);
        
        for (extensions) |ext| {
            if (!ext.enabled) continue;
            
            if (ext.hooks.onCalculate) |hookFn| {
                const new_result = hookFn(allocator, result) catch |err| {
                    std.debug.print("Error in calculation hook for {s}: {}\n", .{ ext.id, err });
                    continue;
                };
                
                allocator.free(result);
                result = new_result;
            }
        }
        
        return result;
    }
    
    /// Handle extension API request
    pub fn handleExtensionRequest(
        self: *ExtensionRegistry,
        allocator: std.mem.Allocator,
        extension_id: []const u8,
        path: []const u8,
        method: []const u8,
        body: []const u8,
    ) ![]const u8 {
        const extension = self.get(extension_id) orelse return error.ExtensionNotFound;
        
        if (!extension.enabled) {
            return error.ExtensionDisabled;
        }
        
        if (extension.hooks.handleRequest) |handler| {
            return try handler(allocator, path, method, body);
        }
        
        return error.HandlerNotImplemented;
    }
    
    /// Enable an extension
    pub fn enable(self: *ExtensionRegistry, extension_id: []const u8) !void {
        const ext = self.get(extension_id) orelse return error.ExtensionNotFound;
        ext.enabled = true;
        std.debug.print("Enabled extension: {s}\n", .{extension_id});
    }
    
    /// Disable an extension
    pub fn disable(self: *ExtensionRegistry, extension_id: []const u8) !void {
        const ext = self.get(extension_id) orelse return error.ExtensionNotFound;
        ext.enabled = false;
        std.debug.print("Disabled extension: {s}\n", .{extension_id});
    }
    
    /// Get extension statistics
    pub fn getStats(self: *ExtensionRegistry) ExtensionStats {
        var total: usize = 0;
        var enabled: usize = 0;
        var by_type = std.mem.zeroes([5]usize); // One for each ExtensionType
        
        var iter = self.extensions.valueIterator();
        while (iter.next()) |ext| {
            total += 1;
            if (ext.*.enabled) enabled += 1;
            
            const type_index = @intFromEnum(ext.*.ext_type);
            by_type[type_index] += 1;
        }
        
        return ExtensionStats{
            .total = total,
            .enabled = enabled,
            .data_sources = by_type[@intFromEnum(ExtensionType.data_source)],
            .calculators = by_type[@intFromEnum(ExtensionType.calculator)],
            .transformers = by_type[@intFromEnum(ExtensionType.transformer)],
            .api_handlers = by_type[@intFromEnum(ExtensionType.api_handler)],
            .middleware = by_type[@intFromEnum(ExtensionType.middleware)],
        };
    }
};

/// Extension statistics
pub const ExtensionStats = struct {
    total: usize,
    enabled: usize,
    data_sources: usize,
    calculators: usize,
    transformers: usize,
    api_handlers: usize,
    middleware: usize,
};

/// Comparison function for sorting extensions by priority (descending)
fn extensionPriorityDesc(_: void, a: *Extension, b: *Extension) bool {
    return a.priority > b.priority;
}

// Tests
test "ExtensionRegistry - basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var registry = try ExtensionRegistry.init(allocator);
    defer registry.deinit();
    
    // Create test extension
    var ext = Extension.init(
        "test-ext",
        "Test Extension",
        "1.0.0",
        .calculator,
    );
    ext.priority = 10;
    
    // Register
    try registry.register(&ext);
    
    // Verify registration
    const retrieved = registry.get("test-ext");
    try testing.expect(retrieved != null);
    try testing.expectEqualStrings("Test Extension", retrieved.?.name);
    
    // Check stats
    const stats = registry.getStats();
    try testing.expectEqual(@as(usize, 1), stats.total);
    try testing.expectEqual(@as(usize, 1), stats.enabled);
    try testing.expectEqual(@as(usize, 1), stats.calculators);
}

test "ExtensionRegistry - enable/disable" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var registry = try ExtensionRegistry.init(allocator);
    defer registry.deinit();
    
    var ext = Extension.init("test-ext", "Test", "1.0.0", .calculator);
    try registry.register(&ext);
    
    // Disable
    try registry.disable("test-ext");
    const retrieved = registry.get("test-ext");
    try testing.expect(!retrieved.?.enabled);
    
    // Re-enable
    try registry.enable("test-ext");
    try testing.expect(retrieved.?.enabled);
}

test "ExtensionRegistry - priority ordering" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var registry = try ExtensionRegistry.init(allocator);
    defer registry.deinit();
    
    var ext1 = Extension.init("ext1", "Extension 1", "1.0.0", .calculator);
    ext1.priority = 5;
    
    var ext2 = Extension.init("ext2", "Extension 2", "1.0.0", .calculator);
    ext2.priority = 10;
    
    var ext3 = Extension.init("ext3", "Extension 3", "1.0.0", .calculator);
    ext3.priority = 1;
    
    try registry.register(&ext1);
    try registry.register(&ext2);
    try registry.register(&ext3);
    
    // Get calculators - should be sorted by priority
    const calculators = registry.getByType(.calculator).?;
    try testing.expectEqual(@as(usize, 3), calculators.len);
    try testing.expectEqual(@as(i32, 10), calculators[0].priority); // ext2 first
    try testing.expectEqual(@as(i32, 5), calculators[1].priority);  // ext1 second
    try testing.expectEqual(@as(i32, 1), calculators[2].priority);  // ext3 last
}