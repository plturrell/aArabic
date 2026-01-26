//! Component Registry - Day 16
//! 
//! Central registry for managing workflow components.
//! Provides registration, lookup, and factory capabilities.
//!
//! Key Features:
//! - Dynamic component registration
//! - Lookup by ID or category
//! - Search by name/tags
//! - Node creation via factory functions
//! - Configuration validation
//! - Built-in component registration

const std = @import("std");
const node_types = @import("node_types");
const metadata_mod = @import("component_metadata");
const Allocator = std.mem.Allocator;

pub const ComponentMetadata = metadata_mod.ComponentMetadata;
pub const ComponentCategory = metadata_mod.ComponentCategory;
pub const PortMetadata = metadata_mod.PortMetadata;
pub const ConfigSchemaField = metadata_mod.ConfigSchemaField;

/// Component registry error types
pub const RegistryError = error{
    ComponentAlreadyExists,
    ComponentNotFound,
    InvalidComponentId,
    InvalidConfig,
};

/// Component registry manages all available components
pub const ComponentRegistry = struct {
    allocator: Allocator,
    components: std.StringHashMap(ComponentMetadata),
    // Category index for fast filtering
    category_index: std.AutoHashMap(ComponentCategory, std.ArrayList([]const u8)),
    
    pub fn init(allocator: Allocator) ComponentRegistry {
        return ComponentRegistry{
            .allocator = allocator,
            .components = std.StringHashMap(ComponentMetadata).init(allocator),
            .category_index = std.AutoHashMap(ComponentCategory, std.ArrayList([]const u8)){},
        };
    }
    
    pub fn deinit(self: *ComponentRegistry) void {
        // Clean up category index
        var cat_iter = self.category_index.valueIterator();
        while (cat_iter.next()) |comp_list| {
            comp_list.deinit(self.allocator);
        }
        self.category_index.deinit();
        
        // Clean up components map
        self.components.deinit();
    }
    
    /// Register a component in the registry
    pub fn register(self: *ComponentRegistry, component: ComponentMetadata) !void {
        // Check if component already exists
        if (self.components.contains(component.id)) {
            return RegistryError.ComponentAlreadyExists;
        }
        
        // Add to main registry
        try self.components.put(component.id, component);
        
        // Update category index
        const result = try self.category_index.getOrPut(component.category);
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList([]const u8){};
        }
        
        try result.value_ptr.append(self.allocator, component.id);
    }
    
    /// Get component by ID
    pub fn get(self: *const ComponentRegistry, id: []const u8) ?*const ComponentMetadata {
        return self.components.getPtr(id);
    }
    
    /// Check if component exists
    pub fn has(self: *const ComponentRegistry, id: []const u8) bool {
        return self.components.contains(id);
    }
    
    /// Get component count
    pub fn count(self: *const ComponentRegistry) usize {
        return self.components.count();
    }
    
    /// List all components
    pub fn list(self: *const ComponentRegistry, allocator: Allocator) ![]ComponentMetadata {
        var result = try std.ArrayList(ComponentMetadata).initCapacity(
            allocator,
            self.components.count(),
        );
        errdefer result.deinit(allocator);
        
        var iter = self.components.valueIterator();
        while (iter.next()) |component| {
            try result.append(allocator, component.*);
        }
        
        return result.toOwnedSlice(allocator);
    }
    
    /// List components by category
    pub fn listByCategory(
        self: *const ComponentRegistry,
        allocator: Allocator,
        category: ComponentCategory,
    ) ![]ComponentMetadata {
        const comp_ids = self.category_index.get(category) orelse {
            // Return empty slice if category not found
            return try allocator.alloc(ComponentMetadata, 0);
        };
        
        var result = try std.ArrayList(ComponentMetadata).initCapacity(
            allocator,
            comp_ids.items.len,
        );
        errdefer result.deinit(allocator);
        
        for (comp_ids.items) |id| {
            if (self.components.get(id)) |component| {
                try result.append(allocator, component);
            }
        }
        
        return result.toOwnedSlice(allocator);
    }
    
    /// Search components by query (matches name, description, or tags)
    pub fn search(
        self: *const ComponentRegistry,
        allocator: Allocator,
        query: []const u8,
    ) ![]ComponentMetadata {
        var result = std.ArrayList(ComponentMetadata){};
        errdefer result.deinit(allocator);
        
        const query_lower = try std.ascii.allocLowerString(allocator, query);
        defer allocator.free(query_lower);
        
        var iter = self.components.valueIterator();
        while (iter.next()) |component| {
            // Check name
            const name_lower = try std.ascii.allocLowerString(allocator, component.name);
            defer allocator.free(name_lower);
            if (std.mem.indexOf(u8, name_lower, query_lower) != null) {
                try result.append(allocator, component.*);
                continue;
            }
            
            // Check description
            const desc_lower = try std.ascii.allocLowerString(allocator, component.description);
            defer allocator.free(desc_lower);
            if (std.mem.indexOf(u8, desc_lower, query_lower) != null) {
                try result.append(allocator, component.*);
                continue;
            }
            
            // Check tags
            for (component.tags) |tag| {
                const tag_lower = try std.ascii.allocLowerString(allocator, tag);
                defer allocator.free(tag_lower);
                if (std.mem.indexOf(u8, tag_lower, query_lower) != null) {
                    try result.append(allocator, component.*);
                    break;
                }
            }
        }
        
        return result.toOwnedSlice(allocator);
    }
    
    /// Create node instance from component
    pub fn createNode(
        self: *ComponentRegistry,
        component_id: []const u8,
        node_id: []const u8,
        node_name: []const u8,
        config: std.json.Value,
    ) !*node_types.NodeInterface {
        const component = self.components.getPtr(component_id) orelse {
            return RegistryError.ComponentNotFound;
        };
        
        // Validate configuration against schema
        try component.validateConfig(config);
        
        // Create node using factory function
        return try component.factory_fn(self.allocator, node_id, node_name, config);
    }
    
    /// Validate component configuration without creating node
    pub fn validateConfig(
        self: *const ComponentRegistry,
        component_id: []const u8,
        config: std.json.Value,
    ) !void {
        const component = self.components.get(component_id) orelse {
            return RegistryError.ComponentNotFound;
        };
        
        try component.validateConfig(config);
    }
    
    /// Unregister a component
    pub fn unregister(self: *ComponentRegistry, component_id: []const u8) !void {
        const component = self.components.get(component_id) orelse {
            return RegistryError.ComponentNotFound;
        };
        
        // Remove from category index
        if (self.category_index.getPtr(component.category)) |comp_list| {
            for (comp_list.items, 0..) |id, i| {
                if (std.mem.eql(u8, id, component_id)) {
                    _ = comp_list.orderedRemove(i);
                    break;
                }
            }
        }
        
        // Remove from main registry
        _ = self.components.remove(component_id);
    }
    
    /// Get all categories with component counts
    pub fn getCategoryCounts(self: *const ComponentRegistry, allocator: Allocator) !std.AutoHashMap(ComponentCategory, usize) {
        var counts = std.AutoHashMap(ComponentCategory, usize).init(allocator);
        
        var iter = self.category_index.iterator();
        while (iter.next()) |entry| {
            try counts.put(entry.key_ptr.*, entry.value_ptr.items.len);
        }
        
        return counts;
    }
    
    /// Register built-in components
    pub fn registerBuiltins(self: *ComponentRegistry) !void {
        // Will be populated with built-in components in subsequent implementations
        // For now, this is a placeholder for future built-in components
        _ = self;
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "ComponentRegistry init and deinit" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), registry.count());
}

test "Register and retrieve component" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    // Create dummy factory function
    const dummyFactory = struct {
        fn create(_: Allocator, _: []const u8, _: []const u8, _: std.json.Value) !*node_types.NodeInterface {
            return error.NotImplemented;
        }
    }.create;
    
    const component = ComponentMetadata{
        .id = "test_component",
        .name = "Test Component",
        .version = "1.0.0",
        .description = "A test component",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🧪",
        .color = "#FF0000",
        .tags = &[_][]const u8{"test"},
        .help_text = "This is a test component",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    try registry.register(component);
    
    try std.testing.expectEqual(@as(usize, 1), registry.count());
    try std.testing.expect(registry.has("test_component"));
    
    const retrieved = registry.get("test_component").?;
    try std.testing.expectEqualStrings("Test Component", retrieved.name);
    try std.testing.expectEqual(ComponentCategory.action, retrieved.category);
}

test "Register duplicate component" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    const dummyFactory = struct {
        fn create(_: Allocator, _: []const u8, _: []const u8, _: std.json.Value) !*node_types.NodeInterface {
            return error.NotImplemented;
        }
    }.create;
    
    const component = ComponentMetadata{
        .id = "test",
        .name = "Test",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🧪",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "Help",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    try registry.register(component);
    
    // Try to register again
    try std.testing.expectError(RegistryError.ComponentAlreadyExists, registry.register(component));
}

test "List components by category" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    const dummyFactory = struct {
        fn create(_: Allocator, _: []const u8, _: []const u8, _: std.json.Value) !*node_types.NodeInterface {
            return error.NotImplemented;
        }
    }.create;
    
    // Register action component
    const action_comp = ComponentMetadata{
        .id = "action1",
        .name = "Action",
        .version = "1.0.0",
        .description = "Action component",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "⚡",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    // Register LLM component
    const llm_comp = ComponentMetadata{
        .id = "llm1",
        .name = "LLM",
        .version = "1.0.0",
        .description = "LLM component",
        .category = .llm,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🤖",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    try registry.register(action_comp);
    try registry.register(llm_comp);
    
    const action_list = try registry.listByCategory(allocator, .action);
    defer allocator.free(action_list);
    
    try std.testing.expectEqual(@as(usize, 1), action_list.len);
    try std.testing.expectEqualStrings("action1", action_list[0].id);
    
    const llm_list = try registry.listByCategory(allocator, .llm);
    defer allocator.free(llm_list);
    
    try std.testing.expectEqual(@as(usize, 1), llm_list.len);
    try std.testing.expectEqualStrings("llm1", llm_list[0].id);
}

test "Search components" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    const dummyFactory = struct {
        fn create(_: Allocator, _: []const u8, _: []const u8, _: std.json.Value) !*node_types.NodeInterface {
            return error.NotImplemented;
        }
    }.create;
    
    const tags1 = [_][]const u8{ "http", "api" };
    const comp1 = ComponentMetadata{
        .id = "http_req",
        .name = "HTTP Request",
        .version = "1.0.0",
        .description = "Make HTTP requests",
        .category = .integration,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🌐",
        .color = "#000000",
        .tags = &tags1,
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    const tags2 = [_][]const u8{ "graphql", "api" };
    const comp2 = ComponentMetadata{
        .id = "graphql_req",
        .name = "GraphQL Request",
        .version = "1.0.0",
        .description = "Query GraphQL APIs",
        .category = .integration,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "📊",
        .color = "#000000",
        .tags = &tags2,
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    try registry.register(comp1);
    try registry.register(comp2);
    
    // Search by name
    const http_results = try registry.search(allocator, "HTTP");
    defer allocator.free(http_results);
    try std.testing.expectEqual(@as(usize, 1), http_results.len);
    try std.testing.expectEqualStrings("http_req", http_results[0].id);
    
    // Search by tag
    const api_results = try registry.search(allocator, "api");
    defer allocator.free(api_results);
    try std.testing.expectEqual(@as(usize, 2), api_results.len);
}

test "Validate configuration" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    const dummyFactory = struct {
        fn create(_: Allocator, _: []const u8, _: []const u8, _: std.json.Value) !*node_types.NodeInterface {
            return error.NotImplemented;
        }
    }.create;
    
    const config_schema = [_]ConfigSchemaField{
        ConfigSchemaField.stringField("url", true, "URL", null),
    };
    
    const component = ComponentMetadata{
        .id = "test",
        .name = "Test",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &config_schema,
        .icon = "🧪",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    try registry.register(component);
    
    // Valid config
    var valid_config = std.json.ObjectMap.init(allocator);
    defer valid_config.deinit();
    try valid_config.put("url", std.json.Value{ .string = "https://example.com" });
    
    try registry.validateConfig("test", std.json.Value{ .object = valid_config });
    
    // Invalid config (missing required field)
    var invalid_config = std.json.ObjectMap.init(allocator);
    defer invalid_config.deinit();
    
    try std.testing.expectError(
        error.MissingRequiredField,
        registry.validateConfig("test", std.json.Value{ .object = invalid_config }),
    );
}

test "Unregister component" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    const dummyFactory = struct {
        fn create(_: Allocator, _: []const u8, _: []const u8, _: std.json.Value) !*node_types.NodeInterface {
            return error.NotImplemented;
        }
    }.create;
    
    const component = ComponentMetadata{
        .id = "test",
        .name = "Test",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🧪",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    try registry.register(component);
    try std.testing.expect(registry.has("test"));
    
    try registry.unregister("test");
    try std.testing.expect(!registry.has("test"));
}

test "Get category counts" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    const dummyFactory = struct {
        fn create(_: Allocator, _: []const u8, _: []const u8, _: std.json.Value) !*node_types.NodeInterface {
            return error.NotImplemented;
        }
    }.create;
    
    // Register 2 action components
    const action1 = ComponentMetadata{
        .id = "action1",
        .name = "Action 1",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🧪",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    const action2 = ComponentMetadata{
        .id = "action2",
        .name = "Action 2",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🧪",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    // Register 1 LLM component
    const llm1 = ComponentMetadata{
        .id = "llm1",
        .name = "LLM",
        .version = "1.0.0",
        .description = "Test",
        .category = .llm,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🤖",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    try registry.register(action1);
    try registry.register(action2);
    try registry.register(llm1);
    
    var counts = try registry.getCategoryCounts(allocator);
    defer counts.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), counts.get(.action).?);
    try std.testing.expectEqual(@as(usize, 1), counts.get(.llm).?);
}

test "List all components" {
    const allocator = std.testing.allocator;
    
    var registry = ComponentRegistry.init(allocator);
    defer registry.deinit();
    
    const dummyFactory = struct {
        fn create(_: Allocator, _: []const u8, _: []const u8, _: std.json.Value) !*node_types.NodeInterface {
            return error.NotImplemented;
        }
    }.create;
    
    const comp1 = ComponentMetadata{
        .id = "comp1",
        .name = "Component 1",
        .version = "1.0.0",
        .description = "Test",
        .category = .action,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🧪",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    const comp2 = ComponentMetadata{
        .id = "comp2",
        .name = "Component 2",
        .version = "1.0.0",
        .description = "Test",
        .category = .llm,
        .inputs = &[_]PortMetadata{},
        .outputs = &[_]PortMetadata{},
        .config_schema = &[_]ConfigSchemaField{},
        .icon = "🤖",
        .color = "#000000",
        .tags = &[_][]const u8{},
        .help_text = "",
        .examples = &[_][]const u8{},
        .factory_fn = dummyFactory,
    };
    
    try registry.register(comp1);
    try registry.register(comp2);
    
    const all_components = try registry.list(allocator);
    defer allocator.free(all_components);
    
    try std.testing.expectEqual(@as(usize, 2), all_components.len);
}
