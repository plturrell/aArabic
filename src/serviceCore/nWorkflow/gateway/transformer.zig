// Request/Response Transformation for APISIX
// Day 32: APISIX Gateway Integration (Continued)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Transformation type
pub const TransformationType = enum {
    header_add,
    header_remove,
    header_rename,
    body_json_path,
    body_template,
    body_base64_encode,
    body_base64_decode,
    query_param_add,
    query_param_remove,
    uri_rewrite,
    method_override,
};

/// Header transformation
pub const HeaderTransform = struct {
    action: enum { add, remove, rename },
    header_name: []const u8,
    header_value: ?[]const u8 = null,
    new_header_name: ?[]const u8 = null, // For rename
};

/// Body transformation
pub const BodyTransform = struct {
    action: enum { json_path, template, base64_encode, base64_decode },
    json_path: ?[]const u8 = null, // JSONPath expression
    template: ?[]const u8 = null, // Template string
};

/// Query parameter transformation
pub const QueryTransform = struct {
    action: enum { add, remove },
    param_name: []const u8,
    param_value: ?[]const u8 = null,
};

/// URI rewrite rule
pub const UriRewrite = struct {
    regex: []const u8,
    replacement: []const u8,
    options: ?[]const u8 = null, // e.g., "i" for case-insensitive
};

/// Complete transformation configuration
pub const TransformConfig = struct {
    headers: []HeaderTransform = &[_]HeaderTransform{},
    body: ?BodyTransform = null,
    query_params: []QueryTransform = &[_]QueryTransform{},
    uri_rewrite: ?UriRewrite = null,
    method_override: ?[]const u8 = null,
};

/// Transformer manager
pub const TransformerManager = struct {
    allocator: Allocator,
    transformations: std.StringHashMap(TransformConfig),
    arena: std.heap.ArenaAllocator,

    pub fn init(allocator: Allocator) !TransformerManager {
        return TransformerManager{
            .allocator = allocator,
            .transformations = std.StringHashMap(TransformConfig).init(allocator),
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *TransformerManager) void {
        var it = self.transformations.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.transformations.deinit();
        self.arena.deinit();
    }

    /// Register transformation for a route
    pub fn registerTransformation(
        self: *TransformerManager,
        route_id: []const u8,
        config: TransformConfig,
    ) !void {
        const key = try self.allocator.dupe(u8, route_id);
        try self.transformations.put(key, config);
    }

    /// Get transformation for route
    pub fn getTransformation(self: *const TransformerManager, route_id: []const u8) ?TransformConfig {
        return self.transformations.get(route_id);
    }

    /// Remove transformation
    pub fn removeTransformation(self: *TransformerManager, route_id: []const u8) !void {
        if (self.transformations.fetchRemove(route_id)) |kv| {
            self.allocator.free(kv.key);
        } else {
            return error.TransformationNotFound;
        }
    }

    /// Serialize to APISIX plugin format (request-transformer)
    pub fn serializeRequestTransformer(self: *TransformerManager, config: TransformConfig) ![]const u8 {
        const arena_allocator = self.arena.allocator();
        var string: std.ArrayListUnmanaged(u8) = .{};
        var writer = string.writer(arena_allocator);

        try writer.writeAll("{\"request-transformer\":{");

        var has_content = false;

        // Header transformations
        if (config.headers.len > 0) {
            if (has_content) try writer.writeAll(",");
            try writer.writeAll("\"set\":[");

            var header_idx: usize = 0;
            for (config.headers) |transform| {
                if (transform.action == .add) {
                    if (header_idx > 0) try writer.writeAll(",");
                    try writer.print("[\"{s}\",\"{s}\"]", .{ transform.header_name, transform.header_value orelse "" });
                    header_idx += 1;
                }
            }

            try writer.writeAll("],\"remove\":[");

            header_idx = 0;
            for (config.headers) |transform| {
                if (transform.action == .remove) {
                    if (header_idx > 0) try writer.writeAll(",");
                    try writer.print("\"{s}\"", .{transform.header_name});
                    header_idx += 1;
                }
            }

            try writer.writeAll("]");
            has_content = true;
        }

        // Query parameter transformations
        if (config.query_params.len > 0) {
            if (has_content) try writer.writeAll(",");
            try writer.writeAll("\"add\":[");

            var query_idx: usize = 0;
            for (config.query_params) |transform| {
                if (transform.action == .add) {
                    if (query_idx > 0) try writer.writeAll(",");
                    try writer.print("[\"{s}\",\"{s}\"]", .{ transform.param_name, transform.param_value orelse "" });
                    query_idx += 1;
                }
            }

            try writer.writeAll("],\"remove_query\":[");

            query_idx = 0;
            for (config.query_params) |transform| {
                if (transform.action == .remove) {
                    if (query_idx > 0) try writer.writeAll(",");
                    try writer.print("\"{s}\"", .{transform.param_name});
                    query_idx += 1;
                }
            }

            try writer.writeAll("]");
            has_content = true;
        }

        try writer.writeAll("}}");
        return string.toOwnedSlice(arena_allocator);
    }

    /// Serialize to APISIX plugin format (response-rewrite)
    pub fn serializeResponseRewrite(self: *TransformerManager, config: TransformConfig) ![]const u8 {
        const arena_allocator = self.arena.allocator();
        var string: std.ArrayListUnmanaged(u8) = .{};
        var writer = string.writer(arena_allocator);

        try writer.writeAll("{\"response-rewrite\":{");

        var has_content = false;

        // Response headers
        if (config.headers.len > 0) {
            if (has_content) try writer.writeAll(",");
            try writer.writeAll("\"headers\":{");

            var header_idx: usize = 0;
            for (config.headers) |transform| {
                if (transform.action == .add) {
                    if (header_idx > 0) try writer.writeAll(",");
                    try writer.print("\"{s}\":\"{s}\"", .{ transform.header_name, transform.header_value orelse "" });
                    header_idx += 1;
                }
            }

            try writer.writeAll("}");
            has_content = true;
        }

        // Body template
        if (config.body) |body_transform| {
            if (body_transform.template) |template| {
                if (has_content) try writer.writeAll(",");
                try writer.print("\"body\":\"{s}\"", .{template});
                has_content = true;
            }
        }

        try writer.writeAll("}}");
        return string.toOwnedSlice(arena_allocator);
    }

    /// Serialize URI rewrite plugin
    pub fn serializeUriRewrite(self: *TransformerManager, rewrite: UriRewrite) ![]const u8 {
        const arena_allocator = self.arena.allocator();
        var string: std.ArrayListUnmanaged(u8) = .{};
        var writer = string.writer(arena_allocator);

        try writer.writeAll("{\"proxy-rewrite\":{");
        try writer.print("\"regex_uri\":[\"{s}\",\"{s}\"", .{ rewrite.regex, rewrite.replacement });

        if (rewrite.options) |options| {
            try writer.print(",\"{s}\"", .{options});
        }

        try writer.writeAll("]}}");
        return string.toOwnedSlice(arena_allocator);
    }
};

/// Template engine for body transformation
pub const TemplateEngine = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) TemplateEngine {
        return TemplateEngine{ .allocator = allocator };
    }

    /// Render template with variables
    /// Supports {{variable}} syntax
    pub fn render(self: *TemplateEngine, template: []const u8, variables: std.StringHashMap([]const u8)) ![]const u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        var writer = result.writer(self.allocator);

        var i: usize = 0;
        while (i < template.len) {
            if (i + 1 < template.len and template[i] == '{' and template[i + 1] == '{') {
                // Find closing }}
                const start = i + 2;
                var end: ?usize = null;
                var j = start;
                while (j + 1 < template.len) {
                    if (template[j] == '}' and template[j + 1] == '}') {
                        end = j;
                        break;
                    }
                    j += 1;
                }

                if (end) |e| {
                    const var_name = template[start..e];
                    // Lookup variable
                    if (variables.get(var_name)) |value| {
                        try writer.writeAll(value);
                    } else {
                        // Variable not found, keep placeholder
                        try writer.print("{{{{{s}}}}}", .{var_name});
                    }
                    i = e + 2;
                } else {
                    // No closing }}, treat as literal
                    try writer.writeByte(template[i]);
                    i += 1;
                }
            } else {
                try writer.writeByte(template[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }
};

/// JSON Path evaluator (simplified)
pub const JsonPathEvaluator = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) JsonPathEvaluator {
        return JsonPathEvaluator{ .allocator = allocator };
    }

    /// Evaluate JSON path expression
    /// Supports simple paths like $.user.name or $.items[0].id
    pub fn evaluate(self: *JsonPathEvaluator, json_str: []const u8, path: []const u8) !?[]const u8 {
        // Parse JSON
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_str, .{});
        defer parsed.deinit();

        // Evaluate path
        return try self.evaluatePath(parsed.value, path);
    }

    fn evaluatePath(self: *JsonPathEvaluator, value: std.json.Value, path: []const u8) !?[]const u8 {
        if (path.len == 0 or path[0] != '$') {
            return null;
        }

        var current = value;
        var i: usize = 1; // Skip $

        while (i < path.len) {
            if (path[i] == '.') {
                i += 1;
                // Find next . or [ or end
                const start = i;
                while (i < path.len and path[i] != '.' and path[i] != '[') {
                    i += 1;
                }
                const field_name = path[start..i];

                // Navigate to field
                if (current == .object) {
                    if (current.object.get(field_name)) |next_value| {
                        current = next_value;
                    } else {
                        return null;
                    }
                } else {
                    return null;
                }
            } else if (path[i] == '[') {
                // Array index
                i += 1;
                const start = i;
                while (i < path.len and path[i] != ']') {
                    i += 1;
                }
                const index_str = path[start..i];
                const index = try std.fmt.parseInt(usize, index_str, 10);
                i += 1; // Skip ]

                if (current == .array) {
                    if (index < current.array.items.len) {
                        current = current.array.items[index];
                    } else {
                        return null;
                    }
                } else {
                    return null;
                }
            } else {
                i += 1;
            }
        }

        // Serialize final value using Stringify
        var out: std.io.Writer.Allocating = .init(self.allocator);
        errdefer out.deinit();
        var jw: std.json.Stringify = .{ .writer = &out.writer };
        try jw.write(current);
        return try out.toOwnedSlice();
    }
};

// Tests
test "TransformerManager: init and deinit" {
    const allocator = std.testing.allocator;

    var manager = try TransformerManager.init(allocator);
    defer manager.deinit();

    try std.testing.expect(manager.transformations.count() == 0);
}

test "TransformerManager: register transformation" {
    const allocator = std.testing.allocator;

    var manager = try TransformerManager.init(allocator);
    defer manager.deinit();

    var headers = [_]HeaderTransform{
        .{ .action = .add, .header_name = "X-Custom-Header", .header_value = "value" },
    };

    const config = TransformConfig{
        .headers = headers[0..],
    };

    try manager.registerTransformation("route-1", config);
    try std.testing.expect(manager.transformations.count() == 1);

    const retrieved = manager.getTransformation("route-1");
    try std.testing.expect(retrieved != null);
}

test "TransformerManager: serialize request transformer" {
    const allocator = std.testing.allocator;

    var manager = try TransformerManager.init(allocator);
    defer manager.deinit();

    var headers = [_]HeaderTransform{
        .{ .action = .add, .header_name = "X-Custom", .header_value = "test" },
        .{ .action = .remove, .header_name = "X-Remove" },
    };

    const config = TransformConfig{
        .headers = headers[0..],
    };

    const json = try manager.serializeRequestTransformer(config);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"request-transformer\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"X-Custom\"") != null);
}

test "TransformerManager: serialize URI rewrite" {
    const allocator = std.testing.allocator;

    var manager = try TransformerManager.init(allocator);
    defer manager.deinit();

    const rewrite = UriRewrite{
        .regex = "^/old/(.*)",
        .replacement = "/new/$1",
    };

    const json = try manager.serializeUriRewrite(rewrite);

    try std.testing.expect(std.mem.indexOf(u8, json, "proxy-rewrite") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "^/old/(.*)") != null);
}

test "TemplateEngine: render simple template" {
    const allocator = std.testing.allocator;

    var engine = TemplateEngine.init(allocator);

    var vars = std.StringHashMap([]const u8).init(allocator);
    defer vars.deinit();

    try vars.put("name", "Alice");
    try vars.put("age", "30");

    const template = "Hello {{name}}, you are {{age}} years old";
    const result = try engine.render(template, vars);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello Alice, you are 30 years old", result);
}

test "TemplateEngine: missing variable" {
    const allocator = std.testing.allocator;

    var engine = TemplateEngine.init(allocator);

    var vars = std.StringHashMap([]const u8).init(allocator);
    defer vars.deinit();

    try vars.put("name", "Bob");

    const template = "Hello {{name}}, your role is {{role}}";
    const result = try engine.render(template, vars);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello Bob, your role is {{role}}", result);
}

test "JsonPathEvaluator: simple field access" {
    const allocator = std.testing.allocator;

    var evaluator = JsonPathEvaluator.init(allocator);

    const json = "{\"user\":{\"name\":\"Alice\",\"age\":30}}";
    const result = try evaluator.evaluate(json, "$.user.name");

    if (result) |value| {
        defer allocator.free(value);
        try std.testing.expectEqualStrings("\"Alice\"", value);
    } else {
        try std.testing.expect(false);
    }
}

test "JsonPathEvaluator: array access" {
    const allocator = std.testing.allocator;

    var evaluator = JsonPathEvaluator.init(allocator);

    const json = "{\"items\":[{\"id\":1},{\"id\":2}]}";
    const result = try evaluator.evaluate(json, "$.items[0].id");

    if (result) |value| {
        defer allocator.free(value);
        try std.testing.expectEqualStrings("1", value);
    } else {
        try std.testing.expect(false);
    }
}
