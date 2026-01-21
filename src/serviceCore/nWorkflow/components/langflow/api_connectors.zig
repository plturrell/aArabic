// API Connector Components for nWorkflow
// Day 29: Langflow Component Parity (Part 2/3)
// Implements: WebSocketNode, GraphQLNode, RESTClientNode

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// ============================================================================
// WebSocket Node - Bidirectional WebSocket Communication
// ============================================================================

pub const WebSocketNode = struct {
    allocator: Allocator,
    url: []const u8,
    headers: StringHashMap([]const u8),
    auto_reconnect: bool,
    reconnect_delay_ms: u32,
    max_reconnects: u32,
    message_queue: ArrayList([]const u8),
    
    pub fn init(allocator: Allocator, url: []const u8) !WebSocketNode {
        return WebSocketNode{
            .allocator = allocator,
            .url = try allocator.dupe(u8, url),
            .headers = StringHashMap([]const u8).init(allocator),
            .auto_reconnect = true,
            .reconnect_delay_ms = 1000,
            .max_reconnects = 5,
            .message_queue = ArrayList([]const u8){},
        };
    }
    
    pub fn deinit(self: *WebSocketNode) void {
        self.allocator.free(self.url);
        
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
        
        for (self.message_queue.items) |msg| {
            self.allocator.free(msg);
        }
        self.message_queue.deinit(self.allocator);
    }
    
    pub fn setHeader(self: *WebSocketNode, name: []const u8, value: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        
        try self.headers.put(name_copy, value_copy);
    }
    
    pub fn connect(self: *WebSocketNode) !void {
        // Placeholder for actual WebSocket connection
        // In production, would use a WebSocket client library
        _ = self;
    }
    
    pub fn send(self: *WebSocketNode, message: []const u8) !void {
        const msg_copy = try self.allocator.dupe(u8, message);
        try self.message_queue.append(self.allocator, msg_copy);
    }
    
    pub fn receive(self: *WebSocketNode) !?[]const u8 {
        if (self.message_queue.items.len > 0) {
            return self.message_queue.items[0];
        }
        return null;
    }
    
    pub fn close(self: *WebSocketNode) !void {
        // Placeholder for actual WebSocket close
        _ = self;
    }
};

// ============================================================================
// GraphQL Node - GraphQL Query and Mutation Support
// ============================================================================

pub const GraphQLOperation = enum {
    query,
    mutation,
    subscription,
};

pub const GraphQLNode = struct {
    allocator: Allocator,
    endpoint: []const u8,
    operation_type: GraphQLOperation,
    query: []const u8,
    variables: StringHashMap([]const u8),
    headers: StringHashMap([]const u8),
    
    pub fn init(
        allocator: Allocator,
        endpoint: []const u8,
        operation_type: GraphQLOperation,
        query: []const u8,
    ) !GraphQLNode {
        return GraphQLNode{
            .allocator = allocator,
            .endpoint = try allocator.dupe(u8, endpoint),
            .operation_type = operation_type,
            .query = try allocator.dupe(u8, query),
            .variables = StringHashMap([]const u8).init(allocator),
            .headers = StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *GraphQLNode) void {
        self.allocator.free(self.endpoint);
        self.allocator.free(self.query);
        
        var var_it = self.variables.iterator();
        while (var_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.variables.deinit();
        
        var header_it = self.headers.iterator();
        while (header_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
    }
    
    pub fn setVariable(self: *GraphQLNode, name: []const u8, value: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        
        try self.variables.put(name_copy, value_copy);
    }
    
    pub fn setHeader(self: *GraphQLNode, name: []const u8, value: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        
        try self.headers.put(name_copy, value_copy);
    }
    
    pub fn buildRequest(self: *const GraphQLNode) ![]const u8 {
        var request = ArrayList(u8){};
        errdefer request.deinit(self.allocator);
        
        var writer = request.writer(self.allocator);
        
        try writer.writeAll("{\"query\":\"");
        try writer.writeAll(self.query);
        try writer.writeAll("\"");
        
        if (self.variables.count() > 0) {
            try writer.writeAll(",\"variables\":{");
            var first = true;
            var it = self.variables.iterator();
            while (it.next()) |entry| {
                if (!first) try writer.writeAll(",");
                first = false;
                try writer.print("\"{s}\":\"{s}\"", .{ entry.key_ptr.*, entry.value_ptr.* });
            }
            try writer.writeAll("}");
        }
        
        try writer.writeAll("}");
        
        return try request.toOwnedSlice(self.allocator);
    }
    
    pub fn execute(self: *const GraphQLNode) ![]const u8 {
        const request = try self.buildRequest();
        defer self.allocator.free(request);
        
        // Placeholder for actual HTTP request
        // In production, would use an HTTP client to POST to endpoint
        return try self.allocator.dupe(u8, "{\"data\":{}}");
    }
};

// ============================================================================
// REST Client Node - Advanced REST API Client
// ============================================================================

pub const HttpMethod = enum {
    GET,
    POST,
    PUT,
    PATCH,
    DELETE,
    HEAD,
    OPTIONS,
    
    pub fn toString(self: HttpMethod) []const u8 {
        return switch (self) {
            .GET => "GET",
            .POST => "POST",
            .PUT => "PUT",
            .PATCH => "PATCH",
            .DELETE => "DELETE",
            .HEAD => "HEAD",
            .OPTIONS => "OPTIONS",
        };
    }
};

pub const RESTClientNode = struct {
    allocator: Allocator,
    base_url: []const u8,
    default_headers: StringHashMap([]const u8),
    timeout_ms: u32,
    retry_count: u32,
    retry_delay_ms: u32,
    
    pub fn init(allocator: Allocator, base_url: []const u8) !RESTClientNode {
        return RESTClientNode{
            .allocator = allocator,
            .base_url = try allocator.dupe(u8, base_url),
            .default_headers = StringHashMap([]const u8).init(allocator),
            .timeout_ms = 30000,
            .retry_count = 3,
            .retry_delay_ms = 1000,
        };
    }
    
    pub fn deinit(self: *RESTClientNode) void {
        self.allocator.free(self.base_url);
        
        var it = self.default_headers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.default_headers.deinit();
    }
    
    pub fn setDefaultHeader(self: *RESTClientNode, name: []const u8, value: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        
        try self.default_headers.put(name_copy, value_copy);
    }
    
    pub fn buildUrl(self: *const RESTClientNode, path: []const u8) ![]const u8 {
        var url = ArrayList(u8){};
        errdefer url.deinit(self.allocator);
        
        var writer = url.writer(self.allocator);
        try writer.writeAll(self.base_url);
        
        if (self.base_url[self.base_url.len - 1] != '/' and path[0] != '/') {
            try writer.writeAll("/");
        }
        
        try writer.writeAll(path);
        
        return try url.toOwnedSlice(self.allocator);
    }
    
    pub fn request(
        self: *const RESTClientNode,
        method: HttpMethod,
        path: []const u8,
        body: ?[]const u8,
        headers: ?StringHashMap([]const u8),
    ) ![]const u8 {
        const url = try self.buildUrl(path);
        defer self.allocator.free(url);
        
        // Merge default headers with request-specific headers
        var merged_headers = StringHashMap([]const u8).init(self.allocator);
        defer merged_headers.deinit();
        
        var default_it = self.default_headers.iterator();
        while (default_it.next()) |entry| {
            try merged_headers.put(entry.key_ptr.*, entry.value_ptr.*);
        }
        
        if (headers) |h| {
            var header_it = h.iterator();
            while (header_it.next()) |entry| {
                try merged_headers.put(entry.key_ptr.*, entry.value_ptr.*);
            }
        }
        
        // Placeholder for actual HTTP request
        // In production, would use an HTTP client library
        _ = method;
        _ = body;
        
        return try self.allocator.dupe(u8, "{\"status\":\"success\"}");
    }
    
    pub fn get(self: *const RESTClientNode, path: []const u8) ![]const u8 {
        return try self.request(.GET, path, null, null);
    }
    
    pub fn post(self: *const RESTClientNode, path: []const u8, body: []const u8) ![]const u8 {
        return try self.request(.POST, path, body, null);
    }
    
    pub fn put(self: *const RESTClientNode, path: []const u8, body: []const u8) ![]const u8 {
        return try self.request(.PUT, path, body, null);
    }
    
    pub fn patch(self: *const RESTClientNode, path: []const u8, body: []const u8) ![]const u8 {
        return try self.request(.PATCH, path, body, null);
    }
    
    pub fn delete(self: *const RESTClientNode, path: []const u8) ![]const u8 {
        return try self.request(.DELETE, path, null, null);
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "WebSocketNode: initialization and cleanup" {
    const allocator = std.testing.allocator;
    
    var ws = try WebSocketNode.init(allocator, "ws://localhost:8080");
    defer ws.deinit();
    
    try std.testing.expectEqualStrings("ws://localhost:8080", ws.url);
    try std.testing.expect(ws.auto_reconnect);
    try std.testing.expectEqual(@as(u32, 1000), ws.reconnect_delay_ms);
}

test "WebSocketNode: set headers" {
    const allocator = std.testing.allocator;
    
    var ws = try WebSocketNode.init(allocator, "ws://localhost:8080");
    defer ws.deinit();
    
    try ws.setHeader("Authorization", "Bearer token123");
    try ws.setHeader("X-Custom-Header", "value");
    
    try std.testing.expectEqual(@as(usize, 2), ws.headers.count());
}

test "WebSocketNode: message queue" {
    const allocator = std.testing.allocator;
    
    var ws = try WebSocketNode.init(allocator, "ws://localhost:8080");
    defer ws.deinit();
    
    try ws.send("Hello");
    try ws.send("World");
    
    try std.testing.expectEqual(@as(usize, 2), ws.message_queue.items.len);
    
    const msg = try ws.receive();
    try std.testing.expect(msg != null);
    try std.testing.expectEqualStrings("Hello", msg.?);
}

test "GraphQLNode: initialization" {
    const allocator = std.testing.allocator;
    
    var gql = try GraphQLNode.init(
        allocator,
        "https://api.example.com/graphql",
        .query,
        "{ user(id: \"1\") { name email } }",
    );
    defer gql.deinit();
    
    try std.testing.expectEqualStrings("https://api.example.com/graphql", gql.endpoint);
    try std.testing.expectEqual(GraphQLOperation.query, gql.operation_type);
}

test "GraphQLNode: set variables" {
    const allocator = std.testing.allocator;
    
    var gql = try GraphQLNode.init(
        allocator,
        "https://api.example.com/graphql",
        .mutation,
        "mutation($name: String!) { createUser(name: $name) { id } }",
    );
    defer gql.deinit();
    
    try gql.setVariable("name", "John Doe");
    
    try std.testing.expectEqual(@as(usize, 1), gql.variables.count());
}

test "GraphQLNode: build request" {
    const allocator = std.testing.allocator;
    
    var gql = try GraphQLNode.init(
        allocator,
        "https://api.example.com/graphql",
        .query,
        "{ users { id } }",
    );
    defer gql.deinit();
    
    const request = try gql.buildRequest();
    defer allocator.free(request);
    
    try std.testing.expect(std.mem.indexOf(u8, request, "query") != null);
    try std.testing.expect(std.mem.indexOf(u8, request, "{ users { id } }") != null);
}

test "GraphQLNode: build request with variables" {
    const allocator = std.testing.allocator;
    
    var gql = try GraphQLNode.init(
        allocator,
        "https://api.example.com/graphql",
        .query,
        "query($id: ID!) { user(id: $id) { name } }",
    );
    defer gql.deinit();
    
    try gql.setVariable("id", "123");
    
    const request = try gql.buildRequest();
    defer allocator.free(request);
    
    try std.testing.expect(std.mem.indexOf(u8, request, "variables") != null);
    try std.testing.expect(std.mem.indexOf(u8, request, "123") != null);
}

test "RESTClientNode: initialization" {
    const allocator = std.testing.allocator;
    
    var client = try RESTClientNode.init(allocator, "https://api.example.com");
    defer client.deinit();
    
    try std.testing.expectEqualStrings("https://api.example.com", client.base_url);
    try std.testing.expectEqual(@as(u32, 30000), client.timeout_ms);
}

test "RESTClientNode: set default headers" {
    const allocator = std.testing.allocator;
    
    var client = try RESTClientNode.init(allocator, "https://api.example.com");
    defer client.deinit();
    
    try client.setDefaultHeader("Authorization", "Bearer token");
    try client.setDefaultHeader("Content-Type", "application/json");
    
    try std.testing.expectEqual(@as(usize, 2), client.default_headers.count());
}

test "RESTClientNode: build URL" {
    const allocator = std.testing.allocator;
    
    var client = try RESTClientNode.init(allocator, "https://api.example.com");
    defer client.deinit();
    
    const url1 = try client.buildUrl("/users");
    defer allocator.free(url1);
    try std.testing.expectEqualStrings("https://api.example.com/users", url1);
    
    const url2 = try client.buildUrl("users");
    defer allocator.free(url2);
    try std.testing.expectEqualStrings("https://api.example.com/users", url2);
}

test "RESTClientNode: GET request" {
    const allocator = std.testing.allocator;
    
    var client = try RESTClientNode.init(allocator, "https://api.example.com");
    defer client.deinit();
    
    const response = try client.get("/users");
    defer allocator.free(response);
    
    try std.testing.expect(response.len > 0);
}

test "RESTClientNode: POST request" {
    const allocator = std.testing.allocator;
    
    var client = try RESTClientNode.init(allocator, "https://api.example.com");
    defer client.deinit();
    
    const response = try client.post("/users", "{\"name\":\"John\"}");
    defer allocator.free(response);
    
    try std.testing.expect(response.len > 0);
}

test "HttpMethod: toString" {
    try std.testing.expectEqualStrings("GET", HttpMethod.GET.toString());
    try std.testing.expectEqualStrings("POST", HttpMethod.POST.toString());
    try std.testing.expectEqualStrings("PUT", HttpMethod.PUT.toString());
    try std.testing.expectEqualStrings("DELETE", HttpMethod.DELETE.toString());
}
