//! nCode Client Library for Zig
//! Provides a complete API client for the nCode SCIP-based code intelligence platform
//!
//! Usage:
//!   const client = try NCodeClient.init(allocator, "http://localhost:18003");
//!   defer client.deinit();
//!   const symbols = try client.getSymbols(file_path);

const std = @import("std");
const http = std.http;
const json = std.json;
const Allocator = std.mem.Allocator;

/// Main nCode client for interacting with the API
pub const NCodeClient = struct {
    allocator: Allocator,
    base_url: []const u8,
    http_client: http.Client,
    timeout_ms: u64,

    pub const Config = struct {
        base_url: []const u8 = "http://localhost:18003",
        timeout_ms: u64 = 30000,
    };

    /// Initialize a new nCode client
    pub fn init(allocator: Allocator, config: Config) !*NCodeClient {
        const client = try allocator.create(NCodeClient);
        errdefer allocator.destroy(client);

        const base_url = try allocator.dupe(u8, config.base_url);
        errdefer allocator.free(base_url);

        client.* = .{
            .allocator = allocator,
            .base_url = base_url,
            .http_client = http.Client{ .allocator = allocator },
            .timeout_ms = config.timeout_ms,
        };

        return client;
    }

    pub fn deinit(self: *NCodeClient) void {
        self.allocator.free(self.base_url);
        self.http_client.deinit();
        self.allocator.destroy(self);
    }

    /// Health check endpoint
    pub fn health(self: *NCodeClient) !HealthResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/health", .{self.base_url});
        defer self.allocator.free(url);

        var response = try self.makeRequest(.GET, url, null);
        defer response.deinit();

        const body = try response.readAll(self.allocator);
        defer self.allocator.free(body);

        const parsed = try json.parseFromSlice(HealthResponse, self.allocator, body, .{});
        defer parsed.deinit();

        return parsed.value;
    }

    /// Load a SCIP index file
    pub fn loadIndex(self: *NCodeClient, scip_path: []const u8) !LoadIndexResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/index/load", .{self.base_url});
        defer self.allocator.free(url);

        const payload = try std.fmt.allocPrint(
            self.allocator,
            "{{\"path\":\"{s}\"}}",
            .{scip_path},
        );
        defer self.allocator.free(payload);

        var response = try self.makeRequest(.POST, url, payload);
        defer response.deinit();

        const body = try response.readAll(self.allocator);
        defer self.allocator.free(body);

        const parsed = try json.parseFromSlice(LoadIndexResponse, self.allocator, body, .{});
        defer parsed.deinit();

        return parsed.value;
    }

    /// Find the definition of a symbol
    pub fn findDefinition(self: *NCodeClient, request: DefinitionRequest) !DefinitionResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/definition", .{self.base_url});
        defer self.allocator.free(url);

        const payload = try json.stringifyAlloc(self.allocator, request, .{});
        defer self.allocator.free(payload);

        var response = try self.makeRequest(.POST, url, payload);
        defer response.deinit();

        const body = try response.readAll(self.allocator);
        defer self.allocator.free(body);

        const parsed = try json.parseFromSlice(DefinitionResponse, self.allocator, body, .{});
        defer parsed.deinit();

        return parsed.value;
    }

    /// Find all references to a symbol
    pub fn findReferences(self: *NCodeClient, request: ReferencesRequest) !ReferencesResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/references", .{self.base_url});
        defer self.allocator.free(url);

        const payload = try json.stringifyAlloc(self.allocator, request, .{});
        defer self.allocator.free(payload);

        var response = try self.makeRequest(.POST, url, payload);
        defer response.deinit();

        const body = try response.readAll(self.allocator);
        defer self.allocator.free(body);

        const parsed = try json.parseFromSlice(ReferencesResponse, self.allocator, body, .{});
        defer parsed.deinit();

        return parsed.value;
    }

    /// Get hover information for a symbol
    pub fn getHover(self: *NCodeClient, request: HoverRequest) !HoverResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/hover", .{self.base_url});
        defer self.allocator.free(url);

        const payload = try json.stringifyAlloc(self.allocator, request, .{});
        defer self.allocator.free(payload);

        var response = try self.makeRequest(.POST, url, payload);
        defer response.deinit();

        const body = try response.readAll(self.allocator);
        defer self.allocator.free(body);

        const parsed = try json.parseFromSlice(HoverResponse, self.allocator, body, .{});
        defer parsed.deinit();

        return parsed.value;
    }

    /// Get all symbols in a file
    pub fn getSymbols(self: *NCodeClient, file_path: []const u8) !SymbolsResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/symbols", .{self.base_url});
        defer self.allocator.free(url);

        const payload = try std.fmt.allocPrint(
            self.allocator,
            "{{\"file\":\"{s}\"}}",
            .{file_path},
        );
        defer self.allocator.free(payload);

        var response = try self.makeRequest(.POST, url, payload);
        defer response.deinit();

        const body = try response.readAll(self.allocator);
        defer self.allocator.free(body);

        const parsed = try json.parseFromSlice(SymbolsResponse, self.allocator, body, .{});
        defer parsed.deinit();

        return parsed.value;
    }

    /// Get document outline (hierarchical symbol tree)
    pub fn getDocumentSymbols(self: *NCodeClient, file_path: []const u8) !DocumentSymbolsResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/v1/document-symbols", .{self.base_url});
        defer self.allocator.free(url);

        const payload = try std.fmt.allocPrint(
            self.allocator,
            "{{\"file\":\"{s}\"}}",
            .{file_path},
        );
        defer self.allocator.free(payload);

        var response = try self.makeRequest(.POST, url, payload);
        defer response.deinit();

        const body = try response.readAll(self.allocator);
        defer self.allocator.free(body);

        const parsed = try json.parseFromSlice(DocumentSymbolsResponse, self.allocator, body, .{});
        defer parsed.deinit();

        return parsed.value;
    }

    // Private helper to make HTTP requests
    fn makeRequest(self: *NCodeClient, method: http.Method, url: []const u8, body: ?[]const u8) !http.Client.Response {
        const uri = try std.Uri.parse(url);
        
        var headers = http.Headers{ .allocator = self.allocator };
        defer headers.deinit();
        
        try headers.append("Content-Type", "application/json");
        try headers.append("Accept", "application/json");

        const request = try self.http_client.request(method, uri, headers, .{});
        defer request.deinit();

        if (body) |b| {
            try request.writer().writeAll(b);
        }

        try request.finish();
        try request.wait();

        return request.response;
    }
};

// Response types

pub const HealthResponse = struct {
    status: []const u8,
    version: []const u8,
    uptime_seconds: ?f64 = null,
    index_loaded: ?bool = null,
};

pub const LoadIndexResponse = struct {
    success: bool,
    message: []const u8,
    documents: ?usize = null,
    symbols: ?usize = null,
};

pub const Position = struct {
    line: i32,
    character: i32,
};

pub const Range = struct {
    start: Position,
    end: Position,
};

pub const Location = struct {
    uri: []const u8,
    range: Range,
};

pub const DefinitionRequest = struct {
    file: []const u8,
    line: i32,
    character: i32,
};

pub const DefinitionResponse = struct {
    location: ?Location = null,
    symbol: ?[]const u8 = null,
};

pub const ReferencesRequest = struct {
    file: []const u8,
    line: i32,
    character: i32,
    include_declaration: bool = true,
};

pub const ReferencesResponse = struct {
    locations: []Location,
    symbol: ?[]const u8 = null,
};

pub const HoverRequest = struct {
    file: []const u8,
    line: i32,
    character: i32,
};

pub const HoverResponse = struct {
    contents: []const u8,
    range: ?Range = null,
};

pub const SymbolInfo = struct {
    name: []const u8,
    kind: []const u8,
    range: Range,
    detail: ?[]const u8 = null,
};

pub const SymbolsResponse = struct {
    symbols: []SymbolInfo,
    file: []const u8,
};

pub const DocumentSymbol = struct {
    name: []const u8,
    kind: []const u8,
    range: Range,
    selection_range: Range,
    children: ?[]DocumentSymbol = null,
};

pub const DocumentSymbolsResponse = struct {
    symbols: []DocumentSymbol,
    file: []const u8,
};

// Database query helpers

pub const QdrantClient = struct {
    allocator: Allocator,
    base_url: []const u8,
    collection_name: []const u8,

    pub fn init(allocator: Allocator, base_url: []const u8, collection_name: []const u8) !*QdrantClient {
        const client = try allocator.create(QdrantClient);
        client.* = .{
            .allocator = allocator,
            .base_url = try allocator.dupe(u8, base_url),
            .collection_name = try allocator.dupe(u8, collection_name),
        };
        return client;
    }

    pub fn deinit(self: *QdrantClient) void {
        self.allocator.free(self.base_url);
        self.allocator.free(self.collection_name);
        self.allocator.destroy(self);
    }

    pub fn semanticSearch(self: *QdrantClient, query: []const u8, limit: usize) ![]const u8 {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/collections/{s}/points/search",
            .{ self.base_url, self.collection_name },
        );
        defer self.allocator.free(url);

        // This is a simplified version - in production you'd embed the query
        const payload = try std.fmt.allocPrint(
            self.allocator,
            "{{\"vector\":[],\"limit\":{d},\"with_payload\":true}}",
            .{limit},
        );
        defer self.allocator.free(payload);

        // Make HTTP request (simplified)
        return try self.allocator.dupe(u8, "{}");
    }
};

pub const MemgraphClient = struct {
    allocator: Allocator,
    connection_string: []const u8,

    pub fn init(allocator: Allocator, connection_string: []const u8) !*MemgraphClient {
        const client = try allocator.create(MemgraphClient);
        client.* = .{
            .allocator = allocator,
            .connection_string = try allocator.dupe(u8, connection_string),
        };
        return client;
    }

    pub fn deinit(self: *MemgraphClient) void {
        self.allocator.free(self.connection_string);
        self.allocator.destroy(self);
    }

    pub fn findDefinitions(self: *MemgraphClient, symbol_name: []const u8) ![]const u8 {
        _ = self;
        _ = symbol_name;
        // Simplified - in production would use Bolt protocol
        return "[]";
    }

    pub fn findReferences(self: *MemgraphClient, symbol_name: []const u8) ![]const u8 {
        _ = self;
        _ = symbol_name;
        // Simplified - in production would use Bolt protocol
        return "[]";
    }

    pub fn getCallGraph(self: *MemgraphClient, function_name: []const u8, depth: usize) ![]const u8 {
        _ = self;
        _ = function_name;
        _ = depth;
        // Simplified - in production would use Bolt protocol
        return "[]";
    }
};

// Example usage
pub fn example() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create client
    const client = try NCodeClient.init(allocator, .{
        .base_url = "http://localhost:18003",
        .timeout_ms = 30000,
    });
    defer client.deinit();

    // Check health
    const health_result = try client.health();
    std.debug.print("Health: {s}\n", .{health_result.status});

    // Load index
    const load_result = try client.loadIndex("index.scip");
    std.debug.print("Loaded: {}\n", .{load_result.success});

    // Find definition
    const def_request = DefinitionRequest{
        .file = "src/main.zig",
        .line = 10,
        .character = 5,
    };
    const def_result = try client.findDefinition(def_request);
    if (def_result.symbol) |sym| {
        std.debug.print("Symbol: {s}\n", .{sym});
    }

    // Get symbols
    const symbols = try client.getSymbols("src/main.zig");
    std.debug.print("Found {d} symbols\n", .{symbols.symbols.len});
}
