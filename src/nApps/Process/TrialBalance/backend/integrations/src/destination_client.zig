const std = @import("std");

/// SAP Destination Service Client
/// Handles communication with nServices through SAP BTP Destinations

pub const DestinationType = enum {
    AGENT_FLOW,
    AGENT_META,
    LOCAL_MODELS,
    GROUNDING,
    HANA_CLOUD,
};

pub const DestinationClient = struct {
    allocator: std.mem.Allocator,
    base_url: []const u8,
    auth_token: ?[]const u8,
    client: std.http.Client,

    pub fn init(allocator: std.mem.Allocator, destination_type: DestinationType) !DestinationClient {
        const base_url = switch (destination_type) {
            .AGENT_FLOW => "http://localhost:8090/api/v1", // Mock for local dev
            .AGENT_META => "http://localhost:8091/api/v1",
            .LOCAL_MODELS => "http://localhost:8006", // Matching llm_http.zig port
            .GROUNDING => "http://localhost:8093/api/v1",
            .HANA_CLOUD => "http://localhost:8094/api/v1/data",
        };

        return .{
            .allocator = allocator,
            .base_url = base_url,
            .auth_token = null,
            .client = std.http.Client.init(allocator, .{}),
        };
    }

    pub fn deinit(self: *DestinationClient) void {
        if (self.auth_token) |token| {
            self.allocator.free(token);
        }
        self.client.deinit();
    }

    /// Authenticate and get token
    pub fn authenticate(self: *DestinationClient) !void {
        // For local dev/mock, we just set a dummy token
        // In prod, this would call XSUAA
        if (self.auth_token) |token| self.allocator.free(token);
        self.auth_token = try self.allocator.dupe(u8, "dummy-token");
    }

    /// Make HTTP GET request
    pub fn get(self: *DestinationClient, path: []const u8) ![]u8 {
        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, path });
        defer self.allocator.free(url);

        const uri = try std.Uri.parse(url);
        var header_buffer: [4096]u8 = undefined;
        var request = try self.client.open(.GET, uri, .{ .server_header_buffer = &header_buffer });
        defer request.deinit();

        if (self.auth_token) |token| {
             var auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
             defer self.allocator.free(auth_header);
             try request.headers.append("Authorization", auth_header);
        }

        try request.send();
        try request.wait();

        if (request.response.status != .ok) {
            return error.RequestFailed;
        }

        const body = try request.reader().readAllAlloc(self.allocator, 10 * 1024 * 1024);
        return body;
    }

    /// Make HTTP POST request
    pub fn post(self: *DestinationClient, path: []const u8, body: []const u8) ![]u8 {
        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, path });
        defer self.allocator.free(url);

        const uri = try std.Uri.parse(url);
        var header_buffer: [4096]u8 = undefined;
        var request = try self.client.open(.POST, uri, .{ .server_header_buffer = &header_buffer });
        defer request.deinit();

        request.transfer_encoding = .chunked;
        try request.headers.append("Content-Type", "application/json");
        if (self.auth_token) |token| {
             var auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
             defer self.allocator.free(auth_header);
             try request.headers.append("Authorization", auth_header);
        }

        try request.send();
        try request.writeAll(body);
        try request.finish();
        try request.wait();

        if (request.response.status != .ok) {
            return error.RequestFailed;
        }

        const response_body = try request.reader().readAllAlloc(self.allocator, 10 * 1024 * 1024);
        return response_body;
    }
};

/// nAgentFlow Service Client
pub const AgentFlowClient = struct {
    client: DestinationClient,

    pub fn init(allocator: std.mem.Allocator) !AgentFlowClient {
        return .{
            .client = try DestinationClient.init(allocator, .AGENT_FLOW),
        };
    }

    pub fn deinit(self: *AgentFlowClient) void {
        self.client.deinit();
    }

    /// Start workflow for trial balance entry
    pub fn startWorkflow(self: *AgentFlowClient, workflow_type: []const u8, data: []const u8) ![]u8 {
        return self.client.post("/workflows/execute", data);
    }
};

/// nLocalModels Service Client
pub const LocalModelsClient = struct {
    client: DestinationClient,

    pub fn init(allocator: std.mem.Allocator) !LocalModelsClient {
        return .{
            .client = try DestinationClient.init(allocator, .LOCAL_MODELS),
        };
    }

    pub fn deinit(self: *LocalModelsClient) void {
        self.client.deinit();
    }

    /// Generate AI narrative for trial balance
    /// Uses the /extract-workflow endpoint as a proxy for generation or a custom chat endpoint
    pub fn generateNarrative(self: *LocalModelsClient, data: []const u8) ![]u8 {
        // Construct a prompt payload
        // In a real scenario, we might use a different endpoint or format
        const prompt_struct = struct {
            markdown: []const u8, // reusing the field name expected by llm_http.zig
            temperature: f32 = 0.7,
        };
        
        // Wrap the data in a narrative request
        const prompt = try std.fmt.allocPrint(self.client.allocator, "Analyze this trial balance data and provide a narrative summary of variances: {s}", .{data});
        defer self.client.allocator.free(prompt);

        const payload = prompt_struct{ .markdown = prompt };
        const json_body = try std.json.stringifyAlloc(self.client.allocator, payload, .{});
        defer self.client.allocator.free(json_body);

        return self.client.post("/extract-workflow", json_body);
    }

    /// Get AI suggestions for reconciliation
    pub fn getSuggestions(self: *LocalModelsClient, account_data: []const u8) ![]u8 {
         const prompt = try std.fmt.allocPrint(self.client.allocator, "Suggest reconciliation steps for: {s}", .{account_data});
        defer self.client.allocator.free(prompt);
        
        const payload = struct { markdown: []const u8, temperature: f32 = 0.5 }{ .markdown = prompt };
        const json_body = try std.json.stringifyAlloc(self.client.allocator, payload, .{});
        defer self.client.allocator.free(json_body);

        return self.client.post("/extract-workflow", json_body);
    }
};

test "DestinationClient init" {
    const allocator = std.testing.allocator;
    var client = try DestinationClient.init(allocator, .AGENT_FLOW);
    defer client.deinit();
}