// LSP Testing Infrastructure
// Day 77: Test framework for LSP implementation

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// JSON-RPC Message Types
// ============================================================================

pub const MessageType = enum {
    Request,
    Response,
    Notification,
};

pub const JsonRpcMessage = struct {
    jsonrpc: []const u8 = "2.0",
    id: ?i64 = null,
    method: ?[]const u8 = null,
    params: ?[]const u8 = null,
    result: ?[]const u8 = null,
    error_info: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn deinit(self: *JsonRpcMessage) void {
        if (self.method) |m| self.allocator.free(m);
        if (self.params) |p| self.allocator.free(p);
        if (self.result) |r| self.allocator.free(r);
        if (self.error_info) |e| self.allocator.free(e);
    }
};

// ============================================================================
// Mock LSP Client
// ============================================================================

pub const MockLspClient = struct {
    sent_messages: std.ArrayList(JsonRpcMessage),
    received_messages: std.ArrayList(JsonRpcMessage),
    next_id: i64 = 1,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) MockLspClient {
        return MockLspClient{
            .sent_messages = std.ArrayList(JsonRpcMessage){},
            .received_messages = std.ArrayList(JsonRpcMessage){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MockLspClient) void {
        for (self.sent_messages.items) |*msg| {
            msg.deinit();
        }
        self.sent_messages.deinit(self.allocator);
        
        for (self.received_messages.items) |*msg| {
            msg.deinit();
        }
        self.received_messages.deinit(self.allocator);
    }
    
    /// Send a request to the server
    pub fn sendRequest(self: *MockLspClient, method: []const u8, params: ?[]const u8) !i64 {
        const id = self.next_id;
        self.next_id += 1;
        
        const msg = JsonRpcMessage{
            .id = id,
            .method = try self.allocator.dupe(u8, method),
            .params = if (params) |p| try self.allocator.dupe(u8, p) else null,
            .allocator = self.allocator,
        };
        
        try self.sent_messages.append(self.allocator, msg);
        return id;
    }
    
    /// Send a notification to the server
    pub fn sendNotification(self: *MockLspClient, method: []const u8, params: ?[]const u8) !void {
        const msg = JsonRpcMessage{
            .method = try self.allocator.dupe(u8, method),
            .params = if (params) |p| try self.allocator.dupe(u8, p) else null,
            .allocator = self.allocator,
        };
        
        try self.sent_messages.append(self.allocator, msg);
    }
    
    /// Receive a response from the server
    pub fn receiveResponse(self: *MockLspClient, id: i64, result: []const u8) !void {
        const msg = JsonRpcMessage{
            .id = id,
            .result = try self.allocator.dupe(u8, result),
            .allocator = self.allocator,
        };
        
        try self.received_messages.append(self.allocator, msg);
    }
    
    /// Get number of sent messages
    pub fn getSentCount(self: *MockLspClient) usize {
        return self.sent_messages.items.len;
    }
    
    /// Get number of received messages
    pub fn getReceivedCount(self: *MockLspClient) usize {
        return self.received_messages.items.len;
    }
    
    /// Clear all messages
    pub fn clear(self: *MockLspClient) void {
        for (self.sent_messages.items) |*msg| {
            msg.deinit();
        }
        self.sent_messages.clearRetainingCapacity();
        
        for (self.received_messages.items) |*msg| {
            msg.deinit();
        }
        self.received_messages.clearRetainingCapacity();
    }
};

// ============================================================================
// Test Message Builders
// ============================================================================

pub const TestMessageBuilder = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) TestMessageBuilder {
        return TestMessageBuilder{ .allocator = allocator };
    }
    
    /// Build initialize request
    pub fn buildInitialize(self: TestMessageBuilder) ![]const u8 {
        return try std.fmt.allocPrint(self.allocator,
            \\{{
            \\  "processId": 1234,
            \\  "rootUri": "file:///test/workspace",
            \\  "capabilities": {{}}
            \\}}
        , .{});
    }
    
    /// Build textDocument/didOpen notification
    pub fn buildDidOpen(self: TestMessageBuilder, uri: []const u8, content: []const u8) ![]const u8 {
        return try std.fmt.allocPrint(self.allocator,
            \\{{
            \\  "textDocument": {{
            \\    "uri": "{s}",
            \\    "languageId": "mojo",
            \\    "version": 1,
            \\    "text": "{s}"
            \\  }}
            \\}}
        , .{ uri, content });
    }
    
    /// Build textDocument/didChange notification
    pub fn buildDidChange(self: TestMessageBuilder, uri: []const u8, version: i32, content: []const u8) ![]const u8 {
        return try std.fmt.allocPrint(self.allocator,
            \\{{
            \\  "textDocument": {{
            \\    "uri": "{s}",
            \\    "version": {d}
            \\  }},
            \\  "contentChanges": [
            \\    {{ "text": "{s}" }}
            \\  ]
            \\}}
        , .{ uri, version, content });
    }
    
    /// Build textDocument/documentSymbol request
    pub fn buildDocumentSymbol(self: TestMessageBuilder, uri: []const u8) ![]const u8 {
        return try std.fmt.allocPrint(self.allocator,
            \\{{
            \\  "textDocument": {{
            \\    "uri": "{s}"
            \\  }}
            \\}}
        , .{uri});
    }
};

// ============================================================================
// Response Validators
// ============================================================================

pub const ResponseValidator = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ResponseValidator {
        return ResponseValidator{ .allocator = allocator };
    }
    
    /// Validate response has required fields
    pub fn validateResponse(self: ResponseValidator, response: []const u8) !bool {
        _ = self;
        // Simple validation: check if response contains "result" or "error"
        return std.mem.indexOf(u8, response, "result") != null or
            std.mem.indexOf(u8, response, "error") != null;
    }
    
    /// Validate initialize response
    pub fn validateInitializeResponse(self: ResponseValidator, response: []const u8) !bool {
        _ = self;
        return std.mem.indexOf(u8, response, "capabilities") != null;
    }
    
    /// Validate diagnostics notification
    pub fn validateDiagnostics(self: ResponseValidator, notification: []const u8) !bool {
        _ = self;
        return std.mem.indexOf(u8, notification, "textDocument/publishDiagnostics") != null and
            std.mem.indexOf(u8, notification, "diagnostics") != null;
    }
};

// ============================================================================
// Snapshot Testing
// ============================================================================

pub const Snapshot = struct {
    name: []const u8,
    content: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, content: []const u8) !Snapshot {
        return Snapshot{
            .name = try allocator.dupe(u8, name),
            .content = try allocator.dupe(u8, content),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Snapshot) void {
        self.allocator.free(self.name);
        self.allocator.free(self.content);
    }
    
    /// Compare snapshot with actual output
    pub fn compare(self: *Snapshot, actual: []const u8) bool {
        return std.mem.eql(u8, self.content, actual);
    }
};

pub const SnapshotStore = struct {
    snapshots: std.StringHashMap(Snapshot),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) SnapshotStore {
        return SnapshotStore{
            .snapshots = std.StringHashMap(Snapshot).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SnapshotStore) void {
        var iter = self.snapshots.iterator();
        while (iter.next()) |entry| {
            var snapshot = entry.value_ptr;
            snapshot.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.snapshots.deinit();
    }
    
    /// Add or update snapshot
    pub fn addSnapshot(self: *SnapshotStore, name: []const u8, content: []const u8) !void {
        const snapshot = try Snapshot.init(self.allocator, name, content);
        const key = try self.allocator.dupe(u8, name);
        try self.snapshots.put(key, snapshot);
    }
    
    /// Compare with snapshot
    pub fn compareWithSnapshot(self: *SnapshotStore, name: []const u8, actual: []const u8) !bool {
        if (self.snapshots.getPtr(name)) |snapshot| {
            return snapshot.compare(actual);
        }
        return error.SnapshotNotFound;
    }
};

// ============================================================================
// Integration Test Framework
// ============================================================================

pub const TestResult = struct {
    name: []const u8,
    passed: bool,
    message: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn deinit(self: *TestResult) void {
        self.allocator.free(self.name);
        if (self.message) |msg| {
            self.allocator.free(msg);
        }
    }
};

pub const IntegrationTestRunner = struct {
    results: std.ArrayList(TestResult),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) IntegrationTestRunner {
        return IntegrationTestRunner{
            .results = std.ArrayList(TestResult){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *IntegrationTestRunner) void {
        for (self.results.items) |*result| {
            result.deinit();
        }
        self.results.deinit(self.allocator);
    }
    
    /// Record test result
    pub fn recordResult(self: *IntegrationTestRunner, name: []const u8, passed: bool, message: ?[]const u8) !void {
        const result = TestResult{
            .name = try self.allocator.dupe(u8, name),
            .passed = passed,
            .message = if (message) |msg| try self.allocator.dupe(u8, msg) else null,
            .allocator = self.allocator,
        };
        
        try self.results.append(self.allocator, result);
    }
    
    /// Get pass count
    pub fn getPassCount(self: *IntegrationTestRunner) usize {
        var count: usize = 0;
        for (self.results.items) |result| {
            if (result.passed) count += 1;
        }
        return count;
    }
    
    /// Get fail count
    pub fn getFailCount(self: *IntegrationTestRunner) usize {
        var count: usize = 0;
        for (self.results.items) |result| {
            if (!result.passed) count += 1;
        }
        return count;
    }
    
    /// Get total count
    pub fn getTotalCount(self: *IntegrationTestRunner) usize {
        return self.results.items.len;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "MockLspClient: send request" {
    var client = MockLspClient.init(std.testing.allocator);
    defer client.deinit();
    
    const id = try client.sendRequest("initialize", null);
    
    try std.testing.expectEqual(@as(i64, 1), id);
    try std.testing.expectEqual(@as(usize, 1), client.getSentCount());
}

test "MockLspClient: send notification" {
    var client = MockLspClient.init(std.testing.allocator);
    defer client.deinit();
    
    try client.sendNotification("initialized", null);
    
    try std.testing.expectEqual(@as(usize, 1), client.getSentCount());
}

test "MockLspClient: receive response" {
    var client = MockLspClient.init(std.testing.allocator);
    defer client.deinit();
    
    try client.receiveResponse(1, "{}");
    
    try std.testing.expectEqual(@as(usize, 1), client.getReceivedCount());
}

test "MockLspClient: clear messages" {
    var client = MockLspClient.init(std.testing.allocator);
    defer client.deinit();
    
    _ = try client.sendRequest("test", null);
    try client.receiveResponse(1, "{}");
    
    client.clear();
    
    try std.testing.expectEqual(@as(usize, 0), client.getSentCount());
    try std.testing.expectEqual(@as(usize, 0), client.getReceivedCount());
}

test "TestMessageBuilder: initialize" {
    const builder = TestMessageBuilder.init(std.testing.allocator);
    
    const msg = try builder.buildInitialize();
    defer std.testing.allocator.free(msg);
    
    try std.testing.expect(std.mem.indexOf(u8, msg, "processId") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "rootUri") != null);
}

test "TestMessageBuilder: didOpen" {
    const builder = TestMessageBuilder.init(std.testing.allocator);
    
    const msg = try builder.buildDidOpen("file:///test.mojo", "fn main() {}");
    defer std.testing.allocator.free(msg);
    
    try std.testing.expect(std.mem.indexOf(u8, msg, "textDocument") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "file:///test.mojo") != null);
}

test "TestMessageBuilder: didChange" {
    const builder = TestMessageBuilder.init(std.testing.allocator);
    
    const msg = try builder.buildDidChange("file:///test.mojo", 2, "fn main() { print() }");
    defer std.testing.allocator.free(msg);
    
    try std.testing.expect(std.mem.indexOf(u8, msg, "contentChanges") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "version") != null);
}

test "TestMessageBuilder: documentSymbol" {
    const builder = TestMessageBuilder.init(std.testing.allocator);
    
    const msg = try builder.buildDocumentSymbol("file:///test.mojo");
    defer std.testing.allocator.free(msg);
    
    try std.testing.expect(std.mem.indexOf(u8, msg, "textDocument") != null);
}

test "ResponseValidator: validate response" {
    const validator = ResponseValidator.init(std.testing.allocator);
    
    const valid_result = "{ \"result\": {} }";
    const valid_error = "{ \"error\": {} }";
    const invalid = "{ \"something\": {} }";
    
    try std.testing.expect(try validator.validateResponse(valid_result));
    try std.testing.expect(try validator.validateResponse(valid_error));
    try std.testing.expect(!try validator.validateResponse(invalid));
}

test "ResponseValidator: validate initialize" {
    const validator = ResponseValidator.init(std.testing.allocator);
    
    const valid = "{ \"result\": { \"capabilities\": {} } }";
    const invalid = "{ \"result\": {} }";
    
    try std.testing.expect(try validator.validateInitializeResponse(valid));
    try std.testing.expect(!try validator.validateInitializeResponse(invalid));
}

test "Snapshot: compare" {
    var snapshot = try Snapshot.init(std.testing.allocator, "test1", "expected output");
    defer snapshot.deinit();
    
    try std.testing.expect(snapshot.compare("expected output"));
    try std.testing.expect(!snapshot.compare("different output"));
}

test "SnapshotStore: add and compare" {
    var store = SnapshotStore.init(std.testing.allocator);
    defer store.deinit();
    
    try store.addSnapshot("test1", "expected output");
    
    try std.testing.expect(try store.compareWithSnapshot("test1", "expected output"));
    try std.testing.expect(!try store.compareWithSnapshot("test1", "different"));
}

test "IntegrationTestRunner: record results" {
    var runner = IntegrationTestRunner.init(std.testing.allocator);
    defer runner.deinit();
    
    try runner.recordResult("test1", true, null);
    try runner.recordResult("test2", false, "Error message");
    try runner.recordResult("test3", true, null);
    
    try std.testing.expectEqual(@as(usize, 3), runner.getTotalCount());
    try std.testing.expectEqual(@as(usize, 2), runner.getPassCount());
    try std.testing.expectEqual(@as(usize, 1), runner.getFailCount());
}

test "IntegrationTestRunner: empty" {
    var runner = IntegrationTestRunner.init(std.testing.allocator);
    defer runner.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), runner.getTotalCount());
    try std.testing.expectEqual(@as(usize, 0), runner.getPassCount());
    try std.testing.expectEqual(@as(usize, 0), runner.getFailCount());
}

test "JsonRpcMessage: lifecycle" {
    var msg = JsonRpcMessage{
        .id = 1,
        .method = try std.testing.allocator.dupe(u8, "test"),
        .params = try std.testing.allocator.dupe(u8, "{}"),
        .allocator = std.testing.allocator,
    };
    defer msg.deinit();
    
    try std.testing.expectEqual(@as(i64, 1), msg.id.?);
    try std.testing.expectEqualStrings("test", msg.method.?);
}
