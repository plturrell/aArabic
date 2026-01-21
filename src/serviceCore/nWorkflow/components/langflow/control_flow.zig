// Day 28: Langflow Component Parity - Control Flow Components
// If/Else, Switch, Loop, and Delay nodes for workflow control

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

/// Comparison operators for conditions
pub const CompareOp = enum {
    equals,
    not_equals,
    greater_than,
    less_than,
    greater_or_equal,
    less_or_equal,
    contains,
    starts_with,
    ends_with,

    pub fn toString(self: CompareOp) []const u8 {
        return switch (self) {
            .equals => "equals",
            .not_equals => "not_equals",
            .greater_than => "greater_than",
            .less_than => "less_than",
            .greater_or_equal => "greater_or_equal",
            .less_or_equal => "less_or_equal",
            .contains => "contains",
            .starts_with => "starts_with",
            .ends_with => "ends_with",
        };
    }
};

/// If/Else Node - Conditional branching
pub const IfElseNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    operator: CompareOp,
    compare_value: []const u8,

    pub fn init(allocator: Allocator, node_id: []const u8, operator: CompareOp, compare_value: []const u8) !*IfElseNode {
        const node = try allocator.create(IfElseNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .operator = operator,
            .compare_value = try allocator.dupe(u8, compare_value),
        };
        return node;
    }

    pub fn deinit(self: *IfElseNode) void {
        self.allocator.free(self.node_id);
        self.allocator.free(self.compare_value);
        self.allocator.destroy(self);
    }

    pub fn evaluate(self: *IfElseNode, input: []const u8) !bool {
        return switch (self.operator) {
            .equals => std.mem.eql(u8, input, self.compare_value),
            .not_equals => !std.mem.eql(u8, input, self.compare_value),
            .greater_than => try self.compareNumeric(input, .greater),
            .less_than => try self.compareNumeric(input, .less),
            .greater_or_equal => try self.compareNumeric(input, .greater_or_equal),
            .less_or_equal => try self.compareNumeric(input, .less_or_equal),
            .contains => std.mem.indexOf(u8, input, self.compare_value) != null,
            .starts_with => std.mem.startsWith(u8, input, self.compare_value),
            .ends_with => std.mem.endsWith(u8, input, self.compare_value),
        };
    }

    fn compareNumeric(self: *IfElseNode, input: []const u8, comptime op: enum { greater, less, greater_or_equal, less_or_equal }) !bool {
        const input_num = std.fmt.parseFloat(f64, input) catch return false;
        const compare_num = std.fmt.parseFloat(f64, self.compare_value) catch return false;

        return switch (op) {
            .greater => input_num > compare_num,
            .less => input_num < compare_num,
            .greater_or_equal => input_num >= compare_num,
            .less_or_equal => input_num <= compare_num,
        };
    }
};

/// Switch Node - Multi-way branching
pub const SwitchNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    cases: StringHashMap([]const u8), // value -> output_id
    default_output: ?[]const u8,

    pub fn init(allocator: Allocator, node_id: []const u8) !*SwitchNode {
        const node = try allocator.create(SwitchNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .cases = StringHashMap([]const u8).init(allocator),
            .default_output = null,
        };
        return node;
    }

    pub fn deinit(self: *SwitchNode) void {
        var iter = self.cases.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.cases.deinit();

        if (self.default_output) |default| {
            self.allocator.free(default);
        }

        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }

    pub fn addCase(self: *SwitchNode, value: []const u8, output_id: []const u8) !void {
        const key = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(key);
        const val = try self.allocator.dupe(u8, output_id);
        errdefer self.allocator.free(val);

        try self.cases.put(key, val);
    }

    pub fn setDefault(self: *SwitchNode, output_id: []const u8) !void {
        if (self.default_output) |default| {
            self.allocator.free(default);
        }
        self.default_output = try self.allocator.dupe(u8, output_id);
    }

    pub fn evaluate(self: *SwitchNode, input: []const u8) ?[]const u8 {
        if (self.cases.get(input)) |output_id| {
            return output_id;
        }
        return self.default_output;
    }
};

/// Loop Node - Iterate over collection
pub const LoopNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    max_iterations: usize,
    current_iteration: usize,

    pub fn init(allocator: Allocator, node_id: []const u8, max_iterations: usize) !*LoopNode {
        const node = try allocator.create(LoopNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .max_iterations = max_iterations,
            .current_iteration = 0,
        };
        return node;
    }

    pub fn deinit(self: *LoopNode) void {
        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }

    pub fn reset(self: *LoopNode) void {
        self.current_iteration = 0;
    }

    pub fn shouldContinue(self: *LoopNode) bool {
        return self.current_iteration < self.max_iterations;
    }

    pub fn next(self: *LoopNode) usize {
        const current = self.current_iteration;
        self.current_iteration += 1;
        return current;
    }

    pub fn executeLoop(self: *LoopNode, items: []const []const u8, callback: *const fn ([]const u8, usize) anyerror!void) !void {
        self.reset();
        for (items, 0..) |item, idx| {
            if (!self.shouldContinue()) break;
            try callback(item, idx);
            _ = self.next();
        }
    }
};

/// Delay Node - Add delay/throttle to workflow
pub const DelayNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    delay_ms: u64,

    pub fn init(allocator: Allocator, node_id: []const u8, delay_ms: u64) !*DelayNode {
        const node = try allocator.create(DelayNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .delay_ms = delay_ms,
        };
        return node;
    }

    pub fn deinit(self: *DelayNode) void {
        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }

    pub fn execute(self: *DelayNode) void {
        std.Thread.sleep(self.delay_ms * std.time.ns_per_ms);
    }

    pub fn executeAsync(self: *DelayNode) !void {
        // For async workflows - would integrate with event loop
        std.Thread.sleep(self.delay_ms * std.time.ns_per_ms);
    }
};

/// Retry Node - Retry failed operations
pub const RetryNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    max_retries: usize,
    backoff_ms: u64,
    current_attempt: usize,

    pub fn init(allocator: Allocator, node_id: []const u8, max_retries: usize, backoff_ms: u64) !*RetryNode {
        const node = try allocator.create(RetryNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .max_retries = max_retries,
            .backoff_ms = backoff_ms,
            .current_attempt = 0,
        };
        return node;
    }

    pub fn deinit(self: *RetryNode) void {
        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }

    pub fn reset(self: *RetryNode) void {
        self.current_attempt = 0;
    }

    pub fn shouldRetry(self: *RetryNode) bool {
        return self.current_attempt < self.max_retries;
    }

    pub fn executeWithRetry(self: *RetryNode, operation: *const fn () anyerror!void) !void {
        self.reset();

        while (self.shouldRetry()) {
            operation() catch |err| {
                self.current_attempt += 1;
                if (!self.shouldRetry()) {
                    return err;
                }
                // Exponential backoff
                const delay = self.backoff_ms * (@as(u64, 1) << @intCast(self.current_attempt - 1));
                std.Thread.sleep(delay * std.time.ns_per_ms);
                continue;
            };
            return; // Success
        }
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "IfElseNode - equals" {
    const allocator = std.testing.allocator;

    var node = try IfElseNode.init(allocator, "if-1", .equals, "test");
    defer node.deinit();

    try std.testing.expect(try node.evaluate("test"));
    try std.testing.expect(!try node.evaluate("other"));
}

test "IfElseNode - greater than" {
    const allocator = std.testing.allocator;

    var node = try IfElseNode.init(allocator, "if-2", .greater_than, "10");
    defer node.deinit();

    try std.testing.expect(try node.evaluate("15"));
    try std.testing.expect(!try node.evaluate("5"));
}

test "IfElseNode - contains" {
    const allocator = std.testing.allocator;

    var node = try IfElseNode.init(allocator, "if-3", .contains, "world");
    defer node.deinit();

    try std.testing.expect(try node.evaluate("hello world"));
    try std.testing.expect(!try node.evaluate("hello"));
}

test "IfElseNode - starts with" {
    const allocator = std.testing.allocator;

    var node = try IfElseNode.init(allocator, "if-4", .starts_with, "hello");
    defer node.deinit();

    try std.testing.expect(try node.evaluate("hello world"));
    try std.testing.expect(!try node.evaluate("world hello"));
}

test "SwitchNode - basic cases" {
    const allocator = std.testing.allocator;

    var node = try SwitchNode.init(allocator, "switch-1");
    defer node.deinit();

    try node.addCase("red", "output-1");
    try node.addCase("green", "output-2");
    try node.addCase("blue", "output-3");
    try node.setDefault("output-default");

    try std.testing.expectEqualStrings("output-1", node.evaluate("red").?);
    try std.testing.expectEqualStrings("output-2", node.evaluate("green").?);
    try std.testing.expectEqualStrings("output-default", node.evaluate("yellow").?);
}

test "LoopNode - iteration control" {
    const allocator = std.testing.allocator;

    var node = try LoopNode.init(allocator, "loop-1", 5);
    defer node.deinit();

    try std.testing.expect(node.shouldContinue());
    try std.testing.expectEqual(@as(usize, 0), node.next());
    try std.testing.expectEqual(@as(usize, 1), node.next());

    node.reset();
    try std.testing.expectEqual(@as(usize, 0), node.current_iteration);
}

test "LoopNode - execute with items" {
    const allocator = std.testing.allocator;

    var node = try LoopNode.init(allocator, "loop-2", 10);
    defer node.deinit();

    const items = [_][]const u8{ "a", "b", "c" };

    const callback = struct {
        fn call(item: []const u8, idx: usize) !void {
            _ = item;
            _ = idx;
        }
    }.call;

    try node.executeLoop(&items, &callback);
}

test "DelayNode - basic delay" {
    const allocator = std.testing.allocator;

    var node = try DelayNode.init(allocator, "delay-1", 10);
    defer node.deinit();

    const start = std.time.milliTimestamp();
    node.execute();
    const end = std.time.milliTimestamp();

    try std.testing.expect((end - start) >= 10);
}

test "RetryNode - retry control" {
    const allocator = std.testing.allocator;

    var node = try RetryNode.init(allocator, "retry-1", 3, 10);
    defer node.deinit();

    try std.testing.expect(node.shouldRetry());
    node.current_attempt = 3;
    try std.testing.expect(!node.shouldRetry());

    node.reset();
    try std.testing.expect(node.shouldRetry());
}

test "RetryNode - execute with retry success" {
    const allocator = std.testing.allocator;

    var node = try RetryNode.init(allocator, "retry-2", 3, 5);
    defer node.deinit();

    const operation = struct {
        fn call() !void {
            // Success on first try
        }
    }.call;

    try node.executeWithRetry(&operation);
    try std.testing.expectEqual(@as(usize, 0), node.current_attempt);
}
