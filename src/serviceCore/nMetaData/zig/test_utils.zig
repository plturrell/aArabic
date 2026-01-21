const std = @import("std");
const client = @import("db/client.zig");
const DbClient = client.DbClient;
const Transaction = client.Transaction;
const ResultSet = client.ResultSet;
const Value = client.Value;
const IsolationLevel = client.IsolationLevel;

/// Mock database client for testing
pub const MockDbClient = struct {
    allocator: std.mem.Allocator,
    should_fail: bool,
    call_count: usize,

    pub fn init(allocator: std.mem.Allocator) MockDbClient {
        return MockDbClient{
            .allocator = allocator,
            .should_fail = false,
            .call_count = 0,
        };
    }

    pub fn setShouldFail(self: *MockDbClient, should_fail: bool) void {
        self.should_fail = should_fail;
    }

    pub fn getCallCount(self: MockDbClient) usize {
        return self.call_count;
    }

    pub fn resetCallCount(self: *MockDbClient) void {
        self.call_count = 0;
    }
};

/// Mock result set for testing
pub const MockResultSet = struct {
    rows: []const []const Value,
    current_row: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, rows: []const []const Value) MockResultSet {
        return MockResultSet{
            .rows = rows,
            .current_row = 0,
            .allocator = allocator,
        };
    }

    pub fn next(self: *MockResultSet) ?[]const Value {
        if (self.current_row >= self.rows.len) {
            return null;
        }
        const row = self.rows[self.current_row];
        self.current_row += 1;
        return row;
    }

    pub fn len(self: MockResultSet) usize {
        return self.rows.len;
    }

    pub fn reset(self: *MockResultSet) void {
        self.current_row = 0;
    }
};

/// Test data generator
pub const TestDataGenerator = struct {
    allocator: std.mem.Allocator,
    random: std.rand.Random,

    pub fn init(allocator: std.mem.Allocator) TestDataGenerator {
        var prng = std.rand.DefaultPrng.init(0);
        return TestDataGenerator{
            .allocator = allocator,
            .random = prng.random(),
        };
    }

    /// Generate random string of specified length
    pub fn randomString(self: *TestDataGenerator, len: usize) ![]const u8 {
        const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        var buf = try self.allocator.alloc(u8, len);
        for (buf) |*c| {
            c.* = chars[self.random.intRangeAtMost(usize, 0, chars.len - 1)];
        }
        return buf;
    }

    /// Generate random integer in range
    pub fn randomInt(self: *TestDataGenerator, comptime T: type, min: T, max: T) T {
        return self.random.intRangeAtMost(T, min, max);
    }

    /// Generate random email address
    pub fn randomEmail(self: *TestDataGenerator) ![]const u8 {
        const username = try self.randomString(8);
        defer self.allocator.free(username);
        
        const domain = try self.randomString(6);
        defer self.allocator.free(domain);
        
        return std.fmt.allocPrint(self.allocator, "{s}@{s}.com", .{ username, domain });
    }

    /// Generate test user data
    pub fn generateUser(self: *TestDataGenerator, id: i64) !TestUser {
        const name = try self.randomString(10);
        const email = try self.randomEmail();
        const age = self.randomInt(i32, 18, 80);
        
        return TestUser{
            .id = id,
            .name = name,
            .email = email,
            .age = age,
        };
    }
};

/// Test user structure
pub const TestUser = struct {
    id: i64,
    name: []const u8,
    email: []const u8,
    age: i32,

    pub fn deinit(self: TestUser, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.email);
    }
};

/// Test assertion helpers
pub const TestAssert = struct {
    /// Assert that two values are approximately equal (for floats)
    pub fn expectApproxEq(expected: f64, actual: f64, tolerance: f64) !void {
        const diff = @abs(expected - actual);
        if (diff > tolerance) {
            std.debug.print("Expected {d}, got {d} (diff: {d}, tolerance: {d})\n", .{ expected, actual, diff, tolerance });
            return error.TestExpectedApprox;
        }
    }

    /// Assert that a value is within a range
    pub fn expectInRange(comptime T: type, value: T, min: T, max: T) !void {
        if (value < min or value > max) {
            std.debug.print("Value {d} not in range [{d}, {d}]\n", .{ value, min, max });
            return error.TestExpectedInRange;
        }
    }

    /// Assert that a string contains a substring
    pub fn expectContains(haystack: []const u8, needle: []const u8) !void {
        if (std.mem.indexOf(u8, haystack, needle) == null) {
            std.debug.print("String '{s}' does not contain '{s}'\n", .{ haystack, needle });
            return error.TestExpectedContains;
        }
    }
};

/// Performance timer for benchmarking
pub const PerfTimer = struct {
    start: i64,

    pub fn start() PerfTimer {
        return PerfTimer{
            .start = std.time.milliTimestamp(),
        };
    }

    pub fn elapsed(self: PerfTimer) i64 {
        return std.time.milliTimestamp() - self.start;
    }

    pub fn elapsedMs(self: PerfTimer) f64 {
        return @as(f64, @floatFromInt(self.elapsed()));
    }
};

/// Test context for integration tests
pub const TestContext = struct {
    allocator: std.mem.Allocator,
    temp_dir: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator) TestContext {
        return TestContext{
            .allocator = allocator,
            .temp_dir = null,
        };
    }

    pub fn deinit(self: *TestContext) void {
        if (self.temp_dir) |dir| {
            self.allocator.free(dir);
        }
    }

    /// Create a temporary directory for testing
    pub fn createTempDir(self: *TestContext) ![]const u8 {
        const timestamp = std.time.milliTimestamp();
        const dir = try std.fmt.allocPrint(
            self.allocator,
            "/tmp/nmetadata_test_{d}",
            .{timestamp},
        );
        self.temp_dir = dir;
        return dir;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "MockDbClient - basic functionality" {
    var mock = MockDbClient.init(std.testing.allocator);
    
    try std.testing.expectEqual(@as(usize, 0), mock.getCallCount());
    try std.testing.expect(!mock.should_fail);
    
    mock.setShouldFail(true);
    try std.testing.expect(mock.should_fail);
}

test "MockResultSet - iteration" {
    const allocator = std.testing.allocator;
    
    const row1 = [_]Value{
        Value{ .int64 = 1 },
        Value{ .string = "test" },
    };
    const row2 = [_]Value{
        Value{ .int64 = 2 },
        Value{ .string = "test2" },
    };
    
    const rows = [_][]const Value{ &row1, &row2 };
    
    var result = MockResultSet.init(allocator, &rows);
    
    try std.testing.expectEqual(@as(usize, 2), result.len());
    
    if (result.next()) |row| {
        try std.testing.expectEqual(@as(i64, 1), row[0].int64);
    } else {
        return error.TestUnexpectedNull;
    }
    
    if (result.next()) |row| {
        try std.testing.expectEqual(@as(i64, 2), row[0].int64);
    } else {
        return error.TestUnexpectedNull;
    }
    
    try std.testing.expect(result.next() == null);
}

test "MockResultSet - reset" {
    const allocator = std.testing.allocator;
    
    const row1 = [_]Value{ Value{ .int64 = 1 } };
    const rows = [_][]const Value{&row1};
    
    var result = MockResultSet.init(allocator, &rows);
    
    _ = result.next();
    try std.testing.expect(result.next() == null);
    
    result.reset();
    try std.testing.expect(result.next() != null);
}

test "TestDataGenerator - random string" {
    const allocator = std.testing.allocator;
    var gen = TestDataGenerator.init(allocator);
    
    const str = try gen.randomString(10);
    defer allocator.free(str);
    
    try std.testing.expectEqual(@as(usize, 10), str.len);
}

test "TestDataGenerator - random int" {
    const allocator = std.testing.allocator;
    var gen = TestDataGenerator.init(allocator);
    
    const val = gen.randomInt(i32, 1, 100);
    try std.testing.expect(val >= 1 and val <= 100);
}

test "TestDataGenerator - random email" {
    const allocator = std.testing.allocator;
    var gen = TestDataGenerator.init(allocator);
    
    const email = try gen.randomEmail();
    defer allocator.free(email);
    
    try std.testing.expect(std.mem.indexOf(u8, email, "@") != null);
    try std.testing.expect(std.mem.indexOf(u8, email, ".com") != null);
}

test "TestDataGenerator - generate user" {
    const allocator = std.testing.allocator;
    var gen = TestDataGenerator.init(allocator);
    
    const user = try gen.generateUser(42);
    defer user.deinit(allocator);
    
    try std.testing.expectEqual(@as(i64, 42), user.id);
    try std.testing.expect(user.name.len > 0);
    try std.testing.expect(user.email.len > 0);
    try std.testing.expect(user.age >= 18 and user.age <= 80);
}

test "TestAssert - expectApproxEq" {
    try TestAssert.expectApproxEq(1.0, 1.001, 0.01);
    try TestAssert.expectApproxEq(100.0, 100.5, 1.0);
    
    const result = TestAssert.expectApproxEq(1.0, 2.0, 0.1);
    try std.testing.expectError(error.TestExpectedApprox, result);
}

test "TestAssert - expectInRange" {
    try TestAssert.expectInRange(i32, 50, 0, 100);
    try TestAssert.expectInRange(i32, 0, 0, 100);
    try TestAssert.expectInRange(i32, 100, 0, 100);
    
    const result = TestAssert.expectInRange(i32, 150, 0, 100);
    try std.testing.expectError(error.TestExpectedInRange, result);
}

test "TestAssert - expectContains" {
    try TestAssert.expectContains("hello world", "world");
    try TestAssert.expectContains("testing", "test");
    
    const result = TestAssert.expectContains("hello", "goodbye");
    try std.testing.expectError(error.TestExpectedContains, result);
}

test "PerfTimer - elapsed time" {
    const timer = PerfTimer.start();
    
    std.time.sleep(10 * std.time.ns_per_ms); // Sleep 10ms
    
    const elapsed = timer.elapsed();
    try std.testing.expect(elapsed >= 10);
}

test "TestContext - init and deinit" {
    var ctx = TestContext.init(std.testing.allocator);
    defer ctx.deinit();
    
    try std.testing.expect(ctx.temp_dir == null);
}

test "TestContext - create temp dir" {
    var ctx = TestContext.init(std.testing.allocator);
    defer ctx.deinit();
    
    const dir = try ctx.createTempDir();
    try std.testing.expect(dir.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, dir, "/tmp/nmetadata_test_") != null);
}
