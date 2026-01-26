const std = @import("std");

/// Load test configuration
pub const LoadTestConfig = struct {
    /// Number of concurrent users
    concurrent_users: u32,
    /// Duration of test in seconds
    duration_seconds: u32,
    /// Requests per second target (0 = unlimited)
    target_rps: u32,
    /// Base URL for API
    base_url: []const u8,
    /// Authentication token
    auth_token: ?[]const u8,

    pub fn default(base_url: []const u8) LoadTestConfig {
        return LoadTestConfig{
            .concurrent_users = 10,
            .duration_seconds = 60,
            .target_rps = 0,
            .base_url = base_url,
            .auth_token = null,
        };
    }
};

/// Load test metrics
pub const LoadTestMetrics = struct {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    total_latency_ms: u64,
    min_latency_ms: u64,
    max_latency_ms: u64,
    status_codes: std.AutoHashMap(u16, u64),
    
    pub fn init(allocator: std.mem.Allocator) LoadTestMetrics {
        return LoadTestMetrics{
            .total_requests = 0,
            .successful_requests = 0,
            .failed_requests = 0,
            .total_latency_ms = 0,
            .min_latency_ms = std.math.maxInt(u64),
            .max_latency_ms = 0,
            .status_codes = std.AutoHashMap(u16, u64).init(allocator),
        };
    }

    pub fn deinit(self: *LoadTestMetrics) void {
        self.status_codes.deinit();
    }

    /// Record request result
    pub fn recordRequest(self: *LoadTestMetrics, status_code: u16, latency_ms: u64) !void {
        self.total_requests += 1;
        
        if (status_code >= 200 and status_code < 300) {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }

        self.total_latency_ms += latency_ms;
        self.min_latency_ms = @min(self.min_latency_ms, latency_ms);
        self.max_latency_ms = @max(self.max_latency_ms, latency_ms);

        // Track status code distribution
        const entry = try self.status_codes.getOrPut(status_code);
        if (entry.found_existing) {
            entry.value_ptr.* += 1;
        } else {
            entry.value_ptr.* = 1;
        }
    }

    /// Calculate average latency
    pub fn avgLatency(self: LoadTestMetrics) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_latency_ms)) / @as(f64, @floatFromInt(self.total_requests));
    }

    /// Calculate success rate
    pub fn successRate(self: LoadTestMetrics) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.successful_requests)) / @as(f64, @floatFromInt(self.total_requests)) * 100.0;
    }

    /// Calculate requests per second
    pub fn rps(self: LoadTestMetrics, duration_seconds: u32) f64 {
        if (duration_seconds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_requests)) / @as(f64, @floatFromInt(duration_seconds));
    }

    /// Print metrics summary
    pub fn printSummary(self: LoadTestMetrics, duration_seconds: u32) void {
        std.debug.print("\n=== Load Test Results ===\n", .{});
        std.debug.print("Duration: {d}s\n", .{duration_seconds});
        std.debug.print("Total Requests: {d}\n", .{self.total_requests});
        std.debug.print("Successful: {d}\n", .{self.successful_requests});
        std.debug.print("Failed: {d}\n", .{self.failed_requests});
        std.debug.print("Success Rate: {d:.2}%\n", .{self.successRate()});
        std.debug.print("Requests/sec: {d:.2}\n", .{self.rps(duration_seconds)});
        std.debug.print("\nLatency:\n", .{});
        std.debug.print("  Min: {d}ms\n", .{self.min_latency_ms});
        std.debug.print("  Avg: {d:.2}ms\n", .{self.avgLatency()});
        std.debug.print("  Max: {d}ms\n", .{self.max_latency_ms});
        
        std.debug.print("\nStatus Code Distribution:\n", .{});
        var iter = self.status_codes.iterator();
        while (iter.next()) |entry| {
            const percentage = @as(f64, @floatFromInt(entry.value_ptr.*)) / 
                             @as(f64, @floatFromInt(self.total_requests)) * 100.0;
            std.debug.print("  {d}: {d} ({d:.1}%)\n", .{ entry.key_ptr.*, entry.value_ptr.*, percentage });
        }
    }
};

/// Load test scenario
pub const LoadTestScenario = struct {
    name: []const u8,
    endpoint: []const u8,
    method: []const u8,
    body: ?[]const u8,
    weight: u32, // Probability weight (1-100)

    pub fn init(name: []const u8, method: []const u8, endpoint: []const u8, body: ?[]const u8, weight: u32) LoadTestScenario {
        return LoadTestScenario{
            .name = name,
            .endpoint = endpoint,
            .method = method,
            .body = body,
            .weight = weight,
        };
    }
};

/// Load test runner
pub const LoadTestRunner = struct {
    allocator: std.mem.Allocator,
    config: LoadTestConfig,
    metrics: LoadTestMetrics,
    scenarios: []const LoadTestScenario,
    random: std.rand.Random,

    pub fn init(allocator: std.mem.Allocator, config: LoadTestConfig, scenarios: []const LoadTestScenario) LoadTestRunner {
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        return LoadTestRunner{
            .allocator = allocator,
            .config = config,
            .metrics = LoadTestMetrics.init(allocator),
            .scenarios = scenarios,
            .random = prng.random(),
        };
    }

    pub fn deinit(self: *LoadTestRunner) void {
        self.metrics.deinit();
    }

    /// Select random scenario based on weights
    fn selectScenario(self: *LoadTestRunner) LoadTestScenario {
        var total_weight: u32 = 0;
        for (self.scenarios) |scenario| {
            total_weight += scenario.weight;
        }

        const rand_value = self.random.intRangeAtMost(u32, 0, total_weight - 1);
        var cumulative: u32 = 0;
        
        for (self.scenarios) |scenario| {
            cumulative += scenario.weight;
            if (rand_value < cumulative) {
                return scenario;
            }
        }

        return self.scenarios[0];
    }

    /// Simulate single request
    fn simulateRequest(self: *LoadTestRunner, scenario: LoadTestScenario) !void {
        const start = std.time.milliTimestamp();
        
        // Simulate HTTP request (in real scenario, would make actual HTTP call)
        const base_latency: u64 = 10 + self.random.intRangeAtMost(u64, 0, 50);
        
        // Add latency based on endpoint complexity
        const latency_ms = if (std.mem.indexOf(u8, scenario.endpoint, "graphql") != null)
            base_latency + 20
        else if (std.mem.indexOf(u8, scenario.endpoint, "lineage") != null)
            base_latency + 30
        else
            base_latency;

        std.time.sleep(latency_ms * std.time.ns_per_ms);

        const end = std.time.milliTimestamp();
        const actual_latency: u64 = @intCast(end - start);

        // Simulate status codes (95% success rate)
        const status_code: u16 = if (self.random.intRangeAtMost(u32, 0, 99) < 95) 200 else 500;

        try self.metrics.recordRequest(status_code, actual_latency);
    }

    /// Run load test
    pub fn run(self: *LoadTestRunner) !void {
        std.debug.print("\n=== Starting Load Test ===\n", .{});
        std.debug.print("Concurrent Users: {d}\n", .{self.config.concurrent_users});
        std.debug.print("Duration: {d}s\n", .{self.config.duration_seconds});
        std.debug.print("Scenarios: {d}\n", .{self.scenarios.len});
        
        const start_time = std.time.milliTimestamp();
        const end_time = start_time + @as(i64, self.config.duration_seconds) * 1000;

        // Simulate concurrent users
        var user: u32 = 0;
        while (user < self.config.concurrent_users) : (user += 1) {
            var current_time = std.time.milliTimestamp();
            
            while (current_time < end_time) {
                const scenario = self.selectScenario();
                try self.simulateRequest(scenario);
                
                // Apply rate limiting if configured
                if (self.config.target_rps > 0) {
                    const delay_ms = 1000 / self.config.target_rps;
                    std.time.sleep(delay_ms * std.time.ns_per_ms);
                }
                
                current_time = std.time.milliTimestamp();
            }
        }

        self.metrics.printSummary(self.config.duration_seconds);
    }
};

/// Predefined load test scenarios
pub const Scenarios = struct {
    /// Authentication scenarios
    pub fn authentication(allocator: std.mem.Allocator) ![]const LoadTestScenario {
        var scenarios = std.ArrayList(LoadTestScenario){};
        
        try scenarios.append(LoadTestScenario.init(
            "Login",
            "POST",
            "/api/v1/auth/login",
            \\{"username": "admin", "password": "admin123"}
            ,
            30,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Get Current User",
            "GET",
            "/api/v1/auth/me",
            null,
            40,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Verify Token",
            "GET",
            "/api/v1/auth/verify",
            null,
            20,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Refresh Token",
            "POST",
            "/api/v1/auth/refresh",
            null,
            10,
        ));
        
        return scenarios.toOwnedSlice();
    }

    /// Dataset CRUD scenarios
    pub fn datasetCRUD(allocator: std.mem.Allocator) ![]const LoadTestScenario {
        var scenarios = std.ArrayList(LoadTestScenario){};
        
        try scenarios.append(LoadTestScenario.init(
            "List Datasets",
            "GET",
            "/api/v1/datasets?limit=20&offset=0",
            null,
            50,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Get Dataset",
            "GET",
            "/api/v1/datasets/1",
            null,
            30,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Create Dataset",
            "POST",
            "/api/v1/datasets",
            \\{"name": "test_ds", "type": "table", "schema": "public"}
            ,
            10,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Update Dataset",
            "PUT",
            "/api/v1/datasets/1",
            \\{"description": "Updated"}
            ,
            5,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Delete Dataset",
            "DELETE",
            "/api/v1/datasets/1",
            null,
            5,
        ));
        
        return scenarios.toOwnedSlice();
    }

    /// Lineage tracking scenarios
    pub fn lineage(allocator: std.mem.Allocator) ![]const LoadTestScenario {
        var scenarios = std.ArrayList(LoadTestScenario){};
        
        try scenarios.append(LoadTestScenario.init(
            "Get Upstream Lineage",
            "GET",
            "/api/v1/lineage/upstream/1?depth=5",
            null,
            45,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Get Downstream Lineage",
            "GET",
            "/api/v1/lineage/downstream/1?depth=5",
            null,
            45,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Create Lineage Edge",
            "POST",
            "/api/v1/lineage/edges",
            \\{"from_id": 1, "to_id": 2, "edge_type": "derives_from"}
            ,
            10,
        ));
        
        return scenarios.toOwnedSlice();
    }

    /// GraphQL scenarios
    pub fn graphql(allocator: std.mem.Allocator) ![]const LoadTestScenario {
        var scenarios = std.ArrayList(LoadTestScenario){};
        
        try scenarios.append(LoadTestScenario.init(
            "Query Datasets",
            "POST",
            "/api/v1/graphql",
            \\{"query": "{ datasets { id name type } }"}
            ,
            40,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Query Lineage",
            "POST",
            "/api/v1/graphql",
            \\{"query": "{ dataset(id: 1) { upstream { id name } } }"}
            ,
            30,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Schema Introspection",
            "POST",
            "/api/v1/graphql",
            \\{"query": "{ __schema { types { name } } }"}
            ,
            20,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "GraphiQL Playground",
            "GET",
            "/api/v1/graphiql",
            null,
            10,
        ));
        
        return scenarios.toOwnedSlice();
    }

    /// Mixed workload scenarios
    pub fn mixed(allocator: std.mem.Allocator) ![]const LoadTestScenario {
        var scenarios = std.ArrayList(LoadTestScenario){};
        
        // Read-heavy workload (70%)
        try scenarios.append(LoadTestScenario.init(
            "List Datasets",
            "GET",
            "/api/v1/datasets?limit=20",
            null,
            30,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Get Dataset",
            "GET",
            "/api/v1/datasets/1",
            null,
            20,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Get Lineage",
            "GET",
            "/api/v1/lineage/upstream/1",
            null,
            20,
        ));
        
        // Write operations (20%)
        try scenarios.append(LoadTestScenario.init(
            "Create Dataset",
            "POST",
            "/api/v1/datasets",
            \\{"name": "test", "type": "table"}
            ,
            10,
        ));
        
        try scenarios.append(LoadTestScenario.init(
            "Update Dataset",
            "PUT",
            "/api/v1/datasets/1",
            \\{"description": "Updated"}
            ,
            10,
        ));
        
        // GraphQL (10%)
        try scenarios.append(LoadTestScenario.init(
            "GraphQL Query",
            "POST",
            "/api/v1/graphql",
            \\{"query": "{ datasets { id name } }"}
            ,
            10,
        ));
        
        return scenarios.toOwnedSlice();
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "LoadTestConfig - default" {
    const config = LoadTestConfig.default("http://localhost:3000");
    
    try std.testing.expectEqual(@as(u32, 10), config.concurrent_users);
    try std.testing.expectEqual(@as(u32, 60), config.duration_seconds);
    try std.testing.expectEqual(@as(u32, 0), config.target_rps);
    try std.testing.expectEqualStrings("http://localhost:3000", config.base_url);
}

test "LoadTestMetrics - record request" {
    var metrics = LoadTestMetrics.init(std.testing.allocator);
    defer metrics.deinit();
    
    try metrics.recordRequest(200, 50);
    try metrics.recordRequest(200, 100);
    try metrics.recordRequest(500, 75);
    
    try std.testing.expectEqual(@as(u64, 3), metrics.total_requests);
    try std.testing.expectEqual(@as(u64, 2), metrics.successful_requests);
    try std.testing.expectEqual(@as(u64, 1), metrics.failed_requests);
    try std.testing.expectEqual(@as(u64, 50), metrics.min_latency_ms);
    try std.testing.expectEqual(@as(u64, 100), metrics.max_latency_ms);
}

test "LoadTestMetrics - calculations" {
    var metrics = LoadTestMetrics.init(std.testing.allocator);
    defer metrics.deinit();
    
    try metrics.recordRequest(200, 100);
    try metrics.recordRequest(200, 200);
    
    const avg = metrics.avgLatency();
    try std.testing.expectApproxEqAbs(@as(f64, 150.0), avg, 0.1);
    
    const success = metrics.successRate();
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), success, 0.1);
    
    const requests_per_sec = metrics.rps(10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), requests_per_sec, 0.1);
}

test "LoadTestScenario - init" {
    const scenario = LoadTestScenario.init(
        "Test",
        "GET",
        "/test",
        null,
        50,
    );
    
    try std.testing.expectEqualStrings("Test", scenario.name);
    try std.testing.expectEqualStrings("GET", scenario.method);
    try std.testing.expectEqualStrings("/test", scenario.endpoint);
    try std.testing.expectEqual(@as(u32, 50), scenario.weight);
}

test "Scenarios - authentication" {
    const scenarios = try Scenarios.authentication(std.testing.allocator);
    defer std.testing.allocator.free(scenarios);
    
    try std.testing.expect(scenarios.len > 0);
}

test "Scenarios - dataset CRUD" {
    const scenarios = try Scenarios.datasetCRUD(std.testing.allocator);
    defer std.testing.allocator.free(scenarios);
    
    try std.testing.expect(scenarios.len > 0);
}
