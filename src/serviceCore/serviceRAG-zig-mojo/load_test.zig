// Load Testing Suite for Zig + Mojo RAG Service
// Phase 3: Performance testing, benchmarking, stress testing

const std = @import("std");
const net = std.net;
const Thread = std.Thread;
const time = std.time;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Test configuration
pub const LoadTestConfig = struct {
    target_url: []const u8,
    num_clients: usize,
    requests_per_client: usize,
    request_delay_ms: u64,
    test_duration_seconds: u64,
};

// Test results
pub const LoadTestResults = struct {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    total_duration_ms: f64,
    requests_per_second: f64,
    avg_latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
};

// Request stats
const RequestStats = struct {
    latency_ms: f64,
    success: bool,
    timestamp: i64,
};

var test_results: std.ArrayList(RequestStats) = .{};
var results_mutex = Thread.Mutex{};

/// Run load test
pub fn runLoadTest(config: LoadTestConfig) !LoadTestResults {
    std.debug.print("🚀 Starting load test...\n", .{});
    std.debug.print("   Target: {s}\n", .{config.target_url});
    std.debug.print("   Clients: {d}\n", .{config.num_clients});
    std.debug.print("   Requests/client: {d}\n", .{config.requests_per_client});
    std.debug.print("   Total requests: {d}\n", .{config.num_clients * config.requests_per_client});
    
    test_results.clearRetainingCapacity();
    
    const start_time = time.milliTimestamp();
    
    // Spawn client threads
    const threads = try allocator.alloc(Thread, config.num_clients);
    defer allocator.free(threads);
    
    for (threads, 0..) |*thread, i| {
        thread.* = try Thread.spawn(.{}, clientWorker, .{ config, i });
    }
    
    // Wait for all threads
    for (threads) |thread| {
        thread.join();
    }
    
    const end_time = time.milliTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time));
    
    // Calculate statistics
    return calculateResults(duration_ms);
}

fn clientWorker(config: LoadTestConfig, client_id: usize) void {
    std.debug.print("   Client {d} started\n", .{client_id});
    
    for (0..config.requests_per_client) |req_num| {
        const stats = sendRequest(config.target_url) catch |err| {
            std.debug.print("   Client {d} error: {any}\n", .{ client_id, err });
            continue;
        };
        
        // Record stats
        results_mutex.lock();
        test_results.append(allocator, stats) catch {};
        results_mutex.unlock();
        
        // Delay between requests
        if (config.request_delay_ms > 0) {
            // Delay removed for Zig 0.15.2
        }
        
        if (req_num % 100 == 0 and req_num > 0) {
            std.debug.print("   Client {d}: {d} requests sent\n", .{ client_id, req_num });
        }
    }
    
    std.debug.print("   Client {d} completed\n", .{client_id});
}

fn sendRequest(url: []const u8) !RequestStats {
    const request_start = time.milliTimestamp();
    
    // Parse URL
    const uri = try std.Uri.parse(url);
    
    // Connect
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 8009
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Send request
    const request_body = "{\"query\":\"test load\",\"top_k\":10}";
    const http_request = try std.fmt.allocPrint(
        allocator,
        "POST {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{ uri.path.percent_encoded, uri.host.?.percent_encoded, request_body.len, request_body }
    );
    defer allocator.free(http_request);
    
    _ = try conn.writeAll(http_request);
    
    // Read response
    var buffer: [8192]u8 = undefined;
    _ = try conn.read(&buffer);
    
    const request_end = time.milliTimestamp();
    const latency = @as(f64, @floatFromInt(request_end - request_start));
    
    return RequestStats{
        .latency_ms = latency,
        .success = true,
        .timestamp = request_start,
    };
}

fn calculateResults(duration_ms: f64) !LoadTestResults {
    var successful: u64 = 0;
    var failed: u64 = 0;
    var total_latency: f64 = 0;
    var min_latency: f64 = 1000000;
    var max_latency: f64 = 0;
    
    var latencies = try allocator.alloc(f64, test_results.items.len);
    defer allocator.free(latencies);
    
    for (test_results.items, 0..) |stat, i| {
        if (stat.success) {
            successful += 1;
        } else {
            failed += 1;
        }
        
        total_latency += stat.latency_ms;
        latencies[i] = stat.latency_ms;
        
        if (stat.latency_ms < min_latency) min_latency = stat.latency_ms;
        if (stat.latency_ms > max_latency) max_latency = stat.latency_ms;
    }
    
    // Sort latencies for percentiles
    std.sort.pdq(f64, latencies, {}, std.sort.asc(f64));
    
    const total = successful + failed;
    const avg_latency = if (total > 0) total_latency / @as(f64, @floatFromInt(total)) else 0;
    const rps = @as(f64, @floatFromInt(total)) / (duration_ms / 1000.0);
    
    // Calculate percentiles
    const p50_idx = (total * 50) / 100;
    const p95_idx = (total * 95) / 100;
    const p99_idx = (total * 99) / 100;
    
    return LoadTestResults{
        .total_requests = total,
        .successful_requests = successful,
        .failed_requests = failed,
        .total_duration_ms = duration_ms,
        .requests_per_second = rps,
        .avg_latency_ms = avg_latency,
        .min_latency_ms = min_latency,
        .max_latency_ms = max_latency,
        .p50_latency_ms = if (total > 0) latencies[p50_idx] else 0,
        .p95_latency_ms = if (total > 0) latencies[p95_idx] else 0,
        .p99_latency_ms = if (total > 0) latencies[p99_idx] else 0,
    };
}

fn printResults(results: LoadTestResults) void {
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("📊 LOAD TEST RESULTS\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("\n", .{});
    
    std.debug.print("Total Requests:      {d}\n", .{results.total_requests});
    std.debug.print("Successful:          {d} ({d:.1}%)\n", 
        .{ results.successful_requests, @as(f64, @floatFromInt(results.successful_requests)) / 
           @as(f64, @floatFromInt(results.total_requests)) * 100 });
    std.debug.print("Failed:              {d} ({d:.1}%)\n", 
        .{ results.failed_requests, @as(f64, @floatFromInt(results.failed_requests)) / 
           @as(f64, @floatFromInt(results.total_requests)) * 100 });
    std.debug.print("\n", .{});
    
    std.debug.print("Duration:            {d:.2} seconds\n", .{results.total_duration_ms / 1000});
    std.debug.print("Throughput:          {d:.2} requests/second\n", .{results.requests_per_second});
    std.debug.print("\n", .{});
    
    std.debug.print("Latency Statistics:\n", .{});
    std.debug.print("  Min:               {d:.2} ms\n", .{results.min_latency_ms});
    std.debug.print("  Avg:               {d:.2} ms\n", .{results.avg_latency_ms});
    std.debug.print("  Max:               {d:.2} ms\n", .{results.max_latency_ms});
    std.debug.print("  P50:               {d:.2} ms\n", .{results.p50_latency_ms});
    std.debug.print("  P95:               {d:.2} ms\n", .{results.p95_latency_ms});
    std.debug.print("  P99:               {d:.2} ms\n", .{results.p99_latency_ms});
    std.debug.print("\n", .{});
    
    std.debug.print("=" ** 80 ++ "\n", .{});
}

pub fn main() !void {
    std.debug.print("🧪 Zig + Mojo RAG Load Testing Suite\n", .{});
    std.debug.print("=" ** 80 ++ "\n\n", .{});
    
    // Test configurations
    const tests = [_]struct {
        name: []const u8,
        config: LoadTestConfig,
    }{
        .{
            .name = "Light Load (10 clients, 100 req each)",
            .config = .{
                .target_url = "http://localhost:8009/search",
                .num_clients = 10,
                .requests_per_client = 100,
                .request_delay_ms = 10,
                .test_duration_seconds = 60,
            },
        },
        .{
            .name = "Medium Load (50 clients, 200 req each)",
            .config = .{
                .target_url = "http://localhost:8009/search",
                .num_clients = 50,
                .requests_per_client = 200,
                .request_delay_ms = 5,
                .test_duration_seconds = 120,
            },
        },
        .{
            .name = "Heavy Load (100 clients, 500 req each)",
            .config = .{
                .target_url = "http://localhost:8009/search",
                .num_clients = 100,
                .requests_per_client = 500,
                .request_delay_ms = 1,
                .test_duration_seconds = 300,
            },
        },
    };
    
    // Run each test
    for (tests) |test_case| {
        std.debug.print("🔬 Test: {s}\n", .{test_case.name});
        std.debug.print("-" ** 80 ++ "\n", .{});
        
        const results = try runLoadTest(test_case.config);
        printResults(results);
        
        std.debug.print("\n\n", .{});
    }
    
    std.debug.print("✅ All load tests complete!\n", .{});
}
