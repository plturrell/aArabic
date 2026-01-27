//! ============================================================================
//! Enhanced Metrics Extension
//! Example backend extension for advanced metric calculations
//! ============================================================================
//!
//! [CODE:file=enhanced_metrics.zig]
//! [CODE:module=extensions/examples]
//! [CODE:language=zig]
//!
//! [RELATION:uses=CODE:extension_registry.zig]
//!
//! Note: Example extension code demonstrating the extension framework.

const std = @import("std");
const Extension = @import("../extension_registry.zig").Extension;
const ExtensionType = @import("../extension_registry.zig").ExtensionType;
const HookFn = @import("../extension_registry.zig").HookFn;

/// Enhanced Metrics Extension
/// Example backend extension that provides advanced metric calculations
/// for the Enhanced Network Graph frontend extension
pub const EnhancedMetricsExtension = struct {
    allocator: std.mem.Allocator,
    cache: std.StringHashMap([]const u8),
    
    pub fn create(allocator: std.mem.Allocator) !*Extension {
        const ext = try allocator.create(Extension);
        
        ext.* = Extension.init(
            "enhanced-metrics",
            "Enhanced Metrics Calculator",
            "1.0.0",
            .calculator,
        );
        
        ext.priority = 10;
        ext.hooks.init = init;
        ext.hooks.deinit = deinit;
        ext.hooks.onCalculate = onCalculate;
        ext.hooks.handleRequest = handleRequest;
        
        // Allocate user data
        const user_data = try allocator.create(EnhancedMetricsExtension);
        user_data.* = .{
            .allocator = allocator,
            .cache = std.StringHashMap([]const u8).init(allocator),
        };
        ext.user_data = user_data;
        
        return ext;
    }
};

/// Initialize extension
fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("EnhancedMetrics extension initialized\n", .{});
}

/// Cleanup extension
fn deinit() void {
    std.debug.print("EnhancedMetrics extension deinitialized\n", .{});
}

/// Calculate enhanced metrics
fn onCalculate(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    std.debug.print("Calculating enhanced metrics for input: {s}\n", .{input});
    
    // Example: Parse input JSON and add calculated fields
    // In real implementation, would use JSON parser
    
    // For now, return enhanced JSON string
    const result = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "original": {s},
        \\  "metrics": {{
        \\    "centrality": 0.75,
        \\    "riskScore": 0.42,
        \\    "trend": "increasing",
        \\    "confidence": 0.89
        \\  }},
        \\  "enhanced": true,
        \\  "timestamp": "{d}"
        \\}}
    , .{ input, std.time.timestamp() });
    
    return result;
}

/// Handle HTTP requests for this extension
fn handleRequest(
    allocator: std.mem.Allocator,
    path: []const u8,
    method: []const u8,
    body: []const u8,
) ![]const u8 {
    std.debug.print("EnhancedMetrics handling request: {s} {s}\n", .{ method, path });
    
    // Route: GET /config - Return extension configuration
    if (std.mem.endsWith(u8, path, "/config") and std.mem.eql(u8, method, "GET")) {
        return try allocator.dupe(u8,
            \\{
            \\  "autoLayout": true,
            \\  "showMetrics": true,
            \\  "animationEnabled": true,
            \\  "metricsRefreshInterval": 30000
            \\}
        );
    }
    
    // Route: GET /node/{id} - Return node details
    if (std.mem.indexOf(u8, path, "/node/")) |idx| {
        const node_id_start = idx + 6;
        if (node_id_start < path.len) {
            const node_id = path[node_id_start..];
            
            return try std.fmt.allocPrint(allocator,
                \\{{
                \\  "nodeId": "{s}",
                \\  "detailedMetrics": {{
                \\    "balance": 125000.50,
                \\    "variance": 5420.30,
                \\    "variancePercent": 4.34,
                \\    "transactions": 847,
                \\    "lastUpdate": "{d}",
                \\    "riskFactors": [
                \\      "High transaction volume",
                \\      "Unusual pattern detected"
                \\    ]
                \\  }},
                \\  "relatedAccounts": ["ACC001", "ACC042", "ACC103"],
                \\  "recommendations": [
                \\    "Review high-variance transactions",
                \\    "Verify account reconciliation"
                \\  ]
                \\}}
            , .{ node_id, std.time.timestamp() });
        }
    }
    
    // Route: POST /calculate - Calculate custom metrics
    if (std.mem.endsWith(u8, path, "/calculate") and std.mem.eql(u8, method, "POST")) {
        return try onCalculate(allocator, body);
    }
    
    // Default: Method not found
    return try allocator.dupe(u8,
        \\{
        \\  "error": "Endpoint not found",
        \\  "path": "unknown",
        \\  "availableEndpoints": [
        \\    "GET /config",
        \\    "GET /node/{id}",
        \\    "POST /calculate"
        \\  ]
        \\}
    );
}

// ========== Tests ==========

test "EnhancedMetrics - initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const ext = try EnhancedMetricsExtension.create(allocator);
    defer allocator.destroy(ext);
    
    try testing.expectEqualStrings("enhanced-metrics", ext.id);
    try testing.expectEqualStrings("Enhanced Metrics Calculator", ext.name);
    try testing.expectEqual(ExtensionType.calculator, ext.ext_type);
    try testing.expectEqual(@as(i32, 10), ext.priority);
    try testing.expect(ext.hooks.init != null);
    try testing.expect(ext.hooks.handleRequest != null);
}

test "EnhancedMetrics - config endpoint" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const response = try handleRequest(
        allocator,
        "/api/v1/extensions/enhanced-metrics/config",
        "GET",
        "",
    );
    defer allocator.free(response);
    
    try testing.expect(std.mem.indexOf(u8, response, "autoLayout") != null);
    try testing.expect(std.mem.indexOf(u8, response, "true") != null);
}

test "EnhancedMetrics - node endpoint" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const response = try handleRequest(
        allocator,
        "/api/v1/extensions/enhanced-metrics/node/ACC001",
        "GET",
        "",
    );
    defer allocator.free(response);
    
    try testing.expect(std.mem.indexOf(u8, response, "ACC001") != null);
    try testing.expect(std.mem.indexOf(u8, response, "detailedMetrics") != null);
}

test "EnhancedMetrics - calculate endpoint" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const input = "{\"account\": \"ACC001\", \"balance\": 1000}";
    const response = try handleRequest(
        allocator,
        "/api/v1/extensions/enhanced-metrics/calculate",
        "POST",
        input,
    );
    defer allocator.free(response);
    
    try testing.expect(std.mem.indexOf(u8, response, "metrics") != null);
    try testing.expect(std.mem.indexOf(u8, response, "enhanced") != null);
}