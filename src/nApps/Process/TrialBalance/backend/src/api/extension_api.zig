//! ============================================================================
//! Extension API Handler
//! REST endpoints for frontend extension communication
//! ============================================================================
//!
//! [CODE:file=extension_api.zig]
//! [CODE:module=api]
//! [CODE:language=zig]
//!
//! [TABLE:reads=TB_EXTENSION_REGISTRY]
//!
//! [API:produces=/api/v1/extensions,/api/v1/extensions/discover]
//!
//! [RELATION:calls=CODE:extension_registry.zig]
//! [RELATION:called_by=CODE:main.zig]
//!
//! This API provides extension discovery and management for the frontend.
//! Note: Infrastructure code - no ODPS business rules implemented here.

const std = @import("std");
const ExtensionRegistry = @import("../extensions/extension_registry.zig").ExtensionRegistry;
const ExtensionType = @import("../extensions/extension_registry.zig").ExtensionType;

/// Extension API Handler
/// Provides REST endpoints for frontend extension communication
pub const ExtensionApi = struct {
    allocator: std.mem.Allocator,
    registry: *ExtensionRegistry,

    pub fn init(allocator: std.mem.Allocator, registry: *ExtensionRegistry) ExtensionApi {
        return ExtensionApi{
            .allocator = allocator,
            .registry = registry,
        };
    }

    /// Handle extension API request
    pub fn handleRequest(
        self: *ExtensionApi,
        path: []const u8,
        method: []const u8,
        body: []const u8,
    ) ![]const u8 {
        // Route: GET /api/v1/extensions/discover
        if (std.mem.endsWith(u8, path, "/discover") and std.mem.eql(u8, method, "GET")) {
            return try self.handleDiscover();
        }

        // Route: GET /api/v1/extensions/{id}/manifest
        if (std.mem.indexOf(u8, path, "/manifest")) |_| {
            const ext_id = try self.extractExtensionId(path);
            return try self.handleGetManifest(ext_id);
        }

        // Route: GET /api/v1/extensions/{id}/version
        if (std.mem.indexOf(u8, path, "/version")) |_| {
            const ext_id = try self.extractExtensionId(path);
            return try self.handleGetVersion(ext_id);
        }

        // Route: GET /api/v1/extensions/{id}/config
        if (std.mem.indexOf(u8, path, "/config")) |_| {
            const ext_id = try self.extractExtensionId(path);
            return try self.handleGetConfig(ext_id);
        }

        // Route: POST /api/v1/extensions/{id}/register
        if (std.mem.indexOf(u8, path, "/register")) |_| {
            const ext_id = try self.extractExtensionId(path);
            return try self.handleRegister(ext_id, body);
        }

        // Route: Extension-specific endpoints
        if (std.mem.indexOf(u8, path, "/extensions/")) |idx| {
            const ext_path = path[idx + 12..]; // After "/extensions/"
            return try self.routeToExtension(ext_path, method, body);
        }

        return try self.jsonError("Endpoint not found");
    }

    /// Handle extension discovery
    fn handleDiscover(self: *ExtensionApi) ![]const u8 {
        const stats = self.registry.getStats();
        
        var extensions = std.ArrayList(u8).init(self.allocator);
        const writer = extensions.writer();
        
        try writer.writeAll("[");
        
        // List registered extensions
        var first = true;
        inline for (@typeInfo(ExtensionType).@"enum".fields) |field| {
            const ext_type: ExtensionType = @enumFromInt(field.value);
            if (self.registry.getByType(ext_type)) |exts| {
                for (exts) |ext| {
                    if (!first) try writer.writeAll(",");
                    first = false;
                    
                    try std.fmt.format(writer, 
                        \\{{"id":"{s}","name":"{s}","version":"{s}","type":"{s}","enabled":{s}}}
                    , .{ ext.id, ext.name, ext.version, @tagName(ext.ext_type), if (ext.enabled) "true" else "false" });
                }
            }
        }
        
        try writer.writeAll("]");
        
        return try std.fmt.allocPrint(self.allocator,
            \\{{"extensions":{s},"stats":{{"total":{},"enabled":{}}}}}
        , .{ extensions.items, stats.total, stats.enabled });
    }

    /// Handle get manifest
    fn handleGetManifest(self: *ExtensionApi, ext_id: []const u8) ![]const u8 {
        const ext = self.registry.get(ext_id) orelse return try self.jsonError("Extension not found");
        
        return try std.fmt.allocPrint(self.allocator,
            \\{{
            \\  "id": "{s}",
            \\  "name": "{s}",
            \\  "version": "{s}",
            \\  "type": "{s}",
            \\  "enabled": {s},
            \\  "priority": {}
            \\}}
        , .{ ext.id, ext.name, ext.version, @tagName(ext.ext_type), if (ext.enabled) "true" else "false", ext.priority });
    }

    /// Handle get version
    fn handleGetVersion(self: *ExtensionApi, ext_id: []const u8) ![]const u8 {
        const ext = self.registry.get(ext_id) orelse return try self.jsonError("Extension not found");
        
        return try std.fmt.allocPrint(self.allocator,
            \\{{"version":"{s}","autoReload":false}}
        , .{ext.version});
    }

    /// Handle get config
    fn handleGetConfig(self: *ExtensionApi, ext_id: []const u8) ![]const u8 {
        const ext = self.registry.get(ext_id) orelse return try self.jsonError("Extension not found");
        
        // Try to get config from extension handler
        if (ext.hooks.handleRequest) |handler| {
            return try handler(self.allocator, "/config", "GET", "");
        }
        
        // Return default config
        return try self.allocator.dupe(u8,
            \\{"layout":"ForceDirected","minimap":true,"animation":true}
        );
    }

    /// Handle register
    fn handleRegister(self: *ExtensionApi, ext_id: []const u8, body: []const u8) ![]const u8 {
        _ = body;
        
        // Check if already registered
        if (self.registry.get(ext_id)) |_| {
            return try std.fmt.allocPrint(self.allocator,
                \\{{"status":"already_registered","id":"{s}"}}
            , .{ext_id});
        }
        
        return try std.fmt.allocPrint(self.allocator,
            \\{{"status":"registered","id":"{s}"}}
        , .{ext_id});
    }

    /// Route request to extension
    fn routeToExtension(self: *ExtensionApi, ext_path: []const u8, method: []const u8, body: []const u8) ![]const u8 {
        // Extract extension ID from path (e.g., "enhanced-metrics/config" -> "enhanced-metrics")
        const slash_idx = std.mem.indexOf(u8, ext_path, "/") orelse ext_path.len;
        const ext_id = ext_path[0..slash_idx];
        const sub_path = if (slash_idx < ext_path.len) ext_path[slash_idx..] else "/";
        
        return try self.registry.handleExtensionRequest(self.allocator, ext_id, sub_path, method, body);
    }

    /// Extract extension ID from path
    fn extractExtensionId(self: *ExtensionApi, path: []const u8) ![]const u8 {
        _ = self;
        
        // Path format: /api/v1/extensions/{id}/...
        const prefix = "/api/v1/extensions/";
        if (std.mem.indexOf(u8, path, prefix)) |idx| {
            const after_prefix = path[idx + prefix.len..];
            const slash_idx = std.mem.indexOf(u8, after_prefix, "/") orelse after_prefix.len;
            return after_prefix[0..slash_idx];
        }
        
        // Alternative: /extensions/{id}/...
        const alt_prefix = "/extensions/";
        if (std.mem.indexOf(u8, path, alt_prefix)) |idx| {
            const after_prefix = path[idx + alt_prefix.len..];
            const slash_idx = std.mem.indexOf(u8, after_prefix, "/") orelse after_prefix.len;
            return after_prefix[0..slash_idx];
        }
        
        return error.InvalidPath;
    }

    /// Return JSON error
    fn jsonError(self: *ExtensionApi, message: []const u8) ![]const u8 {
        return try std.fmt.allocPrint(self.allocator,
            \\{{"error":"{s}"}}
        , .{message});
    }
};

// Tests
test "ExtensionApi - discover empty" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var registry = try ExtensionRegistry.init(allocator);
    defer registry.deinit();
    
    var api = ExtensionApi.init(allocator, &registry);
    
    const response = try api.handleRequest("/api/v1/extensions/discover", "GET", "");
    defer allocator.free(response);
    
    try testing.expect(std.mem.indexOf(u8, response, "extensions") != null);
}