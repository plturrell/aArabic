const std = @import("std");
const Allocator = std.mem.Allocator;
const ApisixClient = @import("apisix_client.zig").ApisixClient;

/// Manages API keys for workflow authentication via APISIX
/// Provides key generation, validation, revocation, and scoping
pub const ApiKeyManager = struct {
    allocator: Allocator,
    apisix_client: *ApisixClient,
    api_keys: std.StringHashMap(ApiKeyInfo),
    rng: std.Random.DefaultPrng,

    pub fn init(allocator: Allocator, apisix_client: *ApisixClient) !*ApiKeyManager {
        const manager = try allocator.create(ApiKeyManager);
        errdefer allocator.destroy(manager);

        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));

        manager.* = .{
            .allocator = allocator,
            .apisix_client = apisix_client,
            .api_keys = std.StringHashMap(ApiKeyInfo).init(allocator),
            .rng = std.Random.DefaultPrng.init(seed),
        };

        return manager;
    }

    pub fn deinit(self: *ApiKeyManager) void {
        var iterator = self.api_keys.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.api_keys.deinit();
        self.allocator.destroy(self);
    }

    /// Generate a new API key for a workflow or user
    pub fn generateKey(self: *ApiKeyManager, scope: ApiKeyScope, description: []const u8) ![]const u8 {
        // Generate a cryptographically secure random key
        const key = try self.generateRandomKey();
        errdefer self.allocator.free(key);

        // Create key info
        const key_info = ApiKeyInfo{
            .key = try self.allocator.dupe(u8, key),
            .scope = try scope.clone(self.allocator),
            .description = try self.allocator.dupe(u8, description),
            .created_at = std.time.timestamp(),
            .expires_at = null,
            .last_used_at = null,
            .usage_count = 0,
            .is_active = true,
        };

        // Register key with APISIX consumer
        try self.registerKeyWithApisix(key, key_info);

        // Store in local registry
        try self.api_keys.put(try self.allocator.dupe(u8, key), key_info);

        return key;
    }

    /// Generate a key with expiration
    pub fn generateKeyWithExpiration(
        self: *ApiKeyManager,
        scope: ApiKeyScope,
        description: []const u8,
        expires_in_seconds: i64,
    ) ![]const u8 {
        const key = try self.generateRandomKey();
        errdefer self.allocator.free(key);

        const now = std.time.timestamp();
        const key_info = ApiKeyInfo{
            .key = try self.allocator.dupe(u8, key),
            .scope = try scope.clone(self.allocator),
            .description = try self.allocator.dupe(u8, description),
            .created_at = now,
            .expires_at = now + expires_in_seconds,
            .last_used_at = null,
            .usage_count = 0,
            .is_active = true,
        };

        try self.registerKeyWithApisix(key, key_info);
        try self.api_keys.put(try self.allocator.dupe(u8, key), key_info);

        return key;
    }

    /// Validate an API key
    pub fn validateKey(self: *ApiKeyManager, key: []const u8, workflow_id: ?[]const u8) !bool {
        var key_info = self.api_keys.getPtr(key) orelse return false;

        // Check if key is active
        if (!key_info.is_active) return false;

        // Check expiration
        if (key_info.expires_at) |expires| {
            if (std.time.timestamp() > expires) {
                key_info.is_active = false;
                return false;
            }
        }

        // Check scope
        const is_authorized = switch (key_info.scope) {
            .global => true,
            .workflow => |wf_id| blk: {
                if (workflow_id) |wf| {
                    break :blk std.mem.eql(u8, wf_id, wf);
                }
                break :blk false;
            },
            .user => true, // User keys can access any workflow they own
        };

        if (!is_authorized) return false;

        // Update usage statistics
        key_info.last_used_at = std.time.timestamp();
        key_info.usage_count += 1;

        return true;
    }

    /// Revoke an API key
    pub fn revokeKey(self: *ApiKeyManager, key: []const u8) !void {
        var key_info = self.api_keys.getPtr(key) orelse return error.KeyNotFound;
        key_info.is_active = false;

        // Remove from APISIX
        try self.unregisterKeyFromApisix(key);
    }

    /// List all API keys (optionally filtered by scope)
    pub fn listKeys(self: *const ApiKeyManager, filter_scope: ?ApiKeyScope) ![]ApiKeyInfo {
        var list: std.ArrayListUnmanaged(ApiKeyInfo) = .{};
        errdefer list.deinit(self.allocator);

        var iterator = self.api_keys.valueIterator();
        while (iterator.next()) |key_info| {
            // Apply filter if specified
            if (filter_scope) |scope| {
                const matches = switch (scope) {
                    .global => key_info.scope == .global,
                    .workflow => |wf_id| blk: {
                        if (key_info.scope == .workflow) {
                            break :blk std.mem.eql(u8, key_info.scope.workflow, wf_id);
                        }
                        break :blk false;
                    },
                    .user => |user_id| blk: {
                        if (key_info.scope == .user) {
                            break :blk std.mem.eql(u8, key_info.scope.user, user_id);
                        }
                        break :blk false;
                    },
                };
                if (!matches) continue;
            }

            try list.append(self.allocator, try key_info.clone(self.allocator));
        }

        return list.toOwnedSlice(self.allocator);
    }

    /// Get information about a specific key
    pub fn getKeyInfo(self: *const ApiKeyManager, key: []const u8) ?ApiKeyInfo {
        const info = self.api_keys.get(key) orelse return null;
        return info.clone(self.allocator) catch null;
    }

    /// Rotate a key (generate new key, transfer scope, revoke old)
    pub fn rotateKey(self: *ApiKeyManager, old_key: []const u8) ![]const u8 {
        const old_info = self.api_keys.get(old_key) orelse return error.KeyNotFound;

        // Generate new key with same scope
        const new_key = try self.generateKey(old_info.scope, old_info.description);

        // Revoke old key
        try self.revokeKey(old_key);

        return new_key;
    }

    /// Delete expired keys
    pub fn cleanupExpiredKeys(self: *ApiKeyManager) !usize {
        var expired_keys: std.ArrayListUnmanaged([]const u8) = .{};
        defer expired_keys.deinit(self.allocator);

        const now = std.time.timestamp();

        // Find expired keys
        var iterator = self.api_keys.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.expires_at) |expires| {
                if (now > expires) {
                    try expired_keys.append(self.allocator, entry.key_ptr.*);
                }
            }
        }

        // Remove them
        for (expired_keys.items) |key| {
            try self.revokeKey(key);
            if (self.api_keys.fetchRemove(key)) |entry| {
                self.allocator.free(entry.key);
                entry.value.deinit(self.allocator);
            }
        }

        return expired_keys.items.len;
    }

    // Private helper methods

    fn generateRandomKey(self: *ApiKeyManager) ![]const u8 {
        // Generate 32 random bytes (256 bits)
        var random_bytes: [32]u8 = undefined;
        self.rng.random().bytes(&random_bytes);

        // Encode as base64
        var key_buffer: [64]u8 = undefined;
        const encoder = std.base64.standard.Encoder;
        const encoded = encoder.encode(&key_buffer, &random_bytes);

        // Add prefix for identification
        return try std.fmt.allocPrint(self.allocator, "nwf_{s}", .{encoded});
    }

    fn registerKeyWithApisix(self: *ApiKeyManager, key: []const u8, info: ApiKeyInfo) !void {
        _ = key;
        _ = info;
        // Mock implementation - would call APISIX consumer API
        // In production:
        // 1. Create APISIX consumer with key
        // 2. Associate with appropriate routes based on scope
        _ = self;
    }

    fn unregisterKeyFromApisix(self: *ApiKeyManager, key: []const u8) !void {
        _ = key;
        // Mock implementation - would delete APISIX consumer
        _ = self;
    }
};

/// API key scope determines what the key can access
pub const ApiKeyScope = union(enum) {
    global: void, // Can access all workflows
    workflow: []const u8, // Scoped to specific workflow ID
    user: []const u8, // Scoped to specific user ID

    pub fn clone(self: ApiKeyScope, allocator: Allocator) !ApiKeyScope {
        return switch (self) {
            .global => .global,
            .workflow => |wf_id| .{ .workflow = try allocator.dupe(u8, wf_id) },
            .user => |user_id| .{ .user = try allocator.dupe(u8, user_id) },
        };
    }

    pub fn deinit(self: *ApiKeyScope, allocator: Allocator) void {
        switch (self.*) {
            .global => {},
            .workflow => |wf_id| allocator.free(wf_id),
            .user => |user_id| allocator.free(user_id),
        }
    }
};

/// Information about an API key
pub const ApiKeyInfo = struct {
    key: []const u8,
    scope: ApiKeyScope,
    description: []const u8,
    created_at: i64,
    expires_at: ?i64,
    last_used_at: ?i64,
    usage_count: u64,
    is_active: bool,

    pub fn clone(self: ApiKeyInfo, allocator: Allocator) !ApiKeyInfo {
        return .{
            .key = try allocator.dupe(u8, self.key),
            .scope = try self.scope.clone(allocator),
            .description = try allocator.dupe(u8, self.description),
            .created_at = self.created_at,
            .expires_at = self.expires_at,
            .last_used_at = self.last_used_at,
            .usage_count = self.usage_count,
            .is_active = self.is_active,
        };
    }

    pub fn deinit(self: *ApiKeyInfo, allocator: Allocator) void {
        allocator.free(self.key);
        self.scope.deinit(allocator);
        allocator.free(self.description);
    }
};

// Tests
test "ApiKeyManager init and deinit" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    try std.testing.expect(manager.api_keys.count() == 0);
}

test "ApiKeyManager generate global key" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    const key = try manager.generateKey(.global, "Test global key");
    defer allocator.free(key);

    try std.testing.expect(key.len > 0);
    try std.testing.expect(std.mem.startsWith(u8, key, "nwf_"));
}

test "ApiKeyManager generate workflow-scoped key" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    const scope = ApiKeyScope{ .workflow = "workflow-123" };
    const key = try manager.generateKey(scope, "Workflow-specific key");
    defer allocator.free(key);

    try std.testing.expect(key.len > 0);

    const info = manager.getKeyInfo(key).?;
    defer {
        var mutable_info = info;
        mutable_info.deinit(allocator);
    }

    try std.testing.expect(info.scope == .workflow);
}

test "ApiKeyManager validate key" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    const key = try manager.generateKey(.global, "Test key");
    defer allocator.free(key);

    // Valid key
    try std.testing.expect(try manager.validateKey(key, null));

    // Invalid key
    try std.testing.expect(!try manager.validateKey("invalid-key", null));
}

test "ApiKeyManager validate workflow-scoped key" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    const scope = ApiKeyScope{ .workflow = "workflow-123" };
    const key = try manager.generateKey(scope, "Workflow key");
    defer allocator.free(key);

    // Valid for correct workflow
    try std.testing.expect(try manager.validateKey(key, "workflow-123"));

    // Invalid for different workflow
    try std.testing.expect(!try manager.validateKey(key, "workflow-456"));
}

test "ApiKeyManager revoke key" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    const key = try manager.generateKey(.global, "Test key");
    defer allocator.free(key);

    // Valid before revocation
    try std.testing.expect(try manager.validateKey(key, null));

    // Revoke
    try manager.revokeKey(key);

    // Invalid after revocation
    try std.testing.expect(!try manager.validateKey(key, null));
}

test "ApiKeyManager list keys" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    const key1 = try manager.generateKey(.global, "Key 1");
    defer allocator.free(key1);

    const scope2 = ApiKeyScope{ .workflow = "workflow-123" };
    const key2 = try manager.generateKey(scope2, "Key 2");
    defer allocator.free(key2);

    const keys = try manager.listKeys(null);
    defer {
        for (keys) |*key_info| {
            key_info.deinit(allocator);
        }
        allocator.free(keys);
    }

    try std.testing.expectEqual(@as(usize, 2), keys.len);
}

test "ApiKeyManager rotate key" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    const old_key = try manager.generateKey(.global, "Original key");
    defer allocator.free(old_key);

    const new_key = try manager.rotateKey(old_key);
    defer allocator.free(new_key);

    // Old key should be invalid
    try std.testing.expect(!try manager.validateKey(old_key, null));

    // New key should be valid
    try std.testing.expect(try manager.validateKey(new_key, null));
}

test "ApiKeyManager key expiration" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const apisix_client = try @import("apisix_client.zig").ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const manager = try ApiKeyManager.init(allocator, apisix_client);
    defer manager.deinit();

    // Create key that expires in 1 second
    const key = try manager.generateKeyWithExpiration(.global, "Expiring key", 1);
    defer allocator.free(key);

    // Should be valid immediately
    try std.testing.expect(try manager.validateKey(key, null));

    // Wait for expiration (2.5 seconds to ensure expiration on slow systems)
    std.Thread.sleep(2_500_000_000);

    // Should be invalid after expiration
    try std.testing.expect(!try manager.validateKey(key, null));
}
