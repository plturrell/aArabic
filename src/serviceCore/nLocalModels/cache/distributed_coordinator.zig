// ============================================================================
// Distributed Cache Coordinator - Day 57 Implementation
// ============================================================================
// Purpose: Multi-node cache implementation with replication
// Week: Week 12 (Days 56-60) - Distributed Caching
// Phase: Month 4 - HANA Integration & Scalability
// ============================================================================

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// CACHE NODE
// ============================================================================

/// Cache node status
pub const NodeStatus = enum {
    healthy,
    degraded,
    down,
};

/// Cache node in the distributed cluster
pub const CacheNode = struct {
    id: []const u8,
    host: []const u8,
    port: u16,
    status: NodeStatus,
    last_heartbeat: i64,
    stored_keys: u64,
    memory_used_mb: f32,
    hit_rate: f32,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        host: []const u8,
        port: u16,
    ) !CacheNode {
        return .{
            .id = try allocator.dupe(u8, id),
            .host = try allocator.dupe(u8, host),
            .port = port,
            .status = .healthy,
            .last_heartbeat = std.time.milliTimestamp(),
            .stored_keys = 0,
            .memory_used_mb = 0.0,
            .hit_rate = 0.0,
        };
    }
    
    pub fn deinit(self: *CacheNode, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.host);
    }
    
    pub fn updateHeartbeat(self: *CacheNode) void {
        self.last_heartbeat = std.time.milliTimestamp();
        
        // Update status based on heartbeat age
        const age_ms = std.time.milliTimestamp() - self.last_heartbeat;
        if (age_ms > 60000) { // 60 seconds
            self.status = NodeStatus.down;
        } else if (age_ms > 30000) { // 30 seconds
            self.status = NodeStatus.degraded;
        } else {
            self.status = NodeStatus.healthy;
        }
    }
    
    pub fn isHealthy(self: *const CacheNode) bool {
        return self.status == NodeStatus.healthy;
    }
};

// ============================================================================
// CACHE ENTRY
// ============================================================================

/// Versioned cache entry with replication metadata
pub const CacheEntry = struct {
    key: []const u8,
    value: []const u8,
    version: u64,
    created_at: i64,
    expires_at: i64,
    replicated_nodes: [][]const u8,
    
    pub fn init(
        allocator: Allocator,
        key: []const u8,
        value: []const u8,
        ttl_ms: i64,
    ) !CacheEntry {
        const now = std.time.milliTimestamp();
        
        return .{
            .key = try allocator.dupe(u8, key),
            .value = try allocator.dupe(u8, value),
            .version = 1,
            .created_at = now,
            .expires_at = now + ttl_ms,
            .replicated_nodes = try allocator.alloc([]const u8, 0),
        };
    }
    
    pub fn deinit(self: *CacheEntry, allocator: Allocator) void {
        allocator.free(self.key);
        allocator.free(self.value);
        for (self.replicated_nodes) |node_id| {
            allocator.free(node_id);
        }
        allocator.free(self.replicated_nodes);
    }
    
    pub fn isExpired(self: *const CacheEntry) bool {
        return std.time.milliTimestamp() > self.expires_at;
    }
    
    pub fn addReplicatedNode(self: *CacheEntry, allocator: Allocator, node_id: []const u8) !void {
        const new_len = self.replicated_nodes.len + 1;
        const new_nodes = try allocator.alloc([]const u8, new_len);
        
        for (self.replicated_nodes, 0..) |old_node, i| {
            new_nodes[i] = old_node;
        }
        new_nodes[new_len - 1] = try allocator.dupe(u8, node_id);
        
        allocator.free(self.replicated_nodes);
        self.replicated_nodes = new_nodes;
    }
};

// ============================================================================
// DISTRIBUTED COORDINATOR
// ============================================================================

/// Configuration for distributed cache
pub const DistributedCacheConfig = struct {
    replication_factor: u32 = 2, // Replicate to N nodes
    consistency_level: ConsistencyLevel = .eventual,
    heartbeat_interval_ms: u64 = 5000,
    node_timeout_ms: u64 = 30000,
    max_nodes: u32 = 10,
};

/// Consistency level for cache operations
pub const ConsistencyLevel = enum {
    eventual, // Write to primary, async replicate
    quorum, // Write to majority before ack
    strong, // Write to all replicas before ack
};

/// Distributed cache coordinator
pub const DistributedCoordinator = struct {
    allocator: Allocator,
    config: DistributedCacheConfig,
    nodes: std.ArrayList(CacheNode),
    cache: std.StringHashMap(CacheEntry),
    mutex: std.Thread.Mutex,
    
    pub fn init(allocator: Allocator, config: DistributedCacheConfig) !*DistributedCoordinator {
        const coordinator = try allocator.create(DistributedCoordinator);
        coordinator.* = .{
            .allocator = allocator,
            .config = config,
            .nodes = std.ArrayList(CacheNode){},
            .cache = std.StringHashMap(CacheEntry).init(allocator),
            .mutex = .{},
        };
        return coordinator;
    }
    
    pub fn deinit(self: *DistributedCoordinator) void {
        // Clean up nodes
        for (self.nodes.items) |*node| {
            node.deinit(self.allocator);
        }
        self.nodes.deinit();
        
        // Clean up cache entries
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            var mutable_entry = entry.value_ptr.*;
            mutable_entry.deinit(self.allocator);
        }
        self.cache.deinit();
        
        self.allocator.destroy(self);
    }
    
    // ========================================================================
    // NODE MANAGEMENT
    // ========================================================================
    
    /// Register a new cache node in the cluster
    pub fn registerNode(
        self: *DistributedCoordinator,
        id: []const u8,
        host: []const u8,
        port: u16,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.nodes.items.len >= self.config.max_nodes) {
            return error.MaxNodesReached;
        }
        
        // Check if node already exists
        for (self.nodes.items) |*node| {
            if (std.mem.eql(u8, node.id, id)) {
                node.updateHeartbeat();
                return;
            }
        }
        
        // Add new node
        const node = try CacheNode.init(self.allocator, id, host, port);
        try self.nodes.append(node);
        
        std.log.info("Registered cache node: {s} at {s}:{d}", .{ id, host, port });
    }
    
    /// Remove a node from the cluster
    pub fn removeNode(self: *DistributedCoordinator, id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.nodes.items, 0..) |*node, i| {
            if (std.mem.eql(u8, node.id, id)) {
                node.deinit(self.allocator);
                _ = self.nodes.orderedRemove(i);
                std.log.info("Removed cache node: {s}", .{id});
                return;
            }
        }
        
        return error.NodeNotFound;
    }
    
    /// Get healthy nodes for replication
    fn getHealthyNodes(self: *DistributedCoordinator) []CacheNode {
        var healthy = std.ArrayList(CacheNode){};
        
        for (self.nodes.items) |node| {
            if (node.isHealthy()) {
                healthy.append(node) catch continue;
            }
        }
        
        return healthy.toOwnedSlice() catch &[_]CacheNode{};
    }
    
    /// Select N nodes for replication using consistent hashing
    fn selectReplicationNodes(
        self: *DistributedCoordinator,
        key: []const u8,
        count: u32,
    ) ![]CacheNode {
        const healthy = self.getHealthyNodes();
        defer self.allocator.free(healthy);
        
        if (healthy.len == 0) {
            return error.NoHealthyNodes;
        }
        
        // Use hash of key to consistently select nodes
        const hash = std.hash.Wyhash.hash(0, key);
        const start_idx = hash % healthy.len;
        
        const actual_count = @min(count, @as(u32, @intCast(healthy.len)));
        var selected = try self.allocator.alloc(CacheNode, actual_count);
        
        var i: u32 = 0;
        while (i < actual_count) : (i += 1) {
            const idx = (start_idx + i) % healthy.len;
            selected[i] = healthy[idx];
        }
        
        return selected;
    }
    
    // ========================================================================
    // CACHE OPERATIONS WITH REPLICATION
    // ========================================================================
    
    /// Write to cache with replication
    pub fn put(
        self: *DistributedCoordinator,
        key: []const u8,
        value: []const u8,
        ttl_ms: i64,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Create cache entry
        const entry = try CacheEntry.init(self.allocator, key, value, ttl_ms);
        
        // Store locally (primary)
        const gop = try self.cache.getOrPut(key);
        if (gop.found_existing) {
            var old_entry = gop.value_ptr.*;
            old_entry.deinit(self.allocator);
        }
        gop.value_ptr.* = entry;
        
        // Select nodes for replication
        const replication_nodes = try self.selectReplicationNodes(
            key,
            self.config.replication_factor,
        );
        defer self.allocator.free(replication_nodes);
        
        // Replicate based on consistency level
        switch (self.config.consistency_level) {
            .eventual => {
                // Fire and forget (async replication)
                for (replication_nodes) |node| {
                    replicateToNodeAsync(self, node, key, value, ttl_ms) catch {
                        std.log.warn("Async replication failed to node {s}", .{node.id});
                    };
                }
            },
            .quorum => {
                // Wait for majority
                const required = (replication_nodes.len / 2) + 1;
                var successful: u32 = 1; // Primary write succeeded
                
                for (replication_nodes) |node| {
                    if (replicateToNode(self, node, key, value, ttl_ms)) {
                        successful += 1;
                        if (successful >= required) break;
                    } else |_| {}
                }
                
                if (successful < required) {
                    return error.QuorumNotReached;
                }
            },
            .strong => {
                // Wait for all replicas
                for (replication_nodes) |node| {
                    try replicateToNode(self, node, key, value, ttl_ms);
                }
            },
        }
        
        std.log.debug("Cached key '{s}' with {d} replicas", .{ key, replication_nodes.len });
    }
    
    /// Read from cache with load distribution
    pub fn get(self: *DistributedCoordinator, key: []const u8) !?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Try local cache first
        if (self.cache.get(key)) |entry| {
            if (!entry.isExpired()) {
                return try self.allocator.dupe(u8, entry.value);
            } else {
                // Remove expired entry
                var mutable_entry = entry;
                mutable_entry.deinit(self.allocator);
                _ = self.cache.remove(key);
            }
        }
        
        // If not found locally, return null
        // (Read repair from replicas would be implemented here in production)
        return null;
    }
    
    /// Invalidate cache entry across all nodes
    pub fn invalidate(self: *DistributedCoordinator, key: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Remove from local cache
        if (self.cache.fetchRemove(key)) |kv| {
            var entry = kv.value;
            entry.deinit(self.allocator);
        }
        
        // Broadcast invalidation to all nodes
        for (self.nodes.items) |node| {
            invalidateOnNode(self, node, key) catch |err| {
                std.log.warn("Failed to invalidate key '{s}' on node {s}: {}", .{ key, node.id, err });
            };
        }
        
        std.log.debug("Invalidated key '{s}' across cluster", .{key});
    }
    
    // ========================================================================
    // REPLICATION OPERATIONS
    // ========================================================================
    
    /// Replicate cache entry to specific node (synchronous)
    fn replicateToNode(
        self: *DistributedCoordinator,
        node: CacheNode,
        key: []const u8,
        value: []const u8,
        ttl_ms: i64,
    ) !void {
        _ = self; // Used for future HTTP implementation
        _ = key; // Will be used in HTTP body
        _ = value; // Will be used in HTTP body
        _ = ttl_ms; // Will be used in HTTP body
        
        // Real implementation would use HTTP POST to node
        // POST http://{node.host}:{node.port}/cache/replicate
        // Body: { "key": "...", "value": "...", "ttl_ms": 300000 }
        
        // For now, simulate successful replication
        std.log.debug("Replicated to node {s}", .{node.id});
    }
    
    /// Replicate cache entry to specific node (asynchronous)
    fn replicateToNodeAsync(
        self: *DistributedCoordinator,
        node: CacheNode,
        key: []const u8,
        value: []const u8,
        ttl_ms: i64,
    ) !void {
        _ = self; // Used for future async implementation
        _ = key; // Will be used in async call
        _ = value; // Will be used in async call
        _ = ttl_ms; // Will be used in async call
        
        // Real implementation would spawn thread or use event loop
        // For now, just log the async attempt
        std.log.debug("Async replication scheduled for node {s}", .{node.id});
    }
    
    /// Read cache entry from specific node
    fn readFromNode(
        self: *DistributedCoordinator,
        node: CacheNode,
        key: []const u8,
    ) ![]const u8 {
        _ = self;
        _ = node;
        _ = key;
        
        // Real implementation would use HTTP GET from node
        // GET http://{node.host}:{node.port}/cache/get/{key}
        
        return error.NotFoundOnNode;
    }
    
    /// Invalidate cache entry on specific node
    fn invalidateOnNode(
        self: *DistributedCoordinator,
        node: CacheNode,
        key: []const u8,
    ) !void {
        _ = self; // Used for future HTTP implementation
        _ = key; // Will be used in HTTP DELETE
        
        // Real implementation would use HTTP DELETE
        // DELETE http://{node.host}:{node.port}/cache/invalidate/{key}
        
        std.log.debug("Invalidated on node {s}", .{node.id});
    }
    
    // ========================================================================
    // MONITORING
    // ========================================================================
    
    /// Get cluster statistics
    pub fn getClusterStats(self: *DistributedCoordinator) ClusterStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var healthy_count: u32 = 0;
        var total_keys: u64 = 0;
        var total_memory: f32 = 0.0;
        
        for (self.nodes.items) |node| {
            if (node.isHealthy()) healthy_count += 1;
            total_keys += node.stored_keys;
            total_memory += node.memory_used_mb;
        }
        
        return .{
            .total_nodes = @intCast(self.nodes.items.len),
            .healthy_nodes = healthy_count,
            .total_keys = total_keys + self.cache.count(),
            .total_memory_mb = total_memory,
            .local_keys = @intCast(self.cache.count()),
        };
    }
};

/// Cluster statistics
pub const ClusterStats = struct {
    total_nodes: u32,
    healthy_nodes: u32,
    total_keys: u64,
    total_memory_mb: f32,
    local_keys: u32,
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "CacheNode: initialization and cleanup" {
    const allocator = std.testing.allocator;
    
    var node = try CacheNode.init(allocator, "node-1", "localhost", 6379);
    defer node.deinit(allocator);
    
    try std.testing.expectEqualStrings("node-1", node.id);
    try std.testing.expectEqualStrings("localhost", node.host);
    try std.testing.expectEqual(@as(u16, 6379), node.port);
    try std.testing.expect(node.isHealthy());
}

test "CacheEntry: creation and expiration" {
    const allocator = std.testing.allocator;
    
    var entry = try CacheEntry.init(allocator, "key1", "value1", 1000);
    defer entry.deinit(allocator);
    
    try std.testing.expectEqualStrings("key1", entry.key);
    try std.testing.expectEqualStrings("value1", entry.value);
    try std.testing.expectEqual(@as(u64, 1), entry.version);
    try std.testing.expect(!entry.isExpired()); // Should not be expired immediately
}

test "DistributedCoordinator: node registration" {
    const allocator = std.testing.allocator;
    
    const config = DistributedCacheConfig{
        .replication_factor = 2,
        .consistency_level = .eventual,
    };
    
    const coordinator = try DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    try coordinator.registerNode("node-1", "localhost", 6379);
    try coordinator.registerNode("node-2", "localhost", 6380);
    
    const stats = coordinator.getClusterStats();
    try std.testing.expectEqual(@as(u32, 2), stats.total_nodes);
    try std.testing.expectEqual(@as(u32, 2), stats.healthy_nodes);
}

test "DistributedCoordinator: cache put and get" {
    const allocator = std.testing.allocator;
    
    const config = DistributedCacheConfig{
        .replication_factor = 2,
        .consistency_level = .eventual,
    };
    
    const coordinator = try DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    // Register nodes
    try coordinator.registerNode("node-1", "localhost", 6379);
    try coordinator.registerNode("node-2", "localhost", 6380);
    
    // Put value
    try coordinator.put("test-key", "test-value", 300000);
    
    // Get value
    const value = try coordinator.get("test-key");
    try std.testing.expect(value != null);
    
    if (value) |v| {
        defer allocator.free(v);
        try std.testing.expectEqualStrings("test-value", v);
    }
    
    // Check stats
    const stats = coordinator.getClusterStats();
    try std.testing.expectEqual(@as(u32, 1), stats.local_keys);
}

test "DistributedCoordinator: cache invalidation" {
    const allocator = std.testing.allocator;
    
    const config = DistributedCacheConfig{};
    const coordinator = try DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    try coordinator.registerNode("node-1", "localhost", 6379);
    
    // Put and invalidate
    try coordinator.put("key1", "value1", 300000);
    try coordinator.invalidate("key1");
    
    // Should not be found
    const value = try coordinator.get("key1");
    try std.testing.expect(value == null);
}

test "DistributedCoordinator: cluster stats" {
    const allocator = std.testing.allocator;
    
    const config = DistributedCacheConfig{};
    const coordinator = try DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    try coordinator.registerNode("node-1", "localhost", 6379);
    try coordinator.registerNode("node-2", "localhost", 6380);
    try coordinator.registerNode("node-3", "localhost", 6381);
    
    const stats = coordinator.getClusterStats();
    try std.testing.expectEqual(@as(u32, 3), stats.total_nodes);
    try std.testing.expectEqual(@as(u32, 3), stats.healthy_nodes);
}
