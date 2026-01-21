// ============================================================================
// Distributed Cache Coordinator - Day 56 Implementation
// ============================================================================
// Purpose: Coordinate distributed caching across multiple nodes
// Week: Week 12 (Days 56-60) - Distributed Caching
// Phase: Month 4 - HANA Integration & Scalability
// ============================================================================

const std = @import("std");

// ============================================================================
// CACHE NODE TYPES
// ============================================================================

/// Cache node information
pub const CacheNode = struct {
    node_id: []const u8,
    host: []const u8,
    port: u16,
    status: NodeStatus,
    last_heartbeat: i64,
    cache_size: u64,
    max_cache_size: u64,
    cpu_usage: f32,
    memory_usage: f32,
    
    pub const NodeStatus = enum {
        active,
        degraded,
        failed,
        
        pub fn toString(self: NodeStatus) []const u8 {
            return switch (self) {
                .active => "ACTIVE",
                .degraded => "DEGRADED",
                .failed => "FAILED",
            };
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, node_id: []const u8, host: []const u8, port: u16) !CacheNode {
        return .{
            .node_id = try allocator.dupe(u8, node_id),
            .host = try allocator.dupe(u8, host),
            .port = port,
            .status = .active,
            .last_heartbeat = std.time.milliTimestamp(),
            .cache_size = 0,
            .max_cache_size = 1024 * 1024 * 1024, // 1GB default
            .cpu_usage = 0.0,
            .memory_usage = 0.0,
        };
    }
    
    pub fn deinit(self: *CacheNode, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);
        allocator.free(self.host);
    }
    
    pub fn isHealthy(self: *const CacheNode) bool {
        const now = std.time.milliTimestamp();
        const heartbeat_timeout = 30_000; // 30 seconds
        
        return self.status == .active and 
               (now - self.last_heartbeat) < heartbeat_timeout;
    }
    
    pub fn getUtilization(self: *const CacheNode) f32 {
        if (self.max_cache_size == 0) return 0.0;
        return @as(f32, @floatFromInt(self.cache_size)) / 
               @as(f32, @floatFromInt(self.max_cache_size));
    }
};

/// Cache entry with replication metadata
pub const CacheEntry = struct {
    key: []const u8,
    value: []const u8,
    version: u64,
    timestamp: i64,
    primary_node: []const u8,
    replica_nodes: std.ArrayList([]const u8),
    
    pub fn init(
        allocator: std.mem.Allocator,
        key: []const u8,
        value: []const u8,
        primary_node: []const u8,
    ) !CacheEntry {
        return .{
            .key = try allocator.dupe(u8, key),
            .value = try allocator.dupe(u8, value),
            .version = 1,
            .timestamp = std.time.milliTimestamp(),
            .primary_node = try allocator.dupe(u8, primary_node),
            .replica_nodes = std.ArrayList([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *CacheEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.key);
        allocator.free(self.value);
        allocator.free(self.primary_node);
        for (self.replica_nodes.items) |node| {
            allocator.free(node);
        }
        self.replica_nodes.deinit();
    }
};

// ============================================================================
// CONSISTENCY PROTOCOL
// ============================================================================

/// Consistency strategy for distributed cache
pub const ConsistencyStrategy = enum {
    eventual, // Eventual consistency (fastest writes)
    strong,   // Strong consistency (slower writes)
    quorum,   // Quorum-based consistency (balanced)
    
    pub fn toString(self: ConsistencyStrategy) []const u8 {
        return switch (self) {
            .eventual => "EVENTUAL",
            .strong => "STRONG",
            .quorum => "QUORUM",
        };
    }
};

/// Replication configuration
pub const ReplicationConfig = struct {
    replication_factor: u32, // Number of replicas (default: 3)
    consistency_strategy: ConsistencyStrategy,
    sync_timeout_ms: u32, // Timeout for sync replication (default: 100ms)
    async_replication: bool, // Enable async replication
    
    pub fn init() ReplicationConfig {
        return .{
            .replication_factor = 3,
            .consistency_strategy = .eventual,
            .sync_timeout_ms = 100,
            .async_replication = true,
        };
    }
};

// ============================================================================
// DISTRIBUTED COORDINATOR
// ============================================================================

pub const DistributedCoordinator = struct {
    allocator: std.mem.Allocator,
    
    // Node registry
    nodes: std.ArrayList(CacheNode),
    node_map: std.StringHashMap(usize), // node_id -> index in nodes
    
    // Cache metadata
    cache_map: std.StringHashMap(CacheEntry),
    
    // Configuration
    config: ReplicationConfig,
    
    // Statistics
    total_writes: u64,
    total_reads: u64,
    replication_successes: u64,
    replication_failures: u64,
    
    pub fn init(allocator: std.mem.Allocator, config: ReplicationConfig) DistributedCoordinator {
        return .{
            .allocator = allocator,
            .nodes = std.ArrayList(CacheNode).init(allocator),
            .node_map = std.StringHashMap(usize).init(allocator),
            .cache_map = std.StringHashMap(CacheEntry).init(allocator),
            .config = config,
            .total_writes = 0,
            .total_reads = 0,
            .replication_successes = 0,
            .replication_failures = 0,
        };
    }
    
    pub fn deinit(self: *DistributedCoordinator) void {
        for (self.nodes.items) |*node| {
            node.deinit(self.allocator);
        }
        self.nodes.deinit();
        
        var cache_iter = self.cache_map.iterator();
        while (cache_iter.next()) |entry| {
            var value = entry.value_ptr;
            value.deinit(self.allocator);
        }
        self.cache_map.deinit();
        
        self.node_map.deinit();
    }
    
    /// Register a new cache node
    pub fn registerNode(self: *DistributedCoordinator, node: CacheNode) !void {
        const index = self.nodes.items.len;
        try self.nodes.append(node);
        try self.node_map.put(node.node_id, index);
    }
    
    /// Unregister a cache node
    pub fn unregisterNode(self: *DistributedCoordinator, node_id: []const u8) !void {
        if (self.node_map.get(node_id)) |index| {
            var node = &self.nodes.items[index];
            node.status = .failed;
            
            // TODO: Trigger rebalancing and re-replication
        }
    }
    
    /// Update node heartbeat
    pub fn updateHeartbeat(
        self: *DistributedCoordinator,
        node_id: []const u8,
        cpu_usage: f32,
        memory_usage: f32,
        cache_size: u64,
    ) !void {
        if (self.node_map.get(node_id)) |index| {
            var node = &self.nodes.items[index];
            node.last_heartbeat = std.time.milliTimestamp();
            node.cpu_usage = cpu_usage;
            node.memory_usage = memory_usage;
            node.cache_size = cache_size;
            
            // Update status based on health
            if (node.cpu_usage > 0.9 or node.memory_usage > 0.9) {
                node.status = .degraded;
            } else {
                node.status = .active;
            }
        }
    }
    
    /// Get healthy nodes
    pub fn getHealthyNodes(self: *const DistributedCoordinator, allocator: std.mem.Allocator) ![]const *const CacheNode {
        var healthy = std.ArrayList(*const CacheNode).init(allocator);
        
        for (self.nodes.items) |*node| {
            if (node.isHealthy()) {
                try healthy.append(node);
            }
        }
        
        return healthy.toOwnedSlice();
    }
    
    /// Select primary node for a key (consistent hashing)
    pub fn selectPrimaryNode(self: *const DistributedCoordinator, key: []const u8) !*const CacheNode {
        if (self.nodes.items.len == 0) return error.NoNodesAvailable;
        
        // Simple hash-based selection (consistent hashing would be better)
        var hash: u32 = 0;
        for (key) |byte| {
            hash = hash *% 31 +% byte;
        }
        
        // Find first healthy node starting from hash position
        const start_idx = hash % @as(u32, @intCast(self.nodes.items.len));
        var idx: usize = start_idx;
        
        while (true) {
            const node = &self.nodes.items[idx];
            if (node.isHealthy()) {
                return node;
            }
            
            idx = (idx + 1) % self.nodes.items.len;
            if (idx == start_idx) {
                return error.NoHealthyNodes;
            }
        }
    }
    
    /// Select replica nodes for a key
    pub fn selectReplicaNodes(
        self: *const DistributedCoordinator,
        key: []const u8,
        primary_node: *const CacheNode,
        allocator: std.mem.Allocator,
    ) ![]const *const CacheNode {
        _ = key; // unused but kept for future consistent hashing
        
        var replicas = std.ArrayList(*const CacheNode).init(allocator);
        
        const needed_replicas = self.config.replication_factor - 1; // -1 for primary
        if (needed_replicas == 0) return try replicas.toOwnedSlice();
        
        // Find healthy nodes that are not the primary
        for (self.nodes.items) |*node| {
            if (node.isHealthy() and 
                !std.mem.eql(u8, node.node_id, primary_node.node_id)) {
                try replicas.append(node);
                if (replicas.items.len >= needed_replicas) break;
            }
        }
        
        return try replicas.toOwnedSlice();
    }
    
    /// Write to distributed cache
    pub fn write(
        self: *DistributedCoordinator,
        key: []const u8,
        value: []const u8,
    ) !void {
        self.total_writes += 1;
        
        // Select primary node
        const primary = try self.selectPrimaryNode(key);
        
        // Create cache entry
        var entry = try CacheEntry.init(self.allocator, key, value, primary.node_id);
        
        // Select replica nodes
        const replicas = try self.selectReplicaNodes(key, primary, self.allocator);
        defer self.allocator.free(replicas);
        
        // Add replicas to entry
        for (replicas) |replica| {
            const replica_id = try self.allocator.dupe(u8, replica.node_id);
            try entry.replica_nodes.append(replica_id);
        }
        
        // Store metadata
        try self.cache_map.put(key, entry);
        
        // Perform replication based on consistency strategy
        switch (self.config.consistency_strategy) {
            .eventual => {
                // Async replication - return immediately
                // TODO: Queue replication requests
                self.replication_successes += 1;
            },
            .strong => {
                // Sync replication to all replicas
                // TODO: Wait for all replicas to acknowledge
                self.replication_successes += 1;
            },
            .quorum => {
                // Wait for majority acknowledgment
                // TODO: Wait for (replication_factor / 2) + 1 acks
                self.replication_successes += 1;
            },
        }
    }
    
    /// Read from distributed cache
    pub fn read(self: *DistributedCoordinator, key: []const u8) !?[]const u8 {
        self.total_reads += 1;
        
        if (self.cache_map.get(key)) |entry| {
            // Check if primary node is healthy
            if (self.node_map.get(entry.primary_node)) |idx| {
                const primary = &self.nodes.items[idx];
                if (primary.isHealthy()) {
                    return entry.value;
                }
            }
            
            // Try replica nodes
            for (entry.replica_nodes.items) |replica_id| {
                if (self.node_map.get(replica_id)) |idx| {
                    const replica = &self.nodes.items[idx];
                    if (replica.isHealthy()) {
                        return entry.value;
                    }
                }
            }
            
            // All nodes failed
            return error.AllNodesFailed;
        }
        
        return null;
    }
    
    /// Invalidate cache entry across all nodes
    pub fn invalidate(self: *DistributedCoordinator, key: []const u8) !void {
        if (self.cache_map.getPtr(key)) |entry_ptr| {
            entry_ptr.deinit(self.allocator);
            _ = self.cache_map.remove(key);
            
            // TODO: Send invalidation broadcast to all nodes
        }
    }
    
    /// Get coordinator statistics
    pub fn getStatistics(self: *const DistributedCoordinator) Statistics {
        var healthy_count: u32 = 0;
        var total_cache_size: u64 = 0;
        
        for (self.nodes.items) |*node| {
            if (node.isHealthy()) {
                healthy_count += 1;
                total_cache_size += node.cache_size;
            }
        }
        
        const replication_success_rate = if (self.total_writes > 0)
            @as(f32, @floatFromInt(self.replication_successes)) / 
            @as(f32, @floatFromInt(self.total_writes))
        else
            0.0;
        
        return .{
            .total_nodes = @intCast(self.nodes.items.len),
            .healthy_nodes = healthy_count,
            .total_cache_entries = @intCast(self.cache_map.count()),
            .total_cache_size = total_cache_size,
            .total_writes = self.total_writes,
            .total_reads = self.total_reads,
            .replication_success_rate = replication_success_rate,
        };
    }
    
    pub const Statistics = struct {
        total_nodes: u32,
        healthy_nodes: u32,
        total_cache_entries: u32,
        total_cache_size: u64,
        total_writes: u64,
        total_reads: u64,
        replication_success_rate: f32,
    };
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "CacheNode: initialization and health check" {
    const allocator = std.testing.allocator;
    
    var node = try CacheNode.init(allocator, "node-1", "localhost", 8080);
    defer node.deinit(allocator);
    
    try std.testing.expect(node.isHealthy());
    try std.testing.expectEqual(CacheNode.NodeStatus.active, node.status);
}

test "CacheNode: utilization calculation" {
    const allocator = std.testing.allocator;
    
    var node = try CacheNode.init(allocator, "node-1", "localhost", 8080);
    defer node.deinit(allocator);
    
    node.cache_size = 512 * 1024 * 1024; // 512MB
    node.max_cache_size = 1024 * 1024 * 1024; // 1GB
    
    const util = node.getUtilization();
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), util, 0.01); // 50% ±1%
}

test "DistributedCoordinator: node registration" {
    const allocator = std.testing.allocator;
    
    const config = ReplicationConfig.init();
    var coordinator = DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    const node1 = try CacheNode.init(allocator, "node-1", "localhost", 8080);
    const node2 = try CacheNode.init(allocator, "node-2", "localhost", 8081);
    
    try coordinator.registerNode(node1);
    try coordinator.registerNode(node2);
    
    try std.testing.expectEqual(@as(usize, 2), coordinator.nodes.items.len);
}

test "DistributedCoordinator: write and read" {
    const allocator = std.testing.allocator;
    
    var config = ReplicationConfig.init();
    config.replication_factor = 2;
    
    var coordinator = DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    // Register nodes
    const node1 = try CacheNode.init(allocator, "node-1", "localhost", 8080);
    const node2 = try CacheNode.init(allocator, "node-2", "localhost", 8081);
    
    try coordinator.registerNode(node1);
    try coordinator.registerNode(node2);
    
    // Write
    try coordinator.write("test-key", "test-value");
    
    // Read
    const value = try coordinator.read("test-key");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("test-value", value.?);
    
    // Check statistics
    const stats = coordinator.getStatistics();
    try std.testing.expectEqual(@as(u64, 1), stats.total_writes);
    try std.testing.expectEqual(@as(u64, 1), stats.total_reads);
}

test "DistributedCoordinator: primary node selection" {
    const allocator = std.testing.allocator;
    
    const config = ReplicationConfig.init();
    var coordinator = DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    const node1 = try CacheNode.init(allocator, "node-1", "localhost", 8080);
    const node2 = try CacheNode.init(allocator, "node-2", "localhost", 8081);
    
    try coordinator.registerNode(node1);
    try coordinator.registerNode(node2);
    
    const primary = try coordinator.selectPrimaryNode("some-key");
    try std.testing.expect(primary.isHealthy());
}

test "DistributedCoordinator: heartbeat updates" {
    const allocator = std.testing.allocator;
    
    const config = ReplicationConfig.init();
    var coordinator = DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    const node = try CacheNode.init(allocator, "node-1", "localhost", 8080);
    try coordinator.registerNode(node);
    
    // Update heartbeat with high resource usage
    try coordinator.updateHeartbeat("node-1", 0.95, 0.85, 500_000_000);
    
    const updated_node = &coordinator.nodes.items[0];
    try std.testing.expectEqual(CacheNode.NodeStatus.degraded, updated_node.status);
}

test "DistributedCoordinator: cache invalidation" {
    const allocator = std.testing.allocator;
    
    const config = ReplicationConfig.init();
    var coordinator = DistributedCoordinator.init(allocator, config);
    defer coordinator.deinit();
    
    const node = try CacheNode.init(allocator, "node-1", "localhost", 8080);
    try coordinator.registerNode(node);
    
    // Write
    try coordinator.write("test-key", "test-value");
    
    // Invalidate
    try coordinator.invalidate("test-key");
    
    // Read should return null
    const value = try coordinator.read("test-key");
    try std.testing.expect(value == null);
}
