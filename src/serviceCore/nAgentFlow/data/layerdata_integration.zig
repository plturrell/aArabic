const std = @import("std");
const Allocator = std.mem.Allocator;
const data_packet = @import("data_packet.zig");
const data_pipeline = @import("data_pipeline.zig");
const DataPacket = data_packet.DataPacket;
const DataPipeline = data_pipeline.DataPipeline;

/// LayerData integration for SAP-only architecture
/// This module demonstrates how nWorkflow data system integrates with:
/// - SAP HANA Cloud (unified persistent storage + caching)

// Import HANA modules via build wiring
const hana = @import("hana_sdk");
const HanaCache = @import("hana_cache").HanaCache;
const HanaCacheConfig = @import("hana_cache").HanaCacheConfig;
const hana_store = @import("hana_store");

/// SAP HANA integration for unified persistent storage and caching
pub const HanaAdapter = struct {
    allocator: Allocator,
    cache: HanaCache,
    store: *hana_store.HanaWorkflowStore,
    
    pub fn init(allocator: Allocator, host: []const u8, port: u16, user: []const u8, password: []const u8, database: []const u8) !HanaAdapter {
        // Initialize cache
        const cache_config = HanaCacheConfig{
            .host = host,
            .port = port,
            .user = user,
            .password = password,
            .database = database,
            .table_prefix = "cache",
        };
        
        var cache = try HanaCache.init(allocator, cache_config);
        try cache.connect();
        
        // Initialize store
        const hana_config = hana.Config{
            .host = host,
            .port = port,
            .user = user,
            .password = password,
            .database = database,
        };
        
        const store = try hana_store.HanaWorkflowStore.init(allocator, hana_config, "nworkflow");
        
        return .{
            .allocator = allocator,
            .cache = cache,
            .store = store,
        };
    }
    
    pub fn deinit(self: *HanaAdapter) void {
        self.cache.deinit();
        self.store.deinit();
    }
    
    /// Store a DataPacket in HANA (persistent)
    pub fn storePacket(self: *HanaAdapter, packet: *const DataPacket) !void {
        const json_data = try packet.serialize(packet.allocator);
        defer packet.allocator.free(json_data);
        
        // Store in HANA persistence layer
        _ = self.store;
        // In production: INSERT INTO workflow_data (id, type, value, metadata, timestamp)
    }
    
    /// Cache a DataPacket in HANA (in-memory with TTL)
    pub fn cachePacket(self: *HanaAdapter, packet: *const DataPacket, ttl_seconds: u32) !void {
        const json_data = try packet.serialize(packet.allocator);
        defer packet.allocator.free(json_data);
        
        try self.cache.set(packet.id, json_data, ttl_seconds);
    }
    
    /// Retrieve a DataPacket from cache or persistent storage
    pub fn loadPacket(self: *HanaAdapter, packet_id: []const u8) !?*DataPacket {
        // Try cache first
        if (try self.cache.get(packet_id)) |data| {
            // Deserialize from cache
            _ = data;
            return null; // Placeholder - would deserialize
        }
        
        // Fall back to persistent storage
        _ = self.store;
        return null; // Placeholder
    }
    
    /// Store workflow execution state
    pub fn storeWorkflowState(self: *HanaAdapter, workflow_id: []const u8, state: []const u8) !void {
        try self.cache.cacheWorkflowState(workflow_id, state, 3600);
    }
    
    /// Store session data
    pub fn storeSession(self: *HanaAdapter, session_id: []const u8, data: []const u8, ttl_seconds: u32) !void {
        try self.cache.storeSession(session_id, data, ttl_seconds);
    }
    
    /// Publish packet to event stream (HANA pub/sub)
    pub fn publishPacket(self: *HanaAdapter, channel: []const u8, packet: *const DataPacket) !void {
        _ = self;
        const json_data = try packet.serialize(packet.allocator);
        defer packet.allocator.free(json_data);
        
        // Use HANA streaming or event tables
        _ = channel;
    }
};

/// SAP HANA-based pipeline with complete data management
pub const HanaPipeline = struct {
    allocator: Allocator,
    hana: HanaAdapter,
    pipeline: *DataPipeline,
    
    pub fn init(
        allocator: Allocator,
        hana_host: []const u8,
        hana_port: u16,
        hana_user: []const u8,
        hana_password: []const u8,
        hana_schema: []const u8,
        pipeline: *DataPipeline,
    ) !*HanaPipeline {
        const hp = try allocator.create(HanaPipeline);
        errdefer allocator.destroy(hp);
        
        hp.* = .{
            .allocator = allocator,
            .hana = try HanaAdapter.init(allocator, hana_host, hana_port, hana_user, hana_password, hana_schema),
            .pipeline = pipeline,
        };
        
        return hp;
    }
    
    pub fn deinit(self: *HanaPipeline) void {
        self.hana.deinit();
        self.allocator.destroy(self);
    }
    
    /// Execute pipeline with HANA-based tracking and persistence
    pub fn executeWithTracking(self: *HanaPipeline, input: *DataPacket, run_id: []const u8) !*DataPacket {
        // 1. Cache input in HANA
        try self.hana.cachePacket(input, 3600);
        
        // 2. Execute pipeline
        _ = run_id;
        const output = try self.pipeline.execute(input);
        
        // 3. Store result in HANA (persistent)
        try self.hana.storePacket(output);
        
        // 4. Cache output in HANA
        try self.hana.cachePacket(output, 3600);
        
        return output;
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "HanaAdapter creation" {
    const allocator = std.testing.allocator;
    
    var adapter = try HanaAdapter.init(
        allocator,
        "localhost",
        39017,
        "SYSTEM",
        "Password123",
        "NWORKFLOW",
    );
    defer adapter.deinit();
    
    try std.testing.expect(adapter.cache.is_connected);
}

test "HanaPipeline creation" {
    const allocator = std.testing.allocator;
    
    var pipeline = try DataPipeline.init(allocator, "test_pipeline");
    defer pipeline.deinit();
    
    var hana_pipeline = try HanaPipeline.init(
        allocator,
        "localhost",
        39017,
        "SYSTEM",
        "Password123",
        "NWORKFLOW",
        pipeline,
    );
    defer hana_pipeline.deinit();
    
    try std.testing.expect(hana_pipeline.hana.cache.is_connected);
}