const std = @import("std");
const Allocator = std.mem.Allocator;
const data_packet = @import("data_packet.zig");
const data_pipeline = @import("data_pipeline.zig");
const DataPacket = data_packet.DataPacket;
const DataPipeline = data_pipeline.DataPipeline;

/// LayerData integration examples and utilities
/// This module demonstrates how nWorkflow data system integrates with:
/// - SAP HANA (unified persistent storage + caching)
/// - Qdrant (vector storage)
/// - Memgraph (graph relationships)
/// - Marquez (lineage tracking)

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
    
    /// Publish packet to event stream (future: HANA pub/sub)
    pub fn publishPacket(self: *HanaAdapter, channel: []const u8, packet: *const DataPacket) !void {
        _ = self;
        const json_data = try packet.serialize(packet.allocator);
        defer packet.allocator.free(json_data);
        
        // Future: Use HANA streaming or event tables
        _ = channel;
    }
};

/// Qdrant integration for vector storage
pub const QdrantAdapter = struct {
    allocator: Allocator,
    base_url: []const u8,
    collection_name: []const u8,
    
    pub fn init(allocator: Allocator, base_url: []const u8, collection_name: []const u8) !QdrantAdapter {
        return .{
            .allocator = allocator,
            .base_url = try allocator.dupe(u8, base_url),
            .collection_name = try allocator.dupe(u8, collection_name),
        };
    }
    
    pub fn deinit(self: *QdrantAdapter) void {
        self.allocator.free(self.base_url);
        self.allocator.free(self.collection_name);
    }
    
    /// Store vector embedding from DataPacket
    pub fn storeEmbedding(self: *QdrantAdapter, packet_id: []const u8, vector: []const f32) !void {
        _ = self;
        // In production: POST /collections/{collection}/points
        _ = packet_id;
        _ = vector;
    }
    
    /// Search for similar vectors
    pub fn searchSimilar(self: *QdrantAdapter, query_vector: []const f32, limit: usize) ![][]const u8 {
        var results = std.ArrayList([]const u8){};
        _ = query_vector;
        _ = limit;
        return results.toOwnedSlice();
    }
};

/// Memgraph integration for graph relationships
pub const MemgraphAdapter = struct {
    allocator: Allocator,
    connection_string: []const u8,
    
    pub fn init(allocator: Allocator, connection_string: []const u8) !MemgraphAdapter {
        return .{
            .allocator = allocator,
            .connection_string = try allocator.dupe(u8, connection_string),
        };
    }
    
    pub fn deinit(self: *MemgraphAdapter) void {
        self.allocator.free(self.connection_string);
    }
    
    /// Create node for DataPacket
    pub fn createPacketNode(self: *MemgraphAdapter, packet: *const DataPacket) !void {
        _ = self;
        _ = packet;
    }
    
    /// Create relationship between packets (data flow)
    pub fn createFlow(self: *MemgraphAdapter, from_id: []const u8, to_id: []const u8, relationship: []const u8) !void {
        _ = self;
        _ = from_id;
        _ = to_id;
        _ = relationship;
    }
    
    /// Query data lineage
    pub fn getLineage(self: *MemgraphAdapter, packet_id: []const u8) ![][]const u8 {
        var results = std.ArrayList([]const u8){};
        _ = packet_id;
        return results.toOwnedSlice();
    }
};

/// Marquez integration for data lineage tracking
pub const MarquezAdapter = struct {
    allocator: Allocator,
    base_url: []const u8,
    namespace: []const u8,
    
    pub fn init(allocator: Allocator, base_url: []const u8, namespace: []const u8) !MarquezAdapter {
        return .{
            .allocator = allocator,
            .base_url = try allocator.dupe(u8, base_url),
            .namespace = try allocator.dupe(u8, namespace),
        };
    }
    
    pub fn deinit(self: *MarquezAdapter) void {
        self.allocator.free(self.base_url);
        self.allocator.free(self.namespace);
    }
    
    /// Register dataset for lineage tracking
    pub fn registerDataset(self: *MarquezAdapter, dataset_name: []const u8, fields: []const []const u8) !void {
        _ = self;
        _ = dataset_name;
        _ = fields;
    }
    
    /// Start job run for pipeline execution
    pub fn startJobRun(self: *MarquezAdapter, job_name: []const u8, run_id: []const u8) !void {
        _ = self;
        _ = job_name;
        _ = run_id;
    }
    
    /// Complete job run with lineage
    pub fn completeJobRun(self: *MarquezAdapter, run_id: []const u8, inputs: []const []const u8, outputs: []const []const u8) !void {
        _ = self;
        _ = run_id;
        _ = inputs;
        _ = outputs;
    }
};

/// Complete integration example with HANA + layerData services
pub const LayerDataPipeline = struct {
    allocator: Allocator,
    hana: HanaAdapter,
    qdrant: QdrantAdapter,
    memgraph: MemgraphAdapter,
    marquez: MarquezAdapter,
    pipeline: *DataPipeline,
    
    pub fn init(
        allocator: Allocator,
        hana_host: []const u8,
        hana_port: u16,
        hana_user: []const u8,
        hana_password: []const u8,
        hana_schema: []const u8,
        qdrant_url: []const u8,
        memgraph_conn: []const u8,
        marquez_url: []const u8,
        pipeline: *DataPipeline,
    ) !*LayerDataPipeline {
        const ldp = try allocator.create(LayerDataPipeline);
        errdefer allocator.destroy(ldp);
        
        ldp.* = .{
            .allocator = allocator,
            .hana = try HanaAdapter.init(allocator, hana_host, hana_port, hana_user, hana_password, hana_schema),
            .qdrant = try QdrantAdapter.init(allocator, qdrant_url, "workflow_vectors"),
            .memgraph = try MemgraphAdapter.init(allocator, memgraph_conn),
            .marquez = try MarquezAdapter.init(allocator, marquez_url, "nworkflow"),
            .pipeline = pipeline,
        };
        
        return ldp;
    }
    
    pub fn deinit(self: *LayerDataPipeline) void {
        self.hana.deinit();
        self.qdrant.deinit();
        self.memgraph.deinit();
        self.marquez.deinit();
        self.allocator.destroy(self);
    }
    
    /// Execute pipeline with full layerData integration
    pub fn executeWithTracking(self: *LayerDataPipeline, input: *DataPacket, run_id: []const u8) !*DataPacket {
        // 1. Cache input in HANA
        try self.hana.cachePacket(input, 3600);
        
        // 2. Start Marquez job run
        try self.marquez.startJobRun(self.pipeline.id, run_id);
        
        // 3. Create graph node in Memgraph
        try self.memgraph.createPacketNode(input);
        
        // 4. Execute pipeline
        const output = try self.pipeline.execute(input);
        
        // 5. Store result in HANA (persistent)
        try self.hana.storePacket(output);
        
        // 6. Create graph relationship
        try self.memgraph.createFlow(input.id, output.id, "TRANSFORMED");
        
        // 7. Complete Marquez job run with lineage
        const inputs = [_][]const u8{input.id};
        const outputs = [_][]const u8{output.id};
        try self.marquez.completeJobRun(run_id, &inputs, &outputs);
        
        // 8. Cache output in HANA
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

test "QdrantAdapter creation" {
    const allocator = std.testing.allocator;
    
    var adapter = try QdrantAdapter.init(allocator, "http://localhost:6333", "test_collection");
    defer adapter.deinit();
    
    try std.testing.expectEqualStrings("http://localhost:6333", adapter.base_url);
}

test "MemgraphAdapter creation" {
    const allocator = std.testing.allocator;
    
    var adapter = try MemgraphAdapter.init(allocator, "bolt://localhost:7687");
    defer adapter.deinit();
    
    try std.testing.expectEqualStrings("bolt://localhost:7687", adapter.connection_string);
}

test "MarquezAdapter creation" {
    const allocator = std.testing.allocator;
    
    var adapter = try MarquezAdapter.init(allocator, "http://localhost:5000", "nworkflow");
    defer adapter.deinit();
    
    try std.testing.expectEqualStrings("nworkflow", adapter.namespace);
}