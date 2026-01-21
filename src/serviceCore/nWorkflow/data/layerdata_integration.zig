const std = @import("std");
const Allocator = std.mem.Allocator;
const data_packet = @import("data_packet.zig");
const data_pipeline = @import("data_pipeline.zig");
const DataPacket = data_packet.DataPacket;
const DataPipeline = data_pipeline.DataPipeline;

/// LayerData integration examples and utilities
/// This module demonstrates how nWorkflow data system integrates with:
/// - PostgreSQL (persistent storage)
/// - DragonflyDB (caching, sessions)
/// - Qdrant (vector storage)
/// - Memgraph (graph relationships)
/// - Marquez (lineage tracking)

/// PostgreSQL integration for persistent data storage
pub const PostgresAdapter = struct {
    allocator: Allocator,
    connection_string: []const u8,
    
    pub fn init(allocator: Allocator, connection_string: []const u8) !PostgresAdapter {
        return .{
            .allocator = allocator,
            .connection_string = try allocator.dupe(u8, connection_string),
        };
    }
    
    pub fn deinit(self: *PostgresAdapter) void {
        self.allocator.free(self.connection_string);
    }
    
    /// Store a DataPacket in PostgreSQL
    pub fn storePacket(self: *PostgresAdapter, packet: *const DataPacket) !void {
        // Serialize packet to JSON
        const json_data = try packet.serialize(packet.allocator);
        defer packet.allocator.free(json_data);
        
        // In production, this would execute:
        // INSERT INTO workflow_data (id, type, value, metadata, timestamp)
        // VALUES ($1, $2, $3, $4, $5)
        // For now, this is a demonstration
        
        // Simulate storage
        _ = self;
    }
    
    /// Retrieve a DataPacket from PostgreSQL
    pub fn loadPacket(self: *PostgresAdapter, packet_id: []const u8) !?*DataPacket {
        _ = self;
        _ = packet_id;
        // In production, this would execute:
        // SELECT * FROM workflow_data WHERE id = $1
        // Then deserialize the JSON back to DataPacket
        
        return null; // Placeholder
    }
    
    /// Store workflow execution state
    pub fn storeWorkflowState(self: *PostgresAdapter, workflow_id: []const u8, state: []const u8) !void {
        _ = self;
        // In production:
        // INSERT INTO workflow_states (workflow_id, state, updated_at)
        // VALUES ($1, $2, NOW())
        // ON CONFLICT (workflow_id) DO UPDATE SET state = $2, updated_at = NOW()
        
        _ = workflow_id;
        _ = state;
    }
    
    /// Query with Row-Level Security (RLS)
    pub fn queryWithRLS(self: *PostgresAdapter, user_id: []const u8, query: []const u8) !void {
        _ = self;
        // In production:
        // SET app.current_user_id = $1;
        // Then execute the query - RLS policies automatically applied
        
        _ = user_id;
        _ = query;
    }
};

/// DragonflyDB integration for caching and sessions
pub const DragonflyAdapter = struct {
    allocator: Allocator,
    connection_string: []const u8,
    
    pub fn init(allocator: Allocator, connection_string: []const u8) !DragonflyAdapter {
        return .{
            .allocator = allocator,
            .connection_string = try allocator.dupe(u8, connection_string),
        };
    }
    
    pub fn deinit(self: *DragonflyAdapter) void {
        self.allocator.free(self.connection_string);
    }
    
    /// Cache a DataPacket with TTL
    pub fn cachePacket(self: *DragonflyAdapter, packet: *const DataPacket, ttl_seconds: u32) !void {
        _ = self;
        const json_data = try packet.serialize(packet.allocator);
        defer packet.allocator.free(json_data);
        
        // In production: SETEX packet:{id} {ttl} {json_data}
        _ = ttl_seconds;
    }
    
    /// Get cached packet
    pub fn getCachedPacket(self: *DragonflyAdapter, packet_id: []const u8) !?*DataPacket {
        _ = self;
        _ = packet_id;
        // In production: GET packet:{id}
        // Then deserialize if found
        
        return null; // Placeholder
    }
    
    /// Publish packet to pub/sub channel
    pub fn publishPacket(self: *DragonflyAdapter, channel: []const u8, packet: *const DataPacket) !void {
        _ = self;
        const json_data = try packet.serialize(packet.allocator);
        defer packet.allocator.free(json_data);
        
        // In production: PUBLISH {channel} {json_data}
        _ = channel;
    }
    
    /// Store session data
    pub fn storeSession(self: *DragonflyAdapter, session_id: []const u8, data: []const u8, ttl_seconds: u32) !void {
        _ = self;
        // In production: SETEX session:{session_id} {ttl} {data}
        
        _ = session_id;
        _ = data;
        _ = ttl_seconds;
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
        // {
        //   "points": [{
        //     "id": packet_id,
        //     "vector": vector,
        //     "payload": metadata
        //   }]
        // }
        
        _ = packet_id;
        _ = vector;
    }
    
    /// Search for similar vectors
    pub fn searchSimilar(self: *QdrantAdapter, query_vector: []const f32, limit: usize) ![][]const u8 {
        // In production: POST /collections/{collection}/points/search
        // Returns array of packet IDs
        
        var results = std.ArrayList([]const u8).init(self.allocator);
        
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
        // In production: CREATE (n:DataPacket {id: $1, type: $2, timestamp: $3})
        
        _ = self;
        _ = packet;
    }
    
    /// Create relationship between packets (data flow)
    pub fn createFlow(self: *MemgraphAdapter, from_id: []const u8, to_id: []const u8, relationship: []const u8) !void {
        _ = self;
        // In production:
        // MATCH (a:DataPacket {id: $1}), (b:DataPacket {id: $2})
        // CREATE (a)-[r:FLOWS_TO {type: $3}]->(b)
        
        _ = from_id;
        _ = to_id;
        _ = relationship;
    }
    
    /// Query data lineage
    pub fn getLineage(self: *MemgraphAdapter, packet_id: []const u8) ![][]const u8 {
        // In production:
        // MATCH path = (start:DataPacket {id: $1})-[*]->()
        // RETURN nodes(path)
        
        var results = std.ArrayList([]const u8).init(self.allocator);
        
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
        // In production: PUT /api/v1/namespaces/{namespace}/datasets/{name}
        // { "type": "DB_TABLE", "fields": [...] }
        
        _ = dataset_name;
        _ = fields;
    }
    
    /// Start job run for pipeline execution
    pub fn startJobRun(self: *MarquezAdapter, job_name: []const u8, run_id: []const u8) !void {
        _ = self;
        // In production: POST /api/v1/namespaces/{namespace}/jobs/{job}/runs
        // { "id": run_id, "nominalTime": ... }
        
        _ = job_name;
        _ = run_id;
    }
    
    /// Complete job run with lineage
    pub fn completeJobRun(self: *MarquezAdapter, run_id: []const u8, inputs: []const []const u8, outputs: []const []const u8) !void {
        _ = self;
        // In production: POST /api/v1/jobs/runs/{run_id}/complete
        // { "inputs": [...], "outputs": [...] }
        
        _ = run_id;
        _ = inputs;
        _ = outputs;
    }
};

/// Complete integration example with all layerData services
pub const LayerDataPipeline = struct {
    allocator: Allocator,
    postgres: PostgresAdapter,
    dragonfly: DragonflyAdapter,
    qdrant: QdrantAdapter,
    memgraph: MemgraphAdapter,
    marquez: MarquezAdapter,
    pipeline: *DataPipeline,
    
    pub fn init(
        allocator: Allocator,
        postgres_conn: []const u8,
        dragonfly_conn: []const u8,
        qdrant_url: []const u8,
        memgraph_conn: []const u8,
        marquez_url: []const u8,
        pipeline: *DataPipeline,
    ) !*LayerDataPipeline {
        const ldp = try allocator.create(LayerDataPipeline);
        errdefer allocator.destroy(ldp);
        
        ldp.* = .{
            .allocator = allocator,
            .postgres = try PostgresAdapter.init(allocator, postgres_conn),
            .dragonfly = try DragonflyAdapter.init(allocator, dragonfly_conn),
            .qdrant = try QdrantAdapter.init(allocator, qdrant_url, "workflow_vectors"),
            .memgraph = try MemgraphAdapter.init(allocator, memgraph_conn),
            .marquez = try MarquezAdapter.init(allocator, marquez_url, "nworkflow"),
            .pipeline = pipeline,
        };
        
        return ldp;
    }
    
    pub fn deinit(self: *LayerDataPipeline) void {
        self.postgres.deinit();
        self.dragonfly.deinit();
        self.qdrant.deinit();
        self.memgraph.deinit();
        self.marquez.deinit();
        self.allocator.destroy(self);
    }
    
    /// Execute pipeline with full layerData integration
    pub fn executeWithTracking(self: *LayerDataPipeline, input: *DataPacket, run_id: []const u8) !*DataPacket {
        // 1. Cache input in DragonflyDB
        try self.dragonfly.cachePacket(input, 3600);
        
        // 2. Start Marquez job run
        try self.marquez.startJobRun(self.pipeline.id, run_id);
        
        // 3. Create graph node in Memgraph
        try self.memgraph.createPacketNode(input);
        
        // 4. Execute pipeline
        const output = try self.pipeline.execute(input);
        
        // 5. Store result in PostgreSQL
        try self.postgres.storePacket(output);
        
        // 6. Create graph relationship
        try self.memgraph.createFlow(input.id, output.id, "TRANSFORMED");
        
        // 7. Complete Marquez job run with lineage
        const inputs = [_][]const u8{input.id};
        const outputs = [_][]const u8{output.id};
        try self.marquez.completeJobRun(run_id, &inputs, &outputs);
        
        // 8. Cache output
        try self.dragonfly.cachePacket(output, 3600);
        
        return output;
    }
};

// ============================================================================
// USAGE EXAMPLES
// ============================================================================

/// Example: ETL Pipeline with layerData
pub fn exampleETLPipeline(allocator: Allocator) !void {
    // Create transformation pipeline
    var builder = try data_pipeline.PipelineBuilder.init(allocator, "etl-pipeline", "ETL Example");
    
    // Add stages (simplified for example)
    // In production, these would be actual transformation functions
    const extract_stage = try data_pipeline.PipelineStage.init(
        allocator,
        "extract",
        "Extract Data",
        struct {
            fn transform(alloc: Allocator, packet: *DataPacket) !*DataPacket {
                _ = alloc;
                return packet;
            }
        }.transform,
    );
    try builder.pipeline.addStage(extract_stage);
    
    const pipeline = builder.build();
    defer pipeline.deinit();
    
    // Create layerData integration
    const ldp = try LayerDataPipeline.init(
        allocator,
        "postgres://localhost:5432/nworkflow",
        "redis://localhost:6379",
        "http://localhost:6333",
        "bolt://localhost:7687",
        "http://localhost:5000",
        pipeline,
    );
    defer ldp.deinit();
    
    // Execute with full tracking
    const input_value = std.json.Value{ .string = "test data" };
    const input = try DataPacket.init(allocator, "input-1", .string, input_value);
    defer input.deinit();
    
    const output = try ldp.executeWithTracking(input, "run-12345");
    defer output.deinit();
}

// ============================================================================
// TESTS
// ============================================================================

test "PostgresAdapter creation" {
    const allocator = std.testing.allocator;
    
    var adapter = try PostgresAdapter.init(allocator, "postgres://localhost:5432/test");
    defer adapter.deinit();
    
    try std.testing.expectEqualStrings("postgres://localhost:5432/test", adapter.connection_string);
}

test "DragonflyAdapter creation" {
    const allocator = std.testing.allocator;
    
    var adapter = try DragonflyAdapter.init(allocator, "redis://localhost:6379");
    defer adapter.deinit();
    
    try std.testing.expectEqualStrings("redis://localhost:6379", adapter.connection_string);
}

test "QdrantAdapter creation" {
    const allocator = std.testing.allocator;
    
    var adapter = try QdrantAdapter.init(allocator, "http://localhost:6333", "test_collection");
    defer adapter.deinit();
    
    try std.testing.expectEqualStrings("http://localhost:6333", adapter.base_url);
    try std.testing.expectEqualStrings("test_collection", adapter.collection_name);
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
    
    try std.testing.expectEqualStrings("http://localhost:5000", adapter.base_url);
    try std.testing.expectEqualStrings("nworkflow", adapter.namespace);
}

test "LayerDataPipeline integration" {
    const allocator = std.testing.allocator;
    
    const pipeline = try DataPipeline.init(allocator, "test-pipeline", "Integration Test");
    defer pipeline.deinit();
    
    const ldp = try LayerDataPipeline.init(
        allocator,
        "postgres://localhost:5432/test",
        "redis://localhost:6379",
        "http://localhost:6333",
        "bolt://localhost:7687",
        "http://localhost:5000",
        pipeline,
    );
    defer ldp.deinit();
    
    try std.testing.expectEqualStrings("test-pipeline", ldp.pipeline.id);
}
