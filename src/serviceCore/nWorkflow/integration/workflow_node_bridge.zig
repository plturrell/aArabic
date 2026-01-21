//! Workflow-Node Integration Bridge - Day 14
//! 
//! This module bridges the workflow parser with the node type system,
//! enabling compilation of workflow definitions into executable nodes.
//!
//! Key Features:
//! - Convert WorkflowNode to NodeConfig
//! - Validate node connections
//! - Build execution graph from workflow
//! - Port compatibility checking

const std = @import("std");
const Allocator = std.mem.Allocator;
const node_factory = @import("node_factory");
const node_types = @import("node_types");

const NodeFactory = node_factory.NodeFactory;
const NodeConfig = node_factory.NodeConfig;
const NodeInterface = node_types.NodeInterface;
const ExecutionContext = node_types.ExecutionContext;
const Port = node_types.Port;
const PortType = node_types.PortType;

/// Workflow edge connection
pub const EdgeConnection = struct {
    from_node: []const u8,
    from_port: []const u8,
    to_node: []const u8,
    to_port: []const u8,
    condition: ?[]const u8,
    
    pub fn init(
        from_node: []const u8,
        from_port: []const u8,
        to_node: []const u8,
        to_port: []const u8,
        condition: ?[]const u8,
    ) EdgeConnection {
        return EdgeConnection{
            .from_node = from_node,
            .from_port = from_port,
            .to_node = to_node,
            .to_port = to_port,
            .condition = condition,
        };
    }
};

/// Execution graph built from workflow
pub const ExecutionGraph = struct {
    allocator: Allocator,
    nodes: std.StringHashMap(*NodeInterface),
    edges: std.ArrayList(EdgeConnection),
    entry_points: std.ArrayList([]const u8), // Trigger node IDs
    
    pub fn init(allocator: Allocator) ExecutionGraph {
        return ExecutionGraph{
            .allocator = allocator,
            .nodes = std.StringHashMap(*NodeInterface).init(allocator),
            .edges = std.ArrayList(EdgeConnection){},
            .entry_points = std.ArrayList([]const u8){},
        };
    }
    
    pub fn deinit(self: *ExecutionGraph) void {
        self.nodes.deinit();
        self.edges.deinit(self.allocator);
        self.entry_points.deinit(self.allocator);
    }
    
    /// Add a node to the graph
    pub fn addNode(self: *ExecutionGraph, node: *NodeInterface) !void {
        try self.nodes.put(node.id, node);
        
        // Track trigger nodes as entry points
        if (std.mem.eql(u8, node.node_type, "trigger")) {
            try self.entry_points.append(self.allocator, node.id);
        }
    }
    
    /// Add an edge connection
    pub fn addEdge(self: *ExecutionGraph, edge: EdgeConnection) !void {
        try self.edges.append(self.allocator, edge);
    }
    
    /// Get a node by ID
    pub fn getNode(self: *const ExecutionGraph, node_id: []const u8) ?*NodeInterface {
        return self.nodes.get(node_id);
    }
    
    /// Get all edges from a node
    pub fn getOutgoingEdges(self: *const ExecutionGraph, node_id: []const u8) !std.ArrayList(EdgeConnection) {
        var result = std.ArrayList(EdgeConnection){};
        
        for (self.edges.items) |edge| {
            if (std.mem.eql(u8, edge.from_node, node_id)) {
                try result.append(self.allocator, edge);
            }
        }
        
        return result;
    }
    
    /// Validate the execution graph
    pub fn validate(self: *const ExecutionGraph) !void {
        // Check that all edge references exist
        for (self.edges.items) |edge| {
            if (!self.nodes.contains(edge.from_node)) {
                return error.EdgeReferencesNonexistentNode;
            }
            if (!self.nodes.contains(edge.to_node)) {
                return error.EdgeReferencesNonexistentNode;
            }
            
            // Validate port compatibility
            const from_node = self.nodes.get(edge.from_node).?;
            const to_node = self.nodes.get(edge.to_node).?;
            
            // Find ports
            var from_port: ?Port = null;
            for (from_node.outputs) |port| {
                if (std.mem.eql(u8, port.id, edge.from_port)) {
                    from_port = port;
                    break;
                }
            }
            
            var to_port: ?Port = null;
            for (to_node.inputs) |port| {
                if (std.mem.eql(u8, port.id, edge.to_port)) {
                    to_port = port;
                    break;
                }
            }
            
            if (from_port == null) return error.OutputPortNotFound;
            if (to_port == null) return error.InputPortNotFound;
            
            // Check type compatibility
            if (!arePortsCompatible(from_port.?, to_port.?)) {
                return error.IncompatiblePortTypes;
            }
        }
        
        // Must have at least one entry point
        if (self.entry_points.items.len == 0) {
            return error.NoEntryPoints;
        }
    }
};

/// Check if two ports are type-compatible
fn arePortsCompatible(output: Port, input: Port) bool {
    // Any port accepts anything
    if (input.port_type == .any or output.port_type == .any) {
        return true;
    }
    
    // Exact type match
    if (output.port_type == input.port_type) {
        return true;
    }
    
    // Special cases could be added here
    // For example: number can accept integer/float
    
    return false;
}

/// Workflow-to-Node bridge
pub const WorkflowNodeBridge = struct {
    allocator: Allocator,
    factory: NodeFactory,
    
    pub fn init(allocator: Allocator) WorkflowNodeBridge {
        return WorkflowNodeBridge{
            .allocator = allocator,
            .factory = NodeFactory.init(allocator),
        };
    }
    
    pub fn deinit(self: *WorkflowNodeBridge) void {
        self.factory.deinit();
    }
    
    /// Build execution graph from workflow node configurations
    pub fn buildExecutionGraph(
        self: *WorkflowNodeBridge,
        node_configs: []const NodeConfig,
        edges: []const EdgeConnection,
    ) !ExecutionGraph {
        var graph = ExecutionGraph.init(self.allocator);
        errdefer graph.deinit();
        
        // Create all nodes
        for (node_configs) |config| {
            const node = try self.factory.createNode(config);
            errdefer self.factory.destroyNode(node);
            try graph.addNode(node);
        }
        
        // Add all edges
        for (edges) |edge| {
            try graph.addEdge(edge);
        }
        
        // Validate the graph
        try graph.validate();
        
        return graph;
    }
    
    /// Execute a workflow graph
    pub fn executeGraph(
        self: *WorkflowNodeBridge,
        graph: *ExecutionGraph,
        ctx: *ExecutionContext,
    ) !std.json.Value {
        _ = self;
        
        // Simple sequential execution from entry points
        // In production, this would use the Petri Net executor
        
        if (graph.entry_points.items.len == 0) {
            return error.NoEntryPoints;
        }
        
        // Execute first entry point
        const entry_node_id = graph.entry_points.items[0];
        const entry_node = graph.getNode(entry_node_id) orelse return error.EntryNodeNotFound;
        
        // For trigger nodes, cast and execute
        if (std.mem.eql(u8, entry_node.node_type, "trigger")) {
            const trigger: *node_types.TriggerNode = @alignCast(@ptrCast(entry_node));
            return try trigger.execute(ctx);
        }
        
        return error.InvalidEntryNode;
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "EdgeConnection creation" {
    const edge = EdgeConnection.init("node1", "out1", "node2", "in1", null);
    
    try std.testing.expectEqualStrings("node1", edge.from_node);
    try std.testing.expectEqualStrings("out1", edge.from_port);
    try std.testing.expectEqualStrings("node2", edge.to_node);
    try std.testing.expectEqualStrings("in1", edge.to_port);
    try std.testing.expect(edge.condition == null);
}

test "ExecutionGraph add and retrieve nodes" {
    const allocator = std.testing.allocator;
    
    var graph = ExecutionGraph.init(allocator);
    defer graph.deinit();
    
    // Create a simple trigger node
    const outputs = [_]Port{
        Port.init("output", "Output", "Trigger output", .object, false, null),
    };
    
    var trigger = try allocator.create(node_types.TriggerNode);
    defer allocator.destroy(trigger);
    
    trigger.* = try node_types.TriggerNode.init(
        allocator,
        "trigger1",
        "Test Trigger",
        "Test",
        "manual",
        &outputs,
        std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    );
    
    try graph.addNode(&trigger.base);
    
    const retrieved = graph.getNode("trigger1");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings("trigger1", retrieved.?.id);
    try std.testing.expectEqual(@as(usize, 1), graph.entry_points.items.len);
}

test "Port compatibility checking" {
    const port1 = Port.init("p1", "Port 1", "", .string, false, null);
    const port2 = Port.init("p2", "Port 2", "", .string, false, null);
    
    try std.testing.expect(arePortsCompatible(port1, port2));
    
    const port3 = Port.init("p3", "Port 3", "", .number, false, null);
    try std.testing.expect(!arePortsCompatible(port1, port3));
    
    // Any port is compatible with everything
    const any_port = Port.init("p4", "Port 4", "", .any, false, null);
    try std.testing.expect(arePortsCompatible(port1, any_port));
    try std.testing.expect(arePortsCompatible(any_port, port1));
}

test "WorkflowNodeBridge build simple graph" {
    const allocator = std.testing.allocator;
    
    var bridge = WorkflowNodeBridge.init(allocator);
    defer bridge.deinit();
    
    // Create node configs
    const configs = [_]NodeConfig{
        NodeConfig{
            .id = "trigger1",
            .name = "Start",
            .description = "",
            .node_type = "trigger",
            .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
        },
        NodeConfig{
            .id = "action1",
            .name = "Process",
            .description = "",
            .node_type = "action",
            .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
        },
    };
    
    const edges = [_]EdgeConnection{
        EdgeConnection.init("trigger1", "output", "action1", "input", null),
    };
    
    var graph = try bridge.buildExecutionGraph(&configs, &edges);
    defer {
        // Clean up nodes
        var iter = graph.nodes.valueIterator();
        while (iter.next()) |node| {
            bridge.factory.destroyNode(node.*);
        }
        graph.deinit();
    }
    
    try std.testing.expectEqual(@as(usize, 2), graph.nodes.count());
    try std.testing.expectEqual(@as(usize, 1), graph.edges.items.len);
    try std.testing.expectEqual(@as(usize, 1), graph.entry_points.items.len);
}

test "WorkflowNodeBridge execute simple workflow" {
    const allocator = std.testing.allocator;
    
    var bridge = WorkflowNodeBridge.init(allocator);
    defer bridge.deinit();
    
    // Create simple trigger workflow
    const configs = [_]NodeConfig{
        NodeConfig{
            .id = "trigger1",
            .name = "Start",
            .description = "",
            .node_type = "trigger",
            .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
        },
    };
    
    const edges = [_]EdgeConnection{};
    
    var graph = try bridge.buildExecutionGraph(&configs, &edges);
    defer {
        var iter = graph.nodes.valueIterator();
        while (iter.next()) |node| {
            bridge.factory.destroyNode(node.*);
        }
        graph.deinit();
    }
    
    var ctx = ExecutionContext.init(allocator, "wf1", "exec1", "user1");
    defer ctx.deinit();
    
    var result = try bridge.executeGraph(&graph, &ctx);
    defer result.object.deinit();
    
    try std.testing.expect(result == .object);
    try std.testing.expect(result.object.get("triggered").?.bool);
}

test "ExecutionGraph validation with invalid edges" {
    const allocator = std.testing.allocator;
    
    var graph = ExecutionGraph.init(allocator);
    defer graph.deinit();
    
    // Add node
    const outputs = [_]Port{
        Port.init("output", "Output", "", .object, false, null),
    };
    
    var trigger = try allocator.create(node_types.TriggerNode);
    defer allocator.destroy(trigger);
    
    trigger.* = try node_types.TriggerNode.init(
        allocator,
        "trigger1",
        "Test",
        "",
        "manual",
        &outputs,
        std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    );
    
    try graph.addNode(&trigger.base);
    
    // Add edge to non-existent node
    const invalid_edge = EdgeConnection.init("trigger1", "output", "missing", "input", null);
    try graph.addEdge(invalid_edge);
    
    // Validation should fail
    try std.testing.expectError(error.EdgeReferencesNonexistentNode, graph.validate());
}
