// ============================================================================
// Agent Topology - Day 71 Implementation
// ============================================================================
// Purpose: Agent registration and topology management for orchestration
// Week: Week 15 (Days 71-75) - Orchestration Foundation
// Phase: Month 5 - Orchestration & Training
// ============================================================================

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// AGENT DEFINITION
// ============================================================================

/// Agent type classification
pub const AgentType = enum {
    llm,           // Language model agent
    code,          // Code generation agent
    search,        // Search/retrieval agent
    analysis,      // Data analysis agent
    orchestrator,  // Workflow orchestrator
    tool,          // Tool executor
    custom,        // Custom agent type
};

/// Agent status
pub const AgentStatus = enum {
    active,
    inactive,
    failed,
    maintenance,
};

/// Agent capability definition
pub const AgentCapability = struct {
    name: []const u8,
    description: []const u8,
    version: []const u8,
    
    pub fn init(allocator: Allocator, name: []const u8, description: []const u8, version: []const u8) !AgentCapability {
        return AgentCapability{
            .name = try allocator.dupe(u8, name),
            .description = try allocator.dupe(u8, description),
            .version = try allocator.dupe(u8, version),
        };
    }
    
    pub fn deinit(self: *AgentCapability, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
        allocator.free(self.version);
    }
};

/// Agent in the orchestration system
pub const Agent = struct {
    id: []const u8,
    name: []const u8,
    type: AgentType,
    status: AgentStatus,
    endpoint: []const u8,
    capabilities: std.ArrayList(AgentCapability),
    metadata: std.StringHashMap([]const u8),
    created_at: i64,
    updated_at: i64,
    
    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        agent_type: AgentType,
        endpoint: []const u8,
    ) !Agent {
        return .{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .type = agent_type,
            .status = .active,
            .endpoint = try allocator.dupe(u8, endpoint),
            .capabilities = std.ArrayList(AgentCapability){},
            .metadata = std.StringHashMap([]const u8).init(allocator),
            .created_at = std.time.milliTimestamp(),
            .updated_at = std.time.milliTimestamp(),
        };
    }
    
    pub fn deinit(self: *Agent, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.endpoint);
        
        for (self.capabilities.items) |*cap| {
            cap.deinit(allocator);
        }
        self.capabilities.deinit();
        
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }
    
    pub fn addCapability(self: *Agent, capability: AgentCapability) !void {
        try self.capabilities.append(capability);
        self.updated_at = std.time.milliTimestamp();
    }
    
    pub fn setMetadata(self: *Agent, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        const value_copy = try allocator.dupe(u8, value);
        try self.metadata.put(key_copy, value_copy);
        self.updated_at = std.time.milliTimestamp();
    }
    
    pub fn updateStatus(self: *Agent, status: AgentStatus) void {
        self.status = status;
        self.updated_at = std.time.milliTimestamp();
    }
};

// ============================================================================
// AGENT CONNECTION
// ============================================================================

/// Connection type between agents
pub const ConnectionType = enum {
    sync,        // Synchronous call
    async_msg,   // Asynchronous message
    stream,      // Streaming data
    callback,    // Callback pattern
};

/// Connection between two agents
pub const AgentConnection = struct {
    from_agent_id: []const u8,
    to_agent_id: []const u8,
    connection_type: ConnectionType,
    weight: f32,  // Connection strength/priority
    metadata: std.StringHashMap([]const u8),
    
    pub fn init(
        allocator: Allocator,
        from: []const u8,
        to: []const u8,
        conn_type: ConnectionType,
        weight: f32,
    ) !AgentConnection {
        return .{
            .from_agent_id = try allocator.dupe(u8, from),
            .to_agent_id = try allocator.dupe(u8, to),
            .connection_type = conn_type,
            .weight = weight,
            .metadata = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *AgentConnection, allocator: Allocator) void {
        allocator.free(self.from_agent_id);
        allocator.free(self.to_agent_id);
        
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }
};

// ============================================================================
// TOPOLOGY MANAGER
// ============================================================================

/// Agent topology graph representation
pub const AgentTopology = struct {
    allocator: Allocator,
    agents: std.StringHashMap(Agent),
    connections: std.ArrayList(AgentConnection),
    mutex: std.Thread.Mutex,
    
    pub fn init(allocator: Allocator) !*AgentTopology {
        const topology = try allocator.create(AgentTopology);
        topology.* = .{
            .allocator = allocator,
            .agents = std.StringHashMap(Agent).init(allocator),
            .connections = std.ArrayList(AgentConnection){},
            .mutex = .{},
        };
        return topology;
    }
    
    pub fn deinit(self: *AgentTopology) void {
        var it = self.agents.iterator();
        while (it.next()) |entry| {
            var agent = entry.value_ptr.*;
            agent.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.agents.deinit();
        
        for (self.connections.items) |*conn| {
            conn.deinit(self.allocator);
        }
        self.connections.deinit();
        
        self.allocator.destroy(self);
    }
    
    // ========================================================================
    // AGENT MANAGEMENT
    // ========================================================================
    
    /// Register a new agent
    pub fn registerAgent(self: *AgentTopology, agent: Agent) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Check if agent already exists
        if (self.agents.contains(agent.id)) {
            return error.AgentAlreadyExists;
        }
        
        const id_copy = try self.allocator.dupe(u8, agent.id);
        try self.agents.put(id_copy, agent);
        
        std.log.info("Registered agent: {s} (type: {s})", .{ agent.name, @tagName(agent.type) });
    }
    
    /// Unregister an agent
    pub fn unregisterAgent(self: *AgentTopology, agent_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.agents.fetchRemove(agent_id)) |kv| {
            var agent = kv.value;
            agent.deinit(self.allocator);
            self.allocator.free(kv.key);
            
            // Remove all connections involving this agent
            var i: usize = 0;
            while (i < self.connections.items.len) {
                const conn = &self.connections.items[i];
                if (std.mem.eql(u8, conn.from_agent_id, agent_id) or
                    std.mem.eql(u8, conn.to_agent_id, agent_id))
                {
                    var removed = self.connections.orderedRemove(i);
                    removed.deinit(self.allocator);
                } else {
                    i += 1;
                }
            }
            
            std.log.info("Unregistered agent: {s}", .{agent_id});
        } else {
            return error.AgentNotFound;
        }
    }
    
    /// Get an agent by ID
    pub fn getAgent(self: *AgentTopology, agent_id: []const u8) ?*Agent {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        return self.agents.getPtr(agent_id);
    }
    
    /// Update agent status
    pub fn updateAgentStatus(self: *AgentTopology, agent_id: []const u8, status: AgentStatus) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.agents.getPtr(agent_id)) |agent| {
            agent.updateStatus(status);
        } else {
            return error.AgentNotFound;
        }
    }
    
    /// Get all agents
    pub fn getAllAgents(self: *AgentTopology) !std.ArrayList(Agent) {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var agents_list = std.ArrayList(Agent){};
        
        var it = self.agents.valueIterator();
        while (it.next()) |agent| {
            try agents_list.append(agent.*);
        }
        
        return agents_list;
    }
    
    // ========================================================================
    // CONNECTION MANAGEMENT
    // ========================================================================
    
    /// Add a connection between agents
    pub fn addConnection(self: *AgentTopology, connection: AgentConnection) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Verify both agents exist
        if (!self.agents.contains(connection.from_agent_id)) {
            return error.FromAgentNotFound;
        }
        if (!self.agents.contains(connection.to_agent_id)) {
            return error.ToAgentNotFound;
        }
        
        // Check for duplicate connection
        for (self.connections.items) |*conn| {
            if (std.mem.eql(u8, conn.from_agent_id, connection.from_agent_id) and
                std.mem.eql(u8, conn.to_agent_id, connection.to_agent_id))
            {
                return error.ConnectionAlreadyExists;
            }
        }
        
        try self.connections.append(connection);
        
        std.log.info("Added connection: {s} -> {s}", .{ connection.from_agent_id, connection.to_agent_id });
    }
    
    /// Get connections from an agent
    pub fn getOutgoingConnections(self: *AgentTopology, agent_id: []const u8) !std.ArrayList(AgentConnection) {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var outgoing = std.ArrayList(AgentConnection){};
        
        for (self.connections.items) |conn| {
            if (std.mem.eql(u8, conn.from_agent_id, agent_id)) {
                try outgoing.append(conn);
            }
        }
        
        return outgoing;
    }
    
    /// Get connections to an agent
    pub fn getIncomingConnections(self: *AgentTopology, agent_id: []const u8) !std.ArrayList(AgentConnection) {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var incoming = std.ArrayList(AgentConnection){};
        
        for (self.connections.items) |conn| {
            if (std.mem.eql(u8, conn.to_agent_id, agent_id)) {
                try incoming.append(conn);
            }
        }
        
        return incoming;
    }
    
    // ========================================================================
    // TOPOLOGY QUERY
    // ========================================================================
    
    /// Get topology as JSON for visualization
    pub fn toJSON(self: *AgentTopology, allocator: Allocator) ![]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var json = std.ArrayList(u8){};
        const writer = json.writer();
        
        try writer.writeAll("{\"nodes\":[");
        
        // Write agents as nodes
        var first_node = true;
        var agent_it = self.agents.iterator();
        while (agent_it.next()) |entry| {
            if (!first_node) try writer.writeAll(",");
            first_node = false;
            
            const agent = entry.value_ptr.*;
            try writer.print("{{\"id\":\"{s}\",\"name\":\"{s}\",\"type\":\"{s}\",\"status\":\"{s}\"}}", .{
                agent.id,
                agent.name,
                @tagName(agent.type),
                @tagName(agent.status),
            });
        }
        
        try writer.writeAll("],\"edges\":[");
        
        // Write connections as edges
        var first_edge = true;
        for (self.connections.items) |conn| {
            if (!first_edge) try writer.writeAll(",");
            first_edge = false;
            
            try writer.print("{{\"from\":\"{s}\",\"to\":\"{s}\",\"type\":\"{s}\",\"weight\":{d:.2}}}", .{
                conn.from_agent_id,
                conn.to_agent_id,
                @tagName(conn.connection_type),
                conn.weight,
            });
        }
        
        try writer.writeAll("]}");
        
        return json.toOwnedSlice();
    }
    
    /// Get topology statistics
    pub fn getStats(self: *AgentTopology) TopologyStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var active_count: u32 = 0;
        var agent_it = self.agents.valueIterator();
        while (agent_it.next()) |agent| {
            if (agent.status == .active) active_count += 1;
        }
        
        return .{
            .total_agents = @intCast(self.agents.count()),
            .active_agents = active_count,
            .total_connections = @intCast(self.connections.items.len),
        };
    }
};

/// Topology statistics
pub const TopologyStats = struct {
    total_agents: u32,
    active_agents: u32,
    total_connections: u32,
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "Agent: initialization and cleanup" {
    const allocator = std.testing.allocator;
    
    var agent = try Agent.init(allocator, "agent-1", "Test Agent", .llm, "http://localhost:8080");
    defer agent.deinit(allocator);
    
    try std.testing.expectEqualStrings("agent-1", agent.id);
    try std.testing.expectEqualStrings("Test Agent", agent.name);
    try std.testing.expectEqual(AgentType.llm, agent.type);
    try std.testing.expectEqual(AgentStatus.active, agent.status);
}

test "Agent: add capability" {
    const allocator = std.testing.allocator;
    
    var agent = try Agent.init(allocator, "agent-1", "Test Agent", .llm, "http://localhost:8080");
    defer agent.deinit(allocator);
    
    const cap = try AgentCapability.init(allocator, "generation", "Text generation", "1.0");
    try agent.addCapability(cap);
    
    try std.testing.expectEqual(@as(usize, 1), agent.capabilities.items.len);
}

test "AgentTopology: register and get agent" {
    const allocator = std.testing.allocator;
    
    const topology = try AgentTopology.init(allocator);
    defer topology.deinit();
    
    const agent = try Agent.init(allocator, "agent-1", "Test Agent", .llm, "http://localhost:8080");
    try topology.registerAgent(agent);
    
    const retrieved = topology.getAgent("agent-1");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings("Test Agent", retrieved.?.name);
}

test "AgentTopology: add connection" {
    const allocator = std.testing.allocator;
    
    const topology = try AgentTopology.init(allocator);
    defer topology.deinit();
    
    const agent1 = try Agent.init(allocator, "agent-1", "Agent 1", .llm, "http://localhost:8080");
    const agent2 = try Agent.init(allocator, "agent-2", "Agent 2", .code, "http://localhost:8081");
    
    try topology.registerAgent(agent1);
    try topology.registerAgent(agent2);
    
    const conn = try AgentConnection.init(allocator, "agent-1", "agent-2", .sync, 1.0);
    try topology.addConnection(conn);
    
    const stats = topology.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.total_agents);
    try std.testing.expectEqual(@as(u32, 1), stats.total_connections);
}

test "AgentTopology: unregister agent removes connections" {
    const allocator = std.testing.allocator;
    
    const topology = try AgentTopology.init(allocator);
    defer topology.deinit();
    
    const agent1 = try Agent.init(allocator, "agent-1", "Agent 1", .llm, "http://localhost:8080");
    const agent2 = try Agent.init(allocator, "agent-2", "Agent 2", .code, "http://localhost:8081");
    
    try topology.registerAgent(agent1);
    try topology.registerAgent(agent2);
    
    const conn = try AgentConnection.init(allocator, "agent-1", "agent-2", .sync, 1.0);
    try topology.addConnection(conn);
    
    try topology.unregisterAgent("agent-1");
    
    const stats = topology.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.total_agents);
    try std.testing.expectEqual(@as(u32, 0), stats.total_connections);
}

test "AgentTopology: JSON export" {
    const allocator = std.testing.allocator;
    
    const topology = try AgentTopology.init(allocator);
    defer topology.deinit();
    
    const agent = try Agent.init(allocator, "agent-1", "Test Agent", .llm, "http://localhost:8080");
    try topology.registerAgent(agent);
    
    const json = try topology.toJSON(allocator);
    defer allocator.free(json);
    
    try std.testing.expect(json.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, json, "agent-1") != null);
}
