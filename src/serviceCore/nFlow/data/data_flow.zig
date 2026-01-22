const std = @import("std");
const Allocator = std.mem.Allocator;
const data_packet = @import("data_packet.zig");

pub const DataPacket = data_packet.DataPacket;
pub const DataType = data_packet.DataType;
pub const DataSchema = data_packet.DataSchema;

/// Data flow manager for handling data packets in workflows
pub const DataFlowManager = struct {
    allocator: Allocator,
    packets: std.StringHashMap(*DataPacket),
    connections: std.ArrayList(Connection),
    validators: std.StringHashMap(DataSchema),
    
    pub fn init(allocator: Allocator) DataFlowManager {
        return .{
            .allocator = allocator,
            .packets = std.StringHashMap(*DataPacket).init(allocator),
            .connections = std.ArrayList(Connection){},
            .validators = std.StringHashMap(DataSchema).init(allocator),
        };
    }
    
    pub fn deinit(self: *DataFlowManager) void {
        // Clean up packets and their keys
        var packet_it = self.packets.iterator();
        while (packet_it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.packets.deinit();
        
        // Clean up connections
        for (self.connections.items) |*conn| {
            self.allocator.free(conn.from_node);
            self.allocator.free(conn.from_port);
            self.allocator.free(conn.to_node);
            self.allocator.free(conn.to_port);
        }
        self.connections.deinit(self.allocator);
        
        // Clean up validators
        var validator_it = self.validators.iterator();
        while (validator_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.validators.deinit();
    }
    
    /// Store a data packet
    pub fn storePacket(self: *DataFlowManager, packet: *DataPacket) !void {
        const key = try self.allocator.dupe(u8, packet.id);
        errdefer self.allocator.free(key);
        
        try self.packets.put(key, packet);
    }
    
    /// Retrieve a data packet by ID
    pub fn getPacket(self: *const DataFlowManager, id: []const u8) ?*DataPacket {
        return self.packets.get(id);
    }
    
    /// Remove and return a data packet
    pub fn removePacket(self: *DataFlowManager, id: []const u8) ?*DataPacket {
        if (self.packets.fetchRemove(id)) |entry| {
            self.allocator.free(entry.key);
            return entry.value;
        }
        return null;
    }
    
    /// Add a connection between nodes
    pub fn addConnection(self: *DataFlowManager, from_node: []const u8, from_port: []const u8, to_node: []const u8, to_port: []const u8) !void {
        const conn = Connection{
            .from_node = try self.allocator.dupe(u8, from_node),
            .from_port = try self.allocator.dupe(u8, from_port),
            .to_node = try self.allocator.dupe(u8, to_node),
            .to_port = try self.allocator.dupe(u8, to_port),
        };
        try self.connections.append(self.allocator, conn);
    }
    
    /// Get all connections from a specific node and port
    pub fn getConnectionsFrom(self: *const DataFlowManager, node_id: []const u8, port_id: []const u8) ![]Connection {
        var result = std.ArrayList(Connection){};
        
        for (self.connections.items) |conn| {
            if (std.mem.eql(u8, conn.from_node, node_id) and std.mem.eql(u8, conn.from_port, port_id)) {
                try result.append(self.allocator, conn);
            }
        }
        
        return result.toOwnedSlice(self.allocator);
    }
    
    /// Get all connections to a specific node and port
    pub fn getConnectionsTo(self: *const DataFlowManager, node_id: []const u8, port_id: []const u8) ![]Connection {
        var result = std.ArrayList(Connection){};
        
        for (self.connections.items) |conn| {
            if (std.mem.eql(u8, conn.to_node, node_id) and std.mem.eql(u8, conn.to_port, port_id)) {
                try result.append(self.allocator, conn);
            }
        }
        
        return result.toOwnedSlice(self.allocator);
    }
    
    /// Register a validator for a specific port
    pub fn registerValidator(self: *DataFlowManager, port_key: []const u8, schema: DataSchema) !void {
        const key = try self.allocator.dupe(u8, port_key);
        try self.validators.put(key, schema);
    }
    
    /// Validate a packet against registered schema
    pub fn validatePacket(self: *const DataFlowManager, port_key: []const u8, packet: *const DataPacket) !void {
        if (self.validators.get(port_key)) |schema| {
            try packet.validate(&schema);
        }
    }
    
    /// Send data from one node to connected nodes
    pub fn sendData(self: *DataFlowManager, from_node: []const u8, from_port: []const u8, packet: *DataPacket) ![]RoutedPacket {
        // Validate outgoing data
        const port_key = try std.fmt.allocPrint(self.allocator, "{s}:{s}", .{ from_node, from_port });
        defer self.allocator.free(port_key);
        
        try self.validatePacket(port_key, packet);
        
        // Get connections from this port
        const connections = try self.getConnectionsFrom(from_node, from_port);
        defer self.allocator.free(connections);
        
        // Route packet to all connected nodes
        var routed = std.ArrayList(RoutedPacket){};
        
        for (connections) |conn| {
            const routed_packet = RoutedPacket{
                .packet = packet,
                .target_node = try self.allocator.dupe(u8, conn.to_node),
                .target_port = try self.allocator.dupe(u8, conn.to_port),
            };
            try routed.append(self.allocator, routed_packet);
        }
        
        return routed.toOwnedSlice(self.allocator);
    }
    
    /// Clear all stored packets
    pub fn clearPackets(self: *DataFlowManager) void {
        var it = self.packets.valueIterator();
        while (it.next()) |packet| {
            packet.*.deinit();
        }
        self.packets.clearAndFree();
    }
    
    /// Get statistics about data flow
    pub fn getStats(self: *const DataFlowManager) DataFlowStats {
        return .{
            .total_packets = self.packets.count(),
            .total_connections = self.connections.items.len,
            .total_validators = self.validators.count(),
        };
    }
};

/// Connection between two nodes
pub const Connection = struct {
    from_node: []const u8,
    from_port: []const u8,
    to_node: []const u8,
    to_port: []const u8,
};

/// Routed packet with target information
pub const RoutedPacket = struct {
    packet: *DataPacket,
    target_node: []const u8,
    target_port: []const u8,
    
    pub fn deinit(self: *RoutedPacket, allocator: Allocator) void {
        allocator.free(self.target_node);
        allocator.free(self.target_port);
    }
};

/// Data flow statistics
pub const DataFlowStats = struct {
    total_packets: usize,
    total_connections: usize,
    total_validators: usize,
};

/// Data buffer for temporary storage
pub const DataBuffer = struct {
    allocator: Allocator,
    packets: std.ArrayList(*DataPacket),
    max_size: usize,
    
    pub fn init(allocator: Allocator, max_size: usize) DataBuffer {
        return .{
            .allocator = allocator,
            .packets = std.ArrayList(*DataPacket){},
            .max_size = max_size,
        };
    }
    
    pub fn deinit(self: *DataBuffer) void {
        for (self.packets.items) |packet| {
            packet.deinit();
        }
        self.packets.deinit(self.allocator);
    }
    
    pub fn push(self: *DataBuffer, packet: *DataPacket) !void {
        if (self.packets.items.len >= self.max_size) {
            return error.BufferFull;
        }
        try self.packets.append(self.allocator, packet);
    }
    
    pub fn pop(self: *DataBuffer) ?*DataPacket {
        if (self.packets.items.len == 0) return null;
        return self.packets.pop();
    }
    
    pub fn peek(self: *const DataBuffer) ?*DataPacket {
        if (self.packets.items.len == 0) return null;
        return self.packets.items[self.packets.items.len - 1];
    }
    
    pub fn size(self: *const DataBuffer) usize {
        return self.packets.items.len;
    }
    
    pub fn isFull(self: *const DataBuffer) bool {
        return self.packets.items.len >= self.max_size;
    }
    
    pub fn isEmpty(self: *const DataBuffer) bool {
        return self.packets.items.len == 0;
    }
    
    pub fn clear(self: *DataBuffer) void {
        for (self.packets.items) |packet| {
            packet.deinit();
        }
        self.packets.clearRetainingCapacity();
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "DataFlowManager creation and cleanup" {
    const allocator = std.testing.allocator;
    
    var manager = DataFlowManager.init(allocator);
    defer manager.deinit();
    
    const stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.total_packets);
    try std.testing.expectEqual(@as(usize, 0), stats.total_connections);
}

test "DataFlowManager store and retrieve packet" {
    const allocator = std.testing.allocator;
    
    var manager = DataFlowManager.init(allocator);
    defer manager.deinit();
    
    const value = std.json.Value{ .string = "test" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    // Note: packet ownership transfers to manager, it will be freed in manager.deinit()
    
    try manager.storePacket(packet);
    
    const retrieved = manager.getPacket("p1");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings("p1", retrieved.?.id);
}

test "DataFlowManager add connections" {
    const allocator = std.testing.allocator;
    
    var manager = DataFlowManager.init(allocator);
    defer manager.deinit();
    
    try manager.addConnection("node1", "out1", "node2", "in1");
    try manager.addConnection("node1", "out1", "node3", "in1");
    
    const stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.total_connections);
}

test "DataFlowManager get connections from node" {
    const allocator = std.testing.allocator;
    
    var manager = DataFlowManager.init(allocator);
    defer manager.deinit();
    
    try manager.addConnection("node1", "out1", "node2", "in1");
    try manager.addConnection("node1", "out1", "node3", "in1");
    try manager.addConnection("node2", "out1", "node4", "in1");
    
    const connections = try manager.getConnectionsFrom("node1", "out1");
    defer allocator.free(connections);
    
    try std.testing.expectEqual(@as(usize, 2), connections.len);
}

test "DataFlowManager validate packet" {
    const allocator = std.testing.allocator;
    
    var manager = DataFlowManager.init(allocator);
    defer manager.deinit();
    
    const schema = DataSchema.init(.string, true, .{
        .string_constraints = .{
            .min_length = 3,
            .max_length = 10,
        },
    });
    
    try manager.registerValidator("node1:out1", schema);
    
    const value = std.json.Value{ .string = "hello" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    defer packet.deinit();
    
    try manager.validatePacket("node1:out1", packet);
}

test "DataFlowManager send data" {
    const allocator = std.testing.allocator;
    
    var manager = DataFlowManager.init(allocator);
    defer manager.deinit();
    
    try manager.addConnection("node1", "out1", "node2", "in1");
    try manager.addConnection("node1", "out1", "node3", "in1");
    
    const value = std.json.Value{ .string = "test" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    defer packet.deinit();
    
    const routed = try manager.sendData("node1", "out1", packet);
    defer {
        for (routed) |*r| {
            r.deinit(allocator);
        }
        allocator.free(routed);
    }
    
    try std.testing.expectEqual(@as(usize, 2), routed.len);
    try std.testing.expectEqualStrings("node2", routed[0].target_node);
    try std.testing.expectEqualStrings("node3", routed[1].target_node);
}

test "DataBuffer push and pop" {
    const allocator = std.testing.allocator;
    
    var buffer = DataBuffer.init(allocator, 5);
    defer buffer.deinit();
    
    const value1 = std.json.Value{ .string = "test1" };
    const packet1 = try DataPacket.init(allocator, "p1", .string, value1);
    
    const value2 = std.json.Value{ .string = "test2" };
    const packet2 = try DataPacket.init(allocator, "p2", .string, value2);
    
    try buffer.push(packet1);
    try buffer.push(packet2);
    
    try std.testing.expectEqual(@as(usize, 2), buffer.size());
    
    const popped = buffer.pop();
    try std.testing.expect(popped != null);
    try std.testing.expectEqualStrings("p2", popped.?.id);
    popped.?.deinit(); // Clean up popped packet
    
    try std.testing.expectEqual(@as(usize, 1), buffer.size());
}

test "DataBuffer full detection" {
    const allocator = std.testing.allocator;
    
    var buffer = DataBuffer.init(allocator, 2);
    defer buffer.deinit();
    
    try std.testing.expect(!buffer.isFull());
    
    const value1 = std.json.Value{ .string = "test1" };
    const packet1 = try DataPacket.init(allocator, "p1", .string, value1);
    try buffer.push(packet1);
    
    const value2 = std.json.Value{ .string = "test2" };
    const packet2 = try DataPacket.init(allocator, "p2", .string, value2);
    try buffer.push(packet2);
    
    try std.testing.expect(buffer.isFull());
    
    const value3 = std.json.Value{ .string = "test3" };
    const packet3 = try DataPacket.init(allocator, "p3", .string, value3);
    defer packet3.deinit();
    
    try std.testing.expectError(error.BufferFull, buffer.push(packet3));
}

test "DataBuffer peek" {
    const allocator = std.testing.allocator;
    
    var buffer = DataBuffer.init(allocator, 5);
    defer buffer.deinit();
    
    try std.testing.expect(buffer.peek() == null);
    
    const value = std.json.Value{ .string = "test" };
    const packet = try DataPacket.init(allocator, "p1", .string, value);
    try buffer.push(packet);
    
    const peeked = buffer.peek();
    try std.testing.expect(peeked != null);
    try std.testing.expectEqualStrings("p1", peeked.?.id);
    
    // Buffer should still have the packet
    try std.testing.expectEqual(@as(usize, 1), buffer.size());
}

test "DataBuffer clear" {
    const allocator = std.testing.allocator;
    
    var buffer = DataBuffer.init(allocator, 5);
    defer buffer.deinit();
    
    const value1 = std.json.Value{ .string = "test1" };
    const packet1 = try DataPacket.init(allocator, "p1", .string, value1);
    try buffer.push(packet1);
    
    const value2 = std.json.Value{ .string = "test2" };
    const packet2 = try DataPacket.init(allocator, "p2", .string, value2);
    try buffer.push(packet2);
    
    try std.testing.expectEqual(@as(usize, 2), buffer.size());
    
    buffer.clear();
    
    try std.testing.expectEqual(@as(usize, 0), buffer.size());
    try std.testing.expect(buffer.isEmpty());
}
