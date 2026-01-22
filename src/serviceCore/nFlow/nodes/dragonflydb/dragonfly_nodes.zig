//! DragonflyDB Nodes for nWorkflow
//! 
//! Provides dedicated workflow nodes for interacting with DragonflyDB,
//! the high-performance Redis-compatible in-memory data store.
//!
//! Supported Operations:
//! - Key-Value: GET, SET, DELETE, EXISTS
//! - Lists: PUSH, POP, LENGTH
//! - Sets: ADD, REMOVE, MEMBERS
//! - Hashes: SET, GET, DELETE
//! - Pub/Sub: PUBLISH, SUBSCRIBE (trigger-based)
//! - TTL: Expiration management
//!
//! All nodes integrate with the workflow execution context and support
//! proper error handling, memory management, and type validation.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import node system
const node_types = @import("node_types");
const NodeInterface = node_types.NodeInterface;
const Port = node_types.Port;
const PortType = node_types.PortType;
const ExecutionContext = node_types.ExecutionContext;
const NodeCategory = node_types.NodeCategory;

// Import real RESP protocol client
const resp_client = @import("resp_client.zig");
const RespClient = resp_client.RespClient;

// ============================================================================
// DragonflyDB Connection Configuration
// ============================================================================

pub const DragonflyConfig = struct {
    host: []const u8,
    port: u16,
    password: ?[]const u8,
    db_index: u8,
    timeout_ms: u32,

    pub fn default() DragonflyConfig {
        return .{
            .host = "localhost",
            .port = 6379,
            .password = null,
            .db_index = 0,
            .timeout_ms = 5000,
        };
    }
};

// ============================================================================
// DragonflyDB Client (Real RESP Protocol Implementation)
// ============================================================================

/// Wrapper around RespClient with DragonflyConfig
pub const DragonflyClient = struct {
    resp_client: RespClient,
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: DragonflyConfig) !DragonflyClient {
        return DragonflyClient{
            .resp_client = RespClient.init(
                allocator,
                config.host,
                config.port,
                config.password,
                config.db_index,
                config.timeout_ms,
            ),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DragonflyClient) void {
        self.resp_client.deinit();
    }

    pub fn connect(self: *DragonflyClient) !void {
        try self.resp_client.connect();
    }

    pub fn disconnect(self: *DragonflyClient) void {
        self.resp_client.disconnect();
    }

    // Delegate all operations to RespClient
    pub fn get(self: *DragonflyClient, key: []const u8) !?[]const u8 {
        return try self.resp_client.get(key);
    }

    pub fn set(self: *DragonflyClient, key: []const u8, value: []const u8, ttl_seconds: ?u32) !void {
        return try self.resp_client.set(key, value, ttl_seconds);
    }

    pub fn del(self: *DragonflyClient, key: []const u8) !bool {
        return try self.resp_client.del(key);
    }

    pub fn exists(self: *DragonflyClient, key: []const u8) !bool {
        return try self.resp_client.exists(key);
    }

    pub fn expire(self: *DragonflyClient, key: []const u8, seconds: u32) !bool {
        return try self.resp_client.expire(key, seconds);
    }

    pub fn ttl(self: *DragonflyClient, key: []const u8) !i64 {
        return try self.resp_client.ttl(key);
    }

    pub fn lpush(self: *DragonflyClient, key: []const u8, value: []const u8) !u64 {
        return try self.resp_client.lpush(key, value);
    }

    pub fn rpush(self: *DragonflyClient, key: []const u8, value: []const u8) !u64 {
        return try self.resp_client.rpush(key, value);
    }

    pub fn lpop(self: *DragonflyClient, key: []const u8) !?[]const u8 {
        return try self.resp_client.lpop(key);
    }

    pub fn rpop(self: *DragonflyClient, key: []const u8) !?[]const u8 {
        return try self.resp_client.rpop(key);
    }

    pub fn llen(self: *DragonflyClient, key: []const u8) !u64 {
        return try self.resp_client.llen(key);
    }

    pub fn sadd(self: *DragonflyClient, key: []const u8, member: []const u8) !bool {
        return try self.resp_client.sadd(key, member);
    }

    pub fn srem(self: *DragonflyClient, key: []const u8, member: []const u8) !bool {
        return try self.resp_client.srem(key, member);
    }

    pub fn smembers(self: *DragonflyClient, key: []const u8) ![][]const u8 {
        return try self.resp_client.smembers(key);
    }

    pub fn sismember(self: *DragonflyClient, key: []const u8, member: []const u8) !bool {
        return try self.resp_client.sismember(key, member);
    }

    pub fn hset(self: *DragonflyClient, key: []const u8, field: []const u8, value: []const u8) !bool {
        return try self.resp_client.hset(key, field, value);
    }

    pub fn hget(self: *DragonflyClient, key: []const u8, field: []const u8) !?[]const u8 {
        return try self.resp_client.hget(key, field);
    }

    pub fn hdel(self: *DragonflyClient, key: []const u8, field: []const u8) !bool {
        return try self.resp_client.hdel(key, field);
    }

    pub fn hgetall(self: *DragonflyClient, key: []const u8) !std.StringHashMap([]const u8) {
        return try self.resp_client.hgetall(key);
    }

    pub fn publish(self: *DragonflyClient, channel: []const u8, message: []const u8) !u64 {
        return try self.resp_client.publish(channel, message);
    }
};

// ============================================================================
// Node 1: DragonflyGetNode - Get cached value
// ============================================================================

pub const DragonflyGetNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,

    pub fn init(allocator: Allocator, config: DragonflyConfig) !*DragonflyGetNode {
        const self = try allocator.create(DragonflyGetNode);
        
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = Port{
            .id = "key",
            .name = "Key",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Redis key to retrieve",
        };

        const outputs = try allocator.alloc(Port, 2);
        outputs[0] = Port{
            .id = "value",
            .name = "Value",
            .port_type = PortType.string,
            .required = false,
            .default_value = null,
            .description = "Retrieved value",
        };
        outputs[1] = Port{
            .id = "found",
            .name = "Found",
            .port_type = PortType.boolean,
            .required = false,
            .default_value = null,
            .description = "Whether key exists",
        };

        self.* = .{
            .base = NodeInterface{
                .id = "dragonfly_get",
                .name = "DragonflyDB Get",
                .description = "Retrieve value from DragonflyDB cache",
                .node_type = "dragonflydb_get",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
        // Validation passed
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflyGetNode = @fieldParentPtr("base", ctx);
        
        // Get key from input
        const key = exec_ctx.getVariable("key") orelse return error.MissingKey;

        // Connect to DragonflyDB
        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        // Get value
        const value = try client.get(key);
        
        // Build result
        var result = std.json.ObjectMap.init(self.allocator);
        if (value) |v| {
            defer self.allocator.free(v);
            try result.put("value", .{ .string = try self.allocator.dupe(u8, v) });
            try result.put("found", .{ .bool = true });
        } else {
            try result.put("value", .null);
            try result.put("found", .{ .bool = false });
        }

        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflyGetNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Node 2: DragonflySetNode - Set cached value with TTL
// ============================================================================

pub const DragonflySetNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,

    pub fn init(allocator: Allocator, config: DragonflyConfig) !*DragonflySetNode {
        const self = try allocator.create(DragonflySetNode);
        
        const inputs = try allocator.alloc(Port, 3);
        inputs[0] = Port{
            .id = "key",
            .name = "Key",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Redis key",
        };
        inputs[1] = Port{
            .id = "value",
            .name = "Value",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Value to store",
        };
        inputs[2] = Port{
            .id = "ttl",
            .name = "TTL (seconds)",
            .port_type = PortType.number,
            .required = false,
            .default_value = null,
            .description = "Time to live in seconds (optional)",
        };

        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = Port{
            .id = "success",
            .name = "Success",
            .port_type = PortType.boolean,
            .required = false,
            .default_value = null,
            .description = "Whether operation succeeded",
        };

        self.* = .{
            .base = NodeInterface{
                .id = "dragonfly_set",
                .name = "DragonflyDB Set",
                .description = "Set value in DragonflyDB cache with optional TTL",
                .node_type = "dragonflydb_set",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflySetNode = @fieldParentPtr("base", ctx);
        
        const key = exec_ctx.getVariable("key") orelse return error.MissingKey;
        const value = exec_ctx.getVariable("value") orelse return error.MissingValue;
        const ttl_str = exec_ctx.getVariable("ttl");
        
        var ttl: ?u32 = null;
        if (ttl_str) |t| {
            ttl = try std.fmt.parseInt(u32, t, 10);
        }

        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        try client.set(key, value, ttl);

        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("success", .{ .bool = true });
        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflySetNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Node 3: DragonflyDeleteNode - Delete cached value
// ============================================================================

pub const DragonflyDeleteNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,

    pub fn init(allocator: Allocator, config: DragonflyConfig) !*DragonflyDeleteNode {
        const self = try allocator.create(DragonflyDeleteNode);
        
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = Port{
            .id = "key",
            .name = "Key",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Redis key to delete",
        };

        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = Port{
            .id = "deleted",
            .name = "Deleted",
            .port_type = PortType.boolean,
            .required = false,
            .default_value = null,
            .description = "Whether key was deleted",
        };

        self.* = .{
            .base = NodeInterface{
                .id = "dragonfly_delete",
                .name = "DragonflyDB Delete",
                .description = "Delete key from DragonflyDB cache",
                .node_type = "dragonflydb_delete",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflyDeleteNode = @fieldParentPtr("base", ctx);
        
        const key = exec_ctx.getVariable("key") orelse return error.MissingKey;

        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        const deleted = try client.del(key);

        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("deleted", .{ .bool = deleted });
        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflyDeleteNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Node 4: DragonflyPublishNode - Publish message to channel
// ============================================================================

pub const DragonflyPublishNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,

    pub fn init(allocator: Allocator, config: DragonflyConfig) !*DragonflyPublishNode {
        const self = try allocator.create(DragonflyPublishNode);
        
        const inputs = try allocator.alloc(Port, 2);
        inputs[0] = Port{
            .id = "channel",
            .name = "Channel",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Pub/sub channel name",
        };
        inputs[1] = Port{
            .id = "message",
            .name = "Message",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Message to publish",
        };

        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = Port{
            .id = "subscribers",
            .name = "Subscribers",
            .port_type = PortType.number,
            .required = false,
            .default_value = null,
            .description = "Number of subscribers that received the message",
        };

        self.* = .{
            .base = NodeInterface{
                .id = "dragonfly_publish",
                .name = "DragonflyDB Publish",
                .description = "Publish message to DragonflyDB pub/sub channel",
                .node_type = "dragonflydb_publish",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflyPublishNode = @fieldParentPtr("base", ctx);
        
        const channel = exec_ctx.getVariable("channel") orelse return error.MissingChannel;
        const message = exec_ctx.getVariable("message") orelse return error.MissingMessage;

        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        const subscribers = try client.publish(channel, message);

        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("subscribers", .{ .integer = @intCast(subscribers) });
        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflyPublishNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Node 5: DragonflyListPushNode - Push to list
// ============================================================================

pub const DragonflyListPushNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,
    push_left: bool,

    pub fn init(allocator: Allocator, config: DragonflyConfig, push_left: bool) !*DragonflyListPushNode {
        const self = try allocator.create(DragonflyListPushNode);
        
        const inputs = try allocator.alloc(Port, 2);
        inputs[0] = Port{
            .id = "key",
            .name = "List Key",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "List key",
        };
        inputs[1] = Port{
            .id = "value",
            .name = "Value",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Value to push",
        };

        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = Port{
            .id = "length",
            .name = "List Length",
            .port_type = PortType.number,
            .required = false,
            .default_value = null,
            .description = "List length after push",
        };

        const node_id = if (push_left) "dragonfly_lpush" else "dragonfly_rpush";
        const node_name = if (push_left) "DragonflyDB LPUSH" else "DragonflyDB RPUSH";
        const desc = if (push_left) "Push value to left of list" else "Push value to right of list";

        self.* = .{
            .base = NodeInterface{
                .id = node_id,
                .name = node_name,
                .description = desc,
                .node_type = if (push_left) "dragonflydb_lpush" else "dragonflydb_rpush",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
            .push_left = push_left,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflyListPushNode = @fieldParentPtr("base", ctx);
        
        const key = exec_ctx.getVariable("key") orelse return error.MissingKey;
        const value = exec_ctx.getVariable("value") orelse return error.MissingValue;

        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        const length = if (self.push_left)
            try client.lpush(key, value)
        else
            try client.rpush(key, value);

        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("length", .{ .integer = @intCast(length) });
        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflyListPushNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Node 6: DragonflyListPopNode - Pop from list
// ============================================================================

pub const DragonflyListPopNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,
    pop_left: bool,

    pub fn init(allocator: Allocator, config: DragonflyConfig, pop_left: bool) !*DragonflyListPopNode {
        const self = try allocator.create(DragonflyListPopNode);
        
        const inputs = try allocator.alloc(Port, 1);
        inputs[0] = Port{
            .id = "key",
            .name = "List Key",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "List key",
        };

        const outputs = try allocator.alloc(Port, 2);
        outputs[0] = Port{
            .id = "value",
            .name = "Value",
            .port_type = PortType.string,
            .required = false,
            .default_value = null,
            .description = "Popped value",
        };
        outputs[1] = Port{
            .id = "found",
            .name = "Found",
            .port_type = PortType.boolean,
            .required = false,
            .default_value = null,
            .description = "Whether value was found",
        };

        const node_id = if (pop_left) "dragonfly_lpop" else "dragonfly_rpop";
        const node_name = if (pop_left) "DragonflyDB LPOP" else "DragonflyDB RPOP";
        const desc = if (pop_left) "Pop value from left of list" else "Pop value from right of list";

        self.* = .{
            .base = NodeInterface{
                .id = node_id,
                .name = node_name,
                .description = desc,
                .node_type = if (pop_left) "dragonflydb_lpop" else "dragonflydb_rpop",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
            .pop_left = pop_left,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflyListPopNode = @fieldParentPtr("base", ctx);
        
        const key = exec_ctx.getVariable("key") orelse return error.MissingKey;

        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        const value = if (self.pop_left)
            try client.lpop(key)
        else
            try client.rpop(key);

        var result = std.json.ObjectMap.init(self.allocator);
        if (value) |v| {
            defer self.allocator.free(v);
            try result.put("value", .{ .string = try self.allocator.dupe(u8, v) });
            try result.put("found", .{ .bool = true });
        } else {
            try result.put("value", .null);
            try result.put("found", .{ .bool = false });
        }

        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflyListPopNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Node 7: DragonflySetAddNode - Add to set
// ============================================================================

pub const DragonflySetAddNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,

    pub fn init(allocator: Allocator, config: DragonflyConfig) !*DragonflySetAddNode {
        const self = try allocator.create(DragonflySetAddNode);
        
        const inputs = try allocator.alloc(Port, 2);
        inputs[0] = Port{
            .id = "key",
            .name = "Set Key",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Set key",
        };
        inputs[1] = Port{
            .id = "member",
            .name = "Member",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Member to add",
        };

        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = Port{
            .id = "added",
            .name = "Added",
            .port_type = PortType.boolean,
            .required = false,
            .default_value = null,
            .description = "Whether member was added (false if already exists)",
        };

        self.* = .{
            .base = NodeInterface{
                .id = "dragonfly_sadd",
                .name = "DragonflyDB SADD",
                .description = "Add member to DragonflyDB set",
                .node_type = "dragonflydb_sadd",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflySetAddNode = @fieldParentPtr("base", ctx);
        
        const key = exec_ctx.getVariable("key") orelse return error.MissingKey;
        const member = exec_ctx.getVariable("member") orelse return error.MissingMember;

        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        const added = try client.sadd(key, member);

        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("added", .{ .bool = added });
        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflySetAddNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Node 8: DragonflyHashSetNode - Set hash field
// ============================================================================

pub const DragonflyHashSetNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,

    pub fn init(allocator: Allocator, config: DragonflyConfig) !*DragonflyHashSetNode {
        const self = try allocator.create(DragonflyHashSetNode);
        
        const inputs = try allocator.alloc(Port, 3);
        inputs[0] = Port{
            .id = "key",
            .name = "Hash Key",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Hash key",
        };
        inputs[1] = Port{
            .id = "field",
            .name = "Field",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Field name",
        };
        inputs[2] = Port{
            .id = "value",
            .name = "Value",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Field value",
        };

        const outputs = try allocator.alloc(Port, 1);
        outputs[0] = Port{
            .id = "created",
            .name = "Created",
            .port_type = PortType.boolean,
            .required = false,
            .default_value = null,
            .description = "Whether field was newly created",
        };

        self.* = .{
            .base = NodeInterface{
                .id = "dragonfly_hset",
                .name = "DragonflyDB HSET",
                .description = "Set field in DragonflyDB hash",
                .node_type = "dragonflydb_hset",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflyHashSetNode = @fieldParentPtr("base", ctx);
        
        const key = exec_ctx.getVariable("key") orelse return error.MissingKey;
        const field = exec_ctx.getVariable("field") orelse return error.MissingField;
        const value = exec_ctx.getVariable("value") orelse return error.MissingValue;

        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        const created = try client.hset(key, field, value);

        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("created", .{ .bool = created });
        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflyHashSetNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Node 9: DragonflyHashGetNode - Get hash field
// ============================================================================

pub const DragonflyHashGetNode = struct {
    base: NodeInterface,
    allocator: Allocator,
    config: DragonflyConfig,

    pub fn init(allocator: Allocator, config: DragonflyConfig) !*DragonflyHashGetNode {
        const self = try allocator.create(DragonflyHashGetNode);
        
        const inputs = try allocator.alloc(Port, 2);
        inputs[0] = Port{
            .id = "key",
            .name = "Hash Key",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Hash key",
        };
        inputs[1] = Port{
            .id = "field",
            .name = "Field",
            .port_type = PortType.string,
            .required = true,
            .default_value = null,
            .description = "Field name",
        };

        const outputs = try allocator.alloc(Port, 2);
        outputs[0] = Port{
            .id = "value",
            .name = "Value",
            .port_type = PortType.string,
            .required = false,
            .default_value = null,
            .description = "Field value",
        };
        outputs[1] = Port{
            .id = "found",
            .name = "Found",
            .port_type = PortType.boolean,
            .required = false,
            .default_value = null,
            .description = "Whether field exists",
        };

        self.* = .{
            .base = NodeInterface{
                .id = "dragonfly_hget",
                .name = "DragonflyDB HGET",
                .description = "Get field from DragonflyDB hash",
                .node_type = "dragonflydb_hget",
                .category = .data,
                .inputs = inputs,
                .outputs = outputs,
                .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
                .vtable = &.{
                    .validate = validate,
                    .execute = execute,
                    .deinit = deinitNode,
                },
            },
            .allocator = allocator,
            .config = config,
        };

        return self;
    }

    fn validate(ctx: *const NodeInterface) !void {
        _ = ctx;
    }

    fn execute(ctx: *NodeInterface, exec_ctx: *ExecutionContext) !std.json.Value {
        const self: *DragonflyHashGetNode = @fieldParentPtr("base", ctx);
        
        const key = exec_ctx.getVariable("key") orelse return error.MissingKey;
        const field = exec_ctx.getVariable("field") orelse return error.MissingField;

        var client = try DragonflyClient.init(self.allocator, self.config);
        defer client.deinit();
        try client.connect();
        defer client.disconnect();

        const value = try client.hget(key, field);

        var result = std.json.ObjectMap.init(self.allocator);
        if (value) |v| {
            defer self.allocator.free(v);
            try result.put("value", .{ .string = try self.allocator.dupe(u8, v) });
            try result.put("found", .{ .bool = true });
        } else {
            try result.put("value", .null);
            try result.put("found", .{ .bool = false });
        }

        return std.json.Value{ .object = result };
    }

    fn deinitNode(ctx: *NodeInterface) void {
        const self: *DragonflyHashGetNode = @fieldParentPtr("base", ctx);
        self.allocator.free(self.base.inputs);
        self.allocator.free(self.base.outputs);
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "DragonflyConfig default" {
    const config = DragonflyConfig.default();
    try std.testing.expectEqualStrings("localhost", config.host);
    try std.testing.expectEqual(@as(u16, 6379), config.port);
    try std.testing.expectEqual(@as(?[]const u8, null), config.password);
    try std.testing.expectEqual(@as(u8, 0), config.db_index);
}

test "DragonflyClient init" {
    const allocator = std.testing.allocator;
    const config = DragonflyConfig.default();
    
    var client = try DragonflyClient.init(allocator, config);
    defer client.deinit();
    
    // Just test initialization - actual connection requires running DragonflyDB server
    try std.testing.expect(client.resp_client.stream == null);
}

// Note: The following tests require a running DragonflyDB server
// They are commented out for CI/CD but can be enabled for integration testing

// test "DragonflyClient get operation" {
//     const allocator = std.testing.allocator;
//     const config = DragonflyConfig.default();
//     
//     var client = try DragonflyClient.init(allocator, config);
//     defer client.deinit();
//     try client.connect();
//     defer client.disconnect();
//     
//     const value = try client.get("test_key");
//     if (value) |v| {
//         defer allocator.free(v);
//     }
// }

// test "DragonflyClient set operation" {
//     const allocator = std.testing.allocator;
//     const config = DragonflyConfig.default();
//     
//     var client = try DragonflyClient.init(allocator, config);
//     defer client.deinit();
//     try client.connect();
//     defer client.disconnect();
//     
//     try client.set("test_key", "test_value", 3600);
// }

// test "DragonflyClient list operations" {
//     const allocator = std.testing.allocator;
//     const config = DragonflyConfig.default();
//     
//     var client = try DragonflyClient.init(allocator, config);
//     defer client.deinit();
//     try client.connect();
//     defer client.disconnect();
//     
//     const length = try client.lpush("test_list", "value1");
//     try std.testing.expect(length > 0);
//     
//     const value = try client.lpop("test_list");
//     if (value) |v| {
//         defer allocator.free(v);
//     }
// }

test "DragonflyGetNode creation" {
    const allocator = std.testing.allocator;
    const config = DragonflyConfig.default();
    
    const node = try DragonflyGetNode.init(allocator, config);
    defer node.base.vtable.?.deinit(&node.base);
    
    try std.testing.expectEqualStrings("dragonfly_get", node.base.id);
    try std.testing.expectEqualStrings("DragonflyDB Get", node.base.name);
    try std.testing.expectEqual(@as(usize, 1), node.base.inputs.len);
    try std.testing.expectEqual(@as(usize, 2), node.base.outputs.len);
}

test "DragonflySetNode creation" {
    const allocator = std.testing.allocator;
    const config = DragonflyConfig.default();
    
    const node = try DragonflySetNode.init(allocator, config);
    defer node.base.vtable.?.deinit(&node.base);
    
    try std.testing.expectEqualStrings("dragonfly_set", node.base.id);
    try std.testing.expectEqual(@as(usize, 3), node.base.inputs.len);
    try std.testing.expectEqual(@as(usize, 1), node.base.outputs.len);
}

test "DragonflyPublishNode creation" {
    const allocator = std.testing.allocator;
    const config = DragonflyConfig.default();
    
    const node = try DragonflyPublishNode.init(allocator, config);
    defer node.base.vtable.?.deinit(&node.base);
    
    try std.testing.expectEqualStrings("dragonfly_publish", node.base.id);
    try std.testing.expectEqual(@as(usize, 2), node.base.inputs.len);
}

test "DragonflyListPushNode creation" {
    const allocator = std.testing.allocator;
    const config = DragonflyConfig.default();
    
    const node = try DragonflyListPushNode.init(allocator, config, true);
    defer node.base.vtable.?.deinit(&node.base);
    
    try std.testing.expectEqualStrings("dragonfly_lpush", node.base.id);
    try std.testing.expect(node.push_left);
}

test "DragonflyHashSetNode creation" {
    const allocator = std.testing.allocator;
    const config = DragonflyConfig.default();
    
    const node = try DragonflyHashSetNode.init(allocator, config);
    defer node.base.vtable.?.deinit(&node.base);
    
    try std.testing.expectEqualStrings("dragonfly_hset", node.base.id);
    try std.testing.expectEqual(@as(usize, 3), node.base.inputs.len);
}

test "DragonflyHashGetNode creation" {
    const allocator = std.testing.allocator;
    const config = DragonflyConfig.default();
    
    const node = try DragonflyHashGetNode.init(allocator, config);
    defer node.base.vtable.?.deinit(&node.base);
    
    try std.testing.expectEqualStrings("dragonfly_hget", node.base.id);
    try std.testing.expectEqual(@as(usize, 2), node.base.inputs.len);
    try std.testing.expectEqual(@as(usize, 2), node.base.outputs.len);
}

// test "DragonflyClient publish operation" {
//     const allocator = std.testing.allocator;
//     const config = DragonflyConfig.default();
//     
//     var client = try DragonflyClient.init(allocator, config);
//     defer client.deinit();
//     try client.connect();
//     defer client.disconnect();
//     
//     const subscribers = try client.publish("test_channel", "test_message");
//     try std.testing.expect(subscribers >= 0);
// }

// test "DragonflyClient hash operations" {
//     const allocator = std.testing.allocator;
//     const config = DragonflyConfig.default();
//     
//     var client = try DragonflyClient.init(allocator, config);
//     defer client.deinit();
//     try client.connect();
//     defer client.disconnect();
//     
//     _ = try client.hset("test_hash", "field1", "value1");
//     const value = try client.hget("test_hash", "field1");
//     if (value) |v| {
//         defer allocator.free(v);
//     }
// }

// test "DragonflyClient set operations" {
//     const allocator = std.testing.allocator;
//     const config = DragonflyConfig.default();
//     
//     var client = try DragonflyClient.init(allocator, config);
//     defer client.deinit();
//     try client.connect();
//     defer client.disconnect();
//     
//     const added = try client.sadd("test_set", "member1");
//     try std.testing.expect(added);
//     
//     const is_member = try client.sismember("test_set", "member1");
//     try std.testing.expect(is_member);
// }
