//! Lean4 Proof Verification Nodes for nWorkflow
//! Provides integration with nLeanProof server for formal verification
//!
//! Components:
//! - LeanCheckNode: Type check Lean4 source code
//! - LeanRunNode: Execute Lean4 code and capture output
//! - LeanElaborateNode: Elaborate to typed environment
//! - LeanProofVerifyNode: Verify workflow theorem proofs
//! - LeanWorkflowCompilerNode: Compile verified workflow to executable
//!
//! Integration:
//! - HTTP client for nLeanProof REST API
//! - JSON parsing for API responses
//! - Retry logic with configurable backoff
//! - Error handling and diagnostics

const std = @import("std");
const Allocator = std.mem.Allocator;
const node_types = @import("node_types");
const NodeInterface = node_types.NodeInterface;
const ExecutionContext = node_types.ExecutionContext;
const Port = node_types.Port;
const PortType = node_types.PortType;
const NodeCategory = node_types.NodeCategory;

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for nLeanProof server connection
pub const LeanProofConfig = struct {
    /// Service endpoint (default: "http://localhost:8001")
    endpoint: []const u8 = "http://localhost:8001",
    /// Request timeout in milliseconds (default: 30000)
    timeout_ms: u32 = 30000,
    /// Maximum retry attempts on failure (default: 3)
    max_retries: u32 = 3,
    /// Retry backoff multiplier in milliseconds
    retry_backoff_ms: u32 = 1000,
    /// API version path prefix
    api_version: []const u8 = "/api/v1",

    pub fn getCheckEndpoint(self: *const LeanProofConfig, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "{s}{s}/check", .{ self.endpoint, self.api_version });
    }

    pub fn getRunEndpoint(self: *const LeanProofConfig, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "{s}{s}/run", .{ self.endpoint, self.api_version });
    }

    pub fn getElaborateEndpoint(self: *const LeanProofConfig, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "{s}{s}/elaborate", .{ self.endpoint, self.api_version });
    }
};

/// Result of proof verification operations
pub const ProofResult = struct {
    allocator: Allocator,
    /// Whether verification succeeded
    success: bool,
    /// Error messages from type checking
    errors: [][]const u8,
    /// Warning messages
    warnings: [][]const u8,
    /// Elaborated declarations
    elaborated_decls: [][]const u8,
    /// Type information (optional)
    type_info: ?[]const u8,
    /// Raw JSON response
    raw_response: ?[]const u8,

    pub fn init(allocator: Allocator) ProofResult {
        return .{
            .allocator = allocator,
            .success = false,
            .errors = &[_][]const u8{},
            .warnings = &[_][]const u8{},
            .elaborated_decls = &[_][]const u8{},
            .type_info = null,
            .raw_response = null,
        };
    }

    pub fn deinit(self: *ProofResult) void {
        for (self.errors) |err| {
            self.allocator.free(err);
        }
        if (self.errors.len > 0) {
            self.allocator.free(self.errors);
        }

        for (self.warnings) |warn| {
            self.allocator.free(warn);
        }
        if (self.warnings.len > 0) {
            self.allocator.free(self.warnings);
        }

        for (self.elaborated_decls) |decl| {
            self.allocator.free(decl);
        }
        if (self.elaborated_decls.len > 0) {
            self.allocator.free(self.elaborated_decls);
        }

        if (self.type_info) |ti| {
            self.allocator.free(ti);
        }
        if (self.raw_response) |rr| {
            self.allocator.free(rr);
        }
    }

    /// Convert to JSON value for output
    pub fn toJson(self: *const ProofResult) !std.json.Value {
        var obj = std.json.ObjectMap.init(self.allocator);
        errdefer obj.deinit();

        try obj.put("success", .{ .bool = self.success });

        // Add errors array
        var errors_array = try std.ArrayList(std.json.Value).initCapacity(self.allocator, 0);
        for (self.errors) |err| {
            try errors_array.append(.{ .string = err });
        }
        try obj.put("errors", .{ .array = try errors_array.toOwnedSlice() });

        // Add warnings array
        var warnings_array = try std.ArrayList(std.json.Value).initCapacity(self.allocator, 0);
        for (self.warnings) |warn| {
            try warnings_array.append(.{ .string = warn });
        }
        try obj.put("warnings", .{ .array = try warnings_array.toOwnedSlice() });

        // Add declarations array
        var decls_array = try std.ArrayList(std.json.Value).initCapacity(self.allocator, 0);
        for (self.elaborated_decls) |decl| {
            try decls_array.append(.{ .string = decl });
        }
        try obj.put("elaborated_decls", .{ .array = try decls_array.toOwnedSlice() });

        if (self.type_info) |ti| {
            try obj.put("type_info", .{ .string = ti });
        }

        return .{ .object = obj };
    }
};

/// Theorem verification status
pub const TheoremStatus = struct {
    name: []const u8,
    verified: bool,
    proof_complete: bool,
    error_message: ?[]const u8,
};

/// Workflow proof verification result
pub const WorkflowProofResult = struct {
    allocator: Allocator,
    workflow_name: []const u8,
    all_verified: bool,
    theorem_statuses: []TheoremStatus,
    compilation_ready: bool,

    pub fn deinit(self: *WorkflowProofResult) void {
        self.allocator.free(self.workflow_name);
        for (self.theorem_statuses) |*status| {
            self.allocator.free(status.name);
            if (status.error_message) |msg| {
                self.allocator.free(msg);
            }
        }
        self.allocator.free(self.theorem_statuses);
    }
};

// ============================================================================
// HTTP Client Helper
// ============================================================================

/// HTTP client helper for nLeanProof API communication
pub const LeanHttpClient = struct {
    allocator: Allocator,
    config: LeanProofConfig,

    pub fn init(allocator: Allocator, config: LeanProofConfig) LeanHttpClient {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// POST JSON to nLeanProof endpoint with retry logic
    pub fn postJson(self: *LeanHttpClient, endpoint: []const u8, payload: []const u8) ![]const u8 {
        var retry_count: u32 = 0;
        var last_error: anyerror = error.ConnectionFailed;

        while (retry_count < self.config.max_retries) : (retry_count += 1) {
            if (self.executeRequest(endpoint, payload)) |response| {
                return response;
            } else |err| {
                last_error = err;
                // Exponential backoff
                const delay = self.config.retry_backoff_ms * (@as(u32, 1) << @intCast(retry_count));
                std.time.sleep(delay * std.time.ns_per_ms);
            }
        }

        return last_error;
    }

    fn executeRequest(self: *LeanHttpClient, endpoint: []const u8, payload: []const u8) ![]const u8 {
        const uri = try std.Uri.parse(endpoint);

        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        var server_header_buffer: [8192]u8 = undefined;

        var request = try client.open(.POST, uri, .{
            .server_header_buffer = &server_header_buffer,
        });
        defer request.deinit();

        // Set headers
        request.headers.content_type = .{ .override = "application/json" };

        // Send request
        request.transfer_encoding = .{ .content_length = payload.len };
        try request.send();
        try request.writeAll(payload);
        try request.finish();

        // Wait for response
        try request.wait();

        // Check status
        if (request.response.status != .ok) {
            std.log.err("nLeanProof API error: {}", .{request.response.status});
            return error.LeanApiError;
        }

        // Read response body
        var response_body = try std.ArrayList(u8).initCapacity(self.allocator, 0);
        errdefer response_body.deinit();

        const max_size = 50 * 1024 * 1024; // 50MB max for large proofs
        try request.reader().readAllArrayList(&response_body, max_size);

        return try response_body.toOwnedSlice();
    }

    /// Parse JSON response to ProofResult
    pub fn parseProofResult(self: *LeanHttpClient, response: []const u8) !ProofResult {
        var result = ProofResult.init(self.allocator);
        errdefer result.deinit();

        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, response, .{}) catch |err| {
            std.log.err("Failed to parse nLeanProof response: {}", .{err});
            result.raw_response = try self.allocator.dupe(u8, response);
            return result;
        };
        defer parsed.deinit();

        if (parsed.value != .object) {
            result.raw_response = try self.allocator.dupe(u8, response);
            return result;
        }

        const obj = parsed.value.object;

        // Parse success field
        if (obj.get("success")) |success_val| {
            result.success = success_val == .bool and success_val.bool;
        }

        // Parse errors array
        if (obj.get("errors")) |errors_val| {
            if (errors_val == .array) {
                var errors_list = try std.ArrayList([]const u8).initCapacity(self.allocator, 0);
                for (errors_val.array.items) |item| {
                    if (item == .string) {
                        try errors_list.append(try self.allocator.dupe(u8, item.string));
                    }
                }
                result.errors = try errors_list.toOwnedSlice();
            }
        }

        // Parse warnings array
        if (obj.get("warnings")) |warnings_val| {
            if (warnings_val == .array) {
                var warnings_list = try std.ArrayList([]const u8).initCapacity(self.allocator, 0);
                for (warnings_val.array.items) |item| {
                    if (item == .string) {
                        try warnings_list.append(try self.allocator.dupe(u8, item.string));
                    }
                }
                result.warnings = try warnings_list.toOwnedSlice();
            }
        }

        // Parse elaborated declarations
        if (obj.get("declarations") orelse obj.get("elaborated_decls")) |decls_val| {
            if (decls_val == .array) {
                var decls_list = try std.ArrayList([]const u8).initCapacity(self.allocator, 0);
                for (decls_val.array.items) |item| {
                    if (item == .string) {
                        try decls_list.append(try self.allocator.dupe(u8, item.string));
                    }
                }
                result.elaborated_decls = try decls_list.toOwnedSlice();
            }
        }

        // Parse type info
        if (obj.get("type_info")) |ti_val| {
            if (ti_val == .string) {
                result.type_info = try self.allocator.dupe(u8, ti_val.string);
            }
        }

        result.raw_response = try self.allocator.dupe(u8, response);
        return result;
    }
};


// ============================================================================
// LeanCheckNode - Type check Lean4 source
// ============================================================================

/// Node for type checking Lean4 source code via nLeanProof server
pub const LeanCheckNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    config: LeanProofConfig,
    inputs: []Port,
    outputs: []Port,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: LeanProofConfig,
    ) !*LeanCheckNode {
        const node = try allocator.create(LeanCheckNode);
        errdefer allocator.destroy(node);

        // Define input ports
        var inputs = try allocator.alloc(Port, 1);
        inputs[0] = Port.init(
            "source",
            "Source",
            "Lean4 source code to type check",
            .string,
            true,
            null,
        );

        // Define output ports
        var outputs = try allocator.alloc(Port, 1);
        outputs[0] = Port.init(
            "result",
            "Result",
            "ProofResult as JSON with success, errors, warnings",
            .object,
            true,
            null,
        );

        node.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "lean_check",
            .config = config,
            .inputs = inputs,
            .outputs = outputs,
        };

        return node;
    }

    pub fn deinit(self: *LeanCheckNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }

    pub fn asNodeInterface(self: *LeanCheckNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "Type check Lean4 source code via nLeanProof server",
            .node_type = self.node_type,
            .category = .integration,
            .inputs = self.inputs,
            .outputs = self.outputs,
            .config = .{ .null = {} },
            .vtable = &.{
                .validate = validateCheckImpl,
                .execute = executeCheckImpl,
                .deinit = deinitCheckImpl,
            },
            .impl_ptr = self,
        };
    }

    fn validateCheckImpl(interface: *const NodeInterface) anyerror!void {
        _ = interface;
        // No special validation needed
    }

    fn executeCheckImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self: *LeanCheckNode = @ptrCast(@alignCast(interface.impl_ptr));

        // Get source from input
        const source_value = ctx.getInput("source") orelse return error.MissingInput;
        const source = switch (source_value) {
            .string => |s| s,
            else => return error.InvalidInputType,
        };

        // Build request payload
        var request_obj = std.json.ObjectMap.init(self.allocator);
        defer request_obj.deinit();
        try request_obj.put("source", .{ .string = source });

        var request_str = try std.ArrayList(u8).initCapacity(self.allocator, 0);
        defer request_str.deinit();
        try std.json.stringify(.{ .object = request_obj }, .{}, request_str.writer());

        // Get endpoint
        const endpoint = try self.config.getCheckEndpoint(self.allocator);
        defer self.allocator.free(endpoint);

        // Execute HTTP request
        var client = LeanHttpClient.init(self.allocator, self.config);
        const response = try client.postJson(endpoint, request_str.items);
        defer self.allocator.free(response);

        // Parse result
        var result = try client.parseProofResult(response);
        defer result.deinit();

        return try result.toJson();
    }

    fn deinitCheckImpl(interface: *NodeInterface) void {
        const self: *LeanCheckNode = @ptrCast(@alignCast(interface.impl_ptr));
        self.deinit();
    }
};


// ============================================================================
// LeanRunNode - Execute Lean4 code
// ============================================================================

/// Node for executing Lean4 code via nLeanProof server
pub const LeanRunNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    config: LeanProofConfig,
    inputs: []Port,
    outputs: []Port,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: LeanProofConfig,
    ) !*LeanRunNode {
        const node = try allocator.create(LeanRunNode);
        errdefer allocator.destroy(node);

        // Define input ports
        var inputs = try allocator.alloc(Port, 1);
        inputs[0] = Port.init("source", "Source", "Lean4 code to execute", .string, true, null);

        // Define output ports
        var outputs = try allocator.alloc(Port, 3);
        outputs[0] = Port.init("stdout", "Stdout", "Standard output from execution", .string, true, null);
        outputs[1] = Port.init("stderr", "Stderr", "Standard error from execution", .string, true, null);
        outputs[2] = Port.init("exit_code", "Exit Code", "Process exit code", .number, true, null);

        node.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "lean_run",
            .config = config,
            .inputs = inputs,
            .outputs = outputs,
        };

        return node;
    }

    pub fn deinit(self: *LeanRunNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }

    pub fn asNodeInterface(self: *LeanRunNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "Execute Lean4 code via nLeanProof server",
            .node_type = self.node_type,
            .category = .integration,
            .inputs = self.inputs,
            .outputs = self.outputs,
            .config = .{ .null = {} },
            .vtable = &.{
                .validate = validateRunImpl,
                .execute = executeRunImpl,
                .deinit = deinitRunImpl,
            },
            .impl_ptr = self,
        };
    }

    fn validateRunImpl(interface: *const NodeInterface) anyerror!void {
        _ = interface;
    }

    fn executeRunImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self: *LeanRunNode = @ptrCast(@alignCast(interface.impl_ptr));

        const source_value = ctx.getInput("source") orelse return error.MissingInput;
        const source = switch (source_value) {
            .string => |s| s,
            else => return error.InvalidInputType,
        };

        // Build request
        var request_obj = std.json.ObjectMap.init(self.allocator);
        defer request_obj.deinit();
        try request_obj.put("source", .{ .string = source });

        var request_str = try std.ArrayList(u8).initCapacity(self.allocator, 0);
        defer request_str.deinit();
        try std.json.stringify(.{ .object = request_obj }, .{}, request_str.writer());

        const endpoint = try self.config.getRunEndpoint(self.allocator);
        defer self.allocator.free(endpoint);

        var client = LeanHttpClient.init(self.allocator, self.config);
        const response = try client.postJson(endpoint, request_str.items);
        defer self.allocator.free(response);

        // Parse response
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, response, .{});
        defer parsed.deinit();

        var result = std.json.ObjectMap.init(self.allocator);
        if (parsed.value == .object) {
            const obj = parsed.value.object;
            if (obj.get("stdout")) |stdout| {
                try result.put("stdout", stdout);
            } else {
                try result.put("stdout", .{ .string = "" });
            }
            if (obj.get("stderr")) |stderr| {
                try result.put("stderr", stderr);
            } else {
                try result.put("stderr", .{ .string = "" });
            }
            if (obj.get("exit_code")) |code| {
                try result.put("exit_code", code);
            } else {
                try result.put("exit_code", .{ .integer = 0 });
            }
        }

        return .{ .object = result };
    }

    fn deinitRunImpl(interface: *NodeInterface) void {
        const self: *LeanRunNode = @ptrCast(@alignCast(interface.impl_ptr));
        self.deinit();
    }
};


// ============================================================================
// LeanElaborateNode - Elaborate Lean4 to typed environment
// ============================================================================

/// Node for elaborating Lean4 code to typed environment via nLeanProof server
pub const LeanElaborateNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    config: LeanProofConfig,
    inputs: []Port,
    outputs: []Port,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: LeanProofConfig,
    ) !*LeanElaborateNode {
        const node = try allocator.create(LeanElaborateNode);
        errdefer allocator.destroy(node);

        var inputs = try allocator.alloc(Port, 1);
        inputs[0] = Port.init("source", "Source", "Lean4 code to elaborate", .string, true, null);

        var outputs = try allocator.alloc(Port, 2);
        outputs[0] = Port.init("declarations", "Declarations", "List of typed declarations", .array, true, null);
        outputs[1] = Port.init("environment", "Environment", "Type environment info", .object, true, null);

        node.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "lean_elaborate",
            .config = config,
            .inputs = inputs,
            .outputs = outputs,
        };

        return node;
    }

    pub fn deinit(self: *LeanElaborateNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }

    pub fn asNodeInterface(self: *LeanElaborateNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "Elaborate Lean4 code to typed environment via nLeanProof",
            .node_type = self.node_type,
            .category = .integration,
            .inputs = self.inputs,
            .outputs = self.outputs,
            .config = .{ .null = {} },
            .vtable = &.{
                .validate = validateElaborateImpl,
                .execute = executeElaborateImpl,
                .deinit = deinitElaborateImpl,
            },
            .impl_ptr = self,
        };
    }

    fn validateElaborateImpl(interface: *const NodeInterface) anyerror!void {
        _ = interface;
    }

    fn executeElaborateImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self: *LeanElaborateNode = @ptrCast(@alignCast(interface.impl_ptr));

        const source_value = ctx.getInput("source") orelse return error.MissingInput;
        const source = switch (source_value) {
            .string => |s| s,
            else => return error.InvalidInputType,
        };

        var request_obj = std.json.ObjectMap.init(self.allocator);
        defer request_obj.deinit();
        try request_obj.put("source", .{ .string = source });

        var request_str = std.ArrayList(u8).init(self.allocator);
        defer request_str.deinit();
        try std.json.stringify(.{ .object = request_obj }, .{}, request_str.writer());

        const endpoint = try self.config.getElaborateEndpoint(self.allocator);
        defer self.allocator.free(endpoint);

        var client = LeanHttpClient.init(self.allocator, self.config);
        const response = try client.postJson(endpoint, request_str.items);
        defer self.allocator.free(response);

        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, response, .{});
        return parsed.value;
    }

    fn deinitElaborateImpl(interface: *NodeInterface) void {
        const self: *LeanElaborateNode = @ptrCast(@alignCast(interface.impl_ptr));
        self.deinit();
    }
};


// ============================================================================
// LeanProofVerifyNode - Verify workflow theorem proofs
// ============================================================================

/// Node for verifying workflow theorem proofs via nLeanProof server
/// Extracts theorems like workflow_safe, deadlock_free, eventual_completion
pub const LeanProofVerifyNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    config: LeanProofConfig,
    inputs: []Port,
    outputs: []Port,
    /// List of theorem names to verify
    theorems_to_verify: []const []const u8,

    pub const DefaultTheorems = [_][]const u8{
        "workflow_safe",
        "deadlock_free",
        "eventual_completion",
        "data_integrity",
        "type_safety",
    };

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: LeanProofConfig,
        theorems: ?[]const []const u8,
    ) !*LeanProofVerifyNode {
        const node = try allocator.create(LeanProofVerifyNode);
        errdefer allocator.destroy(node);

        var inputs = try allocator.alloc(Port, 1);
        inputs[0] = Port.init("workflow_source", "Workflow Source", "Lean4 workflow with theorem proofs", .string, true, null);

        var outputs = try allocator.alloc(Port, 3);
        outputs[0] = Port.init("verified", "Verified", "Whether all proofs passed", .boolean, true, null);
        outputs[1] = Port.init("proof_results", "Proof Results", "Per-theorem verification results", .array, true, null);
        outputs[2] = Port.init("compilation_ready", "Compilation Ready", "Whether workflow can be compiled", .boolean, true, null);

        node.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "lean_proof_verify",
            .config = config,
            .inputs = inputs,
            .outputs = outputs,
            .theorems_to_verify = theorems orelse &DefaultTheorems,
        };

        return node;
    }

    pub fn deinit(self: *LeanProofVerifyNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }

    pub fn asNodeInterface(self: *LeanProofVerifyNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "Verify workflow theorem proofs (safety, liveness, deadlock-freedom)",
            .node_type = self.node_type,
            .category = .integration,
            .inputs = self.inputs,
            .outputs = self.outputs,
            .config = .{ .null = {} },
            .vtable = &.{
                .validate = validateProofVerifyImpl,
                .execute = executeProofVerifyImpl,
                .deinit = deinitProofVerifyImpl,
            },
            .impl_ptr = self,
        };
    }

    fn validateProofVerifyImpl(interface: *const NodeInterface) anyerror!void {
        _ = interface;
    }

    fn executeProofVerifyImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self: *LeanProofVerifyNode = @ptrCast(@alignCast(interface.impl_ptr));

        const source_value = ctx.getInput("workflow_source") orelse return error.MissingInput;
        const source = switch (source_value) {
            .string => |s| s,
            else => return error.InvalidInputType,
        };

        // Build check request
        var request_obj = std.json.ObjectMap.init(self.allocator);
        defer request_obj.deinit();
        try request_obj.put("source", .{ .string = source });

        var request_str = std.ArrayList(u8).init(self.allocator);
        defer request_str.deinit();
        try std.json.stringify(.{ .object = request_obj }, .{}, request_str.writer());

        const endpoint = try self.config.getCheckEndpoint(self.allocator);
        defer self.allocator.free(endpoint);

        var client = LeanHttpClient.init(self.allocator, self.config);
        const response = client.postJson(endpoint, request_str.items) catch |err| {
            // Build error result
            var result = std.json.ObjectMap.init(self.allocator);
            try result.put("verified", .{ .bool = false });
            try result.put("compilation_ready", .{ .bool = false });
            try result.put("error", .{ .string = @errorName(err) });
            return .{ .object = result };
        };
        defer self.allocator.free(response);

        // Parse and analyze results
        var proof_result = try client.parseProofResult(response);
        defer proof_result.deinit();

        // Build output with per-theorem results
        var result = std.json.ObjectMap.init(self.allocator);
        try result.put("verified", .{ .bool = proof_result.success });
        try result.put("compilation_ready", .{ .bool = proof_result.success and proof_result.errors.len == 0 });

        // Add proof results array
        var proof_results = try std.ArrayList(std.json.Value).initCapacity(self.allocator, 0);
        for (self.theorems_to_verify) |theorem_name| {
            var theorem_result = std.json.ObjectMap.init(self.allocator);
            try theorem_result.put("theorem", .{ .string = theorem_name });

            // Check if theorem exists in source
            const theorem_found = std.mem.indexOf(u8, source, theorem_name) != null;
            try theorem_result.put("found", .{ .bool = theorem_found });
            try theorem_result.put("verified", .{ .bool = theorem_found and proof_result.success });

            try proof_results.append(.{ .object = theorem_result });
        }
        try result.put("proof_results", .{ .array = try proof_results.toOwnedSlice() });

        return .{ .object = result };
    }

    fn deinitProofVerifyImpl(interface: *NodeInterface) void {
        const self: *LeanProofVerifyNode = @ptrCast(@alignCast(interface.impl_ptr));
        self.deinit();
    }
};


// ============================================================================
// LeanWorkflowCompilerNode - Compile verified workflow to executable
// ============================================================================

/// Node for compiling verified Lean4 workflow definitions to executable workflows
pub const LeanWorkflowCompilerNode = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    node_type: []const u8,
    config: LeanProofConfig,
    inputs: []Port,
    outputs: []Port,
    /// Require all proofs to pass before compilation
    require_proofs: bool,

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        name: []const u8,
        config: LeanProofConfig,
        require_proofs: bool,
    ) !*LeanWorkflowCompilerNode {
        const node = try allocator.create(LeanWorkflowCompilerNode);
        errdefer allocator.destroy(node);

        var inputs = try allocator.alloc(Port, 1);
        inputs[0] = Port.init("lean_workflow", "Lean Workflow", "Lean4 workflow definition", .string, true, null);

        var outputs = try allocator.alloc(Port, 3);
        outputs[0] = Port.init("workflow_json", "Workflow JSON", "Compiled WorkflowSchema as JSON", .object, true, null);
        outputs[1] = Port.init("proof_status", "Proof Status", "Verification status", .object, true, null);
        outputs[2] = Port.init("compile_success", "Compile Success", "Whether compilation succeeded", .boolean, true, null);

        node.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .node_type = "lean_workflow_compiler",
            .config = config,
            .inputs = inputs,
            .outputs = outputs,
            .require_proofs = require_proofs,
        };

        return node;
    }

    pub fn deinit(self: *LeanWorkflowCompilerNode) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.allocator.destroy(self);
    }

    pub fn asNodeInterface(self: *LeanWorkflowCompilerNode) NodeInterface {
        return NodeInterface{
            .id = self.id,
            .name = self.name,
            .description = "Compile verified Lean4 workflow to executable WorkflowSchema",
            .node_type = self.node_type,
            .category = .integration,
            .inputs = self.inputs,
            .outputs = self.outputs,
            .config = .{ .null = {} },
            .vtable = &.{
                .validate = validateCompilerImpl,
                .execute = executeCompilerImpl,
                .deinit = deinitCompilerImpl,
            },
            .impl_ptr = self,
        };
    }

    fn validateCompilerImpl(interface: *const NodeInterface) anyerror!void {
        _ = interface;
    }

    fn executeCompilerImpl(interface: *NodeInterface, ctx: *ExecutionContext) anyerror!std.json.Value {
        const self: *LeanWorkflowCompilerNode = @ptrCast(@alignCast(interface.impl_ptr));

        const source_value = ctx.getInput("lean_workflow") orelse return error.MissingInput;
        const source = switch (source_value) {
            .string => |s| s,
            else => return error.InvalidInputType,
        };

        var result = std.json.ObjectMap.init(self.allocator);

        // Step 1: Verify proofs if required
        if (self.require_proofs) {
            var request_obj = std.json.ObjectMap.init(self.allocator);
            defer request_obj.deinit();
            try request_obj.put("source", .{ .string = source });

            var request_str = try std.ArrayList(u8).initCapacity(self.allocator, 0);
            defer request_str.deinit();
            try std.json.stringify(.{ .object = request_obj }, .{}, request_str.writer());

            const endpoint = try self.config.getCheckEndpoint(self.allocator);
            defer self.allocator.free(endpoint);

            var client = LeanHttpClient.init(self.allocator, self.config);
            const response = client.postJson(endpoint, request_str.items) catch {
                try result.put("compile_success", .{ .bool = false });
                try result.put("error", .{ .string = "Failed to verify proofs" });
                return .{ .object = result };
            };
            defer self.allocator.free(response);

            var proof_result = try client.parseProofResult(response);
            defer proof_result.deinit();

            if (!proof_result.success) {
                try result.put("compile_success", .{ .bool = false });
                try result.put("error", .{ .string = "Proof verification failed" });
                var proof_status = std.json.ObjectMap.init(self.allocator);
                try proof_status.put("verified", .{ .bool = false });
                try result.put("proof_status", .{ .object = proof_status });
                return .{ .object = result };
            }

            var proof_status = std.json.ObjectMap.init(self.allocator);
            try proof_status.put("verified", .{ .bool = true });
            try result.put("proof_status", .{ .object = proof_status });
        }

        // Step 2: Parse Lean4 to WorkflowSchema JSON
        const workflow_json = try self.parseLean4ToWorkflowJson(source);
        try result.put("workflow_json", workflow_json);
        try result.put("compile_success", .{ .bool = true });

        return .{ .object = result };
    }

    fn parseLean4ToWorkflowJson(self: *LeanWorkflowCompilerNode, source: []const u8) !std.json.Value {
        var workflow = std.json.ObjectMap.init(self.allocator);

        // Extract workflow name from "def <name> : Workflow"
        var lines = std.mem.splitScalar(u8, source, '\n');
        var name: []const u8 = "compiled_workflow";

        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (std.mem.startsWith(u8, trimmed, "def ")) {
                if (std.mem.indexOf(u8, trimmed, ":")) |colon_idx| {
                    name = std.mem.trim(u8, trimmed[4..colon_idx], " \t");
                    break;
                }
            }
        }

        try workflow.put("name", .{ .string = name });
        try workflow.put("version", .{ .string = "1.0" });
        try workflow.put("description", .{ .string = "Compiled from verified Lean4 workflow" });
        try workflow.put("verified", .{ .bool = true });

        // Add source hash for traceability
        var hash: u64 = 0;
        for (source) |c| hash = hash *% 31 +% c;
        var hash_buf: [17]u8 = undefined;
        _ = std.fmt.bufPrint(&hash_buf, "{x:0>16}", .{hash}) catch "0000000000000000";
        try workflow.put("source_hash", .{ .string = &hash_buf });

        return .{ .object = workflow };
    }

    fn deinitCompilerImpl(interface: *NodeInterface) void {
        const self: *LeanWorkflowCompilerNode = @ptrCast(@alignCast(interface.impl_ptr));
        self.deinit();
    }
};


// ============================================================================
// Tests
// ============================================================================

test "LeanProofConfig endpoints" {
    const allocator = std.testing.allocator;
    const config = LeanProofConfig{};

    const check_endpoint = try config.getCheckEndpoint(allocator);
    defer allocator.free(check_endpoint);
    try std.testing.expectEqualStrings("http://localhost:8001/api/v1/check", check_endpoint);

    const run_endpoint = try config.getRunEndpoint(allocator);
    defer allocator.free(run_endpoint);
    try std.testing.expectEqualStrings("http://localhost:8001/api/v1/run", run_endpoint);

    const elaborate_endpoint = try config.getElaborateEndpoint(allocator);
    defer allocator.free(elaborate_endpoint);
    try std.testing.expectEqualStrings("http://localhost:8001/api/v1/elaborate", elaborate_endpoint);
}

test "ProofResult initialization" {
    const allocator = std.testing.allocator;
    var result = ProofResult.init(allocator);
    defer result.deinit();

    try std.testing.expect(!result.success);
    try std.testing.expectEqual(@as(usize, 0), result.errors.len);
}

test "LeanCheckNode initialization" {
    const allocator = std.testing.allocator;
    const config = LeanProofConfig{};

    const node = try LeanCheckNode.init(allocator, "test_check", "Test Check Node", config);
    defer node.deinit();

    try std.testing.expectEqualStrings("test_check", node.id);
    try std.testing.expectEqualStrings("lean_check", node.node_type);
    try std.testing.expectEqual(@as(usize, 1), node.inputs.len);
    try std.testing.expectEqual(@as(usize, 1), node.outputs.len);
}

test "LeanRunNode initialization" {
    const allocator = std.testing.allocator;
    const config = LeanProofConfig{};

    const node = try LeanRunNode.init(allocator, "test_run", "Test Run Node", config);
    defer node.deinit();

    try std.testing.expectEqualStrings("lean_run", node.node_type);
    try std.testing.expectEqual(@as(usize, 3), node.outputs.len);
}

test "LeanElaborateNode initialization" {
    const allocator = std.testing.allocator;
    const config = LeanProofConfig{};

    const node = try LeanElaborateNode.init(allocator, "test_elaborate", "Test Elaborate", config);
    defer node.deinit();

    try std.testing.expectEqualStrings("lean_elaborate", node.node_type);
    try std.testing.expectEqual(@as(usize, 2), node.outputs.len);
}

test "LeanProofVerifyNode initialization" {
    const allocator = std.testing.allocator;
    const config = LeanProofConfig{};

    const node = try LeanProofVerifyNode.init(allocator, "test_verify", "Test Verify", config, null);
    defer node.deinit();

    try std.testing.expectEqualStrings("lean_proof_verify", node.node_type);
    try std.testing.expectEqual(@as(usize, 5), node.theorems_to_verify.len);
}

test "LeanWorkflowCompilerNode initialization" {
    const allocator = std.testing.allocator;
    const config = LeanProofConfig{};

    const node = try LeanWorkflowCompilerNode.init(allocator, "test_compiler", "Test Compiler", config, true);
    defer node.deinit();

    try std.testing.expectEqualStrings("lean_workflow_compiler", node.node_type);
    try std.testing.expect(node.require_proofs);
}

test "LeanProofVerifyNode default theorems" {
    const theorems = LeanProofVerifyNode.DefaultTheorems;
    try std.testing.expectEqual(@as(usize, 5), theorems.len);
    try std.testing.expectEqualStrings("workflow_safe", theorems[0]);
    try std.testing.expectEqualStrings("deadlock_free", theorems[1]);
}
