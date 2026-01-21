//! Node-level error handling integration
//! 
//! Integrates error recovery with workflow nodes, providing:
//! - Node-specific error handling
//! - Error propagation through workflow
//! - Retry logic for node execution
//! - Error state persistence
//!
//! Day 23: Error Recovery & Retry Mechanisms - Node Integration

const std = @import("std");
const Allocator = std.mem.Allocator;
const error_recovery = @import("error_recovery.zig");

const ErrorRecoveryManager = error_recovery.ErrorRecoveryManager;
const ErrorContext = error_recovery.ErrorContext;
const RecoveryConfig = error_recovery.RecoveryConfig;
const ErrorCategory = error_recovery.ErrorCategory;
const ErrorSeverity = error_recovery.ErrorSeverity;

/// Node execution state
pub const NodeExecutionState = enum {
    pending,
    running,
    success,
    failed,
    retrying,
    circuit_open,
};

/// Node error handler configuration
pub const NodeErrorConfig = struct {
    /// Node ID
    node_id: []const u8,
    /// Node type for specialized handling
    node_type: []const u8,
    /// Recovery configuration
    recovery_config: RecoveryConfig,
    /// Whether to continue workflow on error
    continue_on_error: bool,
    /// Error output port (send errors here instead of failing)
    error_output_port: ?[]const u8,

    pub fn default(node_id: []const u8, node_type: []const u8) NodeErrorConfig {
        return NodeErrorConfig{
            .node_id = node_id,
            .node_type = node_type,
            .recovery_config = RecoveryConfig.default(),
            .continue_on_error = false,
            .error_output_port = null,
        };
    }

    pub fn resilient(node_id: []const u8, node_type: []const u8) NodeErrorConfig {
        return NodeErrorConfig{
            .node_id = node_id,
            .node_type = node_type,
            .recovery_config = RecoveryConfig.withCircuitBreaker(),
            .continue_on_error = true,
            .error_output_port = "error",
        };
    }
};

/// Node execution result with error information
pub const NodeExecutionResult = struct {
    state: NodeExecutionState,
    output: ?std.json.Value,
    error_ctx: ?ErrorContext,
    retry_count: u32,
    execution_time_ms: u64,

    pub fn success(output: std.json.Value, execution_time_ms: u64) NodeExecutionResult {
        return NodeExecutionResult{
            .state = .success,
            .output = output,
            .error_ctx = null,
            .retry_count = 0,
            .execution_time_ms = execution_time_ms,
        };
    }

    pub fn failure(error_ctx: ErrorContext, retry_count: u32, execution_time_ms: u64) NodeExecutionResult {
        return NodeExecutionResult{
            .state = .failed,
            .output = null,
            .error_ctx = error_ctx,
            .retry_count = retry_count,
            .execution_time_ms = execution_time_ms,
        };
    }
};

/// Node error handler
pub const NodeErrorHandler = struct {
    allocator: Allocator,
    config: NodeErrorConfig,
    recovery_manager: ErrorRecoveryManager,
    execution_history: std.ArrayList(NodeExecutionResult),
    current_state: NodeExecutionState,

    pub fn init(allocator: Allocator, config: NodeErrorConfig) !NodeErrorHandler {
        return NodeErrorHandler{
            .allocator = allocator,
            .config = config,
            .recovery_manager = try ErrorRecoveryManager.init(allocator, config.recovery_config),
            .execution_history = .{},
            .current_state = .pending,
        };
    }

    pub fn deinit(self: *NodeErrorHandler) void {
        self.recovery_manager.deinit();
        for (self.execution_history.items) |*result| {
            if (result.error_ctx) |*ctx| {
                ctx.deinit();
            }
        }
        self.execution_history.deinit(self.allocator);
    }

    /// Execute node with error handling and retry
    pub fn executeNode(
        self: *NodeErrorHandler,
        execute_fn: *const fn (allocator: Allocator) anyerror!std.json.Value,
    ) !NodeExecutionResult {
        const start_time = std.time.milliTimestamp();
        self.current_state = .running;

        // Execute with retry logic
        const result = self.recovery_manager.executeWithRetry(
            std.json.Value,
            execute_fn,
        ) catch |err| {
            const execution_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));
            
            // Create error context
            var error_ctx = try ErrorContext.init(
                self.allocator,
                @errorName(err),
                self.categorizeError(err),
                self.determineSeverity(err),
            );
            try error_ctx.setNodeId(self.config.node_id);

            const retry_count = self.getRetryCount();
            const exec_result = NodeExecutionResult.failure(error_ctx, retry_count, execution_time);
            
            self.current_state = if (err == error.CircuitOpen) .circuit_open else .failed;
            try self.execution_history.append(self.allocator, exec_result);

            if (self.config.continue_on_error) {
                // Convert error to output on error port
                return exec_result;
            }

            return err;
        };

        const execution_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));
        const exec_result = NodeExecutionResult.success(result, execution_time);
        
        self.current_state = .success;
        try self.execution_history.append(self.allocator, exec_result);

        return exec_result;
    }

    /// Get current execution state
    pub fn getState(self: *const NodeErrorHandler) NodeExecutionState {
        return self.current_state;
    }

    /// Get error statistics
    pub fn getErrorStats(self: *const NodeErrorHandler) error_recovery.ErrorStats {
        return self.recovery_manager.getStats();
    }

    /// Get last N execution results
    pub fn getExecutionHistory(self: *const NodeErrorHandler, n: usize) []const NodeExecutionResult {
        const history_len = self.execution_history.items.len;
        const start_idx = if (history_len > n) history_len - n else 0;
        return self.execution_history.items[start_idx..];
    }

    /// Reset error state
    pub fn reset(self: *NodeErrorHandler) void {
        self.recovery_manager.clearHistory();
        self.current_state = .pending;
    }

    fn getRetryCount(self: *const NodeErrorHandler) u32 {
        var count: u32 = 0;
        for (self.execution_history.items) |result| {
            if (result.state == .failed or result.state == .retrying) {
                count += 1;
            }
        }
        return count;
    }

    fn categorizeError(self: *NodeErrorHandler, err: anyerror) ErrorCategory {
        _ = self;
        
        // Categorize based on error type
        return switch (err) {
            error.ConnectionRefused,
            error.NetworkUnreachable,
            error.ConnectionResetByPeer,
            => .network,

            error.Timeout,
            error.TimedOut,
            => .timeout,

            error.AccessDenied,
            error.PermissionDenied,
            error.Unauthorized,
            => .auth,

            error.OutOfMemory,
            error.SystemResources,
            => .resource,

            error.InvalidCharacter,
            error.InvalidFormat,
            error.BadRequest,
            => .validation,

            error.CircuitOpen,
            => .external_service,

            else => .unknown,
        };
    }

    fn determineSeverity(self: *NodeErrorHandler, err: anyerror) ErrorSeverity {
        _ = self;
        
        return switch (err) {
            error.OutOfMemory,
            error.SystemResources,
            => .fatal,

            error.AccessDenied,
            error.PermissionDenied,
            error.InvalidFormat,
            => .persistent,

            error.ConnectionRefused,
            error.Timeout,
            error.CircuitOpen,
            => .transient,

            else => .persistent,
        };
    }
};

/// Workflow-level error coordinator
pub const WorkflowErrorCoordinator = struct {
    allocator: Allocator,
    workflow_id: []const u8,
    node_handlers: std.StringHashMap(*NodeErrorHandler),
    error_count: u64,
    recovery_count: u64,

    pub fn init(allocator: Allocator, workflow_id: []const u8) !WorkflowErrorCoordinator {
        return WorkflowErrorCoordinator{
            .allocator = allocator,
            .workflow_id = try allocator.dupe(u8, workflow_id),
            .node_handlers = std.StringHashMap(*NodeErrorHandler).init(allocator),
            .error_count = 0,
            .recovery_count = 0,
        };
    }

    pub fn deinit(self: *WorkflowErrorCoordinator) void {
        self.allocator.free(self.workflow_id);
        
        var it = self.node_handlers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.node_handlers.deinit();
    }

    /// Register node error handler
    pub fn registerNode(
        self: *WorkflowErrorCoordinator,
        node_id: []const u8,
        config: NodeErrorConfig,
    ) !void {
        const handler = try self.allocator.create(NodeErrorHandler);
        handler.* = try NodeErrorHandler.init(self.allocator, config);

        const key = try self.allocator.dupe(u8, node_id);
        try self.node_handlers.put(key, handler);
    }

    /// Get node error handler
    pub fn getNodeHandler(self: *WorkflowErrorCoordinator, node_id: []const u8) ?*NodeErrorHandler {
        return self.node_handlers.get(node_id);
    }

    /// Record workflow-level error
    pub fn recordError(self: *WorkflowErrorCoordinator, node_id: []const u8) void {
        _ = node_id;
        self.error_count += 1;
    }

    /// Record workflow-level recovery
    pub fn recordRecovery(self: *WorkflowErrorCoordinator, node_id: []const u8) void {
        _ = node_id;
        self.recovery_count += 1;
    }

    /// Get workflow error statistics
    pub fn getWorkflowStats(self: *const WorkflowErrorCoordinator) WorkflowErrorStats {
        var total_node_errors: u64 = 0;
        var total_node_recoveries: u64 = 0;
        var nodes_in_error_state: u32 = 0;

        var it = self.node_handlers.valueIterator();
        while (it.next()) |handler| {
            const stats = handler.*.getErrorStats();
            total_node_errors += stats.total_errors;
            total_node_recoveries += stats.total_recoveries;

            const state = handler.*.getState();
            if (state == .failed or state == .circuit_open) {
                nodes_in_error_state += 1;
            }
        }

        const recovery_rate = if (self.error_count > 0)
            @as(f64, @floatFromInt(self.recovery_count)) / @as(f64, @floatFromInt(self.error_count))
        else
            1.0;

        return WorkflowErrorStats{
            .workflow_errors = self.error_count,
            .workflow_recoveries = self.recovery_count,
            .recovery_rate = recovery_rate,
            .node_errors = total_node_errors,
            .node_recoveries = total_node_recoveries,
            .nodes_in_error = nodes_in_error_state,
            .total_nodes = @as(u32, @intCast(self.node_handlers.count())),
        };
    }
};

/// Workflow error statistics
pub const WorkflowErrorStats = struct {
    workflow_errors: u64,
    workflow_recoveries: u64,
    recovery_rate: f64,
    node_errors: u64,
    node_recoveries: u64,
    nodes_in_error: u32,
    total_nodes: u32,
};

// Tests
const testing = std.testing;

test "NodeErrorConfig presets" {
    const default_config = NodeErrorConfig.default("node1", "http_request");
    try testing.expectEqualStrings("node1", default_config.node_id);
    try testing.expectEqualStrings("http_request", default_config.node_type);
    try testing.expect(!default_config.continue_on_error);

    const resilient_config = NodeErrorConfig.resilient("node2", "llm_chat");
    try testing.expectEqualStrings("node2", resilient_config.node_id);
    try testing.expect(resilient_config.continue_on_error);
    try testing.expect(resilient_config.error_output_port != null);
}

test "NodeExecutionResult success" {
    const output = std.json.Value{ .string = "success" };
    const result = NodeExecutionResult.success(output, 100);

    try testing.expectEqual(NodeExecutionState.success, result.state);
    try testing.expect(result.output != null);
    try testing.expect(result.error_ctx == null);
    try testing.expectEqual(@as(u32, 0), result.retry_count);
}

test "NodeErrorHandler initialization" {
    const config = NodeErrorConfig.default("test_node", "test");
    var handler = try NodeErrorHandler.init(testing.allocator, config);
    defer handler.deinit();

    try testing.expectEqual(NodeExecutionState.pending, handler.getState());
    
    const stats = handler.getErrorStats();
    try testing.expectEqual(@as(u64, 0), stats.total_errors);
}

test "WorkflowErrorCoordinator node registration" {
    var coordinator = try WorkflowErrorCoordinator.init(testing.allocator, "workflow1");
    defer coordinator.deinit();

    const config = NodeErrorConfig.default("node1", "test");
    try coordinator.registerNode("node1", config);

    const handler = coordinator.getNodeHandler("node1");
    try testing.expect(handler != null);
    try testing.expectEqual(NodeExecutionState.pending, handler.?.getState());
}

test "WorkflowErrorCoordinator statistics" {
    var coordinator = try WorkflowErrorCoordinator.init(testing.allocator, "workflow1");
    defer coordinator.deinit();

    coordinator.recordError("node1");
    coordinator.recordError("node2");
    coordinator.recordRecovery("node1");

    const stats = coordinator.getWorkflowStats();
    try testing.expectEqual(@as(u64, 2), stats.workflow_errors);
    try testing.expectEqual(@as(u64, 1), stats.workflow_recoveries);
}

test "NodeErrorHandler execution history" {
    const config = NodeErrorConfig.default("test_node", "test");
    var handler = try NodeErrorHandler.init(testing.allocator, config);
    defer handler.deinit();

    const result1 = NodeExecutionResult.success(std.json.Value{ .string = "ok" }, 50);
    try handler.execution_history.append(testing.allocator, result1);

    const history = handler.getExecutionHistory(1);
    try testing.expectEqual(@as(usize, 1), history.len);
    try testing.expectEqual(NodeExecutionState.success, history[0].state);
}
