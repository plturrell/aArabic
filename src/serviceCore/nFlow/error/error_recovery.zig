//! Error Recovery System for nWorkflow
//! 
//! Provides comprehensive error recovery mechanisms including:
//! - Retry policies with backoff strategies
//! - Error state management
//! - Error propagation and handling
//! - Circuit breaker pattern
//! - Fallback strategies
//!
//! Day 23: Error Recovery & Retry Mechanisms

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Error recovery strategy types
pub const RecoveryStrategy = enum {
    /// Retry with exponential backoff
    exponential_backoff,
    /// Retry with fixed delay
    fixed_delay,
    /// Retry with linear backoff
    linear_backoff,
    /// Retry with jittered backoff (reduces thundering herd)
    jittered_backoff,
    /// Use circuit breaker pattern
    circuit_breaker,
    /// Execute fallback action
    fallback,
    /// Fail fast without retry
    fail_fast,
    /// Custom recovery function
    custom,
};

/// Error severity levels
pub const ErrorSeverity = enum {
    /// Transient error that can be retried
    transient,
    /// Persistent error that might succeed with retry
    persistent,
    /// Fatal error that should not be retried
    fatal,
    /// Warning that doesn't stop execution
    warning,
};

/// Error categories for classification
pub const ErrorCategory = enum {
    /// Network-related errors (timeouts, connection failures)
    network,
    /// Authentication/authorization errors
    auth,
    /// Resource exhaustion (memory, disk, quota)
    resource,
    /// Invalid data or configuration
    validation,
    /// External service errors
    external_service,
    /// Internal logic errors
    internal,
    /// Timeout errors
    timeout,
    /// Unknown errors
    unknown,
};

/// Retry policy configuration
pub const RetryPolicy = struct {
    /// Maximum number of retry attempts
    max_attempts: u32,
    /// Initial delay in milliseconds
    initial_delay_ms: u32,
    /// Maximum delay in milliseconds
    max_delay_ms: u32,
    /// Backoff multiplier (for exponential backoff)
    backoff_multiplier: f32,
    /// Jitter factor (0.0-1.0, randomness added to delay)
    jitter_factor: f32,
    /// Which error categories to retry
    retry_on: []const ErrorCategory,
    /// Which error severities to retry
    retry_severity: []const ErrorSeverity,

    pub fn default() RetryPolicy {
        const retry_on = [_]ErrorCategory{.network, .timeout, .external_service};
        const retry_severity = [_]ErrorSeverity{.transient, .persistent};
        
        return RetryPolicy{
            .max_attempts = 3,
            .initial_delay_ms = 1000,
            .max_delay_ms = 30000,
            .backoff_multiplier = 2.0,
            .jitter_factor = 0.1,
            .retry_on = &retry_on,
            .retry_severity = &retry_severity,
        };
    }

    /// Calculate delay for a given attempt number
    pub fn calculateDelay(self: *const RetryPolicy, attempt: u32) u32 {
        if (attempt == 0) return 0;

        var delay: u32 = self.initial_delay_ms;

        // Apply backoff
        var i: u32 = 1;
        while (i < attempt) : (i += 1) {
            const new_delay = @as(f64, @floatFromInt(delay)) * self.backoff_multiplier;
            delay = @min(@as(u32, @intFromFloat(new_delay)), self.max_delay_ms);
        }

        // Apply jitter to prevent thundering herd
        if (self.jitter_factor > 0.0) {
            const jitter_amount = @as(f64, @floatFromInt(delay)) * self.jitter_factor;
            const random_factor = @as(f64, @floatFromInt(std.crypto.random.int(u32))) / @as(f64, @floatFromInt(std.math.maxInt(u32)));
            const jitter = jitter_amount * random_factor;
            delay = @intFromFloat(@as(f64, @floatFromInt(delay)) + jitter);
        }

        return @min(delay, self.max_delay_ms);
    }

    /// Check if error should be retried
    pub fn shouldRetry(self: *const RetryPolicy, category: ErrorCategory, severity: ErrorSeverity) bool {
        // Check category
        var category_match = false;
        for (self.retry_on) |cat| {
            if (cat == category) {
                category_match = true;
                break;
            }
        }
        if (!category_match) return false;

        // Check severity
        for (self.retry_severity) |sev| {
            if (sev == severity) return true;
        }

        return false;
    }
};

/// Circuit breaker states
pub const CircuitState = enum {
    closed,     // Normal operation
    open,       // Circuit is open, rejecting requests
    half_open,  // Testing if service recovered
};

/// Circuit breaker configuration
pub const CircuitBreakerConfig = struct {
    /// Number of failures before opening circuit
    failure_threshold: u32,
    /// Time in milliseconds before attempting recovery
    timeout_ms: u32,
    /// Number of successful calls to close circuit
    success_threshold: u32,
    /// Rolling window size for failure counting
    window_size: u32,

    pub fn default() CircuitBreakerConfig {
        return CircuitBreakerConfig{
            .failure_threshold = 5,
            .timeout_ms = 60000,
            .success_threshold = 2,
            .window_size = 10,
        };
    }
};

/// Circuit breaker implementation
pub const CircuitBreaker = struct {
    allocator: Allocator,
    config: CircuitBreakerConfig,
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: i64,
    failure_history: std.ArrayList(i64),

    pub fn init(allocator: Allocator, config: CircuitBreakerConfig) !CircuitBreaker {
        return CircuitBreaker{
            .allocator = allocator,
            .config = config,
            .state = .closed,
            .failure_count = 0,
            .success_count = 0,
            .last_failure_time = 0,
            .failure_history = .{},
        };
    }

    pub fn deinit(self: *CircuitBreaker) void {
        self.failure_history.deinit(self.allocator);
    }

    /// Check if circuit allows request
    pub fn allowRequest(self: *CircuitBreaker) bool {
        const now = std.time.milliTimestamp();

        switch (self.state) {
            .closed => return true,
            .open => {
                // Check if timeout has elapsed
                if (now - self.last_failure_time >= self.config.timeout_ms) {
                    self.state = .half_open;
                    self.success_count = 0;
                    return true;
                }
                return false;
            },
            .half_open => return true,
        }
    }

    /// Record successful execution
    pub fn recordSuccess(self: *CircuitBreaker) void {
        switch (self.state) {
            .closed => {
                self.failure_count = 0;
            },
            .half_open => {
                self.success_count += 1;
                if (self.success_count >= self.config.success_threshold) {
                    self.state = .closed;
                    self.failure_count = 0;
                }
            },
            .open => {},
        }
    }

    /// Record failed execution
    pub fn recordFailure(self: *CircuitBreaker) !void {
        const now = std.time.milliTimestamp();
        try self.failure_history.append(self.allocator, now);

        // Remove old failures outside window
        const window_start = now - @as(i64, @intCast(self.config.window_size * 1000));
        var i: usize = 0;
        while (i < self.failure_history.items.len) {
            if (self.failure_history.items[i] < window_start) {
                _ = self.failure_history.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        switch (self.state) {
            .closed => {
                self.failure_count += 1;
                if (self.failure_count >= self.config.failure_threshold) {
                    self.state = .open;
                    self.last_failure_time = now;
                }
            },
            .half_open => {
                self.state = .open;
                self.last_failure_time = now;
            },
            .open => {
                self.last_failure_time = now;
            },
        }
    }

    /// Get current state
    pub fn getState(self: *const CircuitBreaker) CircuitState {
        return self.state;
    }

    /// Reset circuit breaker
    pub fn reset(self: *CircuitBreaker) void {
        self.state = .closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.failure_history.clearRetainingCapacity();
    }
};

/// Error context with recovery information
pub const ErrorContext = struct {
    allocator: Allocator,
    /// Error message
    message: []const u8,
    /// Error category
    category: ErrorCategory,
    /// Error severity
    severity: ErrorSeverity,
    /// Timestamp when error occurred
    timestamp: i64,
    /// Node ID where error occurred
    node_id: ?[]const u8,
    /// Workflow ID
    workflow_id: ?[]const u8,
    /// Attempt number (for retries)
    attempt: u32,
    /// Stack trace (if available)
    stack_trace: ?[]const u8,
    /// Additional metadata
    metadata: std.StringHashMap([]const u8),

    pub fn init(
        allocator: Allocator,
        message: []const u8,
        category: ErrorCategory,
        severity: ErrorSeverity,
    ) !ErrorContext {
        return ErrorContext{
            .allocator = allocator,
            .message = try allocator.dupe(u8, message),
            .category = category,
            .severity = severity,
            .timestamp = std.time.milliTimestamp(),
            .node_id = null,
            .workflow_id = null,
            .attempt = 0,
            .stack_trace = null,
            .metadata = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *ErrorContext) void {
        self.allocator.free(self.message);
        if (self.node_id) |node_id| self.allocator.free(node_id);
        if (self.workflow_id) |workflow_id| self.allocator.free(workflow_id);
        if (self.stack_trace) |trace| self.allocator.free(trace);
        
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn setNodeId(self: *ErrorContext, node_id: []const u8) !void {
        if (self.node_id) |old| self.allocator.free(old);
        self.node_id = try self.allocator.dupe(u8, node_id);
    }

    pub fn setWorkflowId(self: *ErrorContext, workflow_id: []const u8) !void {
        if (self.workflow_id) |old| self.allocator.free(old);
        self.workflow_id = try self.allocator.dupe(u8, workflow_id);
    }

    pub fn addMetadata(self: *ErrorContext, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        const value_copy = try self.allocator.dupe(u8, value);
        try self.metadata.put(key_copy, value_copy);
    }
};

/// Fallback action function type
pub const FallbackFn = *const fn (allocator: Allocator, error_ctx: *const ErrorContext) anyerror!void;

/// Recovery configuration
pub const RecoveryConfig = struct {
    strategy: RecoveryStrategy,
    retry_policy: ?RetryPolicy,
    circuit_breaker_config: ?CircuitBreakerConfig,
    fallback_fn: ?FallbackFn,
    /// Whether to propagate errors up the chain
    propagate_errors: bool,
    /// Whether to log errors
    log_errors: bool,

    pub fn default() RecoveryConfig {
        return RecoveryConfig{
            .strategy = .exponential_backoff,
            .retry_policy = RetryPolicy.default(),
            .circuit_breaker_config = null,
            .fallback_fn = null,
            .propagate_errors = true,
            .log_errors = true,
        };
    }

    pub fn withCircuitBreaker() RecoveryConfig {
        var config = RecoveryConfig.default();
        config.strategy = .circuit_breaker;
        config.circuit_breaker_config = CircuitBreakerConfig.default();
        return config;
    }

    pub fn failFast() RecoveryConfig {
        return RecoveryConfig{
            .strategy = .fail_fast,
            .retry_policy = null,
            .circuit_breaker_config = null,
            .fallback_fn = null,
            .propagate_errors = true,
            .log_errors = true,
        };
    }
};

/// Error recovery manager
pub const ErrorRecoveryManager = struct {
    allocator: Allocator,
    config: RecoveryConfig,
    circuit_breaker: ?CircuitBreaker,
    error_history: std.ArrayList(ErrorContext),
    total_errors: u64,
    total_recoveries: u64,

    pub fn init(allocator: Allocator, config: RecoveryConfig) !ErrorRecoveryManager {
        var circuit_breaker: ?CircuitBreaker = null;
        if (config.circuit_breaker_config) |cb_config| {
            circuit_breaker = try CircuitBreaker.init(allocator, cb_config);
        }

        return ErrorRecoveryManager{
            .allocator = allocator,
            .config = config,
            .circuit_breaker = circuit_breaker,
            .error_history = .{},
            .total_errors = 0,
            .total_recoveries = 0,
        };
    }

    pub fn deinit(self: *ErrorRecoveryManager) void {
        if (self.circuit_breaker) |*cb| {
            cb.deinit();
        }
        for (self.error_history.items) |*ctx| {
            ctx.deinit();
        }
        self.error_history.deinit(self.allocator);
    }

    /// Execute operation with retry logic
    pub fn executeWithRetry(
        self: *ErrorRecoveryManager,
        comptime T: type,
        operation: *const fn (allocator: Allocator) anyerror!T,
    ) !T {
        // Check circuit breaker
        if (self.circuit_breaker) |*cb| {
            if (!cb.allowRequest()) {
                return error.CircuitOpen;
            }
        }

        const policy = self.config.retry_policy orelse {
            // No retry policy, execute once
            return self.executeSingle(T, operation);
        };

        var attempt: u32 = 0;
        var last_error: ?ErrorContext = null;

        while (attempt <= policy.max_attempts) : (attempt += 1) {
            // Calculate delay for this attempt
            if (attempt > 0) {
                const delay_ms = policy.calculateDelay(attempt);
                std.Thread.sleep(delay_ms * std.time.ns_per_ms);
            }

            // Try execution
            const result = operation(self.allocator) catch |err| {
                // Create error context
                var error_ctx = try self.createErrorContext(err, attempt);
                errdefer error_ctx.deinit();

                // Check if should retry
                if (attempt >= policy.max_attempts or
                    !policy.shouldRetry(error_ctx.category, error_ctx.severity))
                {
                    // No more retries or shouldn't retry this error
                    try self.handleFinalError(&error_ctx);
                    return err;
                }

                // Log retry attempt
                if (self.config.log_errors) {
                    std.debug.print("Retry attempt {d}/{d} after error: {s}\n", .{
                        attempt + 1,
                        policy.max_attempts,
                        error_ctx.message,
                    });
                }

                last_error = error_ctx;
                continue;
            };

            // Success
            if (self.circuit_breaker) |*cb| {
                cb.recordSuccess();
            }
            self.total_recoveries += 1;

            if (last_error) |*ctx| {
                ctx.deinit();
            }

            return result;
        }

        // Should never reach here
        return error.MaxRetriesExceeded;
    }

    fn executeSingle(
        self: *ErrorRecoveryManager,
        comptime T: type,
        operation: *const fn (allocator: Allocator) anyerror!T,
    ) !T {
        const result = operation(self.allocator) catch |err| {
            var error_ctx = try self.createErrorContext(err, 0);
            try self.handleFinalError(&error_ctx);
            return err;
        };

        if (self.circuit_breaker) |*cb| {
            cb.recordSuccess();
        }

        return result;
    }

    fn createErrorContext(self: *ErrorRecoveryManager, err: anyerror, attempt: u32) !ErrorContext {
        const message = @errorName(err);
        const category = self.categorizeError(err);
        const severity = self.determineSeverity(category, err);

        var ctx = try ErrorContext.init(self.allocator, message, category, severity);
        ctx.attempt = attempt;
        return ctx;
    }

    fn categorizeError(self: *ErrorRecoveryManager, err: anyerror) ErrorCategory {
        _ = self;
        
        // Map common errors to categories
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

            else => .unknown,
        };
    }

    fn determineSeverity(self: *ErrorRecoveryManager, category: ErrorCategory, err: anyerror) ErrorSeverity {
        _ = self;
        _ = err;
        
        // Determine severity based on category
        return switch (category) {
            .network, .timeout, .external_service => .transient,
            .auth, .validation => .persistent,
            .resource, .internal => .fatal,
            .unknown => .persistent,
        };
    }

    fn handleFinalError(self: *ErrorRecoveryManager, error_ctx: *ErrorContext) !void {
        self.total_errors += 1;

        // Record in circuit breaker
        if (self.circuit_breaker) |*cb| {
            try cb.recordFailure();
        }

        // Add to history
        try self.error_history.append(self.allocator, error_ctx.*);

        // Try fallback
        if (self.config.fallback_fn) |fallback| {
            fallback(self.allocator, error_ctx) catch |err| {
                if (self.config.log_errors) {
                    std.debug.print("Fallback failed: {}\n", .{err});
                }
            };
        }

        // Log error
        if (self.config.log_errors) {
            std.debug.print("Error: {s} (category: {}, severity: {})\n", .{
                error_ctx.message,
                error_ctx.category,
                error_ctx.severity,
            });
        }
    }

    /// Get error statistics
    pub fn getStats(self: *const ErrorRecoveryManager) ErrorStats {
        const recovery_rate = if (self.total_errors > 0)
            @as(f64, @floatFromInt(self.total_recoveries)) / @as(f64, @floatFromInt(self.total_errors))
        else
            0.0;

        return ErrorStats{
            .total_errors = self.total_errors,
            .total_recoveries = self.total_recoveries,
            .recovery_rate = recovery_rate,
            .circuit_state = if (self.circuit_breaker) |*cb| cb.getState() else .closed,
        };
    }

    /// Clear error history
    pub fn clearHistory(self: *ErrorRecoveryManager) void {
        for (self.error_history.items) |*ctx| {
            ctx.deinit();
        }
        self.error_history.clearRetainingCapacity();
    }
};

/// Error statistics
pub const ErrorStats = struct {
    total_errors: u64,
    total_recoveries: u64,
    recovery_rate: f64,
    circuit_state: CircuitState,
};

// Tests
const testing = std.testing;

test "RetryPolicy.calculateDelay exponential backoff" {
    const policy = RetryPolicy{
        .max_attempts = 5,
        .initial_delay_ms = 100,
        .max_delay_ms = 5000,
        .backoff_multiplier = 2.0,
        .jitter_factor = 0.0,
        .retry_on = &[_]ErrorCategory{.network},
        .retry_severity = &[_]ErrorSeverity{.transient},
    };

    try testing.expectEqual(@as(u32, 0), policy.calculateDelay(0));
    try testing.expectEqual(@as(u32, 100), policy.calculateDelay(1));
    try testing.expectEqual(@as(u32, 200), policy.calculateDelay(2));
    try testing.expectEqual(@as(u32, 400), policy.calculateDelay(3));
    try testing.expectEqual(@as(u32, 800), policy.calculateDelay(4));
}

test "RetryPolicy.shouldRetry checks category and severity" {
    const retry_on = [_]ErrorCategory{.network, .timeout};
    const retry_severity = [_]ErrorSeverity{.transient};
    
    const policy = RetryPolicy{
        .max_attempts = 3,
        .initial_delay_ms = 1000,
        .max_delay_ms = 10000,
        .backoff_multiplier = 2.0,
        .jitter_factor = 0.0,
        .retry_on = &retry_on,
        .retry_severity = &retry_severity,
    };

    try testing.expect(policy.shouldRetry(.network, .transient));
    try testing.expect(policy.shouldRetry(.timeout, .transient));
    try testing.expect(!policy.shouldRetry(.network, .fatal));
    try testing.expect(!policy.shouldRetry(.auth, .transient));
}

test "CircuitBreaker state transitions" {
    const config = CircuitBreakerConfig{
        .failure_threshold = 3,
        .timeout_ms = 1000,
        .success_threshold = 2,
        .window_size = 10,
    };

    var cb = try CircuitBreaker.init(testing.allocator, config);
    defer cb.deinit();

    // Initial state: closed
    try testing.expectEqual(CircuitState.closed, cb.getState());
    try testing.expect(cb.allowRequest());

    // Record failures
    try cb.recordFailure();
    try cb.recordFailure();
    try testing.expectEqual(CircuitState.closed, cb.getState());

    try cb.recordFailure();
    // After threshold failures, circuit opens
    try testing.expectEqual(CircuitState.open, cb.getState());
    try testing.expect(!cb.allowRequest());

    // Wait for timeout
    std.Thread.sleep(1100 * std.time.ns_per_ms);
    
    // Should be half-open now
    try testing.expect(cb.allowRequest());
    try testing.expectEqual(CircuitState.half_open, cb.getState());

    // Record successes
    cb.recordSuccess();
    try testing.expectEqual(CircuitState.half_open, cb.getState());
    
    cb.recordSuccess();
    // After success threshold, circuit closes
    try testing.expectEqual(CircuitState.closed, cb.getState());
}

test "ErrorContext creation and metadata" {
    var ctx = try ErrorContext.init(
        testing.allocator,
        "Connection failed",
        .network,
        .transient,
    );
    defer ctx.deinit();

    try testing.expectEqualStrings("Connection failed", ctx.message);
    try testing.expectEqual(ErrorCategory.network, ctx.category);
    try testing.expectEqual(ErrorSeverity.transient, ctx.severity);

    try ctx.setNodeId("node1");
    try ctx.setWorkflowId("workflow1");
    try ctx.addMetadata("host", "localhost");
    try ctx.addMetadata("port", "8080");

    try testing.expectEqualStrings("node1", ctx.node_id.?);
    try testing.expectEqualStrings("workflow1", ctx.workflow_id.?);
    try testing.expectEqualStrings("localhost", ctx.metadata.get("host").?);
    try testing.expectEqualStrings("8080", ctx.metadata.get("port").?);
}

test "ErrorRecoveryManager basic operations" {
    const config = RecoveryConfig.default();
    var manager = try ErrorRecoveryManager.init(testing.allocator, config);
    defer manager.deinit();

    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 0), stats.total_errors);
    try testing.expectEqual(@as(u64, 0), stats.total_recoveries);
}

test "ErrorRecoveryManager with circuit breaker" {
    const config = RecoveryConfig.withCircuitBreaker();
    var manager = try ErrorRecoveryManager.init(testing.allocator, config);
    defer manager.deinit();

    const stats = manager.getStats();
    try testing.expectEqual(CircuitState.closed, stats.circuit_state);
}

test "RecoveryConfig presets" {
    const default_config = RecoveryConfig.default();
    try testing.expectEqual(RecoveryStrategy.exponential_backoff, default_config.strategy);
    try testing.expect(default_config.retry_policy != null);

    const fail_fast_config = RecoveryConfig.failFast();
    try testing.expectEqual(RecoveryStrategy.fail_fast, fail_fast_config.strategy);
    try testing.expect(fail_fast_config.retry_policy == null);

    const cb_config = RecoveryConfig.withCircuitBreaker();
    try testing.expectEqual(RecoveryStrategy.circuit_breaker, cb_config.strategy);
    try testing.expect(cb_config.circuit_breaker_config != null);
}
