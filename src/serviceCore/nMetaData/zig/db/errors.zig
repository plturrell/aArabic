const std = @import("std");

/// Error severity levels
pub const Severity = enum {
    debug,
    info,
    warning,
    error_level,
    critical,

    pub fn toString(self: Severity) []const u8 {
        return switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warning => "WARNING",
            .error_level => "ERROR",
            .critical => "CRITICAL",
        };
    }
};

/// Error category for classification
pub const ErrorCategory = enum {
    connection, // Connection-related errors
    query, // SQL query errors
    transaction, // Transaction errors
    pool, // Connection pool errors
    timeout, // Timeout errors
    validation, // Input validation errors
    internal, // Internal/unexpected errors

    pub fn toString(self: ErrorCategory) []const u8 {
        return switch (self) {
            .connection => "CONNECTION",
            .query => "QUERY",
            .transaction => "TRANSACTION",
            .pool => "POOL",
            .timeout => "TIMEOUT",
            .validation => "VALIDATION",
            .internal => "INTERNAL",
        };
    }
};

/// Error recovery strategy
pub const RecoveryStrategy = enum {
    none, // No automatic recovery
    retry, // Retry the operation
    fallback, // Use fallback behavior
    reconnect, // Reconnect to database
    abort, // Abort and propagate error

    pub fn toString(self: RecoveryStrategy) []const u8 {
        return switch (self) {
            .none => "NONE",
            .retry => "RETRY",
            .fallback => "FALLBACK",
            .reconnect => "RECONNECT",
            .abort => "ABORT",
        };
    }
};

/// Error context with debugging information
pub const ErrorContext = struct {
    message: []const u8,
    category: ErrorCategory,
    severity: Severity,
    recovery_strategy: RecoveryStrategy,
    timestamp: i64,
    source_location: ?std.builtin.SourceLocation,
    additional_info: ?[]const u8,

    pub fn init(
        message: []const u8,
        category: ErrorCategory,
        severity: Severity,
        recovery_strategy: RecoveryStrategy,
    ) ErrorContext {
        return ErrorContext{
            .message = message,
            .category = category,
            .severity = severity,
            .recovery_strategy = recovery_strategy,
            .timestamp = std.time.milliTimestamp(),
            .source_location = null,
            .additional_info = null,
        };
    }

    pub fn withLocation(self: ErrorContext, location: std.builtin.SourceLocation) ErrorContext {
        var ctx = self;
        ctx.source_location = location;
        return ctx;
    }

    pub fn withInfo(self: ErrorContext, info: []const u8) ErrorContext {
        var ctx = self;
        ctx.additional_info = info;
        return ctx;
    }

    pub fn format(self: ErrorContext, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.print("[{s}] [{s}] {s}", .{
            self.severity.toString(),
            self.category.toString(),
            self.message,
        });

        if (self.source_location) |loc| {
            try writer.print(" at {s}:{d}:{d}", .{
                loc.file,
                loc.line,
                loc.column,
            });
        }

        if (self.additional_info) |info| {
            try writer.print(" - {s}", .{info});
        }
    }
};

/// Database-specific error types
pub const DbError = error{
    // Connection errors
    ConnectionFailed,
    ConnectionTimeout,
    ConnectionClosed,
    ConnectionPoolExhausted,

    // Query errors
    QueryFailed,
    InvalidSQL,
    ConstraintViolation,
    UniqueViolation,
    ForeignKeyViolation,
    NotNullViolation,

    // Transaction errors
    TransactionFailed,
    TransactionAlreadyTerminated,
    TransactionNotActive,
    DeadlockDetected,
    SerializationFailure,

    // Savepoint errors
    SavepointNotFound,
    SavepointFailed,

    // Pool errors
    AcquireTimeout,
    InvalidPoolConfig,
    PoolShutdown,

    // Type errors
    TypeMismatch,
    InvalidValue,
    ConversionError,

    // Validation errors
    InvalidIdentifier,
    InvalidParameter,
    MissingParameter,

    // Data errors
    RowNotFound,
    MultipleRowsFound,
    ColumnNotFound,
    IndexOutOfBounds,

    // Internal errors
    OutOfMemory,
    BufferTooSmall,
    UnexpectedError,
};

/// Create error context from standard error
pub fn contextFromError(err: anyerror) ErrorContext {
    return switch (err) {
        // Connection errors
        error.ConnectionFailed => ErrorContext.init(
            "Failed to connect to database",
            .connection,
            .error_level,
            .reconnect,
        ),
        error.ConnectionTimeout => ErrorContext.init(
            "Database connection timed out",
            .connection,
            .error_level,
            .retry,
        ),
        error.ConnectionClosed => ErrorContext.init(
            "Database connection was closed",
            .connection,
            .warning,
            .reconnect,
        ),
        error.ConnectionPoolExhausted => ErrorContext.init(
            "Connection pool exhausted",
            .pool,
            .warning,
            .retry,
        ),

        // Query errors
        error.QueryFailed => ErrorContext.init(
            "SQL query execution failed",
            .query,
            .error_level,
            .none,
        ),
        error.InvalidSQL => ErrorContext.init(
            "Invalid SQL syntax",
            .query,
            .error_level,
            .none,
        ),
        error.ConstraintViolation => ErrorContext.init(
            "Database constraint violated",
            .query,
            .error_level,
            .none,
        ),

        // Transaction errors
        error.TransactionFailed => ErrorContext.init(
            "Transaction failed",
            .transaction,
            .error_level,
            .none,
        ),
        error.TransactionAlreadyTerminated => ErrorContext.init(
            "Transaction already committed or rolled back",
            .transaction,
            .warning,
            .none,
        ),
        error.DeadlockDetected => ErrorContext.init(
            "Database deadlock detected",
            .transaction,
            .error_level,
            .retry,
        ),
        error.SerializationFailure => ErrorContext.init(
            "Serialization failure in transaction",
            .transaction,
            .warning,
            .retry,
        ),

        // Timeout errors
        error.AcquireTimeout => ErrorContext.init(
            "Timeout acquiring connection from pool",
            .timeout,
            .warning,
            .retry,
        ),

        // Type errors
        error.TypeMismatch => ErrorContext.init(
            "Type mismatch in value conversion",
            .validation,
            .error_level,
            .none,
        ),

        // Data errors
        error.RowNotFound => ErrorContext.init(
            "Requested row not found",
            .query,
            .info,
            .none,
        ),

        else => ErrorContext.init(
            "Unexpected error",
            .internal,
            .critical,
            .abort,
        ),
    };
}

/// Check if error is retryable
pub fn isRetryable(err: anyerror) bool {
    const ctx = contextFromError(err);
    return ctx.recovery_strategy == .retry;
}

/// Check if error requires reconnection
pub fn requiresReconnect(err: anyerror) bool {
    const ctx = contextFromError(err);
    return ctx.recovery_strategy == .reconnect;
}

/// Check if error is transient
pub fn isTransient(err: anyerror) bool {
    return switch (err) {
        error.ConnectionTimeout,
        error.AcquireTimeout,
        error.DeadlockDetected,
        error.SerializationFailure,
        => true,
        else => false,
    };
}

/// Check if error is permanent
pub fn isPermanent(err: anyerror) bool {
    return switch (err) {
        error.InvalidSQL,
        error.InvalidIdentifier,
        error.TypeMismatch,
        error.ConstraintViolation,
        => true,
        else => false,
    };
}

/// Log error with context
pub fn logError(err: anyerror, comptime location: std.builtin.SourceLocation) void {
    const ctx = contextFromError(err).withLocation(location);
    
    switch (ctx.severity) {
        .debug => std.log.debug("{}", .{ctx}),
        .info => std.log.info("{}", .{ctx}),
        .warning => std.log.warn("{}", .{ctx}),
        .error_level => std.log.err("{}", .{ctx}),
        .critical => std.log.err("CRITICAL: {}", .{ctx}),
    }
}

/// Retry helper with exponential backoff
pub fn retryWithBackoff(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    args: anytype,
    max_attempts: u32,
    initial_delay_ms: u64,
) !@typeInfo(@TypeOf(func)).Fn.return_type.? {
    var attempt: u32 = 0;
    var delay_ms = initial_delay_ms;

    while (attempt < max_attempts) : (attempt += 1) {
        if (@call(.auto, func, args)) |result| {
            return result;
        } else |err| {
            if (!isRetryable(err) or attempt == max_attempts - 1) {
                return err;
            }

            std.log.warn("Attempt {d} failed, retrying in {d}ms: {}", .{
                attempt + 1,
                delay_ms,
                err,
            });

            std.time.sleep(delay_ms * std.time.ns_per_ms);
            
            // Exponential backoff with jitter
            delay_ms = delay_ms * 2;
            if (delay_ms > 30000) delay_ms = 30000; // Max 30 seconds
        }
    }

    return error.MaxRetriesExceeded;
}

// ============================================================================
// Unit Tests
// ============================================================================

test "Severity - toString" {
    try std.testing.expectEqualStrings("DEBUG", Severity.debug.toString());
    try std.testing.expectEqualStrings("ERROR", Severity.error_level.toString());
    try std.testing.expectEqualStrings("CRITICAL", Severity.critical.toString());
}

test "ErrorCategory - toString" {
    try std.testing.expectEqualStrings("CONNECTION", ErrorCategory.connection.toString());
    try std.testing.expectEqualStrings("QUERY", ErrorCategory.query.toString());
    try std.testing.expectEqualStrings("TRANSACTION", ErrorCategory.transaction.toString());
}

test "RecoveryStrategy - toString" {
    try std.testing.expectEqualStrings("RETRY", RecoveryStrategy.retry.toString());
    try std.testing.expectEqualStrings("RECONNECT", RecoveryStrategy.reconnect.toString());
    try std.testing.expectEqualStrings("ABORT", RecoveryStrategy.abort.toString());
}

test "ErrorContext - basic creation" {
    const ctx = ErrorContext.init(
        "Test error",
        .connection,
        .error_level,
        .retry,
    );

    try std.testing.expectEqualStrings("Test error", ctx.message);
    try std.testing.expectEqual(ErrorCategory.connection, ctx.category);
    try std.testing.expectEqual(Severity.error_level, ctx.severity);
    try std.testing.expectEqual(RecoveryStrategy.retry, ctx.recovery_strategy);
}

test "ErrorContext - with location" {
    const ctx = ErrorContext.init("Test", .query, .error_level, .none)
        .withLocation(@src());

    try std.testing.expect(ctx.source_location != null);
}

test "ErrorContext - with info" {
    const ctx = ErrorContext.init("Test", .query, .error_level, .none)
        .withInfo("Additional details");

    try std.testing.expectEqualStrings("Additional details", ctx.additional_info.?);
}

test "ErrorContext - format" {
    const ctx = ErrorContext.init(
        "Connection failed",
        .connection,
        .error_level,
        .reconnect,
    );

    var buf: [200]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try fbs.writer().print("{}", .{ctx});

    const result = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, result, "ERROR") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "CONNECTION") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Connection failed") != null);
}

test "contextFromError - connection errors" {
    const ctx1 = contextFromError(error.ConnectionFailed);
    try std.testing.expectEqual(ErrorCategory.connection, ctx1.category);
    try std.testing.expectEqual(RecoveryStrategy.reconnect, ctx1.recovery_strategy);

    const ctx2 = contextFromError(error.ConnectionTimeout);
    try std.testing.expectEqual(ErrorCategory.connection, ctx2.category);
    try std.testing.expectEqual(RecoveryStrategy.retry, ctx2.recovery_strategy);
}

test "contextFromError - query errors" {
    const ctx = contextFromError(error.InvalidSQL);
    try std.testing.expectEqual(ErrorCategory.query, ctx.category);
    try std.testing.expectEqual(RecoveryStrategy.none, ctx.recovery_strategy);
}

test "contextFromError - transaction errors" {
    const ctx1 = contextFromError(error.DeadlockDetected);
    try std.testing.expectEqual(ErrorCategory.transaction, ctx1.category);
    try std.testing.expectEqual(RecoveryStrategy.retry, ctx1.recovery_strategy);

    const ctx2 = contextFromError(error.TransactionAlreadyTerminated);
    try std.testing.expectEqual(ErrorCategory.transaction, ctx2.category);
    try std.testing.expectEqual(RecoveryStrategy.none, ctx2.recovery_strategy);
}

test "isRetryable - various errors" {
    try std.testing.expect(isRetryable(error.ConnectionTimeout));
    try std.testing.expect(isRetryable(error.DeadlockDetected));
    try std.testing.expect(isRetryable(error.AcquireTimeout));

    try std.testing.expect(!isRetryable(error.InvalidSQL));
    try std.testing.expect(!isRetryable(error.TypeMismatch));
}

test "requiresReconnect - connection errors" {
    try std.testing.expect(requiresReconnect(error.ConnectionFailed));
    try std.testing.expect(requiresReconnect(error.ConnectionClosed));

    try std.testing.expect(!requiresReconnect(error.QueryFailed));
    try std.testing.expect(!requiresReconnect(error.InvalidSQL));
}

test "isTransient - error classification" {
    try std.testing.expect(isTransient(error.ConnectionTimeout));
    try std.testing.expect(isTransient(error.AcquireTimeout));
    try std.testing.expect(isTransient(error.DeadlockDetected));
    try std.testing.expect(isTransient(error.SerializationFailure));

    try std.testing.expect(!isTransient(error.InvalidSQL));
    try std.testing.expect(!isTransient(error.ConstraintViolation));
}

test "isPermanent - error classification" {
    try std.testing.expect(isPermanent(error.InvalidSQL));
    try std.testing.expect(isPermanent(error.InvalidIdentifier));
    try std.testing.expect(isPermanent(error.TypeMismatch));
    try std.testing.expect(isPermanent(error.ConstraintViolation));

    try std.testing.expect(!isPermanent(error.ConnectionTimeout));
    try std.testing.expect(!isPermanent(error.DeadlockDetected));
}

// Mock function for retry testing
fn flakeyOperation(success_after: *u32) !i32 {
    success_after.* += 1;
    if (success_after.* < 3) {
        return error.ConnectionTimeout;
    }
    return 42;
}

test "retryWithBackoff - successful retry" {
    var attempt_count: u32 = 0;

    const result = try retryWithBackoff(
        std.testing.allocator,
        flakeyOperation,
        .{&attempt_count},
        5,
        10,
    );

    try std.testing.expectEqual(@as(i32, 42), result);
    try std.testing.expectEqual(@as(u32, 3), attempt_count);
}

// Mock function that always fails
fn alwaysFailOperation() !i32 {
    return error.InvalidSQL;
}

test "retryWithBackoff - permanent error" {
    const result = retryWithBackoff(
        std.testing.allocator,
        alwaysFailOperation,
        .{},
        3,
        10,
    );

    try std.testing.expectError(error.InvalidSQL, result);
}

// Mock function that exceeds max retries
fn exceedRetriesOperation(counter: *u32) !i32 {
    counter.* += 1;
    return error.ConnectionTimeout;
}

test "retryWithBackoff - max retries exceeded" {
    var counter: u32 = 0;

    const result = retryWithBackoff(
        std.testing.allocator,
        exceedRetriesOperation,
        .{&counter},
        3,
        5,
    );

    try std.testing.expectError(error.MaxRetriesExceeded, result);
    try std.testing.expectEqual(@as(u32, 3), counter);
}
