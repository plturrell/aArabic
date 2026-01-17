// ============================================================================
// HyperShimmy Error Handling Module
// ============================================================================
// Comprehensive error handling system for HyperShimmy server
// Day 51: Error Handling & Recovery
// ============================================================================

const std = @import("std");
const mem = std.mem;
const json = std.json;

// ============================================================================
// Error Types
// ============================================================================

/// Comprehensive error set for HyperShimmy operations
pub const HyperShimmyError = error{
    // HTTP/Network Errors
    InvalidRequest,
    InvalidContentType,
    RequestTooLarge,
    InvalidJson,
    InvalidMultipart,
    NetworkTimeout,
    ConnectionFailed,

    // Resource Errors
    ResourceNotFound,
    ResourceAlreadyExists,
    ResourceLocked,
    ResourceDeleted,

    // Source Management Errors
    SourceNotFound,
    InvalidSourceType,
    InvalidSourceStatus,
    SourceCreationFailed,
    SourceUpdateFailed,
    SourceDeletionFailed,

    // File Operation Errors
    FileNotFound,
    FileReadError,
    FileWriteError,
    FileDeleteError,
    InvalidFileType,
    FileTooLarge,
    DirectoryNotFound,
    DirectoryCreationFailed,

    // Upload Errors
    UploadFailed,
    NoBoundary,
    NoBoundaryFound,
    NoContentDisposition,
    NoFilename,
    NoEndBoundary,
    InvalidFormat,
    UnsupportedFileType,

    // Parsing Errors
    ParseError,
    InvalidData,
    MissingRequiredField,
    InvalidFieldValue,

    // Database Errors
    DatabaseError,
    QueryFailed,
    TransactionFailed,
    ConstraintViolation,

    // AI/LLM Errors
    LLMError,
    EmbeddingError,
    InferenceError,
    ModelNotFound,
    ContextTooLarge,

    // Processing Errors
    ProcessingFailed,
    ExtractionError,
    IndexingError,
    GenerationError,

    // OData Errors
    ODataParseError,
    ODataValidationError,
    InvalidODataRequest,
    UnsupportedOperation,

    // System Errors
    OutOfMemory,
    AllocationFailed,
    InternalError,
    NotImplemented,
    Timeout,

    // Validation Errors
    ValidationFailed,
    EmptyFileId,
    EmptyText,
    InvalidInput,
};

/// Error severity levels
pub const ErrorSeverity = enum {
    debug,
    info,
    warning,
    error_level,
    critical,

    pub fn toString(self: ErrorSeverity) []const u8 {
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
    client_error, // 4xx - client mistakes
    server_error, // 5xx - server issues
    validation_error, // Input validation failures
    resource_error, // Resource management issues
    system_error, // System-level problems

    pub fn toHttpStatus(self: ErrorCategory) u16 {
        return switch (self) {
            .client_error => 400,
            .validation_error => 422,
            .resource_error => 404,
            .server_error => 500,
            .system_error => 503,
        };
    }
};

// ============================================================================
// Error Context
// ============================================================================

/// Detailed error context for debugging and logging
pub const ErrorContext = struct {
    error_type: []const u8,
    message: []const u8,
    severity: ErrorSeverity,
    category: ErrorCategory,
    timestamp: i64,
    context: ?[]const u8 = null,
    stack_trace: ?[]const u8 = null,
    recoverable: bool = true,

    pub fn deinit(self: ErrorContext, allocator: mem.Allocator) void {
        allocator.free(self.error_type);
        allocator.free(self.message);
        if (self.context) |ctx| allocator.free(ctx);
        if (self.stack_trace) |trace| allocator.free(trace);
    }
};

// ============================================================================
// Error Handler
// ============================================================================

pub const ErrorHandler = struct {
    allocator: mem.Allocator,
    enable_logging: bool = true,
    enable_stack_traces: bool = false,

    pub fn init(allocator: mem.Allocator) ErrorHandler {
        return .{
            .allocator = allocator,
            .enable_logging = true,
            .enable_stack_traces = false,
        };
    }

    /// Create error context from an error
    pub fn createContext(
        self: *ErrorHandler,
        err: anyerror,
        message: []const u8,
        severity: ErrorSeverity,
        context_info: ?[]const u8,
    ) !ErrorContext {
        const error_name = @errorName(err);
        const category = self.categorizeError(err);
        const timestamp = std.time.timestamp();

        return ErrorContext{
            .error_type = try self.allocator.dupe(u8, error_name),
            .message = try self.allocator.dupe(u8, message),
            .severity = severity,
            .category = category,
            .timestamp = timestamp,
            .context = if (context_info) |info| try self.allocator.dupe(u8, info) else null,
            .stack_trace = null, // Would capture in production
            .recoverable = self.isRecoverable(err),
        };
    }

    /// Categorize an error
    pub fn categorizeError(self: *const ErrorHandler, err: anyerror) ErrorCategory {
        _ = self;
        return switch (err) {
            // Client errors
            error.InvalidRequest,
            error.InvalidContentType,
            error.InvalidJson,
            error.InvalidMultipart,
            error.UnsupportedFileType,
            => .client_error,

            // Validation errors
            error.ValidationFailed,
            error.EmptyFileId,
            error.EmptyText,
            error.InvalidInput,
            error.MissingRequiredField,
            error.InvalidFieldValue,
            => .validation_error,

            // Resource errors
            error.ResourceNotFound,
            error.SourceNotFound,
            error.FileNotFound,
            error.DirectoryNotFound,
            => .resource_error,

            // System errors
            error.OutOfMemory,
            error.AllocationFailed,
            error.Timeout,
            error.NetworkTimeout,
            => .system_error,

            // Default to server error
            else => .server_error,
        };
    }

    /// Check if an error is recoverable
    pub fn isRecoverable(self: *const ErrorHandler, err: anyerror) bool {
        _ = self;
        return switch (err) {
            // Non-recoverable
            error.OutOfMemory,
            error.AllocationFailed,
            error.InternalError,
            => false,

            // Recoverable
            else => true,
        };
    }

    /// Log error with context
    pub fn logError(self: *ErrorHandler, ctx: ErrorContext) void {
        if (!self.enable_logging) return;

        const severity_str = ctx.severity.toString();
        const timestamp_str = self.formatTimestamp(ctx.timestamp);

        std.debug.print(
            "\n[{s}] {s} - {s}: {s}\n",
            .{ timestamp_str, severity_str, ctx.error_type, ctx.message },
        );

        if (ctx.context) |context_info| {
            std.debug.print("  Context: {s}\n", .{context_info});
        }

        if (ctx.stack_trace) |trace| {
            std.debug.print("  Stack Trace:\n{s}\n", .{trace});
        }

        std.debug.print("  Recoverable: {}\n\n", .{ctx.recoverable});
    }

    /// Format timestamp for logging
    fn formatTimestamp(self: *ErrorHandler, timestamp: i64) []const u8 {
        _ = self;
        _ = timestamp;
        // Simplified for now - would format properly in production
        return "2026-01-16T20:00:00Z";
    }

    /// Format error as OData error response
    pub fn formatODataError(
        self: *ErrorHandler,
        code: []const u8,
        message: []const u8,
        target: ?[]const u8,
        details: ?[]const []const u8,
    ) ![]const u8 {
        _ = target;
        _ = details;
        
        // Simplified implementation for now
        return try std.fmt.allocPrint(self.allocator,
            \\{{"error":{{"code":"{s}","message":"{s}"}}}}
        , .{code, message});
    }

    /// Escape JSON string
    fn appendEscapedJson(self: *ErrorHandler, buffer: *std.ArrayList(u8), str: []const u8) !void {
        _ = self;
        for (str) |c| {
            switch (c) {
                '"' => try buffer.appendSlice("\\\""),
                '\\' => try buffer.appendSlice("\\\\"),
                '\n' => try buffer.appendSlice("\\n"),
                '\r' => try buffer.appendSlice("\\r"),
                '\t' => try buffer.appendSlice("\\t"),
                else => try buffer.append(c),
            }
        }
    }

    /// Format error as HTTP error response
    pub fn formatHttpError(
        self: *ErrorHandler,
        status: u16,
        message: []const u8,
    ) ![]const u8 {
        return try std.fmt.allocPrint(self.allocator,
            \\{{"error":true,"status":{d},"message":"{s}"}}
        , .{status, message});
    }
};

// ============================================================================
// Error Recovery Strategies
// ============================================================================

pub const RecoveryStrategy = enum {
    retry,
    fallback,
    skip,
    abort,
    log_and_continue,
};

pub const RecoveryConfig = struct {
    strategy: RecoveryStrategy,
    max_retries: u32 = 3,
    retry_delay_ms: u64 = 1000,
    fallback_value: ?[]const u8 = null,
};

pub const RecoveryManager = struct {
    allocator: mem.Allocator,
    error_handler: *ErrorHandler,

    pub fn init(allocator: mem.Allocator, error_handler: *ErrorHandler) RecoveryManager {
        return .{
            .allocator = allocator,
            .error_handler = error_handler,
        };
    }

    /// Execute operation with recovery strategy
    pub fn executeWithRecovery(
        self: *RecoveryManager,
        comptime T: type,
        operation: fn () anyerror!T,
        config: RecoveryConfig,
    ) !T {
        var attempts: u32 = 0;
        
        while (attempts < config.max_retries) : (attempts += 1) {
            const result = operation() catch |err| {
                const ctx = try self.error_handler.createContext(
                    err,
                    "Operation failed, attempting recovery",
                    .warning,
                    null,
                );
                defer ctx.deinit(self.allocator);
                
                self.error_handler.logError(ctx);
                
                switch (config.strategy) {
                    .retry => {
                        if (attempts < config.max_retries - 1) {
                            std.time.sleep(config.retry_delay_ms * std.time.ns_per_ms);
                            continue;
                        }
                        return err;
                    },
                    .abort => return err,
                    else => return err,
                }
            };
            
            return result;
        }
        
        return error.MaxRetriesExceeded;
    }
};

// ============================================================================
// Error Metrics & Monitoring
// ============================================================================

pub const ErrorMetrics = struct {
    total_errors: u64 = 0,
    client_errors: u64 = 0,
    server_errors: u64 = 0,
    validation_errors: u64 = 0,
    resource_errors: u64 = 0,
    system_errors: u64 = 0,

    pub fn recordError(self: *ErrorMetrics, category: ErrorCategory) void {
        self.total_errors += 1;
        switch (category) {
            .client_error => self.client_errors += 1,
            .server_error => self.server_errors += 1,
            .validation_error => self.validation_errors += 1,
            .resource_error => self.resource_errors += 1,
            .system_error => self.system_errors += 1,
        }
    }

    pub fn reset(self: *ErrorMetrics) void {
        self.* = .{};
    }

    pub fn toJson(self: ErrorMetrics, allocator: mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator,
            \\{{"total_errors":{d},"client_errors":{d},"server_errors":{d},"validation_errors":{d},"resource_errors":{d},"system_errors":{d}}}
        , .{
            self.total_errors,
            self.client_errors,
            self.server_errors,
            self.validation_errors,
            self.resource_errors,
            self.system_errors,
        });
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Wrap error with additional context
pub fn wrapError(
    allocator: mem.Allocator,
    err: anyerror,
    context: []const u8,
) !ErrorContext {
    var handler = ErrorHandler.init(allocator);
    return try handler.createContext(err, @errorName(err), .error_level, context);
}

/// Convert error to user-friendly message
pub fn errorToMessage(err: anyerror) []const u8 {
    return switch (err) {
        error.SourceNotFound => "The requested source was not found",
        error.FileNotFound => "The requested file was not found",
        error.InvalidRequest => "The request is invalid or malformed",
        error.UnsupportedFileType => "This file type is not supported",
        error.FileTooLarge => "The file is too large to process",
        error.OutOfMemory => "Server is out of memory",
        error.Timeout => "The operation timed out",
        error.ValidationFailed => "Validation failed for the provided data",
        else => "An unexpected error occurred",
    };
}

// ============================================================================
// Tests
// ============================================================================

test "error handler creation" {
    const allocator = std.testing.allocator;
    const handler = ErrorHandler.init(allocator);
    _ = handler;
}

test "error categorization" {
    const allocator = std.testing.allocator;
    const handler = ErrorHandler.init(allocator);
    
    try std.testing.expectEqual(ErrorCategory.client_error, handler.categorizeError(error.InvalidRequest));
    try std.testing.expectEqual(ErrorCategory.validation_error, handler.categorizeError(error.ValidationFailed));
    try std.testing.expectEqual(ErrorCategory.resource_error, handler.categorizeError(error.ResourceNotFound));
    try std.testing.expectEqual(ErrorCategory.system_error, handler.categorizeError(error.OutOfMemory));
}

test "error recoverability" {
    const allocator = std.testing.allocator;
    const handler = ErrorHandler.init(allocator);
    
    try std.testing.expect(handler.isRecoverable(error.InvalidRequest));
    try std.testing.expect(!handler.isRecoverable(error.OutOfMemory));
}

test "error context creation" {
    const allocator = std.testing.allocator;
    var handler = ErrorHandler.init(allocator);
    
    const ctx = try handler.createContext(
        error.SourceNotFound,
        "Test error message",
        .error_level,
        "Additional context",
    );
    defer ctx.deinit(allocator);
    
    try std.testing.expectEqualStrings("SourceNotFound", ctx.error_type);
    try std.testing.expectEqualStrings("Test error message", ctx.message);
    try std.testing.expectEqual(ErrorSeverity.error_level, ctx.severity);
}

test "OData error formatting" {
    const allocator = std.testing.allocator;
    var handler = ErrorHandler.init(allocator);
    
    const error_json = try handler.formatODataError(
        "SourceNotFound",
        "The source could not be found",
        "Source",
        null,
    );
    defer allocator.free(error_json);
    
    try std.testing.expect(mem.indexOf(u8, error_json, "\"error\"") != null);
    try std.testing.expect(mem.indexOf(u8, error_json, "SourceNotFound") != null);
}

test "HTTP error formatting" {
    const allocator = std.testing.allocator;
    var handler = ErrorHandler.init(allocator);
    
    const error_json = try handler.formatHttpError(404, "Not found");
    defer allocator.free(error_json);
    
    try std.testing.expect(mem.indexOf(u8, error_json, "404") != null);
    try std.testing.expect(mem.indexOf(u8, error_json, "Not found") != null);
}

test "error metrics" {
    var metrics = ErrorMetrics{};
    
    metrics.recordError(.client_error);
    metrics.recordError(.server_error);
    metrics.recordError(.validation_error);
    
    try std.testing.expectEqual(@as(u64, 3), metrics.total_errors);
    try std.testing.expectEqual(@as(u64, 1), metrics.client_errors);
    try std.testing.expectEqual(@as(u64, 1), metrics.server_errors);
    try std.testing.expectEqual(@as(u64, 1), metrics.validation_errors);
}

test "error to message conversion" {
    try std.testing.expectEqualStrings("The requested source was not found", errorToMessage(error.SourceNotFound));
    try std.testing.expectEqualStrings("The request is invalid or malformed", errorToMessage(error.InvalidRequest));
}
