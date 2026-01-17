// Mojo SDK - Error Handling
// Day 27: Result types, Option types, error propagation, try/catch, custom errors

const std = @import("std");

// ============================================================================
// Option Type
// ============================================================================

pub fn Option(comptime T: type) type {
    return union(enum) {
        Some: T,
        None,
        
        pub fn init(value: T) Option(T) {
            return Option(T){ .Some = value };
        }
        
        pub fn none() Option(T) {
            return Option(T){ .None = {} };
        }
        
        pub fn isSome(self: *const Option(T)) bool {
            return switch (self.*) {
                .Some => true,
                .None => false,
            };
        }
        
        pub fn isNone(self: *const Option(T)) bool {
            return !self.isSome();
        }
        
        pub fn unwrap(self: *const Option(T)) T {
            return switch (self.*) {
                .Some => |val| val,
                .None => @panic("called unwrap on None"),
            };
        }
        
        pub fn unwrapOr(self: *const Option(T), default: T) T {
            return switch (self.*) {
                .Some => |val| val,
                .None => default,
            };
        }
    };
}

// ============================================================================
// Result Type
// ============================================================================

pub fn Result(comptime T: type, comptime E: type) type {
    return union(enum) {
        Ok: T,
        Err: E,
        
        pub fn ok(value: T) Result(T, E) {
            return Result(T, E){ .Ok = value };
        }
        
        pub fn err(error_val: E) Result(T, E) {
            return Result(T, E){ .Err = error_val };
        }
        
        pub fn isOk(self: *const Result(T, E)) bool {
            return switch (self.*) {
                .Ok => true,
                .Err => false,
            };
        }
        
        pub fn isErr(self: *const Result(T, E)) bool {
            return !self.isOk();
        }
        
        pub fn unwrap(self: *const Result(T, E)) T {
            return switch (self.*) {
                .Ok => |val| val,
                .Err => @panic("called unwrap on Err"),
            };
        }
        
        pub fn unwrapErr(self: *const Result(T, E)) E {
            return switch (self.*) {
                .Ok => @panic("called unwrapErr on Ok"),
                .Err => |e| e,
            };
        }
        
        pub fn unwrapOr(self: *const Result(T, E), default: T) T {
            return switch (self.*) {
                .Ok => |val| val,
                .Err => default,
            };
        }
    };
}

// ============================================================================
// Custom Error Types
// ============================================================================

pub const ErrorKind = enum {
    IoError,
    ParseError,
    TypeError,
    RuntimeError,
    NetworkError,
    Custom,
};

pub const CustomError = struct {
    kind: ErrorKind,
    message: []const u8,
    source: ?*const CustomError,
    
    pub fn init(kind: ErrorKind, message: []const u8) CustomError {
        return CustomError{
            .kind = kind,
            .message = message,
            .source = null,
        };
    }
    
    pub fn withSource(self: CustomError, source: *const CustomError) CustomError {
        return CustomError{
            .kind = self.kind,
            .message = self.message,
            .source = source,
        };
    }
    
    pub fn isKind(self: *const CustomError, kind: ErrorKind) bool {
        return self.kind == kind;
    }
};

// ============================================================================
// Error Propagation
// ============================================================================

pub const ErrorPropagation = struct {
    errors: std.ArrayList(CustomError),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ErrorPropagation {
        return ErrorPropagation{
            .errors = std.ArrayList(CustomError){},
            .allocator = allocator,
        };
    }
    
    pub fn addError(self: *ErrorPropagation, err: CustomError) !void {
        try self.errors.append(self.allocator, err);
    }
    
    pub fn hasErrors(self: *const ErrorPropagation) bool {
        return self.errors.items.len > 0;
    }
    
    pub fn getErrors(self: *const ErrorPropagation) []const CustomError {
        return self.errors.items;
    }
    
    pub fn clear(self: *ErrorPropagation) void {
        self.errors.clearRetainingCapacity();
    }
    
    pub fn deinit(self: *ErrorPropagation) void {
        self.errors.deinit(self.allocator);
    }
};

// ============================================================================
// Try/Catch Mechanism
// ============================================================================

pub const TryBlock = struct {
    has_error: bool,
    error_value: ?CustomError,
    
    pub fn init() TryBlock {
        return TryBlock{
            .has_error = false,
            .error_value = null,
        };
    }
    
    pub fn setError(self: *TryBlock, err: CustomError) void {
        self.has_error = true;
        self.error_value = err;
    }
    
    pub fn getError(self: *const TryBlock) ?CustomError {
        return self.error_value;
    }
    
    pub fn hasError(self: *const TryBlock) bool {
        return self.has_error;
    }
};

pub const CatchHandler = struct {
    error_kind: ?ErrorKind,
    handled: bool,
    
    pub fn init(error_kind: ?ErrorKind) CatchHandler {
        return CatchHandler{
            .error_kind = error_kind,
            .handled = false,
        };
    }
    
    pub fn canHandle(self: *const CatchHandler, err: *const CustomError) bool {
        if (self.error_kind) |kind| {
            return err.kind == kind;
        }
        return true;  // Catch all
    }
    
    pub fn markHandled(self: *CatchHandler) void {
        self.handled = true;
    }
};

// ============================================================================
// Error Recovery
// ============================================================================

pub const RecoveryStrategy = enum {
    Retry,
    Fallback,
    Ignore,
    Propagate,
};

pub const ErrorRecovery = struct {
    strategy: RecoveryStrategy,
    max_retries: usize,
    current_retries: usize,
    fallback_value: ?[]const u8,
    
    pub fn init(strategy: RecoveryStrategy) ErrorRecovery {
        return ErrorRecovery{
            .strategy = strategy,
            .max_retries = 3,
            .current_retries = 0,
            .fallback_value = null,
        };
    }
    
    pub fn withMaxRetries(self: ErrorRecovery, max: usize) ErrorRecovery {
        return ErrorRecovery{
            .strategy = self.strategy,
            .max_retries = max,
            .current_retries = self.current_retries,
            .fallback_value = self.fallback_value,
        };
    }
    
    pub fn withFallback(self: ErrorRecovery, fallback: []const u8) ErrorRecovery {
        return ErrorRecovery{
            .strategy = self.strategy,
            .max_retries = self.max_retries,
            .current_retries = self.current_retries,
            .fallback_value = fallback,
        };
    }
    
    pub fn shouldRetry(self: *const ErrorRecovery) bool {
        return self.strategy == .Retry and self.current_retries < self.max_retries;
    }
    
    pub fn incrementRetries(self: *ErrorRecovery) void {
        self.current_retries += 1;
    }
    
    pub fn canRecover(self: *const ErrorRecovery) bool {
        return switch (self.strategy) {
            .Retry => self.shouldRetry(),
            .Fallback => self.fallback_value != null,
            .Ignore => true,
            .Propagate => false,
        };
    }
};

// ============================================================================
// Error Context
// ============================================================================

pub const ErrorContext = struct {
    propagation: ErrorPropagation,
    try_blocks: std.ArrayList(TryBlock),
    recovery: ErrorRecovery,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ErrorContext {
        return ErrorContext{
            .propagation = ErrorPropagation.init(allocator),
            .try_blocks = std.ArrayList(TryBlock){},
            .recovery = ErrorRecovery.init(.Propagate),
            .allocator = allocator,
        };
    }
    
    pub fn enterTry(self: *ErrorContext) !void {
        try self.try_blocks.append(self.allocator, TryBlock.init());
    }
    
    pub fn exitTry(self: *ErrorContext) ?CustomError {
        if (self.try_blocks.items.len > 0) {
            const block = self.try_blocks.pop();
            return block.getError();
        }
        return null;
    }
    
    pub fn recordError(self: *ErrorContext, err: CustomError) !void {
        // Add to current try block if exists
        if (self.try_blocks.items.len > 0) {
            const idx = self.try_blocks.items.len - 1;
            self.try_blocks.items[idx].setError(err);
        }
        
        // Add to propagation
        try self.propagation.addError(err);
    }
    
    pub fn hasErrors(self: *const ErrorContext) bool {
        return self.propagation.hasErrors();
    }
    
    pub fn deinit(self: *ErrorContext) void {
        self.propagation.deinit();
        self.try_blocks.deinit(self.allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "errors: option some" {
    const opt = Option(i32).init(42);
    try std.testing.expect(opt.isSome());
    try std.testing.expectEqual(@as(i32, 42), opt.unwrap());
}

test "errors: option none" {
    const opt = Option(i32).none();
    try std.testing.expect(opt.isNone());
    try std.testing.expectEqual(@as(i32, 0), opt.unwrapOr(0));
}

test "errors: result ok" {
    const res = Result(i32, []const u8).ok(42);
    try std.testing.expect(res.isOk());
    try std.testing.expectEqual(@as(i32, 42), res.unwrap());
}

test "errors: result err" {
    const res = Result(i32, []const u8).err("error");
    try std.testing.expect(res.isErr());
    try std.testing.expect(std.mem.eql(u8, "error", res.unwrapErr()));
}

test "errors: custom error" {
    const err = CustomError.init(.IoError, "file not found");
    try std.testing.expect(err.isKind(.IoError));
}

test "errors: error propagation" {
    const allocator = std.testing.allocator;
    var prop = ErrorPropagation.init(allocator);
    defer prop.deinit();
    
    const err = CustomError.init(.ParseError, "syntax error");
    try prop.addError(err);
    
    try std.testing.expect(prop.hasErrors());
}

test "errors: try block" {
    var try_block = TryBlock.init();
    try std.testing.expect(!try_block.hasError());
    
    const err = CustomError.init(.RuntimeError, "runtime error");
    try_block.setError(err);
    
    try std.testing.expect(try_block.hasError());
}

test "errors: catch handler" {
    const handler = CatchHandler.init(.IoError);
    const err = CustomError.init(.IoError, "io error");
    
    try std.testing.expect(handler.canHandle(&err));
}

test "errors: recovery strategy" {
    var recovery = ErrorRecovery.init(.Retry);
    recovery = recovery.withMaxRetries(5);
    
    try std.testing.expect(recovery.shouldRetry());
    recovery.incrementRetries();
    try std.testing.expectEqual(@as(usize, 1), recovery.current_retries);
}

test "errors: error context" {
    const allocator = std.testing.allocator;
    var ctx = ErrorContext.init(allocator);
    defer ctx.deinit();
    
    try ctx.enterTry();
    const err = CustomError.init(.TypeError, "type mismatch");
    try ctx.recordError(err);
    
    try std.testing.expect(ctx.hasErrors());
}
