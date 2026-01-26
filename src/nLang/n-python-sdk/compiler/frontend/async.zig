// Mojo Async/Await System - Day 101
// Provides async/await syntax support in the compiler frontend

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ast = @import("ast.zig");
const types = @import("types.zig");

// ============================================================================
// Async Function Types
// ============================================================================

/// Calling convention for async functions
pub const AsyncCallingConvention = enum {
    Async, // Standard async function
    Sync, // Synchronous function
    AsyncGen, // Async generator (for streams)
};

/// Represents an async function declaration
pub const AsyncFunction = struct {
    name: []const u8,
    params: ArrayList(Param),
    return_type: AsyncReturnType,
    body: ?*ast.BlockStmt,
    calling_conv: AsyncCallingConvention,
    is_unsafe: bool,

    pub const Param = struct {
        name: []const u8,
        type_: types.Type,
        is_mut: bool,
    };

    pub fn init(allocator: Allocator, name: []const u8) !*AsyncFunction {
        const func = try allocator.create(AsyncFunction);
        func.* = .{
            .name = try allocator.dupe(u8, name),
            .params = ArrayList(Param).init(allocator),
            .return_type = .{ .inner = types.Type.void_type },
            .body = null,
            .calling_conv = .Async,
            .is_unsafe = false,
        };
        return func;
    }

    pub fn deinit(self: *AsyncFunction, allocator: Allocator) void {
        allocator.free(self.name);
        for (self.params.items) |*param| {
            allocator.free(param.name);
        }
        self.params.deinit();
        allocator.destroy(self);
    }
};

/// Return type wrapper for async functions
pub const AsyncReturnType = struct {
    inner: types.Type,
    is_result: bool = false,
    error_type: ?types.Type = null,

    pub fn isFuture(self: AsyncReturnType) bool {
        return self.inner.isFutureType();
    }
};

// ============================================================================
// Await Expression
// ============================================================================

/// Represents an await expression
pub const AwaitExpr = struct {
    expr: *ast.Expr,
    location: SourceLocation,

    pub const SourceLocation = struct {
        line: usize,
        column: usize,
        file: []const u8,
    };

    pub fn init(allocator: Allocator, expr: *ast.Expr) !*AwaitExpr {
        const await_expr = try allocator.create(AwaitExpr);
        await_expr.* = .{
            .expr = expr,
            .location = .{ .line = 0, .column = 0, .file = "" },
        };
        return await_expr;
    }

    pub fn deinit(self: *AwaitExpr, allocator: Allocator) void {
        allocator.destroy(self);
    }

    /// Validate that the awaited expression returns a Future
    pub fn validate(self: *AwaitExpr, type_checker: anytype) !void {
        _ = self;
        _ = type_checker;
        // TODO: Implement type validation
    }
};

// ============================================================================
// Async Block
// ============================================================================

/// Represents an async block: async { ... }
pub const AsyncBlock = struct {
    body: *ast.BlockStmt,
    captures: ArrayList(Capture),
    return_type: types.Type,

    pub const Capture = struct {
        name: []const u8,
        type_: types.Type,
        is_move: bool,
    };

    pub fn init(allocator: Allocator, body: *ast.BlockStmt) !*AsyncBlock {
        const block = try allocator.create(AsyncBlock);
        block.* = .{
            .body = body,
            .captures = ArrayList(Capture).init(allocator),
            .return_type = types.Type.void_type,
        };
        return block;
    }

    pub fn deinit(self: *AsyncBlock, allocator: Allocator) void {
        for (self.captures.items) |*capture| {
            allocator.free(capture.name);
        }
        self.captures.deinit();
        allocator.destroy(self);
    }

    pub fn addCapture(self: *AsyncBlock, name: []const u8, type_: types.Type, is_move: bool) !void {
        try self.captures.append(.{
            .name = name,
            .type_ = type_,
            .is_move = is_move,
        });
    }
};

// ============================================================================
// Async State Machine
// ============================================================================

/// Represents the state machine generated for an async function
pub const AsyncStateMachine = struct {
    function: *AsyncFunction,
    states: ArrayList(State),
    current_state: usize,
    allocator: Allocator,

    pub const State = struct {
        id: usize,
        await_points: ArrayList(*AwaitExpr),
        basic_blocks: ArrayList(*ast.BlockStmt),
        transitions: ArrayList(Transition),
    };

    pub const Transition = struct {
        from_state: usize,
        to_state: usize,
        condition: ?*ast.Expr,
    };

    pub fn init(allocator: Allocator, function: *AsyncFunction) !*AsyncStateMachine {
        const machine = try allocator.create(AsyncStateMachine);
        machine.* = .{
            .function = function,
            .states = ArrayList(State).init(allocator),
            .current_state = 0,
            .allocator = allocator,
        };
        return machine;
    }

    pub fn deinit(self: *AsyncStateMachine) void {
        for (self.states.items) |*state| {
            state.await_points.deinit();
            state.basic_blocks.deinit();
            state.transitions.deinit();
        }
        self.states.deinit();
        self.allocator.destroy(self);
    }

    /// Build state machine from async function
    pub fn build(self: *AsyncStateMachine) !void {
        if (self.function.body) |body| {
            try self.analyzeBlock(body);
        }
    }

    fn analyzeBlock(self: *AsyncStateMachine, block: *ast.BlockStmt) !void {
        _ = self;
        _ = block;
        // TODO: Implement block analysis and state extraction
    }

    pub fn addState(self: *AsyncStateMachine) !usize {
        const id = self.states.items.len;
        try self.states.append(.{
            .id = id,
            .await_points = ArrayList(*AwaitExpr).init(self.allocator),
            .basic_blocks = ArrayList(*ast.BlockStmt).init(self.allocator),
            .transitions = ArrayList(Transition).init(self.allocator),
        });
        return id;
    }

    pub fn addTransition(self: *AsyncStateMachine, from: usize, to: usize, condition: ?*ast.Expr) !void {
        if (from >= self.states.items.len) return error.InvalidState;
        try self.states.items[from].transitions.append(.{
            .from_state = from,
            .to_state = to,
            .condition = condition,
        });
    }
};

// ============================================================================
// Async Context
// ============================================================================

/// Tracks async context during compilation
pub const AsyncContext = struct {
    allocator: Allocator,
    current_function: ?*AsyncFunction,
    await_depth: usize,
    in_async_block: bool,
    state_machines: ArrayList(*AsyncStateMachine),

    pub fn init(allocator: Allocator) AsyncContext {
        return .{
            .allocator = allocator,
            .current_function = null,
            .await_depth = 0,
            .in_async_block = false,
            .state_machines = ArrayList(*AsyncStateMachine).init(allocator),
        };
    }

    pub fn deinit(self: *AsyncContext) void {
        for (self.state_machines.items) |machine| {
            machine.deinit();
        }
        self.state_machines.deinit();
    }

    pub fn enterAsyncFunction(self: *AsyncContext, func: *AsyncFunction) !void {
        self.current_function = func;
        const machine = try AsyncStateMachine.init(self.allocator, func);
        try self.state_machines.append(machine);
    }

    pub fn exitAsyncFunction(self: *AsyncContext) void {
        self.current_function = null;
    }

    pub fn enterAwait(self: *AsyncContext) void {
        self.await_depth += 1;
    }

    pub fn exitAwait(self: *AsyncContext) void {
        if (self.await_depth > 0) {
            self.await_depth -= 1;
        }
    }

    pub fn isInAsyncContext(self: *AsyncContext) bool {
        return self.current_function != null or self.in_async_block;
    }

    pub fn canAwait(self: *AsyncContext) bool {
        return self.isInAsyncContext();
    }
};

// ============================================================================
// Async Analyzer
// ============================================================================

/// Analyzes async/await usage and validates correctness
pub const AsyncAnalyzer = struct {
    allocator: Allocator,
    context: AsyncContext,
    errors: ArrayList(AsyncError),

    pub const AsyncError = struct {
        message: []const u8,
        location: AwaitExpr.SourceLocation,
        kind: ErrorKind,

        pub const ErrorKind = enum {
            AwaitOutsideAsync,
            InvalidAwaitTarget,
            AsyncInSyncFunction,
            CircularAsync,
            UnsafeAsyncOperation,
        };
    };

    pub fn init(allocator: Allocator) AsyncAnalyzer {
        return .{
            .allocator = allocator,
            .context = AsyncContext.init(allocator),
            .errors = ArrayList(AsyncError).init(allocator),
        };
    }

    pub fn deinit(self: *AsyncAnalyzer) void {
        self.context.deinit();
        for (self.errors.items) |err| {
            self.allocator.free(err.message);
        }
        self.errors.deinit();
    }

    /// Validate async function
    pub fn validateFunction(self: *AsyncAnalyzer, func: *AsyncFunction) !bool {
        try self.context.enterAsyncFunction(func);
        defer self.context.exitAsyncFunction();

        if (func.body) |body| {
            try self.validateBlock(body);
        }

        return self.errors.items.len == 0;
    }

    fn validateBlock(self: *AsyncAnalyzer, block: *ast.BlockStmt) !void {
        _ = self;
        _ = block;
        // TODO: Implement block validation
    }

    pub fn addError(self: *AsyncAnalyzer, kind: AsyncError.ErrorKind, message: []const u8, location: AwaitExpr.SourceLocation) !void {
        try self.errors.append(.{
            .message = try self.allocator.dupe(u8, message),
            .location = location,
            .kind = kind,
        });
    }

    pub fn hasErrors(self: *AsyncAnalyzer) bool {
        return self.errors.items.len > 0;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "async function creation" {
    const allocator = std.testing.allocator;

    var func = try AsyncFunction.init(allocator, "test_async");
    defer func.deinit(allocator);

    try std.testing.expectEqualStrings("test_async", func.name);
    try std.testing.expectEqual(AsyncCallingConvention.Async, func.calling_conv);
}

test "async context management" {
    const allocator = std.testing.allocator;

    var context = AsyncContext.init(allocator);
    defer context.deinit();

    try std.testing.expect(!context.isInAsyncContext());
    try std.testing.expect(!context.canAwait());

    var func = try AsyncFunction.init(allocator, "test");
    defer func.deinit(allocator);

    try context.enterAsyncFunction(func);
    try std.testing.expect(context.isInAsyncContext());
    try std.testing.expect(context.canAwait());

    context.exitAsyncFunction();
    try std.testing.expect(!context.isInAsyncContext());
}

test "async analyzer validation" {
    const allocator = std.testing.allocator;

    var analyzer = AsyncAnalyzer.init(allocator);
    defer analyzer.deinit();

    try std.testing.expect(!analyzer.hasErrors());
}

test "state machine creation" {
    const allocator = std.testing.allocator;

    var func = try AsyncFunction.init(allocator, "state_test");
    defer func.deinit(allocator);

    var machine = try AsyncStateMachine.init(allocator, func);
    defer machine.deinit();

    const state_id = try machine.addState();
    try std.testing.expectEqual(@as(usize, 0), state_id);
}

test "await depth tracking" {
    const allocator = std.testing.allocator;

    var context = AsyncContext.init(allocator);
    defer context.deinit();

    try std.testing.expectEqual(@as(usize, 0), context.await_depth);

    context.enterAwait();
    try std.testing.expectEqual(@as(usize, 1), context.await_depth);

    context.enterAwait();
    try std.testing.expectEqual(@as(usize, 2), context.await_depth);

    context.exitAwait();
    try std.testing.expectEqual(@as(usize, 1), context.await_depth);

    context.exitAwait();
    try std.testing.expectEqual(@as(usize, 0), context.await_depth);
}

test "async block captures" {
    const allocator = std.testing.allocator;

    const block_stmt = try allocator.create(ast.BlockStmt);
    defer allocator.destroy(block_stmt);

    var async_block = try AsyncBlock.init(allocator, block_stmt);
    defer async_block.deinit(allocator);

    try async_block.addCapture("x", types.Type.i32_type, false);
    try async_block.addCapture("y", types.Type.i64_type, true);

    try std.testing.expectEqual(@as(usize, 2), async_block.captures.items.len);
}
