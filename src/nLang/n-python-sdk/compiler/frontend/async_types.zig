// Mojo Async Types - Day 101
// Defines Future[T], Task, and related async type system

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const types = @import("types.zig");

// ============================================================================
// Future Type
// ============================================================================

/// Represents a Future[T] type - a value that will be available in the future
pub const FutureType = struct {
    inner_type: types.Type,
    is_ready: bool = false,
    error_type: ?types.Type = null,

    pub fn init(inner: types.Type) FutureType {
        return .{
            .inner_type = inner,
            .is_ready = false,
            .error_type = null,
        };
    }

    pub fn withError(inner: types.Type, err_type: types.Type) FutureType {
        return .{
            .inner_type = inner,
            .is_ready = false,
            .error_type = err_type,
        };
    }

    pub fn isResult(self: FutureType) bool {
        return self.error_type != null;
    }

    pub fn format(self: FutureType, allocator: Allocator) ![]const u8 {
        if (self.error_type) |err_type| {
            return std.fmt.allocPrint(allocator, "Future[Result[{s}, {s}]]", .{
                @tagName(self.inner_type),
                @tagName(err_type),
            });
        }
        return std.fmt.allocPrint(allocator, "Future[{s}]", .{@tagName(self.inner_type)});
    }
};

// ============================================================================
// Task Type
// ============================================================================

/// Represents a Task - a unit of async work
pub const TaskType = struct {
    return_type: types.Type,
    priority: Priority,
    cancellable: bool,
    name: ?[]const u8,

    pub const Priority = enum(u8) {
        Low = 0,
        Normal = 1,
        High = 2,
        Critical = 3,
    };

    pub fn init(return_type: types.Type) TaskType {
        return .{
            .return_type = return_type,
            .priority = .Normal,
            .cancellable = true,
            .name = null,
        };
    }

    pub fn withPriority(return_type: types.Type, priority: Priority) TaskType {
        return .{
            .return_type = return_type,
            .priority = priority,
            .cancellable = true,
            .name = null,
        };
    }

    pub fn format(self: TaskType, allocator: Allocator) ![]const u8 {
        if (self.name) |name| {
            return std.fmt.allocPrint(allocator, "Task[{s}](\"{s}\")", .{
                @tagName(self.return_type),
                name,
            });
        }
        return std.fmt.allocPrint(allocator, "Task[{s}]", .{@tagName(self.return_type)});
    }
};

// ============================================================================
// Promise Type
// ============================================================================

/// Represents a Promise - a writable Future
pub const PromiseType = struct {
    future_type: FutureType,
    is_fulfilled: bool,
    is_rejected: bool,

    pub fn init(inner: types.Type) PromiseType {
        return .{
            .future_type = FutureType.init(inner),
            .is_fulfilled = false,
            .is_rejected = false,
        };
    }

    pub fn isSettled(self: PromiseType) bool {
        return self.is_fulfilled or self.is_rejected;
    }

    pub fn format(self: PromiseType, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "Promise[{s}]", .{@tagName(self.future_type.inner_type)});
    }
};

// ============================================================================
// Stream Type
// ============================================================================

/// Represents a Stream[T] - an async sequence of values
pub const StreamType = struct {
    element_type: types.Type,
    is_bounded: bool,
    buffer_size: ?usize,

    pub fn init(element: types.Type) StreamType {
        return .{
            .element_type = element,
            .is_bounded = false,
            .buffer_size = null,
        };
    }

    pub fn withBuffer(element: types.Type, size: usize) StreamType {
        return .{
            .element_type = element,
            .is_bounded = true,
            .buffer_size = size,
        };
    }

    pub fn format(self: StreamType, allocator: Allocator) ![]const u8 {
        if (self.buffer_size) |size| {
            return std.fmt.allocPrint(allocator, "Stream[{s}, buffer={d}]", .{
                @tagName(self.element_type),
                size,
            });
        }
        return std.fmt.allocPrint(allocator, "Stream[{s}]", .{@tagName(self.element_type)});
    }
};

// ============================================================================
// Channel Type
// ============================================================================

/// Represents a Channel[T] - bidirectional async communication
pub const ChannelType = struct {
    message_type: types.Type,
    capacity: usize,
    is_mpsc: bool, // Multiple Producer Single Consumer
    is_spmc: bool, // Single Producer Multiple Consumer

    pub const ChannelKind = enum {
        SPSC, // Single Producer Single Consumer
        MPSC, // Multiple Producer Single Consumer
        SPMC, // Single Producer Multiple Consumer
        MPMC, // Multiple Producer Multiple Consumer
    };

    pub fn init(message_type: types.Type, capacity: usize) ChannelType {
        return .{
            .message_type = message_type,
            .capacity = capacity,
            .is_mpsc = false,
            .is_spmc = false,
        };
    }

    pub fn withKind(message_type: types.Type, capacity: usize, kind: ChannelKind) ChannelType {
        return .{
            .message_type = message_type,
            .capacity = capacity,
            .is_mpsc = kind == .MPSC or kind == .MPMC,
            .is_spmc = kind == .SPMC or kind == .MPMC,
        };
    }

    pub fn getKind(self: ChannelType) ChannelKind {
        if (self.is_mpsc and self.is_spmc) return .MPMC;
        if (self.is_mpsc) return .MPSC;
        if (self.is_spmc) return .SPMC;
        return .SPSC;
    }

    pub fn format(self: ChannelType, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "Channel[{s}, capacity={d}]", .{
            @tagName(self.message_type),
            self.capacity,
        });
    }
};

// ============================================================================
// Async Type Registry
// ============================================================================

/// Registry for async types
pub const AsyncTypeRegistry = struct {
    allocator: Allocator,
    futures: ArrayList(FutureType),
    tasks: ArrayList(TaskType),
    promises: ArrayList(PromiseType),
    streams: ArrayList(StreamType),
    channels: ArrayList(ChannelType),

    pub fn init(allocator: Allocator) AsyncTypeRegistry {
        return .{
            .allocator = allocator,
            .futures = ArrayList(FutureType).init(allocator),
            .tasks = ArrayList(TaskType).init(allocator),
            .promises = ArrayList(PromiseType).init(allocator),
            .streams = ArrayList(StreamType).init(allocator),
            .channels = ArrayList(ChannelType).init(allocator),
        };
    }

    pub fn deinit(self: *AsyncTypeRegistry) void {
        self.futures.deinit();
        self.tasks.deinit();
        self.promises.deinit();
        self.streams.deinit();
        self.channels.deinit();
    }

    pub fn registerFuture(self: *AsyncTypeRegistry, future: FutureType) !void {
        try self.futures.append(future);
    }

    pub fn registerTask(self: *AsyncTypeRegistry, task: TaskType) !void {
        try self.tasks.append(task);
    }

    pub fn registerPromise(self: *AsyncTypeRegistry, promise: PromiseType) !void {
        try self.promises.append(promise);
    }

    pub fn registerStream(self: *AsyncTypeRegistry, stream: StreamType) !void {
        try self.streams.append(stream);
    }

    pub fn registerChannel(self: *AsyncTypeRegistry, channel: ChannelType) !void {
        try self.channels.append(channel);
    }

    pub fn findFuture(self: *AsyncTypeRegistry, inner: types.Type) ?*FutureType {
        for (self.futures.items) |*future| {
            if (std.meta.eql(future.inner_type, inner)) {
                return future;
            }
        }
        return null;
    }
};

// ============================================================================
// Async Type Checker
// ============================================================================

/// Type checker for async expressions
pub const AsyncTypeChecker = struct {
    registry: AsyncTypeRegistry,
    allocator: Allocator,

    pub fn init(allocator: Allocator) AsyncTypeChecker {
        return .{
            .registry = AsyncTypeRegistry.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AsyncTypeChecker) void {
        self.registry.deinit();
    }

    /// Check if a type is awaitable
    pub fn isAwaitable(self: *AsyncTypeChecker, type_: types.Type) bool {
        _ = self;
        _ = type_;
        // TODO: Implement awaitable check
        return true;
    }

    /// Infer the type of an await expression
    pub fn inferAwaitType(self: *AsyncTypeChecker, future_type: FutureType) types.Type {
        _ = self;
        return future_type.inner_type;
    }

    /// Check if two async types are compatible
    pub fn areCompatible(self: *AsyncTypeChecker, a: types.Type, b: types.Type) bool {
        _ = self;
        return std.meta.eql(a, b);
    }

    /// Convert a synchronous type to its async equivalent
    pub fn toAsyncType(self: *AsyncTypeChecker, sync_type: types.Type) !FutureType {
        const future = FutureType.init(sync_type);
        try self.registry.registerFuture(future);
        return future;
    }
};

// ============================================================================
// Async Type Builder
// ============================================================================

/// Builder for constructing complex async types
pub const AsyncTypeBuilder = struct {
    allocator: Allocator,
    current_type: ?AsyncTypeVariant = null,

    pub const AsyncTypeVariant = union(enum) {
        future: FutureType,
        task: TaskType,
        promise: PromiseType,
        stream: StreamType,
        channel: ChannelType,
    };

    pub fn init(allocator: Allocator) AsyncTypeBuilder {
        return .{
            .allocator = allocator,
            .current_type = null,
        };
    }

    pub fn future(self: *AsyncTypeBuilder, inner: types.Type) *AsyncTypeBuilder {
        self.current_type = .{ .future = FutureType.init(inner) };
        return self;
    }

    pub fn task(self: *AsyncTypeBuilder, return_type: types.Type) *AsyncTypeBuilder {
        self.current_type = .{ .task = TaskType.init(return_type) };
        return self;
    }

    pub fn promise(self: *AsyncTypeBuilder, inner: types.Type) *AsyncTypeBuilder {
        self.current_type = .{ .promise = PromiseType.init(inner) };
        return self;
    }

    pub fn stream(self: *AsyncTypeBuilder, element: types.Type) *AsyncTypeBuilder {
        self.current_type = .{ .stream = StreamType.init(element) };
        return self;
    }

    pub fn channel(self: *AsyncTypeBuilder, message: types.Type, capacity: usize) *AsyncTypeBuilder {
        self.current_type = .{ .channel = ChannelType.init(message, capacity) };
        return self;
    }

    pub fn withError(self: *AsyncTypeBuilder, err_type: types.Type) *AsyncTypeBuilder {
        if (self.current_type) |*variant| {
            switch (variant.*) {
                .future => |*f| f.error_type = err_type,
                else => {},
            }
        }
        return self;
    }

    pub fn withPriority(self: *AsyncTypeBuilder, priority: TaskType.Priority) *AsyncTypeBuilder {
        if (self.current_type) |*variant| {
            switch (variant.*) {
                .task => |*t| t.priority = priority,
                else => {},
            }
        }
        return self;
    }

    pub fn build(self: *AsyncTypeBuilder) ?AsyncTypeVariant {
        return self.current_type;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "future type creation" {
    const future = FutureType.init(types.Type.i32_type);
    try std.testing.expect(!future.is_ready);
    try std.testing.expect(future.error_type == null);
    try std.testing.expect(!future.isResult());
}

test "future with error type" {
    const future = FutureType.withError(types.Type.i32_type, types.Type.string_type);
    try std.testing.expect(future.error_type != null);
    try std.testing.expect(future.isResult());
}

test "task type creation" {
    const task = TaskType.init(types.Type.void_type);
    try std.testing.expectEqual(TaskType.Priority.Normal, task.priority);
    try std.testing.expect(task.cancellable);
}

test "task with priority" {
    const task = TaskType.withPriority(types.Type.i64_type, .High);
    try std.testing.expectEqual(TaskType.Priority.High, task.priority);
}

test "promise type creation" {
    const promise = PromiseType.init(types.Type.f32_type);
    try std.testing.expect(!promise.isSettled());
    try std.testing.expect(!promise.is_fulfilled);
    try std.testing.expect(!promise.is_rejected);
}

test "stream type creation" {
    const stream = StreamType.init(types.Type.u8_type);
    try std.testing.expect(!stream.is_bounded);
    try std.testing.expect(stream.buffer_size == null);
}

test "stream with buffer" {
    const stream = StreamType.withBuffer(types.Type.i32_type, 100);
    try std.testing.expect(stream.is_bounded);
    try std.testing.expectEqual(@as(?usize, 100), stream.buffer_size);
}

test "channel type creation" {
    const channel = ChannelType.init(types.Type.string_type, 10);
    try std.testing.expectEqual(@as(usize, 10), channel.capacity);
    try std.testing.expectEqual(ChannelType.ChannelKind.SPSC, channel.getKind());
}

test "channel with kind" {
    const channel = ChannelType.withKind(types.Type.i32_type, 5, .MPMC);
    try std.testing.expectEqual(ChannelType.ChannelKind.MPMC, channel.getKind());
    try std.testing.expect(channel.is_mpsc);
    try std.testing.expect(channel.is_spmc);
}

test "async type registry" {
    const allocator = std.testing.allocator;

    var registry = AsyncTypeRegistry.init(allocator);
    defer registry.deinit();

    const future = FutureType.init(types.Type.i32_type);
    try registry.registerFuture(future);

    try std.testing.expectEqual(@as(usize, 1), registry.futures.items.len);
}

test "async type checker" {
    const allocator = std.testing.allocator;

    var checker = AsyncTypeChecker.init(allocator);
    defer checker.deinit();

    try std.testing.expect(checker.isAwaitable(types.Type.i32_type));
}

test "async type builder - future" {
    const allocator = std.testing.allocator;

    var builder = AsyncTypeBuilder.init(allocator);
    _ = builder.future(types.Type.i32_type);

    const result = builder.build();
    try std.testing.expect(result != null);
    try std.testing.expect(result.? == .future);
}

test "async type builder - task with priority" {
    const allocator = std.testing.allocator;

    var builder = AsyncTypeBuilder.init(allocator);
    _ = builder.task(types.Type.void_type).withPriority(.High);

    const result = builder.build();
    try std.testing.expect(result != null);
    if (result) |r| {
        switch (r) {
            .task => |t| try std.testing.expectEqual(TaskType.Priority.High, t.priority),
            else => try std.testing.expect(false),
        }
    }
}

test "async type builder - future with error" {
    const allocator = std.testing.allocator;

    var builder = AsyncTypeBuilder.init(allocator);
    _ = builder.future(types.Type.i32_type).withError(types.Type.string_type);

    const result = builder.build();
    try std.testing.expect(result != null);
    if (result) |r| {
        switch (r) {
            .future => |f| {
                try std.testing.expect(f.error_type != null);
                try std.testing.expect(f.isResult());
            },
            else => try std.testing.expect(false),
        }
    }
}
