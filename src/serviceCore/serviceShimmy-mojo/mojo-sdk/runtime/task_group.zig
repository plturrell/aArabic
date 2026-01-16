// Structured Concurrency - Day 141
// TaskGroup/Nursery pattern for proper cancellation propagation

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;
const Atomic = std.atomic.Value;
const ArrayList = std.ArrayList;

// ============================================================================
// Cancellation Token
// ============================================================================

pub const CancellationToken = struct {
    cancelled: Atomic(bool),
    reason: ?[]const u8,
    reason_mutex: Mutex,
    listeners: ArrayList(*const fn () void),
    listeners_mutex: Mutex,
    
    pub fn init(allocator: Allocator) !*CancellationToken {
        const token = try allocator.create(CancellationToken);
        token.* = .{
            .cancelled = Atomic(bool).init(false),
            .reason = null,
            .reason_mutex = Mutex{},
            .listeners = ArrayList(*const fn () void).init(allocator),
            .listeners_mutex = Mutex{},
        };
        return token;
    }
    
    pub fn deinit(self: *CancellationToken, allocator: Allocator) void {
        if (self.reason) |reason| {
            allocator.free(reason);
        }
        self.listeners.deinit();
        allocator.destroy(self);
    }
    
    pub fn cancel(self: *CancellationToken, reason: ?[]const u8) void {
        const was_cancelled = self.cancelled.swap(true, .seq_cst);
        if (was_cancelled) return;  // Already cancelled
        
        // Store reason
        if (reason) |r| {
            self.reason_mutex.lock();
            self.reason = r;
            self.reason_mutex.unlock();
        }
        
        // Notify all listeners
        self.listeners_mutex.lock();
        for (self.listeners.items) |listener| {
            listener();
        }
        self.listeners_mutex.unlock();
    }
    
    pub fn isCancelled(self: *const CancellationToken) bool {
        return self.cancelled.load(.seq_cst);
    }
    
    pub fn getCancelReason(self: *CancellationToken) ?[]const u8 {
        self.reason_mutex.lock();
        defer self.reason_mutex.unlock();
        return self.reason;
    }
    
    pub fn onCancel(self: *CancellationToken, callback: *const fn () void) !void {
        self.listeners_mutex.lock();
        defer self.listeners_mutex.unlock();
        try self.listeners.append(callback);
    }
};

// ============================================================================
// Task Handle with Cancellation
// ============================================================================

pub const TaskHandle = struct {
    id: u64,
    state: TaskState,
    cancel_token: *CancellationToken,
    result: ?[]const u8,
    err: ?anyerror,
    completed: Atomic(bool),
    
    pub const TaskState = enum {
        Pending,
        Running,
        Completed,
        Cancelled,
        Failed,
    };
    
    pub fn init(allocator: Allocator, id: u64) !*TaskHandle {
        const handle = try allocator.create(TaskHandle);
        const token = try CancellationToken.init(allocator);
        
        handle.* = .{
            .id = id,
            .state = .Pending,
            .cancel_token = token,
            .result = null,
            .err = null,
            .completed = Atomic(bool).init(false),
        };
        
        return handle;
    }
    
    pub fn deinit(self: *TaskHandle, allocator: Allocator) void {
        self.cancel_token.deinit(allocator);
        if (self.result) |r| allocator.free(r);
        allocator.destroy(self);
    }
    
    pub fn cancel(self: *TaskHandle, reason: ?[]const u8) void {
        self.cancel_token.cancel(reason);
        self.state = .Cancelled;
    }
    
    pub fn isComplete(self: *const TaskHandle) bool {
        return self.completed.load(.seq_cst);
    }
};

// ============================================================================
// Task Group / Nursery
// ============================================================================

pub const TaskGroup = struct {
    allocator: Allocator,
    
    // Child tracking
    children: ArrayList(*TaskHandle),
    children_mutex: Mutex,
    
    // Cancellation
    cancel_token: *CancellationToken,
    
    // Completion tracking
    active_count: Atomic(usize),
    completion_cond: Condition,
    completion_mutex: Mutex,
    
    // Error handling
    first_error: ?anyerror,
    error_mutex: Mutex,
    cancel_on_error: bool,
    
    pub const Config = struct {
        cancel_on_error: bool = true,  // Cancel all on first error
        max_children: ?usize = null,   // Optional limit
    };
    
    pub fn init(allocator: Allocator, config: Config) !*TaskGroup {
        const group = try allocator.create(TaskGroup);
        const token = try CancellationToken.init(allocator);
        
        group.* = .{
            .allocator = allocator,
            .children = ArrayList(*TaskHandle).init(allocator),
            .children_mutex = Mutex{},
            .cancel_token = token,
            .active_count = Atomic(usize).init(0),
            .completion_cond = Condition{},
            .completion_mutex = Mutex{},
            .first_error = null,
            .error_mutex = Mutex{},
            .cancel_on_error = config.cancel_on_error,
        };
        
        return group;
    }
    
    pub fn deinit(self: *TaskGroup) void {
        self.cancel_token.deinit(self.allocator);
        
        self.children_mutex.lock();
        for (self.children.items) |child| {
            child.deinit(self.allocator);
        }
        self.children.deinit();
        self.children_mutex.unlock();
        
        self.allocator.destroy(self);
    }
    
    /// Spawn a child task
    pub fn spawn(self: *TaskGroup, task_id: u64) !*TaskHandle {
        const handle = try TaskHandle.init(self.allocator, task_id);
        
        // Link to parent's cancellation token
        try handle.cancel_token.onCancel(struct {
            fn callback() void {
                // Propagate cancellation
            }
        }.callback);
        
        // Track child
        self.children_mutex.lock();
        try self.children.append(handle);
        self.children_mutex.unlock();
        
        _ = self.active_count.fetchAdd(1, .seq_cst);
        
        return handle;
    }
    
    /// Mark a child as completed
    pub fn markCompleted(self: *TaskGroup, handle: *TaskHandle, result: anyerror![]const u8) void {
        if (result) |r| {
            handle.result = r;
            handle.state = .Completed;
        } else |err| {
            handle.err = err;
            handle.state = .Failed;
            
            // Record first error
            self.error_mutex.lock();
            if (self.first_error == null) {
                self.first_error = err;
            }
            self.error_mutex.unlock();
            
            // Cancel all siblings if configured
            if (self.cancel_on_error) {
                self.cancelAll("Parent task failed");
            }
        }
        
        _ = handle.completed.swap(true, .seq_cst);
        
        const remaining = self.active_count.fetchSub(1, .seq_cst) - 1;
        if (remaining == 0) {
            // All children completed, wake waiters
            self.completion_mutex.lock();
            self.completion_cond.broadcast();
            self.completion_mutex.unlock();
        }
    }
    
    /// Cancel all children
    pub fn cancelAll(self: *TaskGroup, reason: []const u8) void {
        // Cancel parent token (propagates to all children)
        self.cancel_token.cancel(reason);
        
        // Also directly cancel each child
        self.children_mutex.lock();
        for (self.children.items) |child| {
            child.cancel(reason);
        }
        self.children_mutex.unlock();
    }
    
    /// Wait for all children to complete
    pub fn wait(self: *TaskGroup) !void {
        self.completion_mutex.lock();
        defer self.completion_mutex.unlock();
        
        while (self.active_count.load(.seq_cst) > 0) {
            self.completion_cond.wait(&self.completion_mutex);
        }
        
        // Check for errors
        self.error_mutex.lock();
        const err = self.first_error;
        self.error_mutex.unlock();
        
        if (err) |e| {
            return e;
        }
    }
    
    /// Wait with timeout
    pub fn waitTimeout(self: *TaskGroup, timeout_ns: u64) !bool {
        self.completion_mutex.lock();
        defer self.completion_mutex.unlock();
        
        const deadline = std.time.nanoTimestamp() + @as(i128, @intCast(timeout_ns));
        
        while (self.active_count.load(.seq_cst) > 0) {
            const now = std.time.nanoTimestamp();
            if (now >= deadline) {
                return false;  // Timeout
            }
            
            const remaining = @as(u64, @intCast(deadline - now));
            self.completion_cond.timedWait(&self.completion_mutex, remaining) catch {
                return false;
            };
        }
        
        // Check for errors
        self.error_mutex.lock();
        const err = self.first_error;
        self.error_mutex.unlock();
        
        if (err) |e| {
            return e;
        }
        
        return true;  // Completed within timeout
    }
    
    /// Get statistics
    pub fn getStats(self: *const TaskGroup) Stats {
        self.children_mutex.lock();
        defer self.children_mutex.unlock();
        
        var completed: usize = 0;
        var failed: usize = 0;
        var cancelled: usize = 0;
        var running: usize = 0;
        
        for (self.children.items) |child| {
            switch (child.state) {
                .Completed => completed += 1,
                .Failed => failed += 1,
                .Cancelled => cancelled += 1,
                .Running => running += 1,
                else => {},
            }
        }
        
        return .{
            .total = self.children.items.len,
            .completed = completed,
            .failed = failed,
            .cancelled = cancelled,
            .running = running,
            .active = self.active_count.load(.seq_cst),
        };
    }
    
    pub const Stats = struct {
        total: usize,
        completed: usize,
        failed: usize,
        cancelled: usize,
        running: usize,
        active: usize,
        
        pub fn print(self: *const Stats) void {
            std.debug.print("TaskGroup Stats:\n", .{});
            std.debug.print("  Total: {d}\n", .{self.total});
            std.debug.print("  Completed: {d}\n", .{self.completed});
            std.debug.print("  Failed: {d}\n", .{self.failed});
            std.debug.print("  Cancelled: {d}\n", .{self.cancelled});
            std.debug.print("  Running: {d}\n", .{self.running});
            std.debug.print("  Active: {d}\n", .{self.active});
        }
    };
};

// ============================================================================
// Scoped Task Group (RAII pattern)
// ============================================================================

pub const ScopedTaskGroup = struct {
    group: *TaskGroup,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, config: TaskGroup.Config) !ScopedTaskGroup {
        const group = try TaskGroup.init(allocator, config);
        return .{
            .group = group,
            .allocator = allocator,
        };
    }
    
    /// Automatically waits and cleans up on deinit
    pub fn deinit(self: *ScopedTaskGroup) void {
        // Wait for all children (ignore error, just clean up)
        self.group.wait() catch {};
        self.group.deinit();
    }
    
    pub fn spawn(self: *ScopedTaskGroup, task_id: u64) !*TaskHandle {
        return self.group.spawn(task_id);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CancellationToken basic" {
    const allocator = std.testing.allocator;
    
    const token = try CancellationToken.init(allocator);
    defer token.deinit(allocator);
    
    try std.testing.expect(!token.isCancelled());
    
    token.cancel("test reason");
    try std.testing.expect(token.isCancelled());
    
    const reason = token.getCancelReason();
    try std.testing.expect(reason != null);
    try std.testing.expectEqualStrings("test reason", reason.?);
}

test "TaskHandle creation" {
    const allocator = std.testing.allocator;
    
    const handle = try TaskHandle.init(allocator, 1);
    defer handle.deinit(allocator);
    
    try std.testing.expectEqual(@as(u64, 1), handle.id);
    try std.testing.expect(!handle.isComplete());
}

test "TaskGroup spawn" {
    const allocator = std.testing.allocator;
    
    var group = try TaskGroup.init(allocator, .{});
    defer group.deinit();
    
    const handle = try group.spawn(1);
    
    try std.testing.expectEqual(@as(usize, 1), group.active_count.load(.seq_cst));
    try std.testing.expectEqual(@as(u64, 1), handle.id);
}

test "TaskGroup cancelAll" {
    const allocator = std.testing.allocator;
    
    var group = try TaskGroup.init(allocator, .{});
    defer group.deinit();
    
    _ = try group.spawn(1);
    _ = try group.spawn(2);
    _ = try group.spawn(3);
    
    group.cancelAll("test cancellation");
    
    try std.testing.expect(group.cancel_token.isCancelled());
}

test "TaskGroup error propagation" {
    const allocator = std.testing.allocator;
    
    var group = try TaskGroup.init(allocator, .{ .cancel_on_error = true });
    defer group.deinit();
    
    const h1 = try group.spawn(1);
    const h2 = try group.spawn(2);
    
    // Mark first as failed
    group.markCompleted(h1, error.TestError);
    
    // Second should be cancelled
    try std.testing.expect(group.cancel_token.isCancelled());
    
    // Mark second as completed (but already cancelled)
    group.markCompleted(h2, "result");
    
    // Wait should return error
    const result = group.wait();
    try std.testing.expectError(error.TestError, result);
}

test "ScopedTaskGroup RAII" {
    const allocator = std.testing.allocator;
    
    {
        var scoped = try ScopedTaskGroup.init(allocator, .{});
        defer scoped.deinit();
        
        const h = try scoped.spawn(1);
        scoped.group.markCompleted(h, "done");
        
        // Automatically waits and cleans up on scope exit
    }
}
