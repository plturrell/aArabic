const std = @import("std");
const async_runtime = @import("../async_runtime.zig");
const Executor = async_runtime.Executor;
const Future = async_runtime.Future;
const Waker = async_runtime.Waker;
const Allocator = std.mem.Allocator;

// A future that sleeps for a specific duration
const SleepFuture = struct {
    duration_ms: u64,
    timer_id: ?u64,
    start_time: i64,
    
    fn poll(ptr: *anyopaque, waker: *Waker) Future.PollResult {
        const self = @as(*SleepFuture, @ptrCast(@alignCast(ptr)));
        
        if (self.timer_id == null) {
            // First poll: schedule timer
            self.start_time = std.time.milliTimestamp();
            // Use internal API to schedule (in real usage, we'd wrap this)
            self.timer_id = waker.executor.timer_driver.schedule(self.duration_ms, waker.task_id, waker.clone(waker.executor.allocator) catch @panic("OOM")) catch @panic("Timer fail");
            return .Pending;
        } else {
            // Second poll: timer fired
            // Verify duration (approximate)
            const now = std.time.milliTimestamp();
            const elapsed = now - self.start_time;
            if (elapsed < @as(i64, @intCast(self.duration_ms)) - 50) { // Allow some slack
                 // Spurious wake?
                 return .Pending;
            }
            return .{ .Ready = "" };
        }
    }
    
    fn deinit(ptr: *anyopaque, allocator: Allocator) void {
        const self = @as(*SleepFuture, @ptrCast(@alignCast(ptr)));
        allocator.destroy(self);
    }
};

fn createSleepFuture(allocator: Allocator, duration_ms: u64) !Future {
    const ptr = try allocator.create(SleepFuture);
    ptr.* = .{
        .duration_ms = duration_ms,
        .timer_id = null,
        .start_time = 0,
    };
    return Future{
        .ptr = ptr,
        .vtable = &.{
            .poll = SleepFuture.poll,
            .deinit = SleepFuture.deinit,
        },
    };
}

test "timer integration" {
    const allocator = std.testing.allocator;
    
    var executor = try Executor.init(allocator, 2);
    defer executor.deinit();
    
    try executor.run();
    
    const f = try createSleepFuture(allocator, 100);
    const id = try executor.spawn(f);
    
    const start = std.time.milliTimestamp();
    const res = try executor.blockOn(id);
    const end = std.time.milliTimestamp();
    
    // Should be at least 100ms
    try std.testing.expect((end - start) >= 90); 
    // Should not be too long (e.g. 2 seconds)
    try std.testing.expect((end - start) < 2000); 
}
