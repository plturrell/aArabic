const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Config = struct {
    num_threads: usize = 4,
};

pub const WorkItem = struct {
    work_fn: *const fn (*anyopaque) void,
    context: *anyopaque,
};

pub const ThreadPool = struct {
    allocator: Allocator,
    config: Config,
    threads: []std.Thread,
    queue: std.ArrayList(WorkItem),
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,
    pending_count: std.atomic.Value(usize),
    done_cond: std.Thread.Condition,
    shutdown: std.atomic.Value(bool),
    serial_mode: bool,

    pub fn init(allocator: Allocator) !*ThreadPool {
        return initWithConfig(allocator, .{});
    }

    pub fn initWithConfig(allocator: Allocator, config: Config) !*ThreadPool {
        const self = try allocator.create(ThreadPool);
        errdefer allocator.destroy(self);

        const num_threads = if (config.num_threads == 0) 1 else config.num_threads;
        self.* = .{
            .allocator = allocator,
            .config = .{ .num_threads = num_threads },
            .threads = &.{},
            .queue = std.ArrayList(WorkItem).initCapacity(allocator, 16) catch return error.OutOfMemory,
            .mutex = .{},
            .cond = .{},
            .pending_count = std.atomic.Value(usize).init(0),
            .done_cond = .{},
            .shutdown = std.atomic.Value(bool).init(false),
            .serial_mode = num_threads == 1,
        };

        if (!self.serial_mode) {
            self.threads = try allocator.alloc(std.Thread, num_threads);
            for (self.threads, 0..) |*t, i| {
                t.* = std.Thread.spawn(.{}, workerLoop, .{self}) catch |err| {
                    for (self.threads[0..i]) |prev| prev.join();
                    allocator.free(self.threads);
                    return err;
                };
            }
        }
        return self;
    }

    pub fn submit(self: *ThreadPool, work_item: WorkItem) void {
        if (self.serial_mode) {
            work_item.work_fn(work_item.context);
            return;
        }
        _ = self.pending_count.fetchAdd(1, .seq_cst);
        self.mutex.lock();
        defer self.mutex.unlock();
        self.queue.append(self.allocator, work_item) catch {
            _ = self.pending_count.fetchSub(1, .seq_cst);
            return;
        };
        self.cond.signal();
    }

    pub fn waitAll(self: *ThreadPool) void {
        if (self.serial_mode) return;
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.pending_count.load(.seq_cst) > 0) {
            self.done_cond.wait(&self.mutex);
        }
    }

    pub fn deinit(self: *ThreadPool) void {
        if (!self.serial_mode) {
            self.shutdown.store(true, .seq_cst);
            self.mutex.lock();
            self.cond.broadcast();
            self.mutex.unlock();
            for (self.threads) |t| t.join();
            self.allocator.free(self.threads);
        }
        self.queue.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn workerLoop(self: *ThreadPool) void {
        while (true) {
            const item = blk: {
                self.mutex.lock();
                defer self.mutex.unlock();
                while (self.queue.items.len == 0) {
                    if (self.shutdown.load(.seq_cst)) return;
                    self.cond.wait(&self.mutex);
                }
                break :blk self.queue.orderedRemove(0);
            };
            item.work_fn(item.context);
            if (self.pending_count.fetchSub(1, .seq_cst) == 1) {
                self.mutex.lock();
                self.done_cond.broadcast();
                self.mutex.unlock();
            }
        }
    }
};

// Tests
test "thread pool basic" {
    const allocator = std.testing.allocator;
    const pool = try ThreadPool.init(allocator);
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);
    const Context = struct {
        counter: *std.atomic.Value(usize),
        fn work(ctx: *anyopaque) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            _ = c.counter.fetchAdd(1, .seq_cst);
        }
    };
    var ctx = Context{ .counter = &counter };
    for (0..10) |_| pool.submit(.{ .work_fn = Context.work, .context = @ptrCast(&ctx) });
    pool.waitAll();
    try std.testing.expectEqual(@as(usize, 10), counter.load(.seq_cst));
}

test "thread pool serial mode" {
    const allocator = std.testing.allocator;
    const pool = try ThreadPool.initWithConfig(allocator, .{ .num_threads = 1 });
    defer pool.deinit();

    var value: usize = 0;
    const Context = struct {
        value: *usize,
        fn work(ctx: *anyopaque) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.value.* += 1;
        }
    };
    var ctx = Context{ .value = &value };
    pool.submit(.{ .work_fn = Context.work, .context = @ptrCast(&ctx) });
    try std.testing.expectEqual(@as(usize, 1), value);
}
