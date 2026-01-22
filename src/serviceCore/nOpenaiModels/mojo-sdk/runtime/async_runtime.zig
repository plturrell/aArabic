// Mojo Async Runtime - Day 101+ (Optimized)
// Work-stealing executor with epoll/kqueue integration

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;
const Atomic = std.atomic.Value;

const core = @import("core.zig");
const RuntimeAllocator = core.RuntimeAllocator;
const Poller = @import("io_poller.zig").Poller;
const Event = @import("io_poller.zig").Event;
const EventType = @import("io_poller.zig").EventType;
const LockFree = @import("lockfree_queue.zig");
const TimerDriver = @import("timer.zig").TimerDriver;
const BlockingPool = @import("blocking_pool.zig").BlockingPool;
const BlockingFn = @import("blocking_pool.zig").BlockingFn;

// ============================================================================ 
// Task Handle
// ============================================================================ 

/// Unique identifier for a task
pub const TaskId = u64;

/// Task state
pub const TaskState = enum {
    Pending,
    Running,
    Suspended,
    Completed,
    Failed,
    Cancelled,
};

/// Priority levels for task scheduling
pub const TaskPriority = enum(u8) {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
};

/// Represents a handle to an async task
pub const TaskHandle = struct {
    id: TaskId,
    state: TaskState,
    priority: TaskPriority,
    result: ?[]const u8,
    error_msg: ?[]const u8,
    waker: ?*Waker,

    pub fn init(id: TaskId) TaskHandle {
        return .{
            .id = id,
            .state = .Pending,
            .priority = .Normal,
            .result = null,
            .error_msg = null,
            .waker = null,
        };
    }

    pub fn isComplete(self: *TaskHandle) bool {
        return self.state == .Completed or self.state == .Failed or self.state == .Cancelled;
    }

    pub fn wake(self: *TaskHandle) void {
        if (self.waker) |waker| {
            waker.wake();
        }
    }
};

// ============================================================================ 
// Waker
// ============================================================================ 

/// Waker to signal task readiness
pub const Waker = struct {
    task_id: TaskId,
    executor: *Executor,

    pub fn init(task_id: TaskId, executor: *Executor) Waker {
        return .{
            .task_id = task_id,
            .executor = executor,
        };
    }

    pub fn wake(self: *Waker) void {
        self.executor.wakeTask(self.task_id);
    }

    pub fn clone(self: *Waker, allocator: Allocator) !*Waker {
        const new_waker = try allocator.create(Waker);
        new_waker.* = self.*;
        return new_waker;
    }
};

// ============================================================================ 
// Future Trait
// ============================================================================ 

/// Future trait for async computations
pub const Future = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        poll: *const fn (ptr: *anyopaque, waker: *Waker) PollResult,
        deinit: *const fn (ptr: *anyopaque, allocator: Allocator) void,
    };

    pub const PollResult = union(enum) {
        Ready: []const u8,
        Pending: void,
    };

    pub fn poll(self: *Future, waker: *Waker) PollResult {
        return self.vtable.poll(self.ptr, waker);
    }

    pub fn deinit(self: *Future, allocator: Allocator) void {
        self.vtable.deinit(self.ptr, allocator);
    }
};

// ============================================================================ 
// Task
// ============================================================================ 

/// Represents an async task
pub const Task = struct {
    handle: TaskHandle,
    future: Future,
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: TaskId, future: Future) !*Task {
        const task = try allocator.create(Task);
        task.* = .{
            .handle = TaskHandle.init(id),
            .future = future,
            .allocator = allocator,
        };
        return task;
    }

    pub fn deinit(self: *Task) void {
        self.future.deinit(self.allocator);
        if (self.handle.result) |result| {
            self.allocator.free(result);
        }
        if (self.handle.error_msg) |err| {
            self.allocator.free(err);
        }
        self.allocator.destroy(self);
    }

    pub fn poll(self: *Task, waker: *Waker) Future.PollResult {
        self.handle.state = .Running;
        const result = self.future.poll(waker);
        
        switch (result) {
            .Ready => |value| {
                self.handle.state = .Completed;
                self.handle.result = value;
            },
            .Pending => {
                self.handle.state = .Suspended;
            },
        }
        
        return result;
    }
};

// ============================================================================ 
// Executor
// ============================================================================ 

const LocalQueue = LockFree.WorkStealingDeque(*Task);
const GlobalQueue = LockFree.MPSCQueue(*Task);

/// Work-Stealing Executor
pub const Executor = struct {
    allocator: Allocator,
    
    // Queues
    global_queue: *GlobalQueue,
    local_queues: []*LocalQueue,
    
    // I/O
    poller: Poller,
    timer_driver: TimerDriver,
    blocking_pool: *BlockingPool,
    
    // Task Management
    tasks: std.AutoHashMapUnmanaged(TaskId, *Task),
    tasks_mutex: Mutex, // Protects tasks map
    next_task_id: Atomic(u64),
    
    // Thread Management
    worker_threads: std.ArrayListUnmanaged(Thread),
    num_threads: usize,
    running: Atomic(bool),
    polling_mutex: Mutex,
    
    // Sleeping/Parking
    parked_threads: Atomic(usize),
    cond: Condition,
    cond_mutex: Mutex,

    pub fn init(allocator: Allocator, num_threads: usize) !*Executor {
        const executor = try allocator.create(Executor);
        
        // Init Global Queue
        const global_q = try GlobalQueue.init(allocator);
        
        // Init Local Queues
        const local_qs = try allocator.alloc(*LocalQueue, num_threads);
        for (local_qs) |*q| {
            q.* = try LocalQueue.init(allocator);
        }
        
        // Init Poller
        const poller = try Poller.init();
        
        // Init Timer
        const timer_driver = TimerDriver.init(allocator);
        
        // Init Blocking Pool (start with max 512 threads)
        const blocking_pool = try BlockingPool.init(allocator, 512);
        
        executor.* = .{
            .allocator = allocator,
            .global_queue = global_q,
            .local_queues = local_qs,
            .poller = poller,
            .timer_driver = timer_driver,
            .blocking_pool = blocking_pool,
            .tasks = .{},
            .tasks_mutex = Mutex{}, 
            .next_task_id = Atomic(u64).init(1),
            .worker_threads = .{} ,
            .num_threads = num_threads,
            .running = Atomic(bool).init(false),
            .polling_mutex = Mutex{}, 
            .parked_threads = Atomic(usize).init(0),
            .cond = Condition{}, 
            .cond_mutex = Mutex{}, 
        };
        
        return executor;
    }

    pub fn deinit(self: *Executor) void {
        self.shutdown();
        
        for (self.worker_threads.items) |thread| {
            thread.join();
        }
        self.worker_threads.deinit(self.allocator);
        
        self.global_queue.deinit();
        for (self.local_queues) |q| {
            q.deinit();
        }
        self.allocator.free(self.local_queues);
        
        self.poller.deinit();
        self.timer_driver.deinit();
        self.blocking_pool.deinit();
        
        // Clean up tasks
        self.tasks_mutex.lock();
        var it = self.tasks.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        self.tasks.deinit(self.allocator);
        self.tasks_mutex.unlock();
        
        self.allocator.destroy(self);
    }

    /// Spawn a new task
    pub fn spawn(self: *Executor, future: Future) !TaskId {
        const id = self.next_task_id.fetchAdd(1, .monotonic);
        const task = try Task.init(self.allocator, id, future);
        
        // Track task
        self.tasks_mutex.lock();
        try self.tasks.put(self.allocator, id, task);
        self.tasks_mutex.unlock();
        
        // Push to global queue
        try self.global_queue.push(task);
        
        // Wake a thread
        self.signalWorker();
        
        return id;
    }

    /// Wake a task by ID
    pub fn wakeTask(self: *Executor, task_id: TaskId) void {
        self.tasks_mutex.lock();
        const task_ptr = self.tasks.get(task_id);
        self.tasks_mutex.unlock();
        
        if (task_ptr) |task| {
            // Push to global queue (simplification: could push to local if thread-local storage was available)
            self.global_queue.push(task) catch |err| {
                std.debug.print("Error waking task {d}: {}\n", .{task_id, err});
                return;
            };
            self.signalWorker();
        }
    }

    /// Register interest in an I/O event
    pub fn registerIo(self: *Executor, fd: std.posix.fd_t, event: EventType, task_id: TaskId) !void {
        try self.poller.register(fd, event, @intCast(task_id));
    }

    /// Spawn a blocking task on the dedicated thread pool.
    /// Returns a TaskId for the async wrapper that awaits the blocking result.
    pub fn spawnBlocking(self: *Executor, func: BlockingFn, context: ?*anyopaque) !TaskId {
        // Create the blocking future
        const blocking_fut = try self.blocking_pool.spawn(func, context);
        
        // Wrap it in a standard Future interface
        const future = blocking_fut.asFuture();
        
        // Spawn it as a normal async task (which will just await the blocking future)
        return self.spawn(future);
    }

    /// Run the executor
    pub fn run(self: *Executor) !void {
        self.running.store(true, .seq_cst);
        
        for (0..self.num_threads) |i| {
            const thread = try Thread.spawn(.{}, workerLoop, .{ self, i });
            try self.worker_threads.append(self.allocator, thread);
        }
    }

    /// Shutdown
    pub fn shutdown(self: *Executor) void {
        self.running.store(false, .seq_cst);
        self.cond_mutex.lock();
        self.cond.broadcast();
        self.cond_mutex.unlock();
    }

    fn signalWorker(self: *Executor) void {
        self.cond_mutex.lock();
        defer self.cond_mutex.unlock();
        self.cond.signal();
    }

    fn workerLoop(executor: *Executor, thread_idx: usize) void {
        const local_q = executor.local_queues[thread_idx];
        var events_buf: [16]Event = undefined;
        
        while (executor.running.load(.monotonic)) {
            var task_opt: ?*Task = null;
            
            // 1. Try local queue
            task_opt = local_q.pop();
            
            // 2. Try global queue
            if (task_opt == null) {
                task_opt = executor.global_queue.pop();
            }
            
            // 3. Work stealing
            if (task_opt == null) {
                // Try stealing from other threads (random start)
                var steal_idx = (thread_idx + 1) % executor.num_threads;
                for (0..executor.num_threads - 1) |_| {
                    if (executor.local_queues[steal_idx].steal()) |stolen| {
                        task_opt = stolen;
                        break;
                    }
                    steal_idx = (steal_idx + 1) % executor.num_threads;
                }
            }
            
            // 4. Process task if found
            if (task_opt) |task| {
                var waker = Waker.init(task.handle.id, executor);
                const result = task.poll(&waker);
                
                if (result == .Ready) {
                    // Task finished. Leave in map for blockOn to retrieve.
                }
                // If Pending, it's suspended and will be woken later
                continue;
            }
            
            // 5. Poll I/O if no work found
            // Try to acquire the polling lock so only one thread polls at a time
            if (executor.polling_mutex.tryLock()) {
                 // Process timers first
                 const expired_count = executor.timer_driver.tick();
                 if (expired_count > 0) {
                     executor.signalWorker(); // Wake others for timer tasks
                 }

                 // Determine poll timeout based on next timer
                 // Default 10ms if no timers, or min(next_timer, 10ms)
                 var timeout_ms: i32 = 10;
                 if (executor.timer_driver.nextTimeoutMs()) |next_ms| {
                     if (next_ms < 10) timeout_ms = @intCast(next_ms);
                 }
                 
                 const count = executor.poller.poll(timeout_ms, &events_buf) catch 0;
                 if (count > 0) {
                     for (0..count) |i| {
                         const ev = events_buf[i];
                         // User data is task_id
                         executor.wakeTask(@intCast(ev.user_data));
                     }
                     executor.polling_mutex.unlock();
                     // Found IO events, so tasks are woken, loop again
                     continue;
                 }
                 executor.polling_mutex.unlock();
            }

            // 6. Park if nothing to do
            executor.cond_mutex.lock();
            // check again before sleeping
            if (executor.global_queue.isEmpty() and local_q.pop() == null) {
                 // Wait with timeout to periodically check IO/shutdown
                 // In a real impl, poller would be integrated into the wait
                 executor.cond.timedWait(&executor.cond_mutex, 10 * std.time.ns_per_ms) catch {};
            }
            executor.cond_mutex.unlock();
        }
    }
    
    /// Block until a specific task completes
    pub fn blockOn(self: *Executor, task_id: TaskId) ![]const u8 {
        while (true) {
            self.tasks_mutex.lock();
            const task_ptr = self.tasks.get(task_id);
            
            if (task_ptr) |task| {
                if (task.handle.isComplete()) {
                    // Remove from map since we are consuming it
                    _ = self.tasks.remove(task_id);
                    self.tasks_mutex.unlock();
                    
                    const result = task.handle.result orelse error.TaskFailed;
                    // Steal result ownership to prevent deinit from freeing it
                    task.handle.result = null;
                    task.deinit();
                    
                    return result;
                }
                self.tasks_mutex.unlock();
            } else {
                self.tasks_mutex.unlock();
                return error.TaskNotFound;
            }
            
            // Sleep briefly
            std.posix.nanosleep(0, 1 * std.time.ns_per_ms);
        }
    }
};

// ============================================================================ 
// Utilities
// ============================================================================ 

/// Create a simple future that immediately returns a value
pub fn readyFuture(allocator: Allocator, value: []const u8) !Future {
    const ReadyFuture = struct {
        value: []const u8,
        
        fn poll(ptr: *anyopaque, waker: *Waker) Future.PollResult {
            const self = @as(*const @This(), @ptrCast(@alignCast(ptr)));
            const duped = waker.executor.allocator.dupe(u8, self.value) catch @panic("OOM");
            return .{ .Ready = duped };
        }
        
        fn deinit_impl(ptr: *anyopaque, alloc: Allocator) void {
            const self = @as(*@This(), @ptrCast(@alignCast(ptr)));
            alloc.free(self.value);
            alloc.destroy(self);
        }
    };
    
    const rf = try allocator.create(ReadyFuture);
    rf.* = .{ .value = try allocator.dupe(u8, value) };
    
    return Future{
        .ptr = rf,
        .vtable = &.{
            .poll = ReadyFuture.poll,
            .deinit = ReadyFuture.deinit_impl,
        },
    };
}

// ============================================================================ 
// Tests
// ============================================================================ 

test "executor basics" {
    const allocator = std.testing.allocator;
    
    var executor = try Executor.init(allocator, 2);
    defer executor.deinit();
    
    try executor.run();
    
    const f = try readyFuture(allocator, "hello");
    const id = try executor.spawn(f);
    
    const res = try executor.blockOn(id);
    defer allocator.free(res);
    try std.testing.expectEqualStrings("hello", res);
}

// ============================================================================ 
// Timer Test
// ============================================================================ 

const SleepFuture = struct {
    duration_ms: u64,
    timer_id: ?u64,
    start_time: i64,
    
    fn poll(ptr: *anyopaque, waker: *Waker) Future.PollResult {
        const self = @as(*SleepFuture, @ptrCast(@alignCast(ptr)));
        
        if (self.timer_id == null) {
            self.start_time = std.time.milliTimestamp();
            const w_ptr = waker.clone(waker.executor.allocator) catch @panic("OOM");
            const w_val = w_ptr.*;
            waker.executor.allocator.destroy(w_ptr);
            
            self.timer_id = waker.executor.timer_driver.schedule(self.duration_ms, waker.task_id, w_val) catch @panic("Timer fail");
            return .Pending;
        } else {
            const now = std.time.milliTimestamp();
            const elapsed = now - self.start_time;
            if (elapsed < @as(i64, @intCast(self.duration_ms)) - 50) { 
                 return .Pending;
            }
            // Return allocated string to prevent double-free
            const res = waker.executor.allocator.dupe(u8, "") catch @panic("OOM");
            return .{ .Ready = res };
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
    defer allocator.free(res);
    const end = std.time.milliTimestamp();
    
    try std.testing.expect((end - start) >= 50); 
}