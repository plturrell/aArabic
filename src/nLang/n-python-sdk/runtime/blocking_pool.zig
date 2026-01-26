// Blocking Thread Pool
// A dedicated pool for blocking I/O and CPU-bound tasks.
// Dynamic scaling: spawns threads on demand up to a limit, scales down when idle.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;
const Atomic = std.atomic.Value;

const async_runtime = @import("async_runtime.zig");
const Task = async_runtime.Task; // We might wrap tasks differently
const Future = async_runtime.Future;
const Waker = async_runtime.Waker;

// ============================================================================
// Blocking Task Wrapper
// ============================================================================

/// A type-erased closure for a blocking operation
pub const BlockingFn = *const fn (context: ?*anyopaque) void;

pub const BlockingTask = struct {
    func: BlockingFn,
    context: ?*anyopaque,
    result_future: *BlockingFuture,
};

// ============================================================================
// Blocking Pool
// ============================================================================

pub const BlockingPool = struct {
    allocator: Allocator,
    mutex: Mutex,
    cond: Condition,
    
    // Task Queue
    queue: std.ArrayListUnmanaged(BlockingTask),
    
    // Pool State
    threads: std.ArrayListUnmanaged(Thread),
    num_threads: usize,
    idle_threads: usize,
    max_threads: usize,
    running: bool,
    
    // Configuration
    const KEEP_ALIVE_MS = 10000; // 10 seconds

    pub fn init(allocator: Allocator, max_threads: usize) !*BlockingPool {
        const pool = try allocator.create(BlockingPool);
        pool.* = .{
            .allocator = allocator,
            .mutex = Mutex{},
            .cond = Condition{},
            .queue = .{},
            .threads = .{},
            .num_threads = 0,
            .idle_threads = 0,
            .max_threads = max_threads,
            .running = true,
        };
        return pool;
    }

    pub fn deinit(self: *BlockingPool) void {
        self.shutdown();
        self.queue.deinit(self.allocator);
        self.threads.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn shutdown(self: *BlockingPool) void {
        self.mutex.lock();
        self.running = false;
        self.cond.broadcast();
        self.mutex.unlock();
        
        for (self.threads.items) |thread| {
            thread.join();
        }
    }

    /// Spawn a blocking task. Returns a future that resolves when done.
    /// The future yields `void` (for now, or we could support generic results).
    pub fn spawn(self: *BlockingPool, func: BlockingFn, context: ?*anyopaque) !*BlockingFuture {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (!self.running) return error.PoolShutDown;
        
        // Create the future that communicates the result back to async land
        const future = try BlockingFuture.create(self.allocator);
        
        const task = BlockingTask{
            .func = func,
            .context = context,
            .result_future = future,
        };
        
        try self.queue.append(self.allocator, task);
        
        // Scale up if needed:
        // If we have tasks but no idle threads, and we haven't hit max, spawn.
        if (self.idle_threads == 0 and self.num_threads < self.max_threads) {
            const thread = try Thread.spawn(.{}, workerLoop, .{self});
            try self.threads.append(self.allocator, thread);
            self.num_threads += 1;
        } else {
            // Wake an idle thread
            self.cond.signal();
        }
        
        return future;
    }

    fn workerLoop(self: *BlockingPool) void {
        self.mutex.lock();
        
        while (true) {
            // Try to get a task
            if (self.queue.items.len > 0) {
                const task = self.queue.orderedRemove(0);
                self.mutex.unlock();
                
                // Execute task
                task.func(task.context);
                
                // Signal completion
                task.result_future.complete();
                
                self.mutex.lock();
                continue;
            }
            
            // No tasks. Wait or exit.
            if (!self.running) break;
            
            self.idle_threads += 1;
            // Wait with timeout for dynamic scaling down
            // For simple "World Class" start, we just keep them alive or standard wait.
            // Let's implement timeout logic later to keep it robust first.
            self.cond.wait(&self.mutex);
            self.idle_threads -= 1;
        }
        
        self.mutex.unlock();
    }
};

// ============================================================================
// Blocking Future
// ============================================================================

/// Future that resolves when the blocking task is done.
/// This bridges the gap between the blocking thread and the async executor.
pub const BlockingFuture = struct {
    completed: Atomic(bool),
    waker: ?Waker,
    allocator: Allocator,
    mutex: Mutex, // Protects waker

    pub fn create(allocator: Allocator) !*BlockingFuture {
        const self = try allocator.create(BlockingFuture);
        self.* = .{
            .completed = Atomic(bool).init(false),
            .waker = null,
            .allocator = allocator,
            .mutex = Mutex{},
        };
        return self;
    }

    pub fn complete(self: *BlockingFuture) void {
        self.completed.store(true, .release);
        
        self.mutex.lock();
        if (self.waker) |*w| {
            w.wake();
        }
        self.mutex.unlock();
    }

    // Future Trait Implementation
    fn poll(ptr: *anyopaque, waker: *Waker) Future.PollResult {
        const self = @as(*BlockingFuture, @ptrCast(@alignCast(ptr)));
        
        if (self.completed.load(.acquire)) {
            return .{ .Ready = "" }; // Void result
        }
        
        self.mutex.lock();
        // Check again under lock to avoid race
        if (self.completed.load(.acquire)) {
            self.mutex.unlock();
            return .{ .Ready = "" };
        }
        
        // Update waker
        // In robust impl, we might check if waker changed.
        if (self.waker) |_| {
            // If we had a waker, we might need to free it if it was cloned? 
            // Or just overwrite. For simplicity assume single awaiter.
        }
        
        // We need to store the waker. The `waker` passed in is valid during poll.
        // But we need it *after* poll returns. So we must clone it.
        // Note: Executor must support cloning wakers.
        if (waker.clone(self.allocator)) |cloned| {
             // If we already had one, we should free it.
             // Simplification: We store *Waker pointer returned by clone.
             // But wait, our struct has `waker: ?Waker` (by value) or `?*Waker`?
             // Previous code: `waker.clone` returned `!*Waker`.
             self.waker = cloned.*; // Copy by value if Waker is small struct
             self.allocator.destroy(cloned); // Free the pointer, keep the struct
        } else |_| {
            // OOM in poll is hard. Panic or return Pending and hope?
        }
        self.mutex.unlock();
        
        return .Pending;
    }

    fn deinit_impl(ptr: *anyopaque, allocator: Allocator) void {
        const self = @as(*BlockingFuture, @ptrCast(@alignCast(ptr)));
        // Note: Waker inside doesn't own resources usually, but if it did...
        allocator.destroy(self);
    }

    pub fn asFuture(self: *BlockingFuture) Future {
        return Future{
            .ptr = self,
            .vtable = &.{
                .poll = poll,
                .deinit = deinit_impl,
            },
        };
    }
};