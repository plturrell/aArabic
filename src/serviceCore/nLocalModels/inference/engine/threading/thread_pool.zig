const std = @import("std");

/// Basic Thread Pool Implementation
/// Provides parallel execution for inference tasks

// ============================================================================
// Thread Pool Configuration
// ============================================================================

pub const ThreadPoolConfig = struct {
    num_threads: u32 = 4,
    queue_size: u32 = 128,
    
    pub fn default() ThreadPoolConfig {
        // Default to number of CPU cores
        return .{
            .num_threads = @min(@as(u32, @intCast(std.Thread.getCpuCount() catch 4)), 16),
            .queue_size = 128,
        };
    }
    
    pub fn withThreads(count: u32) ThreadPoolConfig {
        return .{
            .num_threads = count,
            .queue_size = 128,
        };
    }
};

// ============================================================================
// Task Definition
// ============================================================================

pub const Task = struct {
    work_fn: *const fn (*anyopaque) void,
    context: *anyopaque,
};

// ============================================================================
// Thread Pool
// ============================================================================

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    config: ThreadPoolConfig,
    threads: []std.Thread,
    task_queue: std.ArrayList(Task),
    mutex: std.Thread.Mutex,
    condition: std.Thread.Condition,
    shutdown: bool,
    active_tasks: u32, // CRITICAL: Track tasks currently being executed
    
    pub fn init(allocator: std.mem.Allocator, config: ThreadPoolConfig) !ThreadPool {
        // Allocate threads array
        const threads = try allocator.alloc(std.Thread, config.num_threads);
        errdefer allocator.free(threads);
        
        // Create pool structure - fully initialized before starting threads
        var pool = ThreadPool{
            .allocator = allocator,
            .config = config,
            .threads = threads,
            .task_queue = undefined,
            .mutex = .{},
            .condition = .{},
            .shutdown = false,
            .active_tasks = 0, // Initialize active task counter
        };
        
        // Initialize ArrayList properly
        pool.task_queue = try std.ArrayList(Task).initCapacity(allocator, config.queue_size);
        
        // DON'T start threads yet - they would have invalid pointer after return
        // Threads will be started after pool is in its final memory location
        
        return pool;
    }
    
    /// Start the worker threads after pool is in final location
    pub fn start(self: *ThreadPool) !void {
        for (self.threads) |*thread| {
            thread.* = try std.Thread.spawn(.{}, workerThread, .{self});
        }
    }
    
    pub fn deinit(self: *ThreadPool) void {
        // Signal shutdown
        self.mutex.lock();
        self.shutdown = true;
        self.mutex.unlock();
        self.condition.broadcast();
        
        // Wait for all threads to finish
        for (self.threads) |thread| {
            thread.join();
        }
        
        self.allocator.free(self.threads);
        self.task_queue.deinit(self.allocator);
    }
    
    /// Submit a task to the thread pool
    pub fn submit(self: *ThreadPool, task: Task) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.shutdown) {
            return error.PoolShutdown;
        }
        
        try self.task_queue.append(self.allocator, task);
        self.condition.signal();
    }
    
    /// Wait for all tasks to complete
    pub fn waitAll(self: *ThreadPool) void {
        while (true) {
            self.mutex.lock();
            const all_done = self.task_queue.items.len == 0 and self.active_tasks == 0;
            self.mutex.unlock();
            
            if (all_done) {
                break;
            }
            
            std.Thread.sleep(100_000); // 100μs (reduced from 1ms for faster response)
        }
    }
    
    /// Worker thread function
    fn workerThread(pool: *ThreadPool) void {
        while (true) {
            pool.mutex.lock();
            
            // Wait for task or shutdown
            while (pool.task_queue.items.len == 0 and !pool.shutdown) {
                pool.condition.wait(&pool.mutex);
            }
            
            // Check shutdown
            if (pool.shutdown and pool.task_queue.items.len == 0) {
                pool.mutex.unlock();
                break;
            }
            
            // Get task
            const task = if (pool.task_queue.items.len > 0)
                pool.task_queue.orderedRemove(0)
            else {
                pool.mutex.unlock();
                continue;
            };
            
            // Increment active task counter BEFORE releasing mutex
            pool.active_tasks += 1;
            pool.mutex.unlock();
            
            // Execute task
            task.work_fn(task.context);
            
            // Decrement active task counter after completion
            pool.mutex.lock();
            pool.active_tasks -= 1;
            pool.mutex.unlock();
        }
    }
};

// ============================================================================
// Parallel Map
// ============================================================================

/// Execute a function in parallel over a slice
pub fn parallelMap(
    comptime T: type,
    comptime R: type,
    pool: *ThreadPool,
    input: []const T,
    output: []R,
    map_fn: *const fn (T) R,
) !void {
    std.debug.assert(input.len == output.len);
    
    const Context = struct {
        input: []const T,
        output: []R,
        map_fn: *const fn (T) R,
        index: usize,
    };
    
    const work_fn = struct {
        fn work(ctx: *anyopaque) void {
            const context: *Context = @ptrCast(@alignCast(ctx));
            context.output[context.index] = context.map_fn(context.input[context.index]);
        }
    }.work;
    
    // Submit tasks
    var contexts = try pool.allocator.alloc(Context, input.len);
    defer pool.allocator.free(contexts);
    
    for (input, output, 0..) |_, _, i| {
        contexts[i] = Context{
            .input = input,
            .output = output,
            .map_fn = map_fn,
            .index = i,
        };
        
        try pool.submit(.{
            .work_fn = work_fn,
            .context = &contexts[i],
        });
    }
    
    // Wait for completion
    pool.waitAll();
}

// ============================================================================
// Parallel Reduce
// ============================================================================

/// Execute a reduction in parallel
pub fn parallelReduce(
    comptime T: type,
    comptime R: type,
    pool: *ThreadPool,
    input: []const T,
    initial: R,
    reduce_fn: *const fn (R, T) R,
) !R {
    if (input.len == 0) return initial;
    
    const chunk_size = @max(1, input.len / pool.config.num_threads);
    const num_chunks = (input.len + chunk_size - 1) / chunk_size;
    
    var partial_results = try pool.allocator.alloc(R, num_chunks);
    defer pool.allocator.free(partial_results);
    
    const Context = struct {
        input: []const T,
        result: *R,
        reduce_fn: *const fn (R, T) R,
        initial: R,
        start: usize,
        end: usize,
    };
    
    const work_fn = struct {
        fn work(ctx: *anyopaque) void {
            const context: *Context = @ptrCast(@alignCast(ctx));
            var acc = context.initial;
            for (context.input[context.start..context.end]) |item| {
                acc = context.reduce_fn(acc, item);
            }
            context.result.* = acc;
        }
    }.work;
    
    // Submit chunk reduction tasks
    var contexts = try pool.allocator.alloc(Context, num_chunks);
    defer pool.allocator.free(contexts);
    
    for (0..num_chunks) |i| {
        const start = i * chunk_size;
        const end = @min(start + chunk_size, input.len);
        
        contexts[i] = Context{
            .input = input,
            .result = &partial_results[i],
            .reduce_fn = reduce_fn,
            .initial = initial,
            .start = start,
            .end = end,
        };
        
        try pool.submit(.{
            .work_fn = work_fn,
            .context = &contexts[i],
        });
    }
    
    pool.waitAll();
    
    // Combine partial results
    var final_result = initial;
    for (partial_results) |partial| {
        final_result = reduce_fn(final_result, partial);
    }
    
    return final_result;
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_thread_pool(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Thread Pool Module\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: Basic task submission
    {
        std.debug.print("\n1️⃣  Testing basic task submission...\n", .{});
        
        var pool = try ThreadPool.init(allocator, ThreadPoolConfig.withThreads(2));
        defer pool.deinit();
        try pool.start(); // Start worker threads
        
        std.debug.print("   Thread pool created with {d} threads\n", .{pool.config.num_threads});
        
        const Context = struct {
            counter: *u32,
            mutex: *std.Thread.Mutex,
        };
        
        var counter: u32 = 0;
        var mutex = std.Thread.Mutex{};
        
        const work_fn = struct {
            fn work(ctx: *anyopaque) void {
                const context: *Context = @ptrCast(@alignCast(ctx));
                context.mutex.lock();
                context.counter.* += 1;
                context.mutex.unlock();
            }
        }.work;
        
        var context = Context{ .counter = &counter, .mutex = &mutex };
        
        // Submit 10 tasks
        for (0..10) |_| {
            try pool.submit(.{
                .work_fn = work_fn,
                .context = &context,
            });
        }
        
        pool.waitAll();
        
        std.debug.print("   Submitted 10 tasks, counter = {d}\n", .{counter});
        
        if (counter != 10) {
            std.debug.print("   ❌ Expected counter to be 10\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Task submission working\n", .{});
    }
    
    // Test 2: Parallel map
    {
        std.debug.print("\n2️⃣  Testing parallel map...\n", .{});
        
        var pool = try ThreadPool.init(allocator, ThreadPoolConfig.default());
        defer pool.deinit();
        try pool.start(); // Start worker threads
        
        const n = 100;
        const input = try allocator.alloc(i32, n);
        defer allocator.free(input);
        const output = try allocator.alloc(i32, n);
        defer allocator.free(output);
        
        for (input, 0..) |*val, i| {
            val.* = @intCast(i);
        }
        
        const map_fn = struct {
            fn map(x: i32) i32 {
                return x * x;
            }
        }.map;
        
        try parallelMap(i32, i32, &pool, input, output, map_fn);
        
        // Verify results
        for (output, 0..) |val, i| {
            const expected = @as(i32, @intCast(i)) * @as(i32, @intCast(i));
            if (val != expected) {
                std.debug.print("   ❌ output[{d}] = {d}, expected {d}\n", .{ i, val, expected });
                return error.TestFailed;
            }
        }
        
        std.debug.print("   Mapped {d} elements in parallel\n", .{n});
        std.debug.print("   ✅ Parallel map working\n", .{});
    }
    
    // Test 3: Parallel reduce
    {
        std.debug.print("\n3️⃣  Testing parallel reduce...\n", .{});
        
        var pool = try ThreadPool.init(allocator, ThreadPoolConfig.default());
        defer pool.deinit();
        try pool.start(); // Start worker threads
        
        const n = 1000;
        const input = try allocator.alloc(i32, n);
        defer allocator.free(input);
        
        for (input, 0..) |*val, i| {
            val.* = @intCast(i + 1);
        }
        
        const reduce_fn = struct {
            fn reduce(acc: i32, x: i32) i32 {
                return acc + x;
            }
        }.reduce;
        
        const result = try parallelReduce(i32, i32, &pool, input, 0, reduce_fn);
        
        // Expected: sum of 1 to 1000 = n * (n+1) / 2 = 500500
        const expected = (n * (n + 1)) / 2;
        
        std.debug.print("   Reduced {d} elements: sum = {d}\n", .{ n, result });
        std.debug.print("   Expected: {d}\n", .{expected});
        
        if (result != expected) {
            std.debug.print("   ❌ Sum incorrect\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Parallel reduce working\n", .{});
    }
    
    // Test 4: Performance comparison
    {
        std.debug.print("\n4️⃣  Testing performance comparison...\n", .{});
        
        const n = 10000;
        const input = try allocator.alloc(f32, n);
        defer allocator.free(input);
        const output_serial = try allocator.alloc(f32, n);
        defer allocator.free(output_serial);
        const output_parallel = try allocator.alloc(f32, n);
        defer allocator.free(output_parallel);
        
        for (input, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i));
        }
        
        const map_fn = struct {
            fn map(x: f32) f32 {
                // Simulate some work
                var result: f32 = x;
                for (0..10) |_| {
                    result = @sqrt(result + 1.0);
                }
                return result;
            }
        }.map;
        
        // Serial execution
        var serial_timer = try std.time.Timer.start();
        for (input, output_serial) |in_val, *out_val| {
            out_val.* = map_fn(in_val);
        }
        const serial_time = serial_timer.read();
        
        // Parallel execution
        var pool = try ThreadPool.init(allocator, ThreadPoolConfig.default());
        defer pool.deinit();
        try pool.start(); // Start worker threads
        
        var parallel_timer = try std.time.Timer.start();
        try parallelMap(f32, f32, &pool, input, output_parallel, map_fn);
        const parallel_time = parallel_timer.read();
        
        const speedup = @as(f64, @floatFromInt(serial_time)) / @as(f64, @floatFromInt(parallel_time));
        
        std.debug.print("   Serial time: {d:.2} ms\n", .{@as(f64, @floatFromInt(serial_time)) / 1_000_000.0});
        std.debug.print("   Parallel time: {d:.2} ms\n", .{@as(f64, @floatFromInt(parallel_time)) / 1_000_000.0});
        std.debug.print("   Speedup: {d:.2}x\n", .{speedup});
        std.debug.print("   ✅ Performance comparison complete\n", .{});
    }
    
    std.debug.print("\n✅ All thread pool tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
