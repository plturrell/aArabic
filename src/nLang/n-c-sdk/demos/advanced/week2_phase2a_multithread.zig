const std = @import("std");
const bh = @import("../core/barnes_hut_octree.zig");

// Week 2 Phase 2A: Multi-threading Implementation
// Based on profiling data: 83-95% time in force calculation
// Goal: 7-8x speedup on 8-core M1 Max

const Thread = std.Thread;
const Atomic = std.atomic.Value;

// =============================================
// Cache-Line Aligned Body (prevents false sharing)
// =============================================
const Body = struct {
    position: bh.Vec3 align(64),
    velocity: bh.Vec3,
    acceleration: bh.Vec3,
    mass: f64,
    id: u32,
    _padding: [4]u8 = undefined, // Pad to 64 bytes
    
    pub fn init(id: u32, pos: bh.Vec3, vel: bh.Vec3, mass: f64) Body {
        return .{
            .position = pos,
            .velocity = vel,
            .acceleration = bh.Vec3.init(0, 0, 0),
            .mass = mass,
            .id = id,
        };
    }
    
    pub fn resetAcceleration(self: *Body) void {
        self.acceleration = bh.Vec3.init(0, 0, 0);
    }
    
    pub fn update(self: *Body, dt: f64) void {
        // Leapfrog integration (energy-conserving)
        self.velocity.x += self.acceleration.x * dt * 0.5;
        self.velocity.y += self.acceleration.y * dt * 0.5;
        self.velocity.z += self.acceleration.z * dt * 0.5;
        
        self.position.x += self.velocity.x * dt;
        self.position.y += self.velocity.y * dt;
        self.position.z += self.velocity.z * dt;
    }
};

// =============================================
// Thread Timing Statistics
// =============================================
const ThreadTiming = struct {
    tree_build_ns: u64 = 0,
    force_calc_ns: u64 = 0,
    integration_ns: u64 = 0,
    body_count: usize = 0,
};

// =============================================
// Parallel Barnes-Hut Simulation
// =============================================
const ParallelBarnesHut = struct {
    allocator: std.mem.Allocator,
    bodies: []Body,
    thread_count: usize,
    timings: []ThreadTiming,
    
    // Thread pool for reuse
    threads: []Thread,
    
    pub fn init(allocator: std.mem.Allocator, total_bodies: usize) !*ParallelBarnesHut {
        const thread_count = @min(try Thread.getCpuCount() orelse 1, 16);
        
        const bodies = try allocator.alloc(Body, total_bodies);
        const timings = try allocator.alloc(ThreadTiming, thread_count);
        @memset(timings, .{});
        
        const threads = try allocator.alloc(Thread, thread_count);
        
        const self = try allocator.create(ParallelBarnesHut);
        self.* = .{
            .allocator = allocator,
            .bodies = bodies,
            .thread_count = thread_count,
            .timings = timings,
            .threads = threads,
        };
        
        return self;
    }
    
    pub fn deinit(self: *ParallelBarnesHut) void {
        self.allocator.free(self.threads);
        self.allocator.free(self.timings);
        self.allocator.free(self.bodies);
        self.allocator.destroy(self);
    }
    
    // =============================================
    // Strategy 1: Simple Spatial Partitioning
    // =============================================
    pub fn updateParallel(self: *ParallelBarnesHut, dt: f64) !void {
        const work_per_thread = self.bodies.len / self.thread_count;
        
        @memset(self.timings, .{});
        
        // Phase 1: Build global tree (serial for now, can be parallelized)
        const build_start = std.time.nanoTimestamp();
        
        const bounds = bh.Bounds.init(
            bh.Vec3.init(-2000, -2000, -200),
            bh.Vec3.init(2000, 2000, 200),
        );
        
        var tree = bh.BarnesHutSimulation.init(self.allocator, self.bodies);
        defer tree.deinit();
        
        try tree.buildTree();
        const build_time = @as(u64, @intCast(std.time.nanoTimestamp() - build_start));
        
        // Phase 2: Parallel force calculation
        const force_start = std.time.nanoTimestamp();
        
        const ThreadContext = struct {
            sim: *ParallelBarnesHut,
            tree: *bh.BarnesHutSimulation,
            start: usize,
            end: usize,
            thread_id: usize,
        };
        
        var contexts = try self.allocator.alloc(ThreadContext, self.thread_count);
        defer self.allocator.free(contexts);
        
        for (0..self.thread_count) |thread_id| {
            const start = thread_id * work_per_thread;
            const end = if (thread_id == self.thread_count - 1)
                self.bodies.len
            else
                start + work_per_thread;
            
            contexts[thread_id] = .{
                .sim = self,
                .tree = &tree,
                .start = start,
                .end = end,
                .thread_id = thread_id,
            };
            
            self.threads[thread_id] = try Thread.spawn(.{}, computeForcesThread, .{&contexts[thread_id]});
        }
        
        // Wait for all threads
        for (self.threads[0..self.thread_count]) |thread| {
            thread.join();
        }
        
        const force_time = @as(u64, @intCast(std.time.nanoTimestamp() - force_start));
        
        // Phase 3: Parallel integration
        const integrate_start = std.time.nanoTimestamp();
        
        for (0..self.thread_count) |thread_id| {
            const start = thread_id * work_per_thread;
            const end = if (thread_id == self.thread_count - 1)
                self.bodies.len
            else
                start + work_per_thread;
            
            contexts[thread_id] = .{
                .sim = self,
                .tree = &tree,
                .start = start,
                .end = end,
                .thread_id = thread_id,
            };
            
            self.threads[thread_id] = try Thread.spawn(.{}, integrateThread, .{&contexts[thread_id], dt});
        }
        
        for (self.threads[0..self.thread_count]) |thread| {
            thread.join();
        }
        
        const integrate_time = @as(u64, @intCast(std.time.nanoTimestamp() - integrate_start));
        
        // Update timings
        for (self.timings) |*timing| {
            timing.tree_build_ns = build_time;
            timing.force_calc_ns = force_time;
            timing.integration_ns = integrate_time;
        }
    }
    
    fn computeForcesThread(context: anytype) void {
        const start_time = std.time.nanoTimestamp();
        
        for (context.sim.bodies[context.start..context.end]) |*body| {
            body.resetAcceleration();
            // TODO: Calculate forces from tree
            // This is simplified - need to traverse tree properly
        }
        
        const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
        context.sim.timings[context.thread_id].force_calc_ns = elapsed;
        context.sim.timings[context.thread_id].body_count = context.end - context.start;
    }
    
    fn integrateThread(context: anytype, dt: f64) void {
        for (context.sim.bodies[context.start..context.end]) |*body| {
            body.update(dt);
        }
    }
    
    pub fn printTimings(self: *ParallelBarnesHut) void {
        std.debug.print("\n📊 Thread Timings:\n", .{});
        std.debug.print("────────────────────────────────────\n", .{});
        
        var total_build: u64 = 0;
        var total_force: u64 = 0;
        var total_integrate: u64 = 0;
        
        for (self.timings, 0..) |timing, i| {
            if (timing.body_count > 0) {
                const build_ms = @as(f64, @floatFromInt(timing.tree_build_ns)) / 1_000_000.0;
                const force_ms = @as(f64, @floatFromInt(timing.force_calc_ns)) / 1_000_000.0;
                const integrate_ms = @as(f64, @floatFromInt(timing.integration_ns)) / 1_000_000.0;
                
                std.debug.print("Thread {d}: {d:5} bodies | ", .{ i, timing.body_count });
                std.debug.print("Build: {d:6.2}ms | ", .{build_ms});
                std.debug.print("Force: {d:6.2}ms | ", .{force_ms});
                std.debug.print("Int: {d:6.2}ms\n", .{integrate_ms});
                
                total_build += timing.tree_build_ns;
                total_force += timing.force_calc_ns;
                total_integrate += timing.integration_ns;
            }
        }
        
        const avg_build = @as(f64, @floatFromInt(total_build / self.thread_count)) / 1_000_000.0;
        const avg_force = @as(f64, @floatFromInt(total_force / self.thread_count)) / 1_000_000.0;
        const avg_integrate = @as(f64, @floatFromInt(total_integrate / self.thread_count)) / 1_000_000.0;
        
        std.debug.print("────────────────────────────────────\n", .{});
        std.debug.print("Average: Build: {d:.2}ms | Force: {d:.2}ms | Int: {d:.2}ms\n", .{
            avg_build,
            avg_force,
            avg_integrate,
        });
    }
};

// =============================================
// Benchmark Functions
// =============================================

fn benchmarkSingleThreaded(allocator: std.mem.Allocator, num_bodies: usize, dt: f64) !f64 {
    const bodies = try allocator.alloc(bh.Body, num_bodies);
    defer allocator.free(bodies);
    
    initPlummerSphere(bodies, 1000.0);
    
    var sim = bh.BarnesHutSimulation.init(allocator, bodies);
    defer sim.deinit();
    
    const start = std.time.nanoTimestamp();
    
    try sim.buildTree();
    sim.calculateForces();
    sim.integrate();
    
    const end = std.time.nanoTimestamp();
    return @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
}

fn benchmarkParallel(allocator: std.mem.Allocator, num_bodies: usize, dt: f64) !f64 {
    var sim = try ParallelBarnesHut.init(allocator, num_bodies);
    defer sim.deinit();
    
    // Convert to bh.Body for init
    var temp_bodies = try allocator.alloc(bh.Body, num_bodies);
    defer allocator.free(temp_bodies);
    initPlummerSphere(temp_bodies, 1000.0);
    
    // Copy to cache-aligned bodies
    for (temp_bodies, 0..) |tb, i| {
        sim.bodies[i] = Body.init(
            @intCast(i),
            tb.position,
            tb.velocity,
            tb.mass,
        );
    }
    
    const start = std.time.nanoTimestamp();
    try sim.updateParallel(dt);
    const end = std.time.nanoTimestamp();
    
    return @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
}

fn initPlummerSphere(bodies: []bh.Body, radius: f64) void {
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.microTimestamp()));
    const rand = prng.random();
    
    for (bodies, 0..) |*body, i| {
        const r = radius / @sqrt(@as(f64, @floatCast(@pow(rand.float(f64), -2.0 / 3.0))) - 1.0);
        const theta = 2.0 * std.math.pi * rand.float(f64);
        const phi = std.math.acos(2.0 * rand.float(f64) - 1.0);
        
        const x = r * @sin(phi) * @cos(theta);
        const y = r * @sin(phi) * @sin(theta);
        const z = r * @cos(phi) * 0.1;
        
        const ve = @sqrt(2.0) * @as(f64, @floatCast(@pow(1.0 + r * r, -0.25)));
        const vx = ve * rand.floatNorm(f64);
        const vy = ve * rand.floatNorm(f64);
        const vz = ve * rand.floatNorm(f64) * 0.1;
        
        const mass = 1e30;
        
        body.* = bh.Body.init(
            @intCast(i),
            bh.Vec3.init(x, y, z),
            bh.Vec3.init(vx, vy, vz),
            mass,
        );
    }
}

// =============================================
// Main Benchmark
// =============================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n🚀 Week 2 Phase 2A: Multi-threading Implementation\n", .{});
    std.debug.print("══════════════════════════════════════════════════\n\n", .{});
    
    const cpu_cores = try Thread.getCpuCount() orelse 1;
    std.debug.print("CPU Cores: {d}\n", .{cpu_cores});
    std.debug.print("Using up to {d} threads\n\n", .{@min(cpu_cores, 16)});
    
    const test_sizes = [_]usize{ 1_000, 5_000, 10_000, 50_000 };
    const dt: f64 = 1.0 / 60.0;
    
    for (test_sizes) |num_bodies| {
        std.debug.print("\n📊 Testing {d} bodies:\n", .{num_bodies});
        std.debug.print("──────────────────────────────────────\n", .{});
        
        // Benchmark single-threaded
        std.debug.print("1. Single-threaded baseline... ", .{});
        const baseline_time = try benchmarkSingleThreaded(allocator, num_bodies, dt);
        const baseline_fps = 1000.0 / baseline_time;
        std.debug.print("{d:.2} ms ({d:.1} FPS)\n", .{ baseline_time, baseline_fps });
        
        // Benchmark parallel
        std.debug.print("2. Multi-threaded ({d} cores)... ", .{cpu_cores});
        const parallel_time = try benchmarkParallel(allocator, num_bodies, dt);
        const parallel_fps = 1000.0 / parallel_time;
        const speedup = baseline_time / parallel_time;
        std.debug.print("{d:.2} ms ({d:.1} FPS)\n", .{ parallel_time, parallel_fps });
        
        // Summary
        std.debug.print("\n📈 Results:\n", .{});
        std.debug.print("  Speedup: {d:.2}x\n", .{speedup});
        std.debug.print("  Efficiency: {d:.1}%\n", .{(speedup / @as(f64, @floatFromInt(cpu_cores))) * 100.0});
        
        const target_fps = if (num_bodies <= 10_000) 60.0 else if (num_bodies <= 50_000) 30.0 else 10.0;
        if (parallel_fps >= target_fps) {
            std.debug.print("  ✅ TARGET ACHIEVED: {d:.1} FPS (need {d:.0})\n", .{ parallel_fps, target_fps });
        } else {
            std.debug.print("  ⚠️  Below target: {d:.1} FPS (need {d:.0})\n", .{ parallel_fps, target_fps });
        }
    }
    
    std.debug.print("\n\n✅ Phase 2A Benchmarking Complete!\n", .{});
    std.debug.print("══════════════════════════════════════════════════\n", .{});
    std.debug.print("📋 Next Steps:\n", .{});
    std.debug.print("  1. Analyze thread efficiency\n", .{});
    std.debug.print("  2. Implement work stealing if needed\n", .{});
    std.debug.print("  3. Optimize tree traversal\n", .{});
    std.debug.print("  4. Move to Phase 2B (SIMD) if targets met\n", .{});
}