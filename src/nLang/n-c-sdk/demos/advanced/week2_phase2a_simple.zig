const std = @import("std");

// Week 2 Phase 2A: Multi-threading Implementation (Simplified)
// Demonstrates parallel force calculation without external dependencies
// Goal: 7-8x speedup on 8-core M1 Max

const Thread = std.Thread;

// =============================================
// Simple 3D Vector
// =============================================
const Vec3 = struct {
    x: f64,
    y: f64,
    z: f64,

    pub fn init(x: f64, y: f64, z: f64) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn add(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x + other.x, .y = self.y + other.y, .z = self.z + other.z };
    }

    pub fn sub(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x - other.x, .y = self.y - other.y, .z = self.z - other.z };
    }

    pub fn scale(self: Vec3, s: f64) Vec3 {
        return .{ .x = self.x * s, .y = self.y * s, .z = self.z * s };
    }

    pub fn lengthSquared(self: Vec3) f64 {
        return self.x * self.x + self.y * self.y + self.z * self.z;
    }

    pub fn length(self: Vec3) f64 {
        return @sqrt(self.lengthSquared());
    }
};

// =============================================
// Body Structure
// =============================================
const Body = struct {
    position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    mass: f64,
    id: u32,

    pub fn init(id: u32, pos: Vec3, vel: Vec3, mass: f64) Body {
        return .{
            .position = pos,
            .velocity = vel,
            .acceleration = Vec3.init(0, 0, 0),
            .mass = mass,
            .id = id,
        };
    }

    pub fn resetAcceleration(self: *Body) void {
        self.acceleration = Vec3.init(0, 0, 0);
    }

    pub fn update(self: *Body, dt: f64) void {
        // Leapfrog integration
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
    force_calc_ns: u64 = 0,
    integration_ns: u64 = 0,
    body_count: usize = 0,
};

// =============================================
// Parallel N-Body Simulation (Direct O(N²))
// =============================================
const ParallelNBody = struct {
    allocator: std.mem.Allocator,
    bodies: []Body,
    thread_count: usize,
    timings: []ThreadTiming,
    threads: []Thread,
    softening: f64 = 1.0, // Softening parameter

    pub fn init(allocator: std.mem.Allocator, total_bodies: usize) !*ParallelNBody {
        const thread_count = @min(try Thread.getCpuCount(), 16);

        const bodies = try allocator.alloc(Body, total_bodies);
        const timings = try allocator.alloc(ThreadTiming, thread_count);
        @memset(timings, .{});

        const threads = try allocator.alloc(Thread, thread_count);

        const self = try allocator.create(ParallelNBody);
        self.* = .{
            .allocator = allocator,
            .bodies = bodies,
            .thread_count = thread_count,
            .timings = timings,
            .threads = threads,
        };

        return self;
    }

    pub fn deinit(self: *ParallelNBody) void {
        self.allocator.free(self.threads);
        self.allocator.free(self.timings);
        self.allocator.free(self.bodies);
        self.allocator.destroy(self);
    }

    pub fn updateParallel(self: *ParallelNBody, dt: f64) !void {
        const work_per_thread = self.bodies.len / self.thread_count;

        @memset(self.timings, .{});

        // Phase 1: Parallel force calculation
        const force_start = std.time.nanoTimestamp();

        const ThreadContext = struct {
            sim: *ParallelNBody,
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

        // Phase 2: Parallel integration
        const integrate_start = std.time.nanoTimestamp();

        for (0..self.thread_count) |thread_id| {
            const start = thread_id * work_per_thread;
            const end = if (thread_id == self.thread_count - 1)
                self.bodies.len
            else
                start + work_per_thread;

            contexts[thread_id] = .{
                .sim = self,
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
            timing.force_calc_ns = force_time;
            timing.integration_ns = integrate_time;
        }
    }

    fn computeForcesThread(context: anytype) void {
        const start_time = std.time.nanoTimestamp();
        const G: f64 = 6.67430e-11; // Gravitational constant

        // Each thread computes forces for its slice
        for (context.sim.bodies[context.start..context.end]) |*body_i| {
            body_i.resetAcceleration();

            // Calculate force from all other bodies
            for (context.sim.bodies) |body_j| {
                if (body_i.id == body_j.id) continue;

                const dx = body_j.position.x - body_i.position.x;
                const dy = body_j.position.y - body_i.position.y;
                const dz = body_j.position.z - body_i.position.z;

                const dist_sq = dx * dx + dy * dy + dz * dz + context.sim.softening * context.sim.softening;
                const dist = @sqrt(dist_sq);
                const force_magnitude = G * body_j.mass / (dist_sq * dist);

                body_i.acceleration.x += force_magnitude * dx;
                body_i.acceleration.y += force_magnitude * dy;
                body_i.acceleration.z += force_magnitude * dz;
            }
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

    pub fn printTimings(self: *ParallelNBody) void {
        std.debug.print("\n📊 Thread Timings:\n", .{});
        std.debug.print("────────────────────────────────────\n", .{});

        var total_force: u64 = 0;
        var total_integrate: u64 = 0;

        for (self.timings, 0..) |timing, i| {
            if (timing.body_count > 0) {
                const force_ms = @as(f64, @floatFromInt(timing.force_calc_ns)) / 1_000_000.0;
                const integrate_ms = @as(f64, @floatFromInt(timing.integration_ns)) / 1_000_000.0;

                std.debug.print("Thread {d}: {d:5} bodies | ", .{ i, timing.body_count });
                std.debug.print("Force: {d:6.2}ms | ", .{force_ms});
                std.debug.print("Int: {d:6.2}ms\n", .{integrate_ms});

                total_force += timing.force_calc_ns;
                total_integrate += timing.integration_ns;
            }
        }

        const avg_force = @as(f64, @floatFromInt(total_force / self.thread_count)) / 1_000_000.0;
        const avg_integrate = @as(f64, @floatFromInt(total_integrate / self.thread_count)) / 1_000_000.0;

        std.debug.print("────────────────────────────────────\n", .{});
        std.debug.print("Average: Force: {d:.2}ms | Int: {d:.2}ms\n", .{
            avg_force,
            avg_integrate,
        });
    }
};

// =============================================
// Benchmark Functions
// =============================================

fn benchmarkSingleThreaded(allocator: std.mem.Allocator, num_bodies: usize, dt: f64) !f64 {
    const bodies = try allocator.alloc(Body, num_bodies);
    defer allocator.free(bodies);

    initPlummerSphere(bodies, 1000.0);

    const G: f64 = 6.67430e-11;
    const softening: f64 = 1.0;

    const start = std.time.nanoTimestamp();

    // Force calculation
    for (bodies) |*body_i| {
        body_i.resetAcceleration();
        for (bodies) |body_j| {
            if (body_i.id == body_j.id) continue;

            const dx = body_j.position.x - body_i.position.x;
            const dy = body_j.position.y - body_i.position.y;
            const dz = body_j.position.z - body_i.position.z;

            const dist_sq = dx * dx + dy * dy + dz * dz + softening * softening;
            const dist = @sqrt(dist_sq);
            const force_magnitude = G * body_j.mass / (dist_sq * dist);

            body_i.acceleration.x += force_magnitude * dx;
            body_i.acceleration.y += force_magnitude * dy;
            body_i.acceleration.z += force_magnitude * dz;
        }
    }

    // Integration
    for (bodies) |*body| {
        body.update(dt);
    }

    const end = std.time.nanoTimestamp();
    return @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
}

fn benchmarkParallel(allocator: std.mem.Allocator, num_bodies: usize, dt: f64) !f64 {
    var sim = try ParallelNBody.init(allocator, num_bodies);
    defer sim.deinit();

    initPlummerSphere(sim.bodies, 1000.0);

    const start = std.time.nanoTimestamp();
    try sim.updateParallel(dt);
    const end = std.time.nanoTimestamp();

    return @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
}

fn initPlummerSphere(bodies: []Body, radius: f64) void {
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.microTimestamp()));
    const rand = prng.random();

    for (bodies, 0..) |*body, i| {
        const r_frac = rand.float(f64);
        const r = radius / @sqrt(std.math.pow(f64, r_frac, -2.0 / 3.0) - 1.0);
        const theta = 2.0 * std.math.pi * rand.float(f64);
        const phi = std.math.acos(2.0 * rand.float(f64) - 1.0);

        const x = r * @sin(phi) * @cos(theta);
        const y = r * @sin(phi) * @sin(theta);
        const z = r * @cos(phi) * 0.1;

        const ve = @sqrt(2.0) * std.math.pow(f64, 1.0 + r * r, -0.25);
        const vx = ve * rand.floatNorm(f64);
        const vy = ve * rand.floatNorm(f64);
        const vz = ve * rand.floatNorm(f64) * 0.1;

        const mass = 1e30;

        body.* = Body.init(
            @intCast(i),
            Vec3.init(x, y, z),
            Vec3.init(vx, vy, vz),
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

    std.debug.print("\n🚀 Week 2 Phase 2A: Multi-threading Proof-of-Concept\n", .{});
    std.debug.print("══════════════════════════════════════════════════\n\n", .{});

    const cpu_cores = try Thread.getCpuCount();
    std.debug.print("CPU Cores: {d}\n", .{cpu_cores});
    std.debug.print("Using up to {d} threads\n\n", .{@min(cpu_cores, 16)});

    const test_sizes = [_]usize{ 100, 500, 1_000, 2_000 };
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

        if (speedup >= 4.0) {
            std.debug.print("  ✅ GOOD SPEEDUP: {d:.2}x (4x+ is excellent)\n", .{speedup});
        } else if (speedup >= 2.0) {
            std.debug.print("  ⚠️  MODERATE SPEEDUP: {d:.2}x (room for improvement)\n", .{speedup});
        } else {
            std.debug.print("  ❌ LOW SPEEDUP: {d:.2}x (needs optimization)\n", .{speedup});
        }
    }

    std.debug.print("\n\n✅ Phase 2A Proof-of-Concept Complete!\n", .{});
    std.debug.print("══════════════════════════════════════════════════\n", .{});
    std.debug.print("📋 Observations:\n", .{});
    std.debug.print("  • This uses O(N²) direct force calculation\n", .{});
    std.debug.print("  • With Barnes-Hut O(N log N), we can handle 50K+ bodies\n", .{});
    std.debug.print("  • Expected 6-8x speedup with proper implementation\n", .{});
}