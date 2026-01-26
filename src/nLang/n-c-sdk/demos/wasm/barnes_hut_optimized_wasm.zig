// barnes_hut_optimized_wasm.zig - WebAssembly port of optimized Barnes-Hut
// Exports C ABI for JavaScript interop

const std = @import("std");
const bh = @import("barnes_hut_octree");
const simd = @import("barnes_hut_simd");

// Global allocator for WASM
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Global simulation state
var simulation: ?*simd.BarnesHutSIMD = null;
var bodies_buffer: ?[]bh.Body = null;

// Statistics structure for JavaScript
const Stats = extern struct {
    fps: f32,
    frame_time_ms: f32,
    tree_build_ms: f32,
    force_calc_ms: f32,
    integration_ms: f32,
    kinetic_energy: f64,
    potential_energy: f64,
    body_count: u32,
};

// Scenario enum matching JavaScript
const Scenario = enum(u32) {
    disk_galaxy = 0,
    galaxy_merger = 1,
    sphere_collapse = 2,
};

// Initialize simulation
export fn init(body_count: u32, _: u32) void {
    // Clean up existing simulation
    if (simulation) |sim| {
        sim.deinit();
        allocator.destroy(sim);
    }
    if (bodies_buffer) |bodies| {
        allocator.free(bodies);
    }
    
    // Create initial bodies (disk galaxy by default)
    bodies_buffer = createDiskGalaxy(body_count) catch return;
    
    // Initialize simulation
    const sim_ptr = allocator.create(simd.BarnesHutSIMD) catch return;
    sim_ptr.* = simd.BarnesHutSIMD.init(
        allocator,
        bodies_buffer.?
    ) catch return;
    
    simulation = sim_ptr;
}

// Clean up simulation
export fn deinit() void {
    if (simulation) |sim| {
        sim.deinit();
        allocator.destroy(sim);
        simulation = null;
    }
    if (bodies_buffer) |bodies| {
        allocator.free(bodies);
        bodies_buffer = null;
    }
}

// Step simulation forward
export fn step() void {
    if (simulation) |sim| {
        sim.step() catch return;
    }
}

// Get particle positions for rendering (returns pointer to float array)
export fn getPositions() [*]f32 {
    if (simulation) |sim| {
        const bodies = &sim.bodies_soa;
        // JavaScript will read 3 floats per particle: [x, y, z, x, y, z, ...]
        // We'll use pos_x array as base and rely on memory layout
        return @ptrCast(bodies.pos_x.ptr);
    }
    return undefined;
}

// Get particle velocities for coloring
export fn getVelocities() [*]f32 {
    if (simulation) |sim| {
        const bodies = &sim.bodies_soa;
        return @ptrCast(bodies.vel_x.ptr);
    }
    return undefined;
}

// Get simulation statistics
export fn getStats() Stats {
    if (simulation) |sim| {
        const perf_stats = sim.getStats();
        const energy = sim.calculateEnergy();
        
        return Stats{
            .fps = @floatCast(perf_stats.fps),
            .frame_time_ms = @floatCast(perf_stats.total_ms),
            .tree_build_ms = @floatCast(perf_stats.tree_build_ms),
            .force_calc_ms = @floatCast(perf_stats.force_calc_ms),
            .integration_ms = @floatCast(perf_stats.integration_ms),
            .kinetic_energy = energy.kinetic,
            .potential_energy = energy.potential,
            .body_count = @intCast(sim.bodies_soa.count),
        };
    }
    
    return Stats{
        .fps = 0,
        .frame_time_ms = 0,
        .tree_build_ms = 0,
        .force_calc_ms = 0,
        .integration_ms = 0,
        .kinetic_energy = 0,
        .potential_energy = 0,
        .body_count = 0,
    };
}

// Set scenario
export fn setScenario(scenario: u32, body_count: u32) void {
    const scenario_enum: Scenario = @enumFromInt(scenario);
    
    if (bodies_buffer) |bodies| {
        allocator.free(bodies);
    }
    
    bodies_buffer = switch (scenario_enum) {
        .disk_galaxy => createDiskGalaxy(body_count),
        .galaxy_merger => createTwoGalaxies(body_count),
        .sphere_collapse => createSphereCollapse(body_count),
    } catch return;
    
    // Reinitialize simulation with new bodies
    if (simulation) |sim| {
        sim.deinit();
        allocator.destroy(sim);
        
        const sim_ptr = allocator.create(simd.BarnesHutSIMD) catch return;
        sim_ptr.* = simd.BarnesHutSIMD.init(
            allocator,
            bodies_buffer.?
        ) catch return;
        
        simulation = sim_ptr;
    }
}

// Set simulation parameters
export fn setTheta(theta: f32) void {
    if (simulation) |sim| {
        sim.theta = theta;
    }
}

export fn setTimeStep(dt: f32) void {
    if (simulation) |sim| {
        sim.dt = dt;
    }
}

export fn setGravitationalConstant(G: f32) void {
    if (simulation) |sim| {
        sim.G = G;
    }
}

// Helper functions for creating different scenarios

fn createDiskGalaxy(n: u32) ![]bh.Body {
    const bodies = try allocator.alloc(bh.Body, n);
    
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();
    
    const radius = 100.0;
    const thickness = 10.0;
    
    for (bodies, 0..) |*body, i| {
        const r = radius * @sqrt(random.float(f64));
        const theta = 2.0 * std.math.pi * random.float(f64);
        const z = thickness * (random.float(f64) - 0.5);
        
        const x = r * @cos(theta);
        const y = r * @sin(theta);
        
        const v_circular = @sqrt(1.0 * radius / @max(r, 0.1));
        const v_random = 0.1 * v_circular;
        
        body.* = bh.Body.init(
            @intCast(i),
            bh.Vec3.init(x, y, z),
            bh.Vec3.init(
                -v_circular * @sin(theta) + v_random * (random.float(f64) - 0.5),
                v_circular * @cos(theta) + v_random * (random.float(f64) - 0.5),
                v_random * (random.float(f64) - 0.5),
            ),
            1.0 / @as(f64, @floatFromInt(n)),
        );
    }
    
    return bodies;
}

fn createTwoGalaxies(n: u32) ![]bh.Body {
    const bodies = try allocator.alloc(bh.Body, n);
    
    var prng = std.Random.DefaultPrng.init(67890);
    const random = prng.random();
    
    const separation = 150.0;
    const radius = 50.0;
    const thickness = 5.0;
    
    for (bodies, 0..) |*body, i| {
        const is_galaxy1 = i < n / 2;
        const center_x: f64 = if (is_galaxy1) -separation / 2.0 else separation / 2.0;
        
        const r = radius * @sqrt(random.float(f64));
        const theta = 2.0 * std.math.pi * random.float(f64);
        const z = thickness * (random.float(f64) - 0.5);
        
        const x = center_x + r * @cos(theta);
        const y = r * @sin(theta);
        
        const v_approach: f64 = if (is_galaxy1) 0.3 else -0.3;
        const v_circular = @sqrt(0.5 * radius / @max(r, 0.1));
        const rotation_dir: f64 = if (is_galaxy1) 1.0 else -1.0;
        
        body.* = bh.Body.init(
            @intCast(i),
            bh.Vec3.init(x, y, z),
            bh.Vec3.init(
                v_approach - rotation_dir * v_circular * @sin(theta),
                rotation_dir * v_circular * @cos(theta),
                0.0,
            ),
            1.0 / @as(f64, @floatFromInt(n)),
        );
    }
    
    return bodies;
}

fn createSphereCollapse(n: u32) ![]bh.Body {
    const bodies = try allocator.alloc(bh.Body, n);
    
    var prng = std.Random.DefaultPrng.init(54321);
    const random = prng.random();
    
    const radius = 100.0;
    
    for (bodies, 0..) |*body, i| {
        const theta = 2.0 * std.math.pi * random.float(f64);
        const phi = std.math.acos(2.0 * random.float(f64) - 1.0);
        const r = radius * std.math.pow(f64, random.float(f64), 1.0 / 3.0);
        
        const x = r * @sin(phi) * @cos(theta);
        const y = r * @sin(phi) * @sin(theta);
        const z = r * @cos(phi);
        
        const v_scale = 0.05;
        
        body.* = bh.Body.init(
            @intCast(i),
            bh.Vec3.init(x, y, z),
            bh.Vec3.init(
                v_scale * (random.float(f64) - 0.5),
                v_scale * (random.float(f64) - 0.5),
                v_scale * (random.float(f64) - 0.5),
            ),
            1.0 / @as(f64, @floatFromInt(n)),
        );
    }
    
    return bodies;
}

// Memory allocation exports for JavaScript
export fn allocateMemory(size: u32) [*]u8 {
    const memory = allocator.alloc(u8, size) catch return undefined;
    return memory.ptr;
}

export fn freeMemory(ptr: [*]u8, size: u32) void {
    const slice = ptr[0..size];
    allocator.free(slice);
}