const std = @import("std");
const math = std.math;
const bh = @import("barnes_hut_octree.zig");

// SIMD vector type for 8-wide f64 operations (AVX-512 or 2x AVX2)
const Vec8f64 = @Vector(8, f64);
const Vec8bool = @Vector(8, bool);

// Named return type for force calculations
const ForceVec8 = struct {
    fx: Vec8f64,
    fy: Vec8f64,
    fz: Vec8f64,
};

// Structure of Arrays (SoA) layout for SIMD-friendly access
pub const BodiesSOA = struct {
    allocator: std.mem.Allocator,
    
    // Separate arrays for each component (cache-aligned)
    pos_x: []f64,
    pos_y: []f64,
    pos_z: []f64,
    
    vel_x: []f64,
    vel_y: []f64,
    vel_z: []f64,
    
    acc_x: []f64,
    acc_y: []f64,
    acc_z: []f64,
    
    mass: []f64,
    id: []u32,
    
    count: usize,
    capacity: usize,
    
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !BodiesSOA {
        // Allocate with 64-byte alignment for cache lines
        return BodiesSOA{
            .allocator = allocator,
            .pos_x = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity), // 2^6 = 64
            .pos_y = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .pos_z = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .vel_x = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .vel_y = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .vel_z = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .acc_x = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .acc_y = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .acc_z = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .mass = try allocator.alignedAlloc(f64, @enumFromInt(6), capacity),
            .id = try allocator.alignedAlloc(u32, @enumFromInt(6), capacity),
            .count = 0,
            .capacity = capacity,
        };
    }
    
    pub fn deinit(self: *BodiesSOA) void {
        self.allocator.free(self.pos_x);
        self.allocator.free(self.pos_y);
        self.allocator.free(self.pos_z);
        self.allocator.free(self.vel_x);
        self.allocator.free(self.vel_y);
        self.allocator.free(self.vel_z);
        self.allocator.free(self.acc_x);
        self.allocator.free(self.acc_y);
        self.allocator.free(self.acc_z);
        self.allocator.free(self.mass);
        self.allocator.free(self.id);
    }
    
    // Convert from AoS (Array of Structures) to SoA
    pub fn fromBodies(allocator: std.mem.Allocator, bodies: []bh.Body) !BodiesSOA {
        var soa = try BodiesSOA.init(allocator, bodies.len);
        
        for (bodies, 0..) |body, i| {
            soa.pos_x[i] = body.position.x;
            soa.pos_y[i] = body.position.y;
            soa.pos_z[i] = body.position.z;
            soa.vel_x[i] = body.velocity.x;
            soa.vel_y[i] = body.velocity.y;
            soa.vel_z[i] = body.velocity.z;
            soa.acc_x[i] = body.acceleration.x;
            soa.acc_y[i] = body.acceleration.y;
            soa.acc_z[i] = body.acceleration.z;
            soa.mass[i] = body.mass;
            soa.id[i] = body.id;
        }
        
        soa.count = bodies.len;
        return soa;
    }
    
    // Convert back to AoS for compatibility
    pub fn toBodies(self: *BodiesSOA, allocator: std.mem.Allocator) ![]bh.Body {
        const bodies = try allocator.alloc(bh.Body, self.count);
        
        for (0..self.count) |i| {
            bodies[i] = bh.Body{
                .position = bh.Vec3.init(self.pos_x[i], self.pos_y[i], self.pos_z[i]),
                .velocity = bh.Vec3.init(self.vel_x[i], self.vel_y[i], self.vel_z[i]),
                .acceleration = bh.Vec3.init(self.acc_x[i], self.acc_y[i], self.acc_z[i]),
                .mass = self.mass[i],
                .id = self.id[i],
            };
        }
        
        return bodies;
    }
};

// SIMD-optimized Barnes-Hut simulation
pub const BarnesHutSIMD = struct {
    bodies_soa: BodiesSOA,
    root: ?*bh.OctreeNode,
    allocator: std.mem.Allocator,
    
    // Simulation parameters
    theta: f64,
    G: f64,
    dt: f64,
    
    // Statistics
    tree_depth: u32,
    node_count: u64,
    
    pub fn init(allocator: std.mem.Allocator, bodies: []bh.Body) !BarnesHutSIMD {
        return BarnesHutSIMD{
            .bodies_soa = try BodiesSOA.fromBodies(allocator, bodies),
            .root = null,
            .allocator = allocator,
            .theta = 0.5,
            .G = 1.0,
            .dt = 0.01,
            .tree_depth = 0,
            .node_count = 0,
        };
    }
    
    pub fn deinit(self: *BarnesHutSIMD) void {
        if (self.root) |root| {
            root.deinit();
        }
        self.bodies_soa.deinit();
    }
    
    // Build octree (same as scalar version)
    pub fn buildTree(self: *BarnesHutSIMD) !void {
        // Clear old tree
        if (self.root) |root| {
            root.deinit();
        }
        
        // Convert SoA back to AoS for tree building
        const bodies_aos = try self.bodies_soa.toBodies(self.allocator);
        defer self.allocator.free(bodies_aos);
        
        // Find bounding box
        var min_pos = bh.Vec3.init(math.inf(f64), math.inf(f64), math.inf(f64));
        var max_pos = bh.Vec3.init(-math.inf(f64), -math.inf(f64), -math.inf(f64));
        
        for (bodies_aos) |body| {
            min_pos.x = @min(min_pos.x, body.position.x);
            min_pos.y = @min(min_pos.y, body.position.y);
            min_pos.z = @min(min_pos.z, body.position.z);
            max_pos.x = @max(max_pos.x, body.position.x);
            max_pos.y = @max(max_pos.y, body.position.y);
            max_pos.z = @max(max_pos.z, body.position.z);
        }
        
        const center = bh.Vec3.init(
            (min_pos.x + max_pos.x) / 2.0,
            (min_pos.y + max_pos.y) / 2.0,
            (min_pos.z + max_pos.z) / 2.0,
        );
        
        const size_x = max_pos.x - min_pos.x;
        const size_y = max_pos.y - min_pos.y;
        const size_z = max_pos.z - min_pos.z;
        const size = @max(size_x, @max(size_y, size_z)) * 1.1;
        
        self.root = try bh.OctreeNode.init(self.allocator, center, size);
        
        for (bodies_aos) |*body| {
            try self.root.?.insert(body);
        }
    }
    
    // SIMD-vectorized force calculation (8 bodies at once!)
    fn calculateForce8(
        self: *BarnesHutSIMD,
        idx: usize,
        node: *bh.OctreeNode,
    ) ForceVec8 {
        const bodies = &self.bodies_soa;
        
        // Load 8 body positions
        const px: Vec8f64 = bodies.pos_x[idx..][0..8].*;
        const py: Vec8f64 = bodies.pos_y[idx..][0..8].*;
        const pz: Vec8f64 = bodies.pos_z[idx..][0..8].*;
        
        // Displacement vectors to center of mass
        const dx = @as(Vec8f64, @splat(node.center_of_mass.x)) - px;
        const dy = @as(Vec8f64, @splat(node.center_of_mass.y)) - py;
        const dz = @as(Vec8f64, @splat(node.center_of_mass.z)) - pz;
        
        // Distance calculation
        const dist_sq = dx * dx + dy * dy + dz * dz;
        const softening: Vec8f64 = @splat(0.01 * 0.01);
        const dist_soft_sq = dist_sq + softening;
        const dist = @sqrt(dist_soft_sq);
        
        // Check Barnes-Hut criterion
        const size: Vec8f64 = @splat(node.size);
        const theta_vec: Vec8f64 = @splat(self.theta);
        const ratio = size / dist;
        const use_approx = ratio < theta_vec;
        
        // Force magnitude: F = G * m1 * m2 / r²
        const masses: Vec8f64 = bodies.mass[idx..][0..8].*;
        const node_mass: Vec8f64 = @splat(node.total_mass);
        const G: Vec8f64 = @splat(self.G);
        
        const force_mag = (G * masses * node_mass) / dist_soft_sq;
        
        // Force components (normalized direction × magnitude)
        var fx = (dx / dist) * force_mag;
        var fy = (dy / dist) * force_mag;
        var fz = (dz / dist) * force_mag;
        
        // If criterion not met, need to recurse (handled in traverseTree8)
        // For now, return the approximation
        const zero: Vec8f64 = @splat(0.0);
        fx = @select(f64, use_approx, fx, zero);
        fy = @select(f64, use_approx, fy, zero);
        fz = @select(f64, use_approx, fz, zero);
        
        return .{ .fx = fx, .fy = fy, .fz = fz };
    }
    
    // Traverse tree and accumulate forces for 8 bodies
    fn traverseTree8(
        self: *BarnesHutSIMD,
        idx: usize,
        node: *bh.OctreeNode,
    ) ForceVec8 {
        // Base case: leaf node or passes Barnes-Hut criterion
        if (node.is_leaf or node.total_mass == 0) {
            if (node.total_mass > 0) {
                return self.calculateForce8(idx, node);
            }
            const zero: Vec8f64 = @splat(0.0);
            return .{ .fx = zero, .fy = zero, .fz = zero };
        }
        
        // Check if node passes theta criterion for all 8 bodies
        const bodies = &self.bodies_soa;
        const px: Vec8f64 = bodies.pos_x[idx..][0..8].*;
        const py: Vec8f64 = bodies.pos_y[idx..][0..8].*;
        const pz: Vec8f64 = bodies.pos_z[idx..][0..8].*;
        
        const dx = @as(Vec8f64, @splat(node.center_of_mass.x)) - px;
        const dy = @as(Vec8f64, @splat(node.center_of_mass.y)) - py;
        const dz = @as(Vec8f64, @splat(node.center_of_mass.z)) - pz;
        
        const dist = @sqrt(dx * dx + dy * dy + dz * dz);
        const size: Vec8f64 = @splat(node.size);
        const theta_vec: Vec8f64 = @splat(self.theta);
        const ratio = size / dist;
        
        // If all 8 bodies pass theta criterion, use approximation
        const all_pass = @reduce(.And, ratio < theta_vec);
        if (all_pass) {
            return self.calculateForce8(idx, node);
        }
        
        // Otherwise, recurse into children
        var total_fx: Vec8f64 = @splat(0.0);
        var total_fy: Vec8f64 = @splat(0.0);
        var total_fz: Vec8f64 = @splat(0.0);
        
        if (node.children) |children| {
            for (children) |child| {
                if (child.total_mass > 0) {
                    const child_force = self.traverseTree8(idx, child);
                    total_fx += child_force.fx;
                    total_fy += child_force.fy;
                    total_fz += child_force.fz;
                }
            }
        }
        
        return .{ .fx = total_fx, .fy = total_fy, .fz = total_fz };
    }
    
    // Calculate forces using SIMD
    pub fn calculateForcesSIMD(self: *BarnesHutSIMD) void {
        if (self.root == null) return;
        
        const bodies = &self.bodies_soa;
        const n = bodies.count;
        const vec_width = 8;
        
        // Process in chunks of 8
        var i: usize = 0;
        while (i + vec_width <= n) : (i += vec_width) {
            const forces = self.traverseTree8(i, self.root.?);
            
            // Load masses for acceleration calculation
            const masses: Vec8f64 = bodies.mass[i..][0..8].*;
            
            // a = F / m
            const acc_x = forces.fx / masses;
            const acc_y = forces.fy / masses;
            const acc_z = forces.fz / masses;
            
            // Store results
            for (0..8) |j| {
                bodies.acc_x[i + j] = acc_x[j];
                bodies.acc_y[i + j] = acc_y[j];
                bodies.acc_z[i + j] = acc_z[j];
            }
        }
        
        // Handle remainder (< 8 bodies) with scalar code
        while (i < n) : (i += 1) {
            const pos = bh.Vec3.init(bodies.pos_x[i], bodies.pos_y[i], bodies.pos_z[i]);
            var body_temp = bh.Body{
                .position = pos,
                .velocity = bh.Vec3.zero(),
                .acceleration = bh.Vec3.zero(),
                .mass = bodies.mass[i],
                .id = bodies.id[i],
            };
            
            const force = self.root.?.calculateForce(&body_temp, self.theta, self.G);
            bodies.acc_x[i] = force.x / bodies.mass[i];
            bodies.acc_y[i] = force.y / bodies.mass[i];
            bodies.acc_z[i] = force.z / bodies.mass[i];
        }
    }
    
    // SIMD-optimized integration
    pub fn integrate(self: *BarnesHutSIMD) void {
        const bodies = &self.bodies_soa;
        const n = bodies.count;
        const vec_width = 8;
        
        const half_dt: Vec8f64 = @splat(self.dt / 2.0);
        const dt_vec: Vec8f64 = @splat(self.dt);
        
        // Leapfrog integration - first half-step
        var i: usize = 0;
        while (i + vec_width <= n) : (i += vec_width) {
            // Load current state
            var vx: Vec8f64 = bodies.vel_x[i..][0..8].*;
            var vy: Vec8f64 = bodies.vel_y[i..][0..8].*;
            var vz: Vec8f64 = bodies.vel_z[i..][0..8].*;
            
            const ax: Vec8f64 = bodies.acc_x[i..][0..8].*;
            const ay: Vec8f64 = bodies.acc_y[i..][0..8].*;
            const az: Vec8f64 = bodies.acc_z[i..][0..8].*;
            
            // v(t + dt/2) = v(t) + a(t) * dt/2
            vx += ax * half_dt;
            vy += ay * half_dt;
            vz += az * half_dt;
            
            // Update position: x(t + dt) = x(t) + v(t + dt/2) * dt
            const px: Vec8f64 = bodies.pos_x[i..][0..8].*;
            const py: Vec8f64 = bodies.pos_y[i..][0..8].*;
            const pz: Vec8f64 = bodies.pos_z[i..][0..8].*;
            
            const new_px = px + vx * dt_vec;
            const new_py = py + vy * dt_vec;
            const new_pz = pz + vz * dt_vec;
            
            // Store results
            for (0..8) |j| {
                bodies.vel_x[i + j] = vx[j];
                bodies.vel_y[i + j] = vy[j];
                bodies.vel_z[i + j] = vz[j];
                bodies.pos_x[i + j] = new_px[j];
                bodies.pos_y[i + j] = new_py[j];
                bodies.pos_z[i + j] = new_pz[j];
            }
        }
        
        // Handle remainder
        while (i < n) : (i += 1) {
            bodies.vel_x[i] += bodies.acc_x[i] * self.dt / 2.0;
            bodies.vel_y[i] += bodies.acc_y[i] * self.dt / 2.0;
            bodies.vel_z[i] += bodies.acc_z[i] * self.dt / 2.0;
            
            bodies.pos_x[i] += bodies.vel_x[i] * self.dt;
            bodies.pos_y[i] += bodies.vel_y[i] * self.dt;
            bodies.pos_z[i] += bodies.vel_z[i] * self.dt;
        }
    }
    
    // Complete simulation step
    pub fn step(self: *BarnesHutSIMD) !void {
        try self.buildTree();
        self.calculateForcesSIMD();
        self.integrate();
        
        // Second half of leapfrog
        const bodies = &self.bodies_soa;
        const n = bodies.count;
        const vec_width = 8;
        const half_dt: Vec8f64 = @splat(self.dt / 2.0);
        
        var i: usize = 0;
        while (i + vec_width <= n) : (i += vec_width) {
            var vx: Vec8f64 = bodies.vel_x[i..][0..8].*;
            var vy: Vec8f64 = bodies.vel_y[i..][0..8].*;
            var vz: Vec8f64 = bodies.vel_z[i..][0..8].*;
            
            const ax: Vec8f64 = bodies.acc_x[i..][0..8].*;
            const ay: Vec8f64 = bodies.acc_y[i..][0..8].*;
            const az: Vec8f64 = bodies.acc_z[i..][0..8].*;
            
            vx += ax * half_dt;
            vy += ay * half_dt;
            vz += az * half_dt;
            
            for (0..8) |j| {
                bodies.vel_x[i + j] = vx[j];
                bodies.vel_y[i + j] = vy[j];
                bodies.vel_z[i + j] = vz[j];
            }
        }
        
        while (i < n) : (i += 1) {
            bodies.vel_x[i] += bodies.acc_x[i] * self.dt / 2.0;
            bodies.vel_y[i] += bodies.acc_y[i] * self.dt / 2.0;
            bodies.vel_z[i] += bodies.acc_z[i] * self.dt / 2.0;
        }
    }
    
    // Calculate energy for verification
    pub fn calculateEnergy(self: *BarnesHutSIMD) struct { kinetic: f64, potential: f64 } {
        const bodies = &self.bodies_soa;
        const n = bodies.count;
        
        var kinetic: f64 = 0;
        var potential: f64 = 0;
        
        // Kinetic energy (vectorized)
        const vec_width = 8;
        var ke_vec: Vec8f64 = @splat(0.0);
        const half: Vec8f64 = @splat(0.5);
        
        var i: usize = 0;
        while (i + vec_width <= n) : (i += vec_width) {
            const vx: Vec8f64 = bodies.vel_x[i..][0..8].*;
            const vy: Vec8f64 = bodies.vel_y[i..][0..8].*;
            const vz: Vec8f64 = bodies.vel_z[i..][0..8].*;
            const m: Vec8f64 = bodies.mass[i..][0..8].*;
            
            const v_sq = vx * vx + vy * vy + vz * vz;
            ke_vec += half * m * v_sq;
        }
        
        kinetic = @reduce(.Add, ke_vec);
        
        // Handle remainder
        while (i < n) : (i += 1) {
            const vx = bodies.vel_x[i];
            const vy = bodies.vel_y[i];
            const vz = bodies.vel_z[i];
            const v_sq = vx * vx + vy * vy + vz * vz;
            kinetic += 0.5 * bodies.mass[i] * v_sq;
        }
        
        // Potential energy (still O(N²), but less frequent)
        for (0..n) |j| {
            for (j + 1..n) |k| {
                const dx = bodies.pos_x[k] - bodies.pos_x[j];
                const dy = bodies.pos_y[k] - bodies.pos_y[j];
                const dz = bodies.pos_z[k] - bodies.pos_z[j];
                const dist = @sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > 0) {
                    potential -= self.G * bodies.mass[j] * bodies.mass[k] / dist;
                }
            }
        }
        
        return .{ .kinetic = kinetic, .potential = potential };
    }
};