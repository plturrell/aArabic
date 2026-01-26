const std = @import("std");
const math = std.math;

// 3D Vector for positions, velocities, forces
pub const Vec3 = struct {
    x: f64,
    y: f64,
    z: f64,

    pub fn init(x: f64, y: f64, z: f64) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn zero() Vec3 {
        return .{ .x = 0, .y = 0, .z = 0 };
    }

    pub fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }

    pub fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
    }

    pub fn mul(a: Vec3, scalar: f64) Vec3 {
        return .{ .x = a.x * scalar, .y = a.y * scalar, .z = a.z * scalar };
    }

    pub fn div(a: Vec3, scalar: f64) Vec3 {
        return .{ .x = a.x / scalar, .y = a.y / scalar, .z = a.z / scalar };
    }

    pub fn lengthSquared(self: Vec3) f64 {
        return self.x * self.x + self.y * self.y + self.z * self.z;
    }

    pub fn length(self: Vec3) f64 {
        return @sqrt(self.lengthSquared());
    }

    pub fn normalize(self: Vec3) Vec3 {
        const len = self.length();
        if (len == 0) return Vec3.zero();
        return self.div(len);
    }

    pub fn dot(a: Vec3, b: Vec3) f64 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }
};

// A gravitational body (star, dark matter particle, etc.)
pub const Body = struct {
    position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    mass: f64,
    id: u32,

    pub fn init(id: u32, position: Vec3, velocity: Vec3, mass: f64) Body {
        return .{
            .id = id,
            .position = position,
            .velocity = velocity,
            .acceleration = Vec3.zero(),
            .mass = mass,
        };
    }
};

// Octree node for spatial partitioning
pub const OctreeNode = struct {
    // Geometric bounds
    center: Vec3,
    size: f64,

    // Mass distribution
    center_of_mass: Vec3,
    total_mass: f64,

    // Tree structure
    children: ?[8]*OctreeNode,
    body: ?*Body,
    is_leaf: bool,

    // For memory management
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, center: Vec3, size: f64) !*OctreeNode {
        const node = try allocator.create(OctreeNode);
        node.* = .{
            .center = center,
            .size = size,
            .center_of_mass = Vec3.zero(),
            .total_mass = 0,
            .children = null,
            .body = null,
            .is_leaf = true,
            .allocator = allocator,
        };
        return node;
    }

    pub fn deinit(self: *OctreeNode) void {
        if (self.children) |children| {
            // Recursively free all child nodes first
            for (children) |child| {
                child.deinit();
            }
        }
        // Free this node
        self.allocator.destroy(self);
    }
    
    // Helper methods for profiling
    pub fn calculateDepth(self: *OctreeNode) u32 {
        if (self.children == null) return 1;
        
        var max_depth: u32 = 0;
        for (self.children.?) |child| {
            const depth = child.calculateDepth();
            max_depth = @max(max_depth, depth);
        }
        return max_depth + 1;
    }
    
    pub fn countNodes(self: *OctreeNode) u64 {
        var count: u64 = 1;
        if (self.children) |children| {
            for (children) |child| {
                count += child.countNodes();
            }
        }
        return count;
    }
    
    pub fn countLeaves(self: *OctreeNode) u64 {
        if (self.children == null) return 1;
        
        var count: u64 = 0;
        for (self.children.?) |child| {
            count += child.countLeaves();
        }
        return count;
    }

    // Determine which octant a position belongs to
    fn getOctant(self: *OctreeNode, pos: Vec3) u3 {
        var octant: u3 = 0;
        if (pos.x >= self.center.x) octant |= 1;
        if (pos.y >= self.center.y) octant |= 2;
        if (pos.z >= self.center.z) octant |= 4;
        return octant;
    }

    // Get the center of a child octant
    fn getOctantCenter(self: *OctreeNode, octant: u3) Vec3 {
        const half_size = self.size / 4.0;
        return Vec3.init(
            self.center.x + if (octant & 1 != 0) half_size else -half_size,
            self.center.y + if (octant & 2 != 0) half_size else -half_size,
            self.center.z + if (octant & 4 != 0) half_size else -half_size,
        );
    }

    // Insert a body into the tree
    pub fn insert(self: *OctreeNode, body: *Body) !void {
        // If node is empty, just place the body here
        if (self.total_mass == 0) {
            self.body = body;
            self.center_of_mass = body.position;
            self.total_mass = body.mass;
            return;
        }

        // Update center of mass
        const total_mass_new = self.total_mass + body.mass;
        self.center_of_mass = Vec3.add(
            self.center_of_mass.mul(self.total_mass),
            body.position.mul(body.mass),
        ).div(total_mass_new);
        self.total_mass = total_mass_new;

        // If this is a leaf with one body, subdivide
        if (self.is_leaf and self.body != null) {
            // Create children array
            const children = try self.allocator.create([8]*OctreeNode);
            const half_size = self.size / 2.0;

            // Initialize all 8 octants
            for (0..8) |i| {
                const octant: u3 = @intCast(i);
                const octant_center = self.getOctantCenter(octant);
                children[i] = try OctreeNode.init(self.allocator, octant_center, half_size);
            }

            self.children = children.*;
            self.is_leaf = false;

            // Re-insert the existing body
            const existing_body = self.body.?;
            const existing_octant = self.getOctant(existing_body.position);
            try self.children.?[existing_octant].insert(existing_body);

            self.body = null;
        }

        // Insert new body into appropriate child
        if (self.children) |children| {
            const octant = self.getOctant(body.position);
            try children[octant].insert(body);
        }
    }

    // Calculate gravitational force on a body using Barnes-Hut approximation
    pub fn calculateForce(self: *OctreeNode, body: *Body, theta: f64, G: f64) Vec3 {
        // Don't calculate force from a body on itself
        if (self.body) |node_body| {
            if (node_body.id == body.id) {
                return Vec3.zero();
            }
        }

        // Vector from body to center of mass
        const r = Vec3.sub(self.center_of_mass, body.position);
        const distance_sq = r.lengthSquared();
        const distance = @sqrt(distance_sq);

        // Barnes-Hut criterion: s/d < theta
        // s = size of region, d = distance to center of mass
        // If criterion is met, treat as single body
        if (self.is_leaf or (self.size / distance < theta)) {
            // Softening parameter to avoid singularities
            const softening = 0.01;
            const distance_soft = distance_sq + softening * softening;

            // F = G * m1 * m2 / r²
            const force_magnitude = G * body.mass * self.total_mass / distance_soft;
            return r.normalize().mul(force_magnitude);
        }

        // Otherwise, recursively calculate force from children
        var total_force = Vec3.zero();
        if (self.children) |children| {
            for (children) |child| {
                if (child.total_mass > 0) {
                    const force = child.calculateForce(body, theta, G);
                    total_force = Vec3.add(total_force, force);
                }
            }
        }

        return total_force;
    }
};

// Barnes-Hut simulation manager
pub const BarnesHutSimulation = struct {
    bodies: []Body,
    root: ?*OctreeNode,
    allocator: std.mem.Allocator,

    // Simulation parameters
    theta: f64, // Barnes-Hut opening angle
    G: f64, // Gravitational constant
    dt: f64, // Time step

    // Statistics
    tree_depth: u32,
    node_count: u64,

    pub fn init(allocator: std.mem.Allocator, bodies: []Body) BarnesHutSimulation {
        return .{
            .bodies = bodies,
            .root = null,
            .allocator = allocator,
            .theta = 0.5,
            .G = 1.0, // Normalized units
            .dt = 0.01,
            .tree_depth = 0,
            .node_count = 0,
        };
    }

    pub fn deinit(self: *BarnesHutSimulation) void {
        if (self.root) |root| {
            root.deinit();
        }
    }

    // Build the octree from current body positions
    pub fn buildTree(self: *BarnesHutSimulation) !void {
        // Clear old tree
        if (self.root) |root| {
            root.deinit();
        }

        // Find bounding box of all bodies
        var min_pos = Vec3.init(math.inf(f64), math.inf(f64), math.inf(f64));
        var max_pos = Vec3.init(-math.inf(f64), -math.inf(f64), -math.inf(f64));

        for (self.bodies) |body| {
            min_pos.x = @min(min_pos.x, body.position.x);
            min_pos.y = @min(min_pos.y, body.position.y);
            min_pos.z = @min(min_pos.z, body.position.z);
            max_pos.x = @max(max_pos.x, body.position.x);
            max_pos.y = @max(max_pos.y, body.position.y);
            max_pos.z = @max(max_pos.z, body.position.z);
        }

        // Calculate center and size
        const center = Vec3.init(
            (min_pos.x + max_pos.x) / 2.0,
            (min_pos.y + max_pos.y) / 2.0,
            (min_pos.z + max_pos.z) / 2.0,
        );

        const size_x = max_pos.x - min_pos.x;
        const size_y = max_pos.y - min_pos.y;
        const size_z = max_pos.z - min_pos.z;
        const size = @max(size_x, @max(size_y, size_z)) * 1.1; // 10% padding

        // Create root node
        self.root = try OctreeNode.init(self.allocator, center, size);

        // Insert all bodies
        for (self.bodies) |*body| {
            try self.root.?.insert(body);
        }

        self.node_count = self.countNodes(self.root.?);
        self.tree_depth = self.calculateDepth(self.root.?, 0);
    }

    fn countNodes(self: *BarnesHutSimulation, node: *OctreeNode) u64 {
        var count: u64 = 1;
        if (node.children) |children| {
            for (children) |child| {
                count += self.countNodes(child);
            }
        }
        return count;
    }

    fn calculateDepth(self: *BarnesHutSimulation, node: *OctreeNode, current_depth: u32) u32 {
        if (node.children) |children| {
            var max_depth = current_depth;
            for (children) |child| {
                const child_depth = self.calculateDepth(child, current_depth + 1);
                max_depth = @max(max_depth, child_depth);
            }
            return max_depth;
        }
        return current_depth;
    }

    // Calculate forces on all bodies
    pub fn calculateForces(self: *BarnesHutSimulation) void {
        if (self.root == null) return;

        for (self.bodies) |*body| {
            const force = self.root.?.calculateForce(body, self.theta, self.G);
            body.acceleration = force.div(body.mass);
        }
    }

    // Integrate equations of motion (Leapfrog integrator for better energy conservation)
    pub fn integrate(self: *BarnesHutSimulation) void {
        for (self.bodies) |*body| {
            // Leapfrog: v(t + dt/2) = v(t) + a(t) * dt/2
            const half_dt = self.dt / 2.0;
            body.velocity = Vec3.add(body.velocity, body.acceleration.mul(half_dt));

            // Update position: x(t + dt) = x(t) + v(t + dt/2) * dt
            body.position = Vec3.add(body.position, body.velocity.mul(self.dt));
        }
    }

    // Complete simulation step
    pub fn step(self: *BarnesHutSimulation) !void {
        try self.buildTree();
        self.calculateForces();
        self.integrate();

        // Second half of leapfrog
        for (self.bodies) |*body| {
            body.velocity = Vec3.add(body.velocity, body.acceleration.mul(self.dt / 2.0));
        }
    }

    // Calculate total energy for verification
    pub fn calculateEnergy(self: *BarnesHutSimulation) struct { kinetic: f64, potential: f64 } {
        var kinetic: f64 = 0;
        var potential: f64 = 0;

        // Kinetic energy: KE = 0.5 * m * v²
        for (self.bodies) |body| {
            const v_sq = body.velocity.lengthSquared();
            kinetic += 0.5 * body.mass * v_sq;
        }

        // Potential energy: PE = -G * m1 * m2 / r (summed over all pairs)
        for (self.bodies, 0..) |body1, i| {
            for (self.bodies[i + 1 ..]) |body2| {
                const r = Vec3.sub(body2.position, body1.position);
                const distance = r.length();
                if (distance > 0) {
                    potential -= self.G * body1.mass * body2.mass / distance;
                }
            }
        }

        return .{ .kinetic = kinetic, .potential = potential };
    }

    // Calculate angular momentum for verification
    pub fn calculateAngularMomentum(self: *BarnesHutSimulation) Vec3 {
        var L = Vec3.zero();

        for (self.bodies) |body| {
            // L = r × (m * v)
            const momentum = body.velocity.mul(body.mass);
            const angular = Vec3.cross(body.position, momentum);
            L = Vec3.add(L, angular);
        }

        return L;
    }
};

// Helper function to create initial conditions
pub fn createGalaxy(allocator: std.mem.Allocator, n_bodies: usize, radius: f64, thickness: f64) ![]Body {
    const bodies = try allocator.alloc(Body, n_bodies);

    const Random = std.Random.DefaultPrng;
    var prng = Random.init(@intCast(std.time.microTimestamp()));
    const rand = prng.random();

    for (bodies, 0..) |*body, i| {
        // Random position in disk
        const r = radius * @sqrt(rand.float(f64)); // Sqrt for uniform disk distribution
        const theta = 2.0 * math.pi * rand.float(f64);
        const z = thickness * (rand.float(f64) - 0.5);

        const x = r * @cos(theta);
        const y = r * @sin(theta);

        // Circular velocity profile: v = sqrt(G * M(<r) / r)
        // For simplicity, assume flat rotation curve (like dark matter halo)
        const v_circular = @sqrt(1.0 * radius / @max(r, 0.1));

        // Add some random motion
        const v_random = 0.1 * v_circular;

        body.* = Body.init(
            @intCast(i),
            Vec3.init(x, y, z),
            Vec3.init(
                -v_circular * @sin(theta) + v_random * (rand.float(f64) - 0.5),
                v_circular * @cos(theta) + v_random * (rand.float(f64) - 0.5),
                v_random * (rand.float(f64) - 0.5),
            ),
            1.0 / @as(f64, @floatFromInt(n_bodies)), // Normalize total mass to 1
        );
    }

    return bodies;
}