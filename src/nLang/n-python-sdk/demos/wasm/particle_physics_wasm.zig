const std = @import("std");

const Particle = extern struct {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    mass: f32,
    color: u32,
};

var particles: []Particle = &.{};
var allocator: std.mem.Allocator = undefined;

// Configuration (can be set from JS)
var width: f32 = 800.0;
var height: f32 = 600.0;

export fn init(w: f32, h: f32, count: usize) usize {
    allocator = std.heap.wasm_allocator;

    width = w;
    height = h;

    particles = allocator.alloc(Particle, count) catch return 0;

    var prng = std.Random.DefaultPrng.init(0); // Fixed seed for determinism or pass seed
    const random = prng.random();

    for (particles) |*p| {
        p.x = random.float(f32) * width;
        p.y = random.float(f32) * height;
        p.vx = (random.float(f32) - 0.5) * 2.0;
        p.vy = (random.float(f32) - 0.5) * 2.0;
        p.mass = 1.0 + random.float(f32) * 2.0;

        // Random hue
        const hue = random.float(f32) * 360.0;
        p.color = hsvToRgb(hue, 0.8, 1.0);
    }

    return particles.len;
}

export fn update(dt: f32, mouse_x: f32, mouse_y: f32, mouse_active: bool) void {
    const GRAVITY = 0.0001;
    const DAMPING = 0.999;
    const MOUSE_FORCE = 5.0;

    for (particles) |*p| {
        if (mouse_active) {
            const dx = mouse_x - p.x;
            const dy = mouse_y - p.y;
            const dist_sq = dx * dx + dy * dy;
            if (dist_sq > 0.1) {
                const dist = @sqrt(dist_sq);
                const force = MOUSE_FORCE / (dist_sq + 100.0);
                p.vx += dx / dist * force;
                p.vy += dy / dist * force;
            }
        }

        const center_x = width / 2.0;
        const center_y = height / 2.0;
        const dx = center_x - p.x;
        const dy = center_y - p.y;
        p.vx += dx * GRAVITY * dt;
        p.vy += dy * GRAVITY * dt;

        p.vx *= DAMPING;
        p.vy *= DAMPING;

        p.x += p.vx * dt;
        p.y += p.vy * dt;

        if (p.x < 0 or p.x > width) {
            p.vx *= -0.8;
            p.x = std.math.clamp(p.x, 0, width);
        }
        if (p.y < 0 or p.y > height) {
            p.vy *= -0.8;
            p.y = std.math.clamp(p.y, 0, height);
        }

        const speed = @sqrt(p.vx * p.vx + p.vy * p.vy);
        const hue = @mod(speed * 10.0, 360.0);
        p.color = hsvToRgb(hue, 0.8, 1.0);
    }
}

export fn getParticlesPtr() [*]Particle {
    return particles.ptr;
}

export fn getParticlesLen() usize {
    return particles.len;
}

fn hsvToRgb(h: f32, s: f32, v: f32) u32 {
    const chroma = v * s;
    const x = chroma * (1.0 - @abs(@mod(h / 60.0, 2.0) - 1.0));
    const m = v - chroma;

    var r: f32 = 0;
    var g: f32 = 0;
    var b: f32 = 0;

    if (h < 60) {
        r = chroma;
        g = x;
        b = 0;
    } else if (h < 120) {
        r = x;
        g = chroma;
        b = 0;
    } else if (h < 180) {
        r = 0;
        g = chroma;
        b = x;
    } else if (h < 240) {
        r = 0;
        g = x;
        b = chroma;
    } else if (h < 300) {
        r = x;
        g = 0;
        b = chroma;
    } else {
        r = chroma;
        g = 0;
        b = x;
    }

    const ri = @as(u32, @intFromFloat((r + m) * 255.0));
    const gi = @as(u32, @intFromFloat((g + m) * 255.0));
    const bi = @as(u32, @intFromFloat((b + m) * 255.0));

    // ABGR packed for easy use in JS DataView/TypedArray (little-endian)
    return (0xFF << 24) | (bi << 16) | (gi << 8) | ri;
}
