const std = @import("std");
const sdl = @import("sdl");

const WIDTH: f32 = 1280;
const HEIGHT: f32 = 720;
const PARTICLE_COUNT = 50_000;

const TRAIL_LENGTH = 12;

const Particle = struct {
    pos: [2]f32,
    vel: [2]f32,
    hue: f32,
    target: [2]f32,
    trail: [TRAIL_LENGTH][2]f32,  // Previous positions for trail effect
    trail_idx: usize,
};

fn hsvToRgb(h: f32) [3]u8 {
    const s: f32 = 0.85;
    const v: f32 = 1.0;
    const c = v * s;
    const x = c * (1.0 - @abs(@mod(h / 60.0, 2.0) - 1.0));
    const m = v - c;

    var r: f32 = 0;
    var g: f32 = 0;
    var b: f32 = 0;

    if (h < 60) {
        r = c; g = x;
    } else if (h < 120) {
        r = x; g = c;
    } else if (h < 180) {
        g = c; b = x;
    } else if (h < 240) {
        g = x; b = c;
    } else if (h < 300) {
        r = x; b = c;
    } else {
        r = c; b = x;
    }

    return .{
        @as(u8, @intFromFloat((r + m) * 255.0)),
        @as(u8, @intFromFloat((g + m) * 255.0)),
        @as(u8, @intFromFloat((b + m) * 255.0)),
    };
}

// Simple text rasterization - creates target points for particles
fn createTextTargets(allocator: std.mem.Allocator, text: []const u8) ![][2]f32 {
    // Pre-allocate estimated size (about 15 points per character)
    var target_list = try std.ArrayList([2]f32).initCapacity(allocator, text.len * 15);
    defer target_list.deinit(allocator);
    
    const char_width: f32 = 40.0;
    const char_height: f32 = 60.0;
    const spacing: f32 = 8.0;
    const start_x: f32 = 80.0;
    const start_y: f32 = HEIGHT * 0.5 - char_height * 0.5;
    
    var x_offset: f32 = start_x;
    
    for (text) |char| {
        if (char == ' ') {
            x_offset += char_width * 0.5;
            continue;
        }
        if (char == '-') {
            // Draw horizontal line for dash
            for (0..8) |i| {
                const x = x_offset + @as(f32, @floatFromInt(i)) * spacing;
                const y = start_y + char_height * 0.5;
                try target_list.append(allocator, .{ x, y });
            }
            x_offset += char_width * 0.6;
            continue;
        }
        
        // Simplified character drawing using line segments
        try drawChar(char, x_offset, start_y, char_width, char_height, spacing, allocator, &target_list);
        x_offset += char_width;
    }
    
    return try target_list.toOwnedSlice(allocator);
}

fn drawChar(char: u8, x: f32, y: f32, w: f32, h: f32, s: f32, allocator: std.mem.Allocator, targets: *std.ArrayList([2]f32)) !void {
    const mx = x + w * 0.5;
    const my = y + h * 0.5;
    
    switch (char) {
        'A' => {
            // Triangle shape for A
            for (0..12) |i| {
                const t = @as(f32, @floatFromInt(i)) / 11.0;
                try targets.append(allocator, .{ x + w * t, y + h });  // Bottom
                try targets.append(allocator, .{ x + w * 0.5, y });  // Top
                try targets.append(allocator, .{ mx, my });  // Middle bar
            }
        },
        'I' => {
            // Vertical line
            for (0..15) |i| {
                const t = @as(f32, @floatFromInt(i)) / 14.0;
                try targets.append(allocator, .{ mx, y + h * t });
            }
        },
        'N' => {
            // Two verticals and diagonal
            for (0..12) |i| {
                const t = @as(f32, @floatFromInt(i)) / 11.0;
                try targets.append(allocator, .{ x + s, y + h * t });  // Left
                try targets.append(allocator, .{ x + w - s, y + h * t });  // Right
                try targets.append(allocator, .{ x + w * t, y + h * t });  // Diagonal
            }
        },
        'U' => {
            // U shape
            for (0..15) |i| {
                const t = @as(f32, @floatFromInt(i)) / 14.0;
                if (t > 0.3) {
                    try targets.append(allocator, .{ x + s, y + h * t });  // Left
                    try targets.append(allocator, .{ x + w - s, y + h * t });  // Right
                }
                if (t > 0.7) {
                    try targets.append(allocator, .{ x + w * t, y + h - s });  // Bottom
                }
            }
        },
        'C' => {
            // C shape (arc)
            for (0..15) |i| {
                const t = @as(f32, @floatFromInt(i)) / 14.0;
                try targets.append(allocator, .{ x + s, y + h * t });  // Left
                if (t < 0.3 or t > 0.7) {
                    try targets.append(allocator, .{ x + w * t, y + h * t });
                }
            }
        },
        'L' => {
            // L shape
            for (0..12) |i| {
                const t = @as(f32, @floatFromInt(i)) / 11.0;
                try targets.append(allocator, .{ x + s, y + h * t });  // Vertical
                if (t > 0.7) {
                    try targets.append(allocator, .{ x + w * t, y + h - s });  // Bottom
                }
            }
        },
        'E' => {
            // E shape
            for (0..12) |i| {
                const t = @as(f32, @floatFromInt(i)) / 11.0;
                try targets.append(allocator, .{ x + s, y + h * t });  // Vertical
                if (t < 0.2 or (t > 0.4 and t < 0.6) or t > 0.8) {
                    try targets.append(allocator, .{ x + w * t, y + h * t });  // Horizontals
                }
            }
        },
        'S' => {
            // S curve
            for (0..18) |i| {
                const t = @as(f32, @floatFromInt(i)) / 17.0;
                const curve_x = if (t < 0.5) x + w * (1.0 - t * 2.0) else x + w * ((t - 0.5) * 2.0);
                try targets.append(allocator, .{ curve_x, y + h * t });
            }
        },
        'D', 'd' => {
            // D shape
            for (0..15) |i| {
                const t = @as(f32, @floatFromInt(i)) / 14.0;
                try targets.append(allocator, .{ x + s, y + h * t });  // Left
                const angle = (t - 0.5) * std.math.pi;
                try targets.append(allocator, .{ x + w * 0.5 + @cos(angle) * w * 0.4, my + @sin(angle) * h * 0.4 });
            }
        },
        'K', 'k' => {
            // K shape
            for (0..12) |i| {
                const t = @as(f32, @floatFromInt(i)) / 11.0;
                try targets.append(allocator, .{ x + s, y + h * t });  // Vertical
                if (t < 0.5) {
                    try targets.append(allocator, .{ x + w * (1.0 - t * 2.0), y + h * t });  // Top diagonal
                } else {
                    try targets.append(allocator, .{ x + w * ((t - 0.5) * 2.0), y + h * t });  // Bottom diagonal
                }
            }
        },
        else => {
            // Default: small cluster
            for (0..5) |i| {
                const t = @as(f32, @floatFromInt(i)) / 4.0;
                try targets.append(allocator, .{ mx, y + h * t });
            }
        },
    }
}

fn initParticles(allocator: std.mem.Allocator) ![]Particle {
    const particles = try allocator.alloc(Particle, PARTICLE_COUNT);
    
    // Create text targets
    const text_targets = try createTextTargets(allocator, "AI NUCLEUS - n-c-sdk");
    defer allocator.free(text_targets);

    var prng = std.Random.DefaultPrng.init(@as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))));
    const random = prng.random();

    for (particles, 0..) |*p, i| {
        // Start from random positions
        p.pos = .{ random.float(f32) * WIDTH, random.float(f32) * HEIGHT };
        p.vel = .{ (random.float(f32) - 0.5) * 80.0, (random.float(f32) - 0.5) * 80.0 };
        
        // Assign target from text
        const target_idx = i % text_targets.len;
        p.target = .{ text_targets[target_idx][0], text_targets[target_idx][1] };
        
        // Color based on position in text
        p.hue = @as(f32, @floatFromInt(i % 360));
        
        // Initialize trail
        p.trail_idx = 0;
        for (&p.trail) |*t| {
            t.* = p.pos;
        }
    }

    return particles;
}

fn updateParticles(particles: []Particle, dt: f32) void {
    for (particles) |*p| {
        // Store old position in trail
        p.trail[p.trail_idx] = p.pos;
        p.trail_idx = (p.trail_idx + 1) % TRAIL_LENGTH;
        
        // Attract to target position (text formation)
        const dx = p.target[0] - p.pos[0];
        const dy = p.target[1] - p.pos[1];
        const dist = @sqrt(dx * dx + dy * dy);
        
        // Strong attraction to target with smooth approach
        const attraction_strength: f32 = 2.5;
        p.vel[0] += dx * attraction_strength * dt;
        p.vel[1] += dy * attraction_strength * dt;
        
        // Damping for smooth settling
        p.vel[0] *= 0.92;
        p.vel[1] *= 0.92;
        
        // Add slight noise when near target for organic look
        if (dist < 50.0) {
            var prng = std.Random.DefaultPrng.init(@as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))));
            const random = prng.random();
            p.vel[0] += (random.float(f32) - 0.5) * 20.0 * dt;
            p.vel[1] += (random.float(f32) - 0.5) * 20.0 * dt;
        }

        p.pos[0] += p.vel[0] * dt;
        p.pos[1] += p.vel[1] * dt;

        // Soft boundaries
        if (p.pos[0] < 0 or p.pos[0] > WIDTH) {
            p.vel[0] *= -0.7;
            p.pos[0] = std.math.clamp(p.pos[0], 0.0, WIDTH);
        }
        if (p.pos[1] < 0 or p.pos[1] > HEIGHT) {
            p.vel[1] *= -0.7;
            p.pos[1] = std.math.clamp(p.pos[1], 0.0, HEIGHT);
        }

        // Color cycle based on speed and position
        const speed = @sqrt(p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]);
        p.hue = @mod(p.hue + speed * 4.0 * dt + dt * 15.0, 360.0);
    }
}

fn renderParticles(renderer: *sdl.c.SDL_Renderer, particles: []const Particle) void {
    for (particles) |p| {
        const base_color = hsvToRgb(p.hue);
        
        // Draw trail with fading alpha
        for (0..TRAIL_LENGTH) |i| {
            const trail_idx = (p.trail_idx + i) % TRAIL_LENGTH;
            const trail_pos = p.trail[trail_idx];
            
            // Fade alpha based on age of trail point
            const alpha_factor = 1.0 - (@as(f32, @floatFromInt(i)) / @as(f32, TRAIL_LENGTH));
            const alpha = @as(u8, @intFromFloat(180.0 * alpha_factor));
            
            _ = sdl.c.SDL_SetRenderDrawColor(renderer, base_color[0], base_color[1], base_color[2], alpha);
            
            const trail_rect = sdl.c.SDL_Rect{
                .x = @as(c_int, @intFromFloat(trail_pos[0])) - 1,
                .y = @as(c_int, @intFromFloat(trail_pos[1])) - 1,
                .w = 2,
                .h = 2,
            };
            _ = sdl.c.SDL_RenderFillRect(renderer, &trail_rect);
        }
        
        // Draw current particle position brighter
        _ = sdl.c.SDL_SetRenderDrawColor(renderer, base_color[0], base_color[1], base_color[2], 255);
        const rect = sdl.c.SDL_Rect{
            .x = @as(c_int, @intFromFloat(p.pos[0])) - 1,
            .y = @as(c_int, @intFromFloat(p.pos[1])) - 1,
            .w = 3,
            .h = 3,
        };
        _ = sdl.c.SDL_RenderFillRect(renderer, &rect);
    }
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    sdl.c.SDL_SetMainReady();
    if (sdl.c.SDL_Init(sdl.c.SDL_INIT_VIDEO | sdl.c.SDL_INIT_EVENTS) != 0)
        return error.SDLInitFailed;
    defer sdl.c.SDL_Quit();

    const window = sdl.c.SDL_CreateWindow(
        "Zig Visual Particles",
        sdl.c.SDL_WINDOWPOS_CENTERED,
        sdl.c.SDL_WINDOWPOS_CENTERED,
        @as(c_int, @intFromFloat(WIDTH)),
        @as(c_int, @intFromFloat(HEIGHT)),
        sdl.c.SDL_WINDOW_SHOWN | sdl.c.SDL_WINDOW_RESIZABLE,
    ) orelse return error.CreateWindowFailed;
    defer sdl.c.SDL_DestroyWindow(window);

    const renderer = sdl.c.SDL_CreateRenderer(
        window,
        -1,
        sdl.c.SDL_RENDERER_ACCELERATED | sdl.c.SDL_RENDERER_PRESENTVSYNC,
    ) orelse return error.CreateRendererFailed;
    defer sdl.c.SDL_DestroyRenderer(renderer);

    _ = sdl.c.SDL_SetRenderDrawBlendMode(renderer, sdl.c.SDL_BLENDMODE_BLEND);

    const particles = try initParticles(allocator);
    defer allocator.free(particles);

    var e: sdl.c.SDL_Event = undefined;
    var running = true;
    var last = std.time.nanoTimestamp();
    var last_log = last;

    while (running) {
        while (sdl.c.SDL_PollEvent(&e) != 0) {
            switch (e.type) {
                sdl.c.SDL_QUIT => running = false,
                sdl.c.SDL_KEYDOWN => switch (e.key.keysym.sym) {
                    sdl.c.SDLK_ESCAPE => running = false,
                    else => {},
                },
                else => {},
            }
        }

        const now = std.time.nanoTimestamp();
        const dt = @as(f32, @floatFromInt(now - last)) / 1_000_000_000.0;
        last = now;

        updateParticles(particles, dt);

        // Clear screen to dark background
        _ = sdl.c.SDL_SetRenderDrawColor(renderer, 8, 12, 24, 255);
        _ = sdl.c.SDL_RenderClear(renderer);
        
        // Render all particles
        renderParticles(renderer, particles);
        
        sdl.c.SDL_RenderPresent(renderer);

        if (now - last_log >= 2_000_000_000) {
            const elapsed = @as(f64, @floatFromInt(now - last_log)) / 1_000_000_000.0;
            const fps = 1.0 / (@as(f64, @floatCast(dt)));
            std.log.info("fps: {d:.1} (avg over ~{d:.1}s)", .{ fps, elapsed });
            last_log = now;
        }
    }
}
