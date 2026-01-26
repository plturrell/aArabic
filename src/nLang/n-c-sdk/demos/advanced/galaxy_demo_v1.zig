const std = @import("std");
const bh = @import("barnes_hut_octree.zig");
const math = std.math;

const c = @cImport({
    @cInclude("SDL2/SDL.h");
});

// Configuration
const WINDOW_WIDTH = 1600;
const WINDOW_HEIGHT = 1200;
const INITIAL_BODIES = 10_000; // Start small, scale up
const VIEW_SCALE = 200.0; // Zoom factor for visualization

// 3D camera for visualization
const Camera = struct {
    position: bh.Vec3,
    rotation_x: f64, // Pitch
    rotation_y: f64, // Yaw
    zoom: f64,

    pub fn init() Camera {
        return .{
            .position = bh.Vec3.init(0, 0, 5.0),
            .rotation_x = 0.3, // Look down slightly
            .rotation_y = 0,
            .zoom = 1.0,
        };
    }

    // Project 3D point to 2D screen
    pub fn project(self: Camera, point: bh.Vec3) struct { x: i32, y: i32, visible: bool } {
        // Rotate around Y axis (yaw)
        const cos_y = @cos(self.rotation_y);
        const sin_y = @sin(self.rotation_y);
        const x1 = point.x * cos_y - point.z * sin_y;
        const z1 = point.x * sin_y + point.z * cos_y;

        // Rotate around X axis (pitch)
        const cos_x = @cos(self.rotation_x);
        const sin_x = @sin(self.rotation_x);
        const y2 = point.y * cos_x - z1 * sin_x;
        const z2 = point.y * sin_x + z1 * cos_x;

        // Simple perspective projection
        const depth = z2 + self.position.z;
        if (depth <= 0.1) return .{ .x = 0, .y = 0, .visible = false };

        const scale = (VIEW_SCALE * self.zoom) / depth;
        const screen_x = @as(i32, @intFromFloat(x1 * scale + WINDOW_WIDTH / 2));
        const screen_y = @as(i32, @intFromFloat(y2 * scale + WINDOW_HEIGHT / 2));

        const visible = screen_x >= 0 and screen_x < WINDOW_WIDTH and
            screen_y >= 0 and screen_y < WINDOW_HEIGHT;

        return .{ .x = screen_x, .y = screen_y, .visible = visible };
    }
};

// Performance metrics
const Metrics = struct {
    tree_build_time_ms: f64 = 0,
    force_calc_time_ms: f64 = 0,
    integration_time_ms: f64 = 0,
    total_frame_time_ms: f64 = 0,
    fps: f64 = 0,
    
    tree_depth: u32 = 0,
    node_count: u64 = 0,
    body_count: usize = 0,
    
    energy_kinetic: f64 = 0,
    energy_potential: f64 = 0,
    energy_total: f64 = 0,
    energy_initial: f64 = 0,
    energy_drift: f64 = 0,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n🌌 Barnes-Hut Galaxy Simulation - Week 1 Demo\n", .{});
    std.debug.print("═══════════════════════════════════════════════════\n\n", .{});
    std.debug.print("Initializing {d} bodies with disk galaxy distribution...\n", .{INITIAL_BODIES});

    // Initialize SDL
    if (c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_EVENTS) != 0) {
        std.debug.print("SDL_Init failed: {s}\n", .{c.SDL_GetError()});
        return error.SDLInitFailed;
    }
    defer c.SDL_Quit();

    // Create window
    const window = c.SDL_CreateWindow(
        "🌌 Barnes-Hut Galaxy Simulation",
        c.SDL_WINDOWPOS_CENTERED,
        c.SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        c.SDL_WINDOW_SHOWN,
    ) orelse {
        std.debug.print("SDL_CreateWindow failed: {s}\n", .{c.SDL_GetError()});
        return error.WindowCreationFailed;
    };
    defer c.SDL_DestroyWindow(window);

    // Create renderer
    const renderer = c.SDL_CreateRenderer(
        window,
        -1,
        c.SDL_RENDERER_ACCELERATED,
    ) orelse {
        std.debug.print("SDL_CreateRenderer failed: {s}\n", .{c.SDL_GetError()});
        return error.RendererCreationFailed;
    };
    defer c.SDL_DestroyRenderer(renderer);

    // Create galaxy
    const bodies = try bh.createGalaxy(allocator, INITIAL_BODIES, 2.0, 0.2);
    defer allocator.free(bodies);

    // Initialize simulation
    var simulation = bh.BarnesHutSimulation.init(allocator, bodies);
    defer simulation.deinit();

    // Calculate initial energy
    const initial_energy = simulation.calculateEnergy();
    const energy_initial = initial_energy.kinetic + initial_energy.potential;

    std.debug.print("✅ Initialization complete!\n", .{});
    std.debug.print("Initial Energy: {d:.6}\n", .{energy_initial});
    std.debug.print("Starting simulation...\n\n", .{});

    // Camera
    var camera = Camera.init();

    // Metrics
    var metrics = Metrics{
        .body_count = INITIAL_BODIES,
        .energy_initial = energy_initial,
    };

    // Main loop
    var running = true;
    var frame_count: u64 = 0;
    var last_time = std.time.nanoTimestamp();
    var fps_timer: f64 = 0;
    var fps_counter: u32 = 0;
    
    var paused = false;
    var show_help = true;

    while (running) {
        const frame_start = std.time.nanoTimestamp();

        // Handle events
        var event: c.SDL_Event = undefined;
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => running = false,
                c.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        c.SDLK_ESCAPE, c.SDLK_q => running = false,
                        c.SDLK_SPACE => paused = !paused,
                        c.SDLK_h => show_help = !show_help,
                        c.SDLK_r => {
                            // Reset simulation
                            allocator.free(simulation.bodies);
                            const new_bodies = try bh.createGalaxy(allocator, INITIAL_BODIES, 2.0, 0.2);
                            simulation.bodies = new_bodies;
                            const new_initial = simulation.calculateEnergy();
                            metrics.energy_initial = new_initial.kinetic + new_initial.potential;
                        },
                        c.SDLK_LEFT => camera.rotation_y -= 0.05,
                        c.SDLK_RIGHT => camera.rotation_y += 0.05,
                        c.SDLK_UP => camera.rotation_x -= 0.05,
                        c.SDLK_DOWN => camera.rotation_x += 0.05,
                        c.SDLK_EQUALS, c.SDLK_PLUS => camera.zoom *= 1.1,
                        c.SDLK_MINUS => camera.zoom /= 1.1,
                        else => {},
                    }
                },
                else => {},
            }
        }

        // Update simulation
        if (!paused) {
            // Build tree
            const tree_start = std.time.nanoTimestamp();
            try simulation.buildTree();
            const tree_end = std.time.nanoTimestamp();
            metrics.tree_build_time_ms = @as(f64, @floatFromInt(tree_end - tree_start)) / 1_000_000.0;
            metrics.tree_depth = simulation.tree_depth;
            metrics.node_count = simulation.node_count;

            // Calculate forces
            const force_start = std.time.nanoTimestamp();
            simulation.calculateForces();
            const force_end = std.time.nanoTimestamp();
            metrics.force_calc_time_ms = @as(f64, @floatFromInt(force_end - force_start)) / 1_000_000.0;

            // Integrate
            const integrate_start = std.time.nanoTimestamp();
            simulation.integrate();
            const integrate_end = std.time.nanoTimestamp();
            metrics.integration_time_ms = @as(f64, @floatFromInt(integrate_end - integrate_start)) / 1_000_000.0;

            // Calculate energy drift
            if (frame_count % 10 == 0) {
                const current_energy = simulation.calculateEnergy();
                metrics.energy_kinetic = current_energy.kinetic;
                metrics.energy_potential = current_energy.potential;
                metrics.energy_total = current_energy.kinetic + current_energy.potential;
                metrics.energy_drift = (metrics.energy_total - metrics.energy_initial) / metrics.energy_initial * 100.0;
            }
        }

        // Render
        _ = c.SDL_SetRenderDrawColor(renderer, 0, 0, 5, 255);
        _ = c.SDL_RenderClear(renderer);

        // Draw bodies
        for (simulation.bodies) |body| {
            const projected = camera.project(body.position);
            if (!projected.visible) continue;

            // Color based on velocity (hotter = faster)
            const speed = body.velocity.length();
            const color_value = @min(speed * 50.0, 255.0);
            _ = c.SDL_SetRenderDrawColor(
                renderer,
                @intFromFloat(color_value),
                @intFromFloat(@min(150.0, color_value * 0.5)),
                @intFromFloat(@max(0, 255.0 - color_value)),
                255,
            );

            // Draw point (or small circle for close-up)
            const point_size: i32 = @intFromFloat(@max(1, 3.0 / camera.zoom));
            if (point_size == 1) {
                _ = c.SDL_RenderDrawPoint(renderer, projected.x, projected.y);
            } else {
                const rect = c.SDL_Rect{
                    .x = projected.x - @divTrunc(point_size, 2),
                    .y = projected.y - @divTrunc(point_size, 2),
                    .w = point_size,
                    .h = point_size,
                };
                _ = c.SDL_RenderFillRect(renderer, &rect);
            }
        }

        // Draw metrics overlay
        drawMetrics(renderer, metrics, paused);

        // Draw help overlay
        if (show_help) {
            drawHelp(renderer);
        }

        c.SDL_RenderPresent(renderer);

        // Calculate frame time
        const frame_end = std.time.nanoTimestamp();
        const current_time = frame_end;
        const dt = @as(f64, @floatFromInt(current_time - last_time)) / 1_000_000_000.0;
        last_time = current_time;

        metrics.total_frame_time_ms = @as(f64, @floatFromInt(frame_end - frame_start)) / 1_000_000.0;

        // Update FPS
        fps_timer += dt;
        fps_counter += 1;

        if (fps_timer >= 0.5) {
            metrics.fps = @as(f64, @floatFromInt(fps_counter)) / fps_timer;
            fps_timer = 0;
            fps_counter = 0;
        }

        frame_count += 1;

        // Target 60 FPS
        const target_frame_time_ns = 16_666_666; // ~60 FPS
        const elapsed_ns = frame_end - frame_start;
        if (elapsed_ns < target_frame_time_ns) {
            std.Thread.sleep(@intCast(target_frame_time_ns - elapsed_ns));
        }
    }

    // Final statistics
    std.debug.print("\n\n✨ Simulation Complete!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════\n", .{});
    std.debug.print("Total Frames:          {d}\n", .{frame_count});
    std.debug.print("Final FPS:             {d:.1}\n", .{metrics.fps});
    std.debug.print("Avg Tree Build:        {d:.2} ms\n", .{metrics.tree_build_time_ms});
    std.debug.print("Avg Force Calc:        {d:.2} ms\n", .{metrics.force_calc_time_ms});
    std.debug.print("Avg Integration:       {d:.2} ms\n", .{metrics.integration_time_ms});
    std.debug.print("Avg Total Frame:       {d:.2} ms\n", .{metrics.total_frame_time_ms});
    std.debug.print("Final Energy Drift:    {d:.2}%%\n", .{metrics.energy_drift});
    std.debug.print("\n🎯 Week 1 Target: 100K bodies at 30 FPS\n", .{});
    std.debug.print("Current: {d} bodies at {d:.1} FPS\n", .{ INITIAL_BODIES, metrics.fps });
}

fn drawMetrics(renderer: *c.SDL_Renderer, metrics: Metrics, paused: bool) void {
    // Semi-transparent background
    const panel = c.SDL_Rect{ .x = 10, .y = 10, .w = 480, .h = 380 };
    _ = c.SDL_SetRenderDrawBlendMode(renderer, c.SDL_BLENDMODE_BLEND);
    _ = c.SDL_SetRenderDrawColor(renderer, 10, 10, 20, 220);
    _ = c.SDL_RenderFillRect(renderer, &panel);

    // Border
    _ = c.SDL_SetRenderDrawColor(renderer, 100, 150, 255, 255);
    _ = c.SDL_RenderDrawRect(renderer, &panel);

    var y: i32 = 20;
    const line_height: i32 = 22;

    // Title
    drawText(renderer, "BARNES-HUT GALAXY SIMULATION", 20, y, 2, .{ .r = 100, .g = 200, .b = 255 });
    y += line_height * 2;

    if (paused) {
        drawText(renderer, "*** PAUSED ***", 20, y, 2, .{ .r = 255, .g = 200, .b = 100 });
        y += line_height * 2;
    }

    // Performance metrics
    drawText(renderer, "PERFORMANCE:", 20, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height + 5;

    var buf: [64]u8 = undefined;

    const fps_str = std.fmt.bufPrint(&buf, "FPS: {d:.1}", .{metrics.fps}) catch "FPS: N/A";
    drawText(renderer, fps_str, 30, y, 1, .{ .r = 100, .g = 255, .b = 100 });
    y += line_height;

    const frame_str = std.fmt.bufPrint(&buf, "Frame Time: {d:.2} ms", .{metrics.total_frame_time_ms}) catch "Frame: N/A";
    drawText(renderer, frame_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height;

    const tree_str = std.fmt.bufPrint(&buf, "Tree Build: {d:.2} ms", .{metrics.tree_build_time_ms}) catch "Tree: N/A";
    drawText(renderer, tree_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height;

    const force_str = std.fmt.bufPrint(&buf, "Force Calc: {d:.2} ms", .{metrics.force_calc_time_ms}) catch "Force: N/A";
    drawText(renderer, force_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height;

    const integrate_str = std.fmt.bufPrint(&buf, "Integration: {d:.2} ms", .{metrics.integration_time_ms}) catch "Int: N/A";
    drawText(renderer, integrate_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height + 10;

    // Tree statistics
    drawText(renderer, "TREE STATISTICS:", 20, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height + 5;

    const bodies_str = std.fmt.bufPrint(&buf, "Bodies: {d}", .{metrics.body_count}) catch "Bodies: N/A";
    drawText(renderer, bodies_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height;

    const depth_str = std.fmt.bufPrint(&buf, "Tree Depth: {d} levels", .{metrics.tree_depth}) catch "Depth: N/A";
    drawText(renderer, depth_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height;

    const nodes_str = std.fmt.bufPrint(&buf, "Nodes: {d}", .{metrics.node_count}) catch "Nodes: N/A";
    drawText(renderer, nodes_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height + 10;

    // Energy conservation
    drawText(renderer, "PHYSICS VERIFICATION:", 20, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height + 5;

    const ke_str = std.fmt.bufPrint(&buf, "Kinetic: {d:.4}", .{metrics.energy_kinetic}) catch "KE: N/A";
    drawText(renderer, ke_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height;

    const pe_str = std.fmt.bufPrint(&buf, "Potential: {d:.4}", .{metrics.energy_potential}) catch "PE: N/A";
    drawText(renderer, pe_str, 30, y, 1, .{ .r = 200, .g = 200, .b = 200 });
    y += line_height;

    const drift_str = std.fmt.bufPrint(&buf, "Energy Drift: {d:.3}%%", .{metrics.energy_drift}) catch "Drift: N/A";
    const drift_color = if (@abs(metrics.energy_drift) < 1.0)
        SDL_Color{ .r = 100, .g = 255, .b = 100 }
    else if (@abs(metrics.energy_drift) < 5.0)
        SDL_Color{ .r = 255, .g = 200, .b = 100 }
    else
        SDL_Color{ .r = 255, .g = 100, .b = 100 };
    drawText(renderer, drift_str, 30, y, 1, drift_color);
}

fn drawHelp(renderer: *c.SDL_Renderer) void {
    const panel = c.SDL_Rect{ .x = WINDOW_WIDTH - 410, .y = 10, .w = 400, .h = 260 };
    _ = c.SDL_SetRenderDrawBlendMode(renderer, c.SDL_BLENDMODE_BLEND);
    _ = c.SDL_SetRenderDrawColor(renderer, 10, 10, 20, 220);
    _ = c.SDL_RenderFillRect(renderer, &panel);

    _ = c.SDL_SetRenderDrawColor(renderer, 100, 150, 255, 255);
    _ = c.SDL_RenderDrawRect(renderer, &panel);

    var y: i32 = 20;
    const line_height: i32 = 22;

    drawText(renderer, "CONTROLS:", WINDOW_WIDTH - 400, y, 1, .{ .r = 100, .g = 200, .b = 255 });
    y += line_height + 5;

    const controls = [_][]const u8{
        "SPACE - Pause/Resume",
        "R - Reset simulation",
        "H - Toggle help",
        "ESC/Q - Exit",
        "",
        "ARROWS - Rotate camera",
        "+/- - Zoom in/out",
        "",
        "Color Legend:",
        "Blue - Slow particles",
        "Red - Fast particles",
    };

    for (controls) |text| {
        drawText(renderer, text, WINDOW_WIDTH - 390, y, 1, .{ .r = 200, .g = 200, .b = 200 });
        y += line_height;
    }
}

const SDL_Color = struct { r: u8, g: u8, b: u8 };

fn drawText(renderer: *c.SDL_Renderer, text: []const u8, x: i32, y: i32, scale: u8, color: SDL_Color) void {
    _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 255);

    var current_x = x;
    const char_width = 8 * @as(i32, scale);

    for (text) |char| {
        if (char == ' ') {
            current_x += char_width;
            continue;
        }

        const char_data = getCharBitmap(char);

        for (0..7) |row| {
            for (0..5) |col| {
                if ((char_data[row] >> @as(u3, @intCast(4 - col))) & 1 != 0) {
                    const rect = c.SDL_Rect{
                        .x = current_x + @as(i32, @intCast(col)) * @as(i32, scale),
                        .y = y + @as(i32, @intCast(row)) * @as(i32, scale),
                        .w = @as(i32, scale),
                        .h = @as(i32, scale),
                    };
                    _ = c.SDL_RenderFillRect(renderer, &rect);
                }
            }
        }

        current_x += char_width;
    }
}

fn getCharBitmap(char: u8) [7]u8 {
    return switch (char) {
        'A'...'Z' => blk: {
            const patterns = [_][7]u8{
                .{ 0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 }, // A
                .{ 0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110 }, // B
                .{ 0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110 }, // C
                .{ 0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110 }, // D
                .{ 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111 }, // E
                .{ 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000 }, // F
                .{ 0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110 }, // G
                .{ 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 }, // H
                .{ 0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 }, // I
                .{ 0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100 }, // J
                .{ 0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001 }, // K
                .{ 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111 }, // L
                .{ 0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001 }, // M
                .{ 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001 }, // N
                .{ 0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 }, // O
                .{ 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000 }, // P
                .{ 0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101 }, // Q
                .{ 0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001 }, // R
                .{ 0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110 }, // S
                .{ 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 }, // T
                .{ 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 }, // U
                .{ 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100 }, // V
                .{ 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001 }, // W
                .{ 0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001 }, // X
                .{ 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100 }, // Y
                .{ 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111 }, // Z
            };
            const offset = char - 'A';
            if (offset < 26) break :blk patterns[offset];
            break :blk .{ 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000 };
        },
        '0'...'9' => blk: {
            const patterns = [_][7]u8{
                .{ 0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110 }, // 0
                .{ 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 }, // 1
                .{ 0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111 }, // 2
                .{ 0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110 }, // 3
                .{ 0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010 }, // 4
                .{ 0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110 }, // 5
                .{ 0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110 }, // 6
                .{ 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000 }, // 7
                .{ 0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110 }, // 8
                .{ 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100 }, // 9
            };
            break :blk patterns[char - '0'];
        },
        '.' => .{ 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100 },
        ',' => .{ 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00100 },
        ':' => .{ 0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000 },
        '%' => .{ 0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011 },
        '(' => .{ 0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010 },
        ')' => .{ 0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000 },
        '+' => .{ 0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000 },
        '-' => .{ 0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000 },
        '/' => .{ 0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000 },
        '*' => .{ 0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000 },
        '=' => .{ 0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000 },
        '<' => .{ 0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010 },
        '>' => .{ 0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000 },
        else => .{ 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000 },
    };
}