const std = @import("std");

const c = @cImport({
    @cInclude("SDL2/SDL.h");
});

// Configuration
const WIDTH = 1600;
const HEIGHT = 1000;
const PARTICLE_COUNT = 50_000;
const TARGET_FPS = 60;

// Performance comparison data (estimated based on typical benchmarks)
const LanguageMultiplier = struct {
    name: []const u8,
    multiplier: f32,
    color: SDL_Color,
};

const language_comparisons = [_]LanguageMultiplier{
    .{ .name = "Zig", .multiplier = 1.0, .color = .{ .r = 100, .g = 255, .b = 100, .a = 255 } },
    .{ .name = "C", .multiplier = 1.05, .color = .{ .r = 150, .g = 200, .b = 255, .a = 255 } },
    .{ .name = "Rust", .multiplier = 1.15, .color = .{ .r = 255, .g = 150, .b = 100, .a = 255 } },
    .{ .name = "Go", .multiplier = 3.5, .color = .{ .r = 100, .g = 200, .b = 255, .a = 255 } },
    .{ .name = "JavaScript", .multiplier = 45.0, .color = .{ .r = 255, .g = 255, .b = 100, .a = 255 } },
    .{ .name = "Python", .multiplier = 150.0, .color = .{ .r = 255, .g = 200, .b = 100, .a = 255 } },
};

const Vec2 = struct {
    x: f32,
    y: f32,
    
    pub fn add(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }
    
    pub fn sub(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x - b.x, .y = a.y - b.y };
    }
    
    pub fn mul(a: Vec2, scalar: f32) Vec2 {
        return .{ .x = a.x * scalar, .y = a.y * scalar };
    }
    
    pub fn length(self: Vec2) f32 {
        return @sqrt(self.x * self.x + self.y * self.y);
    }
    
    pub fn normalize(self: Vec2) Vec2 {
        const len = self.length();
        if (len == 0) return Vec2{ .x = 0, .y = 0 };
        return .{ .x = self.x / len, .y = self.y / len };
    }
};

const SDL_Color = struct { r: u8, g: u8, b: u8, a: u8 };

const Particle = struct {
    position: Vec2,
    velocity: Vec2,
    color: SDL_Color,
    radius: f32,
    mass: f32,
    
    pub fn update(self: *Particle, dt: f32) void {
        // Apply gravity
        self.velocity.y += 500.0 * dt;
        
        // Update position
        self.position = self.position.add(self.velocity.mul(dt));
        
        // Boundary collisions
        if (self.position.x < self.radius) {
            self.position.x = self.radius;
            self.velocity.x = -self.velocity.x * 0.95;
        }
        if (self.position.x > WIDTH - self.radius) {
            self.position.x = WIDTH - self.radius;
            self.velocity.x = -self.velocity.x * 0.95;
        }
        if (self.position.y < self.radius) {
            self.position.y = self.radius;
            self.velocity.y = -self.velocity.y * 0.95;
        }
        if (self.position.y > HEIGHT - self.radius) {
            self.position.y = HEIGHT - self.radius;
            self.velocity.y = -self.velocity.y * 0.95;
        }
        
        // Air resistance
        self.velocity = self.velocity.mul(0.999);
    }
    
    pub fn applyForce(self: *Particle, force: Vec2) void {
        self.velocity = self.velocity.add(force.mul(1.0 / self.mass));
    }
    
    pub fn draw(self: Particle, renderer: *c.SDL_Renderer) void {
        _ = c.SDL_SetRenderDrawColor(renderer, self.color.r, self.color.g, self.color.b, self.color.a);
        
        // Draw filled circle
        const x = @as(i32, @intFromFloat(self.position.x));
        const y = @as(i32, @intFromFloat(self.position.y));
        const r = @as(i32, @intFromFloat(self.radius));
        
        var offset_x: i32 = 0;
        var offset_y = r;
        var d: i32 = 3 - 2 * r;
        
        while (offset_x <= offset_y) {
            // Draw horizontal lines for filled circle
            _ = c.SDL_RenderDrawLine(renderer, x - offset_y, y + offset_x, x + offset_y, y + offset_x);
            _ = c.SDL_RenderDrawLine(renderer, x - offset_y, y - offset_x, x + offset_y, y - offset_x);
            _ = c.SDL_RenderDrawLine(renderer, x - offset_x, y + offset_y, x + offset_x, y + offset_y);
            _ = c.SDL_RenderDrawLine(renderer, x - offset_x, y - offset_y, x + offset_x, y - offset_y);
            
            if (d < 0) {
                d = d + 4 * offset_x + 6;
            } else {
                d = d + 4 * (offset_x - offset_y) + 10;
                offset_y -= 1;
            }
            offset_x += 1;
        }
    }
};

const PerformanceMetrics = struct {
    fps: f32 = 0,
    frame_time_ms: f32 = 0,
    update_time_ms: f32 = 0,
    render_time_ms: f32 = 0,
    particles_per_sec: f64 = 0,
    memory_mb: f32 = 0,
    
    // Comparison data
    estimated_fps: [6]f32 = undefined,
    
    pub fn updateComparisons(self: *PerformanceMetrics) void {
        for (language_comparisons, 0..) |lang, i| {
            self.estimated_fps[i] = self.fps / lang.multiplier;
        }
    }
};

const ParticleSystem = struct {
    particles: []Particle,
    allocator: std.mem.Allocator,
    window: *c.SDL_Window,
    renderer: *c.SDL_Renderer,
    metrics: PerformanceMetrics,
    
    frame_count: u64 = 0,
    total_update_time: u64 = 0,
    total_render_time: u64 = 0,
    
    mouse_x: i32 = 0,
    mouse_y: i32 = 0,
    mouse_attract: bool = false,
    mouse_repel: bool = false,
    
    show_metrics: bool = true,
    paused: bool = false,
    
    pub fn init(allocator: std.mem.Allocator) !*ParticleSystem {
        // Initialize SDL
        if (c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_EVENTS) != 0) {
            return error.SDLInitFailed;
        }
        
        const window = c.SDL_CreateWindow(
            "🎯 Zig Particle Physics - Performance Showcase",
            c.SDL_WINDOWPOS_CENTERED,
            c.SDL_WINDOWPOS_CENTERED,
            WIDTH,
            HEIGHT,
            c.SDL_WINDOW_SHOWN
        ) orelse return error.WindowCreationFailed;
        
        const renderer = c.SDL_CreateRenderer(
            window,
            -1,
            c.SDL_RENDERER_ACCELERATED
        ) orelse return error.RendererCreationFailed;
        
        // Allocate particles
        const particles = try allocator.alloc(Particle, PARTICLE_COUNT);
        
        // Initialize particles
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const rand = prng.random();
        
        for (particles) |*p| {
            p.* = .{
                .position = .{
                    .x = rand.float(f32) * WIDTH,
                    .y = rand.float(f32) * HEIGHT * 0.5,
                },
                .velocity = .{
                    .x = (rand.float(f32) - 0.5) * 400.0,
                    .y = (rand.float(f32) - 0.5) * 400.0,
                },
                .color = .{
                    .r = @intCast(rand.intRangeAtMost(u8, 100, 255)),
                    .g = @intCast(rand.intRangeAtMost(u8, 100, 255)),
                    .b = @intCast(rand.intRangeAtMost(u8, 100, 255)),
                    .a = 255,
                },
                .radius = 2.0 + rand.float(f32) * 2.0,
                .mass = 1.0,
            };
        }
        
        const system = try allocator.create(ParticleSystem);
        system.* = .{
            .particles = particles,
            .allocator = allocator,
            .window = window,
            .renderer = renderer,
            .metrics = .{
                .memory_mb = @as(f32, @floatFromInt(PARTICLE_COUNT * @sizeOf(Particle))) / (1024.0 * 1024.0),
            },
        };
        
        return system;
    }
    
    pub fn deinit(self: *ParticleSystem) void {
        const allocator = self.allocator;
        allocator.free(self.particles);
        c.SDL_DestroyRenderer(self.renderer);
        c.SDL_DestroyWindow(self.window);
        c.SDL_Quit();
        allocator.destroy(self);
    }
    
    pub fn update(self: *ParticleSystem, dt: f32) void {
        const update_start = std.time.nanoTimestamp();
        
        if (!self.paused) {
            // Update all particles
            for (self.particles) |*p| {
                p.update(dt);
                
                // Mouse interaction
                if (self.mouse_attract or self.mouse_repel) {
                    const mouse_pos = Vec2{
                        .x = @floatFromInt(self.mouse_x),
                        .y = @floatFromInt(self.mouse_y),
                    };
                    const to_mouse = mouse_pos.sub(p.position);
                    const distance = to_mouse.length();
                    
                    if (distance > 0 and distance < 300.0) {
                        const direction = to_mouse.normalize();
                        const strength: f32 = if (self.mouse_attract) 10000.0 else -10000.0;
                        const force = direction.mul(strength / (distance * distance));
                        p.applyForce(force);
                    }
                }
            }
        }
        
        const update_end = std.time.nanoTimestamp();
        const update_time_ns = @as(u64, @intCast(update_end - update_start));
        self.total_update_time += update_time_ns;
        self.metrics.update_time_ms = @as(f32, @floatFromInt(update_time_ns)) / 1_000_000.0;
    }
    
    pub fn render(self: *ParticleSystem) void {
        const render_start = std.time.nanoTimestamp();
        
        // Clear screen
        _ = c.SDL_SetRenderDrawColor(self.renderer, 10, 10, 20, 255);
        _ = c.SDL_RenderClear(self.renderer);
        
        // Draw particles
        for (self.particles) |p| {
            p.draw(self.renderer);
        }
        
        // Draw metrics overlay
        if (self.show_metrics) {
            self.drawMetricsOverlay();
        }
        
        // Draw instructions
        self.drawInstructions();
        
        // Draw mouse interaction zone
        if (self.mouse_attract or self.mouse_repel) {
            const color: SDL_Color = if (self.mouse_attract)
                .{ .r = 100, .g = 255, .b = 100, .a = 100 }
            else
                .{ .r = 255, .g = 100, .b = 100, .a = 100 };
            
            _ = c.SDL_SetRenderDrawBlendMode(self.renderer, c.SDL_BLENDMODE_BLEND);
            _ = c.SDL_SetRenderDrawColor(self.renderer, color.r, color.g, color.b, color.a);
            
            // Draw interaction radius
            drawCircle(self.renderer, self.mouse_x, self.mouse_y, 300);
        }
        
        c.SDL_RenderPresent(self.renderer);
        
        const render_end = std.time.nanoTimestamp();
        const render_time_ns = @as(u64, @intCast(render_end - render_start));
        self.total_render_time += render_time_ns;
        self.metrics.render_time_ms = @as(f32, @floatFromInt(render_time_ns)) / 1_000_000.0;
    }
    
    fn drawMetricsOverlay(self: *ParticleSystem) void {
        // Semi-transparent background for metrics
        const panel_rect = c.SDL_Rect{ .x = 10, .y = 10, .w = 550, .h = 480 };
        _ = c.SDL_SetRenderDrawBlendMode(self.renderer, c.SDL_BLENDMODE_BLEND);
        _ = c.SDL_SetRenderDrawColor(self.renderer, 20, 20, 40, 220);
        _ = c.SDL_RenderFillRect(self.renderer, &panel_rect);
        
        // Border
        _ = c.SDL_SetRenderDrawColor(self.renderer, 100, 150, 255, 255);
        _ = c.SDL_RenderDrawRect(self.renderer, &panel_rect);
        
        // Title
        drawTextSimple(self.renderer, "⚡ PERFORMANCE METRICS", 20, 20, .{ .r = 100, .g = 200, .b = 255, .a = 255 }, 2);
        
        // Current metrics
        var y: i32 = 60;
        const spacing: i32 = 25;
        
        drawTextSimple(self.renderer, "═══ CURRENT PERFORMANCE ═══", 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 }, 1);
        y += spacing + 10;
        
        drawMetricLine(self.renderer, "FPS:", self.metrics.fps, 20, y, .{ .r = 255, .g = 255, .b = 255, .a = 255 });
        y += spacing;
        
        drawMetricLine(self.renderer, "Frame Time:", self.metrics.frame_time_ms, 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 });
        y += spacing;
        
        drawMetricLine(self.renderer, "Update Time:", self.metrics.update_time_ms, 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 });
        y += spacing;
        
        drawMetricLine(self.renderer, "Render Time:", self.metrics.render_time_ms, 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 });
        y += spacing;
        
        const particles_per_sec = @as(f64, @floatFromInt(PARTICLE_COUNT)) * self.metrics.fps;
        var buf: [64]u8 = undefined;
        const pps_str = std.fmt.bufPrint(&buf, "{d:.2}M/s", .{particles_per_sec / 1_000_000.0}) catch "N/A";
        drawTextSimple(self.renderer, "Particles/sec:", 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 }, 1);
        drawTextSimple(self.renderer, pps_str, 280, y, .{ .r = 100, .g = 255, .b = 100, .a = 255 }, 1);
        y += spacing;
        
        const mem_str = std.fmt.bufPrint(&buf, "{d:.2} MB", .{self.metrics.memory_mb}) catch "N/A";
        drawTextSimple(self.renderer, "Memory:", 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 }, 1);
        drawTextSimple(self.renderer, mem_str, 280, y, .{ .r = 255, .g = 200, .b = 100, .a = 255 }, 1);
        y += spacing + 15;
        
        // Language comparison
        drawTextSimple(self.renderer, "═══ LANGUAGE COMPARISON ═══", 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 }, 1);
        y += spacing + 10;
        
        self.metrics.updateComparisons();
        
        for (language_comparisons, 0..) |lang, i| {
            const fps = self.metrics.estimated_fps[i];
            const fps_str = std.fmt.bufPrint(&buf, "{d:.1} FPS", .{fps}) catch "N/A";
            
            // Language name
            drawTextSimple(self.renderer, lang.name, 20, y, lang.color, 1);
            
            // FPS value
            drawTextSimple(self.renderer, fps_str, 150, y, lang.color, 1);
            
            // Performance bar
            const bar_ratio: f32 = if (self.metrics.fps > 0) fps / self.metrics.fps else 0.0;
            const bar_width = @as(i32, @intFromFloat(@min(300.0 * bar_ratio, 300.0)));
            const bar_rect = c.SDL_Rect{
                .x = 250,
                .y = y + 2,
                .w = bar_width,
                .h = 18,
            };
            _ = c.SDL_SetRenderDrawBlendMode(self.renderer, c.SDL_BLENDMODE_BLEND);
            _ = c.SDL_SetRenderDrawColor(self.renderer, lang.color.r, lang.color.g, lang.color.b, 180);
            _ = c.SDL_RenderFillRect(self.renderer, &bar_rect);
            
            y += spacing;
        }
    }
    
    fn drawInstructions(self: *ParticleSystem) void {
        const panel_rect = c.SDL_Rect{ .x = 10, .y = HEIGHT - 150, .w = 550, .h = 140 };
        _ = c.SDL_SetRenderDrawBlendMode(self.renderer, c.SDL_BLENDMODE_BLEND);
        _ = c.SDL_SetRenderDrawColor(self.renderer, 20, 20, 40, 220);
        _ = c.SDL_RenderFillRect(self.renderer, &panel_rect);
        
        _ = c.SDL_SetRenderDrawColor(self.renderer, 100, 150, 255, 255);
        _ = c.SDL_RenderDrawRect(self.renderer, &panel_rect);
        
        var y: i32 = HEIGHT - 140;
        const spacing: i32 = 22;
        
        drawTextSimple(self.renderer, "CONTROLS:", 20, y, .{ .r = 100, .g = 200, .b = 255, .a = 255 }, 1);
        y += spacing + 5;
        
        drawTextSimple(self.renderer, "• LEFT CLICK: Attract particles", 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 }, 1);
        y += spacing;
        
        drawTextSimple(self.renderer, "• RIGHT CLICK: Repel particles", 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 }, 1);
        y += spacing;
        
        drawTextSimple(self.renderer, "• SPACE: Pause/Resume", 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 }, 1);
        y += spacing;
        
        drawTextSimple(self.renderer, "• M: Toggle metrics | R: Reset | ESC: Exit", 20, y, .{ .r = 200, .g = 200, .b = 200, .a = 255 }, 1);
    }
    
    pub fn handleEvents(self: *ParticleSystem) bool {
        var event: c.SDL_Event = undefined;
        
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => return false,
                c.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        c.SDLK_ESCAPE, c.SDLK_q => return false,
                        c.SDLK_SPACE => self.paused = !self.paused,
                        c.SDLK_m => self.show_metrics = !self.show_metrics,
                        c.SDLK_r => self.reset(),
                        else => {},
                    }
                },
                c.SDL_MOUSEMOTION => {
                    self.mouse_x = event.motion.x;
                    self.mouse_y = event.motion.y;
                },
                c.SDL_MOUSEBUTTONDOWN => {
                    if (event.button.button == c.SDL_BUTTON_LEFT) {
                        self.mouse_attract = true;
                    } else if (event.button.button == c.SDL_BUTTON_RIGHT) {
                        self.mouse_repel = true;
                    }
                },
                c.SDL_MOUSEBUTTONUP => {
                    if (event.button.button == c.SDL_BUTTON_LEFT) {
                        self.mouse_attract = false;
                    } else if (event.button.button == c.SDL_BUTTON_RIGHT) {
                        self.mouse_repel = false;
                    }
                },
                else => {},
            }
        }
        
        return true;
    }
    
    fn reset(self: *ParticleSystem) void {
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const rand = prng.random();
        
        for (self.particles) |*p| {
            p.position.x = rand.float(f32) * WIDTH;
            p.position.y = rand.float(f32) * HEIGHT * 0.5;
            p.velocity.x = (rand.float(f32) - 0.5) * 400.0;
            p.velocity.y = (rand.float(f32) - 0.5) * 400.0;
        }
    }
    
    pub fn run(self: *ParticleSystem) !void {
        var last_time = std.time.nanoTimestamp();
        var fps_timer: f64 = 0;
        var fps_counter: u32 = 0;
        
        var running = true;
        while (running) {
            const current_time = std.time.nanoTimestamp();
            const dt = @as(f32, @floatFromInt(current_time - last_time)) / 1_000_000_000.0;
            last_time = current_time;
            
            // Handle events
            running = self.handleEvents();
            
            // Update physics
            self.update(dt);
            
            // Render
            self.render();
            
            // Calculate FPS
            fps_timer += dt;
            fps_counter += 1;
            
            if (fps_timer >= 0.5) {
                self.metrics.fps = @as(f32, @floatFromInt(fps_counter)) / @as(f32, @floatCast(fps_timer));
                self.metrics.frame_time_ms = (@as(f32, @floatCast(fps_timer)) / @as(f32, @floatFromInt(fps_counter))) * 1000.0;
                self.metrics.particles_per_sec = @as(f64, @floatFromInt(PARTICLE_COUNT)) * self.metrics.fps;
                fps_timer = 0;
                fps_counter = 0;
            }
            
            self.frame_count += 1;
            
            // Frame limiting
            const frame_time_ns: i128 = std.time.nanoTimestamp() - current_time;
            const target_frame_time_ns = 1_000_000_000 / TARGET_FPS;
            if (frame_time_ns < target_frame_time_ns) {
                const sleep_ns: u64 = @intCast(target_frame_time_ns - frame_time_ns);
                std.Thread.sleep(sleep_ns);
            }
        }
        
        // Final statistics
        const avg_update_ms = @as(f64, @floatFromInt(self.total_update_time)) / @as(f64, @floatFromInt(self.frame_count)) / 1_000_000.0;
        const avg_render_ms = @as(f64, @floatFromInt(self.total_render_time)) / @as(f64, @floatFromInt(self.frame_count)) / 1_000_000.0;
        
        std.debug.print("\n✨ Session Summary\n", .{});
        std.debug.print("══════════════════════════════════════\n", .{});
        std.debug.print("Total Frames:      {d}\n", .{self.frame_count});
        std.debug.print("Avg Update Time:   {d:.2} ms\n", .{avg_update_ms});
        std.debug.print("Avg Render Time:   {d:.2} ms\n", .{avg_render_ms});
        std.debug.print("Final FPS:         {d:.1}\n", .{self.metrics.fps});
        std.debug.print("Particles:         {d}\n", .{PARTICLE_COUNT});
        std.debug.print("Total Updates:     {d} million\n", .{self.frame_count * PARTICLE_COUNT / 1_000_000});
    }
};

fn drawCircle(renderer: *c.SDL_Renderer, cx: i32, cy: i32, radius: i32) void {
    var x: i32 = 0;
    var y = radius;
    var d: i32 = 3 - 2 * radius;
    
    while (x <= y) {
        // Draw 8 octants
        _ = c.SDL_RenderDrawPoint(renderer, cx + x, cy + y);
        _ = c.SDL_RenderDrawPoint(renderer, cx - x, cy + y);
        _ = c.SDL_RenderDrawPoint(renderer, cx + x, cy - y);
        _ = c.SDL_RenderDrawPoint(renderer, cx - x, cy - y);
        _ = c.SDL_RenderDrawPoint(renderer, cx + y, cy + x);
        _ = c.SDL_RenderDrawPoint(renderer, cx - y, cy + x);
        _ = c.SDL_RenderDrawPoint(renderer, cx + y, cy - x);
        _ = c.SDL_RenderDrawPoint(renderer, cx - y, cy - x);
        
        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y -= 1;
        }
        x += 1;
    }
}

fn drawMetricLine(renderer: *c.SDL_Renderer, label: []const u8, value: f32, x: i32, y: i32, color: SDL_Color) void {
    drawTextSimple(renderer, label, x, y, color, 1);
    
    var buf: [32]u8 = undefined;
    const value_str = std.fmt.bufPrint(&buf, "{d:.1}", .{value}) catch "N/A";
    drawTextSimple(renderer, value_str, x + 260, y, .{ .r = 100, .g = 255, .b = 100, .a = 255 }, 1);
}

fn drawTextSimple(renderer: *c.SDL_Renderer, text: []const u8, x: i32, y: i32, color: SDL_Color, scale: u8) void {
    _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    
    var current_x = x;
    const char_width = 8 * @as(i32, scale);
    
    for (text) |char| {
        if (char == ' ') {
            current_x += char_width;
            continue;
        }
        
        // Simple 5x7 pixel font (ultra-minimal)
        const char_data = getCharBitmap(char);
        
        for (0..7) |row| {
            for (0..5) |col| {
                if ((char_data[row] >> @as(u3, @intCast(4 - col))) & 1 != 0) {
                    const rect = c.SDL_Rect{
                        .x = current_x + @as(i32, @intCast(col)) * scale,
                        .y = y + @as(i32, @intCast(row)) * scale,
                        .w = scale,
                        .h = scale,
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
        'A' => .{ 0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
        'C' => .{ 0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110 },
        'E' => .{ 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111 },
        'F' => .{ 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000 },
        'G' => .{ 0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110 },
        'I' => .{ 0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 },
        'L' => .{ 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111 },
        'M' => .{ 0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001 },
        'N' => .{ 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001 },
        'O' => .{ 0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 },
        'P' => .{ 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000 },
        'R' => .{ 0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001 },
        'S' => .{ 0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110 },
        'T' => .{ 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 },
        'U' => .{ 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 },
        'Z' => .{ 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111 },
        '0' => .{ 0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110 },
        '1' => .{ 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 },
        '2' => .{ 0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111 },
        '3' => .{ 0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110 },
        '4' => .{ 0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010 },
        '5' => .{ 0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110 },
        '6' => .{ 0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110 },
        '7' => .{ 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000 },
        '8' => .{ 0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110 },
        '9' => .{ 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100 },
        '.' => .{ 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100 },
        ':' => .{ 0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000 },
        '/' => .{ 0b00000, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000 },
        '-' => .{ 0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000 },
        '=' => .{ 0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000 },
        '(' => .{ 0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010 },
        ')' => .{ 0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000 },
        '|' => .{ 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 },
        // Use ASCII equivalents for special characters
        '*' => .{ 0b00000, 0b00000, 0b00000, 0b01110, 0b01110, 0b01110, 0b00000 },
        '#' => .{ 0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010 },
        '<' => .{ 0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010 },
        else => .{ 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000 },
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n🚀 Zig Visual Particle Physics Demo\n", .{});
    std.debug.print("══════════════════════════════════════\n\n", .{});
    std.debug.print("Initializing {d} particles...\n\n", .{PARTICLE_COUNT});
    
    var system = try ParticleSystem.init(allocator);
    defer system.deinit();
    
    std.debug.print("✨ System ready!\n", .{});
    std.debug.print("Window: {d}x{d}\n", .{ WIDTH, HEIGHT });
    std.debug.print("Particles: {d}\n", .{PARTICLE_COUNT});
    std.debug.print("Memory: {d:.2} MB\n\n", .{system.metrics.memory_mb});
    
    try system.run();
}