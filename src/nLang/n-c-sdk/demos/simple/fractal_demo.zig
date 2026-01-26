const std = @import("std");

// SDL2 Integration
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});

// Fractal Configuration
const WIDTH = 1200;
const HEIGHT = 900;
const MAX_ITERATIONS = 512;
const ZOOM_FACTOR = 1.2;
const PAN_SPEED = 0.1;

// Complex number type for performance
const Complex = extern struct {
    real: f64,
    imag: f64,

    comptime {
        @compileLog("Using aligned Complex struct for SIMD optimization");
    }

    pub inline fn add(a: Complex, b: Complex) Complex {
        return .{ .real = a.real + b.real, .imag = a.imag + b.imag };
    }

    pub inline fn mul(a: Complex, b: Complex) Complex {
        return .{
            .real = a.real * b.real - a.imag * b.imag,
            .imag = a.real * b.imag + a.imag * b.real,
        };
    }

    pub inline fn magnitudeSquared(a: Complex) f64 {
        return a.real * a.real + a.imag * a.imag;
    }
};

// Color gradient with smooth interpolation
const Color = extern struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8 = 255,

    pub fn fromHSV(h: f32, s: f32, v: f32) Color {
        const chroma = v * s;
        const x = chroma * (1.0 - @abs(@mod(h / 60.0, 2.0) - 1.0));
        const m = v - chroma;

        var r1: f32 = 0;
        var g1: f32 = 0;
        var b1: f32 = 0;

        if (h < 60) {
            r1 = chroma; g1 = x; b1 = 0;
        } else if (h < 120) {
            r1 = x; g1 = chroma; b1 = 0;
        } else if (h < 180) {
            r1 = 0; g1 = chroma; b1 = x;
        } else if (h < 240) {
            r1 = 0; g1 = x; b1 = chroma;
        } else if (h < 300) {
            r1 = x; g1 = 0; b1 = chroma;
        } else {
            r1 = chroma; g1 = 0; b1 = x;
        }

        return .{
            .r = @intFromFloat(255 * (r1 + m)),
            .g = @intFromFloat(255 * (g1 + m)),
            .b = @intFromFloat(255 * (b1 + m)),
        };
    }

    pub fn toU32(self: Color) u32 {
        return (@as(u32, self.r) << 24) |
               (@as(u32, self.g) << 16) |
               (@as(u32, self.b) << 8) |
               self.a;
    }
};

// Thread-safe fractal computation context
const RenderContext = struct {
    allocator: std.mem.Allocator,
    buffer: []u32,
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    max_iterations: u32,
    
    // Multi-threading
    start_y: usize,
    end_y: usize,
    thread_id: usize,
    
    // Performance tracking
    pixels_computed: usize = 0,
    time_taken_ns: u64 = 0,
    
    pub fn renderSlice(self: *RenderContext) void {
        const start_time = std.time.nanoTimestamp();
        var pixels_computed: usize = 0;
        
        const scale = 4.0 / (self.zoom * @as(f64, @floatFromInt(self.width)));
        
        for (self.start_y..self.end_y) |y| {
            const imag = self.center_y + (@as(f64, @floatFromInt(y)) - @as(f64, @floatFromInt(self.height)) / 2.0) * scale;
            
            for (0..self.width) |x| {
                const real = self.center_x + (@as(f64, @floatFromInt(x)) - @as(f64, @floatFromInt(self.width)) / 2.0) * scale;
                
                var z = Complex{ .real = 0, .imag = 0 };
                const complex_c = Complex{ .real = real, .imag = imag };
                var iteration: u32 = 0;
                
                // Optimized escape algorithm
                while (iteration < self.max_iterations) {
                    const z_real_sq = z.real * z.real;
                    const z_imag_sq = z.imag * z.imag;
                    
                    if (z_real_sq + z_imag_sq > 4.0) break;
                    
                    // Optimized: z = z*z + c
                    const new_real = z_real_sq - z_imag_sq + complex_c.real;
                    const new_imag = 2.0 * z.real * z.imag + complex_c.imag;
                    
                    z.real = new_real;
                    z.imag = new_imag;
                    iteration += 1;
                }
                
                // Smooth coloring algorithm
                const index = y * self.width + x;
                if (iteration == self.max_iterations) {
                    self.buffer[index] = 0xFF000000; // Black for interior
                } else {
                    // Smoothed iteration count
                    const zn = @sqrt(z.real * z.real + z.imag * z.imag);
                    const nu = @log(@log(zn) / 2.0) / @log(2.0);
                    const iter_smooth = @as(f32, @floatFromInt(iteration)) + 1.0 - nu;
                    
                    // Color mapping with smooth gradient
                    const t = iter_smooth / @as(f32, @floatFromInt(self.max_iterations));
                    const hue = t * 360.0 * 3.0; // Triple hue range for more colors
                    const color = Color.fromHSV(@mod(hue, 360.0), 0.8, 1.0);
                    self.buffer[index] = color.toU32();
                }
                
                pixels_computed += 1;
            }
        }
        
        self.pixels_computed = pixels_computed;
        self.time_taken_ns = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
    }
};

// SDL2 Window Manager
const FractalWindow = struct {
    window: *c.SDL_Window,
    renderer: *c.SDL_Renderer,
    texture: *c.SDL_Texture,
    buffer: []u32,
    allocator: std.mem.Allocator,
    
    // Fractal state
    center_x: f64 = -0.5,
    center_y: f64 = 0.0,
    zoom: f64 = 1.0,
    max_iterations: u32 = MAX_ITERATIONS,
    
    // Performance tracking
    frame_count: u64 = 0,
    last_frame_time: u64 = 0,
    fps: f64 = 0,
    total_render_time: u64 = 0,
    
    // Thread pool
    thread_pool: std.Thread.Pool,
    thread_count: usize,
    
    pub fn init(allocator: std.mem.Allocator) !*FractalWindow {
        // Initialize SDL
        if (c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_EVENTS) != 0) {
            std.debug.print("SDL_Init Error: {s}\n", .{c.SDL_GetError()});
            return error.SDLInitFailed;
        }
        
        // Create window
        const window = c.SDL_CreateWindow(
            "🌀 Zig Fractal Visualizer - 1M Pixels Real-time",
            c.SDL_WINDOWPOS_CENTERED,
            c.SDL_WINDOWPOS_CENTERED,
            @intCast(WIDTH),
            @intCast(HEIGHT),
            c.SDL_WINDOW_SHOWN | c.SDL_WINDOW_RESIZABLE
        ) orelse {
            std.debug.print("SDL_CreateWindow Error: {s}\n", .{c.SDL_GetError()});
            return error.WindowCreationFailed;
        };
        
        // Create renderer
        const renderer = c.SDL_CreateRenderer(
            window,
            -1,
            c.SDL_RENDERER_ACCELERATED | c.SDL_RENDERER_PRESENTVSYNC
        ) orelse {
            std.debug.print("SDL_CreateRenderer Error: {s}\n", .{c.SDL_GetError()});
            return error.RendererCreationFailed;
        };
        
        // Create texture for fast pixel updates
        const texture = c.SDL_CreateTexture(
            renderer,
            c.SDL_PIXELFORMAT_ARGB8888,
            c.SDL_TEXTUREACCESS_STREAMING,
            @intCast(WIDTH),
            @intCast(HEIGHT)
        ) orelse {
            std.debug.print("SDL_CreateTexture Error: {s}\n", .{c.SDL_GetError()});
            return error.TextureCreationFailed;
        };
        
        // Allocate pixel buffer
        const buffer = try allocator.alloc(u32, WIDTH * HEIGHT);
        
        // Initialize thread pool (use all CPU cores)
        const thread_count = @min(try std.Thread.getCpuCount(), 16);
        var thread_pool: std.Thread.Pool = undefined;
        try thread_pool.init(.{ 
            .allocator = allocator,
            .n_jobs = thread_count,
        });
        
        const fractal_window = try allocator.create(FractalWindow);
        fractal_window.* = .{
            .allocator = allocator,
            .window = window,
            .renderer = renderer,
            .texture = texture,
            .buffer = buffer,
            .thread_pool = thread_pool,
            .thread_count = thread_count,
        };
        
        return fractal_window;
    }
    
    pub fn deinit(self: *FractalWindow) void {
        const allocator = self.allocator;
        self.thread_pool.deinit();
        c.SDL_DestroyTexture(self.texture);
        c.SDL_DestroyRenderer(self.renderer);
        c.SDL_DestroyWindow(self.window);
        c.SDL_Quit();
        allocator.free(self.buffer);
        allocator.destroy(self);
    }
    
    pub fn renderFractal(self: *FractalWindow) !void {
        const start_time = std.time.nanoTimestamp();
        
        // Split work among threads
        const rows_per_thread = HEIGHT / self.thread_count;
        var contexts = try self.allocator.alloc(RenderContext, self.thread_count);
        defer self.allocator.free(contexts);
        
        // Prepare work items
        for (0..self.thread_count) |i| {
            const start_y = i * rows_per_thread;
            const end_y = if (i == self.thread_count - 1) HEIGHT else start_y + rows_per_thread;
            
            contexts[i] = .{
                .allocator = self.allocator,
                .buffer = self.buffer,
                .width = WIDTH,
                .height = HEIGHT,
                .center_x = self.center_x,
                .center_y = self.center_y,
                .zoom = self.zoom,
                .max_iterations = self.max_iterations,
                .start_y = start_y,
                .end_y = end_y,
                .thread_id = i,
            };
        }
        
        // Dispatch work to thread pool
        var wait_group: std.Thread.WaitGroup = undefined;
        wait_group.reset();
        
        for (contexts) |*ctx| {
            wait_group.start();
            try self.thread_pool.spawn(renderThread, .{ctx});
        }
        
        // Wait for all threads to complete
        self.thread_pool.waitAndWork(&wait_group);
        
        // Update texture with computed pixels
        var pitch: c_int = 0;
        var pixels_ptr: ?*anyopaque = null;
        
        if (c.SDL_LockTexture(self.texture, null, &pixels_ptr, &pitch) == 0) {
            if (pixels_ptr) |ptr| {
                const dest = @as([*]u32, @ptrCast(@alignCast(ptr)));
                @memcpy(dest[0..self.buffer.len], self.buffer);
            }
            c.SDL_UnlockTexture(self.texture);
        }
        
        // Calculate performance metrics
        const render_time = std.time.nanoTimestamp() - start_time;
        self.total_render_time += render_time;
        
        // Update FPS calculation
        const current_time = std.time.nanoTimestamp();
        if (self.last_frame_time > 0) {
            const frame_time = current_time - self.last_frame_time;
            self.fps = 1_000_000_000.0 / @as(f64, @floatFromInt(frame_time));
        }
        self.last_frame_time = current_time;
        self.frame_count += 1;
    }
    
    fn renderThread(ctx: *RenderContext) void {
        ctx.renderSlice();
    }
    
    pub fn renderToScreen(self: *FractalWindow) void {
        // Clear screen
        _ = c.SDL_SetRenderDrawColor(self.renderer, 0, 0, 0, 255);
        _ = c.SDL_RenderClear(self.renderer);
        
        // Draw fractal texture
        _ = c.SDL_RenderCopy(self.renderer, self.texture, null, null);
        
        // Draw UI overlay
        self.renderUI();
        
        // Present frame
        c.SDL_RenderPresent(self.renderer);
    }
    
    fn renderUI(self: *FractalWindow) void {
        // Draw performance stats
        const avg_render_time = if (self.frame_count > 0) 
            @as(f64, @floatFromInt(self.total_render_time)) / @as(f64, @floatFromInt(self.frame_count)) / 1_000_000.0 
        else 0;
        
        const pixels_per_frame = WIDTH * HEIGHT;
        const mpps = pixels_per_frame / (avg_render_time * 1000.0); // Million pixels per second
        
        // Create UI text
        var text_buffer: [1024]u8 = undefined;
        const text = std.fmt.bufPrint(&text_buffer, 
            \\🌀 Zig Fractal Visualizer
            \\══════════════════════════════════════
            \\FPS:          {d:.1}
            \\Threads:      {d}
            \\Zoom:         {d:.2e}
            \\Center:       ({d:.10}, {d:.10})
            \\Iterations:   {d}
            \\Render Time:  {d:.2} ms
            \\Pixels/Frame: {d}
            \\Throughput:   {d:.1} Mpx/s
            \\Frame:        {d}
            \\
            \\CONTROLS:
            \\• Mouse Wheel: Zoom In/Out
            \\• Mouse Drag:  Pan
            \\• +/-:         Change Iterations
            \\• R:           Reset View
            \\• C:           Cycle Color Schemes
            \\• SPACE:       Auto-zoom Animation
            \\• ESC:         Exit
        , .{
            self.fps,
            self.thread_count,
            self.zoom,
            self.center_x,
            self.center_y,
            self.max_iterations,
            avg_render_time,
            pixels_per_frame,
            mpps,
            self.frame_count,
        }) catch "Error formatting text";
        
        // Draw text to screen (simplified - in practice use SDL2_ttf)
        self.drawSimpleText(text, 10, 10);
    }
    
    fn drawSimpleText(self: *FractalWindow, text: []const u8, x: i32, y: i32) void {
        // Simple ASCII text rendering for demo
        // In a real application, use SDL2_ttf
        _ = self; _ = text; _ = x; _ = y;
        // Implementation would go here
    }
    
    pub fn handleEvents(self: *FractalWindow) bool {
        var event: c.SDL_Event = undefined;
        var needs_redraw = false;
        
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDLK_q, c.SDL_QUIT => return false,
                c.SDL_KEYDOWN => {
                    const key = event.key.keysym.sym;
                    switch (key) {
                        c.SDLK_ESCAPE => return false,
                        c.SDLK_r => { // Reset view
                            self.center_x = -0.5;
                            self.center_y = 0.0;
                            self.zoom = 1.0;
                            self.max_iterations = MAX_ITERATIONS;
                            needs_redraw = true;
                        },
                        c.SDLK_EQUALS, c.SDLK_PLUS => { // Increase iterations
                            self.max_iterations = @min(self.max_iterations * 2, 8192);
                            needs_redraw = true;
                        },
                        c.SDLK_MINUS => { // Decrease iterations
                            self.max_iterations = @max(self.max_iterations / 2, 16);
                            needs_redraw = true;
                        },
                        c.SDLK_SPACE => { // Auto-zoom
                            self.zoom *= 1.05;
                            needs_redraw = true;
                        },
                        else => {},
                    }
                },
                c.SDL_MOUSEWHEEL => { // Zoom
                    const zoom_delta = if (event.wheel.y > 0) ZOOM_FACTOR else 1.0 / ZOOM_FACTOR;
                    self.zoom *= zoom_delta;
                    needs_redraw = true;
                },
                c.SDL_MOUSEBUTTONDOWN => {
                    if (event.button.button == c.SDL_BUTTON_LEFT) {
                        // Start panning
                        var mouse_x: c_int = 0;
                        var mouse_y: c_int = 0;
                        _ = c.SDL_GetMouseState(&mouse_x, &mouse_y);
                        
                        // Convert screen coords to fractal coords
                        const scale = 4.0 / (self.zoom * @as(f64, @floatFromInt(WIDTH)));
                        const fractal_x = self.center_x + 
                            (@as(f64, @floatFromInt(mouse_x)) - @as(f64, @floatFromInt(WIDTH)) / 2.0) * scale;
                        const fractal_y = self.center_y + 
                            (@as(f64, @floatFromInt(mouse_y)) - @as(f64, @floatFromInt(HEIGHT)) / 2.0) * scale;
                        
                        // Center on clicked point
                        self.center_x = fractal_x;
                        self.center_y = fractal_y;
                        needs_redraw = true;
                    }
                },
                c.SDL_MOUSEMOTION => {
                    if (event.motion.state & c.SDL_BUTTON_LMASK != 0) {
                        // Pan with mouse drag
                        const dx = @as(f64, @floatFromInt(event.motion.xrel));
                        const dy = @as(f64, @floatFromInt(event.motion.yrel));
                        
                        const scale = 4.0 / (self.zoom * @as(f64, @floatFromInt(WIDTH)));
                        self.center_x -= dx * scale * PAN_SPEED;
                        self.center_y -= dy * scale * PAN_SPEED;
                        needs_redraw = true;
                    }
                },
                else => {},
            }
        }
        
        return true;
    }
};

// Performance benchmark function
fn benchmarkFractal(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🚀 Performance Benchmark Mode\n", .{});
    std.debug.print("══════════════════════════════════════\n\n", .{});
    
    const test_cases = [_]struct {
        name: []const u8,
        width: usize,
        height: usize,
        iterations: u32,
    }{
        .{ .name = "Small (640x480)", .width = 640, .height = 480, .iterations = 256 },
        .{ .name = "HD (1280x720)", .width = 1280, .height = 720, .iterations = 512 },
        .{ .name = "Full HD (1920x1080)", .width = 1920, .height = 1080, .iterations = 512 },
        .{ .name = "4K (3840x2160)", .width = 3840, .height = 2160, .iterations = 1024 },
    };
    
    for (test_cases) |test_case| {
        std.debug.print("Testing: {s}\n", .{test_case.name});
        std.debug.print("Pixels: {d}\n", .{test_case.width * test_case.height});
        std.debug.print("Iterations: {d}\n", .{test_case.iterations});
        
        const buffer = try allocator.alloc(u32, test_case.width * test_case.height);
        defer allocator.free(buffer);
        
        const start_time = std.time.nanoTimestamp();
        
        // Single-threaded render for benchmarking
        const scale = 4.0 / (@as(f64, @floatFromInt(test_case.width)));
        const center_x: f64 = -0.5;
        const center_y: f64 = 0.0;
        
        for (0..test_case.height) |y| {
            const imag = center_y + (@as(f64, @floatFromInt(y)) - @as(f64, @floatFromInt(test_case.height)) / 2.0) * scale;
            
            for (0..test_case.width) |x| {
                const real = center_x + (@as(f64, @floatFromInt(x)) - @as(f64, @floatFromInt(test_case.width)) / 2.0) * scale;
                
                var z = Complex{ .real = 0, .imag = 0 };
                const complex_c = Complex{ .real = real, .imag = imag };
                var iteration: u32 = 0;
                
                while (iteration < test_case.iterations and z.magnitudeSquared() < 4.0) {
                    z = z.mul(z).add(complex_c);
                    iteration += 1;
                }
            }
        }
        
        const end_time = std.time.nanoTimestamp();
        const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        const pixels_per_sec = (@as(f64, @floatFromInt(test_case.width * test_case.height)) / elapsed_ms) * 1000.0;
        
        std.debug.print("Time: {d:.2} ms\n", .{elapsed_ms});
        std.debug.print("Speed: {d:.2} million pixels/sec\n", .{pixels_per_sec / 1_000_000.0});
        std.debug.print("Operations: {d:.2} billion\n", .{@as(f64, @floatFromInt(test_case.width * test_case.height * test_case.iterations)) / 1_000_000_000.0});
        std.debug.print("Operations/sec: {d:.2} billion\n\n", .{(pixels_per_sec * @as(f64, @floatFromInt(test_case.iterations))) / 1_000_000_000.0});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .enable_memory_limit = true,
        .stack_trace_frames = 0,
    }){};
    defer _ = gpa.deinit();
    
    const allocator = gpa.allocator();
    
    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    var benchmark_mode = false;
    var headless_mode = false;
    
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--benchmark") or std.mem.eql(u8, arg, "-b")) {
            benchmark_mode = true;
        } else if (std.mem.eql(u8, arg, "--headless") or std.mem.eql(u8, arg, "-h")) {
            headless_mode = true;
        }
    }
    
    if (benchmark_mode) {
        return benchmarkFractal(allocator);
    }
    
    if (headless_mode) {
        std.debug.print("Running in headless mode...\n", .{});
        // Run fractal computation without graphics
        const buffer = try allocator.alloc(u32, WIDTH * HEIGHT);
        defer allocator.free(buffer);
        
        const start_time = std.time.nanoTimestamp();
        
        // Perform computation
        const scale = 4.0 / (@as(f64, @floatFromInt(WIDTH)));
        const center_x: f64 = -0.5;
        const center_y: f64 = 0.0;
        
        for (0..HEIGHT) |y| {
            const imag = center_y + (@as(f64, @floatFromInt(y)) - @as(f64, @floatFromInt(HEIGHT)) / 2.0) * scale;
            
            for (0..WIDTH) |x| {
                const real = center_x + (@as(f64, @floatFromInt(x)) - @as(f64, @floatFromInt(WIDTH)) / 2.0) * scale;
                
                var z = Complex{ .real = 0, .imag = 0 };
                const complex_c = Complex{ .real = real, .imag = imag };
                var iteration: u32 = 0;
                
                while (iteration < MAX_ITERATIONS and z.magnitudeSquared() < 4.0) {
                    z = z.mul(z).add(complex_c);
                    iteration += 1;
                }
            }
        }
        
        const end_time = std.time.nanoTimestamp();
        const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        const pixels_per_sec = (@as(f64, @floatFromInt(WIDTH * HEIGHT)) / elapsed_ms) * 1000.0;
        
        std.debug.print("Headless computation complete!\n", .{});
        std.debug.print("Time: {d:.2} ms\n", .{elapsed_ms});
        std.debug.print("Speed: {d:.2} million pixels/sec\n", .{pixels_per_sec / 1_000_000.0});
        
        return;
    }
    
    // Create SDL2 window
    var fractal_window = try FractalWindow.init(allocator);
    defer fractal_window.deinit();
    
    // Initial render
    try fractal_window.renderFractal();
    fractal_window.renderToScreen();
    
    std.debug.print("\n🌀 Zig Fractal Visualizer\n", .{});
    std.debug.print("══════════════════════════════════════\n\n", .{});
    std.debug.print("Real-time Mandelbrot fractal renderer\n", .{});
    std.debug.print("Resolution: {}x{} ({} million pixels)\n", .{ WIDTH, HEIGHT, WIDTH * HEIGHT / 1_000_000 });
    std.debug.print("Threads: {}\n", .{fractal_window.thread_count});
    std.debug.print("\nControls:\n", .{});
    std.debug.print("• Mouse Wheel: Zoom In/Out\n", .{});
    std.debug.print("• Mouse Drag: Pan around\n", .{});
    std.debug.print("• +/-: Change iteration count\n", .{});
    std.debug.print("• R: Reset view\n", .{});
    std.debug.print("• SPACE: Auto-zoom animation\n", .{});
    std.debug.print("• ESC: Exit\n\n", .{});
    
    // Main loop
    var running = true;
    var last_update_time: u64 = 0;
    const target_frame_time = 1_000_000_000 / 60; // 60 FPS target
    
    while (running) {
        const frame_start = std.time.nanoTimestamp();
        
        // Handle events
        running = fractal_window.handleEvents();
        
        // Check if we need to redraw
        const current_time = std.time.nanoTimestamp();
        if (current_time - last_update_time > 16_000_000) { // ~60 FPS
            try fractal_window.renderFractal();
            fractal_window.renderToScreen();
            last_update_time = current_time;
        }
        
        // Frame rate limiting
        const frame_time = std.time.nanoTimestamp() - frame_start;
        if (frame_time < target_frame_time) {
            const sleep_time = target_frame_time - frame_time;
            std.time.sleep(sleep_time);
        }
    }
    
    // Display final statistics
    const avg_render_time = if (fractal_window.frame_count > 0)
        @as(f64, @floatFromInt(fractal_window.total_render_time)) / 
        @as(f64, @floatFromInt(fractal_window.frame_count)) / 1_000_000.0 
    else 0;
    
    const pixels_per_sec = (@as(f64, @floatFromInt(WIDTH * HEIGHT)) / avg_render_time) * 1000.0;
    const total_operations = @as(f64, @floatFromInt(WIDTH * HEIGHT * fractal_window.max_iterations * fractal_window.frame_count));
    
    std.debug.print("\n\n✨ Performance Summary\n", .{});
    std.debug.print("══════════════════════════════════════\n", .{});
    std.debug.print("Frames rendered:       {}\n", .{fractal_window.frame_count});
    std.debug.print("Average render time:   {d:.2} ms\n", .{avg_render_time});
    std.debug.print("Average FPS:           {d:.1}\n", .{fractal_window.fps});
    std.debug.print("Pixels per frame:      {} million\n", .{WIDTH * HEIGHT / 1_000_000});
    std.debug.print("Throughput:            {d:.1} million pixels/sec\n", .{pixels_per_sec / 1_000_000.0});
    std.debug.print("Thread count:          {}\n", .{fractal_window.thread_count});
    std.debug.print("Total operations:      {d:.2} billion\n", .{total_operations / 1_000_000_000.0});
    std.debug.print("\n💡 This demonstrates:\n", .{});
    std.debug.print("• Real-time fractal computation\n", .{});
    std.debug.print("• Multi-threaded pixel processing\n", .{});
    std.debug.print("• GPU-like performance on CPU\n", .{});
    std.debug.print("• Zero-cost abstractions\n", .{});
    std.debug.print("• Memory-safe parallel processing\n", .{});
}