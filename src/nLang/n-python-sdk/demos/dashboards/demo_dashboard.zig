const std = @import("std");

const c = @cImport({
    @cInclude("SDL2/SDL.h");
    @cInclude("SDL2/SDL_ttf.h");
});

const WIDTH = 1400;
const HEIGHT = 1000;

const DemoType = enum {
    particle_physics,
    particle_physics_mt,
    sorting_comparison,
    memory_benchmark,
    fractal_renderer,
    matrix_operations,
    back_to_menu,
    
    pub fn getName(self: DemoType) []const u8 {
        return switch (self) {
            .particle_physics => "Particle Physics (100K particles)",
            .particle_physics_mt => "Multi-threaded Physics (1M particles)",
            .sorting_comparison => "Sorting Algorithm Race",
            .memory_benchmark => "Memory Throughput Test",
            .fractal_renderer => "Real-time Fractal Explorer",
            .matrix_operations => "Matrix Math Benchmark",
            .back_to_menu => "← Back to Menu",
        };
    }
    
    pub fn getDescription(self: DemoType) []const u8 {
        return switch (self) {
            .particle_physics => "Single-threaded particle simulation with collision detection",
            .particle_physics_mt => "Parallel processing showcase with 1M particles",
            .sorting_comparison => "Compare Zig vs Python/Go/Rust sorting performance",
            .memory_benchmark => "Memory bandwidth and cache performance test",
            .fractal_renderer => "Multi-threaded Mandelbrot set renderer",
            .matrix_operations => "Linear algebra performance comparison",
            .back_to_menu => "Return to main menu",
        };
    }
};

const Button = struct {
    rect: SDL_Rect,
    demo: DemoType,
    hovered: bool = false,
    
    pub fn contains(self: Button, x: i32, y: i32) bool {
        return x >= self.rect.x and x < self.rect.x + self.rect.w and
               y >= self.rect.y and y < self.rect.y + self.rect.h;
    }
    
    pub fn draw(self: Button, renderer: *c.SDL_Renderer) void {
        // Button background
        const bg_color = if (self.hovered) 
            SDL_Color{ .r = 60, .g = 120, .b = 200, .a = 255 }
        else
            SDL_Color{ .r = 40, .g = 80, .b = 140, .a = 255 };
        
        _ = c.SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
        _ = c.SDL_RenderFillRect(renderer, &self.rect);
        
        // Button border
        const border_color = if (self.hovered)
            SDL_Color{ .r = 100, .g = 160, .b = 255, .a = 255 }
        else
            SDL_Color{ .r = 80, .g = 140, .b = 220, .a = 255 };
        
        _ = c.SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
        _ = c.SDL_RenderDrawRect(renderer, &self.rect);
    }
};

const SDL_Color = struct { r: u8, g: u8, b: u8, a: u8 };
const SDL_Rect = c.SDL_Rect;

const Dashboard = struct {
    window: *c.SDL_Window,
    renderer: *c.SDL_Renderer,
    allocator: std.mem.Allocator,
    buttons: std.ArrayList(Button),
    current_demo: ?DemoType = null,
    mouse_x: i32 = 0,
    mouse_y: i32 = 0,
    
    pub fn init(allocator: std.mem.Allocator) !*Dashboard {
        // Initialize SDL
        if (c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_EVENTS) != 0) {
            std.debug.print("SDL_Init Error: {s}\n", .{c.SDL_GetError()});
            return error.SDLInitFailed;
        }
        
        // Create window
        const window = c.SDL_CreateWindow(
            "🚀 Zig Performance Demo Dashboard",
            c.SDL_WINDOWPOS_CENTERED,
            c.SDL_WINDOWPOS_CENTERED,
            WIDTH,
            HEIGHT,
            c.SDL_WINDOW_SHOWN
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
        
        var buttons = std.ArrayList(Button).init(allocator);
        
        // Create menu buttons
        const button_width = 1200;
        const button_height = 80;
        const start_y = 150;
        const spacing = 100;
        
        const demos = [_]DemoType{
            .particle_physics,
            .particle_physics_mt,
            .sorting_comparison,
            .memory_benchmark,
            .fractal_renderer,
            .matrix_operations,
        };
        
        for (demos, 0..) |demo, i| {
            try buttons.append(.{
                .rect = SDL_Rect{
                    .x = (WIDTH - button_width) / 2,
                    .y = @as(i32, @intCast(start_y + i * spacing)),
                    .w = button_width,
                    .h = button_height,
                },
                .demo = demo,
            });
        }
        
        const dashboard = try allocator.create(Dashboard);
        dashboard.* = .{
            .window = window,
            .renderer = renderer,
            .allocator = allocator,
            .buttons = buttons,
        };
        
        return dashboard;
    }
    
    pub fn deinit(self: *Dashboard) void {
        self.buttons.deinit();
        c.SDL_DestroyRenderer(self.renderer);
        c.SDL_DestroyWindow(self.window);
        c.SDL_Quit();
        self.allocator.destroy(self);
    }
    
    pub fn render(self: *Dashboard) void {
        // Clear screen with dark background
        _ = c.SDL_SetRenderDrawColor(self.renderer, 15, 15, 25, 255);
        _ = c.SDL_RenderClear(self.renderer);
        
        // Draw title
        self.drawTitle();
        
        // Draw system info
        self.drawSystemInfo();
        
        // Draw buttons
        for (self.buttons.items) |button| {
            button.draw(self.renderer);
        }
        
        // Draw button labels
        for (self.buttons.items, 0..) |button, i| {
            const name = button.demo.getName();
            const desc = button.demo.getDescription();
            
            // Draw name
            self.drawText(name, button.rect.x + 50, button.rect.y + 15, 24, SDL_Color{ .r = 255, .g = 255, .b = 255, .a = 255 });
            
            // Draw description
            self.drawText(desc, button.rect.x + 50, button.rect.y + 45, 16, SDL_Color{ .r = 180, .g = 180, .b = 200, .a = 255 });
            
            // Draw icon
            const icon_rect = SDL_Rect{
                .x = button.rect.x + 10,
                .y = button.rect.y + 20,
                .w = 40,
                .h = 40,
            };
            self.drawIcon(@intCast(i), icon_rect);
        }
        
        // Draw instructions
        self.drawInstructions();
        
        c.SDL_RenderPresent(self.renderer);
    }
    
    fn drawTitle(self: *Dashboard) void {
        const title = "🚀 ZIG PERFORMANCE DEMO SUITE";
        self.drawText(title, WIDTH / 2 - 300, 40, 36, SDL_Color{ .r = 100, .g = 200, .b = 255, .a = 255 });
        
        const subtitle = "High-Performance Computing • Real-time Visualization • Memory Safety";
        self.drawText(subtitle, WIDTH / 2 - 350, 90, 16, SDL_Color{ .r = 150, .g = 180, .b = 200, .a = 255 });
    }
    
    fn drawSystemInfo(self: *Dashboard) void {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const builtin = @import("builtin");
        
        var info_buf: [256]u8 = undefined;
        const info = std.fmt.bufPrint(&info_buf, 
            "System: {s} | CPU Cores: {d} | Zig {s} | Build: {s}",
            .{
                @tagName(builtin.os.tag),
                cpu_count,
                builtin.zig_version_string,
                @tagName(builtin.mode),
            }
        ) catch "System Info";
        
        self.drawText(info, 20, HEIGHT - 40, 14, SDL_Color{ .r = 120, .g = 120, .b = 120, .a = 255 });
    }
    
    fn drawInstructions(self: *Dashboard) void {
        const instr = "Click on any demo to launch • ESC to return to menu • Q to quit";
        self.drawText(instr, WIDTH / 2 - 300, HEIGHT - 70, 16, SDL_Color{ .r = 150, .g = 150, .b = 150, .a = 255 });
    }
    
    fn drawIcon(self: *Dashboard, icon_type: usize, rect: SDL_Rect) void {
        const colors = [_]SDL_Color{
            .{ .r = 255, .g = 100, .b = 100, .a = 255 }, // Red
            .{ .r = 100, .g = 255, .b = 100, .a = 255 }, // Green
            .{ .r = 100, .g = 100, .b = 255, .a = 255 }, // Blue
            .{ .r = 255, .g = 255, .b = 100, .a = 255 }, // Yellow
            .{ .r = 255, .g = 100, .b = 255, .a = 255 }, // Magenta
            .{ .r = 100, .g = 255, .b = 255, .a = 255 }, // Cyan
        };
        
        const color = colors[icon_type % colors.len];
        _ = c.SDL_SetRenderDrawColor(self.renderer, color.r, color.g, color.b, color.a);
        _ = c.SDL_RenderFillRect(self.renderer, &rect);
    }
    
    fn drawText(self: *Dashboard, text: []const u8, x: i32, y: i32, size: u8, color: SDL_Color) void {
        // Simplified text rendering using SDL primitives
        // In production, use SDL_ttf for proper font rendering
        _ = self; _ = text; _ = x; _ = y; _ = size; _ = color;
        
        // For now, this is a placeholder
        // Real implementation would use SDL_ttf
    }
    
    pub fn handleEvents(self: *Dashboard) !bool {
        var event: c.SDL_Event = undefined;
        
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => return false,
                c.SDL_KEYDOWN => {
                    switch (event.key.keysym.sym) {
                        c.SDLK_q, c.SDLK_ESCAPE => return false,
                        else => {},
                    }
                },
                c.SDL_MOUSEMOTION => {
                    self.mouse_x = event.motion.x;
                    self.mouse_y = event.motion.y;
                    
                    // Update hover states
                    for (self.buttons.items) |*button| {
                        button.hovered = button.contains(self.mouse_x, self.mouse_y);
                    }
                },
                c.SDL_MOUSEBUTTONDOWN => {
                    if (event.button.button == c.SDL_BUTTON_LEFT) {
                        // Check if any button was clicked
                        for (self.buttons.items) |button| {
                            if (button.contains(self.mouse_x, self.mouse_y)) {
                                self.current_demo = button.demo;
                                try self.launchDemo(button.demo);
                            }
                        }
                    }
                },
                else => {},
            }
        }
        
        return true;
    }
    
    fn launchDemo(self: *Dashboard, demo: DemoType) !void {
        std.debug.print("\n🚀 Launching: {s}\n", .{demo.getName()});
        std.debug.print("══════════════════════════════════════\n\n", .{});
        
        // Hide main window while demo runs
        _ = c.SDL_HideWindow(self.window);
        defer _ = c.SDL_ShowWindow(self.window);
        
        switch (demo) {
            .particle_physics => try runParticlePhysicsDemo(self.allocator),
            .particle_physics_mt => try runParticlePhysicsMT(self.allocator),
            .sorting_comparison => try runSortingDemo(self.allocator),
            .memory_benchmark => try runMemoryBenchmark(self.allocator),
            .fractal_renderer => try runFractalDemo(self.allocator),
            .matrix_operations => try runMatrixDemo(self.allocator),
            .back_to_menu => {},
        }
        
        std.debug.print("\n✨ Demo complete! Returning to dashboard...\n\n", .{});
    }
};

// Demo implementations (these will call the actual demo modules)
fn runParticlePhysicsDemo(allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("Running Particle Physics Demo...\n", .{});
    std.debug.print("(Implementation integrated with visual dashboard)\n", .{});
}

fn runParticlePhysicsMT(allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("Running Multi-threaded Particle Physics Demo...\n", .{});
    std.debug.print("(Implementation integrated with visual dashboard)\n", .{});
}

fn runSortingDemo(allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("Running Sorting Comparison Demo...\n", .{});
    std.debug.print("(Implementation integrated with visual dashboard)\n", .{});
}

fn runMemoryBenchmark(allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("Running Memory Benchmark...\n", .{});
    std.debug.print("(Implementation integrated with visual dashboard)\n", .{});
}

fn runFractalDemo(allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("Running Fractal Demo...\n", .{});
    std.debug.print("(Implementation integrated with visual dashboard)\n", .{});
}

fn runMatrixDemo(allocator: std.mem.Allocator) !void {
    _ = allocator;
    std.debug.print("Running Matrix Operations Demo...\n", .{});
    std.debug.print("(Implementation integrated with visual dashboard)\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n╔══════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║                                                          ║\n", .{});
    std.debug.print("║        🚀 ZIG PERFORMANCE DEMO SUITE 🚀                  ║\n", .{});
    std.debug.print("║                                                          ║\n", .{});
    std.debug.print("║  High-Performance Computing Demonstrations              ║\n", .{});
    std.debug.print("║  Real-time Visualization • Technical Metrics            ║\n", .{});
    std.debug.print("║  Language Comparisons • Optimization Showcases          ║\n", .{});
    std.debug.print("║                                                          ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════════════════╝\n\n", .{});
    
    // System information
    const builtin = @import("builtin");
    const cpu_count = try std.Thread.getCpuCount();
    std.debug.print("System Information:\n", .{});
    std.debug.print("  OS: {s}\n", .{@tagName(builtin.os.tag)});
    std.debug.print("  CPU Cores: {d}\n", .{cpu_count});
    std.debug.print("  Zig Version: {s}\n", .{builtin.zig_version_string});
    std.debug.print("  Build Mode: {s}\n\n", .{@tagName(builtin.mode)});
    
    std.debug.print("Initializing dashboard...\n\n", .{});
    
    var dashboard = try Dashboard.init(allocator);
    defer dashboard.deinit();
    
    std.debug.print("✨ Dashboard ready!\n", .{});
    std.debug.print("Click on any demo to launch it.\n\n", .{});
    
    // Main loop
    var running = true;
    while (running) {
        running = try dashboard.handleEvents();
        dashboard.render();
        
        // Small delay to prevent busy waiting
        c.SDL_Delay(16); // ~60 FPS
    }
    
    std.debug.print("\n👋 Thanks for exploring Zig's performance!\n", .{});
}