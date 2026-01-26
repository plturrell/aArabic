const std = @import("std");
const builtin = @import("builtin");
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});

const WIDTH = 1200;
const HEIGHT = 800;

const DemoInfo = struct {
    name: []const u8,
    description: []const u8,
    file: []const u8,
    available: bool = true,
};

const demos = [_]DemoInfo{
    .{
        .name = "1. Visual Particle Physics",
        .description = "50K particles • Real-time metrics • Language comparisons",
        .file = "visual_particle_demo_complete",
    },
    .{
        .name = "2. Console Particle Physics",
        .description = "100K particles • Terminal output • Performance stats",
        .file = "particle_physics_demo",
    },
    .{
        .name = "3. Performance Benchmarks",
        .description = "CPU tests • Memory throughput • Algorithm comparisons",
        .file = "benchmark_suite",
    },
};

const Button = struct {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    demo_index: usize,
    hovered: bool = false,
    
    pub fn contains(self: Button, mx: i32, my: i32) bool {
        return mx >= self.x and mx < self.x + self.width and
               my >= self.y and my < self.y + self.height;
    }
    
    pub fn draw(self: Button, renderer: *c.SDL_Renderer) void {
        const rect = c.SDL_Rect{
            .x = self.x,
            .y = self.y,
            .w = self.width,
            .h = self.height,
        };
        
        // Background
        if (self.hovered) {
            _ = c.SDL_SetRenderDrawColor(renderer, 60, 120, 200, 255);
        } else {
            _ = c.SDL_SetRenderDrawColor(renderer, 40, 80, 140, 255);
        }
        _ = c.SDL_RenderFillRect(renderer, &rect);
        
        // Border
        if (self.hovered) {
            _ = c.SDL_SetRenderDrawColor(renderer, 100, 160, 255, 255);
        } else {
            _ = c.SDL_SetRenderDrawColor(renderer, 60, 100, 180, 255);
        }
        _ = c.SDL_RenderDrawRect(renderer, &rect);
        
        // Draw demo info
        const demo = demos[self.demo_index];
        drawText(renderer, demo.name, self.x + 20, self.y + 15, 2, .{ .r = 255, .g = 255, .b = 255, .a = 255 });
        drawText(renderer, demo.description, self.x + 20, self.y + 45, 1, .{ .r = 200, .g = 200, .b = 220, .a = 255 });
        
        // Draw indicator
        const indicator_color = if (demos[self.demo_index].available)
            SDL_Color{ .r = 100, .g = 255, .b = 100, .a = 255 }
        else
            SDL_Color{ .r = 255, .g = 100, .b = 100, .a = 255 };
        
        const indicator = c.SDL_Rect{
            .x = self.x + 10,
            .y = self.y + 25,
            .w = 8,
            .h = 40,
        };
        _ = c.SDL_SetRenderDrawColor(renderer, indicator_color.r, indicator_color.g, indicator_color.b, indicator_color.a);
        _ = c.SDL_RenderFillRect(renderer, &indicator);
    }
};

const SDL_Color = struct { r: u8, g: u8, b: u8, a: u8 };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Initialize SDL
    if (c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_EVENTS) != 0) {
        std.debug.print("SDL_Init failed: {s}\n", .{c.SDL_GetError()});
        return error.SDLInitFailed;
    }
    defer c.SDL_Quit();
    
    // Create window
    const window = c.SDL_CreateWindow(
        "🚀 Zig Performance Demo Dashboard",
        c.SDL_WINDOWPOS_CENTERED,
        c.SDL_WINDOWPOS_CENTERED,
        WIDTH,
        HEIGHT,
        c.SDL_WINDOW_SHOWN
    ) orelse {
        std.debug.print("SDL_CreateWindow failed: {s}\n", .{c.SDL_GetError()});
        return error.WindowCreationFailed;
    };
    defer c.SDL_DestroyWindow(window);
    
    // Create renderer
    const renderer = c.SDL_CreateRenderer(
        window,
        -1,
        c.SDL_RENDERER_ACCELERATED | c.SDL_RENDERER_PRESENTVSYNC
    ) orelse {
        std.debug.print("SDL_CreateRenderer failed: {s}\n", .{c.SDL_GetError()});
        return error.RendererCreationFailed;
    };
    defer c.SDL_DestroyRenderer(renderer);
    
    // Create buttons
    var buttons: [demos.len]Button = undefined;
    const button_height = 80;
    const button_spacing = 20;
    const start_y = 200;
    
    for (&buttons, 0..) |*btn, i| {
        btn.* = .{
            .x = 50,
            .y = start_y + @as(i32, @intCast(i)) * (button_height + button_spacing),
            .width = WIDTH - 100,
            .height = button_height,
            .demo_index = i,
        };
    }
    
    var mouse_x: i32 = 0;
    var mouse_y: i32 = 0;
    var running = true;
    
    std.debug.print("\n🚀 Zig Performance Demo Dashboard\n", .{});
    std.debug.print("══════════════════════════════════════\n", .{});
    std.debug.print("Dashboard window opened!\n", .{});
    std.debug.print("Click on any demo to launch it.\n\n", .{});
    
    while (running) {
        // Handle events
        var event: c.SDL_Event = undefined;
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => running = false,
                c.SDL_KEYDOWN => {
                    if (event.key.keysym.sym == c.SDLK_ESCAPE or event.key.keysym.sym == c.SDLK_q) {
                        running = false;
                    }
                },
                c.SDL_MOUSEMOTION => {
                    mouse_x = event.motion.x;
                    mouse_y = event.motion.y;
                    
                    // Update hover states
                    for (&buttons) |*btn| {
                        btn.hovered = btn.contains(mouse_x, mouse_y);
                    }
                },
                c.SDL_MOUSEBUTTONDOWN => {
                    if (event.button.button == c.SDL_BUTTON_LEFT) {
                        // Check which button was clicked
                        for (buttons) |btn| {
                            if (btn.contains(mouse_x, mouse_y)) {
                                try launchDemo(allocator, btn.demo_index, window);
                            }
                        }
                    }
                },
                else => {},
            }
        }
        
        // Render
        _ = c.SDL_SetRenderDrawColor(renderer, 15, 15, 25, 255);
        _ = c.SDL_RenderClear(renderer);
        
        // Draw title
        drawText(renderer, "ZIG PERFORMANCE DEMO SUITE", WIDTH / 2 - 300, 50, 3, .{ .r = 100, .g = 200, .b = 255, .a = 255 });
        drawText(renderer, "Select a demo to launch", WIDTH / 2 - 180, 110, 1, .{ .r = 150, .g = 180, .b = 200, .a = 255 });
        
        // Draw buttons
        for (buttons) |btn| {
            btn.draw(renderer);
        }
        
        // Draw instructions
        drawText(renderer, "Click demo to launch • ESC/Q to quit • Demos return here when closed", 
            20, HEIGHT - 40, 1, .{ .r = 120, .g = 120, .b = 140, .a = 255 });
        
        // System info
        const cpu_count = std.Thread.getCpuCount() catch 1;
        var info_buf: [128]u8 = undefined;
        const info = std.fmt.bufPrint(&info_buf, "System: {s} | {d} CPU cores | Zig {s}",
            .{ @tagName(builtin.os.tag), cpu_count, builtin.zig_version_string }) catch "System Info";
        drawText(renderer, info, 20, HEIGHT - 70, 1, .{ .r = 100, .g = 100, .b = 120, .a = 255 });
        
        c.SDL_RenderPresent(renderer);
        c.SDL_Delay(16); // ~60 FPS
    }
    
    std.debug.print("\n👋 Dashboard closed. Goodbye!\n\n", .{});
}

fn launchDemo(allocator: std.mem.Allocator, demo_index: usize, main_window: *c.SDL_Window) !void {
    const demo = demos[demo_index];
    
    std.debug.print("\n🚀 Launching: {s}\n", .{demo.name});
    std.debug.print("══════════════════════════════════════\n\n", .{});
    
    // Hide main dashboard
    _ = c.SDL_HideWindow(main_window);
    defer _ = c.SDL_ShowWindow(main_window);
    
    // Check if it's a binary or build command
    if (std.mem.startsWith(u8, demo.file, "zig build")) {
        // It's a build command - execute it
        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "sh", "-c", demo.file },
            .cwd = ".",
        }) catch |err| {
            std.debug.print("❌ Failed to run demo: {}\n", .{err});
            std.debug.print("Returning to dashboard in 3 seconds...\n", .{});
            std.Thread.sleep(3_000_000_000);
            return;
        };
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        
        std.debug.print("{s}", .{result.stdout});
        if (result.stderr.len > 0) {
            std.debug.print("{s}", .{result.stderr});
        }
    } else {
        // Check if binary exists
        const file = std.fs.cwd().openFile(demo.file, .{}) catch |err| {
            std.debug.print("❌ Demo not found: {s}\n", .{demo.file});
            std.debug.print("Error: {}\n", .{err});
            std.debug.print("\n💡 Build it first with: zig build-exe {s}.zig -lc -lSDL2 -I/opt/homebrew/include/SDL2 -L/opt/homebrew/lib -OReleaseFast\n", .{demo.file});
            std.debug.print("Returning to dashboard in 5 seconds...\n", .{});
            std.Thread.sleep(5_000_000_000);
            return;
        };
        file.close();
        
        // Execute the binary
        var child = std.process.Child.init(&[_][]const u8{demo.file}, allocator);
        child.cwd = ".";
        
        const term = child.spawnAndWait() catch |err| {
            std.debug.print("❌ Failed to run demo: {}\n", .{err});
            std.debug.print("Returning to dashboard in 3 seconds...\n", .{});
            std.Thread.sleep(3_000_000_000);
            return;
        };
        
        _ = term;
    }
    
    std.debug.print("\n✨ Demo finished! Returning to dashboard in 2 seconds...\n", .{});
    std.Thread.sleep(2_000_000_000);
}

fn drawText(renderer: *c.SDL_Renderer, text: []const u8, x: i32, y: i32, scale: u8, color: SDL_Color) void {
    _ = c.SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    
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
            const offset = char - 'A';
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
            if (offset < patterns.len) {
                break :blk patterns[offset];
            }
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
        ';' => .{ 0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00110, 0b00100 },
        '-' => .{ 0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000 },
        '|' => .{ 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 },
        '+' => .{ 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000, 0b00000 },
        else => .{ 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000 },
    };
}