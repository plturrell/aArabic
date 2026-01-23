const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load .env
    _ = loadDotEnv(allocator, "../../.env") catch |err| {
        std.debug.print("⚠️  .env not loaded: {}\n", .{err});
    };

    // Start HANA bridge
    const bridge_pid = try spawn(allocator, &[_][]const u8{
        "node",
        "src/hana_bridge/server.js",
    }, ".");
    std.debug.print("📡 HANA bridge PID {d}\n", .{bridge_pid});

    // Start admin server
    const admin_pid = try spawn(allocator, &[_][]const u8{
        "./bin/agents_admin_server",
    }, ".");
    std.debug.print("🛠  Admin server PID {d}\n", .{admin_pid});

    // Start production server
    const prod_pid = try spawn(allocator, &[_][]const u8{
        "./bin/production_server",
    }, ".");
    std.debug.print("🌐 Production server PID {d}\n", .{prod_pid});

    std.debug.print("✅ Runner started all services. Press Ctrl+C to exit.\n", .{});

    while (true) {
        std.Thread.sleep(10 * std.time.ns_per_s);
    }
}

fn spawn(allocator: std.mem.Allocator, args: []const []const u8, workdir: []const u8) !std.process.Child.Id {
    var proc = std.process.Child.init(args, allocator);
    proc.cwd = workdir;
    proc.stdin_behavior = .Inherit;
    proc.stdout_behavior = .Inherit;
    proc.stderr_behavior = .Inherit;
    try proc.spawn();
    return proc.id;
}

fn loadDotEnv(allocator: std.mem.Allocator, path: []const u8) !void {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();
    const data = try file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');
    while (it.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.indexOfScalar(u8, line, '=')) |eq| {
            _ = eq; // env loading skipped in runner for now
        }
    }
}
