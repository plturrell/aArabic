// Mojo Package Manager - Main Entry Point
// Day 96: Complete CLI application

const std = @import("std");
const cli = @import("cli.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Get command line arguments
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    
    // Skip program name
    _ = args.next();
    
    // Collect remaining args
    var arg_list = std.ArrayList([]const u8){};
    defer arg_list.deinit(allocator);
    
    while (args.next()) |arg| {
        try arg_list.append(allocator, arg);
    }
    
    // Parse command
    var parser = cli.CliParser.init(allocator);
    const command = parser.parse(arg_list.items) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        std.debug.print("Run 'mojo help' for usage information\n", .{});
        return;
    };
    
    // Execute command
    var runner = cli.CliRunner.init(allocator);
    defer runner.deinit();
    
    runner.run(command) catch |err| {
        std.debug.print("Error executing command: {}\n", .{err});
        return;
    };
}
