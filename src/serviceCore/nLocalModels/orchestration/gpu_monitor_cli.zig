//! GPU Monitor CLI Tool
//! Real-time GPU monitoring with logging and health status
//!
//! Replaces bash script with native Zig implementation for better performance
//! and integration with the orchestration system

const std = @import("std");
const GPUMonitor = @import("gpu_monitor.zig").GPUMonitor;
const GPUState = @import("gpu_monitor.zig").GPUState;

const ANSI_RED = "\x1b[31m";
const ANSI_GREEN = "\x1b[32m";
const ANSI_YELLOW = "\x1b[33m";
const ANSI_RESET = "\x1b[0m";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    // Parse arguments
    var interval_ms: u32 = 5000; // Default: 5 seconds
    var log_file: []const u8 = "logs/gpu_monitor.log";
    
    if (args.len > 1) {
        interval_ms = try std.fmt.parseInt(u32, args[1], 10) * 1000;
    }
    if (args.len > 2) {
        log_file = args[2];
    }
    
    const stdout = std.io.getStdOut().writer();
    
    try stdout.print("=== GPU Selection Monitoring ===\n", .{});
    try stdout.print("Interval: {d} seconds\n", .{interval_ms / 1000});
    try stdout.print("Log File: {s}\n", .{log_file});
    try stdout.print("Press Ctrl+C to stop\n\n", .{});
    
    // Create log directory
    const log_dir = std.fs.path.dirname(log_file) orelse ".";
    try std.fs.cwd().makePath(log_dir);
    
    // Create GPU monitor
    const monitor = try GPUMonitor.init(allocator, interval_ms);
    defer monitor.deinit();
    
    // Open log file
    const log = try std.fs.cwd().createFile(log_file, .{ .truncate = false });
    defer log.close();
    
    // Check if file is empty, write header
    const stat = try log.stat();
    if (stat.size == 0) {
        try log.writer().print("timestamp,gpu_id,name,utilization_%,memory_used_mb,memory_free_mb,memory_total_mb,memory_percent,temperature_c,power_w,power_limit_w,health_status\n", .{});
    }
    
    // Monitoring loop
    var iteration: usize = 0;
    while (true) : (iteration += 1) {
        // Refresh GPU states
        monitor.refresh() catch |err| {
            try stdout.print("{s}ERROR: Failed to query GPUs: {any}{s}\n", 
                .{ANSI_RED, err, ANSI_RESET});
            std.time.sleep(interval_ms * std.time.ns_per_ms);
            continue;
        };
        
        // Clear screen and show header
        if (iteration > 0) {
            try stdout.print("\x1b[2J\x1b[H", .{}); // Clear screen
        }
        
        try stdout.print("=== GPU Selection Monitor ===\n", .{});
        try stdout.print("Time: {d}\n", .{std.time.timestamp()});
        try stdout.print("Log: {s}\n\n", .{log_file});
        
        // Get timestamp
        const timestamp = std.time.timestamp();
        
        // Display and log each GPU
        for (monitor.gpu_states.items) |state| {
            // Determine health status
            const health_status = if (state.isHealthy()) "HEALTHY" else blk: {
                if (state.temperature_c > 85.0) break :blk "HOT";
                if (state.utilization_percent > 95.0) break :blk "OVERLOADED";
                break :blk "HIGH_POWER";
            };
            
            // Choose color
            const color = if (std.mem.eql(u8, health_status, "HEALTHY"))
                ANSI_GREEN
            else
                ANSI_RED;
            
            // Calculate memory percentage
            const mem_percent = (@as(f32, @floatFromInt(state.used_memory_mb)) / 
                                @as(f32, @floatFromInt(state.total_memory_mb))) * 100.0;
            
            // Display to console
            try stdout.print("{s}GPU {d}: {s}{s} | Util: {d:3}% | Mem: {d:5}/{d:5}MB ({d:3.0}%) | Temp: {d:2}°C | Status: {s}\n",
                .{color, state.device_id, state.name, ANSI_RESET,
                  @as(u32, @intFromFloat(state.utilization_percent)),
                  state.used_memory_mb, state.total_memory_mb, mem_percent,
                  @as(u32, @intFromFloat(state.temperature_c)), health_status});
            
            // Log to file
            try log.writer().print("{d},{d},{s},{d:.1},{d},{d},{d},{d:.1},{d:.1},{d:.1},{d:.1},{s}\n",
                .{timestamp, state.device_id, state.name, state.utilization_percent,
                  state.used_memory_mb, state.free_memory_mb, state.total_memory_mb, mem_percent,
                  state.temperature_c, state.power_usage_w, state.power_limit_w, health_status});
        }
        
        try stdout.print("\nNext update in {d} seconds... (Ctrl+C to stop)\n", .{interval_ms / 1000});
        
        // Sleep
        std.time.sleep(interval_ms * std.time.ns_per_ms);
    }
}
