const std = @import("std");
const petri_net = @import("petri_net");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== nWorkflow Basic Example ===\n\n", .{});

    // Create a simple workflow: Input -> Process -> Output
    var net = try petri_net.PetriNet.init(allocator, "Simple Workflow");
    defer net.deinit();

    // Add places
    _ = try net.addPlace("input", "Input Queue", null);
    _ = try net.addPlace("processing", "Processing", 1);
    _ = try net.addPlace("output", "Output Queue", null);

    // Add transitions
    _ = try net.addTransition("start_process", "Start Processing", 0);
    _ = try net.addTransition("finish_process", "Finish Processing", 0);

    // Add arcs
    _ = try net.addArc("a1", .input, 1, "input", "start_process");
    _ = try net.addArc("a2", .output, 1, "start_process", "processing");
    _ = try net.addArc("a3", .input, 1, "processing", "finish_process");
    _ = try net.addArc("a4", .output, 1, "finish_process", "output");

    // Add initial tokens
    try net.addTokenToPlace("input", "{\"task\": \"Process document 1\"}");
    try net.addTokenToPlace("input", "{\"task\": \"Process document 2\"}");

    std.debug.print("Initial state:\n", .{});
    printStats(&net);

    // Execute workflow
    var step: usize = 1;
    while (!net.isDeadlocked() and step <= 10) {
        var enabled = try net.getEnabledTransitions();
        defer enabled.deinit(allocator);

        if (enabled.items.len == 0) break;

        std.debug.print("\nStep {d}: Firing transition '{s}'\n", .{ step, enabled.items[0] });
        try net.fireTransition(enabled.items[0]);
        printStats(&net);

        step += 1;
    }

    std.debug.print("\n=== Workflow Complete ===\n", .{});
}

fn printStats(net: *const petri_net.PetriNet) void {
    const stats = net.getStats();
    std.debug.print("  Places: {d}, Transitions: {d}, Tokens: {d}\n", .{
        stats.place_count,
        stats.transition_count,
        stats.total_tokens,
    });
}
