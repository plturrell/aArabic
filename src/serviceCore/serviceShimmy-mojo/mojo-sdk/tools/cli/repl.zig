// Mojo CLI - Interactive REPL
// Read-Eval-Print Loop for interactive Mojo

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ReplOptions = struct {
    verbose: bool = false,
};

pub fn startRepl(allocator: Allocator, options: ReplOptions) !void {
    var repl_state = ReplState.init(allocator);
    defer repl_state.deinit();

    try printWelcome(options.verbose);
    
    var stdin = std.io.getStdIn().reader();
    var stdout = std.io.getStdOut().writer();
    
    var line_buffer: [4096]u8 = undefined;
    
    while (true) {
        // Print prompt
        try stdout.writeAll("mojo> ");
        
        // Read line
        const line = (try stdin.readUntilDelimiterOrEof(&line_buffer, '\n')) orelse break;
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        
        if (trimmed.len == 0) continue;
        
        // Handle REPL commands
        if (std.mem.startsWith(u8, trimmed, ":")) {
            const continue_repl = try handleCommand(allocator, &repl_state, trimmed, options);
            if (!continue_repl) break;
            continue;
        }
        
        // Evaluate expression
        evaluateExpression(allocator, &repl_state, trimmed, options) catch |err| {
            try stdout.print("Error: {}\n", .{err});
            continue;
        };
    }
    
    try stdout.writeAll("\nGoodbye!\n");
}

const ReplState = struct {
    variables: std.StringHashMap(Value),
    history: std.ArrayList([]const u8),
    allocator: Allocator,
    
    const Value = union(enum) {
        int: i64,
        float: f64,
        string: []const u8,
        bool_: bool,
    };
    
    fn init(allocator: Allocator) ReplState {
        return .{
            .variables = std.StringHashMap(Value).init(allocator),
            .history = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
        };
    }
    
    fn deinit(self: *ReplState) void {
        self.variables.deinit();
        for (self.history.items) |line| {
            self.allocator.free(line);
        }
        self.history.deinit();
    }
    
    fn setVariable(self: *ReplState, name: []const u8, value: Value) !void {
        try self.variables.put(name, value);
    }
    
    fn getVariable(self: *const ReplState, name: []const u8) ?Value {
        return self.variables.get(name);
    }
    
    fn addHistory(self: *ReplState, line: []const u8) !void {
        const copy = try self.allocator.dupe(u8, line);
        try self.history.append(copy);
    }
};

fn printWelcome(verbose: bool) !void {
    const stdout = std.io.getStdOut().writer();
    
    try stdout.writeAll("Mojo REPL v0.1.0\n");
    try stdout.writeAll("Type :help for help, :quit to exit\n");
    
    if (verbose) {
        try stdout.writeAll("Verbose mode enabled\n");
    }
    
    try stdout.writeAll("\n");
}

fn handleCommand(allocator: Allocator, state: *ReplState, command: []const u8, options: ReplOptions) !bool {
    const stdout = std.io.getStdOut().writer();
    
    if (std.mem.eql(u8, command, ":quit") or std.mem.eql(u8, command, ":q")) {
        return false; // Exit REPL
    } else if (std.mem.eql(u8, command, ":help") or std.mem.eql(u8, command, ":h")) {
        try printHelp();
    } else if (std.mem.eql(u8, command, ":clear") or std.mem.eql(u8, command, ":c")) {
        try clearScreen();
    } else if (std.mem.eql(u8, command, ":reset") or std.mem.eql(u8, command, ":r")) {
        state.variables.clearRetainingCapacity();
        try stdout.writeAll("REPL state reset\n");
    } else if (std.mem.eql(u8, command, ":vars")) {
        try printVariables(state);
    } else if (std.mem.startsWith(u8, command, ":type ")) {
        const expr = command[6..];
        try printType(allocator, state, expr, options);
    } else {
        try stdout.print("Unknown command: {s}\n", .{command});
        try stdout.writeAll("Type :help for available commands\n");
    }
    
    return true; // Continue REPL
}

fn printHelp() !void {
    const stdout = std.io.getStdOut().writer();
    
    try stdout.writeAll("\nREPL Commands:\n");
    try stdout.writeAll("  :quit, :q       Exit REPL\n");
    try stdout.writeAll("  :help, :h       Show this help\n");
    try stdout.writeAll("  :clear, :c      Clear screen\n");
    try stdout.writeAll("  :reset, :r      Reset REPL state\n");
    try stdout.writeAll("  :vars           Show variables\n");
    try stdout.writeAll("  :type <expr>    Show type of expression\n");
    try stdout.writeAll("\n");
}

fn clearScreen() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("\x1B[2J\x1B[H");
}

fn printVariables(state: *const ReplState) !void {
    const stdout = std.io.getStdOut().writer();
    
    if (state.variables.count() == 0) {
        try stdout.writeAll("No variables defined\n");
        return;
    }
    
    try stdout.writeAll("\nVariables:\n");
    
    var iter = state.variables.iterator();
    while (iter.next()) |entry| {
        try stdout.print("  {s} = ", .{entry.key_ptr.*});
        switch (entry.value_ptr.*) {
            .int => |val| try stdout.print("{d}\n", .{val}),
            .float => |val| try stdout.print("{d}\n", .{val}),
            .string => |val| try stdout.print("\"{s}\"\n", .{val}),
            .bool_ => |val| try stdout.print("{}\n", .{val}),
        }
    }
    
    try stdout.writeAll("\n");
}

fn printType(allocator: Allocator, state: *const ReplState, expr: []const u8, options: ReplOptions) !void {
    _ = state;
    _ = allocator;
    _ = options;
    
    const stdout = std.io.getStdOut().writer();
    
    // Infer type of expression
    // In real implementation, would use compiler type inference
    
    try stdout.print("Type of '{s}': Int\n", .{expr});
}

fn evaluateExpression(allocator: Allocator, state: *ReplState, expr: []const u8, options: ReplOptions) !void {
    // Add to history
    try state.addHistory(expr);
    
    // Parse and evaluate expression
    // In real implementation, would use compiler frontend
    
    const result = try evaluate(allocator, state, expr, options);
    defer result.deinit(allocator);
    
    // Print result
    try result.print();
}

const EvalResult = struct {
    value: ReplState.Value,
    type_name: []const u8,
    allocator: Allocator,
    
    fn deinit(self: *const EvalResult, allocator: Allocator) void {
        _ = self;
        _ = allocator;
    }
    
    fn print(self: *const EvalResult) !void {
        const stdout = std.io.getStdOut().writer();
        
        switch (self.value) {
            .int => |val| try stdout.print("{d} : {s}\n", .{ val, self.type_name }),
            .float => |val| try stdout.print("{d} : {s}\n", .{ val, self.type_name }),
            .string => |val| try stdout.print("\"{s}\" : {s}\n", .{ val, self.type_name }),
            .bool_ => |val| try stdout.print("{} : {s}\n", .{ val, self.type_name }),
        }
    }
};

fn evaluate(allocator: Allocator, state: *ReplState, expr: []const u8, options: ReplOptions) !EvalResult {
    _ = options;
    
    // Simple expression evaluation
    // In real implementation, would compile and execute using JIT
    
    // Check if it's a variable assignment
    if (std.mem.indexOf(u8, expr, "=")) |eq_pos| {
        const var_name = std.mem.trim(u8, expr[0..eq_pos], &std.ascii.whitespace);
        const value_str = std.mem.trim(u8, expr[eq_pos + 1 ..], &std.ascii.whitespace);
        
        // Parse value (very simplified)
        const value = if (std.fmt.parseInt(i64, value_str, 10)) |int_val| 
            ReplState.Value{ .int = int_val }
        else |_| 
            ReplState.Value{ .string = try allocator.dupe(u8, value_str) };
        
        try state.setVariable(var_name, value);
        
        return EvalResult{
            .value = value,
            .type_name = "Int",
            .allocator = allocator,
        };
    }
    
    // Check if it's a variable reference
    if (state.getVariable(expr)) |value| {
        return EvalResult{
            .value = value,
            .type_name = "Int",
            .allocator = allocator,
        };
    }
    
    // Try to parse as integer
    if (std.fmt.parseInt(i64, expr, 10)) |int_val| {
        return EvalResult{
            .value = .{ .int = int_val },
            .type_name = "Int",
            .allocator = allocator,
        };
    } else |_| {}
    
    // Default: treat as string
    return EvalResult{
        .value = .{ .string = try allocator.dupe(u8, expr) },
        .type_name = "String",
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "repl state init" {
    const allocator = std.testing.allocator;
    
    var state = ReplState.init(allocator);
    defer state.deinit();
    
    try state.setVariable("x", .{ .int = 42 });
    
    const value = state.getVariable("x");
    try std.testing.expect(value != null);
}

test "repl variable assignment" {
    const allocator = std.testing.allocator;
    
    var state = ReplState.init(allocator);
    defer state.deinit();
    
    try state.setVariable("test", .{ .int = 123 });
    
    const val = state.getVariable("test");
    try std.testing.expect(val != null);
    try std.testing.expectEqual(@as(i64, 123), val.?.int);
}

test "repl history" {
    const allocator = std.testing.allocator;
    
    var state = ReplState.init(allocator);
    defer state.deinit();
    
    try state.addHistory("let x = 42");
    try state.addHistory("print(x)");
    
    try std.testing.expectEqual(@as(usize, 2), state.history.items.len);
}
