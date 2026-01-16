// Debug Adapter Protocol (DAP)
// Day 89: Foundation for debugging support

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// DAP Protocol Types
// ============================================================================

pub const MessageType = enum {
    Request,
    Response,
    Event,
    
    pub fn toString(self: MessageType) []const u8 {
        return switch (self) {
            .Request => "request",
            .Response => "response",
            .Event => "event",
        };
    }
};

pub const Message = struct {
    seq: i32,
    type_: MessageType,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, seq: i32, type_: MessageType) Message {
        return Message{
            .seq = seq,
            .type_ = type_,
            .allocator = allocator,
        };
    }
};

// ============================================================================
// Source Location
// ============================================================================

pub const Source = struct {
    path: []const u8,
    name: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, path: []const u8) !Source {
        return Source{
            .path = try allocator.dupe(u8, path),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Source) void {
        self.allocator.free(self.path);
        if (self.name) |name| {
            self.allocator.free(name);
        }
    }
};

// ============================================================================
// Breakpoint Management
// ============================================================================

pub const Breakpoint = struct {
    id: u32,
    verified: bool,
    source: Source,
    line: u32,
    column: ?u32 = null,
    condition: ?[]const u8 = null,
    hit_condition: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, id: u32, source: Source, line: u32) Breakpoint {
        return Breakpoint{
            .id = id,
            .verified = false,
            .source = source,
            .line = line,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Breakpoint) void {
        if (self.condition) |cond| {
            self.allocator.free(cond);
        }
        if (self.hit_condition) |hit| {
            self.allocator.free(hit);
        }
        self.source.deinit();
    }
    
    pub fn verify(self: *Breakpoint) void {
        self.verified = true;
    }
    
    pub fn withCondition(self: Breakpoint, condition: []const u8) !Breakpoint {
        var bp = self;
        bp.condition = try self.allocator.dupe(u8, condition);
        return bp;
    }
};

pub const BreakpointManager = struct {
    breakpoints: std.ArrayList(Breakpoint),
    next_id: u32,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) BreakpointManager {
        return BreakpointManager{
            .breakpoints = std.ArrayList(Breakpoint){},
            .next_id = 1,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *BreakpointManager) void {
        for (self.breakpoints.items) |*bp| {
            bp.deinit();
        }
        self.breakpoints.deinit(self.allocator);
    }
    
    pub fn addBreakpoint(self: *BreakpointManager, source: Source, line: u32) !u32 {
        const id = self.next_id;
        self.next_id += 1;
        
        var bp = Breakpoint.init(self.allocator, id, source, line);
        bp.verify(); // Auto-verify for now
        
        try self.breakpoints.append(self.allocator, bp);
        return id;
    }
    
    pub fn removeBreakpoint(self: *BreakpointManager, id: u32) bool {
        for (self.breakpoints.items, 0..) |bp, i| {
            if (bp.id == id) {
                var removed = self.breakpoints.orderedRemove(i);
                removed.deinit();
                return true;
            }
        }
        return false;
    }
    
    pub fn getBreakpoint(self: *BreakpointManager, id: u32) ?*Breakpoint {
        for (self.breakpoints.items) |*bp| {
            if (bp.id == id) return bp;
        }
        return null;
    }
    
    pub fn getBreakpointsForFile(self: *BreakpointManager, path: []const u8) std.ArrayList(*Breakpoint) {
        var result = std.ArrayList(*Breakpoint){};
        for (self.breakpoints.items) |*bp| {
            if (std.mem.eql(u8, bp.source.path, path)) {
                result.append(self.allocator, bp) catch {};
            }
        }
        return result;
    }
};

// ============================================================================
// Stack Frame
// ============================================================================

pub const StackFrame = struct {
    id: u32,
    name: []const u8,
    source: Source,
    line: u32,
    column: u32,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, id: u32, name: []const u8, source: Source, line: u32, column: u32) !StackFrame {
        return StackFrame{
            .id = id,
            .name = try allocator.dupe(u8, name),
            .source = source,
            .line = line,
            .column = column,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *StackFrame) void {
        self.allocator.free(self.name);
        self.source.deinit();
    }
};

// ============================================================================
// Debug Adapter
// ============================================================================

pub const DebugAdapter = struct {
    allocator: Allocator,
    breakpoint_manager: BreakpointManager,
    initialized: bool = false,
    sequence: i32 = 1,
    
    pub fn init(allocator: Allocator) DebugAdapter {
        return DebugAdapter{
            .allocator = allocator,
            .breakpoint_manager = BreakpointManager.init(allocator),
        };
    }
    
    pub fn deinit(self: *DebugAdapter) void {
        self.breakpoint_manager.deinit();
    }
    
    pub fn nextSequence(self: *DebugAdapter) i32 {
        const seq = self.sequence;
        self.sequence += 1;
        return seq;
    }
    
    /// Handle initialize request
    pub fn handleInitialize(self: *DebugAdapter) !void {
        self.initialized = true;
    }
    
    /// Handle setBreakpoints request
    pub fn handleSetBreakpoints(
        self: *DebugAdapter,
        source_path: []const u8,
        lines: []const u32,
    ) !std.ArrayList(u32) {
        var breakpoint_ids = std.ArrayList(u32){};
        
        for (lines) |line| {
            const source = try Source.init(self.allocator, source_path);
            const id = try self.breakpoint_manager.addBreakpoint(source, line);
            try breakpoint_ids.append(self.allocator, id);
        }
        
        return breakpoint_ids;
    }
    
    /// Handle stackTrace request
    pub fn handleStackTrace(self: *DebugAdapter, thread_id: u32) !std.ArrayList(StackFrame) {
        _ = thread_id;
        
        var frames = std.ArrayList(StackFrame){};
        
        // Example stack frame
        const source = try Source.init(self.allocator, "main.mojo");
        const frame = try StackFrame.init(
            self.allocator,
            1,
            "main",
            source,
            10,
            5,
        );
        try frames.append(self.allocator, frame);
        
        return frames;
    }
    
    /// Handle continue request
    pub fn handleContinue(self: *DebugAdapter) !void {
        _ = self;
        // Resume execution
    }
    
    /// Handle pause request
    pub fn handlePause(self: *DebugAdapter) !void {
        _ = self;
        // Pause execution
    }
    
    /// Handle stepIn request
    pub fn handleStepIn(self: *DebugAdapter) !void {
        _ = self;
        // Step into function
    }
    
    /// Handle stepOut request
    pub fn handleStepOut(self: *DebugAdapter) !void {
        _ = self;
        // Step out of function
    }
    
    /// Handle next request (step over)
    pub fn handleNext(self: *DebugAdapter) !void {
        _ = self;
        // Step over line
    }
};

// ============================================================================
// Tests
// ============================================================================

test "MessageType: toString" {
    try std.testing.expectEqualStrings("request", MessageType.Request.toString());
    try std.testing.expectEqualStrings("response", MessageType.Response.toString());
    try std.testing.expectEqualStrings("event", MessageType.Event.toString());
}

test "Source: creation" {
    var source = try Source.init(std.testing.allocator, "test.mojo");
    defer source.deinit();
    
    try std.testing.expectEqualStrings("test.mojo", source.path);
}

test "Breakpoint: creation and verification" {
    const source = try Source.init(std.testing.allocator, "test.mojo");
    var bp = Breakpoint.init(std.testing.allocator, 1, source, 10);
    defer bp.deinit();
    
    try std.testing.expect(!bp.verified);
    bp.verify();
    try std.testing.expect(bp.verified);
}

test "BreakpointManager: add and remove" {
    var manager = BreakpointManager.init(std.testing.allocator);
    defer manager.deinit();
    
    const source = try Source.init(std.testing.allocator, "test.mojo");
    const id = try manager.addBreakpoint(source, 10);
    
    try std.testing.expect(manager.getBreakpoint(id) != null);
    
    const removed = manager.removeBreakpoint(id);
    try std.testing.expect(removed);
    try std.testing.expect(manager.getBreakpoint(id) == null);
}

test "BreakpointManager: multiple breakpoints" {
    var manager = BreakpointManager.init(std.testing.allocator);
    defer manager.deinit();
    
    const source1 = try Source.init(std.testing.allocator, "test.mojo");
    const id1 = try manager.addBreakpoint(source1, 10);
    
    const source2 = try Source.init(std.testing.allocator, "test.mojo");
    const id2 = try manager.addBreakpoint(source2, 20);
    
    try std.testing.expect(id1 != id2);
    try std.testing.expectEqual(@as(usize, 2), manager.breakpoints.items.len);
}

test "StackFrame: creation" {
    const source = try Source.init(std.testing.allocator, "main.mojo");
    var frame = try StackFrame.init(std.testing.allocator, 1, "main", source, 10, 5);
    defer frame.deinit();
    
    try std.testing.expectEqualStrings("main", frame.name);
    try std.testing.expectEqual(@as(u32, 10), frame.line);
}

test "DebugAdapter: initialization" {
    var adapter = DebugAdapter.init(std.testing.allocator);
    defer adapter.deinit();
    
    try std.testing.expect(!adapter.initialized);
    try adapter.handleInitialize();
    try std.testing.expect(adapter.initialized);
}

test "DebugAdapter: set breakpoints" {
    var adapter = DebugAdapter.init(std.testing.allocator);
    defer adapter.deinit();
    
    const lines = [_]u32{ 10, 20, 30 };
    var ids = try adapter.handleSetBreakpoints("test.mojo", &lines);
    defer ids.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(@as(usize, 3), ids.items.len);
}
