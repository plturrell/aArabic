//! RPA Action Recorder for nWorkflow
//! Captures user actions and converts them to bot scripts

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Event types that can be recorded
pub const EventType = enum {
    CLICK,
    DOUBLE_CLICK,
    RIGHT_CLICK,
    MOUSE_MOVE,
    MOUSE_DOWN,
    MOUSE_UP,
    SCROLL,
    KEYSTROKE,
    KEY_DOWN,
    KEY_UP,
    TEXT_INPUT,
    WINDOW_OPEN,
    WINDOW_CLOSE,
    WINDOW_FOCUS,
    WINDOW_RESIZE,
    WINDOW_MOVE,
    CLIPBOARD_COPY,
    CLIPBOARD_PASTE,
    ELEMENT_FOCUS,
    ELEMENT_CHANGE,

    pub fn toString(self: EventType) []const u8 {
        return @tagName(self);
    }

    pub fn isMouseEvent(self: EventType) bool {
        return switch (self) {
            .CLICK, .DOUBLE_CLICK, .RIGHT_CLICK, .MOUSE_MOVE, .MOUSE_DOWN, .MOUSE_UP, .SCROLL => true,
            else => false,
        };
    }

    pub fn isKeyboardEvent(self: EventType) bool {
        return switch (self) {
            .KEYSTROKE, .KEY_DOWN, .KEY_UP, .TEXT_INPUT => true,
            else => false,
        };
    }

    pub fn isWindowEvent(self: EventType) bool {
        return switch (self) {
            .WINDOW_OPEN, .WINDOW_CLOSE, .WINDOW_FOCUS, .WINDOW_RESIZE, .WINDOW_MOVE => true,
            else => false,
        };
    }
};

/// Mouse button identifiers
pub const MouseButton = enum {
    LEFT,
    MIDDLE,
    RIGHT,

    pub fn toString(self: MouseButton) []const u8 {
        return @tagName(self);
    }
};

/// Recorded event data
pub const RecordedEvent = struct {
    event_type: EventType,
    timestamp: i64,
    x: i32 = 0,
    y: i32 = 0,
    button: MouseButton = .LEFT,
    key_code: ?u32 = null,
    modifiers: Modifiers = .{},
    text: ?[]const u8 = null,
    window_handle: ?u64 = null,
    window_title: ?[]const u8 = null,
    element_id: ?[]const u8 = null,

    pub const Modifiers = struct {
        ctrl: bool = false,
        alt: bool = false,
        shift: bool = false,
        meta: bool = false,

        pub fn hasAny(self: *const Modifiers) bool {
            return self.ctrl or self.alt or self.shift or self.meta;
        }
    };

    pub fn click(x: i32, y: i32, button: MouseButton) RecordedEvent {
        return RecordedEvent{
            .event_type = .CLICK,
            .timestamp = std.time.timestamp(),
            .x = x,
            .y = y,
            .button = button,
        };
    }

    pub fn doubleClick(x: i32, y: i32) RecordedEvent {
        return RecordedEvent{
            .event_type = .DOUBLE_CLICK,
            .timestamp = std.time.timestamp(),
            .x = x,
            .y = y,
        };
    }

    pub fn keystroke(key_code: u32, modifiers: Modifiers) RecordedEvent {
        return RecordedEvent{
            .event_type = .KEYSTROKE,
            .timestamp = std.time.timestamp(),
            .key_code = key_code,
            .modifiers = modifiers,
        };
    }

    pub fn textInput(text: []const u8) RecordedEvent {
        return RecordedEvent{
            .event_type = .TEXT_INPUT,
            .timestamp = std.time.timestamp(),
            .text = text,
        };
    }

    pub fn windowFocus(handle: u64, title: []const u8) RecordedEvent {
        return RecordedEvent{
            .event_type = .WINDOW_FOCUS,
            .timestamp = std.time.timestamp(),
            .window_handle = handle,
            .window_title = title,
        };
    }

    pub fn mouseMove(x: i32, y: i32) RecordedEvent {
        return RecordedEvent{
            .event_type = .MOUSE_MOVE,
            .timestamp = std.time.timestamp(),
            .x = x,
            .y = y,
        };
    }

    pub fn scroll(x: i32, y: i32, delta: i32) RecordedEvent {
        _ = delta;
        return RecordedEvent{
            .event_type = .SCROLL,
            .timestamp = std.time.timestamp(),
            .x = x,
            .y = y,
        };
    }
};

/// Recording session status
pub const RecordingStatus = enum {
    IDLE,
    RECORDING,
    PAUSED,
    STOPPED,

    pub fn toString(self: RecordingStatus) []const u8 {
        return @tagName(self);
    }
};

/// Recording session configuration
pub const RecordingConfig = struct {
    capture_mouse_moves: bool = false,
    capture_screenshots: bool = false,
    merge_text_events: bool = true,
    min_event_interval_ms: u32 = 50,
    max_events: u32 = 10000,
    filter_redundant: bool = true,
};

/// Recording session
pub const RecordingSession = struct {
    id: []const u8,
    name: []const u8,
    config: RecordingConfig,
    status: RecordingStatus = .IDLE,
    events: std.ArrayList(RecordedEvent),
    start_time: ?i64 = null,
    end_time: ?i64 = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: RecordingConfig) !RecordingSession {
        return RecordingSession{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .config = config,
            .events = std.ArrayList(RecordedEvent){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RecordingSession) void {
        self.events.deinit(self.allocator);
        self.allocator.free(self.id);
        self.allocator.free(self.name);
    }

    pub fn startRecording(self: *RecordingSession) !void {
        if (self.status == .RECORDING) return error.AlreadyRecording;
        self.status = .RECORDING;
        self.start_time = std.time.timestamp();
    }

    pub fn pauseRecording(self: *RecordingSession) void {
        if (self.status == .RECORDING) self.status = .PAUSED;
    }

    pub fn resumeRecording(self: *RecordingSession) void {
        if (self.status == .PAUSED) self.status = .RECORDING;
    }

    pub fn stopRecording(self: *RecordingSession) void {
        self.status = .STOPPED;
        self.end_time = std.time.timestamp();
    }

    pub fn addEvent(self: *RecordingSession, event: RecordedEvent) !void {
        if (self.status != .RECORDING) return error.NotRecording;
        if (self.events.items.len >= self.config.max_events) return error.MaxEventsReached;
        if (!self.config.capture_mouse_moves and event.event_type == .MOUSE_MOVE) return;
        try self.events.append(self.allocator, event);
    }

    pub fn getEventCount(self: *const RecordingSession) usize {
        return self.events.items.len;
    }

    pub fn getDuration(self: *const RecordingSession) ?i64 {
        if (self.start_time) |start| {
            const end = self.end_time orelse std.time.timestamp();
            return end - start;
        }
        return null;
    }

    pub fn clear(self: *RecordingSession) void {
        self.events.clearRetainingCapacity();
        self.status = .IDLE;
        self.start_time = null;
        self.end_time = null;
    }
};

/// Action recorder
pub const ActionRecorder = struct {
    sessions: std.StringHashMap(*RecordingSession),
    active_session: ?*RecordingSession = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ActionRecorder {
        return ActionRecorder{
            .sessions = std.StringHashMap(*RecordingSession).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ActionRecorder) void {
        var iter = self.sessions.valueIterator();
        while (iter.next()) |session| {
            session.*.deinit();
            self.allocator.destroy(session.*);
        }
        self.sessions.deinit();
    }

    pub fn createSession(self: *ActionRecorder, id: []const u8, name: []const u8, config: RecordingConfig) !*RecordingSession {
        const session = try self.allocator.create(RecordingSession);
        session.* = try RecordingSession.init(self.allocator, id, name, config);
        try self.sessions.put(session.id, session);
        return session;
    }

    pub fn getSession(self: *ActionRecorder, id: []const u8) ?*RecordingSession {
        return self.sessions.get(id);
    }

    pub fn startRecording(self: *ActionRecorder, session_id: []const u8) !void {
        if (self.sessions.get(session_id)) |session| {
            try session.startRecording();
            self.active_session = session;
        } else {
            return error.SessionNotFound;
        }
    }

    pub fn stopRecording(self: *ActionRecorder) void {
        if (self.active_session) |session| {
            session.stopRecording();
            self.active_session = null;
        }
    }

    pub fn recordEvent(self: *ActionRecorder, event: RecordedEvent) !void {
        if (self.active_session) |session| {
            try session.addEvent(event);
        }
    }

    pub fn getSessionCount(self: *const ActionRecorder) usize {
        return self.sessions.count();
    }
};

/// Script generator - converts recordings to executable scripts
pub const ScriptGenerator = struct {
    output_format: OutputFormat = .ZIG,
    include_comments: bool = true,
    include_delays: bool = true,
    allocator: Allocator,

    pub const OutputFormat = enum {
        ZIG,
        JSON,
        YAML,

        pub fn toString(self: OutputFormat) []const u8 {
            return @tagName(self);
        }

        pub fn extension(self: OutputFormat) []const u8 {
            return switch (self) {
                .ZIG => ".zig",
                .JSON => ".json",
                .YAML => ".yaml",
            };
        }
    };

    pub fn init(allocator: Allocator) ScriptGenerator {
        return ScriptGenerator{ .allocator = allocator };
    }

    pub fn generateScript(self: *const ScriptGenerator, session: *const RecordingSession) ![]const u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        errdefer buffer.deinit(self.allocator);
        var writer = buffer.writer();
        try writer.print("// Generated from recording: {s}\n", .{session.name});
        try writer.print("// Events: {d}\n\n", .{session.events.items.len});
        for (session.events.items) |event| {
            try self.writeEvent(&writer, event);
        }
        return buffer.toOwnedSlice(self.allocator);
    }

    fn writeEvent(self: *const ScriptGenerator, writer: anytype, event: RecordedEvent) !void {
        _ = self;
        switch (event.event_type) {
            .CLICK => try writer.print("bot.click({d}, {d});\n", .{ event.x, event.y }),
            .DOUBLE_CLICK => try writer.print("bot.doubleClick({d}, {d});\n", .{ event.x, event.y }),
            .KEYSTROKE => try writer.print("bot.pressKey({d});\n", .{event.key_code orelse 0}),
            .TEXT_INPUT => try writer.print("bot.typeText(\"{s}\");\n", .{event.text orelse ""}),
            else => try writer.print("// {s} event\n", .{event.event_type.toString()}),
        }
    }

    pub fn generateJson(self: *const ScriptGenerator, session: *const RecordingSession) ![]const u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        errdefer buffer.deinit(self.allocator);
        var writer = buffer.writer();
        try writer.print("{{\"name\":\"{s}\",\"events\":[", .{session.name});
        for (session.events.items, 0..) |event, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{{\"type\":\"{s}\",\"x\":{d},\"y\":{d}}}", .{
                event.event_type.toString(),
                event.x,
                event.y,
            });
        }
        try writer.writeAll("]}");
        return buffer.toOwnedSlice(self.allocator);
    }
};

// Tests
test "EventType properties" {
    try std.testing.expect(EventType.CLICK.isMouseEvent());
    try std.testing.expect(!EventType.CLICK.isKeyboardEvent());
    try std.testing.expect(EventType.KEYSTROKE.isKeyboardEvent());
    try std.testing.expect(EventType.WINDOW_FOCUS.isWindowEvent());
}

test "RecordedEvent creation" {
    const event = RecordedEvent.click(100, 200, .LEFT);
    try std.testing.expectEqual(EventType.CLICK, event.event_type);
    try std.testing.expectEqual(@as(i32, 100), event.x);
    try std.testing.expectEqual(@as(i32, 200), event.y);
}

test "RecordingSession lifecycle" {
    const allocator = std.testing.allocator;
    var session = try RecordingSession.init(allocator, "sess-1", "Test Session", RecordingConfig{});
    defer session.deinit();
    try std.testing.expectEqual(RecordingStatus.IDLE, session.status);
    try session.startRecording();
    try std.testing.expectEqual(RecordingStatus.RECORDING, session.status);
    try session.addEvent(RecordedEvent.click(10, 20, .LEFT));
    try std.testing.expectEqual(@as(usize, 1), session.getEventCount());
    session.stopRecording();
    try std.testing.expectEqual(RecordingStatus.STOPPED, session.status);
}

test "ActionRecorder operations" {
    const allocator = std.testing.allocator;
    var recorder = ActionRecorder.init(allocator);
    defer recorder.deinit();
    _ = try recorder.createSession("sess-1", "Test", RecordingConfig{});
    try std.testing.expectEqual(@as(usize, 1), recorder.getSessionCount());
}

test "ScriptGenerator output formats" {
    try std.testing.expectEqualStrings(".zig", ScriptGenerator.OutputFormat.ZIG.extension());
    try std.testing.expectEqualStrings(".json", ScriptGenerator.OutputFormat.JSON.extension());
}

test "Modifiers check" {
    const mods = RecordedEvent.Modifiers{ .ctrl = true };
    try std.testing.expect(mods.hasAny());
    const no_mods = RecordedEvent.Modifiers{};
    try std.testing.expect(!no_mods.hasAny());
}

