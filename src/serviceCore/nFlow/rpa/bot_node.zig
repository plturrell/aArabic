//! RPA Bot Node for nWorkflow
//! Provides desktop automation capabilities

const std = @import("std");
const Allocator = std.mem.Allocator;

// RPA Action types
pub const RpaActionType = enum {
    CLICK,
    DOUBLE_CLICK,
    RIGHT_CLICK,
    TYPE_TEXT,
    PRESS_KEY,
    KEY_COMBO,
    MOUSE_MOVE,
    DRAG_DROP,
    SCROLL,
    SCREENSHOT,
    WAIT,
    WAIT_FOR_ELEMENT,
    READ_TEXT,
    GET_ATTRIBUTE,
    SET_VALUE,
    SELECT_OPTION,
    CHECK_CHECKBOX,
    FOCUS,
    HOVER,

    pub fn toString(self: RpaActionType) []const u8 {
        return @tagName(self);
    }
};

// Element locator strategies
pub const LocatorStrategy = enum {
    ID,
    NAME,
    CLASS,
    XPATH,
    CSS,
    TEXT,
    PARTIAL_TEXT,
    IMAGE,
    COORDINATES,

    pub fn toString(self: LocatorStrategy) []const u8 {
        return @tagName(self);
    }
};

// Element locator
pub const ElementLocator = struct {
    strategy: LocatorStrategy,
    value: []const u8,
    timeout_ms: u32 = 5000,
    index: u32 = 0,

    pub fn byId(value: []const u8) ElementLocator {
        return ElementLocator{ .strategy = .ID, .value = value };
    }

    pub fn byXpath(value: []const u8) ElementLocator {
        return ElementLocator{ .strategy = .XPATH, .value = value };
    }

    pub fn byCss(value: []const u8) ElementLocator {
        return ElementLocator{ .strategy = .CSS, .value = value };
    }

    pub fn byText(value: []const u8) ElementLocator {
        return ElementLocator{ .strategy = .TEXT, .value = value };
    }

    pub fn byName(value: []const u8) ElementLocator {
        return ElementLocator{ .strategy = .NAME, .value = value };
    }

    pub fn byClass(value: []const u8) ElementLocator {
        return ElementLocator{ .strategy = .CLASS, .value = value };
    }

    pub fn byCoordinates(x: u32, y: u32, allocator: Allocator) !ElementLocator {
        const value = try std.fmt.allocPrint(allocator, "{d},{d}", .{ x, y });
        return ElementLocator{
            .strategy = .COORDINATES,
            .value = value,
        };
    }
};

// RPA Action definition
pub const RpaAction = struct {
    action_type: RpaActionType,
    target: ?ElementLocator = null,
    value: ?[]const u8 = null,
    delay_before_ms: u32 = 0,
    delay_after_ms: u32 = 100,
    retry_count: u32 = 3,
    screenshot_on_error: bool = true,

    pub fn click(target: ElementLocator) RpaAction {
        return RpaAction{ .action_type = .CLICK, .target = target };
    }

    pub fn doubleClick(target: ElementLocator) RpaAction {
        return RpaAction{ .action_type = .DOUBLE_CLICK, .target = target };
    }

    pub fn rightClick(target: ElementLocator) RpaAction {
        return RpaAction{ .action_type = .RIGHT_CLICK, .target = target };
    }

    pub fn typeText(target: ElementLocator, text: []const u8) RpaAction {
        return RpaAction{ .action_type = .TYPE_TEXT, .target = target, .value = text };
    }

    pub fn pressKey(key: []const u8) RpaAction {
        return RpaAction{ .action_type = .PRESS_KEY, .value = key };
    }

    pub fn keyCombo(combo: []const u8) RpaAction {
        return RpaAction{ .action_type = .KEY_COMBO, .value = combo };
    }

    pub fn mouseMove(target: ElementLocator) RpaAction {
        return RpaAction{ .action_type = .MOUSE_MOVE, .target = target };
    }

    pub fn scroll(direction: []const u8) RpaAction {
        return RpaAction{ .action_type = .SCROLL, .value = direction };
    }

    pub fn wait(ms: u32, allocator: Allocator) !RpaAction {
        const value = try std.fmt.allocPrint(allocator, "{d}", .{ms});
        return RpaAction{ .action_type = .WAIT, .value = value };
    }

    pub fn waitForElement(target: ElementLocator) RpaAction {
        return RpaAction{ .action_type = .WAIT_FOR_ELEMENT, .target = target };
    }

    pub fn screenshot() RpaAction {
        return RpaAction{ .action_type = .SCREENSHOT };
    }

    pub fn readText(target: ElementLocator) RpaAction {
        return RpaAction{ .action_type = .READ_TEXT, .target = target };
    }

    pub fn getAttribute(target: ElementLocator, attr: []const u8) RpaAction {
        return RpaAction{ .action_type = .GET_ATTRIBUTE, .target = target, .value = attr };
    }

    pub fn setValue(target: ElementLocator, val: []const u8) RpaAction {
        return RpaAction{ .action_type = .SET_VALUE, .target = target, .value = val };
    }

    pub fn selectOption(target: ElementLocator, option: []const u8) RpaAction {
        return RpaAction{ .action_type = .SELECT_OPTION, .target = target, .value = option };
    }
};

// Action result
pub const ActionResult = struct {
    success: bool,
    action_type: RpaActionType,
    duration_ms: u64,
    output: ?[]const u8 = null,
    error_message: ?[]const u8 = null,
    screenshot_path: ?[]const u8 = null,

    pub fn succeeded(action_type: RpaActionType, duration_ms: u64) ActionResult {
        return ActionResult{
            .success = true,
            .action_type = action_type,
            .duration_ms = duration_ms,
        };
    }

    pub fn succeededWithOutput(action_type: RpaActionType, duration_ms: u64, output: []const u8) ActionResult {
        return ActionResult{
            .success = true,
            .action_type = action_type,
            .duration_ms = duration_ms,
            .output = output,
        };
    }

    pub fn failed(action_type: RpaActionType, duration_ms: u64, error_msg: []const u8) ActionResult {
        return ActionResult{
            .success = false,
            .action_type = action_type,
            .duration_ms = duration_ms,
            .error_message = error_msg,
        };
    }
};

// Bot execution status
pub const BotStatus = enum {
    IDLE,
    RUNNING,
    PAUSED,
    COMPLETED,
    FAILED,
    STOPPED,

    pub fn toString(self: BotStatus) []const u8 {
        return @tagName(self);
    }

    pub fn isTerminal(self: BotStatus) bool {
        return self == .COMPLETED or self == .FAILED or self == .STOPPED;
    }
};

// RPA Bot configuration
pub const BotConfig = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    max_retries: u32 = 3,
    retry_delay_ms: u32 = 1000,
    default_timeout_ms: u32 = 30000,
    screenshot_on_error: bool = true,
    screenshot_on_success: bool = false,
    target_platform: Platform = .WINDOWS,
    headless: bool = false,

    pub const Platform = enum {
        WINDOWS,
        MACOS,
        LINUX,
        WEB,

        pub fn toString(self: Platform) []const u8 {
            return @tagName(self);
        }
    };
};

// RPA Bot
pub const RpaBot = struct {
    id: []const u8,
    config: BotConfig,
    status: BotStatus = .IDLE,
    actions: std.ArrayList(RpaAction),
    results: std.ArrayList(ActionResult),
    current_action_index: usize = 0,
    start_time: ?i64 = null,
    end_time: ?i64 = null,
    variables: std.StringHashMap([]const u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, config: BotConfig) !RpaBot {
        return RpaBot{
            .id = try allocator.dupe(u8, id),
            .config = config,
            .actions = std.ArrayList(RpaAction){},
            .results = std.ArrayList(ActionResult){},
            .variables = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RpaBot) void {
        self.actions.deinit(self.allocator);
        self.results.deinit(self.allocator);
        self.variables.deinit();
        self.allocator.free(self.id);
    }

    pub fn addAction(self: *RpaBot, action: RpaAction) !void {
        try self.actions.append(self.allocator, action);
    }

    pub fn addActions(self: *RpaBot, actions_list: []const RpaAction) !void {
        for (actions_list) |action| {
            try self.actions.append(self.allocator, action);
        }
    }

    pub fn start(self: *RpaBot) !void {
        if (self.status == .RUNNING) return error.BotAlreadyRunning;
        self.status = .RUNNING;
        self.start_time = std.time.timestamp();
        self.current_action_index = 0;
    }

    pub fn pause(self: *RpaBot) void {
        if (self.status == .RUNNING) self.status = .PAUSED;
    }

    pub fn resume_(self: *RpaBot) void {
        if (self.status == .PAUSED) self.status = .RUNNING;
    }

    pub fn stop(self: *RpaBot) void {
        self.status = .STOPPED;
        self.end_time = std.time.timestamp();
    }

    pub fn reset(self: *RpaBot) void {
        self.status = .IDLE;
        self.current_action_index = 0;
        self.start_time = null;
        self.end_time = null;
        self.results.clearRetainingCapacity();
    }

    pub fn executeNext(self: *RpaBot) !?ActionResult {
        if (self.status != .RUNNING) return null;
        if (self.current_action_index >= self.actions.items.len) {
            self.status = .COMPLETED;
            self.end_time = std.time.timestamp();
            return null;
        }
        const action = self.actions.items[self.current_action_index];
        const result = try self.executeAction(action);
        try self.results.append(self.allocator, result);
        if (!result.success) {
            self.status = .FAILED;
            self.end_time = std.time.timestamp();
        } else {
            self.current_action_index += 1;
        }
        return result;
    }

    fn executeAction(self: *RpaBot, action: RpaAction) !ActionResult {
        const start_time = std.time.milliTimestamp();
        _ = self;
        // Simulated action execution - in real implementation would perform the action
        const action_type = action.action_type;
        const duration: u64 = @intCast(@max(0, std.time.milliTimestamp() - start_time));
        return ActionResult.succeeded(action_type, duration);
    }

    pub fn getProgress(self: *const RpaBot) f32 {
        if (self.actions.items.len == 0) return 0.0;
        return @as(f32, @floatFromInt(self.current_action_index)) /
            @as(f32, @floatFromInt(self.actions.items.len));
    }

    pub fn getSuccessRate(self: *const RpaBot) f32 {
        if (self.results.items.len == 0) return 0.0;
        var success_count: usize = 0;
        for (self.results.items) |result| {
            if (result.success) success_count += 1;
        }
        return @as(f32, @floatFromInt(success_count)) /
            @as(f32, @floatFromInt(self.results.items.len));
    }

    pub fn setVariable(self: *RpaBot, name: []const u8, value: []const u8) !void {
        try self.variables.put(name, value);
    }

    pub fn getVariable(self: *const RpaBot, name: []const u8) ?[]const u8 {
        return self.variables.get(name);
    }

    pub fn getActionCount(self: *const RpaBot) usize {
        return self.actions.items.len;
    }

    pub fn getCompletedCount(self: *const RpaBot) usize {
        return self.results.items.len;
    }

    pub fn toJson(self: *const RpaBot, allocator: Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        errdefer buffer.deinit(allocator);
        var writer = buffer.writer();
        try writer.print(
            \\{{"id":"{s}","status":"{s}","progress":{d:.2},"actions":{d},"completed":{d}}}
        , .{
            self.id,
            self.status.toString(),
            self.getProgress(),
            self.actions.items.len,
            self.results.items.len,
        });
        return buffer.toOwnedSlice(allocator);
    }
};

// Bot Manager
pub const BotManager = struct {
    bots: std.StringHashMap(*RpaBot),
    allocator: Allocator,

    pub fn init(allocator: Allocator) BotManager {
        return BotManager{
            .bots = std.StringHashMap(*RpaBot).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BotManager) void {
        var iter = self.bots.valueIterator();
        while (iter.next()) |bot| {
            bot.*.deinit();
            self.allocator.destroy(bot.*);
        }
        self.bots.deinit();
    }

    pub fn createBot(self: *BotManager, id: []const u8, config: BotConfig) !*RpaBot {
        const bot = try self.allocator.create(RpaBot);
        bot.* = try RpaBot.init(self.allocator, id, config);
        try self.bots.put(bot.id, bot);
        return bot;
    }

    pub fn getBot(self: *BotManager, id: []const u8) ?*RpaBot {
        return self.bots.get(id);
    }

    pub fn removeBot(self: *BotManager, id: []const u8) bool {
        if (self.bots.fetchRemove(id)) |entry| {
            entry.value.deinit();
            self.allocator.destroy(entry.value.*);
            return true;
        }
        return false;
    }

    pub fn getRunningBotCount(self: *const BotManager) usize {
        var count: usize = 0;
        var iter = self.bots.valueIterator();
        while (iter.next()) |bot| {
            if (bot.*.status == .RUNNING) count += 1;
        }
        return count;
    }

    pub fn getBotCount(self: *const BotManager) usize {
        return self.bots.count();
    }
};

// Tests
test "RpaBot creation" {
    const allocator = std.testing.allocator;
    const config = BotConfig{ .name = "Test Bot" };
    var bot = try RpaBot.init(allocator, "bot-1", config);
    defer bot.deinit();
    try std.testing.expectEqualStrings("bot-1", bot.id);
    try std.testing.expectEqual(BotStatus.IDLE, bot.status);
}

test "RpaBot actions" {
    const allocator = std.testing.allocator;
    const config = BotConfig{ .name = "Test Bot" };
    var bot = try RpaBot.init(allocator, "bot-1", config);
    defer bot.deinit();
    const locator = ElementLocator.byId("submit-btn");
    try bot.addAction(RpaAction.click(locator));
    try bot.addAction(RpaAction.typeText(locator, "Hello"));
    try std.testing.expectEqual(@as(usize, 2), bot.actions.items.len);
}

test "RpaBot execution" {
    const allocator = std.testing.allocator;
    const config = BotConfig{ .name = "Test Bot" };
    var bot = try RpaBot.init(allocator, "bot-1", config);
    defer bot.deinit();
    try bot.addAction(RpaAction.click(ElementLocator.byId("btn")));
    try bot.start();
    try std.testing.expectEqual(BotStatus.RUNNING, bot.status);
    const result = try bot.executeNext();
    try std.testing.expect(result != null);
    try std.testing.expect(result.?.success);
}

test "BotManager operations" {
    const allocator = std.testing.allocator;
    var manager = BotManager.init(allocator);
    defer manager.deinit();
    const config = BotConfig{ .name = "Bot 1" };
    _ = try manager.createBot("bot-1", config);
    try std.testing.expectEqual(@as(usize, 1), manager.bots.count());
}

test "ElementLocator factories" {
    const locator1 = ElementLocator.byId("my-id");
    try std.testing.expectEqual(LocatorStrategy.ID, locator1.strategy);
    const locator2 = ElementLocator.byXpath("//div[@class='test']");
    try std.testing.expectEqual(LocatorStrategy.XPATH, locator2.strategy);
}

test "RpaActionType toString" {
    try std.testing.expectEqualStrings("CLICK", RpaActionType.CLICK.toString());
    try std.testing.expectEqualStrings("TYPE_TEXT", RpaActionType.TYPE_TEXT.toString());
}

test "BotStatus transitions" {
    const allocator = std.testing.allocator;
    const config = BotConfig{ .name = "Test Bot" };
    var bot = try RpaBot.init(allocator, "bot-1", config);
    defer bot.deinit();
    try std.testing.expectEqual(BotStatus.IDLE, bot.status);
    try bot.start();
    try std.testing.expectEqual(BotStatus.RUNNING, bot.status);
    bot.pause();
    try std.testing.expectEqual(BotStatus.PAUSED, bot.status);
    bot.resume_();
    try std.testing.expectEqual(BotStatus.RUNNING, bot.status);
    bot.stop();
    try std.testing.expectEqual(BotStatus.STOPPED, bot.status);
}

test "RpaBot variables" {
    const allocator = std.testing.allocator;
    const config = BotConfig{ .name = "Test Bot" };
    var bot = try RpaBot.init(allocator, "bot-1", config);
    defer bot.deinit();
    try bot.setVariable("username", "admin");
    try std.testing.expectEqualStrings("admin", bot.getVariable("username").?);
    try std.testing.expectEqual(@as(?[]const u8, null), bot.getVariable("nonexistent"));
}

