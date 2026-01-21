//! RPA Bot Scheduler for nWorkflow
//! Manages scheduled and event-triggered bot runs

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Schedule types
pub const ScheduleType = enum {
    ONCE,
    HOURLY,
    DAILY,
    WEEKLY,
    MONTHLY,
    CRON,
    ON_DEMAND,

    pub fn toString(self: ScheduleType) []const u8 {
        return @tagName(self);
    }
};

/// Days of week
pub const DayOfWeek = enum(u3) {
    SUNDAY = 0,
    MONDAY = 1,
    TUESDAY = 2,
    WEDNESDAY = 3,
    THURSDAY = 4,
    FRIDAY = 5,
    SATURDAY = 6,

    pub fn toString(self: DayOfWeek) []const u8 {
        return @tagName(self);
    }
};

/// Schedule configuration
pub const BotSchedule = struct {
    schedule_type: ScheduleType,
    hour: u8 = 0,
    minute: u8 = 0,
    day_of_week: ?DayOfWeek = null,
    day_of_month: ?u8 = null,
    cron_expression: ?[]const u8 = null,
    timezone: []const u8 = "UTC",
    enabled: bool = true,
    start_date: ?i64 = null,
    end_date: ?i64 = null,

    pub fn once(hour: u8, minute: u8) BotSchedule {
        return BotSchedule{ .schedule_type = .ONCE, .hour = hour, .minute = minute };
    }

    pub fn daily(hour: u8, minute: u8) BotSchedule {
        return BotSchedule{ .schedule_type = .DAILY, .hour = hour, .minute = minute };
    }

    pub fn weekly(day: DayOfWeek, hour: u8, minute: u8) BotSchedule {
        return BotSchedule{ .schedule_type = .WEEKLY, .hour = hour, .minute = minute, .day_of_week = day };
    }

    pub fn monthly(day: u8, hour: u8, minute: u8) BotSchedule {
        return BotSchedule{ .schedule_type = .MONTHLY, .hour = hour, .minute = minute, .day_of_month = day };
    }

    pub fn hourly(minute: u8) BotSchedule {
        return BotSchedule{ .schedule_type = .HOURLY, .minute = minute };
    }

    pub fn cron(expression: []const u8) BotSchedule {
        return BotSchedule{ .schedule_type = .CRON, .cron_expression = expression };
    }

    pub fn onDemand() BotSchedule {
        return BotSchedule{ .schedule_type = .ON_DEMAND };
    }

    pub fn isActive(self: *const BotSchedule, current_time: i64) bool {
        if (!self.enabled) return false;
        if (self.start_date) |start| {
            if (current_time < start) return false;
        }
        if (self.end_date) |end| {
            if (current_time > end) return false;
        }
        return true;
    }
};

/// Trigger condition types
pub const TriggerType = enum {
    FILE_CREATED,
    FILE_MODIFIED,
    FILE_DELETED,
    EMAIL_RECEIVED,
    HTTP_REQUEST,
    DATABASE_CHANGE,
    QUEUE_MESSAGE,
    WORKFLOW_COMPLETED,
    MANUAL,
    API_CALL,

    pub fn toString(self: TriggerType) []const u8 {
        return @tagName(self);
    }
};

/// Trigger condition
pub const TriggerCondition = struct {
    trigger_type: TriggerType,
    filter: ?[]const u8 = null,
    enabled: bool = true,
    priority: u8 = 5,
    cooldown_ms: u32 = 0,
    last_triggered: ?i64 = null,

    pub fn fileCreated(filter: ?[]const u8) TriggerCondition {
        return TriggerCondition{ .trigger_type = .FILE_CREATED, .filter = filter };
    }

    pub fn fileModified(filter: ?[]const u8) TriggerCondition {
        return TriggerCondition{ .trigger_type = .FILE_MODIFIED, .filter = filter };
    }

    pub fn emailReceived(filter: ?[]const u8) TriggerCondition {
        return TriggerCondition{ .trigger_type = .EMAIL_RECEIVED, .filter = filter };
    }

    pub fn httpRequest() TriggerCondition {
        return TriggerCondition{ .trigger_type = .HTTP_REQUEST };
    }

    pub fn manual() TriggerCondition {
        return TriggerCondition{ .trigger_type = .MANUAL };
    }

    pub fn apiCall() TriggerCondition {
        return TriggerCondition{ .trigger_type = .API_CALL };
    }

    pub fn canTrigger(self: *const TriggerCondition, current_time: i64) bool {
        if (!self.enabled) return false;
        if (self.last_triggered) |last| {
            const elapsed: u64 = @intCast(current_time - last);
            if (elapsed < self.cooldown_ms) return false;
        }
        return true;
    }
};

/// Schedule entry status
pub const EntryStatus = enum {
    ACTIVE,
    PAUSED,
    DISABLED,
    COMPLETED,
    FAILED,

    pub fn toString(self: EntryStatus) []const u8 {
        return @tagName(self);
    }
};

/// Schedule entry
pub const ScheduleEntry = struct {
    id: []const u8,
    bot_id: []const u8,
    name: []const u8,
    schedule: BotSchedule,
    triggers: std.ArrayList(TriggerCondition),
    status: EntryStatus = .ACTIVE,
    run_count: u32 = 0,
    success_count: u32 = 0,
    failure_count: u32 = 0,
    last_run: ?i64 = null,
    next_run: ?i64 = null,
    created_at: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, bot_id: []const u8, name: []const u8, schedule: BotSchedule) !ScheduleEntry {
        return ScheduleEntry{
            .id = try allocator.dupe(u8, id),
            .bot_id = try allocator.dupe(u8, bot_id),
            .name = try allocator.dupe(u8, name),
            .schedule = schedule,
            .triggers = std.ArrayList(TriggerCondition){},
            .created_at = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ScheduleEntry) void {
        self.triggers.deinit(self.allocator);
        self.allocator.free(self.id);
        self.allocator.free(self.bot_id);
        self.allocator.free(self.name);
    }

    pub fn addTrigger(self: *ScheduleEntry, trigger: TriggerCondition) !void {
        try self.triggers.append(self.allocator, trigger);
    }

    pub fn recordRun(self: *ScheduleEntry, success: bool) void {
        self.run_count += 1;
        if (success) {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        self.last_run = std.time.timestamp();
    }

    pub fn getSuccessRate(self: *const ScheduleEntry) f32 {
        if (self.run_count == 0) return 0.0;
        return @as(f32, @floatFromInt(self.success_count)) / @as(f32, @floatFromInt(self.run_count));
    }

    pub fn pause(self: *ScheduleEntry) void {
        if (self.status == .ACTIVE) self.status = .PAUSED;
    }

    pub fn resume_(self: *ScheduleEntry) void {
        if (self.status == .PAUSED) self.status = .ACTIVE;
    }

    pub fn disable(self: *ScheduleEntry) void {
        self.status = .DISABLED;
    }
};

/// Bot scheduler
pub const BotScheduler = struct {
    entries: std.StringHashMap(*ScheduleEntry),
    running_bots: std.ArrayList([]const u8),
    max_concurrent: u32 = 5,
    allocator: Allocator,

    pub fn init(allocator: Allocator) BotScheduler {
        return BotScheduler{
            .entries = std.StringHashMap(*ScheduleEntry).init(allocator),
            .running_bots = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BotScheduler) void {
        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            entry.*.deinit();
            self.allocator.destroy(entry.*);
        }
        self.entries.deinit();
        self.running_bots.deinit(self.allocator);
    }

    pub fn addEntry(self: *BotScheduler, id: []const u8, bot_id: []const u8, name: []const u8, schedule: BotSchedule) !*ScheduleEntry {
        const entry = try self.allocator.create(ScheduleEntry);
        entry.* = try ScheduleEntry.init(self.allocator, id, bot_id, name, schedule);
        try self.entries.put(entry.id, entry);
        return entry;
    }

    pub fn getEntry(self: *BotScheduler, id: []const u8) ?*ScheduleEntry {
        return self.entries.get(id);
    }

    pub fn removeEntry(self: *BotScheduler, id: []const u8) bool {
        if (self.entries.fetchRemove(id)) |kv| {
            kv.value.deinit();
            self.allocator.destroy(kv.value.*);
            return true;
        }
        return false;
    }

    pub fn getActiveEntries(self: *const BotScheduler) !std.ArrayList(*ScheduleEntry) {
        var result = std.ArrayList(*ScheduleEntry){};
        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            if (entry.*.status == .ACTIVE) {
                try result.append(self.allocator, entry.*);
            }
        }
        return result;
    }

    pub fn getEntryCount(self: *const BotScheduler) usize {
        return self.entries.count();
    }

    pub fn getRunningCount(self: *const BotScheduler) usize {
        return self.running_bots.items.len;
    }

    pub fn canStartBot(self: *const BotScheduler) bool {
        return self.running_bots.items.len < self.max_concurrent;
    }
};

// Tests
test "BotSchedule creation" {
    const daily = BotSchedule.daily(9, 30);
    try std.testing.expectEqual(ScheduleType.DAILY, daily.schedule_type);
    try std.testing.expectEqual(@as(u8, 9), daily.hour);
    try std.testing.expectEqual(@as(u8, 30), daily.minute);
}

test "BotSchedule weekly" {
    const weekly = BotSchedule.weekly(.MONDAY, 10, 0);
    try std.testing.expectEqual(ScheduleType.WEEKLY, weekly.schedule_type);
    try std.testing.expectEqual(DayOfWeek.MONDAY, weekly.day_of_week.?);
}

test "TriggerCondition creation" {
    const trigger = TriggerCondition.fileCreated("*.csv");
    try std.testing.expectEqual(TriggerType.FILE_CREATED, trigger.trigger_type);
    try std.testing.expectEqualStrings("*.csv", trigger.filter.?);
}

test "ScheduleEntry lifecycle" {
    const allocator = std.testing.allocator;
    var entry = try ScheduleEntry.init(allocator, "entry-1", "bot-1", "Daily Report", BotSchedule.daily(9, 0));
    defer entry.deinit();
    try std.testing.expectEqual(EntryStatus.ACTIVE, entry.status);
    entry.recordRun(true);
    try std.testing.expectEqual(@as(u32, 1), entry.run_count);
    try std.testing.expectEqual(@as(u32, 1), entry.success_count);
}

test "BotScheduler operations" {
    const allocator = std.testing.allocator;
    var scheduler = BotScheduler.init(allocator);
    defer scheduler.deinit();
    _ = try scheduler.addEntry("entry-1", "bot-1", "Test Entry", BotSchedule.daily(9, 0));
    try std.testing.expectEqual(@as(usize, 1), scheduler.getEntryCount());
    try std.testing.expect(scheduler.canStartBot());
}

test "ScheduleEntry success rate" {
    const allocator = std.testing.allocator;
    var entry = try ScheduleEntry.init(allocator, "entry-1", "bot-1", "Test", BotSchedule.daily(9, 0));
    defer entry.deinit();
    entry.recordRun(true);
    entry.recordRun(true);
    entry.recordRun(false);
    try std.testing.expectEqual(@as(f32, 2.0 / 3.0), entry.getSuccessRate());
}

test "BotSchedule isActive" {
    var schedule = BotSchedule.daily(9, 0);
    try std.testing.expect(schedule.isActive(std.time.timestamp()));
    schedule.enabled = false;
    try std.testing.expect(!schedule.isActive(std.time.timestamp()));
}

test "EntryStatus pause resume" {
    const allocator = std.testing.allocator;
    var entry = try ScheduleEntry.init(allocator, "entry-1", "bot-1", "Test", BotSchedule.daily(9, 0));
    defer entry.deinit();
    try std.testing.expectEqual(EntryStatus.ACTIVE, entry.status);
    entry.pause();
    try std.testing.expectEqual(EntryStatus.PAUSED, entry.status);
    entry.resume_();
    try std.testing.expectEqual(EntryStatus.ACTIVE, entry.status);
}

