// Day 26: Enhanced State Management - Versioning & Migration
// State versioning, migration framework, and advanced recovery

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

/// State version for tracking schema evolution
pub const StateVersion = struct {
    major: u32,
    minor: u32,
    patch: u32,

    pub fn init(major: u32, minor: u32, patch: u32) StateVersion {
        return .{
            .major = major,
            .minor = minor,
            .patch = patch,
        };
    }

    pub fn toString(self: StateVersion, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "{d}.{d}.{d}", .{ self.major, self.minor, self.patch });
    }

    pub fn fromString(allocator: Allocator, version_str: []const u8) !StateVersion {
        _ = allocator;
        var iter = std.mem.splitScalar(u8, version_str, '.');
        
        const major_str = iter.next() orelse return error.InvalidVersion;
        const minor_str = iter.next() orelse return error.InvalidVersion;
        const patch_str = iter.next() orelse return error.InvalidVersion;

        return .{
            .major = try std.fmt.parseInt(u32, major_str, 10),
            .minor = try std.fmt.parseInt(u32, minor_str, 10),
            .patch = try std.fmt.parseInt(u32, patch_str, 10),
        };
    }

    pub fn isCompatible(self: StateVersion, other: StateVersion) bool {
        // Compatible if same major version
        return self.major == other.major;
    }

    pub fn compare(self: StateVersion, other: StateVersion) std.math.Order {
        if (self.major != other.major) {
            return std.math.order(self.major, other.major);
        }
        if (self.minor != other.minor) {
            return std.math.order(self.minor, other.minor);
        }
        return std.math.order(self.patch, other.patch);
    }
};

/// Versioned state with metadata
pub const VersionedState = struct {
    version: StateVersion,
    workflow_id: []const u8,
    execution_id: []const u8,
    state_data: []const u8,
    timestamp: i64,
    checksum: u64,
    metadata: StringHashMap([]const u8),

    pub fn init(
        allocator: Allocator,
        version: StateVersion,
        workflow_id: []const u8,
        execution_id: []const u8,
        state_data: []const u8,
    ) !VersionedState {
        const checksum = std.hash.Wyhash.hash(0, state_data);
        
        return .{
            .version = version,
            .workflow_id = try allocator.dupe(u8, workflow_id),
            .execution_id = try allocator.dupe(u8, execution_id),
            .state_data = try allocator.dupe(u8, state_data),
            .timestamp = std.time.timestamp(),
            .checksum = checksum,
            .metadata = StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *VersionedState, allocator: Allocator) void {
        allocator.free(self.workflow_id);
        allocator.free(self.execution_id);
        allocator.free(self.state_data);
        
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn validateChecksum(self: *const VersionedState) bool {
        const computed = std.hash.Wyhash.hash(0, self.state_data);
        return computed == self.checksum;
    }

    pub fn addMetadata(self: *VersionedState, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);
        
        const value_copy = try allocator.dupe(u8, value);
        errdefer allocator.free(value_copy);
        
        try self.metadata.put(key_copy, value_copy);
    }

    pub fn serialize(self: *const VersionedState, allocator: Allocator) ![]const u8 {
        const version_str = try self.version.toString(allocator);
        defer allocator.free(version_str);

        return std.fmt.allocPrint(
            allocator,
            "{{\"version\":\"{s}\",\"workflow_id\":\"{s}\",\"execution_id\":\"{s}\",\"timestamp\":{d},\"checksum\":{d}}}",
            .{ version_str, self.workflow_id, self.execution_id, self.timestamp, self.checksum },
        );
    }
};

/// Migration strategy for state schema changes
pub const MigrationStrategy = union(enum) {
    none: void,
    transform: *const fn (Allocator, []const u8) anyerror![]const u8,
    rebuild: *const fn (Allocator, []const u8) anyerror![]const u8,
    custom: struct {
        migrate_fn: *const fn (Allocator, []const u8, StateVersion, StateVersion) anyerror![]const u8,
    },

    pub fn apply(
        self: MigrationStrategy,
        allocator: Allocator,
        data: []const u8,
        from_version: StateVersion,
        to_version: StateVersion,
    ) ![]const u8 {
        return switch (self) {
            .none => try allocator.dupe(u8, data),
            .transform => |fn_ptr| try fn_ptr(allocator, data),
            .rebuild => |fn_ptr| try fn_ptr(allocator, data),
            .custom => |custom| try custom.migrate_fn(allocator, data, from_version, to_version),
        };
    }
};

/// State migration record
pub const MigrationRecord = struct {
    from_version: StateVersion,
    to_version: StateVersion,
    strategy: MigrationStrategy,
    applied_at: i64,
    success: bool,
    error_message: ?[]const u8,

    pub fn init(
        from_version: StateVersion,
        to_version: StateVersion,
        strategy: MigrationStrategy,
    ) MigrationRecord {
        return .{
            .from_version = from_version,
            .to_version = to_version,
            .strategy = strategy,
            .applied_at = std.time.timestamp(),
            .success = false,
            .error_message = null,
        };
    }

    pub fn markSuccess(self: *MigrationRecord) void {
        self.success = true;
        self.error_message = null;
    }

    pub fn markFailure(self: *MigrationRecord, allocator: Allocator, error_msg: []const u8) !void {
        self.success = false;
        self.error_message = try allocator.dupe(u8, error_msg);
    }

    pub fn deinit(self: *MigrationRecord, allocator: Allocator) void {
        if (self.error_message) |msg| {
            allocator.free(msg);
        }
    }
};

/// State migrator for handling version upgrades
pub const StateMigrator = struct {
    allocator: Allocator,
    migrations: ArrayList(MigrationPath),
    current_version: StateVersion,

    const MigrationPath = struct {
        from_version: StateVersion,
        to_version: StateVersion,
        strategy: MigrationStrategy,
    };

    pub fn init(allocator: Allocator, current_version: StateVersion) StateMigrator {
        return .{
            .allocator = allocator,
            .migrations = ArrayList(MigrationPath){},
            .current_version = current_version,
        };
    }

    pub fn deinit(self: *StateMigrator) void {
        self.migrations.deinit(self.allocator);
    }

    pub fn registerMigration(
        self: *StateMigrator,
        from_version: StateVersion,
        to_version: StateVersion,
        strategy: MigrationStrategy,
    ) !void {
        try self.migrations.append(self.allocator, .{
            .from_version = from_version,
            .to_version = to_version,
            .strategy = strategy,
        });
    }

    pub fn findMigrationPath(
        self: *const StateMigrator,
        from_version: StateVersion,
        to_version: StateVersion,
    ) !?ArrayList(MigrationPath) {
        // Simple direct path lookup (can be enhanced with graph search)
        var path = ArrayList(MigrationPath){};
        errdefer path.deinit(self.allocator);

        for (self.migrations.items) |migration| {
            if (migration.from_version.major == from_version.major and
                migration.from_version.minor == from_version.minor and
                migration.to_version.major == to_version.major and
                migration.to_version.minor == to_version.minor)
            {
                try path.append(self.allocator, migration);
                return path;
            }
        }

        return null;
    }

    pub fn migrate(
        self: *StateMigrator,
        state: *VersionedState,
        target_version: StateVersion,
    ) !MigrationRecord {
        var record = MigrationRecord.init(state.version, target_version, .{ .none = {} });

        // Check if migration is needed
        if (state.version.major == target_version.major and
            state.version.minor == target_version.minor and
            state.version.patch == target_version.patch)
        {
            record.markSuccess();
            return record;
        }

        // Find migration path
        const path = try self.findMigrationPath(state.version, target_version);
        if (path == null) {
            try record.markFailure(self.allocator, "No migration path found");
            return record;
        }
        defer path.?.deinit();

        // Apply migrations
        var current_data = try self.allocator.dupe(u8, state.state_data);
        defer self.allocator.free(current_data);

        for (path.?.items) |migration| {
            const migrated_data = migration.strategy.apply(
                self.allocator,
                current_data,
                migration.from_version,
                migration.to_version,
            ) catch |err| {
                const err_msg = try std.fmt.allocPrint(self.allocator, "Migration failed: {}", .{err});
                defer self.allocator.free(err_msg);
                try record.markFailure(self.allocator, err_msg);
                return record;
            };
            
            self.allocator.free(current_data);
            current_data = migrated_data;
        }

        // Update state with migrated data
        self.allocator.free(state.state_data);
        state.state_data = try self.allocator.dupe(u8, current_data);
        state.version = target_version;
        state.checksum = std.hash.Wyhash.hash(0, state.state_data);
        state.timestamp = std.time.timestamp();

        record.markSuccess();
        return record;
    }
};

/// Advanced recovery strategy
pub const RecoveryStrategy = enum {
    immediate, // Recover immediately from last checkpoint
    delayed, // Wait before recovery (e.g., check if transient failure)
    conditional, // Recover based on conditions
    manual, // Require manual intervention
    automatic_retry, // Retry with exponential backoff
};

/// Recovery configuration
pub const RecoveryConfig = struct {
    strategy: RecoveryStrategy,
    max_retries: u32,
    retry_delay_ms: u64,
    checkpoint_interval: u64, // milliseconds
    enable_auto_checkpoints: bool,

    pub fn default() RecoveryConfig {
        return .{
            .strategy = .immediate,
            .max_retries = 3,
            .retry_delay_ms = 1000,
            .checkpoint_interval = 60000, // 1 minute
            .enable_auto_checkpoints = true,
        };
    }

    pub fn withStrategy(strategy: RecoveryStrategy) RecoveryConfig {
        var config = RecoveryConfig.default();
        config.strategy = strategy;
        return config;
    }
};

/// Recovery point with enhanced metadata
pub const EnhancedRecoveryPoint = struct {
    id: []const u8,
    workflow_id: []const u8,
    execution_id: []const u8,
    versioned_state: VersionedState,
    recovery_config: RecoveryConfig,
    created_at: i64,
    tags: ArrayList([]const u8),

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        workflow_id: []const u8,
        execution_id: []const u8,
        versioned_state: VersionedState,
        recovery_config: RecoveryConfig,
    ) !EnhancedRecoveryPoint {
        return .{
            .id = try allocator.dupe(u8, id),
            .workflow_id = try allocator.dupe(u8, workflow_id),
            .execution_id = try allocator.dupe(u8, execution_id),
            .versioned_state = versioned_state,
            .recovery_config = recovery_config,
            .created_at = std.time.timestamp(),
            .tags = ArrayList([]const u8){},
        };
    }

    pub fn deinit(self: *EnhancedRecoveryPoint, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.workflow_id);
        allocator.free(self.execution_id);
        
        for (self.tags.items) |tag| {
            allocator.free(tag);
        }
        self.tags.deinit(allocator);
    }

    pub fn addTag(self: *EnhancedRecoveryPoint, allocator: Allocator, tag: []const u8) !void {
        const tag_copy = try allocator.dupe(u8, tag);
        try self.tags.append(allocator, tag_copy);
    }

    pub fn hasTag(self: *const EnhancedRecoveryPoint, tag: []const u8) bool {
        for (self.tags.items) |t| {
            if (std.mem.eql(u8, t, tag)) return true;
        }
        return false;
    }
};

/// State snapshot manager with versioning
pub const StateSnapshotManager = struct {
    allocator: Allocator,
    snapshots: StringHashMap(ArrayList(VersionedState)),
    max_snapshots_per_workflow: usize,

    pub fn init(allocator: Allocator, max_snapshots: usize) StateSnapshotManager {
        return .{
            .allocator = allocator,
            .snapshots = StringHashMap(ArrayList(VersionedState)).init(allocator),
            .max_snapshots_per_workflow = max_snapshots,
        };
    }

    pub fn deinit(self: *StateSnapshotManager) void {
        var iter = self.snapshots.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.items) |*state| {
                state.deinit(self.allocator);
            }
            entry.value_ptr.deinit(self.allocator);
        }
        self.snapshots.deinit();
    }

    pub fn saveSnapshot(self: *StateSnapshotManager, state: VersionedState) !void {
        const key = try std.fmt.allocPrint(
            self.allocator,
            "{s}:{s}",
            .{ state.workflow_id, state.execution_id },
        );
        errdefer self.allocator.free(key);

        const result = try self.snapshots.getOrPut(key);
        if (!result.found_existing) {
            result.key_ptr.* = key;
            result.value_ptr.* = ArrayList(VersionedState).empty;
        } else {
            self.allocator.free(key);
        }

        // Add snapshot
        try result.value_ptr.append(self.allocator, state);

        // Enforce max snapshots limit (FIFO)
        while (result.value_ptr.items.len > self.max_snapshots_per_workflow) {
            var oldest = result.value_ptr.orderedRemove(0);
            oldest.deinit(self.allocator);
        }
    }

    pub fn getLatestSnapshot(
        self: *const StateSnapshotManager,
        workflow_id: []const u8,
        execution_id: []const u8,
    ) !?VersionedState {
        const key = try std.fmt.allocPrint(
            self.allocator,
            "{s}:{s}",
            .{ workflow_id, execution_id },
        );
        defer self.allocator.free(key);

        if (self.snapshots.get(key)) |snapshots| {
            if (snapshots.items.len > 0) {
                return snapshots.items[snapshots.items.len - 1];
            }
        }
        return null;
    }

    pub fn getSnapshotHistory(
        self: *const StateSnapshotManager,
        workflow_id: []const u8,
        execution_id: []const u8,
    ) !?ArrayList(VersionedState) {
        const key = try std.fmt.allocPrint(
            self.allocator,
            "{s}:{s}",
            .{ workflow_id, execution_id },
        );
        defer self.allocator.free(key);

        if (self.snapshots.get(key)) |snapshots| {
            return snapshots;
        }
        return null;
    }

    pub fn deleteSnapshots(
        self: *StateSnapshotManager,
        workflow_id: []const u8,
        execution_id: []const u8,
    ) !void {
        const key = try std.fmt.allocPrint(
            self.allocator,
            "{s}:{s}",
            .{ workflow_id, execution_id },
        );
        defer self.allocator.free(key);

        if (self.snapshots.fetchRemove(key)) |kv| {
            self.allocator.free(kv.key);
            for (kv.value.items) |*state| {
                state.deinit(self.allocator);
            }
            kv.value.deinit(self.allocator);
        }
    }

    pub fn getStats(self: *const StateSnapshotManager) SnapshotStats {
        var total_snapshots: usize = 0;
        var iter = self.snapshots.valueIterator();
        while (iter.next()) |snapshots| {
            total_snapshots += snapshots.items.len;
        }

        return .{
            .workflow_count = self.snapshots.count(),
            .total_snapshots = total_snapshots,
            .max_per_workflow = self.max_snapshots_per_workflow,
        };
    }
};

pub const SnapshotStats = struct {
    workflow_count: usize,
    total_snapshots: usize,
    max_per_workflow: usize,
};

/// Performance optimization cache
pub const StateCache = struct {
    allocator: Allocator,
    cache: StringHashMap(CachedState),
    max_size: usize,
    hit_count: u64,
    miss_count: u64,

    const CachedState = struct {
        state: VersionedState,
        last_accessed: i64,
        access_count: u64,
    };

    pub fn init(allocator: Allocator, max_size: usize) StateCache {
        return .{
            .allocator = allocator,
            .cache = StringHashMap(CachedState).init(allocator),
            .max_size = max_size,
            .hit_count = 0,
            .miss_count = 0,
        };
    }

    pub fn deinit(self: *StateCache) void {
        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var mutable_state = entry.value_ptr.state;
            mutable_state.deinit(self.allocator);
        }
        self.cache.deinit();
    }

    pub fn put(self: *StateCache, key: []const u8, state: VersionedState) !void {
        // Evict if at capacity (LRU)
        if (self.cache.count() >= self.max_size) {
            try self.evictLRU();
        }

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        try self.cache.put(key_copy, .{
            .state = state,
            .last_accessed = std.time.timestamp(),
            .access_count = 0,
        });
    }

    pub fn get(self: *StateCache, key: []const u8) ?VersionedState {
        if (self.cache.getPtr(key)) |cached| {
            cached.last_accessed = std.time.timestamp();
            cached.access_count += 1;
            self.hit_count += 1;
            return cached.state;
        }
        self.miss_count += 1;
        return null;
    }

    fn evictLRU(self: *StateCache) !void {
        if (self.cache.count() == 0) return;

        var oldest_key: ?[]const u8 = null;
        var oldest_time: i64 = std.math.maxInt(i64);

        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.last_accessed < oldest_time) {
                oldest_time = entry.value_ptr.last_accessed;
                oldest_key = entry.key_ptr.*;
            }
        }

        if (oldest_key) |key| {
            if (self.cache.fetchRemove(key)) |kv| {
                self.allocator.free(kv.key);
                var mutable_state = kv.value.state;
                mutable_state.deinit(self.allocator);
            }
        }
    }

    pub fn getHitRate(self: *const StateCache) f64 {
        const total = self.hit_count + self.miss_count;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hit_count)) / @as(f64, @floatFromInt(total));
    }

    pub fn clear(self: *StateCache) void {
        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var mutable_state = entry.value_ptr.state;
            mutable_state.deinit(self.allocator);
        }
        self.cache.clearRetainingCapacity();
        self.hit_count = 0;
        self.miss_count = 0;
    }
};

// === Tests ===

test "StateVersion operations" {
    const allocator = std.testing.allocator;

    const v1 = StateVersion.init(1, 0, 0);
    const v2 = StateVersion.init(1, 1, 0);
    const v3 = StateVersion.init(2, 0, 0);

    // Compatible versions
    try std.testing.expect(v1.isCompatible(v2));
    try std.testing.expect(!v1.isCompatible(v3));

    // Version comparison
    try std.testing.expectEqual(std.math.Order.lt, v1.compare(v2));
    try std.testing.expectEqual(std.math.Order.lt, v2.compare(v3));

    // String conversion
    const v1_str = try v1.toString(allocator);
    defer allocator.free(v1_str);
    try std.testing.expectEqualStrings("1.0.0", v1_str);

    // Parse from string
    const parsed = try StateVersion.fromString(allocator, "1.2.3");
    try std.testing.expectEqual(@as(u32, 1), parsed.major);
    try std.testing.expectEqual(@as(u32, 2), parsed.minor);
    try std.testing.expectEqual(@as(u32, 3), parsed.patch);
}

test "VersionedState creation and validation" {
    const allocator = std.testing.allocator;

    const version = StateVersion.init(1, 0, 0);
    const data = "test state data";

    var state = try VersionedState.init(allocator, version, "wf123", "exec456", data);
    defer state.deinit(allocator);

    // Validate checksum
    try std.testing.expect(state.validateChecksum());

    // Add metadata
    try state.addMetadata(allocator, "author", "test");
    try std.testing.expectEqual(@as(usize, 1), state.metadata.count());

    // Serialize
    const serialized = try state.serialize(allocator);
    defer allocator.free(serialized);
    try std.testing.expect(serialized.len > 0);
}

test "StateMigrator registration and migration" {
    const allocator = std.testing.allocator;

    const v1 = StateVersion.init(1, 0, 0);
    const v2 = StateVersion.init(1, 1, 0);

    var migrator = StateMigrator.init(allocator, v2);
    defer migrator.deinit();

    // Register migration
    const strategy = MigrationStrategy{ .none = {} };
    try migrator.registerMigration(v1, v2, strategy);

    try std.testing.expectEqual(@as(usize, 1), migrator.migrations.items.len);
}

test "RecoveryConfig creation" {
    const config = RecoveryConfig.default();
    try std.testing.expectEqual(RecoveryStrategy.immediate, config.strategy);
    try std.testing.expectEqual(@as(u32, 3), config.max_retries);

    const custom = RecoveryConfig.withStrategy(.manual);
    try std.testing.expectEqual(RecoveryStrategy.manual, custom.strategy);
}

test "EnhancedRecoveryPoint with tags" {
    const allocator = std.testing.allocator;

    const version = StateVersion.init(1, 0, 0);
    const data = "recovery data";
    const versioned_state = try VersionedState.init(allocator, version, "wf123", "exec456", data);
    
    const config = RecoveryConfig.default();
    var recovery_point = try EnhancedRecoveryPoint.init(
        allocator,
        "rp001",
        "wf123",
        "exec456",
        versioned_state,
        config,
    );
    defer {
        var mutable_state = recovery_point.versioned_state;
        mutable_state.deinit(allocator);
        recovery_point.deinit(allocator);
    }

    // Add tags
    try recovery_point.addTag(allocator, "stable");
    try recovery_point.addTag(allocator, "pre-api-call");

    try std.testing.expect(recovery_point.hasTag("stable"));
    try std.testing.expect(recovery_point.hasTag("pre-api-call"));
    try std.testing.expect(!recovery_point.hasTag("unstable"));
}

test "StateSnapshotManager operations" {
    const allocator = std.testing.allocator;

    var manager = StateSnapshotManager.init(allocator, 5);
    defer manager.deinit();

    const version = StateVersion.init(1, 0, 0);

    // Save snapshots
    for (0..3) |i| {
        const data = try std.fmt.allocPrint(allocator, "data_{d}", .{i});
        defer allocator.free(data);

        const state = try VersionedState.init(allocator, version, "wf123", "exec456", data);
        try manager.saveSnapshot(state);
    }

    // Get latest snapshot
    const latest = try manager.getLatestSnapshot("wf123", "exec456");
    try std.testing.expect(latest != null);

    // Check stats
    const stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.workflow_count);
    try std.testing.expectEqual(@as(usize, 3), stats.total_snapshots);
}

test "StateSnapshotManager max snapshots limit" {
    const allocator = std.testing.allocator;

    var manager = StateSnapshotManager.init(allocator, 2);
    defer manager.deinit();

    const version = StateVersion.init(1, 0, 0);

    // Save 3 snapshots (exceeds limit of 2)
    for (0..3) |i| {
        const data = try std.fmt.allocPrint(allocator, "data_{d}", .{i});
        defer allocator.free(data);

        const state = try VersionedState.init(allocator, version, "wf123", "exec456", data);
        try manager.saveSnapshot(state);
    }

    // Should only have 2 snapshots (oldest evicted)
    const history = try manager.getSnapshotHistory("wf123", "exec456");
    try std.testing.expect(history != null);
    try std.testing.expectEqual(@as(usize, 2), history.?.items.len);
}

test "StateCache operations" {
    const allocator = std.testing.allocator;

    var cache = StateCache.init(allocator, 10);
    defer cache.deinit();

    const version = StateVersion.init(1, 0, 0);
    const state = try VersionedState.init(allocator, version, "wf123", "exec456", "test data");

    // Put state in cache
    try cache.put("key1", state);

    // Get from cache (hit)
    const cached = cache.get("key1");
    try std.testing.expect(cached != null);
    try std.testing.expectEqual(@as(u64, 1), cache.hit_count);

    // Get non-existent key (miss)
    const missing = cache.get("key2");
    try std.testing.expect(missing == null);
    try std.testing.expectEqual(@as(u64, 1), cache.miss_count);

    // Hit rate
    const hit_rate = cache.getHitRate();
    try std.testing.expectEqual(@as(f64, 0.5), hit_rate);
}

test "StateCache LRU eviction" {
    const allocator = std.testing.allocator;

    var cache = StateCache.init(allocator, 2);
    defer cache.deinit();

    const version = StateVersion.init(1, 0, 0);

    // Fill cache to capacity
    const state1 = try VersionedState.init(allocator, version, "wf1", "exec1", "data1");
    try cache.put("key1", state1);

    const state2 = try VersionedState.init(allocator, version, "wf2", "exec2", "data2");
    try cache.put("key2", state2);

    // Access key2 to make key1 the LRU
    _ = cache.get("key2");

    // Add new item (should evict key1)
    const state3 = try VersionedState.init(allocator, version, "wf3", "exec3", "data3");
    try cache.put("key3", state3);

    // key2 and key3 should exist, key1 should be evicted
    try std.testing.expect(cache.get("key1") == null);
    try std.testing.expect(cache.get("key2") != null);
    try std.testing.expect(cache.get("key3") != null);
}

test "StateCache clear operation" {
    const allocator = std.testing.allocator;

    var cache = StateCache.init(allocator, 10);
    defer cache.deinit();

    const version = StateVersion.init(1, 0, 0);
    const state = try VersionedState.init(allocator, version, "wf123", "exec456", "test data");

    try cache.put("key1", state);
    try std.testing.expectEqual(@as(usize, 1), cache.cache.count());

    cache.clear();
    try std.testing.expectEqual(@as(usize, 0), cache.cache.count());
    try std.testing.expectEqual(@as(u64, 0), cache.hit_count);
    try std.testing.expectEqual(@as(u64, 0), cache.miss_count);
}
