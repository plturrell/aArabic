// Mojo Runtime Memory Types
// String, List, Dict, and Set runtime implementations
//
// These are the actual runtime representations of Mojo's built-in collection types.
// They are reference-counted and managed by the runtime core.

const std = @import("std");
const core = @import("core.zig");

// ============================================================================
// MojoString - Dynamic String with Small String Optimization (SSO)
// ============================================================================

pub const MojoString = extern struct {
    /// Union of small string buffer or heap pointer
    data: Data,

    /// Length of the string (not including null terminator)
    len: u32,

    /// Capacity (for heap strings) or SSO flag
    cap_or_sso: u32,

    const SSO_CAPACITY = 23; // Fits in 24 bytes with null terminator
    const SSO_FLAG: u32 = 0x80000000;

    const Data = extern union {
        /// Small string stored inline (SSO)
        small: [SSO_CAPACITY + 1]u8,

        /// Heap-allocated string
        heap: HeapData,
    };

    const HeapData = extern struct {
        ptr: [*]u8,
        _padding: [SSO_CAPACITY + 1 - @sizeOf([*]u8)]u8,
    };

    /// Create an empty string
    pub fn empty() MojoString {
        var s = MojoString{
            .data = undefined,
            .len = 0,
            .cap_or_sso = SSO_FLAG,
        };
        s.data.small[0] = 0;
        return s;
    }

    /// Create from a slice
    pub fn fromSlice(slice: []const u8) !MojoString {
        var s = MojoString{
            .data = undefined,
            .len = @intCast(slice.len),
            .cap_or_sso = 0,
        };

        if (slice.len <= SSO_CAPACITY) {
            // Use SSO
            s.cap_or_sso = SSO_FLAG;
            @memcpy(s.data.small[0..slice.len], slice);
            s.data.small[slice.len] = 0;
        } else {
            // Allocate on heap
            const allocator = core.getAllocator();
            const buf = try allocator.alloc(u8, slice.len + 1);
            @memcpy(buf[0..slice.len], slice);
            buf[slice.len] = 0;
            s.data.heap.ptr = buf.ptr;
            s.cap_or_sso = @intCast(slice.len + 1);
        }

        return s;
    }

    /// Check if using SSO
    pub fn isSSO(self: *const MojoString) bool {
        return (self.cap_or_sso & SSO_FLAG) != 0;
    }

    /// Get the string as a slice
    pub fn asSlice(self: *const MojoString) []const u8 {
        if (self.isSSO()) {
            return self.data.small[0..self.len];
        } else {
            return self.data.heap.ptr[0..self.len];
        }
    }

    /// Get mutable pointer to data
    pub fn dataPtr(self: *MojoString) [*]u8 {
        if (self.isSSO()) {
            return &self.data.small;
        } else {
            return self.data.heap.ptr;
        }
    }

    /// Get capacity
    pub fn capacity(self: *const MojoString) usize {
        if (self.isSSO()) {
            return SSO_CAPACITY;
        } else {
            return self.cap_or_sso;
        }
    }

    /// Append another string
    pub fn append(self: *MojoString, other: *const MojoString) !void {
        try self.appendSlice(other.asSlice());
    }

    /// Append a slice
    pub fn appendSlice(self: *MojoString, slice: []const u8) !void {
        const new_len = self.len + @as(u32, @intCast(slice.len));

        if (self.isSSO() and new_len <= SSO_CAPACITY) {
            // Still fits in SSO
            @memcpy(self.data.small[self.len..][0..slice.len], slice);
            self.data.small[new_len] = 0;
            self.len = new_len;
        } else {
            // Need heap allocation
            try self.growTo(new_len + 1);
            const ptr = self.data.heap.ptr;
            @memcpy(ptr[self.len..][0..slice.len], slice);
            ptr[new_len] = 0;
            self.len = new_len;
        }
    }

    /// Grow to at least the given capacity
    fn growTo(self: *MojoString, min_cap: usize) !void {
        if (self.capacity() >= min_cap) return;

        const allocator = core.getAllocator();
        const new_cap = @max(min_cap, self.capacity() * 2);

        const new_buf = try allocator.alloc(u8, new_cap);

        // Copy existing data
        const old_data = self.asSlice();
        @memcpy(new_buf[0..old_data.len], old_data);
        new_buf[old_data.len] = 0;

        // Free old heap buffer if not SSO
        if (!self.isSSO()) {
            allocator.free(u8, self.data.heap.ptr[0..self.cap_or_sso]);
        }

        self.data.heap.ptr = new_buf.ptr;
        self.cap_or_sso = @intCast(new_cap);
    }

    /// Free the string
    pub fn deinit(self: *MojoString) void {
        if (!self.isSSO()) {
            const allocator = core.getAllocator();
            allocator.free(u8, self.data.heap.ptr[0..self.cap_or_sso]);
        }
        self.* = empty();
    }

    /// Clone the string
    pub fn clone(self: *const MojoString) !MojoString {
        return fromSlice(self.asSlice());
    }

    /// Compare two strings
    pub fn eql(self: *const MojoString, other: *const MojoString) bool {
        return std.mem.eql(u8, self.asSlice(), other.asSlice());
    }

    /// Hash the string
    pub fn hash(self: *const MojoString) u64 {
        return std.hash.Wyhash.hash(0, self.asSlice());
    }
};

// ============================================================================
// MojoList - Dynamic Array
// ============================================================================

pub fn MojoList(comptime T: type) type {
    return extern struct {
        const Self = @This();

        /// Pointer to data
        data: ?[*]T,

        /// Number of elements
        len: u32,

        /// Capacity
        cap: u32,

        /// Type ID for runtime type info
        type_id: u32,

        const INITIAL_CAPACITY = 8;

        /// Create an empty list
        pub fn empty() Self {
            return .{
                .data = null,
                .len = 0,
                .cap = 0,
                .type_id = @intFromEnum(core.BuiltinTypeId.List),
            };
        }

        /// Create with initial capacity
        pub fn withCapacity(cap: usize) !Self {
            var list = empty();
            try list.reserve(cap);
            return list;
        }

        /// Get length
        pub fn length(self: *const Self) usize {
            return self.len;
        }

        /// Check if empty
        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        /// Get element at index
        pub fn get(self: *const Self, index: usize) ?T {
            if (index >= self.len) return null;
            return self.data.?[index];
        }

        /// Set element at index
        pub fn set(self: *Self, index: usize, value: T) bool {
            if (index >= self.len) return false;
            self.data.?[index] = value;
            return true;
        }

        /// Get as slice
        pub fn asSlice(self: *const Self) []const T {
            if (self.data) |d| {
                return d[0..self.len];
            }
            return &[_]T{};
        }

        /// Get as mutable slice
        pub fn asMutSlice(self: *Self) []T {
            if (self.data) |d| {
                return d[0..self.len];
            }
            return &[_]T{};
        }

        /// Append an element
        pub fn append(self: *Self, value: T) !void {
            try self.ensureCapacity(self.len + 1);
            self.data.?[self.len] = value;
            self.len += 1;
        }

        /// Insert at index
        pub fn insert(self: *Self, index: usize, value: T) !void {
            if (index > self.len) return error.IndexOutOfBounds;

            try self.ensureCapacity(self.len + 1);

            // Shift elements right
            if (index < self.len) {
                const src = self.data.?[index..self.len];
                const dst = self.data.?[index + 1 .. self.len + 1];
                std.mem.copyBackwards(T, dst, src);
            }

            self.data.?[index] = value;
            self.len += 1;
        }

        /// Remove at index and return value
        pub fn remove(self: *Self, index: usize) ?T {
            if (index >= self.len) return null;

            const value = self.data.?[index];

            // Shift elements left
            if (index < self.len - 1) {
                const src = self.data.?[index + 1 .. self.len];
                const dst = self.data.?[index .. self.len - 1];
                @memcpy(dst, src);
            }

            self.len -= 1;
            return value;
        }

        /// Remove last element and return it
        pub fn pop(self: *Self) ?T {
            if (self.len == 0) return null;
            self.len -= 1;
            return self.data.?[self.len];
        }

        /// Clear all elements
        pub fn clear(self: *Self) void {
            self.len = 0;
        }

        /// Reserve capacity
        pub fn reserve(self: *Self, min_cap: usize) !void {
            if (self.cap >= min_cap) return;

            const allocator = core.getAllocator();
            const new_cap = @max(min_cap, @max(INITIAL_CAPACITY, self.cap * 2));

            const new_data = try allocator.alloc(T, new_cap);

            // Copy existing data
            if (self.data) |old_data| {
                @memcpy(new_data[0..self.len], old_data[0..self.len]);
                allocator.free(T, old_data[0..self.cap]);
            }

            self.data = new_data.ptr;
            self.cap = @intCast(new_cap);
        }

        fn ensureCapacity(self: *Self, needed: usize) !void {
            if (self.cap >= needed) return;
            try self.reserve(needed);
        }

        /// Reverse the list in place
        pub fn reverse(self: *Self) void {
            if (self.len <= 1) return;
            std.mem.reverse(T, self.asMutSlice());
        }

        /// Sort the list (requires T to be comparable)
        pub fn sort(self: *Self) void {
            if (self.len <= 1) return;
            std.mem.sort(T, self.asMutSlice(), {}, std.sort.asc(T));
        }

        /// Find index of element
        pub fn indexOf(self: *const Self, value: T) ?usize {
            for (self.asSlice(), 0..) |item, i| {
                if (item == value) return i;
            }
            return null;
        }

        /// Check if contains element
        pub fn contains(self: *const Self, value: T) bool {
            return self.indexOf(value) != null;
        }

        /// Free the list
        pub fn deinit(self: *Self) void {
            if (self.data) |data| {
                const allocator = core.getAllocator();
                allocator.free(T, data[0..self.cap]);
            }
            self.* = empty();
        }

        /// Clone the list
        pub fn clone(self: *const Self) !Self {
            var new_list = try withCapacity(self.len);
            if (self.data) |data| {
                @memcpy(new_list.data.?[0..self.len], data[0..self.len]);
            }
            new_list.len = self.len;
            return new_list;
        }
    };
}

// ============================================================================
// MojoDict - Hash Map
// ============================================================================

pub fn MojoDict(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        /// Internal hash map
        map: std.AutoHashMap(K, V),

        /// Type ID
        type_id: u32,

        /// Create an empty dict
        pub fn empty() Self {
            return .{
                .map = std.AutoHashMap(K, V).init(std.heap.page_allocator),
                .type_id = @intFromEnum(core.BuiltinTypeId.Dict),
            };
        }

        /// Get length
        pub fn length(self: *const Self) usize {
            return self.map.count();
        }

        /// Check if empty
        pub fn isEmpty(self: *const Self) bool {
            return self.map.count() == 0;
        }

        /// Get value for key
        pub fn get(self: *const Self, key: K) ?V {
            return self.map.get(key);
        }

        /// Set value for key
        pub fn set(self: *Self, key: K, value: V) !void {
            try self.map.put(key, value);
        }

        /// Remove key and return value
        pub fn remove(self: *Self, key: K) ?V {
            return self.map.fetchRemove(key) orelse return null;
        }

        /// Check if contains key
        pub fn containsKey(self: *const Self, key: K) bool {
            return self.map.contains(key);
        }

        /// Clear all entries
        pub fn clear(self: *Self) void {
            self.map.clearRetainingCapacity();
        }

        /// Get keys iterator
        pub fn keys(self: *const Self) std.AutoHashMap(K, V).KeyIterator {
            return self.map.keyIterator();
        }

        /// Get values iterator
        pub fn values(self: *const Self) std.AutoHashMap(K, V).ValueIterator {
            return self.map.valueIterator();
        }

        /// Free the dict
        pub fn deinit(self: *Self) void {
            self.map.deinit();
        }
    };
}

// ============================================================================
// MojoSet - Hash Set
// ============================================================================

pub fn MojoSet(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Internal hash map (value is void)
        map: std.AutoHashMap(T, void),

        /// Type ID
        type_id: u32,

        /// Create an empty set
        pub fn empty() Self {
            return .{
                .map = std.AutoHashMap(T, void).init(std.heap.page_allocator),
                .type_id = @intFromEnum(core.BuiltinTypeId.Set),
            };
        }

        /// Get length
        pub fn length(self: *const Self) usize {
            return self.map.count();
        }

        /// Check if empty
        pub fn isEmpty(self: *const Self) bool {
            return self.map.count() == 0;
        }

        /// Add element
        pub fn add(self: *Self, value: T) !void {
            try self.map.put(value, {});
        }

        /// Remove element
        pub fn remove(self: *Self, value: T) bool {
            return self.map.remove(value);
        }

        /// Discard element (remove if exists, no error if not)
        pub fn discard(self: *Self, value: T) void {
            _ = self.map.remove(value);
        }

        /// Check if contains element
        pub fn contains(self: *const Self, value: T) bool {
            return self.map.contains(value);
        }

        /// Clear all elements
        pub fn clear(self: *Self) void {
            self.map.clearRetainingCapacity();
        }

        /// Union with another set
        pub fn unionWith(self: *Self, other: *const Self) !void {
            var it = other.map.keyIterator();
            while (it.next()) |key| {
                try self.add(key.*);
            }
        }

        /// Intersection with another set
        pub fn intersectionWith(self: *Self, other: *const Self) void {
            var to_remove = std.ArrayList(T).init(std.heap.page_allocator);
            defer to_remove.deinit();

            var it = self.map.keyIterator();
            while (it.next()) |key| {
                if (!other.contains(key.*)) {
                    to_remove.append(key.*) catch continue;
                }
            }

            for (to_remove.items) |key| {
                _ = self.map.remove(key);
            }
        }

        /// Difference with another set (self - other)
        pub fn differenceWith(self: *Self, other: *const Self) void {
            var it = other.map.keyIterator();
            while (it.next()) |key| {
                _ = self.map.remove(key.*);
            }
        }

        /// Check if subset of another set
        pub fn isSubsetOf(self: *const Self, other: *const Self) bool {
            var it = self.map.keyIterator();
            while (it.next()) |key| {
                if (!other.contains(key.*)) return false;
            }
            return true;
        }

        /// Check if superset of another set
        pub fn isSupersetOf(self: *const Self, other: *const Self) bool {
            return other.isSubsetOf(self);
        }

        /// Get iterator
        pub fn iterator(self: *const Self) std.AutoHashMap(T, void).KeyIterator {
            return self.map.keyIterator();
        }

        /// Free the set
        pub fn deinit(self: *Self) void {
            self.map.deinit();
        }
    };
}

// ============================================================================
// C ABI Exports for String
// ============================================================================

export fn mojo_string_new() callconv(.c) ?*MojoString {
    const allocator = core.getAllocator();
    const str = allocator.backing.create(MojoString) catch return null;
    str.* = MojoString.empty();
    return str;
}

export fn mojo_string_from_cstr(cstr: [*:0]const u8) callconv(.c) ?*MojoString {
    const allocator = core.getAllocator();
    const str = allocator.backing.create(MojoString) catch return null;
    const len = std.mem.len(cstr);
    str.* = MojoString.fromSlice(cstr[0..len]) catch {
        allocator.backing.destroy(str);
        return null;
    };
    return str;
}

export fn mojo_string_len(str: *const MojoString) callconv(.c) u32 {
    return str.len;
}

export fn mojo_string_cstr(str: *const MojoString) callconv(.c) [*:0]const u8 {
    const slice = str.asSlice();
    if (str.isSSO()) {
        return @ptrCast(&str.data.small);
    } else {
        return @ptrCast(str.data.heap.ptr);
    }
    _ = slice;
}

export fn mojo_string_free(str: *MojoString) callconv(.c) void {
    str.deinit();
    const allocator = core.getAllocator();
    allocator.backing.destroy(str);
}

// ============================================================================
// C ABI Exports for List (i64 specialization)
// ============================================================================

const IntList = MojoList(i64);

export fn mojo_list_int_new() callconv(.c) ?*IntList {
    const allocator = core.getAllocator();
    const list = allocator.backing.create(IntList) catch return null;
    list.* = IntList.empty();
    return list;
}

export fn mojo_list_int_append(list: *IntList, value: i64) callconv(.c) i32 {
    list.append(value) catch return -1;
    return 0;
}

export fn mojo_list_int_get(list: *const IntList, index: u32) callconv(.c) i64 {
    return list.get(index) orelse 0;
}

export fn mojo_list_int_len(list: *const IntList) callconv(.c) u32 {
    return list.len;
}

export fn mojo_list_int_free(list: *IntList) callconv(.c) void {
    list.deinit();
    const allocator = core.getAllocator();
    allocator.backing.destroy(list);
}

// ============================================================================
// Tests
// ============================================================================

test "MojoString SSO" {
    try core.initDefault();
    defer core.deinit();

    var s = try MojoString.fromSlice("Hello");
    defer s.deinit();

    try std.testing.expect(s.isSSO());
    try std.testing.expectEqualStrings("Hello", s.asSlice());
}

test "MojoString heap" {
    try core.initDefault();
    defer core.deinit();

    const long_str = "This is a very long string that exceeds SSO capacity";
    var s = try MojoString.fromSlice(long_str);
    defer s.deinit();

    try std.testing.expect(!s.isSSO());
    try std.testing.expectEqualStrings(long_str, s.asSlice());
}

test "MojoString append" {
    try core.initDefault();
    defer core.deinit();

    var s = try MojoString.fromSlice("Hello");
    defer s.deinit();

    try s.appendSlice(" World");
    try std.testing.expectEqualStrings("Hello World", s.asSlice());
}

test "MojoList basic" {
    try core.initDefault();
    defer core.deinit();

    var list = MojoList(i32).empty();
    defer list.deinit();

    try list.append(1);
    try list.append(2);
    try list.append(3);

    try std.testing.expect(list.length() == 3);
    try std.testing.expect(list.get(0).? == 1);
    try std.testing.expect(list.get(1).? == 2);
    try std.testing.expect(list.get(2).? == 3);
}

test "MojoList operations" {
    try core.initDefault();
    defer core.deinit();

    var list = MojoList(i32).empty();
    defer list.deinit();

    try list.append(3);
    try list.append(1);
    try list.append(2);

    list.sort();
    try std.testing.expect(list.get(0).? == 1);
    try std.testing.expect(list.get(1).? == 2);
    try std.testing.expect(list.get(2).? == 3);

    list.reverse();
    try std.testing.expect(list.get(0).? == 3);
}

test "MojoDict basic" {
    try core.initDefault();
    defer core.deinit();

    var dict = MojoDict(i32, []const u8).empty();
    defer dict.deinit();

    try dict.set(1, "one");
    try dict.set(2, "two");

    try std.testing.expect(dict.length() == 2);
    try std.testing.expectEqualStrings("one", dict.get(1).?);
}

test "MojoSet basic" {
    try core.initDefault();
    defer core.deinit();

    var set = MojoSet(i32).empty();
    defer set.deinit();

    try set.add(1);
    try set.add(2);
    try set.add(1); // Duplicate

    try std.testing.expect(set.length() == 2);
    try std.testing.expect(set.contains(1));
    try std.testing.expect(!set.contains(3));
}

test "MojoSet operations" {
    try core.initDefault();
    defer core.deinit();

    var set1 = MojoSet(i32).empty();
    defer set1.deinit();
    try set1.add(1);
    try set1.add(2);

    var set2 = MojoSet(i32).empty();
    defer set2.deinit();
    try set2.add(2);
    try set2.add(3);

    // Union
    var union_set = MojoSet(i32).empty();
    defer union_set.deinit();
    try union_set.add(1);
    try union_set.add(2);
    try union_set.unionWith(&set2);
    try std.testing.expect(union_set.length() == 3);
}
