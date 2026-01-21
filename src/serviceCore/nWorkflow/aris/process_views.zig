//! ARIS Process Views - Swim lanes, organizational charts, process hierarchies for ARIS-style visualization

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const SwimLane = struct {
    id: []const u8, name: []const u8, participant: []const u8,
    elements: std.ArrayListUnmanaged([]const u8), y_position: f32, height: f32, allocator: Allocator,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, participant: []const u8) !SwimLane {
        return .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .participant = try a.dupe(u8, participant),
            .elements = .{}, .y_position = 0.0, .height = 100.0, .allocator = a };
    }
    pub fn deinit(self: *SwimLane) void {
        self.allocator.free(self.id); self.allocator.free(self.name); self.allocator.free(self.participant);
        for (self.elements.items) |e| self.allocator.free(e); self.elements.deinit(self.allocator);
    }
    pub fn addElement(self: *SwimLane, elem_id: []const u8) !void {
        try self.elements.append(self.allocator, try self.allocator.dupe(u8, elem_id));
    }
    pub fn removeElement(self: *SwimLane, elem_id: []const u8) bool {
        for (self.elements.items, 0..) |e, i| if (std.mem.eql(u8, e, elem_id)) {
            self.allocator.free(e); _ = self.elements.orderedRemove(i); return true;
        };
        return false;
    }
};

pub const LaneOrientation = enum { horizontal, vertical };

pub const SwimLaneLayout = struct {
    lanes: std.ArrayListUnmanaged(SwimLane), orientation: LaneOrientation, allocator: Allocator,

    pub fn init(a: Allocator, orientation: LaneOrientation) SwimLaneLayout {
        return .{ .lanes = .{}, .orientation = orientation, .allocator = a };
    }
    pub fn deinit(self: *SwimLaneLayout) void {
        for (self.lanes.items) |*l| l.deinit(); self.lanes.deinit(self.allocator);
    }
    pub fn addLane(self: *SwimLaneLayout, lane: SwimLane) !void { try self.lanes.append(self.allocator, lane); }
    pub fn removeLane(self: *SwimLaneLayout, lane_id: []const u8) bool {
        for (self.lanes.items, 0..) |*l, i| if (std.mem.eql(u8, l.id, lane_id)) {
            l.deinit(); _ = self.lanes.orderedRemove(i); return true;
        };
        return false;
    }
    pub fn findLane(self: *const SwimLaneLayout, lane_id: []const u8) ?*SwimLane {
        for (self.lanes.items) |*l| if (std.mem.eql(u8, l.id, lane_id)) return l;
        return null;
    }
    pub fn assignElementToLane(self: *SwimLaneLayout, lane_id: []const u8, elem_id: []const u8) !bool {
        if (self.findLane(lane_id)) |l| { try l.addElement(elem_id); return true; }
        return false;
    }
    pub fn calculateLayout(self: *SwimLaneLayout) void {
        var pos: f32 = 0.0;
        for (self.lanes.items) |*l| { l.y_position = pos; if (l.height < 50.0) l.height = 100.0; pos += l.height + 10.0; }
    }
    pub fn toJson(self: *const SwimLaneLayout) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .{}; errdefer buf.deinit(self.allocator);
        try buf.appendSlice(self.allocator, "{\"orientation\":\"");
        try buf.appendSlice(self.allocator, if (self.orientation == .horizontal) "horizontal" else "vertical");
        try buf.appendSlice(self.allocator, "\",\"lanes\":[");
        for (self.lanes.items, 0..) |l, i| {
            if (i > 0) try buf.append(self.allocator, ',');
            var tmp: [32]u8 = undefined;
            try buf.appendSlice(self.allocator, "{\"id\":\""); try buf.appendSlice(self.allocator, l.id);
            try buf.appendSlice(self.allocator, "\",\"name\":\""); try buf.appendSlice(self.allocator, l.name);
            try buf.appendSlice(self.allocator, "\",\"participant\":\""); try buf.appendSlice(self.allocator, l.participant);
            try buf.appendSlice(self.allocator, "\",\"y_position\":"); try buf.appendSlice(self.allocator, std.fmt.bufPrint(&tmp, "{d:.2}", .{l.y_position}) catch "0");
            try buf.appendSlice(self.allocator, ",\"height\":"); try buf.appendSlice(self.allocator, std.fmt.bufPrint(&tmp, "{d:.2}", .{l.height}) catch "0");
            try buf.appendSlice(self.allocator, ",\"elements\":[");
            for (l.elements.items, 0..) |e, j| { if (j > 0) try buf.append(self.allocator, ','); try buf.append(self.allocator, '"'); try buf.appendSlice(self.allocator, e); try buf.append(self.allocator, '"'); }
            try buf.appendSlice(self.allocator, "]}");
        }
        try buf.appendSlice(self.allocator, "]}");
        return try buf.toOwnedSlice(self.allocator);
    }
};

pub const OrgUnitType = enum { company, division, department, team, role, person,
    pub fn toString(self: OrgUnitType) []const u8 {
        return switch (self) { .company => "company", .division => "division", .department => "department", .team => "team", .role => "role", .person => "person" };
    }
};

pub const OrgUnit = struct {
    id: []const u8, name: []const u8, parent_id: ?[]const u8, unit_type: OrgUnitType,
    children: std.ArrayListUnmanaged(*OrgUnit), allocator: Allocator,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, unit_type: OrgUnitType) !*OrgUnit {
        const u = try a.create(OrgUnit);
        u.* = .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .parent_id = null, .unit_type = unit_type, .children = .{}, .allocator = a };
        return u;
    }
    pub fn deinit(self: *OrgUnit) void {
        self.allocator.free(self.id); self.allocator.free(self.name);
        if (self.parent_id) |pid| self.allocator.free(pid);
        for (self.children.items) |c| c.deinit();
        self.children.deinit(self.allocator); self.allocator.destroy(self);
    }
    pub fn setParent(self: *OrgUnit, pid: []const u8) !void {
        if (self.parent_id) |old| self.allocator.free(old);
        self.parent_id = try self.allocator.dupe(u8, pid);
    }
    pub fn addChild(self: *OrgUnit, child: *OrgUnit) !void {
        try self.children.append(self.allocator, child); try child.setParent(self.id);
    }
};

pub const OrgChart = struct {
    root: ?*OrgUnit, units: std.StringHashMap(*OrgUnit), allocator: Allocator,

    pub fn init(a: Allocator) OrgChart { return .{ .root = null, .units = std.StringHashMap(*OrgUnit).init(a), .allocator = a }; }
    pub fn deinit(self: *OrgChart) void { if (self.root) |r| r.deinit(); self.units.deinit(); }
    pub fn addUnit(self: *OrgChart, unit: *OrgUnit) !void {
        try self.units.put(unit.id, unit);
        if (self.root == null and unit.parent_id == null) self.root = unit;
    }
    pub fn removeUnit(self: *OrgChart, uid: []const u8) bool {
        if (self.units.get(uid)) |u| {
            if (u.parent_id) |pid| if (self.units.get(pid)) |p| {
                for (p.children.items, 0..) |c, i| if (std.mem.eql(u8, c.id, uid)) { _ = p.children.orderedRemove(i); break; };
            };
            _ = self.units.remove(uid); if (self.root == u) self.root = null; u.deinit(); return true;
        }
        return false;
    }
    pub fn moveUnit(self: *OrgChart, uid: []const u8, new_pid: []const u8) !bool {
        const u = self.units.get(uid) orelse return false;
        const np = self.units.get(new_pid) orelse return false;
        if (u.parent_id) |old_pid| if (self.units.get(old_pid)) |op| {
            for (op.children.items, 0..) |c, i| if (std.mem.eql(u8, c.id, uid)) { _ = op.children.orderedRemove(i); break; };
        };
        try np.addChild(u); return true;
    }
    pub fn findPath(self: *const OrgChart, from_id: []const u8, to_id: []const u8) !?[]const []const u8 {
        const from = self.units.get(from_id) orelse return null;
        const to = self.units.get(to_id) orelse return null;
        var from_path: std.ArrayListUnmanaged([]const u8) = .{}; defer from_path.deinit(self.allocator);
        var to_path: std.ArrayListUnmanaged([]const u8) = .{}; defer to_path.deinit(self.allocator);
        var cur: ?*const OrgUnit = from;
        while (cur) |c| { try from_path.append(self.allocator, c.id); cur = if (c.parent_id) |pid| self.units.get(pid) else null; }
        cur = to;
        while (cur) |c| { try to_path.append(self.allocator, c.id); cur = if (c.parent_id) |pid| self.units.get(pid) else null; }
        var result: std.ArrayListUnmanaged([]const u8) = .{}; errdefer result.deinit(self.allocator);
        for (from_path.items) |id| try result.append(self.allocator, try self.allocator.dupe(u8, id));
        outer: for (from_path.items) |fid| for (to_path.items, 0..) |tid, ti| if (std.mem.eql(u8, fid, tid)) {
            var j: usize = ti; while (j > 0) { j -= 1; try result.append(self.allocator, try self.allocator.dupe(u8, to_path.items[j])); }
            break :outer;
        };
        return try result.toOwnedSlice(self.allocator);
    }
    pub fn toJson(self: *const OrgChart) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .{}; errdefer buf.deinit(self.allocator);
        try buf.appendSlice(self.allocator, "{\"units\":[");
        var iter = self.units.iterator(); var first = true;
        while (iter.next()) |e| {
            if (!first) try buf.append(self.allocator, ','); first = false;
            const u = e.value_ptr.*;
            try buf.appendSlice(self.allocator, "{\"id\":\""); try buf.appendSlice(self.allocator, u.id);
            try buf.appendSlice(self.allocator, "\",\"name\":\""); try buf.appendSlice(self.allocator, u.name);
            try buf.appendSlice(self.allocator, "\",\"type\":\""); try buf.appendSlice(self.allocator, u.unit_type.toString());
            try buf.appendSlice(self.allocator, "\",\"parent_id\":");
            if (u.parent_id) |pid| { try buf.append(self.allocator, '"'); try buf.appendSlice(self.allocator, pid); try buf.append(self.allocator, '"'); }
            else try buf.appendSlice(self.allocator, "null");
            try buf.append(self.allocator, '}');
        }
        try buf.appendSlice(self.allocator, "]}");
        return try buf.toOwnedSlice(self.allocator);
    }
};

pub const ProcessHierarchy = struct {
    id: []const u8, name: []const u8, parent_process_id: ?[]const u8,
    child_processes: std.ArrayListUnmanaged(*ProcessHierarchy), level: u32, allocator: Allocator,

    pub fn init(a: Allocator, id: []const u8, name: []const u8, level: u32) !*ProcessHierarchy {
        const ph = try a.create(ProcessHierarchy);
        ph.* = .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .parent_process_id = null, .child_processes = .{}, .level = level, .allocator = a };
        return ph;
    }
    pub fn deinit(self: *ProcessHierarchy) void {
        self.allocator.free(self.id); self.allocator.free(self.name);
        if (self.parent_process_id) |pid| self.allocator.free(pid);
        for (self.child_processes.items) |c| c.deinit();
        self.child_processes.deinit(self.allocator); self.allocator.destroy(self);
    }
    pub fn addChild(self: *ProcessHierarchy, child: *ProcessHierarchy) !void {
        if (child.parent_process_id) |old| child.allocator.free(old);
        child.parent_process_id = try child.allocator.dupe(u8, self.id);
        child.level = self.level + 1;
        try self.child_processes.append(self.allocator, child);
    }
    pub fn removeChild(self: *ProcessHierarchy, cid: []const u8) bool {
        for (self.child_processes.items, 0..) |c, i| if (std.mem.eql(u8, c.id, cid)) { _ = self.child_processes.orderedRemove(i); return true; };
        return false;
    }
    pub fn findDescendant(self: *ProcessHierarchy, pid: []const u8) ?*ProcessHierarchy {
        if (std.mem.eql(u8, self.id, pid)) return self;
        for (self.child_processes.items) |c| if (c.findDescendant(pid)) |f| return f;
        return null;
    }
    pub fn getDepth(self: *const ProcessHierarchy) u32 {
        var max: u32 = 0;
        for (self.child_processes.items) |c| { const d = c.getDepth() + 1; if (d > max) max = d; }
        return max;
    }
    pub fn toJson(self: *const ProcessHierarchy) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .{}; errdefer buf.deinit(self.allocator);
        try self.writeJson(&buf); return try buf.toOwnedSlice(self.allocator);
    }
    fn writeJson(self: *const ProcessHierarchy, buf: *std.ArrayListUnmanaged(u8)) !void {
        var tmp: [16]u8 = undefined;
        try buf.appendSlice(self.allocator, "{\"id\":\""); try buf.appendSlice(self.allocator, self.id);
        try buf.appendSlice(self.allocator, "\",\"name\":\""); try buf.appendSlice(self.allocator, self.name);
        try buf.appendSlice(self.allocator, "\",\"level\":"); try buf.appendSlice(self.allocator, std.fmt.bufPrint(&tmp, "{d}", .{self.level}) catch "0");
        try buf.appendSlice(self.allocator, ",\"children\":[");
        for (self.child_processes.items, 0..) |c, i| { if (i > 0) try buf.append(self.allocator, ','); try c.writeJson(buf); }
        try buf.appendSlice(self.allocator, "]}");
    }
};

pub const Activity = struct {
    id: []const u8, name: []const u8, description: []const u8, allocator: Allocator,
    pub fn init(a: Allocator, id: []const u8, name: []const u8, desc: []const u8) !Activity {
        return .{ .id = try a.dupe(u8, id), .name = try a.dupe(u8, name), .description = try a.dupe(u8, desc), .allocator = a };
    }
    pub fn deinit(self: *Activity) void {
        self.allocator.free(self.id); self.allocator.free(self.name); self.allocator.free(self.description);
    }
};

pub const ValueChain = struct {
    primary_activities: std.ArrayListUnmanaged(Activity), support_activities: std.ArrayListUnmanaged(Activity), allocator: Allocator,

    pub fn init(a: Allocator) ValueChain { return .{ .primary_activities = .{}, .support_activities = .{}, .allocator = a }; }
    pub fn deinit(self: *ValueChain) void {
        for (self.primary_activities.items) |*a| a.deinit(); self.primary_activities.deinit(self.allocator);
        for (self.support_activities.items) |*a| a.deinit(); self.support_activities.deinit(self.allocator);
    }
    pub fn addPrimaryActivity(self: *ValueChain, act: Activity) !void { try self.primary_activities.append(self.allocator, act); }
    pub fn addSupportActivity(self: *ValueChain, act: Activity) !void { try self.support_activities.append(self.allocator, act); }

    pub fn toEpc(self: *const ValueChain) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .{}; errdefer buf.deinit(self.allocator);
        try buf.appendSlice(self.allocator, "<Model id=\"value_chain\" name=\"ValueChain\">\n");
        var prev_id: ?[]const u8 = null;
        for (self.primary_activities.items, 0..) |act, i| {
            try buf.appendSlice(self.allocator, "  <Function id=\""); try buf.appendSlice(self.allocator, act.id);
            try buf.appendSlice(self.allocator, "\" name=\""); try buf.appendSlice(self.allocator, act.name);
            try buf.appendSlice(self.allocator, "\" category=\"primary\"/>\n");
            if (prev_id) |pid| {
                var tmp: [16]u8 = undefined;
                try buf.appendSlice(self.allocator, "  <Connection id=\"conn_p_");
                try buf.appendSlice(self.allocator, std.fmt.bufPrint(&tmp, "{d}", .{i}) catch "0");
                try buf.appendSlice(self.allocator, "\" source=\""); try buf.appendSlice(self.allocator, pid);
                try buf.appendSlice(self.allocator, "\" target=\""); try buf.appendSlice(self.allocator, act.id);
                try buf.appendSlice(self.allocator, "\"/>\n");
            }
            prev_id = act.id;
        }
        for (self.support_activities.items) |act| {
            try buf.appendSlice(self.allocator, "  <OrgUnit id=\""); try buf.appendSlice(self.allocator, act.id);
            try buf.appendSlice(self.allocator, "\" name=\""); try buf.appendSlice(self.allocator, act.name);
            try buf.appendSlice(self.allocator, "\" category=\"support\"/>\n");
        }
        try buf.appendSlice(self.allocator, "</Model>");
        return try buf.toOwnedSlice(self.allocator);
    }
    pub fn toJson(self: *const ValueChain) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .{}; errdefer buf.deinit(self.allocator);
        try buf.appendSlice(self.allocator, "{\"primary_activities\":[");
        for (self.primary_activities.items, 0..) |a, i| {
            if (i > 0) try buf.append(self.allocator, ',');
            try buf.appendSlice(self.allocator, "{\"id\":\""); try buf.appendSlice(self.allocator, a.id);
            try buf.appendSlice(self.allocator, "\",\"name\":\""); try buf.appendSlice(self.allocator, a.name);
            try buf.appendSlice(self.allocator, "\",\"description\":\""); try buf.appendSlice(self.allocator, a.description);
            try buf.appendSlice(self.allocator, "\"}");
        }
        try buf.appendSlice(self.allocator, "],\"support_activities\":[");
        for (self.support_activities.items, 0..) |a, i| {
            if (i > 0) try buf.append(self.allocator, ',');
            try buf.appendSlice(self.allocator, "{\"id\":\""); try buf.appendSlice(self.allocator, a.id);
            try buf.appendSlice(self.allocator, "\",\"name\":\""); try buf.appendSlice(self.allocator, a.name);
            try buf.appendSlice(self.allocator, "\",\"description\":\""); try buf.appendSlice(self.allocator, a.description);
            try buf.appendSlice(self.allocator, "\"}");
        }
        try buf.appendSlice(self.allocator, "]}");
        return try buf.toOwnedSlice(self.allocator);
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "SwimLane creation and element management" {
    const allocator = std.testing.allocator;
    var lane = try SwimLane.init(allocator, "lane1", "Sales", "Sales Team");
    defer lane.deinit();
    try std.testing.expectEqualStrings("lane1", lane.id);
    try lane.addElement("task1"); try lane.addElement("task2");
    try std.testing.expectEqual(@as(usize, 2), lane.elements.items.len);
    try std.testing.expect(lane.removeElement("task1"));
    try std.testing.expect(!lane.removeElement("nonexistent"));
}

test "SwimLaneLayout operations and JSON serialization" {
    const allocator = std.testing.allocator;
    var layout = SwimLaneLayout.init(allocator, .horizontal);
    defer layout.deinit();
    try layout.addLane(try SwimLane.init(allocator, "l1", "Customer", "Customer"));
    try layout.addLane(try SwimLane.init(allocator, "l2", "Support", "Support Team"));
    _ = try layout.assignElementToLane("l1", "elem1");
    layout.calculateLayout();
    try std.testing.expect(layout.lanes.items[1].y_position > 0);
    const json = try layout.toJson();
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"orientation\":\"horizontal\"") != null);
    try std.testing.expect(layout.removeLane("l1"));
}

test "OrgUnit and OrgChart hierarchy" {
    const allocator = std.testing.allocator;
    var chart = OrgChart.init(allocator);
    defer chart.deinit();
    const company = try OrgUnit.init(allocator, "c1", "Acme Corp", .company);
    const dept = try OrgUnit.init(allocator, "d1", "Engineering", .department);
    const team = try OrgUnit.init(allocator, "t1", "Platform", .team);
    try company.addChild(dept); try dept.addChild(team);
    try chart.addUnit(company); try chart.addUnit(dept); try chart.addUnit(team);
    try std.testing.expect(chart.root == company);
    try std.testing.expectEqualStrings("d1", team.parent_id.?);
    const json = try chart.toJson();
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"units\":[") != null);
    _ = try chart.moveUnit("t1", "c1");
    try std.testing.expectEqualStrings("c1", team.parent_id.?);
}

test "ProcessHierarchy tree operations" {
    const allocator = std.testing.allocator;
    const root = try ProcessHierarchy.init(allocator, "p0", "Main Process", 0);
    defer root.deinit();
    const sub1 = try ProcessHierarchy.init(allocator, "p1", "SubProcess A", 0);
    const sub1_1 = try ProcessHierarchy.init(allocator, "p1_1", "SubSub A1", 0);
    try root.addChild(sub1); try sub1.addChild(sub1_1);
    try std.testing.expectEqual(@as(u32, 1), sub1.level);
    try std.testing.expectEqual(@as(u32, 2), sub1_1.level);
    try std.testing.expectEqual(@as(u32, 2), root.getDepth());
    try std.testing.expect(root.findDescendant("p1_1") != null);
    const json = try root.toJson();
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"level\":0") != null);
}

test "ValueChain Porter model and EPC conversion" {
    const allocator = std.testing.allocator;
    var chain = ValueChain.init(allocator);
    defer chain.deinit();
    try chain.addPrimaryActivity(try Activity.init(allocator, "il", "Inbound Logistics", "Receiving"));
    try chain.addPrimaryActivity(try Activity.init(allocator, "op", "Operations", "Manufacturing"));
    try chain.addPrimaryActivity(try Activity.init(allocator, "ol", "Outbound Logistics", "Distribution"));
    try chain.addSupportActivity(try Activity.init(allocator, "hr", "HR Management", "Human resources"));
    try std.testing.expectEqual(@as(usize, 3), chain.primary_activities.items.len);
    const epc = try chain.toEpc();
    defer allocator.free(epc);
    try std.testing.expect(std.mem.indexOf(u8, epc, "<Model id=\"value_chain\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, epc, "<Connection id=\"conn_p_") != null);
    const json = try chain.toJson();
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"primary_activities\":[") != null);
}

test "OrgChart findPath between units" {
    const allocator = std.testing.allocator;
    var chart = OrgChart.init(allocator);
    defer chart.deinit();
    const root = try OrgUnit.init(allocator, "root", "Company", .company);
    const div1 = try OrgUnit.init(allocator, "div1", "Division A", .division);
    const team1 = try OrgUnit.init(allocator, "team1", "Team A1", .team);
    try root.addChild(div1); try div1.addChild(team1);
    try chart.addUnit(root); try chart.addUnit(div1); try chart.addUnit(team1);
    const path = try chart.findPath("team1", "root");
    if (path) |p| {
        defer { for (p) |id| allocator.free(id); allocator.free(p); }
        try std.testing.expect(p.len > 0);
    }
    try std.testing.expect((try chart.findPath("nonexistent", "div1")) == null);
}
