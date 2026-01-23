//! ARIS EPC (Event-driven Process Chain) Parser - Parses ARIS AML format
//! EPC to Petri Net: event→place, function→transition, xor→exclusive, and→parallel

const std = @import("std");
const Allocator = std.mem.Allocator;
const workflow_parser = @import("workflow_parser");
const WorkflowSchema = workflow_parser.WorkflowSchema;
const WorkflowNode = workflow_parser.WorkflowNode;
const WorkflowEdge = workflow_parser.WorkflowEdge;
const NodeType = workflow_parser.NodeType;

pub const EpcElementType = enum {
    event, function, organizational_unit, information_object,
    xor_connector, or_connector, and_connector, process_interface,

    pub fn toString(self: EpcElementType) []const u8 {
        return switch (self) {
            .event => "Event", .function => "Function", .organizational_unit => "OrgUnit",
            .information_object => "InfoObject", .xor_connector => "XOR", .or_connector => "OR",
            .and_connector => "AND", .process_interface => "ProcessInterface",
        };
    }

    pub fn fromString(str: []const u8) !EpcElementType {
        const map = .{
            .{ "Event", .event }, .{ "OT_EVENT", .event }, .{ "Function", .function },
            .{ "OT_FUNC", .function }, .{ "OrgUnit", .organizational_unit },
            .{ "OT_ORG_UNIT", .organizational_unit }, .{ "InfoObject", .information_object },
            .{ "OT_INFO_CARR", .information_object }, .{ "XOR", .xor_connector },
            .{ "OT_RULE_XOR", .xor_connector }, .{ "OR", .or_connector },
            .{ "OT_RULE_OR", .or_connector }, .{ "AND", .and_connector },
            .{ "OT_RULE_AND", .and_connector }, .{ "ProcessInterface", .process_interface },
            .{ "OT_PROC_IF", .process_interface },
        };
        inline for (map) |entry| if (std.mem.eql(u8, str, entry[0])) return entry[1];
        return error.InvalidElementType;
    }
};

pub const EpcConnectionType = enum {
    control_flow, org_relation, info_flow,

    pub fn toString(self: EpcConnectionType) []const u8 {
        return switch (self) { .control_flow => "ControlFlow", .org_relation => "OrgRelation", .info_flow => "InfoFlow" };
    }

    pub fn fromString(str: []const u8) !EpcConnectionType {
        if (std.mem.eql(u8, str, "ControlFlow") or std.mem.eql(u8, str, "CT_LEADS_TO_1")) return .control_flow;
        if (std.mem.eql(u8, str, "OrgRelation") or std.mem.eql(u8, str, "CT_EXEC_1")) return .org_relation;
        if (std.mem.eql(u8, str, "InfoFlow") or std.mem.eql(u8, str, "CT_PROV_INP_FOR")) return .info_flow;
        return error.InvalidConnectionType;
    }
};

pub const EpcElement = struct {
    id: []const u8, name: []const u8, element_type: EpcElementType,
    attributes: std.StringHashMap([]const u8), allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, element_type: EpcElementType) !EpcElement {
        return .{ .id = try allocator.dupe(u8, id), .name = try allocator.dupe(u8, name),
            .element_type = element_type, .attributes = std.StringHashMap([]const u8).init(allocator), .allocator = allocator };
    }

    pub fn deinit(self: *EpcElement) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        var iter = self.attributes.iterator();
        while (iter.next()) |e| { self.allocator.free(e.key_ptr.*); self.allocator.free(e.value_ptr.*); }
        self.attributes.deinit();
    }

    pub fn setAttribute(self: *EpcElement, key: []const u8, value: []const u8) !void {
        try self.attributes.put(try self.allocator.dupe(u8, key), try self.allocator.dupe(u8, value));
    }
};

pub const EpcConnection = struct {
    id: []const u8, source_id: []const u8, target_id: []const u8,
    connection_type: EpcConnectionType, allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, source_id: []const u8, target_id: []const u8, connection_type: EpcConnectionType) !EpcConnection {
        return .{ .id = try allocator.dupe(u8, id), .source_id = try allocator.dupe(u8, source_id),
            .target_id = try allocator.dupe(u8, target_id), .connection_type = connection_type, .allocator = allocator };
    }

    pub fn deinit(self: *EpcConnection) void {
        self.allocator.free(self.id); self.allocator.free(self.source_id); self.allocator.free(self.target_id);
    }
};

pub const EpcProcess = struct {
    id: []const u8, name: []const u8,
    elements: std.ArrayListUnmanaged(EpcElement), connections: std.ArrayListUnmanaged(EpcConnection), allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8) !EpcProcess {
        return .{ .id = try allocator.dupe(u8, id), .name = try allocator.dupe(u8, name),
            .elements = .{}, .connections = .{}, .allocator = allocator };
    }

    pub fn deinit(self: *EpcProcess) void {
        self.allocator.free(self.id); self.allocator.free(self.name);
        for (self.elements.items) |*e| e.deinit();
        self.elements.deinit(self.allocator);
        for (self.connections.items) |*c| c.deinit();
        self.connections.deinit(self.allocator);
    }

    pub fn addElement(self: *EpcProcess, element: EpcElement) !void { try self.elements.append(self.allocator, element); }
    pub fn addConnection(self: *EpcProcess, connection: EpcConnection) !void { try self.connections.append(self.allocator, connection); }

    pub fn findElement(self: *const EpcProcess, id: []const u8) ?*const EpcElement {
        for (self.elements.items) |*e| if (std.mem.eql(u8, e.id, id)) return e;
        return null;
    }
};

pub const EpcParseError = error{ InvalidXml, MissingModelElement, MissingId, OutOfMemory, InvalidElementType, InvalidConnectionType };

pub const EpcParser = struct {
    allocator: Allocator, source: []const u8, pos: usize,

    pub fn init(allocator: Allocator) EpcParser { return .{ .allocator = allocator, .source = "", .pos = 0 }; }
    pub fn deinit(self: *EpcParser) void { _ = self; }

    pub fn parseAml(self: *EpcParser, xml: []const u8) !EpcProcess {
        self.source = xml; self.pos = 0;
        self.skipXmlDecl(); self.skipWs();

        const model_start = self.findElem("Model") orelse self.findElem("Group") orelse return EpcParseError.MissingModelElement;
        self.pos = model_start;

        const model_id = self.getAttr("Model.ID") orelse self.getAttr("id") orelse "epc_model";
        var process = try EpcProcess.init(self.allocator, model_id, self.getAttr("name") orelse model_id);
        errdefer process.deinit();

        try self.parseObjects(&process);
        try self.parseConnections(&process);
        return process;
    }

    fn parseObjects(self: *EpcParser, process: *EpcProcess) !void {
        const start = self.pos;
        self.pos = start;
        while (self.findElem("ObjDef")) |p| { self.pos = p; if (try self.parseObjDef()) |e| try process.addElement(e); self.pos = p + 1; }

        inline for (.{ "Event", "Function", "XOR", "OR", "AND", "OrgUnit", "InfoObject", "ProcessInterface" }) |tag| {
            self.pos = start;
            while (self.findElem(tag)) |p| {
                self.pos = p;
                const id = self.getAttr("id") orelse { self.pos = p + 1; continue; };
                try process.addElement(try EpcElement.init(self.allocator, id, self.getAttr("name") orelse id, EpcElementType.fromString(tag) catch { self.pos = p + 1; continue; }));
                self.pos = p + 1;
            }
        }
    }

    fn parseObjDef(self: *EpcParser) !?EpcElement {
        const id = self.getAttr("ObjDef.ID") orelse self.getAttr("id") orelse return null;
        const t = self.getAttr("TypeNum") orelse self.getAttr("type") orelse return null;
        return try EpcElement.init(self.allocator, id, self.getAttr("name") orelse id, EpcElementType.fromString(t) catch return null);
    }

    fn parseConnections(self: *EpcParser, process: *EpcProcess) !void {
        const start = self.pos;
        self.pos = start;
        while (self.findElem("CxnDef")) |p| { self.pos = p; if (try self.parseCxnDef()) |c| try process.addConnection(c); self.pos = p + 1; }

        self.pos = start;
        while (self.findElem("Connection")) |p| {
            self.pos = p;
            const id = self.getAttr("id") orelse { self.pos = p + 1; continue; };
            const src = self.getAttr("sourceRef") orelse self.getAttr("source") orelse { self.pos = p + 1; continue; };
            const tgt = self.getAttr("targetRef") orelse self.getAttr("target") orelse { self.pos = p + 1; continue; };
            try process.addConnection(try EpcConnection.init(self.allocator, id, src, tgt, EpcConnectionType.fromString(self.getAttr("type") orelse "ControlFlow") catch .control_flow));
            self.pos = p + 1;
        }
    }

    fn parseCxnDef(self: *EpcParser) !?EpcConnection {
        const id = self.getAttr("CxnDef.ID") orelse self.getAttr("id") orelse return null;
        const src = self.getAttr("SourceObjDef.IdRef") orelse self.getAttr("source") orelse return null;
        const tgt = self.getAttr("TargetObjDef.IdRef") orelse self.getAttr("target") orelse return null;
        return try EpcConnection.init(self.allocator, id, src, tgt, EpcConnectionType.fromString(self.getAttr("CxnDef.Type") orelse "CT_LEADS_TO_1") catch .control_flow);
    }

    fn findElem(self: *EpcParser, tag: []const u8) ?usize {
        var p = self.pos;
        while (p < self.source.len) {
            const open = std.mem.indexOfPos(u8, self.source, p, "<") orelse return null;
            if (open + 1 >= self.source.len) return null;
            if (std.mem.startsWith(u8, self.source[open + 1 ..], tag)) {
                const after = open + 1 + tag.len;
                if (after < self.source.len and (self.source[after] == ' ' or self.source[after] == '>' or self.source[after] == '/' or self.source[after] == '\n')) return open;
            }
            p = open + 1;
        }
        return null;
    }

    fn getAttr(self: *EpcParser, name: []const u8) ?[]const u8 {
        const end = std.mem.indexOfPos(u8, self.source, self.pos, ">") orelse return null;
        const tag = self.source[self.pos..end];
        var buf: [128]u8 = undefined;
        const len = @min(name.len + 2, 126);
        @memcpy(buf[0..name.len], name); buf[name.len] = '='; buf[name.len + 1] = '"';
        const attr_pos = std.mem.indexOf(u8, tag, buf[0..len]) orelse return null;
        const val_start = attr_pos + len;
        const val_end = std.mem.indexOfPos(u8, tag, val_start, "\"") orelse return null;
        return tag[val_start..val_end];
    }

    fn skipXmlDecl(self: *EpcParser) void {
        if (std.mem.startsWith(u8, self.source, "<?xml")) if (std.mem.indexOf(u8, self.source, "?>")) |e| { self.pos = e + 2; };
    }

    fn skipWs(self: *EpcParser) void {
        while (self.pos < self.source.len and (self.source[self.pos] == ' ' or self.source[self.pos] == '\t' or self.source[self.pos] == '\n' or self.source[self.pos] == '\r')) self.pos += 1;
    }

    pub fn toWorkflowSchema(self: *EpcParser, process: *const EpcProcess) !WorkflowSchema {
        var schema = WorkflowSchema.init(self.allocator);
        schema.version = try self.allocator.dupe(u8, "1.0");
        schema.name = try self.allocator.dupe(u8, process.name);
        schema.description = try self.allocator.dupe(u8, "Imported from ARIS EPC");

        var nodes: std.ArrayListUnmanaged(WorkflowNode) = .{};
        var edges: std.ArrayListUnmanaged(WorkflowEdge) = .{};

        for (process.elements.items) |e| {
            const nt: NodeType = switch (e.element_type) {
                .event => .trigger, .function => .action, .xor_connector => .condition, .or_connector => .condition,
                .and_connector => .split, .process_interface => .action, .organizational_unit, .information_object => continue,
            };
            try nodes.append(self.allocator, .{ .id = try self.allocator.dupe(u8, e.id), .node_type = nt, .name = try self.allocator.dupe(u8, e.name), .config = std.json.Value{ .object = std.json.ObjectMap.init(self.allocator) } });
        }

        for (process.connections.items) |c| {
            if (c.connection_type != .control_flow) continue;
            try edges.append(self.allocator, .{ .from = try self.allocator.dupe(u8, c.source_id), .to = try self.allocator.dupe(u8, c.target_id), .condition = null });
        }

        schema.nodes = try nodes.toOwnedSlice(self.allocator);
        schema.edges = try edges.toOwnedSlice(self.allocator);
        return schema;
    }

    pub fn fromWorkflowSchema(self: *EpcParser, schema: *const WorkflowSchema) !EpcProcess {
        var process = try EpcProcess.init(self.allocator, schema.name, schema.name);
        errdefer process.deinit();

        for (schema.nodes) |n| {
            const et: EpcElementType = switch (n.node_type) { .trigger => .event, .action, .transform => .function, .condition => .xor_connector, .split, .join => .and_connector };
            try process.addElement(try EpcElement.init(self.allocator, n.id, n.name, et));
        }

        var i: usize = 0;
        for (schema.edges) |e| {
            const cid = try std.fmt.allocPrint(self.allocator, "Cxn_{d}", .{i});
            defer self.allocator.free(cid);
            try process.addConnection(try EpcConnection.init(self.allocator, cid, e.from, e.to, .control_flow));
            i += 1;
        }
        return process;
    }
};

pub fn parseAml(allocator: Allocator, xml: []const u8) !EpcProcess { var p = EpcParser.init(allocator); return p.parseAml(xml); }
pub fn toWorkflowSchema(allocator: Allocator, process: *const EpcProcess) !WorkflowSchema { var p = EpcParser.init(allocator); return p.toWorkflowSchema(process); }
pub fn fromWorkflowSchema(allocator: Allocator, schema: *const WorkflowSchema) !EpcProcess { var p = EpcParser.init(allocator); return p.fromWorkflowSchema(schema); }

// ============================================================================
// TESTS
// ============================================================================

test "EpcElementType toString and fromString" {
    try std.testing.expectEqualStrings("Event", EpcElementType.event.toString());
    try std.testing.expectEqualStrings("XOR", EpcElementType.xor_connector.toString());
    try std.testing.expectEqual(EpcElementType.event, try EpcElementType.fromString("OT_EVENT"));
    try std.testing.expectEqual(EpcElementType.function, try EpcElementType.fromString("OT_FUNC"));
}

test "EpcConnectionType operations" {
    try std.testing.expectEqualStrings("ControlFlow", EpcConnectionType.control_flow.toString());
    try std.testing.expectEqual(EpcConnectionType.control_flow, try EpcConnectionType.fromString("CT_LEADS_TO_1"));
}

test "EpcElement and EpcConnection structs" {
    const allocator = std.testing.allocator;
    var elem = try EpcElement.init(allocator, "e1", "Event1", .event);
    defer elem.deinit();
    try std.testing.expectEqualStrings("e1", elem.id);
    try elem.setAttribute("key", "value");
    try std.testing.expectEqualStrings("value", elem.attributes.get("key").?);

    var conn = try EpcConnection.init(allocator, "c1", "e1", "f1", .control_flow);
    defer conn.deinit();
    try std.testing.expectEqualStrings("e1", conn.source_id);
}

test "EpcProcess operations" {
    const allocator = std.testing.allocator;
    var proc = try EpcProcess.init(allocator, "p1", "Process");
    defer proc.deinit();

    try proc.addElement(try EpcElement.init(allocator, "e1", "Start", .event));
    try proc.addConnection(try EpcConnection.init(allocator, "c1", "e1", "f1", .control_flow));

    try std.testing.expectEqual(@as(usize, 1), proc.elements.items.len);
    try std.testing.expect(proc.findElement("e1") != null);
}

test "parse simple EPC AML" {
    const allocator = std.testing.allocator;
    const xml =
        \\<?xml version="1.0"?>
        \\<Model id="epc1" name="OrderProcess">
        \\  <Event id="start" name="Order Received"/>
        \\  <Function id="check" name="Check Stock"/>
        \\  <XOR id="xor1" name="Available?"/>
        \\  <Event id="end" name="Done"/>
        \\  <Connection id="c1" source="start" target="check"/>
        \\  <Connection id="c2" source="check" target="xor1"/>
        \\</Model>
    ;

    var proc = try parseAml(allocator, xml);
    defer proc.deinit();

    try std.testing.expectEqualStrings("epc1", proc.id);
    try std.testing.expectEqual(@as(usize, 4), proc.elements.items.len);
    try std.testing.expectEqual(@as(usize, 2), proc.connections.items.len);
}

test "EPC to WorkflowSchema conversion" {
    const allocator = std.testing.allocator;
    const xml =
        \\<Model id="epc" name="Test">
        \\  <Event id="s" name="Start"/>
        \\  <Function id="t" name="Task"/>
        \\  <Connection id="c1" source="s" target="t"/>
        \\</Model>
    ;

    var proc = try parseAml(allocator, xml);
    defer proc.deinit();

    var schema = try toWorkflowSchema(allocator, &proc);
    defer schema.deinit();

    try std.testing.expectEqualStrings("Test", schema.name);
    try std.testing.expectEqual(@as(usize, 2), schema.nodes.len);
    try std.testing.expectEqual(@as(usize, 1), schema.edges.len);
}

test "WorkflowSchema to EPC conversion" {
    const allocator = std.testing.allocator;
    var schema = WorkflowSchema.init(allocator);
    schema.version = try allocator.dupe(u8, "1.0");
    schema.name = try allocator.dupe(u8, "WF");
    schema.description = try allocator.dupe(u8, "");

    var nodes = try allocator.alloc(WorkflowNode, 2);
    nodes[0] = .{ .id = try allocator.dupe(u8, "n1"), .node_type = .trigger, .name = try allocator.dupe(u8, "N1"), .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) } };
    nodes[1] = .{ .id = try allocator.dupe(u8, "n2"), .node_type = .action, .name = try allocator.dupe(u8, "N2"), .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) } };
    schema.nodes = nodes;

    var edges = try allocator.alloc(WorkflowEdge, 1);
    edges[0] = .{ .from = try allocator.dupe(u8, "n1"), .to = try allocator.dupe(u8, "n2"), .condition = null };
    schema.edges = edges;
    defer schema.deinit();

    var proc = try fromWorkflowSchema(allocator, &schema);
    defer proc.deinit();

    try std.testing.expectEqual(@as(usize, 2), proc.elements.items.len);
    try std.testing.expectEqual(@as(usize, 1), proc.connections.items.len);
    try std.testing.expectEqual(EpcElementType.event, proc.elements.items[0].element_type);
    try std.testing.expectEqual(EpcElementType.function, proc.elements.items[1].element_type);
}

