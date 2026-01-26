//! BPMN 2.0 Parser for nWorkflow
//!
//! Parses Business Process Model and Notation (BPMN) 2.0 XML documents
//! and converts them to the internal WorkflowSchema format.
//!
//! Supported BPMN elements:
//! - bpmn:process - process definitions
//! - bpmn:startEvent, bpmn:endEvent - events
//! - bpmn:task, bpmn:userTask, bpmn:serviceTask, bpmn:scriptTask - tasks
//! - bpmn:exclusiveGateway, bpmn:parallelGateway, bpmn:inclusiveGateway - gateways
//! - bpmn:sequenceFlow - connections
//! - bpmn:lane, bpmn:laneSet - swim lanes
//! - bpmndi:BPMNDiagram - diagram coordinates

const std = @import("std");
const Allocator = std.mem.Allocator;
const workflow_parser = @import("../core/workflow_parser.zig");
const WorkflowSchema = workflow_parser.WorkflowSchema;
const WorkflowNode = workflow_parser.WorkflowNode;
const WorkflowEdge = workflow_parser.WorkflowEdge;
const NodeType = workflow_parser.NodeType;

// ============================================================================
// BPMN ELEMENT TYPES
// ============================================================================

/// Task type in BPMN
pub const TaskType = enum {
    task,
    user_task,
    service_task,
    script_task,
    manual_task,
    send_task,
    receive_task,

    pub fn toString(self: TaskType) []const u8 {
        return switch (self) {
            .task => "task",
            .user_task => "userTask",
            .service_task => "serviceTask",
            .script_task => "scriptTask",
            .manual_task => "manualTask",
            .send_task => "sendTask",
            .receive_task => "receiveTask",
        };
    }
};

/// Gateway type in BPMN
pub const GatewayType = enum {
    exclusive,
    parallel,
    inclusive,
    event_based,
    complex,

    pub fn toString(self: GatewayType) []const u8 {
        return switch (self) {
            .exclusive => "exclusiveGateway",
            .parallel => "parallelGateway",
            .inclusive => "inclusiveGateway",
            .event_based => "eventBasedGateway",
            .complex => "complexGateway",
        };
    }
};

/// Event type in BPMN
pub const EventType = enum {
    start,
    end,
    intermediate_catch,
    intermediate_throw,
    boundary,

    pub fn toString(self: EventType) []const u8 {
        return switch (self) {
            .start => "startEvent",
            .end => "endEvent",
            .intermediate_catch => "intermediateCatchEvent",
            .intermediate_throw => "intermediateThrowEvent",
            .boundary => "boundaryEvent",
        };
    }
};

// ============================================================================
// BPMN ELEMENT STRUCTS
// ============================================================================

/// Base BPMN element with common attributes
pub const BpmnElement = struct {
    id: []const u8,
    name: ?[]const u8,

    pub fn init(allocator: Allocator, id: []const u8, name: ?[]const u8) !BpmnElement {
        return BpmnElement{
            .id = try allocator.dupe(u8, id),
            .name = if (name) |n| try allocator.dupe(u8, n) else null,
        };
    }

    pub fn deinit(self: *BpmnElement, allocator: Allocator) void {
        allocator.free(self.id);
        if (self.name) |n| allocator.free(n);
    }
};

/// BPMN Task element
pub const BpmnTask = struct {
    base: BpmnElement,
    task_type: TaskType,
    incoming: std.ArrayListUnmanaged([]const u8),
    outgoing: std.ArrayListUnmanaged([]const u8),
    script: ?[]const u8,
    implementation: ?[]const u8,

    pub fn init(allocator: Allocator, id: []const u8, name: ?[]const u8, task_type: TaskType) !BpmnTask {
        return BpmnTask{
            .base = try BpmnElement.init(allocator, id, name),
            .task_type = task_type,
            .incoming = .{},
            .outgoing = .{},
            .script = null,
            .implementation = null,
        };
    }

    pub fn deinit(self: *BpmnTask, allocator: Allocator) void {
        self.base.deinit(allocator);
        for (self.incoming.items) |item| allocator.free(item);
        self.incoming.deinit(allocator);
        for (self.outgoing.items) |item| allocator.free(item);
        self.outgoing.deinit(allocator);
        if (self.script) |s| allocator.free(s);
        if (self.implementation) |i| allocator.free(i);
    }
};

/// BPMN Gateway element
pub const BpmnGateway = struct {
    base: BpmnElement,
    gateway_type: GatewayType,
    incoming: std.ArrayListUnmanaged([]const u8),
    outgoing: std.ArrayListUnmanaged([]const u8),
    default_flow: ?[]const u8,

    pub fn init(allocator: Allocator, id: []const u8, name: ?[]const u8, gateway_type: GatewayType) !BpmnGateway {
        return BpmnGateway{
            .base = try BpmnElement.init(allocator, id, name),
            .gateway_type = gateway_type,
            .incoming = .{},
            .outgoing = .{},
            .default_flow = null,
        };
    }

    pub fn deinit(self: *BpmnGateway, allocator: Allocator) void {
        self.base.deinit(allocator);
        for (self.incoming.items) |item| allocator.free(item);
        self.incoming.deinit(allocator);
        for (self.outgoing.items) |item| allocator.free(item);
        self.outgoing.deinit(allocator);
        if (self.default_flow) |d| allocator.free(d);
    }
};

/// BPMN Event element
pub const BpmnEvent = struct {
    base: BpmnElement,
    event_type: EventType,
    incoming: std.ArrayListUnmanaged([]const u8),
    outgoing: std.ArrayListUnmanaged([]const u8),
    event_definition: ?[]const u8,

    pub fn init(allocator: Allocator, id: []const u8, name: ?[]const u8, event_type: EventType) !BpmnEvent {
        return BpmnEvent{
            .base = try BpmnElement.init(allocator, id, name),
            .event_type = event_type,
            .incoming = .{},
            .outgoing = .{},
            .event_definition = null,
        };
    }

    pub fn deinit(self: *BpmnEvent, allocator: Allocator) void {
        self.base.deinit(allocator);
        for (self.incoming.items) |item| allocator.free(item);
        self.incoming.deinit(allocator);
        for (self.outgoing.items) |item| allocator.free(item);
        self.outgoing.deinit(allocator);
        if (self.event_definition) |e| allocator.free(e);
    }
};

/// BPMN Sequence Flow (connection)
pub const BpmnSequenceFlow = struct {
    base: BpmnElement,
    source_ref: []const u8,
    target_ref: []const u8,
    condition_expression: ?[]const u8,

    pub fn init(allocator: Allocator, id: []const u8, name: ?[]const u8, source: []const u8, target: []const u8) !BpmnSequenceFlow {
        return BpmnSequenceFlow{
            .base = try BpmnElement.init(allocator, id, name),
            .source_ref = try allocator.dupe(u8, source),
            .target_ref = try allocator.dupe(u8, target),
            .condition_expression = null,
        };
    }

    pub fn deinit(self: *BpmnSequenceFlow, allocator: Allocator) void {
        self.base.deinit(allocator);
        allocator.free(self.source_ref);
        allocator.free(self.target_ref);
        if (self.condition_expression) |c| allocator.free(c);
    }
};

/// BPMN Lane for swim lanes
pub const BpmnLane = struct {
    base: BpmnElement,
    participant: ?[]const u8,
    flow_node_refs: std.ArrayListUnmanaged([]const u8),

    pub fn init(allocator: Allocator, id: []const u8, name: ?[]const u8) !BpmnLane {
        return BpmnLane{
            .base = try BpmnElement.init(allocator, id, name),
            .participant = null,
            .flow_node_refs = .{},
        };
    }

    pub fn deinit(self: *BpmnLane, allocator: Allocator) void {
        self.base.deinit(allocator);
        if (self.participant) |p| allocator.free(p);
        for (self.flow_node_refs.items) |item| allocator.free(item);
        self.flow_node_refs.deinit(allocator);
    }
};

/// BPMN Diagram coordinates
pub const BpmnBounds = struct {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
};

pub const BpmnWaypoint = struct {
    x: f64,
    y: f64,
};

pub const BpmnShape = struct {
    bpmn_element: []const u8,
    bounds: BpmnBounds,

    pub fn deinit(self: *BpmnShape, allocator: Allocator) void {
        allocator.free(self.bpmn_element);
    }
};

pub const BpmnEdge = struct {
    bpmn_element: []const u8,
    waypoints: std.ArrayListUnmanaged(BpmnWaypoint),

    pub fn deinit(self: *BpmnEdge, allocator: Allocator) void {
        allocator.free(self.bpmn_element);
        self.waypoints.deinit(allocator);
    }
};

pub const BpmnDiagram = struct {
    id: []const u8,
    shapes: std.ArrayListUnmanaged(BpmnShape),
    edges: std.ArrayListUnmanaged(BpmnEdge),

    pub fn init(allocator: Allocator, id: []const u8) !BpmnDiagram {
        return BpmnDiagram{
            .id = try allocator.dupe(u8, id),
            .shapes = .{},
            .edges = .{},
        };
    }

    pub fn deinit(self: *BpmnDiagram, allocator: Allocator) void {
        allocator.free(self.id);
        for (self.shapes.items) |*shape| shape.deinit(allocator);
        self.shapes.deinit(allocator);
        for (self.edges.items) |*edge| edge.deinit(allocator);
        self.edges.deinit(allocator);
    }
};

/// Complete BPMN Process
pub const BpmnProcess = struct {
    allocator: Allocator,
    id: []const u8,
    name: ?[]const u8,
    is_executable: bool,
    events: std.ArrayListUnmanaged(BpmnEvent),
    tasks: std.ArrayListUnmanaged(BpmnTask),
    gateways: std.ArrayListUnmanaged(BpmnGateway),
    sequence_flows: std.ArrayListUnmanaged(BpmnSequenceFlow),
    lanes: std.ArrayListUnmanaged(BpmnLane),
    diagram: ?BpmnDiagram,

    pub fn init(allocator: Allocator, id: []const u8, name: ?[]const u8) !BpmnProcess {
        return BpmnProcess{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = if (name) |n| try allocator.dupe(u8, n) else null,
            .is_executable = true,
            .events = .{},
            .tasks = .{},
            .gateways = .{},
            .sequence_flows = .{},
            .lanes = .{},
            .diagram = null,
        };
    }

    pub fn deinit(self: *BpmnProcess) void {
        self.allocator.free(self.id);
        if (self.name) |n| self.allocator.free(n);
        for (self.events.items) |*e| e.deinit(self.allocator);
        self.events.deinit(self.allocator);
        for (self.tasks.items) |*t| t.deinit(self.allocator);
        self.tasks.deinit(self.allocator);
        for (self.gateways.items) |*g| g.deinit(self.allocator);
        self.gateways.deinit(self.allocator);
        for (self.sequence_flows.items) |*f| f.deinit(self.allocator);
        self.sequence_flows.deinit(self.allocator);
        for (self.lanes.items) |*l| l.deinit(self.allocator);
        self.lanes.deinit(self.allocator);
        if (self.diagram) |*d| d.deinit(self.allocator);
    }
};

// ============================================================================
// BPMN PARSER
// ============================================================================

pub const BpmnParseError = error{
    InvalidXml,
    MissingProcessElement,
    MissingId,
    MissingSourceRef,
    MissingTargetRef,
    InvalidElement,
    UnexpectedEndOfInput,
    OutOfMemory,
};

/// BPMN 2.0 XML Parser
pub const BpmnParser = struct {
    allocator: Allocator,
    source: []const u8,
    pos: usize,

    pub fn init(allocator: Allocator) BpmnParser {
        return BpmnParser{
            .allocator = allocator,
            .source = "",
            .pos = 0,
        };
    }

    /// Parse BPMN XML and return a BpmnProcess
    pub fn parseBpmn(self: *BpmnParser, xml: []const u8) !BpmnProcess {
        self.source = xml;
        self.pos = 0;

        // Skip XML declaration
        self.skipXmlDeclaration();
        self.skipWhitespace();

        // Find process element
        const process_start = self.findElement("bpmn:process") orelse
            self.findElement("process") orelse
            return BpmnParseError.MissingProcessElement;

        self.pos = process_start;

        // Parse process attributes
        const process_id = self.getAttribute("id") orelse return BpmnParseError.MissingId;
        const process_name = self.getAttribute("name");

        var process = try BpmnProcess.init(self.allocator, process_id, process_name);
        errdefer process.deinit();

        // Check isExecutable
        if (self.getAttribute("isExecutable")) |exec| {
            process.is_executable = std.mem.eql(u8, exec, "true");
        }

        // Parse all child elements
        try self.parseProcessChildren(&process);

        // Parse diagram if present
        self.pos = 0;
        if (self.findElement("bpmndi:BPMNDiagram")) |diag_start| {
            self.pos = diag_start;
            process.diagram = try self.parseDiagram();
        }

        return process;
    }

    fn parseProcessChildren(self: *BpmnParser, process: *BpmnProcess) !void {
        const start_pos = self.pos;

        // Parse start events
        self.pos = start_pos;
        while (self.findElement("bpmn:startEvent")) |elem_pos| {
            self.pos = elem_pos;
            var event = try self.parseEvent(.start);
            try process.events.append(self.allocator, event);
            self.pos = elem_pos + 1;
        }

        // Parse end events
        self.pos = start_pos;
        while (self.findElement("bpmn:endEvent")) |elem_pos| {
            self.pos = elem_pos;
            var event = try self.parseEvent(.end);
            try process.events.append(self.allocator, event);
            self.pos = elem_pos + 1;
        }

        // Parse tasks
        self.pos = start_pos;
        try self.parseTasks(process, "bpmn:task", .task);
        self.pos = start_pos;
        try self.parseTasks(process, "bpmn:userTask", .user_task);
        self.pos = start_pos;
        try self.parseTasks(process, "bpmn:serviceTask", .service_task);
        self.pos = start_pos;
        try self.parseTasks(process, "bpmn:scriptTask", .script_task);

        // Parse gateways
        self.pos = start_pos;
        try self.parseGateways(process, "bpmn:exclusiveGateway", .exclusive);
        self.pos = start_pos;
        try self.parseGateways(process, "bpmn:parallelGateway", .parallel);
        self.pos = start_pos;
        try self.parseGateways(process, "bpmn:inclusiveGateway", .inclusive);

        // Parse sequence flows
        self.pos = start_pos;
        while (self.findElement("bpmn:sequenceFlow")) |elem_pos| {
            self.pos = elem_pos;
            var flow = try self.parseSequenceFlow();
            try process.sequence_flows.append(self.allocator, flow);
            self.pos = elem_pos + 1;
        }

        // Parse lanes
        self.pos = start_pos;
        while (self.findElement("bpmn:lane")) |elem_pos| {
            self.pos = elem_pos;
            var lane = try self.parseLane();
            try process.lanes.append(self.allocator, lane);
            self.pos = elem_pos + 1;
        }
    }

    fn parseTasks(self: *BpmnParser, process: *BpmnProcess, tag: []const u8, task_type: TaskType) !void {
        while (self.findElement(tag)) |elem_pos| {
            self.pos = elem_pos;
            var task = try self.parseTask(task_type);
            try process.tasks.append(self.allocator, task);
            self.pos = elem_pos + 1;
        }
    }

    fn parseGateways(self: *BpmnParser, process: *BpmnProcess, tag: []const u8, gw_type: GatewayType) !void {
        while (self.findElement(tag)) |elem_pos| {
            self.pos = elem_pos;
            var gateway = try self.parseGateway(gw_type);
            try process.gateways.append(self.allocator, gateway);
            self.pos = elem_pos + 1;
        }
    }

    fn parseEvent(self: *BpmnParser, event_type: EventType) !BpmnEvent {
        const id = self.getAttribute("id") orelse return BpmnParseError.MissingId;
        const name = self.getAttribute("name");
        return try BpmnEvent.init(self.allocator, id, name, event_type);
    }

    fn parseTask(self: *BpmnParser, task_type: TaskType) !BpmnTask {
        const id = self.getAttribute("id") orelse return BpmnParseError.MissingId;
        const name = self.getAttribute("name");
        return try BpmnTask.init(self.allocator, id, name, task_type);
    }

    fn parseGateway(self: *BpmnParser, gateway_type: GatewayType) !BpmnGateway {
        const id = self.getAttribute("id") orelse return BpmnParseError.MissingId;
        const name = self.getAttribute("name");
        var gateway = try BpmnGateway.init(self.allocator, id, name, gateway_type);
        gateway.default_flow = if (self.getAttribute("default")) |d| try self.allocator.dupe(u8, d) else null;
        return gateway;
    }

    fn parseSequenceFlow(self: *BpmnParser) !BpmnSequenceFlow {
        const id = self.getAttribute("id") orelse return BpmnParseError.MissingId;
        const name = self.getAttribute("name");
        const source = self.getAttribute("sourceRef") orelse return BpmnParseError.MissingSourceRef;
        const target = self.getAttribute("targetRef") orelse return BpmnParseError.MissingTargetRef;
        return try BpmnSequenceFlow.init(self.allocator, id, name, source, target);
    }

    fn parseLane(self: *BpmnParser) !BpmnLane {
        const id = self.getAttribute("id") orelse return BpmnParseError.MissingId;
        const name = self.getAttribute("name");
        return try BpmnLane.init(self.allocator, id, name);
    }

    fn parseDiagram(self: *BpmnParser) !BpmnDiagram {
        const id = self.getAttribute("id") orelse "diagram";
        var diagram = try BpmnDiagram.init(self.allocator, id);
        errdefer diagram.deinit(self.allocator);

        const start_pos = self.pos;

        // Parse shapes
        while (self.findElement("bpmndi:BPMNShape")) |elem_pos| {
            self.pos = elem_pos;
            if (self.getAttribute("bpmnElement")) |bpmn_elem| {
                var shape = BpmnShape{
                    .bpmn_element = try self.allocator.dupe(u8, bpmn_elem),
                    .bounds = BpmnBounds{ .x = 0, .y = 0, .width = 100, .height = 80 },
                };
                try diagram.shapes.append(self.allocator, shape);
            }
            self.pos = elem_pos + 1;
        }

        // Parse edges
        self.pos = start_pos;
        while (self.findElement("bpmndi:BPMNEdge")) |elem_pos| {
            self.pos = elem_pos;
            if (self.getAttribute("bpmnElement")) |bpmn_elem| {
                var edge = BpmnEdge{
                    .bpmn_element = try self.allocator.dupe(u8, bpmn_elem),
                    .waypoints = .{},
                };
                try diagram.edges.append(self.allocator, edge);
            }
            self.pos = elem_pos + 1;
        }

        return diagram;
    }

    // ========================================================================
    // XML PARSING HELPERS
    // ========================================================================

    fn findElement(self: *BpmnParser, tag: []const u8) ?usize {
        const search_start = "<" ++ tag[0..0];
        _ = search_start;
        var search_pos = self.pos;

        while (search_pos < self.source.len) {
            if (std.mem.indexOfPos(u8, self.source, search_pos, "<")) |open_pos| {
                const after_open = open_pos + 1;
                if (after_open >= self.source.len) return null;

                // Check if this matches our tag
                const remaining = self.source[after_open..];
                if (std.mem.startsWith(u8, remaining, tag)) {
                    const after_tag = after_open + tag.len;
                    if (after_tag < self.source.len) {
                        const next_char = self.source[after_tag];
                        if (next_char == ' ' or next_char == '>' or next_char == '/' or next_char == '\n' or next_char == '\r' or next_char == '\t') {
                            return open_pos;
                        }
                    }
                }
                search_pos = open_pos + 1;
            } else {
                return null;
            }
        }
        return null;
    }

    fn getAttribute(self: *BpmnParser, attr_name: []const u8) ?[]const u8 {
        // Find the end of the current tag
        const tag_end = std.mem.indexOfPos(u8, self.source, self.pos, ">") orelse return null;
        const tag_content = self.source[self.pos..tag_end];

        // Search for attribute
        var search_str: [64]u8 = undefined;
        const search_len = @min(attr_name.len + 2, 62);
        @memcpy(search_str[0..attr_name.len], attr_name);
        search_str[attr_name.len] = '=';
        search_str[attr_name.len + 1] = '"';

        if (std.mem.indexOf(u8, tag_content, search_str[0..search_len])) |attr_pos| {
            const value_start = attr_pos + search_len;
            if (value_start < tag_content.len) {
                if (std.mem.indexOfPos(u8, tag_content, value_start, "\"")) |value_end| {
                    return tag_content[value_start..value_end];
                }
            }
        }
        return null;
    }

    fn skipXmlDeclaration(self: *BpmnParser) void {
        if (std.mem.startsWith(u8, self.source, "<?xml")) {
            if (std.mem.indexOf(u8, self.source, "?>")) |end| {
                self.pos = end + 2;
            }
        }
    }

    fn skipWhitespace(self: *BpmnParser) void {
        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
};

// ============================================================================
// CONVERSION FUNCTIONS
// ============================================================================

/// Convert BpmnProcess to internal WorkflowSchema format
pub fn toWorkflowSchema(allocator: Allocator, process: *const BpmnProcess) !WorkflowSchema {
    var schema = WorkflowSchema.init(allocator);

    schema.version = try allocator.dupe(u8, "1.0");
    schema.name = try allocator.dupe(u8, process.name orelse process.id);
    schema.description = try allocator.dupe(u8, "Imported from BPMN");

    var nodes_list = std.ArrayList(WorkflowNode){};
    errdefer nodes_list.deinit();

    var edges_list = std.ArrayList(WorkflowEdge){};
    errdefer edges_list.deinit();

    // Convert events to nodes
    for (process.events.items) |event| {
        const node_type: NodeType = switch (event.event_type) {
            .start => .trigger,
            .end => .action,
            else => .action,
        };
        try nodes_list.append(WorkflowNode{
            .id = try allocator.dupe(u8, event.base.id),
            .node_type = node_type,
            .name = try allocator.dupe(u8, event.base.name orelse event.base.id),
            .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
        });
    }

    // Convert tasks to nodes
    for (process.tasks.items) |task| {
        try nodes_list.append(WorkflowNode{
            .id = try allocator.dupe(u8, task.base.id),
            .node_type = .action,
            .name = try allocator.dupe(u8, task.base.name orelse task.base.id),
            .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
        });
    }

    // Convert gateways to nodes
    for (process.gateways.items) |gateway| {
        const node_type: NodeType = switch (gateway.gateway_type) {
            .exclusive => .condition,
            .parallel => .split,
            .inclusive => .condition,
            else => .condition,
        };
        try nodes_list.append(WorkflowNode{
            .id = try allocator.dupe(u8, gateway.base.id),
            .node_type = node_type,
            .name = try allocator.dupe(u8, gateway.base.name orelse gateway.base.id),
            .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
        });
    }

    // Convert sequence flows to edges
    for (process.sequence_flows.items) |flow| {
        try edges_list.append(WorkflowEdge{
            .from = try allocator.dupe(u8, flow.source_ref),
            .to = try allocator.dupe(u8, flow.target_ref),
            .condition = if (flow.condition_expression) |c| try allocator.dupe(u8, c) else null,
        });
    }

    schema.nodes = try nodes_list.toOwnedSlice();
    schema.edges = try edges_list.toOwnedSlice();

    return schema;
}

/// Convert internal WorkflowSchema to BpmnProcess
pub fn fromWorkflowSchema(allocator: Allocator, schema: *const WorkflowSchema) !BpmnProcess {
    var process = try BpmnProcess.init(allocator, schema.name, schema.name);
    errdefer process.deinit();

    // Convert nodes to BPMN elements
    for (schema.nodes) |node| {
        switch (node.node_type) {
            .trigger => {
                var event = try BpmnEvent.init(allocator, node.id, node.name, .start);
                try process.events.append(allocator, event);
            },
            .action, .transform => {
                var task = try BpmnTask.init(allocator, node.id, node.name, .task);
                try process.tasks.append(allocator, task);
            },
            .condition => {
                var gateway = try BpmnGateway.init(allocator, node.id, node.name, .exclusive);
                try process.gateways.append(allocator, gateway);
            },
            .split => {
                var gateway = try BpmnGateway.init(allocator, node.id, node.name, .parallel);
                try process.gateways.append(allocator, gateway);
            },
            .join => {
                var gateway = try BpmnGateway.init(allocator, node.id, node.name, .parallel);
                try process.gateways.append(allocator, gateway);
            },
        }
    }

    // Convert edges to sequence flows
    var flow_count: usize = 0;
    for (schema.edges) |edge| {
        const flow_id = try std.fmt.allocPrint(allocator, "Flow_{d}", .{flow_count});
        var flow = try BpmnSequenceFlow.init(allocator, flow_id, null, edge.from, edge.to);
        allocator.free(flow_id);
        if (edge.condition) |c| {
            flow.condition_expression = try allocator.dupe(u8, c);
        }
        try process.sequence_flows.append(allocator, flow);
        flow_count += 1;
    }

    return process;
}

/// Parse BPMN XML - convenience function
pub fn parseBpmn(allocator: Allocator, xml: []const u8) !BpmnProcess {
    var parser = BpmnParser.init(allocator);
    return try parser.parseBpmn(xml);
}

// ============================================================================
// TESTS
// ============================================================================

test "parse simple BPMN process" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const bpmn_xml =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
        \\  <bpmn:process id="Process_1" name="Simple Process" isExecutable="true">
        \\    <bpmn:startEvent id="StartEvent_1" name="Start"/>
        \\    <bpmn:task id="Task_1" name="Do Something"/>
        \\    <bpmn:endEvent id="EndEvent_1" name="End"/>
        \\    <bpmn:sequenceFlow id="Flow_1" sourceRef="StartEvent_1" targetRef="Task_1"/>
        \\    <bpmn:sequenceFlow id="Flow_2" sourceRef="Task_1" targetRef="EndEvent_1"/>
        \\  </bpmn:process>
        \\</bpmn:definitions>
    ;

    var process = try parseBpmn(allocator, bpmn_xml);
    defer process.deinit();

    try testing.expectEqualStrings("Process_1", process.id);
    try testing.expectEqualStrings("Simple Process", process.name.?);
    try testing.expect(process.is_executable);
    try testing.expectEqual(@as(usize, 2), process.events.items.len);
    try testing.expectEqual(@as(usize, 1), process.tasks.items.len);
    try testing.expectEqual(@as(usize, 2), process.sequence_flows.items.len);
}

test "parse BPMN with gateways" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const bpmn_xml =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
        \\  <bpmn:process id="Process_2" name="Gateway Process">
        \\    <bpmn:startEvent id="Start"/>
        \\    <bpmn:exclusiveGateway id="Gateway_1" name="Decision"/>
        \\    <bpmn:parallelGateway id="Gateway_2" name="Split"/>
        \\    <bpmn:endEvent id="End"/>
        \\    <bpmn:sequenceFlow id="Flow_1" sourceRef="Start" targetRef="Gateway_1"/>
        \\    <bpmn:sequenceFlow id="Flow_2" sourceRef="Gateway_1" targetRef="Gateway_2"/>
        \\  </bpmn:process>
        \\</bpmn:definitions>
    ;

    var process = try parseBpmn(allocator, bpmn_xml);
    defer process.deinit();

    try testing.expectEqual(@as(usize, 2), process.gateways.items.len);
    try testing.expectEqual(GatewayType.exclusive, process.gateways.items[0].gateway_type);
    try testing.expectEqual(GatewayType.parallel, process.gateways.items[1].gateway_type);
}

test "parse BPMN with different task types" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const bpmn_xml =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
        \\  <bpmn:process id="Process_3" name="Task Types">
        \\    <bpmn:userTask id="UserTask_1" name="Review"/>
        \\    <bpmn:serviceTask id="ServiceTask_1" name="API Call"/>
        \\    <bpmn:scriptTask id="ScriptTask_1" name="Calculate"/>
        \\  </bpmn:process>
        \\</bpmn:definitions>
    ;

    var process = try parseBpmn(allocator, bpmn_xml);
    defer process.deinit();

    try testing.expectEqual(@as(usize, 3), process.tasks.items.len);

    var found_user = false;
    var found_service = false;
    var found_script = false;

    for (process.tasks.items) |task| {
        switch (task.task_type) {
            .user_task => found_user = true,
            .service_task => found_service = true,
            .script_task => found_script = true,
            else => {},
        }
    }

    try testing.expect(found_user);
    try testing.expect(found_service);
    try testing.expect(found_script);
}

test "convert BPMN to WorkflowSchema" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const bpmn_xml =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
        \\  <bpmn:process id="Process_1" name="Test Process">
        \\    <bpmn:startEvent id="Start" name="Begin"/>
        \\    <bpmn:task id="Task_1" name="Work"/>
        \\    <bpmn:endEvent id="End" name="Finish"/>
        \\    <bpmn:sequenceFlow id="Flow_1" sourceRef="Start" targetRef="Task_1"/>
        \\    <bpmn:sequenceFlow id="Flow_2" sourceRef="Task_1" targetRef="End"/>
        \\  </bpmn:process>
        \\</bpmn:definitions>
    ;

    var process = try parseBpmn(allocator, bpmn_xml);
    defer process.deinit();

    var schema = try toWorkflowSchema(allocator, &process);
    defer schema.deinit();

    try testing.expectEqualStrings("Test Process", schema.name);
    try testing.expectEqual(@as(usize, 3), schema.nodes.len);
    try testing.expectEqual(@as(usize, 2), schema.edges.len);

    // Check that start event became a trigger
    var found_trigger = false;
    for (schema.nodes) |node| {
        if (node.node_type == .trigger) {
            found_trigger = true;
            break;
        }
    }
    try testing.expect(found_trigger);
}

test "convert WorkflowSchema to BPMN" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a simple WorkflowSchema
    var schema = WorkflowSchema.init(allocator);
    schema.version = try allocator.dupe(u8, "1.0");
    schema.name = try allocator.dupe(u8, "My Workflow");
    schema.description = try allocator.dupe(u8, "Test workflow");

    var nodes = try allocator.alloc(WorkflowNode, 2);
    nodes[0] = WorkflowNode{
        .id = try allocator.dupe(u8, "start"),
        .node_type = .trigger,
        .name = try allocator.dupe(u8, "Start"),
        .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    };
    nodes[1] = WorkflowNode{
        .id = try allocator.dupe(u8, "task1"),
        .node_type = .action,
        .name = try allocator.dupe(u8, "Task 1"),
        .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) },
    };
    schema.nodes = nodes;

    var edges = try allocator.alloc(WorkflowEdge, 1);
    edges[0] = WorkflowEdge{
        .from = try allocator.dupe(u8, "start"),
        .to = try allocator.dupe(u8, "task1"),
        .condition = null,
    };
    schema.edges = edges;

    defer schema.deinit();

    var process = try fromWorkflowSchema(allocator, &schema);
    defer process.deinit();

    try testing.expectEqualStrings("My Workflow", process.id);
    try testing.expectEqual(@as(usize, 1), process.events.items.len);
    try testing.expectEqual(@as(usize, 1), process.tasks.items.len);
    try testing.expectEqual(@as(usize, 1), process.sequence_flows.items.len);
}

test "parse BPMN with lanes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const bpmn_xml =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
        \\  <bpmn:process id="Process_1" name="Lane Process">
        \\    <bpmn:laneSet id="LaneSet_1">
        \\      <bpmn:lane id="Lane_1" name="Manager"/>
        \\      <bpmn:lane id="Lane_2" name="Employee"/>
        \\    </bpmn:laneSet>
        \\    <bpmn:startEvent id="Start"/>
        \\  </bpmn:process>
        \\</bpmn:definitions>
    ;

    var process = try parseBpmn(allocator, bpmn_xml);
    defer process.deinit();

    try testing.expectEqual(@as(usize, 2), process.lanes.items.len);
    try testing.expectEqualStrings("Manager", process.lanes.items[0].base.name.?);
    try testing.expectEqualStrings("Employee", process.lanes.items[1].base.name.?);
}

test "BpmnElement struct operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var elem = try BpmnElement.init(allocator, "test_id", "Test Name");
    defer elem.deinit(allocator);

    try testing.expectEqualStrings("test_id", elem.id);
    try testing.expectEqualStrings("Test Name", elem.name.?);
}

test "BpmnTask struct operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var task = try BpmnTask.init(allocator, "task_1", "My Task", .service_task);
    defer task.deinit(allocator);

    try testing.expectEqualStrings("task_1", task.base.id);
    try testing.expectEqual(TaskType.service_task, task.task_type);
}

test "BpmnGateway struct operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var gateway = try BpmnGateway.init(allocator, "gw_1", "Decision", .exclusive);
    defer gateway.deinit(allocator);

    try testing.expectEqualStrings("gw_1", gateway.base.id);
    try testing.expectEqual(GatewayType.exclusive, gateway.gateway_type);
}

test "BpmnSequenceFlow struct operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var flow = try BpmnSequenceFlow.init(allocator, "flow_1", "Connection", "start", "end");
    defer flow.deinit(allocator);

    try testing.expectEqualStrings("flow_1", flow.base.id);
    try testing.expectEqualStrings("start", flow.source_ref);
    try testing.expectEqualStrings("end", flow.target_ref);
}

test "TaskType toString" {
    try std.testing.expectEqualStrings("userTask", TaskType.user_task.toString());
    try std.testing.expectEqualStrings("serviceTask", TaskType.service_task.toString());
}

test "GatewayType toString" {
    try std.testing.expectEqualStrings("exclusiveGateway", GatewayType.exclusive.toString());
    try std.testing.expectEqualStrings("parallelGateway", GatewayType.parallel.toString());
}

test "EventType toString" {
    try std.testing.expectEqualStrings("startEvent", EventType.start.toString());
    try std.testing.expectEqualStrings("endEvent", EventType.end.toString());
}

