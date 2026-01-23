//! BPMN 2.0 Exporter for nWorkflow
//!
//! Exports nWorkflow definitions to standard BPMN 2.0 XML format.
//! Compatible with Camunda Modeler, bpmn.io, ARIS, and Signavio.

const std = @import("std");
const Allocator = std.mem.Allocator;
const workflow_parser = @import("workflow_parser");
const WorkflowSchema = workflow_parser.WorkflowSchema;
const WorkflowNode = workflow_parser.WorkflowNode;
const WorkflowEdge = workflow_parser.WorkflowEdge;
const NodeType = workflow_parser.NodeType;

// ============================================================================
// BPMN EXPORTER
// ============================================================================

pub const BpmnExporter = struct {
    allocator: Allocator,
    xml_buffer: std.ArrayListUnmanaged(u8),

    pub fn init(allocator: Allocator) BpmnExporter {
        return BpmnExporter{
            .allocator = allocator,
            .xml_buffer = .{},
        };
    }

    pub fn deinit(self: *BpmnExporter) void {
        self.xml_buffer.deinit(self.allocator);
    }

    /// Export WorkflowSchema to BPMN 2.0 XML string
    pub fn exportToXml(self: *BpmnExporter, schema: WorkflowSchema) ![]const u8 {
        self.xml_buffer.clearRetainingCapacity();

        try self.writeXmlHeader();
        try self.writeDefinitionsOpen(schema.name);
        try self.writeProcessElement(schema.name, schema.nodes, schema.edges);
        try self.writeDiagram(schema.name, schema.nodes, schema.edges);
        try self.writeDefinitionsClose();

        return try self.allocator.dupe(u8, self.xml_buffer.items);
    }

    /// Export from raw node/connection arrays
    pub fn exportProcessDefinition(
        self: *BpmnExporter,
        process_id: []const u8,
        nodes: []const WorkflowNode,
        connections: []const WorkflowEdge,
    ) ![]const u8 {
        self.xml_buffer.clearRetainingCapacity();

        try self.writeXmlHeader();
        try self.writeDefinitionsOpen(process_id);
        try self.writeProcessElement(process_id, nodes, connections);
        try self.writeDiagram(process_id, nodes, connections);
        try self.writeDefinitionsClose();

        return try self.allocator.dupe(u8, self.xml_buffer.items);
    }

    // ========================================================================
    // HELPER FUNCTIONS
    // ========================================================================

    fn writeXmlHeader(self: *BpmnExporter) !void {
        try self.xml_buffer.appendSlice(self.allocator,
            \\<?xml version="1.0" encoding="UTF-8"?>
            \\
        );
    }

    fn writeDefinitionsOpen(self: *BpmnExporter, process_id: []const u8) !void {
        try self.xml_buffer.appendSlice(self.allocator,
            \\<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
            \\                  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
            \\                  xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"
            \\                  xmlns:di="http://www.omg.org/spec/DD/20100524/DI"
            \\                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            \\                  id="Definitions_1"
            \\                  targetNamespace="http://bpmn.io/schema/bpmn"
            \\                  exporter="nWorkflow BPMN Exporter"
            \\                  exporterVersion="1.0">
            \\
        );
        _ = process_id;
    }

    fn writeDefinitionsClose(self: *BpmnExporter) !void {
        try self.xml_buffer.appendSlice(self.allocator, "</bpmn:definitions>\n");
    }

    fn writeProcessElement(
        self: *BpmnExporter,
        process_id: []const u8,
        nodes: []const WorkflowNode,
        edges: []const WorkflowEdge,
    ) !void {
        try self.xml_buffer.appendSlice(self.allocator, "  <bpmn:process id=\"");
        try self.writeEscapedXml(process_id);
        try self.xml_buffer.appendSlice(self.allocator, "\" isExecutable=\"true\">\n");

        // Write flow elements (nodes)
        for (nodes) |node| {
            try self.writeFlowElement(node, edges);
        }

        // Write sequence flows (edges)
        for (edges, 0..) |edge, idx| {
            try self.writeSequenceFlow(edge, idx);
        }

        try self.xml_buffer.appendSlice(self.allocator, "  </bpmn:process>\n");
    }

    fn writeFlowElement(self: *BpmnExporter, node: WorkflowNode, edges: []const WorkflowEdge) !void {
        const element_type = nodeTypeToBpmnElement(node.node_type);
        const is_gateway = node.node_type == .condition or node.node_type == .split or node.node_type == .join;

        try self.xml_buffer.appendSlice(self.allocator, "    <bpmn:");
        try self.xml_buffer.appendSlice(self.allocator, element_type);
        try self.xml_buffer.appendSlice(self.allocator, " id=\"");
        try self.writeEscapedXml(node.id);
        try self.xml_buffer.appendSlice(self.allocator, "\" name=\"");
        try self.writeEscapedXml(node.name);
        try self.xml_buffer.appendSlice(self.allocator, "\"");

        // Add gateway direction if applicable
        if (is_gateway) {
            try self.xml_buffer.appendSlice(self.allocator, " gatewayDirection=\"Unspecified\"");
        }

        try self.xml_buffer.appendSlice(self.allocator, ">\n");

        // Write incoming/outgoing references
        for (edges, 0..) |edge, idx| {
            if (std.mem.eql(u8, edge.to, node.id)) {
                try self.xml_buffer.appendSlice(self.allocator, "      <bpmn:incoming>Flow_");
                try self.writeNumber(idx);
                try self.xml_buffer.appendSlice(self.allocator, "</bpmn:incoming>\n");
            }
        }
        for (edges, 0..) |edge, idx| {
            if (std.mem.eql(u8, edge.from, node.id)) {
                try self.xml_buffer.appendSlice(self.allocator, "      <bpmn:outgoing>Flow_");
                try self.writeNumber(idx);
                try self.xml_buffer.appendSlice(self.allocator, "</bpmn:outgoing>\n");
            }
        }

        try self.xml_buffer.appendSlice(self.allocator, "    </bpmn:");
        try self.xml_buffer.appendSlice(self.allocator, element_type);
        try self.xml_buffer.appendSlice(self.allocator, ">\n");
    }

    fn writeSequenceFlow(self: *BpmnExporter, edge: WorkflowEdge, idx: usize) !void {
        try self.xml_buffer.appendSlice(self.allocator, "    <bpmn:sequenceFlow id=\"Flow_");
        try self.writeNumber(idx);
        try self.xml_buffer.appendSlice(self.allocator, "\" sourceRef=\"");
        try self.writeEscapedXml(edge.from);
        try self.xml_buffer.appendSlice(self.allocator, "\" targetRef=\"");
        try self.writeEscapedXml(edge.to);
        try self.xml_buffer.appendSlice(self.allocator, "\"");

        if (edge.condition) |condition| {
            try self.xml_buffer.appendSlice(self.allocator, ">\n");
            try self.xml_buffer.appendSlice(self.allocator, "      <bpmn:conditionExpression xsi:type=\"bpmn:tFormalExpression\">");
            try self.writeEscapedXml(condition);
            try self.xml_buffer.appendSlice(self.allocator, "</bpmn:conditionExpression>\n");
            try self.xml_buffer.appendSlice(self.allocator, "    </bpmn:sequenceFlow>\n");
        } else {
            try self.xml_buffer.appendSlice(self.allocator, "/>\n");
        }
    }

    fn writeDiagram(
        self: *BpmnExporter,
        process_id: []const u8,
        nodes: []const WorkflowNode,
        edges: []const WorkflowEdge,
    ) !void {
        try self.xml_buffer.appendSlice(self.allocator, "  <bpmndi:BPMNDiagram id=\"BPMNDiagram_1\">\n");
        try self.xml_buffer.appendSlice(self.allocator, "    <bpmndi:BPMNPlane id=\"BPMNPlane_1\" bpmnElement=\"");
        try self.writeEscapedXml(process_id);
        try self.xml_buffer.appendSlice(self.allocator, "\">\n");

        // Write shapes for nodes
        const base_x: f64 = 150;
        const base_y: f64 = 100;
        const spacing_x: f64 = 150;
        const spacing_y: f64 = 100;

        for (nodes, 0..) |node, idx| {
            const x = base_x + @as(f64, @floatFromInt(idx % 4)) * spacing_x;
            const y = base_y + @as(f64, @floatFromInt(idx / 4)) * spacing_y;
            try self.writeShape(node, x, y);
        }

        // Write edges
        for (edges, 0..) |edge, idx| {
            try self.writeEdge(edge, idx, nodes);
        }

        try self.xml_buffer.appendSlice(self.allocator, "    </bpmndi:BPMNPlane>\n");
        try self.xml_buffer.appendSlice(self.allocator, "  </bpmndi:BPMNDiagram>\n");
    }

    fn writeShape(self: *BpmnExporter, node: WorkflowNode, x: f64, y: f64) !void {
        const is_event = node.node_type == .trigger;
        const is_gateway = node.node_type == .condition or node.node_type == .split or node.node_type == .join;

        const width: f64 = if (is_event) 36 else if (is_gateway) 50 else 100;
        const height: f64 = if (is_event) 36 else if (is_gateway) 50 else 80;

        try self.xml_buffer.appendSlice(self.allocator, "      <bpmndi:BPMNShape id=\"");
        try self.writeEscapedXml(node.id);
        try self.xml_buffer.appendSlice(self.allocator, "_di\" bpmnElement=\"");
        try self.writeEscapedXml(node.id);
        try self.xml_buffer.appendSlice(self.allocator, "\"");

        if (is_gateway) {
            try self.xml_buffer.appendSlice(self.allocator, " isMarkerVisible=\"true\"");
        }

        try self.xml_buffer.appendSlice(self.allocator, ">\n");
        try self.xml_buffer.appendSlice(self.allocator, "        <dc:Bounds x=\"");
        try self.writeFloat(x);
        try self.xml_buffer.appendSlice(self.allocator, "\" y=\"");
        try self.writeFloat(y);
        try self.xml_buffer.appendSlice(self.allocator, "\" width=\"");
        try self.writeFloat(width);
        try self.xml_buffer.appendSlice(self.allocator, "\" height=\"");
        try self.writeFloat(height);
        try self.xml_buffer.appendSlice(self.allocator, "\"/>\n");
        try self.xml_buffer.appendSlice(self.allocator, "      </bpmndi:BPMNShape>\n");
    }

    fn writeEdge(self: *BpmnExporter, edge: WorkflowEdge, idx: usize, nodes: []const WorkflowNode) !void {
        // Calculate waypoints based on node positions
        var source_idx: usize = 0;
        var target_idx: usize = 0;

        for (nodes, 0..) |node, i| {
            if (std.mem.eql(u8, node.id, edge.from)) source_idx = i;
            if (std.mem.eql(u8, node.id, edge.to)) target_idx = i;
        }

        const base_x: f64 = 150;
        const base_y: f64 = 100;
        const spacing_x: f64 = 150;
        const spacing_y: f64 = 100;

        const source_x = base_x + @as(f64, @floatFromInt(source_idx % 4)) * spacing_x + 50;
        const source_y = base_y + @as(f64, @floatFromInt(source_idx / 4)) * spacing_y + 40;
        const target_x = base_x + @as(f64, @floatFromInt(target_idx % 4)) * spacing_x;
        const target_y = base_y + @as(f64, @floatFromInt(target_idx / 4)) * spacing_y + 40;

        try self.xml_buffer.appendSlice(self.allocator, "      <bpmndi:BPMNEdge id=\"Flow_");
        try self.writeNumber(idx);
        try self.xml_buffer.appendSlice(self.allocator, "_di\" bpmnElement=\"Flow_");
        try self.writeNumber(idx);
        try self.xml_buffer.appendSlice(self.allocator, "\">\n");

        try self.xml_buffer.appendSlice(self.allocator, "        <di:waypoint x=\"");
        try self.writeFloat(source_x);
        try self.xml_buffer.appendSlice(self.allocator, "\" y=\"");
        try self.writeFloat(source_y);
        try self.xml_buffer.appendSlice(self.allocator, "\"/>\n");

        try self.xml_buffer.appendSlice(self.allocator, "        <di:waypoint x=\"");
        try self.writeFloat(target_x);
        try self.xml_buffer.appendSlice(self.allocator, "\" y=\"");
        try self.writeFloat(target_y);
        try self.xml_buffer.appendSlice(self.allocator, "\"/>\n");

        try self.xml_buffer.appendSlice(self.allocator, "      </bpmndi:BPMNEdge>\n");
    }

    // ========================================================================
    // XML UTILITIES
    // ========================================================================

    fn writeEscapedXml(self: *BpmnExporter, str: []const u8) !void {
        for (str) |c| {
            switch (c) {
                '<' => try self.xml_buffer.appendSlice(self.allocator, "&lt;"),
                '>' => try self.xml_buffer.appendSlice(self.allocator, "&gt;"),
                '&' => try self.xml_buffer.appendSlice(self.allocator, "&amp;"),
                '"' => try self.xml_buffer.appendSlice(self.allocator, "&quot;"),
                '\'' => try self.xml_buffer.appendSlice(self.allocator, "&apos;"),
                else => try self.xml_buffer.append(self.allocator, c),
            }
        }
    }

    fn writeNumber(self: *BpmnExporter, num: usize) !void {
        var buf: [20]u8 = undefined;
        const str = std.fmt.bufPrint(&buf, "{d}", .{num}) catch return;
        try self.xml_buffer.appendSlice(self.allocator, str);
    }

    fn writeFloat(self: *BpmnExporter, num: f64) !void {
        var buf: [32]u8 = undefined;
        const str = std.fmt.bufPrint(&buf, "{d:.0}", .{num}) catch return;
        try self.xml_buffer.appendSlice(self.allocator, str);
    }
};

/// Map NodeType to BPMN element name
fn nodeTypeToBpmnElement(node_type: NodeType) []const u8 {
    return switch (node_type) {
        .trigger => "startEvent",
        .action, .transform => "task",
        .condition => "exclusiveGateway",
        .split, .join => "parallelGateway",
    };
}

/// Map string node type to BPMN element (for external use)
pub fn mapNodeTypeToBpmn(type_str: []const u8) []const u8 {
    if (std.mem.eql(u8, type_str, "start")) return "startEvent";
    if (std.mem.eql(u8, type_str, "end")) return "endEvent";
    if (std.mem.eql(u8, type_str, "task")) return "task";
    if (std.mem.eql(u8, type_str, "decision")) return "exclusiveGateway";
    if (std.mem.eql(u8, type_str, "parallel")) return "parallelGateway";
    return "task";
}

// ============================================================================
// TESTS
// ============================================================================

test "BpmnExporter init/deinit" {
    const allocator = std.testing.allocator;
    var exporter = BpmnExporter.init(allocator);
    defer exporter.deinit();
}

test "export workflow to BPMN" {
    const allocator = std.testing.allocator;
    var exporter = BpmnExporter.init(allocator);
    defer exporter.deinit();

    var schema = WorkflowSchema.init(allocator);
    schema.version = try allocator.dupe(u8, "1.0");
    schema.name = try allocator.dupe(u8, "TestProcess");
    schema.description = try allocator.dupe(u8, "");

    var nodes = try allocator.alloc(WorkflowNode, 2);
    nodes[0] = WorkflowNode{ .id = try allocator.dupe(u8, "start"), .node_type = .trigger, .name = try allocator.dupe(u8, "Start"), .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) } };
    nodes[1] = WorkflowNode{ .id = try allocator.dupe(u8, "task1"), .node_type = .action, .name = try allocator.dupe(u8, "Task"), .config = std.json.Value{ .object = std.json.ObjectMap.init(allocator) } };
    schema.nodes = nodes;

    var edges = try allocator.alloc(WorkflowEdge, 1);
    edges[0] = WorkflowEdge{ .from = try allocator.dupe(u8, "start"), .to = try allocator.dupe(u8, "task1"), .condition = try allocator.dupe(u8, "${x > 1}") };
    schema.edges = edges;
    defer schema.deinit();

    const xml = try exporter.exportToXml(schema);
    defer allocator.free(xml);

    // Verify BPMN 2.0 structure
    try std.testing.expect(std.mem.indexOf(u8, xml, "<?xml version=\"1.0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "bpmn:definitions") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "bpmn:process") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "bpmn:startEvent") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "bpmn:task") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "bpmn:sequenceFlow") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "bpmndi:BPMNDiagram") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "bpmn:conditionExpression") != null);
    try std.testing.expect(std.mem.indexOf(u8, xml, "${x &gt; 1}") != null);
}

test "XML escaping" {
    const allocator = std.testing.allocator;
    var exporter = BpmnExporter.init(allocator);
    defer exporter.deinit();
    try exporter.writeEscapedXml("Test <>&\"' chars");
    try std.testing.expectEqualStrings("Test &lt;&gt;&amp;&quot;&apos; chars", exporter.xml_buffer.items);
}

test "node type mappings" {
    // Internal NodeType mapping
    try std.testing.expectEqualStrings("startEvent", nodeTypeToBpmnElement(.trigger));
    try std.testing.expectEqualStrings("task", nodeTypeToBpmnElement(.action));
    try std.testing.expectEqualStrings("exclusiveGateway", nodeTypeToBpmnElement(.condition));
    try std.testing.expectEqualStrings("parallelGateway", nodeTypeToBpmnElement(.split));
    // String mapping
    try std.testing.expectEqualStrings("startEvent", mapNodeTypeToBpmn("start"));
    try std.testing.expectEqualStrings("endEvent", mapNodeTypeToBpmn("end"));
    try std.testing.expectEqualStrings("exclusiveGateway", mapNodeTypeToBpmn("decision"));
}
