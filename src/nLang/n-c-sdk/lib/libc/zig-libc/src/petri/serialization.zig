//! Petri Net Serialization Module
//! JSON and PNML export/import for Petri nets

const std = @import("std");
const types = @import("types.zig");
const core = @import("core.zig");
const Allocator = std.mem.Allocator;

// Import internal types from core
const PetriNet = core.PetriNet;
const Place = core.Place;
const Transition = core.Transition;
const Arc = core.Arc;

/// Serialization format
pub const SerializationFormat = enum {
    json,
    pnml, // Petri Net Markup Language
    dot, // GraphViz DOT format
};

/// Error types for serialization
pub const SerializationError = error{
    InvalidFormat,
    MissingField,
    InvalidArcType,
    OutOfMemory,
    NotImplemented,
};

/// Helper to get null-terminated string from fixed buffer
fn getStringFromBuffer(buf: *const [256]u8) []const u8 {
    return std.mem.sliceTo(buf, 0);
}

/// Helper to escape JSON strings
fn writeJsonString(writer: anytype, str: []const u8) !void {
    try writer.writeByte('"');
    for (str) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (c < 0x20) {
                    try writer.print("\\u{x:0>4}", .{c});
                } else {
                    try writer.writeByte(c);
                }
            },
        }
    }
    try writer.writeByte('"');
}

/// Helper to escape XML strings
fn writeXmlEscaped(writer: anytype, str: []const u8) !void {
    for (str) |c| {
        switch (c) {
            '<' => try writer.writeAll("&lt;"),
            '>' => try writer.writeAll("&gt;"),
            '&' => try writer.writeAll("&amp;"),
            '"' => try writer.writeAll("&quot;"),
            '\'' => try writer.writeAll("&apos;"),
            else => try writer.writeByte(c),
        }
    }
}

/// Export Petri net to JSON
pub fn exportToJSON(
    allocator: Allocator,
    net: *const anyopaque,
) ![]const u8 {
    const pn: *const PetriNet = @ptrCast(@alignCast(net));

    var json = std.ArrayList(u8).init(allocator);
    errdefer json.deinit();
    const writer = json.writer();

    try writer.writeAll("{\n");
    try writer.writeAll("  \"format\": \"petri-net-json\",\n");
    try writer.writeAll("  \"version\": \"1.0\",\n");

    // Serialize net name
    try writer.writeAll("  \"name\": ");
    try writeJsonString(writer, getStringFromBuffer(&pn.name));
    try writer.writeAll(",\n");

    // Serialize places
    try writer.writeAll("  \"places\": [\n");
    var place_idx: usize = 0;
    var place_it = pn.places.valueIterator();
    while (place_it.next()) |place_ptr| {
        const place = place_ptr.*;
        if (place_idx > 0) try writer.writeAll(",\n");

        const place_id = getStringFromBuffer(&place.id);
        const place_name = getStringFromBuffer(&place.name);
        const token_count = place.tokens.items.len;

        try writer.writeAll("    {\"id\": ");
        try writeJsonString(writer, place_id);
        try writer.writeAll(", \"name\": ");
        try writeJsonString(writer, place_name);
        try writer.print(", \"tokens\": {d}}}", .{token_count});

        place_idx += 1;
    }
    if (place_idx > 0) try writer.writeAll("\n");
    try writer.writeAll("  ],\n");

    // Serialize transitions
    try writer.writeAll("  \"transitions\": [\n");
    var trans_idx: usize = 0;
    var trans_it = pn.transitions.valueIterator();
    while (trans_it.next()) |trans_ptr| {
        const trans = trans_ptr.*;
        if (trans_idx > 0) try writer.writeAll(",\n");

        const trans_id = getStringFromBuffer(&trans.id);
        const trans_name = getStringFromBuffer(&trans.name);

        try writer.writeAll("    {\"id\": ");
        try writeJsonString(writer, trans_id);
        try writer.writeAll(", \"name\": ");
        try writeJsonString(writer, trans_name);
        try writer.print(", \"enabled\": {s}}}", .{if (trans.enabled) "true" else "false"});

        trans_idx += 1;
    }
    if (trans_idx > 0) try writer.writeAll("\n");
    try writer.writeAll("  ],\n");

    // Serialize arcs
    try writer.writeAll("  \"arcs\": [\n");
    for (pn.arcs.items, 0..) |arc, arc_idx| {
        if (arc_idx > 0) try writer.writeAll(",\n");

        const arc_id = getStringFromBuffer(&arc.id);
        const source_id = getStringFromBuffer(&arc.source_id);
        const target_id = getStringFromBuffer(&arc.target_id);
        const arc_type_str = switch (arc.arc_type) {
            .input => "input",
            .output => "output",
            .inhibitor => "inhibitor",
        };

        try writer.writeAll("    {\"id\": ");
        try writeJsonString(writer, arc_id);
        try writer.writeAll(", \"source\": ");
        try writeJsonString(writer, source_id);
        try writer.writeAll(", \"target\": ");
        try writeJsonString(writer, target_id);
        try writer.print(", \"weight\": {d}, \"type\": \"{s}\"}}", .{ arc.weight, arc_type_str });
    }
    if (pn.arcs.items.len > 0) try writer.writeAll("\n");
    try writer.writeAll("  ]\n");

    try writer.writeAll("}\n");

    return json.toOwnedSlice();
}

/// Import Petri net from JSON
pub fn importFromJSON(
    allocator: Allocator,
    json_data: []const u8,
) !*types.pn_net_t {
    _ = allocator;

    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        core.allocator,
        json_data,
        .{},
    );
    defer parsed.deinit();

    const root = parsed.value;

    // Get net name (optional, default to "imported_net")
    const name_val = root.object.get("name");
    const name_str: []const u8 = if (name_val) |nv| switch (nv) {
        .string => |s| s,
        else => "imported_net",
    } else "imported_net";

    // Create a null-terminated name
    var name_buf: [256]u8 = undefined;
    @memset(&name_buf, 0);
    const copy_len = @min(name_str.len, 255);
    @memcpy(name_buf[0..copy_len], name_str[0..copy_len]);

    // Create the net
    const net = core.pn_create(@ptrCast(&name_buf), 0) orelse
        return SerializationError.OutOfMemory;

    // Import places
    if (root.object.get("places")) |places_val| {
        if (places_val == .array) {
            for (places_val.array.items) |place_val| {
                if (place_val != .object) continue;

                const pid = place_val.object.get("id") orelse continue;
                const pname = place_val.object.get("name") orelse continue;

                if (pid != .string or pname != .string) continue;

                var pid_buf: [256]u8 = undefined;
                var pname_buf: [256]u8 = undefined;
                @memset(&pid_buf, 0);
                @memset(&pname_buf, 0);

                const pid_len = @min(pid.string.len, 255);
                const pname_len = @min(pname.string.len, 255);
                @memcpy(pid_buf[0..pid_len], pid.string[0..pid_len]);
                @memcpy(pname_buf[0..pname_len], pname.string[0..pname_len]);

                const place = core.pn_place_create(net, @ptrCast(&pid_buf), @ptrCast(&pname_buf));

                // Add tokens if specified
                if (place_val.object.get("tokens")) |tokens_val| {
                    if (tokens_val == .integer) {
                        const token_count: usize = @intCast(@max(0, tokens_val.integer));
                        for (0..token_count) |_| {
                            const token = core.pn_token_create(null, 0);
                            _ = core.pn_token_put(place, token);
                        }
                    }
                }
            }
        }
    }

    // Import transitions
    if (root.object.get("transitions")) |trans_val| {
        if (trans_val == .array) {
            for (trans_val.array.items) |t_val| {
                if (t_val != .object) continue;

                const tid = t_val.object.get("id") orelse continue;
                const tname = t_val.object.get("name") orelse continue;

                if (tid != .string or tname != .string) continue;

                var tid_buf: [256]u8 = undefined;
                var tname_buf: [256]u8 = undefined;
                @memset(&tid_buf, 0);
                @memset(&tname_buf, 0);

                const tid_len = @min(tid.string.len, 255);
                const tname_len = @min(tname.string.len, 255);
                @memcpy(tid_buf[0..tid_len], tid.string[0..tid_len]);
                @memcpy(tname_buf[0..tname_len], tname.string[0..tname_len]);

                const trans = core.pn_trans_create(net, @ptrCast(&tid_buf), @ptrCast(&tname_buf));

                // Set enabled state if specified
                if (t_val.object.get("enabled")) |enabled_val| {
                    if (enabled_val == .bool) {
                        if (!enabled_val.bool) {
                            _ = core.pn_trans_disable(trans);
                        }
                    }
                }
            }
        }
    }

    // Import arcs
    if (root.object.get("arcs")) |arcs_val| {
        if (arcs_val == .array) {
            for (arcs_val.array.items) |arc_val| {
                if (arc_val != .object) continue;

                const aid = arc_val.object.get("id") orelse continue;
                const source = arc_val.object.get("source") orelse continue;
                const target = arc_val.object.get("target") orelse continue;

                if (aid != .string or source != .string or target != .string) continue;

                // Determine arc type
                var arc_type: types.pn_arc_type_t = .input;
                if (arc_val.object.get("type")) |type_val| {
                    if (type_val == .string) {
                        if (std.mem.eql(u8, type_val.string, "output")) {
                            arc_type = .output;
                        } else if (std.mem.eql(u8, type_val.string, "inhibitor")) {
                            arc_type = .inhibitor;
                        }
                    }
                }

                var aid_buf: [256]u8 = undefined;
                var src_buf: [256]u8 = undefined;
                var tgt_buf: [256]u8 = undefined;
                @memset(&aid_buf, 0);
                @memset(&src_buf, 0);
                @memset(&tgt_buf, 0);

                const aid_len = @min(aid.string.len, 255);
                const src_len = @min(source.string.len, 255);
                const tgt_len = @min(target.string.len, 255);
                @memcpy(aid_buf[0..aid_len], aid.string[0..aid_len]);
                @memcpy(src_buf[0..src_len], source.string[0..src_len]);
                @memcpy(tgt_buf[0..tgt_len], target.string[0..tgt_len]);

                const arc = core.pn_arc_create(net, @ptrCast(&aid_buf), arc_type);
                _ = core.pn_arc_connect(arc, @ptrCast(&src_buf), @ptrCast(&tgt_buf));

                // Set weight if specified
                if (arc_val.object.get("weight")) |weight_val| {
                    if (weight_val == .integer) {
                        _ = core.pn_arc_set_weight(arc, @intCast(@max(1, weight_val.integer)));
                    }
                }
            }
        }
    }

    return net;
}

/// Export to PNML format
pub fn exportToPNML(
    allocator: Allocator,
    net: *const anyopaque,
) ![]const u8 {
    const pn: *const PetriNet = @ptrCast(@alignCast(net));

    var xml = std.ArrayList(u8).init(allocator);
    errdefer xml.deinit();
    const writer = xml.writer();

    try writer.writeAll("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    try writer.writeAll("<pnml xmlns=\"http://www.pnml.org/version-2009/grammar/pnml\">\n");

    // Net element with id from name
    try writer.writeAll("  <net id=\"");
    try writeXmlEscaped(writer, getStringFromBuffer(&pn.name));
    try writer.writeAll("\" type=\"http://www.pnml.org/version-2009/grammar/ptnet\">\n");
    try writer.writeAll("    <page id=\"page1\">\n");

    // Serialize places
    var place_it = pn.places.valueIterator();
    while (place_it.next()) |place_ptr| {
        const place = place_ptr.*;
        const place_id = getStringFromBuffer(&place.id);
        const place_name = getStringFromBuffer(&place.name);
        const token_count = place.tokens.items.len;

        try writer.writeAll("      <place id=\"");
        try writeXmlEscaped(writer, place_id);
        try writer.writeAll("\">\n");
        try writer.writeAll("        <name>\n");
        try writer.writeAll("          <text>");
        try writeXmlEscaped(writer, place_name);
        try writer.writeAll("</text>\n");
        try writer.writeAll("        </name>\n");
        try writer.print("        <initialMarking>\n          <text>{d}</text>\n        </initialMarking>\n", .{token_count});
        try writer.writeAll("      </place>\n");
    }

    // Serialize transitions
    var trans_it = pn.transitions.valueIterator();
    while (trans_it.next()) |trans_ptr| {
        const trans = trans_ptr.*;
        const trans_id = getStringFromBuffer(&trans.id);
        const trans_name = getStringFromBuffer(&trans.name);

        try writer.writeAll("      <transition id=\"");
        try writeXmlEscaped(writer, trans_id);
        try writer.writeAll("\">\n");
        try writer.writeAll("        <name>\n");
        try writer.writeAll("          <text>");
        try writeXmlEscaped(writer, trans_name);
        try writer.writeAll("</text>\n");
        try writer.writeAll("        </name>\n");
        try writer.writeAll("      </transition>\n");
    }

    // Serialize arcs
    for (pn.arcs.items) |arc| {
        const arc_id = getStringFromBuffer(&arc.id);
        const source_id = getStringFromBuffer(&arc.source_id);
        const target_id = getStringFromBuffer(&arc.target_id);

        try writer.writeAll("      <arc id=\"");
        try writeXmlEscaped(writer, arc_id);
        try writer.writeAll("\" source=\"");
        try writeXmlEscaped(writer, source_id);
        try writer.writeAll("\" target=\"");
        try writeXmlEscaped(writer, target_id);
        try writer.writeAll("\">\n");
        try writer.print("        <inscription>\n          <text>{d}</text>\n        </inscription>\n", .{arc.weight});
        try writer.writeAll("      </arc>\n");
    }

    try writer.writeAll("    </page>\n");
    try writer.writeAll("  </net>\n");
    try writer.writeAll("</pnml>\n");

    return xml.toOwnedSlice();
}

/// Export to DOT (GraphViz) format
pub fn exportToDOT(
    allocator: Allocator,
    net: *const anyopaque,
) ![]const u8 {
    const pn: *const PetriNet = @ptrCast(@alignCast(net));

    var dot = std.ArrayList(u8).init(allocator);
    errdefer dot.deinit();
    const writer = dot.writer();

    try writer.writeAll("digraph PetriNet {\n");
    try writer.writeAll("  rankdir=LR;\n");
    try writer.writeAll("  \n");

    // Places as circles (double circle for places with tokens)
    try writer.writeAll("  // Places\n");
    var place_it = pn.places.valueIterator();
    while (place_it.next()) |place_ptr| {
        const place = place_ptr.*;
        const place_id = getStringFromBuffer(&place.id);
        const place_name = getStringFromBuffer(&place.name);
        const token_count = place.tokens.items.len;

        // Include token count in label if > 0
        try writer.writeAll("  ");
        try writer.writeAll(place_id);
        try writer.writeAll(" [label=\"");
        try writer.writeAll(place_name);
        if (token_count > 0) {
            try writer.print("\\n({d})", .{token_count});
        }
        try writer.writeAll("\" shape=circle");
        if (token_count > 0) {
            try writer.writeAll(" style=filled fillcolor=lightblue");
        }
        try writer.writeAll("];\n");
    }

    try writer.writeAll("  \n");

    // Transitions as boxes
    try writer.writeAll("  // Transitions\n");
    var trans_it = pn.transitions.valueIterator();
    while (trans_it.next()) |trans_ptr| {
        const trans = trans_ptr.*;
        const trans_id = getStringFromBuffer(&trans.id);
        const trans_name = getStringFromBuffer(&trans.name);

        try writer.writeAll("  ");
        try writer.writeAll(trans_id);
        try writer.writeAll(" [label=\"");
        try writer.writeAll(trans_name);
        try writer.writeAll("\" shape=box");
        if (!trans.enabled) {
            try writer.writeAll(" style=filled fillcolor=gray");
        }
        try writer.writeAll("];\n");
    }

    try writer.writeAll("  \n");

    // Arcs as edges
    try writer.writeAll("  // Arcs\n");
    for (pn.arcs.items) |arc| {
        const source_id = getStringFromBuffer(&arc.source_id);
        const target_id = getStringFromBuffer(&arc.target_id);

        try writer.writeAll("  ");
        try writer.writeAll(source_id);
        try writer.writeAll(" -> ");
        try writer.writeAll(target_id);
        try writer.writeAll(" [label=\"");
        try writer.print("{d}", .{arc.weight});
        try writer.writeAll("\"");

        // Style inhibitor arcs differently
        if (arc.arc_type == .inhibitor) {
            try writer.writeAll(" arrowhead=odot style=dashed");
        }

        try writer.writeAll("];\n");
    }

    try writer.writeAll("}\n");

    return dot.toOwnedSlice();
}
