///! JSON serialization utilities for OData responses
///! Simple JSON generation for Source entities

const std = @import("std");
const sources = @import("sources.zig");

/// Escape a string for JSON
fn escapeJson(allocator: std.mem.Allocator, s: []const u8) ![]const u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();

    for (s) |c| {
        switch (c) {
            '"' => try list.appendSlice("\\\""),
            '\\' => try list.appendSlice("\\\\"),
            '\n' => try list.appendSlice("\\n"),
            '\r' => try list.appendSlice("\\r"),
            '\t' => try list.appendSlice("\\t"),
            else => try list.append(c),
        }
    }

    return try list.toOwnedSlice();
}

/// Serialize a single source to JSON
pub fn serializeSource(allocator: std.mem.Allocator, source: sources.Source) ![]const u8 {
    const escaped_title = try escapeJson(allocator, source.title);
    defer allocator.free(escaped_title);
    
    const escaped_url = try escapeJson(allocator, source.url);
    defer allocator.free(escaped_url);
    
    const escaped_content = try escapeJson(allocator, source.content);
    defer allocator.free(escaped_content);

    return try std.fmt.allocPrint(allocator,
        \\{{
        \\  "Id": "{s}",
        \\  "Title": "{s}",
        \\  "SourceType": "{s}",
        \\  "Url": "{s}",
        \\  "Content": "{s}",
        \\  "Status": "{s}",
        \\  "CreatedAt": "{s}",
        \\  "UpdatedAt": "{s}"
        \\}}
    , .{
        source.id,
        escaped_title,
        source.source_type.toString(),
        escaped_url,
        escaped_content,
        source.status.toString(),
        source.created_at,
        source.updated_at,
    });
}

/// Serialize multiple sources to JSON array
pub fn serializeSourceArray(allocator: std.mem.Allocator, source_list: []const sources.Source) ![]const u8 {
    if (source_list.len == 0) {
        return try allocator.dupe(u8, "[]");
    }

    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();

    try list.appendSlice("[\n");

    for (source_list, 0..) |source, i| {
        const json = try serializeSource(allocator, source);
        defer allocator.free(json);

        try list.appendSlice(json);

        if (i < source_list.len - 1) {
            try list.appendSlice(",\n");
        } else {
            try list.appendSlice("\n");
        }
    }

    try list.appendSlice("]");

    return try list.toOwnedSlice();
}

/// Serialize OData response with value array
pub fn serializeODataResponse(allocator: std.mem.Allocator, source_list: []const sources.Source) ![]const u8 {
    const value_json = try serializeSourceArray(allocator, source_list);
    defer allocator.free(value_json);

    return try std.fmt.allocPrint(allocator,
        \\{{
        \\  "@odata.context": "/odata/v4/research/$metadata#Sources",
        \\  "value": {s}
        \\}}
    , .{value_json});
}

/// Parse JSON to extract source fields (simplified parser)
pub const ParsedSource = struct {
    title: []const u8,
    source_type: []const u8,
    url: []const u8,
    content: []const u8,
};

/// Simple JSON field extraction (not a full parser, just for MVP)
pub fn parseSourceJson(allocator: std.mem.Allocator, json: []const u8) !ParsedSource {
    _ = allocator;
    
    var result = ParsedSource{
        .title = "",
        .source_type = "URL",
        .url = "",
        .content = "",
    };

    // Very simple field extraction (would use proper JSON parser in production)
    // This is a placeholder for Day 7 - will be improved later
    
    if (std.mem.indexOf(u8, json, "\"Title\":")) |idx| {
        const start = idx + 9; // Skip "Title":"
        if (std.mem.indexOfPos(u8, json, start, "\"")) |end| {
            result.title = json[start..end];
        }
    }

    if (std.mem.indexOf(u8, json, "\"SourceType\":")) |idx| {
        const start = idx + 14; // Skip "SourceType":"
        if (std.mem.indexOfPos(u8, json, start, "\"")) |end| {
            result.source_type = json[start..end];
        }
    }

    if (std.mem.indexOf(u8, json, "\"Url\":")) |idx| {
        const start = idx + 7; // Skip "Url":"
        if (std.mem.indexOfPos(u8, json, start, "\"")) |end| {
            result.url = json[start..end];
        }
    }

    if (std.mem.indexOf(u8, json, "\"Content\":")) |idx| {
        const start = idx + 11; // Skip "Content":"
        if (std.mem.indexOfPos(u8, json, start, "\"")) |end| {
            result.content = json[start..end];
        }
    }

    return result;
}

// Tests
test "Serialize single source" {
    const allocator = std.testing.allocator;

    const source = try sources.Source.init(
        allocator,
        "test_id",
        "Test Source",
        .url,
        "https://example.com",
        "Test content",
        .ready,
        "2026-01-16T00:00:00Z",
        "2026-01-16T00:00:00Z",
    );
    defer source.deinit(allocator);

    const json = try serializeSource(allocator, source);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"test_id\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"Title\": \"Test Source\"") != null);
}

test "Serialize source array" {
    const allocator = std.testing.allocator;

    var source_list = [_]sources.Source{
        try sources.Source.init(
            allocator,
            "id1",
            "Source 1",
            .url,
            "https://example.com/1",
            "Content 1",
            .ready,
            "2026-01-16T00:00:00Z",
            "2026-01-16T00:00:00Z",
        ),
        try sources.Source.init(
            allocator,
            "id2",
            "Source 2",
            .pdf,
            "https://example.com/2",
            "Content 2",
            .ready,
            "2026-01-16T00:00:00Z",
            "2026-01-16T00:00:00Z",
        ),
    };
    defer for (source_list) |s| s.deinit(allocator);

    const json = try serializeSourceArray(allocator, &source_list);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"id1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"id2\"") != null);
}
