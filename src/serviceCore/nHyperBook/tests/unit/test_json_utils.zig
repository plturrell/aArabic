// ============================================================================
// HyperShimmy Unit Tests - JSON Utilities
// ============================================================================
// Day 56: Comprehensive unit tests for JSON serialization/deserialization
// ============================================================================

const std = @import("std");
const testing = std.testing;
const json_utils = @import("../../server/json_utils.zig");
const sources = @import("../../server/sources.zig");

// ============================================================================
// JSON Escaping Tests
// ============================================================================

test "JSON escape - special characters" {
    const allocator = testing.allocator;
    var sanitizer = json_utils.Sanitizer.init(allocator);
    
    const input = "Text with \"quotes\" and \n newlines";
    const output = try json_utils.escapeJson(allocator, input);
    defer allocator.free(output);
    
    try testing.expect(std.mem.indexOf(u8, output, "\\\"") != null);
    try testing.expect(std.mem.indexOf(u8, output, "\\n") != null);
}

test "JSON escape - backslashes" {
    const allocator = testing.allocator;
    
    const input = "Path\\with\\backslashes";
    const output = try json_utils.escapeJson(allocator, input);
    defer allocator.free(output);
    
    try testing.expect(std.mem.indexOf(u8, output, "\\\\") != null);
}

test "JSON escape - tabs and carriage returns" {
    const allocator = testing.allocator;
    
    const input = "Text\twith\ttabs\rand\rreturns";
    const output = try json_utils.escapeJson(allocator, input);
    defer allocator.free(output);
    
    try testing.expect(std.mem.indexOf(u8, output, "\\t") != null);
    try testing.expect(std.mem.indexOf(u8, output, "\\r") != null);
}

test "JSON escape - no special characters" {
    const allocator = testing.allocator;
    
    const input = "Normal text without special characters";
    const output = try json_utils.escapeJson(allocator, input);
    defer allocator.free(output);
    
    try testing.expectEqualStrings(input, output);
}

test "JSON escape - empty string" {
    const allocator = testing.allocator;
    
    const input = "";
    const output = try json_utils.escapeJson(allocator, input);
    defer allocator.free(output);
    
    try testing.expectEqualStrings("", output);
}

// ============================================================================
// Source Serialization Tests
// ============================================================================

test "serializeSource - basic fields" {
    const allocator = testing.allocator;
    
    const source = try sources.Source.init(
        allocator,
        "src-001",
        "Test Document",
        .pdf,
        "https://example.com/doc.pdf",
        "Sample content",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);
    
    const json = try json_utils.serializeSource(allocator, source);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"src-001\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"Title\": \"Test Document\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"SourceType\": \"PDF\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"Status\": \"Ready\"") != null);
}

test "serializeSource - URL source type" {
    const allocator = testing.allocator;
    
    const source = try sources.Source.init(
        allocator,
        "src-002",
        "Web Article",
        .url,
        "https://example.com/article",
        "Article content",
        .processing,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);
    
    const json = try json_utils.serializeSource(allocator, source);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "\"SourceType\": \"URL\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"Status\": \"Processing\"") != null);
}

test "serializeSource - with special characters" {
    const allocator = testing.allocator;
    
    const source = try sources.Source.init(
        allocator,
        "src-003",
        "Title with \"quotes\" and \n newlines",
        .text,
        "https://example.com/path?query=value&other=123",
        "Content with special: \n\r\t\"'",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);
    
    const json = try json_utils.serializeSource(allocator, source);
    defer allocator.free(json);
    
    // Should properly escape special characters
    try testing.expect(std.mem.indexOf(u8, json, "\\\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\\n") != null);
}

test "serializeSource - empty content" {
    const allocator = testing.allocator;
    
    const source = try sources.Source.init(
        allocator,
        "src-004",
        "Empty Source",
        .url,
        "https://example.com",
        "",
        .pending,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);
    
    const json = try json_utils.serializeSource(allocator, source);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "\"Content\": \"\"") != null);
}

// ============================================================================
// Source Array Serialization Tests
// ============================================================================

test "serializeSourceArray - empty array" {
    const allocator = testing.allocator;
    
    const sources_array: []const sources.Source = &[_]sources.Source{};
    const json = try json_utils.serializeSourceArray(allocator, sources_array);
    defer allocator.free(json);
    
    try testing.expectEqualStrings("[]", json);
}

test "serializeSourceArray - single source" {
    const allocator = testing.allocator;
    
    const source = try sources.Source.init(
        allocator,
        "src-001",
        "Test Source",
        .url,
        "https://example.com",
        "Content",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);
    
    const sources_array = [_]sources.Source{source};
    const json = try json_utils.serializeSourceArray(allocator, &sources_array);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"src-001\"") != null);
    try testing.expect(std.mem.startsWith(u8, json, "[\n"));
    try testing.expect(std.mem.endsWith(u8, json, "\n]"));
}

test "serializeSourceArray - multiple sources" {
    const allocator = testing.allocator;
    
    var source_list = [_]sources.Source{
        try sources.Source.init(
            allocator,
            "src-001",
            "Source 1",
            .url,
            "https://example.com/1",
            "Content 1",
            .ready,
            "2026-01-16T12:00:00Z",
            "2026-01-16T12:00:00Z",
        ),
        try sources.Source.init(
            allocator,
            "src-002",
            "Source 2",
            .pdf,
            "https://example.com/2",
            "Content 2",
            .processing,
            "2026-01-16T12:00:00Z",
            "2026-01-16T12:00:00Z",
        ),
        try sources.Source.init(
            allocator,
            "src-003",
            "Source 3",
            .text,
            "https://example.com/3",
            "Content 3",
            .pending,
            "2026-01-16T12:00:00Z",
            "2026-01-16T12:00:00Z",
        ),
    };
    defer for (source_list) |s| s.deinit(allocator);
    
    const json = try json_utils.serializeSourceArray(allocator, &source_list);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"src-001\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"src-002\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"src-003\"") != null);
}

// ============================================================================
// OData Response Serialization Tests
// ============================================================================

test "serializeODataResponse - empty collection" {
    const allocator = testing.allocator;
    
    const sources_array: []const sources.Source = &[_]sources.Source{};
    const json = try json_utils.serializeODataResponse(allocator, sources_array);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "\"@odata.context\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"value\": []") != null);
}

test "serializeODataResponse - with sources" {
    const allocator = testing.allocator;
    
    var source_list = [_]sources.Source{
        try sources.Source.init(
            allocator,
            "src-001",
            "Test Source",
            .url,
            "https://example.com",
            "Content",
            .ready,
            "2026-01-16T12:00:00Z",
            "2026-01-16T12:00:00Z",
        ),
    };
    defer for (source_list) |s| s.deinit(allocator);
    
    const json = try json_utils.serializeODataResponse(allocator, &source_list);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "\"@odata.context\": \"/odata/v4/research/$metadata#Sources\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"value\":") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"Id\": \"src-001\"") != null);
}

test "serializeODataResponse - metadata URL format" {
    const allocator = testing.allocator;
    
    const sources_array: []const sources.Source = &[_]sources.Source{};
    const json = try json_utils.serializeODataResponse(allocator, sources_array);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "$metadata#Sources") != null);
}

// ============================================================================
// JSON Parsing Tests
// ============================================================================

test "parseSourceJson - basic fields" {
    const allocator = testing.allocator;
    
    const json_input =
        \\{
        \\  "Title": "Test Document",
        \\  "SourceType": "PDF",
        \\  "Url": "https://example.com/doc.pdf",
        \\  "Content": "Sample content"
        \\}
    ;
    
    const parsed = try json_utils.parseSourceJson(allocator, json_input);
    
    try testing.expectEqualStrings("Test Document", parsed.title);
    try testing.expectEqualStrings("PDF", parsed.source_type);
    try testing.expectEqualStrings("https://example.com/doc.pdf", parsed.url);
    try testing.expectEqualStrings("Sample content", parsed.content);
}

test "parseSourceJson - missing fields" {
    const allocator = testing.allocator;
    
    const json_input =
        \\{
        \\  "Title": "Partial Data"
        \\}
    ;
    
    const parsed = try json_utils.parseSourceJson(allocator, json_input);
    
    try testing.expectEqualStrings("Partial Data", parsed.title);
    try testing.expectEqualStrings("URL", parsed.source_type); // default
    try testing.expectEqualStrings("", parsed.url);
    try testing.expectEqualStrings("", parsed.content);
}

test "parseSourceJson - empty JSON" {
    const allocator = testing.allocator;
    
    const json_input = "{}";
    const parsed = try json_utils.parseSourceJson(allocator, json_input);
    
    try testing.expectEqualStrings("", parsed.title);
    try testing.expectEqualStrings("URL", parsed.source_type);
}

test "parseSourceJson - URL source type" {
    const allocator = testing.allocator;
    
    const json_input =
        \\{
        \\  "Title": "Web Article",
        \\  "SourceType": "URL",
        \\  "Url": "https://example.com/article"
        \\}
    ;
    
    const parsed = try json_utils.parseSourceJson(allocator, json_input);
    
    try testing.expectEqualStrings("Web Article", parsed.title);
    try testing.expectEqualStrings("URL", parsed.source_type);
    try testing.expectEqualStrings("https://example.com/article", parsed.url);
}

// ============================================================================
// Round-trip Tests
// ============================================================================

test "round-trip serialization" {
    const allocator = testing.allocator;
    
    // Create a source
    const original = try sources.Source.init(
        allocator,
        "src-roundtrip",
        "Round Trip Test",
        .pdf,
        "https://example.com/test.pdf",
        "Test content",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer original.deinit(allocator);
    
    // Serialize to JSON
    const json = try json_utils.serializeSource(allocator, original);
    defer allocator.free(json);
    
    // Parse back
    const parsed = try json_utils.parseSourceJson(allocator, json);
    
    // Verify fields match
    try testing.expectEqualStrings(original.title, parsed.title);
    try testing.expectEqualStrings(original.url, parsed.url);
    try testing.expectEqualStrings(original.content, parsed.content);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

test "serialize - very long content" {
    const allocator = testing.allocator;
    
    // Create large content (100KB)
    const large_content = try allocator.alloc(u8, 100 * 1024);
    defer allocator.free(large_content);
    @memset(large_content, 'A');
    
    const source = try sources.Source.init(
        allocator,
        "src-large",
        "Large Content",
        .text,
        "https://example.com",
        large_content,
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);
    
    const json = try json_utils.serializeSource(allocator, source);
    defer allocator.free(json);
    
    // Should still serialize successfully
    try testing.expect(json.len > 100 * 1024);
}

test "serialize - unicode characters" {
    const allocator = testing.allocator;
    
    const source = try sources.Source.init(
        allocator,
        "src-unicode",
        "Title with 中文 and العربية",
        .text,
        "https://example.com",
        "Content with émojis 🚀 and हिन्दी",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);
    
    const json = try json_utils.serializeSource(allocator, source);
    defer allocator.free(json);
    
    // Unicode should be preserved
    try testing.expect(std.mem.indexOf(u8, json, "中文") != null);
    try testing.expect(std.mem.indexOf(u8, json, "🚀") != null);
}

test "parse - malformed JSON" {
    const allocator = testing.allocator;
    
    const malformed = "{ not valid json";
    
    // Should still attempt to parse (simplified parser)
    const parsed = try json_utils.parseSourceJson(allocator, malformed);
    
    // Should return defaults for missing fields
    try testing.expectEqualStrings("", parsed.title);
}

test "array serialization - many sources" {
    const allocator = testing.allocator;
    
    // Create 100 sources
    var source_list = std.ArrayList(sources.Source).init(allocator);
    defer {
        for (source_list.items) |s| s.deinit(allocator);
        source_list.deinit();
    }
    
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "src-{d:0>3}", .{i});
        defer allocator.free(id);
        
        const source = try sources.Source.init(
            allocator,
            id,
            "Test Source",
            .url,
            "https://example.com",
            "Content",
            .ready,
            "2026-01-16T12:00:00Z",
            "2026-01-16T12:00:00Z",
        );
        try source_list.append(source);
    }
    
    const json = try json_utils.serializeSourceArray(allocator, source_list.items);
    defer allocator.free(json);
    
    // Should contain all sources
    try testing.expect(std.mem.indexOf(u8, json, "src-000") != null);
    try testing.expect(std.mem.indexOf(u8, json, "src-099") != null);
}
