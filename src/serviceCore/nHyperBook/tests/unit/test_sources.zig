// ============================================================================
// HyperShimmy Unit Tests - Source Management
// ============================================================================
// Day 56: Comprehensive unit tests for source entity operations
// ============================================================================

const std = @import("std");
const testing = std.testing;
const sources = @import("../../server/sources.zig");

// ============================================================================
// Source Creation Tests
// ============================================================================

test "Source creation with valid data" {
    const allocator = testing.allocator;

    const source = try sources.Source.init(
        allocator,
        "test-001",
        "Test Document",
        .pdf,
        "https://example.com/doc.pdf",
        "Sample content",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);

    try testing.expectEqualStrings("test-001", source.id);
    try testing.expectEqualStrings("Test Document", source.title);
    try testing.expectEqual(sources.SourceType.pdf, source.source_type);
    try testing.expectEqualStrings("https://example.com/doc.pdf", source.url);
    try testing.expectEqualStrings("Sample content", source.content);
    try testing.expectEqual(sources.SourceStatus.ready, source.status);
}

test "Source creation with empty strings" {
    const allocator = testing.allocator;

    const source = try sources.Source.init(
        allocator,
        "",
        "",
        .url,
        "",
        "",
        .pending,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);

    try testing.expectEqualStrings("", source.id);
    try testing.expectEqualStrings("", source.title);
}

test "Source creation with special characters" {
    const allocator = testing.allocator;

    const source = try sources.Source.init(
        allocator,
        "test-\n\t\"special\"",
        "Title with \"quotes\" and \n newlines",
        .text,
        "https://example.com/path?query=value&other=123",
        "Content with special chars: \n\r\t\"'",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);

    try testing.expect(std.mem.indexOf(u8, source.title, "quotes") != null);
    try testing.expect(std.mem.indexOf(u8, source.url, "query=value") != null);
}

// ============================================================================
// Source Type Tests
// ============================================================================

test "SourceType toString conversion" {
    try testing.expectEqualStrings("URL", sources.SourceType.url.toString());
    try testing.expectEqualStrings("PDF", sources.SourceType.pdf.toString());
    try testing.expectEqualStrings("Text", sources.SourceType.text.toString());
    try testing.expectEqualStrings("YouTube", sources.SourceType.youtube.toString());
    try testing.expectEqualStrings("File", sources.SourceType.file.toString());
}

test "SourceType fromString conversion" {
    try testing.expectEqual(sources.SourceType.url, try sources.SourceType.fromString("URL"));
    try testing.expectEqual(sources.SourceType.pdf, try sources.SourceType.fromString("PDF"));
    try testing.expectEqual(sources.SourceType.text, try sources.SourceType.fromString("Text"));
    try testing.expectEqual(sources.SourceType.youtube, try sources.SourceType.fromString("YouTube"));
    try testing.expectEqual(sources.SourceType.file, try sources.SourceType.fromString("File"));
}

test "SourceType fromString case insensitive" {
    try testing.expectEqual(sources.SourceType.url, try sources.SourceType.fromString("url"));
    try testing.expectEqual(sources.SourceType.pdf, try sources.SourceType.fromString("pdf"));
    try testing.expectEqual(sources.SourceType.text, try sources.SourceType.fromString("text"));
}

test "SourceType fromString invalid" {
    try testing.expectError(error.InvalidSourceType, sources.SourceType.fromString("invalid"));
    try testing.expectError(error.InvalidSourceType, sources.SourceType.fromString(""));
    try testing.expectError(error.InvalidSourceType, sources.SourceType.fromString("PD"));
}

// ============================================================================
// Source Status Tests
// ============================================================================

test "SourceStatus toString conversion" {
    try testing.expectEqualStrings("Pending", sources.SourceStatus.pending.toString());
    try testing.expectEqualStrings("Processing", sources.SourceStatus.processing.toString());
    try testing.expectEqualStrings("Ready", sources.SourceStatus.ready.toString());
    try testing.expectEqualStrings("Error", sources.SourceStatus.error_status.toString());
}

test "SourceStatus fromString conversion" {
    try testing.expectEqual(sources.SourceStatus.pending, try sources.SourceStatus.fromString("Pending"));
    try testing.expectEqual(sources.SourceStatus.processing, try sources.SourceStatus.fromString("Processing"));
    try testing.expectEqual(sources.SourceStatus.ready, try sources.SourceStatus.fromString("Ready"));
    try testing.expectEqual(sources.SourceStatus.error_status, try sources.SourceStatus.fromString("Error"));
}

test "SourceStatus fromString invalid" {
    try testing.expectError(error.InvalidSourceStatus, sources.SourceStatus.fromString("invalid"));
    try testing.expectError(error.InvalidSourceStatus, sources.SourceStatus.fromString(""));
}

// ============================================================================
// Source Manager Tests
// ============================================================================

test "SourceManager initialization" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    try testing.expectEqual(@as(usize, 0), manager.sources.items.len);
}

test "SourceManager add and retrieve source" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    const source = try sources.Source.init(
        allocator,
        "test-001",
        "Test Source",
        .url,
        "https://example.com",
        "Content",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );

    try manager.addSource(source);
    try testing.expectEqual(@as(usize, 1), manager.sources.items.len);

    const retrieved = manager.getSource("test-001");
    try testing.expect(retrieved != null);
    try testing.expectEqualStrings("test-001", retrieved.?.id);
}

test "SourceManager get non-existent source" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    const result = manager.getSource("non-existent");
    try testing.expect(result == null);
}

test "SourceManager delete source" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    const source = try sources.Source.init(
        allocator,
        "test-001",
        "Test Source",
        .url,
        "https://example.com",
        "Content",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );

    try manager.addSource(source);
    try testing.expectEqual(@as(usize, 1), manager.sources.items.len);

    const deleted = try manager.deleteSource("test-001");
    try testing.expect(deleted);
    try testing.expectEqual(@as(usize, 0), manager.sources.items.len);
}

test "SourceManager delete non-existent source" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    const deleted = try manager.deleteSource("non-existent");
    try testing.expect(!deleted);
}

test "SourceManager multiple sources" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    // Add multiple sources
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "test-{d:0>3}", .{i});
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
        try manager.addSource(source);
    }

    try testing.expectEqual(@as(usize, 5), manager.sources.items.len);
}

test "SourceManager list all sources" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    // Add sources
    const source1 = try sources.Source.init(
        allocator,
        "id-1",
        "Source 1",
        .url,
        "https://example.com/1",
        "Content 1",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    try manager.addSource(source1);

    const source2 = try sources.Source.init(
        allocator,
        "id-2",
        "Source 2",
        .pdf,
        "https://example.com/2",
        "Content 2",
        .processing,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    try manager.addSource(source2);

    const all_sources = manager.listSources();
    try testing.expectEqual(@as(usize, 2), all_sources.len);
}

// ============================================================================
// Source Update Tests
// ============================================================================

test "Source update status" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    const source = try sources.Source.init(
        allocator,
        "test-001",
        "Test Source",
        .url,
        "https://example.com",
        "Content",
        .pending,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    try manager.addSource(source);

    try manager.updateSourceStatus("test-001", .ready);

    const updated = manager.getSource("test-001");
    try testing.expect(updated != null);
    try testing.expectEqual(sources.SourceStatus.ready, updated.?.status);
}

test "Source update non-existent" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);
    defer manager.deinit();

    try testing.expectError(error.SourceNotFound, manager.updateSourceStatus("non-existent", .ready));
}

// ============================================================================
// Edge Cases and Error Conditions
// ============================================================================

test "Source with very long content" {
    const allocator = testing.allocator;

    // Create a large content string (10KB)
    const large_content = try allocator.alloc(u8, 10 * 1024);
    defer allocator.free(large_content);
    @memset(large_content, 'A');

    const source = try sources.Source.init(
        allocator,
        "test-large",
        "Large Content Test",
        .text,
        "https://example.com",
        large_content,
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);

    try testing.expectEqual(@as(usize, 10 * 1024), source.content.len);
}

test "Source with unicode characters" {
    const allocator = testing.allocator;

    const source = try sources.Source.init(
        allocator,
        "test-unicode",
        "Title with émojis 🚀 and 中文",
        .text,
        "https://example.com/unicode",
        "Content with العربية and हिन्दी",
        .ready,
        "2026-01-16T12:00:00Z",
        "2026-01-16T12:00:00Z",
    );
    defer source.deinit(allocator);

    try testing.expect(std.mem.indexOf(u8, source.title, "🚀") != null);
    try testing.expect(std.mem.indexOf(u8, source.content, "العربية") != null);
}

test "SourceManager memory cleanup" {
    const allocator = testing.allocator;
    var manager = sources.SourceManager.init(allocator);

    // Add many sources
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "test-{d:0>3}", .{i});
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
        try manager.addSource(source);
    }

    // Cleanup should not leak memory
    manager.deinit();
}
