// nExtract - Document Extraction Engine
// Main library entry point for Zig implementation
//
// This module exports the core functionality of nExtract for FFI consumption
// by the Mojo layer and provides the unified API surface.

const std = @import("std");

// Version information
pub const version = std.SemanticVersion{
    .major = 1,
    .minor = 0,
    .patch = 0,
    .pre = "dev",
};

// Export core modules
pub const core = @import("core/types.zig");

// Export parser modules
pub const csv = @import("parsers/csv.zig");
pub const markdown = @import("parsers/markdown.zig");
pub const xml = @import("parsers/xml.zig");
pub const html = @import("parsers/html.zig");
pub const json = @import("parsers/json.zig");

// Export compression modules
pub const deflate = @import("parsers/deflate.zig");
// pub const zip = @import("parsers/zip.zig");
// pub const gzip = @import("parsers/gzip.zig");

// Export image modules (to be implemented)
// pub const png = @import("parsers/png.zig");
// pub const jpeg = @import("parsers/jpeg.zig");

// Export OCR modules (to be implemented)
// pub const ocr = @import("ocr/ocr.zig");

// Export ML modules (to be implemented)
// pub const ml = @import("ml/tensor.zig");

// Export PDF modules (to be implemented)
// pub const pdf = @import("pdf/objects.zig");

// Library initialization
pub fn init() void {
    // Placeholder for library initialization
    // This will be used for:
    // - Setting up memory allocators
    // - Initializing thread pools
    // - Loading ML models
    // - Setting up logging
}

// Library cleanup
pub fn deinit() void {
    // Placeholder for library cleanup
    // This will be used for:
    // - Freeing memory allocators
    // - Shutting down thread pools
    // - Unloading ML models
    // - Flushing logs
}

// Basic smoke test
test "nExtract library version" {
    const testing = std.testing;
    try testing.expectEqual(@as(u32, 1), version.major);
    try testing.expectEqual(@as(u32, 0), version.minor);
    try testing.expectEqual(@as(u32, 0), version.patch);
}

test "nExtract init/deinit" {
    init();
    deinit();
    // Should not crash
}
