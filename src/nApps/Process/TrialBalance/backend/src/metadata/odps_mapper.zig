//! ============================================================================
//! ODPS v4.1 Mapper - Convert between ODPS, ORD, and CSN formats
//! Enables dual metadata compliance for vendor-neutral and SAP-specific catalogs
//! ============================================================================
//!
//! [CODE:file=odps_mapper.zig]
//! [CODE:module=metadata]
//! [CODE:language=zig]
//!
//! [ODPS:product=*]
//! Note: This module processes all ODPS products for metadata conversion.
//!
//! [TABLE:reads=ODPS_YAML_FILES]
//! [TABLE:writes=ODPS_YAML_FILES,ORD_JSON_FILES,CSN_JSON_FILES]
//!
//! [RELATION:called_by=CODE:odps_api.zig]
//! [RELATION:called_by=CODE:odps_quality_service.zig]
//! [RELATION:called_by=CODE:odps_petrinet_bridge.zig]
//! [RELATION:calls=CODE:yaml_parser.zig]
//!
//! This module provides ODPS↔ORD↔CSN conversion and YAML reading/writing.
//! It is infrastructure code that does not implement ODPS business rules directly.

const std = @import("std");
const Allocator = std.mem.Allocator;
const yaml_parser = @import("yaml_parser");

/// ODPS Product (simplified representation)
pub const ODPSProduct = struct {
    product_id: []const u8,
    name: []const u8,
    description: []const u8,
    version: []const u8,
    status: []const u8,
    quality_score: f64,
    
    // References to SAP artifacts
    ord_ref: ?[]const u8,
    csn_ref: ?[]const u8,
};

/// ORD Document (simplified representation)
pub const ORDDocument = struct {
    ord_id: []const u8,
    title: []const u8,
    description: []const u8,
    version: []const u8,
    product_type: []const u8,
    category: []const u8,
};

/// CSN Schema (simplified representation)
pub const CSNSchema = struct {
    entity_name: []const u8,
    title: []const u8,
    description: []const u8,
    elements: []const u8, // JSON string of elements
    keys: []const []const u8,
};

/// Load ODPS YAML file and parse into struct
pub fn loadODPS(allocator: Allocator, file_path: []const u8) !ODPSProduct {
    // Read file
    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(file_content);
    
    // Parse YAML
    var parser = try yaml_parser.YAMLParser.init(allocator);
    defer parser.deinit();
    try parser.parse(file_content);
    
    // Extract ODPS fields
    const product_id = parser.getValue("product.productID") orelse return error.MissingRequiredField;
    const name = parser.getValue("product.name") orelse return error.MissingRequiredField;
    const description = parser.getValue("product.description") orelse return error.MissingRequiredField;
    const version = parser.getValue("product.version") orelse "1.0.0";
    const status = parser.getValue("product.status") orelse "active";
    
    const quality_str = parser.getValue("product.dataQuality.dataQualityScore") orelse "0.0";
    const quality_score = std.fmt.parseFloat(f64, quality_str) catch 0.0;
    
    const ord_ref = parser.getValue("product.extensions.sapORD");
    const csn_ref = parser.getValue("product.extensions.sapCSN");
    
    return ODPSProduct{
        .product_id = try allocator.dupe(u8, product_id),
        .name = try allocator.dupe(u8, name),
        .description = try allocator.dupe(u8, description),
        .version = try allocator.dupe(u8, version),
        .status = try allocator.dupe(u8, status),
        .quality_score = quality_score,
        .ord_ref = if (ord_ref) |ref| try allocator.dupe(u8, ref) else null,
        .csn_ref = if (csn_ref) |ref| try allocator.dupe(u8, ref) else null,
    };
}

/// Convert ODPS to ORD format
pub fn toORD(odps: ODPSProduct, allocator: Allocator) !ORDDocument {
    return ORDDocument{
        .ord_id = try std.fmt.allocPrint(allocator, "sap.nApps.Process:dataProduct:{s}", .{odps.product_id}),
        .title = try allocator.dupe(u8, odps.name),
        .description = try allocator.dupe(u8, odps.description),
        .version = try allocator.dupe(u8, odps.version),
        .product_type = try allocator.dupe(u8, if (std.mem.indexOf(u8, odps.description, "derived") != null) "derived" else "primary"),
        .category = try allocator.dupe(u8, "transactional"),
    };
}

/// Convert ODPS to CSN format
pub fn toCSN(odps: ODPSProduct, allocator: Allocator) !CSNSchema {
    return CSNSchema{
        .entity_name = try extractEntityName(allocator, odps.product_id),
        .title = try allocator.dupe(u8, odps.name),
        .description = try allocator.dupe(u8, odps.description),
        .elements = try allocator.dupe(u8, "{}"), // TODO: Extract from contract
        .keys = &[_][]const u8{},
    };
}

/// Extract entity name from product ID
fn extractEntityName(allocator: Allocator, product_id: []const u8) ![]u8 {
    // Extract from URN: urn:uuid:acdoca-journal-entries-v1 → JournalEntry
    if (std.mem.indexOf(u8, product_id, "acdoca")) |_| {
        return try allocator.dupe(u8, "JournalEntry");
    } else if (std.mem.indexOf(u8, product_id, "exchange-rates")) |_| {
        return try allocator.dupe(u8, "ExchangeRate");
    } else if (std.mem.indexOf(u8, product_id, "trial-balance")) |_| {
        return try allocator.dupe(u8, "TrialBalanceEntry");
    } else if (std.mem.indexOf(u8, product_id, "variances")) |_| {
        return try allocator.dupe(u8, "VarianceEntry");
    } else if (std.mem.indexOf(u8, product_id, "account-master")) |_| {
        return try allocator.dupe(u8, "AccountMaster");
    }
    
    return try allocator.dupe(u8, "Unknown");
}

/// Validate ODPS product against v4.1 schema
pub fn validateODPS(odps: *const ODPSProduct) !bool {
    // Check required fields
    if (odps.product_id.len == 0) return false;
    if (odps.name.len == 0) return false;
    if (odps.description.len == 0) return false;
    if (odps.version.len == 0) return false;
    
    // Check status is valid
    const valid_statuses = [_][]const u8{ "active", "beta", "deprecated" };
    var status_valid = false;
    for (valid_statuses) |valid_status| {
        if (std.mem.eql(u8, odps.status, valid_status)) {
            status_valid = true;
            break;
        }
    }
    if (!status_valid) return false;
    
    // Check quality score range
    if (odps.quality_score < 0.0 or odps.quality_score > 100.0) return false;
    
    return true;
}

/// Update ODPS quality score from runtime metrics
pub fn updateODPSQuality(
    allocator: Allocator,
    file_path: []const u8,
    quality_score: f64,
    verified_count: usize,
    total_count: usize,
) !void {
    // Read original file
    const file_content = try std.fs.cwd().readFileAlloc(allocator, file_path, 10 * 1024 * 1024);
    defer allocator.free(file_content);
    
    // Replace quality score line
    var updated_content = std.ArrayList(u8).init(allocator);
    defer updated_content.deinit();
    
    var lines = std.mem.splitScalar(u8, file_content, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "dataQualityScore:")) |_| {
            // Replace quality score
            try updated_content.writer().print("    dataQualityScore: {d:.1}\n", .{quality_score});
        } else if (std.mem.indexOf(u8, line, "score:") != null and 
                   std.mem.indexOf(u8, line, "completeness") != null) {
            // Update completeness score
            const completeness = @as(f64, @floatFromInt(verified_count)) / @as(f64, @floatFromInt(total_count)) * 100.0;
            try updated_content.writer().print("        score: {d:.0}\n", .{completeness});
        } else {
            try updated_content.appendSlice(line);
            try updated_content.append('\n');
        }
    }
    
    // Write back to file
    try std.fs.cwd().writeFile(.{
        .sub_path = file_path,
        .data = updated_content.items,
    });
}

/// Generate ORD JSON from all ODPS files in a directory
pub fn generateORDFromDirectory(
    allocator: Allocator,
    odps_dir: []const u8,
    output_file: []const u8,
) !void {
    // TODO: Scan directory, load all ODPS files, generate consolidated ORD JSON
    _ = allocator;
    _ = odps_dir;
    _ = output_file;
}

/// Generate CSN JSON from all ODPS files in a directory
pub fn generateCSNFromDirectory(
    allocator: Allocator,
    odps_dir: []const u8,
    output_file: []const u8,
) !void {
    // TODO: Scan directory, load all ODPS files, generate consolidated CSN JSON
    _ = allocator;
    _ = odps_dir;
    _ = output_file;
}

// Tests
test "ODPS validation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var valid_odps = ODPSProduct{
        .product_id = try allocator.dupe(u8, "urn:uuid:test-v1"),
        .name = try allocator.dupe(u8, "Test Product"),
        .description = try allocator.dupe(u8, "Test description"),
        .version = try allocator.dupe(u8, "1.0.0"),
        .status = try allocator.dupe(u8, "active"),
        .quality_score = 95.0,
        .ord_ref = null,
        .csn_ref = null,
    };
    defer {
        allocator.free(valid_odps.product_id);
        allocator.free(valid_odps.name);
        allocator.free(valid_odps.description);
        allocator.free(valid_odps.version);
        allocator.free(valid_odps.status);
    }
    
    try testing.expect(try validateODPS(&valid_odps));
}

test "ODPS to ORD conversion" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const odps = ODPSProduct{
        .product_id = try allocator.dupe(u8, "urn:uuid:acdoca-v1"),
        .name = try allocator.dupe(u8, "ACDOCA Journal Entries"),
        .description = try allocator.dupe(u8, "Journal entries derived from ACDOCA"),
        .version = try allocator.dupe(u8, "1.0.0"),
        .status = try allocator.dupe(u8, "active"),
        .quality_score = 95.0,
        .ord_ref = null,
        .csn_ref = null,
    };
    defer {
        allocator.free(odps.product_id);
        allocator.free(odps.name);
        allocator.free(odps.description);
        allocator.free(odps.version);
        allocator.free(odps.status);
    }
    
    const ord = try toORD(odps, allocator);
    defer {
        allocator.free(ord.ord_id);
        allocator.free(ord.title);
        allocator.free(ord.description);
        allocator.free(ord.version);
        allocator.free(ord.product_type);
        allocator.free(ord.category);
    }
    
    try testing.expectEqualStrings("ACDOCA Journal Entries", ord.title);
    try testing.expectEqualStrings("derived", ord.product_type);
}

test "entity name extraction" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    {
        const name = try extractEntityName(allocator, "urn:uuid:acdoca-journal-entries-v1");
        defer allocator.free(name);
        try testing.expectEqualStrings("JournalEntry", name);
    }
    
    {
        const name = try extractEntityName(allocator, "urn:uuid:exchange-rates-v1");
        defer allocator.free(name);
        try testing.expectEqualStrings("ExchangeRate", name);
    }
}