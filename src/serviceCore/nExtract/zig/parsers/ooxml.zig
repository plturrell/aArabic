// Office Open XML (ISO 29500) Package Structure Parser
// Handles ZIP-based OOXML packages for DOCX, XLSX, PPTX

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// Import our ZIP parser
const zip = @import("zip.zig");
const xml = @import("xml.zig");

/// OOXML Package - represents the root package structure
pub const OOXMLPackage = struct {
    allocator: Allocator,
    content_types: ContentTypes,
    relationships: RelationshipMap,
    parts: StringHashMap(*Part),
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return .{
            .allocator = allocator,
            .content_types = ContentTypes.init(allocator),
            .relationships = RelationshipMap.init(allocator),
            .parts = StringHashMap(*Part).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        // Free all parts
        var part_iter = self.parts.valueIterator();
        while (part_iter.next()) |part| {
            part.*.deinit();
            self.allocator.destroy(part.*);
        }
        self.parts.deinit();
        
        self.content_types.deinit();
        self.relationships.deinit();
    }
    
    /// Parse an OOXML package from a ZIP archive
    pub fn fromZip(allocator: Allocator, zip_path: []const u8) !Self {
        var package = Self.init(allocator);
        errdefer package.deinit();
        
        // Open ZIP archive
        var archive = try zip.ZipArchive.open(allocator, zip_path);
        defer archive.close();
        
        // Parse [Content_Types].xml
        if (archive.findFile("[Content_Types].xml")) |content_types_entry| {
            const content_data = try archive.extractFile(content_types_entry);
            defer allocator.free(content_data);
            try package.parseContentTypes(content_data);
        } else {
            return error.MissingContentTypes;
        }
        
        // Parse _rels/.rels (root relationships)
        if (archive.findFile("_rels/.rels")) |rels_entry| {
            const rels_data = try archive.extractFile(rels_entry);
            defer allocator.free(rels_data);
            try package.parseRelationships("", rels_data);
        } else {
            return error.MissingRootRelationships;
        }
        
        // Parse all relationship files
        for (archive.entries.items) |entry| {
            if (std.mem.endsWith(u8, entry.filename, ".rels")) {
                if (std.mem.eql(u8, entry.filename, "_rels/.rels")) continue;
                
                const rels_data = try archive.extractFile(&entry);
                defer allocator.free(rels_data);
                
                // Extract source part name from relationship file path
                // e.g., "word/_rels/document.xml.rels" -> "word/document.xml"
                const source_part = try extractSourcePartName(allocator, entry.filename);
                defer allocator.free(source_part);
                
                try package.parseRelationships(source_part, rels_data);
            }
        }
        
        return package;
    }
    
    /// Parse [Content_Types].xml
    fn parseContentTypes(self: *Self, data: []const u8) !void {
        var doc = try xml.parseDocument(self.allocator, data);
        defer doc.deinit();
        
        const root = doc.root orelse return error.InvalidContentTypes;
        
        // Parse Default elements
        for (root.children.items) |child| {
            if (child != .element) continue;
            const elem = child.element;
            
            if (std.mem.eql(u8, elem.name, "Default")) {
                const extension = elem.getAttribute("Extension") orelse continue;
                const content_type = elem.getAttribute("ContentType") orelse continue;
                
                try self.content_types.defaults.put(
                    try self.allocator.dupe(u8, extension),
                    try self.allocator.dupe(u8, content_type)
                );
            } else if (std.mem.eql(u8, elem.name, "Override")) {
                const part_name = elem.getAttribute("PartName") orelse continue;
                const content_type = elem.getAttribute("ContentType") orelse continue;
                
                try self.content_types.overrides.put(
                    try self.allocator.dupe(u8, part_name),
                    try self.allocator.dupe(u8, content_type)
                );
            }
        }
    }
    
    /// Parse a relationships file
    fn parseRelationships(self: *Self, source_part: []const u8, data: []const u8) !void {
        var doc = try xml.parseDocument(self.allocator, data);
        defer doc.deinit();
        
        const root = doc.root orelse return error.InvalidRelationships;
        
        var relationships = ArrayList(Relationship).init(self.allocator);
        
        for (root.children.items) |child| {
            if (child != .element) continue;
            const elem = child.element;
            
            if (std.mem.eql(u8, elem.name, "Relationship")) {
                const id = elem.getAttribute("Id") orelse continue;
                const rel_type = elem.getAttribute("Type") orelse continue;
                const target = elem.getAttribute("Target") orelse continue;
                const target_mode = elem.getAttribute("TargetMode") orelse "Internal";
                
                try relationships.append(.{
                    .id = try self.allocator.dupe(u8, id),
                    .type = try self.allocator.dupe(u8, rel_type),
                    .target = try self.allocator.dupe(u8, target),
                    .target_mode = if (std.mem.eql(u8, target_mode, "External"))
                        .external
                    else
                        .internal,
                });
            }
        }
        
        const source_key = try self.allocator.dupe(u8, source_part);
        try self.relationships.put(source_key, relationships);
    }
    
    /// Get content type for a part
    pub fn getContentType(self: *const Self, part_name: []const u8) ?[]const u8 {
        // Check overrides first
        if (self.content_types.overrides.get(part_name)) |content_type| {
            return content_type;
        }
        
        // Check defaults by extension
        if (std.mem.lastIndexOfScalar(u8, part_name, '.')) |dot_pos| {
            const extension = part_name[dot_pos + 1..];
            if (self.content_types.defaults.get(extension)) |content_type| {
                return content_type;
            }
        }
        
        return null;
    }
    
    /// Get relationships for a part
    pub fn getRelationships(self: *const Self, source_part: []const u8) ?ArrayList(Relationship) {
        return self.relationships.get(source_part);
    }
    
    /// Resolve a relationship target to absolute part name
    pub fn resolveTarget(self: *Self, source_part: []const u8, target: []const u8) ![]const u8 {
        // If target starts with '/', it's already absolute
        if (target.len > 0 and target[0] == '/') {
            return try self.allocator.dupe(u8, target[1..]);
        }
        
        // Resolve relative to source part directory
        var result = ArrayList(u8).init(self.allocator);
        errdefer result.deinit();
        
        // Get source directory
        if (std.mem.lastIndexOfScalar(u8, source_part, '/')) |slash_pos| {
            try result.appendSlice(source_part[0..slash_pos + 1]);
        }
        
        // Append target
        try result.appendSlice(target);
        
        return result.toOwnedSlice();
    }
};

/// Content Types mapping
pub const ContentTypes = struct {
    defaults: StringHashMap([]const u8),  // extension -> content type
    overrides: StringHashMap([]const u8), // part name -> content type
    
    pub fn init(allocator: Allocator) ContentTypes {
        return .{
            .defaults = StringHashMap([]const u8).init(allocator),
            .overrides = StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *ContentTypes) void {
        // Free keys and values
        var default_iter = self.defaults.iterator();
        while (default_iter.next()) |entry| {
            self.defaults.allocator.free(entry.key_ptr.*);
            self.defaults.allocator.free(entry.value_ptr.*);
        }
        
        var override_iter = self.overrides.iterator();
        while (override_iter.next()) |entry| {
            self.overrides.allocator.free(entry.key_ptr.*);
            self.overrides.allocator.free(entry.value_ptr.*);
        }
        
        self.defaults.deinit();
        self.overrides.deinit();
    }
};

/// Relationship between parts
pub const Relationship = struct {
    id: []const u8,
    type: []const u8,
    target: []const u8,
    target_mode: TargetMode,
    
    pub const TargetMode = enum {
        internal,
        external,
    };
    
    pub fn deinit(self: *Relationship, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.type);
        allocator.free(self.target);
    }
};

/// Map of source part -> relationships
pub const RelationshipMap = StringHashMap(ArrayList(Relationship));

/// OOXML Part (file within the package)
pub const Part = struct {
    name: []const u8,
    content_type: []const u8,
    data: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, content_type: []const u8, data: []const u8) !*Part {
        const part = try allocator.create(Part);
        part.* = .{
            .name = try allocator.dupe(u8, name),
            .content_type = try allocator.dupe(u8, content_type),
            .data = try allocator.dupe(u8, data),
            .allocator = allocator,
        };
        return part;
    }
    
    pub fn deinit(self: *Part) void {
        self.allocator.free(self.name);
        self.allocator.free(self.content_type);
        self.allocator.free(self.data);
    }
};

/// Common OOXML relationship types
pub const RelationshipTypes = struct {
    pub const office_document = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument";
    pub const extended_properties = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties";
    pub const core_properties = "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties";
    pub const styles = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles";
    pub const theme = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme";
    pub const font_table = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable";
    pub const settings = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings";
    pub const numbering = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering";
    pub const image = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image";
    pub const hyperlink = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink";
    pub const header = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header";
    pub const footer = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer";
    pub const worksheet = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet";
    pub const shared_strings = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings";
    pub const slide = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide";
    pub const slide_master = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster";
    pub const slide_layout = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout";
};

/// Common OOXML content types
pub const ContentTypeValues = struct {
    pub const word_document = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml";
    pub const excel_worksheet = "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml";
    pub const excel_workbook = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml";
    pub const powerpoint_presentation = "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml";
    pub const powerpoint_slide = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml";
    pub const shared_strings = "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml";
    pub const styles = "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml";
    pub const theme = "application/vnd.openxmlformats-officedocument.theme+xml";
    pub const core_properties = "application/vnd.openxmlformats-package.core-properties+xml";
    pub const extended_properties = "application/vnd.openxmlformats-officedocument.extended-properties+xml";
};

/// Extract source part name from relationship file path
/// e.g., "word/_rels/document.xml.rels" -> "word/document.xml"
fn extractSourcePartName(allocator: Allocator, rels_path: []const u8) ![]const u8 {
    // Remove "_rels/" and ".rels" suffix
    var result = ArrayList(u8).init(allocator);
    errdefer result.deinit();
    
    // Find last occurrence of "_rels/"
    if (std.mem.lastIndexOf(u8, rels_path, "_rels/")) |rels_pos| {
        // Add everything before "_rels/"
        if (rels_pos > 0) {
            try result.appendSlice(rels_path[0..rels_pos]);
        }
        
        // Add filename without .rels extension
        const filename_start = rels_pos + "_rels/".len;
        if (std.mem.endsWith(u8, rels_path, ".rels")) {
            const filename_end = rels_path.len - ".rels".len;
            try result.appendSlice(rels_path[filename_start..filename_end]);
        } else {
            try result.appendSlice(rels_path[filename_start..]);
        }
    } else {
        try result.appendSlice(rels_path);
    }
    
    return result.toOwnedSlice();
}

/// Validate OOXML package structure
pub fn validatePackage(package: *const OOXMLPackage) !void {
    // Check for required content types
    if (package.content_types.defaults.count() == 0 and 
        package.content_types.overrides.count() == 0) {
        return error.MissingContentTypes;
    }
    
    // Check for root relationships
    if (package.relationships.get("")) |root_rels| {
        if (root_rels.items.len == 0) {
            return error.MissingRootRelationships;
        }
        
        // Check for office document relationship
        var has_office_doc = false;
        for (root_rels.items) |rel| {
            if (std.mem.eql(u8, rel.type, RelationshipTypes.office_document)) {
                has_office_doc = true;
                break;
            }
        }
        
        if (!has_office_doc) {
            return error.MissingOfficeDocumentRelationship;
        }
    } else {
        return error.MissingRootRelationships;
    }
}

/// Find main document part (officeDocument relationship)
pub fn findMainDocument(package: *const OOXMLPackage) ![]const u8 {
    const root_rels = package.relationships.get("") orelse return error.MissingRootRelationships;
    
    for (root_rels.items) |rel| {
        if (std.mem.eql(u8, rel.type, RelationshipTypes.office_document)) {
            return rel.target;
        }
    }
    
    return error.MainDocumentNotFound;
}

/// Find all parts of a specific type
pub fn findPartsByType(package: *const OOXMLPackage, allocator: Allocator, rel_type: []const u8) !ArrayList([]const u8) {
    var parts = ArrayList([]const u8).init(allocator);
    errdefer parts.deinit();
    
    var rel_iter = package.relationships.iterator();
    while (rel_iter.next()) |entry| {
        for (entry.value_ptr.items) |rel| {
            if (std.mem.eql(u8, rel.type, rel_type)) {
                const part_name = try allocator.dupe(u8, rel.target);
                try parts.append(part_name);
            }
        }
    }
    
    return parts;
}

/// Get package type (DOCX, XLSX, PPTX)
pub fn getPackageType(package: *const OOXMLPackage) PackageType {
    const main_doc = findMainDocument(package) catch return .unknown;
    
    // Check content type of main document
    const content_type = package.getContentType(main_doc) orelse return .unknown;
    
    if (std.mem.indexOf(u8, content_type, "wordprocessingml") != null) {
        return .docx;
    } else if (std.mem.indexOf(u8, content_type, "spreadsheetml") != null) {
        return .xlsx;
    } else if (std.mem.indexOf(u8, content_type, "presentationml") != null) {
        return .pptx;
    }
    
    return .unknown;
}

pub const PackageType = enum {
    docx,
    xlsx,
    pptx,
    unknown,
};

/// Check if package has digital signatures
pub fn hasDigitalSignatures(package: *const OOXMLPackage) bool {
    // Check for signature relationships
    const root_rels = package.relationships.get("") orelse return false;
    
    for (root_rels.items) |rel| {
        if (std.mem.indexOf(u8, rel.type, "digital-signature") != null) {
            return true;
        }
    }
    
    return false;
}

/// Extract metadata from core properties
pub const Metadata = struct {
    title: ?[]const u8 = null,
    creator: ?[]const u8 = null,
    subject: ?[]const u8 = null,
    description: ?[]const u8 = null,
    keywords: ?[]const u8 = null,
    last_modified_by: ?[]const u8 = null,
    created: ?[]const u8 = null,
    modified: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) Metadata {
        return .{
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Metadata) void {
        if (self.title) |t| self.allocator.free(t);
        if (self.creator) |c| self.allocator.free(c);
        if (self.subject) |s| self.allocator.free(s);
        if (self.description) |d| self.allocator.free(d);
        if (self.keywords) |k| self.allocator.free(k);
        if (self.last_modified_by) |l| self.allocator.free(l);
        if (self.created) |c| self.allocator.free(c);
        if (self.modified) |m| self.allocator.free(m);
    }
};

/// Extract core properties metadata
pub fn extractMetadata(package: *const OOXMLPackage, allocator: Allocator, core_props_data: []const u8) !Metadata {
    var metadata = Metadata.init(allocator);
    errdefer metadata.deinit();
    
    var doc = try xml.parseDocument(allocator, core_props_data);
    defer doc.deinit();
    
    const root = doc.root orelse return metadata;
    
    // Parse DC (Dublin Core) and CP (Core Properties) elements
    for (root.children.items) |child| {
        if (child != .element) continue;
        const elem = child.element;
        
        const text = elem.getTextContent() orelse continue;
        
        if (std.mem.endsWith(u8, elem.name, ":title") or std.mem.eql(u8, elem.name, "title")) {
            metadata.title = try allocator.dupe(u8, text);
        } else if (std.mem.endsWith(u8, elem.name, ":creator") or std.mem.eql(u8, elem.name, "creator")) {
            metadata.creator = try allocator.dupe(u8, text);
        } else if (std.mem.endsWith(u8, elem.name, ":subject") or std.mem.eql(u8, elem.name, "subject")) {
            metadata.subject = try allocator.dupe(u8, text);
        } else if (std.mem.endsWith(u8, elem.name, ":description") or std.mem.eql(u8, elem.name, "description")) {
            metadata.description = try allocator.dupe(u8, text);
        } else if (std.mem.endsWith(u8, elem.name, ":keywords") or std.mem.eql(u8, elem.name, "keywords")) {
            metadata.keywords = try allocator.dupe(u8, text);
        } else if (std.mem.endsWith(u8, elem.name, ":lastModifiedBy")) {
            metadata.last_modified_by = try allocator.dupe(u8, text);
        } else if (std.mem.endsWith(u8, elem.name, ":created")) {
            metadata.created = try allocator.dupe(u8, text);
        } else if (std.mem.endsWith(u8, elem.name, ":modified")) {
            metadata.modified = try allocator.dupe(u8, text);
        }
    }
    
    return metadata;
}

/// Statistics about the package
pub const PackageStats = struct {
    total_parts: usize,
    total_relationships: usize,
    content_type_count: usize,
    has_images: bool,
    has_external_refs: bool,
    package_type: PackageType,
};

/// Get package statistics
pub fn getPackageStats(package: *const OOXMLPackage) PackageStats {
    var stats = PackageStats{
        .total_parts = package.parts.count(),
        .total_relationships = 0,
        .content_type_count = package.content_types.defaults.count() + package.content_types.overrides.count(),
        .has_images = false,
        .has_external_refs = false,
        .package_type = getPackageType(package),
    };
    
    // Count relationships
    var rel_iter = package.relationships.iterator();
    while (rel_iter.next()) |entry| {
        stats.total_relationships += entry.value_ptr.items.len;
        
        // Check for images and external references
        for (entry.value_ptr.items) |rel| {
            if (std.mem.eql(u8, rel.type, RelationshipTypes.image)) {
                stats.has_images = true;
            }
            if (rel.target_mode == .external) {
                stats.has_external_refs = true;
            }
        }
    }
    
    return stats;
}

// Export functions for FFI
export fn nExtract_OOXML_parse(path: [*:0]const u8) ?*OOXMLPackage {
    const allocator = std.heap.c_allocator;
    const path_slice = std.mem.span(path);
    
    var package = OOXMLPackage.fromZip(allocator, path_slice) catch return null;
    const package_ptr = allocator.create(OOXMLPackage) catch return null;
    package_ptr.* = package;
    
    return package_ptr;
}

export fn nExtract_OOXML_destroy(package: *OOXMLPackage) void {
    const allocator = package.allocator;
    package.deinit();
    allocator.destroy(package);
}

export fn nExtract_OOXML_getContentType(
    package: *const OOXMLPackage,
    part_name: [*:0]const u8,
) ?[*:0]const u8 {
    const part_name_slice = std.mem.span(part_name);
    const content_type = package.getContentType(part_name_slice) orelse return null;
    
    // Return as C string (already null-terminated in our storage)
    return @ptrCast(content_type.ptr);
}

export fn nExtract_OOXML_validate(package: *const OOXMLPackage) bool {
    validatePackage(package) catch return false;
    return true;
}

// Tests
test "OOXML content types parsing" {
    const allocator = std.testing.allocator;
    
    const content_types_xml =
        \\<?xml version="1.0"?>
        \\<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
        \\  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
        \\  <Default Extension="xml" ContentType="application/xml"/>
        \\  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
        \\</Types>
    ;
    
    var package = OOXMLPackage.init(allocator);
    defer package.deinit();
    
    try package.parseContentTypes(content_types_xml);
    
    // Check defaults
    try std.testing.expect(package.content_types.defaults.count() == 2);
    
    // Check overrides
    try std.testing.expect(package.content_types.overrides.count() == 1);
    const doc_content_type = package.getContentType("/word/document.xml").?;
    try std.testing.expect(std.mem.eql(u8, doc_content_type, ContentTypeValues.word_document));
}

test "OOXML relationships parsing" {
    const allocator = std.testing.allocator;
    
    const rels_xml =
        \\<?xml version="1.0"?>
        \\<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
        \\  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
        \\  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
        \\</Relationships>
    ;
    
    var package = OOXMLPackage.init(allocator);
    defer package.deinit();
    
    try package.parseRelationships("", rels_xml);
    
    const rels = package.getRelationships("").?;
    try std.testing.expect(rels.items.len == 2);
    
    const first_rel = rels.items[0];
    try std.testing.expect(std.mem.eql(u8, first_rel.id, "rId1"));
    try std.testing.expect(std.mem.eql(u8, first_rel.type, RelationshipTypes.office_document));
}

test "Extract source part name" {
    const allocator = std.testing.allocator;
    
    const result1 = try extractSourcePartName(allocator, "word/_rels/document.xml.rels");
    defer allocator.free(result1);
    try std.testing.expect(std.mem.eql(u8, result1, "word/document.xml"));
    
    const result2 = try extractSourcePartName(allocator, "_rels/.rels");
    defer allocator.free(result2);
    try std.testing.expect(std.mem.eql(u8, result2, ""));
}
