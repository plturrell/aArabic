const std = @import("std");
const fs = std.fs;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// ============================================================================
// Global State
// ============================================================================
var loaded_index: ?*ScipIndex = null;

// ============================================================================
// Protobuf Wire Types
// ============================================================================
const WIRE_VARINT: u3 = 0;
const WIRE_FIXED64: u3 = 1;
const WIRE_LENGTH_DELIMITED: u3 = 2;
const WIRE_FIXED32: u3 = 5;

// ============================================================================
// SCIP Field Numbers (from scip.proto)
// ============================================================================
// Index fields
const INDEX_METADATA: u32 = 1;
const INDEX_DOCUMENTS: u32 = 2;
const INDEX_EXTERNAL_SYMBOLS: u32 = 3;

// Metadata fields
const METADATA_VERSION: u32 = 1;
const METADATA_TOOL_INFO: u32 = 2;
const METADATA_PROJECT_ROOT: u32 = 3;
const METADATA_TEXT_ENCODING: u32 = 4;

// ToolInfo fields
const TOOLINFO_NAME: u32 = 1;
const TOOLINFO_VERSION: u32 = 2;
const TOOLINFO_ARGUMENTS: u32 = 3;

// Document fields
const DOC_RELATIVE_PATH: u32 = 1;
const DOC_OCCURRENCES: u32 = 2;
const DOC_SYMBOLS: u32 = 3;
const DOC_LANGUAGE: u32 = 4;
const DOC_TEXT: u32 = 5;
const DOC_POSITION_ENCODING: u32 = 6;

// Occurrence fields
const OCC_RANGE: u32 = 1;
const OCC_SYMBOL: u32 = 2;
const OCC_SYMBOL_ROLES: u32 = 3;
const OCC_OVERRIDE_DOC: u32 = 4;
const OCC_SYNTAX_KIND: u32 = 5;
const OCC_DIAGNOSTICS: u32 = 6;
const OCC_ENCLOSING_RANGE: u32 = 7;

// SymbolInformation fields
const SYMINFO_SYMBOL: u32 = 1;
const SYMINFO_DOCUMENTATION: u32 = 3;
const SYMINFO_RELATIONSHIPS: u32 = 4;
const SYMINFO_KIND: u32 = 5;
const SYMINFO_DISPLAY_NAME: u32 = 6;
const SYMINFO_SIGNATURE_DOC: u32 = 7;
const SYMINFO_ENCLOSING_SYMBOL: u32 = 8;

// Relationship fields
const REL_SYMBOL: u32 = 1;
const REL_IS_REFERENCE: u32 = 2;
const REL_IS_IMPLEMENTATION: u32 = 3;
const REL_IS_TYPE_DEFINITION: u32 = 4;
const REL_IS_DEFINITION: u32 = 5;

// SymbolRole values (bitflags)
pub const SymbolRole = struct {
    pub const Definition: i32 = 0x1;
    pub const Import: i32 = 0x2;
    pub const WriteAccess: i32 = 0x4;
    pub const ReadAccess: i32 = 0x8;
    pub const Generated: i32 = 0x10;
    pub const Test: i32 = 0x20;
    pub const ForwardDefinition: i32 = 0x40;
};

// ============================================================================
// Protobuf Reader Helpers
// ============================================================================

/// Read an unsigned varint from bytes, returns value and bytes consumed
pub fn readVarint(bytes: []const u8) !struct { value: u64, consumed: usize } {
    var result: u64 = 0;
    var shift: u6 = 0;
    var i: usize = 0;

    while (i < bytes.len) {
        const byte = bytes[i];
        result |= @as(u64, byte & 0x7F) << shift;
        i += 1;

        if (byte & 0x80 == 0) {
            return .{ .value = result, .consumed = i };
        }

        if (shift >= 63) return error.VarintOverflow;
        shift += 7;
    }
    return error.UnexpectedEndOfData;
}

/// Read a ZigZag encoded signed varint
pub fn readSignedVarint(bytes: []const u8) !struct { value: i64, consumed: usize } {
    const result = try readVarint(bytes);
    // ZigZag decode: (n >> 1) ^ -(n & 1)
    const n = result.value;
    const decoded: i64 = @bitCast((n >> 1) ^ (~(n & 1) +% 1));
    return .{ .value = decoded, .consumed = result.consumed };
}

/// Read a length-delimited field (returns slice and bytes consumed)
pub fn readLengthDelimited(bytes: []const u8) !struct { data: []const u8, consumed: usize } {
    const len_result = try readVarint(bytes);
    const length: usize = @intCast(len_result.value);
    const start = len_result.consumed;

    if (start + length > bytes.len) return error.UnexpectedEndOfData;

    return .{
        .data = bytes[start .. start + length],
        .consumed = start + length,
    };
}

/// Parse a protobuf tag into field number and wire type
pub fn readTag(bytes: []const u8) !struct { field_number: u32, wire_type: u3, consumed: usize } {
    const result = try readVarint(bytes);
    return .{
        .field_number = @intCast(result.value >> 3),
        .wire_type = @intCast(result.value & 0x7),
        .consumed = result.consumed,
    };
}

/// Read packed repeated int32 values
fn readPackedInt32(bytes: []const u8) !std.ArrayListUnmanaged(i32) {
    var list = std.ArrayListUnmanaged(i32){};
    errdefer list.deinit(allocator);

    var pos: usize = 0;
    while (pos < bytes.len) {
        const result = try readVarint(bytes[pos..]);
        try list.append(allocator, @intCast(@as(i64, @bitCast(result.value))));
        pos += result.consumed;
    }
    return list;
}

// ============================================================================
// SCIP Data Structures for In-Memory Representation
// ============================================================================

/// Represents a relationship between symbols
pub const ScipRelationship = struct {
    symbol: []const u8,
    is_reference: bool,
    is_implementation: bool,
    is_type_definition: bool,
    is_definition: bool,

    pub fn deinit(self: *ScipRelationship) void {
        if (self.symbol.len > 0) allocator.free(@constCast(self.symbol));
    }
};

/// Represents an occurrence of a symbol in source code
pub const ScipOccurrence = struct {
    start_line: i32,
    start_char: i32,
    end_line: i32,
    end_char: i32,
    symbol: []const u8,
    symbol_roles: i32,
    syntax_kind: i32,
    override_documentation: std.ArrayListUnmanaged([]const u8),
    enclosing_range: ?[4]i32,

    pub fn init() ScipOccurrence {
        return .{
            .start_line = 0,
            .start_char = 0,
            .end_line = 0,
            .end_char = 0,
            .symbol = "",
            .symbol_roles = 0,
            .syntax_kind = 0,
            .override_documentation = std.ArrayListUnmanaged([]const u8){},
            .enclosing_range = null,
        };
    }

    pub fn deinit(self: *ScipOccurrence) void {
        if (self.symbol.len > 0) allocator.free(@constCast(self.symbol));
        for (self.override_documentation.items) |doc| {
            allocator.free(@constCast(doc));
        }
        self.override_documentation.deinit(allocator);
    }

    pub fn isDefinition(self: *const ScipOccurrence) bool {
        return (self.symbol_roles & SymbolRole.Definition) != 0;
    }

    pub fn isReference(self: *const ScipOccurrence) bool {
        return (self.symbol_roles & SymbolRole.Definition) == 0;
    }
};

/// Represents symbol information (documentation, kind, relationships)
pub const ScipSymbolInfo = struct {
    symbol: []const u8,
    documentation: std.ArrayListUnmanaged([]const u8),
    kind: i32,
    display_name: []const u8,
    enclosing_symbol: []const u8,
    relationships: std.ArrayListUnmanaged(ScipRelationship),

    pub fn init() ScipSymbolInfo {
        return .{
            .symbol = "",
            .documentation = std.ArrayListUnmanaged([]const u8){},
            .kind = 0,
            .display_name = "",
            .enclosing_symbol = "",
            .relationships = std.ArrayListUnmanaged(ScipRelationship){},
        };
    }

    pub fn deinit(self: *ScipSymbolInfo) void {
        if (self.symbol.len > 0) allocator.free(@constCast(self.symbol));
        if (self.display_name.len > 0) allocator.free(@constCast(self.display_name));
        if (self.enclosing_symbol.len > 0) allocator.free(@constCast(self.enclosing_symbol));
        for (self.documentation.items) |doc| {
            allocator.free(@constCast(doc));
        }
        self.documentation.deinit(allocator);
        for (self.relationships.items) |*rel| {
            rel.deinit();
        }
        self.relationships.deinit(allocator);
    }

    pub fn getDocumentation(self: *const ScipSymbolInfo) []const u8 {
        if (self.documentation.items.len > 0) {
            return self.documentation.items[0];
        }
        return "";
    }
};

/// Represents a document (source file) in the index
pub const ScipDocument = struct {
    language: []const u8,
    relative_path: []const u8,
    text: []const u8,
    position_encoding: i32,
    occurrences: std.ArrayListUnmanaged(ScipOccurrence),
    symbols: std.ArrayListUnmanaged(ScipSymbolInfo),

    pub fn init() ScipDocument {
        return .{
            .language = "",
            .relative_path = "",
            .text = "",
            .position_encoding = 0,
            .occurrences = std.ArrayListUnmanaged(ScipOccurrence){},
            .symbols = std.ArrayListUnmanaged(ScipSymbolInfo){},
        };
    }

    pub fn deinit(self: *ScipDocument) void {
        if (self.language.len > 0) allocator.free(@constCast(self.language));
        if (self.relative_path.len > 0) allocator.free(@constCast(self.relative_path));
        if (self.text.len > 0) allocator.free(@constCast(self.text));
        for (self.occurrences.items) |*occ| {
            occ.deinit();
        }
        self.occurrences.deinit(allocator);
        for (self.symbols.items) |*sym| {
            sym.deinit();
        }
        self.symbols.deinit(allocator);
    }

    /// Find symbol info by symbol string
    pub fn findSymbol(self: *const ScipDocument, symbol: []const u8) ?*const ScipSymbolInfo {
        for (self.symbols.items) |*info| {
            if (std.mem.eql(u8, info.symbol, symbol)) {
                return info;
            }
        }
        return null;
    }
};

/// Tool information from metadata
pub const ScipToolInfo = struct {
    name: []const u8,
    version: []const u8,
    arguments: std.ArrayListUnmanaged([]const u8),

    pub fn init() ScipToolInfo {
        return .{
            .name = "",
            .version = "",
            .arguments = std.ArrayListUnmanaged([]const u8){},
        };
    }

    pub fn deinit(self: *ScipToolInfo) void {
        if (self.name.len > 0) allocator.free(@constCast(self.name));
        if (self.version.len > 0) allocator.free(@constCast(self.version));
        for (self.arguments.items) |arg| {
            allocator.free(@constCast(arg));
        }
        self.arguments.deinit(allocator);
    }
};

/// Metadata about the index
pub const ScipMetadata = struct {
    version: i32,
    tool_info: ScipToolInfo,
    project_root: []const u8,
    text_encoding: i32,

    pub fn init() ScipMetadata {
        return .{
            .version = 0,
            .tool_info = ScipToolInfo.init(),
            .project_root = "",
            .text_encoding = 0,
        };
    }

    pub fn deinit(self: *ScipMetadata) void {
        self.tool_info.deinit();
        if (self.project_root.len > 0) allocator.free(@constCast(self.project_root));
    }
};

/// The complete SCIP index
pub const ScipIndex = struct {
    metadata: ScipMetadata,
    documents: std.ArrayListUnmanaged(ScipDocument),
    external_symbols: std.ArrayListUnmanaged(ScipSymbolInfo),

    pub fn init() ScipIndex {
        return .{
            .metadata = ScipMetadata.init(),
            .documents = std.ArrayListUnmanaged(ScipDocument){},
            .external_symbols = std.ArrayListUnmanaged(ScipSymbolInfo){},
        };
    }

    pub fn deinit(self: *ScipIndex) void {
        self.metadata.deinit();
        for (self.documents.items) |*doc| {
            doc.deinit();
        }
        self.documents.deinit(allocator);
        for (self.external_symbols.items) |*sym| {
            sym.deinit();
        }
        self.external_symbols.deinit(allocator);
    }

    /// Find a document by relative path
    pub fn findDocument(self: *const ScipIndex, path: []const u8) ?*const ScipDocument {
        for (self.documents.items) |*doc| {
            if (std.mem.eql(u8, doc.relative_path, path)) {
                return doc;
            }
        }
        return null;
    }

    /// Find external symbol info
    pub fn findExternalSymbol(self: *const ScipIndex, symbol: []const u8) ?*const ScipSymbolInfo {
        for (self.external_symbols.items) |*info| {
            if (std.mem.eql(u8, info.symbol, symbol)) {
                return info;
            }
        }
        return null;
    }
};


// ============================================================================
// Parser Functions
// ============================================================================

/// Allocate a copy of a string slice
fn dupeString(data: []const u8) ![]const u8 {
    if (data.len == 0) return "";
    const copy = try allocator.alloc(u8, data.len);
    @memcpy(copy, data);
    return copy;
}

/// Parse a Relationship message
fn parseRelationship(bytes: []const u8) !ScipRelationship {
    var rel = ScipRelationship{
        .symbol = "",
        .is_reference = false,
        .is_implementation = false,
        .is_type_definition = false,
        .is_definition = false,
    };
    errdefer rel.deinit();

    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readTag(bytes[pos..]);
        pos += tag.consumed;

        switch (tag.field_number) {
            REL_SYMBOL => {
                const str = try readLengthDelimited(bytes[pos..]);
                rel.symbol = try dupeString(str.data);
                pos += str.consumed;
            },
            REL_IS_REFERENCE => {
                const v = try readVarint(bytes[pos..]);
                rel.is_reference = v.value != 0;
                pos += v.consumed;
            },
            REL_IS_IMPLEMENTATION => {
                const v = try readVarint(bytes[pos..]);
                rel.is_implementation = v.value != 0;
                pos += v.consumed;
            },
            REL_IS_TYPE_DEFINITION => {
                const v = try readVarint(bytes[pos..]);
                rel.is_type_definition = v.value != 0;
                pos += v.consumed;
            },
            REL_IS_DEFINITION => {
                const v = try readVarint(bytes[pos..]);
                rel.is_definition = v.value != 0;
                pos += v.consumed;
            },
            else => {
                // Skip unknown field
                pos += try skipField(bytes[pos..], tag.wire_type);
            },
        }
    }
    return rel;
}

/// Skip an unknown field based on wire type
fn skipField(bytes: []const u8, wire_type: u3) !usize {
    switch (wire_type) {
        WIRE_VARINT => {
            const v = try readVarint(bytes);
            return v.consumed;
        },
        WIRE_FIXED64 => return 8,
        WIRE_LENGTH_DELIMITED => {
            const ld = try readLengthDelimited(bytes);
            return ld.consumed;
        },
        WIRE_FIXED32 => return 4,
        else => return error.UnknownWireType,
    }
}

/// Parse an Occurrence message
pub fn parseOccurrence(bytes: []const u8) !ScipOccurrence {
    var occ = ScipOccurrence.init();
    errdefer occ.deinit();

    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readTag(bytes[pos..]);
        pos += tag.consumed;

        switch (tag.field_number) {
            OCC_RANGE => {
                const packed_range = try readLengthDelimited(bytes[pos..]);
                var range_list = try readPackedInt32(packed_range.data);
                defer range_list.deinit(allocator);

                if (range_list.items.len >= 3) {
                    occ.start_line = range_list.items[0];
                    occ.start_char = range_list.items[1];
                    if (range_list.items.len == 3) {
                        // Same line
                        occ.end_line = occ.start_line;
                        occ.end_char = range_list.items[2];
                    } else {
                        occ.end_line = range_list.items[2];
                        occ.end_char = range_list.items[3];
                    }
                }
                pos += packed_range.consumed;
            },
            OCC_SYMBOL => {
                const str = try readLengthDelimited(bytes[pos..]);
                occ.symbol = try dupeString(str.data);
                pos += str.consumed;
            },
            OCC_SYMBOL_ROLES => {
                const v = try readVarint(bytes[pos..]);
                occ.symbol_roles = @intCast(@as(i64, @bitCast(v.value)));
                pos += v.consumed;
            },
            OCC_OVERRIDE_DOC => {
                const str = try readLengthDelimited(bytes[pos..]);
                try occ.override_documentation.append(allocator, try dupeString(str.data));
                pos += str.consumed;
            },
            OCC_SYNTAX_KIND => {
                const v = try readVarint(bytes[pos..]);
                occ.syntax_kind = @intCast(@as(i64, @bitCast(v.value)));
                pos += v.consumed;
            },
            OCC_ENCLOSING_RANGE => {
                const enc_range = try readLengthDelimited(bytes[pos..]);
                var range_list = try readPackedInt32(enc_range.data);
                defer range_list.deinit(allocator);

                if (range_list.items.len >= 4) {
                    occ.enclosing_range = .{
                        range_list.items[0],
                        range_list.items[1],
                        range_list.items[2],
                        range_list.items[3],
                    };
                }
                pos += enc_range.consumed;
            },
            else => {
                pos += try skipField(bytes[pos..], tag.wire_type);
            },
        }
    }
    return occ;
}

/// Parse a SymbolInformation message
pub fn parseSymbolInfo(bytes: []const u8) !ScipSymbolInfo {
    var info = ScipSymbolInfo.init();
    errdefer info.deinit();

    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readTag(bytes[pos..]);
        pos += tag.consumed;

        switch (tag.field_number) {
            SYMINFO_SYMBOL => {
                const str = try readLengthDelimited(bytes[pos..]);
                info.symbol = try dupeString(str.data);
                pos += str.consumed;
            },
            SYMINFO_DOCUMENTATION => {
                const str = try readLengthDelimited(bytes[pos..]);
                try info.documentation.append(allocator, try dupeString(str.data));
                pos += str.consumed;
            },
            SYMINFO_RELATIONSHIPS => {
                const nested = try readLengthDelimited(bytes[pos..]);
                const rel = try parseRelationship(nested.data);
                try info.relationships.append(allocator, rel);
                pos += nested.consumed;
            },
            SYMINFO_KIND => {
                const v = try readVarint(bytes[pos..]);
                info.kind = @intCast(@as(i64, @bitCast(v.value)));
                pos += v.consumed;
            },
            SYMINFO_DISPLAY_NAME => {
                const str = try readLengthDelimited(bytes[pos..]);
                info.display_name = try dupeString(str.data);
                pos += str.consumed;
            },
            SYMINFO_ENCLOSING_SYMBOL => {
                const str = try readLengthDelimited(bytes[pos..]);
                info.enclosing_symbol = try dupeString(str.data);
                pos += str.consumed;
            },
            else => {
                pos += try skipField(bytes[pos..], tag.wire_type);
            },
        }
    }
    return info;
}

/// Parse a ToolInfo message
fn parseToolInfo(bytes: []const u8) !ScipToolInfo {
    var info = ScipToolInfo.init();
    errdefer info.deinit();

    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readTag(bytes[pos..]);
        pos += tag.consumed;

        switch (tag.field_number) {
            TOOLINFO_NAME => {
                const str = try readLengthDelimited(bytes[pos..]);
                info.name = try dupeString(str.data);
                pos += str.consumed;
            },
            TOOLINFO_VERSION => {
                const str = try readLengthDelimited(bytes[pos..]);
                info.version = try dupeString(str.data);
                pos += str.consumed;
            },
            TOOLINFO_ARGUMENTS => {
                const str = try readLengthDelimited(bytes[pos..]);
                try info.arguments.append(allocator, try dupeString(str.data));
                pos += str.consumed;
            },
            else => {
                pos += try skipField(bytes[pos..], tag.wire_type);
            },
        }
    }
    return info;
}

/// Parse a Metadata message
pub fn parseMetadata(bytes: []const u8) !ScipMetadata {
    var meta = ScipMetadata.init();
    errdefer meta.deinit();

    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readTag(bytes[pos..]);
        pos += tag.consumed;

        switch (tag.field_number) {
            METADATA_VERSION => {
                const v = try readVarint(bytes[pos..]);
                meta.version = @intCast(@as(i64, @bitCast(v.value)));
                pos += v.consumed;
            },
            METADATA_TOOL_INFO => {
                const nested = try readLengthDelimited(bytes[pos..]);
                meta.tool_info.deinit();
                meta.tool_info = try parseToolInfo(nested.data);
                pos += nested.consumed;
            },
            METADATA_PROJECT_ROOT => {
                const str = try readLengthDelimited(bytes[pos..]);
                meta.project_root = try dupeString(str.data);
                pos += str.consumed;
            },
            METADATA_TEXT_ENCODING => {
                const v = try readVarint(bytes[pos..]);
                meta.text_encoding = @intCast(@as(i64, @bitCast(v.value)));
                pos += v.consumed;
            },
            else => {
                pos += try skipField(bytes[pos..], tag.wire_type);
            },
        }
    }
    return meta;
}

/// Parse a Document message
pub fn parseDocument(bytes: []const u8) !ScipDocument {
    var doc = ScipDocument.init();
    errdefer doc.deinit();

    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readTag(bytes[pos..]);
        pos += tag.consumed;

        switch (tag.field_number) {
            DOC_RELATIVE_PATH => {
                const str = try readLengthDelimited(bytes[pos..]);
                doc.relative_path = try dupeString(str.data);
                pos += str.consumed;
            },
            DOC_OCCURRENCES => {
                const nested = try readLengthDelimited(bytes[pos..]);
                const occ = try parseOccurrence(nested.data);
                try doc.occurrences.append(allocator, occ);
                pos += nested.consumed;
            },
            DOC_SYMBOLS => {
                const nested = try readLengthDelimited(bytes[pos..]);
                const sym = try parseSymbolInfo(nested.data);
                try doc.symbols.append(allocator, sym);
                pos += nested.consumed;
            },
            DOC_LANGUAGE => {
                const str = try readLengthDelimited(bytes[pos..]);
                doc.language = try dupeString(str.data);
                pos += str.consumed;
            },
            DOC_TEXT => {
                const str = try readLengthDelimited(bytes[pos..]);
                doc.text = try dupeString(str.data);
                pos += str.consumed;
            },
            DOC_POSITION_ENCODING => {
                const v = try readVarint(bytes[pos..]);
                doc.position_encoding = @intCast(@as(i64, @bitCast(v.value)));
                pos += v.consumed;
            },
            else => {
                pos += try skipField(bytes[pos..], tag.wire_type);
            },
        }
    }
    return doc;
}

/// Parse a complete SCIP Index from bytes
pub fn parseIndex(bytes: []const u8) !*ScipIndex {
    var index = try allocator.create(ScipIndex);
    index.* = ScipIndex.init();
    errdefer {
        index.deinit();
        allocator.destroy(index);
    }

    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readTag(bytes[pos..]);
        pos += tag.consumed;

        switch (tag.field_number) {
            INDEX_METADATA => {
                const nested = try readLengthDelimited(bytes[pos..]);
                index.metadata.deinit();
                index.metadata = try parseMetadata(nested.data);
                pos += nested.consumed;
            },
            INDEX_DOCUMENTS => {
                const nested = try readLengthDelimited(bytes[pos..]);
                const doc = try parseDocument(nested.data);
                try index.documents.append(allocator, doc);
                pos += nested.consumed;
            },
            INDEX_EXTERNAL_SYMBOLS => {
                const nested = try readLengthDelimited(bytes[pos..]);
                const sym = try parseSymbolInfo(nested.data);
                try index.external_symbols.append(allocator, sym);
                pos += nested.consumed;
            },
            else => {
                pos += try skipField(bytes[pos..], tag.wire_type);
            },
        }
    }
    return index;
}

// ============================================================================
// C ABI Query Result Structures
// ============================================================================

/// Result for definition/reference queries - holds location info
pub const ScipLocation = extern struct {
    file_path: [*:0]const u8,
    start_line: c_int,
    start_char: c_int,
    end_line: c_int,
    end_char: c_int,
};

/// Result for multi-location queries
pub const ScipLocationList = extern struct {
    locations: [*]ScipLocation,
    count: c_int,
};

/// Result for hover queries
pub const ScipHoverResult = extern struct {
    documentation: [*:0]const u8,
    symbol: [*:0]const u8,
    kind: c_int,
};

// ============================================================================
// C ABI Export Functions for Mojo
// ============================================================================

/// Load and parse a SCIP index file
/// Returns 0 on success, -1 on failure
export fn scip_load_index(path: [*:0]const u8) callconv(.c) c_int {
    // Free existing index if any
    if (loaded_index) |idx| {
        idx.deinit();
        allocator.destroy(idx);
        loaded_index = null;
    }

    const filename = std.mem.span(path);
    std.debug.print("SCIP Reader: Loading index from {s}\n", .{filename});

    // Read file
    const file = fs.cwd().openFile(filename, .{}) catch |err| {
        std.debug.print("SCIP Reader: Failed to open file: {any}\n", .{err});
        return -1;
    };
    defer file.close();

    const file_size = file.getEndPos() catch |err| {
        std.debug.print("SCIP Reader: Failed to get file size: {any}\n", .{err});
        return -1;
    };

    const bytes = allocator.alloc(u8, file_size) catch |err| {
        std.debug.print("SCIP Reader: Failed to allocate: {any}\n", .{err});
        return -1;
    };
    defer allocator.free(bytes);

    const bytes_read = file.readAll(bytes) catch |err| {
        std.debug.print("SCIP Reader: Failed to read file: {any}\n", .{err});
        return -1;
    };

    // Parse index
    loaded_index = parseIndex(bytes[0..bytes_read]) catch |err| {
        std.debug.print("SCIP Reader: Failed to parse index: {any}\n", .{err});
        return -1;
    };

    std.debug.print("SCIP Reader: Loaded {d} documents\n", .{loaded_index.?.documents.items.len});
    return 0;
}

/// Find where a symbol is defined
/// Returns null if not found
export fn scip_find_definition(symbol_ptr: [*:0]const u8) callconv(.c) ?*ScipLocation {
    const index = loaded_index orelse return null;
    const symbol = std.mem.span(symbol_ptr);

    // Search through all documents for a definition occurrence
    for (index.documents.items) |*doc| {
        for (doc.occurrences.items) |*occ| {
            if (std.mem.eql(u8, occ.symbol, symbol) and occ.isDefinition()) {
                // Found definition - create result
                const result = allocator.create(ScipLocation) catch return null;

                // Copy path to null-terminated string
                const path_copy = allocator.allocSentinel(u8, doc.relative_path.len, 0) catch {
                    allocator.destroy(result);
                    return null;
                };
                @memcpy(path_copy, doc.relative_path);

                result.* = .{
                    .file_path = path_copy,
                    .start_line = occ.start_line,
                    .start_char = occ.start_char,
                    .end_line = occ.end_line,
                    .end_char = occ.end_char,
                };
                return result;
            }
        }
    }
    return null;
}

/// Find all references to a symbol
/// Caller must free the result using scip_free_locations
export fn scip_find_references(symbol_ptr: [*:0]const u8) callconv(.c) ?*ScipLocationList {
    const index = loaded_index orelse return null;
    const symbol = std.mem.span(symbol_ptr);

    var locations = std.ArrayListUnmanaged(ScipLocation){};
    defer locations.deinit(allocator);

    // Search through all documents for reference occurrences
    for (index.documents.items) |*doc| {
        for (doc.occurrences.items) |*occ| {
            if (std.mem.eql(u8, occ.symbol, symbol)) {
                // Copy path to null-terminated string
                const path_copy = allocator.allocSentinel(u8, doc.relative_path.len, 0) catch continue;
                @memcpy(path_copy, doc.relative_path);

                locations.append(allocator, .{
                    .file_path = path_copy,
                    .start_line = occ.start_line,
                    .start_char = occ.start_char,
                    .end_line = occ.end_line,
                    .end_char = occ.end_char,
                }) catch continue;
            }
        }
    }

    if (locations.items.len == 0) return null;

    // Create result structure
    const result = allocator.create(ScipLocationList) catch return null;
    const locs = allocator.alloc(ScipLocation, locations.items.len) catch {
        allocator.destroy(result);
        return null;
    };
    @memcpy(locs, locations.items);

    result.* = .{
        .locations = locs.ptr,
        .count = @intCast(locations.items.len),
    };
    return result;
}

/// Free a location list returned by scip_find_references
export fn scip_free_locations(list: ?*ScipLocationList) callconv(.c) void {
    const lst = list orelse return;

    // Free path strings
    const count: usize = @intCast(lst.count);
    for (0..count) |i| {
        const path = lst.locations[i].file_path;
        allocator.free(std.mem.span(path));
    }

    // Free locations array
    allocator.free(lst.locations[0..count]);
    allocator.destroy(lst);
}

/// Get documentation (hover info) at a specific file position
/// Returns null if no symbol found at position
export fn scip_get_hover(
    file_path: [*:0]const u8,
    line: c_int,
    char_pos: c_int,
) callconv(.c) ?*ScipHoverResult {
    const index = loaded_index orelse return null;
    const path = std.mem.span(file_path);

    // Find the document
    const doc = index.findDocument(path) orelse return null;

    // Find occurrence at position
    for (doc.occurrences.items) |*occ| {
        if (positionInRange(line, char_pos, occ)) {
            // Found occurrence - get documentation
            const result = allocator.create(ScipHoverResult) catch return null;

            // Get documentation from symbol info
            var documentation: []const u8 = "";
            var kind: i32 = 0;

            // Check document-local symbols first
            if (doc.findSymbol(occ.symbol)) |info| {
                documentation = info.getDocumentation();
                kind = info.kind;
            } else if (index.findExternalSymbol(occ.symbol)) |info| {
                // Check external symbols
                documentation = info.getDocumentation();
                kind = info.kind;
            }

            // Copy strings
            const doc_copy = allocator.allocSentinel(u8, documentation.len, 0) catch {
                allocator.destroy(result);
                return null;
            };
            if (documentation.len > 0) @memcpy(doc_copy, documentation);

            const sym_copy = allocator.allocSentinel(u8, occ.symbol.len, 0) catch {
                allocator.free(doc_copy[0 .. documentation.len + 1]);
                allocator.destroy(result);
                return null;
            };
            if (occ.symbol.len > 0) @memcpy(sym_copy, occ.symbol);

            result.* = .{
                .documentation = doc_copy,
                .symbol = sym_copy,
                .kind = kind,
            };
            return result;
        }
    }
    return null;
}

/// Check if a position (line, char) is within an occurrence's range
fn positionInRange(line: c_int, char_pos: c_int, occ: *const ScipOccurrence) bool {
    // Check if position is within the occurrence range
    if (line < occ.start_line or line > occ.end_line) return false;

    if (line == occ.start_line and char_pos < occ.start_char) return false;
    if (line == occ.end_line and char_pos >= occ.end_char) return false;

    return true;
}

/// Get the symbol at a specific file position
/// Returns null-terminated symbol string, or null if no symbol at position
export fn scip_get_symbol_at(
    file_path: [*:0]const u8,
    line: c_int,
    char_pos: c_int,
) callconv(.c) ?[*:0]const u8 {
    const index = loaded_index orelse return null;
    const path = std.mem.span(file_path);

    // Find the document
    const doc = index.findDocument(path) orelse return null;

    // Find occurrence at position
    for (doc.occurrences.items) |*occ| {
        if (positionInRange(line, char_pos, occ)) {
            // Copy symbol to null-terminated string
            const sym_copy = allocator.allocSentinel(u8, occ.symbol.len, 0) catch return null;
            if (occ.symbol.len > 0) @memcpy(sym_copy, occ.symbol);
            return sym_copy;
        }
    }
    return null;
}

/// Free a single location returned by scip_find_definition
export fn scip_free_location(loc: ?*ScipLocation) callconv(.c) void {
    const location = loc orelse return;
    allocator.free(std.mem.span(location.file_path));
    allocator.destroy(location);
}

/// Free a hover result
export fn scip_free_hover(hover: ?*ScipHoverResult) callconv(.c) void {
    const h = hover orelse return;
    allocator.free(std.mem.span(h.documentation));
    allocator.free(std.mem.span(h.symbol));
    allocator.destroy(h);
}

/// Free a symbol string returned by scip_get_symbol_at
export fn scip_free_symbol(symbol: ?[*:0]const u8) callconv(.c) void {
    const sym = symbol orelse return;
    allocator.free(std.mem.span(sym));
}

/// Free the loaded index and release all memory
export fn scip_free_index() callconv(.c) void {
    if (loaded_index) |idx| {
        idx.deinit();
        allocator.destroy(idx);
        loaded_index = null;
        std.debug.print("SCIP Reader: Index freed\n", .{});
    }
}

// ============================================================================
// Additional Query Helpers (Zig API)
// ============================================================================

/// Get the number of documents in the loaded index
export fn scip_get_document_count() callconv(.c) c_int {
    const index = loaded_index orelse return 0;
    return @intCast(index.documents.items.len);
}

/// Get the number of external symbols in the loaded index
export fn scip_get_external_symbol_count() callconv(.c) c_int {
    const index = loaded_index orelse return 0;
    return @intCast(index.external_symbols.items.len);
}

/// Get the project root from metadata
export fn scip_get_project_root() callconv(.c) ?[*:0]const u8 {
    const index = loaded_index orelse return null;
    if (index.metadata.project_root.len == 0) return null;

    const copy = allocator.allocSentinel(u8, index.metadata.project_root.len, 0) catch return null;
    @memcpy(copy, index.metadata.project_root);
    return copy;
}

/// Get the tool name from metadata
export fn scip_get_tool_name() callconv(.c) ?[*:0]const u8 {
    const index = loaded_index orelse return null;
    if (index.metadata.tool_info.name.len == 0) return null;

    const copy = allocator.allocSentinel(u8, index.metadata.tool_info.name.len, 0) catch return null;
    @memcpy(copy, index.metadata.tool_info.name);
    return copy;
}

/// Get the tool version from metadata
export fn scip_get_tool_version() callconv(.c) ?[*:0]const u8 {
    const index = loaded_index orelse return null;
    if (index.metadata.tool_info.version.len == 0) return null;

    const copy = allocator.allocSentinel(u8, index.metadata.tool_info.version.len, 0) catch return null;
    @memcpy(copy, index.metadata.tool_info.version);
    return copy;
}

/// Get document path by index
export fn scip_get_document_path(doc_index: c_int) callconv(.c) ?[*:0]const u8 {
    const index = loaded_index orelse return null;
    const idx: usize = @intCast(doc_index);

    if (idx >= index.documents.items.len) return null;

    const path = index.documents.items[idx].relative_path;
    if (path.len == 0) return null;

    const copy = allocator.allocSentinel(u8, path.len, 0) catch return null;
    @memcpy(copy, path);
    return copy;
}

/// Get document language by index
export fn scip_get_document_language(doc_index: c_int) callconv(.c) ?[*:0]const u8 {
    const index = loaded_index orelse return null;
    const idx: usize = @intCast(doc_index);

    if (idx >= index.documents.items.len) return null;

    const lang = index.documents.items[idx].language;
    if (lang.len == 0) return null;

    const copy = allocator.allocSentinel(u8, lang.len, 0) catch return null;
    @memcpy(copy, lang);
    return copy;
}

/// Get occurrence count for a document
export fn scip_get_occurrence_count(doc_index: c_int) callconv(.c) c_int {
    const index = loaded_index orelse return 0;
    const idx: usize = @intCast(doc_index);

    if (idx >= index.documents.items.len) return 0;

    return @intCast(index.documents.items[idx].occurrences.items.len);
}

/// Get symbol count for a document
export fn scip_get_symbol_count(doc_index: c_int) callconv(.c) c_int {
    const index = loaded_index orelse return 0;
    const idx: usize = @intCast(doc_index);

    if (idx >= index.documents.items.len) return 0;

    return @intCast(index.documents.items[idx].symbols.items.len);
}
