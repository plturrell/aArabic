const std = @import("std");
const fs = std.fs;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// ============================================================================
// Global State
// ============================================================================
var output_file: ?fs.File = null;

// Current document state for streaming writes
var current_document: ?DocumentBuilder = null;

// ============================================================================
// File Writer Wrapper
// ============================================================================
// Wraps fs.File to provide writeByte/writeAll interface expected by helpers
const FileWriter = struct {
    file: fs.File,

    pub fn writeByte(self: FileWriter, byte: u8) !void {
        try self.file.writeAll(&[1]u8{byte});
    }

    pub fn writeAll(self: FileWriter, bytes: []const u8) !void {
        try self.file.writeAll(bytes);
    }
};

// ============================================================================
// Document Builder - accumulates occurrences and symbols for a document
// ============================================================================
const DocumentBuilder = struct {
    language: []const u8,
    relative_path: []const u8,
    occurrences: std.ArrayListUnmanaged(u8),
    symbols: std.ArrayListUnmanaged(u8),
    occurrence_count: u32,
    symbol_count: u32,

    fn init(language: []const u8, relative_path: []const u8) DocumentBuilder {
        return .{
            .language = language,
            .relative_path = relative_path,
            .occurrences = std.ArrayListUnmanaged(u8){},
            .symbols = std.ArrayListUnmanaged(u8){},
            .occurrence_count = 0,
            .symbol_count = 0,
        };
    }

    fn deinit(self: *DocumentBuilder) void {
        self.occurrences.deinit(allocator);
        self.symbols.deinit(allocator);
    }
};

// ============================================================================
// Protobuf Wire Types
// ============================================================================
const WIRE_VARINT: u3 = 0;
const WIRE_FIXED64: u3 = 1;
const WIRE_LENGTH_DELIMITED: u3 = 2;
const WIRE_FIXED32: u3 = 5;

// ============================================================================
// Protobuf Helpers
// ============================================================================

fn writeVarint(writer: anytype, value: u64) !void {
    var v = value;
    while (v >= 0x80) {
        try writer.writeByte(@as(u8, @intCast((v & 0x7F) | 0x80)));
        v >>= 7;
    }
    try writer.writeByte(@as(u8, @intCast(v)));
}

/// Writes a signed varint using ZigZag encoding (for sint32/sint64 fields)
fn writeSignedVarint(writer: anytype, value: i64) !void {
    // ZigZag encoding: (n << 1) ^ (n >> 63)
    const encoded: u64 = @bitCast((value << 1) ^ (value >> 63));
    try writeVarint(writer, encoded);
}

fn writeTag(writer: anytype, field_number: u32, wire_type: u3) !void {
    const tag = (@as(u64, field_number) << 3) | wire_type;
    try writeVarint(writer, tag);
}

fn writeStringField(writer: anytype, field_number: u32, value: []const u8) !void {
    try writeTag(writer, field_number, WIRE_LENGTH_DELIMITED);
    try writeVarint(writer, value.len);
    try writer.writeAll(value);
}

/// Writes a nested message field (length-delimited)
fn writeNestedMessage(writer: anytype, field_number: u32, message_bytes: []const u8) !void {
    try writeTag(writer, field_number, WIRE_LENGTH_DELIMITED);
    try writeVarint(writer, message_bytes.len);
    try writer.writeAll(message_bytes);
}

/// Writes an int32/int64 field (varint encoding)
fn writeVarintField(writer: anytype, field_number: u32, value: u64) !void {
    if (value == 0) return; // Skip default values
    try writeTag(writer, field_number, WIRE_VARINT);
    try writeVarint(writer, value);
}

/// Writes a signed int32/int64 field (ZigZag varint encoding)
fn writeSignedVarintField(writer: anytype, field_number: u32, value: i64) !void {
    if (value == 0) return; // Skip default values
    try writeTag(writer, field_number, WIRE_VARINT);
    try writeSignedVarint(writer, value);
}

/// Writes a bool field
fn writeBoolField(writer: anytype, field_number: u32, value: bool) !void {
    if (!value) return; // Skip default values
    try writeTag(writer, field_number, WIRE_VARINT);
    try writer.writeByte(1);
}

/// Writes a packed repeated int32 field
fn writePackedInt32Field(writer: anytype, field_number: u32, values: []const i32) !void {
    if (values.len == 0) return;

    // First calculate the size of the packed data
    var size_buf = std.ArrayListUnmanaged(u8){};
    defer size_buf.deinit(allocator);
    const size_writer = size_buf.writer(allocator);

    for (values) |v| {
        try writeVarint(size_writer, @as(u64, @bitCast(@as(i64, v))));
    }

    try writeTag(writer, field_number, WIRE_LENGTH_DELIMITED);
    try writeVarint(writer, size_buf.items.len);
    try writer.writeAll(size_buf.items);
}

// ============================================================================ 
// C ABI Exports for Mojo
// ============================================================================ 

pub export fn scip_init(path: [*:0]const u8) callconv(.c) c_int {
    const filename = std.mem.span(path);
    std.debug.print("Zig: Initializing SCIP index at {s}\n", .{filename});

    output_file = fs.cwd().createFile(filename, .{}) catch |err| {
        std.debug.print("Zig: Failed to create file: {any}\n", .{err});
        return -1;
    };

    return 0;
}

fn writeMetadataInternal(
    tool_name: []const u8,
    tool_version: []const u8,
    project_root: []const u8,
) !void {
    if (output_file == null) return error.NoFileOpen;
    const file = output_file.?;
    const file_writer = FileWriter{ .file = file };

    // 1. Build Metadata content in memory first
    var meta_buf = std.ArrayListUnmanaged(u8){};
    defer meta_buf.deinit(allocator);
    const meta_writer = meta_buf.writer(allocator);

    // --- ToolInfo (Field 2 of Metadata) ---
    var tool_info_buf = std.ArrayListUnmanaged(u8){};
    defer tool_info_buf.deinit(allocator);

    // ToolInfo.name (Field 1)
    try writeStringField(tool_info_buf.writer(allocator), 1, tool_name);
    // ToolInfo.version (Field 2)
    try writeStringField(tool_info_buf.writer(allocator), 2, tool_version);

    // Write Metadata.tool_info (Field 2) to meta_buf
    try writeNestedMessage(meta_writer, 2, tool_info_buf.items);

    // Write Metadata.project_root (Field 3) to meta_buf
    try writeStringField(meta_writer, 3, project_root);

    // 2. Write Index message to file
    // Index.metadata is Field 1
    try writeNestedMessage(file_writer, 1, meta_buf.items);
}

pub export fn scip_write_metadata(
    tool_name: [*:0]const u8,
    tool_version: [*:0]const u8,
    project_root: [*:0]const u8,
) callconv(.c) c_int {
    writeMetadataInternal(
        std.mem.span(tool_name),
        std.mem.span(tool_version),
        std.mem.span(project_root),
    ) catch |err| {
        std.debug.print("Zig: Error writing metadata: {any}\n", .{err});
        return -1;
    };
    return 0;
}

// ============================================================================
// Document Writing Functions
// ============================================================================

/// Begins a new document. Must be called before adding occurrences or symbols.
fn beginDocumentInternal(language: []const u8, relative_path: []const u8) !void {
    // Clean up any existing document
    if (current_document != null) {
        var doc = current_document.?;
        doc.deinit();
        current_document = null;
    }

    current_document = DocumentBuilder.init(language, relative_path);
}

pub export fn scip_begin_document(
    language: [*:0]const u8,
    relative_path: [*:0]const u8,
) callconv(.c) c_int {
    beginDocumentInternal(
        std.mem.span(language),
        std.mem.span(relative_path),
    ) catch |err| {
        std.debug.print("Zig: Error beginning document: {any}\n", .{err});
        return -1;
    };
    return 0;
}

/// Writes a complete Document message to the output file
fn writeDocumentInternal(language: []const u8, relative_path: []const u8) !void {
    if (output_file == null) return error.NoFileOpen;
    const file = output_file.?;
    const file_writer = FileWriter{ .file = file };

    var doc_buf = std.ArrayListUnmanaged(u8){};
    defer doc_buf.deinit(allocator);
    const doc_writer = doc_buf.writer(allocator);

    // Document.relative_path (Field 1)
    try writeStringField(doc_writer, 1, relative_path);

    // Document.language (Field 4)
    if (language.len > 0) {
        try writeStringField(doc_writer, 4, language);
    }

    // Include occurrences and symbols from current document builder if available
    if (current_document) |doc| {
        // Document.occurrences (Field 2) - already serialized as repeated messages
        if (doc.occurrences.items.len > 0) {
            try doc_writer.writeAll(doc.occurrences.items);
        }
        // Document.symbols (Field 3) - already serialized as repeated messages
        if (doc.symbols.items.len > 0) {
            try doc_writer.writeAll(doc.symbols.items);
        }
    }

    // Write Index.documents (Field 2)
    try writeNestedMessage(file_writer, 2, doc_buf.items);
}

/// Writes a Document message with language and relative_path
pub export fn scip_write_document(
    language: [*:0]const u8,
    relative_path: [*:0]const u8,
) callconv(.c) c_int {
    writeDocumentInternal(
        std.mem.span(language),
        std.mem.span(relative_path),
    ) catch |err| {
        std.debug.print("Zig: Error writing document: {any}\n", .{err});
        return -1;
    };

    // Clean up current document
    if (current_document != null) {
        var doc = current_document.?;
        doc.deinit();
        current_document = null;
    }

    return 0;
}

// ============================================================================
// Occurrence Writing Functions
// ============================================================================

/// Adds an occurrence to the current document
fn addOccurrenceInternal(
    start_line: i32,
    start_char: i32,
    end_line: i32,
    end_char: i32,
    symbol: []const u8,
    symbol_roles: i32,
) !void {
    if (current_document == null) return error.NoDocumentOpen;
    var doc = &current_document.?;

    var occ_buf = std.ArrayListUnmanaged(u8){};
    defer occ_buf.deinit(allocator);
    const occ_writer = occ_buf.writer(allocator);

    // Occurrence.range (Field 1) - packed repeated int32
    // Use 3 elements if same line, 4 elements otherwise
    if (start_line == end_line) {
        const range = [_]i32{ start_line, start_char, end_char };
        try writePackedInt32Field(occ_writer, 1, &range);
    } else {
        const range = [_]i32{ start_line, start_char, end_line, end_char };
        try writePackedInt32Field(occ_writer, 1, &range);
    }

    // Occurrence.symbol (Field 2)
    if (symbol.len > 0) {
        try writeStringField(occ_writer, 2, symbol);
    }

    // Occurrence.symbol_roles (Field 3)
    if (symbol_roles != 0) {
        try writeVarintField(occ_writer, 3, @as(u64, @bitCast(@as(i64, symbol_roles))));
    }

    // Write as a repeated field entry for Document.occurrences (Field 2)
    try writeNestedMessage(doc.occurrences.writer(allocator), 2, occ_buf.items);
    doc.occurrence_count += 1;
}

/// Adds an occurrence to the current document (range, symbol, roles)
pub export fn scip_add_occurrence(
    start_line: c_int,
    start_char: c_int,
    end_line: c_int,
    end_char: c_int,
    symbol: [*:0]const u8,
    symbol_roles: c_int,
) callconv(.c) c_int {
    addOccurrenceInternal(
        @intCast(start_line),
        @intCast(start_char),
        @intCast(end_line),
        @intCast(end_char),
        std.mem.span(symbol),
        @intCast(symbol_roles),
    ) catch |err| {
        std.debug.print("Zig: Error adding occurrence: {any}\n", .{err});
        return -1;
    };
    return 0;
}

// ============================================================================
// Symbol Information Writing Functions
// ============================================================================

/// Adds symbol information to the current document
fn addSymbolInfoInternal(
    symbol: []const u8,
    documentation: []const u8,
    kind: i32,
) !void {
    if (current_document == null) return error.NoDocumentOpen;
    var doc = &current_document.?;

    var sym_buf = std.ArrayListUnmanaged(u8){};
    defer sym_buf.deinit(allocator);
    const sym_writer = sym_buf.writer(allocator);

    // SymbolInformation.symbol (Field 1)
    try writeStringField(sym_writer, 1, symbol);

    // SymbolInformation.documentation (Field 3) - repeated string
    if (documentation.len > 0) {
        try writeStringField(sym_writer, 3, documentation);
    }

    // SymbolInformation.kind (Field 5)
    if (kind != 0) {
        try writeVarintField(sym_writer, 5, @as(u64, @bitCast(@as(i64, kind))));
    }

    // Write as a repeated field entry for Document.symbols (Field 3)
    try writeNestedMessage(doc.symbols.writer(allocator), 3, sym_buf.items);
    doc.symbol_count += 1;
}

/// Adds symbol information (symbol string, documentation, kind)
pub export fn scip_add_symbol_info(
    symbol: [*:0]const u8,
    documentation: [*:0]const u8,
    kind: c_int,
) callconv(.c) c_int {
    addSymbolInfoInternal(
        std.mem.span(symbol),
        std.mem.span(documentation),
        @intCast(kind),
    ) catch |err| {
        std.debug.print("Zig: Error adding symbol info: {any}\n", .{err});
        return -1;
    };
    return 0;
}

// ============================================================================
// Relationship Writing Functions
// ============================================================================

/// Builds a Relationship message
fn buildRelationship(
    symbol: []const u8,
    is_reference: bool,
    is_implementation: bool,
    is_type_definition: bool,
    is_definition: bool,
) ![]u8 {
    var rel_buf = std.ArrayListUnmanaged(u8){};
    errdefer rel_buf.deinit(allocator);
    const rel_writer = rel_buf.writer(allocator);

    // Relationship.symbol (Field 1)
    try writeStringField(rel_writer, 1, symbol);

    // Relationship.is_reference (Field 2)
    try writeBoolField(rel_writer, 2, is_reference);

    // Relationship.is_implementation (Field 3)
    try writeBoolField(rel_writer, 3, is_implementation);

    // Relationship.is_type_definition (Field 4)
    try writeBoolField(rel_writer, 4, is_type_definition);

    // Relationship.is_definition (Field 5)
    try writeBoolField(rel_writer, 5, is_definition);

    return rel_buf.toOwnedSlice(allocator);
}

/// Adds a relationship to a symbol in the current document
fn addRelationshipInternal(
    symbol: []const u8,
    related_symbol: []const u8,
    is_reference: bool,
    is_implementation: bool,
    is_type_definition: bool,
    is_definition: bool,
) !void {
    if (current_document == null) return error.NoDocumentOpen;
    var doc = &current_document.?;

    // Build SymbolInformation with relationship
    var sym_buf = std.ArrayListUnmanaged(u8){};
    defer sym_buf.deinit(allocator);
    const sym_writer = sym_buf.writer(allocator);

    // SymbolInformation.symbol (Field 1)
    try writeStringField(sym_writer, 1, symbol);

    // Build the relationship
    const rel_bytes = try buildRelationship(
        related_symbol,
        is_reference,
        is_implementation,
        is_type_definition,
        is_definition,
    );
    defer allocator.free(rel_bytes);

    // SymbolInformation.relationships (Field 4) - repeated Relationship
    try writeNestedMessage(sym_writer, 4, rel_bytes);

    // Write as a repeated field entry for Document.symbols (Field 3)
    try writeNestedMessage(doc.symbols.writer(allocator), 3, sym_buf.items);
}

/// Adds a relationship between symbols
pub export fn scip_add_relationship(
    symbol: [*:0]const u8,
    related_symbol: [*:0]const u8,
    is_reference: c_int,
    is_implementation: c_int,
    is_type_definition: c_int,
    is_definition: c_int,
) callconv(.c) c_int {
    addRelationshipInternal(
        std.mem.span(symbol),
        std.mem.span(related_symbol),
        is_reference != 0,
        is_implementation != 0,
        is_type_definition != 0,
        is_definition != 0,
    ) catch |err| {
        std.debug.print("Zig: Error adding relationship: {any}\n", .{err});
        return -1;
    };
    return 0;
}

// ============================================================================
// External Symbol Writing Functions
// ============================================================================

/// Writes an external symbol reference directly to the index
fn writeExternalSymbolInternal(
    symbol: []const u8,
    documentation: []const u8,
    kind: i32,
) !void {
    if (output_file == null) return error.NoFileOpen;
    const file = output_file.?;
    const file_writer = FileWriter{ .file = file };

    var sym_buf = std.ArrayListUnmanaged(u8){};
    defer sym_buf.deinit(allocator);
    const sym_writer = sym_buf.writer(allocator);

    // SymbolInformation.symbol (Field 1)
    try writeStringField(sym_writer, 1, symbol);

    // SymbolInformation.documentation (Field 3) - repeated string
    if (documentation.len > 0) {
        try writeStringField(sym_writer, 3, documentation);
    }

    // SymbolInformation.kind (Field 5)
    if (kind != 0) {
        try writeVarintField(sym_writer, 5, @as(u64, @bitCast(@as(i64, kind))));
    }

    // Write Index.external_symbols (Field 3)
    try writeNestedMessage(file_writer, 3, sym_buf.items);
}

/// Writes an external symbol reference
pub export fn scip_write_external_symbol(
    symbol: [*:0]const u8,
    documentation: [*:0]const u8,
    kind: c_int,
) callconv(.c) c_int {
    writeExternalSymbolInternal(
        std.mem.span(symbol),
        std.mem.span(documentation),
        @intCast(kind),
    ) catch |err| {
        std.debug.print("Zig: Error writing external symbol: {any}\n", .{err});
        return -1;
    };
    return 0;
}

// ============================================================================
// Close Function
// ============================================================================

pub export fn scip_close() callconv(.c) c_int {
    // Clean up any open document
    if (current_document != null) {
        var doc = current_document.?;
        doc.deinit();
        current_document = null;
    }

    if (output_file) |file| {
        file.close();
        output_file = null;
        std.debug.print("Zig: SCIP index closed.\n", .{});
        return 0;
    }
    return -1;
}