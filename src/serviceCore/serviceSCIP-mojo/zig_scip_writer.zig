const std = @import("std");
const fs = std.fs;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// ============================================================================ 
// Global State
// ============================================================================ 
var output_file: ?fs.File = null;

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

fn writeTag(writer: anytype, field_number: u32, wire_type: u3) !void {
    const tag = (@as(u64, field_number) << 3) | wire_type;
    try writeVarint(writer, tag);
}

fn writeStringField(writer: anytype, field_number: u32, value: []const u8) !void {
    try writeTag(writer, field_number, 2); // Wire type 2 = Length Delimited
    try writeVarint(writer, value.len);
    try writer.writeAll(value);
}

// ============================================================================ 
// C ABI Exports for Mojo
// ============================================================================ 

export fn scip_init(path: [*:0]const u8) callconv(.c) c_int {
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
    project_root: []const u8
) !void {
    if (output_file == null) return error.NoFileOpen;
    const file = output_file.?;
    const file_writer = FileWriter{ .file = file };

    // 1. Build Metadata content in memory first
    var meta_buf = std.ArrayListUnmanaged(u8){};
    defer meta_buf.deinit(allocator);
    const meta_writer = meta_buf.writer(allocator);

    // --- ToolInfo (Field 1 of Metadata) ---
    var tool_info_buf = std.ArrayListUnmanaged(u8){};
    defer tool_info_buf.deinit(allocator);
    
    // ToolInfo.name (Field 1)
    try writeStringField(tool_info_buf.writer(allocator), 1, tool_name);
    // ToolInfo.version (Field 2)
    try writeStringField(tool_info_buf.writer(allocator), 2, tool_version);

    // Write Metadata.tool_info (Field 1) to meta_buf
    try writeTag(meta_writer, 1, 2);
    try writeVarint(meta_writer, tool_info_buf.items.len);
    try meta_writer.writeAll(tool_info_buf.items);

    // Write Metadata.project_root (Field 2) to meta_buf
    try writeStringField(meta_writer, 2, project_root);

    // 2. Write Index message to file
    // Index.metadata is Field 1
    try writeTag(file_writer, 1, 2);
    try writeVarint(file_writer, meta_buf.items.len);
    try file_writer.writeAll(meta_buf.items);
}

export fn scip_write_metadata(
    tool_name: [*:0]const u8,
    tool_version: [*:0]const u8,
    project_root: [*:0]const u8
) callconv(.c) c_int {
    writeMetadataInternal(
        std.mem.span(tool_name),
        std.mem.span(tool_version),
        std.mem.span(project_root)
    ) catch |err| {
        std.debug.print("Zig: Error writing metadata: {any}\n", .{err});
        return -1;
    };
    return 0;
}

export fn scip_close() callconv(.c) c_int {
    if (output_file) |file| {
        file.close();
        output_file = null;
        std.debug.print("Zig: SCIP index closed.\n", .{});
        return 0;
    }
    return -1;
}