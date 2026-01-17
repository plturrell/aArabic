const std = @import("std");
const mem = std.mem;
const fs = std.fs;

// ============================================================================
// HyperShimmy File Upload Handler
// ============================================================================
//
// Day 16 Implementation: File upload endpoint with multipart/form-data support
//
// Features:
// - Parse multipart/form-data requests
// - Validate file types (PDF, TXT, HTML)
// - Save uploaded files
// - Extract text content using parsers
// - Return upload metadata
//
// Endpoints:
// - POST /api/upload - Upload file
// ============================================================================

const pdf_parser = @import("pdf_parser");
const html_parser = @import("html_parser");

/// File upload result
pub const UploadResult = struct {
    success: bool,
    file_id: []const u8,
    filename: []const u8,
    file_type: []const u8,
    size: usize,
    text_length: usize,
    error_message: ?[]const u8 = null,
    
    pub fn deinit(self: *UploadResult, allocator: mem.Allocator) void {
        allocator.free(self.file_id);
        allocator.free(self.filename);
        allocator.free(self.file_type);
        if (self.error_message) |msg| {
            allocator.free(msg);
        }
    }
};

/// Supported file types
pub const FileType = enum {
    PDF,
    Text,
    HTML,
    Unknown,
    
    pub fn fromExtension(ext: []const u8) FileType {
        if (mem.eql(u8, ext, ".pdf")) return .PDF;
        if (mem.eql(u8, ext, ".txt")) return .Text;
        if (mem.eql(u8, ext, ".html") or mem.eql(u8, ext, ".htm")) return .HTML;
        return .Unknown;
    }
    
    pub fn toString(self: FileType) []const u8 {
        return switch (self) {
            .PDF => "application/pdf",
            .Text => "text/plain",
            .HTML => "text/html",
            .Unknown => "application/octet-stream",
        };
    }
};

/// Multipart boundary parser
pub const MultipartParser = struct {
    boundary: []const u8,
    allocator: mem.Allocator,
    
    pub fn init(allocator: mem.Allocator, content_type: []const u8) !MultipartParser {
        // Extract boundary from Content-Type header
        // Example: "multipart/form-data; boundary=----WebKitFormBoundary..."
        const boundary_start = mem.indexOf(u8, content_type, "boundary=") orelse return error.NoBoundary;
        const boundary = content_type[boundary_start + 9..];
        
        return MultipartParser{
            .boundary = try allocator.dupe(u8, boundary),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MultipartParser) void {
        self.allocator.free(self.boundary);
    }
    
    /// Parse multipart data and extract file
    pub fn parseFile(self: *MultipartParser, data: []const u8) !struct {
        filename: []const u8,
        content: []const u8,
    } {
        // Find the boundary markers
        const boundary_marker = try std.fmt.allocPrint(
            self.allocator,
            "--{s}",
            .{self.boundary},
        );
        defer self.allocator.free(boundary_marker);
        
        // Find first boundary
        const first_boundary = mem.indexOf(u8, data, boundary_marker) orelse return error.NoBoundaryFound;
        const after_first = data[first_boundary + boundary_marker.len..];
        
        // Skip to headers
        const headers_start = mem.indexOf(u8, after_first, "\r\n") orelse return error.InvalidFormat;
        const headers_section = after_first[headers_start + 2..];
        
        // Find Content-Disposition header
        const content_disp = mem.indexOf(u8, headers_section, "Content-Disposition:") orelse return error.NoContentDisposition;
        const disp_line_end = mem.indexOf(u8, headers_section[content_disp..], "\r\n") orelse return error.InvalidFormat;
        const disp_line = headers_section[content_disp .. content_disp + disp_line_end];
        
        // Extract filename
        const filename_start = mem.indexOf(u8, disp_line, "filename=\"") orelse return error.NoFilename;
        const filename_value = disp_line[filename_start + 10..];
        const filename_end = mem.indexOf(u8, filename_value, "\"") orelse return error.NoFilename;
        const filename = filename_value[0..filename_end];
        
        // Find start of file content (after \r\n\r\n)
        const content_start = mem.indexOf(u8, headers_section, "\r\n\r\n") orelse return error.InvalidFormat;
        const file_content_section = headers_section[content_start + 4..];
        
        // Find end boundary
        const end_boundary = mem.indexOf(u8, file_content_section, boundary_marker) orelse return error.NoEndBoundary;
        const file_content = file_content_section[0..end_boundary - 2]; // Remove trailing \r\n
        
        return .{
            .filename = try self.allocator.dupe(u8, filename),
            .content = try self.allocator.dupe(u8, file_content),
        };
    }
};

/// File upload handler
pub const UploadHandler = struct {
    allocator: mem.Allocator,
    upload_dir: []const u8,
    
    pub fn init(allocator: mem.Allocator, upload_dir: []const u8) !UploadHandler {
        // Create upload directory if it doesn't exist
        fs.cwd().makeDir(upload_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };
        
        return UploadHandler{
            .allocator = allocator,
            .upload_dir = upload_dir,
        };
    }
    
    /// Handle file upload request
    pub fn handleUpload(
        self: *UploadHandler,
        content_type: []const u8,
        body: []const u8,
    ) !UploadResult {
        // Parse multipart data
        var parser = try MultipartParser.init(self.allocator, content_type);
        defer parser.deinit();
        
        const file_data = try parser.parseFile(body);
        defer {
            self.allocator.free(file_data.filename);
            self.allocator.free(file_data.content);
        }
        
        // Validate file type
        const ext = fs.path.extension(file_data.filename);
        const file_type = FileType.fromExtension(ext);
        
        if (file_type == .Unknown) {
            return UploadResult{
                .success = false,
                .file_id = try self.allocator.dupe(u8, ""),
                .filename = try self.allocator.dupe(u8, file_data.filename),
                .file_type = try self.allocator.dupe(u8, "unknown"),
                .size = 0,
                .text_length = 0,
                .error_message = try self.allocator.dupe(u8, "Unsupported file type. Supported: PDF, TXT, HTML"),
            };
        }
        
        // Generate unique file ID
        const file_id = try self.generateFileId();
        
        // Save file
        const file_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}{s}",
            .{ self.upload_dir, file_id, ext },
        );
        defer self.allocator.free(file_path);
        
        const file = try fs.cwd().createFile(file_path, .{});
        defer file.close();
        try file.writeAll(file_data.content);
        
        // Extract text content
        const text_content = try self.extractText(file_type, file_data.content);
        defer self.allocator.free(text_content);
        
        // Save text content
        const text_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}.txt",
            .{ self.upload_dir, file_id },
        );
        defer self.allocator.free(text_path);
        
        const text_file = try fs.cwd().createFile(text_path, .{});
        defer text_file.close();
        try text_file.writeAll(text_content);
        
        return UploadResult{
            .success = true,
            .file_id = file_id,
            .filename = try self.allocator.dupe(u8, file_data.filename),
            .file_type = try self.allocator.dupe(u8, file_type.toString()),
            .size = file_data.content.len,
            .text_length = text_content.len,
            .error_message = null,
        };
    }
    
    /// Generate unique file ID
    fn generateFileId(self: *UploadHandler) ![]const u8 {
        const timestamp = std.time.timestamp();
        const random = std.crypto.random.int(u32);
        
        return try std.fmt.allocPrint(
            self.allocator,
            "{d}_{x}",
            .{ timestamp, random },
        );
    }
    
    /// Extract text from file based on type
    fn extractText(self: *UploadHandler, file_type: FileType, content: []const u8) ![]const u8 {
        return switch (file_type) {
            .PDF => try self.extractPdfText(content),
            .HTML => try self.extractHtmlText(content),
            .Text => try self.allocator.dupe(u8, content),
            .Unknown => try self.allocator.dupe(u8, ""),
        };
    }
    
    /// Extract text from PDF
    fn extractPdfText(self: *UploadHandler, pdf_data: []const u8) ![]const u8 {
        var parser = pdf_parser.PdfParser.init(self.allocator);
        defer parser.deinit();
        
        var doc = parser.parse(pdf_data) catch |err| {
            std.debug.print("PDF parse error: {any}\n", .{err});
            return try std.fmt.allocPrint(
                self.allocator,
                "[PDF parsing failed: {any}]",
                .{err},
            );
        };
        defer doc.deinit();
        
        const text = doc.getText() catch |err| {
            std.debug.print("PDF text extraction error: {any}\n", .{err});
            return try std.fmt.allocPrint(
                self.allocator,
                "[PDF text extraction failed: {any}]",
                .{err},
            );
        };
        
        return text;
    }
    
    /// Extract text from HTML
    fn extractHtmlText(self: *UploadHandler, html_data: []const u8) ![]const u8 {
        var parser = html_parser.HtmlParser.init(self.allocator);
        defer parser.deinit();
        
        var doc = parser.parse(html_data) catch |err| {
            std.debug.print("HTML parse error: {any}\n", .{err});
            return try std.fmt.allocPrint(
                self.allocator,
                "[HTML parsing failed: {any}]",
                .{err},
            );
        };
        defer doc.deinit();
        
        var text_buffer = std.ArrayListUnmanaged(u8){};
        defer text_buffer.deinit(self.allocator);
        
        try doc.getText(&text_buffer);
        
        return try text_buffer.toOwnedSlice(self.allocator);
    }
    
    /// Format upload result as JSON
    pub fn resultToJson(self: *UploadHandler, result: *const UploadResult) ![]const u8 {
        if (!result.success) {
            return try std.fmt.allocPrint(
                self.allocator,
                \\{{"success":false,"error":"{s}","filename":"{s}"}}
                ,
                .{ result.error_message.?, result.filename },
            );
        }
        
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{"success":true,"fileId":"{s}","filename":"{s}","fileType":"{s}","size":{d},"textLength":{d}}}
            ,
            .{
                result.file_id,
                result.filename,
                result.file_type,
                result.size,
                result.text_length,
            },
        );
    }
};

// ============================================================================
// Tests
// ============================================================================

test "file type from extension" {
    try std.testing.expectEqual(FileType.PDF, FileType.fromExtension(".pdf"));
    try std.testing.expectEqual(FileType.Text, FileType.fromExtension(".txt"));
    try std.testing.expectEqual(FileType.HTML, FileType.fromExtension(".html"));
    try std.testing.expectEqual(FileType.Unknown, FileType.fromExtension(".exe"));
}

test "file type to string" {
    try std.testing.expectEqualStrings("application/pdf", FileType.PDF.toString());
    try std.testing.expectEqualStrings("text/plain", FileType.Text.toString());
    try std.testing.expectEqualStrings("text/html", FileType.HTML.toString());
}

test "upload handler init" {
    const test_dir = "test_uploads";
    defer fs.cwd().deleteTree(test_dir) catch {};
    
    const handler = try UploadHandler.init(std.testing.allocator, test_dir);
    _ = handler;
    
    // Check directory was created
    const dir = try fs.cwd().openDir(test_dir, .{});
    dir.close();
}
