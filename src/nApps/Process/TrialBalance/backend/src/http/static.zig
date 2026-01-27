//! ============================================================================
//! Static File Server Handler
//! Serves webapp static files (HTML, CSS, JS, etc.)
//! ============================================================================
//!
//! [CODE:file=static.zig]
//! [CODE:module=http]
//! [CODE:language=zig]
//!
//! [RELATION:called_by=CODE:main.zig]
//! [RELATION:calls=CODE:mime.zig]
//!
//! Note: Infrastructure code - no ODPS business rules implemented here.

const std = @import("std");
const mime = @import("mime.zig");

pub fn serveStaticFile(allocator: std.mem.Allocator, root_dir: []const u8, path: []const u8) ![]u8 {
    // Security: Prevent directory traversal
    if (std.mem.indexOf(u8, path, "..")) |_| {
        return try allocator.dupe(u8, "HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\n\r\nDirectory traversal not allowed");
    }

    // Remove query parameters
    var clean_path = path;
    if (std.mem.indexOf(u8, path, "?")) |idx| {
        clean_path = path[0..idx];
    }

    // Default to index.html for root
    var file_path: []const u8 = undefined;
    if (std.mem.eql(u8, clean_path, "/")) {
        file_path = try std.fmt.allocPrint(allocator, "{s}/index.html", .{root_dir});
    } else {
        // Remove leading slash
        const safe_path = if (clean_path.len > 0 and clean_path[0] == '/') clean_path[1..] else clean_path;
        file_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root_dir, safe_path });
    }
    defer allocator.free(file_path);

    // Try to open and read the file
    const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            // Try adding index.html if it's a directory request
            const index_path = try std.fmt.allocPrint(allocator, "{s}/index.html", .{file_path});
            defer allocator.free(index_path);
            
            const index_file = std.fs.cwd().openFile(index_path, .{}) catch {
                const response = try std.fmt.allocPrint(allocator, 
                    "HTTP/1.1 404 Not Found\r\n" ++
                    "Content-Type: text/html\r\n" ++
                    "Access-Control-Allow-Origin: *\r\n" ++
                    "\r\n" ++
                    "<html><body><h1>404 Not Found</h1><p>{s}</p></body></html>",
                    .{clean_path});
                return response;
            };
            defer index_file.close();
            
            const content = try index_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
            defer allocator.free(content);
            
            const mime_type = mime.getType(index_path);
            const response = try std.fmt.allocPrint(allocator,
                "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: {s}\r\n" ++
                "Content-Length: {d}\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "Cache-Control: public, max-age=3600\r\n" ++
                "\r\n" ++
                "{s}",
                .{ mime_type, content.len, content });
            return response;
        }
        
        const response = try std.fmt.allocPrint(allocator,
            "HTTP/1.1 500 Internal Server Error\r\n" ++
            "Content-Type: text/plain\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "\r\n" ++
            "Error opening file: {any}",
            .{err});
        return response;
    };
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024); // 10MB max
    defer allocator.free(content);

    const mime_type = mime.getType(file_path);
    
    std.debug.print("Served: {s} ({s}, {d} bytes)\n", .{ clean_path, mime_type, content.len });

    const response = try std.fmt.allocPrint(allocator,
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: {s}\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Cache-Control: public, max-age=3600\r\n" ++
        "\r\n" ++
        "{s}",
        .{ mime_type, content.len, content });
    
    return response;
}