//! ============================================================================
//! MIME Type Detection
//! Maps file extensions to Content-Type headers
//! ============================================================================
//!
//! [CODE:file=mime.zig]
//! [CODE:module=http]
//! [CODE:language=zig]
//!
//! [RELATION:called_by=CODE:static.zig]
//!
//! Note: Infrastructure code - no ODPS business rules implemented here.

const std = @import("std");

pub fn getType(path: []const u8) []const u8 {
    if (std.mem.endsWith(u8, path, ".html")) return "text/html; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".htm")) return "text/html; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".css")) return "text/css; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".js")) return "application/javascript; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".json")) return "application/json; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".xml")) return "application/xml; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".txt")) return "text/plain; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".properties")) return "text/plain; charset=utf-8";
    
    // Images
    if (std.mem.endsWith(u8, path, ".png")) return "image/png";
    if (std.mem.endsWith(u8, path, ".jpg")) return "image/jpeg";
    if (std.mem.endsWith(u8, path, ".jpeg")) return "image/jpeg";
    if (std.mem.endsWith(u8, path, ".gif")) return "image/gif";
    if (std.mem.endsWith(u8, path, ".svg")) return "image/svg+xml";
    if (std.mem.endsWith(u8, path, ".ico")) return "image/x-icon";
    
    // Fonts
    if (std.mem.endsWith(u8, path, ".woff")) return "font/woff";
    if (std.mem.endsWith(u8, path, ".woff2")) return "font/woff2";
    if (std.mem.endsWith(u8, path, ".ttf")) return "font/ttf";
    if (std.mem.endsWith(u8, path, ".eot")) return "application/vnd.ms-fontobject";
    
    // Documents  
    if (std.mem.endsWith(u8, path, ".pdf")) return "application/pdf";
    if (std.mem.endsWith(u8, path, ".zip")) return "application/zip";
    
    // UI5 specific
    if (std.mem.endsWith(u8, path, ".yaml")) return "text/yaml; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".yml")) return "text/yaml; charset=utf-8";
    
    return "application/octet-stream";
}