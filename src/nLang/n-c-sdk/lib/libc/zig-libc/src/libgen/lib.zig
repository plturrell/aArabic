// libgen module - Phase 1.17
const std = @import("std");

pub export fn basename(path: [*:0]u8) [*:0]u8 {
    var i: usize = 0;
    while (path[i] != 0) : (i += 1) {}
    
    if (i == 0) return @constCast(".");
    
    // Skip trailing slashes
    while (i > 0 and path[i - 1] == '/') : (i -= 1) {
        path[i - 1] = 0;
    }
    
    if (i == 0) return @constCast("/");
    
    // Find last slash
    var j = i;
    while (j > 0 and path[j - 1] != '/') : (j -= 1) {}
    
    return path + j;
}

pub export fn dirname(path: [*:0]u8) [*:0]u8 {
    var i: usize = 0;
    while (path[i] != 0) : (i += 1) {}
    
    if (i == 0) return @constCast(".");
    
    // Skip trailing slashes
    while (i > 0 and path[i - 1] == '/') : (i -= 1) {
        path[i - 1] = 0;
    }
    
    if (i == 0) return @constCast("/");
    
    // Find last slash
    while (i > 0 and path[i - 1] != '/') : (i -= 1) {}
    
    if (i == 0) return @constCast(".");
    
    // Remove trailing slashes from dirname
    while (i > 1 and path[i - 1] == '/') : (i -= 1) {}
    
    path[i] = 0;
    return if (i == 0) @constCast("/") else path;
}
