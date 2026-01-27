// Time functions for stdlib
// Phase 1.2 - Week 29
// Implements time, difftime

const std = @import("std");

/// Get current time
/// C signature: time_t time(time_t *tloc);
pub export fn time(tloc: ?*i64) i64 {
    const timestamp = std.time.timestamp();
    if (tloc) |ptr| {
        ptr.* = timestamp;
    }
    return timestamp;
}

/// Calculate difference between two times
/// C signature: double difftime(time_t time1, time_t time0);
pub export fn difftime(time1: i64, time0: i64) f64 {
    return @as(f64, @floatFromInt(time1 - time0));
}

// Time structure (simplified)
pub const tm = extern struct {
    tm_sec: c_int,
    tm_min: c_int,
    tm_hour: c_int,
    tm_mday: c_int,
    tm_mon: c_int,
    tm_year: c_int,
    tm_wday: c_int,
    tm_yday: c_int,
    tm_isdst: c_int,
};

var global_tm: tm = undefined;

/// Convert to UTC
pub export fn gmtime(timep: *const i64) ?*tm {
    const t = timep.*;
    const days = @divTrunc(t, 86400);
    const secs = @rem(t, 86400);
    
    global_tm.tm_hour = @intCast(@divTrunc(secs, 3600));
    global_tm.tm_min = @intCast(@divTrunc(@rem(secs, 3600), 60));
    global_tm.tm_sec = @intCast(@rem(secs, 60));
    global_tm.tm_mday = @intCast(@rem(days, 31) + 1);
    global_tm.tm_mon = 0;
    global_tm.tm_year = 70 + @as(c_int, @intCast(@divTrunc(days, 365)));
    global_tm.tm_wday = @intCast(@rem(days + 4, 7));
    global_tm.tm_yday = @intCast(@rem(days, 365));
    global_tm.tm_isdst = 0;
    
    return &global_tm;
}

/// Convert to local time
pub export fn localtime(timep: *const i64) ?*tm {
    return gmtime(timep); // Simplified: same as GMT
}

/// Convert tm to time_t
pub export fn mktime(timeptr: *tm) i64 {
    var result: i64 = 0;
    result += @as(i64, timeptr.tm_year - 70) * 365 * 86400;
    result += @as(i64, timeptr.tm_yday) * 86400;
    result += @as(i64, timeptr.tm_hour) * 3600;
    result += @as(i64, timeptr.tm_min) * 60;
    result += @as(i64, timeptr.tm_sec);
    return result;
}

/// Convert to string - FIXED: Proper NUL termination
const ctime_buffer: [26:0]u8 = [_:0]u8{'T','h','u',' ','J','a','n',' ',' ','1',' ','0','0',':','0','0',':','0','0',' ','1','9','7','0','\n',0};

pub export fn ctime(timep: *const i64) [*:0]const u8 {
    _ = gmtime(timep);
    return @ptrCast(@constCast(&ctime_buffer));
}
