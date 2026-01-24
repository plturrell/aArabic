const std = @import("std");
const json = std.json;

pub fn findBearerToken(header: []const u8) ?[]const u8 {
    if (std.mem.startsWith(u8, header, "Bearer ")) {
        return header["Bearer ".len..];
    }
    return null;
}

/// Minimal validator: decodes payload and optionally checks exp; signature is NOT verified.
pub fn validateWithKey(token: []const u8, _: []const u8) bool {
    var parts = std.mem.splitScalar(u8, token, '.');
    _ = parts.next() orelse return false; // header
    const payload_b64 = parts.next() orelse return false;
    _ = parts.next() orelse return false; // signature ignored

    var buf: [4096]u8 = undefined;
    const decoded_len = @min(buf.len, (payload_b64.len * 3 + 3) / 4);
    const out_slice = buf[0..decoded_len];
    std.base64.url_safe_no_pad.Decoder.decode(out_slice, payload_b64) catch return false;
    const payload = out_slice;

    var tree = json.parseFromSlice(json.Value, std.heap.page_allocator, payload, .{}) catch return false;
    defer tree.deinit();
    if (tree.value == .object) {
        if (tree.value.object.get("exp")) |exp_val| {
            const now = std.time.timestamp();
            const exp = switch (exp_val) {
                .integer => |i| i,
                .number_string => |s| std.fmt.parseInt(i64, s, 10) catch now,
                else => now,
            };
            if (now > exp) return false;
        }
    }
    return true;
}
