// net/if module - Phase 1.21
const std = @import("std");

pub const IF_NAMESIZE: usize = 16;

pub const if_nameindex = extern struct {
    if_index: c_uint,
    if_name: [*:0]u8,
};

pub export fn if_nametoindex(ifname: [*:0]const u8) c_uint {
    _ = ifname;
    return 1;
}

pub export fn if_indextoname(ifindex: c_uint, ifname: [*:0]u8) ?[*:0]u8 {
    _ = ifindex;
    ifname[0] = 'l';
    ifname[1] = 'o';
    ifname[2] = 0;
    return ifname;
}

pub export fn if_nameindex_get() ?[*]if_nameindex {
    return null;
}

pub export fn if_freenameindex(ptr: ?[*]if_nameindex) void {
    _ = ptr;
}
