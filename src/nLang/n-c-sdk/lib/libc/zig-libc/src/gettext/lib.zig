// gettext module - Internationalization - Phase 1.33
const std = @import("std");

pub export fn gettext(msgid: [*:0]const u8) [*:0]u8 {
    return @constCast(msgid);
}

pub export fn dgettext(domainname: ?[*:0]const u8, msgid: [*:0]const u8) [*:0]u8 {
    _ = domainname;
    return @constCast(msgid);
}

pub export fn dcgettext(domainname: ?[*:0]const u8, msgid: [*:0]const u8, category: c_int) [*:0]u8 {
    _ = domainname; _ = category;
    return @constCast(msgid);
}

pub export fn ngettext(msgid: [*:0]const u8, msgid_plural: [*:0]const u8, n: c_ulong) [*:0]u8 {
    if (n == 1) return @constCast(msgid);
    return @constCast(msgid_plural);
}

pub export fn dngettext(domainname: ?[*:0]const u8, msgid: [*:0]const u8, msgid_plural: [*:0]const u8, n: c_ulong) [*:0]u8 {
    _ = domainname;
    if (n == 1) return @constCast(msgid);
    return @constCast(msgid_plural);
}

pub export fn dcngettext(domainname: ?[*:0]const u8, msgid: [*:0]const u8, msgid_plural: [*:0]const u8, n: c_ulong, category: c_int) [*:0]u8 {
    _ = domainname; _ = category;
    if (n == 1) return @constCast(msgid);
    return @constCast(msgid_plural);
}

pub export fn textdomain(domainname: ?[*:0]const u8) [*:0]u8 {
    _ = domainname;
    return @constCast("messages");
}

pub export fn bindtextdomain(domainname: [*:0]const u8, dirname: ?[*:0]const u8) [*:0]u8 {
    _ = domainname;
    if (dirname) |d| return @constCast(d);
    return @constCast("/usr/share/locale");
}

pub export fn bind_textdomain_codeset(domainname: [*:0]const u8, codeset: ?[*:0]const u8) ?[*:0]u8 {
    _ = domainname;
    if (codeset) |c| return @constCast(c);
    return null;
}
