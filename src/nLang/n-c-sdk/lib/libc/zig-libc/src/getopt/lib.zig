// getopt module - Phase 1.16
const std = @import("std");

pub var optarg: ?[*:0]u8 = null;
pub var optind: c_int = 1;
pub var opterr: c_int = 1;
pub var optopt: c_int = 0;

// Internal state for tracking position within grouped short options
var optpos: usize = 0;

pub const option = extern struct {
    name: ?[*:0]const u8,
    has_arg: c_int,
    flag: ?*c_int,
    val: c_int,
};

pub const no_argument: c_int = 0;
pub const required_argument: c_int = 1;
pub const optional_argument: c_int = 2;

// Helper to get the length of a null-terminated string
fn strlen(s: [*:0]const u8) usize {
    var len: usize = 0;
    while (s[len] != 0) : (len += 1) {}
    return len;
}

// Helper to compare two null-terminated strings
fn streq(a: [*:0]const u8, b: [*:0]const u8) bool {
    var i: usize = 0;
    while (a[i] != 0 and b[i] != 0) : (i += 1) {
        if (a[i] != b[i]) return false;
    }
    return a[i] == b[i];
}

// Check if string starts with prefix, returns remaining length matched
fn strPrefixLen(s: [*]const u8, prefix: [*:0]const u8) usize {
    var i: usize = 0;
    while (prefix[i] != 0) : (i += 1) {
        if (s[i] != prefix[i]) return 0;
    }
    return i;
}

pub export fn getopt(argc: c_int, argv: [*:null]?[*:0]u8, optstring: [*:0]const u8) c_int {
    return getoptImpl(argc, argv, optstring, null, null, false);
}

pub export fn getopt_long(argc: c_int, argv: [*:null]?[*:0]u8, optstring: [*:0]const u8, longopts: ?[*:null]?*const option, longindex: ?*c_int) c_int {
    return getoptLongImpl(argc, argv, optstring, longopts, longindex, false);
}

pub export fn getopt_long_only(argc: c_int, argv: [*:null]?[*:0]u8, optstring: [*:0]const u8, longopts: ?[*:null]?*const option, longindex: ?*c_int) c_int {
    return getoptLongImpl(argc, argv, optstring, longopts, longindex, true);
}

fn getoptLongImpl(argc: c_int, argv: [*:null]?[*:0]u8, optstring: [*:0]const u8, longopts: ?[*:null]?*const option, longindex: ?*c_int, longonly: bool) c_int {
    // Reset state if optind is 0
    if (optind == 0) {
        optind = 1;
        optpos = 0;
    }

    const uargc: usize = if (argc < 0) 0 else @intCast(argc);
    const uoptind: usize = if (optind < 0) 0 else @intCast(optind);

    if (uoptind >= uargc) return -1;

    const arg_opt = argv[uoptind];
    if (arg_opt == null) return -1;
    const arg = arg_opt.?;

    // Not an option
    if (arg[0] != '-') {
        // Check for '-' mode in optstring (return non-options as argument to char code 1)
        if (optstring[0] == '-') {
            optarg = arg;
            optind += 1;
            return 1;
        }
        return -1;
    }

    // Just "-" by itself
    if (arg[1] == 0) return -1;

    // "--" terminates option processing
    if (arg[1] == '-' and arg[2] == 0) {
        optind += 1;
        return -1;
    }

    // Check for long option: starts with "--" or (longonly and starts with "-" and more chars)
    if (longopts) |lopts| {
        const is_long = (arg[1] == '-' and arg[2] != 0);
        const is_longonly_candidate = longonly and arg[1] != '-' and arg[1] != 0;

        if (is_long or is_longonly_candidate) {
            const result = tryLongOption(argc, argv, optstring, lopts, longindex, is_long);
            if (result != null) return result.?;
            // If it was a -- option and didn't match, it's an error
            if (is_long) {
                optopt = 0;
                if (opterr != 0 and optstringColon(optstring) == 0) {
                    // Error: unrecognized option
                }
                optind += 1;
                return '?';
            }
            // For longonly, fall through to try short option
        }
    }

    // Handle short option
    return getoptImpl(argc, argv, optstring, longopts, longindex, longonly);
}

fn optstringColon(optstring: [*:0]const u8) u8 {
    var idx: usize = 0;
    if (optstring[0] == '+' or optstring[0] == '-') idx = 1;
    return optstring[idx];
}

fn tryLongOption(argc: c_int, argv: [*:null]?[*:0]u8, optstring: [*:0]const u8, longopts: [*:null]?*const option, longindex: ?*c_int, is_double_dash: bool) ?c_int {
    const uoptind: usize = @intCast(optind);
    const arg = argv[uoptind].?;

    // Skip leading dashes
    var start: usize = 1;
    if (is_double_dash) start = 2;

    // Find '=' if present
    var name_end: usize = start;
    while (arg[name_end] != 0 and arg[name_end] != '=') : (name_end += 1) {}
    const name_len = name_end - start;

    if (name_len == 0) return null;

    // Search for matching long option
    var match_idx: ?usize = null;
    var match_count: usize = 0;

    var i: usize = 0;
    while (true) : (i += 1) {
        const lopt_ptr = longopts[i];
        if (lopt_ptr == null) break;
        const lopt = lopt_ptr.?;
        const lopt_name = lopt.name orelse continue;

        // Check if name matches (exact or prefix)
        var j: usize = 0;
        var matches = true;
        while (j < name_len) : (j += 1) {
            if (lopt_name[j] == 0 or lopt_name[j] != arg[start + j]) {
                matches = false;
                break;
            }
        }

        if (matches) {
            // Check for exact match
            if (lopt_name[j] == 0) {
                // Exact match
                match_idx = i;
                match_count = 1;
                break;
            } else {
                // Prefix match
                match_idx = i;
                match_count += 1;
            }
        }
    }

    // Ambiguous match
    if (match_count > 1) {
        optopt = 0;
        if (opterr != 0 and optstringColon(optstring) != ':') {
            // Could print error message here
        }
        optind += 1;
        return '?';
    }

    // No match
    if (match_count == 0 or match_idx == null) {
        return null;
    }

    const idx = match_idx.?;
    const lopt = longopts[idx].?.?;

    optind += 1;
    optpos = 0;

    // Handle argument
    if (arg[name_end] == '=') {
        // Argument provided with =
        if (lopt.has_arg == no_argument) {
            optopt = lopt.val;
            if (opterr != 0 and optstringColon(optstring) != ':') {
                // Option doesn't take argument
            }
            return '?';
        }
        // Point to argument after '='
        optarg = @ptrCast(@constCast(&arg[name_end + 1]));
    } else if (lopt.has_arg == required_argument) {
        // Need next argv as argument
        const uargc: usize = @intCast(argc);
        const next_optind: usize = @intCast(optind);
        if (next_optind >= uargc or argv[next_optind] == null) {
            optopt = lopt.val;
            if (optstringColon(optstring) == ':') return ':';
            return '?';
        }
        optarg = argv[next_optind];
        optind += 1;
    } else {
        optarg = null;
    }

    if (longindex) |lidx| {
        lidx.* = @intCast(idx);
    }

    if (lopt.flag) |flag| {
        flag.* = lopt.val;
        return 0;
    }

    return lopt.val;
}

fn getoptImpl(argc: c_int, argv: [*:null]?[*:0]u8, optstring: [*:0]const u8, longopts: ?[*:null]?*const option, longindex: ?*c_int, longonly: bool) c_int {
    _ = longopts;
    _ = longindex;
    _ = longonly;

    // Reset state if optind is 0
    if (optind == 0) {
        optind = 1;
        optpos = 0;
    }

    const uargc: usize = if (argc < 0) 0 else @intCast(argc);
    const uoptind: usize = if (optind < 0) 0 else @intCast(optind);

    if (uoptind >= uargc) return -1;

    const arg_opt = argv[uoptind];
    if (arg_opt == null) return -1;
    const arg = arg_opt.?;

    // Not an option
    if (arg[0] != '-') {
        if (optstring[0] == '-') {
            optarg = arg;
            optind += 1;
            return 1;
        }
        return -1;
    }

    // Just "-" by itself
    if (arg[1] == 0) return -1;

    // "--" terminates option processing
    if (arg[1] == '-' and arg[2] == 0) {
        optind += 1;
        return -1;
    }

    // Skip "--something" in short-only getopt
    if (arg[1] == '-') {
        optopt = 0;
        if (opterr != 0 and optstringColon(optstring) != ':') {
            // Unrecognized option
        }
        optind += 1;
        return '?';
    }

    // Position in current argument for grouped options
    if (optpos == 0) optpos = 1;

    const c = arg[optpos];
    optpos += 1;

    // Move to next arg if we've consumed all chars in this one
    if (arg[optpos] == 0) {
        optind += 1;
        optpos = 0;
    }

    // Skip leading '+' or '-' in optstring
    var os_start: usize = 0;
    if (optstring[0] == '+' or optstring[0] == '-') os_start = 1;

    // Search for option character in optstring
    var i: usize = os_start;
    while (optstring[i] != 0) : (i += 1) {
        if (optstring[i] == c and c != ':') {
            // Found the option
            if (optstring[i + 1] == ':') {
                // Option takes an argument
                if (optstring[i + 2] == ':') {
                    // Optional argument (only if attached)
                    if (optpos != 0) {
                        // Argument is rest of current argv
                        optarg = @ptrCast(@constCast(&arg[optpos]));
                        optind += 1;
                        optpos = 0;
                    } else {
                        optarg = null;
                    }
                } else {
                    // Required argument
                    if (optpos != 0) {
                        // Argument is rest of current argv
                        optarg = @ptrCast(@constCast(&arg[optpos]));
                        optind += 1;
                        optpos = 0;
                    } else {
                        // Argument is next argv
                        const next_uoptind: usize = @intCast(optind);
                        if (next_uoptind >= uargc or argv[next_uoptind] == null) {
                            optopt = c;
                            if (optstringColon(optstring) == ':') return ':';
                            if (opterr != 0) {
                                // Missing argument error
                            }
                            return '?';
                        }
                        optarg = argv[next_uoptind];
                        optind += 1;
                    }
                }
            } else {
                optarg = null;
            }
            return c;
        }
    }

    // Unknown option
    optopt = c;
    if (opterr != 0 and optstringColon(optstring) != ':') {
        // Unrecognized option error
    }
    return '?';
}
