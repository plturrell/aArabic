// regex module - Phase 1.10 Priority 8 - POSIX Regular Expressions (Pure Zig)
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Regex types
pub const regex_t = extern struct {
    re_nsub: usize,
    __opaque: ?*anyopaque, // Pointer to our internal CompiledRegex
    __allocated: usize,
    __used: usize,
    __syntax: c_ulong,
    __fastmap: ?*u8,
    __translate: ?*u8,
    __can_be_null: c_int,
    __regs_allocated: c_int,
    __fastmap_accurate: c_int,
    __no_sub: c_int,
    __not_bol: c_int,
    __not_eol: c_int,
    __newline_anchor: c_int,
};

pub const regmatch_t = extern struct {
    rm_so: isize,
    rm_eo: isize,
};

pub const regoff_t = isize;

// Flags for regcomp
pub const REG_EXTENDED: c_int = 1;
pub const REG_ICASE: c_int = 2;
pub const REG_NOSUB: c_int = 4;
pub const REG_NEWLINE: c_int = 8;

// Flags for regexec
pub const REG_NOTBOL: c_int = 1;
pub const REG_NOTEOL: c_int = 2;

// Error codes
pub const REG_NOMATCH: c_int = 1;
pub const REG_BADPAT: c_int = 2;
pub const REG_ECOLLATE: c_int = 3;
pub const REG_ECTYPE: c_int = 4;
pub const REG_EESCAPE: c_int = 5;
pub const REG_ESUBREG: c_int = 6;
pub const REG_EBRACK: c_int = 7;
pub const REG_EPAREN: c_int = 8;
pub const REG_EBRACE: c_int = 9;
pub const REG_BADBR: c_int = 10;
pub const REG_ERANGE: c_int = 11;
pub const REG_ESPACE: c_int = 12;
pub const REG_BADRPT: c_int = 13;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Internal Regex Engine
const OpCode = enum {
    Char, // Match specific char
    Any, // Match any char (.)
    Split, // Split execution (alternation)
    Jmp, // Jump
    Match, // Success
    Save, // Save position for capture group (val = slot number)
    Class, // Character class [a-z]
    BeginLine, // ^ anchor - match at start
    EndLine, // $ anchor - match at end
};

const CharRange = struct { min: u8, max: u8 };

const ClassData = struct {
    ranges: []CharRange,
    negated: bool,
};

const Instruction = struct {
    op: OpCode,
    val: u8 = 0,
    target: usize = 0, // Jump target
    class_data: ?ClassData = null,
};

const CompiledRegex = struct {
    prog: std.ArrayList(Instruction),
    nsub: usize,
    anchored_start: bool, // True if pattern starts with ^
};

// Parse a character class [abc], [a-z], [^abc]
fn parseCharClass(pattern: []const u8, start: usize) !struct { class_data: ClassData, consumed: usize } {
    if (start >= pattern.len or pattern[start] != '[') return error.InvalidPattern;

    var idx: usize = start + 1;
    var negated = false;

    // Check for negation
    if (idx < pattern.len and (pattern[idx] == '^' or pattern[idx] == '!')) {
        negated = true;
        idx += 1;
    }

    // Empty class is invalid
    if (idx >= pattern.len or pattern[idx] == ']') return error.InvalidPattern;

    var ranges = std.ArrayList(CharRange).init(allocator);

    while (idx < pattern.len and pattern[idx] != ']') {
        var c = pattern[idx];

        // Handle escape sequences
        if (c == '\\' and idx + 1 < pattern.len) {
            idx += 1;
            c = pattern[idx];
        }

        // Check for range (a-z)
        if (idx + 2 < pattern.len and pattern[idx + 1] == '-' and pattern[idx + 2] != ']') {
            var end_c = pattern[idx + 2];

            // Handle escape in range end
            if (end_c == '\\' and idx + 3 < pattern.len) {
                end_c = pattern[idx + 3];
                idx += 1;
            }

            try ranges.append(.{ .min = c, .max = end_c });
            idx += 3;
        } else {
            try ranges.append(.{ .min = c, .max = c });
            idx += 1;
        }
    }

    // Check we found closing bracket
    if (idx >= pattern.len or pattern[idx] != ']') {
        ranges.deinit();
        return error.InvalidPattern;
    }

    return .{
        .class_data = .{
            .ranges = try ranges.toOwnedSlice(),
            .negated = negated,
        },
        .consumed = idx + 1 - start,
    };
}

// Compiler
fn compile(pattern: []const u8, flags: c_int) !*CompiledRegex {
    _ = flags;
    const compiled = try allocator.create(CompiledRegex);
    compiled.prog = std.ArrayList(Instruction).init(allocator);
    compiled.nsub = 0;
    compiled.anchored_start = false;

    var group_count: u8 = 0; // Current capture group number
    var i: usize = 0;

    while (i < pattern.len) {
        const c = pattern[i];
        switch (c) {
            '^' => {
                try compiled.prog.append(.{ .op = .BeginLine });
                if (i == 0) compiled.anchored_start = true;
                i += 1;
            },
            '$' => {
                try compiled.prog.append(.{ .op = .EndLine });
                i += 1;
            },
            '(' => {
                // Start capture group
                group_count += 1;
                compiled.nsub = group_count;
                // Save slot = group_count * 2 - 2 for start (0, 2, 4, ...)
                try compiled.prog.append(.{ .op = .Save, .val = (group_count - 1) * 2 });
                i += 1;
            },
            ')' => {
                // End capture group - use the matching group
                // Save slot = group_count * 2 - 1 for end (1, 3, 5, ...)
                if (group_count > 0) {
                    try compiled.prog.append(.{ .op = .Save, .val = (group_count - 1) * 2 + 1 });
                }
                i += 1;
            },
            '.' => {
                try compiled.prog.append(.{ .op = .Any });
                i += 1;
            },
            '[' => {
                // Character class
                const result = parseCharClass(pattern, i) catch return error.OutOfMemory;
                try compiled.prog.append(.{ .op = .Class, .class_data = result.class_data });
                i += result.consumed;
            },
            '*' => {
                // Zero or more (greedy)
                if (compiled.prog.items.len > 0) {
                    const last_idx = compiled.prog.items.len - 1;
                    // Insert Split before last instruction
                    try compiled.prog.insert(last_idx, .{ .op = .Split, .target = last_idx + 2 });
                    // Add Jump back after last instruction
                    try compiled.prog.append(.{ .op = .Jmp, .target = last_idx });
                    // Update split target to skip the loop
                    compiled.prog.items[last_idx].target = compiled.prog.items.len;
                }
                i += 1;
            },
            '+' => {
                // One or more (greedy): match once, then zero or more
                if (compiled.prog.items.len > 0) {
                    const last_idx = compiled.prog.items.len - 1;
                    // Add Split: either jump back to last_idx or continue
                    try compiled.prog.append(.{ .op = .Split, .target = compiled.prog.items.len + 1 });
                    try compiled.prog.append(.{ .op = .Jmp, .target = last_idx });
                }
                i += 1;
            },
            '?' => {
                // Zero or one (greedy)
                if (compiled.prog.items.len > 0) {
                    const last_idx = compiled.prog.items.len - 1;
                    // Insert Split before last instruction: skip or match
                    try compiled.prog.insert(last_idx, .{ .op = .Split, .target = last_idx + 2 });
                    // Update split target to skip the optional element
                    compiled.prog.items[last_idx].target = compiled.prog.items.len;
                }
                i += 1;
            },
            '\\' => {
                i += 1;
                if (i < pattern.len) {
                    try compiled.prog.append(.{ .op = .Char, .val = pattern[i] });
                    i += 1;
                }
            },
            else => {
                try compiled.prog.append(.{ .op = .Char, .val = c });
                i += 1;
            },
        }
    }

    try compiled.prog.append(.{ .op = .Match });
    return compiled;
}

// Maximum number of capture slots (pairs of start/end positions)
const MAX_CAPTURE_SLOTS: usize = 20; // 10 capture groups * 2

// Match state for tracking captures
const MatchState = struct {
    captures: [MAX_CAPTURE_SLOTS]isize,

    fn init() MatchState {
        return .{ .captures = [_]isize{-1} ** MAX_CAPTURE_SLOTS };
    }
};

// Check if character matches a character class
fn matchClass(class_data: ClassData, char: u8) bool {
    var matched = false;
    for (class_data.ranges) |range| {
        if (char >= range.min and char <= range.max) {
            matched = true;
            break;
        }
    }
    if (class_data.negated) return !matched;
    return matched;
}

// VM-based Executor with capture support
fn execute(compiled: *CompiledRegex, str: []const u8, start_pos: usize, state: *MatchState) ?usize {
    // Returns the end position of match, or null if no match
    return matchRecursive(compiled, 0, str, start_pos, state);
}

fn matchRecursive(compiled: *CompiledRegex, pc: usize, str: []const u8, sp: usize, state: *MatchState) ?usize {
    const prog = compiled.prog.items;
    if (pc >= prog.len) return null;

    const inst = prog[pc];
    switch (inst.op) {
        .Char => {
            if (sp < str.len and str[sp] == inst.val) {
                return matchRecursive(compiled, pc + 1, str, sp + 1, state);
            }
            return null;
        },
        .Any => {
            if (sp < str.len) {
                return matchRecursive(compiled, pc + 1, str, sp + 1, state);
            }
            return null;
        },
        .Class => {
            if (inst.class_data) |class_data| {
                if (sp < str.len and matchClass(class_data, str[sp])) {
                    return matchRecursive(compiled, pc + 1, str, sp + 1, state);
                }
            }
            return null;
        },
        .BeginLine => {
            // Match only at start of string
            if (sp == 0) {
                return matchRecursive(compiled, pc + 1, str, sp, state);
            }
            return null;
        },
        .EndLine => {
            // Match only at end of string
            if (sp == str.len) {
                return matchRecursive(compiled, pc + 1, str, sp, state);
            }
            return null;
        },
        .Save => {
            // Save current position to capture slot
            const slot = inst.val;
            if (slot < MAX_CAPTURE_SLOTS) {
                const old_val = state.captures[slot];
                state.captures[slot] = @intCast(sp);
                if (matchRecursive(compiled, pc + 1, str, sp, state)) |end_pos| {
                    return end_pos;
                }
                // Backtrack: restore old value
                state.captures[slot] = old_val;
            }
            return null;
        },
        .Split => {
            // Try first branch (greedy)
            if (matchRecursive(compiled, pc + 1, str, sp, state)) |end_pos| {
                return end_pos;
            }
            return matchRecursive(compiled, inst.target, str, sp, state);
        },
        .Jmp => {
            return matchRecursive(compiled, inst.target, str, sp, state);
        },
        .Match => {
            return sp; // Return the end position
        },
    }
}

// --- Public API ---

pub export fn regcomp(preg: *regex_t, pattern: [*:0]const u8, cflags: c_int) c_int {
    const pat = std.mem.span(pattern);
    const compiled = compile(pat, cflags) catch return REG_ESPACE;
    
    preg.__opaque = @ptrCast(compiled);
    preg.re_nsub = compiled.nsub;
    
    return 0;
}

pub export fn regexec(preg: *const regex_t, string: [*:0]const u8, nmatch: usize, pmatch: [*]regmatch_t, eflags: c_int) c_int {
    _ = eflags;
    const compiled: *CompiledRegex = @ptrCast(@alignCast(preg.__opaque));
    const str = std.mem.span(string);

    // Initialize pmatch to -1
    if (nmatch > 0 and pmatch != @as([*]regmatch_t, undefined)) {
        for (0..nmatch) |j| {
            pmatch[j].rm_so = -1;
            pmatch[j].rm_eo = -1;
        }
    }

    // If pattern is anchored at start, only try from position 0
    const max_start: usize = if (compiled.anchored_start) 1 else str.len + 1;

    var i: usize = 0;
    while (i < max_start) : (i += 1) {
        var state = MatchState.init();
        if (execute(compiled, str, i, &state)) |end_pos| {
            // Found match!
            if (nmatch > 0) {
                // pmatch[0] = entire match
                pmatch[0].rm_so = @intCast(i);
                pmatch[0].rm_eo = @intCast(end_pos);

                // pmatch[1..n] = capture groups
                var group: usize = 1;
                while (group < nmatch and (group - 1) * 2 + 1 < MAX_CAPTURE_SLOTS) : (group += 1) {
                    const start_slot = (group - 1) * 2;
                    const end_slot = start_slot + 1;
                    pmatch[group].rm_so = state.captures[start_slot];
                    pmatch[group].rm_eo = state.captures[end_slot];
                }
            }
            return 0;
        }
    }

    return REG_NOMATCH;
}

pub export fn regerror(errcode: c_int, preg: *const regex_t, errbuf: [*]u8, errbuf_size: usize) usize {
    _ = preg;
    const msg = switch (errcode) {
        REG_NOMATCH => "No match",
        REG_BADPAT => "Invalid regular expression",
        else => "Unknown error",
    };
    const len = std.mem.len(msg);
    const copy_len = @min(len, errbuf_size - 1);
    @memcpy(errbuf[0..copy_len], msg[0..copy_len]);
    errbuf[copy_len] = 0;
    return len + 1;
}

pub export fn regfree(preg: *regex_t) void {
    if (preg.__opaque) |ptr| {
        const compiled: *CompiledRegex = @ptrCast(@alignCast(ptr));
        // Free character class data
        for (compiled.prog.items) |inst| {
            if (inst.class_data) |class_data| {
                allocator.free(class_data.ranges);
            }
        }
        compiled.prog.deinit();
        allocator.destroy(compiled);
        preg.__opaque = null;
    }
}