// fnmatch module - Phase 1.17 - Real Implementation with character class support
const std = @import("std");

pub const FNM_NOMATCH: c_int = 1;
pub const FNM_PATHNAME: c_int = 1 << 0;
pub const FNM_NOESCAPE: c_int = 1 << 1;
pub const FNM_PERIOD: c_int = 1 << 2;
pub const FNM_LEADING_DIR: c_int = 1 << 3;
pub const FNM_CASEFOLD: c_int = 1 << 4;

pub export fn fnmatch(pattern: [*:0]const u8, string: [*:0]const u8, flags: c_int) c_int {
    const pat = std.mem.span(pattern);
    const str = std.mem.span(string);

    if (match(pat, str, flags)) return 0;
    return FNM_NOMATCH;
}

// Match a character class pattern like [abc], [a-z], [!abc], [^a-z]
fn matchCharClass(pattern: []const u8, char: u8, noescape: bool) ?usize {
    if (pattern.len < 2 or pattern[0] != '[') return null;

    var idx: usize = 1;
    var negate = false;

    // Check for negation
    if (idx < pattern.len and (pattern[idx] == '!' or pattern[idx] == '^')) {
        negate = true;
        idx += 1;
    }

    // Empty class is invalid
    if (idx >= pattern.len or pattern[idx] == ']') return null;

    var matched = false;

    while (idx < pattern.len and pattern[idx] != ']') {
        var c = pattern[idx];

        // Handle escape sequences
        if (!noescape and c == '\\' and idx + 1 < pattern.len) {
            idx += 1;
            c = pattern[idx];
        }

        // Check for range (a-z)
        if (idx + 2 < pattern.len and pattern[idx + 1] == '-' and pattern[idx + 2] != ']') {
            var end_c = pattern[idx + 2];

            // Handle escape in range end
            if (!noescape and end_c == '\\' and idx + 3 < pattern.len) {
                end_c = pattern[idx + 3];
                idx += 1;
            }

            if (char >= c and char <= end_c) {
                matched = true;
            }
            idx += 3;
        } else {
            if (char == c) {
                matched = true;
            }
            idx += 1;
        }
    }

    // Check we found closing bracket
    if (idx >= pattern.len or pattern[idx] != ']') return null;

    // Return match result (accounting for negation)
    if (negate) {
        if (!matched) return idx + 1 else return null;
    } else {
        if (matched) return idx + 1 else return null;
    }
}

fn match(pattern: []const u8, string: []const u8, flags: c_int) bool {
    var p_idx: usize = 0;
    var s_idx: usize = 0;

    const pathname = (flags & FNM_PATHNAME) != 0;
    const noescape = (flags & FNM_NOESCAPE) != 0;
    const period = (flags & FNM_PERIOD) != 0;

    // Check FNM_PERIOD: leading period in string must be matched by period in pattern
    if (period and string.len > 0 and string[0] == '.') {
        if (pattern.len == 0 or pattern[0] != '.') return false;
    }

    while (p_idx < pattern.len) {
        if (s_idx >= string.len) {
            // String exhausted, pattern must be all '*'
            while (p_idx < pattern.len and pattern[p_idx] == '*') {
                p_idx += 1;
            }
            return p_idx >= pattern.len;
        }

        const p_char = pattern[p_idx];

        switch (p_char) {
            '[' => {
                // Character class matching
                if (pathname and string[s_idx] == '/') return false;
                if (period and string[s_idx] == '.' and s_idx == 0) return false;

                if (matchCharClass(pattern[p_idx..], string[s_idx], noescape)) |consumed| {
                    p_idx += consumed;
                    s_idx += 1;
                } else {
                    return false;
                }
            },
            '?' => {
                if (pathname and string[s_idx] == '/') return false;
                if (period and string[s_idx] == '.' and s_idx == 0) return false;
                p_idx += 1;
                s_idx += 1;
            },
            '*' => {
                // Collapse multiple stars
                while (p_idx < pattern.len and pattern[p_idx] == '*') {
                    p_idx += 1;
                }

                if (p_idx >= pattern.len) {
                    // Trailing star matches everything (unless pathname logic applies?)
                    if (pathname) {
                        // If pathname, * matches until next /
                        while (s_idx < string.len) {
                            if (string[s_idx] == '/') return false; // Star cannot match slash
                            s_idx += 1;
                        }
                        return true;
                    }
                    return true;
                }

                // Backtracking match
                while (s_idx < string.len) {
                    if (match(pattern[p_idx..], string[s_idx..], flags)) return true;
                    if (pathname and string[s_idx] == '/') return false; // * cannot cross /
                    if (period and string[s_idx] == '.' and s_idx == 0) return false;
                    s_idx += 1;
                }
                // Try matching empty string at end
                return match(pattern[p_idx..], string[s_idx..], flags);
            },
            '\\' => {
                if (!noescape and p_idx + 1 < pattern.len) {
                    p_idx += 1;
                    if (pattern[p_idx] != string[s_idx]) return false;
                    p_idx += 1;
                    s_idx += 1;
                } else {
                    if (pattern[p_idx] != string[s_idx]) return false;
                    p_idx += 1;
                    s_idx += 1;
                }
            },
            else => {
                if (p_char != string[s_idx]) return false;
                p_idx += 1;
                s_idx += 1;
            },
        }
    }

    return s_idx == string.len;
}
