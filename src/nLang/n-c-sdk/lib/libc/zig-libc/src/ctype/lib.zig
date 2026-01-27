// Character classification module
// Phase 1.1: Foundation - Basic character functions

const std = @import("std");

// ============================================================================
// Phase 1.1 Functions (Month 3)
// ============================================================================

/// Check if character is alphabetic
/// POSIX: int isalpha(int c);
pub fn isalpha(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return (uc >= 'A' and uc <= 'Z') or (uc >= 'a' and uc <= 'z');
}

/// Check if character is a digit
/// POSIX: int isdigit(int c);
pub fn isdigit(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return uc >= '0' and uc <= '9';
}

/// Check if character is alphanumeric
/// POSIX: int isalnum(int c);
pub fn isalnum(c: i32) bool {
    return isalpha(c) or isdigit(c);
}

/// Check if character is whitespace
/// POSIX: int isspace(int c);
pub fn isspace(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return uc == ' ' or uc == '\t' or uc == '\n' or 
           uc == '\r' or uc == '\x0B' or uc == '\x0C';
}

/// Convert to uppercase
/// POSIX: int toupper(int c);
pub fn toupper(c: i32) i32 {
    const uc = @as(u8, @intCast(c & 0xFF));
    if (uc >= 'a' and uc <= 'z') {
        return c - 32;
    }
    return c;
}

/// Convert to lowercase
/// POSIX: int tolower(int c);
pub fn tolower(c: i32) i32 {
    const uc = @as(u8, @intCast(c & 0xFF));
    if (uc >= 'A' and uc <= 'Z') {
        return c + 32;
    }
    return c;
}

/// Check if character is uppercase
/// POSIX: int isupper(int c);
pub fn isupper(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return uc >= 'A' and uc <= 'Z';
}

/// Check if character is lowercase
/// POSIX: int islower(int c);
pub fn islower(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return uc >= 'a' and uc <= 'z';
}

/// Check if character is hexadecimal digit
/// POSIX: int isxdigit(int c);
pub fn isxdigit(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return (uc >= '0' and uc <= '9') or
           (uc >= 'A' and uc <= 'F') or
           (uc >= 'a' and uc <= 'f');
}

/// Check if character is punctuation
/// POSIX: int ispunct(int c);
pub fn ispunct(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return (uc >= '!' and uc <= '/') or
           (uc >= ':' and uc <= '@') or
           (uc >= '[' and uc <= '`') or
           (uc >= '{' and uc <= '~');
}

/// Check if character is printable (including space)
/// POSIX: int isprint(int c);
pub fn isprint(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return uc >= ' ' and uc <= '~';
}

/// Check if character is graphical (printable except space)
/// POSIX: int isgraph(int c);
pub fn isgraph(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return uc >= '!' and uc <= '~';
}

/// Check if character is control character
/// POSIX: int iscntrl(int c);
pub fn iscntrl(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return (uc < ' ') or (uc == 0x7F);
}

/// Check if character is blank (space or tab)
/// POSIX: int isblank(int c);
pub fn isblank(c: i32) bool {
    const uc = @as(u8, @intCast(c & 0xFF));
    return uc == ' ' or uc == '\t';
}

// ============================================================================
// C-Compatible Exports (TODO: Add in Phase 1.2 with proper FFI)
// ============================================================================

// Note: C exports commented out for Phase 1.1
// Will be re-added with proper ABI compatibility testing in Phase 1.2

// ============================================================================
// Unit Tests
// ============================================================================

test "isalpha - letters" {
    try std.testing.expect(isalpha('A'));
    try std.testing.expect(isalpha('z'));
    try std.testing.expect(!isalpha('0'));
    try std.testing.expect(!isalpha(' '));
}

test "isdigit - numbers" {
    try std.testing.expect(isdigit('0'));
    try std.testing.expect(isdigit('9'));
    try std.testing.expect(!isdigit('a'));
}

test "isalnum - alphanumeric" {
    try std.testing.expect(isalnum('A'));
    try std.testing.expect(isalnum('0'));
    try std.testing.expect(!isalnum(' '));
}

test "isspace - whitespace" {
    try std.testing.expect(isspace(' '));
    try std.testing.expect(isspace('\t'));
    try std.testing.expect(isspace('\n'));
    try std.testing.expect(!isspace('a'));
}

test "toupper - conversion" {
    try std.testing.expectEqual(@as(i32, 'A'), toupper('a'));
    try std.testing.expectEqual(@as(i32, 'Z'), toupper('z'));
    try std.testing.expectEqual(@as(i32, 'A'), toupper('A'));
}

test "tolower - conversion" {
    try std.testing.expectEqual(@as(i32, 'a'), tolower('A'));
    try std.testing.expectEqual(@as(i32, 'z'), tolower('Z'));
    try std.testing.expectEqual(@as(i32, 'a'), tolower('a'));
}

test "isupper - uppercase check" {
    try std.testing.expect(isupper('A'));
    try std.testing.expect(isupper('Z'));
    try std.testing.expect(!isupper('a'));
    try std.testing.expect(!isupper('0'));
}

test "islower - lowercase check" {
    try std.testing.expect(islower('a'));
    try std.testing.expect(islower('z'));
    try std.testing.expect(!islower('A'));
    try std.testing.expect(!islower('0'));
}

test "isxdigit - hex digit check" {
    try std.testing.expect(isxdigit('0'));
    try std.testing.expect(isxdigit('9'));
    try std.testing.expect(isxdigit('A'));
    try std.testing.expect(isxdigit('F'));
    try std.testing.expect(isxdigit('a'));
    try std.testing.expect(isxdigit('f'));
    try std.testing.expect(!isxdigit('G'));
    try std.testing.expect(!isxdigit('g'));
}

test "ispunct - punctuation check" {
    try std.testing.expect(ispunct('!'));
    try std.testing.expect(ispunct('.'));
    try std.testing.expect(ispunct(','));
    try std.testing.expect(ispunct('?'));
    try std.testing.expect(!ispunct('a'));
    try std.testing.expect(!ispunct('0'));
}

test "isprint - printable check" {
    try std.testing.expect(isprint(' '));
    try std.testing.expect(isprint('a'));
    try std.testing.expect(isprint('~'));
    try std.testing.expect(!isprint('\n'));
    try std.testing.expect(!isprint(0x7F));
}

test "isgraph - graphical check" {
    try std.testing.expect(isgraph('a'));
    try std.testing.expect(isgraph('!'));
    try std.testing.expect(isgraph('~'));
    try std.testing.expect(!isgraph(' '));
    try std.testing.expect(!isgraph('\n'));
}

test "iscntrl - control character check" {
    try std.testing.expect(iscntrl(0));
    try std.testing.expect(iscntrl('\n'));
    try std.testing.expect(iscntrl('\t'));
    try std.testing.expect(iscntrl(0x7F));
    try std.testing.expect(!iscntrl('a'));
    try std.testing.expect(!iscntrl(' '));
}

test "isblank - blank check" {
    try std.testing.expect(isblank(' '));
    try std.testing.expect(isblank('\t'));
    try std.testing.expect(!isblank('\n'));
    try std.testing.expect(!isblank('a'));
}

// ============================================================================
// Character Classification Complete! ✅
// ============================================================================

// Week 11-12: Implemented ✅
// - [x] isalpha
// - [x] isdigit
// - [x] isalnum
// - [x] isspace
// - [x] toupper
// - [x] tolower

// Week 15-16: Implemented ✅
// - [x] isupper
// - [x] islower
// - [x] isxdigit
// - [x] ispunct
// - [x] isprint
// - [x] isgraph
// - [x] iscntrl
// - [x] isblank

// Phase 1.1: 14/20 character functions complete (70%)
// Remaining 6 for Phase 1.2+:
// - [ ] isascii
// - [ ] toascii
// - [ ] _tolower (macro version)
// - [ ] _toupper (macro version)
// - [ ] isalpha_l (locale-aware)
// - [ ] isdigit_l (locale-aware)
