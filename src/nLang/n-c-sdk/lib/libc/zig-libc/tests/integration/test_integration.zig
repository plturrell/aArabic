// Integration tests for zig-libc Phase 1.1
// Tests multiple functions working together in real-world scenarios

const std = @import("std");
const testing = std.testing;
const zig_libc = @import("zig-libc");

// Test string operations working together
test "string: copy, concatenate, and compare workflow" {
    var buffer: [100]u8 = undefined;
    @memset(&buffer, 0);
    
    const src1 = "Hello";
    const src2 = ", World!";
    
    // Copy first string
    _ = zig_libc.string.strcpy(@ptrCast(&buffer), src1);
    try testing.expectEqual(@as(usize, 5), zig_libc.string.strlen(@ptrCast(&buffer)));
    
    // Concatenate second string
    _ = zig_libc.string.strcat(@ptrCast(&buffer), src2);
    try testing.expectEqual(@as(usize, 13), zig_libc.string.strlen(@ptrCast(&buffer)));
    
    // Compare with expected result
    const expected = "Hello, World!";
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strcmp(@ptrCast(&buffer), expected));
}

// Test string search in processed text
test "string: process text and search for patterns" {
    var buffer: [200]u8 = undefined;
    @memset(&buffer, 0);
    
    const text1 = "The quick brown fox ";
    const text2 = "jumps over the lazy dog";
    
    // Build complete sentence
    _ = zig_libc.string.strcpy(@ptrCast(&buffer), text1);
    _ = zig_libc.string.strcat(@ptrCast(&buffer), text2);
    
    // Search for patterns
    const fox_ptr = zig_libc.string.strstr(@ptrCast(&buffer), "fox");
    try testing.expect(fox_ptr != null);
    
    const dog_ptr = zig_libc.string.strstr(@ptrCast(&buffer), "dog");
    try testing.expect(dog_ptr != null);
    
    // Search for character
    const space_ptr = zig_libc.string.strchr(@ptrCast(&buffer), ' ');
    try testing.expect(space_ptr != null);
}

// Test memory operations with strings
test "memory: copy strings and compare buffers" {
    var buf1: [50]u8 = undefined;
    var buf2: [50]u8 = undefined;
    
    const test_str = "Testing memory operations";
    const len = zig_libc.string.strlen(test_str);
    
    // Copy string to first buffer using memcpy
    _ = zig_libc.memory.memcpy(&buf1, test_str.ptr, len + 1);
    
    // Copy from first to second buffer
    _ = zig_libc.memory.memcpy(&buf2, &buf1, len + 1);
    
    // Compare buffers
    try testing.expectEqual(@as(c_int, 0), zig_libc.memory.memcmp(&buf1, &buf2, len));
    
    // Verify string equality
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strcmp(@ptrCast(&buf1), @ptrCast(&buf2)));
}

// Test character classification with string processing
test "ctype: validate and transform string content" {
    var input: [50]u8 = undefined;
    @memset(&input, 0);
    _ = zig_libc.string.strcpy(@ptrCast(&input), "Hello123World");
    
    var alpha_count: usize = 0;
    var digit_count: usize = 0;
    var i: usize = 0;
    
    while (input[i] != 0) : (i += 1) {
        if (zig_libc.ctype.isalpha(input[i])) {
            alpha_count += 1;
        }
        if (zig_libc.ctype.isdigit(input[i])) {
            digit_count += 1;
        }
    }
    
    try testing.expectEqual(@as(usize, 10), alpha_count);
    try testing.expectEqual(@as(usize, 3), digit_count);
}

// Test case transformation workflow
test "ctype: convert string case" {
    var buffer: [50]u8 = undefined;
    @memset(&buffer, 0);
    _ = zig_libc.string.strcpy(@ptrCast(&buffer), "Hello World");
    
    // Convert to uppercase
    var i: usize = 0;
    while (buffer[i] != 0) : (i += 1) {
        buffer[i] = @intCast(zig_libc.ctype.toupper(buffer[i]));
    }
    
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strcmp(@ptrCast(&buffer), "HELLO WORLD"));
    
    // Convert to lowercase
    i = 0;
    while (buffer[i] != 0) : (i += 1) {
        buffer[i] = @intCast(zig_libc.ctype.tolower(buffer[i]));
    }
    
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strcmp(@ptrCast(&buffer), "hello world"));
}

// Test string tokenization workflow
test "string: tokenize and process tokens" {
    var buffer: [100]u8 = undefined;
    @memset(&buffer, 0);
    _ = zig_libc.string.strcpy(@ptrCast(&buffer), "apple,banana,cherry");
    
    var token_count: usize = 0;
    var saveptr: ?[*:0]u8 = null;
    
    var token = zig_libc.string.strtok_r(@ptrCast(&buffer), ",", &saveptr);
    while (token != null) {
        token_count += 1;
        const len = zig_libc.string.strlen(token.?);
        try testing.expect(len > 0);
        token = zig_libc.string.strtok_r(null, ",", &saveptr);
    }
    
    try testing.expectEqual(@as(usize, 3), token_count);
}

// Test string span operations
test "string: span and complement operations" {
    const text = "abc123def456";
    
    // Count initial alphabetic characters
    const alpha_span = zig_libc.string.strspn(text, "abcdefghijklmnopqrstuvwxyz");
    try testing.expectEqual(@as(usize, 3), alpha_span);
    
    // Count initial non-digit characters  
    const non_digit_span = zig_libc.string.strcspn(text, "0123456789");
    try testing.expectEqual(@as(usize, 3), non_digit_span);
}

// Test string search with character sets
test "string: find characters from set" {
    const text = "hello world";
    
    // Find first vowel
    const vowel = zig_libc.string.strpbrk(text, "aeiou");
    try testing.expect(vowel != null);
    try testing.expectEqual(@as(u8, 'e'), vowel.?[0]);
    
    // Find first punctuation
    const punct = zig_libc.string.strpbrk(text, ".,!?");
    try testing.expect(punct == null); // No punctuation in text
}

// Test memory operations with overlapping regions
test "memory: handle overlapping regions with memmove" {
    var buffer: [20]u8 = undefined;
    const src = "ABCDEFGHIJ";
    @memcpy(buffer[0..10], src[0..10]);
    
    // Move overlapping region forward
    _ = zig_libc.memory.memmove(@ptrCast(&buffer[5]), @ptrCast(&buffer[0]), 5);
    
    // Verify data integrity
    try testing.expectEqual(@as(u8, 'A'), buffer[5]);
    try testing.expectEqual(@as(u8, 'B'), buffer[6]);
}

// Test memory search operations
test "memory: search for byte in buffer" {
    var buffer: [100]u8 = undefined;
    @memset(&buffer, 0xAA);
    buffer[50] = 0xFF;
    buffer[75] = 0xFF;
    
    // Find first occurrence
    const first = zig_libc.memory.memchr(&buffer, 0xFF, 100);
    try testing.expect(first != null);
    
    // Find from reverse
    const last = zig_libc.memory.memrchr(&buffer, 0xFF, 100);
    try testing.expect(last != null);
}

// Test combined string and memory operations
test "integration: full text processing workflow" {
    var workspace: [500]u8 = undefined;
    @memset(&workspace, 0);
    
    // Phase 1: Build text
    const part1 = "Processing text with ";
    const part2 = "string and memory functions";
    
    _ = zig_libc.string.strcpy(@ptrCast(&workspace), part1);
    _ = zig_libc.string.strcat(@ptrCast(&workspace), part2);
    
    // Phase 2: Validate content
    const total_len = zig_libc.string.strlen(@ptrCast(&workspace));
    try testing.expect(total_len > 0);
    
    // Phase 3: Search and verify
    const found = zig_libc.string.strstr(@ptrCast(&workspace), "memory");
    try testing.expect(found != null);
    
    // Phase 4: Character analysis
    var letter_count: usize = 0;
    var i: usize = 0;
    while (workspace[i] != 0) : (i += 1) {
        if (zig_libc.ctype.isalpha(workspace[i])) {
            letter_count += 1;
        }
    }
    try testing.expect(letter_count > 0);
}

// Test error handling and edge cases
test "integration: handle edge cases gracefully" {
    // Empty strings
    var empty: [10]u8 = undefined;
    @memset(&empty, 0);
    try testing.expectEqual(@as(usize, 0), zig_libc.string.strlen(@ptrCast(&empty)));
    
    // Single character
    var single: [2]u8 = .{ 'A', 0 };
    try testing.expectEqual(@as(usize, 1), zig_libc.string.strlen(@ptrCast(&single)));
    
    // Self comparison
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strcmp(@ptrCast(&single), @ptrCast(&single)));
}

// Test case-insensitive operations
test "string: case-insensitive operations" {
    const str1 = "Hello World";
    const str2 = "HELLO WORLD";
    const str3 = "hello world";
    
    // Case-insensitive comparison
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strcasecmp(str1, str2));
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strcasecmp(str1, str3));
    
    // Partial case-insensitive comparison
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strncasecmp(str1, str2, 5));
    
    // Case-insensitive search
    const result = zig_libc.string.strcasestr(str1, "WORLD");
    try testing.expect(result != null);
}

// Test string length with limit
test "string: length with maximum limit" {
    const long_string = "This is a very long string for testing";
    
    // Get limited length
    const len_full = zig_libc.string.strnlen(long_string, 100);
    const len_limited = zig_libc.string.strnlen(long_string, 10);
    
    try testing.expect(len_full > len_limited);
    try testing.expectEqual(@as(usize, 10), len_limited);
}

// Real-world scenario: Parse CSV-like data
test "integration: parse structured data" {
    var data: [100]u8 = undefined;
    @memset(&data, 0);
    _ = zig_libc.string.strcpy(@ptrCast(&data), "John,30,Engineer");
    
    var saveptr: ?[*:0]u8 = null;
    
    // Parse name
    const name = zig_libc.string.strtok_r(@ptrCast(&data), ",", &saveptr);
    try testing.expect(name != null);
    try testing.expectEqual(@as(c_int, 0), zig_libc.string.strcmp(name.?, "John"));
    
    // Parse age
    const age = zig_libc.string.strtok_r(null, ",", &saveptr);
    try testing.expect(age != null);
    try testing.expect(zig_libc.ctype.isdigit(age.?[0]));
    
    // Parse occupation
    const occupation = zig_libc.string.strtok_r(null, ",", &saveptr);
    try testing.expect(occupation != null);
    try testing.expect(zig_libc.ctype.isalpha(occupation.?[0]));
}

// Test memory pattern finding
test "memory: find pattern in buffer" {
    var haystack: [100]u8 = undefined;
    @memset(&haystack, 0xAA);
    
    const needle = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF };
    @memcpy(haystack[50..54], &needle);
    
    // Find the pattern
    const found = zig_libc.memory.memmem(&haystack, 100, &needle, 4);
    try testing.expect(found != null);
}

// Stress test: Process large text
test "integration: handle larger data volumes" {
    const allocator = testing.allocator;
    var large_buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(large_buffer);
    
    @memset(large_buffer, 0);
    
    // Fill with repeated pattern
    var i: usize = 0;
    while (i < 900) : (i += 10) {
        @memcpy(large_buffer[i..][0..10], "ABCDEFGHIJ");
    }
    large_buffer[900] = 0; // Null terminate
    
    // Verify operations work on large data
    const len = zig_libc.string.strlen(@ptrCast(large_buffer.ptr));
    try testing.expect(len == 900);
    
    // Search in large buffer
    const found = zig_libc.string.strstr(@ptrCast(large_buffer.ptr), "DEFG");
    try testing.expect(found != null);
}
