// Thread safety tests for zig-libc Phase 1.1
// Phase 1.1 Month 6 Week 23-24
// Tests concurrent function usage and thread safety

const std = @import("std");
const testing = std.testing;
const Thread = std.Thread;
const zig_libc = @import("zig-libc");

// Test data structure for thread tests
const ThreadTestData = struct {
    buffer: []u8,
    input: []const u8,
    result: std.atomic.Value(bool),
    
    fn init(allocator: std.mem.Allocator, size: usize, input_str: []const u8) !*ThreadTestData {
        const data = try allocator.create(ThreadTestData);
        data.buffer = try allocator.alloc(u8, size);
        data.input = input_str;
        data.result = std.atomic.Value(bool).init(true);
        return data;
    }
    
    fn deinit(self: *ThreadTestData, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
        allocator.destroy(self);
    }
};

// Test concurrent string length operations
test "thread_safety: concurrent strlen calls" {
    const test_strings = [_][]const u8{
        "Thread safety test string 1",
        "Thread safety test string 2",
        "Thread safety test string 3",
        "Thread safety test string 4",
    };
    
    var threads: [4]Thread = undefined;
    var results: [4]usize = undefined;
    
    const ThreadContext = struct {
        str: []const u8,
        result: *usize,
        
        fn run(ctx: *const @This()) void {
            // Perform many strlen operations
            var i: usize = 0;
            while (i < 1000) : (i += 1) {
                const len = zig_libc.string.strlen(@ptrCast(ctx.str.ptr));
                if (i == 0) ctx.result.* = len;
            }
        }
    };
    
    // Launch threads
    for (test_strings, 0..) |str, i| {
        const ctx = ThreadContext{
            .str = str,
            .result = &results[i],
        };
        threads[i] = try Thread.spawn(.{}, ThreadContext.run, .{&ctx});
    }
    
    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify results
    for (test_strings, 0..) |str, i| {
        try testing.expectEqual(str.len, results[i]);
    }
}

// Test concurrent string comparison
test "thread_safety: concurrent strcmp calls" {
    const str1 = "Concurrent string comparison test";
    const str2 = "Concurrent string comparison test";
    const str3 = "Different string for comparison";
    
    var results = [_]std.atomic.Value(i32){
        std.atomic.Value(i32).init(0),
        std.atomic.Value(i32).init(0),
    } ** 10;
    
    const ThreadContext = struct {
        s1: []const u8,
        s2: []const u8,
        result: *std.atomic.Value(i32),
        
        fn run(ctx: *const @This()) void {
            var i: usize = 0;
            while (i < 1000) : (i += 1) {
                const cmp = zig_libc.string.strcmp(@ptrCast(ctx.s1.ptr), @ptrCast(ctx.s2.ptr));
                ctx.result.store(cmp, .monotonic);
            }
        }
    };
    
    var threads: [10]Thread = undefined;
    
    // Launch threads with different comparison pairs
    for (0..10) |i| {
        const ctx = if (i % 2 == 0)
            ThreadContext{ .s1 = str1, .s2 = str2, .result = &results[i] }
        else
            ThreadContext{ .s1 = str1, .s2 = str3, .result = &results[i] };
        threads[i] = try Thread.spawn(.{}, ThreadContext.run, .{&ctx});
    }
    
    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify results
    for (0..10) |i| {
        if (i % 2 == 0) {
            try testing.expectEqual(@as(i32, 0), results[i].load(.monotonic));
        } else {
            try testing.expect(results[i].load(.monotonic) != 0);
        }
    }
}

// Test concurrent character classification
test "thread_safety: concurrent ctype operations" {
    const test_string = "AbC123XyZ!@#";
    
    var alpha_counts = [_]std.atomic.Value(usize){std.atomic.Value(usize).init(0)} ** 8;
    var digit_counts = [_]std.atomic.Value(usize){std.atomic.Value(usize).init(0)} ** 8;
    
    const ThreadContext = struct {
        str: []const u8,
        alpha_count: *std.atomic.Value(usize),
        digit_count: *std.atomic.Value(usize),
        
        fn run(ctx: *const @This()) void {
            var i: usize = 0;
            while (i < 100) : (i += 1) {
                var alpha: usize = 0;
                var digits: usize = 0;
                
                for (ctx.str) |ch| {
                    if (zig_libc.ctype.isalpha(@intCast(ch))) alpha += 1;
                    if (zig_libc.ctype.isdigit(@intCast(ch))) digits += 1;
                }
                
                ctx.alpha_count.store(alpha, .monotonic);
                ctx.digit_count.store(digits, .monotonic);
            }
        }
    };
    
    var threads: [8]Thread = undefined;
    
    // Launch threads
    for (0..8) |i| {
        const ctx = ThreadContext{
            .str = test_string,
            .alpha_count = &alpha_counts[i],
            .digit_count = &digit_counts[i],
        };
        threads[i] = try Thread.spawn(.{}, ThreadContext.run, .{&ctx});
    }
    
    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify all threads got same counts
    const expected_alpha: usize = 6;
    const expected_digits: usize = 3;
    
    for (0..8) |i| {
        try testing.expectEqual(expected_alpha, alpha_counts[i].load(.monotonic));
        try testing.expectEqual(expected_digits, digit_counts[i].load(.monotonic));
    }
}

// Test concurrent memory operations
test "thread_safety: concurrent memcpy operations" {
    const allocator = testing.allocator;
    
    const src_data = "Concurrent memory copy test data";
    var destinations: [8][]u8 = undefined;
    
    // Allocate destination buffers
    for (&destinations) |*dest| {
        dest.* = try allocator.alloc(u8, src_data.len + 1);
    }
    defer {
        for (destinations) |dest| {
            allocator.free(dest);
        }
    }
    
    const ThreadContext = struct {
        src: []const u8,
        dest: []u8,
        
        fn run(ctx: *const @This()) void {
            var i: usize = 0;
            while (i < 100) : (i += 1) {
                _ = zig_libc.memory.memcpy(ctx.dest.ptr, ctx.src.ptr, ctx.src.len);
                ctx.dest[ctx.src.len] = 0;
            }
        }
    };
    
    var threads: [8]Thread = undefined;
    
    // Launch threads
    for (0..8) |i| {
        const ctx = ThreadContext{
            .src = src_data,
            .dest = destinations[i],
        };
        threads[i] = try Thread.spawn(.{}, ThreadContext.run, .{&ctx});
    }
    
    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify all destinations have correct data
    for (destinations) |dest| {
        try testing.expectEqualStrings(src_data, dest[0..src_data.len]);
    }
}

// Test reentrant strtok_r with concurrent threads
test "thread_safety: concurrent strtok_r (reentrant)" {
    const allocator = testing.allocator;
    
    const TokenResult = struct {
        tokens: std.ArrayList([]const u8),
        
        allocator: std.mem.Allocator,
        
        fn init(alloc: std.mem.Allocator) @This() {
            return .{ .tokens = std.ArrayList([]const u8){}, .allocator = alloc };
        }
        
        fn deinit(self: *@This()) void {
            self.tokens.deinit(self.allocator);
        }
    };
    
    var results: [4]TokenResult = undefined;
    for (&results) |*r| {
        r.* = TokenResult.init(allocator);
    }
    defer {
        for (&results) |*r| {
            r.deinit();
        }
    }
    
    const ThreadContext = struct {
        allocator: std.mem.Allocator,
        input: []const u8,
        result: *TokenResult,
        
        fn run(ctx: *const @This()) void {
            // Each thread has its own buffer and saveptr
            var buffer = ctx.allocator.alloc(u8, ctx.input.len + 1) catch return;
            defer ctx.allocator.free(buffer);
            
            @memcpy(buffer[0..ctx.input.len], ctx.input);
            buffer[ctx.input.len] = 0;
            
            var saveptr: ?[*:0]u8 = null;
            const delim: [*:0]const u8 = ",";
            
            var token = zig_libc.string.strtok_r(@ptrCast(buffer.ptr), delim, &saveptr);
            while (token != null) {
                const len = zig_libc.string.strlen(token.?);
                const token_copy = ctx.allocator.alloc(u8, len) catch return;
                @memcpy(token_copy, token.?[0..len]);
                ctx.result.tokens.append(ctx.allocator, token_copy) catch return;
                token = zig_libc.string.strtok_r(null, delim, &saveptr);
            }
        }
    };
    
    const test_inputs = [_][]const u8{
        "apple,banana,cherry",
        "one,two,three,four",
        "red,green,blue",
        "alpha,beta,gamma,delta",
    };
    
    var threads: [4]Thread = undefined;
    
    // Launch threads with different inputs
    for (0..4) |i| {
        const ctx = ThreadContext{
            .allocator = allocator,
            .input = test_inputs[i],
            .result = &results[i],
        };
        threads[i] = try Thread.spawn(.{}, ThreadContext.run, .{&ctx});
    }
    
    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify results
    const expected_counts = [_]usize{ 3, 4, 3, 4 };
    for (0..4) |i| {
        try testing.expectEqual(expected_counts[i], results[i].tokens.items.len);
        // Clean up token copies
        for (results[i].tokens.items) |token| {
            allocator.free(token);
        }
    }
}

// Test case transformation under concurrent load
test "thread_safety: concurrent toupper/tolower" {
    const test_string = "ConCurrent CaSe TranSformatioN TeSt";
    
    var results_upper: [6][]u8 = undefined;
    var results_lower: [6][]u8 = undefined;
    
    const allocator = testing.allocator;
    
    // Allocate result buffers
    for (&results_upper) |*r| {
        r.* = try allocator.alloc(u8, test_string.len + 1);
    }
    for (&results_lower) |*r| {
        r.* = try allocator.alloc(u8, test_string.len + 1);
    }
    defer {
        for (results_upper) |r| allocator.free(r);
        for (results_lower) |r| allocator.free(r);
    }
    
    const ThreadContext = struct {
        src: []const u8,
        dest_upper: []u8,
        dest_lower: []u8,
        
        fn run(ctx: *const @This()) void {
            var i: usize = 0;
            while (i < 100) : (i += 1) {
                // Transform to uppercase
                for (ctx.src, 0..) |ch, j| {
                    ctx.dest_upper[j] = @intCast(zig_libc.ctype.toupper(@intCast(ch)));
                }
                ctx.dest_upper[ctx.src.len] = 0;
                
                // Transform to lowercase  
                for (ctx.src, 0..) |ch, j| {
                    ctx.dest_lower[j] = @intCast(zig_libc.ctype.tolower(@intCast(ch)));
                }
                ctx.dest_lower[ctx.src.len] = 0;
            }
        }
    };
    
    var threads: [6]Thread = undefined;
    
    // Launch threads
    for (0..6) |i| {
        const ctx = ThreadContext{
            .src = test_string,
            .dest_upper = results_upper[i],
            .dest_lower = results_lower[i],
        };
        threads[i] = try Thread.spawn(.{}, ThreadContext.run, .{&ctx});
    }
    
    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify all transformations are correct
    const expected_upper = "CONCURRENT CASE TRANSFORMATION TEST";
    const expected_lower = "concurrent case transformation test";
    
    for (results_upper) |result| {
        try testing.expectEqualStrings(expected_upper, result[0..expected_upper.len]);
    }
    for (results_lower) |result| {
        try testing.expectEqualStrings(expected_lower, result[0..expected_lower.len]);
    }
}

// Test memory search operations under concurrent load
test "thread_safety: concurrent memchr operations" {
    const data = "Finding bytes in concurrent threads is important for thread safety";
    
    var results: [8]std.atomic.Value(usize) = undefined;
    for (&results) |*r| {
        r.* = std.atomic.Value(usize).init(0);
    }
    
    const ThreadContext = struct {
        data: []const u8,
        target: u8,
        result: *std.atomic.Value(usize),
        
        fn run(ctx: *const @This()) void {
            var i: usize = 0;
            while (i < 1000) : (i += 1) {
                const found = zig_libc.memory.memchr(ctx.data.ptr, @intCast(ctx.target), ctx.data.len);
                if (found) |ptr| {
                    const offset = @intFromPtr(ptr) - @intFromPtr(ctx.data.ptr);
                    ctx.result.store(offset + 1, .monotonic); // Store offset+1 to distinguish from initial 0
                }
            }
        }
    };
    
    var threads: [8]Thread = undefined;
    
    // Launch threads searching for different targets
    const targets = "Ficdnbts";
    for (0..8) |i| {
        const ctx = ThreadContext{
            .data = data,
            .target = targets[i],
            .result = &results[i],
        };
        threads[i] = try Thread.spawn(.{}, ThreadContext.run, .{&ctx});
    }
    
    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify all threads found their targets  
    for (results) |result| {
        const offset = result.load(.monotonic);
        try testing.expect(offset > 0); // Should be set (offset+1)
    }
}

// Stress test: Many threads performing various operations
test "thread_safety: stress test with mixed operations" {
    const allocator = testing.allocator;
    const thread_count = 16;
    
    var success_count = std.atomic.Value(usize).init(0);
    
    const ThreadContext = struct {
        thread_id: usize,
        success: *std.atomic.Value(usize),
        allocator: std.mem.Allocator,
        
        fn run(ctx: *const @This()) void {
            var buffer: [200]u8 = undefined;
            @memset(&buffer, 0);
            
            // Perform mixed operations
            var i: usize = 0;
            while (i < 100) : (i += 1) {
                // String operations
                const src1 = "Thread";
                _ = zig_libc.string.strcpy(@ptrCast(&buffer), src1);
                
                const src2 = " Safety";
                _ = zig_libc.string.strcat(@ptrCast(&buffer), src2);
                
                const len = zig_libc.string.strlen(@ptrCast(&buffer));
                if (len != 13) return;
                
                // Character classification
                if (!zig_libc.ctype.isalpha('T')) return;
                if (!zig_libc.ctype.isspace(' ')) return;
                
                // Memory operations
                var mem_buf: [50]u8 = undefined;
                _ = zig_libc.memory.memset(&mem_buf, 'A', 50);
                _ = zig_libc.memory.memcpy(@ptrCast(&buffer[20]), &mem_buf, 10);
                
                const found = zig_libc.memory.memchr(&mem_buf, 'A', 50);
                if (found == null) return;
            }
            
            _ = ctx.success.fetchAdd(1, .monotonic);
        }
    };
    
    var threads: [thread_count]Thread = undefined;
    
    // Launch stress test threads
    for (0..thread_count) |i| {
        const ctx = ThreadContext{
            .thread_id = i,
            .success = &success_count,
            .allocator = allocator,
        };
        threads[i] = try Thread.spawn(.{}, ThreadContext.run, .{&ctx});
    }
    
    // Wait for all threads
    for (threads) |thread| {
        thread.join();
    }
    
    // All threads should complete successfully
    try testing.expectEqual(@as(usize, thread_count), success_count.load(.monotonic));
}
