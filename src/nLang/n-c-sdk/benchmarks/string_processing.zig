const std = @import("std");
const framework = @import("framework");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    framework.printHeader();
    std.debug.print("Running String Processing Benchmarks...\n\n", .{});

    // Benchmark 1: String concatenation (100K operations)
    const concat_result = try framework.benchmark(
        allocator,
        "String Concatenation (100K ops)",
        50,
        5,
        struct {
            alloc: std.mem.Allocator,
            pub fn run(self: @This()) void {
                var list = std.ArrayList(u8){};
                defer list.deinit(self.alloc);
                
                var i: usize = 0;
                while (i < 100_000) : (i += 1) {
                    list.appendSlice(self.alloc, "test") catch unreachable;
                }
                std.mem.doNotOptimizeAway(list.items.ptr);
            }
        }{ .alloc = allocator },
    );
    framework.printResult(concat_result);

    // Benchmark 2: String search (in 1MB text)
    const text_size = 1024 * 1024; // 1MB
    const text = try allocator.alloc(u8, text_size);
    defer allocator.free(text);
    
    // Fill with pattern
    var i: usize = 0;
    while (i < text_size) : (i += 1) {
        text[i] = @intCast('a' + (i % 26));
    }
    
    const search_result = try framework.benchmark(
        allocator,
        "String Search (1MB text)",
        100,
        10,
        struct {
            haystack: []const u8,
            pub fn run(self: @This()) void {
                const needle = "zyxwvu";
                var count: usize = 0;
                var idx: usize = 0;
                while (idx < self.haystack.len) {
                    if (std.mem.indexOf(u8, self.haystack[idx..], needle)) |pos| {
                        count += 1;
                        idx += pos + needle.len;
                    } else {
                        break;
                    }
                }
                std.mem.doNotOptimizeAway(&count);
            }
        }{ .haystack = text },
    );
    framework.printResult(search_result);

    // Benchmark 3: String parsing (100K integers)
    const parse_result = try framework.benchmark(
        allocator,
        "String to Int Parsing (100K ops)",
        50,
        5,
        struct {
            pub fn run(_: @This()) void {
                var sum: i64 = 0;
                var j: usize = 0;
                while (j < 100_000) : (j += 1) {
                    const num_str = "12345";
                    const num = std.fmt.parseInt(i64, num_str, 10) catch unreachable;
                    sum += num;
                }
                std.mem.doNotOptimizeAway(&sum);
            }
        }{},
    );
    framework.printResult(parse_result);

    std.debug.print("\n✅ String processing benchmarks complete!\n", .{});
}
