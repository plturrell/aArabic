const std = @import("std");
const framework = @import("framework");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    framework.printHeader();
    std.debug.print("Running Array Operations Benchmarks...\n\n", .{});

    // Benchmark 1: Array sum (1M elements)
    const array_size = 1_000_000;
    const data = try allocator.alloc(u64, array_size);
    defer allocator.free(data);
    
    for (data, 0..) |*item, i| {
        item.* = i;
    }

    const sum_result = try framework.benchmark(
        allocator,
        "Array Sum (1M elements)",
        100,
        10,
        struct {
            arr: []u64,
            pub fn run(self: @This()) void {
                var sum: u64 = 0;
                for (self.arr) |val| {
                    sum +%= val;
                }
                std.mem.doNotOptimizeAway(&sum);
            }
        }{ .arr = data },
    );
    framework.printResult(sum_result);

    // Benchmark 2: Array multiply and add (1M elements)
    const multiply_result = try framework.benchmark(
        allocator,
        "Array Multiply-Add (1M elements)",
        100,
        10,
        struct {
            arr: []u64,
            pub fn run(self: @This()) void {
                for (self.arr) |*val| {
                    val.* = val.* * 2 + 1;
                }
            }
        }{ .arr = data },
    );
    framework.printResult(multiply_result);

    // Benchmark 3: Array sort (100K elements)
    const sort_size = 100_000;
    const sort_data = try allocator.alloc(u64, sort_size);
    defer allocator.free(sort_data);
    
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    
    const sort_result = try framework.benchmark(
        allocator,
        "Array Sort (100K elements)",
        20,
        3,
        struct {
            arr: []u64,
            rng: std.Random,
            pub fn run(self: @This()) void {
                // Randomize before each sort
                for (self.arr) |*item| {
                    item.* = self.rng.int(u64);
                }
                std.mem.sort(u64, self.arr, {}, comptime std.sort.asc(u64));
            }
        }{ .arr = sort_data, .rng = random },
    );
    framework.printResult(sort_result);

    std.debug.print("\n✅ Array operations benchmarks complete!\n", .{});
}
