const std = @import("std");
const framework = @import("framework");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    framework.printHeader();
    std.debug.print("Running Computation Benchmarks...\n\n", .{});

    // Benchmark 1: Fibonacci (recursive, n=35)
    const fib_result = try framework.benchmark(
        allocator,
        "Fibonacci(35) - Recursive",
        10,
        2,
        struct {
            pub fn run(_: @This()) void {
                const result = fibonacci(35);
                std.mem.doNotOptimizeAway(&result);
            }
            
            fn fibonacci(n: u32) u64 {
                if (n <= 1) return n;
                return fibonacci(n - 1) + fibonacci(n - 2);
            }
        }{},
    );
    framework.printResult(fib_result);

    // Benchmark 2: Prime number sieve (up to 1M)
    const sieve_result = try framework.benchmark(
        allocator,
        "Prime Sieve (up to 1M)",
        20,
        3,
        struct {
            alloc: std.mem.Allocator,
            pub fn run(self: @This()) void {
                const limit = 1_000_000;
                var is_prime = self.alloc.alloc(bool, limit + 1) catch unreachable;
                defer self.alloc.free(is_prime);
                
                @memset(is_prime, true);
                is_prime[0] = false;
                is_prime[1] = false;
                
                var i: usize = 2;
                while (i * i <= limit) : (i += 1) {
                    if (is_prime[i]) {
                        var j = i * i;
                        while (j <= limit) : (j += i) {
                            is_prime[j] = false;
                        }
                    }
                }
                
                var count: usize = 0;
                for (is_prime) |prime| {
                    if (prime) count += 1;
                }
                std.mem.doNotOptimizeAway(&count);
            }
        }{ .alloc = allocator },
    );
    framework.printResult(sieve_result);

    // Benchmark 3: Matrix multiplication (100x100)
    const matrix_size = 100;
    const matrix_result = try framework.benchmark(
        allocator,
        "Matrix Multiply (100x100)",
        30,
        3,
        struct {
            alloc: std.mem.Allocator,
            size: usize,
            pub fn run(self: @This()) void {
                const s = self.size;
                const a = self.alloc.alloc(f64, s * s) catch unreachable;
                defer self.alloc.free(a);
                const b = self.alloc.alloc(f64, s * s) catch unreachable;
                defer self.alloc.free(b);
                const c = self.alloc.alloc(f64, s * s) catch unreachable;
                defer self.alloc.free(c);
                
                // Initialize matrices
                for (a, 0..) |*val, i| val.* = @floatFromInt(i % 10);
                for (b, 0..) |*val, i| val.* = @floatFromInt((i + 1) % 10);
                @memset(c, 0.0);
                
                // Matrix multiplication
                var i: usize = 0;
                while (i < s) : (i += 1) {
                    var j: usize = 0;
                    while (j < s) : (j += 1) {
                        var sum: f64 = 0.0;
                        var k: usize = 0;
                        while (k < s) : (k += 1) {
                            sum += a[i * s + k] * b[k * s + j];
                        }
                        c[i * s + j] = sum;
                    }
                }
                std.mem.doNotOptimizeAway(c);
            }
        }{ .alloc = allocator, .size = matrix_size },
    );
    framework.printResult(matrix_result);

    // Benchmark 4: Hash computation (1M iterations)
    const hash_result = try framework.benchmark(
        allocator,
        "Hash Computation (1M ops)",
        50,
        5,
        struct {
            pub fn run(_: @This()) void {
                var hasher = std.hash.Wyhash.init(0);
                var i: usize = 0;
                while (i < 1_000_000) : (i += 1) {
                    const data = std.mem.asBytes(&i);
                    hasher.update(data);
                }
                const hash = hasher.final();
                std.mem.doNotOptimizeAway(&hash);
            }
        }{},
    );
    framework.printResult(hash_result);

    std.debug.print("\n✅ Computation benchmarks complete!\n", .{});
}
