const std = @import("std");
const bindgen = @import("zig_bindgen.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <zig_file>\n", .{args[0]});
        return;
    }

    const file_path = args[1];
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const source_size = (try file.stat()).size;
    const source = try allocator.alloc(u8, source_size);
    defer allocator.free(source);

    _ = try file.readAll(source);

    var parser = bindgen.ZigParser.init(allocator, source);
    var exports = try parser.parseExports();
    defer {
        for (exports.items) |*e| e.deinit(allocator);
        exports.deinit(allocator);
    }

    var generator = bindgen.MojoGenerator.init(allocator);
    const mojo_code = try generator.generate(exports);
    defer allocator.free(mojo_code);

    std.debug.print("{s}\n", .{mojo_code});
}
