const std = @import("std");

pub export fn inference_load_model(path: [*:0]const u8) callconv(.c) i32 {
    _ = path;
    return 0;
}

pub export fn inference_is_loaded() callconv(.c) i32 {
    return 1;
}

pub export fn inference_get_info(buf: [*]u8, buf_size: usize) callconv(.c) i32 {
    const msg = "max_output=4096;embedding_dims=64";
    if (msg.len > buf_size) return -1;
    std.mem.copyForwards(u8, buf[0..msg.len], msg);
    return @intCast(msg.len);
}

pub export fn inference_generate(
    prompt: [*]const u8,
    prompt_len: usize,
    max_tokens: u32,
    temperature: f32,
    out_buf: [*]u8,
    out_buf_size: usize,
) callconv(.c) i32 {
    _ = max_tokens;
    _ = temperature;
    const prefix = "fixture: ";
    const total: usize = if (out_buf_size < prefix.len + prompt_len) out_buf_size else prefix.len + prompt_len;
    var i: usize = 0;
    while (i < prefix.len and i < total) : (i += 1) {
        out_buf[i] = prefix[i];
    }
    var j: usize = 0;
    while (i < total and j < prompt_len) : (i += 1) {
        out_buf[i] = prompt[j];
        j += 1;
    }
    return @intCast(total);
}

pub export fn inference_embed(input: [*]const u8, input_len: usize, out: [*]f32, out_len: usize) callconv(.c) i32 {
    _ = input;
    _ = input_len;
    if (out_len < 4) return -1;
    out[0] = 0.11;
    out[1] = 0.22;
    out[2] = 0.33;
    out[3] = 0.44;
    return 4;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .name = "inference_fixture",
        .root_module = .{
            .root_source_file = .{ .src_path = "server/inference_fixture.zig" },
            .target = target,
            .optimize = optimize,
        },
        .linkage = .dynamic,
    });
    b.installArtifact(lib);
}
