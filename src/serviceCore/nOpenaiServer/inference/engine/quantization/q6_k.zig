const std = @import("std");
const common = @import("common");

/// Q6_K quantization (K-quants, 6-bit) as defined in ggml
/// Block size: 256 values packed into:
/// - ql: lower 4 bits for each value (128 bytes)
/// - qh: upper 2 bits packed (64 bytes)
/// - scales: 16 int8 scales (one per 16-value sub-block)
/// - d: fp16 super-block scale

pub const QK_K: usize = 256;

pub const BlockQ6_K = extern struct {
    ql: [QK_K / 2]u8,
    qh: [QK_K / 4]u8,
    scales: [QK_K / 16]i8,
    d: u16, // fp16 super-block scale
};

/// Debug flag for first block
var debug_first_block: bool = true;

/// Dequantize a single Q6_K block (256 values) into `output`.
pub fn dequantizeBlock(output: []f32, block: *const BlockQ6_K) void {
    std.debug.assert(output.len == QK_K);

    var d = common.f16_to_f32(block.d);

    // Debug: print first block's d value
    if (debug_first_block) {
        std.debug.print("🔬 Q6_K dequant: d_raw=0x{x:0>4}, d_f32={d:.10}\n", .{ block.d, d });
        std.debug.print("   ql[0:4]={d},{d},{d},{d} qh[0:4]={d},{d},{d},{d}\n", .{
            block.ql[0], block.ql[1], block.ql[2], block.ql[3],
            block.qh[0], block.qh[1], block.qh[2], block.qh[3],
        });
        std.debug.print("   scales[0:4]={d},{d},{d},{d}\n", .{
            block.scales[0], block.scales[1], block.scales[2], block.scales[3],
        });
        debug_first_block = false;
    }

    // Handle NaN/Inf scales - replace with 0 to avoid propagation
    if (std.math.isNan(d) or std.math.isInf(d)) d = 0.0;
    const ql = &block.ql;
    const qh = &block.qh;
    const sc = &block.scales;

    var out_idx: usize = 0;
    var ql_idx: usize = 0;
    var qh_idx: usize = 0;
    var sc_idx: usize = 0;

    // Two chunks of 128 values per block
    var n: usize = 0;
    while (n < QK_K) : (n += 128) {
        for (0..32) |l| {
            const is: usize = l / 16;

            const q1 = @as(i8, @intCast((ql[ql_idx + l + 0] & 0xF) | (((qh[qh_idx + l] >> 0) & 0x3) << 4))) - 32;
            const q2 = @as(i8, @intCast((ql[ql_idx + l + 32] & 0xF) | (((qh[qh_idx + l] >> 2) & 0x3) << 4))) - 32;
            const q3 = @as(i8, @intCast((ql[ql_idx + l + 0] >> 4) | (((qh[qh_idx + l] >> 4) & 0x3) << 4))) - 32;
            const q4 = @as(i8, @intCast((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 0x3) << 4))) - 32;

            const s0 = @as(f32, @floatFromInt(sc[sc_idx + is + 0]));
            const s1 = @as(f32, @floatFromInt(sc[sc_idx + is + 2]));
            const s2 = @as(f32, @floatFromInt(sc[sc_idx + is + 4]));
            const s3 = @as(f32, @floatFromInt(sc[sc_idx + is + 6]));

            output[out_idx + l + 0] = d * s0 * @as(f32, @floatFromInt(q1));
            output[out_idx + l + 32] = d * s1 * @as(f32, @floatFromInt(q2));
            output[out_idx + l + 64] = d * s2 * @as(f32, @floatFromInt(q3));
            output[out_idx + l + 96] = d * s3 * @as(f32, @floatFromInt(q4));
        }

        out_idx += 128;
        ql_idx += 64;
        qh_idx += 32;
        sc_idx += 8;
    }
}
