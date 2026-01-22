const std = @import("std");
const compute = @import("compute");
const matrix_ops = @import("matrix_ops");
const thread_pool = @import("thread_pool");
const gguf = @import("gguf_loader");

pub const CpuBackend = struct {
    allocator: std.mem.Allocator,
    pool: *thread_pool.ThreadPool,

    pub fn init(allocator: std.mem.Allocator, pool: *thread_pool.ThreadPool) !compute.ComputeBackend {
        const self = try allocator.create(CpuBackend);
        self.* = .{
            .allocator = allocator,
            .pool = pool,
        };
        
        return compute.ComputeBackend{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    fn deinit(ctx: *anyopaque) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        self.allocator.destroy(self);
    }

    fn alloc(ctx: *anyopaque, size: usize) ![]u8 {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        return self.allocator.alloc(u8, size);
    }

    fn free(ctx: *anyopaque, ptr: []u8) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        self.allocator.free(ptr);
    }

    fn copyToDevice(ctx: *anyopaque, dest: []u8, src: []const u8) !void {
        _ = ctx;
        @memcpy(dest, src);
    }

    fn copyFromDevice(ctx: *anyopaque, dest: []u8, src: []const u8) !void {
        _ = ctx;
        @memcpy(dest, src);
    }

    fn matmul(
        ctx: *anyopaque,
        c: []f32,
        a_data: []const u8,
        a_type: gguf.QuantizationType,
        b: []const f32,
        m: usize,
        n: usize,
        k: usize
    ) !void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        try matrix_ops.matmul_quantized(c, a_data, a_type, b, m, n, k, self.allocator, self.pool);
    }

    fn rms_norm(
        ctx: *anyopaque,
        output: []f32,
        input: []const f32,
        weight: []const f32,
        eps: f32
    ) void {
        _ = ctx;
        matrix_ops.rms_norm(output, input, weight, eps);
    }

    fn softmax(
        ctx: *anyopaque,
        output: []f32,
        input: []const f32
    ) void {
        _ = ctx;
        matrix_ops.softmax(output, input);
    }

    fn rope(
        ctx: *anyopaque,
        output: []f32,
        input: []const f32,
        pos: u32,
        _freqs: []const f32,
        head_dim: u32
    ) void {
        _ = ctx;
        _ = _freqs;
        matrix_ops.apply_rope(output, @constCast(input), pos, head_dim * 2, 10000.0); // Temporary adaptation
        // Note: apply_rope signature in matrix_ops is specialized, we should generalize it later.
        // For now, we assume CpuBackend uses the optimized matrix_ops directly.
    }

    const vtable = compute.ComputeBackend.VTable{
        .deinit = deinit,
        .alloc = alloc,
        .free = free,
        .copyToDevice = copyToDevice,
        .copyFromDevice = copyFromDevice,
        .matmul = matmul,
        .rms_norm = rms_norm,
        .softmax = softmax,
        .rope = rope,
    };
};
