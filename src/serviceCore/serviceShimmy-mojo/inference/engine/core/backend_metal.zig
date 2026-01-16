const std = @import("std");
const compute = @import("compute");
const gguf = @import("gguf_loader");

// On non-Darwin systems, this backend is a stub.
const is_darwin = @import("builtin").os.tag == .macos;

pub const MetalBackend = struct {
    allocator: std.mem.Allocator,
    device: if (is_darwin) *anyopaque else void,
    queue: if (is_darwin) *anyopaque else void,
    
    // In a real implementation, we would hold references to compiled pipelines here
    // matmul_pipeline: *MTLComputePipelineState, 

    pub fn init(allocator: std.mem.Allocator) !compute.ComputeBackend {
        if (!is_darwin) return error.UnsupportedPlatform;

        // In a real implementation, we would use objc_msgSend to:
        // 1. MTLCreateSystemDefaultDevice()
        // 2. device.newCommandQueue()
        // 3. Compile .metal shaders
        
        std.debug.print("⚡ Initializing Metal Backend (Apple Silicon)...\n", .{});
        
        // Mocking the successful initialization for this demonstration
        const self = try allocator.create(MetalBackend);
        self.* = .{ 
            .allocator = allocator,
            .device = undefined, // Placeholder for MTLDevice
            .queue = undefined,  // Placeholder for MTLCommandQueue
        };
        
        return compute.ComputeBackend{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    fn deinit(ctx: *anyopaque) void {
        const self: *MetalBackend = @ptrCast(@alignCast(ctx));
        // Release Metal objects
        self.allocator.destroy(self);
    }

    fn alloc(ctx: *anyopaque, size: usize) ![]u8 {
        // Metal: [device newBufferWithLength:size options:MTLResourceStorageModeShared]
        // This allocates "Unified Memory" accessible by both CPU and GPU
        const self: *MetalBackend = @ptrCast(@alignCast(ctx));
        return self.allocator.alloc(u8, size); // Fallback to RAM for prototype
    }

    fn free(ctx: *anyopaque, ptr: []u8) void {
        const self: *MetalBackend = @ptrCast(@alignCast(ctx));
        self.allocator.free(ptr);
    }

    fn copyToDevice(ctx: *anyopaque, dest: []u8, src: []const u8) !void {
        _ = ctx;
        // For Shared memory, this is just memcpy.
        // For Private memory, this would involve a BlitEncoder.
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
        _ = ctx;
        // Metal Implementation Flow:
        // 1. Get Command Buffer from Queue.
        // 2. Get Compute Encoder.
        // 3. Set Pipeline State (Q4_0 Matmul Shader).
        // 4. Set Buffers (A, B, C).
        // 5. Dispatch Threadgroups (m/8, n/8, 1).
        // 6. Commit and Wait.
        
        // For this prototype, we fallback to CPU math to allow compilation without Metal SDK
        // In a real "World Class" engine, this line calls the GPU.
        const matrix_ops = @import("matrix_ops");
        try matrix_ops.matmul_quantized(c, a_data, a_type, b, m, n, k, std.heap.c_allocator, null);
    }

    fn rms_norm(ctx: *anyopaque, out: []f32, in: []const f32, w: []const f32, eps: f32) void {
        _ = ctx;
        // Metal: Dispatch RMS Norm Kernel
        const matrix_ops = @import("matrix_ops");
        matrix_ops.rms_norm(out, in, w, eps);
    }

    fn softmax(ctx: *anyopaque, out: []f32, in: []const f32) void {
        _ = ctx;
        const matrix_ops = @import("matrix_ops");
        matrix_ops.softmax(out, in);
    }

    pub fn rope(ctx: *anyopaque, out: []f32, in: []const f32, pos: u32, _freqs: []const f32, dim: u32) void {
        _ = ctx;
        _ = _freqs;
        const matrix_ops = @import("matrix_ops");
        // Adaptation for different signature
        matrix_ops.apply_rope(out, @constCast(in), pos, dim * 2, 10000.0);
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
