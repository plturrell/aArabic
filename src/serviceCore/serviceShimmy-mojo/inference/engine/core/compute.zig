const std = @import("std");
const thread_pool = @import("thread_pool");
const gguf = @import("gguf_loader");

/// Abstract interface for compute operations.
/// This allows swapping the CPU implementation for Metal/CUDA/Vulkan.
pub const ComputeBackend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        
        /// Allocate memory on the device
        alloc: *const fn (ctx: *anyopaque, size: usize) anyerror![]u8,
        /// Free memory on the device
        free: *const fn (ctx: *anyopaque, ptr: []u8) void,
        
        /// Copy data to device
        copyToDevice: *const fn (ctx: *anyopaque, dest: []u8, src: []const u8) anyerror!void,
        /// Copy data from device
        copyFromDevice: *const fn (ctx: *anyopaque, dest: []u8, src: []const u8) anyerror!void,
        
        /// Operations
        matmul: *const fn (
            ctx: *anyopaque, 
            c: []f32, a_data: []const u8, a_type: gguf.QuantizationType, b: []const f32, 
            m: usize, n: usize, k: usize
        ) anyerror!void,
        
        rms_norm: *const fn (
            ctx: *anyopaque,
            output: []f32, input: []const f32, weight: []const f32, eps: f32
        ) void,
        
        softmax: *const fn (
            ctx: *anyopaque,
            output: []f32, input: []const f32
        ) void,
        
        rope: *const fn (
            ctx: *anyopaque,
            output: []f32, input: []const f32, 
            pos: u32, freqs: []const f32, head_dim: u32
        ) void,
    };

    pub fn deinit(self: ComputeBackend) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn alloc(self: ComputeBackend, size: usize) ![]u8 {
        return self.vtable.alloc(self.ptr, size);
    }

    pub fn free(self: ComputeBackend, ptr: []u8) void {
        self.vtable.free(self.ptr, ptr);
    }
    
    // Typed allocation helper
    pub fn allocFloats(self: ComputeBackend, count: usize) ![]f32 {
        const bytes = try self.alloc(count * @sizeOf(f32));
        return std.mem.bytesAsSlice(f32, bytes);
    }
    
    pub fn freeFloats(self: ComputeBackend, ptr: []f32) void {
        self.free(std.mem.sliceAsBytes(ptr));
    }

    pub fn matmul(
        self: ComputeBackend,
        c: []f32,
        a_data: []const u8,
        a_type: gguf.QuantizationType,
        b: []const f32,
        m: usize,
        n: usize,
        k: usize
    ) !void {
        return self.vtable.matmul(self.ptr, c, a_data, a_type, b, m, n, k);
    }

    pub fn rms_norm(self: ComputeBackend, out: []f32, in: []const f32, w: []const f32, eps: f32) void {
        self.vtable.rms_norm(self.ptr, out, in, w, eps);
    }
    
    pub fn softmax(self: ComputeBackend, out: []f32, in: []const f32) void {
        self.vtable.softmax(self.ptr, out, in);
    }
    
    pub fn rope(self: ComputeBackend, out: []f32, in: []const f32, pos: u32, freqs: []const f32, hd: u32) void {
        self.vtable.rope(self.ptr, out, in, pos, freqs, hd);
    }
};
