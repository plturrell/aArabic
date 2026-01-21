// GPU Tensor - Tensor that lives entirely on GPU
// Enables GPU-to-GPU operations without host transfers
//
// Key features:
// - Data remains on GPU between operations
// - FP16 storage for Tensor Core compatibility
// - Efficient cuBLAS integration
// - Minimal host-device transfers

const std = @import("std");
const cuda = @import("cuda_bindings");

pub const GpuTensor = struct {
    /// GPU device pointer (FP16 data)
    ptr: [*]f16,
    /// Number of elements
    len: usize,
    /// Byte size on GPU
    byte_size: usize,
    /// Whether we own the memory (should free on deinit)
    owned: bool,

    const Self = @This();

    /// Allocate a new GPU tensor with given number of FP16 elements
    pub fn alloc(num_elements: usize) !Self {
        const byte_size = num_elements * @sizeOf(f16);
        var device_ptr: *anyopaque = undefined;

        const result = cuda.cudaMalloc(@ptrCast(&device_ptr), byte_size);
        if (result != 0) {
            std.debug.print("cudaMalloc failed: error code {}\n", .{result});
            return error.CudaAllocFailed;
        }

        return Self{
            .ptr = @ptrCast(@alignCast(device_ptr)),
            .len = num_elements,
            .byte_size = byte_size,
            .owned = true,
        };
    }

    /// Create from existing GPU pointer (does not own memory)
    pub fn fromPtr(ptr: [*]f16, num_elements: usize) Self {
        return Self{
            .ptr = ptr,
            .len = num_elements,
            .byte_size = num_elements * @sizeOf(f16),
            .owned = false,
        };
    }

    /// Free GPU memory if owned
    pub fn deinit(self: *Self) void {
        if (self.owned) {
            _ = cuda.cudaFree(@ptrCast(self.ptr));
            self.ptr = undefined;
            self.len = 0;
            self.byte_size = 0;
        }
    }

    /// Copy FP32 data from host to GPU as FP16
    pub fn copyFromHostF32(self: *Self, host_data: []const f32) !void {
        if (host_data.len != self.len) {
            return error.SizeMismatch;
        }

        // Convert F32 to F16 on CPU first (TODO: do this on GPU for better perf)
        const fp16_buffer = try std.heap.page_allocator.alloc(f16, host_data.len);
        defer std.heap.page_allocator.free(fp16_buffer);

        for (host_data, 0..) |val, i| {
            fp16_buffer[i] = @floatCast(val);
        }

        const result = cuda.cudaMemcpy(
            @ptrCast(self.ptr),
            fp16_buffer.ptr,
            self.byte_size,
            cuda.cudaMemcpyHostToDevice,
        );
        if (result != 0) {
            return error.CudaCopyFailed;
        }
    }

    /// Copy FP16 data from host to GPU
    pub fn copyFromHostF16(self: *Self, host_data: []const f16) !void {
        if (host_data.len != self.len) {
            return error.SizeMismatch;
        }

        const result = cuda.cudaMemcpy(
            @ptrCast(self.ptr),
            host_data.ptr,
            self.byte_size,
            cuda.cudaMemcpyHostToDevice,
        );
        if (result != 0) {
            return error.CudaCopyFailed;
        }
    }

    /// Copy GPU data to host as FP32
    pub fn copyToHostF32(self: *const Self, host_buffer: []f32) !void {
        if (host_buffer.len < self.len) {
            return error.BufferTooSmall;
        }

        // Copy FP16 from GPU
        const fp16_buffer = try std.heap.page_allocator.alloc(f16, self.len);
        defer std.heap.page_allocator.free(fp16_buffer);

        const result = cuda.cudaMemcpy(
            fp16_buffer.ptr,
            @ptrCast(self.ptr),
            self.byte_size,
            cuda.cudaMemcpyDeviceToHost,
        );
        if (result != 0) {
            return error.CudaCopyFailed;
        }

        // Convert F16 to F32
        for (fp16_buffer, 0..) |val, i| {
            host_buffer[i] = @floatCast(val);
        }
    }

    /// Zero out the tensor
    pub fn zero(self: *Self) !void {
        const result = cuda.cudaMemset(@ptrCast(self.ptr), 0, self.byte_size);
        if (result != 0) {
            return error.CudaMemsetFailed;
        }
    }

    /// Get raw device pointer for cuBLAS operations
    pub fn devicePtr(self: *const Self) *anyopaque {
        return @ptrCast(@constCast(self.ptr));
    }

    /// Get memory usage in bytes
    pub fn memoryUsage(self: *const Self) usize {
        return self.byte_size;
    }
};

