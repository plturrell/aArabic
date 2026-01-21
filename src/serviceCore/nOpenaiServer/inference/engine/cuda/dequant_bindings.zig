// Dequantization Kernel Bindings for Zig
// FFI layer for Mojo-compiled dequantization CUDA kernels
//
// Links against: libdequant_kernels.so (compiled from dequant_kernels.mojo)
// Provides: Q4_0, Q8_0, Q4_K → FP16 GPU dequantization for Tensor Core input
//
// Usage Flow:
// 1. Quantized weights arrive in GGUF format (Q4_K, Q8_0, etc.)
// 2. Dequant kernel converts to FP16 on GPU
// 3. cuBLAS GemmEx uses FP16 with Tensor Cores
// 4. Result in FP32 for numerical stability

const std = @import("std");
const cuda = @import("cuda_bindings");

const log = std.log.scoped(.dequant);

// ============================================================================
// Quantization Constants (matching GGUF spec)
// ============================================================================

/// Q4_0: 32 weights per block, 18 bytes (2B f16 scale + 16B packed 4-bit)
pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Q8_0: 32 weights per block, 36 bytes (4B f32 scale + 32B int8)
pub const Q8_0_BLOCK_SIZE: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 36;

/// Q4_K: 256 weights per block, 144 bytes (complex K-quant format)
pub const Q4_K_BLOCK_SIZE: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Q6_K: 256 weights per block, 210 bytes
pub const Q6_K_BLOCK_SIZE: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;

// ============================================================================
// External CUDA Dequantization Kernels (from libdequant_kernels.so)
// ============================================================================
// These link against the compiled CUDA kernels in cuda/kernels/libdequant_kernels.so
// The kernels are compiled with nvcc from dequant_kernels.cu
//
// Build the library: cd cuda/kernels && make
// The library must be in LD_LIBRARY_PATH or the same directory as the executable

/// Dequantize Q4_0 blocks to FP16 on GPU
/// Format: 18 bytes -> 32 FP16 values per block
/// Returns 0 on success, -1 on error
pub extern "dequant_kernels" fn dequant_q4_0_fp16(
    input: [*]const u8,
    output: [*]f16,
    num_blocks: c_int,
    stream: ?*anyopaque,
) c_int;

/// Dequantize Q8_0 blocks to FP16 on GPU
/// Format: 36 bytes -> 32 FP16 values per block
pub extern "dequant_kernels" fn dequant_q8_0_fp16(
    input: [*]const u8,
    output: [*]f16,
    num_blocks: c_int,
    stream: ?*anyopaque,
) c_int;

/// Dequantize Q4_K blocks to FP16 on GPU
/// Format: 144 bytes -> 256 FP16 values per block
pub extern "dequant_kernels" fn dequant_q4_k_fp16(
    input: [*]const u8,
    output: [*]f16,
    num_blocks: c_int,
    stream: ?*anyopaque,
) c_int;

/// Dequantize Q6_K blocks to FP16 on GPU
/// Format: 210 bytes -> 256 FP16 values per block
pub extern "dequant_kernels" fn dequant_q6_k_fp16(
    input: [*]const u8,
    output: [*]f16,
    num_blocks: c_int,
    stream: ?*anyopaque,
) c_int;

// ============================================================================
// Wrapper functions with Zig-friendly names (for backwards compatibility)
// ============================================================================

pub fn mojo_dequant_q4_0_fp16(
    input: [*]const u8,
    output: [*]f16,
    num_blocks: i32,
    stream: ?*anyopaque,
) i32 {
    return dequant_q4_0_fp16(input, output, num_blocks, stream);
}

pub fn mojo_dequant_q8_0_fp16(
    input: [*]const u8,
    output: [*]f16,
    num_blocks: i32,
    stream: ?*anyopaque,
) i32 {
    return dequant_q8_0_fp16(input, output, num_blocks, stream);
}

pub fn mojo_dequant_q4_k_fp16(
    input: [*]const u8,
    output: [*]f16,
    num_blocks: i32,
    stream: ?*anyopaque,
) i32 {
    return dequant_q4_k_fp16(input, output, num_blocks, stream);
}

pub fn mojo_dequant_q6_k_fp16(
    input: [*]const u8,
    output: [*]f16,
    num_blocks: i32,
    stream: ?*anyopaque,
) i32 {
    return dequant_q6_k_fp16(input, output, num_blocks, stream);
}

/// Get output buffer size in FP16 elements
/// quant_type: 2=Q4_0, 8=Q8_0, 12=Q4_K
pub fn mojo_dequant_get_output_size(
    quant_type: i32,
    num_blocks: i32,
) i32 {
    const block_size: i32 = switch (quant_type) {
        2 => @intCast(Q4_0_BLOCK_SIZE),
        8 => @intCast(Q8_0_BLOCK_SIZE),
        12 => @intCast(Q4_K_BLOCK_SIZE),
        else => 32,
    };
    return num_blocks * block_size;
}

/// Get input buffer size in bytes
pub fn mojo_dequant_get_input_size(
    quant_type: i32,
    num_blocks: i32,
) i32 {
    const block_bytes: i32 = switch (quant_type) {
        2 => @intCast(Q4_0_BLOCK_BYTES),
        8 => @intCast(Q8_0_BLOCK_BYTES),
        12 => @intCast(Q4_K_BLOCK_BYTES),
        else => 18,
    };
    return num_blocks * block_bytes;
}

// ============================================================================
// Quantization Type Enum (matches GGUF)
// ============================================================================

pub const QuantType = enum(i32) {
    Q4_0 = 2,
    Q8_0 = 8,
    Q4_K = 12,
    Q6_K = 14,
    F16 = 1,
    F32 = 0,

    pub fn blockSize(self: QuantType) usize {
        return switch (self) {
            .Q4_0 => Q4_0_BLOCK_SIZE,
            .Q8_0 => Q8_0_BLOCK_SIZE,
            .Q4_K => Q4_K_BLOCK_SIZE,
            .Q6_K => Q6_K_BLOCK_SIZE,
            .F16 => 1,
            .F32 => 1,
        };
    }

    pub fn blockBytes(self: QuantType) usize {
        return switch (self) {
            .Q4_0 => Q4_0_BLOCK_BYTES,
            .Q8_0 => Q8_0_BLOCK_BYTES,
            .Q4_K => Q4_K_BLOCK_BYTES,
            .Q6_K => Q6_K_BLOCK_BYTES,
            .F16 => 2,
            .F32 => 4,
        };
    }

    /// Convert from GGUF QuantizationType to dequant QuantType
    /// Returns null for unsupported quantization types (that don't have GPU dequant kernels)
    const gguf = @import("gguf_loader");
    pub fn fromGguf(quant_type: gguf.QuantizationType) ?QuantType {
        return switch (quant_type) {
            .Q4_0 => .Q4_0,
            .Q8_0 => .Q8_0,
            .Q4_K => .Q4_K,
            .Q6_K => .Q6_K,
            .F16 => .F16,
            .F32 => .F32,
            // These don't have GPU dequant kernels yet
            .Q4_1, .Q5_0, .Q5_1, .Q8_1, .Q2_K, .Q3_K, .Q5_K, .Q8_K => null,
        };
    }
};

// ============================================================================
// High-Level Dequantization Context
// ============================================================================

pub const DequantContext = struct {
    stream: ?*anyopaque,
    // Reusable GPU buffers for dequantized output (FP16)
    fp16_buffer: ?[*]f16,
    fp16_buffer_size: usize,
    // Reusable GPU buffer for quantized input (u8)
    input_buffer: ?[*]u8,
    input_buffer_size: usize,

    const Self = @This();

    pub fn init(stream: ?*anyopaque) Self {
        return Self{
            .stream = stream,
            .fp16_buffer = null,
            .fp16_buffer_size = 0,
            .input_buffer = null,
            .input_buffer_size = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.fp16_buffer) |buf| {
            _ = cuda.cudaFree(@ptrCast(buf));
            self.fp16_buffer = null;
            self.fp16_buffer_size = 0;
        }
        if (self.input_buffer) |buf| {
            _ = cuda.cudaFree(@ptrCast(buf));
            self.input_buffer = null;
            self.input_buffer_size = 0;
        }
    }

    /// Ensure FP16 output buffer is large enough
    pub fn ensureBuffer(self: *Self, num_elements: usize) !void {
        if (self.fp16_buffer_size >= num_elements) return;

        // Free old buffer if exists
        if (self.fp16_buffer) |buf| {
            _ = cuda.cudaFree(@ptrCast(buf));
        }

        // Allocate new buffer
        var ptr: *anyopaque = undefined;
        try cuda.checkCudaError(
            cuda.cudaMalloc(&ptr, num_elements * @sizeOf(f16)),
            "DequantContext FP16 buffer alloc",
        );
        self.fp16_buffer = @ptrCast(@alignCast(ptr));
        self.fp16_buffer_size = num_elements;
    }

    /// Ensure input buffer is large enough for quantized data
    pub fn ensureInputBuffer(self: *Self, num_bytes: usize) !void {
        if (self.input_buffer_size >= num_bytes) return;

        // Free old buffer if exists
        if (self.input_buffer) |buf| {
            _ = cuda.cudaFree(@ptrCast(buf));
        }

        // Allocate new buffer
        var ptr: *anyopaque = undefined;
        try cuda.checkCudaError(
            cuda.cudaMalloc(&ptr, num_bytes),
            "DequantContext input buffer alloc",
        );
        self.input_buffer = @ptrCast(@alignCast(ptr));
        self.input_buffer_size = num_bytes;
    }

    /// Copy quantized data from host to GPU and return GPU pointer
    fn copyInputToGpu(self: *Self, host_input: [*]const u8, num_bytes: usize) ![*]const u8 {
        try self.ensureInputBuffer(num_bytes);

        // Copy host -> device
        try cuda.checkCudaError(
            cuda.cudaMemcpy(
                @ptrCast(self.input_buffer.?),
                @ptrCast(host_input),
                num_bytes,
                cuda.cudaMemcpyHostToDevice,
            ),
            "DequantContext host->device copy",
        );

        return self.input_buffer.?;
    }

    /// Dequantize Q4_0 data to FP16 on GPU
    /// Input is HOST memory - will be copied to GPU first
    pub fn dequantQ4_0(self: *Self, host_input: [*]const u8, num_blocks: usize) ![*]f16 {
        const num_elements = num_blocks * Q4_0_BLOCK_SIZE;
        const input_bytes = num_blocks * Q4_0_BLOCK_BYTES;

        try self.ensureBuffer(num_elements);
        const gpu_input = try self.copyInputToGpu(host_input, input_bytes);

        const result = mojo_dequant_q4_0_fp16(
            gpu_input,
            self.fp16_buffer.?,
            @intCast(num_blocks),
            self.stream,
        );
        if (result != 0) return error.DequantKernelFailed;

        return self.fp16_buffer.?;
    }

    /// Dequantize Q8_0 data to FP16 on GPU
    /// Input is HOST memory - will be copied to GPU first
    pub fn dequantQ8_0(self: *Self, host_input: [*]const u8, num_blocks: usize) ![*]f16 {
        const num_elements = num_blocks * Q8_0_BLOCK_SIZE;
        const input_bytes = num_blocks * Q8_0_BLOCK_BYTES;

        try self.ensureBuffer(num_elements);
        const gpu_input = try self.copyInputToGpu(host_input, input_bytes);

        const result = mojo_dequant_q8_0_fp16(
            gpu_input,
            self.fp16_buffer.?,
            @intCast(num_blocks),
            self.stream,
        );
        if (result != 0) return error.DequantKernelFailed;

        return self.fp16_buffer.?;
    }

    /// Dequantize Q4_K data to FP16 on GPU
    /// Input is HOST memory - will be copied to GPU first
    pub fn dequantQ4_K(self: *Self, host_input: [*]const u8, num_blocks: usize) ![*]f16 {
        const num_elements = num_blocks * Q4_K_BLOCK_SIZE;
        const input_bytes = num_blocks * Q4_K_BLOCK_BYTES;

        log.debug("🔶 Q4_K dequant: blocks={} elements={} bytes={}", .{ num_blocks, num_elements, input_bytes });

        try self.ensureBuffer(num_elements);
        const gpu_input = try self.copyInputToGpu(host_input, input_bytes);

        log.debug("🔶 Q4_K calling kernel: gpu_input={*} fp16_buf={*}", .{ gpu_input, self.fp16_buffer.? });

        const result = mojo_dequant_q4_k_fp16(
            gpu_input,
            self.fp16_buffer.?,
            @intCast(num_blocks),
            self.stream,
        );

        log.debug("🔶 Q4_K kernel returned: {}", .{result});

        if (result != 0) {
            log.err("❌ Q4_K kernel failed with code: {}", .{result});
            return error.DequantKernelFailed;
        }

        return self.fp16_buffer.?;
    }

    /// Dequantize Q6_K data to FP16 on GPU
    /// Input is HOST memory - will be copied to GPU first
    pub fn dequantQ6_K(self: *Self, host_input: [*]const u8, num_blocks: usize) ![*]f16 {
        const num_elements = num_blocks * Q6_K_BLOCK_SIZE;
        const input_bytes = num_blocks * Q6_K_BLOCK_BYTES;

        log.debug("🔷 Q6_K dequant: blocks={} elements={} bytes={}", .{ num_blocks, num_elements, input_bytes });

        try self.ensureBuffer(num_elements);
        const gpu_input = try self.copyInputToGpu(host_input, input_bytes);

        log.debug("🔷 Q6_K calling kernel: gpu_input={*} fp16_buf={*}", .{ gpu_input, self.fp16_buffer.? });

        const result = mojo_dequant_q6_k_fp16(
            gpu_input,
            self.fp16_buffer.?,
            @intCast(num_blocks),
            self.stream,
        );

        log.debug("🔷 Q6_K kernel returned: {}", .{result});

        if (result != 0) {
            log.err("❌ Q6_K kernel failed with code: {}", .{result});
            return error.DequantKernelFailed;
        }

        return self.fp16_buffer.?;
    }

    /// Generic dequantization dispatcher based on QuantType
    /// Input is HOST memory - will be copied to GPU first
    pub fn dequant(self: *Self, host_input: [*]const u8, quant_type: QuantType, num_blocks: usize) ![*]f16 {
        return switch (quant_type) {
            .Q4_0 => self.dequantQ4_0(host_input, num_blocks),
            .Q8_0 => self.dequantQ8_0(host_input, num_blocks),
            .Q4_K => self.dequantQ4_K(host_input, num_blocks),
            .Q6_K => self.dequantQ6_K(host_input, num_blocks),
            else => error.UnsupportedQuantType,
        };
    }

    /// Calculate number of blocks for a given element count and quant type
    pub fn calculateNumBlocks(quant_type: QuantType, num_elements: usize) usize {
        const block_size = quant_type.blockSize();
        return (num_elements + block_size - 1) / block_size;
    }
};

