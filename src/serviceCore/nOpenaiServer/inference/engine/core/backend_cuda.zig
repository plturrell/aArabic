// CUDA Compute Backend for T4 GPU
// Implements the ComputeBackend interface for NVIDIA GPUs with Tensor Core support
//
// Target: NVIDIA Tesla T4 (16GB VRAM, Compute 7.5, 320 Tensor Cores)
// Features: FP16 mixed precision, cuBLAS integration, async streams

const std = @import("std");
const compute = @import("compute");
const gguf = @import("gguf_loader");
const cuda_bindings = @import("cuda_bindings");
const cuda_memory = @import("cuda_memory");
const cuda_streams = @import("cuda_streams");
const cuda_context = @import("cuda_context");
const cublas = @import("cublas_bindings");
const dequant = @import("dequant_bindings");

const log = std.log.scoped(.backend_cuda);

pub const CudaBackend = struct {
    allocator: std.mem.Allocator,
    context: *cuda_context.CudaContext,
    stream: cuda_streams.CudaStream,
    cublas_ctx: cublas.CublasContext,
    device_id: i32,
    has_tensor_cores: bool,
    fp16_supported: bool,
    total_memory: usize,
    free_memory: usize,
    // GPU memory buffers for matmul (reused to avoid allocation overhead)
    gpu_buffer_a: ?[]u8,
    gpu_buffer_b: ?[]u8,
    gpu_buffer_c: ?[]u8,
    buffer_capacity: usize,
    // GPU dequantization context for Tensor Core FP16 path
    dequant_ctx: dequant.DequantContext,

    pub const DeviceInfo = struct {
        name: []const u8,
        compute_capability: struct { major: i32, minor: i32 },
        total_memory_mb: u32,
        tensor_cores: bool,
        is_t4: bool,
    };

    pub fn init(allocator: std.mem.Allocator) !compute.ComputeBackend {
        log.info("⚡ Initializing CUDA Backend (NVIDIA GPU)...", .{});

        // Initialize CUDA context
        var device_count: c_int = 0;
        const count_result = cuda_bindings.cudaGetDeviceCount(&device_count);
        if (count_result != cuda_bindings.cudaSuccess or device_count == 0) {
            return error.NoCudaDevices;
        }

        // Use first CUDA device (can be made configurable)
        const device_id: i32 = 0;
        const set_result = cuda_bindings.cudaSetDevice(device_id);
        if (set_result != cuda_bindings.cudaSuccess) {
            return error.CudaSetDeviceFailed;
        }

        // Get device properties
        var props: cuda_bindings.CudaDeviceProp = undefined;
        const props_result = cuda_bindings.cudaGetDeviceProperties(&props, device_id);
        if (props_result != cuda_bindings.cudaSuccess) {
            return error.CudaGetPropertiesFailed;
        }

        // Check for Tensor Core support (Compute 7.0+)
        const has_tensor_cores = props.major >= 7;
        const fp16_supported = props.major >= 6;

        // Check if this is a T4 GPU
        const device_name = std.mem.sliceTo(&props.name, 0);
        const is_t4 = std.mem.indexOf(u8, device_name, "T4") != null;

        if (is_t4) {
            log.info("🎯 Detected NVIDIA Tesla T4 - Enabling T4 optimizations", .{});
        }

        log.info("GPU: {s}, Compute: {d}.{d}, VRAM: {d}MB, Tensor Cores: {}", .{
            device_name,
            props.major,
            props.minor,
            props.totalGlobalMem / (1024 * 1024),
            has_tensor_cores,
        });

        // Create CUDA context (init returns a pointer)
        const context = try cuda_context.CudaContext.init(allocator, device_id);

        // Create default stream
        const stream = try cuda_streams.CudaStream.init(allocator);

        // Initialize cuBLAS with Tensor Core support
        log.info("⚡ Initializing cuBLAS (Tensor Cores: {})...", .{has_tensor_cores});
        var cublas_ctx = try cublas.CublasContext.init(has_tensor_cores);
        try cublas_ctx.setStream(stream.handle);

        // Initialize GPU dequantization context for Tensor Core path
        const dequant_ctx = dequant.DequantContext.init(stream.handle);

        const self = try allocator.create(CudaBackend);
        self.* = .{
            .allocator = allocator,
            .context = context,
            .stream = stream,
            .cublas_ctx = cublas_ctx,
            .device_id = device_id,
            .has_tensor_cores = has_tensor_cores,
            .fp16_supported = fp16_supported,
            .total_memory = props.totalGlobalMem,
            .free_memory = props.totalGlobalMem,
            .gpu_buffer_a = null,
            .gpu_buffer_b = null,
            .gpu_buffer_c = null,
            .buffer_capacity = 0,
            .dequant_ctx = dequant_ctx,
        };

        log.info("✅ CUDA Backend initialized with cuBLAS + GPU dequant (using real GPU compute)", .{});

        return compute.ComputeBackend{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    fn deinit(ctx: *anyopaque) void {
        const self: *CudaBackend = @ptrCast(@alignCast(ctx));

        // Free GPU buffers
        if (self.gpu_buffer_a) |buf| cuda_memory.freeDevice(buf);
        if (self.gpu_buffer_b) |buf| cuda_memory.freeDevice(buf);
        if (self.gpu_buffer_c) |buf| cuda_memory.freeDevice(buf);

        // Cleanup dequant context
        self.dequant_ctx.deinit();

        // Cleanup cuBLAS
        self.cublas_ctx.deinit();

        self.stream.deinit();
        self.context.deinit();
        self.allocator.destroy(self.context);
        self.allocator.destroy(self);
        log.info("CUDA Backend deinitialized", .{});
    }

    fn alloc(ctx: *anyopaque, size: usize) ![]u8 {
        const self: *CudaBackend = @ptrCast(@alignCast(ctx));
        return cuda_memory.allocDevice(self.allocator, size);
    }

    fn free(ctx: *anyopaque, ptr: []u8) void {
        _ = ctx;
        cuda_memory.freeDevice(ptr);
    }

    fn copyToDevice(ctx: *anyopaque, dest: []u8, src: []const u8) !void {
        _ = ctx;
        try cuda_memory.copyHostToDevice(dest, src);
    }

    fn copyFromDevice(ctx: *anyopaque, dest: []u8, src: []const u8) !void {
        _ = ctx;
        try cuda_memory.copyDeviceToHost(dest, src);
    }

    fn matmul(
        ctx: *anyopaque,
        c: []f32,
        a_data: []const u8,
        a_type: gguf.QuantizationType,
        b: []const f32,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        const self: *CudaBackend = @ptrCast(@alignCast(ctx));

        // Decide whether to use Tensor Cores (FP16) or standard FP32 path
        // Tensor Cores are ~8x faster but require FP16 and work best with aligned dimensions
        const use_tensor_cores = self.has_tensor_cores and
            self.fp16_supported and
            isTensorCoreOptimal(m, n, k);

        // For quantized weights, we need to dequantize first
        // Use GPU dequantization for supported quant types (Q4_0, Q8_0, Q4_K)
        if (a_type != .F32 and a_type != .F16) {
            // Check if we can use GPU dequantization (Tensor Core path)
            const quant_type = dequant.QuantType.fromGguf(a_type);
            if (use_tensor_cores and quant_type != null) {
                // GPU dequantization + FP16 Tensor Core matmul
                // This is the fastest path: dequant on GPU → FP16 → Tensor Core GEMM
                const num_elements = m * k;
                const num_blocks = dequant.DequantContext.calculateNumBlocks(quant_type.?, num_elements);
                const a_fp16 = try self.dequant_ctx.dequant(a_data.ptr, quant_type.?, num_blocks);

                // Convert B from FP32 to FP16 for Tensor Core path
                const b_fp16 = try self.allocator.alloc(f16, k * n);
                defer self.allocator.free(b_fp16);
                for (b, 0..) |val, i| {
                    b_fp16[i] = @floatCast(val);
                }

                // Execute FP16 Tensor Core GEMM
                try self.gpuMatmulFp16TensorCoreRaw(c, a_fp16, b_fp16.ptr, m, n, k);
            } else {
                // Fallback: dequantize on CPU, then GPU matmul
                const a_fp32 = try self.allocator.alloc(f32, m * k);
                defer self.allocator.free(a_fp32);

                // Dequantize weights on CPU
                const matrix_ops = @import("matrix_ops");
                matrix_ops.dequantize(a_fp32, a_data, a_type, m * k);

                // Use Tensor Core path if available and optimal
                if (use_tensor_cores) {
                    try self.gpuMatmulFp16TensorCore(c, a_fp32, b, m, n, k);
                } else {
                    try self.gpuMatmulFp32(c, a_fp32, b, m, n, k);
                }
            }
        } else {
            // FP32/FP16 path: direct GPU matmul
            const a_fp32 = std.mem.bytesAsSlice(f32, @constCast(a_data));
            if (use_tensor_cores) {
                try self.gpuMatmulFp16TensorCore(c, a_fp32, b, m, n, k);
            } else {
                try self.gpuMatmulFp32(c, a_fp32, b, m, n, k);
            }
        }
    }

    /// GPU matrix multiplication using cuBLAS
    /// C[m,n] = A[m,k] @ B[k,n]
    fn gpuMatmulFp32(self: *CudaBackend, c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) !void {
        const size_a = m * k * @sizeOf(f32);
        const size_b = k * n * @sizeOf(f32);
        const size_c = m * n * @sizeOf(f32);

        // Ensure GPU buffers are large enough
        try self.ensureBufferCapacity(size_a, size_b, size_c);

        const gpu_a = self.gpu_buffer_a.?;
        const gpu_b = self.gpu_buffer_b.?;
        const gpu_c = self.gpu_buffer_c.?;

        // Copy inputs to GPU
        try cuda_memory.copyHostToDevice(gpu_a[0..size_a], std.mem.sliceAsBytes(a));
        try cuda_memory.copyHostToDevice(gpu_b[0..size_b], std.mem.sliceAsBytes(b));

        // Execute cuBLAS SGEMM on GPU
        const a_ptr: [*]const f32 = @ptrCast(@alignCast(gpu_a.ptr));
        const b_ptr: [*]const f32 = @ptrCast(@alignCast(gpu_b.ptr));
        const c_ptr: [*]f32 = @ptrCast(@alignCast(gpu_c.ptr));

        try self.cublas_ctx.sgemm(c_ptr, a_ptr, b_ptr, m, n, k);

        // Synchronize and copy result back
        _ = cuda_bindings.cudaDeviceSynchronize();
        try cuda_memory.copyDeviceToHost(std.mem.sliceAsBytes(c), gpu_c[0..size_c]);
    }

    /// GPU matrix multiplication using FP16 Tensor Cores via cublasGemmEx
    /// C[m,n] = A[m,k] @ B[k,n] with FP16 compute and FP32 accumulation
    /// This provides ~8x speedup over FP32 SGEMM on T4 (65 TFLOPS vs 8.1 TFLOPS)
    fn gpuMatmulFp16TensorCore(self: *CudaBackend, c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) !void {
        // FP16 sizes (half the FP32 sizes)
        const size_a_fp16 = m * k * @sizeOf(f16);
        const size_b_fp16 = k * n * @sizeOf(f16);
        const size_c_fp16 = m * n * @sizeOf(f16);
        const size_c_fp32 = m * n * @sizeOf(f32);

        // Ensure GPU buffers are large enough (use max of FP16 and FP32 sizes)
        try self.ensureBufferCapacity(size_a_fp16, size_b_fp16, @max(size_c_fp16, size_c_fp32));

        const gpu_a = self.gpu_buffer_a.?;
        const gpu_b = self.gpu_buffer_b.?;
        const gpu_c = self.gpu_buffer_c.?;

        // Convert FP32 inputs to FP16 on CPU and copy to GPU
        // TODO: Do this conversion on GPU for better performance
        const a_fp16 = try self.allocator.alloc(f16, m * k);
        defer self.allocator.free(a_fp16);
        const b_fp16 = try self.allocator.alloc(f16, k * n);
        defer self.allocator.free(b_fp16);

        // Convert to FP16
        for (a, 0..) |val, i| {
            a_fp16[i] = @floatCast(val);
        }
        for (b, 0..) |val, i| {
            b_fp16[i] = @floatCast(val);
        }

        // Copy FP16 inputs to GPU
        try cuda_memory.copyHostToDevice(gpu_a[0..size_a_fp16], std.mem.sliceAsBytes(a_fp16));
        try cuda_memory.copyHostToDevice(gpu_b[0..size_b_fp16], std.mem.sliceAsBytes(b_fp16));

        // Execute cuBLAS GemmEx with Tensor Cores (FP16 input, FP32 output)
        try self.cublas_ctx.gemmEx_fp16(
            @ptrCast(gpu_c.ptr),
            @ptrCast(gpu_a.ptr),
            @ptrCast(gpu_b.ptr),
            m,
            n,
            k,
            cublas.CUDA_R_32F, // Output in FP32 for precision
        );

        // Synchronize and copy FP32 result back
        _ = cuda_bindings.cudaDeviceSynchronize();
        try cuda_memory.copyDeviceToHost(std.mem.sliceAsBytes(c), gpu_c[0..size_c_fp32]);
    }

    /// GPU matrix multiplication using FP16 Tensor Cores with raw FP16 pointers
    /// Used when data is already dequantized to FP16 on GPU (no CPU conversion needed)
    /// This is the fastest path: GPU dequant → FP16 already on GPU → Tensor Core GEMM
    fn gpuMatmulFp16TensorCoreRaw(self: *CudaBackend, c: []f32, a_fp16_gpu: [*]const f16, b_fp16: [*]const f16, m: usize, n: usize, k: usize) !void {
        const size_b_fp16 = k * n * @sizeOf(f16);
        const size_c_fp32 = m * n * @sizeOf(f32);

        // Ensure GPU buffers are large enough for B and C
        try self.ensureBufferCapacity(0, size_b_fp16, size_c_fp32);

        const gpu_b = self.gpu_buffer_b.?;
        const gpu_c = self.gpu_buffer_c.?;

        // Copy B (FP16) to GPU - A is already on GPU from dequant kernel
        try cuda_memory.copyHostToDevice(gpu_b[0..size_b_fp16], @as([*]const u8, @ptrCast(b_fp16))[0..size_b_fp16]);

        // Execute cuBLAS GemmEx with Tensor Cores
        // A is already on GPU (from dequant), B copied, C is output
        try self.cublas_ctx.gemmEx_fp16(
            @ptrCast(gpu_c.ptr), // C output (FP32)
            a_fp16_gpu, // A already on GPU from dequant
            @ptrCast(gpu_b.ptr), // B on GPU
            m,
            n,
            k,
            cublas.CUDA_R_32F, // Output in FP32 for precision
        );

        // Synchronize and copy FP32 result back
        _ = cuda_bindings.cudaDeviceSynchronize();
        try cuda_memory.copyDeviceToHost(std.mem.sliceAsBytes(c), gpu_c[0..size_c_fp32]);
    }

    /// Check if dimensions are optimal for Tensor Core usage
    /// Tensor Cores work best with dimensions that are multiples of 8 (FP16) or 16 (INT8)
    fn isTensorCoreOptimal(m: usize, n: usize, k: usize) bool {
        return (m % 8 == 0) and (n % 8 == 0) and (k % 8 == 0);
    }

    /// Ensure GPU buffers are large enough for the given matrix sizes
    fn ensureBufferCapacity(self: *CudaBackend, size_a: usize, size_b: usize, size_c: usize) !void {
        const required = @max(size_a, @max(size_b, size_c));
        if (required > self.buffer_capacity) {
            // Free old buffers
            if (self.gpu_buffer_a) |buf| cuda_memory.freeDevice(buf);
            if (self.gpu_buffer_b) |buf| cuda_memory.freeDevice(buf);
            if (self.gpu_buffer_c) |buf| cuda_memory.freeDevice(buf);

            // Allocate new buffers with some headroom
            const new_capacity = required * 2;
            self.gpu_buffer_a = try cuda_memory.allocDevice(self.allocator, new_capacity);
            self.gpu_buffer_b = try cuda_memory.allocDevice(self.allocator, new_capacity);
            self.gpu_buffer_c = try cuda_memory.allocDevice(self.allocator, new_capacity);
            self.buffer_capacity = new_capacity;

            log.debug("Resized GPU buffers to {} bytes", .{new_capacity});
        }
    }

    fn rms_norm(ctx: *anyopaque, out: []f32, in: []const f32, w: []const f32, eps: f32) void {
        _ = ctx;
        const matrix_ops = @import("matrix_ops");
        matrix_ops.rms_norm(out, in, w, eps);
    }

    fn softmax(ctx: *anyopaque, out: []f32, in: []const f32) void {
        _ = ctx;
        const matrix_ops = @import("matrix_ops");
        matrix_ops.softmax(out, in);
    }

    fn rope(ctx: *anyopaque, out: []f32, in: []const f32, pos: u32, _freqs: []const f32, dim: u32) void {
        _ = ctx;
        _ = _freqs;
        const matrix_ops = @import("matrix_ops");
        matrix_ops.apply_rope(out, @constCast(in), pos, dim * 2, 10000.0);
    }

    /// VTable for ComputeBackend interface
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
