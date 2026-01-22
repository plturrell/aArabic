// GPU Weight Cache - Pre-loads and caches all model weights on GPU
// 
// This eliminates the per-matmul weight transfer bottleneck by:
// 1. Dequantizing all quantized weights to FP16 at model load time
// 2. Keeping the FP16 weights resident on GPU memory
// 3. Providing direct GPU pointers for cuBLAS matmul operations
//
// Memory usage for TinyLlama 1.1B (Q4_K): ~2.2GB FP16 on GPU

const std = @import("std");
const cuda = @import("cuda_bindings");
const GpuTensor = @import("gpu_tensor").GpuTensor;
const dequant = @import("dequant_bindings");
const gguf = @import("gguf_loader");

pub const GpuLayerWeights = struct {
    /// Normalization weights (F32 on GPU)
    attn_norm: GpuTensor, // [embed_dim] - for pre-attention RMSNorm
    ffn_norm: GpuTensor, // [embed_dim] - for pre-FFN RMSNorm

    /// Attention weights
    wq: GpuTensor, // [n_heads * head_dim, embed_dim]
    wk: GpuTensor, // [n_kv_heads * head_dim, embed_dim]
    wv: GpuTensor, // [n_kv_heads * head_dim, embed_dim]
    wo: GpuTensor, // [embed_dim, n_heads * head_dim]

    /// FFN weights
    w_gate: GpuTensor, // [hidden_dim, embed_dim]
    w_up: GpuTensor, // [hidden_dim, embed_dim]
    w_down: GpuTensor, // [embed_dim, hidden_dim]

    pub fn deinit(self: *GpuLayerWeights) void {
        self.attn_norm.deinit();
        self.ffn_norm.deinit();
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        self.w_gate.deinit();
        self.w_up.deinit();
        self.w_down.deinit();
    }

    pub fn memoryUsage(self: *const GpuLayerWeights) usize {
        return self.attn_norm.memoryUsage() + self.ffn_norm.memoryUsage() +
            self.wq.memoryUsage() + self.wk.memoryUsage() +
            self.wv.memoryUsage() + self.wo.memoryUsage() +
            self.w_gate.memoryUsage() + self.w_up.memoryUsage() +
            self.w_down.memoryUsage();
    }
};

pub const GpuWeightCache = struct {
    allocator: std.mem.Allocator,

    /// Token embedding (F16 on GPU)
    token_embedding: GpuTensor,

    /// Output normalization weight (F32 on GPU for RMSNorm)
    output_norm: GpuTensor,

    /// Output projection weight
    output_weight: GpuTensor,

    /// Per-layer weights
    layers: []GpuLayerWeights,

    /// Dequant context for GPU operations
    dequant_ctx: dequant.DequantContext,

    /// Stats
    total_gpu_memory: usize,
    n_layers: usize,

    /// RMSNorm epsilon (from model metadata)
    rms_norm_eps: f32,

    const Self = @This();

    /// Initialize GPU weight cache and load all weights from model
    pub fn init(
        allocator: std.mem.Allocator,
        model: *gguf.GGUFModel,
        n_layers: u32,
        vocab_size: u32,
        embed_dim: u32,
        hidden_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
    ) !Self {
        std.debug.print("\n🚀 Initializing GPU Weight Cache...\n", .{});

        var dequant_ctx = dequant.DequantContext.init(null);
        var total_memory: usize = 0;

        // Load token embeddings
        std.debug.print("   Loading token embeddings...\n", .{});
        const token_emb = try loadTensorToGpu(
            model,
            "token_embd.weight",
            @as(usize, vocab_size) * embed_dim,
            &dequant_ctx,
        );
        total_memory += token_emb.memoryUsage();

        // Load output norm (always F32)
        std.debug.print("   Loading output norm...\n", .{});
        const output_norm_tensor = try loadTensorToGpuF32(model, "output_norm.weight", embed_dim);
        total_memory += output_norm_tensor.memoryUsage();

        // Load output weight
        std.debug.print("   Loading output weight...\n", .{});
        const output_w = try loadTensorToGpu(
            model,
            "output.weight",
            @as(usize, vocab_size) * embed_dim,
            &dequant_ctx,
        );
        total_memory += output_w.memoryUsage();

        // Allocate layer weights array
        const layers = try allocator.alloc(GpuLayerWeights, n_layers);

        const head_dim = embed_dim / n_heads;

        // Load per-layer weights
        for (0..n_layers) |layer_idx| {
            std.debug.print("   Loading layer {}/{}...\r", .{ layer_idx + 1, n_layers });

            var name_buf: [128]u8 = undefined;

            // Load normalization weights (F32)
            layers[layer_idx].attn_norm = try loadTensorToGpuF32Fmt(model, "blk.{}.attn_norm.weight", layer_idx, embed_dim, &name_buf);
            layers[layer_idx].ffn_norm = try loadTensorToGpuF32Fmt(model, "blk.{}.ffn_norm.weight", layer_idx, embed_dim, &name_buf);

            // Attention weights
            const wq_name = std.fmt.bufPrint(&name_buf, "blk.{}.attn_q.weight", .{layer_idx}) catch unreachable;
            layers[layer_idx].wq = try loadTensorToGpu(
                model,
                wq_name,
                @as(usize, n_heads) * head_dim * embed_dim,
                &dequant_ctx,
            );

            // More weights loaded in continuation...
            layers[layer_idx].wk = try loadTensorToGpuFmt(model, "blk.{}.attn_k.weight", layer_idx, @as(usize, n_kv_heads) * head_dim * embed_dim, &dequant_ctx, &name_buf);
            layers[layer_idx].wv = try loadTensorToGpuFmt(model, "blk.{}.attn_v.weight", layer_idx, @as(usize, n_kv_heads) * head_dim * embed_dim, &dequant_ctx, &name_buf);
            layers[layer_idx].wo = try loadTensorToGpuFmt(model, "blk.{}.attn_output.weight", layer_idx, @as(usize, embed_dim) * n_heads * head_dim, &dequant_ctx, &name_buf);

            // FFN weights
            layers[layer_idx].w_gate = try loadTensorToGpuFmt(model, "blk.{}.ffn_gate.weight", layer_idx, @as(usize, hidden_dim) * embed_dim, &dequant_ctx, &name_buf);
            layers[layer_idx].w_up = try loadTensorToGpuFmt(model, "blk.{}.ffn_up.weight", layer_idx, @as(usize, hidden_dim) * embed_dim, &dequant_ctx, &name_buf);
            layers[layer_idx].w_down = try loadTensorToGpuFmt(model, "blk.{}.ffn_down.weight", layer_idx, @as(usize, embed_dim) * hidden_dim, &dequant_ctx, &name_buf);

            total_memory += layers[layer_idx].memoryUsage();
        }

        // Get RMS norm epsilon from model metadata
        const rms_eps = model.metadata.rms_norm_eps;

        std.debug.print("\n   ✅ GPU Weight Cache initialized: {d:.2} GB\n", .{
            @as(f64, @floatFromInt(total_memory)) / (1024.0 * 1024.0 * 1024.0),
        });

        return Self{
            .allocator = allocator,
            .token_embedding = token_emb,
            .output_norm = output_norm_tensor,
            .output_weight = output_w,
            .layers = layers,
            .dequant_ctx = dequant_ctx,
            .total_gpu_memory = total_memory,
            .n_layers = n_layers,
            .rms_norm_eps = rms_eps,
        };
    }

    pub fn deinit(self: *Self) void {
        self.token_embedding.deinit();
        self.output_norm.deinit();
        self.output_weight.deinit();
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.dequant_ctx.deinit();
    }

    /// Get layer weights for GPU-to-GPU matmul
    pub fn getLayer(self: *const Self, layer_idx: usize) *const GpuLayerWeights {
        return &self.layers[layer_idx];
    }
};

// Helper functions for loading tensors

fn loadTensorToGpu(
    model: *gguf.GGUFModel,
    name: []const u8,
    num_elements: usize,
    dequant_ctx: *dequant.DequantContext,
) !GpuTensor {
    // Find tensor index
    const tensor_idx = model.findTensor(name) orelse {
        std.debug.print("ERROR: Tensor not found: {s}\n", .{name});
        return error.TensorNotFound;
    };

    const tensor_info = &model.tensors[tensor_idx];
    const quant_type = dequant.QuantType.fromGguf(tensor_info.quant_type);

    // Get tensor data from file
    const data_bytes = try model.getTensorData(tensor_idx);
    defer model.allocator.free(data_bytes);

    // Allocate GPU tensor
    var gpu_tensor = try GpuTensor.alloc(num_elements);
    errdefer gpu_tensor.deinit();

    if (quant_type) |qt| {
        // Quantized tensor - dequantize to FP16 on GPU
        if (qt == .F16 or qt == .F32) {
            // Already float - just copy
            if (qt == .F32) {
                const f32_data: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data_bytes));
                try gpu_tensor.copyFromHostF32(f32_data);
            } else {
                const f16_data: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, data_bytes));
                try gpu_tensor.copyFromHostF16(f16_data);
            }
        } else {
            // Quantized - use GPU dequant
            const num_blocks = dequant.DequantContext.calculateNumBlocks(qt, num_elements);

            // Ensure dequant context has enough buffer space
            try dequant_ctx.ensureBuffer(num_elements);
            try dequant_ctx.ensureInputBuffer(data_bytes.len);

            // Dequant on GPU
            const fp16_ptr = dequant_ctx.dequant(data_bytes.ptr, qt, num_blocks) catch {
                // Fallback to CPU dequant
                return loadTensorToGpuCpuFallback(model.allocator, data_bytes, tensor_info.quant_type, num_elements);
            };

            // Copy from dequant buffer to our tensor
            const result = cuda.cudaMemcpy(
                gpu_tensor.devicePtr(),
                @ptrCast(fp16_ptr),
                num_elements * @sizeOf(f16),
                cuda.cudaMemcpyDeviceToDevice,
            );
            if (result != 0) return error.CudaCopyFailed;
        }
    } else {
        return error.UnsupportedQuantType;
    }

    return gpu_tensor;
}

fn loadTensorToGpuFmt(
    model: *gguf.GGUFModel,
    comptime fmt: []const u8,
    layer_idx: usize,
    num_elements: usize,
    dequant_ctx: *dequant.DequantContext,
    buf: *[128]u8,
) !GpuTensor {
    const name = std.fmt.bufPrint(buf, fmt, .{layer_idx}) catch unreachable;
    return loadTensorToGpu(model, name, num_elements, dequant_ctx);
}

fn loadTensorToGpuCpuFallback(
    allocator: std.mem.Allocator,
    data_bytes: []const u8,
    quant_type: gguf.QuantizationType,
    num_elements: usize,
) !GpuTensor {
    // CPU dequant fallback
    const f32_buf = try allocator.alloc(f32, num_elements);
    defer allocator.free(f32_buf);

    const matrix_ops = @import("matrix_ops");
    matrix_ops.dequantize(f32_buf, data_bytes, quant_type, num_elements);

    var gpu_tensor = try GpuTensor.alloc(num_elements);
    try gpu_tensor.copyFromHostF32(f32_buf);
    return gpu_tensor;
}

/// Load F32 tensor directly to GPU (for normalization weights)
fn loadTensorToGpuF32(
    model: *gguf.GGUFModel,
    name: []const u8,
    num_elements: usize,
) !GpuTensor {
    // Find tensor index
    const tensor_idx = model.findTensor(name) orelse {
        std.debug.print("ERROR: Tensor not found: {s}\n", .{name});
        return error.TensorNotFound;
    };

    // Get tensor data from file
    const data_bytes = try model.getTensorData(tensor_idx);
    defer model.allocator.free(data_bytes);

    // Norms are always F32 in GGUF
    const f32_data: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data_bytes));

    // Allocate GPU tensor and copy data
    var gpu_tensor = try GpuTensor.alloc(num_elements);
    errdefer gpu_tensor.deinit();
    try gpu_tensor.copyFromHostF32(f32_data);

    return gpu_tensor;
}

fn loadTensorToGpuF32Fmt(
    model: *gguf.GGUFModel,
    comptime fmt: []const u8,
    layer_idx: usize,
    num_elements: usize,
    buf: *[128]u8,
) !GpuTensor {
    const name = std.fmt.bufPrint(buf, fmt, .{layer_idx}) catch unreachable;
    return loadTensorToGpuF32(model, name, num_elements);
}
