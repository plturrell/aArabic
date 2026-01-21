// GPU-Native Inference Engine
// 
// This module provides a fully GPU-resident inference path that:
// 1. Keeps all activations on GPU between layers
// 2. Uses pre-cached FP16 weights (no per-matmul transfers)
// 3. Only transfers input embeddings to GPU and logits back to CPU
// 4. Uses cuBLAS for all matmul operations
//
// Target: 500+ tokens/second on Tesla T4

const std = @import("std");
const cuda = @import("cuda_bindings");
const cublas = @import("cublas_bindings");
const GpuTensor = @import("gpu_tensor").GpuTensor;
const GpuWeightCache = @import("gpu_weight_cache").GpuWeightCache;

pub const GpuInference = struct {
    allocator: std.mem.Allocator,
    
    // cuBLAS handle
    cublas_handle: cublas.cublasHandle_t,
    
    // Model dimensions
    embed_dim: u32,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    n_layers: u32,
    
    // Pre-allocated GPU buffers for activations (reused each forward pass)
    hidden_state: GpuTensor,      // [embed_dim] - current hidden state
    residual: GpuTensor,          // [embed_dim] - residual connection
    attn_out: GpuTensor,          // [embed_dim] - attention output
    q_proj: GpuTensor,            // [n_heads * head_dim]
    k_proj: GpuTensor,            // [n_kv_heads * head_dim]  
    v_proj: GpuTensor,            // [n_kv_heads * head_dim]
    ffn_gate: GpuTensor,          // [hidden_dim]
    ffn_up: GpuTensor,            // [hidden_dim]
    ffn_out: GpuTensor,           // [embed_dim]
    logits: GpuTensor,            // [vocab_size]
    
    // Scratch buffers
    scratch1: GpuTensor,
    scratch2: GpuTensor,
    
    const Self = @This();
    
    pub fn init(
        allocator: std.mem.Allocator,
        embed_dim: u32,
        hidden_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        vocab_size: u32,
        n_layers: u32,
    ) !Self {
        std.debug.print("\n🔧 Initializing GPU Inference Engine...\n", .{});
        
        // Create cuBLAS handle
        var handle: cublas.cublasHandle_t = undefined;
        const status = cublas.cublasCreate_v2(&handle);
        if (status != 0) {
            std.debug.print("cuBLAS create failed: {}\n", .{status});
            return error.CublasInitFailed;
        }
        
        const head_dim = embed_dim / n_heads;
        
        // Allocate activation buffers
        const hidden_state = try GpuTensor.alloc(embed_dim);
        const residual = try GpuTensor.alloc(embed_dim);
        const attn_out = try GpuTensor.alloc(embed_dim);
        const q_proj = try GpuTensor.alloc(n_heads * head_dim);
        const k_proj = try GpuTensor.alloc(n_kv_heads * head_dim);
        const v_proj = try GpuTensor.alloc(n_kv_heads * head_dim);
        const ffn_gate = try GpuTensor.alloc(hidden_dim);
        const ffn_up = try GpuTensor.alloc(hidden_dim);
        const ffn_out = try GpuTensor.alloc(embed_dim);
        const logits = try GpuTensor.alloc(vocab_size);
        const scratch1 = try GpuTensor.alloc(@max(hidden_dim, vocab_size));
        const scratch2 = try GpuTensor.alloc(@max(hidden_dim, vocab_size));
        
        const total_mem = hidden_state.memoryUsage() + residual.memoryUsage() +
            attn_out.memoryUsage() + q_proj.memoryUsage() + k_proj.memoryUsage() +
            v_proj.memoryUsage() + ffn_gate.memoryUsage() + ffn_up.memoryUsage() +
            ffn_out.memoryUsage() + logits.memoryUsage() + scratch1.memoryUsage() +
            scratch2.memoryUsage();
        
        std.debug.print("   ✅ GPU activation buffers: {d:.2} MB\n", .{
            @as(f64, @floatFromInt(total_mem)) / (1024.0 * 1024.0),
        });
        
        return Self{
            .allocator = allocator,
            .cublas_handle = handle,
            .embed_dim = embed_dim,
            .hidden_dim = hidden_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .vocab_size = vocab_size,
            .n_layers = n_layers,
            .hidden_state = hidden_state,
            .residual = residual,
            .attn_out = attn_out,
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .ffn_gate = ffn_gate,
            .ffn_up = ffn_up,
            .ffn_out = ffn_out,
            .logits = logits,
            .scratch1 = scratch1,
            .scratch2 = scratch2,
        };
    }
    
    pub fn deinit(self: *Self) void {
        _ = cublas.cublasDestroy_v2(self.cublas_handle);
        self.hidden_state.deinit();
        self.residual.deinit();
        self.attn_out.deinit();
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.ffn_gate.deinit();
        self.ffn_up.deinit();
        self.ffn_out.deinit();
        self.logits.deinit();
        self.scratch1.deinit();
        self.scratch2.deinit();
    }

    /// GPU-to-GPU matmul using cuBLAS with FP16 Tensor Cores
    /// C = A @ B where A[m,k], B[k,n], C[m,n] - all FP16 on GPU
    pub fn gpuMatmul(
        self: *Self,
        c: *GpuTensor,
        a: *const GpuTensor,
        b: *const GpuTensor,
        m: u32,
        n: u32,
        k: u32,
    ) !void {
        // cuBLAS uses column-major, so we compute C^T = B^T @ A^T
        // which gives us row-major C = A @ B
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;

        const status = cublas.cublasGemmEx(
            self.cublas_handle,
            cublas.CUBLAS_OP_N, // B not transposed
            cublas.CUBLAS_OP_N, // A not transposed
            @intCast(n), // rows of B^T = cols of B = n
            @intCast(m), // cols of A^T = rows of A = m
            @intCast(k), // shared dim
            &alpha,
            b.devicePtr(), // B
            cublas.CUDA_R_16F,
            @intCast(n), // ldb
            a.devicePtr(), // A
            cublas.CUDA_R_16F,
            @intCast(k), // lda
            &beta,
            c.devicePtr(), // C
            cublas.CUDA_R_16F,
            @intCast(n), // ldc
            cublas.CUBLAS_COMPUTE_32F, // Compute in FP32
            cublas.CUBLAS_GEMM_DEFAULT_TENSOR_OP, // Use Tensor Cores
        );

        if (status != 0) {
            std.debug.print("cuBLAS GemmEx failed: {}\n", .{status});
            return error.CublasGemmFailed;
        }
    }

    /// Forward pass for a single token - all on GPU
    /// Input: token embedding loaded into hidden_state
    /// Output: logits tensor
    pub fn forward(
        self: *Self,
        weights: *const GpuWeightCache,
        _: u32, // position (for RoPE - TODO)
    ) !*GpuTensor {
        // Process each transformer layer
        for (0..self.n_layers) |layer_idx| {
            const layer = weights.getLayer(layer_idx);

            // Save residual
            try self.copyTensor(&self.residual, &self.hidden_state);

            // TODO: RMSNorm on GPU (currently skipped)

            // Attention projections (GPU-to-GPU matmul)
            // Q = hidden @ Wq
            try self.gpuMatmul(&self.q_proj, &self.hidden_state, &layer.wq, 1, self.n_heads * self.head_dim, self.embed_dim);
            // K = hidden @ Wk
            try self.gpuMatmul(&self.k_proj, &self.hidden_state, &layer.wk, 1, self.n_kv_heads * self.head_dim, self.embed_dim);
            // V = hidden @ Wv
            try self.gpuMatmul(&self.v_proj, &self.hidden_state, &layer.wv, 1, self.n_kv_heads * self.head_dim, self.embed_dim);

            // TODO: RoPE on GPU
            // TODO: Attention score computation on GPU
            // TODO: KV cache update on GPU

            // For now: output projection (simplified - no actual attention)
            try self.gpuMatmul(&self.attn_out, &self.v_proj, &layer.wo, 1, self.embed_dim, self.n_kv_heads * self.head_dim);

            // Add residual
            try self.addTensors(&self.hidden_state, &self.attn_out, &self.residual);

            // FFN
            try self.copyTensor(&self.residual, &self.hidden_state);

            // TODO: RMSNorm on GPU

            // gate = hidden @ W_gate
            try self.gpuMatmul(&self.ffn_gate, &self.hidden_state, &layer.w_gate, 1, self.hidden_dim, self.embed_dim);
            // up = hidden @ W_up
            try self.gpuMatmul(&self.ffn_up, &self.hidden_state, &layer.w_up, 1, self.hidden_dim, self.embed_dim);

            // TODO: SiLU and elementwise mul on GPU

            // down = (gate * up) @ W_down
            try self.gpuMatmul(&self.ffn_out, &self.ffn_gate, &layer.w_down, 1, self.embed_dim, self.hidden_dim);

            // Add residual
            try self.addTensors(&self.hidden_state, &self.ffn_out, &self.residual);
        }

        // Final output projection
        try self.gpuMatmul(&self.logits, &self.hidden_state, &weights.output_weight, 1, self.vocab_size, self.embed_dim);

        return &self.logits;
    }

    /// Copy tensor data (GPU-to-GPU)
    fn copyTensor(self: *Self, dst: *GpuTensor, src: *const GpuTensor) !void {
        _ = self;
        const result = cuda.cudaMemcpy(
            dst.devicePtr(),
            src.devicePtr(),
            @min(dst.byte_size, src.byte_size),
            cuda.cudaMemcpyDeviceToDevice,
        );
        if (result != 0) return error.CudaCopyFailed;
    }

    /// Add two tensors on GPU (TODO: implement as CUDA kernel)
    fn addTensors(self: *Self, out: *GpuTensor, a: *const GpuTensor, b: *const GpuTensor) !void {
        _ = self;
        // For now, just copy a (proper implementation needs CUDA kernel)
        const result = cuda.cudaMemcpy(
            out.devicePtr(),
            a.devicePtr(),
            @min(out.byte_size, a.byte_size),
            cuda.cudaMemcpyDeviceToDevice,
        );
        _ = b;
        if (result != 0) return error.CudaCopyFailed;
    }

    /// Load token embedding to GPU hidden state
    pub fn loadEmbedding(self: *Self, weights: *const GpuWeightCache, token_id: u32) !void {
        // Copy one row from token_embedding to hidden_state
        const offset = @as(usize, token_id) * self.embed_dim * @sizeOf(f16);
        const src_ptr: [*]u8 = @ptrCast(weights.token_embedding.ptr);

        const result = cuda.cudaMemcpy(
            self.hidden_state.devicePtr(),
            @ptrCast(src_ptr + offset),
            self.embed_dim * @sizeOf(f16),
            cuda.cudaMemcpyDeviceToDevice,
        );
        if (result != 0) return error.CudaCopyFailed;
    }

    /// Get logits back to CPU
    pub fn getLogits(self: *Self, host_buffer: []f32) !void {
        try self.logits.copyToHostF32(host_buffer);
    }
};

