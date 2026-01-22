// GPU-Native Inference Engine
//
// This module provides a fully GPU-resident inference path that:
// 1. Keeps all activations on GPU between layers
// 2. Uses pre-cached FP16 weights (no per-matmul transfers)
// 3. Only transfers input embeddings to GPU and logits back to CPU
// 4. Uses cuBLAS for all matmul operations
// 5. Uses CUDA streams for async operation and CUDA Graphs to reduce launch overhead
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

    // CUDA stream for async operations
    stream: ?*anyopaque,

    // CUDA Graph for replaying forward pass (reduces launch overhead)
    graph: ?cuda.cudaGraph_t,
    graph_exec: ?cuda.cudaGraphExec_t,
    graph_captured: bool,

    // Model dimensions
    embed_dim: u32,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    n_layers: u32,

    // Batch size for parallel token processing
    batch_size: u32,

    // Pre-allocated GPU buffers for activations (reused each forward pass)
    // All buffers are sized for batch_size tokens
    hidden_state: GpuTensor,      // [batch_size * embed_dim] - current hidden state
    residual: GpuTensor,          // [batch_size * embed_dim] - residual connection
    attn_out: GpuTensor,          // [batch_size * embed_dim] - attention output
    q_proj: GpuTensor,            // [batch_size * n_heads * head_dim]
    k_proj: GpuTensor,            // [batch_size * n_kv_heads * head_dim]
    v_proj: GpuTensor,            // [batch_size * n_kv_heads * head_dim]
    ffn_gate: GpuTensor,          // [batch_size * hidden_dim]
    ffn_up: GpuTensor,            // [batch_size * hidden_dim]
    ffn_out: GpuTensor,           // [batch_size * embed_dim]
    logits: GpuTensor,            // [batch_size * vocab_size]

    // Scratch buffers
    scratch1: GpuTensor,
    scratch2: GpuTensor,
    
    const Self = @This();
    
    /// Initialize with default batch_size=1 (single token mode)
    pub fn init(
        allocator: std.mem.Allocator,
        embed_dim: u32,
        hidden_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        vocab_size: u32,
        n_layers: u32,
    ) !Self {
        return initWithBatchSize(allocator, embed_dim, hidden_dim, n_heads, n_kv_heads, vocab_size, n_layers, 1);
    }

    /// Initialize with configurable batch size for parallel token processing
    pub fn initWithBatchSize(
        allocator: std.mem.Allocator,
        embed_dim: u32,
        hidden_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        vocab_size: u32,
        n_layers: u32,
        batch_size: u32,
    ) !Self {
        std.debug.print("\n🔧 Initializing GPU Inference Engine (batch_size={})...\n", .{batch_size});

        // Create cuBLAS handle
        var handle: cublas.cublasHandle_t = undefined;
        const status = cublas.cublasCreate_v2(&handle);
        if (status != 0) {
            std.debug.print("cuBLAS create failed: {}\n", .{status});
            return error.CublasInitFailed;
        }

        // Create CUDA stream for async operations
        var stream: ?*anyopaque = null;
        const stream_result = cuda.cudaStreamCreate(@ptrCast(&stream));
        if (stream_result != 0) {
            std.debug.print("CUDA stream create failed: {}\n", .{stream_result});
            return error.CudaStreamCreateFailed;
        }

        // Set cuBLAS to use this stream
        _ = cublas.cublasSetStream_v2(handle, stream);

        const head_dim = embed_dim / n_heads;

        // Allocate activation buffers sized for batch_size tokens
        const hidden_state = try GpuTensor.alloc(batch_size * embed_dim);
        const residual = try GpuTensor.alloc(batch_size * embed_dim);
        const attn_out = try GpuTensor.alloc(batch_size * embed_dim);
        const q_proj = try GpuTensor.alloc(batch_size * n_heads * head_dim);
        const k_proj = try GpuTensor.alloc(batch_size * n_kv_heads * head_dim);
        const v_proj = try GpuTensor.alloc(batch_size * n_kv_heads * head_dim);
        const ffn_gate = try GpuTensor.alloc(batch_size * hidden_dim);
        const ffn_up = try GpuTensor.alloc(batch_size * hidden_dim);
        const ffn_out = try GpuTensor.alloc(batch_size * embed_dim);
        const logits = try GpuTensor.alloc(batch_size * vocab_size);
        const scratch1 = try GpuTensor.alloc(batch_size * @max(hidden_dim, vocab_size));
        const scratch2 = try GpuTensor.alloc(batch_size * @max(hidden_dim, vocab_size));

        const total_mem = hidden_state.memoryUsage() + residual.memoryUsage() +
            attn_out.memoryUsage() + q_proj.memoryUsage() + k_proj.memoryUsage() +
            v_proj.memoryUsage() + ffn_gate.memoryUsage() + ffn_up.memoryUsage() +
            ffn_out.memoryUsage() + logits.memoryUsage() + scratch1.memoryUsage() +
            scratch2.memoryUsage();

        std.debug.print("   ✅ GPU activation buffers: {d:.2} MB (batch_size={})\n", .{
            @as(f64, @floatFromInt(total_mem)) / (1024.0 * 1024.0), batch_size,
        });
        
        return Self{
            .allocator = allocator,
            .cublas_handle = handle,
            .stream = stream,
            .graph = null,
            .graph_exec = null,
            .graph_captured = false,
            .embed_dim = embed_dim,
            .hidden_dim = hidden_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .vocab_size = vocab_size,
            .n_layers = n_layers,
            .batch_size = batch_size,
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
        // Destroy CUDA graph if captured
        if (self.graph_exec) |ge| {
            _ = cuda.cudaGraphExecDestroy(ge);
        }
        if (self.graph) |g| {
            _ = cuda.cudaGraphDestroy(g);
        }
        // Destroy stream
        if (self.stream) |s| {
            _ = cuda.cudaStreamDestroy(s);
        }
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

        // Use CUBLAS_COMPUTE_32F_FAST_16F for best Tensor Core performance
        // This uses FP16 operations with FP32 accumulation - ideal for Turing GPUs
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
            cublas.CUBLAS_COMPUTE_32F_FAST_16F, // FP16 Tensor Cores with FP32 accumulate
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

            // Save residual (GPU D2D copy - very fast)
            try self.copyTensor(&self.residual, &self.hidden_state);

            // TODO: RMSNorm on GPU (currently skipped)

            // Attention projections (GPU-to-GPU matmul)
            // Note: M=1 means these are GEMV operations - could use cublasSgemv for better perf
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

            // Add residual (TODO: implement proper CUDA add kernel)
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

    /// Forward pass with CUDA Graph - captures on first call, replays on subsequent calls
    /// This significantly reduces kernel launch overhead (154 launches → 1 graph launch)
    pub fn forwardWithGraph(
        self: *Self,
        weights: *const GpuWeightCache,
        position: u32,
    ) !*GpuTensor {
        if (self.graph_captured) {
            // Replay captured graph - much faster!
            if (self.graph_exec) |ge| {
                if (self.stream) |s| {
                    const result = cuda.cudaGraphLaunch(ge, s);
                    if (result != 0) {
                        std.debug.print("cudaGraphLaunch failed: {}\n", .{result});
                        return error.GraphLaunchFailed;
                    }
                    return &self.logits;
                }
            }
        }

        // First call - capture the graph
        if (self.stream) |s| {
            std.debug.print("   📸 Capturing CUDA Graph...\n", .{});

            // Begin capture
            const begin_result = cuda.cudaStreamBeginCapture(s, cuda.cudaStreamCaptureModeGlobal);
            if (begin_result != 0) {
                std.debug.print("cudaStreamBeginCapture failed: {}, falling back to regular forward\n", .{begin_result});
                return self.forward(weights, position);
            }

            // Execute forward pass (operations are captured, not executed)
            _ = try self.forward(weights, position);

            // End capture and get graph
            var graph: cuda.cudaGraph_t = undefined;
            const end_result = cuda.cudaStreamEndCapture(s, &graph);
            if (end_result != 0) {
                std.debug.print("cudaStreamEndCapture failed: {}, falling back to regular forward\n", .{end_result});
                return self.forward(weights, position);
            }
            self.graph = graph;

            // Instantiate graph into executable
            var graph_exec: cuda.cudaGraphExec_t = undefined;
            const inst_result = cuda.cudaGraphInstantiate(&graph_exec, graph, null, null, 0);
            if (inst_result != 0) {
                std.debug.print("cudaGraphInstantiate failed: {}, falling back to regular forward\n", .{inst_result});
                return self.forward(weights, position);
            }
            self.graph_exec = graph_exec;
            self.graph_captured = true;

            std.debug.print("   ✅ CUDA Graph captured! Future calls will replay graph.\n", .{});

            // Need to run forward again since capture doesn't produce output
            return self.forward(weights, position);
        }

        // No stream, fall back to regular forward
        return self.forward(weights, position);
    }

    /// Copy tensor data (GPU-to-GPU) - async on stream
    fn copyTensor(self: *Self, dst: *GpuTensor, src: *const GpuTensor) !void {
        const result = cuda.cudaMemcpyAsync(
            dst.devicePtr(),
            src.devicePtr(),
            @min(dst.byte_size, src.byte_size),
            cuda.cudaMemcpyDeviceToDevice,
            self.stream,
        );
        if (result != 0) return error.CudaCopyFailed;
    }

    /// Add two tensors on GPU (TODO: implement as CUDA kernel)
    fn addTensors(self: *Self, out: *GpuTensor, a: *const GpuTensor, b: *const GpuTensor) !void {
        // For now, just copy a (proper implementation needs CUDA kernel)
        const result = cuda.cudaMemcpyAsync(
            out.devicePtr(),
            a.devicePtr(),
            @min(out.byte_size, a.byte_size),
            cuda.cudaMemcpyDeviceToDevice,
            self.stream,
        );
        _ = b;
        if (result != 0) return error.CudaCopyFailed;
    }

    /// Load token embedding to GPU hidden state (single token)
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

    /// Load multiple token embeddings for batched inference
    /// token_ids should have exactly batch_size elements
    pub fn loadEmbeddingsBatched(self: *Self, weights: *const GpuWeightCache, token_ids: []const u32) !void {
        if (token_ids.len != self.batch_size) {
            std.debug.print("loadEmbeddingsBatched: expected {} tokens, got {}\n", .{self.batch_size, token_ids.len});
            return error.BatchSizeMismatch;
        }

        const src_ptr: [*]u8 = @ptrCast(weights.token_embedding.ptr);
        const dst_ptr: [*]u8 = @ptrCast(self.hidden_state.devicePtr());
        const row_bytes = self.embed_dim * @sizeOf(f16);

        // Copy each token's embedding to the hidden_state buffer
        for (token_ids, 0..) |token_id, i| {
            const src_offset = @as(usize, token_id) * row_bytes;
            const dst_offset = i * row_bytes;

            const result = cuda.cudaMemcpyAsync(
                @ptrCast(dst_ptr + dst_offset),
                @ptrCast(src_ptr + src_offset),
                row_bytes,
                cuda.cudaMemcpyDeviceToDevice,
                self.stream,
            );
            if (result != 0) return error.CudaCopyFailed;
        }
    }

    /// Forward pass for batch of tokens - uses M=batch_size for better GPU utilization
    /// Input: token embeddings loaded into hidden_state via loadEmbeddingsBatched
    /// Output: logits tensor with batch_size * vocab_size values
    pub fn forwardBatched(
        self: *Self,
        weights: *const GpuWeightCache,
        _: u32, // position (for RoPE - TODO)
    ) !*GpuTensor {
        const M = self.batch_size; // Number of tokens in batch

        // Process each transformer layer
        for (0..self.n_layers) |layer_idx| {
            const layer = weights.getLayer(layer_idx);

            // Save residual
            try self.copyTensor(&self.residual, &self.hidden_state);

            // Attention projections with M=batch_size (true GEMM, not GEMV!)
            // Q = hidden @ Wq  [batch_size × embed_dim] @ [embed_dim × n_heads*head_dim]
            try self.gpuMatmul(&self.q_proj, &self.hidden_state, &layer.wq, M, self.n_heads * self.head_dim, self.embed_dim);
            // K = hidden @ Wk
            try self.gpuMatmul(&self.k_proj, &self.hidden_state, &layer.wk, M, self.n_kv_heads * self.head_dim, self.embed_dim);
            // V = hidden @ Wv
            try self.gpuMatmul(&self.v_proj, &self.hidden_state, &layer.wv, M, self.n_kv_heads * self.head_dim, self.embed_dim);

            // Output projection (simplified - no actual attention)
            try self.gpuMatmul(&self.attn_out, &self.v_proj, &layer.wo, M, self.embed_dim, self.n_kv_heads * self.head_dim);

            // Add residual
            try self.addTensors(&self.hidden_state, &self.attn_out, &self.residual);

            // FFN
            try self.copyTensor(&self.residual, &self.hidden_state);

            // gate = hidden @ W_gate
            try self.gpuMatmul(&self.ffn_gate, &self.hidden_state, &layer.w_gate, M, self.hidden_dim, self.embed_dim);
            // up = hidden @ W_up
            try self.gpuMatmul(&self.ffn_up, &self.hidden_state, &layer.w_up, M, self.hidden_dim, self.embed_dim);

            // down = (gate * up) @ W_down
            try self.gpuMatmul(&self.ffn_out, &self.ffn_gate, &layer.w_down, M, self.embed_dim, self.hidden_dim);

            // Add residual
            try self.addTensors(&self.hidden_state, &self.ffn_out, &self.residual);
        }

        // Final output projection
        try self.gpuMatmul(&self.logits, &self.hidden_state, &weights.output_weight, M, self.vocab_size, self.embed_dim);

        return &self.logits;
    }

    /// Get logits back to CPU
    pub fn getLogits(self: *Self, host_buffer: []f32) !void {
        try self.logits.copyToHostF32(host_buffer);
    }
};

