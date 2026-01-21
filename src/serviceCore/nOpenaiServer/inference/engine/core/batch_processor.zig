const std = @import("std");
const llama = @import("llama_model");
const transformer = @import("transformer");
const matrix_ops = @import("matrix_ops");
const kv_cache = @import("kv_cache");
const compute = @import("compute");
const gguf = @import("gguf_loader");

/// Batch Processing for Efficient Multi-Token Inference
/// Processes multiple tokens in parallel to reduce overhead

// ============================================================================
// Batch Configuration
// ============================================================================

pub const BatchConfig = struct {
    max_batch_size: usize = 32,
    enable_parallel: bool = true,

    pub fn default() BatchConfig {
        return .{
            .max_batch_size = 32,
            .enable_parallel = true,
        };
    }
};

// ============================================================================
// Batch State
// ============================================================================

pub const BatchState = struct {
    allocator: std.mem.Allocator,
    max_batch_size: usize,
    embed_dim: usize,

    // Shared buffers for batch processing
    batch_embeddings: []f32, // [max_batch_size, embed_dim]
    batch_hidden: []f32, // [max_batch_size, embed_dim]
    batch_output: []f32, // [max_batch_size, embed_dim]

    pub fn init(
        allocator: std.mem.Allocator,
        config: BatchConfig,
        embed_dim: usize,
    ) !BatchState {
        std.debug.print("\n📦 Initializing Batch Processor\n", .{});
        std.debug.print("   Max batch size: {d}\n", .{config.max_batch_size});
        std.debug.print("   Embedding dim: {d}\n", .{embed_dim});
        std.debug.print("   Parallel: {}\n", .{config.enable_parallel});

        const batch_embeddings = try allocator.alloc(f32, config.max_batch_size * embed_dim);
        errdefer allocator.free(batch_embeddings);

        const batch_hidden = try allocator.alloc(f32, config.max_batch_size * embed_dim);
        errdefer allocator.free(batch_hidden);

        const batch_output = try allocator.alloc(f32, config.max_batch_size * embed_dim);
        errdefer allocator.free(batch_output);

        const total_mb = (3 * config.max_batch_size * embed_dim * @sizeOf(f32)) / (1024 * 1024);
        std.debug.print("   Batch buffers: {d} MB\n", .{total_mb});
        std.debug.print("   ✅ Batch processor initialized\n", .{});

        return BatchState{
            .allocator = allocator,
            .max_batch_size = config.max_batch_size,
            .embed_dim = embed_dim,
            .batch_embeddings = batch_embeddings,
            .batch_hidden = batch_hidden,
            .batch_output = batch_output,
        };
    }

    pub fn deinit(self: *BatchState) void {
        self.allocator.free(self.batch_embeddings);
        self.allocator.free(self.batch_hidden);
        self.allocator.free(self.batch_output);
    }
};

// ============================================================================
// Batch Forward Pass
// ============================================================================

/// Process multiple tokens through embedding layer
/// Handles both f32 and quantized embeddings
pub fn batchGetEmbeddings(
    batch_state: *BatchState,
    token_ids: []const u32,
    token_embedding: matrix_ops.Weight,
    embed_dim: usize,
) void {
    for (token_ids, 0..) |token_id, i| {
        const output_start = i * embed_dim;
        const output = batch_state.batch_embeddings[output_start .. output_start + embed_dim];

        // Use get_row which handles dequantization for quantized weights
        matrix_ops.get_row(output, token_embedding, token_id, embed_dim);
    }
}

/// Process batch through a transformer layer
pub fn batchTransformerLayer(
    allocator: std.mem.Allocator,
    batch_state: *BatchState,
    batch_size: usize,
    layer_weights: transformer.TransformerWeights,
    kv_caches: []kv_cache.KVCache,
    layer_idx: u32,
    positions: []const u32,
    config: transformer.TransformerConfig,
    rope_freqs: []const f32,
) !void {
    const embed_dim = config.embed_dim;

    // Process each token in batch sequentially
    // (Parallel processing would require more complex KV cache management)
    for (0..batch_size) |batch_idx| {
        const input_start = batch_idx * embed_dim;
        const input = batch_state.batch_embeddings[input_start .. input_start + embed_dim];

        const output_start = batch_idx * embed_dim;
        const output = batch_state.batch_output[output_start .. output_start + embed_dim];

        // Process single token through transformer
        try transformer.computeTransformerLayer(
            allocator,
            output,
            input,
            layer_weights,
            &kv_caches[batch_idx],
            layer_idx,
            positions[batch_idx],
            config,
            rope_freqs,
        );
    }

    // Copy output back to embeddings for next layer
    const total_size = batch_size * embed_dim;
    @memcpy(
        batch_state.batch_embeddings[0..total_size],
        batch_state.batch_output[0..total_size],
    );
}

/// Process batch through a transformer layer using GPU backend
/// Uses ComputeBackend for accelerated matmul operations
pub fn batchTransformerLayerGpu(
    allocator: std.mem.Allocator,
    batch_state: *BatchState,
    batch_size: usize,
    layer_weights: transformer.TransformerWeights,
    kv_caches: []kv_cache.KVCache,
    layer_idx: u32,
    positions: []const u32,
    config: transformer.TransformerConfig,
    rope_freqs: []const f32,
    backend: compute.ComputeBackend,
) !void {
    const embed_dim = config.embed_dim;

    // Process each token in batch sequentially
    // (Parallel processing would require more complex KV cache management)
    for (0..batch_size) |batch_idx| {
        const input_start = batch_idx * embed_dim;
        const input = batch_state.batch_embeddings[input_start .. input_start + embed_dim];

        const output_start = batch_idx * embed_dim;
        const output = batch_state.batch_output[output_start .. output_start + embed_dim];

        // Process single token through transformer using GPU backend
        try transformer.computeTransformerLayerGpu(
            allocator,
            output,
            input,
            layer_weights,
            &kv_caches[batch_idx],
            layer_idx,
            positions[batch_idx],
            config,
            rope_freqs,
            backend,
        );
    }

    // Copy output back to embeddings for next layer
    const total_size = batch_size * embed_dim;
    @memcpy(
        batch_state.batch_embeddings[0..total_size],
        batch_state.batch_output[0..total_size],
    );
}

/// Apply final normalization to batch
pub fn batchFinalNorm(
    batch_state: *BatchState,
    batch_size: usize,
    output_norm: []const f32,
    embed_dim: usize,
    rms_norm_eps: f32,
) void {
    for (0..batch_size) |batch_idx| {
        const input_start = batch_idx * embed_dim;
        const input = batch_state.batch_embeddings[input_start .. input_start + embed_dim];

        const output_start = batch_idx * embed_dim;
        const output = batch_state.batch_output[output_start .. output_start + embed_dim];

        matrix_ops.rms_norm(output, input, output_norm, rms_norm_eps);
    }
}

/// Project batch to vocabulary (handles quantized output weights)
/// Uses GPU backend for acceleration when available
pub fn batchOutputProjection(
    allocator: std.mem.Allocator,
    batch_state: *BatchState,
    batch_size: usize,
    output_weight: matrix_ops.Weight,
    embed_dim: usize,
    vocab_size: usize,
    backend: compute.ComputeBackend,
) ![]f32 {
    // Allocate output logits for all batch items
    const total_logits = try allocator.alloc(f32, batch_size * vocab_size);

    // Extract weight data and type from Weight union for backend interface
    var w_ptr: []const u8 = undefined;
    var w_type: gguf.QuantizationType = undefined;

    switch (output_weight) {
        .f32 => |data| {
            w_ptr = std.mem.sliceAsBytes(data);
            w_type = .F32;
        },
        .q4_0 => |data| {
            w_ptr = data;
            w_type = .Q4_0;
        },
        .q4_k => |data| {
            w_ptr = data;
            w_type = .Q4_K;
        },
        .q6_k => |data| {
            w_ptr = data;
            w_type = .Q6_K;
        },
    }

    for (0..batch_size) |batch_idx| {
        const input_start = batch_idx * embed_dim;
        const input = batch_state.batch_output[input_start .. input_start + embed_dim];

        const logits_start = batch_idx * vocab_size;
        const logits = total_logits[logits_start .. logits_start + vocab_size];

        // Use backend.matmul for GPU acceleration (cuBLAS + Tensor Cores)
        try backend.matmul(logits, w_ptr, w_type, input, vocab_size, 1, embed_dim);
    }

    return total_logits;
}

// ============================================================================
// Batch Model Extension
// ============================================================================

pub const BatchLlamaModel = struct {
    model: *llama.LlamaModel,
    batch_state: BatchState,
    batch_config: BatchConfig,
    batch_kv_caches: []kv_cache.KVCache,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *llama.LlamaModel,
        batch_config: BatchConfig,
    ) !BatchLlamaModel {
        const batch_state = try BatchState.init(
            allocator,
            batch_config,
            model.config.embed_dim,
        );

        // Initialize KV caches for batch processing
        const batch_kv_caches = try allocator.alloc(kv_cache.KVCache, batch_config.max_batch_size);
        errdefer allocator.free(batch_kv_caches);

        for (batch_kv_caches) |*cache| {
            cache.* = try kv_cache.KVCache.init(
                allocator,
                model.config.n_layers,
                model.config.n_kv_heads,
                model.config.head_dim,
                model.config.max_seq_len,
            );
        }

        return BatchLlamaModel{
            .model = model,
            .batch_state = batch_state,
            .batch_config = batch_config,
            .batch_kv_caches = batch_kv_caches,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BatchLlamaModel) void {
        self.batch_state.deinit();
        for (self.batch_kv_caches) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.batch_kv_caches);
    }

    /// Forward pass for multiple tokens
    pub fn forwardBatch(
        self: *BatchLlamaModel,
        token_ids: []const u32,
        positions: []const u32,
    ) ![]f32 {
        if (token_ids.len != positions.len) {
            return error.BatchSizeMismatch;
        }

        if (token_ids.len > self.batch_config.max_batch_size) {
            return error.BatchTooLarge;
        }

        const batch_size = token_ids.len;
        const config = self.model.config;

        // Get embeddings for all tokens (handles quantized weights)
        batchGetEmbeddings(
            &self.batch_state,
            token_ids,
            self.model.weights.token_embedding,
            config.embed_dim,
        );

        // Process through all transformer layers
        const layer_config = transformer.TransformerConfig{
            .embed_dim = config.embed_dim,
            .ffn_dim = config.ffn_dim,
            .n_heads = config.n_heads,
            .n_kv_heads = config.n_kv_heads,
            .head_dim = config.head_dim,
            .rope_theta = config.rope_theta,
            .rms_norm_eps = config.rms_norm_eps,
        };

        for (0..config.n_layers) |layer_idx| {
            try batchTransformerLayerGpu(
                self.model.allocator,
                &self.batch_state,
                batch_size,
                self.model.weights.layer_weights[layer_idx],
                self.batch_kv_caches,
                @intCast(layer_idx),
                positions,
                layer_config,
                self.model.rope_freqs,
                self.model.backend,
            );
        }

        // Final normalization
        batchFinalNorm(
            &self.batch_state,
            batch_size,
            self.model.weights.output_norm,
            config.embed_dim,
            config.rms_norm_eps,
        );

        // Project to vocabulary using GPU backend (cuBLAS + Tensor Cores)
        const logits = try batchOutputProjection(
            self.model.allocator,
            &self.batch_state,
            batch_size,
            self.model.weights.output_weight,
            config.embed_dim,
            config.vocab_size,
            self.model.backend,
        );

        return logits;
    }

    /// Process prompt tokens in batches
    pub fn processPromptBatch(
        self: *BatchLlamaModel,
        prompt_tokens: []const u32,
        batch_size: usize,
    ) !void {
        std.debug.print("\n📦 Processing prompt in batches of {d}...\n", .{batch_size});

        var position: u32 = 0;
        var i: usize = 0;

        while (i < prompt_tokens.len) {
            const remaining = prompt_tokens.len - i;
            const current_batch_size = @min(batch_size, remaining);

            const token_batch = prompt_tokens[i .. i + current_batch_size];

            // Create position array
            const positions = try self.model.allocator.alloc(u32, current_batch_size);
            defer self.model.allocator.free(positions);

            for (0..current_batch_size) |j| {
                positions[j] = position + @as(u32, @intCast(j));
            }

            // Process batch
            const logits = try self.forwardBatch(token_batch, positions);
            self.model.allocator.free(logits);

            // Advance position and caches
            for (0..current_batch_size) |batch_idx| {
                self.batch_kv_caches[batch_idx].advance();
                position += 1;
            }

            i += current_batch_size;

            if (i % (batch_size * 4) == 0 or i >= prompt_tokens.len) {
                std.debug.print("   Processed {d}/{d} tokens\n", .{ i, prompt_tokens.len });
            }
        }

        std.debug.print("   ✅ Prompt processing complete\n", .{});
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_batch_processor(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Batch Processor\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    // Test 1: Batch state initialization
    {
        std.debug.print("\n1️⃣  Testing batch state initialization...\n", .{});

        const config = BatchConfig{
            .max_batch_size = 16,
            .enable_parallel = true,
        };

        var batch_state = try BatchState.init(allocator, config, 64);
        defer batch_state.deinit();

        if (batch_state.batch_embeddings.len != 16 * 64) {
            return error.TestFailed;
        }

        std.debug.print("   ✅ Batch state initialized correctly\n", .{});
    }

    // Test 2: Batch embedding retrieval
    {
        std.debug.print("\n2️⃣  Testing batch embedding retrieval...\n", .{});

        const config = BatchConfig.default();
        var batch_state = try BatchState.init(allocator, config, 64);
        defer batch_state.deinit();

        // Create dummy embeddings
        const token_embedding_data = try allocator.alloc(f32, 100 * 64);
        defer allocator.free(token_embedding_data);
        @memset(token_embedding_data, 1.0);

        const token_ids = [_]u32{ 5, 10, 15 };

        // Wrap as Weight.f32 for the function call
        const token_embedding = matrix_ops.Weight{ .f32 = token_embedding_data };
        batchGetEmbeddings(&batch_state, &token_ids, token_embedding, 64);

        // Verify embeddings were copied
        for (0..3) |i| {
            const start = i * 64;
            const embedding = batch_state.batch_embeddings[start .. start + 64];

            for (embedding) |val| {
                if (val != 1.0) {
                    return error.TestFailed;
                }
            }
        }

        std.debug.print("   ✅ Batch embeddings retrieved correctly\n", .{});
    }

    std.debug.print("\n✅ All batch processor tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
