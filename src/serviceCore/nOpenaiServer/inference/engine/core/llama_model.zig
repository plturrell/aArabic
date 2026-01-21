const std = @import("std");
const gguf = @import("gguf_loader");
const transformer = @import("transformer");
const attention = @import("attention");
const tokenizer = @import("tokenizer");
const kv_cache = @import("kv_cache");
const matrix_ops = @import("matrix_ops");
const thread_pool = @import("thread_pool");
const compute = @import("compute");
const backend_cpu = @import("backend_cpu");
const backend_metal = @import("backend_metal");
const backend_cuda = @import("backend_cuda");

var log_enabled: ?bool = null;

fn logEnabled() bool {
    if (log_enabled) |enabled| {
        return enabled;
    }
    log_enabled = std.posix.getenv("SHIMMY_DEBUG") != null;
    return log_enabled.?;
}

fn log(comptime fmt: []const u8, args: anytype) void {
    if (logEnabled()) {
        std.debug.print(fmt, args);
    }
}

/// Complete Llama model for text generation
/// Implements multi-layer transformer with embeddings and output head

// ============================================================================
// Model Configuration
// ============================================================================

pub const LlamaConfig = struct {
    vocab_size: u32,
    n_layers: u32,
    embed_dim: u32,
    ffn_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-5,

    pub fn fromGGUF(model: *gguf.GGUFModel) LlamaConfig {
        return LlamaConfig{
            .vocab_size = model.metadata.vocab_size,
            .n_layers = model.metadata.n_layers,
            .embed_dim = model.metadata.hidden_size,
            .ffn_dim = model.metadata.intermediate_size,
            .n_heads = model.metadata.n_heads,
            .n_kv_heads = model.metadata.n_kv_heads,
            .head_dim = model.metadata.hidden_size / model.metadata.n_heads,
            .max_seq_len = model.metadata.max_seq_len,
            .rope_theta = model.metadata.rope_theta,
            .rms_norm_eps = 1e-5,
        };
    }
};

// ============================================================================
// Model Weights
// ============================================================================

pub const LlamaWeights = struct {
    allocator: std.mem.Allocator,
    token_embedding: matrix_ops.Weight, // [vocab_size, embed_dim]
    output_norm: []const f32, // [embed_dim]
    output_weight: matrix_ops.Weight, // [vocab_size, embed_dim]
    share_output_with_embedding: bool = false,
    share_output_norm_with_embedding: bool = false,

    // Per-layer weights (allocated as array)
    layer_weights: []transformer.TransformerWeights,

    pub fn deinit(self: *LlamaWeights) void {
        switch (self.token_embedding) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => |data| self.allocator.free(data),
        }
        if (!self.share_output_norm_with_embedding) {
            self.allocator.free(self.output_norm);
        }
        if (!self.share_output_with_embedding) {
            switch (self.output_weight) {
                .f32 => |data| self.allocator.free(data),
                .q4_0 => |data| self.allocator.free(data),
                .q4_k => |data| self.allocator.free(data),
                .q6_k => |data| self.allocator.free(data),
            }
        }

        for (self.layer_weights) |*lw| {
            lw.deinit();
        }
        self.allocator.free(self.layer_weights);
    }
};

// ============================================================================
// Llama Model
// ============================================================================

pub const LlamaModel = struct {
    allocator: std.mem.Allocator,
    config: LlamaConfig,
    weights: LlamaWeights,
    rope_freqs: []f32,
    kv_caches: []kv_cache.KVCache,
    tok: tokenizer.Tokenizer,
    pool: *thread_pool.ThreadPool,
    backend: compute.ComputeBackend,
    scratch_hidden: []f32,
    scratch_layer_output: []f32,
    scratch_logits: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        config: LlamaConfig,
        weights: LlamaWeights,
        tok: tokenizer.Tokenizer,
    ) !LlamaModel {
        log("\nInitializing Llama Model\n", .{});
        log("   Layers: {d}\n", .{config.n_layers});
        log("   Embedding: {d}\n", .{config.embed_dim});
        log("   Heads: {d} (KV: {d})\n", .{ config.n_heads, config.n_kv_heads });
        log("   Vocab: {d}\n", .{config.vocab_size});
        log("   Context: {d}\n", .{config.max_seq_len});

        // Initialize Thread Pool
        const pool = try allocator.create(thread_pool.ThreadPool);
        errdefer allocator.destroy(pool);
        pool.* = try thread_pool.ThreadPool.init(allocator, thread_pool.ThreadPoolConfig.default());
        try pool.start();
        log("   Thread pool: {d} threads\n", .{pool.config.num_threads});

        // Initialize Backend
        const builtin = @import("builtin");
        const is_macos = builtin.os.tag == .macos;
        const is_linux = builtin.os.tag == .linux;
        const backend: compute.ComputeBackend = blk: {
            if (is_macos) {
                // Try Metal first on macOS
                break :blk backend_metal.MetalBackend.init(allocator) catch |err| {
                    std.debug.print("Metal backend initialization failed: {s}. Falling back to CPU.\n", .{@errorName(err)});
                    break :blk try backend_cpu.CpuBackend.init(allocator, pool);
                };
            } else if (is_linux) {
                // Try CUDA first on Linux (T4/A100/H100 support)
                break :blk backend_cuda.CudaBackend.init(allocator) catch |err| {
                    std.debug.print("CUDA backend initialization failed: {s}. Falling back to CPU.\n", .{@errorName(err)});
                    break :blk try backend_cpu.CpuBackend.init(allocator, pool);
                };
            } else {
                break :blk try backend_cpu.CpuBackend.init(allocator, pool);
            }
        };

        // Precompute RoPE frequencies
        const rope_freqs = try attention.precomputeRopeFreqs(
            allocator,
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
        );
        errdefer allocator.free(rope_freqs);

        // Initialize per-layer KV caches
        // Each cache stores only 1 layer's worth of data (not n_layers!)
        // This allows independent memory management per layer
        var kv_caches = try allocator.alloc(kv_cache.KVCache, config.n_layers);
        errdefer allocator.free(kv_caches);

        for (0..config.n_layers) |i| {
            kv_caches[i] = try kv_cache.KVCache.init(
                allocator,
                1,  // Single layer per cache - NOT n_layers!
                config.n_kv_heads,
                config.head_dim,
                config.max_seq_len,
            );
            errdefer {
                for (0..i) |j| kv_caches[j].deinit();
            }
        }

        const scratch_hidden = try allocator.alloc(f32, @intCast(config.embed_dim));
        errdefer allocator.free(scratch_hidden);
        const scratch_layer_output = try allocator.alloc(f32, @intCast(config.embed_dim));
        errdefer allocator.free(scratch_layer_output);
        const scratch_logits = try allocator.alloc(f32, @intCast(config.vocab_size));
        errdefer allocator.free(scratch_logits);

        log("   Model initialized\n", .{});

        return LlamaModel{
            .allocator = allocator,
            .config = config,
            .weights = weights,
            .rope_freqs = rope_freqs,
            .kv_caches = kv_caches,
            .tok = tok,
            .pool = pool,
            .backend = backend,
            .scratch_hidden = scratch_hidden,
            .scratch_layer_output = scratch_layer_output,
            .scratch_logits = scratch_logits,
        };
    }

    pub fn deinit(self: *LlamaModel) void {
        self.allocator.free(self.rope_freqs);
        for (self.kv_caches) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.kv_caches);
        self.allocator.free(self.scratch_hidden);
        self.allocator.free(self.scratch_layer_output);
        self.allocator.free(self.scratch_logits);
        self.tok.deinit();
        self.weights.deinit();
        self.backend.deinit();
        self.pool.deinit();
        self.allocator.destroy(self.pool);
    }

    /// Forward pass for a single token
    pub fn forward(
        self: *LlamaModel,
        token_id: u32,
        position: u32,
    ) ![]f32 {
        const embed_dim = self.config.embed_dim;

        // Allocate hidden state buffer
        // OPTIMIZATION: Use pre-allocated buffer in struct instead of alloc/free every token
        const hidden = try self.allocator.alloc(f32, embed_dim);
        errdefer self.allocator.free(hidden);

        // Get token embedding (lookup)
        matrix_ops.get_row(hidden, self.weights.token_embedding, token_id, embed_dim);

        // Pass through each transformer layer
        const layer_config = transformer.TransformerConfig{
            .embed_dim = self.config.embed_dim,
            .ffn_dim = self.config.ffn_dim,
            .n_heads = self.config.n_heads,
            .n_kv_heads = self.config.n_kv_heads,
            .head_dim = self.config.head_dim,
            .rope_theta = self.config.rope_theta,
            .rms_norm_eps = self.config.rms_norm_eps,
        };

        const layer_output = try self.allocator.alloc(f32, embed_dim);
        defer self.allocator.free(layer_output);

        for (0..self.config.n_layers) |layer_idx| {
            try transformer.computeTransformerLayer(
                self.allocator,
                layer_output,
                hidden,
                self.weights.layer_weights[layer_idx],
                &self.kv_caches[layer_idx],
                0,  // Each per-layer cache only has 1 layer at index 0
                position,
                layer_config,
                self.rope_freqs,
            );

            // Copy output to hidden for next layer
            @memcpy(hidden, layer_output);
        }

        // Final layer norm
        matrix_ops.rms_norm(
            layer_output,
            hidden,
            self.weights.output_norm,
            self.config.rms_norm_eps,
        );

        // Project to vocabulary
        const logits = try self.allocator.alloc(f32, self.config.vocab_size);

        // Use Backend for final projection (Demonstrates offloading architecture)
        // Extract raw pointer and type from Weight union for backend interface
        var w_ptr: []const u8 = undefined;
        var w_type: gguf.QuantizationType = undefined;

        switch (self.weights.output_weight) {
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

        try self.backend.matmul(logits, w_ptr, w_type, layer_output, self.config.vocab_size, 1, embed_dim);

        self.allocator.free(hidden);

        return logits;
    }

    /// Forward pass using preallocated scratch buffers (not thread-safe).
    pub fn forwardReuse(
        self: *LlamaModel,
        token_id: u32,
        position: u32,
    ) ![]f32 {
        const embed_dim = self.config.embed_dim;

        const hidden = self.scratch_hidden;
        const layer_output = self.scratch_layer_output;

        // Get token embedding (lookup)
        matrix_ops.get_row(hidden, self.weights.token_embedding, token_id, embed_dim);

        // Debug: check embedding values for first few tokens
        if (position == 0) {
            var sum: f32 = 0;
            var max_val: f32 = -1e30;
            var min_val: f32 = 1e30;
            for (hidden[0..@min(embed_dim, 896)]) |v| {
                sum += v;
                if (v > max_val) max_val = v;
                if (v < min_val) min_val = v;
            }
            log("   [DEBUG] Token {d} embedding: sum={d:.4}, min={d:.4}, max={d:.4}\n", .{token_id, sum, min_val, max_val});
        }

        const layer_config = transformer.TransformerConfig{
            .embed_dim = self.config.embed_dim,
            .ffn_dim = self.config.ffn_dim,
            .n_heads = self.config.n_heads,
            .n_kv_heads = self.config.n_kv_heads,
            .head_dim = self.config.head_dim,
            .rope_theta = self.config.rope_theta,
            .rms_norm_eps = self.config.rms_norm_eps,
        };

        for (0..self.config.n_layers) |layer_idx| {
            try transformer.computeTransformerLayer(
                self.allocator,
                layer_output,
                hidden,
                self.weights.layer_weights[layer_idx],
                &self.kv_caches[layer_idx],
                0,  // Each per-layer cache only has 1 layer at index 0
                position,
                layer_config,
                self.rope_freqs,
            );

            // Debug: check hidden state after first and last layer
            if (position == 0 and (layer_idx == 0 or layer_idx == self.config.n_layers - 1)) {
                var sum: f32 = 0;
                var max_val: f32 = -1e30;
                var min_val: f32 = 1e30;
                for (layer_output) |v| {
                    sum += v;
                    if (v > max_val) max_val = v;
                    if (v < min_val) min_val = v;
                }
                log("   [DEBUG] Layer {d} output: sum={d:.4}, min={d:.4}, max={d:.4}\n", .{layer_idx, sum, min_val, max_val});
            }

            @memcpy(hidden, layer_output);
        }

        matrix_ops.rms_norm(
            layer_output,
            hidden,
            self.weights.output_norm,
            self.config.rms_norm_eps,
        );

        var w_ptr: []const u8 = undefined;
        var w_type: gguf.QuantizationType = undefined;

        switch (self.weights.output_weight) {
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

        try self.backend.matmul(
            self.scratch_logits,
            w_ptr,
            w_type,
            layer_output,
            self.config.vocab_size,
            1,
            embed_dim,
        );

        // Debug: check logit distribution
        if (position == 0) {
            var max_logit: f32 = -1e30;
            var min_logit: f32 = 1e30;
            var max_idx: usize = 0;
            var sum: f32 = 0;
            for (self.scratch_logits, 0..) |logit, i| {
                sum += logit;
                if (logit > max_logit) {
                    max_logit = logit;
                    max_idx = i;
                }
                if (logit < min_logit) min_logit = logit;
            }
            const mean = sum / @as(f32, @floatFromInt(self.scratch_logits.len));
            log("   [DEBUG] Logits: mean={d:.4}, min={d:.4}, max={d:.4}, argmax={d}\n", .{mean, min_logit, max_logit, max_idx});
        }

        return self.scratch_logits;
    }

    /// Reset KV caches for new sequence
    pub fn resetCaches(self: *LlamaModel) void {
        for (self.kv_caches) |*cache| {
            cache.reset();
        }
    }

    /// Advance all caches to next position
    pub fn advanceCaches(self: *LlamaModel) void {
        for (self.kv_caches) |*cache| {
            cache.advance();
        }
    }

    /// Generate text from prompt
    pub fn generate(
        self: *LlamaModel,
        prompt: []const u8,
        max_tokens: u32,
        temperature: f32,
        top_k: ?usize,
        top_p: ?f32,
    ) ![]u8 {
        log("\nGenerating text...\n", .{});
        log("   Prompt: \"{s}\"\n", .{prompt});
        log("   Max tokens: {d}\n", .{max_tokens});
        log("   Temperature: {d:.2}\n", .{temperature});

        // Reset caches
        self.resetCaches();

        // Encode prompt
        const prompt_tokens = try self.tok.encode(prompt, self.allocator);
        defer self.allocator.free(prompt_tokens);

        log("   Prompt tokens: {d}\n", .{prompt_tokens.len});

        // Process prompt tokens
        for (prompt_tokens, 0..) |token, pos| {
            _ = try self.forwardReuse(token, @intCast(pos));
            if (pos < prompt_tokens.len - 1) {
                self.advanceCaches();
            }
        }

        // Generate new tokens
        var generated_tokens = try self.allocator.alloc(u32, max_tokens);
        defer self.allocator.free(generated_tokens);

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();

        var gen_count: usize = 0;
        var position = prompt_tokens.len;

        log("   Generating", .{});

        const probs = try self.allocator.alloc(f32, @intCast(self.config.vocab_size));
        defer self.allocator.free(probs);

        while (gen_count < max_tokens) {
            // Get next token
            const prev_token = if (gen_count == 0)
                prompt_tokens[prompt_tokens.len - 1]
            else
                generated_tokens[gen_count - 1];

            // Calculate probabilities
            const logits = try self.forwardReuse(prev_token, @intCast(position));

            // Debug: show logits for first generated token
            if (gen_count == 0) {
                var max_logit: f32 = -1e30;
                var argmax: usize = 0;
                for (logits, 0..) |l, i| {
                    if (l > max_logit) {
                        max_logit = l;
                        argmax = i;
                    }
                }
                log("\n   [GEN] pos={d} prev_tok={d} logits_argmax={d} logit={d:.4}\n", .{position, prev_token, argmax, max_logit});
            }

            tokenizer.calculateProbs(probs, logits, temperature);

            // Debug: show top probs
            if (gen_count == 0) {
                var max_prob: f32 = 0;
                var max_prob_idx: usize = 0;
                for (probs, 0..) |p, i| {
                    if (p > max_prob) {
                        max_prob = p;
                        max_prob_idx = i;
                    }
                }
                log("   [GEN] After softmax: top_prob_idx={d} prob={d:.6}\n", .{max_prob_idx, max_prob});
            }

            // Apply sampling
            if (top_k) |k| {
                tokenizer.topK(probs, k);
            }
            if (top_p) |p| {
                tokenizer.topP(probs, p);
            }

            // Sample next token
            const next_token = tokenizer.sampleToken(probs, random);

            if (gen_count == 0) {
                log("   [GEN] Sampled token: {d}\n", .{next_token});
            }
            generated_tokens[gen_count] = next_token;
            gen_count += 1;

            // Check for EOS
            if (next_token == self.tok.eos_token) {
                log(" [EOS]\n", .{});
                break;
            }

            // Progress indicator
            if (gen_count % 10 == 0) {
                log(".", .{});
            }

            self.advanceCaches();
            position += 1;
        }

        log("   Generated {d} tokens\n", .{gen_count});

        // Debug: show generated token IDs and their vocab entries
        log("   Token IDs: ", .{});
        for (generated_tokens[0..gen_count]) |tid| {
            log("{d} ", .{tid});
        }
        log("\n", .{});

        // Show first few token texts
        for (generated_tokens[0..@min(gen_count, 5)]) |tid| {
            const text = self.tok.vocab[tid].text;
            log("   Token {d}: \"{s}\" (len={d})\n", .{tid, text, text.len});
        }

        // Decode generated tokens
        const output = try self.tok.decode(generated_tokens[0..gen_count], self.allocator);
        log("   Decoded output ({d} bytes): \"{s}\"\n", .{output.len, output});

        return output;
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_llama_model(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Llama Model\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    // Create minimal test configuration
    const config = LlamaConfig{
        .vocab_size = 100,
        .n_layers = 2,
        .embed_dim = 64,
        .ffn_dim = 256,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .max_seq_len = 32,
        .rope_theta = 10000.0,
        .rms_norm_eps = 1e-5,
    };

    std.debug.print("\n1️⃣  Creating test model weights...\n", .{});

    // Create dummy weights
    const token_embedding = try allocator.alloc(f32, config.vocab_size * config.embed_dim);
    // defer allocator.free(token_embedding); // Freed by LlamaWeights.deinit
    @memset(token_embedding, 0.1);

    const output_norm = try allocator.alloc(f32, config.embed_dim);
    // defer allocator.free(output_norm); // Freed by LlamaWeights.deinit
    for (output_norm) |*w| w.* = 1.0;

    const output_weight = try allocator.alloc(f32, config.embed_dim * config.vocab_size);
    // defer allocator.free(output_weight); // Freed by LlamaWeights.deinit
    @memset(output_weight, 0.1);

    // Create layer weights
    const layer_weights = try allocator.alloc(transformer.TransformerWeights, config.n_layers);
    // defer allocator.free(layer_weights); // Freed by LlamaWeights.deinit

    const q_dim = config.n_heads * config.head_dim;
    const kv_dim = config.n_kv_heads * config.head_dim;

    for (0..config.n_layers) |layer_idx| {
        const attn_norm = try allocator.alloc(f32, config.embed_dim);
        for (attn_norm) |*w| w.* = 1.0;

        const wq = try allocator.alloc(f32, config.embed_dim * q_dim);
        @memset(wq, 0.1);
        const wk = try allocator.alloc(f32, config.embed_dim * kv_dim);
        @memset(wk, 0.1);
        const wv = try allocator.alloc(f32, config.embed_dim * kv_dim);
        @memset(wv, 0.1);
        const wo = try allocator.alloc(f32, q_dim * config.embed_dim);
        @memset(wo, 0.1);

        const ffn_norm = try allocator.alloc(f32, config.embed_dim);
        for (ffn_norm) |*w| w.* = 1.0;

        const w_gate = try allocator.alloc(f32, config.embed_dim * config.ffn_dim);
        @memset(w_gate, 0.1);
        const w_up = try allocator.alloc(f32, config.embed_dim * config.ffn_dim);
        @memset(w_up, 0.1);
        const w_down = try allocator.alloc(f32, config.ffn_dim * config.embed_dim);
        @memset(w_down, 0.1);

        layer_weights[layer_idx] = transformer.TransformerWeights{
            .allocator = allocator,
            .attn_norm = attn_norm,
            .wq = .{ .f32 = wq },
            .wk = .{ .f32 = wk },
            .wv = .{ .f32 = wv },
            .wo = .{ .f32 = wo },
            .ffn_norm = ffn_norm,
            .w_gate = .{ .f32 = w_gate },
            .w_up = .{ .f32 = w_up },
            .w_down = .{ .f32 = w_down },
        };
    }

    const weights = LlamaWeights{
        .allocator = allocator,
        .token_embedding = .{ .f32 = token_embedding },
        .output_norm = output_norm,
        .output_weight = .{ .f32 = output_weight },
        .layer_weights = layer_weights,
    };

    // Create simple tokenizer with dummy metadata
    var dummy_model = gguf.GGUFModel{
        .allocator = allocator,
        .file = undefined,
        .header = undefined,
        .metadata = .{
            .architecture = .Llama,
            .vocab_size = config.vocab_size,
            .n_layers = config.n_layers,
            .n_heads = config.n_heads,
            .n_kv_heads = config.n_kv_heads,
            .hidden_size = config.embed_dim,
            .intermediate_size = config.ffn_dim,
            .max_seq_len = config.max_seq_len,
            .rope_theta = config.rope_theta,
            .rms_norm_eps = config.rms_norm_eps,
            .conv_kernel = 3,
        },
        .tensors = &[_]gguf.TensorInfo{},
        .vocab_tokens = &[_][]u8{},
        .vocab_scores = &[_]f32{},
        .tensor_data_offset = 0,
    };

    const tok = try tokenizer.Tokenizer.loadFromModel(allocator, &dummy_model);

    std.debug.print("   ✅ Weights created\n", .{});

    std.debug.print("\n2️⃣  Initializing model...\n", .{});

    var model = try LlamaModel.init(allocator, config, weights, tok);
    defer model.deinit();

    std.debug.print("   ✅ Model initialized\n", .{});

    std.debug.print("\n3️⃣  Testing forward pass...\n", .{});

    const logits = try model.forward(5, 0);
    defer allocator.free(logits);

    if (logits.len != config.vocab_size) {
        std.debug.print("   ❌ Logits size mismatch: {d} vs {d}\n", .{ logits.len, config.vocab_size });
        return error.TestFailed;
    }

    std.debug.print("   Logits size: {d}\n", .{logits.len});
    std.debug.print("   ✅ Forward pass working\n", .{});

    std.debug.print("\n✅ All Llama model tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
