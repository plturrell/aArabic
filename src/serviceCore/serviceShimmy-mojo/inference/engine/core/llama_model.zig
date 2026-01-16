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

    // Per-layer weights (allocated as array)
    layer_weights: []transformer.TransformerWeights,

    pub fn deinit(self: *LlamaWeights) void {
        switch (self.token_embedding) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            else => {},
        }
        self.allocator.free(self.output_norm);
        switch (self.output_weight) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            else => {},
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

    pub fn init(
        allocator: std.mem.Allocator,
        config: LlamaConfig,
        weights: LlamaWeights,
        tok: tokenizer.Tokenizer,
    ) !LlamaModel {
        std.debug.print("\n🦙 Initializing Llama Model\n", .{});
        std.debug.print("   Layers: {d}\n", .{config.n_layers});
        std.debug.print("   Embedding: {d}\n", .{config.embed_dim});
        std.debug.print("   Heads: {d} (KV: {d})\n", .{ config.n_heads, config.n_kv_heads });
        std.debug.print("   Vocab: {d}\n", .{config.vocab_size});
        std.debug.print("   Context: {d}\n", .{config.max_seq_len});

        // Initialize Thread Pool
        const pool = try allocator.create(thread_pool.ThreadPool);
        errdefer allocator.destroy(pool);
        pool.* = try thread_pool.ThreadPool.init(allocator, thread_pool.ThreadPoolConfig.default());
        try pool.start();
        std.debug.print("   Thread pool: {d} threads\n", .{pool.config.num_threads});

        // Initialize Backend
        const is_macos = @import("builtin").os.tag == .macos;
        const backend: compute.ComputeBackend = blk: {
            if (is_macos) {
                // Try Metal first on macOS
                break :blk backend_metal.MetalBackend.init(allocator) catch |err| {
                    std.debug.print("⚠️  Metal backend initialization failed: {s}. Falling back to CPU.\n", .{@errorName(err)});
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

        // Initialize KV caches for each layer
        var kv_caches = try allocator.alloc(kv_cache.KVCache, config.n_layers);
        errdefer allocator.free(kv_caches);

        for (0..config.n_layers) |i| {
            kv_caches[i] = try kv_cache.KVCache.init(
                allocator,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                config.max_seq_len,
            );
            errdefer {
                for (0..i) |j| kv_caches[j].deinit();
                allocator.free(kv_caches);
            }
        }

        std.debug.print("   ✅ Model initialized\n", .{});

        return LlamaModel{
            .allocator = allocator,
            .config = config,
            .weights = weights,
            .rope_freqs = rope_freqs,
            .kv_caches = kv_caches,
            .tok = tok,
            .pool = pool,
            .backend = backend,
        };
    }

    pub fn deinit(self: *LlamaModel) void {
        self.allocator.free(self.rope_freqs);
        for (self.kv_caches) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.kv_caches);
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
                @intCast(layer_idx),
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
            }, // Added Q4_K support
        }

        try self.backend.matmul(logits, w_ptr, w_type, layer_output, self.config.vocab_size, 1, embed_dim);

        self.allocator.free(hidden);

        return logits;
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
        std.debug.print("\n🔮 Generating text...\n", .{});
        std.debug.print("   Prompt: \"{s}\"\n", .{prompt});
        std.debug.print("   Max tokens: {d}\n", .{max_tokens});
        std.debug.print("   Temperature: {d:.2}\n", .{temperature});

        // Reset caches
        self.resetCaches();

        // Encode prompt
        const prompt_tokens = try self.tok.encode(prompt, self.allocator);
        defer self.allocator.free(prompt_tokens);

        std.debug.print("   Prompt tokens: {d}\n", .{prompt_tokens.len});

        // Process prompt tokens
        for (prompt_tokens, 0..) |token, pos| {
            const logits = try self.forward(token, @intCast(pos));
            self.allocator.free(logits);
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

        std.debug.print("   Generating", .{});

        while (gen_count < max_tokens) {
            // Get next token
            const prev_token = if (gen_count == 0)
                prompt_tokens[prompt_tokens.len - 1]
            else
                generated_tokens[gen_count - 1];

            const logits = try self.forward(prev_token, @intCast(position));
            defer self.allocator.free(logits);

            // Calculate probabilities
            const probs = try self.allocator.alloc(f32, self.config.vocab_size);
            defer self.allocator.free(probs);

            tokenizer.calculateProbs(probs, logits, temperature);

            // Apply sampling
            if (top_k) |k| {
                tokenizer.topK(probs, k);
            }
            if (top_p) |p| {
                tokenizer.topP(probs, p);
            }

            // Sample next token
            const next_token = tokenizer.sampleToken(probs, random);
            generated_tokens[gen_count] = next_token;
            gen_count += 1;

            // Check for EOS
            if (next_token == self.tok.eos_token) {
                std.debug.print(" [EOS]\n", .{});
                break;
            }

            // Progress indicator
            if (gen_count % 10 == 0) {
                std.debug.print(".", .{});
            }

            self.advanceCaches();
            position += 1;
        }

        std.debug.print("   Generated {d} tokens\n", .{gen_count});

        // Decode generated tokens
        const output = try self.tok.decode(generated_tokens[0..gen_count], self.allocator);

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
        },
        .tensors = &[_]gguf.TensorInfo{},
        .vocab_tokens = &[_][]u8{},
        .vocab_scores = &[_]f32{},
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
