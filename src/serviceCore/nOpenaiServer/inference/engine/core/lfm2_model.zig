const std = @import("std");
const gguf = @import("gguf_loader");
const tokenizer = @import("tokenizer");
const matrix_ops = @import("matrix_ops");
const attention = @import("attention");
const kv_cache = @import("kv_cache");
const thread_pool = @import("thread_pool");

// LFM2 configuration derived from GGUF / config.json
pub const Lfm2Config = struct {
    vocab_size: u32,
    n_layers: u32,
    hidden_size: u32,
    intermediate_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    rope_theta: f32 = 1_000_000.0,
    norm_eps: f32 = 1e-5,
    conv_kernel: u32 = 3,
    conv_bias: bool = false,
    layer_types: []const LayerType,
    full_attn_idxs: []const u32,
};

pub const LayerType = enum { conv, ffn, full_attention };

pub const Lfm2LayerWeights = struct {
    // shortconv
    conv_weight: []const f32,        // [kernel, hidden]
    in_proj: matrix_ops.Weight,      // [hidden, 2*hidden]
    out_proj: matrix_ops.Weight,     // [hidden, hidden]

    // attention (optional per layer)
    attn_norm: []const f32,
    has_attn: bool,
    wq: matrix_ops.Weight,
    wk: matrix_ops.Weight,
    wv: matrix_ops.Weight,
    wo: matrix_ops.Weight,
    q_norm: []const f32,
    k_norm: []const f32,

    // FFN
    ffn_gate: matrix_ops.Weight,
    ffn_up: matrix_ops.Weight,
    ffn_down: matrix_ops.Weight,
    ffn_norm: []const f32,
};

pub const Lfm2Weights = struct {
    allocator: std.mem.Allocator,
    token_embedding: matrix_ops.Weight,
    output_weight: matrix_ops.Weight,
    output_tied: bool = true,
    layers: []Lfm2LayerWeights,

    pub fn deinit(self: *Lfm2Weights) void {
        // Free embedding only if not tied to external buffer
        switch (self.token_embedding) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => |data| self.allocator.free(data),
        }
        if (!self.output_tied) {
            switch (self.output_weight) {
                .f32 => |data| self.allocator.free(data),
                .q4_0 => |data| self.allocator.free(data),
                .q4_k => |data| self.allocator.free(data),
                .q6_k => |data| self.allocator.free(data),
            }
        }
        for (self.layers) |*lw| {
            switch (lw.in_proj) {
                .f32 => |data| self.allocator.free(data),
                .q4_0 => |data| self.allocator.free(data),
                .q4_k => |data| self.allocator.free(data),
                .q6_k => |data| self.allocator.free(data),
            }
            switch (lw.out_proj) {
                .f32 => |data| self.allocator.free(data),
                .q4_0 => |data| self.allocator.free(data),
                .q4_k => |data| self.allocator.free(data),
                .q6_k => |data| self.allocator.free(data),
            }
            if (lw.has_attn) {
                const freeW = struct {
                    fn go(w: matrix_ops.Weight, alloc: std.mem.Allocator) void {
                        switch (w) {
                            .f32 => |data| alloc.free(data),
                            .q4_0 => |data| alloc.free(data),
                            .q4_k => |data| alloc.free(data),
                            .q6_k => |data| alloc.free(data),
                        }
                    }
                }.go;
                freeW(lw.wq, self.allocator);
                freeW(lw.wk, self.allocator);
                freeW(lw.wv, self.allocator);
                freeW(lw.wo, self.allocator);
                self.allocator.free(lw.q_norm);
                self.allocator.free(lw.k_norm);
            }
            self.allocator.free(lw.attn_norm);
            self.allocator.free(lw.ffn_norm);
            switch (lw.ffn_gate) {
                .f32 => |data| self.allocator.free(data),
                .q4_0 => |data| self.allocator.free(data),
                .q4_k => |data| self.allocator.free(data),
                .q6_k => |data| self.allocator.free(data),
            }
            switch (lw.ffn_up) {
                .f32 => |data| self.allocator.free(data),
                .q4_0 => |data| self.allocator.free(data),
                .q4_k => |data| self.allocator.free(data),
                .q6_k => |data| self.allocator.free(data),
            }
            switch (lw.ffn_down) {
                .f32 => |data| self.allocator.free(data),
                .q4_0 => |data| self.allocator.free(data),
                .q4_k => |data| self.allocator.free(data),
                .q6_k => |data| self.allocator.free(data),
            }
            self.allocator.free(lw.conv_weight);
        }
        self.allocator.free(self.layers);
    }
};

pub const Lfm2Model = struct {
    allocator: std.mem.Allocator,
    config: Lfm2Config,
    weights: Lfm2Weights,
    tok: tokenizer.Tokenizer,
    pool: *thread_pool.ThreadPool,
    rope_freqs: []f32,
    kv_caches: []kv_cache.KVCache,
    conv_cache: [][]f32, // per-layer causal conv cache (kernel-1 x hidden)

    pub fn deinit(self: *Lfm2Model) void {
        self.allocator.free(self.config.layer_types);
        self.allocator.free(self.config.full_attn_idxs);
        self.allocator.free(self.rope_freqs);
        for (self.kv_caches) |*c| c.deinit();
        self.allocator.free(self.kv_caches);
        for (self.conv_cache) |buf| self.allocator.free(buf);
        self.allocator.free(self.conv_cache);
        self.weights.deinit();
        self.tok.deinit();
        self.pool.deinit();
        self.allocator.destroy(self.pool);
    }

    /// Initialize model (without inference wiring yet)
    pub fn init(
        allocator: std.mem.Allocator,
        config: Lfm2Config,
        weights: Lfm2Weights,
        tok: tokenizer.Tokenizer,
    ) !Lfm2Model {
        // Thread pool
        const pool_ptr = try allocator.create(thread_pool.ThreadPool);
        errdefer allocator.destroy(pool_ptr);
        pool_ptr.* = try thread_pool.ThreadPool.init(allocator, thread_pool.ThreadPoolConfig.default());
        try pool_ptr.start();

        // RoPE
        const rope_freqs = try attention.precomputeRopeFreqs(
            allocator,
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
        );
        errdefer allocator.free(rope_freqs);

        // KV caches
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
            }
        }

        // Conv cache (kernel-1 timesteps per layer)
        const cache_len = (config.conv_kernel - 1) * config.hidden_size;
        var conv_cache = try allocator.alloc([]f32, config.n_layers);
        errdefer allocator.free(conv_cache);
        for (0..config.n_layers) |i| {
            conv_cache[i] = try allocator.alloc(f32, cache_len);
            @memset(conv_cache[i], 0);
        }

        return .{
            .allocator = allocator,
            .config = config,
            .weights = weights,
            .tok = tok,
            .pool = pool_ptr,
            .rope_freqs = rope_freqs,
            .kv_caches = kv_caches,
            .conv_cache = conv_cache,
        };
    }

    /// Single-token forward (attention to be added in next step)
    pub fn forward(
        self: *Lfm2Model,
        input: []const f32, // hidden_size
        position: u32,
    ) ![]f32 {
        var hidden = try self.allocator.alloc(f32, self.config.hidden_size);
        @memcpy(hidden, input);
        errdefer self.allocator.free(hidden);

        try self.runLayers(&hidden, position);
        return hidden;
    }

    /// Forward from token id -> logits (weight tying by default)
    pub fn forwardToken(self: *Lfm2Model, token_id: u32, position: u32) ![]f32 {
        const t_start = std.time.nanoTimestamp();
        var hidden = try self.allocator.alloc(f32, self.config.hidden_size);
        errdefer self.allocator.free(hidden);
        matrix_ops.get_row(hidden, self.weights.token_embedding, token_id, self.config.hidden_size);
        const t_embed = std.time.nanoTimestamp();
        std.debug.print("   ⏱️  Embedding: {d}ms\n", .{@divFloor(t_embed - t_start, 1_000_000)});

        try self.runLayers(&hidden, position);
        const t_layers = std.time.nanoTimestamp();
        std.debug.print("   ⏱️  Layers: {d}ms\n", .{@divFloor(t_layers - t_embed, 1_000_000)});

        const logits = try self.allocator.alloc(f32, self.config.vocab_size);
        const head_weight: matrix_ops.Weight = if (self.weights.output_tied)
            self.weights.token_embedding
        else
            self.weights.output_weight;

        try matrix_ops.matmul(
            logits,
            head_weight,
            hidden,
            self.config.vocab_size,
            1,
            self.config.hidden_size,
            self.allocator,
            self.pool,
        );

        // Sanitize logits: replace NaN/Inf with large negative value
        for (logits) |*v| {
            if (std.math.isNan(v.*) or std.math.isInf(v.*)) {
                v.* = -1e10; // Very low probability but valid
            }
        }

        // Debug: Check for NaN in hidden and logits (after sanitization)
        var hidden_nan_count: usize = 0;
        var logits_nan_count: usize = 0;
        for (hidden) |v| {
            if (std.math.isNan(v)) hidden_nan_count += 1;
        }
        for (logits) |v| {
            if (std.math.isNan(v)) logits_nan_count += 1;
        }
        if (hidden_nan_count > 0 or logits_nan_count > 0) {
            std.debug.print("   ⚠️  NaN detected: hidden={d}/{d}, logits={d}/{d}\n", .{
                hidden_nan_count, hidden.len, logits_nan_count, logits.len
            });
        }

        self.allocator.free(hidden);
        return logits;
    }

    fn runLayers(self: *Lfm2Model, hidden: *[]f32, position: u32) !void {
        for (self.weights.layers, 0..) |*lw, idx| {
            const t_layer = std.time.nanoTimestamp();
            
            // ShortConv block (depthwise causal conv + GLU)
            try self.applyShortConv(lw, hidden, idx);
            const t_conv = std.time.nanoTimestamp();
            
            // Check for NaN after conv
            if (idx == 0) {
                var nan_count: usize = 0;
                for (hidden.*) |v| {
                    if (std.math.isNan(v)) nan_count += 1;
                }
                if (nan_count > 0) {
                    std.debug.print("   ❌ NaN after conv: {d}/{d}\n", .{ nan_count, hidden.len });
                }
            }

            // Optional attention
            if (lw.has_attn) {
                try self.applyAttention(lw, hidden, idx, position);
                
                // Check for NaN after attention
                if (idx == 0) {
                    var nan_count: usize = 0;
                    for (hidden.*) |v| {
                        if (std.math.isNan(v)) nan_count += 1;
                    }
                    if (nan_count > 0) {
                        std.debug.print("   ❌ NaN after attn: {d}/{d}\n", .{ nan_count, hidden.len });
                    }
                }
            }
            const t_attn = std.time.nanoTimestamp();

            // FFN
            try self.applyFfn(lw, hidden);
            const t_ffn = std.time.nanoTimestamp();
            
            // Check for NaN after FFN
            if (idx == 0) {
                var nan_count: usize = 0;
                for (hidden.*) |v| {
                    if (std.math.isNan(v)) nan_count += 1;
                }
                if (nan_count > 0) {
                    std.debug.print("   ❌ NaN after FFN: {d}/{d}\n", .{ nan_count, hidden.len });
                }
                
                std.debug.print("   ⏱️  Layer {d}: conv={d}ms attn={d}ms ffn={d}ms\n", .{
                    idx,
                    @divFloor(t_conv - t_layer, 1_000_000),
                    @divFloor(t_attn - t_conv, 1_000_000),
                    @divFloor(t_ffn - t_attn, 1_000_000),
                });
            }
        }
    }

    pub fn resetCaches(self: *Lfm2Model) void {
        for (self.kv_caches) |*cache| {
            cache.reset();
        }
        for (self.conv_cache) |buf| {
            @memset(buf, 0.0);
        }
    }

    pub fn advanceCaches(self: *Lfm2Model) void {
        for (self.kv_caches) |*cache| {
            cache.advance();
        }
    }

    fn applyShortConv(self: *Lfm2Model, lw: *const Lfm2LayerWeights, hidden: *[]f32, layer_idx: usize) !void {
        const hid = self.config.hidden_size;
        const kernel = self.config.conv_kernel;
        const cache = self.conv_cache[layer_idx];
        // If conv weights are absent (FFN-only layer), skip.
        if (lw.conv_weight.len == 0) {
            return;
        }
        const input = hidden.*;

        // Save residual input
        const residual = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(residual);
        @memcpy(residual, input);

        // Input buffer for conv (cache + current token)
        const window = kernel - 1;
        const total = window + 1;
        var conv_in = try self.allocator.alloc(f32, total * hid);
        defer self.allocator.free(conv_in);
        
        // Initialize to zero to prevent any uninitialized memory issues
        @memset(conv_in, 0.0);

        // Shift cache: cache[0..window-1] already holds previous tokens; append current
        // Cache layout: contiguous (window * hidden)
        // Copy cache (with NaN guard)
        for (cache, 0..) |v, i| {
            conv_in[i] = if (std.math.isNan(v) or std.math.isInf(v)) 0.0 else v;
        }
        // Append current hidden (with NaN guard)
        for (input, 0..) |v, i| {
            const val = if (std.math.isNan(v) or std.math.isInf(v)) 0.0 else v;
            conv_in[window * hid + i] = val;
        }

        // Depthwise causal conv: per-channel dot with kernel elements
        const conv_out = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(conv_out);

        // conv_weight shape: [kernel, hidden] (per-channel)
        for (0..hid) |c| {
            var acc: f32 = 0;
            var k: usize = 0;
            while (k < kernel) : (k += 1) {
                const w = lw.conv_weight[k * hid + c];
                const x = conv_in[(window - k) * hid + c]; // reversed for causal (padding already handled)
                
                // Guard against NaN/Inf in weights or inputs
                if (std.math.isNan(w) or std.math.isInf(w)) continue;
                if (std.math.isNan(x) or std.math.isInf(x)) continue;
                
                acc += w * x;
            }
            // Clamp accumulator to prevent overflow
            conv_out[c] = std.math.clamp(acc, -1e10, 1e10);
        }

        // Update cache: drop oldest, append current
        // Shift left by hid and append current hidden at end
        if (window > 0) {
            const cache_slice = cache[0 .. window * hid];
            if (window > 1) {
                // shift left by one slot (hid)
                @memmove(cache_slice[0.. (window - 1) * hid], cache_slice[hid .. window * hid]);
            }
            @memcpy(cache_slice[(window - 1) * hid ..], input);
        }

        // In-proj: [hidden, 2*hidden]
        const inproj_out = try self.allocator.alloc(f32, hid * 2);
        defer self.allocator.free(inproj_out);
        
        // Debug: Check conv_out for NaN before matmul
        if (layer_idx == 0) {
            var conv_nan: usize = 0;
            for (conv_out) |v| {
                if (std.math.isNan(v)) conv_nan += 1;
            }
            if (conv_nan > 0) std.debug.print("   🔴 NaN in conv_out: {d}/{d}\n", .{ conv_nan, conv_out.len });
            
            // Check in_proj weights for NaN
            var weight_nan: usize = 0;
            var weight_inf: usize = 0;
            switch (lw.in_proj) {
                .f32 => |data| {
                    for (data) |v| {
                        if (std.math.isNan(v)) weight_nan += 1;
                        if (std.math.isInf(v)) weight_inf += 1;
                    }
                },
                else => {},
            }
            if (weight_nan > 0 or weight_inf > 0) {
                std.debug.print("   🔴 in_proj weights: NaN={d}, Inf={d}\n", .{ weight_nan, weight_inf });
            }
        }
        
        try matrix_ops.matmul(inproj_out, lw.in_proj, conv_out, hid * 2, 1, hid, self.allocator, self.pool);
        
        // Debug: Check inproj_out for NaN after matmul
        if (layer_idx == 0) {
            var inproj_nan: usize = 0;
            for (inproj_out) |v| {
                if (std.math.isNan(v)) inproj_nan += 1;
            }
            if (inproj_nan > 0) std.debug.print("   🔴 NaN in inproj_out: {d}/{d}\n", .{ inproj_nan, inproj_out.len });
        }

        // GLU: split A,B with stable sigmoid
        const glu_out = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(glu_out);
        for (0..hid) |i| {
            const a = inproj_out[i];
            const b = inproj_out[hid + i];
            
            // Skip NaN/Inf inputs
            if (std.math.isNan(a) or std.math.isInf(a) or std.math.isNan(b) or std.math.isInf(b)) {
                glu_out[i] = 0.0;
                continue;
            }
            
            // Numerically stable sigmoid: clip to prevent overflow
            const b_clipped = std.math.clamp(b, -88.0, 88.0);
            const sb = 1.0 / (1.0 + @exp(-b_clipped));
            const result = a * sb;
            
            // Final guard: clamp result
            glu_out[i] = if (std.math.isNan(result) or std.math.isInf(result)) 
                0.0 
            else 
                std.math.clamp(result, -1e10, 1e10);
        }

        // Out-proj back to hidden
        const proj_out = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(proj_out);
        try matrix_ops.matmul(proj_out, lw.out_proj, glu_out, hid, 1, hid, self.allocator, self.pool);

        // Residual add with NaN guard
        for (0..hid) |i| {
            const res_val = residual[i];
            const proj_val = proj_out[i];
            
            if (std.math.isNan(res_val) or std.math.isInf(res_val) or 
                std.math.isNan(proj_val) or std.math.isInf(proj_val)) {
                hidden.*[i] = 0.0;
            } else {
                hidden.*[i] = res_val + proj_val;
            }
        }
    }

    fn applyFfn(self: *Lfm2Model, lw: *const Lfm2LayerWeights, hidden: *[]f32) !void {
        const hid = self.config.hidden_size;
        const ffn = self.config.intermediate_size;
        const residual = hidden.*;

        // Norm pre-ffn (RMSNorm)
        const normed = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(normed);
        matrix_ops.rms_norm(normed, hidden.*, lw.ffn_norm, self.config.norm_eps);

        // gate and up projections (weights pre-transposed during load)
        const gate_out = try self.allocator.alloc(f32, ffn);
        defer self.allocator.free(gate_out);
        const up_out = try self.allocator.alloc(f32, ffn);
        defer self.allocator.free(up_out);
        try matrix_ops.matmul(gate_out, lw.ffn_gate, normed, ffn, 1, hid, self.allocator, self.pool);
        try matrix_ops.matmul(up_out, lw.ffn_up, normed, ffn, 1, hid, self.allocator, self.pool);

        // SwiGLU: silu(gate) * up
        var ffn_act = try self.allocator.alloc(f32, ffn);
        defer self.allocator.free(ffn_act);
        for (0..ffn) |i| {
            const g = gate_out[i];
            const silu = g / (1.0 + @exp(-g));
            ffn_act[i] = silu * up_out[i];
        }

        // Down projection back to hidden (weights pre-transposed during load)
        const down = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(down);
        try matrix_ops.matmul(down, lw.ffn_down, ffn_act, hid, 1, ffn, self.allocator, self.pool);

        // Residual add
        for (0..hid) |i| {
            hidden.*[i] = residual[i] + down[i];
        }
    }

    fn applyAttention(
        self: *Lfm2Model,
        lw: *const Lfm2LayerWeights,
        hidden: *[]f32,
        layer_idx: usize,
        position: u32,
    ) !void {
        const hid = self.config.hidden_size;
        const head_dim = self.config.head_dim;
        const n_heads = self.config.n_heads;
        const n_kv_heads = self.config.n_kv_heads;
        const q_dim = n_heads * head_dim;
        const kv_dim = n_kv_heads * head_dim;
        const head_dim_usize = @as(usize, head_dim);
        const q_dim_usize = @as(usize, q_dim);
        const kv_dim_usize = @as(usize, kv_dim);
        const cache = &self.kv_caches[layer_idx];

        // Residual input
        const residual = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(residual);
        @memcpy(residual, hidden.*);

        // Pre-attn RMSNorm
        const normed = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(normed);
        matrix_ops.rms_norm(normed, hidden.*, lw.attn_norm, self.config.norm_eps);

        // Projections
        var q = try self.allocator.alloc(f32, q_dim_usize);
        defer self.allocator.free(q);
        var k = try self.allocator.alloc(f32, kv_dim_usize);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, kv_dim_usize);
        defer self.allocator.free(v);

        try matrix_ops.matmul(q, lw.wq, normed, q_dim, 1, hid, self.allocator, self.pool);
        try matrix_ops.matmul(k, lw.wk, normed, kv_dim, 1, hid, self.allocator, self.pool);
        try matrix_ops.matmul(v, lw.wv, normed, kv_dim, 1, hid, self.allocator, self.pool);

        // Q/K RMSNorm per head
        const q_normed = try self.allocator.alloc(f32, q_dim_usize);
        defer self.allocator.free(q_normed);
        const k_normed = try self.allocator.alloc(f32, kv_dim_usize);
        defer self.allocator.free(k_normed);

        for (0..n_heads) |h| {
            const src = q[h * head_dim_usize .. (h + 1) * head_dim_usize];
            const dst = q_normed[h * head_dim_usize .. (h + 1) * head_dim_usize];
            matrix_ops.rms_norm(dst, src, lw.q_norm, self.config.norm_eps);
        }
        for (0..n_kv_heads) |h| {
            const src = k[h * head_dim_usize .. (h + 1) * head_dim_usize];
            const dst = k_normed[h * head_dim_usize .. (h + 1) * head_dim_usize];
            matrix_ops.rms_norm(dst, src, lw.k_norm, self.config.norm_eps);
        }

        // Apply RoPE
        const q_rope = try self.allocator.alloc(f32, q_dim_usize);
        defer self.allocator.free(q_rope);
        const k_rope = try self.allocator.alloc(f32, kv_dim_usize);
        defer self.allocator.free(k_rope);

        for (0..n_heads) |h| {
            const src = q_normed[h * head_dim_usize .. (h + 1) * head_dim_usize];
            const dst = q_rope[h * head_dim_usize .. (h + 1) * head_dim_usize];
            attention.applyRope(dst, src, position, self.rope_freqs, head_dim);
        }
        for (0..n_kv_heads) |h| {
            const src = k_normed[h * head_dim_usize .. (h + 1) * head_dim_usize];
            const dst = k_rope[h * head_dim_usize .. (h + 1) * head_dim_usize];
            attention.applyRope(dst, src, position, self.rope_freqs, head_dim);
        }

        // Store into KV cache
        cache.store(@intCast(layer_idx), k_rope, v);

        // Attention computation (serial per head)
        const attn_out = try self.allocator.alloc(f32, q_dim_usize);
        defer self.allocator.free(attn_out);
        @memset(attn_out, 0);

        const seq_len = cache.getSequenceLength();
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const BLOCK: u32 = 128;

        const k_block = try self.allocator.alloc(f32, @as(usize, BLOCK) * head_dim_usize);
        defer self.allocator.free(k_block);
        const v_block = try self.allocator.alloc(f32, @as(usize, BLOCK) * head_dim_usize);
        defer self.allocator.free(v_block);
        const scores = try self.allocator.alloc(f32, BLOCK);
        defer self.allocator.free(scores);

        for (0..n_heads) |h| {
            const kv_head_idx: u32 = @intCast(h * n_kv_heads / n_heads);
            const q_head = q_rope[h * head_dim_usize .. (h + 1) * head_dim_usize];
            const out_head = attn_out[h * head_dim_usize .. (h + 1) * head_dim_usize];

            var max_score: f32 = -std.math.inf(f32);
            var sum_exp: f32 = 0.0;
            @memset(out_head, 0.0);

            var start: u32 = 0;
            while (start < seq_len) : (start += BLOCK) {
                const end = @min(start + BLOCK, seq_len);
                const current: u32 = end - start;

                cache.gatherHeadKeys(@intCast(layer_idx), kv_head_idx, start, end, k_block[0 .. @as(usize, current) * head_dim_usize]);
                cache.gatherHeadValues(@intCast(layer_idx), kv_head_idx, start, end, v_block[0 .. @as(usize, current) * head_dim_usize]);

                // Dot products
                var local_max: f32 = -std.math.inf(f32);
                for (0..current) |b_idx| {
                    var dot: f32 = 0;
                    const k_ptr = k_block[@as(usize, b_idx) * head_dim_usize .. @as(usize, b_idx + 1) * head_dim_usize];
                    for (0..head_dim_usize) |d| {
                        dot += q_head[d] * k_ptr[d];
                    }
                    const score = dot * scale;
                    scores[b_idx] = score;
                    if (score > local_max) local_max = score;
                }

                const new_max = @max(max_score, local_max);
                const factor_old = @exp(max_score - new_max);

                // Rescale accumulator
                for (out_head) |*val| {
                    val.* *= factor_old;
                }
                sum_exp *= factor_old;

                // Accumulate current block
                for (0..current) |b_idx| {
                    const score = scores[b_idx];
                    const weight = @exp(score - new_max);
                    sum_exp += weight;

                    const v_ptr = v_block[@as(usize, b_idx) * head_dim_usize .. @as(usize, b_idx + 1) * head_dim_usize];
                    for (0..head_dim_usize) |d| {
                        out_head[d] += weight * v_ptr[d];
                    }
                }

                max_score = new_max;
            }

            if (sum_exp > 0) {
                const inv_sum = 1.0 / sum_exp;
                matrix_ops.vec_scale(out_head, out_head, inv_sum);
            }
        }

        // Project back to hidden
        const attn_proj = try self.allocator.alloc(f32, hid);
        defer self.allocator.free(attn_proj);
        try matrix_ops.matmul(attn_proj, lw.wo, attn_out, hid, 1, q_dim, self.allocator, self.pool);

        for (0..hid) |i| {
            hidden.*[i] = residual[i] + attn_proj[i];
        }
    }
};
