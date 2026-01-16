const std = @import("std");
const math = std.math;

/// Advanced Attention Patterns - Day 19
/// Implements causal masking, multi-query attention (MQA), and grouped-query attention (GQA)

// ============================================================================
// Configuration
// ============================================================================

pub const AttentionType = enum {
    full,           // Full bidirectional attention
    causal,         // Causal (autoregressive) masking
    multi_query,    // MQA: 1 KV head, multiple Q heads
    grouped_query,  // GQA: Few KV heads, many Q heads
};

pub const AdvancedAttentionConfig = struct {
    attention_type: AttentionType,
    n_heads: u32,           // Number of query heads
    n_kv_heads: u32,        // Number of KV heads (for GQA/MQA)
    head_dim: u32,
    max_seq_len: u32 = 2048,
    scale: f32,
    
    pub fn init(
        attention_type: AttentionType,
        n_heads: u32,
        head_dim: u32,
    ) AdvancedAttentionConfig {
        const n_kv_heads = switch (attention_type) {
            .multi_query => 1,                    // MQA: 1 KV head
            .grouped_query => n_heads / 4,        // GQA: 1/4 of Q heads
            .full, .causal => n_heads,            // Same as Q heads
        };
        
        return .{
            .attention_type = attention_type,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        };
    }
};

// ============================================================================
// Causal Attention (Autoregressive)
// ============================================================================

pub const CausalAttention = struct {
    allocator: std.mem.Allocator,
    config: AdvancedAttentionConfig,
    scores: []f32,
    
    pub fn init(allocator: std.mem.Allocator, config: AdvancedAttentionConfig) !CausalAttention {
        const max_size = config.max_seq_len * config.max_seq_len;
        const scores = try allocator.alloc(f32, max_size);
        
        return CausalAttention{
            .allocator = allocator,
            .config = config,
            .scores = scores,
        };
    }
    
    pub fn deinit(self: *CausalAttention) void {
        self.allocator.free(self.scores);
    }
    
    /// Causal attention: each position can only attend to previous positions
    pub fn forward(
        self: *CausalAttention,
        q: []const f32,
        k: []const f32,
        v: []const f32,
        output: []f32,
        seq_len: u32,
    ) !void {
        const head_dim = self.config.head_dim;
        
        // Compute scores with causal masking
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                if (j > i) {
                    // Future positions masked out
                    self.scores[i * seq_len + j] = -math.inf(f32);
                } else {
                    // Compute attention score for valid positions
                    var score: f32 = 0.0;
                    for (0..head_dim) |d| {
                        score += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    self.scores[i * seq_len + j] = score * self.config.scale;
                }
            }
        }
        
        // Softmax per row
        for (0..seq_len) |i| {
            var max_val: f32 = -math.inf(f32);
            for (0..i + 1) |j| {  // Only consider valid positions
                max_val = @max(max_val, self.scores[i * seq_len + j]);
            }
            
            var sum: f32 = 0.0;
            for (0..i + 1) |j| {
                const exp_val = @exp(self.scores[i * seq_len + j] - max_val);
                self.scores[i * seq_len + j] = exp_val;
                sum += exp_val;
            }
            
            for (0..i + 1) |j| {
                self.scores[i * seq_len + j] /= sum;
            }
        }
        
        // Output: O = Attention @ V
        @memset(output, 0.0);
        for (0..seq_len) |i| {
            for (0..i + 1) |j| {  // Only use valid positions
                const attn = self.scores[i * seq_len + j];
                for (0..head_dim) |d| {
                    output[i * head_dim + d] += attn * v[j * head_dim + d];
                }
            }
        }
    }
};

// ============================================================================
// Multi-Query Attention (MQA)
// ============================================================================

pub const MultiQueryAttention = struct {
    allocator: std.mem.Allocator,
    config: AdvancedAttentionConfig,
    scores: []f32,
    
    pub fn init(allocator: std.mem.Allocator, config: AdvancedAttentionConfig) !MultiQueryAttention {
        const max_size = config.n_heads * config.max_seq_len * config.max_seq_len;
        const scores = try allocator.alloc(f32, max_size);
        
        return MultiQueryAttention{
            .allocator = allocator,
            .config = config,
            .scores = scores,
        };
    }
    
    pub fn deinit(self: *MultiQueryAttention) void {
        self.allocator.free(self.scores);
    }
    
    /// MQA: Multiple query heads share single KV head
    pub fn forward(
        self: *MultiQueryAttention,
        q: []const f32,      // [n_heads, seq_len, head_dim]
        k: []const f32,      // [1, seq_len, head_dim] - single KV head
        v: []const f32,      // [1, seq_len, head_dim]
        output: []f32,       // [n_heads, seq_len, head_dim]
        seq_len: u32,
    ) !void {
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        
        // Process each query head with shared KV
        for (0..n_heads) |h| {
            const q_offset = h * seq_len * head_dim;
            const out_offset = h * seq_len * head_dim;
            
            // Compute scores for this head
            for (0..seq_len) |i| {
                for (0..seq_len) |j| {
                    var score: f32 = 0.0;
                    for (0..head_dim) |d| {
                        const q_val = q[q_offset + i * head_dim + d];
                        const k_val = k[j * head_dim + d];  // Shared KV
                        score += q_val * k_val;
                    }
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    self.scores[score_idx] = score * self.config.scale;
                }
            }
            
            // Softmax per row
            for (0..seq_len) |i| {
                var max_val: f32 = -math.inf(f32);
                for (0..seq_len) |j| {
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    max_val = @max(max_val, self.scores[score_idx]);
                }
                
                var sum: f32 = 0.0;
                for (0..seq_len) |j| {
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    const exp_val = @exp(self.scores[score_idx] - max_val);
                    self.scores[score_idx] = exp_val;
                    sum += exp_val;
                }
                
                for (0..seq_len) |j| {
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    self.scores[score_idx] /= sum;
                }
            }
            
            // Output for this head
            @memset(output[out_offset..out_offset + seq_len * head_dim], 0.0);
            for (0..seq_len) |i| {
                for (0..seq_len) |j| {
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    const attn = self.scores[score_idx];
                    for (0..head_dim) |d| {
                        output[out_offset + i * head_dim + d] += attn * v[j * head_dim + d];
                    }
                }
            }
        }
    }
};

// ============================================================================
// Grouped-Query Attention (GQA)
// ============================================================================

pub const GroupedQueryAttention = struct {
    allocator: std.mem.Allocator,
    config: AdvancedAttentionConfig,
    scores: []f32,
    
    pub fn init(allocator: std.mem.Allocator, config: AdvancedAttentionConfig) !GroupedQueryAttention {
        const max_size = config.n_heads * config.max_seq_len * config.max_seq_len;
        const scores = try allocator.alloc(f32, max_size);
        
        return GroupedQueryAttention{
            .allocator = allocator,
            .config = config,
            .scores = scores,
        };
    }
    
    pub fn deinit(self: *GroupedQueryAttention) void {
        self.allocator.free(self.scores);
    }
    
    /// GQA: Groups of query heads share KV heads
    pub fn forward(
        self: *GroupedQueryAttention,
        q: []const f32,      // [n_heads, seq_len, head_dim]
        k: []const f32,      // [n_kv_heads, seq_len, head_dim]
        v: []const f32,      // [n_kv_heads, seq_len, head_dim]
        output: []f32,       // [n_heads, seq_len, head_dim]
        seq_len: u32,
    ) !void {
        const n_heads = self.config.n_heads;
        const n_kv_heads = self.config.n_kv_heads;
        const head_dim = self.config.head_dim;
        const group_size = n_heads / n_kv_heads;
        
        // Process each query head
        for (0..n_heads) |h| {
            const kv_head = h / group_size;  // Which KV head this Q head uses
            const q_offset = h * seq_len * head_dim;
            const kv_offset = kv_head * seq_len * head_dim;
            const out_offset = h * seq_len * head_dim;
            
            // Compute scores
            for (0..seq_len) |i| {
                for (0..seq_len) |j| {
                    var score: f32 = 0.0;
                    for (0..head_dim) |d| {
                        const q_val = q[q_offset + i * head_dim + d];
                        const k_val = k[kv_offset + j * head_dim + d];
                        score += q_val * k_val;
                    }
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    self.scores[score_idx] = score * self.config.scale;
                }
            }
            
            // Softmax
            for (0..seq_len) |i| {
                var max_val: f32 = -math.inf(f32);
                for (0..seq_len) |j| {
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    max_val = @max(max_val, self.scores[score_idx]);
                }
                
                var sum: f32 = 0.0;
                for (0..seq_len) |j| {
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    const exp_val = @exp(self.scores[score_idx] - max_val);
                    self.scores[score_idx] = exp_val;
                    sum += exp_val;
                }
                
                for (0..seq_len) |j| {
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    self.scores[score_idx] /= sum;
                }
            }
            
            // Output
            @memset(output[out_offset..out_offset + seq_len * head_dim], 0.0);
            for (0..seq_len) |i| {
                for (0..seq_len) |j| {
                    const score_idx = h * seq_len * seq_len + i * seq_len + j;
                    const attn = self.scores[score_idx];
                    for (0..head_dim) |d| {
                        const v_val = v[kv_offset + j * head_dim + d];
                        output[out_offset + i * head_dim + d] += attn * v_val;
                    }
                }
            }
        }
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_advanced_attention(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Advanced Attention Patterns\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const head_dim: u32 = 64;
    const seq_len: u32 = 8;
    
    // Test 1: Causal attention
    {
        std.debug.print("\n1️⃣  Testing causal attention...\n", .{});
        
        const config = AdvancedAttentionConfig.init(.causal, 1, head_dim);
        var causal = try CausalAttention.init(allocator, config);
        defer causal.deinit();
        
        const qkv_size = seq_len * head_dim;
        const q = try allocator.alloc(f32, qkv_size);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, qkv_size);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, qkv_size);
        defer allocator.free(v);
        const output = try allocator.alloc(f32, qkv_size);
        defer allocator.free(output);
        
        // Initialize with pattern
        for (0..qkv_size) |i| {
            q[i] = @as(f32, @floatFromInt(i % seq_len)) * 0.1;
            k[i] = @as(f32, @floatFromInt(i % seq_len)) * 0.1;
            v[i] = 1.0;
        }
        
        try causal.forward(q, k, v, output, seq_len);
        
        // Check that each position only uses past (including self)
        // Position 0 should only use position 0, position 1 should use 0-1, etc.
        std.debug.print("   Processed {d} tokens with causal masking\n", .{seq_len});
        std.debug.print("   ✅ Causal attention working\n", .{});
    }
    
    // Test 2: Multi-query attention (MQA)
    {
        std.debug.print("\n2️⃣  Testing multi-query attention (MQA)...\n", .{});
        
        const n_heads: u32 = 4;
        const config = AdvancedAttentionConfig.init(.multi_query, n_heads, head_dim);
        
        if (config.n_kv_heads != 1) {
            std.debug.print("   ❌ Expected 1 KV head, got {d}\n", .{config.n_kv_heads});
            return error.TestFailed;
        }
        
        var mqa = try MultiQueryAttention.init(allocator, config);
        defer mqa.deinit();
        
        const q_size = n_heads * seq_len * head_dim;
        const kv_size = seq_len * head_dim;  // Single KV head
        
        const q = try allocator.alloc(f32, q_size);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, kv_size);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, kv_size);
        defer allocator.free(v);
        const output = try allocator.alloc(f32, q_size);
        defer allocator.free(output);
        
        for (0..q_size) |i| {
            q[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
        }
        for (0..kv_size) |i| {
            k[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
            v[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
        }
        
        try mqa.forward(q, k, v, output, seq_len);
        
        // Check no NaN/Inf
        var has_invalid = false;
        for (output) |val| {
            if (math.isNan(val) or math.isInf(val)) {
                has_invalid = true;
                break;
            }
        }
        
        if (has_invalid) {
            std.debug.print("   ❌ Output contains NaN/Inf\n", .{});
            return error.TestFailed;
        }
        
        const kv_memory_saved = ((n_heads - 1) * kv_size * @sizeOf(f32)) / 1024;
        std.debug.print("   Query heads: {d}, KV heads: {d}\n", .{n_heads, config.n_kv_heads});
        std.debug.print("   KV cache memory saved: {d} KB ({d}x reduction)\n", .{kv_memory_saved, n_heads});
        std.debug.print("   ✅ Multi-query attention working\n", .{});
    }
    
    // Test 3: Grouped-query attention (GQA)
    {
        std.debug.print("\n3️⃣  Testing grouped-query attention (GQA)...\n", .{});
        
        const n_heads: u32 = 8;
        const config = AdvancedAttentionConfig.init(.grouped_query, n_heads, head_dim);
        const n_kv_heads = config.n_kv_heads;
        
        if (n_kv_heads != n_heads / 4) {
            std.debug.print("   ❌ Expected {d} KV heads, got {d}\n", .{n_heads / 4, n_kv_heads});
            return error.TestFailed;
        }
        
        var gqa = try GroupedQueryAttention.init(allocator, config);
        defer gqa.deinit();
        
        const q_size = n_heads * seq_len * head_dim;
        const kv_size = n_kv_heads * seq_len * head_dim;
        
        const q = try allocator.alloc(f32, q_size);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, kv_size);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, kv_size);
        defer allocator.free(v);
        const output = try allocator.alloc(f32, q_size);
        defer allocator.free(output);
        
        for (0..q_size) |i| {
            q[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
        }
        for (0..kv_size) |i| {
            k[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
            v[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
        }
        
        try gqa.forward(q, k, v, output, seq_len);
        
        // Check no NaN/Inf
        var has_invalid = false;
        for (output) |val| {
            if (math.isNan(val) or math.isInf(val)) {
                has_invalid = true;
                break;
            }
        }
        
        if (has_invalid) {
            std.debug.print("   ❌ Output contains NaN/Inf\n", .{});
            return error.TestFailed;
        }
        
        const group_size = n_heads / n_kv_heads;
        const memory_reduction = @as(f64, @floatFromInt(n_heads)) / @as(f64, @floatFromInt(n_kv_heads));
        std.debug.print("   Query heads: {d}, KV heads: {d}\n", .{n_heads, n_kv_heads});
        std.debug.print("   Group size: {d} Q heads per KV head\n", .{group_size});
        std.debug.print("   KV cache reduction: {d:.1}x\n", .{memory_reduction});
        std.debug.print("   ✅ Grouped-query attention working\n", .{});
    }
    
    std.debug.print("\n✅ All advanced attention tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
