const std = @import("std");
const math = std.math;

/// Flash Attention Implementation - Day 18
/// Memory-efficient attention with tiling and online softmax
/// Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

// ============================================================================
// Configuration
// ============================================================================

pub const FlashAttentionConfig = struct {
    n_heads: u32,
    head_dim: u32,
    block_size_q: u32 = 64,  // Query block size (tunable)
    block_size_kv: u32 = 64, // KV block size (tunable)
    scale: f32,              // 1/sqrt(head_dim)
    
    pub fn init(n_heads: u32, head_dim: u32) FlashAttentionConfig {
        return .{
            .n_heads = n_heads,
            .head_dim = head_dim,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        };
    }
};

// ============================================================================
// Flash Attention Context
// ============================================================================

pub const FlashAttention = struct {
    allocator: std.mem.Allocator,
    config: FlashAttentionConfig,
    
    // Workspace buffers (reused across calls)
    q_block: []f32,      // Query block [block_size_q, head_dim]
    k_block: []f32,      // Key block [block_size_kv, head_dim]
    v_block: []f32,      // Value block [block_size_kv, head_dim]
    s_block: []f32,      // Scores [block_size_q, block_size_kv]
    o_block: []f32,      // Output accumulator [block_size_q, head_dim]
    
    // Online statistics
    m_block: []f32,      // Row-wise max [block_size_q]
    l_block: []f32,      // Row-wise sum exp [block_size_q]
    
    pub fn init(allocator: std.mem.Allocator, config: FlashAttentionConfig) !FlashAttention {
        const q_size = config.block_size_q * config.head_dim;
        const kv_size = config.block_size_kv * config.head_dim;
        const s_size = config.block_size_q * config.block_size_kv;
        const o_size = config.block_size_q * config.head_dim;
        const stat_size = config.block_size_q;
        
        const q_block = try allocator.alloc(f32, q_size);
        errdefer allocator.free(q_block);
        
        const k_block = try allocator.alloc(f32, kv_size);
        errdefer allocator.free(k_block);
        
        const v_block = try allocator.alloc(f32, kv_size);
        errdefer allocator.free(v_block);
        
        const s_block = try allocator.alloc(f32, s_size);
        errdefer allocator.free(s_block);
        
        const o_block = try allocator.alloc(f32, o_size);
        errdefer allocator.free(o_block);
        
        const m_block = try allocator.alloc(f32, stat_size);
        errdefer allocator.free(m_block);
        
        const l_block = try allocator.alloc(f32, stat_size);
        errdefer allocator.free(l_block);
        
        return FlashAttention{
            .allocator = allocator,
            .config = config,
            .q_block = q_block,
            .k_block = k_block,
            .v_block = v_block,
            .s_block = s_block,
            .o_block = o_block,
            .m_block = m_block,
            .l_block = l_block,
        };
    }
    
    pub fn deinit(self: *FlashAttention) void {
        self.allocator.free(self.q_block);
        self.allocator.free(self.k_block);
        self.allocator.free(self.v_block);
        self.allocator.free(self.s_block);
        self.allocator.free(self.o_block);
        self.allocator.free(self.m_block);
        self.allocator.free(self.l_block);
    }
    
    /// Forward pass: O = softmax(Q * K^T / sqrt(d)) * V
    /// Uses tiled computation to minimize HBM traffic
    pub fn forward(
        self: *FlashAttention,
        q: []const f32,      // [seq_len_q, head_dim]
        k: []const f32,      // [seq_len_kv, head_dim]
        v: []const f32,      // [seq_len_kv, head_dim]
        output: []f32,       // [seq_len_q, head_dim]
        seq_len_q: u32,
        seq_len_kv: u32,
    ) !void {
        if (q.len != seq_len_q * self.config.head_dim) return error.InvalidQuerySize;
        if (k.len != seq_len_kv * self.config.head_dim) return error.InvalidKeySize;
        if (v.len != seq_len_kv * self.config.head_dim) return error.InvalidValueSize;
        if (output.len != seq_len_q * self.config.head_dim) return error.InvalidOutputSize;
        
        // Initialize output to zero
        @memset(output, 0.0);
        
        // Process in blocks to fit in SRAM
        const num_q_blocks = (seq_len_q + self.config.block_size_q - 1) / self.config.block_size_q;
        const num_kv_blocks = (seq_len_kv + self.config.block_size_kv - 1) / self.config.block_size_kv;
        
        for (0..num_q_blocks) |q_block_idx| {
            const q_start = @as(u32, @intCast(q_block_idx)) * self.config.block_size_q;
            const q_end = @min(q_start + self.config.block_size_q, seq_len_q);
            const q_block_size = q_end - q_start;
            
            // Load query block
            try self.loadBlock(q, self.q_block, q_start, q_block_size, self.config.head_dim);
            
            // Initialize statistics for this query block
            @memset(self.m_block[0..q_block_size], -math.inf(f32));
            @memset(self.l_block[0..q_block_size], 0.0);
            @memset(self.o_block[0..q_block_size * self.config.head_dim], 0.0);
            
            // Process KV blocks
            for (0..num_kv_blocks) |kv_block_idx| {
                const kv_start = @as(u32, @intCast(kv_block_idx)) * self.config.block_size_kv;
                const kv_end = @min(kv_start + self.config.block_size_kv, seq_len_kv);
                const kv_block_size = kv_end - kv_start;
                
                // Load KV block
                try self.loadBlock(k, self.k_block, kv_start, kv_block_size, self.config.head_dim);
                try self.loadBlock(v, self.v_block, kv_start, kv_block_size, self.config.head_dim);
                
                // Compute S = Q @ K^T (with scaling)
                try self.computeScores(q_block_size, kv_block_size);
                
                // Online softmax: update statistics and accumulate
                try self.onlineSoftmax(q_block_size, kv_block_size);
            }
            
            // Finalize output block: O = O / l
            try self.finalizeOutput(q_block_size);
            
            // Store output block
            try self.storeBlock(self.o_block, output, q_start, q_block_size, self.config.head_dim);
        }
    }
    
    /// Load a block from source to workspace
    fn loadBlock(
        self: *FlashAttention,
        src: []const f32,
        dst: []f32,
        start_row: u32,
        num_rows: u32,
        row_size: u32,
    ) !void {
        _ = self;
        for (0..num_rows) |i| {
            const src_offset = (start_row + @as(u32, @intCast(i))) * row_size;
            const dst_offset = @as(u32, @intCast(i)) * row_size;
            @memcpy(dst[dst_offset..dst_offset + row_size], src[src_offset..src_offset + row_size]);
        }
    }
    
    /// Store a block from workspace to destination
    fn storeBlock(
        self: *FlashAttention,
        src: []const f32,
        dst: []f32,
        start_row: u32,
        num_rows: u32,
        row_size: u32,
    ) !void {
        _ = self;
        for (0..num_rows) |i| {
            const src_offset = @as(u32, @intCast(i)) * row_size;
            const dst_offset = (start_row + @as(u32, @intCast(i))) * row_size;
            @memcpy(dst[dst_offset..dst_offset + row_size], src[src_offset..src_offset + row_size]);
        }
    }
    
    /// Compute attention scores: S = Q @ K^T * scale
    fn computeScores(self: *FlashAttention, q_rows: u32, kv_rows: u32) !void {
        // S[i,j] = sum_k(Q[i,k] * K[j,k]) * scale
        for (0..q_rows) |i| {
            for (0..kv_rows) |j| {
                var score: f32 = 0.0;
                for (0..self.config.head_dim) |k| {
                    const q_val = self.q_block[i * self.config.head_dim + k];
                    const k_val = self.k_block[j * self.config.head_dim + k];
                    score += q_val * k_val;
                }
                self.s_block[i * self.config.block_size_kv + j] = score * self.config.scale;
            }
        }
    }
    
    /// Online softmax with accumulation
    fn onlineSoftmax(self: *FlashAttention, q_rows: u32, kv_rows: u32) !void {
        for (0..q_rows) |i| {
            // Find max in current block
            var m_new: f32 = -math.inf(f32);
            for (0..kv_rows) |j| {
                const s_ij = self.s_block[i * self.config.block_size_kv + j];
                m_new = @max(m_new, s_ij);
            }
            
            const m_old = self.m_block[i];
            const l_old = self.l_block[i];
            
            // Update max
            const m_global = @max(m_old, m_new);
            
            // Compute correction factors
            const exp_diff_old = @exp(m_old - m_global);
            
            // Update sum
            var l_new: f32 = 0.0;
            for (0..kv_rows) |j| {
                const s_ij = self.s_block[i * self.config.block_size_kv + j];
                l_new += @exp(s_ij - m_global);
            }
            
            const l_global = exp_diff_old * l_old + l_new;
            
            // Update output: O = (l_old * exp(m_old - m_global) * O_old + exp(S - m_global) * V) / l_global
            // First rescale old output
            const rescale_old = exp_diff_old * l_old;
            for (0..self.config.head_dim) |d| {
                self.o_block[i * self.config.head_dim + d] *= rescale_old;
            }
            
            // Add new contribution
            for (0..kv_rows) |j| {
                const s_ij = self.s_block[i * self.config.block_size_kv + j];
                const exp_s = @exp(s_ij - m_global);
                
                for (0..self.config.head_dim) |d| {
                    const v_val = self.v_block[j * self.config.head_dim + d];
                    self.o_block[i * self.config.head_dim + d] += exp_s * v_val;
                }
            }
            
            // Store updated statistics
            self.m_block[i] = m_global;
            self.l_block[i] = l_global;
        }
    }
    
    /// Finalize output by dividing by sum
    fn finalizeOutput(self: *FlashAttention, q_rows: u32) !void {
        for (0..q_rows) |i| {
            const l = self.l_block[i];
            if (l > 0.0) {
                for (0..self.config.head_dim) |d| {
                    self.o_block[i * self.config.head_dim + d] /= l;
                }
            }
        }
    }
};

// ============================================================================
// Standard (Naive) Attention for Comparison
// ============================================================================

pub const StandardAttention = struct {
    allocator: std.mem.Allocator,
    config: FlashAttentionConfig,
    scores: []f32,  // Full attention matrix [seq_len_q, seq_len_kv]
    
    pub fn init(allocator: std.mem.Allocator, config: FlashAttentionConfig, max_seq_len: u32) !StandardAttention {
        const scores = try allocator.alloc(f32, max_seq_len * max_seq_len);
        return StandardAttention{
            .allocator = allocator,
            .config = config,
            .scores = scores,
        };
    }
    
    pub fn deinit(self: *StandardAttention) void {
        self.allocator.free(self.scores);
    }
    
    pub fn forward(
        self: *StandardAttention,
        q: []const f32,
        k: []const f32,
        v: []const f32,
        output: []f32,
        seq_len_q: u32,
        seq_len_kv: u32,
    ) !void {
        if (q.len != seq_len_q * self.config.head_dim) return error.InvalidQuerySize;
        if (k.len != seq_len_kv * self.config.head_dim) return error.InvalidKeySize;
        if (v.len != seq_len_kv * self.config.head_dim) return error.InvalidValueSize;
        if (output.len != seq_len_q * self.config.head_dim) return error.InvalidOutputSize;
        
        // Step 1: Compute scores S = Q @ K^T * scale
        for (0..seq_len_q) |i| {
            for (0..seq_len_kv) |j| {
                var score: f32 = 0.0;
                for (0..self.config.head_dim) |k_idx| {
                    const q_val = q[i * self.config.head_dim + k_idx];
                    const k_val = k[j * self.config.head_dim + k_idx];
                    score += q_val * k_val;
                }
                self.scores[i * seq_len_kv + j] = score * self.config.scale;
            }
        }
        
        // Step 2: Softmax per row
        for (0..seq_len_q) |i| {
            // Find max
            var max_val: f32 = -math.inf(f32);
            for (0..seq_len_kv) |j| {
                max_val = @max(max_val, self.scores[i * seq_len_kv + j]);
            }
            
            // Exp and sum
            var sum: f32 = 0.0;
            for (0..seq_len_kv) |j| {
                const exp_val = @exp(self.scores[i * seq_len_kv + j] - max_val);
                self.scores[i * seq_len_kv + j] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (0..seq_len_kv) |j| {
                self.scores[i * seq_len_kv + j] /= sum;
            }
        }
        
        // Step 3: Output O = Attention @ V
        @memset(output, 0.0);
        for (0..seq_len_q) |i| {
            for (0..seq_len_kv) |j| {
                const attn_weight = self.scores[i * seq_len_kv + j];
                for (0..self.config.head_dim) |d| {
                    output[i * self.config.head_dim + d] += attn_weight * v[j * self.config.head_dim + d];
                }
            }
        }
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_flash_attention(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Flash Attention Module\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const head_dim: u32 = 64;
    const config = FlashAttentionConfig.init(1, head_dim);
    
    // Test 1: Small attention (correctness)
    {
        std.debug.print("\n1️⃣  Testing correctness vs standard attention...\n", .{});
        
        const seq_len: u32 = 16;
        const qkv_size = seq_len * head_dim;
        
        // Create test data
        const q = try allocator.alloc(f32, qkv_size);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, qkv_size);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, qkv_size);
        defer allocator.free(v);
        
        // Initialize with simple pattern
        for (0..seq_len) |i| {
            for (0..head_dim) |d| {
                const idx = i * head_dim + d;
                q[idx] = @as(f32, @floatFromInt(i)) * 0.1;
                k[idx] = @as(f32, @floatFromInt(i)) * 0.1;
                v[idx] = @as(f32, @floatFromInt(d)) * 0.01;
            }
        }
        
        // Flash attention
        var flash = try FlashAttention.init(allocator, config);
        defer flash.deinit();
        
        const flash_output = try allocator.alloc(f32, qkv_size);
        defer allocator.free(flash_output);
        
        try flash.forward(q, k, v, flash_output, seq_len, seq_len);
        
        // Standard attention
        var standard = try StandardAttention.init(allocator, config, 256);
        defer standard.deinit();
        
        const std_output = try allocator.alloc(f32, qkv_size);
        defer allocator.free(std_output);
        
        try standard.forward(q, k, v, std_output, seq_len, seq_len);
        
        // Compare outputs
        var max_diff: f32 = 0.0;
        for (0..qkv_size) |i| {
            const diff = @abs(flash_output[i] - std_output[i]);
            max_diff = @max(max_diff, diff);
        }
        
        std.debug.print("   Max difference: {d:.6}\n", .{max_diff});
        
        if (max_diff > 1e-4) {
            std.debug.print("   ❌ Outputs differ too much\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Correctness verified (diff < 1e-4)\n", .{});
    }
    
    // Test 2: Memory efficiency
    {
        std.debug.print("\n2️⃣  Testing memory efficiency...\n", .{});
        
        const seq_len: u32 = 512;  // Longer sequence to show savings
        
        // Flash attention memory
        var flash = try FlashAttention.init(allocator, config);
        defer flash.deinit();
        
        const flash_workspace = flash.q_block.len + flash.k_block.len + flash.v_block.len + 
                               flash.s_block.len + flash.o_block.len + 
                               flash.m_block.len + flash.l_block.len;
        
        // Standard attention memory (full attention matrix)
        const std_memory = seq_len * seq_len;
        
        const flash_kb = (flash_workspace * @sizeOf(f32)) / 1024;
        const std_kb = (std_memory * @sizeOf(f32)) / 1024;
        const savings = (1.0 - @as(f64, @floatFromInt(flash_workspace)) / @as(f64, @floatFromInt(std_memory))) * 100.0;
        
        std.debug.print("   Flash workspace: {d} KB ({d} elements)\n", .{flash_kb, flash_workspace});
        std.debug.print("   Standard memory: {d} KB ({d} elements)\n", .{std_kb, std_memory});
        std.debug.print("   Memory savings: {d:.1}%\n", .{savings});
        
        // Flash should use significantly less memory for long sequences
        if (savings < 90.0) {
            std.debug.print("   ❌ Expected >90% memory savings for seq_len={d}\n", .{seq_len});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Memory efficiency verified\n", .{});
    }
    
    // Test 3: Block processing
    {
        std.debug.print("\n3️⃣  Testing block tiling...\n", .{});
        
        // Use sequence length not divisible by block size
        const seq_len: u32 = 100;
        const qkv_size = seq_len * head_dim;
        
        const q = try allocator.alloc(f32, qkv_size);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, qkv_size);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, qkv_size);
        defer allocator.free(v);
        
        // Random-ish data
        for (0..qkv_size) |i| {
            q[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
            k[i] = @as(f32, @floatFromInt((i * 7) % 100)) * 0.01;
            v[i] = @as(f32, @floatFromInt((i * 13) % 100)) * 0.01;
        }
        
        var flash = try FlashAttention.init(allocator, config);
        defer flash.deinit();
        
        const output = try allocator.alloc(f32, qkv_size);
        defer allocator.free(output);
        
        try flash.forward(q, k, v, output, seq_len, seq_len);
        
        // Check output is valid (no NaN/Inf)
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
        
        std.debug.print("   Processed {d} tokens with block_size={d}\n", .{seq_len, config.block_size_q});
        std.debug.print("   ✅ Block tiling working\n", .{});
    }
    
    std.debug.print("\n✅ All flash attention tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
