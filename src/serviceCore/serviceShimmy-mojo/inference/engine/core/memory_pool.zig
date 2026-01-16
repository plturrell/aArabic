const std = @import("std");

/// Memory Pool for Inference
/// Pre-allocates scratch buffers to eliminate per-token heap allocations

pub const MemoryPool = struct {
    allocator: std.mem.Allocator,
    
    // Pre-allocated scratch buffers
    hidden_state_buffer: []f32,
    logits_buffer: []f32,
    attention_buffer: []f32,
    temp_buffer: []f32,
    
    // Buffer dimensions
    max_batch_size: u32,
    hidden_dim: u32,
    vocab_size: u32,
    max_seq_len: u32,
    
    // Statistics
    forward_calls: u64 = 0,
    allocations_saved: u64 = 0,
    
    pub fn init(
        allocator: std.mem.Allocator,
        max_batch_size: u32,
        hidden_dim: u32,
        vocab_size: u32,
        max_seq_len: u32,
    ) !MemoryPool {
        // Calculate buffer sizes
        const hidden_size = max_batch_size * hidden_dim;
        const logits_size = max_batch_size * vocab_size;
        const attn_size = max_batch_size * max_seq_len * max_seq_len;
        const temp_size = max_batch_size * @max(hidden_dim, vocab_size);
        
        // Allocate buffers
        const hidden_state_buffer = try allocator.alloc(f32, hidden_size);
        errdefer allocator.free(hidden_state_buffer);
        
        const logits_buffer = try allocator.alloc(f32, logits_size);
        errdefer allocator.free(logits_buffer);
        
        const attention_buffer = try allocator.alloc(f32, attn_size);
        errdefer allocator.free(attention_buffer);
        
        const temp_buffer = try allocator.alloc(f32, temp_size);
        errdefer allocator.free(temp_buffer);
        
        // Initialize to zero
        @memset(hidden_state_buffer, 0.0);
        @memset(logits_buffer, 0.0);
        @memset(attention_buffer, 0.0);
        @memset(temp_buffer, 0.0);
        
        return MemoryPool{
            .allocator = allocator,
            .hidden_state_buffer = hidden_state_buffer,
            .logits_buffer = logits_buffer,
            .attention_buffer = attention_buffer,
            .temp_buffer = temp_buffer,
            .max_batch_size = max_batch_size,
            .hidden_dim = hidden_dim,
            .vocab_size = vocab_size,
            .max_seq_len = max_seq_len,
        };
    }
    
    pub fn deinit(self: *MemoryPool) void {
        self.allocator.free(self.hidden_state_buffer);
        self.allocator.free(self.logits_buffer);
        self.allocator.free(self.attention_buffer);
        self.allocator.free(self.temp_buffer);
    }
    
    /// Get hidden state slice for a specific batch index
    pub fn get_hidden_slice(self: *MemoryPool, batch_idx: u32) []f32 {
        if (batch_idx >= self.max_batch_size) {
            @panic("Batch index out of bounds");
        }
        const start = batch_idx * self.hidden_dim;
        return self.hidden_state_buffer[start .. start + self.hidden_dim];
    }
    
    /// Get logits slice for a specific batch index
    pub fn get_logits_slice(self: *MemoryPool, batch_idx: u32) []f32 {
        if (batch_idx >= self.max_batch_size) {
            @panic("Batch index out of bounds");
        }
        const start = batch_idx * self.vocab_size;
        return self.logits_buffer[start .. start + self.vocab_size];
    }
    
    /// Get attention buffer slice for a specific batch index
    pub fn get_attention_slice(self: *MemoryPool, batch_idx: u32) []f32 {
        if (batch_idx >= self.max_batch_size) {
            @panic("Batch index out of bounds");
        }
        const start = batch_idx * self.max_seq_len * self.max_seq_len;
        const size = self.max_seq_len * self.max_seq_len;
        return self.attention_buffer[start .. start + size];
    }
    
    /// Get temporary buffer slice for a specific batch index
    pub fn get_temp_slice(self: *MemoryPool, batch_idx: u32, size: u32) []f32 {
        if (batch_idx >= self.max_batch_size) {
            @panic("Batch index out of bounds");
        }
        const max_size = @max(self.hidden_dim, self.vocab_size);
        if (size > max_size) {
            @panic("Requested size exceeds temp buffer capacity");
        }
        const start = batch_idx * max_size;
        return self.temp_buffer[start .. start + size];
    }
    
    /// Record a forward pass (for statistics)
    pub fn record_forward(self: *MemoryPool) void {
        self.forward_calls += 1;
        // Each forward call would have allocated:
        // - 1 hidden state buffer
        // - 1 logits buffer
        // - Multiple attention buffers
        self.allocations_saved += 3;
    }
    
    /// Get statistics
    pub fn get_stats(self: *const MemoryPool) PoolStats {
        return .{
            .forward_calls = self.forward_calls,
            .allocations_saved = self.allocations_saved,
            .total_memory_bytes = self.get_total_memory(),
            .hidden_buffer_bytes = self.hidden_state_buffer.len * @sizeOf(f32),
            .logits_buffer_bytes = self.logits_buffer.len * @sizeOf(f32),
            .attention_buffer_bytes = self.attention_buffer.len * @sizeOf(f32),
        };
    }
    
    fn get_total_memory(self: *const MemoryPool) usize {
        return (self.hidden_state_buffer.len +
                self.logits_buffer.len +
                self.attention_buffer.len +
                self.temp_buffer.len) * @sizeOf(f32);
    }
};

pub const PoolStats = struct {
    forward_calls: u64,
    allocations_saved: u64,
    total_memory_bytes: usize,
    hidden_buffer_bytes: usize,
    logits_buffer_bytes: usize,
    attention_buffer_bytes: usize,
    
    pub fn format_memory(bytes: usize) []const u8 {
        // Simple formatting helper
        if (bytes >= 1024 * 1024) {
            return "MB";
        } else if (bytes >= 1024) {
            return "KB";
        } else {
            return "B";
        }
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_memory_pool(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Memory Pool\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: Pool initialization
    {
        std.debug.print("\n1️⃣  Testing pool initialization...\n", .{});
        
        var pool = try MemoryPool.init(allocator, 4, 512, 32000, 2048);
        defer pool.deinit();
        
        std.debug.print("   Batch size: {d}\n", .{pool.max_batch_size});
        std.debug.print("   Hidden dim: {d}\n", .{pool.hidden_dim});
        std.debug.print("   Vocab size: {d}\n", .{pool.vocab_size});
        std.debug.print("   Max seq len: {d}\n", .{pool.max_seq_len});
        
        const stats = pool.get_stats();
        std.debug.print("   Total memory: {d} MB\n", .{stats.total_memory_bytes / (1024 * 1024)});
        std.debug.print("   ✅ Pool initialization working\n", .{});
    }
    
    // Test 2: Buffer slicing
    {
        std.debug.print("\n2️⃣  Testing buffer slicing...\n", .{});
        
        var pool = try MemoryPool.init(allocator, 4, 512, 32000, 2048);
        defer pool.deinit();
        
        // Get slices for different batches
        const hidden0 = pool.get_hidden_slice(0);
        const hidden1 = pool.get_hidden_slice(1);
        const logits0 = pool.get_logits_slice(0);
        
        std.debug.print("   Hidden[0] len: {d}\n", .{hidden0.len});
        std.debug.print("   Hidden[1] len: {d}\n", .{hidden1.len});
        std.debug.print("   Logits[0] len: {d}\n", .{logits0.len});
        
        // Write to slices
        hidden0[0] = 1.0;
        hidden1[0] = 2.0;
        logits0[0] = 3.0;
        
        if (hidden0[0] != 1.0 or hidden1[0] != 2.0 or logits0[0] != 3.0) {
            std.debug.print("   ❌ Buffer slicing failed\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Buffer slicing working\n", .{});
    }
    
    // Test 3: Statistics tracking
    {
        std.debug.print("\n3️⃣  Testing statistics tracking...\n", .{});
        
        var pool = try MemoryPool.init(allocator, 4, 512, 32000, 2048);
        defer pool.deinit();
        
        // Simulate some forward passes
        for (0..10) |_| {
            pool.record_forward();
        }
        
        const stats = pool.get_stats();
        std.debug.print("   Forward calls: {d}\n", .{stats.forward_calls});
        std.debug.print("   Allocations saved: {d}\n", .{stats.allocations_saved});
        
        if (stats.forward_calls != 10 or stats.allocations_saved != 30) {
            std.debug.print("   ❌ Statistics tracking failed\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Statistics tracking working\n", .{});
    }
    
    std.debug.print("\n✅ All memory pool tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
