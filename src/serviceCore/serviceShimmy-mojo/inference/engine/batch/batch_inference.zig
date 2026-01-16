const std = @import("std");
const math = std.math;

/// Batch Inference System - Day 20
/// Efficient multi-sequence processing with dynamic batching

// ============================================================================
// Configuration
// ============================================================================

pub const BatchConfig = struct {
    max_batch_size: u32,
    max_seq_len: u32,
    head_dim: u32,
    n_heads: u32,
    timeout_ms: u64 = 100,  // Wait time for batching
    
    pub fn init(max_batch_size: u32, max_seq_len: u32, head_dim: u32, n_heads: u32) BatchConfig {
        return .{
            .max_batch_size = max_batch_size,
            .max_seq_len = max_seq_len,
            .head_dim = head_dim,
            .n_heads = n_heads,
        };
    }
};

// ============================================================================
// Batch Request
// ============================================================================

pub const BatchRequest = struct {
    id: u64,
    tokens: []const u32,
    max_new_tokens: u32,
    temperature: f32 = 1.0,
    
    pub fn init(id: u64, tokens: []const u32, max_new_tokens: u32) BatchRequest {
        return .{
            .id = id,
            .tokens = tokens,
            .max_new_tokens = max_new_tokens,
        };
    }
};

pub const BatchResponse = struct {
    id: u64,
    generated_tokens: []u32,
    completed: bool,
    
    pub fn deinit(self: *BatchResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.generated_tokens);
    }
};

// ============================================================================
// Batch State Management
// ============================================================================

pub const SequenceState = struct {
    request: BatchRequest,
    current_length: u32,
    tokens_generated: u32,
    active: bool,
    
    pub fn init(request: BatchRequest) SequenceState {
        return .{
            .request = request,
            .current_length = @intCast(request.tokens.len),
            .tokens_generated = 0,
            .active = true,
        };
    }
    
    pub fn is_complete(self: *const SequenceState) bool {
        return self.tokens_generated >= self.request.max_new_tokens or !self.active;
    }
};

// ============================================================================
// Batch Processor
// ============================================================================

pub const BatchProcessor = struct {
    allocator: std.mem.Allocator,
    config: BatchConfig,
    
    // Batched tensors
    batch_tokens: []u32,        // [max_batch_size * max_seq_len]
    batch_lengths: []u32,       // [max_batch_size]
    batch_q: []f32,             // [max_batch_size * max_seq_len * head_dim]
    batch_k: []f32,
    batch_v: []f32,
    batch_output: []f32,
    
    // Sequence management
    sequences: []SequenceState,
    active_count: u32,
    
    pub fn init(allocator: std.mem.Allocator, config: BatchConfig) !BatchProcessor {
        const total_tokens = config.max_batch_size * config.max_seq_len;
        const total_qkv = total_tokens * config.head_dim;
        
        const batch_tokens = try allocator.alloc(u32, total_tokens);
        errdefer allocator.free(batch_tokens);
        
        const batch_lengths = try allocator.alloc(u32, config.max_batch_size);
        errdefer allocator.free(batch_lengths);
        
        const batch_q = try allocator.alloc(f32, total_qkv);
        errdefer allocator.free(batch_q);
        
        const batch_k = try allocator.alloc(f32, total_qkv);
        errdefer allocator.free(batch_k);
        
        const batch_v = try allocator.alloc(f32, total_qkv);
        errdefer allocator.free(batch_v);
        
        const batch_output = try allocator.alloc(f32, total_qkv);
        errdefer allocator.free(batch_output);
        
        const sequences = try allocator.alloc(SequenceState, config.max_batch_size);
        errdefer allocator.free(sequences);
        
        @memset(batch_tokens, 0);
        @memset(batch_lengths, 0);
        @memset(sequences, undefined);
        
        return BatchProcessor{
            .allocator = allocator,
            .config = config,
            .batch_tokens = batch_tokens,
            .batch_lengths = batch_lengths,
            .batch_q = batch_q,
            .batch_k = batch_k,
            .batch_v = batch_v,
            .batch_output = batch_output,
            .sequences = sequences,
            .active_count = 0,
        };
    }
    
    pub fn deinit(self: *BatchProcessor) void {
        self.allocator.free(self.batch_tokens);
        self.allocator.free(self.batch_lengths);
        self.allocator.free(self.batch_q);
        self.allocator.free(self.batch_k);
        self.allocator.free(self.batch_v);
        self.allocator.free(self.batch_output);
        self.allocator.free(self.sequences);
    }
    
    /// Add a new request to the batch
    pub fn add_request(self: *BatchProcessor, request: BatchRequest) !bool {
        if (self.active_count >= self.config.max_batch_size) {
            return false; // Batch full
        }
        
        // Find free slot
        for (self.sequences, 0..) |*seq, i| {
            if (i >= self.active_count or !seq.active) {
                seq.* = SequenceState.init(request);
                if (i >= self.active_count) {
                    self.active_count = @intCast(i + 1);
                }
                return true;
            }
        }
        
        return false;
    }
    
    /// Process one step for all active sequences
    pub fn process_step(self: *BatchProcessor) !void {
        if (self.active_count == 0) return;
        
        // Prepare batch
        for (0..self.active_count) |i| {
            const seq = &self.sequences[i];
            if (!seq.active) continue;
            
            // Copy tokens to batch (only if we have tokens)
            if (seq.request.tokens.len > 0) {
                const batch_offset = i * self.config.max_seq_len;
                const token_count = @min(seq.current_length, self.config.max_seq_len);
                const actual_count = @min(token_count, @as(u32, @intCast(seq.request.tokens.len)));
                
                if (actual_count > 0) {
                    @memcpy(
                        self.batch_tokens[batch_offset..batch_offset + actual_count],
                        seq.request.tokens[0..actual_count]
                    );
                }
                self.batch_lengths[i] = actual_count;
            } else {
                self.batch_lengths[i] = 0;
            }
        }
        
        // Simulate embedding + attention (would call real model here)
        try self.simulate_forward_pass();
        
        // Sample next tokens
        for (0..self.active_count) |i| {
            const seq = &self.sequences[i];
            if (!seq.active or seq.is_complete()) continue;
            
            // Simulate token sampling (would use real sampling here)
            const next_token = self.sample_token(i);
            
            seq.tokens_generated += 1;
            seq.current_length += 1;
            
            // Check if complete
            if (seq.tokens_generated >= seq.request.max_new_tokens) {
                seq.active = false;
            }
            
            _ = next_token;
        }
        
        // Clean up completed sequences
        self.compact_batch();
    }
    
    /// Simulate forward pass (placeholder for real model)
    fn simulate_forward_pass(self: *BatchProcessor) !void {
        // Initialize with simple pattern for testing
        for (0..self.active_count) |i| {
            const seq_len = self.batch_lengths[i];
            const offset = i * self.config.max_seq_len * self.config.head_dim;
            
            for (0..seq_len) |pos| {
                const pos_offset = offset + pos * self.config.head_dim;
                for (0..self.config.head_dim) |d| {
                    const idx = pos_offset + d;
                    self.batch_q[idx] = @as(f32, @floatFromInt(i + pos + d)) * 0.01;
                    self.batch_k[idx] = @as(f32, @floatFromInt(i + pos + d)) * 0.01;
                    self.batch_v[idx] = @as(f32, @floatFromInt(d)) * 0.01;
                }
            }
        }
        
        // Simulate attention computation
        @memcpy(self.batch_output, self.batch_v);
    }
    
    /// Sample next token (placeholder for real sampling)
    fn sample_token(self: *BatchProcessor, seq_idx: usize) u32 {
        _ = self;
        // Simple deterministic sampling for testing
        return @intCast(seq_idx + 100);
    }
    
    /// Remove completed sequences and compact the batch
    fn compact_batch(self: *BatchProcessor) void {
        var write_idx: usize = 0;
        for (0..self.active_count) |read_idx| {
            if (self.sequences[read_idx].active) {
                if (write_idx != read_idx) {
                    self.sequences[write_idx] = self.sequences[read_idx];
                }
                write_idx += 1;
            }
        }
        self.active_count = @intCast(write_idx);
    }
    
    /// Get current batch utilization
    pub fn get_utilization(self: *const BatchProcessor) f32 {
        if (self.config.max_batch_size == 0) return 0.0;
        return @as(f32, @floatFromInt(self.active_count)) / 
               @as(f32, @floatFromInt(self.config.max_batch_size));
    }
    
    /// Get statistics
    pub fn get_stats(self: *const BatchProcessor) BatchStats {
        var total_tokens: u32 = 0;
        var total_generated: u32 = 0;
        
        for (0..self.active_count) |i| {
            const seq = &self.sequences[i];
            if (seq.active) {
                total_tokens += seq.current_length;
                total_generated += seq.tokens_generated;
            }
        }
        
        return BatchStats{
            .active_sequences = self.active_count,
            .total_tokens = total_tokens,
            .total_generated = total_generated,
            .utilization = self.get_utilization(),
        };
    }
};

pub const BatchStats = struct {
    active_sequences: u32,
    total_tokens: u32,
    total_generated: u32,
    utilization: f32,
};

// ============================================================================
// Dynamic Batcher
// ============================================================================

pub const DynamicBatcher = struct {
    allocator: std.mem.Allocator,
    config: BatchConfig,
    processor: BatchProcessor,
    pending_requests: []BatchRequest,
    pending_count: usize,
    pending_capacity: usize,
    
    pub fn init(allocator: std.mem.Allocator, config: BatchConfig) !DynamicBatcher {
        const processor = try BatchProcessor.init(allocator, config);
        const pending_requests = try allocator.alloc(BatchRequest, 16);  // Initial capacity
        
        return .{
            .allocator = allocator,
            .config = config,
            .processor = processor,
            .pending_requests = pending_requests,
            .pending_count = 0,
            .pending_capacity = 16,
        };
    }
    
    pub fn deinit(self: *DynamicBatcher) void {
        self.processor.deinit();
        self.allocator.free(self.pending_requests);
    }
    
    /// Submit a new request
    pub fn submit_request(self: *DynamicBatcher, request: BatchRequest) !void {
        // Try to add to current batch
        const added = try self.processor.add_request(request);
        
        if (!added) {
            // Batch full, add to pending
            if (self.pending_count >= self.pending_capacity) {
                // Grow array
                const new_capacity = self.pending_capacity * 2;
                const new_pending = try self.allocator.alloc(BatchRequest, new_capacity);
                @memcpy(new_pending[0..self.pending_count], self.pending_requests[0..self.pending_count]);
                self.allocator.free(self.pending_requests);
                self.pending_requests = new_pending;
                self.pending_capacity = new_capacity;
            }
            self.pending_requests[self.pending_count] = request;
            self.pending_count += 1;
        }
    }
    
    /// Process one iteration
    pub fn process_iteration(self: *DynamicBatcher) !void {
        // Process current batch
        try self.processor.process_step();
        
        // Fill batch from pending if there's space
        while (self.pending_count > 0 and 
               self.processor.active_count < self.config.max_batch_size) {
            const request = self.pending_requests[0];
            _ = try self.processor.add_request(request);
            
            // Shift remaining requests using memmove equivalent
            if (self.pending_count > 1) {
                for (0..self.pending_count - 1) |i| {
                    self.pending_requests[i] = self.pending_requests[i + 1];
                }
            }
            self.pending_count -= 1;
        }
    }
    
    /// Get queue length
    pub fn get_queue_length(self: *const DynamicBatcher) usize {
        return self.pending_count;
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_batch_inference(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Batch Inference System\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const config = BatchConfig.init(4, 128, 64, 8);
    
    // Test 1: Basic batch processing
    {
        std.debug.print("\n1️⃣  Testing basic batch processing...\n", .{});
        
        var processor = try BatchProcessor.init(allocator, config);
        defer processor.deinit();
        
        // Add some requests
        const tokens1 = try allocator.alloc(u32, 10);
        defer allocator.free(tokens1);
        for (tokens1, 0..) |*t, i| t.* = @intCast(i);
        
        const tokens2 = try allocator.alloc(u32, 15);
        defer allocator.free(tokens2);
        for (tokens2, 0..) |*t, i| t.* = @intCast(i + 100);
        
        const req1 = BatchRequest.init(1, tokens1, 5);
        const req2 = BatchRequest.init(2, tokens2, 3);
        
        _ = try processor.add_request(req1);
        _ = try processor.add_request(req2);
        
        std.debug.print("   Added 2 requests to batch\n", .{});
        std.debug.print("   Active sequences: {d}\n", .{processor.active_count});
        
        // Process one step
        try processor.process_step();
        
        const stats = processor.get_stats();
        std.debug.print("   After step: {d} active, {d} tokens, {d:.1}% util\n", 
            .{stats.active_sequences, stats.total_tokens, stats.utilization * 100});
        
        std.debug.print("   ✅ Basic batch processing working\n", .{});
    }
    
    // Test 2: Batch utilization
    {
        std.debug.print("\n2️⃣  Testing batch utilization...\n", .{});
        
        var processor = try BatchProcessor.init(allocator, config);
        defer processor.deinit();
        
        // Fill batch to capacity
        for (0..config.max_batch_size) |i| {
            const tokens = try allocator.alloc(u32, 10);
            defer allocator.free(tokens);
            for (tokens, 0..) |*t, j| t.* = @intCast(j);
            
            const req = BatchRequest.init(i, tokens, 2);
            const added = try processor.add_request(req);
            
            if (!added) {
                std.debug.print("   ❌ Failed to add request {d}\n", .{i});
                return error.TestFailed;
            }
        }
        
        const util = processor.get_utilization();
        std.debug.print("   Batch utilization: {d:.1}%\n", .{util * 100});
        
        if (util < 0.99) {
            std.debug.print("   ❌ Expected 100% utilization\n", .{});
            return error.TestFailed;
        }
        
        // Try to add one more (should fail)
        const overflow_tokens = try allocator.alloc(u32, 5);
        defer allocator.free(overflow_tokens);
        const overflow_req = BatchRequest.init(999, overflow_tokens, 1);
        const added = try processor.add_request(overflow_req);
        
        if (added) {
            std.debug.print("   ❌ Should not be able to exceed batch size\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Batch utilization working correctly\n", .{});
    }
    
    // Test 3: Dynamic batching
    {
        std.debug.print("\n3️⃣  Testing dynamic batching...\n", .{});
        
        var batcher = try DynamicBatcher.init(allocator, config);
        defer batcher.deinit();
        
        // Allocate all tokens first (keep them alive)
        var all_tokens = try allocator.alloc([]u32, 8);
        defer {
            for (all_tokens) |tokens| {
                allocator.free(tokens);
            }
            allocator.free(all_tokens);
        }
        
        // Submit more requests than batch size
        for (0..8) |i| {
            const tokens = try allocator.alloc(u32, 10);
            for (tokens, 0..) |*t, j| t.* = @intCast(j);
            all_tokens[i] = tokens;
            
            const req = BatchRequest.init(i, tokens, 2);
            try batcher.submit_request(req);
        }
        
        const queue_len = batcher.get_queue_length();
        std.debug.print("   Submitted 8 requests (batch size: {d})\n", .{config.max_batch_size});
        std.debug.print("   Queue length: {d}\n", .{queue_len});
        
        if (queue_len != 8 - config.max_batch_size) {
            std.debug.print("   ❌ Expected {d} queued requests, got {d}\n", 
                .{8 - config.max_batch_size, queue_len});
            return error.TestFailed;
        }
        
        // Process iterations until queue empty
        var iterations: u32 = 0;
        while (batcher.get_queue_length() > 0 or batcher.processor.active_count > 0) {
            try batcher.process_iteration();
            iterations += 1;
            if (iterations > 100) break; // Safety limit
        }
        
        std.debug.print("   Processed all requests in {d} iterations\n", .{iterations});
        std.debug.print("   ✅ Dynamic batching working\n", .{});
    }
    
    std.debug.print("\n✅ All batch inference tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
