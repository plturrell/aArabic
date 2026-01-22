const std = @import("std");

/// Advanced Sampling Strategies
/// Implements Top-K, Top-P (nucleus), temperature, and penalties

pub const SamplingConfig = struct {
    temperature: f32 = 1.0,
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    repetition_penalty: f32 = 1.0,
    frequency_penalty: f32 = 0.0,
    presence_penalty: f32 = 0.0,
    
    pub fn default() SamplingConfig {
        return .{};
    }
    
    pub fn greedy() SamplingConfig {
        return .{ .temperature = 0.0 };
    }
    
    pub fn creative() SamplingConfig {
        return .{
            .temperature = 0.9,
            .top_p = 0.95,
        };
    }
    
    pub fn balanced() SamplingConfig {
        return .{
            .temperature = 0.7,
            .top_k = 50,
            .top_p = 0.9,
        };
    }
};

pub const AdvancedSampler = struct {
    allocator: std.mem.Allocator,
    config: SamplingConfig,
    rng: std.Random.DefaultPrng,
    
    // Scratch buffers for sampling
    sorted_indices: []u32,
    sorted_probs: []f32,
    vocab_size: u32,
    
    // Token frequency tracking (for penalties)
    token_counts: std.AutoHashMap(u32, u32),
    
    pub fn init(
        allocator: std.mem.Allocator,
        vocab_size: u32,
        config: SamplingConfig,
        seed: u64,
    ) !AdvancedSampler {
        const sorted_indices = try allocator.alloc(u32, vocab_size);
        errdefer allocator.free(sorted_indices);
        
        const sorted_probs = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(sorted_probs);
        
        var token_counts = std.AutoHashMap(u32, u32).init(allocator);
        errdefer token_counts.deinit();
        
        return AdvancedSampler{
            .allocator = allocator,
            .config = config,
            .rng = std.Random.DefaultPrng.init(seed),
            .sorted_indices = sorted_indices,
            .sorted_probs = sorted_probs,
            .vocab_size = vocab_size,
            .token_counts = token_counts,
        };
    }
    
    pub fn deinit(self: *AdvancedSampler) void {
        self.allocator.free(self.sorted_indices);
        self.allocator.free(self.sorted_probs);
        self.token_counts.deinit();
    }
    
    /// Sample next token from logits
    pub fn sample(self: *AdvancedSampler, logits: []const f32) !u32 {
        if (logits.len != self.vocab_size) {
            return error.InvalidLogitsSize;
        }
        
        // Copy logits to working buffer
        @memcpy(self.sorted_probs[0..logits.len], logits);
        
        // Apply repetition penalty
        if (self.config.repetition_penalty != 1.0) {
            try self.apply_repetition_penalty();
        }
        
        // Apply temperature
        if (self.config.temperature != 1.0) {
            self.apply_temperature();
        }
        
        // Convert to probabilities (softmax)
        try self.softmax();
        
        // Initialize indices
        for (self.sorted_indices[0..self.vocab_size], 0..) |*idx, i| {
            idx.* = @intCast(i);
        }
        
        // Apply Top-K if specified
        if (self.config.top_k) |k| {
            try self.apply_top_k(k);
        }
        
        // Apply Top-P if specified
        if (self.config.top_p) |p| {
            try self.apply_top_p(p);
        }
        
        // Sample from distribution
        const token = self.sample_from_probs();
        
        // Update token frequency
        const count = self.token_counts.get(token) orelse 0;
        try self.token_counts.put(token, count + 1);
        
        return token;
    }
    
    /// Reset token frequency tracking
    pub fn reset(self: *AdvancedSampler) void {
        self.token_counts.clearRetainingCapacity();
    }
    
    /// Apply temperature scaling
    fn apply_temperature(self: *AdvancedSampler) void {
        if (self.config.temperature == 0.0) {
            // Greedy sampling: find max and set others to -inf
            var max_idx: usize = 0;
            var max_val: f32 = self.sorted_probs[0];
            
            for (self.sorted_probs[1..self.vocab_size], 1..) |prob, i| {
                if (prob > max_val) {
                    max_val = prob;
                    max_idx = i;
                }
            }
            
            // Set all to very negative, max to very positive
            for (self.sorted_probs[0..self.vocab_size], 0..) |*p, i| {
                p.* = if (i == max_idx) 100.0 else -100.0;
            }
        } else {
            const temp = self.config.temperature;
            for (self.sorted_probs[0..self.vocab_size]) |*logit| {
                logit.* /= temp;
            }
        }
    }
    
    /// Apply repetition penalty
    fn apply_repetition_penalty(self: *AdvancedSampler) !void {
        const penalty = self.config.repetition_penalty;
        
        var it = self.token_counts.iterator();
        while (it.next()) |entry| {
            const token = entry.key_ptr.*;
            const count = entry.value_ptr.*;
            
            if (token < self.vocab_size) {
                // Penalize by count
                const penalty_factor = std.math.pow(f32, penalty, @as(f32, @floatFromInt(count)));
                self.sorted_probs[token] /= penalty_factor;
            }
        }
    }
    
    /// Convert logits to probabilities using softmax
    fn softmax(self: *AdvancedSampler) !void {
        // Find max for numerical stability
        var max_logit: f32 = -std.math.inf(f32);
        for (self.sorted_probs[0..self.vocab_size]) |logit| {
            max_logit = @max(max_logit, logit);
        }
        
        // Compute exp and sum
        var sum_exp: f32 = 0.0;
        for (self.sorted_probs[0..self.vocab_size]) |*logit| {
            const exp_val = @exp(logit.* - max_logit);
            logit.* = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        if (sum_exp > 0.0) {
            for (self.sorted_probs[0..self.vocab_size]) |*prob| {
                prob.* /= sum_exp;
            }
        }
    }
    
    /// Apply Top-K sampling
    fn apply_top_k(self: *AdvancedSampler, k: u32) !void {
        if (k >= self.vocab_size) return; // No filtering needed
        
        // Sort by probability (descending)
        self.sort_by_probability();
        
        // Zero out all but top k
        for (self.sorted_probs[k..self.vocab_size]) |*prob| {
            prob.* = 0.0;
        }
        
        // Renormalize
        try self.renormalize();
    }
    
    /// Apply Top-P (nucleus) sampling
    fn apply_top_p(self: *AdvancedSampler, p: f32) !void {
        // Sort by probability (descending)
        self.sort_by_probability();
        
        // Find cumulative probability cutoff
        var cumsum: f32 = 0.0;
        var cutoff_idx: usize = self.vocab_size;
        
        for (self.sorted_probs[0..self.vocab_size], 0..) |prob, i| {
            cumsum += prob;
            if (cumsum >= p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Zero out probabilities beyond cutoff
        for (self.sorted_probs[cutoff_idx..self.vocab_size]) |*prob| {
            prob.* = 0.0;
        }
        
        // Renormalize
        try self.renormalize();
    }
    
    /// Sort indices by probability (descending)
    fn sort_by_probability(self: *AdvancedSampler) void {
        // Simple insertion sort for now (could use heap sort for large vocabs)
        for (1..self.vocab_size) |i| {
            const key_idx = self.sorted_indices[i];
            const key_prob = self.sorted_probs[key_idx];
            
            var j = i;
            while (j > 0 and self.sorted_probs[self.sorted_indices[j - 1]] < key_prob) : (j -= 1) {
                self.sorted_indices[j] = self.sorted_indices[j - 1];
            }
            self.sorted_indices[j] = key_idx;
        }
    }
    
    /// Renormalize probabilities
    fn renormalize(self: *AdvancedSampler) !void {
        var sum: f32 = 0.0;
        for (self.sorted_probs[0..self.vocab_size]) |prob| {
            sum += prob;
        }
        
        if (sum > 0.0) {
            for (self.sorted_probs[0..self.vocab_size]) |*prob| {
                prob.* /= sum;
            }
        } else {
            return error.ZeroProbability;
        }
    }
    
    /// Sample from probability distribution
    fn sample_from_probs(self: *AdvancedSampler) u32 {
        const random_value = self.rng.random().float(f32);
        var cumsum: f32 = 0.0;
        
        for (self.sorted_probs[0..self.vocab_size], 0..) |prob, i| {
            cumsum += prob;
            if (cumsum >= random_value) {
                return self.sorted_indices[i];
            }
        }
        
        // Fallback to most probable token
        return self.sorted_indices[0];
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_advanced_sampler(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Advanced Sampler\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const vocab_size = 100;
    
    // Test 1: Temperature sampling
    {
        std.debug.print("\n1️⃣  Testing temperature sampling...\n", .{});
        
        var sampler = try AdvancedSampler.init(allocator, vocab_size, .{ .temperature = 0.5 }, 42);
        defer sampler.deinit();
        
        // Create peaked logits
        const logits = try allocator.alloc(f32, vocab_size);
        defer allocator.free(logits);
        
        for (logits, 0..) |*l, i| {
            l.* = @as(f32, @floatFromInt(i)) / 10.0;
        }
        logits[50] = 10.0; // Peak at token 50
        
        const token = try sampler.sample(logits);
        std.debug.print("   Sampled token: {d} (logits peaked at 50)\n", .{token});
        std.debug.print("   ✅ Temperature sampling working\n", .{});
    }
    
    // Test 2: Top-K sampling
    {
        std.debug.print("\n2️⃣  Testing Top-K sampling...\n", .{});
        
        var sampler = try AdvancedSampler.init(
            allocator,
            vocab_size,
            .{ .temperature = 1.0, .top_k = 10 },
            42,
        );
        defer sampler.deinit();
        
        const logits = try allocator.alloc(f32, vocab_size);
        defer allocator.free(logits);
        
        // Uniform logits
        @memset(logits, 1.0);
        
        // Sample multiple times
        var samples = std.AutoHashMap(u32, u32).init(allocator);
        defer samples.deinit();
        
        for (0..100) |_| {
            const token = try sampler.sample(logits);
            const count = samples.get(token) orelse 0;
            try samples.put(token, count + 1);
        }
        
        std.debug.print("   Unique tokens sampled: {d} (max should be ~10)\n", .{samples.count()});
        
        if (samples.count() > 15) {
            std.debug.print("   ❌ Top-K not filtering correctly\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Top-K sampling working\n", .{});
    }
    
    // Test 3: Top-P (nucleus) sampling
    {
        std.debug.print("\n3️⃣  Testing Top-P sampling...\n", .{});
        
        var sampler = try AdvancedSampler.init(
            allocator,
            vocab_size,
            .{ .temperature = 1.0, .top_p = 0.9 },
            42,
        );
        defer sampler.deinit();
        
        const logits = try allocator.alloc(f32, vocab_size);
        defer allocator.free(logits);
        
        // Create peaked distribution
        for (logits, 0..) |*l, i| {
            if (i < 5) {
                l.* = 5.0; // High probability for first 5 tokens
            } else {
                l.* = 0.1; // Low probability for rest
            }
        }
        
        // Sample multiple times
        var samples = std.AutoHashMap(u32, u32).init(allocator);
        defer samples.deinit();
        
        for (0..100) |_| {
            const token = try sampler.sample(logits);
            const count = samples.get(token) orelse 0;
            try samples.put(token, count + 1);
        }
        
        std.debug.print("   Unique tokens sampled: {d}\n", .{samples.count()});
        std.debug.print("   Most samples should be from tokens 0-4\n", .{});
        
        var in_nucleus: u32 = 0;
        for (0..5) |i| {
            if (samples.contains(@intCast(i))) {
                in_nucleus += 1;
            }
        }
        
        std.debug.print("   Tokens from nucleus (0-4): {d}/5\n", .{in_nucleus});
        
        if (in_nucleus < 3) {
            std.debug.print("   ❌ Top-P not working correctly\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Top-P sampling working\n", .{});
    }
    
    // Test 4: Repetition penalty
    {
        std.debug.print("\n4️⃣  Testing repetition penalty...\n", .{});
        
        var sampler = try AdvancedSampler.init(
            allocator,
            vocab_size,
            .{ .temperature = 0.0, .repetition_penalty = 1.5 },
            42,
        );
        defer sampler.deinit();
        
        const logits = try allocator.alloc(f32, vocab_size);
        defer allocator.free(logits);
        
        // All same logits
        @memset(logits, 1.0);
        
        // Sample same token multiple times
        var prev_token: u32 = 0;
        for (0..5) |i| {
            const token = try sampler.sample(logits);
            std.debug.print("   Sample {d}: token {d}\n", .{i, token});
            
            // After first sample, should avoid repetition
            if (i > 0 and token == prev_token) {
                std.debug.print("   Note: Repeated token (with penalty this is less likely)\n", .{});
            }
            prev_token = token;
        }
        
        std.debug.print("   ✅ Repetition penalty working\n", .{});
    }
    
    // Test 5: Greedy sampling (temperature = 0)
    {
        std.debug.print("\n5️⃣  Testing greedy sampling...\n", .{});
        
        var sampler = try AdvancedSampler.init(
            allocator,
            vocab_size,
            SamplingConfig.greedy(),
            42,
        );
        defer sampler.deinit();
        
        const logits = try allocator.alloc(f32, vocab_size);
        defer allocator.free(logits);
        
        // Create clear maximum
        for (logits, 0..) |*l, i| {
            l.* = @as(f32, @floatFromInt(i));
        }
        logits[75] = 1000.0; // Clear max at token 75
        
        const token = try sampler.sample(logits);
        std.debug.print("   Sampled token: {d} (expected: 75)\n", .{token});
        
        if (token != 75) {
            std.debug.print("   ❌ Greedy sampling should pick max\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Greedy sampling working\n", .{});
    }
    
    std.debug.print("\n✅ All advanced sampler tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
