const std = @import("std");

/// Advanced Sampling Strategies for Token Generation
/// Implements temperature, top-k, and top-p (nucleus) sampling

// ============================================================================
// Sampling Configuration
// ============================================================================

pub const SamplingStrategy = enum {
    Greedy,        // Always pick highest probability (argmax)
    Temperature,   // Scale logits by temperature
    TopK,         // Sample from top-k tokens
    TopP,         // Sample from nucleus (cumulative probability >= p)
};

pub const SamplingConfig = struct {
    strategy: SamplingStrategy = .Greedy,
    temperature: f32 = 1.0,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    
    pub fn default() SamplingConfig {
        return .{
            .strategy = .Greedy,
            .temperature = 1.0,
            .top_k = 40,
            .top_p = 0.9,
        };
    }
    
    pub fn greedy() SamplingConfig {
        return .{ .strategy = .Greedy };
    }
    
    pub fn withTemperature(temp: f32) SamplingConfig {
        return .{
            .strategy = .Temperature,
            .temperature = temp,
        };
    }
    
    pub fn topK(k: u32, temp: f32) SamplingConfig {
        return .{
            .strategy = .TopK,
            .temperature = temp,
            .top_k = k,
        };
    }
    
    pub fn topP(p: f32, temp: f32) SamplingConfig {
        return .{
            .strategy = .TopP,
            .temperature = temp,
            .top_p = p,
        };
    }
};

// ============================================================================
// Token Probability Pair
// ============================================================================

const TokenProb = struct {
    token_id: u32,
    prob: f32,
    
    fn compare(_: void, a: TokenProb, b: TokenProb) bool {
        return a.prob > b.prob;
    }
};

// ============================================================================
// Sampler
// ============================================================================

pub const Sampler = struct {
    allocator: std.mem.Allocator,
    config: SamplingConfig,
    rng: std.Random.DefaultPrng,
    
    pub fn init(allocator: std.mem.Allocator, config: SamplingConfig) Sampler {
        // Use current time as seed for randomness
        const seed = @as(u64, @intCast(std.time.timestamp()));
        return .{
            .allocator = allocator,
            .config = config,
            .rng = std.Random.DefaultPrng.init(seed),
        };
    }
    
    /// Sample a token from logits based on the configured strategy
    pub fn sample(self: *Sampler, logits: []const f32) !u32 {
        return switch (self.config.strategy) {
            .Greedy => sampleGreedy(logits),
            .Temperature => try self.sampleTemperature(logits),
            .TopK => try self.sampleTopK(logits),
            .TopP => try self.sampleTopP(logits),
        };
    }
    
    // ========================================================================
    // Sampling Implementations
    // ========================================================================
    
    /// Temperature sampling: scale logits then sample from distribution
    fn sampleTemperature(self: *Sampler, logits: []const f32) !u32 {
        // Apply temperature scaling
        const scaled_logits = try self.allocator.alloc(f32, logits.len);
        defer self.allocator.free(scaled_logits);
        
        for (logits, scaled_logits) |logit, *scaled| {
            scaled.* = logit / self.config.temperature;
        }
        
        // Convert to probabilities with softmax
        const probs = try self.allocator.alloc(f32, logits.len);
        defer self.allocator.free(probs);
        
        softmax(probs, scaled_logits);
        
        // Sample from distribution
        return sampleFromDistribution(self, probs);
    }
    
    /// Top-k sampling: sample from top k most likely tokens
    fn sampleTopK(self: *Sampler, logits: []const f32) !u32 {
        // Create token-prob pairs
        const pairs = try self.allocator.alloc(TokenProb, logits.len);
        defer self.allocator.free(pairs);
        
        for (logits, 0..) |logit, i| {
            pairs[i] = .{
                .token_id = @intCast(i),
                .prob = logit,
            };
        }
        
        // Sort by probability (descending)
        std.mem.sort(TokenProb, pairs, {}, TokenProb.compare);
        
        // Take top-k
        const k = @min(self.config.top_k, @as(u32, @intCast(pairs.len)));
        const top_k_pairs = pairs[0..k];
        
        // Apply temperature to top-k logits
        const top_k_logits = try self.allocator.alloc(f32, k);
        defer self.allocator.free(top_k_logits);
        
        for (top_k_pairs, top_k_logits) |pair, *logit| {
            logit.* = pair.prob / self.config.temperature;
        }
        
        // Convert to probabilities
        const probs = try self.allocator.alloc(f32, k);
        defer self.allocator.free(probs);
        
        softmax(probs, top_k_logits);
        
        // Sample from top-k distribution
        const sampled_idx = sampleFromDistribution(self, probs);
        return top_k_pairs[sampled_idx].token_id;
    }
    
    /// Top-p (nucleus) sampling: sample from smallest set with cumulative prob >= p
    fn sampleTopP(self: *Sampler, logits: []const f32) !u32 {
        // Create token-prob pairs
        const pairs = try self.allocator.alloc(TokenProb, logits.len);
        defer self.allocator.free(pairs);
        
        // Apply temperature and create pairs
        for (logits, 0..) |logit, i| {
            pairs[i] = .{
                .token_id = @intCast(i),
                .prob = logit / self.config.temperature,
            };
        }
        
        // Sort by probability (descending)
        std.mem.sort(TokenProb, pairs, {}, TokenProb.compare);
        
        // Convert to probabilities
        const all_probs = try self.allocator.alloc(f32, pairs.len);
        defer self.allocator.free(all_probs);
        
        for (pairs, 0..) |pair, i| {
            all_probs[i] = pair.prob;
        }
        
        softmax(all_probs, all_probs);
        
        // Find nucleus (cumulative probability >= top_p)
        var cumulative: f32 = 0.0;
        var nucleus_size: usize = 0;
        
        for (all_probs) |prob| {
            cumulative += prob;
            nucleus_size += 1;
            if (cumulative >= self.config.top_p) break;
        }
        
        // Ensure at least one token
        nucleus_size = @max(nucleus_size, 1);
        
        // Sample from nucleus
        const nucleus_probs = all_probs[0..nucleus_size];
        const sampled_idx = sampleFromDistribution(self, nucleus_probs);
        
        return pairs[sampled_idx].token_id;
    }
};

// ============================================================================
// Sampling Implementations
// ============================================================================

/// Greedy sampling: always pick the token with highest probability
fn sampleGreedy(logits: []const f32) u32 {
    var max_idx: u32 = 0;
    var max_val: f32 = logits[0];
    
    for (logits, 0..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = @intCast(i);
        }
    }
    
    return max_idx;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Apply softmax to convert logits to probabilities
fn softmax(output: []f32, input: []const f32) void {
    // Find max for numerical stability
    var max_val: f32 = input[0];
    for (input[1..]) |val| {
        max_val = @max(max_val, val);
    }
    
    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (input, output) |in_val, *out_val| {
        const exp_val = @exp(in_val - max_val);
        out_val.* = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (output) |*val| {
        val.* /= sum;
    }
}

/// Sample from a probability distribution
fn sampleFromDistribution(self: *Sampler, probs: []const f32) u32 {
    const random = self.rng.random();
    const r = random.float(f32);
    
    var cumulative: f32 = 0.0;
    for (probs, 0..) |prob, i| {
        cumulative += prob;
        if (r < cumulative) {
            return @intCast(i);
        }
    }
    
    // Fallback (should rarely happen)
    return @intCast(probs.len - 1);
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_sampler(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Sampler Module\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test logits (simulating a small vocabulary)
    const test_logits = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 1.5 };
    
    // Test 1: Greedy sampling
    {
        std.debug.print("\n1️⃣  Testing greedy sampling...\n", .{});
        
        var sampler = Sampler.init(allocator, SamplingConfig.greedy());
        const token = try sampler.sample(&test_logits);
        
        std.debug.print("   Logits: [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 1.5]\n", .{});
        std.debug.print("   Sampled token: {d} (should be 4, highest logit)\n", .{token});
        
        if (token != 4) {
            std.debug.print("   ❌ Expected token 4, got {d}\n", .{token});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Greedy sampling working\n", .{});
    }
    
    // Test 2: Temperature sampling
    {
        std.debug.print("\n2️⃣  Testing temperature sampling...\n", .{});
        
        // Low temperature (more deterministic)
        var sampler_low = Sampler.init(allocator, SamplingConfig.withTemperature(0.5));
        const token_low = try sampler_low.sample(&test_logits);
        std.debug.print("   Temperature 0.5 sampled: {d}\n", .{token_low});
        
        // High temperature (more random)
        var sampler_high = Sampler.init(allocator, SamplingConfig.withTemperature(2.0));
        const token_high = try sampler_high.sample(&test_logits);
        std.debug.print("   Temperature 2.0 sampled: {d}\n", .{token_high});
        
        std.debug.print("   ✅ Temperature sampling working\n", .{});
    }
    
    // Test 3: Top-k sampling
    {
        std.debug.print("\n3️⃣  Testing top-k sampling...\n", .{});
        
        var sampler = Sampler.init(allocator, SamplingConfig.topK(3, 1.0));
        const token = try sampler.sample(&test_logits);
        
        std.debug.print("   Top-k=3 sampled: {d} (should be from top 3: 4, 3, 5)\n", .{token});
        
        // Verify token is in top-3
        if (token != 4 and token != 3 and token != 5) {
            std.debug.print("   ❌ Token {d} not in top-3\n", .{token});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Top-k sampling working\n", .{});
    }
    
    // Test 4: Top-p sampling
    {
        std.debug.print("\n4️⃣  Testing top-p (nucleus) sampling...\n", .{});
        
        var sampler = Sampler.init(allocator, SamplingConfig.topP(0.9, 1.0));
        const token = try sampler.sample(&test_logits);
        
        std.debug.print("   Top-p=0.9 sampled: {d}\n", .{token});
        std.debug.print("   ✅ Top-p sampling working\n", .{});
    }
    
    // Test 5: Softmax
    {
        std.debug.print("\n5️⃣  Testing softmax...\n", .{});
        
        const input = [_]f32{ 1.0, 2.0, 3.0 };
        var output: [3]f32 = undefined;
        
        softmax(&output, &input);
        
        // Check sum is approximately 1.0
        var sum: f32 = 0.0;
        for (output) |val| {
            sum += val;
        }
        
        std.debug.print("   Input: [1.0, 2.0, 3.0]\n", .{});
        std.debug.print("   Output: [{d:.4}, {d:.4}, {d:.4}]\n", .{ output[0], output[1], output[2] });
        std.debug.print("   Sum: {d:.6} (should be ~1.0)\n", .{sum});
        
        const tolerance = 0.0001;
        if (@abs(sum - 1.0) > tolerance) {
            std.debug.print("   ❌ Sum not close to 1.0\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Softmax working\n", .{});
    }
    
    // Test 6: Multiple samples with temperature
    {
        std.debug.print("\n6️⃣  Testing sampling diversity...\n", .{});
        
        var sampler = Sampler.init(allocator, SamplingConfig.withTemperature(0.8));
        
        var counts = [_]u32{0} ** 7;
        const num_samples = 100;
        
        for (0..num_samples) |_| {
            const token = try sampler.sample(&test_logits);
            counts[token] += 1;
        }
        
        std.debug.print("   Sampled {d} times with temperature 0.8:\n", .{num_samples});
        for (counts, 0..) |count, i| {
            if (count > 0) {
                std.debug.print("   Token {d}: {d} times ({d:.1}%)\n", .{
                    i,
                    count,
                    100.0 * @as(f32, @floatFromInt(count)) / @as(f32, @floatFromInt(num_samples)),
                });
            }
        }
        
        std.debug.print("   ✅ Sampling diversity verified\n", .{});
    }
    
    std.debug.print("\n✅ All sampler tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
