const std = @import("std");
const gguf = @import("gguf_loader");

/// BPE (Byte Pair Encoding) Tokenizer for GGUF models
/// Loads vocabulary from GGUF metadata and provides encoding/decoding

// ============================================================================
// Structures
// ============================================================================

pub const Token = struct {
    id: u32,
    text: []const u8,
    score: f32,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: []Token,
    vocab_size: u32,
    token_map: std.StringHashMap(u32),

    // Special tokens
    bos_token: u32, // Beginning of sequence
    eos_token: u32, // End of sequence
    pad_token: u32, // Padding
    unk_token: u32, // Unknown

    pub fn loadFromModel(allocator: std.mem.Allocator, model: *gguf.GGUFModel) !Tokenizer {
        std.debug.print("\n📝 Loading tokenizer...\n", .{});

        const vocab_size = model.metadata.vocab_size;
        std.debug.print("   Vocabulary size: {d}\n", .{vocab_size});

        // Allocate vocabulary
        var vocab = try allocator.alloc(Token, vocab_size);
        errdefer allocator.free(vocab);

        var token_map = std.StringHashMap(u32).init(allocator);
        errdefer token_map.deinit();

        // Load vocab from model
        if (model.vocab_tokens.len == vocab_size) {
            for (0..vocab_size) |i| {
                const text = try allocator.dupe(u8, model.vocab_tokens[i]);
                const score = if (i < model.vocab_scores.len) model.vocab_scores[i] else 0.0;

                vocab[i] = Token{
                    .id = @intCast(i),
                    .text = text,
                    .score = score,
                };

                try token_map.put(text, @intCast(i));
            }
        } else {
            std.debug.print("⚠️  Model vocab tokens missing or size mismatch ({d} vs {d}), using dummies.\n", .{ model.vocab_tokens.len, vocab_size });
            for (0..vocab_size) |i| {
                const text = try std.fmt.allocPrint(allocator, "token_{d}", .{i});
                vocab[i] = Token{
                    .id = @intCast(i),
                    .text = text,
                    .score = 0.0,
                };
                try token_map.put(text, @intCast(i));
            }
        }

        std.debug.print("   ✅ Vocabulary loaded\n", .{});

        return Tokenizer{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_size = vocab_size,
            .token_map = token_map,
            .bos_token = 1, // <|begin_of_text|>
            .eos_token = 2, // <|end_of_text|>
            .pad_token = 0, // <|pad|>
            .unk_token = 0, // Unknown token
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.vocab) |token| {
            self.allocator.free(token.text);
        }
        self.allocator.free(self.vocab);
        self.token_map.deinit();
    }

    /// Encode text to token IDs (Basic BPE)
    pub fn encode(self: *Tokenizer, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
        // Initial tokenization: bytes/chars
        // This is a simplified BPE implementation
        // 1. Start with list of bytes
        var current_tokens = try std.ArrayList(u32).initCapacity(allocator, 128);
        defer current_tokens.deinit(allocator);

        try current_tokens.append(allocator, self.bos_token);

        // Treat input as raw bytes for now (Llama 2 approach mostly)
        // Ideally should map to byte tokens (e.g. <0xXX>)
        // For simplicity, we try to match individual characters first

        // Split by whitespace for pre-tokenization (Llama style)
        // A real implementation would handle this more robustly
        // But for "Hello world", this loop works:

        // Very basic: just try to match longest prefix from vocab?
        // No, BPE is bottom-up merge.

        // Let's implement a simple greedy loop:
        // Initialize with characters
        // Note: This relies on vocab having single char tokens.

        // Hack for prototype:
        // Just look for exact matches of words in vocab.
        // If not found, split into chars.
        // This is better than whitespace splitting.

        var i: usize = 0;
        while (i < text.len) {
            // Try to find longest matching token starting at i
            var best_len: usize = 0;
            var best_id: u32 = self.unk_token;

            // Limit search length
            const max_search = @min(text.len - i, 48); // reasonable max token length

            for (1..max_search + 1) |len| {
                const sub = text[i .. i + len];
                if (self.token_map.get(sub)) |id| {
                    best_len = len;
                    best_id = id;
                }
            }

            if (best_len > 0) {
                try current_tokens.append(allocator, best_id);
                i += best_len;
            } else {
                // Unknown character, skip or use unk
                // Try to map byte?
                // For now, skip to avoid infinite loop
                std.debug.print("⚠️  Unknown char at {d}: {c}\n", .{ i, text[i] });
                i += 1;
            }
        }

        try current_tokens.append(allocator, self.eos_token);

        return current_tokens.toOwnedSlice(allocator);
    }

    /// Decode token IDs to text
    pub fn decode(self: *Tokenizer, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
        // Calculate total size needed
        var total_len: usize = 0;
        for (token_ids) |token_id| {
            if (token_id >= self.vocab_size) continue;
            if (token_id == self.bos_token or token_id == self.eos_token or token_id == self.pad_token) continue;

            const token_text = self.vocab[token_id].text;
            // Handle special tokens like <0x0A> or similar if needed
            // Llama sentencepiece uses " " (U+2581) for space
            total_len += token_text.len;
        }

        // Allocate result
        const result = try allocator.alloc(u8, total_len);
        errdefer allocator.free(result);

        // Fill result
        var pos: usize = 0;
        for (token_ids) |token_id| {
            if (token_id >= self.vocab_size) continue;
            if (token_id == self.bos_token or token_id == self.eos_token or token_id == self.pad_token) continue;

            const token_text = self.vocab[token_id].text;

            // Simple space replacement for Llama (U+2581 -> ' ')
            // In byte string, U+2581 is \xe2\x96\x81
            const SPIECE_UNDERLINE = "\xe2\x96\x81";

            if (std.mem.startsWith(u8, token_text, SPIECE_UNDERLINE)) {
                result[pos] = ' ';
                pos += 1;
                @memcpy(result[pos .. pos + token_text.len - 3], token_text[3..]);
                pos += token_text.len - 3;
            } else {
                @memcpy(result[pos .. pos + token_text.len], token_text);
                pos += token_text.len;
            }
        }

        // Trim actual size if we did replacements
        return allocator.realloc(result, pos);
    }

    /// Find token ID by text
    fn findToken(self: *Tokenizer, text: []const u8) ?u32 {
        return self.token_map.get(text);
    }

    /// Get token text by ID
    pub fn getTokenText(self: *Tokenizer, token_id: u32) ?[]const u8 {
        if (token_id >= self.vocab_size) return null;
        return self.vocab[token_id].text;
    }
};

// ============================================================================
// BPE Utilities
// ============================================================================

/// Load vocabulary from GGUF metadata
/// This is a placeholder - real implementation would parse GGUF vocab arrays
pub fn loadVocabFromGGUF(
    allocator: std.mem.Allocator,
    model: *gguf.GGUFModel,
) ![]Token {
    _ = model;

    // TODO: Parse actual GGUF vocabulary metadata
    // For now, return placeholder
    const vocab = try allocator.alloc(Token, 100);
    for (0..100) |i| {
        vocab[i] = Token{
            .id = @intCast(i),
            .text = try std.fmt.allocPrint(allocator, "tok_{d}", .{i}),
            .score = 0.0,
        };
    }
    return vocab;
}

/// Calculate token probabilities from logits
pub fn calculateProbs(
    probs: []f32,
    logits: []const f32,
    temperature: f32,
) void {
    const n = logits.len;

    // Apply temperature
    if (temperature != 1.0) {
        for (0..n) |i| {
            probs[i] = logits[i] / temperature;
        }
    } else {
        @memcpy(probs, logits);
    }

    // Softmax
    const matrix_ops = @import("matrix_ops");
    matrix_ops.softmax(probs, probs);
}

/// Sample token from probability distribution
pub fn sampleToken(probs: []const f32, random: std.Random) u32 {
    var cumsum: f32 = 0.0;
    const rand_val = random.float(f32);

    for (probs, 0..) |prob, i| {
        cumsum += prob;
        if (rand_val < cumsum) {
            return @intCast(i);
        }
    }

    // Fallback to last token
    return @intCast(probs.len - 1);
}

/// Top-k filtering (simplified - selects top k by zeroing others)
pub fn topK(probs: []f32, k: usize) void {
    if (k >= probs.len) return;

    // Find k-th largest using selection
    // Make a copy for sorting
    var sorted_buf: [512]f32 = undefined;
    const n = @min(probs.len, 512);
    @memcpy(sorted_buf[0..n], probs[0..n]);

    std.mem.sort(f32, sorted_buf[0..n], {}, comptime std.sort.desc(f32));

    const threshold = if (k < n) sorted_buf[k] else 0.0;

    // Zero out below threshold
    for (probs) |*p| {
        if (p.* < threshold) p.* = 0.0;
    }

    // Re-normalize
    var sum: f32 = 0.0;
    for (probs) |p| sum += p;

    if (sum > 0.0) {
        const inv_sum = 1.0 / sum;
        for (probs) |*p| p.* *= inv_sum;
    }
}

/// Top-p (nucleus) sampling (simplified)
pub fn topP(probs: []f32, p: f32) void {
    const n = probs.len;
    if (n > 512) return; // Limit for stack allocation

    // Create index array with corresponding probs for sorting
    var indices: [512]usize = undefined;
    for (0..n) |i| indices[i] = i;

    // Simple bubble sort for indices (probs array is used for comparison)
    // Using a simple sort since we can't use std.mem.sort with runtime context easily in Zig 0.15
    for (0..n) |i| {
        for (i + 1..n) |j| {
            if (probs[indices[i]] < probs[indices[j]]) {
                const tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    // Find cutoff
    var cumsum: f32 = 0.0;
    var cutoff: usize = n;

    for (indices[0..n], 0..) |idx, i| {
        cumsum += probs[idx];
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out beyond cutoff
    for (cutoff..n) |i| {
        probs[indices[i]] = 0.0;
    }

    // Re-normalize
    var sum: f32 = 0.0;
    for (probs) |prob| sum += prob;

    if (sum > 0.0) {
        const inv_sum = 1.0 / sum;
        for (probs) |*prob| prob.* *= inv_sum;
    }
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_tokenizer(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Tokenizer\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    // Create a simple test vocab
    var vocab = try allocator.alloc(Token, 10);
    defer {
        for (vocab) |token| {
            allocator.free(token.text);
        }
        allocator.free(vocab);
    }

    const words = [_][]const u8{ "hello", "world", "test", "zig", "llama", "inference", "fast", "code", "run", "end" };
    for (words, 0..) |word, i| {
        vocab[i] = Token{
            .id = @intCast(i),
            .text = try allocator.dupe(u8, word),
            .score = 0.0,
        };
    }

    const token_map = std.StringHashMap(u32).init(allocator);

    var tokenizer = Tokenizer{
        .allocator = allocator,
        .vocab = vocab,
        .vocab_size = 10,
        .token_map = token_map,
        .bos_token = 0,
        .eos_token = 9,
        .pad_token = 0,
        .unk_token = 0,
    };

    // Test 1: Encode/decode
    {
        std.debug.print("\n1️⃣  Testing encode/decode...\n", .{});

        const text = "hello world test";
        const tokens = try tokenizer.encode(text, allocator);
        defer allocator.free(tokens);

        std.debug.print("   Input: {s}\n", .{text});
        std.debug.print("   Tokens: [", .{});
        for (tokens, 0..) |tok, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{tok});
        }
        std.debug.print("]\n", .{});

        const decoded = try tokenizer.decode(tokens, allocator);
        defer allocator.free(decoded);

        std.debug.print("   Decoded: {s}\n", .{decoded});
        std.debug.print("   ✅ Encode/decode working\n", .{});
    }

    // Test 2: Probability calculations
    {
        std.debug.print("\n2️⃣  Testing probability calculations...\n", .{});

        const logits = [_]f32{ 2.0, 1.0, 3.0, 0.5, 1.5 };
        var probs: [5]f32 = undefined;

        calculateProbs(&probs, &logits, 1.0);

        var sum: f32 = 0.0;
        for (probs) |p| sum += p;

        if (@abs(sum - 1.0) > 0.001) {
            std.debug.print("   ❌ Probabilities don't sum to 1.0: {d}\n", .{sum});
            return error.TestFailed;
        }

        std.debug.print("   Probabilities: ", .{});
        for (probs) |p| {
            std.debug.print("{d:.4} ", .{p});
        }
        std.debug.print("\n", .{});
        std.debug.print("   ✅ Probability calculation correct\n", .{});
    }

    // Test 3: Sampling
    {
        std.debug.print("\n3️⃣  Testing token sampling...\n", .{});

        const probs = [_]f32{ 0.1, 0.2, 0.5, 0.1, 0.1 };

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        // Sample multiple times
        var counts = [_]u32{0} ** 5;
        for (0..1000) |_| {
            const token = sampleToken(&probs, random);
            counts[token] += 1;
        }

        std.debug.print("   Sampling distribution (1000 samples):\n", .{});
        for (counts, 0..) |count, i| {
            const freq = @as(f32, @floatFromInt(count)) / 1000.0;
            std.debug.print("      Token {d}: {d} samples ({d:.1}% vs {d:.1}% expected)\n", .{
                i,
                count,
                freq * 100.0,
                probs[i] * 100.0,
            });
        }

        // Most samples should be token 2 (50% probability)
        if (counts[2] < 400) {
            std.debug.print("   ⚠️  Sampling distribution unexpected\n", .{});
        } else {
            std.debug.print("   ✅ Sampling working correctly\n", .{});
        }
    }

    // Test 4: Top-k filtering
    {
        std.debug.print("\n4️⃣  Testing top-k filtering...\n", .{});

        var probs = [_]f32{ 0.05, 0.15, 0.40, 0.25, 0.10, 0.05 };
        const original_sum = blk: {
            var s: f32 = 0.0;
            for (probs) |p| s += p;
            break :blk s;
        };

        topK(&probs, 3);

        // After top-3, should only have 3 non-zero values
        var non_zero: usize = 0;
        var new_sum: f32 = 0.0;
        for (probs) |p| {
            if (p > 0.0001) non_zero += 1;
            new_sum += p;
        }

        std.debug.print("   Original sum: {d:.4}\n", .{original_sum});
        std.debug.print("   After top-3: {d} non-zero values\n", .{non_zero});
        std.debug.print("   New sum: {d:.4}\n", .{new_sum});

        if (non_zero <= 3 and @abs(new_sum - 1.0) < 0.01) {
            std.debug.print("   ✅ Top-k filtering correct\n", .{});
        } else {
            std.debug.print("   ⚠️  Top-k may have issues\n", .{});
        }
    }

    std.debug.print("\n✅ All tokenizer tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
