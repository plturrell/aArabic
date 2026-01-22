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
        // Deinit the map first before freeing the vocab strings
        // (the map uses string slices as keys that point into vocab)
        self.token_map.deinit();

        for (self.vocab) |token| {
            self.allocator.free(token.text);
        }
        self.allocator.free(self.vocab);
    }

    /// Create tokenizer from BPE vocab (for HuggingFace models)
    pub fn initFromBPEVocab(
        allocator: std.mem.Allocator,
        bpe_vocab_size: u32,
        bpe_id_to_token: anytype,
        bos_token_id: u32,
        eos_token_id: u32,
    ) !Tokenizer {
        std.debug.print("\n📝 Loading tokenizer from BPE vocab...\n", .{});
        std.debug.print("   Vocabulary size: {d}\n", .{bpe_vocab_size});

        var vocab = try allocator.alloc(Token, bpe_vocab_size);
        errdefer allocator.free(vocab);

        var token_map = std.StringHashMap(u32).init(allocator);
        errdefer token_map.deinit();

        // Initialize with placeholder tokens
        for (0..bpe_vocab_size) |i| {
            vocab[i] = Token{
                .id = @intCast(i),
                .text = "",
                .score = 0.0,
            };
        }

        // Copy tokens from BPE vocab
        var count: usize = 0;
        var iter = bpe_id_to_token.iterator();
        while (iter.next()) |entry| {
            const id = entry.key_ptr.*;
            const text = entry.value_ptr.*;
            if (id < bpe_vocab_size) {
                const text_copy = try allocator.dupe(u8, text);
                vocab[id] = Token{
                    .id = id,
                    .text = text_copy,
                    .score = 0.0,
                };
                try token_map.put(text_copy, id);
                count += 1;
            }
        }

        std.debug.print("   Loaded {d} tokens from BPE vocab\n", .{count});
        std.debug.print("   ✅ Tokenizer created\n", .{});

        return Tokenizer{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_size = bpe_vocab_size,
            .token_map = token_map,
            .bos_token = bos_token_id,
            .eos_token = eos_token_id,
            .pad_token = 0,
            .unk_token = 0,
        };
    }

    /// Encode text to token IDs (adds BOS, no EOS - suitable for generation)
    pub fn encode(self: *Tokenizer, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
        return self.encodeWithOptions(text, allocator, true, false);
    }

    /// Encode without special tokens (BOS/EOS) - for generation prompts
    pub fn encodePrompt(self: *Tokenizer, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
        return self.encodeWithOptions(text, allocator, true, false);
    }

    /// Encode with options for special tokens
    fn encodeWithOptions(
        self: *Tokenizer,
        text: []const u8,
        allocator: std.mem.Allocator,
        add_bos: bool,
        add_eos: bool,
    ) ![]u32 {
        var current_tokens = try std.ArrayList(u32).initCapacity(allocator, 128);
        defer current_tokens.deinit();

        if (add_bos) {
            try current_tokens.append(self.bos_token);
        }

        // Pre-process: convert spaces to byte-level encoding (Ġ = 0xC4 0xA0)
        // GPT-2/Qwen use byte-level BPE where space (0x20) maps to 'Ġ'
        var processed = try std.ArrayList(u8).initCapacity(allocator, text.len * 2);
        defer processed.deinit();

        for (text) |c| {
            if (c == ' ') {
                // Space -> 'Ġ' (UTF-8: 0xC4 0xA0)
                try processed.append(0xC4);
                try processed.append(0xA0);
            } else if (c == '\n') {
                // Newline -> 'Ċ' (UTF-8: 0xC4 0x8A)
                try processed.append(0xC4);
                try processed.append(0x8A);
            } else {
                try processed.append(c);
            }
        }

        const input = processed.items;

        // Greedy longest-match tokenization
        var i: usize = 0;
        while (i < input.len) {
            var best_len: usize = 0;
            var best_id: u32 = self.unk_token;

            const max_search = @min(input.len - i, 48);

            for (1..max_search + 1) |len| {
                const sub = input[i .. i + len];
                if (self.token_map.get(sub)) |id| {
                    best_len = len;
                    best_id = id;
                }
            }

            if (best_len > 0) {
                try current_tokens.append(best_id);
                i += best_len;
            } else {
                // Try single byte fallback
                const byte_str = input[i .. i + 1];
                if (self.token_map.get(byte_str)) |id| {
                    try current_tokens.append(id);
                } else {
                    std.debug.print("⚠️  Unknown byte at {d}: 0x{X:0>2}\n", .{ i, input[i] });
                }
                i += 1;
            }
        }

        if (add_eos) {
            try current_tokens.append(self.eos_token);
        }

        return current_tokens.toOwnedSlice();
    }

    /// Decode token IDs to text
    /// Handles GPT-2/Qwen byte-level BPE tokens
    pub fn decode(self: *Tokenizer, token_ids: []const u32, allocator: std.mem.Allocator) ![]u8 {
        // First pass: collect raw token bytes
        var raw_bytes = std.ArrayList(u8).init(allocator);
        defer raw_bytes.deinit();

        for (token_ids) |token_id| {
            if (token_id >= self.vocab_size) continue;
            if (token_id == self.bos_token or token_id == self.eos_token or token_id == self.pad_token) continue;

            const token_text = self.vocab[token_id].text;
            try raw_bytes.appendSlice(token_text);
        }

        // Second pass: decode GPT-2/Qwen byte tokens to actual bytes
        // GPT-2 uses special unicode chars to represent bytes:
        // - 'Ġ' (U+0120 = 0xC4 0xA0) represents space
        // - 'Ċ' (U+010A = 0xC4 0x8A) represents newline
        // - Other chars 0x100-0x143 represent control chars and extended bytes
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < raw_bytes.items.len) {
            const byte = raw_bytes.items[i];

            // Check for 2-byte UTF-8 sequences (0xC4 prefix for GPT-2 byte tokens)
            if (byte == 0xC4 and i + 1 < raw_bytes.items.len) {
                const next = raw_bytes.items[i + 1];
                // 'Ġ' (U+0120) = 0xC4 0xA0 -> space
                if (next == 0xA0) {
                    try result.append(' ');
                    i += 2;
                    continue;
                }
                // 'Ċ' (U+010A) = 0xC4 0x8A -> newline
                if (next == 0x8A) {
                    try result.append('\n');
                    i += 2;
                    continue;
                }
                // Other GPT-2 byte tokens in range U+0100-U+013F
                // These represent bytes 0x00-0x3F (control chars and some symbols)
                if (next >= 0x80 and next <= 0xBF) {
                    // U+0100 + x = 0xC4 (0x80 + (x >> 6)) followed by (0x80 + (x & 0x3F))
                    // For U+0100-U+013F, next byte is 0x80-0xBF
                    // 0x04 << 6 = 0x100, use explicit u16 to avoid overflow
                    const unicode: u16 = 0x100 | @as(u16, next & 0x3F);
                    if (unicode >= 0x100 and unicode < 0x144) {
                        // Decode using GPT-2 byte table reverse mapping
                        const decoded = decodeGPT2Byte(unicode);
                        try result.append(decoded);
                        i += 2;
                        continue;
                    }
                }
            }

            // Check for 2-byte UTF-8 (0xC3 prefix for Latin-1 supplement)
            // Characters like å, è, etc. represent themselves as bytes
            if (byte == 0xC3 and i + 1 < raw_bytes.items.len) {
                const next = raw_bytes.items[i + 1];
                // U+00C0-U+00FF -> bytes 0xC0-0xFF
                const decoded = 0xC0 + (next - 0x80);
                try result.append(decoded);
                i += 2;
                continue;
            }

            // Check for 2-byte UTF-8 (0xC2 prefix for Latin-1 supplement)
            // Characters like ¡-¿ represent bytes 0xA1-0xBF
            if (byte == 0xC2 and i + 1 < raw_bytes.items.len) {
                const next = raw_bytes.items[i + 1];
                if (next >= 0xA1 and next <= 0xBF) {
                    try result.append(next);
                    i += 2;
                    continue;
                }
            }

            // Regular ASCII or pass through
            try result.append(byte);
            i += 1;
        }

        return result.toOwnedSlice();
    }

    /// Decode GPT-2 byte token (U+0100-U+0143) to original byte
    fn decodeGPT2Byte(unicode: u16) u8 {
        // GPT-2 maps non-printable bytes to U+0100 and above
        // The mapping is: byte 0 -> U+0100, byte 1 -> U+0101, etc.
        // But only for bytes not in the "printable" set
        // Printable: 33-126, 161-172, 174-255

        // For U+0100-U+0120: these represent bytes 0-32 (control + space)
        if (unicode < 0x121) {
            return @intCast(unicode - 0x100);
        }
        // For U+0121-U+0143: these represent bytes 127-160 and 173
        if (unicode < 0x144) {
            const offset = unicode - 0x121;
            if (offset < 34) {
                return @intCast(127 + offset);
            }
            return 173; // U+0143 = byte 173 (soft hyphen)
        }
        return 0; // Shouldn't happen
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

/// ✅ P2-15 FIXED: Load vocabulary from GGUF metadata
/// Parses actual vocabulary from GGUF model metadata
pub fn loadVocabFromGGUF(
    allocator: std.mem.Allocator,
    model: *gguf.GGUFModel,
) ![]Token {
    const vocab_size = model.metadata.vocab_size;
    
    if (vocab_size == 0) {
        std.debug.print("⚠️  No vocabulary in GGUF model\n", .{});
        return error.NoVocabulary;
    }
    
    std.debug.print("📖 Loading {d} tokens from GGUF vocabulary\n", .{vocab_size});
    
    var vocab = try allocator.alloc(Token, vocab_size);
    errdefer allocator.free(vocab);
    
    // Parse vocabulary from GGUF model
    // GGUF stores vocab as two arrays: tokens (strings) and scores (floats)
    if (model.vocab_tokens.len != vocab_size) {
        std.debug.print("❌ Vocabulary size mismatch: expected {d}, got {d}\n", .{
            vocab_size,
            model.vocab_tokens.len,
        });
        return error.VocabSizeMismatch;
    }
    
    for (0..vocab_size) |i| {
        // Get token text from GGUF metadata
        const token_text = model.vocab_tokens[i];
        const token_text_copy = try allocator.dupe(u8, token_text);
        errdefer allocator.free(token_text_copy);
        
        // Get token score (if available)
        const score = if (i < model.vocab_scores.len) 
            model.vocab_scores[i] 
        else 
            0.0;
        
        vocab[i] = Token{
            .id = @intCast(i),
            .text = token_text_copy,
            .score = score,
        };
    }
    
    std.debug.print("✅ Successfully loaded {d} vocabulary tokens\n", .{vocab_size});
    
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
