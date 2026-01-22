const std = @import("std");
const mem = std.mem;
const fs = std.fs;
const mhc_config = @import("mhc_configuration");
const mhc_constraints = @import("mhc_constraints");
// Note: Avoid importing transformer to prevent circular dependencies
// const transformer = @import("transformer");
const mhc_parser = @import("gguf_mhc_parser");

/// GGUF v3 File Loader with mHC Metadata Support
/// Implements the GGUF format specification for loading GGML models
/// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
/// 
/// Day 38: Enhanced with automatic mHC configuration detection and loading

// ============================================================================
// Constants
// ============================================================================

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_VERSION: u32 = 3;

// ============================================================================
// Enums
// ============================================================================

pub const QuantizationType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
};

// Alias for compatibility
pub const GGMLType = QuantizationType;

pub const Architecture = enum {
    Llama,
    Lfm2,
    Mistral,
    Phi,
    Gemma,
    Qwen,
    Unknown,

    /// Check if this architecture can use the Llama-style loader
    /// Many architectures share the same transformer structure
    pub fn isLlamaCompatible(self: Architecture) bool {
        return switch (self) {
            .Llama, .Mistral, .Phi, .Gemma, .Qwen => true,
            .Lfm2, .Unknown => false,
        };
    }
};

/// Detect architecture from GGUF general.architecture string
/// Maps various model family names to our Architecture enum
pub fn detectArchitecture(arch_str: []const u8) Architecture {
    // Convert to lowercase for case-insensitive matching
    var lower_buf: [64]u8 = undefined;
    const len = @min(arch_str.len, lower_buf.len);
    for (0..len) |i| {
        lower_buf[i] = std.ascii.toLower(arch_str[i]);
    }
    const lower = lower_buf[0..len];

    // Llama family (including CodeLlama, Llama2, Llama3, etc.)
    if (mem.startsWith(u8, lower, "llama") or mem.eql(u8, lower, "llama")) {
        return .Llama;
    }

    // LFM2 (Liquid Foundation Model 2)
    if (mem.startsWith(u8, lower, "lfm2") or mem.eql(u8, lower, "lfm2")) {
        return .Lfm2;
    }

    // Mistral family
    if (mem.startsWith(u8, lower, "mistral") or mem.eql(u8, lower, "mistral")) {
        return .Mistral;
    }

    // Phi family (phi, phi2, phi3, phi-3, phi3.5, etc.)
    if (mem.startsWith(u8, lower, "phi")) {
        return .Phi;
    }

    // Gemma family (gemma, gemma2, etc.)
    if (mem.startsWith(u8, lower, "gemma")) {
        return .Gemma;
    }

    // Qwen family (qwen, qwen2, qwen2.5, etc.)
    if (mem.startsWith(u8, lower, "qwen")) {
        return .Qwen;
    }

    // Falcon - uses similar architecture to Llama
    if (mem.startsWith(u8, lower, "falcon")) {
        return .Llama;
    }

    // StarCoder/StarChat - uses similar architecture
    if (mem.startsWith(u8, lower, "starcoder") or mem.startsWith(u8, lower, "starchat")) {
        return .Llama;
    }

    // Yi models - Llama-compatible
    if (mem.startsWith(u8, lower, "yi")) {
        return .Llama;
    }

    // DeepSeek - Llama-compatible
    if (mem.startsWith(u8, lower, "deepseek")) {
        return .Llama;
    }

    // InternLM - Llama-compatible
    if (mem.startsWith(u8, lower, "internlm")) {
        return .Llama;
    }

    // Baichuan - Llama-compatible
    if (mem.startsWith(u8, lower, "baichuan")) {
        return .Llama;
    }

    // Granite (IBM) - Llama-compatible
    if (mem.startsWith(u8, lower, "granite")) {
        return .Llama;
    }

    std.debug.print("   ⚠️  Unknown architecture: \"{s}\", treating as Llama-compatible\n", .{arch_str});
    return .Llama; // Default to Llama for unknown architectures (most are compatible)
}

pub const MetadataValueType = enum(u32) {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
};

// ============================================================================
// Structures
// ============================================================================

pub const GGUFHeader = struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

pub const TensorInfo = struct {
    name: []const u8,
    n_dimensions: u32,
    dimensions: [4]u64,
    quant_type: QuantizationType,
    offset: u64,

    pub fn deinit(self: *TensorInfo, allocator: mem.Allocator) void {
        allocator.free(self.name);
    }

    pub fn size(self: *const TensorInfo) u64 {
        var total: u64 = 1;
        for (0..self.n_dimensions) |i| {
            total *= self.dimensions[i];
        }
        return total;
    }

    pub fn bytesPerElement(self: *const TensorInfo) usize {
        return switch (self.quant_type) {
            .F32 => 4,
            .F16 => 2,
            .Q4_0 => 18, // 32 values per block: 2 bytes scale + 16 bytes data
            .Q4_1 => 20, // 32 values per block: 2+2 + 16
            .Q5_0 => 22, // 32 values per block: 2 + 4 + 16
            .Q5_1 => 24, // 32 values per block: 2+2 + 4 + 16
            .Q8_0 => 34, // 32 values per block: 2 + 32
            .Q4_K => 144, // 256 values per block: 2+2+12+128
            .Q6_K => 210, // 256 values per block: 128 ql + 64 qh + 16 scales + 2 d
            else => 4, // Default to f32 size
        };
    }

    pub fn dataSize(self: *const TensorInfo) usize {
        const elem_count = self.size();
        const block_size: usize = switch (self.quant_type) {
            .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q8_0 => 32,
            .Q4_K, .Q6_K => 256,
            else => 1,
        };

        const n_blocks = (elem_count + block_size - 1) / block_size;
        return n_blocks * self.bytesPerElement();
    }
};

pub const ModelMetadata = struct {
    // Standard metadata
    architecture: Architecture,
    vocab_size: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    hidden_size: u32,
    intermediate_size: u32,
    max_seq_len: u32,
    rope_theta: f32,
    rms_norm_eps: f32,
    conv_kernel: u32,
    
    // NEW Day 38: mHC metadata
    mhc_enabled: bool = false,
    mhc_version: ?[]const u8 = null,
    mhc_description: ?[]const u8 = null,
    mhc_config: ?mhc_constraints.MHCConfig = null,
    mhc_transformer_config: ?mhc_config.MHCTransformerConfig = null,

    pub fn default() ModelMetadata {
        return .{
            .architecture = .Unknown,
            .vocab_size = 32000,
            .n_layers = 32,
            .n_heads = 32,
            .n_kv_heads = 32,
            .hidden_size = 4096,
            .intermediate_size = 11008,
            .max_seq_len = 2048,
            .rope_theta = 10000.0,
            .rms_norm_eps = 1e-5,
            .conv_kernel = 3,
            // mHC fields default to disabled/null
            .mhc_enabled = false,
            .mhc_version = null,
            .mhc_description = null,
            .mhc_config = null,
            .mhc_transformer_config = null,
        };
    }
    
    pub fn hasMHC(self: *const ModelMetadata) bool {
        return self.mhc_enabled and self.mhc_config != null;
    }
    
    pub fn getMHCConfig(self: *const ModelMetadata) ?mhc_constraints.MHCConfig {
        if (!self.mhc_enabled) return null;
        return self.mhc_config;
    }
};

// ============================================================================
// Day 38: mHC Metadata Detection and Loading
// ============================================================================

pub const MHCDetectionSource = enum {
    None,       // No mHC metadata found
    Explicit,   // Explicit mhc.enabled flag
    Heuristic,  // Inferred from mhc.* keys
};

pub const MHCDetectionResult = struct {
    detected: bool,
    confidence: f32,  // [0.0, 1.0]
    source: MHCDetectionSource,
    mhc_key_count: u32,  // Number of mhc.* keys found
};

/// Temporary storage for mHC metadata during parsing
pub const MHCMetadataBuilder = struct {
    // Detection
    has_enabled_key: bool = false,
    enabled_value: bool = false,
    mhc_key_count: u32 = 0,
    
    // Core config
    version: ?[]const u8 = null,
    description: ?[]const u8 = null,
    sinkhorn_iterations: ?u32 = null,
    manifold_epsilon: ?f32 = null,
    stability_threshold: ?f32 = null,
    manifold_beta: ?f32 = null,
    manifold_type: ?[]const u8 = null,
    early_stopping: ?bool = null,
    
    // Transformer config
    attention_enabled: ?bool = null,
    ffn_enabled: ?bool = null,
    residual_enabled: ?bool = null,
    layer_range_start: ?u32 = null,
    layer_range_end: ?u32 = null,
    
    pub fn init() MHCMetadataBuilder {
        return .{};
    }
    
    pub fn recordKey(self: *MHCMetadataBuilder, key: []const u8) void {
        if (mem.startsWith(u8, key, "mhc.")) {
            self.mhc_key_count += 1;
        }
    }
    
    pub fn detectMHC(self: *const MHCMetadataBuilder) MHCDetectionResult {
        // Level 1: Explicit flag
        if (self.has_enabled_key) {
            return .{
                .detected = self.enabled_value,
                .confidence = 1.0,
                .source = .Explicit,
                .mhc_key_count = self.mhc_key_count,
            };
        }
        
        // Level 2: Heuristic (any mhc.* keys)
        if (self.mhc_key_count > 0) {
            const confidence: f32 = if (self.mhc_key_count >= 3) 0.9 else 0.5;
            return .{
                .detected = true,
                .confidence = confidence,
                .source = .Heuristic,
                .mhc_key_count = self.mhc_key_count,
            };
        }
        
        // Level 3: No mHC metadata
        return .{
            .detected = false,
            .confidence = 1.0,
            .source = .None,
            .mhc_key_count = 0,
        };
    }
    
    pub fn buildMHCConfig(self: *const MHCMetadataBuilder) mhc_constraints.MHCConfig {
        return .{
            .enabled = true,  // Already detected
            .sinkhorn_iterations = self.sinkhorn_iterations orelse 10,
            .manifold_epsilon = self.manifold_epsilon orelse 1e-6,
            .stability_threshold = self.stability_threshold orelse 1e-4,
            .manifold_beta = self.manifold_beta orelse 10.0,
            .log_stability_metrics = false,
            .layer_range = null,  // Set via transformer config
            .early_stopping = self.early_stopping orelse true,
        };
    }
    
    pub fn buildTransformerConfig(
        self: *const MHCMetadataBuilder,
        core_config: mhc_constraints.MHCConfig,
    ) mhc_config.MHCTransformerConfig {
        var config = mhc_config.MHCTransformerConfig{
            .enabled = true,
            .attention_enabled = self.attention_enabled orelse true,
            .ffn_enabled = self.ffn_enabled orelse true,
            .residual_enabled = self.residual_enabled orelse false,
            .layer_range = null,
            .core = .{
                .enabled = core_config.enabled,
                .sinkhorn_iterations = core_config.sinkhorn_iterations,
                .manifold_epsilon = core_config.manifold_epsilon,
                .stability_threshold = core_config.stability_threshold,
                .manifold_beta = core_config.manifold_beta,
                .log_stability_metrics = core_config.log_stability_metrics,
                .early_stopping = core_config.early_stopping,
            },
            .attention_stability_threshold = 1e-4,
            .ffn_stability_threshold = 1e-4,
            .residual_stability_threshold = 1e-4,
            .abort_on_instability = false,
            .stability_callback = null,
        };

        // Set layer range if both start and end are present
        if (self.layer_range_start != null and self.layer_range_end != null) {
            config.layer_range = .{
                .start = self.layer_range_start.?,
                .end = self.layer_range_end.?,
            };
        }

        return config;
    }
};

pub const GGUFModel = struct {
    allocator: mem.Allocator,
    file: fs.File,
    header: GGUFHeader,
    metadata: ModelMetadata,
    tensors: []TensorInfo,
    vocab_tokens: [][]u8,
    vocab_scores: []f32,
    tensor_data_offset: u64, // Base offset where tensor data starts (after header/metadata/tensor info, aligned to 32 bytes)

    pub fn load(allocator: mem.Allocator, path: []const u8) !GGUFModel {
        std.debug.print("\n📂 Loading GGUF model: {s}\n", .{path});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

        // Open file
        var file = try fs.cwd().openFile(path, .{});
        errdefer file.close();

        // Read and validate header
        var raw_header: [@sizeOf(GGUFHeader)]u8 = undefined;
        const bytes_read = try file.readAll(&raw_header);
        if (bytes_read != raw_header.len) {
            std.debug.print("❌ Failed to read header (got {d} bytes)\n", .{bytes_read});
            return error.InvalidGGUFFile;
        }

        const header = GGUFHeader{
            .magic = mem.readInt(u32, raw_header[0..4], .little),
            .version = mem.readInt(u32, raw_header[4..8], .little),
            .tensor_count = mem.readInt(u64, raw_header[8..16], .little),
            .metadata_kv_count = mem.readInt(u64, raw_header[16..24], .little),
        };

        if (header.magic != GGUF_MAGIC) {
            std.debug.print("❌ Invalid magic number: 0x{x} (expected 0x{x})\n", .{ header.magic, GGUF_MAGIC });
            return error.InvalidMagicNumber;
        }

        if (header.version != GGUF_VERSION) {
            std.debug.print("⚠️  Unsupported version: {d} (expected {d})\n", .{ header.version, GGUF_VERSION });
            return error.UnsupportedVersion;
        }

        std.debug.print("✅ Header valid:\n", .{});
        std.debug.print("   Magic: GGUF\n", .{});
        std.debug.print("   Version: {d}\n", .{header.version});
        std.debug.print("   Tensors: {d}\n", .{header.tensor_count});
        std.debug.print("   Metadata keys: {d}\n", .{header.metadata_kv_count});

        // Prepare vocab containers
        var vocab_tokens = try std.ArrayList([]u8).initCapacity(allocator, 256);
        errdefer {
            for (vocab_tokens.items) |t| allocator.free(t);
            vocab_tokens.deinit();
        }
        var vocab_scores = try std.ArrayList(f32).initCapacity(allocator, 256);
        errdefer vocab_scores.deinit();

        // Parse metadata
        const metadata = try parseMetadata(allocator, file, header.metadata_kv_count, &vocab_tokens, &vocab_scores);

        // Parse tensor metadata
        const tensors = try parseTensorMetadata(allocator, file, header.tensor_count);

        // Calculate tensor data offset (current position aligned to 32 bytes)
        const current_pos = try file.getPos();
        const tensor_data_offset = (current_pos + 31) & ~@as(u64, 31);

        std.debug.print("\n✅ GGUF model loaded successfully!\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});

        return GGUFModel{
            .allocator = allocator,
            .file = file,
            .header = header,
            .metadata = metadata,
            .tensors = tensors,
            .vocab_tokens = try vocab_tokens.toOwnedSlice(),
            .vocab_scores = try vocab_scores.toOwnedSlice(),
            .tensor_data_offset = tensor_data_offset,
        };
    }

    pub fn deinit(self: *GGUFModel) void {
        for (self.tensors) |*tensor| {
            tensor.deinit(self.allocator);
        }
        self.allocator.free(self.tensors);

        for (self.vocab_tokens) |token| {
            self.allocator.free(token);
        }
        self.allocator.free(self.vocab_tokens);
        self.allocator.free(self.vocab_scores);

        self.file.close();
    }

    pub fn getTensor(self: *GGUFModel, name: []const u8) ?*TensorInfo {
        for (self.tensors) |*tensor| {
            if (mem.eql(u8, tensor.name, name)) {
                return tensor;
            }
        }
        return null;
    }

    pub fn findTensor(self: *GGUFModel, name: []const u8) ?usize {
        for (self.tensors, 0..) |*tensor, i| {
            if (mem.eql(u8, tensor.name, name)) {
                return i;
            }
        }
        return null;
    }

    pub fn getTensorData(self: *GGUFModel, tensor_idx: usize) ![]const u8 {
        if (tensor_idx >= self.tensors.len) {
            return error.InvalidTensorIndex;
        }

        const tensor = &self.tensors[tensor_idx];
        const data_size = tensor.dataSize();

        // Allocate buffer for tensor data
        const data = try self.allocator.alloc(u8, data_size);

        // Seek to tensor data: base offset + tensor's relative offset
        const absolute_offset = self.tensor_data_offset + tensor.offset;

        // Debug: Print tensor data loading info
        if (tensor_idx < 3 or std.mem.eql(u8, tensor.name, "token_embd.weight")) {
            std.debug.print("📥 getTensorData: {s} idx={d} base={d} rel={d} abs={d} size={d} quant={s}\n", .{
                tensor.name, tensor_idx, self.tensor_data_offset, tensor.offset, absolute_offset, data_size, @tagName(tensor.quant_type),
            });
        }

        try self.file.seekTo(absolute_offset);
        const bytes_read = try self.file.read(data);

        if (bytes_read != data_size) {
            self.allocator.free(data);
            return error.IncompleteTensorData;
        }

        // Debug: Print first few bytes
        if (tensor_idx < 3 or std.mem.eql(u8, tensor.name, "token_embd.weight")) {
            std.debug.print("   First 16 bytes: ", .{});
            for (data[0..@min(16, data.len)]) |b| {
                std.debug.print("{x:0>2} ", .{b});
            }
            std.debug.print("\n", .{});
        }

        return data;
    }

    pub fn printSummary(self: *GGUFModel) void {
        std.debug.print("\n📊 Model Summary\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
        std.debug.print("Architecture: {s}\n", .{@tagName(self.metadata.architecture)});
        std.debug.print("Vocabulary: {d} tokens\n", .{self.metadata.vocab_size});
        std.debug.print("Layers: {d}\n", .{self.metadata.n_layers});
        std.debug.print("Attention heads: {d}\n", .{self.metadata.n_heads});
        std.debug.print("KV heads: {d}\n", .{self.metadata.n_kv_heads});
        std.debug.print("Hidden size: {d}\n", .{self.metadata.hidden_size});
        std.debug.print("Intermediate size: {d}\n", .{self.metadata.intermediate_size});
        std.debug.print("Max sequence: {d}\n", .{self.metadata.max_seq_len});
        std.debug.print("RoPE theta: {d}\n", .{self.metadata.rope_theta});
        std.debug.print("\nTensors: {d}\n", .{self.tensors.len});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
    }
};

// ============================================================================
// Metadata Parsing
// ============================================================================

fn parseMetadata(
    allocator: mem.Allocator,
    file: fs.File,
    count: u64,
    vocab_tokens: *std.ArrayList([]u8),
    vocab_scores: *std.ArrayList(f32),
) !ModelMetadata {
    std.debug.print("\n📋 Parsing metadata ({d} keys)...\n", .{count});

    var metadata = ModelMetadata.default();
    
    // Day 38: mHC metadata builder
    var mhc_builder = MHCMetadataBuilder.init();

    for (0..count) |i| {
        _ = i;

        // Read key name length
        var key_len: u64 = undefined;
        _ = try file.read(mem.asBytes(&key_len));

        // Read key name
        const key_name = try allocator.alloc(u8, key_len);
        defer allocator.free(key_name);
        _ = try file.read(key_name);

        // Read value type
        var value_type: u32 = undefined;
        _ = try file.read(mem.asBytes(&value_type));

        // Day 38: Check if this is an mHC key
        if (mem.startsWith(u8, key_name, "mhc.")) {
            try mhc_parser.parseMHCMetadataKey(
                allocator,
                file,
                key_name,
                @enumFromInt(value_type),
                &mhc_builder,
            );
        } else {
            // Parse standard metadata value
            try parseMetadataValue(allocator, file, key_name, @enumFromInt(value_type), &metadata, vocab_tokens, vocab_scores);
        }
    }

    // If vocab tokens were loaded, trust their size for vocab_size to stay aligned.
    if (vocab_tokens.items.len > 0) {
        metadata.vocab_size = @intCast(vocab_tokens.items.len);
    }

    std.debug.print("✅ Metadata parsed\n", .{});
    
    // Day 38: Finalize mHC metadata
    try mhc_parser.finalizeMHCMetadata(&mhc_builder, &metadata, allocator);
    
    return metadata;
}

fn parseMetadataValue(
    allocator: mem.Allocator,
    file: fs.File,
    key: []const u8,
    value_type: MetadataValueType,
    metadata: *ModelMetadata,
    vocab_tokens: *std.ArrayList([]u8),
    vocab_scores: *std.ArrayList(f32),
) !void {
    switch (value_type) {
        .UInt32 => {
            var value: u32 = undefined;
            _ = try file.read(mem.asBytes(&value));
            updateMetadataU32(key, value, metadata);
        },
        .UInt64 => {
            var value: u64 = undefined;
            _ = try file.read(mem.asBytes(&value));
            updateMetadataU64(key, value, metadata);
        },
        .Float32 => {
            var value: f32 = undefined;
            _ = try file.read(mem.asBytes(&value));
            updateMetadataF32(key, value, metadata);
        },
        .String => {
            var str_len: u64 = undefined;
            _ = try file.read(mem.asBytes(&str_len));

            if (mem.eql(u8, key, "general.architecture")) {
                const buf = try allocator.alloc(u8, str_len);
                defer allocator.free(buf);
                _ = try file.read(buf);
                // Detect architecture from GGUF metadata
                metadata.architecture = detectArchitecture(buf);
                std.debug.print("   Architecture detected: {s} (from \"{s}\")\n", .{ @tagName(metadata.architecture), buf });
            } else {
                // Skip string content for keys we do not explicitly parse
                try file.seekBy(@intCast(str_len));
            }
        },
        .Array => {
            var type_code: u32 = undefined;
            _ = try file.read(mem.asBytes(&type_code));
            var len: u64 = undefined;
            _ = try file.read(mem.asBytes(&len));

            const item_type: MetadataValueType = @enumFromInt(type_code);

            if (mem.eql(u8, key, "tokenizer.ggml.tokens") and item_type == .String) {
                std.debug.print("   Loading vocabulary ({d} tokens)...\n", .{len});
                for (0..len) |_| {
                    var str_len: u64 = undefined;
                    _ = try file.read(mem.asBytes(&str_len));
                    const token_bytes = try allocator.alloc(u8, str_len);
                    _ = try file.read(token_bytes);
                    try vocab_tokens.append(token_bytes);
                }
            } else if (mem.eql(u8, key, "tokenizer.ggml.scores") and item_type == .Float32) {
                std.debug.print("   Loading scores ({d} values)...\n", .{len});
                for (0..len) |_| {
                    var score: f32 = undefined;
                    _ = try file.read(mem.asBytes(&score));
                    try vocab_scores.append(score);
                }
            } else {
                // Skip array
                for (0..len) |_| {
                    try skipMetadataValue(file, item_type);
                }
            }
        },
        else => {
            // Skip unknown types to keep the stream aligned
            std.debug.print("⚠️  Skipping unknown metadata type: {s}\n", .{@tagName(value_type)});
            try skipMetadataValue(file, value_type);
        },
    }
}

fn skipMetadataValue(file: fs.File, value_type: MetadataValueType) !void {
    switch (value_type) {
        .UInt8, .Int8, .Bool => try file.seekBy(1),
        .UInt16, .Int16 => try file.seekBy(2),
        .UInt32, .Int32, .Float32 => try file.seekBy(4),
        .UInt64, .Int64, .Float64 => try file.seekBy(8),
        .String => {
            var str_len: u64 = undefined;
            _ = try file.read(mem.asBytes(&str_len));
            try file.seekBy(@intCast(str_len));
        },
        .Array => {
            var type_code: u32 = undefined;
            _ = try file.read(mem.asBytes(&type_code));
            var len: u64 = undefined;
            _ = try file.read(mem.asBytes(&len));
            const item_type: MetadataValueType = @enumFromInt(type_code);
            for (0..len) |_| {
                try skipMetadataValue(file, item_type);
            }
        },
    }
}

fn updateMetadataU32(key: []const u8, value: u32, metadata: *ModelMetadata) void {
    // llama.* keys
    if (mem.eql(u8, key, "llama.vocab_size")) {
        metadata.vocab_size = value;
    } else if (mem.eql(u8, key, "llama.block_count")) {
        metadata.n_layers = value;
    } else if (mem.eql(u8, key, "llama.attention.head_count")) {
        metadata.n_heads = value;
    } else if (mem.eql(u8, key, "llama.attention.head_count_kv")) {
        metadata.n_kv_heads = value;
    } else if (mem.eql(u8, key, "llama.embedding_length")) {
        metadata.hidden_size = value;
    } else if (mem.eql(u8, key, "llama.feed_forward_length")) {
        metadata.intermediate_size = value;
    } else if (mem.eql(u8, key, "llama.context_length")) {
        metadata.max_seq_len = value;
    // phi3.* keys (Phi-3 models)
    } else if (mem.eql(u8, key, "phi3.vocab_size")) {
        metadata.vocab_size = value;
    } else if (mem.eql(u8, key, "phi3.block_count")) {
        metadata.n_layers = value;
    } else if (mem.eql(u8, key, "phi3.attention.head_count")) {
        metadata.n_heads = value;
    } else if (mem.eql(u8, key, "phi3.attention.head_count_kv")) {
        metadata.n_kv_heads = value;
    } else if (mem.eql(u8, key, "phi3.embedding_length")) {
        metadata.hidden_size = value;
    } else if (mem.eql(u8, key, "phi3.feed_forward_length")) {
        metadata.intermediate_size = value;
    } else if (mem.eql(u8, key, "phi3.context_length")) {
        metadata.max_seq_len = value;
    // qwen2.* keys (Qwen models)
    } else if (mem.eql(u8, key, "qwen2.vocab_size")) {
        metadata.vocab_size = value;
    } else if (mem.eql(u8, key, "qwen2.block_count")) {
        metadata.n_layers = value;
    } else if (mem.eql(u8, key, "qwen2.attention.head_count")) {
        metadata.n_heads = value;
    } else if (mem.eql(u8, key, "qwen2.attention.head_count_kv")) {
        metadata.n_kv_heads = value;
    } else if (mem.eql(u8, key, "qwen2.embedding_length")) {
        metadata.hidden_size = value;
    } else if (mem.eql(u8, key, "qwen2.feed_forward_length")) {
        metadata.intermediate_size = value;
    } else if (mem.eql(u8, key, "qwen2.context_length")) {
        metadata.max_seq_len = value;
    // gemma.* keys (Gemma models)
    } else if (mem.eql(u8, key, "gemma.vocab_size")) {
        metadata.vocab_size = value;
    } else if (mem.eql(u8, key, "gemma.block_count")) {
        metadata.n_layers = value;
    } else if (mem.eql(u8, key, "gemma.attention.head_count")) {
        metadata.n_heads = value;
    } else if (mem.eql(u8, key, "gemma.attention.head_count_kv")) {
        metadata.n_kv_heads = value;
    } else if (mem.eql(u8, key, "gemma.embedding_length")) {
        metadata.hidden_size = value;
    } else if (mem.eql(u8, key, "gemma.feed_forward_length")) {
        metadata.intermediate_size = value;
    } else if (mem.eql(u8, key, "gemma.context_length")) {
        metadata.max_seq_len = value;
    // mistral.* keys (Mistral models)
    } else if (mem.eql(u8, key, "mistral.vocab_size")) {
        metadata.vocab_size = value;
    } else if (mem.eql(u8, key, "mistral.block_count")) {
        metadata.n_layers = value;
    } else if (mem.eql(u8, key, "mistral.attention.head_count")) {
        metadata.n_heads = value;
    } else if (mem.eql(u8, key, "mistral.attention.head_count_kv")) {
        metadata.n_kv_heads = value;
    } else if (mem.eql(u8, key, "mistral.embedding_length")) {
        metadata.hidden_size = value;
    } else if (mem.eql(u8, key, "mistral.feed_forward_length")) {
        metadata.intermediate_size = value;
    } else if (mem.eql(u8, key, "mistral.context_length")) {
        metadata.max_seq_len = value;
    // lfm2.* keys
    } else if (mem.eql(u8, key, "lfm2.vocab_size")) {
        metadata.vocab_size = value;
    } else if (mem.eql(u8, key, "lfm2.block_count")) {
        metadata.n_layers = value;
    } else if (mem.eql(u8, key, "lfm2.attention.head_count")) {
        metadata.n_heads = value;
    } else if (mem.eql(u8, key, "lfm2.attention.head_count_kv")) {
        metadata.n_kv_heads = value;
    } else if (mem.eql(u8, key, "lfm2.embedding_length")) {
        metadata.hidden_size = value;
    } else if (mem.eql(u8, key, "lfm2.feed_forward_length")) {
        metadata.intermediate_size = value;
    } else if (mem.eql(u8, key, "lfm2.context_length")) {
        metadata.max_seq_len = value;
    } else if (mem.eql(u8, key, "lfm2.shortconv.l_cache")) {
        metadata.conv_kernel = value;
    }
}

fn updateMetadataU64(key: []const u8, value: u64, metadata: *ModelMetadata) void {
    // Handle u64 values (cast to u32 where appropriate)
    // llama.*
    if (mem.eql(u8, key, "llama.vocab_size")) {
        metadata.vocab_size = @intCast(value);
    } else if (mem.eql(u8, key, "llama.context_length")) {
        metadata.max_seq_len = @intCast(value);
    // phi3.*
    } else if (mem.eql(u8, key, "phi3.vocab_size")) {
        metadata.vocab_size = @intCast(value);
    } else if (mem.eql(u8, key, "phi3.context_length")) {
        metadata.max_seq_len = @intCast(value);
    // qwen2.*
    } else if (mem.eql(u8, key, "qwen2.vocab_size")) {
        metadata.vocab_size = @intCast(value);
    } else if (mem.eql(u8, key, "qwen2.context_length")) {
        metadata.max_seq_len = @intCast(value);
    // gemma.*
    } else if (mem.eql(u8, key, "gemma.vocab_size")) {
        metadata.vocab_size = @intCast(value);
    } else if (mem.eql(u8, key, "gemma.context_length")) {
        metadata.max_seq_len = @intCast(value);
    // mistral.*
    } else if (mem.eql(u8, key, "mistral.vocab_size")) {
        metadata.vocab_size = @intCast(value);
    } else if (mem.eql(u8, key, "mistral.context_length")) {
        metadata.max_seq_len = @intCast(value);
    // lfm2.*
    } else if (mem.eql(u8, key, "lfm2.vocab_size")) {
        metadata.vocab_size = @intCast(value);
    } else if (mem.eql(u8, key, "lfm2.context_length")) {
        metadata.max_seq_len = @intCast(value);
    }
}

fn updateMetadataF32(key: []const u8, value: f32, metadata: *ModelMetadata) void {
    // llama.*
    if (mem.eql(u8, key, "llama.rope.freq_base")) {
        metadata.rope_theta = value;
    } else if (mem.eql(u8, key, "llama.attention.layer_norm_rms_epsilon")) {
        metadata.rms_norm_eps = value;
    // phi3.*
    } else if (mem.eql(u8, key, "phi3.rope.freq_base")) {
        metadata.rope_theta = value;
    } else if (mem.eql(u8, key, "phi3.attention.layer_norm_rms_epsilon")) {
        metadata.rms_norm_eps = value;
    // qwen2.*
    } else if (mem.eql(u8, key, "qwen2.rope.freq_base")) {
        metadata.rope_theta = value;
    } else if (mem.eql(u8, key, "qwen2.attention.layer_norm_rms_epsilon")) {
        metadata.rms_norm_eps = value;
    // gemma.*
    } else if (mem.eql(u8, key, "gemma.rope.freq_base")) {
        metadata.rope_theta = value;
    } else if (mem.eql(u8, key, "gemma.attention.layer_norm_rms_epsilon")) {
        metadata.rms_norm_eps = value;
    // mistral.*
    } else if (mem.eql(u8, key, "mistral.rope.freq_base")) {
        metadata.rope_theta = value;
    } else if (mem.eql(u8, key, "mistral.attention.layer_norm_rms_epsilon")) {
        metadata.rms_norm_eps = value;
    // lfm2.*
    } else if (mem.eql(u8, key, "lfm2.rope.freq_base")) {
        metadata.rope_theta = value;
    } else if (mem.eql(u8, key, "lfm2.attention.layer_norm_rms_epsilon")) {
        metadata.rms_norm_eps = value;
    }
}

// ============================================================================
// Tensor Metadata Parsing
// ============================================================================

fn parseTensorMetadata(
    allocator: mem.Allocator,
    file: fs.File,
    count: u64,
) ![]TensorInfo {
    std.debug.print("\n📦 Parsing tensor metadata ({d} tensors)...\n", .{count});

    var tensors = try allocator.alloc(TensorInfo, count);
    errdefer allocator.free(tensors);

    for (0..count) |i| {
        // Read tensor name length
        var name_len: u64 = undefined;
        _ = try file.read(mem.asBytes(&name_len));

        // Read tensor name
        const name = try allocator.alloc(u8, name_len);
        errdefer allocator.free(name);
        _ = try file.read(name);

        // Read dimensions
        var n_dims: u32 = undefined;
        _ = try file.read(mem.asBytes(&n_dims));

        var dims: [4]u64 = .{ 0, 0, 0, 0 };
        for (0..n_dims) |d| {
            _ = try file.read(mem.asBytes(&dims[d]));
        }

        // Read quantization type
        var quant_type: u32 = undefined;
        _ = try file.read(mem.asBytes(&quant_type));

        // Read offset
        var offset: u64 = undefined;
        _ = try file.read(mem.asBytes(&offset));

        tensors[i] = TensorInfo{
            .name = name,
            .n_dimensions = n_dims,
            .dimensions = dims,
            .quant_type = @enumFromInt(quant_type),
            .offset = offset,
        };

        if (i < 5) { // Print first 5 tensors
            std.debug.print("   [{d}] {s}: ", .{ i, name });
            std.debug.print("[", .{});
            for (0..n_dims) |d| {
                if (d > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{dims[d]});
            }
            std.debug.print("] {s}\n", .{@tagName(@as(QuantizationType, @enumFromInt(quant_type)))});
        }
    }

    if (count > 5) {
        std.debug.print("   ... and {d} more tensors\n", .{count - 5});
    }

    std.debug.print("✅ Tensor metadata parsed\n", .{});
    return tensors;
}

// ============================================================================
// Tensor Loading
// ============================================================================

pub const Tensor = struct {
    info: TensorInfo,
    data: []u8,
    allocator: mem.Allocator,

    pub fn loadFromFile(
        allocator: mem.Allocator,
        file: fs.File,
        info: *const TensorInfo,
    ) !Tensor {
        // Seek to tensor data
        try file.seekTo(info.offset);

        // Calculate data size
        const data_size = info.dataSize();

        std.debug.print("📥 Loading tensor: {s} ({d} bytes)\n", .{ info.name, data_size });

        // Allocate and read
        const data = try allocator.alloc(u8, data_size);
        errdefer allocator.free(data);

        const bytes_read = try file.read(data);

        if (bytes_read != data_size) {
            std.debug.print("❌ Read {d} bytes, expected {d}\n", .{ bytes_read, data_size });
            return error.IncompleteTensorData;
        }

        return Tensor{
            .info = info.*,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
    }
};

// ============================================================================
// Testing & Validation
// ============================================================================

pub fn validateModel(model: *GGUFModel) !void {
    std.debug.print("\n🔍 Validating model...\n", .{});

    // Check required tensors exist
    const required = [_][]const u8{
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
    };

    for (required) |tensor_name| {
        const tensor = model.getTensor(tensor_name);
        if (tensor == null) {
            std.debug.print("❌ Missing required tensor: {s}\n", .{tensor_name});
            return error.MissingTensor;
        }
        std.debug.print("✅ Found: {s}\n", .{tensor_name});
    }

    // Validate layer tensors
    for (0..model.metadata.n_layers) |i| {
        var buf: [256]u8 = undefined;
        const layer_prefix = try std.fmt.bufPrint(&buf, "blk.{d}.", .{i});

        const layer_tensors = [_][]const u8{
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
        };

        for (layer_tensors) |tensor_suffix| {
            const full_name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ layer_prefix, tensor_suffix });
            const tensor = model.getTensor(full_name);
            if (tensor == null and i == 0) {
                std.debug.print("⚠️  Layer 0 missing: {s}\n", .{full_name});
            }
        }
    }

    std.debug.print("✅ Model validation passed!\n", .{});
}
