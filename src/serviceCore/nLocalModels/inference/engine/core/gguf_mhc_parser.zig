const std = @import("std");
const mem = std.mem;
const fs = std.fs;
// Use module imports
const mhc_constraints = @import("mhc_constraints");
const gguf_loader = @import("gguf_loader");

/// Day 38: mHC Metadata Parser for GGUF Files
/// Handles detection and loading of mHC configuration from GGUF metadata

// ============================================================================
// mHC Metadata Key Parsing
// ============================================================================

/// Parse mHC-specific metadata keys and update the builder
pub fn parseMHCMetadataKey(
    allocator: std.mem.Allocator,
    file: fs.File,
    key: []const u8,
    value_type: gguf_loader.MetadataValueType,
    builder: *gguf_loader.MHCMetadataBuilder,
) !void {
    // Record that we found an mhc.* key
    builder.recordKey(key);
    
    // Parse based on key
    if (mem.eql(u8, key, "mhc.enabled")) {
        // Explicit enabled flag
        if (value_type == .Bool) {
            var value: u8 = undefined;
            _ = try file.read(mem.asBytes(&value));
            builder.has_enabled_key = true;
            builder.enabled_value = value != 0;
            std.debug.print("   mHC enabled: {}\n", .{builder.enabled_value});
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.version")) {
        // Version string
        if (value_type == .String) {
            var str_len: u64 = undefined;
            _ = try file.read(mem.asBytes(&str_len));
            const version_str = try allocator.alloc(u8, str_len);
            _ = try file.read(version_str);
            builder.version = version_str;
            std.debug.print("   mHC version: {s}\n", .{version_str});
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.description")) {
        // Description string
        if (value_type == .String) {
            var str_len: u64 = undefined;
            _ = try file.read(mem.asBytes(&str_len));
            const desc_str = try allocator.alloc(u8, str_len);
            _ = try file.read(desc_str);
            builder.description = desc_str;
            std.debug.print("   mHC description: {s}\n", .{desc_str});
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.config.sinkhorn_iterations")) {
        if (value_type == .UInt32) {
            var value: u32 = undefined;
            _ = try file.read(mem.asBytes(&value));
            if (value >= 1 and value <= 100) {
                builder.sinkhorn_iterations = value;
            } else {
                std.debug.print("   ⚠️  Invalid sinkhorn_iterations: {d}, using default\n", .{value});
            }
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.config.manifold_epsilon")) {
        if (value_type == .Float32) {
            var value: f32 = undefined;
            _ = try file.read(mem.asBytes(&value));
            if (value >= 1e-10 and value <= 1e-3) {
                builder.manifold_epsilon = value;
            } else {
                std.debug.print("   ⚠️  Invalid manifold_epsilon: {e}, using default\n", .{value});
            }
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.config.stability_threshold")) {
        if (value_type == .Float32) {
            var value: f32 = undefined;
            _ = try file.read(mem.asBytes(&value));
            if (value >= 1e-6 and value <= 1e-2) {
                builder.stability_threshold = value;
            } else {
                std.debug.print("   ⚠️  Invalid stability_threshold: {e}, using default\n", .{value});
            }
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.config.manifold_beta")) {
        if (value_type == .Float32) {
            var value: f32 = undefined;
            _ = try file.read(mem.asBytes(&value));
            if (value >= 0.1 and value <= 100.0) {
                builder.manifold_beta = value;
            } else {
                std.debug.print("   ⚠️  Invalid manifold_beta: {d}, using default\n", .{value});
            }
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.config.manifold_type")) {
        if (value_type == .String) {
            var str_len: u64 = undefined;
            _ = try file.read(mem.asBytes(&str_len));
            const type_str = try allocator.alloc(u8, str_len);
            _ = try file.read(type_str);
            builder.manifold_type = type_str;
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.config.early_stopping")) {
        if (value_type == .Bool) {
            var value: u8 = undefined;
            _ = try file.read(mem.asBytes(&value));
            builder.early_stopping = value != 0;
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.transformer.attention_enabled")) {
        if (value_type == .Bool) {
            var value: u8 = undefined;
            _ = try file.read(mem.asBytes(&value));
            builder.attention_enabled = value != 0;
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.transformer.ffn_enabled")) {
        if (value_type == .Bool) {
            var value: u8 = undefined;
            _ = try file.read(mem.asBytes(&value));
            builder.ffn_enabled = value != 0;
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.transformer.residual_enabled")) {
        if (value_type == .Bool) {
            var value: u8 = undefined;
            _ = try file.read(mem.asBytes(&value));
            builder.residual_enabled = value != 0;
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.transformer.layer_range_start")) {
        if (value_type == .UInt32) {
            var value: u32 = undefined;
            _ = try file.read(mem.asBytes(&value));
            builder.layer_range_start = value;
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.eql(u8, key, "mhc.transformer.layer_range_end")) {
        if (value_type == .UInt32) {
            var value: u32 = undefined;
            _ = try file.read(mem.asBytes(&value));
            builder.layer_range_end = value;
        } else {
            try skipValue(file, value_type);
        }
    } else if (mem.startsWith(u8, key, "mhc.training.")) {
        // Training metadata - parse but don't use for config
        try skipValue(file, value_type);
    } else if (mem.startsWith(u8, key, "mhc.")) {
        // Unknown mhc.* key - skip with warning
        std.debug.print("   ⚠️  Unknown mHC key: {s}\n", .{key});
        try skipValue(file, value_type);
    }
}

fn skipValue(file: fs.File, value_type: gguf_loader.MetadataValueType) !void {
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
            const item_type: gguf_loader.MetadataValueType = @enumFromInt(type_code);
            for (0..len) |_| {
                try skipValue(file, item_type);
            }
        },
    }
}

// ============================================================================
// mHC Configuration Finalization
// ============================================================================

/// Finalize mHC metadata and update model metadata
pub fn finalizeMHCMetadata(
    builder: *const gguf_loader.MHCMetadataBuilder,
    metadata: *gguf_loader.ModelMetadata,
    allocator: std.mem.Allocator,
) !void {
    const detection = builder.detectMHC();
    
    if (!detection.detected) {
        std.debug.print("\n📋 No mHC metadata detected\n", .{});
        return;
    }
    
    // Print detection info
    std.debug.print("\n🔍 mHC Detected:\n", .{});
    std.debug.print("   Source: {s}\n", .{@tagName(detection.source)});
    std.debug.print("   Confidence: {d}%\n", .{detection.confidence * 100});
    std.debug.print("   Keys found: {d}\n", .{detection.mhc_key_count});
    
    // Build mHC config
    const mhc_cfg = builder.buildMHCConfig();
    
    // Build transformer config
    const transformer_cfg = builder.buildTransformerConfig(mhc_cfg);
    
    // Update metadata
    metadata.mhc_enabled = true;
    metadata.mhc_config = mhc_cfg;
    metadata.mhc_transformer_config = transformer_cfg;
    
    // Copy version and description (if present)
    if (builder.version) |v| {
        const version_copy = try allocator.alloc(u8, v.len);
        @memcpy(version_copy, v);
        metadata.mhc_version = version_copy;
    }
    
    if (builder.description) |d| {
        const desc_copy = try allocator.alloc(u8, d.len);
        @memcpy(desc_copy, d);
        metadata.mhc_description = desc_copy;
    }
    
    // Print loaded config
    printMHCConfig(metadata);
}

fn printMHCConfig(metadata: *const gguf_loader.ModelMetadata) void {
    if (!metadata.mhc_enabled) return;
    
    std.debug.print("\n✅ mHC Configuration Loaded:\n", .{});
    
    if (metadata.mhc_version) |v| {
        std.debug.print("   Version: {s}\n", .{v});
    }
    
    if (metadata.mhc_description) |d| {
        std.debug.print("   Description: {s}\n", .{d});
    }
    
    if (metadata.mhc_config) |cfg| {
        std.debug.print("   Core Config:\n", .{});
        std.debug.print("      Sinkhorn iterations: {d}\n", .{cfg.sinkhorn_iterations});
        std.debug.print("      Manifold epsilon: {e}\n", .{cfg.manifold_epsilon});
        std.debug.print("      Stability threshold: {e}\n", .{cfg.stability_threshold});
        std.debug.print("      Manifold beta: {d}\n", .{cfg.manifold_beta});
        std.debug.print("      Early stopping: {}\n", .{cfg.early_stopping});
    }
    
    if (metadata.mhc_transformer_config) |tcfg| {
        std.debug.print("   Transformer Config:\n", .{});
        std.debug.print("      Attention: {}\n", .{tcfg.attention_enabled});
        std.debug.print("      FFN: {}\n", .{tcfg.ffn_enabled});
        std.debug.print("      Residual: {}\n", .{tcfg.residual_enabled});
        
        if (tcfg.layer_range) |range| {
            std.debug.print("      Layer range: {d}-{d}\n", .{range.start, range.end});
        } else {
            std.debug.print("      Layer range: all layers\n", .{});
        }
    }
}

// ============================================================================
// Testing
// ============================================================================

test "mhc metadata builder" {
    const testing = std.testing;
    
    var builder = gguf_loader.MHCMetadataBuilder.init();
    
    // Initially no detection
    const detection1 = builder.detectMHC();
    try testing.expect(!detection1.detected);
    try testing.expectEqual(gguf_loader.MHCDetectionSource.None, detection1.source);
    
    // Add some keys
    builder.recordKey("mhc.enabled");
    builder.recordKey("mhc.version");
    builder.recordKey("mhc.config.sinkhorn_iterations");
    
    // Should detect heuristically
    const detection2 = builder.detectMHC();
    try testing.expect(detection2.detected);
    try testing.expectEqual(gguf_loader.MHCDetectionSource.Heuristic, detection2.source);
    try testing.expectEqual(@as(u32, 3), detection2.mhc_key_count);
    
    // Add explicit flag
    builder.has_enabled_key = true;
    builder.enabled_value = true;
    
    // Should detect explicitly
    const detection3 = builder.detectMHC();
    try testing.expect(detection3.detected);
    try testing.expectEqual(gguf_loader.MHCDetectionSource.Explicit, detection3.source);
    try testing.expectApproxEqAbs(@as(f32, 1.0), detection3.confidence, 0.01);
}

test "mhc config building" {
    const testing = std.testing;
    
    var builder = gguf_loader.MHCMetadataBuilder.init();
    builder.sinkhorn_iterations = 15;
    builder.manifold_epsilon = 1e-5;
    builder.attention_enabled = true;
    builder.ffn_enabled = false;
    
    const mhc_cfg = builder.buildMHCConfig();
    try testing.expectEqual(@as(u32, 15), mhc_cfg.sinkhorn_iterations);
    try testing.expectApproxEqAbs(@as(f32, 1e-5), mhc_cfg.manifold_epsilon, 1e-10);
    
    const transformer_cfg = builder.buildTransformerConfig(mhc_cfg);
    try testing.expect(transformer_cfg.attention_enabled);
    try testing.expect(!transformer_cfg.ffn_enabled);
}

test "layer range building" {
    const testing = std.testing;
    
    var builder = gguf_loader.MHCMetadataBuilder.init();
    builder.layer_range_start = 60;
    builder.layer_range_end = 80;
    
    const mhc_cfg = builder.buildMHCConfig();
    const transformer_cfg = builder.buildTransformerConfig(mhc_cfg);
    
    try testing.expect(transformer_cfg.layer_range != null);
    try testing.expectEqual(@as(u32, 60), transformer_cfg.layer_range.?.start);
    try testing.expectEqual(@as(u32, 80), transformer_cfg.layer_range.?.end);
}
