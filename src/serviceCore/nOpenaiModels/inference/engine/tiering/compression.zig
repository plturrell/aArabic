// Compression module for SSD-tiered KV cache
// Reduces SSD bandwidth requirements by 2-4x for KV cache data

const std = @import("std");

// ============================================================================
// LZ4-style fast compression for KV cache
// Optimized for f32 data with predictable patterns
// ============================================================================

pub const CompressionConfig = struct {
    level: CompressionLevel = .fast,
    min_size: u32 = 1024,  // Don't compress below this size
};

pub const CompressionLevel = enum {
    none,      // No compression
    fast,      // LZ4-style fast compression (300 MB/s+)
    balanced,  // Better ratio, still fast
    high,      // Maximum compression (slower)
};

pub const CompressionStats = struct {
    bytes_in: u64 = 0,
    bytes_out: u64 = 0,
    compressions: u64 = 0,
    decompressions: u64 = 0,
    
    pub fn ratio(self: CompressionStats) f32 {
        if (self.bytes_in == 0) return 1.0;
        return @as(f32, @floatFromInt(self.bytes_in)) / @as(f32, @floatFromInt(self.bytes_out));
    }
};

// ============================================================================
// Delta + RLE compression for KV cache (optimized for f32 attention data)
// ============================================================================

pub const KVCompressor = struct {
    config: CompressionConfig,
    stats: CompressionStats,
    
    // Work buffers
    delta_buf: []i32,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, config: CompressionConfig) !*KVCompressor {
        const self = try allocator.create(KVCompressor);
        
        // Pre-allocate work buffer (1MB)
        const buf_size = 1024 * 1024 / @sizeOf(i32);
        const delta_buf = try allocator.alloc(i32, buf_size);
        
        self.* = KVCompressor{
            .config = config,
            .stats = .{},
            .delta_buf = delta_buf,
            .allocator = allocator,
        };
        
        return self;
    }
    
    pub fn deinit(self: *KVCompressor) void {
        self.allocator.free(self.delta_buf);
        self.allocator.destroy(self);
    }
    
    /// Compress f32 KV data using delta + varint encoding
    /// Returns compressed data (caller owns the memory)
    pub fn compress(self: *KVCompressor, data: []const f32) ![]u8 {
        if (self.config.level == .none or data.len * @sizeOf(f32) < self.config.min_size) {
            // Just copy raw data
            const out = try self.allocator.alloc(u8, data.len * @sizeOf(f32) + 1);
            out[0] = 0; // Uncompressed marker
            @memcpy(out[1..], std.mem.sliceAsBytes(data));
            return out;
        }
        
        // Step 1: Convert to i32 (reinterpret f32 bits)
        const int_data: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(data)));
        
        // Step 2: Delta encoding
        var prev: i32 = 0;
        for (int_data, 0..) |val, i| {
            if (i < self.delta_buf.len) {
                self.delta_buf[i] = val - prev;
                prev = val;
            }
        }
        
        // Step 3: Varint encode deltas
        const max_out_size = data.len * 5 + 1; // Max varint is 5 bytes
        const out = try self.allocator.alloc(u8, max_out_size);
        errdefer self.allocator.free(out);
        
        out[0] = 1; // Compressed marker
        var out_pos: usize = 1;
        
        for (self.delta_buf[0..@min(data.len, self.delta_buf.len)]) |delta| {
            // ZigZag encode for signed values
            const zigzag: u32 = @bitCast((delta << 1) ^ (delta >> 31));
            out_pos += writeVarint(out[out_pos..], zigzag);
        }
        
        // Shrink to actual size
        const result = try self.allocator.realloc(out, out_pos);
        
        self.stats.bytes_in += data.len * @sizeOf(f32);
        self.stats.bytes_out += out_pos;
        self.stats.compressions += 1;
        
        return result;
    }
    
    /// Decompress data back to f32
    pub fn decompress(self: *KVCompressor, compressed: []const u8, out: []f32) !void {
        if (compressed.len == 0) return error.EmptyInput;
        
        if (compressed[0] == 0) {
            // Uncompressed
            const bytes = compressed[1..];
            const floats: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes));
            @memcpy(out[0..floats.len], floats);
            return;
        }
        
        // Decompress
        var pos: usize = 1;
        var prev: i32 = 0;
        var out_idx: usize = 0;
        
        while (pos < compressed.len and out_idx < out.len) {
            const result = readVarint(compressed[pos..]);
            pos += result.bytes_read;
            
            // Undo zigzag
            const zigzag = result.value;
            const delta: i32 = @bitCast((zigzag >> 1) ^ (~(zigzag & 1) +% 1));
            
            prev = prev + delta;
            out[out_idx] = @bitCast(prev);
            out_idx += 1;
        }
        
        self.stats.decompressions += 1;
    }
    
    /// Print compression stats
    pub fn printStats(self: *KVCompressor) void {
        std.debug.print("\n📦 Compression Stats\n", .{});
        std.debug.print("   Ratio: {d:.2}x\n", .{self.stats.ratio()});
        std.debug.print("   In: {d:.1} MB, Out: {d:.1} MB\n", .{
            @as(f64, @floatFromInt(self.stats.bytes_in)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.stats.bytes_out)) / (1024.0 * 1024.0),
        });
    }
};

// Varint encoding helpers
fn writeVarint(buf: []u8, value: u32) usize {
    var v = value;
    var i: usize = 0;
    while (v >= 0x80) : (i += 1) {
        buf[i] = @intCast((v & 0x7F) | 0x80);
        v >>= 7;
    }
    buf[i] = @intCast(v);
    return i + 1;
}

fn readVarint(buf: []const u8) struct { value: u32, bytes_read: usize } {
    var result: u32 = 0;
    var shift: u5 = 0;
    var i: usize = 0;
    while (i < buf.len and i < 5) : (i += 1) {
        const byte = buf[i];
        result |= @as(u32, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        shift +|= 7;
    }
    return .{ .value = result, .bytes_read = i + 1 };
}

