const std = @import("std");

/// SafeTensors Format Loader
/// Implements HuggingFace SafeTensors format for safe, efficient tensor storage
/// Format: [8-byte header size][JSON header][tensor data]

// ============================================================================
// Data Types
// ============================================================================

pub const DataType = enum {
    F32,
    F16,
    BF16,
    U8,
    I8,
    I16,
    I32,
    I64,
    U16,
    U32,
    U64,
    
    pub fn fromString(s: []const u8) !DataType {
        if (std.mem.eql(u8, s, "F32")) return .F32;
        if (std.mem.eql(u8, s, "F16")) return .F16;
        if (std.mem.eql(u8, s, "BF16")) return .BF16;
        if (std.mem.eql(u8, s, "U8")) return .U8;
        if (std.mem.eql(u8, s, "I8")) return .I8;
        if (std.mem.eql(u8, s, "I16")) return .I16;
        if (std.mem.eql(u8, s, "I32")) return .I32;
        if (std.mem.eql(u8, s, "I64")) return .I64;
        if (std.mem.eql(u8, s, "U16")) return .U16;
        if (std.mem.eql(u8, s, "U32")) return .U32;
        if (std.mem.eql(u8, s, "U64")) return .U64;
        return error.UnknownDataType;
    }
    
    pub fn sizeInBytes(self: DataType) usize {
        return switch (self) {
            .F32, .I32, .U32 => 4,
            .F16, .BF16, .I16, .U16 => 2,
            .U8, .I8 => 1,
            .I64, .U64 => 8,
        };
    }
};

// ============================================================================
// Tensor Information
// ============================================================================

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DataType,
    shape: []usize,
    data_offsets: [2]u64, // [start_byte, end_byte]
    
    pub fn deinit(self: *TensorInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.shape);
    }
    
    pub fn elementCount(self: TensorInfo) usize {
        var count: usize = 1;
        for (self.shape) |dim| {
            count *= dim;
        }
        return count;
    }
    
    pub fn sizeInBytes(self: TensorInfo) usize {
        return self.elementCount() * self.dtype.sizeInBytes();
    }
};

// ============================================================================
// SafeTensors File Header
// ============================================================================

pub const SafeTensorsHeader = struct {
    tensors: std.StringHashMap(TensorInfo),
    metadata: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) SafeTensorsHeader {
        return .{
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .metadata = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SafeTensorsHeader) void {
        var tensor_it = self.tensors.iterator();
        while (tensor_it.next()) |entry| {
            var tensor = entry.value_ptr;
            tensor.deinit(self.allocator);
        }
        self.tensors.deinit();
        
        var meta_it = self.metadata.iterator();
        while (meta_it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }
};

// ============================================================================
// SafeTensors File Loader
// ============================================================================

pub const SafeTensorsFile = struct {
    allocator: std.mem.Allocator,
    file_path: []const u8,
    header: SafeTensorsHeader,
    data_section_offset: u64,
    file_size: u64,
    
    pub fn init(allocator: std.mem.Allocator, file_path: []const u8) SafeTensorsFile {
        return .{
            .allocator = allocator,
            .file_path = file_path,
            .header = SafeTensorsHeader.init(allocator),
            .data_section_offset = 0,
            .file_size = 0,
        };
    }
    
    pub fn deinit(self: *SafeTensorsFile) void {
        self.header.deinit();
    }
    
    /// Load and parse the SafeTensors file
    pub fn load(self: *SafeTensorsFile) !void {
        std.debug.print("\n📦 Loading SafeTensors file: {s}\n", .{self.file_path});
        
        // Open file
        const file = try std.fs.cwd().openFile(self.file_path, .{});
        defer file.close();
        
        // Get file size
        const stat = try file.stat();
        self.file_size = stat.size;
        std.debug.print("   File size: {d} bytes ({d:.2} MB)\n", .{ self.file_size, @as(f64, @floatFromInt(self.file_size)) / 1024.0 / 1024.0 });
        
        // Read 8-byte header size (little-endian u64)
        var header_size_bytes: [8]u8 = undefined;
        _ = try file.read(&header_size_bytes);
        const header_size = std.mem.readInt(u64, &header_size_bytes, .little);
        
        std.debug.print("   Header size: {d} bytes\n", .{header_size});
        
        if (header_size > 100 * 1024 * 1024) { // Sanity check: header shouldn't be > 100MB
            return error.HeaderTooLarge;
        }
        
        // Read JSON header
        const header_json = try self.allocator.alloc(u8, header_size);
        defer self.allocator.free(header_json);
        
        _ = try file.read(header_json);
        
        // Parse JSON header
        try self.parseHeader(header_json);
        
        // Calculate data section offset
        self.data_section_offset = 8 + header_size;
        
        std.debug.print("   Data section offset: {d} bytes\n", .{self.data_section_offset});
        std.debug.print("   Tensors loaded: {d}\n", .{self.header.tensors.count()});
        std.debug.print("✅ SafeTensors file loaded successfully\n", .{});
    }
    
    /// Parse JSON header and populate tensors map
    fn parseHeader(self: *SafeTensorsFile, json_data: []const u8) !void {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json_data,
            .{},
        );
        defer parsed.deinit();
        
        const root = parsed.value.object;
        
        // Iterate over all keys in JSON
        var it = root.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            const value = entry.value_ptr.*;
            
            // Handle metadata (keys starting with "__")
            if (std.mem.startsWith(u8, key, "__")) {
                // Store metadata key - value as simple string
                const metadata_value = try std.fmt.allocPrint(self.allocator, "{{...}}", .{});
                try self.header.metadata.put(
                    try self.allocator.dupe(u8, key),
                    metadata_value,
                );
                continue;
            }
            
            // Parse tensor info
            if (value == .object) {
                const tensor_obj = value.object;
                
                // Get dtype
                const dtype_str = tensor_obj.get("dtype") orelse continue;
                if (dtype_str != .string) continue;
                const dtype = try DataType.fromString(dtype_str.string);
                
                // Get shape
                const shape_array = tensor_obj.get("shape") orelse continue;
                if (shape_array != .array) continue;
                
                var shape = try self.allocator.alloc(usize, shape_array.array.items.len);
                for (shape_array.array.items, 0..) |dim_val, i| {
                    if (dim_val != .integer) {
                        self.allocator.free(shape);
                        continue;
                    }
                    shape[i] = @intCast(dim_val.integer);
                }
                
                // Get data offsets
                const offsets_array = tensor_obj.get("data_offsets") orelse {
                    self.allocator.free(shape);
                    continue;
                };
                if (offsets_array != .array or offsets_array.array.items.len != 2) {
                    self.allocator.free(shape);
                    continue;
                }
                
                const start_offset: u64 = @intCast(offsets_array.array.items[0].integer);
                const end_offset: u64 = @intCast(offsets_array.array.items[1].integer);
                
                // Create tensor info
                const tensor_info = TensorInfo{
                    .name = try self.allocator.dupe(u8, key),
                    .dtype = dtype,
                    .shape = shape,
                    .data_offsets = .{ start_offset, end_offset },
                };
                
                try self.header.tensors.put(
                    try self.allocator.dupe(u8, key),
                    tensor_info,
                );
            }
        }
    }
    
    /// Get tensor data as f32 array
    pub fn getTensor(self: *SafeTensorsFile, name: []const u8) ![]f32 {
        const tensor_info = self.header.tensors.get(name) orelse return error.TensorNotFound;
        
        // Open file for reading
        const file = try std.fs.cwd().openFile(self.file_path, .{});
        defer file.close();
        
        // Seek to tensor data
        const absolute_offset = self.data_section_offset + tensor_info.data_offsets[0];
        try file.seekTo(absolute_offset);
        
        // Allocate output buffer
        const element_count = tensor_info.elementCount();
        const output = try self.allocator.alloc(f32, element_count);
        errdefer self.allocator.free(output);
        
        // Read and convert based on dtype
        switch (tensor_info.dtype) {
            .F32 => {
                // Direct read for F32
                const bytes = std.mem.sliceAsBytes(output);
                _ = try file.read(bytes);
            },
            .F16 => {
                // Convert F16 to F32
                const f16_data = try self.allocator.alloc(u16, element_count);
                defer self.allocator.free(f16_data);
                
                const bytes = std.mem.sliceAsBytes(f16_data);
                _ = try file.read(bytes);
                
                for (f16_data, 0..) |f16_val, i| {
                    output[i] = f16ToF32(f16_val);
                }
            },
            .BF16 => {
                // Convert BF16 to F32
                const bf16_data = try self.allocator.alloc(u16, element_count);
                defer self.allocator.free(bf16_data);
                
                const bytes = std.mem.sliceAsBytes(bf16_data);
                _ = try file.read(bytes);
                
                for (bf16_data, 0..) |bf16_val, i| {
                    output[i] = bf16ToF32(bf16_val);
                }
            },
            else => return error.UnsupportedDataType,
        }
        
        return output;
    }
    
    /// List all tensors in the file
    pub fn listTensors(self: *SafeTensorsFile) void {
        std.debug.print("\n📊 Tensors in SafeTensors file:\n", .{});
        std.debug.print("   Total: {d} tensors\n\n", .{self.header.tensors.count()});
        
        var it = self.header.tensors.iterator();
        while (it.next()) |entry| {
            const tensor = entry.value_ptr.*;
            
            std.debug.print("   • {s}\n", .{tensor.name});
            std.debug.print("     Shape: [", .{});
            for (tensor.shape, 0..) |dim, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print("]\n", .{});
            std.debug.print("     Dtype: {s}\n", .{@tagName(tensor.dtype)});
            std.debug.print("     Size: {d} elements ({d} bytes)\n", .{ tensor.elementCount(), tensor.sizeInBytes() });
            std.debug.print("     Offsets: [{d}, {d}]\n\n", .{ tensor.data_offsets[0], tensor.data_offsets[1] });
        }
    }
    
    /// Get metadata value
    pub fn getMetadata(self: *SafeTensorsFile, key: []const u8) ?[]const u8 {
        return self.header.metadata.get(key);
    }
};

// ============================================================================
// FP16/BF16 Conversion Utilities
// ============================================================================

/// Convert FP16 to FP32
fn f16ToF32(f16_bits: u16) f32 {
    const sign: u32 = @as(u32, f16_bits >> 15) << 31;
    const exponent_f16: u32 = @as(u32, (f16_bits >> 10) & 0x1F);
    const mantissa_f16: u32 = @as(u32, f16_bits & 0x3FF);
    
    var exponent_f32: u32 = 0;
    var mantissa_f32: u32 = 0;
    
    if (exponent_f16 == 0) {
        // Subnormal or zero
        if (mantissa_f16 == 0) {
            // Zero
            exponent_f32 = 0;
            mantissa_f32 = 0;
        } else {
            // Subnormal - convert to normalized F32
            exponent_f32 = 127 - 14;
            mantissa_f32 = mantissa_f16 << 13;
            
            // Normalize
            while ((mantissa_f32 & 0x00800000) == 0) {
                mantissa_f32 <<= 1;
                exponent_f32 -= 1;
            }
            mantissa_f32 &= 0x007FFFFF;
        }
    } else if (exponent_f16 == 0x1F) {
        // Infinity or NaN
        exponent_f32 = 0xFF;
        mantissa_f32 = mantissa_f16 << 13;
    } else {
        // Normal number
        exponent_f32 = exponent_f16 + (127 - 15);
        mantissa_f32 = mantissa_f16 << 13;
    }
    
    const f32_bits: u32 = sign | (exponent_f32 << 23) | mantissa_f32;
    return @bitCast(f32_bits);
}

/// Convert BF16 to FP32
fn bf16ToF32(bf16_bits: u16) f32 {
    // BF16 is just the upper 16 bits of FP32
    const f32_bits: u32 = @as(u32, bf16_bits) << 16;
    return @bitCast(f32_bits);
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_safetensors_loader(allocator: std.mem.Allocator, file_path: []const u8) !void {
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  SAFETENSORS LOADER TEST\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    var loader = SafeTensorsFile.init(allocator, file_path);
    defer loader.deinit();
    
    try loader.load();
    loader.listTensors();
    
    std.debug.print("✅ SafeTensors loader test complete!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
