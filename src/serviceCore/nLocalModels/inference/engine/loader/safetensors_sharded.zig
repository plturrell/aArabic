const std = @import("std");
const safetensors = @import("safetensors_loader");

/// Multi-Shard SafeTensors Loader
/// Loads sharded models like Qwen3-Coder-30B (16 shards)
/// Uses model.safetensors.index.json to map tensors to shards

// ============================================================================
// Shard Index Information
// ============================================================================

pub const ShardInfo = struct {
    shard_file: []const u8,
    shard_index: usize,
};

pub const ShardIndex = struct {
    weight_map: std.StringHashMap([]const u8), // tensor_name -> shard_file
    metadata: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ShardIndex {
        return .{
            .weight_map = std.StringHashMap([]const u8).init(allocator),
            .metadata = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *ShardIndex) void {
        var it = self.weight_map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.weight_map.deinit();
        
        var meta_it = self.metadata.iterator();
        while (meta_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }
};

// ============================================================================
// Sharded SafeTensors Model
// ============================================================================

pub const SafeTensorsSharded = struct {
    allocator: std.mem.Allocator,
    base_path: []const u8,
    index: ShardIndex,
    shards: std.StringHashMap(safetensors.SafeTensorsFile),
    shard_files: std.ArrayList([]const u8),
    
    pub fn init(allocator: std.mem.Allocator, base_path: []const u8) SafeTensorsSharded {
        return .{
            .allocator = allocator,
            .base_path = base_path,
            .index = ShardIndex.init(allocator),
            .shards = std.StringHashMap(safetensors.SafeTensorsFile).init(allocator),
            .shard_files = .empty,
        };
    }
    
    pub fn deinit(self: *SafeTensorsSharded) void {
        // Deinit all loaded shards
        var it = self.shards.iterator();
        while (it.next()) |entry| {
            var shard = entry.value_ptr;
            shard.deinit();
        }
        self.shards.deinit();
        
        // Free shard file names
        for (self.shard_files.items) |shard_file| {
            self.allocator.free(shard_file);
        }
        self.shard_files.deinit(self.allocator);
        
        self.index.deinit();
    }
    
    /// Load from index file (model.safetensors.index.json)
    pub fn loadFromIndex(self: *SafeTensorsSharded, index_path: []const u8) !void {
        std.debug.print("\n📚 Loading sharded model from index: {s}\n", .{index_path});
        
        // Read index file
        const file = try std.fs.cwd().openFile(index_path, .{});
        defer file.close();
        
        const file_size = (try file.stat()).size;
        const index_json = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(index_json);
        
        _ = try file.read(index_json);
        
        // Parse JSON
        try self.parseIndex(index_json);
        
        std.debug.print("   Total tensors: {d}\n", .{self.index.weight_map.count()});
        std.debug.print("   Unique shards: {d}\n", .{self.shard_files.items.len});
        
        // Load all shard files
        try self.loadAllShards();
        
        std.debug.print("✅ Sharded model loaded successfully\n", .{});
    }
    
    /// Parse index JSON file
    fn parseIndex(self: *SafeTensorsSharded, json_data: []const u8) !void {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json_data,
            .{},
        );
        defer parsed.deinit();
        
        const root = parsed.value.object;
        
        // Parse weight_map
        if (root.get("weight_map")) |weight_map_val| {
            if (weight_map_val == .object) {
                var it = weight_map_val.object.iterator();
                while (it.next()) |entry| {
                    const tensor_name = entry.key_ptr.*;
                    const shard_file = entry.value_ptr.*;
                    
                    if (shard_file == .string) {
                        // Store mapping
                        try self.index.weight_map.put(
                            try self.allocator.dupe(u8, tensor_name),
                            try self.allocator.dupe(u8, shard_file.string),
                        );
                        
                        // Track unique shard files
                        var found = false;
                        for (self.shard_files.items) |existing| {
                            if (std.mem.eql(u8, existing, shard_file.string)) {
                                found = true;
                                break;
                            }
                        }
                        
                        if (!found) {
                            try self.shard_files.append(self.allocator, try self.allocator.dupe(u8, shard_file.string));
                        }
                    }
                }
            }
        }
        
        // Parse metadata (optional)
        if (root.get("metadata")) |metadata_val| {
            if (metadata_val == .object) {
                var it = metadata_val.object.iterator();
                while (it.next()) |entry| {
                    const key = entry.key_ptr.*;
                    
                    // Store metadata placeholder
                    const metadata_value = try std.fmt.allocPrint(self.allocator, "{{...}}", .{});
                    try self.index.metadata.put(
                        try self.allocator.dupe(u8, key),
                        metadata_value,
                    );
                }
            }
        }
    }
    
    /// Load all shard files
    fn loadAllShards(self: *SafeTensorsSharded) !void {
        std.debug.print("\n🔄 Loading {d} shard files...\n", .{self.shard_files.items.len});
        
        for (self.shard_files.items, 0..) |shard_file, i| {
            // Construct full path
            const full_path = try std.fs.path.join(
                self.allocator,
                &[_][]const u8{ self.base_path, shard_file },
            );
            defer self.allocator.free(full_path);
            
            std.debug.print("   [{d}/{d}] Loading {s}...\n", .{ i + 1, self.shard_files.items.len, shard_file });
            
            // Load shard
            var shard = safetensors.SafeTensorsFile.init(self.allocator, full_path);
            try shard.load();
            
            // Store shard
            try self.shards.put(
                try self.allocator.dupe(u8, shard_file),
                shard,
            );
        }
        
        std.debug.print("✅ All shards loaded\n", .{});
    }
    
    /// Get tensor from appropriate shard
    pub fn getTensor(self: *SafeTensorsSharded, tensor_name: []const u8) ![]f32 {
        // Find which shard contains this tensor
        const shard_file = self.index.weight_map.get(tensor_name) orelse return error.TensorNotFound;
        
        // Get the shard
        var shard = self.shards.getPtr(shard_file) orelse return error.ShardNotLoaded;
        
        // Load tensor from shard
        return try shard.getTensor(tensor_name);
    }
    
    /// Check if tensor exists
    pub fn hasTensor(self: *SafeTensorsSharded, tensor_name: []const u8) bool {
        return self.index.weight_map.contains(tensor_name);
    }
    
    /// Get tensor info without loading data
    pub fn getTensorInfo(self: *SafeTensorsSharded, tensor_name: []const u8) !safetensors.TensorInfo {
        const shard_file = self.index.weight_map.get(tensor_name) orelse return error.TensorNotFound;
        var shard = self.shards.getPtr(shard_file) orelse return error.ShardNotLoaded;
        
        const tensor_info = shard.header.tensors.get(tensor_name) orelse return error.TensorNotFound;
        return tensor_info;
    }
    
    /// List all tensors
    pub fn listTensors(self: *SafeTensorsSharded) void {
        std.debug.print("\n📊 Tensors in sharded model:\n", .{});
        std.debug.print("   Total: {d} tensors across {d} shards\n\n", .{ 
            self.index.weight_map.count(), 
            self.shard_files.items.len 
        });
        
        // Group by shard
        for (self.shard_files.items, 0..) |shard_file, shard_idx| {
            std.debug.print("   Shard {d}: {s}\n", .{ shard_idx + 1, shard_file });
            
            var count: usize = 0;
            var it = self.index.weight_map.iterator();
            while (it.next()) |entry| {
                if (std.mem.eql(u8, entry.value_ptr.*, shard_file)) {
                    count += 1;
                }
            }
            
            std.debug.print("     Tensors: {d}\n\n", .{count});
        }
    }
    
    /// List tensors with details
    pub fn listTensorsDetailed(self: *SafeTensorsSharded) void {
        std.debug.print("\n📊 Detailed tensor list:\n\n", .{});
        
        var it = self.index.weight_map.iterator();
        var count: usize = 0;
        
        while (it.next()) |entry| {
            const tensor_name = entry.key_ptr.*;
            const shard_file = entry.value_ptr.*;
            
            std.debug.print("   {d}. {s}\n", .{ count + 1, tensor_name });
            std.debug.print("      Shard: {s}\n", .{shard_file});
            
            // Get tensor info if available
            if (self.getTensorInfo(tensor_name)) |info| {
                std.debug.print("      Shape: [", .{});
                for (info.shape, 0..) |dim, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{d}", .{dim});
                }
                std.debug.print("]\n", .{});
                std.debug.print("      Dtype: {s}\n", .{@tagName(info.dtype)});
                std.debug.print("      Elements: {d}\n\n", .{info.elementCount()});
            } else |_| {
                std.debug.print("      (Info unavailable)\n\n", .{});
            }
            
            count += 1;
            
            // Limit output for large models
            if (count >= 20) {
                std.debug.print("   ... and {d} more tensors\n", .{self.index.weight_map.count() - 20});
                break;
            }
        }
    }
    
    /// Get metadata
    pub fn getMetadata(self: *SafeTensorsSharded, key: []const u8) ?[]const u8 {
        return self.index.metadata.get(key);
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_sharded_loader(allocator: std.mem.Allocator, base_path: []const u8, index_file: []const u8) !void {
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  SHARDED SAFETENSORS LOADER TEST\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    var loader = SafeTensorsSharded.init(allocator, base_path);
    defer loader.deinit();
    
    // Construct index path
    const index_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ base_path, index_file },
    );
    defer allocator.free(index_path);
    
    try loader.loadFromIndex(index_path);
    
    loader.listTensors();
    
    // Test loading a specific tensor
    std.debug.print("\n🧪 Testing tensor loading from shards...\n", .{});
    
    var it = loader.index.weight_map.iterator();
    if (it.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        
        std.debug.print("   Loading: {s}\n", .{tensor_name});
        
        const tensor_data = try loader.getTensor(tensor_name);
        defer allocator.free(tensor_data);
        
        std.debug.print("   Loaded {d} elements\n", .{tensor_data.len});
        std.debug.print("   First value: {d:.6}\n", .{tensor_data[0]});
        std.debug.print("   ✅ Tensor loaded successfully from shard!\n", .{});
    }
    
    std.debug.print("\n✅ Sharded loader test complete!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
