// SSD-Tiered Storage Engine
// Breaks memory constraints by using NVMe SSD as memory tier
// Inspired by DragonflyDB's tiering architecture
//
// Key innovations:
// - Memory-mapped files for zero-copy access
// - Async I/O with io_uring (when available)
// - LRU eviction from RAM to SSD
// - Block-aligned storage for NVMe efficiency
// - Lock-free hot path with atomic reference counting

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Configuration
// ============================================================================

pub const TierConfig = struct {
    // Storage paths
    ssd_path: []const u8 = "/tmp/shimmy_tier",
    
    // Memory limits
    max_ram_mb: u64 = 2048,       // 2GB default RAM budget
    max_ssd_mb: u64 = 32768,      // 32GB default SSD budget
    
    // Testing mode (Day 5: for benchmarks with minimal disk allocation)
    test_mode: bool = false,      // When true, use only 1MB SSD allocation
    
    // Block sizes (aligned to NVMe page size)
    block_size: u32 = 4096,       // 4KB blocks
    large_block_size: u32 = 2 * 1024 * 1024, // 2MB for large allocations
    optimal_block_size: u32 = 65536, // 64KB optimal for NVMe (Day 2 optimization)
    
    // Eviction policy
    eviction_threshold: f32 = 0.85, // Start evicting at 85% RAM usage
    eviction_target: f32 = 0.70,    // Evict down to 70%
    
    // I/O settings (Day 2 optimization)
    max_pending_io: u32 = 64,      // Max concurrent I/O operations
    read_ahead_blocks: u32 = 8,    // Prefetch nearby blocks (increased from 4)
    prefetch_threshold: u32 = 3,   // Sequential reads before prefetch
    merge_distance: u32 = 131072,  // Merge reads within 128KB
    
    // Feature flags
    use_mmap: bool = true,         // Use memory-mapped files
    use_direct_io: bool = true,    // O_DIRECT for bypass page cache
    sync_writes: bool = false,     // O_SYNC for durability (slower)
    enable_prefetch: bool = true,  // Enable read-ahead prefetching
    enable_io_scheduling: bool = true, // Enable I/O request merging
};

// ============================================================================
// Block Descriptor
// ============================================================================

pub const BlockLocation = enum(u8) {
    RAM,           // In memory
    SSD,           // On disk
    SSD_MMAP,      // Memory-mapped from disk
    Evicting,      // Being written to disk
    Loading,       // Being read from disk
};

pub const BlockDescriptor = struct {
    offset: u64,           // Offset in tier (RAM addr or SSD offset)
    size: u32,             // Size in bytes
    location: BlockLocation,
    ref_count: std.atomic.Value(u32),
    last_access: i64,      // Timestamp for LRU
    checksum: u32,         // CRC32 for integrity
    
    pub fn init(offset: u64, size: u32, location: BlockLocation) BlockDescriptor {
        return .{
            .offset = offset,
            .size = size,
            .location = location,
            .ref_count = std.atomic.Value(u32).init(0),
            .last_access = std.time.milliTimestamp(),
            .checksum = 0,
        };
    }
    
    pub fn acquire(self: *BlockDescriptor) void {
        _ = self.ref_count.fetchAdd(1, .acquire);
        self.last_access = std.time.milliTimestamp();
    }
    
    pub fn release(self: *BlockDescriptor) bool {
        return self.ref_count.fetchSub(1, .release) == 1;
    }
    
    pub fn isHot(self: *const BlockDescriptor, threshold_ms: i64) bool {
        const now = std.time.milliTimestamp();
        return (now - self.last_access) < threshold_ms;
    }
};

// ============================================================================
// SSD Storage Backend
// ============================================================================

pub const SSDStorage = struct {
    allocator: std.mem.Allocator,
    config: TierConfig,
    
    // File handle
    file: ?std.fs.File,
    file_size: u64,
    
    // Free space tracking (simple bitmap for now)
    block_bitmap: []u8,
    num_blocks: u64,
    free_blocks: std.atomic.Value(u64),
    
    // Memory mapping
    mmap_base: ?[*]align(4096) u8,  // 4KB page alignment
    mmap_len: usize,
    
    // Statistics
    stats: Stats,
    
    // Day 2: Read-ahead prefetch tracking
    last_read_offset: std.atomic.Value(u64),
    sequential_reads: std.atomic.Value(u32),
    prefetch_cache: std.ArrayList(PrefetchEntry),
    
    pub const PrefetchEntry = struct {
        offset: u64,
        length: u32,
        timestamp: i64,
    };
    
    pub const Stats = struct {
        reads: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        writes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        bytes_read: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        bytes_written: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        evictions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        loads: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        // Day 2: New metrics
        prefetch_hits: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        prefetch_issued: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        io_merges: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    };
    
    pub fn init(allocator: std.mem.Allocator, config: TierConfig) !*SSDStorage {
        const self = try allocator.create(SSDStorage);
        errdefer allocator.destroy(self);
        
        const num_blocks = (config.max_ssd_mb * 1024 * 1024) / config.block_size;
        const bitmap_size = (num_blocks + 7) / 8;
        
        const block_bitmap = try allocator.alloc(u8, bitmap_size);
        @memset(block_bitmap, 0);
        
        self.* = SSDStorage{
            .allocator = allocator,
            .config = config,
            .file = null,
            .file_size = 0,
            .block_bitmap = block_bitmap,
            .num_blocks = num_blocks,
            .free_blocks = std.atomic.Value(u64).init(num_blocks),
            .mmap_base = null,
            .mmap_len = 0,
            .stats = .{},
            // Day 2: Initialize prefetch tracking
            .last_read_offset = std.atomic.Value(u64).init(0),
            .sequential_reads = std.atomic.Value(u32).init(0),
            .prefetch_cache = std.ArrayList(PrefetchEntry){},
        };
        
        return self;
    }
    
    pub fn open(self: *SSDStorage) !void {
        // Create directory if needed
        const dir_path = std.fs.path.dirname(self.config.ssd_path) orelse ".";
        std.fs.cwd().makePath(dir_path) catch {};
        
        // Open/create backing file
        const flags: std.fs.File.CreateFlags = .{
            .read = true,
            .truncate = false,
            .exclusive = false,
        };
        
        self.file = std.fs.cwd().createFile(self.config.ssd_path, flags) catch |err| {
            if (err == error.PathAlreadyExists) {
                return error.FileExists;
            }
            return err;
        };
        
        // Pre-allocate file to max size for performance
        // Day 5: Use only 1MB in test mode to avoid OutOfSpace in benchmarks
        const max_size = if (self.config.test_mode)
            1 * 1024 * 1024  // 1MB for testing
        else
            self.config.max_ssd_mb * 1024 * 1024;  // Full allocation for production
        try self.file.?.setEndPos(max_size);
        self.file_size = max_size;
        
        // Memory map if enabled
        if (self.config.use_mmap) {
            try self.setupMmap();
        }
    }
    
    fn setupMmap(self: *SSDStorage) !void {
        if (self.file == null) return error.FileNotOpen;

        const len = self.file_size;
        const ptr = std.posix.mmap(
            null,
            len,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .TYPE = .SHARED },
            self.file.?.handle,
            0,
        ) catch return error.MmapFailed;

        self.mmap_base = @ptrCast(@alignCast(ptr));
        self.mmap_len = len;
    }

    /// Allocate a block on SSD, returns offset
    pub fn allocBlock(self: *SSDStorage, size: u32) !u64 {
        const blocks_needed = (size + self.config.block_size - 1) / self.config.block_size;

        // Simple first-fit allocation
        var consecutive: u64 = 0;
        var start_block: u64 = 0;

        for (0..self.num_blocks) |block| {
            const byte_idx = block / 8;
            const bit_idx: u3 = @intCast(block % 8);
            const is_used = (self.block_bitmap[byte_idx] >> bit_idx) & 1 == 1;

            if (is_used) {
                consecutive = 0;
                start_block = block + 1;
            } else {
                consecutive += 1;
                if (consecutive >= blocks_needed) {
                    // Found space, mark as used
                    for (start_block..start_block + blocks_needed) |b| {
                        const bi = b / 8;
                        const bb: u3 = @intCast(b % 8);
                        self.block_bitmap[bi] |= @as(u8, 1) << bb;
                    }
                    _ = self.free_blocks.fetchSub(blocks_needed, .monotonic);
                    return start_block * self.config.block_size;
                }
            }
        }

        return error.OutOfSpace;
    }

    /// Free a block on SSD
    pub fn freeBlock(self: *SSDStorage, offset: u64, size: u32) void {
        const start_block = offset / self.config.block_size;
        const blocks = (size + self.config.block_size - 1) / self.config.block_size;

        for (start_block..start_block + blocks) |b| {
            const bi = b / 8;
            const bb: u3 = @intCast(b % 8);
            self.block_bitmap[bi] &= ~(@as(u8, 1) << bb);
        }
        _ = self.free_blocks.fetchAdd(blocks, .monotonic);
    }

    /// Write data to SSD (zero-copy if mmap)
    pub fn write(self: *SSDStorage, offset: u64, data: []const u8) !void {
        if (self.mmap_base) |base| {
            // Zero-copy via mmap
            const dest = base[offset..offset + data.len];
            @memcpy(dest, data);
            _ = self.stats.writes.fetchAdd(1, .monotonic);
            _ = self.stats.bytes_written.fetchAdd(data.len, .monotonic);
        } else if (self.file) |file| {
            // Fall back to pwrite
            try file.seekTo(offset);
            _ = try file.write(data);
            _ = self.stats.writes.fetchAdd(1, .monotonic);
            _ = self.stats.bytes_written.fetchAdd(data.len, .monotonic);
        } else {
            return error.FileNotOpen;
        }
    }

    /// Read data from SSD (zero-copy slice if mmap) - Day 2 optimized
    pub fn read(self: *SSDStorage, offset: u64, len: usize) ![]const u8 {
        if (self.mmap_base) |base| {
            // Day 2: Track sequential access pattern
            if (self.config.enable_prefetch) {
                try self.trackAndPrefetch(offset, len);
            }
            
            _ = self.stats.reads.fetchAdd(1, .monotonic);
            _ = self.stats.bytes_read.fetchAdd(len, .monotonic);
            return base[offset..offset + len];
        } else {
            return error.MmapRequired;
        }
    }

    /// Read into buffer (for non-mmap path)
    pub fn readInto(self: *SSDStorage, offset: u64, buffer: []u8) !void {
        if (self.mmap_base) |base| {
            @memcpy(buffer, base[offset..offset + buffer.len]);
        } else if (self.file) |file| {
            try file.seekTo(offset);
            _ = try file.read(buffer);
        } else {
            return error.FileNotOpen;
        }
        _ = self.stats.reads.fetchAdd(1, .monotonic);
        _ = self.stats.bytes_read.fetchAdd(buffer.len, .monotonic);
    }

    /// Sync to disk (for durability)
    pub fn sync(self: *SSDStorage) !void {
        if (self.mmap_base) |base| {
            std.posix.msync(@alignCast(base[0..self.mmap_len]), .{ .SYNC = true }) catch {};
        }
        if (self.file) |file| {
            try file.sync();
        }
    }

    pub fn close(self: *SSDStorage) void {
        if (self.mmap_base) |base| {
            std.posix.munmap(@alignCast(base[0..self.mmap_len]));
            self.mmap_base = null;
        }
        if (self.file) |file| {
            file.close();
            self.file = null;
        }
    }

    pub fn deinit(self: *SSDStorage) void {
        self.close();
        self.allocator.free(self.block_bitmap);
        self.prefetch_cache.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn getStats(self: *SSDStorage) Stats {
        return self.stats;
    }

    pub fn getUsage(self: *SSDStorage) struct { used_mb: u64, total_mb: u64, pct: f32 } {
        const used = self.num_blocks - self.free_blocks.load(.monotonic);
        const used_bytes = used * self.config.block_size;
        const total_bytes = self.num_blocks * self.config.block_size;
        return .{
            .used_mb = used_bytes / (1024 * 1024),
            .total_mb = total_bytes / (1024 * 1024),
            .pct = @as(f32, @floatFromInt(used)) / @as(f32, @floatFromInt(self.num_blocks)) * 100.0,
        };
    }
    
    // ========================================================================
    // Day 2 Optimizations: Read-ahead prefetching and I/O scheduling
    // ========================================================================
    
    /// Track sequential access patterns and issue prefetch if detected
    fn trackAndPrefetch(self: *SSDStorage, offset: u64, len: usize) !void {
        const last_offset = self.last_read_offset.load(.monotonic);
        const expected_next = last_offset + len;
        
        // Detect sequential access (within optimal block size tolerance)
        const is_sequential = offset >= last_offset and 
                             offset <= expected_next + self.config.optimal_block_size;
        
        if (is_sequential) {
            const seq_count = self.sequential_reads.fetchAdd(1, .monotonic) + 1;
            
            // Issue prefetch after threshold sequential reads
            if (seq_count >= self.config.prefetch_threshold) {
                try self.issuePrefetch(offset + len);
            }
        } else {
            // Reset counter on non-sequential access
            self.sequential_reads.store(0, .monotonic);
        }
        
        self.last_read_offset.store(offset, .monotonic);
    }
    
    /// Issue read-ahead prefetch hint to the kernel
    fn issuePrefetch(self: *SSDStorage, start_offset: u64) !void {
        if (self.mmap_base == null or self.file == null) return;
        
        const prefetch_len = self.config.read_ahead_blocks * self.config.optimal_block_size;
        const end_offset = @min(start_offset + prefetch_len, self.file_size);
        
        if (end_offset <= start_offset) return;
        
        // Check if already prefetched recently
        const now = std.time.milliTimestamp();
        for (self.prefetch_cache.items) |entry| {
            if (entry.offset == start_offset and (now - entry.timestamp) < 1000) {
                _ = self.stats.prefetch_hits.fetchAdd(1, .monotonic);
                return; // Already prefetched
            }
        }
        
        // Use madvise to hint kernel about sequential access
        if (self.mmap_base) |base| {
            const prefetch_slice = base[start_offset..end_offset];
            std.posix.madvise(
                @ptrCast(@alignCast(prefetch_slice.ptr)),
                prefetch_slice.len,
                .WILLNEED
            ) catch {};
            
            // Also hint sequential access pattern
            std.posix.madvise(
                @ptrCast(@alignCast(prefetch_slice.ptr)),
                prefetch_slice.len,
                .SEQUENTIAL
            ) catch {};
        }
        
        // Track prefetch in cache
        try self.prefetch_cache.append(.{
            .offset = start_offset,
            .length = @intCast(end_offset - start_offset),
            .timestamp = now,
        });
        
        // Limit cache size
        if (self.prefetch_cache.items.len > 100) {
            _ = self.prefetch_cache.orderedRemove(0);
        }
        
        _ = self.stats.prefetch_issued.fetchAdd(1, .monotonic);
    }
    
    /// Optimize read size to use optimal block size (64KB)
    pub fn readOptimized(self: *SSDStorage, offset: u64, requested_len: usize) ![]const u8 {
        // Round up to optimal block size for better NVMe performance
        const aligned_len = if (requested_len < self.config.optimal_block_size)
            self.config.optimal_block_size
        else
            ((requested_len + self.config.optimal_block_size - 1) / 
             self.config.optimal_block_size) * self.config.optimal_block_size;
        
        const actual_len = @min(aligned_len, self.file_size - offset);
        const full_data = try self.read(offset, actual_len);
        
        // Return only requested portion
        return full_data[0..requested_len];
    }
    
    /// Batch multiple reads and merge adjacent ones (I/O scheduling)
    pub fn readBatch(
        self: *SSDStorage,
        requests: []const struct { offset: u64, len: usize },
        results: [][]const u8
    ) !void {
        if (requests.len != results.len) return error.LengthMismatch;
        if (requests.len == 0) return;
        
        if (!self.config.enable_io_scheduling or requests.len == 1) {
            // Fast path: no scheduling needed
            for (requests, results) |req, *result| {
                result.* = try self.read(req.offset, req.len);
            }
            return;
        }
        
        // Sort requests by offset for merging
        var sorted_indices = try self.allocator.alloc(usize, requests.len);
        defer self.allocator.free(sorted_indices);
        
        for (sorted_indices, 0..) |*idx, i| {
            idx.* = i;
        }
        
        // Simple insertion sort (fine for small batches)
        for (1..sorted_indices.len) |i| {
            var j = i;
            while (j > 0 and requests[sorted_indices[j]].offset < 
                   requests[sorted_indices[j - 1]].offset) : (j -= 1) {
                const tmp = sorted_indices[j];
                sorted_indices[j] = sorted_indices[j - 1];
                sorted_indices[j - 1] = tmp;
            }
        }
        
        // Process sorted requests and merge adjacent ones
        var i: usize = 0;
        while (i < sorted_indices.len) {
            const req_idx = sorted_indices[i];
            const req = requests[req_idx];
            var merge_end_offset = req.offset + req.len;
            var merge_count: usize = 1;
            
            // Check if next requests can be merged
            while (i + merge_count < sorted_indices.len) {
                const next_idx = sorted_indices[i + merge_count];
                const next_req = requests[next_idx];
                
                // Merge if within threshold distance
                if (next_req.offset <= merge_end_offset + self.config.merge_distance) {
                    merge_end_offset = @max(merge_end_offset, next_req.offset + next_req.len);
                    merge_count += 1;
                } else {
                    break;
                }
            }
            
            if (merge_count > 1) {
                // Merged read
                const merged_data = try self.read(req.offset, merge_end_offset - req.offset);
                
                // Slice results for each original request
                for (0..merge_count) |j| {
                    const orig_idx = sorted_indices[i + j];
                    const orig_req = requests[orig_idx];
                    const slice_start = orig_req.offset - req.offset;
                    results[orig_idx] = merged_data[slice_start..slice_start + orig_req.len];
                }
                
                _ = self.stats.io_merges.fetchAdd(merge_count - 1, .monotonic);
            } else {
                // Single read
                results[req_idx] = try self.read(req.offset, req.len);
            }
            
            i += merge_count;
        }
    }
};
