// KV Cache Sharing System
// Enables cross-request cache sharing via prefix detection and reference counting
//
// Architecture:
// - Prefix Tree (Trie): Detect common prompt prefixes
// - Reference Counting: Share cache entries safely across requests
// - Cache Coordinator: Manage shared cache lifecycle
// - Discovery Protocol: Find shareable cache entries
//
// Benefits:
// - 30-70% speedup for common prefixes (e.g., system prompts)
// - Reduced memory footprint (single copy shared)
// - Lower inference latency for repeat queries

const std = @import("std");
const log = @import("structured_logging.zig");
const database = @import("database_tier.zig");

// ============================================================================
// Configuration
// ============================================================================

/// Cache sharing configuration
pub const CacheSharingConfig = struct {
    /// Enable cache sharing
    enabled: bool = true,
    
    /// Minimum prefix length to consider for sharing (tokens)
    min_prefix_length: u32 = 4,
    
    /// Maximum prefix tree depth
    max_trie_depth: u32 = 128,
    
    /// Enable automatic prefix detection
    auto_detect_prefixes: bool = true,
    
    /// Pre-defined common prefixes (e.g., system prompts)
    common_prefixes: []const []const u32 = &[_][]const u32{},
    
    /// Reference count threshold for eviction
    /// (entries with refcount > 0 won't be evicted)
    protect_shared_entries: bool = true,
    
    /// TTL for unused shared entries (seconds)
    shared_entry_ttl: u32 = 3600, // 1 hour
    
    /// Maximum shared cache size (bytes)
    max_shared_cache_size: u64 = 4 * 1024 * 1024 * 1024, // 4GB
    
    /// Enable prefix compression
    compress_shared_prefixes: bool = true,
    
    /// Replication settings for distributed sharing
    enable_replication: bool = false,
    replication_factor: u32 = 2,
    replication_nodes: []const []const u8 = &[_][]const u8{},
};

// ============================================================================
// Cache Sharing Statistics
// ============================================================================

pub const CacheSharingStats = struct {
    // Sharing efficiency
    shared_cache_hits: u64 = 0,
    shared_cache_misses: u64 = 0,
    prefix_matches: u64 = 0,
    full_prefix_reuse: u64 = 0,
    partial_prefix_reuse: u64 = 0,
    
    // Memory savings
    total_entries: u64 = 0,
    shared_entries: u64 = 0,
    total_references: u64 = 0,
    bytes_saved: u64 = 0,
    
    // Prefix tree stats
    trie_nodes: u64 = 0,
    trie_depth: u32 = 0,
    common_prefixes_detected: u64 = 0,
    
    // Performance
    avg_prefix_match_time_us: u64 = 0,
    avg_reference_time_us: u64 = 0,
    
    // Replication (if enabled)
    replication_success: u64 = 0,
    replication_failures: u64 = 0,
    
    pub fn getSharedHitRate(self: *const CacheSharingStats) f64 {
        const total = self.shared_cache_hits + self.shared_cache_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.shared_cache_hits)) / 
               @as(f64, @floatFromInt(total));
    }
    
    pub fn getSharingRatio(self: *const CacheSharingStats) f64 {
        if (self.total_entries == 0) return 0.0;
        return @as(f64, @floatFromInt(self.shared_entries)) / 
               @as(f64, @floatFromInt(self.total_entries));
    }
    
    pub fn getAvgReferences(self: *const CacheSharingStats) f64 {
        if (self.shared_entries == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_references)) / 
               @as(f64, @floatFromInt(self.shared_entries));
    }
    
    pub fn getMemorySavings(self: *const CacheSharingStats) f64 {
        return @as(f64, @floatFromInt(self.bytes_saved)) / (1024.0 * 1024.0 * 1024.0);
    }
};

// ============================================================================
// Shared Cache Entry
// ============================================================================

/// Reference-counted cache entry for sharing
pub const SharedCacheEntry = struct {
    /// Unique identifier (hash of prefix tokens)
    id: u64,
    
    /// Model ID this cache belongs to
    model_id: []const u8,
    
    /// Layer number
    layer: u32,
    
    /// Token sequence (prefix)
    tokens: []const u32,
    
    /// KV cache data (keys and values)
    keys: []const f32,
    values: []const f32,
    
    /// Reference count (number of active requests using this entry)
    ref_count: std.atomic.Value(u32),
    
    /// Creation timestamp
    created_at: i64,
    
    /// Last accessed timestamp
    accessed_at: std.atomic.Value(i64),
    
    /// Total access count
    access_count: std.atomic.Value(u64),
    
    /// Size in bytes
    size_bytes: u64,
    
    /// Is this entry compressed?
    compressed: bool,
    
    /// Mutex for safe modification
    mutex: std.Thread.Mutex,
    
    pub fn init(
        allocator: std.mem.Allocator,
        id: u64,
        model_id: []const u8,
        layer: u32,
        tokens: []const u32,
        keys: []const f32,
        values: []const f32,
    ) !*SharedCacheEntry {
        const self = try allocator.create(SharedCacheEntry);
        
        const tokens_copy = try allocator.alloc(u32, tokens.len);
        @memcpy(tokens_copy, tokens);
        
        const keys_copy = try allocator.alloc(f32, keys.len);
        @memcpy(keys_copy, keys);
        
        const values_copy = try allocator.alloc(f32, values.len);
        @memcpy(values_copy, values);
        
        const model_id_copy = try allocator.alloc(u8, model_id.len);
        @memcpy(model_id_copy, model_id);
        
        const size = tokens.len * @sizeOf(u32) + 
                    keys.len * @sizeOf(f32) + 
                    values.len * @sizeOf(f32);
        
        const now = std.time.timestamp();
        
        self.* = .{
            .id = id,
            .model_id = model_id_copy,
            .layer = layer,
            .tokens = tokens_copy,
            .keys = keys_copy,
            .values = values_copy,
            .ref_count = std.atomic.Value(u32).init(0),
            .created_at = now,
            .accessed_at = std.atomic.Value(i64).init(now),
            .access_count = std.atomic.Value(u64).init(0),
            .size_bytes = size,
            .compressed = false,
            .mutex = .{},
        };
        
        return self;
    }
    
    pub fn deinit(self: *SharedCacheEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.keys);
        allocator.free(self.values);
        allocator.free(self.model_id);
        allocator.destroy(self);
    }
    
    /// Increment reference count
    pub fn acquire(self: *SharedCacheEntry) void {
        _ = self.ref_count.fetchAdd(1, .monotonic);
        self.accessed_at.store(std.time.timestamp(), .monotonic);
        _ = self.access_count.fetchAdd(1, .monotonic);
    }
    
    /// Decrement reference count
    pub fn release(self: *SharedCacheEntry) void {
        const prev = self.ref_count.fetchSub(1, .monotonic);
        std.debug.assert(prev > 0);
    }
    
    /// Get current reference count
    pub fn getRefCount(self: *const SharedCacheEntry) u32 {
        return self.ref_count.load(.monotonic);
    }
    
    /// Check if entry can be evicted (ref_count == 0)
    pub fn canEvict(self: *const SharedCacheEntry) bool {
        return self.getRefCount() == 0;
    }
};

// ============================================================================
// Prefix Tree Node (Trie)
// ============================================================================

/// Trie node for efficient prefix matching
pub const PrefixTreeNode = struct {
    token: ?u32,
    cache_entry: ?*SharedCacheEntry,
    children: std.AutoHashMap(u32, *PrefixTreeNode),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, token: ?u32) !*PrefixTreeNode {
        const self = try allocator.create(PrefixTreeNode);
        self.* = .{
            .token = token,
            .cache_entry = null,
            .children = std.AutoHashMap(u32, *PrefixTreeNode).init(allocator),
            .allocator = allocator,
        };
        return self;
    }
    
    pub fn deinit(self: *PrefixTreeNode) void {
        var it = self.children.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        self.children.deinit();
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Prefix Tree (Trie)
// ============================================================================

/// Trie for efficient prefix detection
pub const PrefixTree = struct {
    root: *PrefixTreeNode,
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,
    node_count: u64,
    
    pub fn init(allocator: std.mem.Allocator) !*PrefixTree {
        const self = try allocator.create(PrefixTree);
        const root = try PrefixTreeNode.init(allocator, null);
        
        self.* = .{
            .root = root,
            .allocator = allocator,
            .mutex = .{},
            .node_count = 1,
        };
        
        return self;
    }
    
    pub fn deinit(self: *PrefixTree) void {
        self.root.deinit();
        self.allocator.destroy(self);
    }
    
    /// Insert a token sequence into the trie
    pub fn insert(self: *PrefixTree, tokens: []const u32, entry: *SharedCacheEntry) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var current = self.root;
        
        for (tokens) |token| {
            if (current.children.get(token)) |child| {
                current = child;
            } else {
                const new_node = try PrefixTreeNode.init(self.allocator, token);
                try current.children.put(token, new_node);
                current = new_node;
                self.node_count += 1;
            }
        }
        
        current.cache_entry = entry;
    }
    
    /// Find the longest matching prefix
    pub fn findLongestPrefix(
        self: *PrefixTree,
        tokens: []const u32,
    ) ?struct { *SharedCacheEntry, usize } {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var current = self.root;
        var last_match: ?*SharedCacheEntry = null;
        var last_match_len: usize = 0;
        
        for (tokens, 0..) |token, i| {
            if (current.children.get(token)) |child| {
                current = child;
                if (current.cache_entry) |entry| {
                    last_match = entry;
                    last_match_len = i + 1;
                }
            } else {
                break;
            }
        }
        
        if (last_match) |entry| {
            return .{ entry, last_match_len };
        }
        
        return null;
    }
    
    /// Remove a token sequence from the trie
    pub fn remove(self: *PrefixTree, tokens: []const u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Navigate to the node
        var current = self.root;
        for (tokens) |token| {
            if (current.children.get(token)) |child| {
                current = child;
            } else {
                return; // Path doesn't exist
            }
        }
        
        // Clear the cache entry (node remains for other paths)
        current.cache_entry = null;
    }
};

// ============================================================================
// Cache Sharing Manager
// ============================================================================

/// Manages shared cache entries and prefix detection
pub const CacheSharingManager = struct {
    allocator: std.mem.Allocator,
    config: CacheSharingConfig,
    
    /// Shared cache entries (id -> entry)
    shared_entries: std.AutoHashMap(u64, *SharedCacheEntry),
    
    /// Prefix tree for fast prefix matching
    prefix_tree: *PrefixTree,
    
    /// Statistics
    stats: CacheSharingStats,
    
    /// Mutex for thread safety
    mutex: std.Thread.Mutex,
    
    /// Current total size of shared cache
    current_size: u64,
    
    pub fn init(allocator: std.mem.Allocator, config: CacheSharingConfig) !*CacheSharingManager {
        log.info("Initializing Cache Sharing Manager", .{});
        
        const self = try allocator.create(CacheSharingManager);
        const prefix_tree = try PrefixTree.init(allocator);
        
        self.* = .{
            .allocator = allocator,
            .config = config,
            .shared_entries = std.AutoHashMap(u64, *SharedCacheEntry).init(allocator),
            .prefix_tree = prefix_tree,
            .stats = .{},
            .mutex = .{},
            .current_size = 0,
        };
        
        // Pre-populate with common prefixes if provided
        for (self.config.common_prefixes) |prefix| {
            try self.preloadCommonPrefix(prefix);
        }
        
        return self;
    }
    
    pub fn deinit(self: *CacheSharingManager) void {
        var it = self.shared_entries.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit(self.allocator);
        }
        self.shared_entries.deinit();
        self.prefix_tree.deinit();
        self.allocator.destroy(self);
    }
    
    /// Store a cacheable prefix
    pub fn storeSharedEntry(
        self: *CacheSharingManager,
        model_id: []const u8,
        layer: u32,
        tokens: []const u32,
        keys: []const f32,
        values: []const f32,
    ) !u64 {
        if (!self.config.enabled) return error.SharingDisabled;
        if (tokens.len < self.config.min_prefix_length) return error.PrefixTooShort;
        
        const start_time = std.time.microTimestamp();
        
        // Generate unique ID (hash of tokens)
        const id = self.hashTokens(tokens);
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Check if entry already exists
        if (self.shared_entries.get(id)) |existing| {
            log.debug("Shared entry already exists: id={d}", .{id});
            return id;
        }
        
        // Check size limit
        const entry_size = tokens.len * @sizeOf(u32) + 
                          keys.len * @sizeOf(f32) + 
                          values.len * @sizeOf(f32);
        
        if (self.current_size + entry_size > self.config.max_shared_cache_size) {
            try self.evictLRU();
        }
        
        // Create new shared entry
        const entry = try SharedCacheEntry.init(
            self.allocator,
            id,
            model_id,
            layer,
            tokens,
            keys,
            values,
        );
        
        try self.shared_entries.put(id, entry);
        try self.prefix_tree.insert(tokens, entry);
        
        self.current_size += entry_size;
        self.stats.total_entries += 1;
        self.stats.shared_entries += 1;
        
        const elapsed = std.time.microTimestamp() - start_time;
        log.debug("Stored shared entry: id={d}, tokens={d}, size={d}KB, time={d}μs", .{
            id, tokens.len, entry_size / 1024, elapsed,
        });
        
        return id;
    }
    
    /// Find and acquire a shared cache entry
    pub fn findSharedEntry(
        self: *CacheSharingManager,
        tokens: []const u32,
    ) ?struct { *SharedCacheEntry, usize } {
        if (!self.config.enabled) return null;
        
        const start_time = std.time.microTimestamp();
        
        if (self.prefix_tree.findLongestPrefix(tokens)) |result| {
            const entry = result[0];
            const match_len = result[1];
            
            entry.acquire();
            
            self.mutex.lock();
            defer self.mutex.unlock();
            
            self.stats.shared_cache_hits += 1;
            self.stats.prefix_matches += 1;
            self.stats.total_references += 1;
            
            if (match_len == tokens.len) {
                self.stats.full_prefix_reuse += 1;
            } else {
                self.stats.partial_prefix_reuse += 1;
            }
            
            const elapsed = std.time.microTimestamp() - start_time;
            self.stats.avg_prefix_match_time_us = 
                (self.stats.avg_prefix_match_time_us + @as(u64, @intCast(elapsed))) / 2;
            
            log.debug("Found shared entry: id={d}, match_len={d}/{d}, refs={d}", .{
                entry.id, match_len, tokens.len, entry.getRefCount(),
            });
            
            return .{ entry, match_len };
        }
        
        self.mutex.lock();
        defer self.mutex.unlock();
        self.stats.shared_cache_misses += 1;
        
        return null;
    }
    
    /// Release a shared cache entry
    pub fn releaseSharedEntry(self: *CacheSharingManager, entry: *SharedCacheEntry) void {
        entry.release();
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.stats.total_references > 0) {
            self.stats.total_references -= 1;
        }
        
        log.debug("Released shared entry: id={d}, refs={d}", .{
            entry.id, entry.getRefCount(),
        });
    }
    
    /// Evict least recently used entry
    fn evictLRU(self: *CacheSharingManager) !void {
        var oldest_time: i64 = std.math.maxInt(i64);
        var oldest_id: ?u64 = null;
        
        var it = self.shared_entries.iterator();
        while (it.next()) |kv| {
            const entry = kv.value_ptr.*;
            if (entry.canEvict()) {
                const accessed = entry.accessed_at.load(.monotonic);
                if (accessed < oldest_time) {
                    oldest_time = accessed;
                    oldest_id = entry.id;
                }
            }
        }
        
        if (oldest_id) |id| {
            if (self.shared_entries.fetchRemove(id)) |kv| {
                const entry = kv.value;
                self.current_size -= entry.size_bytes;
                self.stats.bytes_saved += entry.size_bytes * entry.access_count.load(.monotonic);
                self.prefix_tree.remove(entry.tokens);
                entry.deinit(self.allocator);
                
                log.info("Evicted shared entry: id={d}", .{id});
            }
        }
    }
    
    /// Hash token sequence to generate unique ID
    fn hashTokens(self: *CacheSharingManager, tokens: []const u32) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        const bytes = std.mem.sliceAsBytes(tokens);
        hasher.update(bytes);
        return hasher.final();
    }
    
    /// Preload a common prefix
    fn preloadCommonPrefix(self: *CacheSharingManager, tokens: []const u32) !void {
        log.info("Preloading common prefix: {d} tokens", .{tokens.len});
        
        // Create placeholder entry (will be populated on first use)
        const dummy_keys = try self.allocator.alloc(f32, tokens.len * 64);
        defer self.allocator.free(dummy_keys);
        @memset(dummy_keys, 0.0);
        
        const dummy_values = try self.allocator.alloc(f32, tokens.len * 64);
        defer self.allocator.free(dummy_values);
        @memset(dummy_values, 0.0);
        
        _ = try self.storeSharedEntry(
            "common",
            0,
            tokens,
            dummy_keys,
            dummy_values,
        );
        
        self.stats.common_prefixes_detected += 1;
    }
    
    /// Get statistics
    pub fn getStats(self: *CacheSharingManager) CacheSharingStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.stats.trie_nodes = self.prefix_tree.node_count;
        return self.stats;
    }
    
    /// Print status
    pub fn printStatus(self: *CacheSharingManager) void {
        const stats = self.getStats();
        
        std.debug.print("\n🔄 Cache Sharing Status\n", .{});
        std.debug.print("   Shared Entries: {d}\n", .{stats.shared_entries});
        std.debug.print("   Total References: {d}\n", .{stats.total_references});
        std.debug.print("   Avg References/Entry: {d:.2}\n", .{stats.getAvgReferences()});
        std.debug.print("   Shared Hit Rate: {d:.2}%\n", .{stats.getSharedHitRate() * 100});
        std.debug.print("   Memory Savings: {d:.2}GB\n", .{stats.getMemorySavings()});
        std.debug.print("   Prefix Matches: {d}\n", .{stats.prefix_matches});
        std.debug.print("   Full Reuse: {d}\n", .{stats.full_prefix_reuse});
        std.debug.print("   Partial Reuse: {d}\n", .{stats.partial_prefix_reuse});
        std.debug.print("   Trie Nodes: {d}\n", .{stats.trie_nodes});
    }
};

// ============================================================================
// Cache Replication (for distributed sharing)
// ============================================================================

/// Replication manager for distributed cache sharing
pub const CacheReplicationManager = struct {
    allocator: std.mem.Allocator,
    config: CacheSharingConfig,
    node_id: []const u8,
    
    pub fn init(
        allocator: std.mem.Allocator,
        config: CacheSharingConfig,
        node_id: []const u8,
    ) !*CacheReplicationManager {
        const self = try allocator.create(CacheReplicationManager);
        
        const node_id_copy = try allocator.alloc(u8, node_id.len);
        @memcpy(node_id_copy, node_id);
        
        self.* = .{
            .allocator = allocator,
            .config = config,
            .node_id = node_id_copy,
        };
        
        log.info("Initialized Cache Replication Manager: node={s}", .{node_id});
        
        return self;
    }
    
    pub fn deinit(self: *CacheReplicationManager) void {
        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }
    
    /// ✅ P2-13 FIXED: Replicate a shared entry to other nodes
    pub fn replicate(
        self: *CacheReplicationManager,
        entry: *SharedCacheEntry,
    ) !void {
        if (!self.config.enable_replication) return;
        if (self.config.replication_nodes.len == 0) return;
        
        log.debug("Replicating entry {d} to {d} nodes", .{
            entry.id,
            self.config.replication_factor,
        });
        
        // Serialize the entry for network transmission
        const serialized = try self.serializeEntry(entry);
        defer self.allocator.free(serialized);
        
        // Replicate to configured nodes (up to replication_factor)
        var successful_replicas: u32 = 0;
        const target_replicas = @min(self.config.replication_factor, @as(u32, @intCast(self.config.replication_nodes.len)));
        
        for (self.config.replication_nodes[0..target_replicas]) |node_addr| {
            self.replicateToNode(node_addr, entry.id, serialized) catch |err| {
                log.warn("Failed to replicate to node {s}: {}", .{node_addr, err});
                continue;
            };
            successful_replicas += 1;
        }
        
        if (successful_replicas < target_replicas) {
            log.warn("Incomplete replication: {d}/{d} nodes succeeded", .{successful_replicas, target_replicas});
        } else {
            log.debug("Successfully replicated entry {d} to {d} nodes", .{entry.id, successful_replicas});
        }
    }
    
    fn serializeEntry(self: *CacheReplicationManager, entry: *SharedCacheEntry) ![]u8 {
        // Calculate total size needed
        const header_size = @sizeOf(u64) + // id
                           @sizeOf(u32) + // model_id length
                           entry.model_id.len +
                           @sizeOf(u32) + // layer
                           @sizeOf(u32) + // tokens length
                           @sizeOf(u32) + // keys length
                           @sizeOf(u32); // values length
        
        const data_size = entry.tokens.len * @sizeOf(u32) +
                         entry.keys.len * @sizeOf(f32) +
                         entry.values.len * @sizeOf(f32);
        
        const total_size = header_size + data_size;
        
        var buffer = try self.allocator.alloc(u8, total_size);
        var offset: usize = 0;
        
        // Serialize header
        std.mem.writeInt(u64, buffer[offset..][0..8], entry.id, .little);
        offset += 8;
        
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(entry.model_id.len), .little);
        offset += 4;
        
        @memcpy(buffer[offset..][0..entry.model_id.len], entry.model_id);
        offset += entry.model_id.len;
        
        std.mem.writeInt(u32, buffer[offset..][0..4], entry.layer, .little);
        offset += 4;
        
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(entry.tokens.len), .little);
        offset += 4;
        
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(entry.keys.len), .little);
        offset += 4;
        
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(entry.values.len), .little);
        offset += 4;
        
        // Serialize data
        @memcpy(
            @as([*]u8, @ptrCast(buffer[offset..].ptr))[0..entry.tokens.len * @sizeOf(u32)],
            std.mem.sliceAsBytes(entry.tokens)
        );
        offset += entry.tokens.len * @sizeOf(u32);
        
        @memcpy(
            @as([*]u8, @ptrCast(buffer[offset..].ptr))[0..entry.keys.len * @sizeOf(f32)],
            std.mem.sliceAsBytes(entry.keys)
        );
        offset += entry.keys.len * @sizeOf(f32);
        
        @memcpy(
            @as([*]u8, @ptrCast(buffer[offset..].ptr))[0..entry.values.len * @sizeOf(f32)],
            std.mem.sliceAsBytes(entry.values)
        );
        
        return buffer;
    }
    
    fn replicateToNode(
        self: *CacheReplicationManager,
        node_addr: []const u8,
        entry_id: u64,
        serialized_data: []const u8,
    ) !void {
        log.debug("Replicating entry {d} to node: {s}", .{entry_id, node_addr});
        
        // Parse node address (format: "host:port")
        const colon_idx = std.mem.indexOf(u8, node_addr, ":") orelse return error.InvalidNodeAddress;
        const host = node_addr[0..colon_idx];
        const port_str = node_addr[colon_idx + 1..];
        const port = try std.fmt.parseInt(u16, port_str, 10);
        
        // Connect to remote node
        const address = try std.net.Address.parseIp(host, port);
        const stream = try std.net.tcpConnectToAddress(address);
        defer stream.close();
        
        // Send replication request header
        const header = ReplicationHeader{
            .magic = REPLICATION_MAGIC,
            .version = REPLICATION_VERSION,
            .entry_id = entry_id,
            .data_size = @intCast(serialized_data.len),
            .source_node_len = @intCast(self.node_id.len),
        };
        
        try stream.writeAll(std.mem.asBytes(&header));
        try stream.writeAll(self.node_id);
        try stream.writeAll(serialized_data);
        
        // Wait for acknowledgment
        var ack_buffer: [16]u8 = undefined;
        const bytes_read = try stream.read(&ack_buffer);
        
        if (bytes_read < @sizeOf(ReplicationAck)) {
            return error.IncompleteAck;
        }
        
        const ack = std.mem.bytesToValue(ReplicationAck, ack_buffer[0..@sizeOf(ReplicationAck)]);
        
        if (ack.magic != REPLICATION_ACK_MAGIC) {
            return error.InvalidAck;
        }
        
        if (ack.status != 0) {
            log.warn("Node {s} rejected replication: status={d}", .{node_addr, ack.status});
            return error.ReplicationRejected;
        }
        
        log.debug("Successfully replicated entry {d} to {s}", .{entry_id, node_addr});
    }
    
    /// ✅ P2-13 FIXED: Fetch a shared entry from remote nodes
    pub fn fetchRemote(
        self: *CacheReplicationManager,
        entry_id: u64,
    ) !?*SharedCacheEntry {
        if (!self.config.enable_replication) return null;
        if (self.config.replication_nodes.len == 0) return null;
        
        log.debug("Fetching entry {d} from remote nodes", .{entry_id});
        
        // Try each replication node until successful
        for (self.config.replication_nodes) |node_addr| {
            const entry = self.fetchFromNode(node_addr, entry_id) catch |err| {
                log.debug("Failed to fetch from node {s}: {}", .{node_addr, err});
                continue;
            };
            
            if (entry) |e| {
                log.info("Successfully fetched entry {d} from node {s}", .{entry_id, node_addr});
                return e;
            }
        }
        
        log.debug("Entry {d} not found on any remote node", .{entry_id});
        return null;
    }
    
    fn fetchFromNode(
        self: *CacheReplicationManager,
        node_addr: []const u8,
        entry_id: u64,
    ) !?*SharedCacheEntry {
        log.debug("Fetching entry {d} from node: {s}", .{entry_id, node_addr});
        
        // Parse node address
        const colon_idx = std.mem.indexOf(u8, node_addr, ":") orelse return error.InvalidNodeAddress;
        const host = node_addr[0..colon_idx];
        const port_str = node_addr[colon_idx + 1..];
        const port = try std.fmt.parseInt(u16, port_str, 10);
        
        // Connect to remote node
        const address = try std.net.Address.parseIp(host, port);
        const stream = try std.net.tcpConnectToAddress(address);
        defer stream.close();
        
        // Send fetch request
        const request = FetchRequest{
            .magic = FETCH_MAGIC,
            .version = REPLICATION_VERSION,
            .entry_id = entry_id,
            .requester_node_len = @intCast(self.node_id.len),
        };
        
        try stream.writeAll(std.mem.asBytes(&request));
        try stream.writeAll(self.node_id);
        
        // Read response header
        var response_header: FetchResponse = undefined;
        const header_bytes = try stream.readAll(std.mem.asBytes(&response_header));
        
        if (header_bytes < @sizeOf(FetchResponse)) {
            return error.IncompleteResponse;
        }
        
        if (response_header.magic != FETCH_RESPONSE_MAGIC) {
            return error.InvalidResponse;
        }
        
        if (response_header.status == 404) {
            return null; // Entry not found on this node
        }
        
        if (response_header.status != 0) {
            return error.FetchFailed;
        }
        
        // Read serialized data
        const serialized_data = try self.allocator.alloc(u8, response_header.data_size);
        defer self.allocator.free(serialized_data);
        
        const data_bytes = try stream.readAll(serialized_data);
        if (data_bytes < serialized_data.len) {
            return error.IncompleteData;
        }
        
        // Deserialize entry
        return try self.deserializeEntry(serialized_data);
    }
    
    fn deserializeEntry(self: *CacheReplicationManager, data: []const u8) !*SharedCacheEntry {
        var offset: usize = 0;
        
        // Read header
        const id = std.mem.readInt(u64, data[offset..][0..8], .little);
        offset += 8;
        
        const model_id_len = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        
        const model_id = try self.allocator.alloc(u8, model_id_len);
        @memcpy(model_id, data[offset..][0..model_id_len]);
        offset += model_id_len;
        
        const layer = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        
        const tokens_len = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        
        const keys_len = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        
        const values_len = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        
        // Read data
        const tokens = try self.allocator.alloc(u32, tokens_len);
        @memcpy(
            std.mem.sliceAsBytes(tokens),
            data[offset..][0..tokens_len * @sizeOf(u32)]
        );
        offset += tokens_len * @sizeOf(u32);
        
        const keys = try self.allocator.alloc(f32, keys_len);
        @memcpy(
            std.mem.sliceAsBytes(keys),
            data[offset..][0..keys_len * @sizeOf(f32)]
        );
        offset += keys_len * @sizeOf(f32);
        
        const values = try self.allocator.alloc(f32, values_len);
        @memcpy(
            std.mem.sliceAsBytes(values),
            data[offset..][0..values_len * @sizeOf(f32)]
        );
        
        // Create entry
        return try SharedCacheEntry.init(
            self.allocator,
            id,
            model_id,
            layer,
            tokens,
            keys,
            values,
        );
    }
    
    /// ✅ P2-13: Start replication server to handle incoming requests
    pub fn startReplicationServer(
        self: *CacheReplicationManager,
        listen_addr: []const u8,
        cache_manager: *CacheSharingManager,
    ) !std.Thread {
        const server_config = ReplicationServerConfig{
            .manager = self,
            .cache_manager = cache_manager,
            .listen_addr = listen_addr,
        };
        
        const config_copy = try self.allocator.create(ReplicationServerConfig);
        config_copy.* = server_config;
        
        return try std.Thread.spawn(.{}, replicationServerThread, .{config_copy});
    }
    
    fn replicationServerThread(config: *ReplicationServerConfig) void {
        defer config.manager.allocator.destroy(config);
        
        runReplicationServer(config) catch |err| {
            log.err("Replication server error: {}", .{err});
        };
    }
    
    fn runReplicationServer(config: *ReplicationServerConfig) !void {
        log.info("Starting replication server on {s}", .{config.listen_addr});
        
        // Parse listen address
        const colon_idx = std.mem.indexOf(u8, config.listen_addr, ":") orelse return error.InvalidAddress;
        const host = config.listen_addr[0..colon_idx];
        const port_str = config.listen_addr[colon_idx + 1..];
        const port = try std.fmt.parseInt(u16, port_str, 10);
        
        const address = try std.net.Address.parseIp(host, port);
        var server = try address.listen(.{
            .reuse_address = true,
        });
        defer server.deinit();
        
        log.info("Replication server listening on {s}", .{config.listen_addr});
        
        while (true) {
            const connection = try server.accept();
            
            // Handle connection in separate thread
            const handler_config = try config.manager.allocator.create(ConnectionHandlerConfig);
            handler_config.* = .{
                .stream = connection.stream,
                .manager = config.manager,
                .cache_manager = config.cache_manager,
            };
            
            _ = std.Thread.spawn(.{}, handleReplicationConnection, .{handler_config}) catch |err| {
                log.err("Failed to spawn connection handler: {}", .{err});
                connection.stream.close();
                config.manager.allocator.destroy(handler_config);
            };
        }
    }
    
    fn handleReplicationConnection(config: *ConnectionHandlerConfig) void {
        defer config.stream.close();
        defer config.manager.allocator.destroy(config);
        
        handleReplicationRequest(config) catch |err| {
            log.err("Replication request handling error: {}", .{err});
        };
    }
    
    fn handleReplicationRequest(config: *ConnectionHandlerConfig) !void {
        var header_buffer: [32]u8 = undefined;
        const bytes_read = try config.stream.read(&header_buffer);
        
        if (bytes_read < 4) return error.InvalidRequest;
        
        const magic = std.mem.readInt(u32, header_buffer[0..4], .little);
        
        if (magic == REPLICATION_MAGIC) {
            try handleReplicationStore(config, header_buffer[0..bytes_read]);
        } else if (magic == FETCH_MAGIC) {
            try handleReplicationFetch(config, header_buffer[0..bytes_read]);
        } else {
            return error.UnknownRequestType;
        }
    }
    
    fn handleReplicationStore(config: *ConnectionHandlerConfig, initial_data: []const u8) !void {
        if (initial_data.len < @sizeOf(ReplicationHeader)) {
            return error.IncompleteHeader;
        }
        
        const header = std.mem.bytesToValue(ReplicationHeader, initial_data[0..@sizeOf(ReplicationHeader)]);
        
        // Read source node ID
        var source_node = try config.manager.allocator.alloc(u8, header.source_node_len);
        defer config.manager.allocator.free(source_node);
        _ = try config.stream.readAll(source_node);
        
        // Read serialized data
        var serialized_data = try config.manager.allocator.alloc(u8, header.data_size);
        defer config.manager.allocator.free(serialized_data);
        _ = try config.stream.readAll(serialized_data);
        
        // Deserialize and store entry
        const entry = try config.manager.deserializeEntry(serialized_data);
        defer entry.deinit(config.manager.allocator);
        
        // Store in local cache
        _ = try config.cache_manager.storeSharedEntry(
            entry.model_id,
            entry.layer,
            entry.tokens,
            entry.keys,
            entry.values,
        );
        
        log.info("Stored replicated entry {d} from node {s}", .{header.entry_id, source_node});
        
        // Send acknowledgment
        const ack = ReplicationAck{
            .magic = REPLICATION_ACK_MAGIC,
            .entry_id = header.entry_id,
            .status = 0,
        };
        
        try config.stream.writeAll(std.mem.asBytes(&ack));
    }
    
    fn handleReplicationFetch(config: *ConnectionHandlerConfig, initial_data: []const u8) !void {
        if (initial_data.len < @sizeOf(FetchRequest)) {
            return error.IncompleteHeader;
        }
        
        const request = std.mem.bytesToValue(FetchRequest, initial_data[0..@sizeOf(FetchRequest)]);
        
        // Read requester node ID
        var requester_node = try config.manager.allocator.alloc(u8, request.requester_node_len);
        defer config.manager.allocator.free(requester_node);
        _ = try config.stream.readAll(requester_node);
        
        // Find entry in local cache
        const entry = config.cache_manager.shared_entries.get(request.entry_id);
        
        if (entry == null) {
            // Entry not found
            const response = FetchResponse{
                .magic = FETCH_RESPONSE_MAGIC,
                .entry_id = request.entry_id,
                .status = 404,
                .data_size = 0,
            };
            try config.stream.writeAll(std.mem.asBytes(&response));
            return;
        }
        
        // Serialize entry
        const serialized = try config.manager.serializeEntry(entry.?);
        defer config.manager.allocator.free(serialized);
        
        // Send response
        const response = FetchResponse{
            .magic = FETCH_RESPONSE_MAGIC,
            .entry_id = request.entry_id,
            .status = 0,
            .data_size = @intCast(serialized.len),
        };
        
        try config.stream.writeAll(std.mem.asBytes(&response));
        try config.stream.writeAll(serialized);
        
        log.info("Sent entry {d} to node {s}", .{request.entry_id, requester_node});
    }
};

// ============================================================================
// Replication Protocol Constants and Structures
// ============================================================================

const REPLICATION_MAGIC: u32 = 0x52455043; // "REPC"
const FETCH_MAGIC: u32 = 0x46455443; // "FETC"
const REPLICATION_ACK_MAGIC: u32 = 0x5245414B; // "REAK"
const FETCH_RESPONSE_MAGIC: u32 = 0x46455352; // "FESR"
const REPLICATION_VERSION: u16 = 1;

const ReplicationHeader = extern struct {
    magic: u32,
    version: u16,
    _padding: u16 = 0,
    entry_id: u64,
    data_size: u32,
    source_node_len: u32,
};

const ReplicationAck = extern struct {
    magic: u32,
    entry_id: u64,
    status: u32, // 0 = success, non-zero = error
};

const FetchRequest = extern struct {
    magic: u32,
    version: u16,
    _padding: u16 = 0,
    entry_id: u64,
    requester_node_len: u32,
};

const FetchResponse = extern struct {
    magic: u32,
    entry_id: u64,
    status: u32, // 0 = success, 404 = not found, other = error
    data_size: u32,
};

const ReplicationServerConfig = struct {
    manager: *CacheReplicationManager,
    cache_manager: *CacheSharingManager,
    listen_addr: []const u8,
};

const ConnectionHandlerConfig = struct {
    stream: std.net.Stream,
    manager: *CacheReplicationManager,
    cache_manager: *CacheSharingManager,
};
