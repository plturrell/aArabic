//! Streaming Extraction Pipeline
//!
//! Connects nExtract → Cache → Inference with backpressure support.
//! Provides streaming document processing with per-stage queues,
//! configurable backpressure, and C ABI exports for Mojo integration.
//!
//! Features:
//! - Multi-stage pipeline: extract → cache → embed → index → inference
//! - Per-stage backpressure with configurable max queue sizes
//! - Metrics: queue depth, processing time per stage
//! - Batch processing for efficiency
//! - C ABI for Mojo integration

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Thread = std.Thread;
const Mutex = Thread.Mutex;

// Import nExtract parsers
const nExtract = @import("../../../../nExtract/zig/nExtract.zig");
const csv = nExtract.csv;
const json = nExtract.json;
const markdown = nExtract.markdown;
const html = nExtract.html;
const xml = nExtract.xml;

// Import HANA cache client
const hana_cache = @import("../cache/hana/hana_cache.zig");
const HanaCache = hana_cache.HanaCache;

// Import unified doc cache
const unified_cache = @import("../document_cache/unified_doc_cache.zig");
const DocType = unified_cache.DocType;
const CachedDocument = unified_cache.CachedDocument;

// ============================================================================
// Pipeline Types
// ============================================================================

/// Pipeline processing stages
pub const Stage = enum(u8) {
    extract = 0, // Parse document with nExtract
    cache = 1, // Store in DragonflyDB
    embed = 2, // Generate embeddings
    index = 3, // Index for search
    inference = 4, // LLM inference

    pub fn toString(self: Stage) []const u8 {
        return switch (self) {
            .extract => "extract",
            .cache => "cache",
            .embed => "embed",
            .index => "index",
            .inference => "inference",
        };
    }

    pub fn next(self: Stage) ?Stage {
        return switch (self) {
            .extract => .cache,
            .cache => .embed,
            .embed => .index,
            .index => .inference,
            .inference => null,
        };
    }
};

/// Pipeline item status
pub const ItemStatus = enum(u8) {
    pending = 0,
    processing = 1,
    completed = 2,
    failed = 3,
};

/// Pipeline item representing a document being processed
pub const PipelineItem = struct {
    id: [32]u8,
    stage: Stage,
    status: ItemStatus,
    doc_type: DocType,
    content: []const u8,
    parsed_content: ?[]const u8,
    embeddings: ?[]f32,
    result: ?[]const u8,
    @"error": ?[]const u8,
    started_at: i64,
    completed_at: ?i64,

    allocator: Allocator,

    pub fn init(allocator: Allocator) PipelineItem {
        return .{
            .id = [_]u8{0} ** 32,
            .stage = .extract,
            .status = .pending,
            .doc_type = .json,
            .content = &[_]u8{},
            .parsed_content = null,
            .embeddings = null,
            .result = null,
            .@"error" = null,
            .started_at = 0,
            .completed_at = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PipelineItem) void {
        if (self.content.len > 0) {
            self.allocator.free(self.content);
        }
        if (self.parsed_content) |p| {
            self.allocator.free(p);
        }
        if (self.embeddings) |e| {
            self.allocator.free(e);
        }
        if (self.result) |r| {
            self.allocator.free(r);
        }
        if (self.@"error") |e| {
            self.allocator.free(e);
        }
    }

    /// Compute item ID from content hash
    pub fn computeId(content: []const u8) [32]u8 {
        var hash: [32]u8 = undefined;
        Sha256.hash(content, &hash, .{});
        return hash;
    }

    /// Get hex-encoded ID
    pub fn getIdHex(self: *const PipelineItem) [64]u8 {
        return std.fmt.bytesToHex(self.id, .lower);
    }

    /// Get processing duration in nanoseconds
    pub fn getProcessingDuration(self: *const PipelineItem) ?i64 {
        if (self.completed_at) |end| {
            return end - self.started_at;
        }
        return null;
    }
};

/// Per-stage queue for backpressure management
pub const StageQueue = struct {
    items: ArrayList(*PipelineItem),
    max_size: usize,
    processing_count: usize,
    total_processed: u64,
    total_processing_time_ns: u64,
    mutex: Mutex,

    pub fn init(allocator: Allocator, max_size: usize) StageQueue {
        return .{
            .items = ArrayList(*PipelineItem).init(allocator),
            .max_size = max_size,
            .processing_count = 0,
            .total_processed = 0,
            .total_processing_time_ns = 0,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *StageQueue) void {
        self.items.deinit();
    }

    pub fn isFull(self: *StageQueue) bool {
        return self.items.items.len >= self.max_size;
    }

    pub fn isEmpty(self: *StageQueue) bool {
        return self.items.items.len == 0;
    }

    pub fn depth(self: *StageQueue) usize {
        return self.items.items.len;
    }

    pub fn avgProcessingTimeNs(self: *StageQueue) u64 {
        if (self.total_processed == 0) return 0;
        return self.total_processing_time_ns / self.total_processed;
    }
};

/// Pipeline metrics for monitoring
pub const PipelineMetrics = struct {
    items_submitted: u64,
    items_completed: u64,
    items_failed: u64,
    total_processing_time_ns: u64,
    stage_metrics: [5]StageMetrics, // One per stage

    pub const StageMetrics = struct {
        queue_depth: usize,
        items_processed: u64,
        avg_time_ns: u64,
        blocked_count: u64, // Times blocked due to backpressure
    };

    pub fn init() PipelineMetrics {
        return .{
            .items_submitted = 0,
            .items_completed = 0,
            .items_failed = 0,
            .total_processing_time_ns = 0,
            .stage_metrics = [_]StageMetrics{.{
                .queue_depth = 0,
                .items_processed = 0,
                .avg_time_ns = 0,
                .blocked_count = 0,
            }} ** 5,
        };
    }
};

/// Pipeline configuration
pub const PipelineConfig = struct {
    /// Max items per stage queue (backpressure threshold)
    max_queue_size: usize = 100,
    /// HANA host
    cache_host: []const u8 = "localhost",
    /// HANA port
    cache_port: u16 = 30015,
    /// HANA database
    cache_database: []const u8 = "NOPENAI_DB",
    /// HANA user
    cache_user: []const u8 = "SHIMMY_USER",
    /// HANA password
    cache_password: []const u8 = "",
    /// Default TTL for cached documents (seconds)
    default_ttl: u32 = 3600,
    /// Batch size for efficient processing
    batch_size: usize = 10,
    /// Enable embedding generation
    enable_embeddings: bool = true,
    /// Enable indexing stage
    enable_indexing: bool = true,
    /// Enable inference stage
    enable_inference: bool = false,
};

// ============================================================================
// Extraction Pipeline
// ============================================================================

pub const PipelineError = error{
    QueueFull,
    StageBlocked,
    ParseError,
    CacheError,
    EmbeddingError,
    IndexError,
    InferenceError,
    InvalidItem,
    ItemNotFound,
    OutOfMemory,
};

/// Extraction Pipeline connecting nExtract → Cache → Inference
pub const ExtractionPipeline = struct {
    allocator: Allocator,
    config: PipelineConfig,
    cache_client: ?*DragonflyClient,

    // Per-stage queues
    queues: [5]StageQueue,

    // Completed items
    completed: ArrayList(*PipelineItem),

    // All tracked items by ID
    items_by_id: std.AutoHashMap([32]u8, *PipelineItem),

    // Metrics
    metrics: PipelineMetrics,

    // Synchronization
    mutex: Mutex,

    const Self = @This();

    /// Initialize the extraction pipeline
    pub fn init(allocator: Allocator, config: PipelineConfig) !*Self {
        const pipeline = try allocator.create(Self);
        errdefer allocator.destroy(pipeline);

        // Initialize cache client
        var cache_client: ?*DragonflyClient = null;
        if (config.cache_host.len > 0) {
            cache_client = DragonflyClient.init(allocator, config.cache_host, config.cache_port) catch null;
        }

        pipeline.* = .{
            .allocator = allocator,
            .config = config,
            .cache_client = cache_client,
            .queues = undefined,
            .completed = ArrayList(*PipelineItem).init(allocator),
            .items_by_id = std.AutoHashMap([32]u8, *PipelineItem).init(allocator),
            .metrics = PipelineMetrics.init(),
            .mutex = .{},
        };

        // Initialize stage queues
        inline for (0..5) |i| {
            pipeline.queues[i] = StageQueue.init(allocator, config.max_queue_size);
        }

        return pipeline;
    }

    /// Deinitialize and free resources
    pub fn deinit(self: *Self) void {
        // Free all tracked items
        var it = self.items_by_id.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.items_by_id.deinit();

        // Free queues
        inline for (0..5) |i| {
            self.queues[i].deinit();
        }

        // Free completed list
        self.completed.deinit();

        // Free cache client
        if (self.cache_client) |client| {
            client.deinit();
        }

        self.allocator.destroy(self);
    }

    /// Submit a document to the pipeline
    pub fn submit(self: *Self, content: []const u8, doc_type: DocType) PipelineError!*PipelineItem {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if extract queue is full (backpressure)
        if (self.queues[0].isFull()) {
            self.metrics.stage_metrics[0].blocked_count += 1;
            return PipelineError.QueueFull;
        }

        // Create new pipeline item
        const item = self.allocator.create(PipelineItem) catch return PipelineError.OutOfMemory;
        item.* = PipelineItem.init(self.allocator);

        // Compute ID and set properties
        item.id = PipelineItem.computeId(content);
        item.doc_type = doc_type;
        item.content = self.allocator.dupe(u8, content) catch {
            self.allocator.destroy(item);
            return PipelineError.OutOfMemory;
        };
        item.started_at = std.time.timestamp();
        item.stage = .extract;
        item.status = .pending;

        // Track item
        self.items_by_id.put(item.id, item) catch {
            item.deinit();
            self.allocator.destroy(item);
            return PipelineError.OutOfMemory;
        };

        // Add to extract queue
        self.queues[0].items.append(item) catch {
            _ = self.items_by_id.remove(item.id);
            item.deinit();
            self.allocator.destroy(item);
            return PipelineError.OutOfMemory;
        };

        self.metrics.items_submitted += 1;
        return item;
    }

    /// Submit from path (reads file content)
    pub fn submitPath(self: *Self, path: []const u8, doc_type: DocType) PipelineError!*PipelineItem {
        const file = std.fs.cwd().openFile(path, .{}) catch return PipelineError.ParseError;
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 10 * 1024 * 1024) catch return PipelineError.ParseError;
        defer self.allocator.free(content);

        return self.submit(content, doc_type);
    }

    /// Process one item from the pipeline (non-blocking)
    pub fn process(self: *Self) !usize {
        var processed: usize = 0;

        // Process each stage in order
        inline for (0..5) |stage_idx| {
            const stage: Stage = @enumFromInt(stage_idx);
            if (try self.processStage(stage)) {
                processed += 1;
            }
        }

        return processed;
    }

    /// Process all pending items in the pipeline
    pub fn processAll(self: *Self) !usize {
        var total_processed: usize = 0;

        while (true) {
            const processed = try self.process();
            if (processed == 0) break;
            total_processed += processed;
        }

        return total_processed;
    }

    /// Process a single stage
    fn processStage(self: *Self, stage: Stage) !bool {
        const stage_idx = @intFromEnum(stage);

        self.mutex.lock();
        defer self.mutex.unlock();

        var queue = &self.queues[stage_idx];

        // Get next item from queue
        if (queue.isEmpty()) return false;

        // Check if next stage is blocked (backpressure)
        if (stage.next()) |next_stage| {
            const next_idx = @intFromEnum(next_stage);

            // Skip disabled stages
            const skip = switch (next_stage) {
                .embed => !self.config.enable_embeddings,
                .index => !self.config.enable_indexing,
                .inference => !self.config.enable_inference,
                else => false,
            };

            if (!skip and self.queues[next_idx].isFull()) {
                self.metrics.stage_metrics[stage_idx].blocked_count += 1;
                return false;
            }
        }

        // Pop item from queue
        const item = queue.items.orderedRemove(0);
        item.status = .processing;
        queue.processing_count += 1;

        // Process based on stage (unlock during processing)
        self.mutex.unlock();
        const start_time = std.time.nanoTimestamp();
        const process_result = self.executeStage(stage, item);
        const end_time = std.time.nanoTimestamp();
        self.mutex.lock();

        queue.processing_count -= 1;
        const duration: u64 = @intCast(end_time - start_time);
        queue.total_processing_time_ns += duration;
        queue.total_processed += 1;

        // Update stage metrics
        self.metrics.stage_metrics[stage_idx].items_processed += 1;
        self.metrics.stage_metrics[stage_idx].avg_time_ns = queue.avgProcessingTimeNs();
        self.metrics.stage_metrics[stage_idx].queue_depth = queue.depth();

        if (process_result) {
            // Move to next stage or complete
            if (stage.next()) |next_stage| {
                // Check if next stage should be skipped
                const skip = switch (next_stage) {
                    .embed => !self.config.enable_embeddings,
                    .index => !self.config.enable_indexing,
                    .inference => !self.config.enable_inference,
                    else => false,
                };

                if (skip) {
                    // Find next enabled stage
                    var target_stage = next_stage;
                    while (target_stage.next()) |s| {
                        const skip_this = switch (s) {
                            .embed => !self.config.enable_embeddings,
                            .index => !self.config.enable_indexing,
                            .inference => !self.config.enable_inference,
                            else => false,
                        };
                        if (!skip_this) {
                            target_stage = s;
                            break;
                        }
                        target_stage = s;
                    }
                    item.stage = target_stage;
                    if (target_stage == .inference and !self.config.enable_inference) {
                        // All remaining stages disabled, complete item
                        item.status = .completed;
                        item.completed_at = std.time.timestamp();
                        self.completed.append(item) catch {};
                        self.metrics.items_completed += 1;
                        return true;
                    }
                } else {
                    item.stage = next_stage;
                }

                item.status = .pending;
                const next_idx = @intFromEnum(item.stage);
                self.queues[next_idx].items.append(item) catch {};
            } else {
                // Pipeline complete
                item.status = .completed;
                item.completed_at = std.time.timestamp();
                self.completed.append(item) catch {};
                self.metrics.items_completed += 1;
            }
        } else |err| {
            // Mark as failed
            item.status = .failed;
            item.completed_at = std.time.timestamp();
            item.@"error" = self.allocator.dupe(u8, @errorName(err)) catch null;
            self.completed.append(item) catch {};
            self.metrics.items_failed += 1;
        }

        return true;
    }

    /// Execute stage-specific processing
    fn executeStage(self: *Self, stage: Stage, item: *PipelineItem) !void {
        switch (stage) {
            .extract => try self.executeExtract(item),
            .cache => try self.executeCache(item),
            .embed => try self.executeEmbed(item),
            .index => try self.executeIndex(item),
            .inference => try self.executeInference(item),
        }
    }

    /// Extract stage: Parse document with nExtract
    fn executeExtract(self: *Self, item: *PipelineItem) !void {
        const parsed = switch (item.doc_type) {
            .csv => blk: {
                var parser = csv.Parser.init(self.allocator, .{});
                var doc = parser.parse(item.content) catch return PipelineError.ParseError;
                doc.deinit();
                break :blk try self.allocator.dupe(u8, item.content);
            },
            .json => blk: {
                var parser = json.Parser.init(self.allocator, .{});
                var doc = parser.parse(item.content) catch return PipelineError.ParseError;
                doc.deinit();
                break :blk try self.allocator.dupe(u8, item.content);
            },
            .markdown => blk: {
                var parser = markdown.Parser.init(self.allocator);
                defer parser.deinit();
                const ast = parser.parse(item.content) catch return PipelineError.ParseError;
                ast.deinit();
                break :blk try self.allocator.dupe(u8, item.content);
            },
            .html => blk: {
                const doc = html.HtmlParser.parse(self.allocator, item.content) catch return PipelineError.ParseError;
                var doc_mut = doc;
                doc_mut.deinit();
                break :blk try self.allocator.dupe(u8, item.content);
            },
            .xml => blk: {
                var parser = xml.Parser.init(self.allocator);
                defer parser.deinit();
                const node = parser.parse(item.content) catch return PipelineError.ParseError;
                if (node) |n| n.deinit();
                break :blk try self.allocator.dupe(u8, item.content);
            },
        };
        item.parsed_content = parsed;
    }

    /// Cache stage: Store in HANA
    fn executeCache(self: *Self, item: *PipelineItem) !void {
        if (self.cache_client) |client| {
            const content = item.parsed_content orelse item.content;
            var key_buf: [70]u8 = undefined;
            @memcpy(key_buf[0..4], "doc:");
            const hex = std.fmt.bytesToHex(item.id, .lower);
            @memcpy(key_buf[4..68], &hex);

            client.set(key_buf[0..68], content, self.config.default_ttl) catch return PipelineError.CacheError;
        }
    }

    /// Embed stage: Generate embeddings (placeholder)
    fn executeEmbed(_: *Self, item: *PipelineItem) !void {
        // Placeholder: In production, call embedding model
        // For now, create dummy embeddings
        _ = item;
        // item.embeddings = ... would be set here
    }

    /// Index stage: Index for search (placeholder)
    fn executeIndex(_: *Self, item: *PipelineItem) !void {
        // Placeholder: In production, send to vector DB
        _ = item;
    }

    /// Inference stage: LLM inference (placeholder)
    fn executeInference(_: *Self, item: *PipelineItem) !void {
        // Placeholder: In production, call LLM
        _ = item;
    }

    /// Get completed items
    pub fn getResults(self: *Self) []*PipelineItem {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.completed.items;
    }

    /// Get item by ID
    pub fn getItem(self: *Self, id: [32]u8) ?*PipelineItem {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.items_by_id.get(id);
    }

    /// Configure backpressure for a stage
    pub fn setPressure(self: *Self, stage: Stage, max_pending: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const idx = @intFromEnum(stage);
        self.queues[idx].max_size = max_pending;
    }

    /// Get current metrics
    pub fn getMetrics(self: *Self) PipelineMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Update queue depths
        inline for (0..5) |i| {
            self.metrics.stage_metrics[i].queue_depth = self.queues[i].depth();
        }

        return self.metrics;
    }

    /// Get pending count for a stage
    pub fn getPendingCount(self: *Self, stage: Stage) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        const idx = @intFromEnum(stage);
        return self.queues[idx].depth();
    }

    /// Get total pending count across all stages
    pub fn getTotalPending(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        var total: usize = 0;
        inline for (0..5) |i| {
            total += self.queues[i].depth();
        }
        return total;
    }

    /// Clear completed items
    pub fn clearCompleted(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.completed.clearRetainingCapacity();
    }
};

// ============================================================================
// C ABI Exports for Mojo Integration
// ============================================================================

const CPipeline = opaque {};

/// C-compatible pipeline config
pub const CPipelineConfig = extern struct {
    max_queue_size: u32,
    cache_host: [*:0]const u8,
    cache_port: u16,
    default_ttl: u32,
    batch_size: u32,
    enable_embeddings: bool,
    enable_indexing: bool,
    enable_inference: bool,
};

/// Create a new extraction pipeline
export fn pipeline_create(config: *const CPipelineConfig) callconv(.c) ?*CPipeline {
    const allocator = std.heap.c_allocator;

    const zig_config = PipelineConfig{
        .max_queue_size = config.max_queue_size,
        .cache_host = mem.span(config.cache_host),
        .cache_port = config.cache_port,
        .default_ttl = config.default_ttl,
        .batch_size = config.batch_size,
        .enable_embeddings = config.enable_embeddings,
        .enable_indexing = config.enable_indexing,
        .enable_inference = config.enable_inference,
    };

    const pipeline = ExtractionPipeline.init(allocator, zig_config) catch return null;
    return @ptrCast(pipeline);
}

/// Create pipeline with default config
export fn pipeline_create_default() callconv(.c) ?*CPipeline {
    const allocator = std.heap.c_allocator;
    const pipeline = ExtractionPipeline.init(allocator, .{}) catch return null;
    return @ptrCast(pipeline);
}

/// Destroy a pipeline
export fn pipeline_destroy(pipeline: *CPipeline) callconv(.c) void {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));
    real_pipeline.deinit();
}

/// Submit content to pipeline
export fn pipeline_submit(
    pipeline: *CPipeline,
    content: [*]const u8,
    len: usize,
    doc_type: u8,
    id_out: *[32]u8,
) callconv(.c) i32 {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));
    const content_slice = content[0..len];
    const dtype: DocType = @enumFromInt(doc_type);

    const item = real_pipeline.submit(content_slice, dtype) catch |err| {
        return switch (err) {
            PipelineError.QueueFull => -2,
            PipelineError.OutOfMemory => -3,
            else => -1,
        };
    };

    id_out.* = item.id;
    return 0;
}

/// Process one item from pipeline
export fn pipeline_process(pipeline: *CPipeline) callconv(.c) i32 {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));
    const count = real_pipeline.process() catch return -1;
    return @intCast(count);
}

/// Process all pending items
export fn pipeline_process_all(pipeline: *CPipeline) callconv(.c) i32 {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));
    const count = real_pipeline.processAll() catch return -1;
    return @intCast(count);
}

/// Get result for an item
export fn pipeline_get_result(
    pipeline: *CPipeline,
    id: *const [32]u8,
    status_out: *u8,
    stage_out: *u8,
    error_out: *[*]const u8,
    error_len_out: *usize,
) callconv(.c) i32 {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));

    if (real_pipeline.getItem(id.*)) |item| {
        status_out.* = @intFromEnum(item.status);
        stage_out.* = @intFromEnum(item.stage);

        if (item.@"error") |err| {
            error_out.* = err.ptr;
            error_len_out.* = err.len;
        } else {
            error_out.* = undefined;
            error_len_out.* = 0;
        }
        return 0;
    }

    return 1; // Not found
}

/// Get completed count
export fn pipeline_get_completed_count(pipeline: *CPipeline) callconv(.c) u64 {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));
    return real_pipeline.metrics.items_completed;
}

/// Get pending count
export fn pipeline_get_pending_count(pipeline: *CPipeline) callconv(.c) u64 {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));
    return @intCast(real_pipeline.getTotalPending());
}

/// Set backpressure for a stage
export fn pipeline_set_pressure(
    pipeline: *CPipeline,
    stage: u8,
    max_pending: u32,
) callconv(.c) void {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));
    const s: Stage = @enumFromInt(stage);
    real_pipeline.setPressure(s, max_pending);
}

/// Get queue depth for a stage
export fn pipeline_get_queue_depth(pipeline: *CPipeline, stage: u8) callconv(.c) u64 {
    const real_pipeline: *ExtractionPipeline = @ptrCast(@alignCast(pipeline));
    const s: Stage = @enumFromInt(stage);
    return @intCast(real_pipeline.getPendingCount(s));
}

// ============================================================================
// Tests
// ============================================================================

test "Stage - transitions" {
    try std.testing.expectEqual(Stage.cache, Stage.extract.next().?);
    try std.testing.expectEqual(Stage.embed, Stage.cache.next().?);
    try std.testing.expectEqual(Stage.index, Stage.embed.next().?);
    try std.testing.expectEqual(Stage.inference, Stage.index.next().?);
    try std.testing.expectEqual(@as(?Stage, null), Stage.inference.next());
}

test "Stage - toString" {
    try std.testing.expectEqualStrings("extract", Stage.extract.toString());
    try std.testing.expectEqualStrings("cache", Stage.cache.toString());
    try std.testing.expectEqualStrings("inference", Stage.inference.toString());
}

test "PipelineItem - computeId" {
    const content = "Hello, Pipeline!";
    const id = PipelineItem.computeId(content);
    try std.testing.expect(id[0] != 0 or id[1] != 0);
}

test "PipelineItem - init and deinit" {
    const allocator = std.testing.allocator;
    var item = PipelineItem.init(allocator);
    defer item.deinit();

    try std.testing.expectEqual(Stage.extract, item.stage);
    try std.testing.expectEqual(ItemStatus.pending, item.status);
    try std.testing.expectEqual(@as(?[]const u8, null), item.@"error");
}

test "StageQueue - basic operations" {
    const allocator = std.testing.allocator;
    var queue = StageQueue.init(allocator, 10);
    defer queue.deinit();

    try std.testing.expect(queue.isEmpty());
    try std.testing.expect(!queue.isFull());
    try std.testing.expectEqual(@as(usize, 0), queue.depth());
}

test "StageQueue - backpressure" {
    const allocator = std.testing.allocator;
    var queue = StageQueue.init(allocator, 2);
    defer queue.deinit();

    // Create dummy items
    var item1 = try allocator.create(PipelineItem);
    item1.* = PipelineItem.init(allocator);
    defer {
        item1.deinit();
        allocator.destroy(item1);
    }

    var item2 = try allocator.create(PipelineItem);
    item2.* = PipelineItem.init(allocator);
    defer {
        item2.deinit();
        allocator.destroy(item2);
    }

    try queue.items.append(item1);
    try std.testing.expect(!queue.isFull());

    try queue.items.append(item2);
    try std.testing.expect(queue.isFull());
    try std.testing.expectEqual(@as(usize, 2), queue.depth());
}

test "PipelineMetrics - init" {
    const metrics = PipelineMetrics.init();

    try std.testing.expectEqual(@as(u64, 0), metrics.items_submitted);
    try std.testing.expectEqual(@as(u64, 0), metrics.items_completed);
    try std.testing.expectEqual(@as(u64, 0), metrics.items_failed);

    for (metrics.stage_metrics) |sm| {
        try std.testing.expectEqual(@as(usize, 0), sm.queue_depth);
        try std.testing.expectEqual(@as(u64, 0), sm.items_processed);
    }
}

test "PipelineConfig - defaults" {
    const config = PipelineConfig{};

    try std.testing.expectEqual(@as(usize, 100), config.max_queue_size);
    try std.testing.expectEqual(@as(u16, 6379), config.cache_port);
    try std.testing.expectEqual(@as(u32, 3600), config.default_ttl);
    try std.testing.expect(config.enable_embeddings);
    try std.testing.expect(!config.enable_inference);
}

test "ExtractionPipeline - init and deinit" {
    const allocator = std.testing.allocator;

    const config = PipelineConfig{
        .cache_host = "", // Disable cache connection for test
        .enable_embeddings = false,
        .enable_indexing = false,
        .enable_inference = false,
    };

    const pipeline = try ExtractionPipeline.init(allocator, config);
    defer pipeline.deinit();

    try std.testing.expectEqual(@as(u64, 0), pipeline.metrics.items_submitted);
}

test "ExtractionPipeline - submit and process JSON" {
    const allocator = std.testing.allocator;

    const config = PipelineConfig{
        .cache_host = "", // Disable cache
        .enable_embeddings = false,
        .enable_indexing = false,
        .enable_inference = false,
    };

    const pipeline = try ExtractionPipeline.init(allocator, config);
    defer pipeline.deinit();

    // Submit valid JSON
    const json_content = "{\"key\": \"value\"}";
    const item = try pipeline.submit(json_content, .json);

    try std.testing.expectEqual(@as(u64, 1), pipeline.metrics.items_submitted);
    try std.testing.expectEqual(Stage.extract, item.stage);
    try std.testing.expectEqual(ItemStatus.pending, item.status);

    // Process all
    const processed = try pipeline.processAll();
    try std.testing.expect(processed > 0);
    try std.testing.expectEqual(@as(u64, 1), pipeline.metrics.items_completed);
}

test "ExtractionPipeline - backpressure enforcement" {
    const allocator = std.testing.allocator;

    const config = PipelineConfig{
        .max_queue_size = 2,
        .cache_host = "",
        .enable_embeddings = false,
        .enable_indexing = false,
        .enable_inference = false,
    };

    const pipeline = try ExtractionPipeline.init(allocator, config);
    defer pipeline.deinit();

    // Fill the extract queue
    _ = try pipeline.submit("{}", .json);
    _ = try pipeline.submit("{}", .json);

    // Third should fail with QueueFull
    const result = pipeline.submit("{}", .json);
    try std.testing.expectError(PipelineError.QueueFull, result);
}

test "ExtractionPipeline - setPressure" {
    const allocator = std.testing.allocator;

    const config = PipelineConfig{
        .cache_host = "",
    };

    const pipeline = try ExtractionPipeline.init(allocator, config);
    defer pipeline.deinit();

    pipeline.setPressure(.extract, 50);
    try std.testing.expectEqual(@as(usize, 50), pipeline.queues[0].max_size);

    pipeline.setPressure(.cache, 25);
    try std.testing.expectEqual(@as(usize, 25), pipeline.queues[1].max_size);
}

test "ExtractionPipeline - getMetrics" {
    const allocator = std.testing.allocator;

    const config = PipelineConfig{
        .cache_host = "",
        .enable_embeddings = false,
        .enable_indexing = false,
        .enable_inference = false,
    };

    const pipeline = try ExtractionPipeline.init(allocator, config);
    defer pipeline.deinit();

    _ = try pipeline.submit("{}", .json);

    const metrics = pipeline.getMetrics();
    try std.testing.expectEqual(@as(u64, 1), metrics.items_submitted);
    try std.testing.expectEqual(@as(usize, 1), metrics.stage_metrics[0].queue_depth);
}