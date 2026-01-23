const std = @import("std");
const Allocator = std.mem.Allocator;
const data_packet = @import("data_packet.zig");
const data_stream = @import("data_stream.zig");
const DataPacket = data_packet.DataPacket;
const DataStream = data_stream.DataStream;

/// Pipeline stage function type
pub const StageFunction = *const fn (Allocator, *DataPacket) anyerror!*DataPacket;

/// Pipeline stage with transformation logic
pub const PipelineStage = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    transform: StageFunction,
    error_handler: ?ErrorHandler,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, transform: StageFunction) !*PipelineStage {
        const stage = try allocator.create(PipelineStage);
        errdefer allocator.destroy(stage);
        
        stage.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .transform = transform,
            .error_handler = null,
        };
        
        return stage;
    }
    
    pub fn deinit(self: *PipelineStage) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.allocator.destroy(self);
    }
    
    /// Execute the stage transformation
    pub fn execute(self: *PipelineStage, packet: *DataPacket) !*DataPacket {
        return self.transform(self.allocator, packet) catch |err| {
            if (self.error_handler) |handler| {
                return handler(self.allocator, packet, err);
            }
            return err;
        };
    }
    
    /// Set error handler for this stage
    pub fn setErrorHandler(self: *PipelineStage, handler: ErrorHandler) void {
        self.error_handler = handler;
    }
};

/// Error handler function type
pub const ErrorHandler = *const fn (Allocator, *DataPacket, anyerror) anyerror!*DataPacket;

/// Data transformation pipeline
pub const DataPipeline = struct {
    allocator: Allocator,
    id: []const u8,
    name: []const u8,
    stages: std.ArrayList(*PipelineStage),
    metrics: PipelineMetrics,
    is_parallel: bool,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8) !*DataPipeline {
        const pipeline = try allocator.create(DataPipeline);
        errdefer allocator.destroy(pipeline);
        
        pipeline.* = .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .stages = std.ArrayList(*PipelineStage){},
            .metrics = PipelineMetrics.init(),
            .is_parallel = false,
        };
        
        return pipeline;
    }
    
    pub fn deinit(self: *DataPipeline) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        
        for (self.stages.items) |stage| {
            stage.deinit();
        }
        self.stages.deinit(self.allocator);
        
        self.allocator.destroy(self);
    }
    
    /// Add a stage to the pipeline
    pub fn addStage(self: *DataPipeline, stage: *PipelineStage) !void {
        try self.stages.append(self.allocator, stage);
    }
    
    /// Execute the pipeline on a packet
    pub fn execute(self: *DataPipeline, input: *DataPacket) !*DataPacket {
        const start_time = std.time.milliTimestamp();
        
        var current = input;
        var stage_index: usize = 0;
        
        errdefer {
            self.metrics.failed_executions += 1;
            self.metrics.last_error_stage = stage_index;
        }
        
        for (self.stages.items, 0..) |stage, i| {
            stage_index = i;
            const stage_start = std.time.milliTimestamp();
            
            current = try stage.execute(current);
            
            const stage_duration = std.time.milliTimestamp() - stage_start;
            self.metrics.stage_durations[i] = stage_duration;
        }
        
        const total_duration = std.time.milliTimestamp() - start_time;
        self.metrics.successful_executions += 1;
        self.metrics.total_execution_time += total_duration;
        self.metrics.last_execution_time = total_duration;
        
        return current;
    }
    
    /// Get pipeline metrics
    pub fn getMetrics(self: *const DataPipeline) PipelineMetrics {
        return self.metrics;
    }
    
    /// Reset metrics
    pub fn resetMetrics(self: *DataPipeline) void {
        self.metrics = PipelineMetrics.init();
    }
};

/// Pipeline execution metrics
pub const PipelineMetrics = struct {
    successful_executions: u64,
    failed_executions: u64,
    total_execution_time: i64,
    last_execution_time: i64,
    stage_durations: [10]i64, // Max 10 stages tracked
    last_error_stage: usize,
    
    pub fn init() PipelineMetrics {
        return .{
            .successful_executions = 0,
            .failed_executions = 0,
            .total_execution_time = 0,
            .last_execution_time = 0,
            .stage_durations = [_]i64{0} ** 10,
            .last_error_stage = 0,
        };
    }
    
    pub fn getAverageExecutionTime(self: *const PipelineMetrics) i64 {
        if (self.successful_executions == 0) return 0;
        return @divTrunc(self.total_execution_time, @as(i64, @intCast(self.successful_executions)));
    }
    
    pub fn getSuccessRate(self: *const PipelineMetrics) f64 {
        const total = self.successful_executions + self.failed_executions;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.successful_executions)) / @as(f64, @floatFromInt(total));
    }
};

/// Common transformation functions
pub const Transformations = struct {
    /// Map transformation - apply function to packet value
    pub fn map(allocator: Allocator, packet: *DataPacket, mapper: *const fn (std.json.Value) anyerror!std.json.Value) !*DataPacket {
        const new_value = try mapper(packet.value);
        const new_packet = try DataPacket.init(allocator, packet.id, packet.data_type, new_value);
        
        // Copy metadata
        var it = packet.metadata.iterator();
        while (it.next()) |entry| {
            try new_packet.setMetadata(entry.key_ptr.*, entry.value_ptr.*);
        }
        
        return new_packet;
    }
    
    /// Filter transformation - only pass packets matching predicate
    pub fn filter(allocator: Allocator, packet: *DataPacket, predicate: *const fn (*DataPacket) bool) !?*DataPacket {
        _ = allocator;
        if (predicate(packet)) {
            return packet;
        }
        return null;
    }
    
    /// Validate transformation - ensure packet meets criteria
    pub fn validate(allocator: Allocator, packet: *DataPacket, schema: *const data_packet.DataSchema) !*DataPacket {
        _ = allocator;
        try packet.validate(schema);
        return packet;
    }
};

/// Pipeline builder for fluent API
pub const PipelineBuilder = struct {
    allocator: Allocator,
    pipeline: *DataPipeline,
    
    pub fn init(allocator: Allocator, id: []const u8, name: []const u8) !PipelineBuilder {
        const pipeline = try DataPipeline.init(allocator, id, name);
        return .{
            .allocator = allocator,
            .pipeline = pipeline,
        };
    }
    
    pub fn addStage(self: *PipelineBuilder, id: []const u8, name: []const u8, transform: StageFunction) !*PipelineBuilder {
        const stage = try PipelineStage.init(self.allocator, id, name, transform);
        try self.pipeline.addStage(stage);
        return self;
    }
    
    pub fn build(self: *PipelineBuilder) *DataPipeline {
        return self.pipeline;
    }
};

/// Parallel pipeline executor for high-throughput scenarios
pub const ParallelExecutor = struct {
    allocator: Allocator,
    pipeline: *DataPipeline,
    worker_count: usize,
    
    pub fn init(allocator: Allocator, pipeline: *DataPipeline, worker_count: usize) !*ParallelExecutor {
        const executor = try allocator.create(ParallelExecutor);
        executor.* = .{
            .allocator = allocator,
            .pipeline = pipeline,
            .worker_count = worker_count,
        };
        return executor;
    }
    
    pub fn deinit(self: *ParallelExecutor) void {
        self.allocator.destroy(self);
    }
    
    /// Execute pipeline on multiple packets in parallel
    pub fn executeBatch(self: *ParallelExecutor, packets: []const *DataPacket) !std.ArrayList(*DataPacket) {
        var results = std.ArrayList(*DataPacket){};
        errdefer results.deinit(self.allocator);
        
        // For now, sequential processing (true parallelism requires threading)
        // This is a placeholder for future async/parallel implementation
        for (packets) |packet| {
            const result = try self.pipeline.execute(packet);
            try results.append(self.allocator, result);
        }
        
        return results;
    }
};

/// Stream processor for continuous data processing
pub const StreamProcessor = struct {
    allocator: Allocator,
    input_stream: *DataStream,
    output_stream: *DataStream,
    pipeline: *DataPipeline,
    is_running: bool,
    
    pub fn init(allocator: Allocator, input: *DataStream, output: *DataStream, pipeline: *DataPipeline) !*StreamProcessor {
        const processor = try allocator.create(StreamProcessor);
        processor.* = .{
            .allocator = allocator,
            .input_stream = input,
            .output_stream = output,
            .pipeline = pipeline,
            .is_running = false,
        };
        return processor;
    }
    
    pub fn deinit(self: *StreamProcessor) void {
        self.allocator.destroy(self);
    }
    
    /// Start processing stream
    pub fn start(self: *StreamProcessor) !void {
        self.is_running = true;
        
        // Process packets from input stream
        while (self.is_running) {
            if (self.input_stream.pull()) |packet| {
                const result = try self.pipeline.execute(packet);
                try self.output_stream.push(result);
            } else {
                // No more packets, exit
                break;
            }
        }
    }
    
    /// Stop processing
    pub fn stop(self: *StreamProcessor) void {
        self.is_running = false;
    }
};

// ============================================================================
// TESTS
// ============================================================================

fn testTransformDouble(allocator: Allocator, packet: *DataPacket) !*DataPacket {
    _ = allocator;
    if (packet.value == .integer) {
        const doubled = packet.value.integer * 2;
        // Modify in place instead of creating new packet
        packet.value = std.json.Value{ .integer = doubled };
        return packet;
    }
    return packet;
}

fn testTransformAddPrefix(allocator: Allocator, packet: *DataPacket) !*DataPacket {
    if (packet.value == .string) {
        const original = packet.value.string;
        const prefixed = try std.fmt.allocPrint(allocator, "PREFIX_{s}", .{original});
        const new_value = std.json.Value{ .string = prefixed };
        const new_packet = try DataPacket.init(allocator, packet.id, .string, new_value);
        return new_packet;
    }
    return packet;
}

test "PipelineStage creation and execution" {
    const allocator = std.testing.allocator;
    
    const stage = try PipelineStage.init(allocator, "stage-1", "Double Numbers", testTransformDouble);
    defer stage.deinit();
    
    const value = std.json.Value{ .integer = 5 };
    const packet = try DataPacket.init(allocator, "p1", .number, value);
    defer packet.deinit();
    
    const result = try stage.execute(packet);
    // Result is the same packet, don't deinit twice
    
    try std.testing.expectEqual(@as(i64, 10), result.value.integer);
}

test "DataPipeline creation and single stage" {
    const allocator = std.testing.allocator;
    
    const pipeline = try DataPipeline.init(allocator, "pipeline-1", "Test Pipeline");
    defer pipeline.deinit();
    
    const stage = try PipelineStage.init(allocator, "stage-1", "Double", testTransformDouble);
    try pipeline.addStage(stage);
    
    const value = std.json.Value{ .integer = 3 };
    const packet = try DataPacket.init(allocator, "p1", .number, value);
    defer packet.deinit();
    
    const result = try pipeline.execute(packet);
    // Result is the same packet, don't deinit twice
    
    try std.testing.expectEqual(@as(i64, 6), result.value.integer);
}

test "DataPipeline multi-stage transformation" {
    const allocator = std.testing.allocator;
    
    const pipeline = try DataPipeline.init(allocator, "pipeline-1", "Multi-Stage");
    defer pipeline.deinit();
    
    const stage1 = try PipelineStage.init(allocator, "stage-1", "Double", testTransformDouble);
    try pipeline.addStage(stage1);
    
    const stage2 = try PipelineStage.init(allocator, "stage-2", "Double Again", testTransformDouble);
    try pipeline.addStage(stage2);
    
    const value = std.json.Value{ .integer = 2 };
    const packet = try DataPacket.init(allocator, "p1", .number, value);
    defer packet.deinit();
    
    const result = try pipeline.execute(packet);
    // Result is the same packet, don't deinit twice
    
    // 2 * 2 * 2 = 8
    try std.testing.expectEqual(@as(i64, 8), result.value.integer);
}

test "PipelineMetrics tracking" {
    const allocator = std.testing.allocator;
    
    const pipeline = try DataPipeline.init(allocator, "pipeline-1", "Metrics Test");
    defer pipeline.deinit();
    
    const stage = try PipelineStage.init(allocator, "stage-1", "Double", testTransformDouble);
    try pipeline.addStage(stage);
    
    const value1 = std.json.Value{ .integer = 1 };
    const packet1 = try DataPacket.init(allocator, "p1", .number, value1);
    defer packet1.deinit();
    
    _ = try pipeline.execute(packet1);
    // Result is the same packet, don't deinit twice
    
    const value2 = std.json.Value{ .integer = 2 };
    const packet2 = try DataPacket.init(allocator, "p2", .number, value2);
    defer packet2.deinit();
    
    _ = try pipeline.execute(packet2);
    // Result is the same packet, don't deinit twice
    
    const metrics = pipeline.getMetrics();
    try std.testing.expectEqual(@as(u64, 2), metrics.successful_executions);
    try std.testing.expectEqual(@as(u64, 0), metrics.failed_executions);
    try std.testing.expectEqual(@as(f64, 1.0), metrics.getSuccessRate());
}

test "PipelineBuilder fluent API" {
    const allocator = std.testing.allocator;
    
    var builder = try PipelineBuilder.init(allocator, "pipeline-1", "Fluent Test");
    _ = try builder.addStage("stage-1", "Double", testTransformDouble);
    _ = try builder.addStage("stage-2", "Double Again", testTransformDouble);
    
    const pipeline = builder.build();
    defer pipeline.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), pipeline.stages.items.len);
}

test "ParallelExecutor batch processing" {
    const allocator = std.testing.allocator;
    
    const pipeline = try DataPipeline.init(allocator, "pipeline-1", "Parallel Test");
    defer pipeline.deinit();
    
    const stage = try PipelineStage.init(allocator, "stage-1", "Double", testTransformDouble);
    try pipeline.addStage(stage);
    
    const executor = try ParallelExecutor.init(allocator, pipeline, 4);
    defer executor.deinit();
    
    var packets: [3]*DataPacket = undefined;
    for (0..3) |i| {
        const value = std.json.Value{ .integer = @as(i64, @intCast(i + 1)) };
        packets[i] = try DataPacket.init(allocator, "p", .number, value);
    }
    defer for (packets) |p| p.deinit();
    
    var results = try executor.executeBatch(&packets);
    defer results.deinit(allocator);
    // Results are the same packets, don't deinit them again
    
    try std.testing.expectEqual(@as(usize, 3), results.items.len);
    try std.testing.expectEqual(@as(i64, 2), results.items[0].value.integer);
    try std.testing.expectEqual(@as(i64, 4), results.items[1].value.integer);
    try std.testing.expectEqual(@as(i64, 6), results.items[2].value.integer);
}

test "StreamProcessor integration" {
    const allocator = std.testing.allocator;
    
    const input_stream = try DataStream.init(allocator, "input", .pull, 10);
    defer input_stream.deinit();
    
    const output_stream = try DataStream.init(allocator, "output", .pull, 10);
    defer output_stream.deinit();
    
    const pipeline = try DataPipeline.init(allocator, "pipeline-1", "Stream Test");
    defer pipeline.deinit();
    
    const stage = try PipelineStage.init(allocator, "stage-1", "Double", testTransformDouble);
    try pipeline.addStage(stage);
    
    const processor = try StreamProcessor.init(allocator, input_stream, output_stream, pipeline);
    defer processor.deinit();
    
    // Add packets to input stream
    const value1 = std.json.Value{ .integer = 1 };
    const packet1 = try DataPacket.init(allocator, "p1", .number, value1);
    defer packet1.deinit();
    try input_stream.push(packet1);

    const value2 = std.json.Value{ .integer = 2 };
    const packet2 = try DataPacket.init(allocator, "p2", .number, value2);
    defer packet2.deinit();
    try input_stream.push(packet2);
    
    // Process stream
    try processor.start();
    
    // Check output stream
    const result1 = output_stream.pull();
    try std.testing.expect(result1 != null);
    try std.testing.expectEqual(@as(i64, 2), result1.?.value.integer);
    
    const result2 = output_stream.pull();
    try std.testing.expect(result2 != null);
    try std.testing.expectEqual(@as(i64, 4), result2.?.value.integer);
}
