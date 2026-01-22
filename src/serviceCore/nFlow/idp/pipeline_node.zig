//! IDP Pipeline Node for nWorkflow
//! Chains all IDP stages: OCR -> Classify -> Extract -> Validate -> Transform

const std = @import("std");
const Allocator = std.mem.Allocator;

const ocr_node = @import("ocr_node.zig");
const classifier_node = @import("classifier_node.zig");
const extractor_node = @import("extractor_node.zig");
const validator_node = @import("validator_node.zig");

// IDP Stage enum
pub const IDPStage = enum {
    OCR,
    CLASSIFY,
    EXTRACT,
    VALIDATE,
    TRANSFORM,

    pub fn toString(self: IDPStage) []const u8 {
        return @tagName(self);
    }

    pub fn fromString(str: []const u8) ?IDPStage {
        inline for (@typeInfo(IDPStage).@"enum".fields) |field| {
            if (std.ascii.eqlIgnoreCase(str, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }
};

// Stage configuration
pub const StageConfig = struct {
    stage: IDPStage,
    enabled: bool = true,
    timeout_ms: u64 = 30000,
    retry_count: u32 = 0,
    on_error: ErrorAction = .CONTINUE,

    pub const ErrorAction = enum {
        STOP,
        CONTINUE,
        SKIP,
    };
};

// IDP Pipeline configuration
pub const IDPPipelineConfig = struct {
    stages: []const StageConfig = &[_]StageConfig{
        .{ .stage = .OCR },
        .{ .stage = .CLASSIFY },
        .{ .stage = .EXTRACT },
        .{ .stage = .VALIDATE },
    },
    parallel_processing: bool = false,
    max_documents: ?u32 = null,
    output_format: OutputFormat = .JSON,

    pub const OutputFormat = enum {
        JSON,
        XML,
        CSV,
    };
};

// Stage result
pub const StageResult = struct {
    stage: IDPStage,
    success: bool,
    processing_time_ms: u64,
    error_message: ?[]const u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, stage: IDPStage, success: bool, time_ms: u64) StageResult {
        return StageResult{
            .stage = stage,
            .success = success,
            .processing_time_ms = time_ms,
            .error_message = null,
            .allocator = allocator,
        };
    }

    pub fn setError(self: *StageResult, msg: []const u8) !void {
        self.error_message = try self.allocator.dupe(u8, msg);
        self.success = false;
    }

    pub fn deinit(self: *StageResult) void {
        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
    }
};

// IDP Result - complete pipeline output
pub const IDPResult = struct {
    document_id: []const u8,
    filename: []const u8,
    success: bool,
    stage_results: std.ArrayList(StageResult),
    total_processing_time_ms: u64,
    allocator: Allocator,

    // Results from each stage
    ocr_text: ?[]const u8 = null,
    document_class: ?classifier_node.DocumentClass = null,
    classification_confidence: f32 = 0.0,
    extracted_entities: std.ArrayList(extractor_node.ExtractedEntity),
    validation_passed: bool = true,
    validation_errors: u32 = 0,

    pub fn init(allocator: Allocator, document_id: []const u8, filename: []const u8) !IDPResult {
        return IDPResult{
            .document_id = try allocator.dupe(u8, document_id),
            .filename = try allocator.dupe(u8, filename),
            .success = true,
            .stage_results = std.ArrayList(StageResult){},
            .total_processing_time_ms = 0,
            .allocator = allocator,
            .extracted_entities = std.ArrayList(extractor_node.ExtractedEntity){},
        };
    }

    pub fn deinit(self: *IDPResult) void {
        self.allocator.free(self.document_id);
        self.allocator.free(self.filename);
        for (self.stage_results.items) |*sr| {
            sr.deinit();
        }
        self.stage_results.deinit(self.allocator);
        if (self.ocr_text) |text| {
            self.allocator.free(text);
        }
        for (self.extracted_entities.items) |*entity| {
            entity.deinit();
        }
        self.extracted_entities.deinit(self.allocator);
    }

    pub fn addStageResult(self: *IDPResult, result: StageResult) !void {
        try self.stage_results.append(self.allocator, result);
        self.total_processing_time_ms += result.processing_time_ms;
        if (!result.success) {
            self.success = false;
        }
    }
};

// IDP Pipeline
pub const IDPPipeline = struct {
    id: []const u8,
    name: []const u8,
    config: IDPPipelineConfig,
    allocator: Allocator,

    // Component nodes
    ocr_node: ?ocr_node.OcrNode = null,
    classifier: ?classifier_node.DocumentClassifier = null,
    extractor: ?extractor_node.EntityExtractor = null,
    validator: ?validator_node.DocumentValidator = null,

    // Stats
    documents_processed: u64 = 0,
    successful_documents: u64 = 0,
    failed_documents: u64 = 0,
    total_processing_time_ms: u64 = 0,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: IDPPipelineConfig) !IDPPipeline {
        var pipeline = IDPPipeline{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .config = config,
            .allocator = allocator,
        };

        // Initialize enabled stages
        for (config.stages) |stage_config| {
            if (!stage_config.enabled) continue;

            switch (stage_config.stage) {
                .OCR => {
                    pipeline.ocr_node = try ocr_node.OcrNode.init(
                        allocator, "ocr", "OCR Node", .{},
                    );
                },
                .CLASSIFY => {
                    pipeline.classifier = try classifier_node.DocumentClassifier.init(
                        allocator, "clf", "Classifier", .{},
                    );
                },
                .EXTRACT => {
                    pipeline.extractor = try extractor_node.EntityExtractor.init(
                        allocator, "ext", "Extractor", .{},
                    );
                },
                .VALIDATE => {
                    pipeline.validator = try validator_node.DocumentValidator.init(
                        allocator, "val", "Validator", .{},
                    );
                },
                .TRANSFORM => {},
            }
        }

        return pipeline;
    }

    pub fn deinit(self: *IDPPipeline) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.ocr_node) |*node| node.deinit();
        if (self.classifier) |*clf| clf.deinit();
        if (self.extractor) |*ext| ext.deinit();
        if (self.validator) |*val| val.deinit();
    }

    pub fn process(self: *IDPPipeline, document_path: []const u8) !IDPResult {
        const start_time = std.time.milliTimestamp();
        const filename = std.fs.path.basename(document_path);

        var id_buf: [64]u8 = undefined;
        const doc_id = std.fmt.bufPrint(&id_buf, "idp-{d}", .{std.time.timestamp()}) catch "idp-0";

        var result = try IDPResult.init(self.allocator, doc_id, filename);
        errdefer result.deinit();

        // Process through each enabled stage
        for (self.config.stages) |stage_config| {
            if (!stage_config.enabled) continue;

            const stage_start = std.time.milliTimestamp();
            var stage_result = StageResult.init(self.allocator, stage_config.stage, true, 0);

            switch (stage_config.stage) {
                .OCR => {
                    if (self.ocr_node) |*node| {
                        var ocr_doc = try node.process(document_path);
                        defer ocr_doc.deinit();
                        result.ocr_text = try ocr_doc.getFullText(self.allocator);
                    }
                },
                .CLASSIFY => {
                    if (self.classifier) |*clf| {
                        if (result.ocr_text) |text| {
                            var clf_result = try clf.classify(text);
                            defer clf_result.deinit();
                            result.document_class = clf_result.document_class;
                            result.classification_confidence = clf_result.confidence;
                        }
                    }
                },
                .EXTRACT => {
                    if (self.extractor) |*ext| {
                        if (result.ocr_text) |text| {
                            var ext_result = try ext.extract(text);
                            defer ext_result.deinit();
                            // Copy entities
                            for (ext_result.entities.items) |entity| {
                                const copy = try extractor_node.ExtractedEntity.init(
                                    self.allocator,
                                    entity.entity_type,
                                    entity.value,
                                    entity.confidence,
                                    entity.position,
                                );
                                try result.extracted_entities.append(self.allocator, copy);
                            }
                        }
                    }
                },
                .VALIDATE => {
                    if (self.validator) |*val| {
                        var data = std.StringHashMap([]const u8).init(self.allocator);
                        defer data.deinit();
                        var val_result = try val.validate(data);
                        defer val_result.deinit();
                        result.validation_passed = val_result.is_valid;
                        result.validation_errors = @intCast(val_result.errors.items.len);
                    }
                },
                .TRANSFORM => {},
            }

            stage_result.processing_time_ms = @intCast(std.time.milliTimestamp() - stage_start);
            try result.addStageResult(stage_result);
        }

        result.total_processing_time_ms = @intCast(std.time.milliTimestamp() - start_time);

        // Update stats
        self.documents_processed += 1;
        if (result.success) {
            self.successful_documents += 1;
        } else {
            self.failed_documents += 1;
        }
        self.total_processing_time_ms += result.total_processing_time_ms;

        return result;
    }

    pub fn getStats(self: *const IDPPipeline) PipelineStats {
        return PipelineStats{
            .documents_processed = self.documents_processed,
            .successful_documents = self.successful_documents,
            .failed_documents = self.failed_documents,
            .total_processing_time_ms = self.total_processing_time_ms,
        };
    }
};

pub const PipelineStats = struct {
    documents_processed: u64,
    successful_documents: u64,
    failed_documents: u64,
    total_processing_time_ms: u64,
};

// Tests
test "IDPStage operations" {
    try std.testing.expectEqualStrings("OCR", IDPStage.OCR.toString());
    try std.testing.expectEqual(IDPStage.CLASSIFY, IDPStage.fromString("CLASSIFY").?);
    try std.testing.expectEqual(@as(?IDPStage, null), IDPStage.fromString("unknown"));
}

test "StageResult initialization" {
    const allocator = std.testing.allocator;

    var result = StageResult.init(allocator, .OCR, true, 100);
    defer result.deinit();

    try std.testing.expectEqual(IDPStage.OCR, result.stage);
    try std.testing.expect(result.success);
}

test "IDPResult initialization" {
    const allocator = std.testing.allocator;

    var result = try IDPResult.init(allocator, "doc-1", "test.pdf");
    defer result.deinit();

    try std.testing.expectEqualStrings("doc-1", result.document_id);
    try std.testing.expect(result.success);
}

test "IDPPipeline initialization" {
    const allocator = std.testing.allocator;

    var pipeline = try IDPPipeline.init(allocator, "pipe-1", "Test Pipeline", .{});
    defer pipeline.deinit();

    try std.testing.expectEqualStrings("pipe-1", pipeline.id);
    try std.testing.expect(pipeline.ocr_node != null);
}

test "IDPPipelineConfig defaults" {
    const config = IDPPipelineConfig{};
    try std.testing.expectEqual(@as(usize, 4), config.stages.len);
    try std.testing.expect(!config.parallel_processing);
}
