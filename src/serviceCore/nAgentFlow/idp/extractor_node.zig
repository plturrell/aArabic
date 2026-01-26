//! Entity Extractor Node for nWorkflow IDP
//! Provides entity extraction using patterns, rules, or LLM-based approaches

const std = @import("std");
const Allocator = std.mem.Allocator;

// Entity type enum
pub const EntityType = enum {
    PERSON,
    ORGANIZATION,
    DATE,
    MONEY,
    ADDRESS,
    EMAIL,
    PHONE,
    URL,
    INVOICE_NUMBER,
    PO_NUMBER,
    ACCOUNT_NUMBER,
    TAX_ID,
    PERCENTAGE,
    QUANTITY,
    CUSTOM,

    pub fn toString(self: EntityType) []const u8 {
        return @tagName(self);
    }

    pub fn fromString(str: []const u8) ?EntityType {
        inline for (@typeInfo(EntityType).@"enum".fields) |field| {
            if (std.ascii.eqlIgnoreCase(str, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }
};

// Position in text
pub const TextPosition = struct {
    start: usize,
    end: usize,
    line: u32,
    column: u32,

    pub fn length(self: TextPosition) usize {
        return self.end - self.start;
    }
};

// Extracted entity
pub const ExtractedEntity = struct {
    entity_type: EntityType,
    value: []const u8,
    normalized_value: ?[]const u8,
    confidence: f32,
    position: TextPosition,
    context: ?[]const u8,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        entity_type: EntityType,
        value: []const u8,
        confidence: f32,
        position: TextPosition,
    ) !ExtractedEntity {
        return ExtractedEntity{
            .entity_type = entity_type,
            .value = try allocator.dupe(u8, value),
            .normalized_value = null,
            .confidence = confidence,
            .position = position,
            .context = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ExtractedEntity) void {
        self.allocator.free(self.value);
        if (self.normalized_value) |nv| {
            self.allocator.free(nv);
        }
        if (self.context) |ctx| {
            self.allocator.free(ctx);
        }
    }

    pub fn setNormalizedValue(self: *ExtractedEntity, value: []const u8) !void {
        if (self.normalized_value) |nv| {
            self.allocator.free(nv);
        }
        self.normalized_value = try self.allocator.dupe(u8, value);
    }

    pub fn setContext(self: *ExtractedEntity, ctx: []const u8) !void {
        if (self.context) |c| {
            self.allocator.free(c);
        }
        self.context = try self.allocator.dupe(u8, ctx);
    }

    pub fn isHighConfidence(self: *const ExtractedEntity) bool {
        return self.confidence >= 0.8;
    }

    pub fn toJson(self: *const ExtractedEntity, allocator: Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit();
        var writer = buffer.writer();
        try writer.print(
            \\{{"type":"{s}","value":"{s}","confidence":{d:.4},"start":{d},"end":{d}}}
        , .{
            self.entity_type.toString(),
            self.value,
            self.confidence,
            self.position.start,
            self.position.end,
        });
        return buffer.toOwnedSlice();
    }
};

// Extraction result
pub const ExtractionResult = struct {
    entities: std.ArrayList(ExtractedEntity),
    processing_time_ms: u64,
    source_text_length: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ExtractionResult {
        return ExtractionResult{
            .entities = std.ArrayList(ExtractedEntity){},
            .processing_time_ms = 0,
            .source_text_length = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ExtractionResult) void {
        for (self.entities.items) |*entity| {
            entity.deinit();
        }
        self.entities.deinit(self.allocator);
    }

    pub fn addEntity(self: *ExtractionResult, entity: ExtractedEntity) !void {
        try self.entities.append(self.allocator, entity);
    }

    pub fn getEntityCount(self: *const ExtractionResult) usize {
        return self.entities.items.len;
    }
};

// Extraction rule for pattern-based extraction
pub const ExtractionRule = struct {
    name: []const u8,
    entity_type: EntityType,
    pattern: []const u8,
    prefix: ?[]const u8,
    suffix: ?[]const u8,
    normalize: bool,

    pub fn matches(self: *const ExtractionRule, text: []const u8, start_pos: usize) ?struct { start: usize, end: usize } {
        var search_start = start_pos;
        if (self.prefix) |prefix| {
            if (std.mem.indexOf(u8, text[search_start..], prefix)) |idx| {
                search_start += idx + prefix.len;
            } else {
                return null;
            }
        }

        if (std.mem.indexOf(u8, text[search_start..], self.pattern)) |idx| {
            const match_start = search_start + idx;
            const match_end = match_start + self.pattern.len;

            if (self.suffix) |suffix| {
                if (!std.mem.startsWith(u8, text[match_end..], suffix)) {
                    return null;
                }
            }
            return .{ .start = match_start, .end = match_end };
        }
        return null;
    }
};

// Pattern extractor configuration
pub const PatternConfig = struct {
    email_pattern: bool = true,
    phone_pattern: bool = true,
    date_pattern: bool = true,
    money_pattern: bool = true,
    url_pattern: bool = true,
    custom_rules: []const ExtractionRule = &[_]ExtractionRule{},
};

fn isEmailChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-' or c == '+';
}

// Pattern-based extractor
pub const PatternExtractor = struct {
    config: PatternConfig,
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: PatternConfig) PatternExtractor {
        return PatternExtractor{
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn extract(self: *const PatternExtractor, text: []const u8) !ExtractionResult {
        var result = ExtractionResult.init(self.allocator);
        result.source_text_length = text.len;

        if (self.config.email_pattern) {
            try self.extractEmails(text, &result);
        }
        if (self.config.phone_pattern) {
            try self.extractPhones(text, &result);
        }
        if (self.config.money_pattern) {
            try self.extractMoney(text, &result);
        }
        return result;
    }

    fn extractEmails(self: *const PatternExtractor, text: []const u8, result: *ExtractionResult) !void {
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            if (text[i] == '@' and i > 0 and i < text.len - 1) {
                var start = i;
                while (start > 0 and isEmailChar(text[start - 1])) {
                    start -= 1;
                }
                var end = i + 1;
                while (end < text.len and isEmailChar(text[end])) {
                    end += 1;
                }
                if (end > i + 1 and start < i) {
                    const email = text[start..end];
                    const entity = try ExtractedEntity.init(
                        self.allocator, .EMAIL, email, 0.9,
                        .{ .start = start, .end = end, .line = 0, .column = 0 },
                    );
                    try result.addEntity(entity);
                    i = end;
                }
            }
        }
    }

    fn extractPhones(self: *const PatternExtractor, text: []const u8, result: *ExtractionResult) !void {
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            if (std.ascii.isDigit(text[i]) or text[i] == '+') {
                const start = i;
                var digit_count: usize = 0;
                var end = i;
                while (end < text.len and (std.ascii.isDigit(text[end]) or
                    text[end] == '-' or text[end] == ' ' or
                    text[end] == '(' or text[end] == ')' or text[end] == '+'))
                {
                    if (std.ascii.isDigit(text[end])) digit_count += 1;
                    end += 1;
                }
                if (digit_count >= 7 and digit_count <= 15) {
                    const phone = text[start..end];
                    const entity = try ExtractedEntity.init(
                        self.allocator, .PHONE, phone, 0.85,
                        .{ .start = start, .end = end, .line = 0, .column = 0 },
                    );
                    try result.addEntity(entity);
                }
                i = end;
            }
        }
    }

    fn extractMoney(self: *const PatternExtractor, text: []const u8, result: *ExtractionResult) !void {
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            if (text[i] == '$' or text[i] == '€' or text[i] == '£') {
                const start = i;
                var end = i + 1;
                while (end < text.len and text[end] == ' ') {
                    end += 1;
                }
                while (end < text.len and (std.ascii.isDigit(text[end]) or
                    text[end] == '.' or text[end] == ','))
                {
                    end += 1;
                }
                if (end > start + 1) {
                    const money = text[start..end];
                    const entity = try ExtractedEntity.init(
                        self.allocator, .MONEY, money, 0.9,
                        .{ .start = start, .end = end, .line = 0, .column = 0 },
                    );
                    try result.addEntity(entity);
                }
                i = end;
            }
        }
    }
};

// Extractor configuration
pub const ExtractorConfig = struct {
    use_patterns: bool = true,
    use_llm: bool = false,
    pattern_config: PatternConfig = .{},
    llm_endpoint: ?[]const u8 = null,
    llm_api_key: ?[]const u8 = null,
    min_confidence: f32 = 0.5,
};

// Entity Extractor Node
pub const EntityExtractor = struct {
    id: []const u8,
    name: []const u8,
    config: ExtractorConfig,
    pattern_extractor: PatternExtractor,
    allocator: Allocator,

    // Stats
    documents_processed: u64 = 0,
    total_entities_extracted: u64 = 0,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: ExtractorConfig) !EntityExtractor {
        return EntityExtractor{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .config = config,
            .pattern_extractor = PatternExtractor.init(allocator, config.pattern_config),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *EntityExtractor) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
    }

    pub fn extract(self: *EntityExtractor, text: []const u8) !ExtractionResult {
        const start_time = std.time.milliTimestamp();

        var result = if (self.config.use_patterns)
            try self.pattern_extractor.extract(text)
        else
            ExtractionResult.init(self.allocator);

        result.processing_time_ms = @intCast(std.time.milliTimestamp() - start_time);

        // Update stats
        self.documents_processed += 1;
        self.total_entities_extracted += result.getEntityCount();

        return result;
    }

    pub fn getStats(self: *const EntityExtractor) ExtractorStats {
        return ExtractorStats{
            .documents_processed = self.documents_processed,
            .total_entities_extracted = self.total_entities_extracted,
        };
    }
};

pub const ExtractorStats = struct {
    documents_processed: u64,
    total_entities_extracted: u64,
};

// Tests
test "EntityType operations" {
    try std.testing.expectEqualStrings("EMAIL", EntityType.EMAIL.toString());
    try std.testing.expectEqual(EntityType.PHONE, EntityType.fromString("PHONE").?);
    try std.testing.expectEqual(@as(?EntityType, null), EntityType.fromString("unknown"));
}

test "TextPosition length" {
    const pos = TextPosition{ .start = 10, .end = 25, .line = 1, .column = 10 };
    try std.testing.expectEqual(@as(usize, 15), pos.length());
}

test "ExtractedEntity initialization" {
    const allocator = std.testing.allocator;

    var entity = try ExtractedEntity.init(
        allocator, .EMAIL, "test@example.com", 0.95,
        .{ .start = 0, .end = 16, .line = 1, .column = 1 },
    );
    defer entity.deinit();

    try std.testing.expectEqualStrings("test@example.com", entity.value);
    try std.testing.expect(entity.isHighConfidence());
}

test "ExtractionResult operations" {
    const allocator = std.testing.allocator;

    var result = ExtractionResult.init(allocator);
    defer result.deinit();

    const entity = try ExtractedEntity.init(
        allocator, .EMAIL, "test@example.com", 0.9,
        .{ .start = 0, .end = 16, .line = 1, .column = 1 },
    );
    try result.addEntity(entity);

    try std.testing.expectEqual(@as(usize, 1), result.getEntityCount());
}

test "PatternExtractor email extraction" {
    const allocator = std.testing.allocator;

    const extractor = PatternExtractor.init(allocator, .{});
    var result = try extractor.extract("Contact us at info@company.com for more info.");
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.getEntityCount());
    try std.testing.expectEqual(EntityType.EMAIL, result.entities.items[0].entity_type);
}

test "PatternExtractor money extraction" {
    const allocator = std.testing.allocator;

    const extractor = PatternExtractor.init(allocator, .{});
    var result = try extractor.extract("The total is $1,234.56 due today.");
    defer result.deinit();

    try std.testing.expect(result.getEntityCount() >= 1);
}

test "EntityExtractor initialization" {
    const allocator = std.testing.allocator;

    var extractor = try EntityExtractor.init(allocator, "ext-1", "Entity Extractor", .{});
    defer extractor.deinit();

    try std.testing.expectEqualStrings("ext-1", extractor.id);
}
