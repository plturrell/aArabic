//! Document Classifier Node for nWorkflow IDP
//! Provides document classification using rules, ML, or LLM-based approaches

const std = @import("std");
const Allocator = std.mem.Allocator;

// Document class enum
pub const DocumentClass = enum {
    INVOICE,
    CONTRACT,
    RECEIPT,
    FORM,
    LETTER,
    REPORT,
    ID_DOCUMENT,
    BANK_STATEMENT,
    TAX_DOCUMENT,
    MEDICAL_RECORD,
    LEGAL_DOCUMENT,
    RESUME,
    OTHER,

    pub fn toString(self: DocumentClass) []const u8 {
        return @tagName(self);
    }

    pub fn fromString(str: []const u8) ?DocumentClass {
        inline for (@typeInfo(DocumentClass).@"enum".fields) |field| {
            if (std.ascii.eqlIgnoreCase(str, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }

    pub fn getDescription(self: DocumentClass) []const u8 {
        return switch (self) {
            .INVOICE => "Commercial invoice or bill",
            .CONTRACT => "Legal agreement or contract",
            .RECEIPT => "Payment receipt or proof of purchase",
            .FORM => "Fillable form or application",
            .LETTER => "Correspondence or formal letter",
            .REPORT => "Business or technical report",
            .ID_DOCUMENT => "Identity document (passport, ID card, etc.)",
            .BANK_STATEMENT => "Bank or financial statement",
            .TAX_DOCUMENT => "Tax-related document",
            .MEDICAL_RECORD => "Medical or health record",
            .LEGAL_DOCUMENT => "Legal filing or court document",
            .RESUME => "CV or resume",
            .OTHER => "Unclassified document",
        };
    }
};

// Classification result
pub const ClassificationResult = struct {
    document_class: DocumentClass,
    confidence: f32,
    alternatives: std.ArrayList(ClassAlternative),
    processing_time_ms: u64,
    model_used: []const u8,
    allocator: Allocator,

    pub const ClassAlternative = struct {
        document_class: DocumentClass,
        confidence: f32,
    };

    pub fn init(allocator: Allocator, doc_class: DocumentClass, confidence: f32, model: []const u8) !ClassificationResult {
        return ClassificationResult{
            .document_class = doc_class,
            .confidence = confidence,
            .alternatives = std.ArrayList(ClassAlternative){},
            .processing_time_ms = 0,
            .model_used = try allocator.dupe(u8, model),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ClassificationResult) void {
        self.alternatives.deinit(self.allocator);
        self.allocator.free(self.model_used);
    }

    pub fn addAlternative(self: *ClassificationResult, doc_class: DocumentClass, confidence: f32) !void {
        try self.alternatives.append(self.allocator, .{
            .document_class = doc_class,
            .confidence = confidence,
        });
    }

    pub fn isHighConfidence(self: *const ClassificationResult) bool {
        return self.confidence >= 0.8;
    }

    pub fn toJson(self: *const ClassificationResult, allocator: Allocator) ![]const u8 {
        var buffer: std.ArrayList(u8) = .{};
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);
        try writer.print(
            \\{{"class":"{s}","confidence":{d:.4},"model":"{s}","alternatives":[
        , .{
            self.document_class.toString(),
            self.confidence,
            self.model_used,
        });

        for (self.alternatives.items, 0..) |alt, i| {
            if (i > 0) try writer.writeByte(',');
            try writer.print(
                \\{{"class":"{s}","confidence":{d:.4}}}
            , .{ alt.document_class.toString(), alt.confidence });
        }
        try writer.writeAll("]}");
        return buffer.toOwnedSlice(allocator);
    }
};

// Classification model type
pub const ClassificationModel = enum {
    RULE_BASED, // Keyword and pattern matching
    ML_BASED, // Machine learning model
    LLM_BASED, // Large language model

    pub fn toString(self: ClassificationModel) []const u8 {
        return @tagName(self);
    }
};

// Keyword rule for rule-based classification
pub const KeywordRule = struct {
    keywords: []const []const u8,
    document_class: DocumentClass,
    weight: f32,
    case_sensitive: bool,

    pub fn matches(self: *const KeywordRule, text: []const u8) f32 {
        var match_count: f32 = 0;
        for (self.keywords) |keyword| {
            if (self.case_sensitive) {
                if (std.mem.indexOf(u8, text, keyword) != null) {
                    match_count += 1;
                }
            } else {
                if (containsIgnoreCase(text, keyword)) {
                    match_count += 1;
                }
            }
        }
        const keyword_count = @as(f32, @floatFromInt(self.keywords.len));
        return if (keyword_count > 0) (match_count / keyword_count) * self.weight else 0;
    }
};

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    if (needle.len == 0) return true;

    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        var match = true;
        for (needle, 0..) |c, j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(c)) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// Classifier configuration
pub const ClassifierConfig = struct {
    model: ClassificationModel = .RULE_BASED,
    min_confidence: f32 = 0.5,
    max_alternatives: u32 = 3,
    timeout_ms: u64 = 10000,

    // LLM settings
    llm_endpoint: ?[]const u8 = null,
    llm_api_key: ?[]const u8 = null,
    llm_model: ?[]const u8 = null,

    // ML settings
    ml_model_path: ?[]const u8 = null,
};

// Rule-based classifier
pub const RuleBasedClassifier = struct {
    rules: std.ArrayList(KeywordRule),
    allocator: Allocator,

    pub fn init(allocator: Allocator) RuleBasedClassifier {
        return RuleBasedClassifier{
            .rules = std.ArrayList(KeywordRule){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RuleBasedClassifier) void {
        self.rules.deinit(self.allocator);
    }

    pub fn addRule(self: *RuleBasedClassifier, rule: KeywordRule) !void {
        try self.rules.append(self.allocator, rule);
    }

    pub fn addDefaultRules(self: *RuleBasedClassifier) !void {
        // Invoice rules
        try self.addRule(.{
            .keywords = &[_][]const u8{ "invoice", "bill to", "amount due", "payment terms", "subtotal", "total" },
            .document_class = .INVOICE,
            .weight = 1.0,
            .case_sensitive = false,
        });

        // Contract rules
        try self.addRule(.{
            .keywords = &[_][]const u8{ "agreement", "contract", "parties", "whereas", "hereby", "terms and conditions" },
            .document_class = .CONTRACT,
            .weight = 1.0,
            .case_sensitive = false,
        });

        // Receipt rules
        try self.addRule(.{
            .keywords = &[_][]const u8{ "receipt", "paid", "thank you", "transaction", "change" },
            .document_class = .RECEIPT,
            .weight = 1.0,
            .case_sensitive = false,
        });

        // Form rules
        try self.addRule(.{
            .keywords = &[_][]const u8{ "form", "please fill", "signature", "date of birth", "applicant" },
            .document_class = .FORM,
            .weight = 1.0,
            .case_sensitive = false,
        });

        // ID Document rules
        try self.addRule(.{
            .keywords = &[_][]const u8{ "passport", "driver license", "identification", "id card", "nationality" },
            .document_class = .ID_DOCUMENT,
            .weight = 1.0,
            .case_sensitive = false,
        });
    }

    pub fn classify(self: *const RuleBasedClassifier, text: []const u8) !ClassificationResult {
        var scores = std.AutoHashMap(DocumentClass, f32).init(self.allocator);
        defer scores.deinit();

        // Calculate scores for each rule
        for (self.rules.items) |rule| {
            const score = rule.matches(text);
            if (score > 0) {
                const current = scores.get(rule.document_class) orelse 0;
                try scores.put(rule.document_class, current + score);
            }
        }

        // Find best match
        var best_class: DocumentClass = .OTHER;
        var best_score: f32 = 0;
        var total_score: f32 = 0;

        var iter = scores.iterator();
        while (iter.next()) |entry| {
            total_score += entry.value_ptr.*;
            if (entry.value_ptr.* > best_score) {
                best_score = entry.value_ptr.*;
                best_class = entry.key_ptr.*;
            }
        }

        // Normalize confidence
        const confidence = if (total_score > 0) best_score / total_score else 0;

        var result = try ClassificationResult.init(self.allocator, best_class, confidence, "rule-based");

        // Add alternatives
        iter = scores.iterator();
        var alt_count: u32 = 0;
        while (iter.next()) |entry| {
            if (entry.key_ptr.* != best_class and alt_count < 3) {
                const alt_confidence = if (total_score > 0) entry.value_ptr.* / total_score else 0;
                try result.addAlternative(entry.key_ptr.*, alt_confidence);
                alt_count += 1;
            }
        }

        return result;
    }
};


// Document Classifier Node
pub const DocumentClassifier = struct {
    id: []const u8,
    name: []const u8,
    config: ClassifierConfig,
    rule_classifier: RuleBasedClassifier,
    allocator: Allocator,

    // Stats
    documents_classified: u64 = 0,
    average_confidence: f32 = 0.0,
    class_distribution: std.AutoHashMap(DocumentClass, u64),

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: ClassifierConfig) !DocumentClassifier {
        var classifier = DocumentClassifier{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .config = config,
            .rule_classifier = RuleBasedClassifier.init(allocator),
            .allocator = allocator,
            .class_distribution = std.AutoHashMap(DocumentClass, u64).init(allocator),
        };

        // Add default rules for rule-based classification
        if (config.model == .RULE_BASED) {
            try classifier.rule_classifier.addDefaultRules();
        }

        return classifier;
    }

    pub fn deinit(self: *DocumentClassifier) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.rule_classifier.deinit();
        self.class_distribution.deinit();
    }

    pub fn classify(self: *DocumentClassifier, text: []const u8) !ClassificationResult {
        const start_time = std.time.milliTimestamp();

        var result = switch (self.config.model) {
            .RULE_BASED => try self.rule_classifier.classify(text),
            .ML_BASED => try self.classifyML(text),
            .LLM_BASED => try self.classifyLLM(text),
        };

        result.processing_time_ms = @intCast(std.time.milliTimestamp() - start_time);

        // Update stats
        self.documents_classified += 1;
        self.updateAverageConfidence(result.confidence);
        const count = self.class_distribution.get(result.document_class) orelse 0;
        try self.class_distribution.put(result.document_class, count + 1);

        return result;
    }

    fn updateAverageConfidence(self: *DocumentClassifier, new_confidence: f32) void {
        if (self.documents_classified == 1) {
            self.average_confidence = new_confidence;
        } else {
            const n = @as(f32, @floatFromInt(self.documents_classified));
            self.average_confidence = ((n - 1) * self.average_confidence + new_confidence) / n;
        }
    }

    fn classifyML(self: *DocumentClassifier, text: []const u8) !ClassificationResult {
        _ = text;
        // ML model integration placeholder
        return ClassificationResult.init(self.allocator, .OTHER, 0.5, "ml-model");
    }

    fn classifyLLM(self: *DocumentClassifier, text: []const u8) !ClassificationResult {
        _ = text;
        // LLM integration placeholder (would call nOpenaiServer)
        return ClassificationResult.init(self.allocator, .OTHER, 0.5, "llm-model");
    }

    pub fn getStats(self: *const DocumentClassifier) ClassifierStats {
        return ClassifierStats{
            .documents_classified = self.documents_classified,
            .average_confidence = self.average_confidence,
        };
    }
};

pub const ClassifierStats = struct {
    documents_classified: u64,
    average_confidence: f32,
};

// Tests
test "DocumentClass operations" {
    try std.testing.expectEqualStrings("INVOICE", DocumentClass.INVOICE.toString());
    try std.testing.expectEqual(DocumentClass.CONTRACT, DocumentClass.fromString("CONTRACT").?);
    try std.testing.expectEqual(DocumentClass.CONTRACT, DocumentClass.fromString("contract").?);
    try std.testing.expectEqual(@as(?DocumentClass, null), DocumentClass.fromString("unknown"));
}

test "ClassificationResult initialization" {
    const allocator = std.testing.allocator;

    var result = try ClassificationResult.init(allocator, .INVOICE, 0.95, "test-model");
    defer result.deinit();

    try std.testing.expectEqual(DocumentClass.INVOICE, result.document_class);
    try std.testing.expect(result.isHighConfidence());
}

test "ClassificationResult alternatives" {
    const allocator = std.testing.allocator;

    var result = try ClassificationResult.init(allocator, .INVOICE, 0.7, "test-model");
    defer result.deinit();

    try result.addAlternative(.RECEIPT, 0.2);
    try result.addAlternative(.OTHER, 0.1);

    try std.testing.expectEqual(@as(usize, 2), result.alternatives.items.len);
}

test "KeywordRule matching" {
    const rule = KeywordRule{
        .keywords = &[_][]const u8{ "invoice", "total", "amount" },
        .document_class = .INVOICE,
        .weight = 1.0,
        .case_sensitive = false,
    };

    const text = "This is an INVOICE with a total amount due.";
    const score = rule.matches(text);
    try std.testing.expect(score > 0.9);
}

test "RuleBasedClassifier" {
    const allocator = std.testing.allocator;

    var classifier = RuleBasedClassifier.init(allocator);
    defer classifier.deinit();

    try classifier.addDefaultRules();

    const invoice_text = "Invoice #12345\nBill To: Customer\nAmount Due: $500\nPayment Terms: Net 30";
    var result = try classifier.classify(invoice_text);
    defer result.deinit();

    try std.testing.expectEqual(DocumentClass.INVOICE, result.document_class);
}

test "DocumentClassifier initialization" {
    const allocator = std.testing.allocator;

    const config = ClassifierConfig{
        .model = .RULE_BASED,
        .min_confidence = 0.6,
    };

    var classifier = try DocumentClassifier.init(allocator, "clf-1", "Doc Classifier", config);
    defer classifier.deinit();

    try std.testing.expectEqualStrings("clf-1", classifier.id);
}

test "containsIgnoreCase" {
    try std.testing.expect(containsIgnoreCase("Hello World", "world"));
    try std.testing.expect(containsIgnoreCase("INVOICE", "invoice"));
    try std.testing.expect(!containsIgnoreCase("Hello", "world"));
}

