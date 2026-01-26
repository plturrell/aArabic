//! Document Validator Node for nWorkflow IDP
//! Provides validation of extracted data using schemas and rules

const std = @import("std");
const Allocator = std.mem.Allocator;

// Validation rule type
pub const ValidationRuleType = enum {
    REQUIRED,
    FORMAT,
    RANGE,
    LENGTH,
    PATTERN,
    ENUM,
    CUSTOM,
    CROSS_FIELD,

    pub fn toString(self: ValidationRuleType) []const u8 {
        return @tagName(self);
    }
};

// Validation severity
pub const ValidationSeverity = enum {
    ERROR,
    WARNING,
    INFO,

    pub fn toString(self: ValidationSeverity) []const u8 {
        return @tagName(self);
    }
};

// Validation error/warning
pub const ValidationIssue = struct {
    field: []const u8,
    message: []const u8,
    severity: ValidationSeverity,
    rule_type: ValidationRuleType,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        field: []const u8,
        message: []const u8,
        severity: ValidationSeverity,
        rule_type: ValidationRuleType,
    ) !ValidationIssue {
        return ValidationIssue{
            .field = try allocator.dupe(u8, field),
            .message = try allocator.dupe(u8, message),
            .severity = severity,
            .rule_type = rule_type,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ValidationIssue) void {
        self.allocator.free(self.field);
        self.allocator.free(self.message);
    }

    pub fn toJson(self: *const ValidationIssue, allocator: Allocator) ![]const u8 {
        _ = allocator;
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit();
        var writer = buffer.writer();
        try writer.print(
            \\{{"field":"{s}","message":"{s}","severity":"{s}","rule":"{s}"}}
        , .{
            self.field,
            self.message,
            self.severity.toString(),
            self.rule_type.toString(),
        });
        return buffer.toOwnedSlice();
    }
};

// Validation rule
pub const ValidationRule = struct {
    field: []const u8,
    rule_type: ValidationRuleType,
    parameters: RuleParameters,
    message: ?[]const u8,
    severity: ValidationSeverity,

    pub const RuleParameters = union(enum) {
        required: bool,
        min_length: usize,
        max_length: usize,
        min_value: f64,
        max_value: f64,
        pattern: []const u8,
        enum_values: []const []const u8,
        cross_field: []const u8,
        none: void,
    };

    pub fn validate(self: *const ValidationRule, value: ?[]const u8) bool {
        const val = value orelse {
            return self.rule_type != .REQUIRED or !self.parameters.required;
        };

        return switch (self.rule_type) {
            .REQUIRED => val.len > 0,
            .LENGTH => switch (self.parameters) {
                .min_length => |min| val.len >= min,
                .max_length => |max| val.len <= max,
                else => true,
            },
            .ENUM => blk: {
                if (self.parameters != .enum_values) break :blk true;
                for (self.parameters.enum_values) |allowed| {
                    if (std.mem.eql(u8, val, allowed)) break :blk true;
                }
                break :blk false;
            },
            else => true,
        };
    }
};

// Validation result
pub const ValidationResult = struct {
    is_valid: bool,
    errors: std.ArrayList(ValidationIssue),
    warnings: std.ArrayList(ValidationIssue),
    fields_validated: u32,
    processing_time_ms: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ValidationResult {
        return ValidationResult{
            .is_valid = true,
            .errors = std.ArrayList(ValidationIssue){},
            .warnings = std.ArrayList(ValidationIssue){},
            .fields_validated = 0,
            .processing_time_ms = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ValidationResult) void {
        for (self.errors.items) |*err| {
            err.deinit();
        }
        self.errors.deinit(self.allocator);
        for (self.warnings.items) |*warn| {
            warn.deinit();
        }
        self.warnings.deinit(self.allocator);
    }

    pub fn addError(self: *ValidationResult, issue: ValidationIssue) !void {
        try self.errors.append(self.allocator, issue);
        self.is_valid = false;
    }

    pub fn addWarning(self: *ValidationResult, issue: ValidationIssue) !void {
        try self.warnings.append(self.allocator, issue);
    }
};

// Schema validator
pub const SchemaValidator = struct {
    rules: std.ArrayList(ValidationRule),
    allocator: Allocator,

    pub fn init(allocator: Allocator) SchemaValidator {
        return SchemaValidator{
            .rules = std.ArrayList(ValidationRule){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SchemaValidator) void {
        self.rules.deinit(self.allocator);
    }

    pub fn addRule(self: *SchemaValidator, rule: ValidationRule) !void {
        try self.rules.append(self.allocator, rule);
    }

    pub fn validate(self: *const SchemaValidator, data: std.StringHashMap([]const u8)) !ValidationResult {
        var result = ValidationResult.init(self.allocator);

        for (self.rules.items) |rule| {
            const value = data.get(rule.field);
            result.fields_validated += 1;

            if (!rule.validate(value)) {
                const msg = rule.message orelse "Validation failed";
                const issue = try ValidationIssue.init(
                    self.allocator,
                    rule.field,
                    msg,
                    rule.severity,
                    rule.rule_type,
                );

                if (rule.severity == .ERROR) {
                    try result.addError(issue);
                } else {
                    try result.addWarning(issue);
                }
            }
        }

        return result;
    }
};

// Cross-field validator
pub const CrossFieldValidator = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) CrossFieldValidator {
        return CrossFieldValidator{
            .allocator = allocator,
        };
    }

    pub fn validateDateRange(
        self: *const CrossFieldValidator,
        start_date: ?[]const u8,
        end_date: ?[]const u8,
        result: *ValidationResult,
    ) !void {
        _ = self;
        if (start_date == null or end_date == null) return;

        // Simple string comparison for dates in ISO format
        if (std.mem.order(u8, start_date.?, end_date.?) == .gt) {
            const issue = try ValidationIssue.init(
                result.allocator,
                "date_range",
                "Start date must be before end date",
                .ERROR,
                .CROSS_FIELD,
            );
            try result.addError(issue);
        }
    }
};

// Validator configuration
pub const ValidatorConfig = struct {
    strict_mode: bool = false,
    fail_fast: bool = false,
    max_errors: u32 = 100,
};

// Document Validator Node
pub const DocumentValidator = struct {
    id: []const u8,
    name: []const u8,
    config: ValidatorConfig,
    schema_validator: SchemaValidator,
    cross_field_validator: CrossFieldValidator,
    allocator: Allocator,

    // Stats
    documents_validated: u64 = 0,
    total_errors: u64 = 0,
    total_warnings: u64 = 0,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, config: ValidatorConfig) !DocumentValidator {
        return DocumentValidator{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .config = config,
            .schema_validator = SchemaValidator.init(allocator),
            .cross_field_validator = CrossFieldValidator.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DocumentValidator) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        self.schema_validator.deinit();
    }

    pub fn addRule(self: *DocumentValidator, rule: ValidationRule) !void {
        try self.schema_validator.addRule(rule);
    }

    pub fn validate(self: *DocumentValidator, data: std.StringHashMap([]const u8)) !ValidationResult {
        const start_time = std.time.milliTimestamp();

        var result = try self.schema_validator.validate(data);
        result.processing_time_ms = @intCast(std.time.milliTimestamp() - start_time);

        // Update stats
        self.documents_validated += 1;
        self.total_errors += result.errors.items.len;
        self.total_warnings += result.warnings.items.len;

        return result;
    }

    pub fn getStats(self: *const DocumentValidator) ValidatorStats {
        return ValidatorStats{
            .documents_validated = self.documents_validated,
            .total_errors = self.total_errors,
            .total_warnings = self.total_warnings,
        };
    }
};

pub const ValidatorStats = struct {
    documents_validated: u64,
    total_errors: u64,
    total_warnings: u64,
};

// Tests
test "ValidationRuleType operations" {
    try std.testing.expectEqualStrings("REQUIRED", ValidationRuleType.REQUIRED.toString());
    try std.testing.expectEqualStrings("ERROR", ValidationSeverity.ERROR.toString());
}

test "ValidationIssue initialization" {
    const allocator = std.testing.allocator;

    var issue = try ValidationIssue.init(
        allocator,
        "email",
        "Invalid email format",
        .ERROR,
        .FORMAT,
    );
    defer issue.deinit();

    try std.testing.expectEqualStrings("email", issue.field);
    try std.testing.expectEqual(ValidationSeverity.ERROR, issue.severity);
}

test "ValidationRule required" {
    const rule = ValidationRule{
        .field = "name",
        .rule_type = .REQUIRED,
        .parameters = .{ .required = true },
        .message = "Name is required",
        .severity = .ERROR,
    };

    try std.testing.expect(rule.validate("John"));
    try std.testing.expect(!rule.validate(null));
    try std.testing.expect(!rule.validate(""));
}

test "ValidationRule length" {
    const rule = ValidationRule{
        .field = "code",
        .rule_type = .LENGTH,
        .parameters = .{ .min_length = 3 },
        .message = "Code must be at least 3 characters",
        .severity = .ERROR,
    };

    try std.testing.expect(rule.validate("ABC"));
    try std.testing.expect(!rule.validate("AB"));
}

test "ValidationResult operations" {
    const allocator = std.testing.allocator;

    var result = ValidationResult.init(allocator);
    defer result.deinit();

    try std.testing.expect(result.is_valid);

    const issue = try ValidationIssue.init(allocator, "field", "error", .ERROR, .REQUIRED);
    try result.addError(issue);

    try std.testing.expect(!result.is_valid);
    try std.testing.expectEqual(@as(usize, 1), result.errors.items.len);
}

test "DocumentValidator initialization" {
    const allocator = std.testing.allocator;

    var validator = try DocumentValidator.init(allocator, "val-1", "Doc Validator", .{});
    defer validator.deinit();

    try std.testing.expectEqualStrings("val-1", validator.id);
}
