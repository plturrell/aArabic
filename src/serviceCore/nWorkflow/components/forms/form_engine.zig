//! Form Engine for nWorkflow Human Tasks
//!
//! This module provides form handling capabilities for human tasks:
//! - FormFieldType: Supported field types (TEXT, NUMBER, DATE, SELECT, etc.)
//! - FormField: Field definition with validation rules
//! - FormDefinition: Complete form with fields array
//! - FormValidator: Validates form data against definition
//! - FormRenderer: Renders forms to HTML and JSON

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Form Field Types
// ============================================================================

/// Form Field Types supported by the form engine
pub const FormFieldType = enum {
    TEXT,
    NUMBER,
    DATE,
    DATETIME,
    SELECT,
    MULTI_SELECT,
    CHECKBOX,
    RADIO,
    TEXTAREA,
    FILE,
    SIGNATURE,
    EMAIL,
    PHONE,
    URL,
    HIDDEN,

    pub fn toString(self: FormFieldType) []const u8 {
        return @tagName(self);
    }

    pub fn fromString(str: []const u8) ?FormFieldType {
        inline for (std.meta.fields(FormFieldType)) |field| {
            if (std.mem.eql(u8, str, field.name)) {
                return @enumFromInt(field.value);
            }
        }
        return null;
    }

    pub fn htmlInputType(self: FormFieldType) []const u8 {
        return switch (self) {
            .TEXT => "text",
            .NUMBER => "number",
            .DATE => "date",
            .DATETIME => "datetime-local",
            .SELECT, .MULTI_SELECT => "select",
            .CHECKBOX => "checkbox",
            .RADIO => "radio",
            .TEXTAREA => "textarea",
            .FILE => "file",
            .SIGNATURE => "signature",
            .EMAIL => "email",
            .PHONE => "tel",
            .URL => "url",
            .HIDDEN => "hidden",
        };
    }

    pub fn isChoiceType(self: FormFieldType) bool {
        return switch (self) {
            .SELECT, .MULTI_SELECT, .RADIO, .CHECKBOX => true,
            else => false,
        };
    }
};

// ============================================================================
// Validation Rule
// ============================================================================

/// Validation rule types
pub const ValidationRuleType = enum {
    REQUIRED,
    MIN_LENGTH,
    MAX_LENGTH,
    MIN_VALUE,
    MAX_VALUE,
    PATTERN,
    EMAIL,
    URL,
    CUSTOM,

    pub fn toString(self: ValidationRuleType) []const u8 {
        return @tagName(self);
    }
};

/// Validation Rule
pub const ValidationRule = struct {
    rule_type: ValidationRuleType,
    value: ?[]const u8 = null, // For min/max/pattern
    message: []const u8, // Error message if validation fails

    pub fn required(message: []const u8) ValidationRule {
        return ValidationRule{
            .rule_type = .REQUIRED,
            .message = message,
        };
    }

    pub fn minLength(length: usize, message: []const u8) ValidationRule {
        var buf: [20]u8 = undefined;
        const len_str = std.fmt.bufPrint(&buf, "{d}", .{length}) catch "0";
        return ValidationRule{
            .rule_type = .MIN_LENGTH,
            .value = len_str,
            .message = message,
        };
    }

    pub fn maxLength(length: usize, message: []const u8) ValidationRule {
        var buf: [20]u8 = undefined;
        const len_str = std.fmt.bufPrint(&buf, "{d}", .{length}) catch "0";
        return ValidationRule{
            .rule_type = .MAX_LENGTH,
            .value = len_str,
            .message = message,
        };
    }

    pub fn pattern(regex: []const u8, message: []const u8) ValidationRule {
        return ValidationRule{
            .rule_type = .PATTERN,
            .value = regex,
            .message = message,
        };
    }
};

// ============================================================================
// Select Option
// ============================================================================

/// Option for select/radio/checkbox fields
pub const SelectOption = struct {
    value: []const u8,
    label: []const u8,
    selected: bool = false,
    disabled: bool = false,
};

// ============================================================================
// Form Field
// ============================================================================

/// Form Field definition
pub const FormField = struct {
    id: []const u8,
    name: []const u8,
    label: []const u8,
    field_type: FormFieldType,
    description: ?[]const u8 = null,
    placeholder: ?[]const u8 = null,
    default_value: ?[]const u8 = null,
    required: bool = false,
    readonly: bool = false,
    hidden: bool = false,
    validation_rules: []const ValidationRule = &[_]ValidationRule{},
    options: []const SelectOption = &[_]SelectOption{}, // For choice types
    min_value: ?f64 = null, // For NUMBER
    max_value: ?f64 = null, // For NUMBER
    min_length: ?usize = null, // For TEXT/TEXTAREA
    max_length: ?usize = null, // For TEXT/TEXTAREA
    pattern: ?[]const u8 = null, // Regex pattern
    accept: ?[]const u8 = null, // For FILE type (mime types)
    order: u32 = 0, // Display order

    pub fn init(id: []const u8, name: []const u8, label: []const u8, field_type: FormFieldType) FormField {
        return FormField{
            .id = id,
            .name = name,
            .label = label,
            .field_type = field_type,
        };
    }

    pub fn isValid(self: *const FormField, value: ?[]const u8) bool {
        // Check required
        if (self.required) {
            if (value == null or value.?.len == 0) {
                return false;
            }
        }

        if (value) |v| {
            // Check length constraints
            if (self.min_length) |min| {
                if (v.len < min) return false;
            }
            if (self.max_length) |max| {
                if (v.len > max) return false;
            }

            // Check numeric constraints
            if (self.field_type == .NUMBER) {
                const num = std.fmt.parseFloat(f64, v) catch return false;
                if (self.min_value) |min| {
                    if (num < min) return false;
                }
                if (self.max_value) |max| {
                    if (num > max) return false;
                }
            }
        }

        return true;
    }
};

// ============================================================================
// Form Definition
// ============================================================================

/// Form Definition
pub const FormDefinition = struct {
    id: []const u8,
    name: []const u8,
    description: ?[]const u8 = null,
    version: u32 = 1,
    fields: []const FormField,
    submit_label: []const u8 = "Submit",
    cancel_label: []const u8 = "Cancel",
    created_at: i64,
    updated_at: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, fields: []const FormField) !FormDefinition {
        const now = std.time.timestamp();
        return FormDefinition{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .fields = fields,
            .created_at = now,
            .updated_at = now,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FormDefinition) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        if (self.description) |d| self.allocator.free(d);
    }

    pub fn getField(self: *const FormDefinition, field_id: []const u8) ?*const FormField {
        for (self.fields) |*field| {
            if (std.mem.eql(u8, field.id, field_id)) {
                return field;
            }
        }
        return null;
    }

    pub fn getRequiredFields(self: *const FormDefinition) []const *const FormField {
        // Note: Would need allocator to return slice, simplified for now
        _ = self;
        return &[_]*const FormField{};
    }

    pub fn fieldCount(self: *const FormDefinition) usize {
        return self.fields.len;
    }
};

// ============================================================================
// Form Validation Error
// ============================================================================

/// Form Validation Error
pub const FormValidationError = struct {
    field_id: []const u8,
    message: []const u8,
    rule_type: ?ValidationRuleType = null,
};

// ============================================================================
// Form Validator
// ============================================================================

/// Form Validator - validates form data against definition
pub const FormValidator = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) FormValidator {
        return FormValidator{
            .allocator = allocator,
        };
    }

    /// Validate form data against a form definition
    /// Returns list of validation errors (empty if valid)
    pub fn validate(
        self: *FormValidator,
        form: *const FormDefinition,
        data: std.StringHashMap([]const u8),
    ) !std.ArrayList(FormValidationError) {
        var errors = std.ArrayList(FormValidationError){};
        errdefer errors.deinit(self.allocator);

        for (form.fields) |field| {
            const value = data.get(field.id);

            // Check required
            if (field.required and (value == null or value.?.len == 0)) {
                try errors.append(self.allocator, FormValidationError{
                    .field_id = field.id,
                    .message = "This field is required",
                    .rule_type = .REQUIRED,
                });
                continue;
            }

            if (value) |v| {
                // Validate field
                try self.validateField(&field, v, &errors);
            }
        }

        return errors;
    }

    fn validateField(
        self: *FormValidator,
        field: *const FormField,
        value: []const u8,
        errors: *std.ArrayList(FormValidationError),
    ) !void {
        // Check min length
        if (field.min_length) |min| {
            if (value.len < min) {
                try errors.append(self.allocator, FormValidationError{
                    .field_id = field.id,
                    .message = "Value is too short",
                    .rule_type = .MIN_LENGTH,
                });
            }
        }

        // Check max length
        if (field.max_length) |max| {
            if (value.len > max) {
                try errors.append(self.allocator, FormValidationError{
                    .field_id = field.id,
                    .message = "Value is too long",
                    .rule_type = .MAX_LENGTH,
                });
            }
        }

        // Check numeric constraints
        if (field.field_type == .NUMBER) {
            if (std.fmt.parseFloat(f64, value)) |num| {
                if (field.min_value) |min| {
                    if (num < min) {
                        try errors.append(self.allocator, FormValidationError{
                            .field_id = field.id,
                            .message = "Value is below minimum",
                            .rule_type = .MIN_VALUE,
                        });
                    }
                }
                if (field.max_value) |max| {
                    if (num > max) {
                        try errors.append(self.allocator, FormValidationError{
                            .field_id = field.id,
                            .message = "Value exceeds maximum",
                            .rule_type = .MAX_VALUE,
                        });
                    }
                }
            } else |_| {
                try errors.append(self.allocator, FormValidationError{
                    .field_id = field.id,
                    .message = "Invalid number format",
                    .rule_type = null,
                });
            }
        }

        // Check email format
        if (field.field_type == .EMAIL) {
            if (std.mem.indexOf(u8, value, "@") == null) {
                try errors.append(self.allocator, FormValidationError{
                    .field_id = field.id,
                    .message = "Invalid email format",
                    .rule_type = .EMAIL,
                });
            }
        }

        // Check URL format
        if (field.field_type == .URL) {
            if (!std.mem.startsWith(u8, value, "http://") and !std.mem.startsWith(u8, value, "https://")) {
                try errors.append(self.allocator, FormValidationError{
                    .field_id = field.id,
                    .message = "Invalid URL format",
                    .rule_type = .URL,
                });
            }
        }
    }

    /// Check if form data is valid
    pub fn isValid(
        self: *FormValidator,
        form: *const FormDefinition,
        data: std.StringHashMap([]const u8),
    ) !bool {
        var errors = try self.validate(form, data);
        defer errors.deinit(self.allocator);
        return errors.items.len == 0;
    }
};

// ============================================================================
// Form Renderer
// ============================================================================

/// Form Renderer - renders forms to HTML and JSON
pub const FormRenderer = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) FormRenderer {
        return FormRenderer{
            .allocator = allocator,
        };
    }

    /// Render form definition to HTML
    pub fn toHtml(self: *FormRenderer, form: *const FormDefinition) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(self.allocator);

        const writer = buffer.writer(self.allocator);

        // Form header
        try writer.print(
            \\<form id="{s}" class="human-task-form">
            \\  <h2>{s}</h2>
        , .{ form.id, form.name });

        if (form.description) |desc| {
            try writer.print(
                \\  <p class="form-description">{s}</p>
            , .{desc});
        }

        // Render each field
        for (form.fields) |field| {
            try self.renderFieldHtml(&field, writer);
        }

        // Submit button
        try writer.print(
            \\  <div class="form-actions">
            \\    <button type="submit" class="btn-primary">{s}</button>
            \\    <button type="button" class="btn-secondary">{s}</button>
            \\  </div>
            \\</form>
        , .{ form.submit_label, form.cancel_label });

        return buffer.toOwnedSlice(self.allocator);
    }

    fn renderFieldHtml(self: *FormRenderer, field: *const FormField, writer: anytype) !void {
        _ = self;

        const required_attr = if (field.required) " required" else "";
        const readonly_attr = if (field.readonly) " readonly" else "";
        const hidden_class = if (field.hidden) " hidden" else "";

        try writer.print(
            \\  <div class="form-field{s}">
            \\    <label for="{s}">{s}
        , .{ hidden_class, field.id, field.label });

        if (field.required) {
            try writer.writeAll(" <span class=\"required\">*</span>");
        }
        try writer.writeAll("</label>\n");

        switch (field.field_type) {
            .TEXTAREA => {
                try writer.print(
                    \\    <textarea id="{s}" name="{s}"{s}{s}
                , .{ field.id, field.name, required_attr, readonly_attr });
                if (field.placeholder) |p| {
                    try writer.print(" placeholder=\"{s}\"", .{p});
                }
                try writer.writeAll(">");
                if (field.default_value) |dv| {
                    try writer.writeAll(dv);
                }
                try writer.writeAll("</textarea>\n");
            },
            .SELECT, .MULTI_SELECT => {
                const multiple = if (field.field_type == .MULTI_SELECT) " multiple" else "";
                try writer.print(
                    \\    <select id="{s}" name="{s}"{s}{s}{s}>
                , .{ field.id, field.name, required_attr, readonly_attr, multiple });

                for (field.options) |opt| {
                    const selected = if (opt.selected) " selected" else "";
                    const disabled = if (opt.disabled) " disabled" else "";
                    try writer.print(
                        \\      <option value="{s}"{s}{s}>{s}</option>
                    , .{ opt.value, selected, disabled, opt.label });
                }
                try writer.writeAll("    </select>\n");
            },
            else => {
                try writer.print(
                    \\    <input type="{s}" id="{s}" name="{s}"{s}{s}
                , .{ field.field_type.htmlInputType(), field.id, field.name, required_attr, readonly_attr });

                if (field.placeholder) |p| {
                    try writer.print(" placeholder=\"{s}\"", .{p});
                }
                if (field.default_value) |dv| {
                    try writer.print(" value=\"{s}\"", .{dv});
                }
                try writer.writeAll(">\n");
            },
        }

        if (field.description) |desc| {
            try writer.print(
                \\    <span class="field-description">{s}</span>
            , .{desc});
        }

        try writer.writeAll("  </div>\n");
    }

    /// Render form definition to JSON
    pub fn toJson(self: *FormRenderer, form: *const FormDefinition) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(self.allocator);

        const writer = buffer.writer(self.allocator);

        try writer.print(
            \\{{"id":"{s}","name":"{s}","version":{d},"fields":[
        , .{ form.id, form.name, form.version });

        for (form.fields, 0..) |field, i| {
            if (i > 0) try writer.writeAll(",");
            try self.renderFieldJson(&field, writer);
        }

        try writer.print(
            \\],"submit_label":"{s}","cancel_label":"{s}"}}
        , .{ form.submit_label, form.cancel_label });

        return buffer.toOwnedSlice(self.allocator);
    }

    fn renderFieldJson(self: *FormRenderer, field: *const FormField, writer: anytype) !void {
        _ = self;

        try writer.print(
            \\{{"id":"{s}","name":"{s}","label":"{s}","type":"{s}","required":{s}
        , .{
            field.id,
            field.name,
            field.label,
            field.field_type.toString(),
            if (field.required) "true" else "false",
        });

        if (field.placeholder) |p| {
            try writer.print(
                \\,"placeholder":"{s}"
            , .{p});
        }
        if (field.default_value) |dv| {
            try writer.print(
                \\,"default_value":"{s}"
            , .{dv});
        }

        try writer.writeAll("}}");
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FormFieldType conversions" {
    try std.testing.expectEqualStrings("TEXT", FormFieldType.TEXT.toString());
    try std.testing.expectEqual(FormFieldType.TEXT, FormFieldType.fromString("TEXT").?);
    try std.testing.expectEqualStrings("text", FormFieldType.TEXT.htmlInputType());

    try std.testing.expect(FormFieldType.SELECT.isChoiceType());
    try std.testing.expect(!FormFieldType.TEXT.isChoiceType());
}

test "FormField validation" {
    const field = FormField{
        .id = "name",
        .name = "name",
        .label = "Name",
        .field_type = .TEXT,
        .required = true,
        .min_length = 2,
        .max_length = 50,
    };

    // Valid value
    try std.testing.expect(field.isValid("John"));

    // Empty value for required field
    try std.testing.expect(!field.isValid(null));
    try std.testing.expect(!field.isValid(""));

    // Too short
    try std.testing.expect(!field.isValid("J"));
}

test "FormField numeric validation" {
    const field = FormField{
        .id = "age",
        .name = "age",
        .label = "Age",
        .field_type = .NUMBER,
        .min_value = 0,
        .max_value = 150,
    };

    try std.testing.expect(field.isValid("25"));
    try std.testing.expect(!field.isValid("invalid"));
}

test "FormDefinition creation" {
    const allocator = std.testing.allocator;

    const fields = [_]FormField{
        FormField.init("name", "name", "Name", .TEXT),
        FormField.init("email", "email", "Email", .EMAIL),
    };

    var form = try FormDefinition.init(allocator, "form-1", "Contact Form", &fields);
    defer form.deinit();

    try std.testing.expectEqualStrings("form-1", form.id);
    try std.testing.expectEqual(@as(usize, 2), form.fieldCount());

    const name_field = form.getField("name");
    try std.testing.expect(name_field != null);
    try std.testing.expectEqualStrings("Name", name_field.?.label);
}

test "FormValidator validate required" {
    const allocator = std.testing.allocator;

    var required_field = FormField.init("name", "name", "Name", .TEXT);
    required_field.required = true;

    const fields = [_]FormField{required_field};
    var form = try FormDefinition.init(allocator, "form-1", "Test Form", &fields);
    defer form.deinit();

    var validator = FormValidator.init(allocator);

    // Missing required field
    var empty_data = std.StringHashMap([]const u8).init(allocator);
    defer empty_data.deinit();

    var errors = try validator.validate(&form, empty_data);
    defer errors.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), errors.items.len);
    try std.testing.expectEqual(ValidationRuleType.REQUIRED, errors.items[0].rule_type.?);
}

test "FormValidator validate email" {
    const allocator = std.testing.allocator;

    const fields = [_]FormField{FormField.init("email", "email", "Email", .EMAIL)};
    var form = try FormDefinition.init(allocator, "form-1", "Test Form", &fields);
    defer form.deinit();

    var validator = FormValidator.init(allocator);

    // Invalid email
    var data = std.StringHashMap([]const u8).init(allocator);
    defer data.deinit();
    try data.put("email", "invalid-email");

    var errors = try validator.validate(&form, data);
    defer errors.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), errors.items.len);
    try std.testing.expectEqual(ValidationRuleType.EMAIL, errors.items[0].rule_type.?);
}

test "FormValidator valid form" {
    const allocator = std.testing.allocator;

    var email_field = FormField.init("email", "email", "Email", .EMAIL);
    email_field.required = true;

    const fields = [_]FormField{email_field};
    var form = try FormDefinition.init(allocator, "form-1", "Test Form", &fields);
    defer form.deinit();

    var validator = FormValidator.init(allocator);

    var data = std.StringHashMap([]const u8).init(allocator);
    defer data.deinit();
    try data.put("email", "test@example.com");

    const is_valid = try validator.isValid(&form, data);
    try std.testing.expect(is_valid);
}

test "FormRenderer toJson" {
    const allocator = std.testing.allocator;

    const fields = [_]FormField{FormField.init("name", "name", "Name", .TEXT)};
    var form = try FormDefinition.init(allocator, "form-1", "Test Form", &fields);
    defer form.deinit();

    var renderer = FormRenderer.init(allocator);
    const json = try renderer.toJson(&form);
    defer allocator.free(json);

    // Check JSON contains expected content
    try std.testing.expect(std.mem.indexOf(u8, json, "\"id\":\"form-1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"Test Form\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"fields\":[") != null);
}

test "FormRenderer toHtml" {
    const allocator = std.testing.allocator;

    const fields = [_]FormField{FormField.init("name", "name", "Name", .TEXT)};
    var form = try FormDefinition.init(allocator, "form-1", "Test Form", &fields);
    defer form.deinit();

    var renderer = FormRenderer.init(allocator);
    const html = try renderer.toHtml(&form);
    defer allocator.free(html);

    // Check HTML contains expected content
    try std.testing.expect(std.mem.indexOf(u8, html, "<form id=\"form-1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "<h2>Test Form</h2>") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "type=\"text\"") != null);
}

