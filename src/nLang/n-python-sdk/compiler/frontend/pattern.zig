// Mojo SDK - Pattern Matching
// Day 23: Match expressions, destructuring, and exhaustiveness checking

const std = @import("std");

// ============================================================================
// Pattern Types
// ============================================================================

pub const Pattern = union(enum) {
    Wildcard,                           // _
    Literal: LiteralPattern,            // 42, "hello"
    Variable: []const u8,               // x, name
    Constructor: ConstructorPattern,    // Point(x, y)
    Tuple: TuplePattern,               // (a, b, c)
    Array: ArrayPattern,               // [1, 2, 3]
    Range: RangePattern,               // 1..10
    Guard: GuardPattern,               // x if x > 0
    Or: OrPattern,                     // A | B
    
    pub fn isWildcard(self: *const Pattern) bool {
        return switch (self.*) {
            .Wildcard => true,
            else => false,
        };
    }
    
    pub fn isVariable(self: *const Pattern) bool {
        return switch (self.*) {
            .Variable => true,
            else => false,
        };
    }
};

// ============================================================================
// Literal Patterns
// ============================================================================

pub const LiteralPattern = union(enum) {
    Int: i64,
    Float: f64,
    Bool: bool,
    String: []const u8,
    
    pub fn matches(self: *const LiteralPattern, value: LiteralValue) bool {
        return switch (self.*) {
            .Int => |expected| value == .Int and value.Int == expected,
            .Float => |expected| value == .Float and value.Float == expected,
            .Bool => |expected| value == .Bool and value.Bool == expected,
            .String => |expected| value == .String and std.mem.eql(u8, value.String, expected),
        };
    }
};

pub const LiteralValue = union(enum) {
    Int: i64,
    Float: f64,
    Bool: bool,
    String: []const u8,
};

// ============================================================================
// Constructor Patterns
// ============================================================================

pub const ConstructorPattern = struct {
    name: []const u8,
    fields: std.ArrayList(Pattern),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) ConstructorPattern {
        return ConstructorPattern{
            .name = name,
            .fields = std.ArrayList(Pattern){},
            .allocator = allocator,
        };
    }
    
    pub fn addField(self: *ConstructorPattern, pattern: Pattern) !void {
        try self.fields.append(self.allocator, pattern);
    }
    
    pub fn deinit(self: *ConstructorPattern) void {
        self.fields.deinit(self.allocator);
    }
};

// ============================================================================
// Tuple Patterns
// ============================================================================

pub const TuplePattern = struct {
    elements: std.ArrayList(Pattern),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) TuplePattern {
        return TuplePattern{
            .elements = std.ArrayList(Pattern){},
            .allocator = allocator,
        };
    }
    
    pub fn addElement(self: *TuplePattern, pattern: Pattern) !void {
        try self.elements.append(self.allocator, pattern);
    }
    
    pub fn deinit(self: *TuplePattern) void {
        self.elements.deinit(self.allocator);
    }
};

// ============================================================================
// Array Patterns
// ============================================================================

pub const ArrayPattern = struct {
    elements: std.ArrayList(Pattern),
    rest: ?[]const u8,  // Variable to bind remaining elements
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ArrayPattern {
        return ArrayPattern{
            .elements = std.ArrayList(Pattern){},
            .rest = null,
            .allocator = allocator,
        };
    }
    
    pub fn addElement(self: *ArrayPattern, pattern: Pattern) !void {
        try self.elements.append(self.allocator, pattern);
    }
    
    pub fn withRest(self: ArrayPattern, rest_var: []const u8) ArrayPattern {
        return ArrayPattern{
            .elements = self.elements,
            .rest = rest_var,
            .allocator = self.allocator,
        };
    }
    
    pub fn deinit(self: *ArrayPattern) void {
        self.elements.deinit(self.allocator);
    }
};

// ============================================================================
// Range Patterns
// ============================================================================

pub const RangePattern = struct {
    start: i64,
    end: i64,
    inclusive: bool = true,
    
    pub fn init(start: i64, end: i64) RangePattern {
        return RangePattern{
            .start = start,
            .end = end,
        };
    }
    
    pub fn exclusive(start: i64, end: i64) RangePattern {
        return RangePattern{
            .start = start,
            .end = end,
            .inclusive = false,
        };
    }
    
    pub fn matches(self: *const RangePattern, value: i64) bool {
        if (self.inclusive) {
            return value >= self.start and value <= self.end;
        } else {
            return value >= self.start and value < self.end;
        }
    }
};

// ============================================================================
// Guard Patterns
// ============================================================================

pub const GuardPattern = struct {
    pattern: *Pattern,
    condition: []const u8,  // Guard expression
    
    pub fn init(allocator: std.mem.Allocator, pattern: Pattern, condition: []const u8) !GuardPattern {
        const pat_ptr = try allocator.create(Pattern);
        pat_ptr.* = pattern;
        return GuardPattern{
            .pattern = pat_ptr,
            .condition = condition,
        };
    }
};

// ============================================================================
// Or Patterns
// ============================================================================

pub const OrPattern = struct {
    alternatives: std.ArrayList(Pattern),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) OrPattern {
        return OrPattern{
            .alternatives = std.ArrayList(Pattern){},
            .allocator = allocator,
        };
    }
    
    pub fn addAlternative(self: *OrPattern, pattern: Pattern) !void {
        try self.alternatives.append(self.allocator, pattern);
    }
    
    pub fn deinit(self: *OrPattern) void {
        self.alternatives.deinit(self.allocator);
    }
};

// ============================================================================
// Match Expression
// ============================================================================

pub const MatchArm = struct {
    pattern: Pattern,
    body: []const u8,  // Simplified: would be AST expression
    
    pub fn init(pattern: Pattern, body: []const u8) MatchArm {
        return MatchArm{
            .pattern = pattern,
            .body = body,
        };
    }
};

pub const MatchExpression = struct {
    scrutinee: []const u8,  // Expression being matched
    arms: std.ArrayList(MatchArm),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, scrutinee: []const u8) MatchExpression {
        return MatchExpression{
            .scrutinee = scrutinee,
            .arms = std.ArrayList(MatchArm){},
            .allocator = allocator,
        };
    }
    
    pub fn addArm(self: *MatchExpression, arm: MatchArm) !void {
        try self.arms.append(self.allocator, arm);
    }
    
    pub fn deinit(self: *MatchExpression) void {
        self.arms.deinit(self.allocator);
    }
};

// ============================================================================
// Exhaustiveness Checker
// ============================================================================

pub const ExhaustivenessChecker = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ExhaustivenessChecker {
        return ExhaustivenessChecker{ .allocator = allocator };
    }
    
    pub fn isExhaustive(self: *ExhaustivenessChecker, match_expr: *const MatchExpression) bool {
        _ = self;
        // Check if patterns cover all possible values
        for (match_expr.arms.items) |arm| {
            if (arm.pattern.isWildcard()) {
                return true;  // Wildcard catches everything
            }
        }
        return false;  // Simplified: would do deeper analysis
    }
    
    pub fn findMissingPatterns(self: *ExhaustivenessChecker, match_expr: *const MatchExpression) !std.ArrayList(Pattern) {
        var missing = std.ArrayList(Pattern).init(self.allocator);
        
        if (!self.isExhaustive(match_expr)) {
            // Simplified: would compute actual missing patterns
            try missing.append(Pattern.Wildcard);
        }
        
        return missing;
    }
};

// ============================================================================
// Pattern Matcher
// ============================================================================

pub const MatchResult = struct {
    matched: bool,
    bindings: std.StringHashMap([]const u8),
    
    pub fn init(allocator: std.mem.Allocator) MatchResult {
        return MatchResult{
            .matched = false,
            .bindings = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *MatchResult) void {
        self.bindings.deinit();
    }
};

pub const PatternMatcher = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) PatternMatcher {
        return PatternMatcher{ .allocator = allocator };
    }
    
    pub fn matchPattern(self: *PatternMatcher, pattern: *const Pattern, value: []const u8) MatchResult {
        var result = MatchResult.init(self.allocator);
        
        switch (pattern.*) {
            .Wildcard => {
                result.matched = true;
            },
            .Variable => |var_name| {
                result.matched = true;
                result.bindings.put(var_name, value) catch {};
            },
            else => {
                // Simplified: would match other pattern types
                result.matched = false;
            },
        }
        
        return result;
    }
    
    pub fn matchLiteral(self: *PatternMatcher, pattern: *const LiteralPattern, value: LiteralValue) bool {
        _ = self;
        return pattern.matches(value);
    }
    
    pub fn matchRange(self: *PatternMatcher, pattern: *const RangePattern, value: i64) bool {
        _ = self;
        return pattern.matches(value);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "pattern: wildcard" {
    var wildcard: Pattern = .Wildcard;
    try std.testing.expect(wildcard.isWildcard());
    try std.testing.expect(!wildcard.isVariable());
}

test "pattern: variable" {
    var var_pattern: Pattern = .{ .Variable = "x" };
    try std.testing.expect(!var_pattern.isWildcard());
    try std.testing.expect(var_pattern.isVariable());
}

test "pattern: literal int match" {
    const pattern = LiteralPattern{ .Int = 42 };
    const value = LiteralValue{ .Int = 42 };
    try std.testing.expect(pattern.matches(value));
    
    const wrong_value = LiteralValue{ .Int = 43 };
    try std.testing.expect(!pattern.matches(wrong_value));
}

test "pattern: literal string match" {
    const pattern = LiteralPattern{ .String = "hello" };
    const value = LiteralValue{ .String = "hello" };
    try std.testing.expect(pattern.matches(value));
}

test "pattern: range inclusive" {
    const range = RangePattern.init(1, 10);
    try std.testing.expect(range.matches(1));
    try std.testing.expect(range.matches(5));
    try std.testing.expect(range.matches(10));
    try std.testing.expect(!range.matches(0));
    try std.testing.expect(!range.matches(11));
}

test "pattern: range exclusive" {
    const range = RangePattern.exclusive(1, 10);
    try std.testing.expect(range.matches(1));
    try std.testing.expect(range.matches(9));
    try std.testing.expect(!range.matches(10));
}

test "pattern: match expression" {
    const allocator = std.testing.allocator;
    var match_expr = MatchExpression.init(allocator, "x");
    defer match_expr.deinit();
    
    const arm = MatchArm.init(Pattern.Wildcard, "default");
    try match_expr.addArm(arm);
    
    try std.testing.expectEqual(@as(usize, 1), match_expr.arms.items.len);
}

test "pattern: exhaustiveness checker" {
    const allocator = std.testing.allocator;
    var checker = ExhaustivenessChecker.init(allocator);
    
    var match_expr = MatchExpression.init(allocator, "x");
    defer match_expr.deinit();
    
    // Not exhaustive without wildcard
    try std.testing.expect(!checker.isExhaustive(&match_expr));
    
    // Add wildcard arm
    const wildcard_arm = MatchArm.init(Pattern.Wildcard, "default");
    try match_expr.addArm(wildcard_arm);
    
    // Now exhaustive
    try std.testing.expect(checker.isExhaustive(&match_expr));
}

test "pattern: pattern matcher wildcard" {
    const allocator = std.testing.allocator;
    var matcher = PatternMatcher.init(allocator);
    
    var pattern: Pattern = .Wildcard;
    var result = matcher.matchPattern(&pattern, "any_value");
    defer result.deinit();
    
    try std.testing.expect(result.matched);
}

test "pattern: pattern matcher variable" {
    const allocator = std.testing.allocator;
    var matcher = PatternMatcher.init(allocator);
    
    const pattern = Pattern{ .Variable = "x" };
    var result = matcher.matchPattern(&pattern, "42");
    defer result.deinit();
    
    try std.testing.expect(result.matched);
    try std.testing.expect(result.bindings.contains("x"));
}
