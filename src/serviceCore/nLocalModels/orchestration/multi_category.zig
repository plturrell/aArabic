//! Multi-Category Model Support Module
//! Enables models to serve multiple task categories with per-category scoring
//!
//! Features:
//! - Models can be assigned to multiple categories
//! - Per-category performance weights and confidence scores
//! - Weighted scoring across categories
//! - Better resource utilization

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Multi-category score for a specific category
pub const MultiCategoryScore = struct {
    category: []const u8,
    base_score: f32,
    benchmark_score: f32,
    confidence: f32, // 0.0-1.0, how confident this model is for this category
    
    pub fn totalScore(self: MultiCategoryScore) f32 {
        return (self.base_score + self.benchmark_score) * self.confidence;
    }
    
    pub fn deinit(self: *MultiCategoryScore, allocator: Allocator) void {
        allocator.free(self.category);
    }
};

/// Model with multi-category support
pub const MultiCategoryModel = struct {
    model_name: []const u8,
    category_scores: []MultiCategoryScore,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, model_name: []const u8) !*MultiCategoryModel {
        const self = try allocator.create(MultiCategoryModel);
        self.* = .{
            .model_name = try allocator.dupe(u8, model_name),
            .category_scores = &[_]MultiCategoryScore{},
            .allocator = allocator,
        };
        return self;
    }
    
    pub fn deinit(self: *MultiCategoryModel) void {
        self.allocator.free(self.model_name);
        for (self.category_scores) |*score| {
            score.deinit(self.allocator);
        }
        self.allocator.free(self.category_scores);
        self.allocator.destroy(self);
    }
    
    /// Add a category with its scoring parameters
    pub fn addCategory(
        self: *MultiCategoryModel,
        category: []const u8,
        base_score: f32,
        benchmark_score: f32,
        confidence: f32,
    ) !void {
        // Create new score
        var new_score = MultiCategoryScore{
            .category = try self.allocator.dupe(u8, category),
            .base_score = base_score,
            .benchmark_score = benchmark_score,
            .confidence = std.math.clamp(confidence, 0.0, 1.0),
        };
        
        // Append to existing scores
        var scores_list = std.ArrayList(MultiCategoryScore).init(self.allocator);
        defer scores_list.deinit();
        
        for (self.category_scores) |score| {
            try scores_list.append(score);
        }
        try scores_list.append(new_score);
        
        // Replace old array
        if (self.category_scores.len > 0) {
            self.allocator.free(self.category_scores);
        }
        self.category_scores = try scores_list.toOwnedSlice();
    }
    
    /// Get score for a specific category
    pub fn scoreForCategory(self: *MultiCategoryModel, category: []const u8) ?f32 {
        for (self.category_scores) |score| {
            if (std.mem.eql(u8, score.category, category)) {
                return score.totalScore();
            }
        }
        return null;
    }
    
    /// Get all categories this model supports
    pub fn getCategories(self: *MultiCategoryModel, allocator: Allocator) ![][]const u8 {
        var categories = std.ArrayList([]const u8).init(allocator);
        errdefer categories.deinit();
        
        for (self.category_scores) |score| {
            try categories.append(try allocator.dupe(u8, score.category));
        }
        
        return try categories.toOwnedSlice();
    }
    
    /// Get best category for this model
    pub fn getBestCategory(self: *MultiCategoryModel) ?[]const u8 {
        if (self.category_scores.len == 0) return null;
        
        var best_category: []const u8 = self.category_scores[0].category;
        var best_score: f32 = self.category_scores[0].totalScore();
        
        for (self.category_scores[1..]) |score| {
            const total = score.totalScore();
            if (total > best_score) {
                best_score = total;
                best_category = score.category;
            }
        }
        
        return best_category;
    }
    
    /// Check if model supports a category
    pub fn supportsCategory(self: *MultiCategoryModel, category: []const u8) bool {
        for (self.category_scores) |score| {
            if (std.mem.eql(u8, score.category, category)) {
                return true;
            }
        }
        return false;
    }
};

/// Multi-category model registry
pub const MultiCategoryRegistry = struct {
    allocator: Allocator,
    models: std.StringHashMap(*MultiCategoryModel),
    
    pub fn init(allocator: Allocator) !*MultiCategoryRegistry {
        const self = try allocator.create(MultiCategoryRegistry);
        self.* = .{
            .allocator = allocator,
            .models = std.StringHashMap(*MultiCategoryModel).init(allocator),
        };
        return self;
    }
    
    pub fn deinit(self: *MultiCategoryRegistry) void {
        var it = self.models.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.models.deinit();
        self.allocator.destroy(self);
    }
    
    /// Register a model with multi-category support
    pub fn registerModel(self: *MultiCategoryRegistry, model: *MultiCategoryModel) !void {
        const key = try self.allocator.dupe(u8, model.model_name);
        try self.models.put(key, model);
    }
    
    /// Get model by name
    pub fn getModel(self: *MultiCategoryRegistry, name: []const u8) ?*MultiCategoryModel {
        return self.models.get(name);
    }
    
    /// Get all models supporting a category
    pub fn getModelsForCategory(
        self: *MultiCategoryRegistry,
        allocator: Allocator,
        category: []const u8,
    ) ![][]const u8 {
        var models = std.ArrayList([]const u8).init(allocator);
        errdefer models.deinit();
        
        var it = self.models.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.*.supportsCategory(category)) {
                try models.append(try allocator.dupe(u8, entry.key_ptr.*));
            }
        }
        
        return try models.toOwnedSlice();
    }
    
    /// Select best model for a category based on scores
    pub fn selectBestForCategory(
        self: *MultiCategoryRegistry,
        category: []const u8,
    ) ?*MultiCategoryModel {
        var best_model: ?*MultiCategoryModel = null;
        var best_score: f32 = 0.0;
        
        var it = self.models.iterator();
        while (it.next()) |entry| {
            const model = entry.value_ptr.*;
            if (model.scoreForCategory(category)) |score| {
                if (score > best_score) {
                    best_score = score;
                    best_model = model;
                }
            }
        }
        
        return best_model;
    }
    
    /// Get model count
    pub fn count(self: *MultiCategoryRegistry) usize {
        return self.models.count();
    }
};

/// Helper to build multi-category model from standard model metadata
pub const MultiCategoryBuilder = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) MultiCategoryBuilder {
        return .{ .allocator = allocator };
    }
    
    /// Build multi-category model with default confidence scores
    pub fn buildWithDefaults(
        self: *MultiCategoryBuilder,
        model_name: []const u8,
        categories: []const []const u8,
        base_score: f32,
        benchmark_score: f32,
    ) !*MultiCategoryModel {
        const model = try MultiCategoryModel.init(self.allocator, model_name);
        errdefer model.deinit();
        
        // Add each category with high confidence (0.9)
        for (categories) |category| {
            try model.addCategory(category, base_score, benchmark_score, 0.9);
        }
        
        return model;
    }
    
    /// Build with custom confidence per category
    pub fn buildWithConfidence(
        self: *MultiCategoryBuilder,
        model_name: []const u8,
        category_configs: []const CategoryConfig,
    ) !*MultiCategoryModel {
        const model = try MultiCategoryModel.init(self.allocator, model_name);
        errdefer model.deinit();
        
        for (category_configs) |config| {
            try model.addCategory(
                config.category,
                config.base_score,
                config.benchmark_score,
                config.confidence,
            );
        }
        
        return model;
    }
    
    pub const CategoryConfig = struct {
        category: []const u8,
        base_score: f32,
        benchmark_score: f32,
        confidence: f32,
    };
};

// Tests
test "MultiCategoryScore calculation" {
    const score = MultiCategoryScore{
        .category = "code",
        .base_score = 50.0,
        .benchmark_score = 40.0,
        .confidence = 0.9,
    };
    
    const total = score.totalScore();
    // (50 + 40) * 0.9 = 81.0
    try std.testing.expectApproxEqAbs(@as(f32, 81.0), total, 0.01);
}

test "MultiCategoryModel creation and category addition" {
    const allocator = std.testing.allocator;
    
    const model = try MultiCategoryModel.init(allocator, "test-model");
    defer model.deinit();
    
    try model.addCategory("code", 50.0, 40.0, 0.9);
    try model.addCategory("math", 45.0, 35.0, 0.7);
    
    try std.testing.expectEqual(@as(usize, 2), model.category_scores.len);
}

test "MultiCategoryModel - score for category" {
    const allocator = std.testing.allocator;
    
    const model = try MultiCategoryModel.init(allocator, "test-model");
    defer model.deinit();
    
    try model.addCategory("code", 50.0, 40.0, 0.9);
    
    const score = model.scoreForCategory("code");
    try std.testing.expect(score != null);
    try std.testing.expectApproxEqAbs(@as(f32, 81.0), score.?, 0.01);
    
    const no_score = model.scoreForCategory("unknown");
    try std.testing.expect(no_score == null);
}

test "MultiCategoryModel - get best category" {
    const allocator = std.testing.allocator;
    
    const model = try MultiCategoryModel.init(allocator, "test-model");
    defer model.deinit();
    
    try model.addCategory("code", 50.0, 40.0, 0.9); // 81.0
    try model.addCategory("math", 45.0, 45.0, 0.8); // 72.0
    try model.addCategory("reasoning", 60.0, 30.0, 0.95); // 85.5
    
    const best = model.getBestCategory();
    try std.testing.expect(best != null);
    try std.testing.expect(std.mem.eql(u8, best.?, "reasoning"));
}

test "MultiCategoryRegistry - register and retrieve" {
    const allocator = std.testing.allocator;
    
    const registry = try MultiCategoryRegistry.init(allocator);
    defer registry.deinit();
    
    const model1 = try MultiCategoryModel.init(allocator, "model-1");
    try model1.addCategory("code", 50.0, 40.0, 0.9);
    try registry.registerModel(model1);
    
    const retrieved = registry.getModel("model-1");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqual(@as(usize, 1), registry.count());
}

test "MultiCategoryRegistry - get models for category" {
    const allocator = std.testing.allocator;
    
    const registry = try MultiCategoryRegistry.init(allocator);
    defer registry.deinit();
    
    const model1 = try MultiCategoryModel.init(allocator, "model-1");
    try model1.addCategory("code", 50.0, 40.0, 0.9);
    try model1.addCategory("math", 45.0, 35.0, 0.8);
    try registry.registerModel(model1);
    
    const model2 = try MultiCategoryModel.init(allocator, "model-2");
    try model2.addCategory("code", 55.0, 45.0, 0.95);
    try registry.registerModel(model2);
    
    const code_models = try registry.getModelsForCategory(allocator, "code");
    defer {
        for (code_models) |m| allocator.free(m);
        allocator.free(code_models);
    }
    
    try std.testing.expectEqual(@as(usize, 2), code_models.len);
    
    const math_models = try registry.getModelsForCategory(allocator, "math");
    defer {
        for (math_models) |m| allocator.free(m);
        allocator.free(math_models);
    }
    
    try std.testing.expectEqual(@as(usize, 1), math_models.len);
}

test "MultiCategoryRegistry - select best for category" {
    const allocator = std.testing.allocator;
    
    const registry = try MultiCategoryRegistry.init(allocator);
    defer registry.deinit();
    
    const model1 = try MultiCategoryModel.init(allocator, "model-1");
    try model1.addCategory("code", 50.0, 40.0, 0.9); // 81.0
    try registry.registerModel(model1);
    
    const model2 = try MultiCategoryModel.init(allocator, "model-2");
    try model2.addCategory("code", 55.0, 45.0, 0.95); // 95.0
    try registry.registerModel(model2);
    
    const best = registry.selectBestForCategory("code");
    try std.testing.expect(best != null);
    try std.testing.expect(std.mem.eql(u8, best.?.model_name, "model-2"));
}

test "MultiCategoryBuilder - build with defaults" {
    const allocator = std.testing.allocator;
    
    var builder = MultiCategoryBuilder.init(allocator);
    
    const categories = [_][]const u8{ "code", "math", "reasoning" };
    const model = try builder.buildWithDefaults("test-model", &categories, 50.0, 40.0);
    defer model.deinit();
    
    try std.testing.expectEqual(@as(usize, 3), model.category_scores.len);
    
    // All should have 0.9 confidence
    for (model.category_scores) |score| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.9), score.confidence, 0.01);
    }
}

test "MultiCategoryBuilder - build with custom confidence" {
    const allocator = std.testing.allocator;
    
    var builder = MultiCategoryBuilder.init(allocator);
    
    const configs = [_]MultiCategoryBuilder.CategoryConfig{
        .{ .category = "code", .base_score = 50.0, .benchmark_score = 40.0, .confidence = 0.95 },
        .{ .category = "math", .base_score = 45.0, .benchmark_score = 35.0, .confidence = 0.7 },
    };
    
    const model = try builder.buildWithConfidence("test-model", &configs);
    defer model.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), model.category_scores.len);
    
    const code_score = model.scoreForCategory("code");
    try std.testing.expect(code_score != null);
    // (50 + 40) * 0.95 = 85.5
    try std.testing.expectApproxEqAbs(@as(f32, 85.5), code_score.?, 0.01);
}
