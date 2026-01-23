//! Test and Example for Model Selection
//! Demonstrates intelligent model routing based on task categories

const std = @import("std");
const ModelSelector = @import("model_selector.zig").ModelSelector;
const SelectionConstraints = @import("model_selector.zig").SelectionConstraints;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.log.info("=== Model Orchestration Selection Test ===\n", .{});
    
    // Initialize model selector
    const selector = try ModelSelector.init(
        allocator,
        "vendor/layerModels/MODEL_REGISTRY.json",
        "src/serviceCore/nOpenaiServer/orchestration/catalog/task_categories.json",
    );
    defer selector.deinit();
    
    // Load registry and categories
    std.log.info("Loading MODEL_REGISTRY.json...", .{});
    try selector.loadRegistry();
    std.log.info("Loaded {d} models\n", .{selector.models.items.len});
    
    std.log.info("Loading task_categories.json...", .{});
    try selector.loadCategories();
    std.log.info("Loaded {d} categories\n", .{selector.categories.count()});
    
    // Test 1: Code generation task with T4 GPU constraints
    try testCodeGeneration(selector);
    
    // Test 2: Translation task with tool requirement
    try testTranslation(selector);
    
    // Test 3: Small model preference
    try testSmallModelPreference(selector);
    
    std.log.info("\n=== All Tests Complete ===", .{});
}

fn testCodeGeneration(selector: *ModelSelector) !void {
    std.log.info("\n--- Test 1: Code Generation (T4 GPU, 14GB constraint) ---", .{});
    
    const constraints = SelectionConstraints{
        .max_gpu_memory_mb = 14 * 1024, // 14GB for T4
        .required_agent_type = "inference",
    };
    
    var result = try selector.selectModel("code", constraints);
    defer result.deinit(selector.allocator);
    
    std.log.info("Selected Model: {s}", .{result.model.name});
    std.log.info("Score: {d:.2}", .{result.score});
    std.log.info("Reason: {s}", .{result.reason});
    std.log.info("GPU Memory: {s}", .{result.model.gpu_memory});
    std.log.info("Categories: {s}", .{result.model.orchestration_categories});
}

fn testTranslation(selector: *ModelSelector) !void {
    std.log.info("\n--- Test 2: Translation (Tool requirement) ---", .{});
    
    const constraints = SelectionConstraints{
        .max_gpu_memory_mb = 16 * 1024, // 16GB
        .required_agent_type = "tool",
    };
    
    var result = try selector.selectModel("relational", constraints);
    defer result.deinit(selector.allocator);
    
    std.log.info("Selected Model: {s}", .{result.model.name});
    std.log.info("Score: {d:.2}", .{result.score});
    std.log.info("Reason: {s}", .{result.reason});
    std.log.info("Agent Types: {s}", .{result.model.agent_types});
}

fn testSmallModelPreference(selector: *ModelSelector) !void {
    std.log.info("\n--- Test 3: Small Model Preference ---", .{});
    
    const preferred = [_][]const u8{"google-gemma-3-270m-it"};
    const constraints = SelectionConstraints{
        .max_gpu_memory_mb = 14 * 1024,
        .preferred_models = &preferred,
    };
    
    var result = try selector.selectModel("code", constraints);
    defer result.deinit(selector.allocator);
    
    std.log.info("Selected Model: {s}", .{result.model.name});
    std.log.info("Score: {d:.2} (bonus for preferred model)", .{result.score});
    std.log.info("Reason: {s}", .{result.reason});
}
