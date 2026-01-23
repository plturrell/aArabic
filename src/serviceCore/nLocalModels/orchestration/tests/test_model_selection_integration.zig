//! Integration Tests for Model Selection with Actual MODEL_REGISTRY.json
//! 
//! These tests validate the complete model selection system using real data:
//! - MODEL_REGISTRY.json (actual model metadata)
//! - task_categories.json (orchestration categories)
//! - All Phase 5 enhancements (benchmarks, multi-category, GPU-aware)

const std = @import("std");
const testing = std.testing;
const ModelSelector = @import("../../src/serviceCore/nLocalModels/orchestration/model_selector.zig").ModelSelector;
const SelectionConstraints = @import("../../src/serviceCore/nLocalModels/orchestration/model_selector.zig").SelectionConstraints;

// Test configuration
const REGISTRY_PATH = "vendor/layerModels/MODEL_REGISTRY.json";
const CATEGORIES_PATH = "src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json";

test "Integration: Load actual MODEL_REGISTRY.json" {
    const allocator = testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        REGISTRY_PATH,
        CATEGORIES_PATH,
    );
    defer selector.deinit();
    
    // Load registry and categories
    try selector.loadRegistry();
    try selector.loadCategories();
    
    // Verify models loaded
    try testing.expect(selector.models.items.len > 0);
    std.debug.print("\n✓ Loaded {d} models from registry\n", .{selector.models.items.len});
    
    // Verify categories loaded
    try testing.expect(selector.categories.count() > 0);
    std.debug.print("✓ Loaded {d} categories\n", .{selector.categories.count()});
    
    // List all loaded models
    std.debug.print("\nLoaded Models:\n", .{});
    for (selector.models.items) |model| {
        std.debug.print("  - {s} (GPU: {s})\n", .{model.name, model.gpu_memory});
        
        // Verify model has required fields
        try testing.expect(model.name.len > 0);
        try testing.expect(model.hf_repo.len > 0);
        try testing.expect(model.gpu_memory.len > 0);
    }
    
    // List all categories
    std.debug.print("\nLoaded Categories:\n", .{});
    var cat_iter = selector.categories.iterator();
    while (cat_iter.next()) |entry| {
        std.debug.print("  - {s}: {d} models\n", .{entry.key_ptr.*, entry.value_ptr.models.len});
    }
}

test "Integration: Select model for code task" {
    const allocator = testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        REGISTRY_PATH,
        CATEGORIES_PATH,
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    // Select model for code task with GPU constraint
    const result = try selector.selectModel("code", .{
        .max_gpu_memory_mb = 16 * 1024, // 16GB
    });
    defer result.deinit(allocator);
    
    // Verify result
    try testing.expect(result.model.name.len > 0);
    try testing.expect(result.score > 0.0);
    try testing.expect(result.reason.len > 0);
    
    std.debug.print("\n=== Code Task Selection ===\n", .{});
    std.debug.print("Selected Model: {s}\n", .{result.model.name});
    std.debug.print("Score: {d:.2}\n", .{result.score});
    std.debug.print("Reason: {s}\n", .{result.reason});
    std.debug.print("GPU Memory: {s}\n", .{result.model.gpu_memory});
    
    // Verify model is suitable for code
    var has_code_category = false;
    for (result.model.orchestration_categories) |cat| {
        if (std.mem.eql(u8, cat, "code")) {
            has_code_category = true;
            break;
        }
    }
    try testing.expect(has_code_category);
}

test "Integration: Select model with benchmark scoring enabled" {
    const allocator = testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        REGISTRY_PATH,
        CATEGORIES_PATH,
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    // Enable benchmark scoring
    try selector.enableBenchmarkScoring();
    
    const result = try selector.selectModel("code", .{
        .max_gpu_memory_mb = 16 * 1024,
    });
    defer result.deinit(allocator);
    
    std.debug.print("\n=== Benchmark-Enhanced Selection ===\n", .{});
    std.debug.print("Selected Model: {s}\n", .{result.model.name});
    std.debug.print("Score: {d:.2} (includes benchmark scoring)\n", .{result.score});
    std.debug.print("Reason: {s}\n", .{result.reason});
    
    try testing.expect(result.score > 0.0);
}

test "Integration: Multi-category model selection" {
    const allocator = testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        REGISTRY_PATH,
        CATEGORIES_PATH,
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    try selector.enableBenchmarkScoring();
    
    // Build multi-category registry
    try selector.buildMultiCategoryRegistry();
    
    // Test multi-category selection
    const result = try selector.selectModelMultiCategory("code", .{
        .max_gpu_memory_mb = 16 * 1024,
    });
    defer result.deinit(allocator);
    
    std.debug.print("\n=== Multi-Category Selection ===\n", .{});
    std.debug.print("Selected Model: {s}\n", .{result.model.name});
    std.debug.print("Score: {d:.2} (confidence-weighted)\n", .{result.score});
    std.debug.print("Reason: {s}\n", .{result.reason});
    
    // Verify score is reasonable for multi-category
    try testing.expect(result.score > 0.0);
    try testing.expect(result.score <= 100.0);
    
    // Check if model serves multiple categories
    if (result.model.orchestration_categories.len > 1) {
        std.debug.print("Multi-category model: supports {d} categories\n", .{
            result.model.orchestration_categories.len
        });
    }
}

test "Integration: Select models for different categories" {
    const allocator = testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        REGISTRY_PATH,
        CATEGORIES_PATH,
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    std.debug.print("\n=== Cross-Category Selection Test ===\n", .{});
    
    // Test multiple categories
    const test_categories = [_][]const u8{ "code", "relational" };
    
    for (test_categories) |category| {
        // Check if category exists
        if (selector.categories.get(category)) |cat_info| {
            if (cat_info.models.len > 0) {
                const result = selector.selectModel(category, .{
                    .max_gpu_memory_mb = 16 * 1024,
                }) catch |err| {
                    std.debug.print("⚠ Could not select for {s}: {any}\n", .{category, err});
                    continue;
                };
                defer result.deinit(allocator);
                
                std.debug.print("\n{s} category:\n", .{category});
                std.debug.print("  Model: {s}\n", .{result.model.name});
                std.debug.print("  Score: {d:.2}\n", .{result.score});
                
                try testing.expect(result.score > 0.0);
            } else {
                std.debug.print("⚠ {s} category has no models\n", .{category});
            }
        } else {
            std.debug.print("⚠ {s} category not found\n", .{category});
        }
    }
}

test "Integration: Constraint validation" {
    const allocator = testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        REGISTRY_PATH,
        CATEGORIES_PATH,
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    std.debug.print("\n=== Constraint Validation Test ===\n", .{});
    
    // Test with very restrictive GPU memory (should exclude larger models)
    const small_gpu_result = try selector.selectModel("code", .{
        .max_gpu_memory_mb = 4 * 1024, // Only 4GB
    });
    defer small_gpu_result.deinit(allocator);
    
    const model_mem = try small_gpu_result.model.getGpuMemoryMB();
    std.debug.print("Selected for 4GB constraint: {s} ({d}MB)\n", .{
        small_gpu_result.model.name,
        model_mem
    });
    
    // Verify model respects constraint
    try testing.expect(model_mem <= 4 * 1024);
    
    // Test with larger GPU memory
    const large_gpu_result = try selector.selectModel("code", .{
        .max_gpu_memory_mb = 32 * 1024, // 32GB
    });
    defer large_gpu_result.deinit(allocator);
    
    std.debug.print("Selected for 32GB constraint: {s}\n", .{large_gpu_result.model.name});
}

test "Integration: Performance comparison across selection methods" {
    const allocator = testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        REGISTRY_PATH,
        CATEGORIES_PATH,
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    std.debug.print("\n=== Selection Method Comparison ===\n", .{});
    
    const constraints = SelectionConstraints{
        .max_gpu_memory_mb = 16 * 1024,
    };
    
    // Method 1: Basic selection
    const basic_start = std.time.milliTimestamp();
    const basic_result = try selector.selectModel("code", constraints);
    defer basic_result.deinit(allocator);
    const basic_duration = std.time.milliTimestamp() - basic_start;
    
    std.debug.print("\n1. Basic Selection:\n", .{});
    std.debug.print("   Model: {s}\n", .{basic_result.model.name});
    std.debug.print("   Score: {d:.2}\n", .{basic_result.score});
    std.debug.print("   Duration: {d}ms\n", .{basic_duration});
    
    // Method 2: With benchmark scoring
    try selector.enableBenchmarkScoring();
    const bench_start = std.time.milliTimestamp();
    const bench_result = try selector.selectModel("code", constraints);
    defer bench_result.deinit(allocator);
    const bench_duration = std.time.milliTimestamp() - bench_start;
    
    std.debug.print("\n2. With Benchmark Scoring:\n", .{});
    std.debug.print("   Model: {s}\n", .{bench_result.model.name});
    std.debug.print("   Score: {d:.2}\n", .{bench_result.score});
    std.debug.print("   Duration: {d}ms\n", .{bench_duration});
    
    // Method 3: Multi-category
    try selector.buildMultiCategoryRegistry();
    const multi_start = std.time.milliTimestamp();
    const multi_result = try selector.selectModelMultiCategory("code", constraints);
    defer multi_result.deinit(allocator);
    const multi_duration = std.time.milliTimestamp() - multi_start;
    
    std.debug.print("\n3. Multi-Category Selection:\n", .{});
    std.debug.print("   Model: {s}\n", .{multi_result.model.name});
    std.debug.print("   Score: {d:.2}\n", .{multi_result.score});
    std.debug.print("   Duration: {d}ms\n", .{multi_duration});
    
    // Verify all methods complete in reasonable time (<100ms)
    try testing.expect(basic_duration < 100);
    try testing.expect(bench_duration < 100);
    try testing.expect(multi_duration < 100);
}

test "Integration: Registry data validation" {
    const allocator = testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        REGISTRY_PATH,
        CATEGORIES_PATH,
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    std.debug.print("\n=== Registry Data Validation ===\n", .{});
    
    var valid_count: usize = 0;
    var invalid_count: usize = 0;
    
    for (selector.models.items) |model| {
        // Validate GPU memory format
        const mem = model.getGpuMemoryMB() catch |err| {
            std.debug.print("⚠ Invalid GPU memory for {s}: {any}\n", .{model.name, err});
            invalid_count += 1;
            continue;
        };
        
        // Validate reasonable memory range (1GB to 96GB)
        if (mem < 1024 or mem > 96 * 1024) {
            std.debug.print("⚠ Suspicious GPU memory for {s}: {d}MB\n", .{model.name, mem});
            invalid_count += 1;
            continue;
        }
        
        // Validate has at least one category
        if (model.orchestration_categories.len == 0) {
            std.debug.print("⚠ No categories for {s}\n", .{model.name});
            invalid_count += 1;
            continue;
        }
        
        valid_count += 1;
    }
    
    std.debug.print("\nValidation Results:\n", .{});
    std.debug.print("  Valid models: {d}\n", .{valid_count});
    std.debug.print("  Invalid models: {d}\n", .{invalid_count});
    std.debug.print("  Total models: {d}\n", .{selector.models.items.len});
    
    // At least 50% of models should be valid
    const valid_percentage = (@as(f64, @floatFromInt(valid_count)) / 
                             @as(f64, @floatFromInt(selector.models.items.len))) * 100.0;
    std.debug.print("  Valid percentage: {d:.1}%\n", .{valid_percentage});
    
    try testing.expect(valid_percentage >= 50.0);
}
