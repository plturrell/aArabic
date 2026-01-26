//! nLocalModels Orchestration Integration
//! 
//! This module provides nFlow with access to the centralized orchestration system
//! located in nLocalModels. This ensures a single source of truth for model selection
//! and routing logic.
//!
//! Architecture:
//! - nLocalModels/orchestration: Central orchestration system (single source of truth)
//! - nFlow/orchestration: Workflow-specific integration (this file)
//! - Both layers share: MODEL_REGISTRY.json, task_categories.json, model_selector.zig

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Import from nLocalModels orchestration via build wiring
const model_selector = @import("nlocalmodels_model_selector");
pub const ModelSelector = model_selector.ModelSelector;
pub const SelectionConstraints = model_selector.SelectionConstraints;
pub const SelectionResult = model_selector.SelectionResult;
pub const Model = model_selector.Model;
pub const TaskCategory = model_selector.TaskCategory;

/// Default paths to shared resources
pub const DEFAULT_REGISTRY_PATH = "vendor/layerModels/MODEL_REGISTRY.json";
pub const DEFAULT_CATEGORIES_PATH = "src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json";

/// Initialize model selector with default paths
pub fn initDefault(allocator: Allocator) !*ModelSelector {
    return try ModelSelector.init(
        allocator,
        DEFAULT_REGISTRY_PATH,
        DEFAULT_CATEGORIES_PATH,
    );
}

/// Workflow-specific helper: Select model for nFlow node
pub fn selectForWorkflow(
    selector: *ModelSelector,
    task_category: []const u8,
    workflow_gpu_limit_mb: ?usize,
) !SelectionResult {
    const constraints = SelectionConstraints{
        .max_gpu_memory_mb = workflow_gpu_limit_mb,
        .required_agent_type = "inference",
    };
    
    return try selector.selectModel(task_category, constraints);
}

/// Workflow-specific helper: Select model with tool capability
pub fn selectToolModel(
    selector: *ModelSelector,
    task_category: []const u8,
    workflow_gpu_limit_mb: ?usize,
) !SelectionResult {
    const constraints = SelectionConstraints{
        .max_gpu_memory_mb = workflow_gpu_limit_mb,
        .required_agent_type = "tool",
    };
    
    return try selector.selectModel(task_category, constraints);
}

/// Get recommended GPU profile for current hardware
pub fn getGPUProfile() GPUProfile {
    // TODO: Detect actual GPU hardware
    // For now, default to T4
    return .t4_16gb;
}

pub const GPUProfile = enum {
    t4_16gb,
    a100_40gb,
    a100_80gb,
    cpu_only,
    
    pub fn getMemoryLimitMB(self: GPUProfile) usize {
        return switch (self) {
            .t4_16gb => 14 * 1024,
            .a100_40gb => 38 * 1024,
            .a100_80gb => 76 * 1024,
            .cpu_only => 16 * 1024, // RAM limit
        };
    }
};

// Re-export for convenience (explicit aliases above)

test "nLocalModels integration" {
    const allocator = std.testing.allocator;
    
    const selector = try initDefault(allocator);
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    try std.testing.expect(selector.models.items.len > 0);
    try std.testing.expect(selector.categories.count() > 0);
}

test "workflow model selection" {
    const allocator = std.testing.allocator;
    
    const selector = try initDefault(allocator);
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    const gpu_profile = getGPUProfile();
    var result = try selectForWorkflow(
        selector,
        "code",
        gpu_profile.getMemoryLimitMB(),
    );
    defer result.deinit(allocator);
    
    try std.testing.expect(result.score > 0);
}
