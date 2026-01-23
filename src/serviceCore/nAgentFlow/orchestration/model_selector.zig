//! Model Selection Module
//! Intelligent model selection based on task categories and constraints
//!
//! Features:
//! - Load MODEL_REGISTRY.json and task_categories.json
//! - Select models based on task category
//! - Apply GPU memory constraints
//! - Support fallback strategies
//! - Track model performance

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Model metadata from registry
pub const Model = struct {
    name: []const u8,
    hf_repo: []const u8,
    orchestration_categories: [][]const u8,
    agent_types: [][]const u8,
    gpu_memory: []const u8,
    specifications: ?Specifications = null,
    benchmarks: ?std.StringHashMap(BenchmarkScore) = null,
    
    pub const Specifications = struct {
        parameters: ?[]const u8 = null,
        context_length: ?[]const u8 = null,
        architecture: ?[]const u8 = null,
        quantization: ?[]const u8 = null,
    };
    
    pub const BenchmarkScore = struct {
        score: f64,
        date: []const u8,
    };
    
    pub fn deinit(self: *Model, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.hf_repo);
        
        for (self.orchestration_categories) |cat| {
            allocator.free(cat);
        }
        allocator.free(self.orchestration_categories);
        
        for (self.agent_types) |agent| {
            allocator.free(agent);
        }
        allocator.free(self.agent_types);
        
        allocator.free(self.gpu_memory);
        
        if (self.specifications) |*specs| {
            if (specs.parameters) |p| allocator.free(p);
            if (specs.context_length) |c| allocator.free(c);
            if (specs.architecture) |a| allocator.free(a);
            if (specs.quantization) |q| allocator.free(q);
        }
        
        if (self.benchmarks) |*bm| {
            var it = bm.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.date);
            }
            bm.deinit();
        }
    }
    
    /// Parse GPU memory string to MB
    pub fn getGpuMemoryMB(self: *const Model) !usize {
        // Parse strings like "2GB", "16GB", "48GB"
        var mem_str = self.gpu_memory;
        
        // Remove "GB" suffix
        if (std.mem.endsWith(u8, mem_str, "GB")) {
            mem_str = mem_str[0 .. mem_str.len - 2];
        }
        
        const gb = try std.fmt.parseInt(usize, mem_str, 10);
        return gb * 1024; // Convert to MB
    }
};

/// Task category definition
pub const TaskCategory = struct {
    id: []const u8,
    name: []const u8,
    description: []const u8,
    models: [][]const u8,
    benchmarks: []Benchmark,
    
    pub const Benchmark = struct {
        name: []const u8,
        description: []const u8,
        metric: []const u8,
    };
    
    pub fn deinit(self: *TaskCategory, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.description);
        
        for (self.models) |model| {
            allocator.free(model);
        }
        allocator.free(self.models);
        
        for (self.benchmarks) |*benchmark| {
            allocator.free(benchmark.name);
            allocator.free(benchmark.description);
            allocator.free(benchmark.metric);
        }
        allocator.free(self.benchmarks);
    }
};

/// Model selection constraints
pub const SelectionConstraints = struct {
    /// Maximum GPU memory available (MB)
    max_gpu_memory_mb: ?usize = null,
    
    /// Required agent type (inference, tool, orchestrator)
    required_agent_type: ?[]const u8 = null,
    
    /// Minimum benchmark score (if available)
    min_benchmark_score: ?f64 = null,
    
    /// Preferred models (prioritize these)
    preferred_models: ?[][]const u8 = null,
    
    /// Exclude specific models
    excluded_models: ?[][]const u8 = null,
};

/// Model selection result
pub const SelectionResult = struct {
    model: *const Model,
    score: f64,
    reason: []const u8,
    
    pub fn deinit(self: *SelectionResult, allocator: Allocator) void {
        allocator.free(self.reason);
    }
};

/// Model Selector - Main orchestration engine
pub const ModelSelector = struct {
    allocator: Allocator,
    models: std.ArrayList(Model),
    categories: std.StringHashMap(TaskCategory),
    registry_path: []const u8,
    categories_path: []const u8,
    
    pub fn init(
        allocator: Allocator,
        registry_path: []const u8,
        categories_path: []const u8,
    ) !*ModelSelector {
        const selector = try allocator.create(ModelSelector);
        
        selector.* = .{
            .allocator = allocator,
            .models = std.ArrayList(Model).init(allocator),
            .categories = std.StringHashMap(TaskCategory).init(allocator),
            .registry_path = try allocator.dupe(u8, registry_path),
            .categories_path = try allocator.dupe(u8, categories_path),
        };
        
        return selector;
    }
    
    pub fn deinit(self: *ModelSelector) void {
        for (self.models.items) |*model| {
            model.deinit(self.allocator);
        }
        self.models.deinit();
        
        var it = self.categories.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.categories.deinit();
        
        self.allocator.free(self.registry_path);
        self.allocator.free(self.categories_path);
        self.allocator.destroy(self);
    }
    
    /// Load MODEL_REGISTRY.json
    pub fn loadRegistry(self: *ModelSelector) !void {
        const file = try std.fs.cwd().openFile(self.registry_path, .{});
        defer file.close();
        
        const content = try file.readToEndAlloc(self.allocator, 10 * 1024 * 1024); // 10MB max
        defer self.allocator.free(content);
        
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            content,
            .{},
        );
        defer parsed.deinit();
        
        if (parsed.value != .object) return error.InvalidRegistryFormat;
        
        const models_array = parsed.value.object.get("models") orelse return error.MissingModelsArray;
        if (models_array != .array) return error.InvalidModelsArray;
        
        for (models_array.array.items) |model_value| {
            if (model_value != .object) continue;
            
            const model = try self.parseModel(model_value.object);
            try self.models.append(model);
        }
    }
    
    fn parseModel(self: *ModelSelector, obj: std.json.ObjectMap) !Model {
        const name = obj.get("name") orelse return error.MissingName;
        const hf_repo = obj.get("hf_repo") orelse return error.MissingHfRepo;
        const gpu_memory = obj.get("gpu_memory") orelse return error.MissingGpuMemory;
        
        var model = Model{
            .name = try self.allocator.dupe(u8, name.string),
            .hf_repo = try self.allocator.dupe(u8, hf_repo.string),
            .orchestration_categories = &[_][]const u8{},
            .agent_types = &[_][]const u8{},
            .gpu_memory = try self.allocator.dupe(u8, gpu_memory.string),
        };
        
        // Parse orchestration_categories
        if (obj.get("orchestration_categories")) |cats| {
            if (cats == .array) {
                var categories = std.ArrayList([]const u8).init(self.allocator);
                for (cats.array.items) |cat| {
                    if (cat == .string) {
                        try categories.append(try self.allocator.dupe(u8, cat.string));
                    }
                }
                model.orchestration_categories = try categories.toOwnedSlice();
            }
        }
        
        // Parse agent_types
        if (obj.get("agent_types")) |types| {
            if (types == .array) {
                var agent_types = std.ArrayList([]const u8).init(self.allocator);
                for (types.array.items) |agent_type| {
                    if (agent_type == .string) {
                        try agent_types.append(try self.allocator.dupe(u8, agent_type.string));
                    }
                }
                model.agent_types = try agent_types.toOwnedSlice();
            }
        }
        
        return model;
    }
    
    /// Load task_categories.json
    pub fn loadCategories(self: *ModelSelector) !void {
        const file = try std.fs.cwd().openFile(self.categories_path, .{});
        defer file.close();
        
        const content = try file.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
        defer self.allocator.free(content);
        
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            content,
            .{},
        );
        defer parsed.deinit();
        
        if (parsed.value != .object) return error.InvalidCategoriesFormat;
        
        const categories_obj = parsed.value.object.get("categories") orelse return error.MissingCategories;
        if (categories_obj != .object) return error.InvalidCategories;
        
        var it = categories_obj.object.iterator();
        while (it.next()) |entry| {
            const category = try self.parseCategory(entry.key_ptr.*, entry.value_ptr.*);
            try self.categories.put(
                try self.allocator.dupe(u8, entry.key_ptr.*),
                category,
            );
        }
    }
    
    fn parseCategory(self: *ModelSelector, id: []const u8, value: std.json.Value) !TaskCategory {
        if (value != .object) return error.InvalidCategory;
        
        const name = value.object.get("name") orelse return error.MissingName;
        const description = value.object.get("description") orelse return error.MissingDescription;
        
        var category = TaskCategory{
            .id = try self.allocator.dupe(u8, id),
            .name = try self.allocator.dupe(u8, name.string),
            .description = try self.allocator.dupe(u8, description.string),
            .models = &[_][]const u8{},
            .benchmarks = &[_]TaskCategory.Benchmark{},
        };
        
        // Parse models
        if (value.object.get("models")) |models| {
            if (models == .array) {
                var model_list = std.ArrayList([]const u8).init(self.allocator);
                for (models.array.items) |model| {
                    if (model == .string) {
                        try model_list.append(try self.allocator.dupe(u8, model.string));
                    }
                }
                category.models = try model_list.toOwnedSlice();
            }
        }
        
        return category;
    }
    
    /// Select best model for a task category
    pub fn selectModel(
        self: *ModelSelector,
        task_category: []const u8,
        constraints: SelectionConstraints,
    ) !SelectionResult {
        // Get category
        const category = self.categories.get(task_category) orelse {
            return error.UnknownCategory;
        };
        
        var best_model: ?*const Model = null;
        var best_score: f64 = 0.0;
        var selection_reason = std.ArrayList(u8).init(self.allocator);
        defer selection_reason.deinit();
        
        // Iterate through models in category
        for (category.models) |model_name| {
            const model = self.findModel(model_name) orelse continue;
            
            // Apply constraints
            if (!try self.meetsConstraints(model, constraints)) {
                continue;
            }
            
            // Calculate score
            const score = try self.calculateScore(model, task_category, constraints);
            
            if (score > best_score) {
                best_score = score;
                best_model = model;
            }
        }
        
        if (best_model == null) {
            // Try fallback
            return self.selectFallbackModel(constraints);
        }
        
        try selection_reason.appendSlice("Selected based on task category '");
        try selection_reason.appendSlice(task_category);
        try selection_reason.appendSlice("' with score ");
        
        var score_buf: [32]u8 = undefined;
        const score_str = try std.fmt.bufPrint(&score_buf, "{d:.2}", .{best_score});
        try selection_reason.appendSlice(score_str);
        
        return SelectionResult{
            .model = best_model.?,
            .score = best_score,
            .reason = try selection_reason.toOwnedSlice(),
        };
    }
    
    fn findModel(self: *ModelSelector, name: []const u8) ?*const Model {
        for (self.models.items) |*model| {
            if (std.mem.eql(u8, model.name, name)) {
                return model;
            }
        }
        return null;
    }
    
    fn meetsConstraints(
        self: *ModelSelector,
        model: *const Model,
        constraints: SelectionConstraints,
    ) !bool {
        _ = self;
        
        // Check GPU memory
        if (constraints.max_gpu_memory_mb) |max_mem| {
            const model_mem = try model.getGpuMemoryMB();
            if (model_mem > max_mem) {
                return false;
            }
        }
        
        // Check required agent type
        if (constraints.required_agent_type) |required_type| {
            var has_type = false;
            for (model.agent_types) |agent_type| {
                if (std.mem.eql(u8, agent_type, required_type)) {
                    has_type = true;
                    break;
                }
            }
            if (!has_type) return false;
        }
        
        // Check excluded models
        if (constraints.excluded_models) |excluded| {
            for (excluded) |excluded_name| {
                if (std.mem.eql(u8, model.name, excluded_name)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    fn calculateScore(
        self: *ModelSelector,
        model: *const Model,
        task_category: []const u8,
        constraints: SelectionConstraints,
    ) !f64 {
        _ = self;
        _ = task_category;
        
        var score: f64 = 50.0; // Base score
        
        // Bonus for preferred models
        if (constraints.preferred_models) |preferred| {
            for (preferred) |preferred_name| {
                if (std.mem.eql(u8, model.name, preferred_name)) {
                    score += 30.0;
                    break;
                }
            }
        }
        
        // Bonus for smaller models (faster inference)
        const model_mem = try model.getGpuMemoryMB();
        if (model_mem < 4096) { // Less than 4GB
            score += 20.0;
        } else if (model_mem < 8192) { // Less than 8GB
            score += 10.0;
        }
        
        // TODO: Add benchmark-based scoring when available
        
        return score;
    }
    
    fn selectFallbackModel(
        self: *ModelSelector,
        constraints: SelectionConstraints,
    ) !SelectionResult {
        // Try to find any model that meets constraints
        for (self.models.items) |*model| {
            if (try self.meetsConstraints(model, constraints)) {
                return SelectionResult{
                    .model = model,
                    .score = 25.0,
                    .reason = try self.allocator.dupe(u8, "Fallback model selection"),
                };
            }
        }
        
        return error.NoSuitableModel;
    }
};

// Tests
test "ModelSelector initialization" {
    const allocator = std.testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        "vendor/layerModels/MODEL_REGISTRY.json",
        "src/serviceCore/nOpenaiServer/orchestration/catalog/task_categories.json",
    );
    defer selector.deinit();
    
    try std.testing.expect(selector.models.items.len == 0);
    try std.testing.expect(selector.categories.count() == 0);
}

test "Model GPU memory parsing" {
    const model = Model{
        .name = "test",
        .hf_repo = "test/test",
        .orchestration_categories = &[_][]const u8{},
        .agent_types = &[_][]const u8{},
        .gpu_memory = "16GB",
    };
    
    const mem_mb = try model.getGpuMemoryMB();
    try std.testing.expectEqual(@as(usize, 16 * 1024), mem_mb);
}
