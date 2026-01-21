const std = @import("std");
const json = std.json;

/// DSPy-style Prompt Optimizer in Zig
/// Based on MIPROv2 and Bootstrap FewShot principles

// ============================================================================
// Core Types
// ============================================================================

pub const Field = struct {
    name: []const u8,
    desc: []const u8,
    prefix: []const u8 = "",
    
    pub fn format(self: Field, value: []const u8) ![]u8 {
        if (self.prefix.len > 0) {
            return try std.fmt.allocPrint(
                std.heap.page_allocator,
                "{s} {s}",
                .{ self.prefix, value }
            );
        }
        return try std.heap.page_allocator.dupe(u8, value);
    }
};

pub const Signature = struct {
    inputs: []const Field,
    outputs: []const Field,
    instructions: []const u8 = "",
    
    pub fn build_prompt(
        self: Signature,
        allocator: std.mem.Allocator,
        input_values: anytype,
    ) ![]u8 {
        var prompt = std.ArrayList(u8).init(allocator);
        defer prompt.deinit();
        
        // Add instructions
        if (self.instructions.len > 0) {
            try prompt.appendSlice(self.instructions);
            try prompt.appendSlice("\n\n");
        }
        
        // Add input fields
        inline for (self.inputs) |field| {
            const value = @field(input_values, field.name);
            const formatted = try field.format(value);
            defer allocator.free(formatted);
            try prompt.appendSlice(formatted);
            try prompt.appendSlice("\n");
        }
        
        // Add output field prefix
        if (self.outputs.len > 0) {
            const out_field = self.outputs[0];
            if (out_field.prefix.len > 0) {
                try prompt.appendSlice(out_field.prefix);
                try prompt.appendSlice(" ");
            }
        }
        
        return try prompt.toOwnedSlice();
    }
};

pub const Example = struct {
    inputs: std.StringHashMap([]const u8),
    outputs: std.StringHashMap([]const u8),
    score: f32 = 0.0,
    
    pub fn init(allocator: std.mem.Allocator) Example {
        return Example{
            .inputs = std.StringHashMap([]const u8).init(allocator),
            .outputs = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *Example) void {
        self.inputs.deinit();
        self.outputs.deinit();
    }
};

// ============================================================================
// Modules (Composable Prompt Components)
// ============================================================================

pub const Module = struct {
    signature: Signature,
    examples: []Example = &[_]Example{},
    temperature: f32 = 0.7,
    
    pub fn forward(
        self: *Module,
        allocator: std.mem.Allocator,
        input: anytype,
        llm_fn: *const fn([]const u8) anyerror![]u8,
    ) ![]u8 {
        // Build prompt from signature
        const prompt = try self.signature.build_prompt(allocator, input);
        defer allocator.free(prompt);
        
        // Call LLM
        return try llm_fn(prompt);
    }
};

pub const ChainOfThought = struct {
    base_module: Module,
    reasoning_field: Field,
    
    pub fn init(sig: Signature) ChainOfThought {
        // Extend signature with reasoning step
        const reasoning = Field{
            .name = "reasoning",
            .desc = "step-by-step reasoning",
            .prefix = "Reasoning:",
        };
        
        return ChainOfThought{
            .base_module = Module{ .signature = sig },
            .reasoning_field = reasoning,
        };
    }
    
    pub fn forward(
        self: *ChainOfThought,
        allocator: std.mem.Allocator,
        input: anytype,
        llm_fn: *const fn([]const u8) anyerror![]u8,
    ) ![]u8 {
        // Build prompt with reasoning step
        var prompt = std.ArrayList(u8).init(allocator);
        defer prompt.deinit();
        
        if (self.base_module.signature.instructions.len > 0) {
            try prompt.appendSlice(self.base_module.signature.instructions);
            try prompt.appendSlice("\n\n");
        }
        
        try prompt.appendSlice("Let's think step by step:\n\n");
        
        // Add input
        const base_prompt = try self.base_module.signature.build_prompt(allocator, input);
        defer allocator.free(base_prompt);
        try prompt.appendSlice(base_prompt);
        
        const final_prompt = try prompt.toOwnedSlice();
        defer allocator.free(final_prompt);
        
        return try llm_fn(final_prompt);
    }
};

// ============================================================================
// Metrics
// ============================================================================

pub const MetricFn = *const fn([]const u8, []const u8) f32;

pub fn exact_match(prediction: []const u8, gold: []const u8) f32 {
    return if (std.mem.eql(u8, prediction, gold)) 1.0 else 0.0;
}

pub fn f1_score(prediction: []const u8, gold: []const u8) f32 {
    // Simple token-level F1
    var pred_tokens = std.mem.tokenize(u8, prediction, " \n\t");
    var gold_tokens = std.mem.tokenize(u8, gold, " \n\t");
    
    var pred_set = std.StringHashMap(void).init(std.heap.page_allocator);
    defer pred_set.deinit();
    
    var gold_set = std.StringHashMap(void).init(std.heap.page_allocator);
    defer gold_set.deinit();
    
    while (pred_tokens.next()) |token| {
        pred_set.put(token, {}) catch {};
    }
    
    while (gold_tokens.next()) |token| {
        gold_set.put(token, {}) catch {};
    }
    
    var tp: u32 = 0;
    var iter = pred_set.iterator();
    while (iter.next()) |entry| {
        if (gold_set.contains(entry.key_ptr.*)) {
            tp += 1;
        }
    }
    
    const precision = @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(pred_set.count()));
    const recall = @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(gold_set.count()));
    
    if (precision + recall == 0) return 0.0;
    return 2.0 * precision * recall / (precision + recall);
}

// ============================================================================
// Optimizer (MIPROv2-style)
// ============================================================================

pub const OptimizerConfig = struct {
    metric: MetricFn,
    num_candidates: u32 = 10,
    init_temperature: f32 = 1.0,
    max_iterations: u32 = 100,
};

pub const Optimizer = struct {
    config: OptimizerConfig,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, config: OptimizerConfig) Optimizer {
        return Optimizer{
            .config = config,
            .allocator = allocator,
        };
    }
    
    /// Optimize prompt module using training data
    pub fn optimize(
        self: *Optimizer,
        module: *Module,
        trainset: []Example,
        llm_fn: *const fn([]const u8) anyerror![]u8,
    ) !Module {
        std.debug.print("🔧 Optimizing module with {d} examples\n", .{trainset.len});
        
        // Phase 1: Instruction optimization
        const best_instructions = try self.optimizeInstructions(
            module,
            trainset,
            llm_fn,
        );
        defer self.allocator.free(best_instructions);
        
        // Phase 2: Few-shot example selection
        const best_examples = try self.selectExamples(
            module,
            trainset,
            self.config.num_candidates,
        );
        
        // Create optimized module
        var optimized = Module{
            .signature = module.signature,
            .examples = best_examples,
            .temperature = module.temperature,
        };
        
        // Update instructions
        optimized.signature.instructions = try self.allocator.dupe(u8, best_instructions);
        
        std.debug.print("✅ Optimization complete\n", .{});
        return optimized;
    }
    
    fn optimizeInstructions(
        self: *Optimizer,
        module: *Module,
        trainset: []Example,
        llm_fn: *const fn([]const u8) anyerror![]u8,
    ) ![]u8 {
        // Generate candidate instructions
        const candidates = try self.generateInstructionCandidates(
            module.signature,
            self.config.num_candidates,
        );
        defer {
            for (candidates) |c| self.allocator.free(c);
            self.allocator.free(candidates);
        }
        
        // Evaluate each candidate
        var best_score: f32 = 0.0;
        var best_idx: usize = 0;
        
        for (candidates, 0..) |candidate, idx| {
            var score: f32 = 0.0;
            
            // Test on training set
            for (trainset) |example| {
                // Build prompt with candidate instructions
                var test_sig = module.signature;
                test_sig.instructions = candidate;
                
                const prompt = try test_sig.build_prompt(self.allocator, example.inputs);
                defer self.allocator.free(prompt);
                
                const output = llm_fn(prompt) catch continue;
                defer self.allocator.free(output);
                
                // Get expected output
                const expected = example.outputs.get("answer") orelse continue;
                
                score += self.config.metric(output, expected);
            }
            
            score /= @as(f32, @floatFromInt(trainset.len));
            
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        
        std.debug.print("   Best instruction score: {d:.2}\n", .{best_score});
        return try self.allocator.dupe(u8, candidates[best_idx]);
    }
    
    fn generateInstructionCandidates(
        self: *Optimizer,
        sig: Signature,
        num: u32,
    ) ![][]u8 {
        // For now, use variations of the base instructions
        // In production, would use LLM to generate variants
        const candidates = try self.allocator.alloc([]u8, num);
        
        const base = sig.instructions;
        for (candidates, 0..) |*c, i| {
            if (i == 0) {
                c.* = try self.allocator.dupe(u8, base);
            } else {
                // Simple variations
                c.* = try std.fmt.allocPrint(
                    self.allocator,
                    "{s} Be precise and accurate.",
                    .{base}
                );
            }
        }
        
        return candidates;
    }
    
    fn selectExamples(
        self: *Optimizer,
        _: *Module,
        trainset: []Example,
        k: u32,
    ) ![]Example {
        // Select top-k most informative examples
        // For now, simple random selection
        // In production, would use embedding similarity
        
        const count = @min(k, trainset.len);
        const selected = try self.allocator.alloc(Example, count);
        
        for (selected, 0..) |*ex, i| {
            ex.* = trainset[i];
        }
        
        return selected;
    }
};

// ============================================================================
// Bootstrap FewShot
// ============================================================================

pub const BootstrapFewShot = struct {
    metric: MetricFn,
    max_bootstrapped: u32 = 10,
    
    pub fn bootstrap(
        self: BootstrapFewShot,
        allocator: std.mem.Allocator,
        module: *Module,
        trainset: []Example,
        llm_fn: *const fn([]const u8) anyerror![]u8,
    ) ![]Example {
        // Generate synthetic examples by self-prompting
        var bootstrapped = std.ArrayList(Example).init(allocator);
        defer bootstrapped.deinit();
        
        for (trainset[0..@min(trainset.len, self.max_bootstrapped)]) |example| {
            const prompt = try module.signature.build_prompt(allocator, example.inputs);
            defer allocator.free(prompt);
            
            const output = llm_fn(prompt) catch continue;
            defer allocator.free(output);
            
            // Evaluate quality
            const expected = example.outputs.get("answer") orelse continue;
            const score = self.metric(output, expected);
            
            if (score > 0.5) {  // Only keep good examples
                var new_example = Example.init(allocator);
                try new_example.inputs.putNoClobber("question", try allocator.dupe(u8, example.inputs.get("question").?));
                try new_example.outputs.putNoClobber("answer", try allocator.dupe(u8, output));
                new_example.score = score;
                
                try bootstrapped.append(new_example);
            }
        }
        
        return try bootstrapped.toOwnedSlice();
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn main() !void {
    std.debug.print("=== Prompt Optimizer Test ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    
    // Define signature
    const qa_sig = Signature{
        .inputs = &[_]Field{
            .{ .name = "question", .desc = "question", .prefix = "Q:" },
        },
        .outputs = &[_]Field{
            .{ .name = "answer", .desc = "answer", .prefix = "A:" },
        },
        .instructions = "Answer the question concisely.",
    };
    
    // Create module
    _ = Module{ .signature = qa_sig };
    
    // Create optimizer
    const config = OptimizerConfig{
        .metric = exact_match,
        .num_candidates = 3,
    };
    _ = Optimizer.init(allocator, config);
    
    // Mock training data
    var trainset = [_]Example{
        Example.init(allocator),
    };
    try trainset[0].inputs.put("question", "What is 2+2?");
    try trainset[0].outputs.put("answer", "4");
    
    std.debug.print("✅ Optimizer initialized\n", .{});
    std.debug.print("   Signature: {d} inputs, {d} outputs\n", .{
        qa_sig.inputs.len,
        qa_sig.outputs.len,
    });
    std.debug.print("   Trainset: {d} examples\n", .{trainset.len});
    std.debug.print("   Config: {d} candidates, metric=exact_match\n", .{config.num_candidates});
}
