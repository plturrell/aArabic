# Model Orchestration - Phase 5 Enhancement Plan

**Date:** 2026-01-23  
**Status:** 🚧 IN PROGRESS  
**Goal:** Advanced orchestration features for production deployment

## Overview

Phase 5 builds on the centralized orchestration system (Phases 1-4) to add production-grade features including multi-category support, real-time GPU monitoring, A/B testing, benchmark-based scoring, and native Zig/Mojo implementations.

## Enhancements

### 1. Multi-Category Model Support with Weighted Scoring

**Current State:** Models assigned to single category only  
**Target:** Models serve multiple categories with per-category performance weights

#### Implementation
```zig
pub const MultiCategoryScore = struct {
    category: []const u8,
    base_score: f32,
    benchmark_score: f32,
    confidence: f32,
    
    pub fn totalScore(self: MultiCategoryScore) f32 {
        return self.base_score + self.benchmark_score * self.confidence;
    }
};

pub const MultiCategoryModel = struct {
    model: Model,
    category_scores: []MultiCategoryScore,
    
    pub fn scoreForCategory(self: *MultiCategoryModel, category: []const u8) ?f32 {
        for (self.category_scores) |score| {
            if (std.mem.eql(u8, score.category, category)) {
                return score.totalScore();
            }
        }
        return null;
    }
};
```

#### Benefits
- Models optimized for multiple use cases
- Better resource utilization
- Finer-grained selection

#### Timeline: 2-3 days

---

### 2. Real-Time GPU Load Monitoring for Dynamic Routing

**Current State:** Static GPU constraints (14GB, 38GB, 76GB)  
**Target:** Dynamic routing based on actual GPU availability

#### Implementation
```zig
pub const GPUMonitor = struct {
    allocator: Allocator,
    update_interval_ms: u32,
    last_update: i64,
    gpu_states: std.ArrayList(GPUState),
    
    pub const GPUState = struct {
        device_id: u32,
        total_memory_mb: usize,
        used_memory_mb: usize,
        temperature_c: f32,
        utilization_percent: f32,
        
        pub fn availableMemoryMB(self: GPUState) usize {
            return self.total_memory_mb - self.used_memory_mb;
        }
        
        pub fn isHealthy(self: GPUState) bool {
            return self.temperature_c < 80.0 and 
                   self.utilization_percent < 95.0;
        }
    };
    
    pub fn init(allocator: Allocator) !*GPUMonitor {
        // Initialize with nvidia-smi or equivalent
        return error.NotImplemented;
    }
    
    pub fn refresh(self: *GPUMonitor) !void {
        // Query GPU state via nvidia-smi, CUDA API, or ROCm
        const now = std.time.milliTimestamp();
        if (now - self.last_update < self.update_interval_ms) {
            return; // Use cached data
        }
        
        // Update GPU states
        self.last_update = now;
    }
    
    pub fn selectBestGPU(self: *GPUMonitor, required_memory_mb: usize) !u32 {
        try self.refresh();
        
        var best_gpu: ?u32 = null;
        var best_available: usize = 0;
        
        for (self.gpu_states.items, 0..) |state, i| {
            if (!state.isHealthy()) continue;
            
            const available = state.availableMemoryMB();
            if (available >= required_memory_mb and available > best_available) {
                best_gpu = @intCast(i);
                best_available = available;
            }
        }
        
        return best_gpu orelse error.NoGPUAvailable;
    }
};
```

#### Integration with ModelSelector
```zig
pub fn selectModelDynamic(
    self: *ModelSelector,
    category: []const u8,
    gpu_monitor: *GPUMonitor,
) !SelectionResult {
    // Get available GPU memory
    const best_gpu = try gpu_monitor.selectBestGPU(0); // Any GPU
    const gpu_state = gpu_monitor.gpu_states.items[best_gpu];
    const available_mb = gpu_state.availableMemoryMB();
    
    // Select model that fits
    const constraints = SelectionConstraints{
        .max_gpu_memory_mb = available_mb,
    };
    
    return try self.selectModel(category, constraints);
}
```

#### Benefits
- Optimal GPU utilization
- Load balancing across GPUs
- Prevent OOM errors
- Better multi-tenant support

#### Timeline: 3-4 days

---

### 3. A/B Testing Framework for Model Comparison

**Current State:** Single model selection per request  
**Target:** A/B test multiple models, track performance metrics

#### Implementation
```zig
pub const ABTest = struct {
    allocator: Allocator,
    test_id: []const u8,
    variants: []ABVariant,
    traffic_split: []f32, // e.g., [0.5, 0.5] for 50/50
    metrics: ABMetrics,
    
    pub const ABVariant = struct {
        name: []const u8,
        model_name: []const u8,
        constraints: SelectionConstraints,
    };
    
    pub const ABMetrics = struct {
        total_requests: usize,
        variant_requests: []usize,
        variant_latencies: []f32, // avg latency per variant
        variant_errors: []usize,
        variant_scores: []f32, // quality scores
        
        pub fn winningVariant(self: ABMetrics) usize {
            var best_idx: usize = 0;
            var best_score: f32 = 0.0;
            
            for (self.variant_scores, 0..) |score, i| {
                // Penalize for errors and latency
                const error_rate = @as(f32, @floatFromInt(self.variant_errors[i])) / 
                                  @as(f32, @floatFromInt(self.variant_requests[i]));
                const adjusted_score = score * (1.0 - error_rate) / 
                                      (1.0 + self.variant_latencies[i] / 1000.0);
                
                if (adjusted_score > best_score) {
                    best_score = adjusted_score;
                    best_idx = i;
                }
            }
            
            return best_idx;
        }
    };
    
    pub fn selectVariant(self: *ABTest, request_id: []const u8) usize {
        // Hash-based consistent variant selection
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(request_id);
        const hash = hasher.final();
        
        // Map hash to variant using traffic split
        const rand_val = @as(f32, @floatFromInt(hash % 10000)) / 10000.0;
        
        var cumulative: f32 = 0.0;
        for (self.traffic_split, 0..) |split, i| {
            cumulative += split;
            if (rand_val < cumulative) {
                return i;
            }
        }
        
        return self.variants.len - 1; // Fallback
    }
    
    pub fn recordMetric(
        self: *ABTest,
        variant_idx: usize,
        latency_ms: f32,
        score: f32,
        is_error: bool,
    ) !void {
        self.metrics.total_requests += 1;
        self.metrics.variant_requests[variant_idx] += 1;
        
        // Update running average
        const n = @as(f32, @floatFromInt(self.metrics.variant_requests[variant_idx]));
        self.metrics.variant_latencies[variant_idx] = 
            (self.metrics.variant_latencies[variant_idx] * (n - 1.0) + latency_ms) / n;
        self.metrics.variant_scores[variant_idx] = 
            (self.metrics.variant_scores[variant_idx] * (n - 1.0) + score) / n;
        
        if (is_error) {
            self.metrics.variant_errors[variant_idx] += 1;
        }
    }
};

pub const ABTestManager = struct {
    allocator: Allocator,
    active_tests: std.StringHashMap(*ABTest),
    
    pub fn createTest(
        self: *ABTestManager,
        test_id: []const u8,
        variants: []ABTest.ABVariant,
        traffic_split: []f32,
    ) !*ABTest {
        const test = try ABTest.init(
            self.allocator,
            test_id,
            variants,
            traffic_split,
        );
        
        try self.active_tests.put(test_id, test);
        return test;
    }
    
    pub fn getTest(self: *ABTestManager, test_id: []const u8) ?*ABTest {
        return self.active_tests.get(test_id);
    }
};
```

#### Benefits
- Data-driven model selection
- Continuous improvement
- Risk mitigation for new models
- Performance tracking

#### Timeline: 4-5 days

---

### 4. Benchmark-Based Scoring Integration

**Current State:** Static scoring (base + size bonuses)  
**Target:** Dynamic scoring based on actual benchmark performance

#### Implementation
```zig
pub const BenchmarkScoring = struct {
    allocator: Allocator,
    benchmark_weights: std.StringHashMap(f32),
    
    pub fn init(allocator: Allocator) !*BenchmarkScoring {
        var self = try allocator.create(BenchmarkScoring);
        self.* = .{
            .allocator = allocator,
            .benchmark_weights = std.StringHashMap(f32).init(allocator),
        };
        
        // Default weights for benchmarks
        try self.benchmark_weights.put("HumanEval", 1.0);
        try self.benchmark_weights.put("MBPP", 0.8);
        try self.benchmark_weights.put("GSM8K", 1.2);
        try self.benchmark_weights.put("MMLU", 1.5);
        try self.benchmark_weights.put("ARC-Challenge", 1.3);
        
        return self;
    }
    
    pub fn scoreModel(
        self: *BenchmarkScoring,
        model: *const Model,
        category: []const u8,
    ) f32 {
        var total_score: f32 = 0.0;
        var weight_sum: f32 = 0.0;
        
        // Get relevant benchmarks for category
        const relevant_benchmarks = self.getRelevantBenchmarks(category);
        
        for (model.benchmarks) |benchmark| {
            if (self.isRelevant(benchmark.name, relevant_benchmarks)) {
                const weight = self.benchmark_weights.get(benchmark.name) orelse 1.0;
                total_score += benchmark.score * weight;
                weight_sum += weight;
            }
        }
        
        if (weight_sum > 0.0) {
            return (total_score / weight_sum) / 100.0 * 50.0; // Scale to 0-50
        }
        
        return 0.0; // No relevant benchmarks
    }
    
    fn getRelevantBenchmarks(self: *BenchmarkScoring, category: []const u8) [][]const u8 {
        // Map categories to relevant benchmarks
        if (std.mem.eql(u8, category, "code")) {
            return &[_][]const u8{ "HumanEval", "MBPP" };
        } else if (std.mem.eql(u8, category, "math")) {
            return &[_][]const u8{ "GSM8K", "MATH" };
        } else if (std.mem.eql(u8, category, "reasoning")) {
            return &[_][]const u8{ "MMLU", "ARC-Challenge", "HellaSwag" };
        }
        return &[_][]const u8{};
    }
    
    fn isRelevant(
        self: *BenchmarkScoring,
        benchmark_name: []const u8,
        relevant: [][]const u8,
    ) bool {
        for (relevant) |rel| {
            if (std.mem.eql(u8, benchmark_name, rel)) {
                return true;
            }
        }
        return false;
    }
};
```

#### Integration
```zig
// In ModelSelector.selectModel()
const benchmark_score = self.benchmark_scoring.scoreModel(model, category);
score += benchmark_score; // Add 0-50 points based on benchmarks
```

#### Benefits
- Data-driven selection
- Objective performance metrics
- Automatic optimization
- Transparent decision-making

#### Timeline: 2-3 days

---

### 5. Extended Taxonomy (Domain-Specific Categories)

**Current State:** 9 generic categories  
**Target:** 20+ domain-specific categories with hierarchical structure

#### New Categories
```json
{
  "categories": {
    "code_python": {
      "parent": "code",
      "specialization": "python",
      "benchmarks": ["HumanEval", "MBPP"]
    },
    "code_javascript": {
      "parent": "code",
      "specialization": "javascript"
    },
    "code_rust": {
      "parent": "code",
      "specialization": "rust"
    },
    "math_algebra": {
      "parent": "math",
      "specialization": "algebra"
    },
    "math_calculus": {
      "parent": "math",
      "specialization": "calculus"
    },
    "nlp_sentiment": {
      "parent": "reasoning",
      "specialization": "sentiment_analysis"
    },
    "nlp_ner": {
      "parent": "reasoning",
      "specialization": "named_entity_recognition"
    },
    "data_forecasting": {
      "parent": "time_series",
      "specialization": "forecasting"
    },
    "data_anomaly": {
      "parent": "time_series",
      "specialization": "anomaly_detection"
    }
  }
}
```

#### Hierarchical Selection
```zig
pub fn selectModelHierarchical(
    self: *ModelSelector,
    category: []const u8,
    constraints: SelectionConstraints,
) !SelectionResult {
    // Try specific category first
    if (self.selectModel(category, constraints)) |result| {
        return result;
    } else |_| {
        // Fallback to parent category
        if (self.getParentCategory(category)) |parent| {
            return try self.selectModel(parent, constraints);
        }
        return error.NoSuitableModel;
    }
}
```

#### Benefits
- Fine-grained specialization
- Better model matching
- Fallback hierarchy
- Extensible taxonomy

#### Timeline: 2-3 days

---

### 6. Convert Python Scripts to Zig/Mojo

**Current State:** Python scripts in scripts/ directory  
**Target:** Native Zig/Mojo implementations in nLocalModels

#### Scripts to Convert

1. **hf_model_card_extractor.py** → **model_enricher.zig**
2. **benchmark_validator.py** → **benchmark_validator.zig**
3. **benchmark_routing_performance.py** → **routing_benchmark.zig**

#### Implementation Structure
```zig
// model_enricher.zig
pub const ModelEnricher = struct {
    allocator: Allocator,
    hf_client: *HFClient,
    
    pub const HFClient = struct {
        // HTTP client for HuggingFace API
        pub fn fetchModelCard(self: *HFClient, repo: []const u8) !ModelCard;
        pub fn fetchBenchmarks(self: *HFClient, repo: []const u8) ![]Benchmark;
    };
    
    pub fn enrichRegistry(
        self: *ModelEnricher,
        registry_path: []const u8,
    ) !void {
        // Load registry
        // For each model, fetch HF data
        // Update registry with enriched data
        // Save back to file
    }
};
```

#### Benefits
- Native performance
- No Python dependency
- Better integration
- Type safety

#### Timeline: 5-6 days

---

## Implementation Timeline

| Enhancement | Duration | Dependencies | Priority |
|-------------|----------|--------------|----------|
| 1. Multi-Category Support | 2-3 days | None | High |
| 2. GPU Monitoring | 3-4 days | None | High |
| 3. A/B Testing | 4-5 days | None | Medium |
| 4. Benchmark Scoring | 2-3 days | None | High |
| 5. Extended Taxonomy | 2-3 days | #1 | Medium |
| 6. Zig/Mojo Conversion | 5-6 days | None | Low |

**Total Estimated Time:** 18-24 days (3-4 weeks)

## Rollout Strategy

### Week 1: Core Enhancements
- Implement multi-category support
- Add benchmark-based scoring
- Update test suite

### Week 2: Dynamic Features
- Implement GPU monitoring
- Add extended taxonomy
- Integration testing

### Week 3: Advanced Features
- Implement A/B testing framework
- Convert Python scripts to Zig
- Documentation updates

### Week 4: Validation & Deployment
- Comprehensive testing
- Performance benchmarks
- Production deployment

## Success Metrics

- [ ] Multi-category models: 80% of models support 2+ categories
- [ ] GPU utilization: 15% improvement
- [ ] A/B test coverage: 100% of model updates
- [ ] Benchmark scoring: Active for all models
- [ ] Extended taxonomy: 20+ categories defined
- [ ] Zero Python dependencies in production

## Next Steps

1. Review and approve plan
2. Start with highest priority items (#1, #2, #4)
3. Implement in parallel where possible
4. Regular progress updates
5. Incremental deployment

---

**Status:** 🚧 Ready to implement  
**Approval:** Pending  
**Start Date:** TBD
