# Phase 5 Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Phase 5 Model Orchestration enhancements to production.

**Phase 5 Enhancements:**
- ✅ Enhancement #1: Multi-Category Model Support
- ✅ Enhancement #2: Real-Time GPU Load Monitoring  
- ✅ Enhancement #4: Benchmark-Based Scoring

**Status:** Development Complete - Ready for Integration & Testing

---

## 📋 Pre-Deployment Checklist

### ✅ Completed (Development)
- [x] benchmark_scoring.zig module created (350+ lines)
- [x] multi_category.zig module created (400+ lines)
- [x] gpu_monitor.zig module created (300+ lines)
- [x] model_selector.zig integration complete (600+ lines)
- [x] 23 unit tests passing
- [x] Documentation updated

### 🔧 Integration Checklist (Current Phase)

#### 1. Update nFlow to Use New Selection Methods

**Location:** `src/serviceCore/nFlow/nodes/llm/llm_nodes.zig`

**Current State:** Using basic model selection

**Required Changes:**

```zig
// Add import
const ModelSelector = @import("../../nLocalModels/orchestration/model_selector.zig").ModelSelector;

// In LLM node initialization
pub fn initWithOrchestration(
    allocator: Allocator,
    registry_path: []const u8,
    categories_path: []const u8,
) !*LLMNode {
    const selector = try ModelSelector.init(allocator, registry_path, categories_path);
    
    // Load registry and categories
    try selector.loadRegistry();
    try selector.loadCategories();
    
    // Enable all Phase 5 features
    try selector.enableBenchmarkScoring();
    try selector.enableMultiCategory();
    try selector.enableGPUMonitoring(5000); // 5 second refresh
    
    return selector;
}

// In model selection logic
pub fn selectModelForTask(
    self: *LLMNode,
    task_category: []const u8,
    constraints: ModelSelector.SelectionConstraints,
) !ModelSelector.SelectionResult {
    // Use GPU-aware selection (combines all enhancements)
    return try self.selector.selectModelGPUAware(task_category, constraints);
}
```

**Files to Modify:**
- [ ] `src/serviceCore/nFlow/nodes/llm/llm_nodes.zig`
- [ ] `src/serviceCore/nFlow/orchestration/nLocalModels_integration.zig`

**Testing:**
- [ ] Build nFlow with new integration
- [ ] Verify model selection works
- [ ] Check GPU monitoring integration
- [ ] Validate benchmark scoring

---

#### 2. Configure Benchmark Weights for Your Models

**Location:** `src/serviceCore/nLocalModels/orchestration/benchmark_scoring.zig`

**Current Configuration:**

```zig
// Code category (lines 30-32)
try weights.put("HumanEval", 1.0);
try weights.put("MBPP", 0.8);

// Math category (lines 36-38)
try weights.put("GSM8K", 1.2);
try weights.put("MATH", 1.5);

// etc...
```

**Action Required:**

1. **Review Current Weights** - Check if weights align with your priorities
2. **Adjust if Needed** - Modify weights based on:
   - Business requirements (which benchmarks matter most?)
   - Model performance profiles
   - Task importance

**Example Adjustment:**

```zig
// If HumanEval is most critical for your code tasks
try weights.put("HumanEval", 1.5);  // Increased from 1.0
try weights.put("MBPP", 0.8);       // Keep as is
```

**Customization Method:**

```zig
// After initialization, update weights
const scoring = try BenchmarkScoring.init(allocator);
try scoring.setBenchmarkWeight("HumanEval", 1.5);
try scoring.setBenchmarkWeight("MMLU", 2.0);
```

**Tasks:**
- [ ] Review all 16 benchmark weights
- [ ] Adjust weights based on priorities
- [ ] Document weight rationale
- [ ] Test with actual model scores

---

#### 3. Set Appropriate GPU Monitoring Refresh Interval

**Location:** `src/serviceCore/nLocalModels/orchestration/model_selector.zig`

**Current Default:** 5000ms (5 seconds)

**Configuration:**

```zig
// In ModelSelector initialization
try selector.enableGPUMonitoring(5000); // 5 second refresh
```

**Recommended Intervals by Use Case:**

| Use Case | Interval | Rationale |
|----------|----------|-----------|
| Development | 10000ms (10s) | Less overhead, slower changes |
| Staging | 5000ms (5s) | Balance testing & overhead |
| Production (Low Load) | 5000ms (5s) | Standard monitoring |
| Production (High Load) | 3000ms (3s) | More responsive |
| Production (Critical) | 1000ms (1s) | Maximum responsiveness |

**Considerations:**
- Shorter intervals = More responsive but higher overhead
- nvidia-smi has ~50-100ms execution time
- Consider your request rate and GPU count

**Tasks:**
- [ ] Determine appropriate interval for your environment
- [ ] Test different intervals under load
- [ ] Monitor nvidia-smi overhead
- [ ] Set production interval
- [ ] Document interval choice

---

#### 4. Update Multi-Category Confidence Scores

**Location:** `src/serviceCore/nLocalModels/orchestration/model_selector.zig` (lines 468-477)

**Current Logic:**

```zig
// Confidence based on whether this is primary category
const confidence: f32 = if (model.orchestration_categories.len == 1) 
    0.95  // Single category model
else if (std.mem.eql(u8, category, model.orchestration_categories[0]))
    0.9   // Primary category
else
    0.7;  // Secondary categories
```

**Customization Based on Performance:**

After collecting production metrics, adjust confidence scores:

```zig
// Example: If a model performs equally well across categories
const confidence: f32 = if (model.orchestration_categories.len == 1) 
    0.95
else if (std.mem.eql(u8, category, model.orchestration_categories[0]))
    0.95  // Increased from 0.9 if primary performance is excellent
else if (std.mem.eql(u8, category, "code") and std.mem.indexOf(u8, model.name, "coder")) 
    0.85  // Higher confidence for code-specialized models
else
    0.7;
```

**Performance-Based Adjustment Process:**

1. **Collect Metrics** (1-2 weeks):
   - Track model accuracy per category
   - Measure inference latency
   - Monitor user satisfaction

2. **Analyze Performance**:
   - Compare primary vs secondary category performance
   - Identify models that excel at multiple tasks
   - Note any surprising results

3. **Adjust Confidence**:
   - Increase confidence for consistently good performers
   - Decrease for models struggling with secondary categories
   - Document reasoning

**Tasks:**
- [ ] Deploy with default confidence scores
- [ ] Collect 1-2 weeks of metrics
- [ ] Analyze multi-category performance
- [ ] Adjust confidence scores
- [ ] Re-test and validate
- [ ] Document adjustments

---

#### 5. Add Integration Tests with Actual MODEL_REGISTRY.json

**Location:** `tests/orchestration/test_model_selection_integration.zig`

**Required Tests:**

```zig
test "Integration: Load actual MODEL_REGISTRY.json" {
    const allocator = std.testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        "vendor/layerModels/MODEL_REGISTRY.json",
        "src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json",
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    // Verify models loaded
    try std.testing.expect(selector.models.items.len > 0);
    try std.testing.expect(selector.categories.count() > 0);
}

test "Integration: Select model for code task" {
    const allocator = std.testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        "vendor/layerModels/MODEL_REGISTRY.json",
        "src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json",
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    try selector.enableBenchmarkScoring();
    
    const result = try selector.selectModel("code", .{
        .max_gpu_memory_mb = 16 * 1024,
    });
    defer result.deinit(allocator);
    
    // Verify result
    try std.testing.expect(result.model.name.len > 0);
    try std.testing.expect(result.score > 0.0);
    
    std.debug.print("\nSelected: {s}\n", .{result.model.name});
    std.debug.print("Score: {d:.2}\n", .{result.score});
    std.debug.print("Reason: {s}\n", .{result.reason});
}

test "Integration: Multi-category selection" {
    const allocator = std.testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        "vendor/layerModels/MODEL_REGISTRY.json",
        "src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json",
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    try selector.enableBenchmarkScoring();
    try selector.buildMultiCategoryRegistry();
    
    const result = try selector.selectModelMultiCategory("code", .{});
    defer result.deinit(allocator);
    
    try std.testing.expect(result.score > 0.0);
}

test "Integration: GPU-aware selection (mock)" {
    // Note: This test requires actual GPU or mock implementation
    // For CI/CD, consider using mock GPU state
    const allocator = std.testing.allocator;
    
    const selector = try ModelSelector.init(
        allocator,
        "vendor/layerModels/MODEL_REGISTRY.json",
        "src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json",
    );
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    // TODO: Add mock GPU monitor for testing
    // const result = try selector.selectModelGPUAware("code", .{});
}
```

**Tasks:**
- [ ] Create integration test file
- [ ] Add MODEL_REGISTRY.json load test
- [ ] Add basic selection test
- [ ] Add multi-category test
- [ ] Add GPU-aware test (with mock)
- [ ] Run tests in CI/CD
- [ ] Verify all tests pass

---

#### 6. Monitor GPU Selection Decisions in Production

**Instrumentation Required:**

```zig
// Add logging to selectModelGPUAware
pub fn selectModelGPUAware(
    self: *ModelSelector,
    task_category: []const u8,
    constraints: SelectionConstraints,
) !SelectionResult {
    const start_time = std.time.milliTimestamp();
    
    // ... selection logic ...
    
    const end_time = std.time.milliTimestamp();
    const duration_ms = end_time - start_time;
    
    // Log selection
    std.log.info(
        "GPU Selection: category={s}, model={s}, gpu={d}, score={d:.2}, duration_ms={d}",
        .{task_category, best_model.?.name, best_gpu_id, best_score, duration_ms}
    );
    
    // ... return result ...
}
```

**Metrics to Track:**

1. **Selection Metrics:**
   - Model selected per category
   - GPU selected per request
   - Selection duration
   - Score distributions

2. **GPU Metrics:**
   - GPU load over time
   - Memory utilization
   - Temperature trends
   - Selection fairness (is one GPU always chosen?)

3. **Performance Metrics:**
   - Inference latency per model
   - Throughput per GPU
   - Error rates
   - OOM incidents

**Monitoring Tools:**

```bash
# Real-time GPU monitoring script
#!/bin/bash
while true; do
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits
    sleep 5
done | tee gpu_monitor.log
```

**Dashboard Metrics:**
- [ ] GPU utilization per device
- [ ] Model selection frequency
- [ ] Average selection latency
- [ ] GPU temperature trends
- [ ] Memory usage patterns

**Tasks:**
- [ ] Add logging to selection methods
- [ ] Set up metrics collection
- [ ] Create monitoring dashboard
- [ ] Configure alerts (high temp, OOM, etc.)
- [ ] Review metrics daily for 1 week
- [ ] Identify optimization opportunities

---

#### 7. Collect Metrics on Multi-Category Routing Effectiveness

**Metrics to Collect:**

1. **Routing Accuracy:**
   - Did the model perform well for the assigned category?
   - How often is fallback triggered?
   - Are confidence scores calibrated correctly?

2. **Performance by Category:**
   - Average latency per category
   - Success rate per category
   - User satisfaction per category

3. **Multi-Category Benefits:**
   - How often are models used across categories?
   - Resource utilization improvement
   - Cost reduction (fewer loaded models)

**Data Collection:**

```zig
// Add to ModelSelector
pub const SelectionMetrics = struct {
    timestamp: i64,
    task_category: []const u8,
    selected_model: []const u8,
    primary_category: bool,
    confidence_score: f32,
    final_score: f64,
    gpu_id: u32,
    selection_duration_ms: i64,
};

// Log metrics
pub fn logMetrics(self: *ModelSelector, metrics: SelectionMetrics) !void {
    // Write to metrics file or send to monitoring system
    std.log.info("METRICS: {}", .{metrics});
}
```

**Analysis Queries:**

```sql
-- Example queries if using SQL for metrics

-- Models used per category
SELECT task_category, selected_model, COUNT(*) as usage_count
FROM selection_metrics
GROUP BY task_category, selected_model
ORDER BY task_category, usage_count DESC;

-- Multi-category effectiveness
SELECT selected_model, 
       COUNT(DISTINCT task_category) as categories_served,
       AVG(confidence_score) as avg_confidence
FROM selection_metrics
GROUP BY selected_model
HAVING categories_served > 1;

-- Performance by category
SELECT task_category,
       AVG(selection_duration_ms) as avg_duration,
       AVG(final_score) as avg_score
FROM selection_metrics
GROUP BY task_category;
```

**Tasks:**
- [ ] Design metrics schema
- [ ] Implement metrics collection
- [ ] Set up storage (database/files)
- [ ] Create analysis scripts
- [ ] Run for 2-4 weeks
- [ ] Generate effectiveness report
- [ ] Share findings with team

---

## 🚀 Deployment Phases

### Phase 1: Development Environment (1 week)
- [ ] Complete all integration checklist items
- [ ] Run integration tests
- [ ] Fix any issues found
- [ ] Document findings

### Phase 2: Staging Environment (2 weeks)
- [ ] Deploy to staging
- [ ] Monitor GPU selection
- [ ] Collect initial metrics
- [ ] Tune configuration
- [ ] Load testing

### Phase 3: Production Canary (1 week)
- [ ] Deploy to 10% of production traffic
- [ ] Monitor closely
- [ ] Compare with baseline
- [ ] Validate improvements
- [ ] Fix any issues

### Phase 4: Full Production (1 week)
- [ ] Roll out to 100% of traffic
- [ ] Continue monitoring
- [ ] Collect performance data
- [ ] Generate deployment report
- [ ] Plan Phase 6 enhancements

---

## 📊 Success Criteria

### Must Have
- ✅ All integration tests passing
- ✅ GPU monitoring working correctly
- ✅ Multi-category selection functional
- ✅ Benchmark scoring validated
- ✅ No increase in error rates
- ✅ No OOM errors

### Should Have
- 📈 10%+ improvement in model selection accuracy
- 📈 Better GPU utilization (>70% average)
- 📈 Reduced manual model configuration
- 📈 Clear metrics dashboard

### Nice to Have
- 🎯 20%+ cost reduction (fewer loaded models)
- 🎯 15%+ latency improvement
- 🎯 Automated model selection for new tasks

---

## 🔧 Rollback Plan

If critical issues arise:

1. **Immediate Actions:**
   - Disable GPU monitoring: Comment out `enableGPUMonitoring()`
   - Revert to basic selection: Use `selectModel()` instead of `selectModelGPUAware()`
   - Fall back to single-category: Disable `buildMultiCategoryRegistry()`

2. **Code Changes:**
```zig
// Emergency rollback - use basic selection
const result = try selector.selectModel(task_category, constraints);
// Instead of:
// const result = try selector.selectModelGPUAware(task_category, constraints);
```

3. **Monitoring:**
   - Check error rates return to baseline
   - Verify system stability
   - Document root cause

---

## 📝 Post-Deployment

### Week 1 Review
- [ ] Analyze selection metrics
- [ ] Review GPU utilization
- [ ] Check error logs
- [ ] Gather team feedback
- [ ] Create status report

### Week 4 Review
- [ ] Complete effectiveness analysis
- [ ] Compare with pre-Phase 5 metrics
- [ ] Document lessons learned
- [ ] Plan configuration tuning
- [ ] Identify Phase 6 opportunities

---

## 📞 Support

**Questions or Issues:**
- Check logs: `logs/model_selection.log`
- Review metrics: GPU monitoring dashboard
- Contact: Phase 5 deployment team
- Escalate: If OOM or critical errors

**Resources:**
- [Phase 5 Plan](../../../docs/01-architecture/ORCHESTRATION_PHASE5_PLAN.md)
- [Model Orchestration Mapping](../../../docs/01-architecture/MODEL_ORCHESTRATION_MAPPING.md)
- [Validation Report](../validation/MODEL_ORCHESTRATION_VALIDATION_REPORT.md)

---

## ✅ Final Checklist

Before marking deployment complete:

- [ ] All 7 integration items completed
- [ ] Integration tests passing
- [ ] Production monitoring active
- [ ] Metrics collection running
- [ ] Team trained on new features
- [ ] Documentation updated
- [ ] Rollback plan tested
- [ ] Success criteria defined
- [ ] Post-deployment review scheduled

**Deployment Complete Date:** _____________

**Deployed By:** _____________

**Sign-off:** _____________
