# Day 61: Uncertainty Quantification for mHC

**Date**: January 19, 2026  
**Focus**: Bootstrap resampling and confidence intervals for geometry detection  
**Duration**: 8 hours  
**Status**: ✅ Complete

## Overview

Day 61 implements uncertainty quantification for the mHC (manifold Hypothesis Classifier) geometry detection system. This extends the base geometry detector with statistical methods to quantify detection reliability, compute confidence intervals, and perform vote-based classification across bootstrap samples.

## Implementation Summary

### New File Created

**`src/serviceCore/nOpenaiServer/inference/engine/core/mhc_uncertainty.zig`**
- **Lines**: 1133+
- **Tests**: 27 (exceeds 20+ target)

### Key Components

#### 1. UncertaintyAwareGeometryDetector Struct

```zig
pub const UncertaintyConfig = struct {
    bootstrap_samples: usize = 100,      // Number of bootstrap resamples
    confidence_level: f32 = 0.95,        // Confidence level for intervals
    detection_threshold: f32 = 0.6,      // Vote confidence threshold
    random_seed: u64 = 42,               // Reproducible resampling
    min_valid_samples: usize = 10,       // Minimum samples for validity
    geometry_config: GeometryDetectorConfig = .{},
};
```

The detector wraps the base geometry detector and adds uncertainty quantification via bootstrap resampling.

#### 2. Bootstrap Resampling

`bootstrap_curvature(points, n_samples)` - Core bootstrap implementation:
- Random sampling with replacement from point cloud
- Runs geometry detection on each resampled dataset
- Collects curvature distribution across samples
- Returns mean, std, and confidence interval

#### 3. Confidence Interval Computation

`compute_confidence_interval(samples, confidence)` - Percentile-based CI:
- Lower bound: `(1-confidence)/2` percentile
- Upper bound: `(1+confidence)/2` percentile
- Returns `ConfidenceInterval` with mean, std, bounds, and sample count

#### 4. Vote-based Classification

`vote_classification(curvatures)` - Majority voting:
- Classifies each bootstrap sample as Euclidean/Hyperbolic/Spherical
- Counts votes for each manifold type
- Returns winning class and vote proportion

### Result Types

#### ConfidenceInterval
```zig
pub const ConfidenceInterval = struct {
    lower: f32,           // Lower bound
    upper: f32,           // Upper bound
    mean: f32,            // Point estimate
    std: f32,             // Standard deviation
    confidence: f32,      // Confidence level used
    sample_count: usize,  // Number of valid samples
    
    pub fn contains(self, value: f32) bool { ... }
    pub fn width(self) f32 { ... }
    pub fn isNarrow(self, threshold: f32) bool { ... }
};
```

#### VoteResult
```zig
pub const VoteResult = struct {
    manifold_type: ManifoldType,    // Winning type
    vote_proportion: f32,           // Winner's proportion
    euclidean_votes: u32,
    hyperbolic_votes: u32,
    spherical_votes: u32,
    total_votes: u32,
    
    pub fn isConfident(self, threshold: f32) bool { ... }
    pub fn getProportions(self) struct { ... } { ... }
};
```

#### UncertaintyResult
```zig
pub const UncertaintyResult = struct {
    vote_result: VoteResult,
    curvature_ci: ConfidenceInterval,
    bootstrap_means: []f32,
    is_reliable: bool,
    computation_time_ns: i64,
    
    pub fn getRecommendedGeometry(self) ?ManifoldType { ... }
    pub fn getVoteEntropy(self) f32 { ... }  // Normalized entropy [0,1]
};
```

### Additional Utilities

- `compute_required_sample_size(std, margin, confidence)` - Sample size estimation
- `compute_calibration_error(predictions, ground_truth, confidences)` - ECE computation

## Test Coverage

All 27 tests pass:

| Test Category | Count | Status |
|--------------|-------|--------|
| Config defaults | 1 | ✅ |
| ConfidenceInterval methods | 3 | ✅ |
| VoteResult methods | 3 | ✅ |
| compute_confidence_interval | 3 | ✅ |
| vote_classification | 5 | ✅ |
| compute_required_sample_size | 2 | ✅ |
| compute_calibration_error | 3 | ✅ |
| UncertaintyAwareGeometryDetector | 3 | ✅ |
| UncertaintyResult methods | 4 | ✅ |

## Usage Example

```zig
const allocator = std.heap.page_allocator;
var detector = UncertaintyAwareGeometryDetector.init(allocator, .{
    .bootstrap_samples = 100,
    .confidence_level = 0.95,
    .detection_threshold = 0.6,
});

var result = try detector.detectWithUncertainty(points, num_points, dim);
defer result.deinit();

if (result.is_reliable) {
    const geometry = result.getRecommendedGeometry().?;
    std.debug.print("Detected: {} (confidence: {d:.2})\n", .{
        geometry, result.vote_result.vote_proportion
    });
    std.debug.print("Curvature CI: [{d:.3}, {d:.3}]\n", .{
        result.curvature_ci.lower, result.curvature_ci.upper
    });
} else {
    std.debug.print("Detection unreliable, falling back to Euclidean\n", .{});
}
```

## Performance Considerations

- Bootstrap resampling is O(n_samples × detection_cost)
- Default 100 samples provides good balance of accuracy and speed
- Computation overhead typically <10ms for small point clouds
- Memory: O(n_samples) for storing bootstrap means

## Success Criteria Met

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Lines of code | 400+ | 1133 | ✅ |
| Test count | 20+ | 27 | ✅ |
| Bootstrap method | Working | Yes | ✅ |
| Confidence intervals | Computed | Yes | ✅ |
| Vote-based classification | Implemented | Yes | ✅ |

## Next Steps

- Day 62: Bayesian curvature estimation with Gaussian priors
- Day 63: Integration with production geometry auto-detection
- Day 64: Performance optimization for large point clouds

