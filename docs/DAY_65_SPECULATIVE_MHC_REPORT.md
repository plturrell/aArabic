# Day 65: Speculative mHC Integration Report

## Overview

This report documents the implementation of speculative mHC (Manifold Hyperbolic Coordinates) integration for speculative decoding in the inference engine. The module provides geometric validation for draft candidates, enabling more accurate acceptance/rejection decisions during speculative decoding.

## Implementation Summary

### File Created
- `src/serviceCore/nOpenaiServer/inference/engine/core/mhc_speculative.zig` (1236 lines)

### Core Components

#### 1. GeometricValidator Struct
Configurable validator for speculative decoding with weighted scoring:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `curvature_weight` | 0.3 | Weight for curvature matching score |
| `distance_weight` | 0.4 | Weight for embedding distance score |
| `stability_weight` | 0.3 | Weight for energy stability score |
| `acceptance_threshold` | 0.5 | Minimum score for acceptance |
| `temperature` | 1.0 | Scaling factor for acceptance probability |
| `max_curvature_deviation` | 0.5 | Maximum allowed curvature difference |
| `max_distance` | 2.0 | Maximum embedding distance |
| `max_energy_delta` | 1.0 | Maximum energy change |

Factory methods provided:
- `default()` - Standard configuration
- `strict()` - High threshold (0.7), tighter constraints
- `lenient()` - Low threshold (0.3), relaxed constraints
- `withWeights(c, d, s)` - Custom weights (auto-normalized)

#### 2. Speculative Acceptance Functions

| Function | Purpose |
|----------|---------|
| `compute_curvature_score()` | Score based on curvature match (1 - normalized deviation) |
| `compute_distance_score()` | Score based on embedding similarity (1 - normalized distance) |
| `compute_stability_score()` | Score based on energy stability (1 - normalized delta) |
| `compute_combined_acceptance()` | Weighted combination of all three scores |

All scores are normalized to [0, 1] range where 1 = perfect match.

#### 3. Speculation Pipeline

**SpeculativeCandidate struct:**
- `token_id`: Token identifier
- `embedding`: Embedding vector slice
- `curvature`: Curvature at sequence point
- `energy`: Energy level
- `log_prob`: Log probability from draft model
- `position`: Position in speculative sequence

**Validation functions:**
- `validate_candidate()` - Validate single candidate, returns `ValidationResult`
- `batch_validate()` - Validate multiple candidates, returns `BatchValidationResult`
- `find_best_candidate()` - Find highest scoring accepted candidate
- `find_longest_accepted_prefix()` - Find longest consecutive accepted prefix

#### 4. GeometricSpeculationContext

Manages geometric state during speculative decoding:

**State tracking:**
- Running mean and variance of curvatures (Welford's algorithm)
- Baseline energy with exponential moving average
- Accumulated curvature and token counts

**Methods:**
- `prepare_speculation()` - Prepare context from draft tokens
- `finalize_speculation()` - Update state after acceptance
- `should_terminate_speculation()` - Early termination check
- `get_curvature_variance()` / `get_curvature_std()` - Statistics

#### 5. SIMD Optimization

Vectorized operations for performance:
- `euclidean_distance_simd()` - 8-wide SIMD distance computation
- `cosine_similarity_simd()` - 8-wide SIMD similarity
- `norm_simd()` - 8-wide SIMD L2 norm

## Test Coverage

**37 tests implemented covering:**

| Category | Tests |
|----------|-------|
| GeometricValidator configuration | 6 |
| Individual score functions | 9 |
| SIMD helper functions | 6 |
| Candidate validation | 4 |
| Batch validation | 3 |
| Candidate selection | 4 |
| GeometricSpeculationContext | 5 |

All 37 tests pass successfully.

## Integration Points

The module integrates with:
- `mhc_constraints.zig` - Core mHC constraint operations
- `mhc_hyperbolic.zig` - Hyperbolic geometry operations
- `mhc_spherical.zig` - Spherical manifold operations

## Usage Example

```zig
const mhc_speculative = @import("mhc_speculative.zig");

// Create validator
const validator = mhc_speculative.GeometricValidator.default();

// Create candidate
const embedding = [_]f32{ 0.1, 0.2, 0.3 };
const candidate = mhc_speculative.SpeculativeCandidate.init(
    token_id,
    &embedding,
    curvature,
    energy,
    log_prob,
    position,
);

// Create target context
const target = mhc_speculative.TargetContext{
    .embedding = &target_embedding,
    .curvature = target_curvature,
    .energy = target_energy,
    .baseline_energy = baseline,
};

// Validate candidate
const result = mhc_speculative.validate_candidate(candidate, target, validator);
if (result.accepted) {
    // Use accepted token
}
```

## Performance Characteristics

- **Scoring functions**: O(1) for scalars, O(n) for embeddings
- **Batch validation**: O(n × d) where n = candidates, d = embedding dim
- **SIMD speedup**: ~4-8x for embedding operations on supported architectures

## Conclusion

Day 65 successfully implements speculative mHC integration with:
- ✅ GeometricValidator with configurable weights and thresholds
- ✅ Four speculative acceptance scoring functions
- ✅ Complete speculation pipeline with batch validation
- ✅ GeometricSpeculationContext for state management
- ✅ 37 comprehensive tests (exceeds 20+ requirement)
- ✅ 1236 lines (exceeds 400+ requirement)
- ✅ SIMD-optimized distance computations

