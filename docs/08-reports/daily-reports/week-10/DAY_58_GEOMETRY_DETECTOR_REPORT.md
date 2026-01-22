# Day 58: Automatic Geometry Detection for mHC

## Overview

This module implements automatic geometry detection for mHC (mixed Hyperbolic Coordinates) using Ollivier-Ricci curvature estimation. Given a point cloud, it automatically determines whether the underlying geometry is Euclidean, Hyperbolic, or Spherical.

## Implementation Details

### Location
`src/serviceCore/nOpenaiServer/inference/engine/core/mhc_geometry_detector.zig`

### Key Components

#### 1. k-NN Graph Construction
- **Function**: `build_knn_graph(points, num_points, dim, k, allocator)`
- Builds a k-nearest neighbors graph from the point cloud
- Uses SIMD-optimized distance computation for efficiency
- Returns a `KNNGraph` structure with neighbor indices and distances

#### 2. Ollivier-Ricci Curvature Estimation
- **Function**: `compute_ollivier_ricci(graph, points, dim, config)`
- Estimates discrete curvature using the Ollivier-Ricci formula: `κ(x, y) = 1 - W(μ_x, μ_y) / d(x, y)`
- Uses Wasserstein distance between uniform distributions on neighbor sets
- Returns mean curvature, standard deviation, and sample count

#### 3. Geometry Classification
- **Function**: `classify_geometry(mean_curvature, thresholds)`
- Classifies geometry based on curvature thresholds:
  - `curvature < -0.1` → **Hyperbolic** (negative curvature)
  - `curvature > 0.1` → **Spherical** (positive curvature)
  - Otherwise → **Euclidean** (flat)

#### 4. Confidence Scoring
- **Function**: `get_detection_confidence(mean, std, type, thresholds)`
- Computes confidence based on:
  - Distance from decision boundary (higher = more confident)
  - Standard deviation of curvature samples (lower = more consistent = more confident)

#### 5. Full Auto-Detection Pipeline
- **Function**: `detect_geometry(points, num_points, dim, config, allocator)`
- Complete pipeline that:
  1. Validates input data
  2. Builds k-NN graph
  3. Computes Ollivier-Ricci curvature
  4. Classifies geometry
  5. Returns `GeometryDetectionResult`

## API Reference

### Types

```zig
pub const ManifoldType = enum {
    Euclidean,
    Hyperbolic,
    Spherical,
};

pub const GeometryDetectionResult = struct {
    manifold_type: ManifoldType,
    mean_curvature: f32,
    std_curvature: f32,
    confidence: f32,
    edges_sampled: u32,
    computation_time_ns: i64,
};

pub const GeometryDetectorConfig = struct {
    k_neighbors: u32 = 10,
    sample_edges: u32 = 100,
    epsilon: f32 = 1e-8,
    thresholds: CurvatureThresholds = .{},
    random_seed: u64 = 42,
};
```

### Functions

| Function | Description |
|----------|-------------|
| `detect_geometry(points, n, dim, config, alloc)` | Full auto-detection pipeline |
| `detect_geometry_auto(points, n, dim, alloc)` | Convenience wrapper with defaults |
| `build_knn_graph(points, n, dim, k, alloc)` | Build k-NN graph |
| `compute_ollivier_ricci(graph, points, dim, cfg)` | Compute curvature statistics |
| `classify_geometry(curvature, thresholds)` | Classify geometry type |
| `get_detection_confidence(...)` | Compute confidence score |

## Usage Example

```zig
const std = @import("std");
const detector = @import("mhc_geometry_detector.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Point cloud data (num_points=100, dim=64)
    const points: []const f32 = ...; // your data

    const result = try detector.detect_geometry(
        points,
        100,  // num_points
        64,   // dimension
        .{},  // default config
        allocator,
    );

    std.debug.print("Detected: {s}\n", .{result.manifold_type.getName()});
    std.debug.print("Curvature: {d:.4}\n", .{result.mean_curvature});
    std.debug.print("Confidence: {d:.2}%\n", .{result.confidence * 100});
}
```

## Algorithm Details

### Ollivier-Ricci Curvature

The Ollivier-Ricci curvature is a discrete analogue of Ricci curvature:

```
κ(x, y) = 1 - W₁(μ_x, μ_y) / d(x, y)
```

Where:
- `W₁` is the 1-Wasserstein (Earth Mover's) distance
- `μ_x` is the uniform distribution over neighbors of x
- `d(x, y)` is the geodesic distance between x and y

**Interpretation**:
- **κ > 0**: Points cluster together (spherical geometry)
- **κ < 0**: Points spread apart (hyperbolic geometry)
- **κ ≈ 0**: Flat geometry (Euclidean)

### Classification Thresholds

| Curvature Range | Classification | Example |
|-----------------|----------------|---------|
| κ < -0.1 | Hyperbolic | Tree-like hierarchies |
| -0.1 ≤ κ ≤ 0.1 | Euclidean | Standard vector spaces |
| κ > 0.1 | Spherical | Cyclic/periodic data |

## Performance

- **SIMD Optimization**: Distance computations use 8-wide SIMD vectors
- **Memory Efficient**: Only stores k neighbors per point
- **Configurable Sampling**: Sample edges for faster estimation on large datasets

## Test Coverage

20 unit tests covering:
- SIMD distance computation
- k-NN graph construction
- Curvature classification
- Confidence scoring
- Full detection pipeline
- Point generation utilities
- Edge cases and error handling

Run tests:
```bash
cd src/serviceCore/nOpenaiServer/inference/engine/core
zig test mhc_geometry_detector.zig
```

## Integration with mHC

This module integrates with:
- `mhc_product_manifold.zig` - Uses `ManifoldType` enum
- `mhc_configuration.zig` - `auto_detect_geometry` flag
- `mhc_hyperbolic.zig` - Hyperbolic distance functions
- `mhc_spherical.zig` - Spherical distance functions

## Statistics

- **Lines of Code**: 878
- **Test Cases**: 20
- **Functions**: 15+
- **Dependencies**: std, mhc_product_manifold

