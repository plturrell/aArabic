# Day 62: Bayesian Curvature Estimation for mHC

## Overview

This module implements Bayesian inference for manifold curvature estimation in mHC (manifold Hyperbolic Constraints). It provides probabilistic curvature estimates with uncertainty quantification, enabling robust decision-making in geometry detection and manifold selection.

## Implementation Details

### Location
`src/serviceCore/nOpenaiServer/inference/engine/core/mhc_bayesian.zig`

### Key Components

#### 1. BayesianCurvatureEstimator Struct
The main estimator with conjugate Gaussian priors:

```zig
pub const BayesianCurvatureEstimator = struct {
    prior_mean: f32 = 0.0,           // Prior curvature
    prior_variance: f32 = 1.0,       // Prior uncertainty
    likelihood_variance: f32 = 0.1,   // Observation noise
    posterior_mean: f32 = 0.0,       // Updated estimate
    posterior_variance: f32 = 1.0,   // Updated uncertainty
    observation_count: u32 = 0,
    observation_sum: f32 = 0.0,
};
```

#### 2. Gaussian Prior/Likelihood Functions
- **`log_prior(curvature)`** - Log Gaussian prior: log N(κ | μ₀, σ₀²)
- **`log_likelihood(curvature, observations)`** - Log likelihood: Σ log N(obs_i | κ, σ²)
- **`log_posterior(curvature, observations)`** - Unnormalized log posterior

#### 3. Posterior Update (Conjugate Gaussian)
Uses closed-form conjugate update formulas:
```
posterior_precision = 1/prior_variance + n/likelihood_variance
posterior_variance = 1/posterior_precision
posterior_mean = posterior_variance × (prior_mean/prior_variance + Σobs/likelihood_variance)
```

#### 4. Credible Intervals
- **`credible_interval(level)`** - Compute symmetric CI using normal quantiles
- **`highest_density_interval(level)`** - HDI (equals CI for Gaussian)
- Returns `Interval{lower, upper}` for specified confidence level

#### 5. Calibration Metrics
- **`expected_calibration_error(predictions, outcomes, n_bins)`** - ECE
- **`maximum_calibration_error(predictions, outcomes, n_bins)`** - MCE
- **`brier_score(predictions, outcomes)`** - Mean squared error of predictions
- **`compute_reliability_diagram(...)`** - Generate reliability diagram data

## API Reference

### Types

| Type | Description |
|------|-------------|
| `BayesianCurvatureEstimator` | Main Bayesian estimator struct |
| `Interval` | Lower/upper bounds for intervals |
| `CalibrationResult` | Calibration error with bin statistics |
| `ReliabilityDiagram` | Data for calibration visualization |
| `BayesianError` | Error enum for invalid inputs |

### Functions

| Function | Description |
|----------|-------------|
| `init()` | Create estimator with defaults |
| `initWithParams(mean, prior_var, lik_var)` | Custom initialization |
| `update_posterior(observations)` | Batch posterior update |
| `update_single(observation)` | Incremental update |
| `credible_interval(level)` | Compute confidence interval |
| `get_map_estimate()` | Get MAP curvature estimate |
| `get_posterior_std()` | Get posterior standard deviation |
| `log_prior(κ)` | Compute log prior |
| `log_likelihood(κ, obs)` | Compute log likelihood |
| `log_posterior(κ, obs)` | Compute log posterior |
| `expected_calibration_error(...)` | Compute ECE |
| `maximum_calibration_error(...)` | Compute MCE |
| `brier_score(pred, outcomes)` | Compute Brier score |

### Helper Functions

| Function | Description |
|----------|-------------|
| `log_gaussian(x, mean, var)` | Log Gaussian PDF |
| `gaussian_pdf(x, mean, var)` | Gaussian PDF |
| `standard_normal_cdf(x)` | Standard normal CDF (Abramowitz-Stegun) |
| `normal_quantile(p)` | Inverse CDF (Beasley-Springer-Moro) |
| `generate_synthetic_observations(...)` | Generate test data |

## Usage Example

```zig
const bayesian = @import("mhc_bayesian.zig");

pub fn estimate_curvature() !void {
    // Create estimator with prior mean=0, variance=1
    var estimator = bayesian.BayesianCurvatureEstimator.init();

    // Update with curvature observations
    const observations = [_]f32{ -0.3, -0.25, -0.35, -0.28, -0.32 };
    try estimator.update_posterior(&observations);

    // Get MAP estimate and uncertainty
    const curvature = estimator.get_map_estimate();
    const uncertainty = estimator.get_posterior_std();

    // Compute 95% credible interval
    const ci = try estimator.credible_interval(0.95);

    std.debug.print("Curvature: {d:.4} ± {d:.4}\n", .{ curvature, uncertainty });
    std.debug.print("95% CI: [{d:.4}, {d:.4}]\n", .{ ci.lower, ci.upper });
}
```

## Calibration Metrics

### Expected Calibration Error (ECE)
```
ECE = Σᵢ (|bin_i| / n) × |accuracy_i - confidence_i|
```
Measures weighted average calibration error across bins.

### Maximum Calibration Error (MCE)
```
MCE = max_i |accuracy_i - confidence_i|
```
Measures worst-case calibration in any bin.

### Brier Score
```
Brier = (1/n) × Σᵢ (prediction_i - outcome_i)²
```
Measures mean squared error of probabilistic predictions.

## Mathematical Background

### Conjugate Gaussian Update
For prior N(μ₀, σ₀²) and likelihood N(x|κ, σ²):
- Posterior precision: τ = 1/σ₀² + n/σ²
- Posterior variance: σₚ² = 1/τ
- Posterior mean: μₚ = σₚ² × (μ₀/σ₀² + Σx/σ²)

### Normal Quantile Approximation
Uses Beasley-Springer-Moro algorithm with rational polynomial approximations for different regions (lower/central/upper).

## Test Coverage

39 unit tests covering:
- Estimator initialization and validation
- Prior/likelihood computation
- Posterior update (single/batch)
- Credible interval calculation
- Calibration metrics (ECE, MCE, Brier)
- Synthetic data generation
- Edge cases and error handling

Run tests:
```bash
cd src/serviceCore/nOpenaiServer/inference/engine/core
zig test mhc_bayesian.zig
```

## Statistics

- **Lines of Code**: 972
- **Test Cases**: 39
- **Functions**: 20+
- **Dependencies**: std library only

## Integration with mHC

This module integrates with:
- `mhc_geometry_detector.zig` - Probabilistic geometry classification
- `mhc_configuration.zig` - Bayesian hyperparameters
- `mhc_hyperbolic.zig` / `mhc_spherical.zig` - Curvature-based operations

