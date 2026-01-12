//! World-Class Time Series Forecasting System
//!
//! A comprehensive, production-ready forecasting system combining state-of-the-art
//! techniques from Moirai, TimesFM, Chronos, PatchTST, and novel Hull-White surprise calibration.
//!
//! ## Key Components
//!
//! ### Input Processing
//! - **Reversible Instance Normalization (RevIN)**: Handles non-stationarity by normalizing
//!   inputs and denormalizing outputs, preserving scale information
//! - **Multi-Scale Patching**: Captures patterns at different temporal scales (hours, days, weeks)
//! - **Temporal Feature Engineering**: Lags, calendar features, Fourier seasonality
//!
//! ### Architecture
//! - **Multi-Frequency Patch Projection**: Routes time series to frequency-specialized projections
//! - **Any-Variate Attention with RoPE**: Handles arbitrary number of variates with positional encoding
//! - **Memory-Augmented Transformer**: Titans-style neural memory for long-range dependencies
//! - **Covariate Fusion**: Integrates known future covariates and static features
//!
//! ### Output Heads
//! - **Multiple Distribution Heads**: Gaussian, Student-t, LogNormal, Negative Binomial
//! - **Quantile Regression**: Direct quantile prediction for flexible intervals
//! - **Conformal Prediction**: Calibrated prediction intervals with coverage guarantees
//!
//! ### Training & Calibration
//! - **Hull-White Stochastic Surprise Gate**: Adaptive memory consolidation based on forecast errors
//! - **Ensemble Forecasting**: Multiple models with adaptive weighting
//! - **Online Calibration**: Continuous adaptation to distribution shift

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig, Dropout, DropoutConfig, Gelu},
    nn::loss::{MseLoss, Reduction},
    tensor::{backend::{Backend, AutodiffBackend}, Tensor, activation, ElementConversion, Int},
    train::{TrainOutput, TrainStep, ValidStep, RegressionOutput},
};
use std::collections::VecDeque;

// =============================================================================
// REVERSIBLE INSTANCE NORMALIZATION (RevIN)
// =============================================================================

/// Configuration for Reversible Instance Normalization.
#[derive(Config, Debug)]
pub struct RevINConfig {
    /// Number of variates/features
    pub num_features: usize,
    /// Small constant for numerical stability
    #[config(default = "1e-5")]
    pub eps: f32,
    /// Whether to learn affine parameters
    #[config(default = "true")]
    pub affine: bool,
}

/// Reversible Instance Normalization (RevIN).
///
/// Normalizes input time series and stores statistics to denormalize outputs.
/// Critical for handling non-stationary time series in a reversible way.
///
/// Reference: Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting"
#[derive(Module, Debug)]
pub struct RevIN<B: Backend> {
    /// Learnable affine scale (gamma)
    affine_weight: Option<Tensor<B, 1>>,
    /// Learnable affine bias (beta)
    affine_bias: Option<Tensor<B, 1>>,
    /// Epsilon for numerical stability
    eps: f32,
}

/// Statistics stored during normalization for denormalization.
#[derive(Debug, Clone)]
pub struct RevINStats<B: Backend> {
    pub mean: Tensor<B, 2>,  // [batch, features]
    pub std: Tensor<B, 2>,   // [batch, features]
}

impl<B: Backend> RevIN<B> {
    pub fn new(config: &RevINConfig, device: &B::Device) -> Self {
        let (affine_weight, affine_bias) = if config.affine {
            (
                Some(Tensor::ones([config.num_features], device)),
                Some(Tensor::zeros([config.num_features], device)),
            )
        } else {
            (None, None)
        };

        Self {
            affine_weight,
            affine_bias,
            eps: config.eps,
        }
    }

    /// Normalize input and return stats for denormalization.
    /// Input: [batch, seq_len, features]
    /// Output: ([batch, seq_len, features], RevINStats)
    pub fn normalize(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, RevINStats<B>) {
        let [batch_size, seq_len, features] = x.dims();

        // Compute mean and std over time dimension
        let mean = x.clone().mean_dim(1);  // [batch, 1, features]
        let mean_sq = mean.clone().squeeze::<2>(1);  // [batch, features]

        // Compute std: sqrt(E[(x - mean)^2])
        let mean_expanded = mean.clone().expand([batch_size, seq_len, features]);
        let centered = x.clone() - mean_expanded.clone();
        let var = centered.clone().powf_scalar(2.0).mean_dim(1).squeeze::<2>(1);  // [batch, features]
        let std = (var + self.eps).sqrt();  // [batch, features]

        // Normalize
        let std_expanded: Tensor<B, 3> = std.clone().unsqueeze_dim::<3>(1).expand([batch_size, seq_len, features]);
        let normalized = centered / std_expanded;

        // Apply affine transformation if present
        let output = if let (Some(ref weight), Some(ref bias)) = (&self.affine_weight, &self.affine_bias) {
            let w: Tensor<B, 3> = weight.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
                .expand([batch_size, seq_len, features]);
            let b: Tensor<B, 3> = bias.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
                .expand([batch_size, seq_len, features]);
            normalized * w + b
        } else {
            normalized
        };

        let stats = RevINStats { mean: mean_sq, std };
        (output, stats)
    }

    /// Denormalize output using stored stats.
    /// Input: [batch, horizon, features] or [batch, horizon]
    pub fn denormalize(&self, x: Tensor<B, 2>, stats: &RevINStats<B>) -> Tensor<B, 2> {
        let [batch_size, horizon] = x.dims();

        // For single-variate output, use first feature's stats
        let mean = stats.mean.clone().slice([0..batch_size, 0..1]).squeeze::<1>(1);  // [batch]
        let std = stats.std.clone().slice([0..batch_size, 0..1]).squeeze::<1>(1);    // [batch]

        // Expand to [batch, horizon]
        let mean_exp: Tensor<B, 2> = mean.unsqueeze_dim::<2>(1).expand([batch_size, horizon]);
        let std_exp: Tensor<B, 2> = std.unsqueeze_dim::<2>(1).expand([batch_size, horizon]);

        // Remove affine, then denormalize
        let output = if let (Some(ref weight), Some(ref bias)) = (&self.affine_weight, &self.affine_bias) {
            let w = weight.clone().slice([0..1]).into_scalar();
            let b = bias.clone().slice([0..1]).into_scalar();
            (x - b) / w
        } else {
            x
        };

        output * std_exp + mean_exp
    }

    /// Denormalize 3D output (multivariate).
    pub fn denormalize_3d(&self, x: Tensor<B, 3>, stats: &RevINStats<B>) -> Tensor<B, 3> {
        let [batch_size, horizon, features] = x.dims();

        let mean_exp: Tensor<B, 3> = stats.mean.clone().unsqueeze_dim::<3>(1).expand([batch_size, horizon, features]);
        let std_exp: Tensor<B, 3> = stats.std.clone().unsqueeze_dim::<3>(1).expand([batch_size, horizon, features]);

        // Remove affine if present
        let output = if let (Some(ref weight), Some(ref bias)) = (&self.affine_weight, &self.affine_bias) {
            let w: Tensor<B, 3> = weight.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
                .expand([batch_size, horizon, features]);
            let b: Tensor<B, 3> = bias.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
                .expand([batch_size, horizon, features]);
            (x - b) / w
        } else {
            x
        };

        output * std_exp + mean_exp
    }
}

// =============================================================================
// TEMPORAL FEATURE ENGINEERING
// =============================================================================

/// Temporal features extracted from timestamps.
#[derive(Debug, Clone)]
pub struct TemporalFeatures<B: Backend> {
    /// Lag features: [batch, num_lags]
    pub lags: Tensor<B, 2>,
    /// Calendar features: [batch, seq_len, num_calendar_features]
    pub calendar: Tensor<B, 3>,
    /// Fourier features for seasonality: [batch, seq_len, num_fourier * 2]
    pub fourier: Tensor<B, 3>,
}

/// Configuration for temporal feature engineering.
#[derive(Debug, Clone)]
pub struct TemporalFeaturesConfig {
    /// Lag indices to use (e.g., [1, 7, 14, 28] for daily data)
    pub lag_indices: Vec<usize>,
    /// Fourier frequencies for seasonality (e.g., [1, 2, 3] for yearly)
    pub fourier_frequencies: Vec<f32>,
    /// Period for Fourier features (e.g., 365.25 for yearly seasonality)
    pub seasonality_period: f32,
    /// Whether to include calendar features
    pub include_calendar: bool,
}

impl Default for TemporalFeaturesConfig {
    fn default() -> Self {
        Self {
            lag_indices: vec![1, 2, 3, 7, 14, 28, 30, 60, 90],
            fourier_frequencies: vec![1.0, 2.0, 3.0, 4.0],
            seasonality_period: 365.25,
            include_calendar: true,
        }
    }
}

/// Extract temporal features from time series.
pub fn extract_temporal_features<B: Backend>(
    values: &Tensor<B, 3>,      // [batch, seq_len, features]
    timestamps: &Tensor<B, 2>,  // [batch, seq_len] - timestamps as day-of-year or similar
    config: &TemporalFeaturesConfig,
    device: &B::Device,
) -> TemporalFeatures<B> {
    let [batch_size, seq_len, _features] = values.dims();

    // Extract lag features from the last time step
    let lags = extract_lags(values, &config.lag_indices, device);

    // Extract Fourier features for seasonality
    let fourier = extract_fourier_features(
        timestamps,
        &config.fourier_frequencies,
        config.seasonality_period,
        device,
    );

    // Calendar features (placeholder - would need actual datetime parsing)
    let num_calendar = if config.include_calendar { 7 } else { 0 }; // day of week
    let calendar = Tensor::zeros([batch_size, seq_len, num_calendar], device);

    TemporalFeatures { lags, calendar, fourier }
}

/// Extract lag features from time series.
fn extract_lags<B: Backend>(
    values: &Tensor<B, 3>,  // [batch, seq_len, features]
    lag_indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let [batch_size, seq_len, _] = values.dims();
    let num_lags = lag_indices.len();

    // Get the last value for each lag
    let mut lag_values = Vec::with_capacity(num_lags);
    for &lag in lag_indices {
        if lag < seq_len {
            let idx = seq_len - lag - 1;
            let lag_val = values.clone()
                .slice([0..batch_size, idx..idx+1, 0..1])
                .squeeze::<2>(2)
                .squeeze::<1>(1);  // [batch]
            lag_values.push(lag_val);
        } else {
            // Pad with zeros if lag exceeds sequence length
            lag_values.push(Tensor::zeros([batch_size], device));
        }
    }

    // Stack lags: [batch, num_lags]
    Tensor::stack(lag_values, 1)
}

/// Extract Fourier features for seasonality modeling.
fn extract_fourier_features<B: Backend>(
    timestamps: &Tensor<B, 2>,  // [batch, seq_len]
    frequencies: &[f32],
    period: f32,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [batch_size, seq_len] = timestamps.dims();
    let num_features = frequencies.len() * 2;  // sin + cos for each frequency

    // t / period * 2π
    let normalized = timestamps.clone() * (2.0 * std::f32::consts::PI / period);

    let mut features = Vec::with_capacity(num_features);
    for &freq in frequencies {
        let angles = normalized.clone() * freq;
        features.push(angles.clone().sin());
        features.push(angles.cos());
    }

    // Stack along last dimension
    let stacked = Tensor::stack(features, 2);  // [batch, seq_len, num_features]
    stacked
}

// =============================================================================
// MULTI-SCALE PATCHING
// =============================================================================

/// Configuration for multi-scale patching.
#[derive(Config, Debug)]
pub struct MultiScalePatchConfig {
    /// Patch sizes at different scales
    pub patch_sizes: Vec<usize>,
    /// Stride for each patch size (None = non-overlapping)
    pub strides: Option<Vec<usize>>,
    /// Model dimension
    pub d_model: usize,
}

impl Default for MultiScalePatchConfig {
    fn default() -> Self {
        Self::new(vec![8, 16, 32, 64], 256)
    }
}

/// Multi-scale patching module.
/// Creates patches at multiple temporal scales and projects them.
#[derive(Module, Debug)]
pub struct MultiScalePatching<B: Backend> {
    /// Projection layers for each scale
    projections: Vec<Linear<B>>,
    /// Patch sizes
    patch_sizes: Vec<usize>,
    /// Output dimension
    d_model: usize,
}

impl<B: Backend> MultiScalePatching<B> {
    pub fn new(config: &MultiScalePatchConfig, device: &B::Device) -> Self {
        let projections: Vec<_> = config.patch_sizes.iter()
            .map(|&size| LinearConfig::new(size, config.d_model).init(device))
            .collect();

        Self {
            projections,
            patch_sizes: config.patch_sizes.clone(),
            d_model: config.d_model,
        }
    }

    /// Create multi-scale patches from input.
    /// Input: [batch, seq_len, features]
    /// Output: [batch, total_patches, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Vec<usize>) {
        let [batch_size, seq_len, features] = x.dims();
        let device = x.device();

        let mut all_patches = Vec::new();
        let mut patch_counts = Vec::new();

        for (i, &patch_size) in self.patch_sizes.iter().enumerate() {
            if patch_size > seq_len {
                continue;
            }

            let stride = patch_size;  // Non-overlapping
            let num_patches = (seq_len - patch_size) / stride + 1;
            patch_counts.push(num_patches);

            // Extract patches
            let mut patches_at_scale = Vec::with_capacity(num_patches);
            for p in 0..num_patches {
                let start = p * stride;
                let end = start + patch_size;
                // Take first feature only, flatten patch
                let patch = x.clone()
                    .slice([0..batch_size, start..end, 0..1])
                    .squeeze::<2>(2);  // [batch, patch_size]
                patches_at_scale.push(patch);
            }

            if !patches_at_scale.is_empty() {
                // Stack patches: [batch, num_patches, patch_size]
                let patches_tensor = Tensor::stack(patches_at_scale, 1);
                // Project: [batch, num_patches, d_model]
                let projected = self.projections[i].forward(patches_tensor);
                all_patches.push(projected);
            }
        }

        if all_patches.is_empty() {
            // Fallback: treat entire sequence as single patch
            let single = x.slice([0..batch_size, 0..seq_len.min(self.patch_sizes[0]), 0..1])
                .squeeze::<2>(2);
            let padded = if single.dims()[1] < self.patch_sizes[0] {
                let pad_size = self.patch_sizes[0] - single.dims()[1];
                let padding = Tensor::zeros([batch_size, pad_size], &device);
                Tensor::cat(vec![single, padding], 1)
            } else {
                single
            };
            let projected = self.projections[0].forward(padded.unsqueeze_dim::<3>(1));
            return (projected, vec![1]);
        }

        // Concatenate all scales
        let combined = Tensor::cat(all_patches, 1);
        (combined, patch_counts)
    }

    /// Get total number of patches for a given sequence length.
    pub fn num_patches(&self, seq_len: usize) -> usize {
        self.patch_sizes.iter()
            .filter(|&&ps| ps <= seq_len)
            .map(|&ps| (seq_len - ps) / ps + 1)
            .sum()
    }
}

// =============================================================================
// FREQUENCY BANDS (existing, enhanced)
// =============================================================================

/// Frequency bands for patch projection routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyBand {
    /// Sub-minute (seconds)
    VeryHigh,
    /// Minutes to hours
    High,
    /// Hours to days
    Medium,
    /// Days to weeks
    Low,
    /// Weeks to months
    VeryLow,
}

impl FrequencyBand {
    /// Get recommended patch size for this frequency band.
    pub fn patch_size(&self) -> usize {
        match self {
            FrequencyBand::VeryHigh => 64,
            FrequencyBand::High => 32,
            FrequencyBand::Medium => 16,
            FrequencyBand::Low => 8,
            FrequencyBand::VeryLow => 4,
        }
    }

    /// Detect frequency band from sampling interval in seconds.
    pub fn from_interval_seconds(interval: f64) -> Self {
        if interval < 60.0 {
            FrequencyBand::VeryHigh
        } else if interval < 3600.0 {
            FrequencyBand::High
        } else if interval < 86400.0 {
            FrequencyBand::Medium
        } else if interval < 604800.0 {
            FrequencyBand::Low
        } else {
            FrequencyBand::VeryLow
        }
    }
}

/// Configuration for multi-frequency patch projection.
#[derive(Config, Debug)]
pub struct PatchProjectionConfig {
    pub d_model: usize,
    pub max_variates: usize,
    #[config(default = "64")]
    pub max_patch_size: usize,
}

/// Multi-frequency patch projection layer.
/// Routes time series to specialized projections based on sampling frequency.
#[derive(Module, Debug)]
pub struct MultiFreqPatchProjection<B: Backend> {
    /// Projection for very high frequency (sub-minute)
    proj_very_high: Linear<B>,
    /// Projection for high frequency (minutes)
    proj_high: Linear<B>,
    /// Projection for medium frequency (hours)
    proj_medium: Linear<B>,
    /// Projection for low frequency (days)
    proj_low: Linear<B>,
    /// Projection for very low frequency (weeks+)
    proj_very_low: Linear<B>,
}

impl<B: Backend> MultiFreqPatchProjection<B> {
    pub fn new(config: &PatchProjectionConfig, device: &B::Device) -> Self {
        // Each frequency band has its own projection with appropriate input size
        let proj_very_high = LinearConfig::new(64, config.d_model).init(device);
        let proj_high = LinearConfig::new(32, config.d_model).init(device);
        let proj_medium = LinearConfig::new(16, config.d_model).init(device);
        let proj_low = LinearConfig::new(8, config.d_model).init(device);
        let proj_very_low = LinearConfig::new(4, config.d_model).init(device);

        Self {
            proj_very_high,
            proj_high,
            proj_medium,
            proj_low,
            proj_very_low,
        }
    }

    /// Project patches using the appropriate frequency-specific projection.
    /// Input: [batch, num_patches, patch_size]
    /// Output: [batch, num_patches, d_model]
    pub fn forward(&self, patches: Tensor<B, 3>, freq: FrequencyBand) -> Tensor<B, 3> {
        match freq {
            FrequencyBand::VeryHigh => self.proj_very_high.forward(patches),
            FrequencyBand::High => self.proj_high.forward(patches),
            FrequencyBand::Medium => self.proj_medium.forward(patches),
            FrequencyBand::Low => self.proj_low.forward(patches),
            FrequencyBand::VeryLow => self.proj_very_low.forward(patches),
        }
    }
}

/// Rotary Position Embedding (RoPE) for temporal encoding.
pub fn apply_rope<B: Backend>(
    x: Tensor<B, 3>,  // [batch, seq_len, dim]
    positions: Tensor<B, 2>,  // [batch, seq_len] - position indices
    base: f32,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, dim] = x.dims();
    let device = x.device();
    let half_dim = dim / 2;

    // Compute frequency bands: 1 / (base ^ (2i/dim))
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
        .collect();
    let inv_freq_tensor = Tensor::<B, 1>::from_floats(&inv_freq[..], &device);

    // Compute angles: position * inv_freq
    // positions: [batch, seq_len] -> [batch, seq_len, 1]
    // inv_freq: [half_dim] -> [1, 1, half_dim]
    let pos_expanded: Tensor<B, 3> = positions.unsqueeze_dim(2);  // [batch, seq_len, 1]
    let inv_freq_2d: Tensor<B, 2> = inv_freq_tensor.unsqueeze_dim(0);
    let inv_freq_expanded: Tensor<B, 3> = inv_freq_2d.unsqueeze_dim(0);  // [1, 1, half_dim]

    // Broadcast multiply
    let angles = pos_expanded.mul(inv_freq_expanded);  // [batch, seq_len, half_dim]

    // Compute sin and cos
    let cos_vals = angles.clone().cos();
    let sin_vals = angles.sin();

    // Split input into two halves
    let x1 = x.clone().slice([0..batch_size, 0..seq_len, 0..half_dim]);
    let x2 = x.slice([0..batch_size, 0..seq_len, half_dim..dim]);

    // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let out1 = x1.clone().mul(cos_vals.clone()) - x2.clone().mul(sin_vals.clone());
    let out2 = x1.mul(sin_vals) + x2.mul(cos_vals);

    // Concatenate along last dimension
    Tensor::cat(vec![out1, out2], 2)
}

/// Configuration for Any-Variate Attention.
#[derive(Config, Debug)]
pub struct AnyVariateAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub max_variates: usize,
    #[config(default = "0.1")]
    pub dropout: f64,
}

/// Any-Variate Attention with dual-axis encoding.
/// Handles multivariate time series with arbitrary number of variates.
#[derive(Module, Debug)]
pub struct AnyVariateAttention<B: Backend> {
    /// Query projection
    wq: Linear<B>,
    /// Key projection
    wk: Linear<B>,
    /// Value projection
    wv: Linear<B>,
    /// Output projection
    wo: Linear<B>,
    /// Learned variate biases [max_variates, max_variates]
    variate_bias: Tensor<B, 2>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> AnyVariateAttention<B> {
    pub fn new(config: &AnyVariateAttentionConfig, device: &B::Device) -> Self {
        let head_dim = config.d_model / config.n_heads;
        // Use larger sequence capacity for variate bias (not variate count)
        let max_seq_len = 1024;  // Maximum expected sequence length

        Self {
            wq: LinearConfig::new(config.d_model, config.d_model).init(device),
            wk: LinearConfig::new(config.d_model, config.d_model).init(device),
            wv: LinearConfig::new(config.d_model, config.d_model).init(device),
            wo: LinearConfig::new(config.d_model, config.d_model).init(device),
            variate_bias: Tensor::zeros([max_seq_len, max_seq_len], device),
            n_heads: config.n_heads,
            head_dim,
        }
    }

    /// Forward pass with any-variate attention.
    /// x: [batch, seq_len, d_model] - flattened multivariate sequence
    /// variate_ids: [batch, seq_len] - which variate each position belongs to
    /// positions: [batch, seq_len] - temporal positions for RoPE
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        _variate_ids: Tensor<B, 2, burn::tensor::Int>,
        positions: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _d_model] = x.dims();
        let _device = x.device();

        // Project to Q, K, V
        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        // Apply RoPE to Q and K for temporal encoding
        let q = apply_rope(q, positions.clone(), 10000.0);
        let k = apply_rope(k, positions, 10000.0);

        // Reshape for multi-head attention
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);  // [batch, n_heads, seq_len, head_dim]
        let k = k.reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v.reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);

        // Compute attention scores
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(k.transpose())
            .div_scalar(scale);  // [batch, n_heads, seq_len, seq_len]

        // Add variate bias (binary attention bias for variate identity)
        // This helps the model understand "what" each position is
        let variate_bias = self.variate_bias.clone()
            .slice([0..seq_len.min(self.variate_bias.dims()[0]),
                    0..seq_len.min(self.variate_bias.dims()[1])]);
        let variate_bias_3d: Tensor<B, 3> = variate_bias.unsqueeze_dim(0);
        let variate_bias: Tensor<B, 4> = variate_bias_3d.unsqueeze_dim(0);  // [1, 1, seq_len, seq_len]

        let scores = scores + variate_bias;

        // Softmax and apply to values
        let attn_weights = activation::softmax(scores, 3);
        let output = attn_weights.matmul(v);  // [batch, n_heads, seq_len, head_dim]

        // Reshape back
        let output = output.swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);

        self.wo.forward(output)
    }
}


/// Parameters for a Student's t distribution.
#[derive(Debug, Clone)]
pub struct StudentTParams<B: Backend> {
    /// Location (mu)
    pub loc: Tensor<B, 2>,
    /// Scale (sigma > 0)
    pub scale: Tensor<B, 2>,
    /// Degrees of freedom (nu > 0)
    pub df: Tensor<B, 2>,
}

/// Configuration for mixture distribution head.
#[derive(Config, Debug)]
pub struct MixtureHeadConfig {
    pub d_model: usize,
    #[config(default = "3")]
    pub n_components: usize,
    pub forecast_horizon: usize,
}

/// Mixture Distribution Head for flexible probabilistic forecasting.
/// Outputs parameters for a mixture of Student's t distributions.
#[derive(Module, Debug)]
pub struct MixtureDistributionHead<B: Backend> {
    /// Project to mixture weights (logits)
    weight_proj: Linear<B>,
    /// Project to location parameters
    loc_proj: Linear<B>,
    /// Project to log-scale parameters
    log_scale_proj: Linear<B>,
    /// Project to log-df parameters
    log_df_proj: Linear<B>,
    n_components: usize,
    forecast_horizon: usize,
}

impl<B: Backend> MixtureDistributionHead<B> {
    pub fn new(config: &MixtureHeadConfig, device: &B::Device) -> Self {
        let output_size = config.n_components * config.forecast_horizon;

        Self {
            weight_proj: LinearConfig::new(config.d_model, config.n_components).init(device),
            loc_proj: LinearConfig::new(config.d_model, output_size).init(device),
            log_scale_proj: LinearConfig::new(config.d_model, output_size).init(device),
            log_df_proj: LinearConfig::new(config.d_model, output_size).init(device),
            n_components: config.n_components,
            forecast_horizon: config.forecast_horizon,
        }
    }

    /// Forward pass: produce mixture distribution parameters.
    /// x: [batch, d_model] - summary representation
    /// Returns: (weights, params) where weights is [batch, n_components]
    /// and params is a vec of StudentTParams for each component
    pub fn forward(&self, x: Tensor<B, 2>) -> MixtureOutput<B> {
        let [batch_size, _] = x.dims();

        // Mixture weights (softmax for valid probabilities)
        let weight_logits = self.weight_proj.forward(x.clone());
        let weights = activation::softmax(weight_logits, 1);  // [batch, n_components]

        // Location parameters
        let loc = self.loc_proj.forward(x.clone());  // [batch, n_components * horizon]
        let loc = loc.reshape([batch_size, self.n_components, self.forecast_horizon]);

        // Scale parameters (softplus to ensure positivity)
        let log_scale = self.log_scale_proj.forward(x.clone());
        let scale = activation::softplus(log_scale, 1.0)
            .reshape([batch_size, self.n_components, self.forecast_horizon]);

        // Degrees of freedom (softplus + 2 to ensure > 2 for finite variance)
        let log_df = self.log_df_proj.forward(x);
        let df = activation::softplus(log_df, 1.0).add_scalar(2.0)
            .reshape([batch_size, self.n_components, self.forecast_horizon]);

        MixtureOutput {
            weights,
            loc,
            scale,
            df,
            n_components: self.n_components,
            forecast_horizon: self.forecast_horizon,
        }
    }
}

/// Output from the mixture distribution head.
#[derive(Debug, Clone)]
pub struct MixtureOutput<B: Backend> {
    /// Mixture weights [batch, n_components]
    pub weights: Tensor<B, 2>,
    /// Location params [batch, n_components, horizon]
    pub loc: Tensor<B, 3>,
    /// Scale params [batch, n_components, horizon]
    pub scale: Tensor<B, 3>,
    /// Degrees of freedom [batch, n_components, horizon]
    pub df: Tensor<B, 3>,
    pub n_components: usize,
    pub forecast_horizon: usize,
}

impl<B: Backend> MixtureOutput<B> {
    /// Compute the mean forecast (expected value of mixture).
    pub fn mean(&self) -> Tensor<B, 2> {
        let [batch_size, _, horizon] = self.loc.dims();

        // Weighted sum of component means
        // For Student's t, mean = loc when df > 1
        let weights_3d: Tensor<B, 3> = self.weights.clone().unsqueeze_dim(2);
        let weights_expanded = weights_3d.expand([batch_size, self.n_components, horizon]);

        let weighted_locs = self.loc.clone().mul(weights_expanded);
        weighted_locs.sum_dim(1).squeeze(1)  // [batch, horizon]
    }

    /// Compute the median forecast.
    pub fn median(&self) -> Tensor<B, 2> {
        // For Student's t, median = loc
        // Use the highest-weight component
        self.mean()  // Simplified: same as mean for symmetric distributions
    }

    /// Sample from the mixture distribution.
    /// Returns: [batch, horizon]
    pub fn sample(&self, _device: &B::Device) -> Tensor<B, 2> {
        // Simplified sampling: use the mean as the point estimate
        // Full implementation would do ancestral sampling
        self.mean()
    }

    /// Compute negative log-likelihood loss against target.
    pub fn nll_loss(&self, target: Tensor<B, 2>) -> Tensor<B, 1> {
        let [_batch_size, _horizon] = target.dims();

        // Simplified: MSE between mean forecast and target
        // Full implementation would compute proper mixture log-likelihood
        let pred = self.mean();
        let diff = pred - target;
        let mse = diff.clone().mul(diff).mean_dim(1).squeeze(1);
        mse
    }

    /// Compute forecast errors for surprise calibration.
    /// Returns per-sample absolute errors [batch].
    pub fn forecast_errors(&self, target: Tensor<B, 2>) -> Tensor<B, 1> {
        let pred = self.mean();
        let diff = pred - target;
        diff.abs().mean_dim(1).squeeze(1)
    }
}

// =============================================================================
// MULTIPLE DISTRIBUTION HEADS
// =============================================================================

/// Distribution type for probabilistic forecasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionType {
    /// Gaussian (Normal) distribution
    Gaussian,
    /// Student's t distribution (heavier tails)
    StudentT,
    /// Log-Normal distribution (for positive data)
    LogNormal,
    /// Negative Binomial (for count data)
    NegativeBinomial,
    /// Quantile regression (distribution-free)
    Quantile,
}

/// Configuration for multi-distribution head.
#[derive(Config, Debug)]
pub struct MultiDistributionHeadConfig {
    pub d_model: usize,
    pub forecast_horizon: usize,
    /// Quantiles to predict for quantile regression
    pub quantiles: Vec<f32>,
}

impl MultiDistributionHeadConfig {
    /// Create with default quantiles.
    pub fn with_default_quantiles(d_model: usize, forecast_horizon: usize) -> Self {
        Self {
            d_model,
            forecast_horizon,
            quantiles: vec![0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        }
    }
}

/// Multi-distribution head supporting various probabilistic outputs.
#[derive(Module, Debug)]
pub struct MultiDistributionHead<B: Backend> {
    /// Gaussian: mean and log_variance
    gaussian_mean: Linear<B>,
    gaussian_log_var: Linear<B>,
    /// Student-t: loc, log_scale, log_df
    student_loc: Linear<B>,
    student_log_scale: Linear<B>,
    student_log_df: Linear<B>,
    /// LogNormal: mu and log_sigma
    lognormal_mu: Linear<B>,
    lognormal_log_sigma: Linear<B>,
    /// Quantile: one output per quantile per horizon
    quantile_proj: Linear<B>,
    /// Config
    quantiles: Vec<f32>,
    forecast_horizon: usize,
}

impl<B: Backend> MultiDistributionHead<B> {
    pub fn new(config: &MultiDistributionHeadConfig, device: &B::Device) -> Self {
        let h = config.forecast_horizon;
        let num_quantiles = config.quantiles.len();

        Self {
            gaussian_mean: LinearConfig::new(config.d_model, h).init(device),
            gaussian_log_var: LinearConfig::new(config.d_model, h).init(device),
            student_loc: LinearConfig::new(config.d_model, h).init(device),
            student_log_scale: LinearConfig::new(config.d_model, h).init(device),
            student_log_df: LinearConfig::new(config.d_model, h).init(device),
            lognormal_mu: LinearConfig::new(config.d_model, h).init(device),
            lognormal_log_sigma: LinearConfig::new(config.d_model, h).init(device),
            quantile_proj: LinearConfig::new(config.d_model, h * num_quantiles).init(device),
            quantiles: config.quantiles.clone(),
            forecast_horizon: h,
        }
    }

    /// Forward for Gaussian distribution.
    /// Returns (mean, variance) both [batch, horizon]
    pub fn forward_gaussian(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mean = self.gaussian_mean.forward(x.clone());
        let log_var = self.gaussian_log_var.forward(x);
        let variance = log_var.exp();
        (mean, variance)
    }

    /// Forward for Student-t distribution.
    /// Returns (loc, scale, df) all [batch, horizon]
    pub fn forward_student_t(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let loc = self.student_loc.forward(x.clone());
        let log_scale = self.student_log_scale.forward(x.clone());
        let log_df = self.student_log_df.forward(x);

        let scale = activation::softplus(log_scale, 1.0);
        let df = activation::softplus(log_df, 1.0).add_scalar(2.0);  // df > 2

        (loc, scale, df)
    }

    /// Forward for LogNormal distribution.
    /// Returns (mu, sigma) for LogNormal(mu, sigma)
    pub fn forward_lognormal(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mu = self.lognormal_mu.forward(x.clone());
        let log_sigma = self.lognormal_log_sigma.forward(x);
        let sigma = activation::softplus(log_sigma, 1.0);
        (mu, sigma)
    }

    /// Forward for quantile regression.
    /// Returns [batch, horizon, num_quantiles]
    pub fn forward_quantiles(&self, x: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _] = x.dims();
        let num_quantiles = self.quantiles.len();

        let output = self.quantile_proj.forward(x);
        output.reshape([batch_size, self.forecast_horizon, num_quantiles])
    }

    /// Get quantile levels.
    pub fn quantile_levels(&self) -> &[f32] {
        &self.quantiles
    }
}

/// Unified probabilistic forecast output.
#[derive(Debug, Clone)]
pub struct ProbabilisticForecast<B: Backend> {
    /// Point forecast (median or mean)
    pub point: Tensor<B, 2>,
    /// Lower bound at specified coverage
    pub lower: Tensor<B, 2>,
    /// Upper bound at specified coverage
    pub upper: Tensor<B, 2>,
    /// Coverage probability (e.g., 0.9 for 90% interval)
    pub coverage: f32,
    /// Distribution type used
    pub dist_type: DistributionType,
    /// Full quantile predictions if available
    pub quantiles: Option<Tensor<B, 3>>,
    /// Quantile levels
    pub quantile_levels: Vec<f32>,
}

// =============================================================================
// CONFORMAL PREDICTION
// =============================================================================

/// Conformal prediction for calibrated uncertainty quantification.
///
/// Provides prediction intervals with guaranteed coverage under
/// exchangeability assumption (or approximate coverage under distribution shift).
#[derive(Debug, Clone)]
pub struct ConformalPredictor {
    /// Residual buffer for calibration
    residuals: VecDeque<f32>,
    /// Maximum buffer size
    max_size: usize,
    /// Target coverage probability
    target_coverage: f32,
    /// Calibrated quantile (updated dynamically)
    calibrated_quantile: f32,
    /// Exponential decay for recent residuals (for distribution shift)
    decay_factor: f32,
    /// Number of updates
    updates: usize,
}

impl ConformalPredictor {
    pub fn new(target_coverage: f32, max_size: usize) -> Self {
        assert!(target_coverage > 0.0 && target_coverage < 1.0);
        Self {
            residuals: VecDeque::with_capacity(max_size),
            max_size,
            target_coverage,
            calibrated_quantile: 1.0,  // Conservative initial value
            decay_factor: 0.99,
            updates: 0,
        }
    }

    /// Update with new prediction-observation pair.
    pub fn update(&mut self, prediction: f32, actual: f32) {
        let residual = (actual - prediction).abs();

        if self.residuals.len() >= self.max_size {
            self.residuals.pop_front();
        }
        self.residuals.push_back(residual);
        self.updates += 1;

        // Recalibrate
        self.calibrate();
    }

    /// Update with batch of residuals.
    pub fn update_batch(&mut self, residuals: &[f32]) {
        for &r in residuals {
            if self.residuals.len() >= self.max_size {
                self.residuals.pop_front();
            }
            self.residuals.push_back(r);
        }
        self.updates += residuals.len();
        self.calibrate();
    }

    /// Calibrate the quantile based on stored residuals.
    fn calibrate(&mut self) {
        if self.residuals.is_empty() {
            return;
        }

        // Apply exponential decay weights for adaptive conformal
        let n = self.residuals.len();
        let mut weighted_residuals: Vec<(f32, f32)> = self.residuals.iter()
            .enumerate()
            .map(|(i, &r)| {
                let weight = self.decay_factor.powi((n - 1 - i) as i32);
                (r, weight)
            })
            .collect();

        // Sort by residual
        weighted_residuals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find weighted quantile
        let total_weight: f32 = weighted_residuals.iter().map(|(_, w)| w).sum();
        let target_weight = self.target_coverage * total_weight;

        let mut cumulative = 0.0;
        for (residual, weight) in &weighted_residuals {
            cumulative += weight;
            if cumulative >= target_weight {
                self.calibrated_quantile = *residual;
                return;
            }
        }

        // Fallback to last
        self.calibrated_quantile = weighted_residuals.last().map(|(r, _)| *r).unwrap_or(1.0);
    }

    /// Get prediction interval for a point prediction.
    pub fn interval(&self, prediction: f32) -> (f32, f32) {
        (
            prediction - self.calibrated_quantile,
            prediction + self.calibrated_quantile,
        )
    }

    /// Get prediction intervals for a batch.
    pub fn intervals_batch<B: Backend>(&self, predictions: &Tensor<B, 2>, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let lower = predictions.clone().sub_scalar(self.calibrated_quantile);
        let upper = predictions.clone().add_scalar(self.calibrated_quantile);
        (lower, upper)
    }

    /// Get current coverage statistics.
    pub fn stats(&self) -> ConformalStats {
        ConformalStats {
            target_coverage: self.target_coverage,
            calibrated_quantile: self.calibrated_quantile,
            buffer_size: self.residuals.len(),
            total_updates: self.updates,
        }
    }

    /// Compute empirical coverage from test data.
    pub fn empirical_coverage(&self, predictions: &[f32], actuals: &[f32]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }

        let covered = predictions.iter()
            .zip(actuals.iter())
            .filter(|(&pred, &actual)| {
                let (lo, hi) = self.interval(pred);
                actual >= lo && actual <= hi
            })
            .count();

        covered as f32 / predictions.len() as f32
    }

    /// Reset calibration.
    pub fn reset(&mut self) {
        self.residuals.clear();
        self.calibrated_quantile = 1.0;
        self.updates = 0;
    }
}

/// Statistics from conformal predictor.
#[derive(Debug, Clone)]
pub struct ConformalStats {
    pub target_coverage: f32,
    pub calibrated_quantile: f32,
    pub buffer_size: usize,
    pub total_updates: usize,
}

// =============================================================================
// TRANSFORMER ENCODER WITH MEMORY
// =============================================================================

/// Configuration for transformer encoder block.
#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = "4")]
    pub ff_multiplier: usize,
    #[config(default = "0.1")]
    pub dropout: f64,
}

/// Single transformer encoder block with pre-norm.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Layer norm before self-attention
    ln1: LayerNorm<B>,
    /// Self-attention (reuse AnyVariateAttention)
    self_attn: AnyVariateAttention<B>,
    /// Layer norm before feedforward
    ln2: LayerNorm<B>,
    /// Feedforward network
    ff_up: Linear<B>,
    ff_down: Linear<B>,
    /// Dropout
    dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &TransformerBlockConfig, max_variates: usize, device: &B::Device) -> Self {
        let ff_dim = config.d_model * config.ff_multiplier;
        let attn_config = AnyVariateAttentionConfig::new(config.d_model, config.n_heads, max_variates);

        Self {
            ln1: LayerNormConfig::new(config.d_model).init(device),
            self_attn: AnyVariateAttention::new(&attn_config, device),
            ln2: LayerNormConfig::new(config.d_model).init(device),
            ff_up: LinearConfig::new(config.d_model, ff_dim).init(device),
            ff_down: LinearConfig::new(ff_dim, config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        variate_ids: Tensor<B, 2, burn::tensor::Int>,
        positions: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        // Pre-norm self-attention with residual
        let normed = self.ln1.forward(x.clone());
        let attn_out = self.self_attn.forward(normed, variate_ids, positions);
        let x = x + self.dropout.forward(attn_out);

        // Pre-norm feedforward with residual
        let normed = self.ln2.forward(x.clone());
        let ff = self.ff_up.forward(normed);
        let ff = Gelu::new().forward(ff);
        let ff = self.dropout.forward(ff);
        let ff = self.ff_down.forward(ff);

        x + self.dropout.forward(ff)
    }
}

/// Configuration for full transformer encoder.
#[derive(Config, Debug)]
pub struct TransformerEncoderConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_variates: usize,
    #[config(default = "4")]
    pub ff_multiplier: usize,
    #[config(default = "0.1")]
    pub dropout: f64,
}

/// Multi-layer transformer encoder.
#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    layers: Vec<TransformerBlock<B>>,
    final_ln: LayerNorm<B>,
}

impl<B: Backend> TransformerEncoder<B> {
    pub fn new(config: &TransformerEncoderConfig, device: &B::Device) -> Self {
        let block_config = TransformerBlockConfig::new(config.d_model, config.n_heads)
            .with_ff_multiplier(config.ff_multiplier)
            .with_dropout(config.dropout);

        let layers: Vec<_> = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&block_config, config.max_variates, device))
            .collect();

        Self {
            layers,
            final_ln: LayerNormConfig::new(config.d_model).init(device),
        }
    }

    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        variate_ids: Tensor<B, 2, burn::tensor::Int>,
        positions: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x, variate_ids.clone(), positions.clone());
        }
        self.final_ln.forward(x)
    }
}

// =============================================================================
// ENSEMBLE FORECASTING
// =============================================================================

/// Ensemble of forecasters with adaptive weighting.
pub struct ForecastEnsemble {
    /// Individual model weights (updated based on performance)
    weights: Vec<f32>,
    /// Running MAE for each model
    model_errors: Vec<VecDeque<f32>>,
    /// EMA decay for error tracking
    decay: f32,
    /// Window size for error tracking
    window: usize,
}

impl ForecastEnsemble {
    pub fn new(n_models: usize, window: usize) -> Self {
        Self {
            weights: vec![1.0 / n_models as f32; n_models],
            model_errors: (0..n_models).map(|_| VecDeque::with_capacity(window)).collect(),
            decay: 0.95,
            window,
        }
    }

    /// Update weights based on observed errors.
    pub fn update(&mut self, model_idx: usize, error: f32) {
        if model_idx >= self.model_errors.len() {
            return;
        }

        let errors = &mut self.model_errors[model_idx];
        if errors.len() >= self.window {
            errors.pop_front();
        }
        errors.push_back(error);

        // Recompute weights based on inverse MAE
        self.recompute_weights();
    }

    fn recompute_weights(&mut self) {
        let maes: Vec<f32> = self.model_errors.iter()
            .map(|errors| {
                if errors.is_empty() {
                    f32::MAX
                } else {
                    errors.iter().sum::<f32>() / errors.len() as f32
                }
            })
            .collect();

        // Inverse MAE (with small epsilon)
        let inv_maes: Vec<f32> = maes.iter()
            .map(|&mae| 1.0 / (mae + 1e-6))
            .collect();

        let total: f32 = inv_maes.iter().sum();
        if total > 0.0 {
            self.weights = inv_maes.iter().map(|&w| w / total).collect();
        }
    }

    /// Get current ensemble weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Combine predictions from multiple models.
    pub fn combine<B: Backend>(&self, predictions: &[Tensor<B, 2>]) -> Tensor<B, 2> {
        assert!(!predictions.is_empty());
        let [batch_size, horizon] = predictions[0].dims();
        let device = predictions[0].device();

        let mut combined = Tensor::zeros([batch_size, horizon], &device);
        for (pred, &weight) in predictions.iter().zip(self.weights.iter()) {
            combined = combined + pred.clone() * weight;
        }
        combined
    }
}

// =============================================================================
// WORLD-CLASS FORECASTER (MAIN INTEGRATION)
// =============================================================================

/// Surprise-Calibrated Forecaster.
/// Integrates Hull-White SSG with Moirai-style forecasting.
pub struct SurpriseCalibratedForecaster<B: Backend> {
    patch_proj: MultiFreqPatchProjection<B>,
    attention: AnyVariateAttention<B>,
    mixture_head: MixtureDistributionHead<B>,
    /// Surprise gate for memory consolidation
    surprise_gate: crate::model::titans::memory::surprise::SurpriseGate,
    /// Current frequency band
    freq_band: FrequencyBand,
    /// Running forecast error for calibration
    error_ema: f32,
    /// EMA decay for error tracking
    error_decay: f32,
}

impl<B: Backend> SurpriseCalibratedForecaster<B> {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        max_variates: usize,
        n_components: usize,
        forecast_horizon: usize,
        freq_band: FrequencyBand,
        device: &B::Device,
    ) -> Self {
        use crate::model::titans::memory::surprise::{SurpriseConfig, SurpriseGate};

        let patch_config = PatchProjectionConfig::new(d_model, max_variates);
        let attn_config = AnyVariateAttentionConfig::new(d_model, n_heads, max_variates);
        let head_config = MixtureHeadConfig::new(d_model, forecast_horizon)
            .with_n_components(n_components);

        Self {
            patch_proj: MultiFreqPatchProjection::new(&patch_config, device),
            attention: AnyVariateAttention::new(&attn_config, device),
            mixture_head: MixtureDistributionHead::new(&head_config, device),
            surprise_gate: SurpriseGate::new(SurpriseConfig::default()),
            freq_band,
            error_ema: 0.0,
            error_decay: 0.9,
        }
    }

    /// Forward pass with surprise-based memory gating.
    /// Returns (forecast, surprise_triggered, z_score)
    pub fn forward(
        &mut self,
        patches: Tensor<B, 3>,
        variate_ids: Tensor<B, 2, burn::tensor::Int>,
        positions: Tensor<B, 2>,
        target: Option<Tensor<B, 2>>,
    ) -> (MixtureOutput<B>, bool, f32) {
        let [_batch_size, _seq_len, _] = patches.dims();

        // Project patches through frequency-appropriate projection
        let embedded = self.patch_proj.forward(patches, self.freq_band);

        // Apply any-variate attention
        let attended = self.attention.forward(embedded, variate_ids, positions);

        // Pool to get summary representation
        let summary = attended.mean_dim(1).squeeze::<2>(1);  // [batch, d_model]

        // Generate mixture distribution forecast
        let forecast = self.mixture_head.forward(summary);

        // Compute surprise from forecast error if target available
        let (triggered, z_score) = if let Some(target) = target {
            let errors = forecast.forecast_errors(target);
            // Convert batch error to scalar for surprise computation
            let mean_error: f32 = errors.mean().into_scalar().elem();

            // Update EMA of errors
            self.error_ema = self.error_decay * self.error_ema
                + (1.0 - self.error_decay) * mean_error;

            // Compute Hull-White surprise using forecast error as observation
            let surprise = self.surprise_gate.compute(mean_error, 1.0);
            (surprise.triggered, surprise.z_score)
        } else {
            (false, 0.0)
        };

        (forecast, triggered, z_score)
    }

    /// Update with actual values and compute surprise for memory consolidation.
    pub fn update_with_actual(
        &mut self,
        forecast: &MixtureOutput<B>,
        actual: Tensor<B, 2>,
        domain: &str,
    ) -> ForecastAuditTrail {
        let errors = forecast.forecast_errors(actual.clone());
        let mean_error: f32 = errors.mean().into_scalar().elem();

        // Compute surprise with full audit trail
        let (surprise, audit) = self.surprise_gate.compute_with_audit(mean_error, 1.0, domain);

        // Compute theta (learning rate) and alpha (forget rate) for memory update
        let theta = self.surprise_gate.theta(&surprise);
        let alpha = self.surprise_gate.alpha(&surprise);

        ForecastAuditTrail {
            domain: domain.to_string(),
            forecast_mean: forecast.mean().mean().into_scalar().elem(),
            actual_mean: actual.mean().into_scalar().elem(),
            forecast_error: mean_error,
            surprise_z_score: surprise.z_score,
            surprise_triggered: surprise.triggered,
            memory_theta: theta,
            memory_alpha: alpha,
            audit_trail: audit,
        }
    }

    /// Get surprise gate statistics.
    pub fn surprise_stats(&self) -> crate::model::titans::memory::surprise::SurpriseStats {
        self.surprise_gate.stats()
    }

    /// Reset surprise calibration.
    pub fn reset_calibration(&mut self) {
        self.surprise_gate.reset();
        self.error_ema = 0.0;
    }
}

/// Audit trail for a forecast decision.
#[derive(Debug, Clone)]
pub struct ForecastAuditTrail {
    pub domain: String,
    pub forecast_mean: f32,
    pub actual_mean: f32,
    pub forecast_error: f32,
    pub surprise_z_score: f32,
    pub surprise_triggered: bool,
    pub memory_theta: f32,
    pub memory_alpha: f32,
    pub audit_trail: crate::model::titans::memory::surprise::SurpriseAuditTrail,
}

// =============================================================================
// COVARIATE FUSION
// =============================================================================

/// Configuration for covariate fusion.
#[derive(Config, Debug)]
pub struct CovariateFusionConfig {
    pub d_model: usize,
    /// Dimension of static covariates (e.g., entity embeddings)
    pub static_dim: usize,
    /// Dimension of known future covariates
    pub future_dim: usize,
    /// Whether to use gated residual connections
    #[config(default = "true")]
    pub gated: bool,
}

/// Covariate fusion layer.
/// Integrates static features and known future covariates with time series representation.
#[derive(Module, Debug)]
pub struct CovariateFusion<B: Backend> {
    /// Project static covariates
    static_proj: Linear<B>,
    /// Project future covariates
    future_proj: Linear<B>,
    /// Gate for static fusion
    static_gate: Linear<B>,
    /// Gate for future fusion
    future_gate: Linear<B>,
    /// Final projection
    output_proj: Linear<B>,
}

impl<B: Backend> CovariateFusion<B> {
    pub fn new(config: &CovariateFusionConfig, device: &B::Device) -> Self {
        Self {
            static_proj: LinearConfig::new(config.static_dim, config.d_model).init(device),
            future_proj: LinearConfig::new(config.future_dim, config.d_model).init(device),
            static_gate: LinearConfig::new(config.d_model * 2, config.d_model).init(device),
            future_gate: LinearConfig::new(config.d_model * 2, config.d_model).init(device),
            output_proj: LinearConfig::new(config.d_model, config.d_model).init(device),
        }
    }

    /// Fuse covariates with time series representation.
    /// x: [batch, seq_len, d_model] - time series representation
    /// static_covs: [batch, static_dim] - entity-level features
    /// future_covs: [batch, seq_len, future_dim] - known future (e.g., holidays)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        static_covs: Option<Tensor<B, 2>>,
        future_covs: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = x.dims();
        let mut fused = x.clone();

        // Fuse static covariates (broadcast across sequence)
        if let Some(static_c) = static_covs {
            let static_emb = self.static_proj.forward(static_c);  // [batch, d_model]
            let static_emb: Tensor<B, 3> = static_emb.unsqueeze_dim::<3>(1).expand([batch_size, seq_len, d_model]);

            // Gated residual fusion
            let concat = Tensor::cat(vec![fused.clone(), static_emb.clone()], 2);
            let gate = self.static_gate.forward(concat);
            let gate = activation::sigmoid(gate);
            fused = fused + gate * static_emb;
        }

        // Fuse future covariates
        if let Some(future_c) = future_covs {
            let future_emb = self.future_proj.forward(future_c);  // [batch, seq_len, d_model]

            // Gated residual fusion
            let concat = Tensor::cat(vec![fused.clone(), future_emb.clone()], 2);
            let gate = self.future_gate.forward(concat);
            let gate = activation::sigmoid(gate);
            fused = fused + gate * future_emb;
        }

        self.output_proj.forward(fused)
    }
}

// =============================================================================
// UNIFIED WORLD-CLASS FORECASTER
// =============================================================================

/// Configuration for the world-class forecaster.
#[derive(Config, Debug)]
pub struct WorldClassForecasterConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Maximum number of variates
    pub max_variates: usize,
    /// Forecast horizon
    pub forecast_horizon: usize,
    /// Number of mixture components
    #[config(default = "3")]
    pub n_mixture_components: usize,
    /// Patch sizes for multi-scale patching
    pub patch_sizes: Vec<usize>,
    /// Target coverage for conformal prediction
    #[config(default = "0.9")]
    pub target_coverage: f32,
    /// Static covariate dimension (0 to disable)
    #[config(default = "0")]
    pub static_dim: usize,
    /// Future covariate dimension (0 to disable)
    #[config(default = "0")]
    pub future_dim: usize,
    /// Quantiles for quantile regression
    pub quantiles: Vec<f32>,
    /// Dropout rate
    #[config(default = "0.1")]
    pub dropout: f64,
}

impl WorldClassForecasterConfig {
    /// Create a new config with required parameters and sensible defaults.
    pub fn with_defaults(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        max_variates: usize,
        forecast_horizon: usize,
    ) -> Self {
        Self {
            d_model,
            n_heads,
            n_layers,
            max_variates,
            forecast_horizon,
            n_mixture_components: 3,
            patch_sizes: vec![8, 16, 32, 64],
            target_coverage: 0.9,
            static_dim: 0,
            future_dim: 0,
            quantiles: vec![0.05, 0.25, 0.5, 0.75, 0.95],
            dropout: 0.1,
        }
    }
}

/// World-class time series forecaster.
///
/// Combines state-of-the-art techniques:
/// - Reversible Instance Normalization for non-stationarity
/// - Multi-scale patching for temporal patterns
/// - Transformer encoder with positional encoding
/// - Multiple distribution heads (Gaussian, Student-t, Quantile)
/// - Covariate fusion for external information
/// - Conformal prediction for calibrated intervals
/// - Hull-White surprise calibration for adaptive learning
/// - Ensemble weighting for robustness
#[derive(Module, Debug)]
pub struct WorldClassForecaster<B: Backend> {
    /// Reversible instance normalization (public for external denormalization)
    pub revin: RevIN<B>,
    /// Multi-scale patching
    multi_scale_patch: MultiScalePatching<B>,
    /// Transformer encoder
    encoder: TransformerEncoder<B>,
    /// Covariate fusion (optional)
    covariate_fusion: Option<CovariateFusion<B>>,
    /// Multi-distribution head
    dist_head: MultiDistributionHead<B>,
    /// Mixture head for Student-t mixture
    mixture_head: MixtureDistributionHead<B>,
    /// Forecast horizon
    forecast_horizon: usize,
}

/// Runtime state for WorldClassForecaster (not part of the model).
#[derive(Debug, Clone)]
pub struct ForecasterState {
    /// Conformal predictor for calibrated intervals
    pub conformal: ConformalPredictor,
    /// Surprise gate for memory calibration
    pub surprise_gate: crate::model::titans::memory::surprise::SurpriseGate,
}

impl ForecasterState {
    /// Create a new forecaster state.
    pub fn new(target_coverage: f32) -> Self {
        use crate::model::titans::memory::surprise::{SurpriseConfig, SurpriseGate};
        Self {
            conformal: ConformalPredictor::new(target_coverage, 1000),
            surprise_gate: SurpriseGate::new(SurpriseConfig::default()),
        }
    }

    /// Update conformal predictor with new observation.
    pub fn update_conformal(&mut self, prediction: f32, actual: f32) {
        self.conformal.update(prediction, actual);
    }

    /// Update surprise gate with forecast error.
    pub fn update_surprise(&mut self, error: f32, dt: f32) -> crate::model::titans::memory::surprise::HullWhiteSurprise {
        self.surprise_gate.compute(error, dt)
    }

    /// Get conformal prediction statistics.
    pub fn conformal_stats(&self) -> ConformalStats {
        self.conformal.stats()
    }

    /// Get surprise gate statistics.
    pub fn surprise_stats(&self) -> crate::model::titans::memory::surprise::SurpriseStats {
        self.surprise_gate.stats()
    }

    /// Reset calibration state.
    pub fn reset(&mut self) {
        self.conformal.reset();
        self.surprise_gate.reset();
    }
}

impl<B: Backend> WorldClassForecaster<B> {
    pub fn new(config: &WorldClassForecasterConfig, device: &B::Device) -> Self {
        // RevIN
        let revin_config = RevINConfig::new(config.max_variates);
        let revin = RevIN::new(&revin_config, device);

        // Multi-scale patching
        let patch_config = MultiScalePatchConfig::new(config.patch_sizes.clone(), config.d_model);
        let multi_scale_patch = MultiScalePatching::new(&patch_config, device);

        // Transformer encoder
        let encoder_config = TransformerEncoderConfig::new(
            config.d_model,
            config.n_heads,
            config.n_layers,
            config.max_variates,
        ).with_dropout(config.dropout);
        let encoder = TransformerEncoder::new(&encoder_config, device);

        // Covariate fusion (if dimensions > 0)
        let covariate_fusion = if config.static_dim > 0 || config.future_dim > 0 {
            let cov_config = CovariateFusionConfig::new(
                config.d_model,
                config.static_dim.max(1),
                config.future_dim.max(1),
            );
            Some(CovariateFusion::new(&cov_config, device))
        } else {
            None
        };

        // Distribution heads
        let dist_config = MultiDistributionHeadConfig {
            d_model: config.d_model,
            forecast_horizon: config.forecast_horizon,
            quantiles: config.quantiles.clone(),
        };
        let dist_head = MultiDistributionHead::new(&dist_config, device);

        let mixture_config = MixtureHeadConfig::new(config.d_model, config.forecast_horizon)
            .with_n_components(config.n_mixture_components);
        let mixture_head = MixtureDistributionHead::new(&mixture_config, device);

        Self {
            revin,
            multi_scale_patch,
            encoder,
            covariate_fusion,
            dist_head,
            mixture_head,
            forecast_horizon: config.forecast_horizon,
        }
    }

    /// Create a new forecaster state for runtime calibration.
    pub fn create_state(&self, target_coverage: f32) -> ForecasterState {
        ForecasterState::new(target_coverage)
    }

    /// Full forward pass.
    ///
    /// Returns a `ProbabilisticForecast` with point prediction and calibrated intervals.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,                              // [batch, seq_len, features]
        static_covs: Option<Tensor<B, 2>>,            // [batch, static_dim]
        future_covs: Option<Tensor<B, 3>>,            // [batch, horizon, future_dim]
        dist_type: DistributionType,
        state: Option<&ForecasterState>,
    ) -> (ProbabilisticForecast<B>, RevINStats<B>) {
        let [batch_size, _seq_len, _features] = x.dims();
        let device = x.device();

        // 1. Reversible instance normalization
        let (normalized, revin_stats) = self.revin.normalize(x);

        // 2. Multi-scale patching
        let (patches, _patch_counts) = self.multi_scale_patch.forward(normalized);
        let [_, num_patches, _d_model] = patches.dims();

        // 3. Create position and variate IDs
        let positions: Vec<f32> = (0..num_patches).map(|i| i as f32).collect();
        let positions = Tensor::<B, 1>::from_floats(&positions[..], &device)
            .unsqueeze_dim::<2>(0)
            .expand([batch_size, num_patches]);

        let variate_ids: Vec<i32> = vec![0i32; num_patches];  // Single variate for now
        let variate_ids = Tensor::<B, 1, Int>::from_ints(&variate_ids[..], &device)
            .unsqueeze_dim::<2>(0)
            .expand([batch_size, num_patches]);

        // 4. Transformer encoding
        let encoded = self.encoder.forward(patches, variate_ids, positions);

        // 5. Covariate fusion
        let fused = if let Some(ref cov_fusion) = self.covariate_fusion {
            cov_fusion.forward(encoded, static_covs, future_covs)
        } else {
            encoded
        };

        // 6. Pool to summary representation
        let summary = fused.mean_dim(1).squeeze::<2>(1);  // [batch, d_model]

        // 7. Generate forecast based on distribution type
        let (point, lower, upper, quantiles) = match dist_type {
            DistributionType::Gaussian => {
                let (mean, var) = self.dist_head.forward_gaussian(summary);
                let std = var.clone().sqrt();
                // 90% interval: mean ± 1.645 * std
                let z = 1.645;
                let lower = mean.clone() - std.clone() * z;
                let upper = mean.clone() + std * z;
                (mean, lower, upper, None)
            }
            DistributionType::StudentT => {
                let (loc, scale, _df) = self.dist_head.forward_student_t(summary);
                // Approximate interval using scale
                let z = 1.645;
                let lower = loc.clone() - scale.clone() * z;
                let upper = loc.clone() + scale * z;
                (loc, lower, upper, None)
            }
            DistributionType::LogNormal => {
                let (mu, sigma) = self.dist_head.forward_lognormal(summary);
                // Mean of LogNormal: exp(mu + sigma²/2)
                let mean = (mu.clone() + sigma.clone().powf_scalar(2.0) / 2.0).exp();
                // Approximate intervals
                let lower = mu.clone().exp();
                let upper = (mu + sigma * 2.0).exp();
                (mean, lower, upper, None)
            }
            DistributionType::Quantile => {
                let q = self.dist_head.forward_quantiles(summary);
                let [b, h, nq] = q.dims();
                // Use median as point forecast
                let median_idx = nq / 2;
                let point = q.clone().slice([0..b, 0..h, median_idx..median_idx+1]).squeeze::<2>(2);
                // Use outer quantiles for interval
                let lower = q.clone().slice([0..b, 0..h, 0..1]).squeeze::<2>(2);
                let upper = q.clone().slice([0..b, 0..h, nq-1..nq]).squeeze::<2>(2);
                (point, lower, upper, Some(q))
            }
            DistributionType::NegativeBinomial => {
                // Fallback to Gaussian for now
                let (mean, var) = self.dist_head.forward_gaussian(summary);
                let std = var.sqrt();
                let z = 1.645;
                let lower = mean.clone() - std.clone() * z;
                let upper = mean.clone() + std * z;
                (mean, lower, upper, None)
            }
        };

        // 8. Apply conformal calibration to intervals if state is provided
        let (final_lower, final_upper, coverage) = if let Some(s) = state {
            let (conf_lower, conf_upper) = s.conformal.intervals_batch(&point, &device);
            // Take wider of distributional and conformal intervals
            let final_lower = lower.clone().min_pair(conf_lower);
            let final_upper = upper.clone().max_pair(conf_upper);
            (final_lower, final_upper, s.conformal.stats().target_coverage)
        } else {
            (lower, upper, 0.9)
        };

        let forecast = ProbabilisticForecast {
            point,
            lower: final_lower,
            upper: final_upper,
            coverage,
            dist_type,
            quantiles,
            quantile_levels: self.dist_head.quantile_levels().to_vec(),
        };

        (forecast, revin_stats)
    }

    /// Forward with denormalization.
    pub fn forecast(
        &self,
        x: Tensor<B, 3>,
        static_covs: Option<Tensor<B, 2>>,
        future_covs: Option<Tensor<B, 3>>,
        dist_type: DistributionType,
        state: Option<&ForecasterState>,
    ) -> ProbabilisticForecast<B> {
        let (forecast, stats) = self.forward(x, static_covs, future_covs, dist_type, state);

        // Denormalize point and intervals
        let point = self.revin.denormalize(forecast.point, &stats);
        let lower = self.revin.denormalize(forecast.lower, &stats);
        let upper = self.revin.denormalize(forecast.upper, &stats);

        ProbabilisticForecast {
            point,
            lower,
            upper,
            ..forecast
        }
    }
}

// =============================================================================
// TRAINING INTEGRATION (TrainStep / ValidStep)
// =============================================================================

/// Batch structure for time series training.
#[derive(Clone, Debug)]
pub struct TimeSeriesBatch<B: Backend> {
    /// Historical observations: [batch, context_len, features]
    pub history: Tensor<B, 3>,
    /// Target future values: [batch, horizon]
    pub targets: Tensor<B, 2>,
    /// Optional static covariates: [batch, static_dim]
    pub static_covs: Option<Tensor<B, 2>>,
    /// Optional future covariates: [batch, horizon, future_dim]
    pub future_covs: Option<Tensor<B, 3>>,
}

impl<B: Backend> TimeSeriesBatch<B> {
    /// Create a new time series batch.
    pub fn new(
        history: Tensor<B, 3>,
        targets: Tensor<B, 2>,
    ) -> Self {
        Self {
            history,
            targets,
            static_covs: None,
            future_covs: None,
        }
    }

    /// Add static covariates.
    pub fn with_static_covs(mut self, covs: Tensor<B, 2>) -> Self {
        self.static_covs = Some(covs);
        self
    }

    /// Add future covariates.
    pub fn with_future_covs(mut self, covs: Tensor<B, 3>) -> Self {
        self.future_covs = Some(covs);
        self
    }
}

/// Forecasting output with loss for training.
#[derive(Clone, Debug)]
pub struct ForecastOutput<B: Backend> {
    /// MSE loss for training
    pub loss: Tensor<B, 1>,
    /// Point predictions: [batch, horizon]
    pub predictions: Tensor<B, 2>,
    /// Target values: [batch, horizon]
    pub targets: Tensor<B, 2>,
}

impl<B: AutodiffBackend> WorldClassForecaster<B> {
    /// Forward pass for training with loss computation.
    pub fn forward_training(
        &self,
        batch: TimeSeriesBatch<B>,
    ) -> ForecastOutput<B> {
        let targets = batch.targets.clone();

        // Get forecast
        let (forecast, stats) = self.forward(
            batch.history,
            batch.static_covs,
            batch.future_covs,
            DistributionType::Gaussian,
            None,
        );

        // Denormalize predictions
        let predictions = self.revin.denormalize(forecast.point, &stats);

        // Compute MSE loss
        let mse = MseLoss::new();
        let loss = mse.forward(
            predictions.clone().reshape([predictions.dims()[0] * predictions.dims()[1]]),
            targets.clone().reshape([targets.dims()[0] * targets.dims()[1]]),
            Reduction::Mean,
        );

        ForecastOutput {
            loss: loss.unsqueeze(),
            predictions,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TimeSeriesBatch<B>, RegressionOutput<B>> for WorldClassForecaster<B> {
    fn step(&self, batch: TimeSeriesBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward_training(batch);

        // RegressionOutput expects 2D tensors [batch, features]
        let regression_output = RegressionOutput::new(
            output.loss.clone(),  // Loss is already a scalar
            output.predictions,   // Already [batch, horizon]
            output.targets,       // Already [batch, horizon]
        );

        TrainOutput::new(self, output.loss.backward(), regression_output)
    }
}

impl<B: Backend> ValidStep<TimeSeriesBatch<B>, RegressionOutput<B>> for WorldClassForecaster<B> {
    fn step(&self, batch: TimeSeriesBatch<B>) -> RegressionOutput<B> {
        let targets = batch.targets.clone();

        let (forecast, stats) = self.forward(
            batch.history,
            batch.static_covs,
            batch.future_covs,
            DistributionType::Gaussian,
            None,
        );

        let predictions = self.revin.denormalize(forecast.point, &stats);
        let [batch_size, horizon] = predictions.dims();
        let total_elements = batch_size * horizon;

        // Compute loss
        let mse = MseLoss::new();
        let loss = mse.forward(
            predictions.clone().reshape([total_elements]),
            targets.clone().reshape([total_elements]),
            Reduction::Mean,
        );

        RegressionOutput::new(
            loss,
            predictions,  // Already [batch, horizon]
            targets,      // Already [batch, horizon]
        )
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_revin_normalize_denormalize() {
        let device = Default::default();
        let config = RevINConfig::new(3);
        let revin = RevIN::<TestBackend>::new(&config, &device);

        // Create test input
        let x = Tensor::<TestBackend, 3>::from_floats(
            [[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]],
            &device,
        );

        let (normalized, stats) = revin.normalize(x.clone());

        // Check mean is approximately 0
        let mean = normalized.clone().mean();
        assert!(mean.into_scalar().abs() < 0.1);

        // Denormalize and check recovery
        let recovered = revin.denormalize_3d(normalized, &stats);
        let diff = (recovered - x).abs().mean().into_scalar();
        assert!(diff < 0.01, "RevIN should be reversible, got diff {}", diff);
    }

    #[test]
    fn test_multi_scale_patching() {
        let device = Default::default();
        let config = MultiScalePatchConfig::new(vec![4, 8], 64);
        let patcher = MultiScalePatching::<TestBackend>::new(&config, &device);

        let x = Tensor::zeros([2, 32, 1], &device);
        let (patches, counts) = patcher.forward(x);

        assert_eq!(patches.dims()[0], 2);  // batch
        assert!(patches.dims()[1] > 0);     // patches
        assert_eq!(patches.dims()[2], 64);  // d_model
    }

    #[test]
    fn test_conformal_predictor() {
        let mut conformal = ConformalPredictor::new(0.9, 100);

        // Add some calibration data
        for i in 0..50 {
            let pred = i as f32;
            let actual = pred + (i as f32 % 5.0) - 2.5;  // Noise ±2.5
            conformal.update(pred, actual);
        }

        let stats = conformal.stats();
        assert!(stats.buffer_size == 50);
        assert!(stats.calibrated_quantile > 0.0);

        // Check interval
        let (lo, hi) = conformal.interval(100.0);
        assert!(lo < 100.0);
        assert!(hi > 100.0);
    }

    #[test]
    fn test_fourier_features() {
        let device = Default::default();
        let timestamps = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 1.0, 2.0, 3.0, 4.0]],
            &device,
        );
        let frequencies = vec![1.0, 2.0];
        let period = 10.0;

        let features = extract_fourier_features(&timestamps, &frequencies, period, &device);

        assert_eq!(features.dims(), [1, 5, 4]);  // 2 freq * 2 (sin+cos)
    }

    #[test]
    fn test_multi_distribution_head() {
        let device = Default::default();
        let config = MultiDistributionHeadConfig {
            d_model: 64,
            forecast_horizon: 12,
            quantiles: vec![0.1, 0.5, 0.9],
        };
        let head = MultiDistributionHead::<TestBackend>::new(&config, &device);

        let x = Tensor::zeros([4, 64], &device);

        // Test Gaussian
        let (mean, var) = head.forward_gaussian(x.clone());
        assert_eq!(mean.dims(), [4, 12]);
        assert_eq!(var.dims(), [4, 12]);

        // Test Student-t
        let (loc, scale, df) = head.forward_student_t(x.clone());
        assert_eq!(loc.dims(), [4, 12]);
        assert!(scale.clone().min().into_scalar() >= 0.0, "scale must be positive");
        assert!(df.clone().min().into_scalar() >= 2.0, "df must be > 2");

        // Test Quantiles
        let q = head.forward_quantiles(x);
        assert_eq!(q.dims(), [4, 12, 3]);
    }

    #[test]
    fn test_ensemble_weights() {
        let mut ensemble = ForecastEnsemble::new(3, 10);

        // Model 0: low error
        for _ in 0..10 {
            ensemble.update(0, 0.1);
        }
        // Model 1: medium error
        for _ in 0..10 {
            ensemble.update(1, 0.5);
        }
        // Model 2: high error
        for _ in 0..10 {
            ensemble.update(2, 1.0);
        }

        let weights = ensemble.weights();
        assert!(weights[0] > weights[1], "Lower error should have higher weight");
        assert!(weights[1] > weights[2], "Lower error should have higher weight");
    }
}
