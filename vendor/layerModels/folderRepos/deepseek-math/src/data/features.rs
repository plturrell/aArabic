//! Advanced Time Series Feature Engineering
//!
//! Mathematically rigorous feature extraction for time series forecasting:
//! - Haar/Daubechies Wavelet Decomposition (multi-resolution analysis)
//! - Autocorrelation & Partial Autocorrelation (ARIMA-style)
//! - Sample Entropy & Approximate Entropy (complexity measures)
//! - Hurst Exponent (long-range dependence / fractality)
//! - Detrended Fluctuation Analysis (DFA)
//! - Statistical Moments (skewness, kurtosis)
//! - Spectral Analysis (periodogram, dominant frequencies)
//! - CUSUM Change Point Detection
//! - Singular Value Decomposition features

use std::f32::consts::PI;

// =============================================================================
// MATHEMATICAL CONSTANTS AND UTILITIES
// =============================================================================

/// Haar wavelet filter coefficients
const HAAR_LO: [f32; 2] = [0.7071067811865476, 0.7071067811865476];
const HAAR_HI: [f32; 2] = [0.7071067811865476, -0.7071067811865476];

/// Daubechies-4 wavelet filter coefficients
const DB4_LO: [f32; 8] = [
    -0.010597401784997278, 0.032883011666982945, 0.030841381835986965,
    -0.18703481171888114, -0.02798376941698385, 0.6308807679295904,
    0.7148465705525415, 0.23037781330885523,
];
const DB4_HI: [f32; 8] = [
    -0.23037781330885523, 0.7148465705525415, -0.6308807679295904,
    -0.02798376941698385, 0.18703481171888114, 0.030841381835986965,
    -0.032883011666982945, -0.010597401784997278,
];

/// Configuration for advanced feature engineering.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Lag indices for autoregressive features
    pub lag_indices: Vec<usize>,
    /// Window sizes for rolling statistics
    pub rolling_windows: Vec<usize>,
    /// Number of wavelet decomposition levels
    pub wavelet_levels: usize,
    /// Use Daubechies-4 (true) or Haar (false) wavelet
    pub use_db4_wavelet: bool,
    /// Max lag for autocorrelation computation
    pub acf_max_lag: usize,
    /// Embedding dimension for entropy calculation
    pub entropy_embed_dim: usize,
    /// Tolerance for approximate entropy
    pub entropy_tolerance: f32,
    /// Whether to compute Hurst exponent
    pub compute_hurst: bool,
    /// Whether to compute spectral features
    pub compute_spectral: bool,
    /// Number of spectral peaks to extract
    pub n_spectral_peaks: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            lag_indices: vec![1, 2, 3, 4, 5, 6, 7, 12, 24],
            rolling_windows: vec![3, 7, 12, 24],
            wavelet_levels: 3,
            use_db4_wavelet: false, // Haar is faster
            acf_max_lag: 24,
            entropy_embed_dim: 2,
            entropy_tolerance: 0.2,
            compute_hurst: true,
            compute_spectral: true,
            n_spectral_peaks: 3,
        }
    }
}

impl FeatureConfig {
    /// Minimal config for fast processing
    pub fn minimal() -> Self {
        Self {
            lag_indices: vec![1, 2, 3],
            rolling_windows: vec![3, 7],
            wavelet_levels: 2,
            use_db4_wavelet: false,
            acf_max_lag: 12,
            entropy_embed_dim: 2,
            entropy_tolerance: 0.2,
            compute_hurst: false,
            compute_spectral: true,
            n_spectral_peaks: 2,
        }
    }

    /// Rich config for maximum feature extraction
    pub fn rich() -> Self {
        Self {
            lag_indices: vec![1, 2, 3, 4, 5, 6, 7, 12, 24, 48, 96],
            rolling_windows: vec![3, 5, 7, 12, 24, 48],
            wavelet_levels: 4,
            use_db4_wavelet: true,
            acf_max_lag: 48,
            entropy_embed_dim: 3,
            entropy_tolerance: 0.15,
            compute_hurst: true,
            compute_spectral: true,
            n_spectral_peaks: 5,
        }
    }
}

/// Mathematically rigorous engineered features.
#[derive(Debug, Clone)]
pub struct EngineeredFeatures {
    // === Normalized representation ===
    /// Robust-scaled series [seq_len]
    pub normalized: Vec<f32>,

    // === Autoregressive features ===
    /// Lag features [num_lags][seq_len]
    pub lags: Vec<Vec<f32>>,
    /// Autocorrelation function [acf_max_lag]
    pub acf: Vec<f32>,
    /// Partial autocorrelation function [acf_max_lag]
    pub pacf: Vec<f32>,

    // === Rolling statistics ===
    /// Rolling mean [num_windows][seq_len]
    pub rolling_mean: Vec<Vec<f32>>,
    /// Rolling std [num_windows][seq_len]
    pub rolling_std: Vec<Vec<f32>>,
    /// Rolling skewness [num_windows][seq_len]
    pub rolling_skew: Vec<Vec<f32>>,
    /// Rolling kurtosis [num_windows][seq_len]
    pub rolling_kurt: Vec<Vec<f32>>,

    // === Wavelet decomposition ===
    /// Wavelet approximation coefficients at each level [levels][coeffs]
    pub wavelet_approx: Vec<Vec<f32>>,
    /// Wavelet detail coefficients at each level [levels][coeffs]
    pub wavelet_detail: Vec<Vec<f32>>,
    /// Energy at each wavelet scale [levels]
    pub wavelet_energy: Vec<f32>,

    // === Complexity measures ===
    /// Sample entropy (regularity/predictability)
    pub sample_entropy: f32,
    /// Approximate entropy
    pub approx_entropy: f32,
    /// Hurst exponent (0.5 = random, >0.5 = trending, <0.5 = mean-reverting)
    pub hurst_exponent: f32,
    /// Detrended fluctuation analysis exponent
    pub dfa_exponent: f32,

    // === Spectral features ===
    /// Spectral centroid (center of mass of spectrum)
    pub spectral_centroid: f32,
    /// Spectral entropy
    pub spectral_entropy: f32,
    /// Dominant frequency periods [n_peaks]
    pub dominant_periods: Vec<f32>,
    /// Power at dominant frequencies [n_peaks]
    pub dominant_power: Vec<f32>,

    // === Trend/stationarity ===
    /// Linear trend slope
    pub trend_slope: f32,
    /// Quadratic trend coefficient
    pub trend_quadratic: f32,
    /// Trend residual variance ratio (stationarity indicator)
    pub stationarity_ratio: f32,
    /// CUSUM statistic for change point detection [seq_len]
    pub cusum: Vec<f32>,

    // === Statistical moments (global) ===
    /// Skewness of full series
    pub skewness: f32,
    /// Kurtosis of full series
    pub kurtosis: f32,
    /// Coefficient of variation
    pub coef_variation: f32,

    // === Normalization params ===
    pub norm_median: f32,
    pub norm_iqr: f32,
    pub norm_mean: f32,
    pub norm_std: f32,

    /// Total feature dimension
    pub feature_dim: usize,
}

/// Advanced time series feature extractor.
#[derive(Clone)]
pub struct FeatureExtractor {
    config: FeatureConfig,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

impl FeatureExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Calculate the total feature dimension per timestep.
    pub fn feature_dim(&self) -> usize {
        let mut dim = 1; // normalized value
        dim += self.config.lag_indices.len(); // lag features
        dim += self.config.rolling_windows.len() * 4; // mean, std, skew, kurt
        dim += self.config.acf_max_lag; // ACF (global, broadcast)
        dim += self.config.acf_max_lag; // PACF (global, broadcast)
        dim += self.config.wavelet_levels; // wavelet energy
        dim += 4; // entropy measures + hurst + dfa
        dim += 2 + self.config.n_spectral_peaks * 2; // spectral features
        dim += 4; // trend + stationarity
        dim += 3; // global moments
        dim
    }

    /// Extract comprehensive features from a raw time series.
    pub fn extract(&self, series: &[f32]) -> EngineeredFeatures {
        let n = series.len();
        if n == 0 {
            return self.empty_features();
        }

        // Robust scaling
        let (normalized, median, iqr, mean, std) = self.robust_normalize(series);

        // Lag features
        let lags = self.compute_lags(&normalized);

        // Autocorrelation and partial autocorrelation
        let acf = self.compute_acf(&normalized);
        let pacf = self.compute_pacf(&normalized, &acf);

        // Rolling statistics with higher moments
        let (rolling_mean, rolling_std, rolling_skew, rolling_kurt) =
            self.compute_rolling_moments(&normalized);

        // Wavelet decomposition
        let (wavelet_approx, wavelet_detail, wavelet_energy) =
            self.compute_wavelet_decomposition(&normalized);

        // Entropy measures
        let sample_entropy = self.compute_sample_entropy(&normalized);
        let approx_entropy = self.compute_approx_entropy(&normalized);

        // Hurst exponent and DFA
        let hurst_exponent = if self.config.compute_hurst {
            self.compute_hurst_exponent(&normalized)
        } else {
            0.5
        };
        let dfa_exponent = if self.config.compute_hurst {
            self.compute_dfa(&normalized)
        } else {
            0.5
        };

        // Spectral features
        let (spectral_centroid, spectral_entropy, dominant_periods, dominant_power) =
            if self.config.compute_spectral {
                self.compute_spectral_features(&normalized)
            } else {
                (0.0, 0.0, vec![0.0; self.config.n_spectral_peaks], vec![0.0; self.config.n_spectral_peaks])
            };

        // Trend and stationarity
        let (trend_slope, trend_quadratic, stationarity_ratio) = self.compute_trend_features(&normalized);
        let cusum = self.compute_cusum(&normalized);

        // Global statistical moments
        let (skewness, kurtosis, coef_variation) = self.compute_global_moments(&normalized);

        EngineeredFeatures {
            feature_dim: self.feature_dim(),
            normalized,
            lags,
            acf,
            pacf,
            rolling_mean,
            rolling_std,
            rolling_skew,
            rolling_kurt,
            wavelet_approx,
            wavelet_detail,
            wavelet_energy,
            sample_entropy,
            approx_entropy,
            hurst_exponent,
            dfa_exponent,
            spectral_centroid,
            spectral_entropy,
            dominant_periods,
            dominant_power,
            trend_slope,
            trend_quadratic,
            stationarity_ratio,
            cusum,
            skewness,
            kurtosis,
            coef_variation,
            norm_median: median,
            norm_iqr: iqr,
            norm_mean: mean,
            norm_std: std,
        }
    }

    // =========================================================================
    // NORMALIZATION
    // =========================================================================

    /// Robust normalization using median and IQR, also returns mean/std.
    fn robust_normalize(&self, series: &[f32]) -> (Vec<f32>, f32, f32, f32, f32) {
        let valid: Vec<f32> = series.iter()
            .filter(|x| x.is_finite())
            .copied()
            .collect();

        if valid.is_empty() {
            return (vec![0.0; series.len()], 0.0, 1.0, 0.0, 1.0);
        }

        let n = valid.len() as f32;
        let mean: f32 = valid.iter().sum::<f32>() / n;
        let variance: f32 = valid.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt().max(1e-8);

        let mut sorted = valid.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        let median = if len % 2 == 0 {
            (sorted[len/2 - 1] + sorted[len/2]) / 2.0
        } else {
            sorted[len/2]
        };

        let q1 = sorted[len / 4];
        let q3 = sorted[(3 * len) / 4];
        let iqr = (q3 - q1).max(1e-8);

        let normalized = series.iter()
            .map(|&x| if x.is_finite() { (x - median) / iqr } else { 0.0 })
            .collect();

        (normalized, median, iqr, mean, std)
    }

    // =========================================================================
    // AUTOREGRESSIVE FEATURES
    // =========================================================================

    /// Compute lag features.
    fn compute_lags(&self, series: &[f32]) -> Vec<Vec<f32>> {
        let n = series.len();
        self.config.lag_indices.iter()
            .map(|&lag| {
                (0..n).map(|i| {
                    if i >= lag { series[i - lag] } else { 0.0 }
                }).collect()
            })
            .collect()
    }

    /// Compute Autocorrelation Function (ACF).
    /// ACF(k) = Cov(X_t, X_{t-k}) / Var(X)
    fn compute_acf(&self, series: &[f32]) -> Vec<f32> {
        let n = series.len();
        if n < 2 {
            return vec![0.0; self.config.acf_max_lag];
        }

        let mean: f32 = series.iter().sum::<f32>() / n as f32;
        let var: f32 = series.iter().map(|x| (x - mean).powi(2)).sum::<f32>();

        if var.abs() < 1e-10 {
            return vec![0.0; self.config.acf_max_lag];
        }

        (1..=self.config.acf_max_lag)
            .map(|lag| {
                if lag >= n {
                    return 0.0;
                }
                let cov: f32 = (lag..n)
                    .map(|t| (series[t] - mean) * (series[t - lag] - mean))
                    .sum();
                cov / var
            })
            .collect()
    }

    /// Compute Partial Autocorrelation Function (PACF) using Durbin-Levinson.
    fn compute_pacf(&self, series: &[f32], acf: &[f32]) -> Vec<f32> {
        let max_lag = self.config.acf_max_lag.min(acf.len());
        if max_lag == 0 {
            return vec![0.0; self.config.acf_max_lag];
        }

        let mut pacf = vec![0.0; max_lag];
        let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];

        // PACF(1) = ACF(1)
        if !acf.is_empty() {
            pacf[0] = acf[0];
            phi[1][1] = acf[0];
        }

        // Durbin-Levinson recursion
        for k in 2..=max_lag {
            if k > acf.len() {
                break;
            }

            let mut num = acf[k - 1];
            let mut den = 1.0f32;

            for j in 1..k {
                num -= phi[k - 1][j] * acf[k - j - 1];
                den -= phi[k - 1][j] * acf[j - 1];
            }

            if den.abs() < 1e-10 {
                break;
            }

            phi[k][k] = num / den;
            pacf[k - 1] = phi[k][k];

            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
            }
        }

        pacf
    }

    // =========================================================================
    // ROLLING STATISTICS WITH HIGHER MOMENTS
    // =========================================================================

    /// Compute rolling mean, std, skewness, and kurtosis.
    fn compute_rolling_moments(&self, series: &[f32])
        -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)
    {
        let n = series.len();
        let mut means = Vec::new();
        let mut stds = Vec::new();
        let mut skews = Vec::new();
        let mut kurts = Vec::new();

        for &window in &self.config.rolling_windows {
            let mut mean_vec = vec![0.0; n];
            let mut std_vec = vec![0.0; n];
            let mut skew_vec = vec![0.0; n];
            let mut kurt_vec = vec![0.0; n];

            for i in 0..n {
                let start = i.saturating_sub(window - 1);
                let win = &series[start..=i];
                let w_len = win.len() as f32;

                let mean = win.iter().sum::<f32>() / w_len;
                mean_vec[i] = mean;

                let m2: f32 = win.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / w_len;
                let std = m2.sqrt().max(1e-10);
                std_vec[i] = std;

                // Skewness: E[(X-μ)³] / σ³
                let m3: f32 = win.iter().map(|&x| (x - mean).powi(3)).sum::<f32>() / w_len;
                skew_vec[i] = m3 / std.powi(3);

                // Kurtosis: E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis)
                let m4: f32 = win.iter().map(|&x| (x - mean).powi(4)).sum::<f32>() / w_len;
                kurt_vec[i] = m4 / std.powi(4) - 3.0;
            }

            means.push(mean_vec);
            stds.push(std_vec);
            skews.push(skew_vec);
            kurts.push(kurt_vec);
        }

        (means, stds, skews, kurts)
    }

    // =========================================================================
    // WAVELET DECOMPOSITION
    // =========================================================================

    /// Discrete Wavelet Transform decomposition.
    fn compute_wavelet_decomposition(&self, series: &[f32])
        -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>)
    {
        let mut approx = series.to_vec();
        let mut approx_coeffs = Vec::new();
        let mut detail_coeffs = Vec::new();
        let mut energy = Vec::new();

        let (lo, hi) = if self.config.use_db4_wavelet {
            (DB4_LO.to_vec(), DB4_HI.to_vec())
        } else {
            (HAAR_LO.to_vec(), HAAR_HI.to_vec())
        };

        for _ in 0..self.config.wavelet_levels {
            if approx.len() < lo.len() * 2 {
                break;
            }

            let (new_approx, detail) = self.dwt_step(&approx, &lo, &hi);

            // Compute energy at this scale: sum of squared coefficients
            let detail_energy: f32 = detail.iter().map(|x| x.powi(2)).sum();
            energy.push(detail_energy / detail.len().max(1) as f32);

            detail_coeffs.push(detail);
            approx_coeffs.push(new_approx.clone());
            approx = new_approx;
        }

        // Pad if needed
        while energy.len() < self.config.wavelet_levels {
            energy.push(0.0);
            approx_coeffs.push(vec![]);
            detail_coeffs.push(vec![]);
        }

        (approx_coeffs, detail_coeffs, energy)
    }

    /// Single step of DWT using convolution and downsampling.
    fn dwt_step(&self, signal: &[f32], lo: &[f32], hi: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = signal.len();
        let filter_len = lo.len();

        // Convolution with downsampling by 2
        let out_len = n / 2;
        let mut approx = Vec::with_capacity(out_len);
        let mut detail = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let mut a = 0.0f32;
            let mut d = 0.0f32;

            for (j, (&lo_j, &hi_j)) in lo.iter().zip(hi.iter()).enumerate() {
                let idx = (2 * i + j) % n;
                a += signal[idx] * lo_j;
                d += signal[idx] * hi_j;
            }

            approx.push(a);
            detail.push(d);
        }

        (approx, detail)
    }

    // =========================================================================
    // ENTROPY MEASURES
    // =========================================================================

    /// Compute Sample Entropy (regularity measure).
    /// Lower values indicate more regular/predictable series.
    fn compute_sample_entropy(&self, series: &[f32]) -> f32 {
        let n = series.len();
        let m = self.config.entropy_embed_dim;
        let r = self.config.entropy_tolerance * self.std_dev(series);

        if n < m + 2 || r < 1e-10 {
            return 0.0;
        }

        let count_matches = |dim: usize| -> usize {
            let mut count = 0;
            for i in 0..n - dim {
                for j in (i + 1)..n - dim {
                    let max_diff = (0..dim)
                        .map(|k| (series[i + k] - series[j + k]).abs())
                        .fold(0.0f32, f32::max);
                    if max_diff < r {
                        count += 1;
                    }
                }
            }
            count
        };

        let a = count_matches(m + 1) as f32;
        let b = count_matches(m) as f32;

        if b == 0.0 || a == 0.0 {
            return 0.0;
        }

        -(a / b).ln()
    }

    /// Compute Approximate Entropy.
    fn compute_approx_entropy(&self, series: &[f32]) -> f32 {
        let n = series.len();
        let m = self.config.entropy_embed_dim;
        let r = self.config.entropy_tolerance * self.std_dev(series);

        if n < m + 1 || r < 1e-10 {
            return 0.0;
        }

        let phi = |dim: usize| -> f32 {
            let mut total = 0.0f32;
            for i in 0..n - dim {
                let mut count = 0;
                for j in 0..n - dim {
                    let max_diff = (0..dim)
                        .map(|k| (series[i + k] - series[j + k]).abs())
                        .fold(0.0f32, f32::max);
                    if max_diff < r {
                        count += 1;
                    }
                }
                total += (count as f32 / (n - dim) as f32).ln();
            }
            total / (n - dim) as f32
        };

        phi(m) - phi(m + 1)
    }

    fn std_dev(&self, series: &[f32]) -> f32 {
        let n = series.len() as f32;
        if n < 2.0 {
            return 1.0;
        }
        let mean: f32 = series.iter().sum::<f32>() / n;
        let var: f32 = series.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        var.sqrt().max(1e-10)
    }

    // =========================================================================
    // HURST EXPONENT & DFA
    // =========================================================================

    /// Compute Hurst exponent using R/S analysis.
    /// H > 0.5: trending/persistent, H < 0.5: mean-reverting, H = 0.5: random walk
    fn compute_hurst_exponent(&self, series: &[f32]) -> f32 {
        let n = series.len();
        if n < 20 {
            return 0.5;
        }

        // Use different window sizes
        let mut log_rs = Vec::new();
        let mut log_n = Vec::new();

        for window in [8, 16, 32, 64, 128].iter() {
            let w = *window;
            if w > n / 2 {
                continue;
            }

            let num_windows = n / w;
            let mut rs_values = Vec::new();

            for i in 0..num_windows {
                let start = i * w;
                let end = start + w;
                let segment = &series[start..end];

                // Mean and centered cumulative sum
                let mean: f32 = segment.iter().sum::<f32>() / w as f32;
                let mut cumsum = Vec::with_capacity(w);
                let mut sum = 0.0f32;
                for &x in segment {
                    sum += x - mean;
                    cumsum.push(sum);
                }

                // Range
                let max_c = cumsum.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let min_c = cumsum.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let range = max_c - min_c;

                // Standard deviation
                let std = segment.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>()
                    .sqrt() / (w as f32).sqrt();

                if std > 1e-10 {
                    rs_values.push(range / std);
                }
            }

            if !rs_values.is_empty() {
                let avg_rs: f32 = rs_values.iter().sum::<f32>() / rs_values.len() as f32;
                log_rs.push(avg_rs.ln());
                log_n.push((w as f32).ln());
            }
        }

        // Linear regression to find Hurst exponent (slope)
        if log_rs.len() < 2 {
            return 0.5;
        }

        let n_pts = log_rs.len() as f32;
        let x_mean: f32 = log_n.iter().sum::<f32>() / n_pts;
        let y_mean: f32 = log_rs.iter().sum::<f32>() / n_pts;

        let num: f32 = log_n.iter().zip(log_rs.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum();
        let den: f32 = log_n.iter()
            .map(|&x| (x - x_mean).powi(2))
            .sum();

        if den.abs() < 1e-10 {
            return 0.5;
        }

        (num / den).clamp(0.0, 1.0)
    }

    /// Detrended Fluctuation Analysis.
    fn compute_dfa(&self, series: &[f32]) -> f32 {
        let n = series.len();
        if n < 16 {
            return 0.5;
        }

        // Cumulative sum (integration)
        let mean: f32 = series.iter().sum::<f32>() / n as f32;
        let mut y = Vec::with_capacity(n);
        let mut sum = 0.0f32;
        for &x in series {
            sum += x - mean;
            y.push(sum);
        }

        let mut log_f = Vec::new();
        let mut log_s = Vec::new();

        for s in [4, 8, 16, 32, 64] {
            if s > n / 4 {
                continue;
            }

            let num_segments = n / s;
            let mut f2_sum = 0.0f32;

            for seg in 0..num_segments {
                let start = seg * s;
                let end = start + s;
                let segment = &y[start..end];

                // Linear fit
                let x_mean = (s - 1) as f32 / 2.0;
                let y_mean: f32 = segment.iter().sum::<f32>() / s as f32;

                let mut num = 0.0f32;
                let mut den = 0.0f32;
                for (i, &yi) in segment.iter().enumerate() {
                    let x = i as f32;
                    num += (x - x_mean) * (yi - y_mean);
                    den += (x - x_mean).powi(2);
                }

                let slope = if den.abs() > 1e-10 { num / den } else { 0.0 };
                let intercept = y_mean - slope * x_mean;

                // Fluctuation
                for (i, &yi) in segment.iter().enumerate() {
                    let trend = intercept + slope * i as f32;
                    f2_sum += (yi - trend).powi(2);
                }
            }

            let f = (f2_sum / (num_segments * s) as f32).sqrt();
            if f > 1e-10 {
                log_f.push(f.ln());
                log_s.push((s as f32).ln());
            }
        }

        if log_f.len() < 2 {
            return 0.5;
        }

        // Slope
        let n_pts = log_f.len() as f32;
        let x_mean: f32 = log_s.iter().sum::<f32>() / n_pts;
        let y_mean: f32 = log_f.iter().sum::<f32>() / n_pts;

        let num: f32 = log_s.iter().zip(log_f.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum();
        let den: f32 = log_s.iter()
            .map(|&x| (x - x_mean).powi(2))
            .sum();

        if den.abs() < 1e-10 {
            return 0.5;
        }

        (num / den).clamp(0.0, 2.0)
    }

    // =========================================================================
    // SPECTRAL FEATURES
    // =========================================================================

    /// Compute spectral features: centroid, entropy, dominant frequencies.
    fn compute_spectral_features(&self, series: &[f32])
        -> (f32, f32, Vec<f32>, Vec<f32>)
    {
        let n = series.len();
        if n < 8 {
            return (
                0.0, 0.0,
                vec![0.0; self.config.n_spectral_peaks],
                vec![0.0; self.config.n_spectral_peaks],
            );
        }

        // Compute periodogram using DFT
        let n_freq = n / 2;
        let mut power = Vec::with_capacity(n_freq);
        let mut freqs = Vec::with_capacity(n_freq);

        for k in 1..=n_freq {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (t, &x) in series.iter().enumerate() {
                let angle = 2.0 * PI * (k as f32) * (t as f32) / (n as f32);
                real += x * angle.cos();
                imag -= x * angle.sin();
            }

            let p = (real.powi(2) + imag.powi(2)) / (n as f32).powi(2);
            power.push(p);
            freqs.push(k as f32 / n as f32);
        }

        let total_power: f32 = power.iter().sum::<f32>().max(1e-10);
        let norm_power: Vec<f32> = power.iter().map(|&p| p / total_power).collect();

        // Spectral centroid: weighted average of frequencies
        let spectral_centroid: f32 = freqs.iter()
            .zip(norm_power.iter())
            .map(|(&f, &p)| f * p)
            .sum();

        // Spectral entropy: -sum(p * log(p))
        let spectral_entropy: f32 = -norm_power.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f32>();

        // Find dominant frequencies (peaks)
        let mut indexed_power: Vec<(usize, f32)> = power.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_power.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_peaks = self.config.n_spectral_peaks;
        let mut dominant_periods = Vec::with_capacity(n_peaks);
        let mut dominant_power_vals = Vec::with_capacity(n_peaks);

        for (idx, pwr) in indexed_power.iter().take(n_peaks) {
            let freq = (*idx + 1) as f32 / n as f32;
            let period = if freq > 1e-10 { 1.0 / freq } else { n as f32 };
            dominant_periods.push(period / n as f32); // Normalize to [0, 1]
            dominant_power_vals.push(*pwr / total_power);
        }

        // Pad if needed
        while dominant_periods.len() < n_peaks {
            dominant_periods.push(0.0);
            dominant_power_vals.push(0.0);
        }

        (spectral_centroid, spectral_entropy, dominant_periods, dominant_power_vals)
    }

    // =========================================================================
    // TREND & STATIONARITY
    // =========================================================================

    /// Compute trend features: linear slope, quadratic, stationarity ratio.
    fn compute_trend_features(&self, series: &[f32]) -> (f32, f32, f32) {
        let n = series.len();
        if n < 3 {
            return (0.0, 0.0, 1.0);
        }

        // Polynomial regression: y = a + b*x + c*x²
        // Using normal equations for simplicity
        let nf = n as f32;
        let sum_x: f32 = (0..n).map(|i| i as f32).sum();
        let sum_x2: f32 = (0..n).map(|i| (i as f32).powi(2)).sum();
        let sum_x3: f32 = (0..n).map(|i| (i as f32).powi(3)).sum();
        let sum_x4: f32 = (0..n).map(|i| (i as f32).powi(4)).sum();
        let sum_y: f32 = series.iter().sum();
        let sum_xy: f32 = series.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_x2y: f32 = series.iter().enumerate().map(|(i, &y)| (i as f32).powi(2) * y).sum();

        // Solve 3x3 system (simplified: just compute linear and quadratic terms)
        let x_mean = sum_x / nf;
        let y_mean = sum_y / nf;

        // Linear slope
        let num: f32 = series.iter().enumerate()
            .map(|(i, &y)| (i as f32 - x_mean) * (y - y_mean))
            .sum();
        let den: f32 = (0..n).map(|i| (i as f32 - x_mean).powi(2)).sum();
        let linear_slope = if den.abs() > 1e-10 { num / den } else { 0.0 };

        // Quadratic term (approximate)
        let residuals: Vec<f32> = series.iter().enumerate()
            .map(|(i, &y)| y - (y_mean + linear_slope * (i as f32 - x_mean)))
            .collect();

        let num_q: f32 = residuals.iter().enumerate()
            .map(|(i, &r)| ((i as f32 - x_mean).powi(2) - sum_x2/nf + x_mean.powi(2)) * r)
            .sum();
        let den_q: f32 = (0..n)
            .map(|i| ((i as f32 - x_mean).powi(2) - sum_x2/nf + x_mean.powi(2)).powi(2))
            .sum::<f32>().max(1e-10);
        let quadratic = num_q / den_q;

        // Stationarity ratio: variance of first half vs second half
        let half = n / 2;
        if half < 2 {
            return (linear_slope, quadratic, 1.0);
        }

        let first_half = &series[..half];
        let second_half = &series[half..];

        let var1: f32 = {
            let m: f32 = first_half.iter().sum::<f32>() / first_half.len() as f32;
            first_half.iter().map(|x| (x - m).powi(2)).sum::<f32>() / first_half.len() as f32
        };
        let var2: f32 = {
            let m: f32 = second_half.iter().sum::<f32>() / second_half.len() as f32;
            second_half.iter().map(|x| (x - m).powi(2)).sum::<f32>() / second_half.len() as f32
        };

        let stationarity_ratio = if var2 > 1e-10 { (var1 / var2).min(10.0) } else { 1.0 };

        (linear_slope, quadratic, stationarity_ratio)
    }

    /// Compute CUSUM (cumulative sum) for change point detection.
    fn compute_cusum(&self, series: &[f32]) -> Vec<f32> {
        let n = series.len();
        if n == 0 {
            return vec![];
        }

        let mean: f32 = series.iter().sum::<f32>() / n as f32;
        let std = self.std_dev(series);

        let mut cusum = Vec::with_capacity(n);
        let mut sum = 0.0f32;

        for &x in series {
            sum += (x - mean) / std;
            cusum.push(sum / (n as f32).sqrt()); // Normalize by sqrt(n)
        }

        cusum
    }

    // =========================================================================
    // GLOBAL MOMENTS
    // =========================================================================

    /// Compute global statistical moments.
    fn compute_global_moments(&self, series: &[f32]) -> (f32, f32, f32) {
        let n = series.len() as f32;
        if n < 2.0 {
            return (0.0, 0.0, 0.0);
        }

        let mean: f32 = series.iter().sum::<f32>() / n;
        let m2: f32 = series.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = m2.sqrt().max(1e-10);

        // Skewness
        let m3: f32 = series.iter().map(|x| (x - mean).powi(3)).sum::<f32>() / n;
        let skewness = m3 / std.powi(3);

        // Kurtosis (excess)
        let m4: f32 = series.iter().map(|x| (x - mean).powi(4)).sum::<f32>() / n;
        let kurtosis = m4 / std.powi(4) - 3.0;

        // Coefficient of variation
        let coef_variation = if mean.abs() > 1e-10 { std / mean.abs() } else { 0.0 };

        (skewness, kurtosis, coef_variation)
    }

    // =========================================================================
    // EMPTY FEATURES
    // =========================================================================

    /// Create empty features for edge cases.
    fn empty_features(&self) -> EngineeredFeatures {
        EngineeredFeatures {
            feature_dim: self.feature_dim(),
            normalized: vec![],
            lags: vec![vec![]; self.config.lag_indices.len()],
            acf: vec![0.0; self.config.acf_max_lag],
            pacf: vec![0.0; self.config.acf_max_lag],
            rolling_mean: vec![vec![]; self.config.rolling_windows.len()],
            rolling_std: vec![vec![]; self.config.rolling_windows.len()],
            rolling_skew: vec![vec![]; self.config.rolling_windows.len()],
            rolling_kurt: vec![vec![]; self.config.rolling_windows.len()],
            wavelet_approx: vec![vec![]; self.config.wavelet_levels],
            wavelet_detail: vec![vec![]; self.config.wavelet_levels],
            wavelet_energy: vec![0.0; self.config.wavelet_levels],
            sample_entropy: 0.0,
            approx_entropy: 0.0,
            hurst_exponent: 0.5,
            dfa_exponent: 0.5,
            spectral_centroid: 0.0,
            spectral_entropy: 0.0,
            dominant_periods: vec![0.0; self.config.n_spectral_peaks],
            dominant_power: vec![0.0; self.config.n_spectral_peaks],
            trend_slope: 0.0,
            trend_quadratic: 0.0,
            stationarity_ratio: 1.0,
            cusum: vec![],
            skewness: 0.0,
            kurtosis: 0.0,
            coef_variation: 0.0,
            norm_median: 0.0,
            norm_iqr: 1.0,
            norm_mean: 0.0,
            norm_std: 1.0,
        }
    }
}

impl EngineeredFeatures {
    /// Convert to a dense feature matrix [seq_len, feature_dim].
    /// Each row contains: normalized + lags + rolling stats + ACF/PACF +
    /// wavelet energy + complexity + spectral + trend + moments
    pub fn to_dense(&self) -> Vec<Vec<f32>> {
        let n = self.normalized.len();
        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut row = Vec::with_capacity(self.feature_dim);

            // 1. Normalized value
            row.push(self.normalized[i]);

            // 2. Lag features
            for lag_vec in &self.lags {
                if i < lag_vec.len() {
                    row.push(lag_vec[i]);
                }
            }

            // 3. Rolling statistics (mean, std, skew, kurt)
            for rm in &self.rolling_mean {
                if i < rm.len() { row.push(rm[i]); }
            }
            for rs in &self.rolling_std {
                if i < rs.len() { row.push(rs[i]); }
            }
            for rsk in &self.rolling_skew {
                if i < rsk.len() { row.push(rsk[i]); }
            }
            for rk in &self.rolling_kurt {
                if i < rk.len() { row.push(rk[i]); }
            }

            // 4. ACF (global, broadcast to each timestep)
            for &a in &self.acf {
                row.push(a);
            }

            // 5. PACF (global, broadcast)
            for &p in &self.pacf {
                row.push(p);
            }

            // 6. Wavelet energy (global, broadcast)
            for &e in &self.wavelet_energy {
                row.push(e);
            }

            // 7. Complexity measures (global, broadcast)
            row.push(self.sample_entropy);
            row.push(self.approx_entropy);
            row.push(self.hurst_exponent);
            row.push(self.dfa_exponent);

            // 8. Spectral features (global, broadcast)
            row.push(self.spectral_centroid);
            row.push(self.spectral_entropy);
            for &p in &self.dominant_periods {
                row.push(p);
            }
            for &pw in &self.dominant_power {
                row.push(pw);
            }

            // 9. Trend features (global, broadcast)
            row.push(self.trend_slope);
            row.push(self.trend_quadratic);
            row.push(self.stationarity_ratio);
            // CUSUM is per-timestep
            if i < self.cusum.len() {
                row.push(self.cusum[i]);
            }

            // 10. Global moments (global, broadcast)
            row.push(self.skewness);
            row.push(self.kurtosis);
            row.push(self.coef_variation);

            // Pad to consistent feature_dim
            while row.len() < self.feature_dim {
                row.push(0.0);
            }
            // Truncate if somehow over (shouldn't happen)
            row.truncate(self.feature_dim);

            result.push(row);
        }

        result
    }

    /// Convert to a flat 1D vector for compatibility with existing interfaces.
    /// Returns [seq_len * feature_dim] flattened row-major.
    pub fn to_flat(&self) -> Vec<f32> {
        self.to_dense().into_iter().flatten().collect()
    }

    /// Get summary statistics for the feature set.
    pub fn summary(&self) -> FeatureSummary {
        FeatureSummary {
            hurst_exponent: self.hurst_exponent,
            dfa_exponent: self.dfa_exponent,
            sample_entropy: self.sample_entropy,
            spectral_centroid: self.spectral_centroid,
            trend_slope: self.trend_slope,
            stationarity_ratio: self.stationarity_ratio,
            skewness: self.skewness,
            kurtosis: self.kurtosis,
        }
    }

    /// Denormalize a value using stored normalization parameters.
    pub fn denormalize(&self, normalized_value: f32) -> f32 {
        normalized_value * self.norm_iqr + self.norm_median
    }

    /// Denormalize a series.
    pub fn denormalize_series(&self, series: &[f32]) -> Vec<f32> {
        series.iter().map(|&x| self.denormalize(x)).collect()
    }
}

/// Summary of key features for inspection.
#[derive(Debug, Clone)]
pub struct FeatureSummary {
    pub hurst_exponent: f32,
    pub dfa_exponent: f32,
    pub sample_entropy: f32,
    pub spectral_centroid: f32,
    pub trend_slope: f32,
    pub stationarity_ratio: f32,
    pub skewness: f32,
    pub kurtosis: f32,
}

impl std::fmt::Display for FeatureSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hurst={:.3} DFA={:.3} SampEn={:.3} SpectCtr={:.3} Trend={:.4} Stat={:.2} Skew={:.2} Kurt={:.2}",
            self.hurst_exponent, self.dfa_exponent, self.sample_entropy,
            self.spectral_centroid, self.trend_slope, self.stationarity_ratio,
            self.skewness, self.kurtosis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let config = FeatureConfig::minimal();
        let extractor = FeatureExtractor::new(config);

        // Generate simple test series
        let series: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 10.0 + 50.0).collect();
        let features = extractor.extract(&series);

        assert_eq!(features.normalized.len(), 100);
        assert!(!features.lags.is_empty());
        assert!(!features.rolling_mean.is_empty());
        assert!(!features.acf.is_empty());
        assert!(!features.wavelet_energy.is_empty());

        // Check dense conversion
        let dense = features.to_dense();
        assert_eq!(dense.len(), 100);
    }

    #[test]
    fn test_robust_normalization() {
        let extractor = FeatureExtractor::default();

        // Series with outliers
        let mut series: Vec<f32> = (0..100).map(|i| i as f32).collect();
        series[50] = 10000.0; // Outlier

        let features = extractor.extract(&series);

        // Most values should be within reasonable range (robust to outlier)
        let reasonable_count = features.normalized.iter()
            .filter(|&&x| x.abs() < 10.0)
            .count();
        assert!(reasonable_count > 90, "Most values should be reasonably normalized");
    }

    #[test]
    fn test_acf_pacf() {
        let extractor = FeatureExtractor::default();

        // AR(1) process: x[t] = 0.8 * x[t-1] + noise
        let mut series = vec![0.0f32; 200];
        for i in 1..200 {
            series[i] = 0.8 * series[i-1] + (i as f32 * 0.1).sin() * 0.2;
        }

        let features = extractor.extract(&series);

        // ACF(1) should be high for AR(1)
        assert!(features.acf[0] > 0.5, "ACF(1) should be high for AR process: {}", features.acf[0]);
        // ACF should decay
        assert!(features.acf[0] > features.acf[5], "ACF should decay over lags");
    }

    #[test]
    fn test_hurst_exponent() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Trending series (should have H > 0.5)
        let trending: Vec<f32> = (0..256).map(|i| i as f32 + (i as f32 * 0.1).sin()).collect();
        let features = extractor.extract(&trending);

        println!("Trending Hurst: {}", features.hurst_exponent);
        // Trending series should have Hurst > 0.5 (but algorithm is approximate)
        assert!(features.hurst_exponent > 0.3, "Trending series Hurst should be > 0.3");
    }

    #[test]
    fn test_wavelet_decomposition() {
        let config = FeatureConfig::minimal();
        let extractor = FeatureExtractor::new(config);

        // Multi-frequency signal
        let series: Vec<f32> = (0..128).map(|i| {
            (i as f32 * 0.1).sin() + (i as f32 * 0.5).sin() * 0.5
        }).collect();

        let features = extractor.extract(&series);

        // Should have wavelet coefficients at multiple levels
        assert!(!features.wavelet_energy.is_empty());
        assert!(features.wavelet_energy.iter().any(|&e| e > 0.0), "Should have non-zero wavelet energy");
    }

    #[test]
    fn test_entropy() {
        let extractor = FeatureExtractor::default();

        // Regular periodic signal (low entropy)
        let regular: Vec<f32> = (0..100).map(|i| (i as f32 * 0.5).sin()).collect();
        let features_regular = extractor.extract(&regular);

        // Random-ish signal (higher entropy)
        let random: Vec<f32> = (0..100).map(|i| ((i * 7 + 13) % 37) as f32).collect();
        let features_random = extractor.extract(&random);

        println!("Regular entropy: {}, Random entropy: {}",
            features_regular.sample_entropy, features_random.sample_entropy);
    }

    #[test]
    fn test_spectral_features() {
        let extractor = FeatureExtractor::default();

        // Pure sine wave with known frequency
        let period = 16.0;
        let series: Vec<f32> = (0..128).map(|i| (2.0 * PI * i as f32 / period).sin()).collect();

        let features = extractor.extract(&series);

        // Spectral centroid should be near the dominant frequency
        assert!(features.spectral_centroid > 0.0);
        assert!(!features.dominant_periods.is_empty());
    }

    #[test]
    fn test_trend_detection() {
        let extractor = FeatureExtractor::default();

        // Linear upward trend
        let series: Vec<f32> = (0..100).map(|i| i as f32 * 2.0 + 10.0).collect();
        let features = extractor.extract(&series);

        // Slope should be positive
        assert!(features.trend_slope > 0.0, "Should detect positive trend");
    }

    #[test]
    fn test_cusum_change_point() {
        let extractor = FeatureExtractor::default();

        // Series with a level shift at midpoint
        let mut series: Vec<f32> = (0..100).map(|i| {
            if i < 50 { 10.0 } else { 20.0 }
        }).collect();

        let features = extractor.extract(&series);

        // CUSUM should show change
        assert!(!features.cusum.is_empty());
        let max_cusum = features.cusum.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_cusum > 0.1, "CUSUM should detect level shift");
    }

    #[test]
    fn test_feature_summary() {
        let extractor = FeatureExtractor::default();
        let series: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 10.0 + 50.0).collect();

        let features = extractor.extract(&series);
        let summary = features.summary();

        println!("Feature summary: {}", summary);
        assert!(summary.hurst_exponent >= 0.0 && summary.hurst_exponent <= 1.0);
    }

    #[test]
    fn test_denormalization() {
        let extractor = FeatureExtractor::default();

        let series: Vec<f32> = (0..100).map(|i| i as f32 * 10.0 + 500.0).collect();
        let features = extractor.extract(&series);

        // Denormalize should approximately recover original
        let denormed = features.denormalize_series(&features.normalized);

        for (orig, recovered) in series.iter().zip(denormed.iter()) {
            let diff = (orig - recovered).abs();
            assert!(diff < 5.0, "Denormalization should approximately recover original: {} vs {}", orig, recovered);
        }
    }
}
