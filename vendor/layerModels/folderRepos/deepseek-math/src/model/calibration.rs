//! Surprise Calibration Module.
//!
//! Implements adaptive calibration of surprise threshold and decay parameters
//! based on observed prediction errors during training and inference.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for surprise calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Window size for computing statistics
    pub window_size: usize,
    /// Target consolidation rate (fraction of samples to consolidate)
    pub target_consolidation_rate: f32,
    /// Learning rate for threshold adaptation
    pub threshold_lr: f32,
    /// Learning rate for decay adaptation
    pub decay_lr: f32,
    /// Minimum threshold value
    pub min_threshold: f32,
    /// Maximum threshold value
    pub max_threshold: f32,
    /// Minimum decay value
    pub min_decay: f32,
    /// Maximum decay value
    pub max_decay: f32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            target_consolidation_rate: 0.2,  // Consolidate ~20% of samples
            threshold_lr: 0.01,
            decay_lr: 0.005,
            min_threshold: 0.1,
            max_threshold: 0.9,
            min_decay: 0.8,
            max_decay: 0.99,
        }
    }
}

/// Statistics from the calibration window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStats {
    pub mean_surprise: f32,
    pub std_surprise: f32,
    pub min_surprise: f32,
    pub max_surprise: f32,
    pub consolidation_rate: f32,
    pub current_threshold: f32,
    pub current_decay: f32,
    pub samples_seen: usize,
}

/// Observation for calibration.
#[derive(Debug, Clone)]
pub struct CalibrationObservation {
    pub surprise: f32,
    pub prediction_error: f32,
    pub was_consolidated: bool,
    pub domain: String,
}

/// Adaptive surprise calibrator.
/// Tunes threshold and decay based on observed behavior.
pub struct SurpriseCalibrator {
    config: CalibrationConfig,
    /// Rolling window of observations
    observations: VecDeque<CalibrationObservation>,
    /// Current threshold (calibrated)
    current_threshold: f32,
    /// Current decay (calibrated)
    current_decay: f32,
    /// Total samples seen
    total_samples: usize,
    /// Per-domain statistics
    domain_stats: std::collections::HashMap<String, DomainCalibrationStats>,
}

/// Per-domain calibration statistics.
#[derive(Debug, Clone, Default)]
struct DomainCalibrationStats {
    count: usize,
    sum_surprise: f32,
    sum_sq_surprise: f32,
    consolidations: usize,
}

impl SurpriseCalibrator {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            current_threshold: 0.3,  // Start with default
            current_decay: 0.9,
            observations: VecDeque::with_capacity(config.window_size),
            total_samples: 0,
            domain_stats: std::collections::HashMap::new(),
            config,
        }
    }

    /// Record an observation and update calibration.
    pub fn observe(&mut self, obs: CalibrationObservation) {
        // Update domain stats
        let domain_stats = self.domain_stats
            .entry(obs.domain.clone())
            .or_default();
        domain_stats.count += 1;
        domain_stats.sum_surprise += obs.surprise;
        domain_stats.sum_sq_surprise += obs.surprise * obs.surprise;
        if obs.was_consolidated {
            domain_stats.consolidations += 1;
        }

        // Add to window (evict oldest if full)
        if self.observations.len() >= self.config.window_size {
            self.observations.pop_front();
        }
        self.observations.push_back(obs);
        self.total_samples += 1;

        // Calibrate after enough samples
        if self.total_samples % 100 == 0 && self.observations.len() >= 100 {
            self.calibrate();
        }
    }

    /// Perform calibration update.
    fn calibrate(&mut self) {
        let stats = self.compute_stats();
        
        // Adapt threshold based on consolidation rate
        let rate_error = stats.consolidation_rate - self.config.target_consolidation_rate;
        
        // If consolidating too much, increase threshold
        // If consolidating too little, decrease threshold
        let threshold_update = rate_error * self.config.threshold_lr;
        self.current_threshold = (self.current_threshold + threshold_update)
            .clamp(self.config.min_threshold, self.config.max_threshold);
        
        // Adapt decay based on surprise variance
        // High variance -> slower decay (more stable running average)
        // Low variance -> faster decay (more responsive)
        let normalized_std = stats.std_surprise / (stats.mean_surprise + 1e-6);
        let decay_target = if normalized_std > 1.0 {
            self.config.max_decay  // High variance, slow decay
        } else if normalized_std < 0.3 {
            self.config.min_decay  // Low variance, fast decay
        } else {
            // Interpolate
            self.config.min_decay + 
                (self.config.max_decay - self.config.min_decay) * (normalized_std - 0.3) / 0.7
        };
        
        self.current_decay += self.config.decay_lr * (decay_target - self.current_decay);
        self.current_decay = self.current_decay.clamp(self.config.min_decay, self.config.max_decay);
    }

