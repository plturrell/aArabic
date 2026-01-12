//! Surprise-Modulated Memory with Hull-White Dynamic Calibration.
//!
//! This is the core TITAN MAC (Memory as Context) implementation where:
//! - Surprise gates modulate memory consolidation gradients
//! - Hull-White dynamics provide calibrated surprise thresholds
//! - Per-sample gradient weighting based on statistical significance
//!
//! The key innovation: surprise isn't just for monitoring - it directly
//! modulates the gradient flow during backpropagation.

use burn::prelude::*;
use burn::module::Param;
use super::surprise::{SurpriseGate, SurpriseConfig};

/// Configuration for surprise-modulated memory.
#[derive(Config, Debug)]
pub struct SurpriseMemoryConfig {
    /// Memory dimension
    pub dim: usize,
    /// Number of memory slots
    pub slots: usize,
    /// Base forgetting rate (α_base)
    #[config(default = 0.01)]
    pub base_alpha: f32,
    /// Base learning rate for memory (η_base)
    #[config(default = 0.1)]
    pub base_eta: f32,
    /// Minimum gradient weight (prevents zero gradients)
    #[config(default = 0.1)]
    pub min_weight: f32,
    /// Maximum gradient weight (prevents exploding gradients)
    #[config(default = 5.0)]
    pub max_weight: f32,
    /// Surprise threshold for significant events
    #[config(default = 2.0)]
    pub z_threshold: f32,
    /// Hull-White mean reversion speed
    #[config(default = 0.1)]
    pub hw_alpha: f32,
    /// Hull-White volatility
    #[config(default = 1.0)]
    pub hw_sigma: f32,
    /// Calibration window size
    #[config(default = 100)]
    pub calibration_window: usize,
}

/// Surprise-modulated neural memory with gradient weighting.
///
/// During forward pass:
/// 1. Compute prediction error for each sample
/// 2. Feed errors through Hull-White surprise gate
/// 3. Compute gradient weights: w = 1 + θ(surprise)
/// 4. Weight the loss: L_weighted = Σ w_i * L_i
///
/// This causes surprising samples to have larger gradients,
/// making the model learn more from unexpected patterns.
#[derive(Module, Debug)]
pub struct SurpriseMemory<B: Backend> {
    /// Memory tensor [slots, dim] - learnable memory slots
    pub memory: Param<Tensor<B, 2>>,
    /// Query projection for memory retrieval
    pub query_proj: nn::Linear<B>,
    /// Key projection for memory addressing
    pub key_proj: nn::Linear<B>,
    /// Value projection for memory content
    pub value_proj: nn::Linear<B>,
    /// Output projection after memory retrieval
    pub output_proj: nn::Linear<B>,
    /// Memory dimension
    pub dim: usize,
    /// Number of slots
    pub slots: usize,
}

impl SurpriseMemoryConfig {
    /// Initialize surprise-modulated memory.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SurpriseMemory<B> {
        // Initialize memory slots with small random values
        let memory = Tensor::random([self.slots, self.dim], burn::tensor::Distribution::Normal(0.0, 0.02), device);
        
        SurpriseMemory {
            memory: Param::from_tensor(memory),
            query_proj: nn::LinearConfig::new(self.dim, self.dim).init(device),
            key_proj: nn::LinearConfig::new(self.dim, self.dim).init(device),
            value_proj: nn::LinearConfig::new(self.dim, self.dim).init(device),
            output_proj: nn::LinearConfig::new(self.dim, self.dim).init(device),
            dim: self.dim,
            slots: self.slots,
        }
    }
    
    /// Create a surprise gate with this config's Hull-White parameters.
    pub fn create_surprise_gate(&self) -> SurpriseGate {
        let surprise_config = SurpriseConfig {
            z_threshold: self.z_threshold,
            hull_white: super::surprise::HullWhiteParams {
                alpha: self.hw_alpha,
                theta: 0.0,  // Will be calibrated online
                sigma: self.hw_sigma,
            },
            calibration_window: self.calibration_window,
            target_consolidation_rate: 0.1,
            adaptive_calibration: true,
            threshold_lr: 0.01,
        };
        SurpriseGate::new(surprise_config)
    }
}

impl<B: Backend> SurpriseMemory<B> {
    /// Forward pass: retrieve from memory using attention.
    ///
    /// Returns (output, attention_weights) where attention_weights
    /// can be used for visualization and surprise computation.
    pub fn forward(&self, query: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, seq, _dim] = query.dims();
        
        // Project query
        let q = self.query_proj.forward(query);  // [batch, seq, dim]
        
        // Project memory to keys and values
        let memory_expanded = self.memory.val()
            .unsqueeze::<3>()  // [1, slots, dim]
            .repeat_dim(0, batch);  // [batch, slots, dim]
        
        let k = self.key_proj.forward(memory_expanded.clone());  // [batch, slots, dim]
        let v = self.value_proj.forward(memory_expanded);  // [batch, slots, dim]
        
        // Scaled dot-product attention
        let scale = (self.dim as f32).sqrt();
        let scores = q.matmul(k.transpose())  // [batch, seq, slots]
            .div_scalar(scale);
        let attn = burn::tensor::activation::softmax(scores.clone(), 2);
        
        // Retrieve values
        let retrieved = attn.clone().matmul(v);  // [batch, seq, dim]
        let output = self.output_proj.forward(retrieved);

        (output, attn)
    }

    /// Update memory with surprise-modulated learning.
    ///
    /// This performs the TITAN memory update:
    /// M_t = (1 - α(s)) * M_{t-1} + η(s) * v
    ///
    /// Where α(s) and η(s) are surprise-modulated:
    /// - High surprise → lower α (preserve memory), higher η (learn more)
    /// - Low surprise → higher α (forget more), lower η (learn less)
    pub fn update_memory(
        &self,
        new_content: Tensor<B, 2>,  // [slots, dim]
        surprise_weights: &[f32],   // Per-slot surprise weights
        base_alpha: f32,
        base_eta: f32,
    ) -> Tensor<B, 2> {
        let device = self.memory.val().device();
        let [slots, dim] = self.memory.val().dims();

        // Convert surprise weights to tensor
        let weights = Tensor::<B, 1>::from_floats(surprise_weights, &device);

        // Compute modulated rates
        // α(s) = base_alpha * (1 - 0.5 * tanh(s))  -- high surprise → lower forgetting
        // η(s) = base_eta * (1 + tanh(s))          -- high surprise → higher learning
        let tanh_weights = weights.clone().tanh();

        let alpha_mod = tanh_weights.clone()
            .mul_scalar(-0.5)
            .add_scalar(1.0)
            .mul_scalar(base_alpha);  // [slots]

        let eta_mod = tanh_weights
            .add_scalar(1.0)
            .mul_scalar(base_eta);  // [slots]

        // Expand to [slots, dim]
        let alpha_2d = alpha_mod.unsqueeze_dim(1).repeat_dim(1, dim);
        let eta_2d = eta_mod.unsqueeze_dim(1).repeat_dim(1, dim);

        // TITAN update: M_t = (1 - α) * M_{t-1} + η * v
        let one = Tensor::<B, 2>::ones([slots, dim], &device);
        let updated = (one - alpha_2d.clone()) * self.memory.val().clone()
            + eta_2d * new_content;

        updated
    }
}

/// Compute gradient weights from prediction errors using Hull-White surprise.
///
/// This is the core function that bridges surprise computation with gradient modulation:
/// 1. Computes per-sample prediction error magnitude
/// 2. Feeds errors through Hull-White surprise gate
/// 3. Returns gradient weights: w = min_w + (max_w - min_w) * sigmoid(z_score - threshold)
///
/// The gradient weights are used to scale the loss before backprop.
pub fn compute_gradient_weights(
    errors: &[f32],
    surprise_gate: &mut SurpriseGate,
    min_weight: f32,
    max_weight: f32,
) -> Vec<f32> {
    let mut weights = Vec::with_capacity(errors.len());

    for &error in errors {
        // Feed error magnitude through surprise gate
        let surprise = surprise_gate.compute(error.abs(), 1.0);

        // Compute weight using sigmoid of (z_score - threshold)
        // This gives smooth transition: below threshold → ~min_weight, above → ~max_weight
        let z = surprise.z_score;
        let threshold = surprise_gate.current_threshold();

        // Sigmoid with temperature for smooth transition
        let temperature = 1.0;
        let sigmoid = 1.0 / (1.0 + (-(z - threshold) / temperature).exp());

        let weight = min_weight + (max_weight - min_weight) * sigmoid;
        weights.push(weight);
    }

    weights
}

/// Compute surprise-weighted loss for backpropagation.
///
/// Given per-sample losses and gradient weights, compute the weighted average.
/// This causes surprising samples to have larger gradients.
pub fn surprise_weighted_loss<B: Backend>(
    per_sample_loss: Tensor<B, 1>,  // [batch]
    gradient_weights: &[f32],
) -> Tensor<B, 1> {
    let device = per_sample_loss.device();
    let weights = Tensor::<B, 1>::from_floats(gradient_weights, &device);

    // Normalize weights to sum to batch_size (preserve loss scale)
    let batch_size = gradient_weights.len() as f32;
    let weight_sum: f32 = gradient_weights.iter().sum();
    let normalized_weights = weights.mul_scalar(batch_size / weight_sum.max(1e-6));

    // Weighted loss
    per_sample_loss * normalized_weights
}

/// Training state that holds the surprise gate across batches.
///
/// This must persist across training steps to maintain Hull-White calibration.
#[derive(Debug)]
pub struct SurpriseTrainingState {
    /// The surprise gate with Hull-White dynamics
    pub gate: SurpriseGate,
    /// Running statistics for monitoring
    pub total_surprises: u64,
    pub total_triggered: u64,
    pub cumulative_z_score: f64,
}

impl SurpriseTrainingState {
    pub fn new(config: &SurpriseMemoryConfig) -> Self {
        Self {
            gate: config.create_surprise_gate(),
            total_surprises: 0,
            total_triggered: 0,
            cumulative_z_score: 0.0,
        }
    }

    /// Process a batch of errors and return gradient weights.
    pub fn process_batch(
        &mut self,
        errors: &[f32],
        min_weight: f32,
        max_weight: f32,
    ) -> Vec<f32> {
        let weights = compute_gradient_weights(errors, &mut self.gate, min_weight, max_weight);

        // Update statistics
        for &error in errors {
            let surprise = self.gate.compute(error.abs(), 1.0);
            self.total_surprises += 1;
            if surprise.triggered {
                self.total_triggered += 1;
            }
            self.cumulative_z_score += surprise.z_score as f64;
        }

        weights
    }

    /// Get consolidation rate (fraction of triggered surprises).
    pub fn consolidation_rate(&self) -> f32 {
        if self.total_surprises == 0 {
            0.0
        } else {
            self.total_triggered as f32 / self.total_surprises as f32
        }
    }

    /// Get average z-score.
    pub fn avg_z_score(&self) -> f32 {
        if self.total_surprises == 0 {
            0.0
        } else {
            (self.cumulative_z_score / self.total_surprises as f64) as f32
        }
    }

    /// Get current Hull-White parameters (for monitoring calibration).
    pub fn hull_white_params(&self) -> (f32, f32, f32) {
        let params = self.gate.current_params();
        (params.alpha, params.theta, params.sigma)
    }
}

