//! Long-term memory for consolidated patterns.

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A consolidated pattern in long-term memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermPattern {
    /// Hash of the pattern
    pub pattern_hash: u64,
    /// Domain classification
    pub domain: String,
    /// Tools that worked well for this pattern
    pub successful_tools: Vec<(String, f32)>,
    /// Summary of outcomes
    pub outcome_summary: String,
    /// Number of times this pattern was accessed
    pub access_count: u32,
    /// Surprise value when consolidated
    pub surprise_at_consolidation: f32,
}

impl LongTermPattern {
    pub fn new(pattern_hash: u64, domain: impl Into<String>) -> Self {
        Self {
            pattern_hash,
            domain: domain.into(),
            successful_tools: Vec::new(),
            outcome_summary: String::new(),
            access_count: 0,
            surprise_at_consolidation: 0.0,
        }
    }

    pub fn with_tools(mut self, tools: Vec<(String, f32)>) -> Self {
        self.successful_tools = tools;
        self
    }

    pub fn with_outcome(mut self, summary: impl Into<String>) -> Self {
        self.outcome_summary = summary.into();
        self
    }

    pub fn with_surprise(mut self, surprise: f32) -> Self {
        self.surprise_at_consolidation = surprise;
        self
    }

    /// Get the best tool for this pattern.
    pub fn best_tool(&self) -> Option<&str> {
        self.successful_tools.first().map(|(name, _)| name.as_str())
    }

    /// Increment access count.
    pub fn access(&mut self) {
        self.access_count += 1;
    }
}

/// Long-term memory with neural embeddings.
pub struct LongTermMemory<B: Backend> {
    /// Pattern storage
    patterns: HashMap<u64, LongTermPattern>,
    /// Neural memory tensor [slots, dim]
    memory_tensor: Tensor<B, 2>,
    /// Memory dimension
    dim: usize,
    /// Number of slots
    slots: usize,
    /// Forgetting rate (alpha)
    forgetting_rate: f32,
    /// Learning rate (eta)
    learning_rate: f32,
}

impl<B: Backend> LongTermMemory<B> {
    pub fn new(slots: usize, dim: usize, device: &B::Device) -> Self {
        Self {
            patterns: HashMap::new(),
            memory_tensor: Tensor::zeros([slots, dim], device),
            dim,
            slots,
            forgetting_rate: 0.01,
            learning_rate: 0.01,
        }
    }

    pub fn with_rates(mut self, forgetting: f32, learning: f32) -> Self {
        self.forgetting_rate = forgetting;
        self.learning_rate = learning;
        self
    }

    /// Store a pattern in memory.
    pub fn store(&mut self, pattern: LongTermPattern, embedding: Tensor<B, 1>) {
        let slot = (pattern.pattern_hash as usize) % self.slots;
        
        // Titans-style update: M_t = (1 - α)M_{t-1} + η * v
        let current_slot = self.memory_tensor.clone().slice([slot..slot+1, 0..self.dim]);
        let embedding_2d: Tensor<B, 2> = embedding.unsqueeze_dim(0);
        
        let updated = current_slot.mul_scalar(1.0 - self.forgetting_rate)
            + embedding_2d.mul_scalar(self.learning_rate);
        
        // Update the memory tensor at this slot
        self.memory_tensor = self.memory_tensor.clone().slice_assign(
            [slot..slot+1, 0..self.dim],
            updated
        );
        
        self.patterns.insert(pattern.pattern_hash, pattern);
    }

    /// Retrieve patterns similar to query embedding.
    pub fn retrieve(&self, query: Tensor<B, 1>, top_k: usize) -> Vec<(&LongTermPattern, f32)> {
        // Compute similarity with all memory slots
        let query_2d: Tensor<B, 2> = query.unsqueeze_dim(0);
        
        // Cosine similarity: (query @ memory.T) / (||query|| * ||memory||)
        let similarities = query_2d.clone().matmul(self.memory_tensor.clone().transpose());
        
        // Get top-k slots
        let sim_data: Vec<f32> = similarities.into_data().to_vec().unwrap();
        
        let mut indexed: Vec<(usize, f32)> = sim_data.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        indexed.into_iter()
            .take(top_k)
            .filter_map(|(slot, score)| {
                // Find pattern that maps to this slot
                self.patterns.values()
                    .find(|p| (p.pattern_hash as usize) % self.slots == slot)
                    .map(|p| (p, score))
            })
            .collect()
    }

    /// Get pattern by hash.
    pub fn get(&self, hash: u64) -> Option<&LongTermPattern> {
        self.patterns.get(&hash)
    }

    /// Get mutable pattern by hash.
    pub fn get_mut(&mut self, hash: u64) -> Option<&mut LongTermPattern> {
        self.patterns.get_mut(&hash)
    }

    /// Number of stored patterns.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.slots
    }

    /// Get dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}
impl<B: Backend> std::fmt::Debug for LongTermMemory<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LongTermMemory")
            .field("dim", &self.dim)
            .field("slots", &self.slots)
            .field("patterns_count", &self.patterns.len())
            .finish()
    }
}
