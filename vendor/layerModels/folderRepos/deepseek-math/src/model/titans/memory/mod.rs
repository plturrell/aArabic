//! Titans-style neural memory system.
//!
//! Implements dual-memory architecture:
//! - Short-term buffer: Recent interactions for context
//! - Long-term memory: Consolidated patterns with surprise-based updates
//! - Surprise-modulated memory: Hull-White gradient weighting for training

mod short_term;
mod long_term;
pub mod surprise;
mod router;
pub mod surprise_memory;

pub use short_term::{ShortTermBuffer, ShortTermEntry};
pub use long_term::{LongTermMemory, LongTermPattern};
pub use surprise::{SurpriseGate, SurpriseConfig, SurpriseStats, HullWhiteParams, compute_surprise};
pub use router::{MemoryRouter, RoutingDecision, ToolScore};
pub use surprise_memory::{
    SurpriseMemory, SurpriseMemoryConfig, SurpriseTrainingState,
    compute_gradient_weights, surprise_weighted_loss,
};
