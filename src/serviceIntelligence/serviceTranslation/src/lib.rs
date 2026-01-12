// Arabic Translation Training System
// High-performance, flexible training pipeline in Rust
// Supports multiple backends: CPU (NdArray), Apple Metal, NVIDIA CUDA (via WGPU)

pub mod data_loader;
pub mod preprocessor;
pub mod trainer;
pub mod evaluator;
pub mod config;
pub mod translator;
pub mod model;
pub mod weight_loader;
pub mod benchmark;
pub mod persistence;
// pub mod backend_selector;  // Disabled - will re-enable with WGPU

pub use data_loader::{DataLoader, DatasetFormat, TranslationPair};
pub use preprocessor::TextPreprocessor;
pub use trainer::TranslationTrainer;
pub use evaluator::MetricsEvaluator;
pub use config::TrainingConfig;
pub use translator::{RustTranslator, TranslatorConfig};
pub use model::M2M100ForConditionalGeneration;
pub use weight_loader::load_model_weights;
// pub use backend_selector::{BackendType, BackendInfo, select_backend, print_backend_info};  // Disabled
