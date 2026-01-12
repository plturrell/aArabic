/// Configuration management for training pipeline

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub data: DataConfig,
    pub model: ModelConfig,
    pub training: TrainConfig,
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub input_paths: Vec<PathBuf>,
    pub max_pairs: Option<usize>,
    pub min_arabic_len: usize,
    pub max_arabic_len: usize,
    pub min_english_len: usize,
    pub max_english_len: usize,
    pub validation_split: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub model_type: String, // "m2m100-418M" or "m2m100-1.2B"
    pub device: String,      // "cpu" or "cuda"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    pub warmup_steps: usize,
    pub gradient_accumulation: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub model_output: PathBuf,
    pub checkpoint_dir: PathBuf,
    pub log_dir: PathBuf,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            data: DataConfig {
                input_paths: vec![],
                max_pairs: Some(10000),
                min_arabic_len: 3,
                max_arabic_len: 512,
                min_english_len: 3,
                max_english_len: 512,
                validation_split: 0.1,
            },
            model: ModelConfig {
                model_path: PathBuf::from("models/m2m100-418M"),
                model_type: "m2m100-418M".to_string(),
                device: "cpu".to_string(),
            },
            training: TrainConfig {
                batch_size: 32,
                learning_rate: 0.0001,
                epochs: 5,
                warmup_steps: 1000,
                gradient_accumulation: 1,
            },
            output: OutputConfig {
                model_output: PathBuf::from("output/model"),
                checkpoint_dir: PathBuf::from("output/checkpoints"),
                log_dir: PathBuf::from("output/logs"),
            },
        }
    }
}
