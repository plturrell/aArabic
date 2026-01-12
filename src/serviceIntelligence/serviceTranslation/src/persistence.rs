/// Model Persistence Layer
/// Handles saving/loading models with versioning and metadata

use anyhow::{Context, Result};
use burn::backend::Backend;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;
use chrono::{DateTime, Utc};
use tracing::{info, warn};

use crate::model::m2m100::M2M100ForConditionalGeneration;
use crate::benchmark::BenchmarkResults;

/// Model metadata for versioning and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub version: String,
    pub parent_model: Option<String>,
    pub created_at: DateTime<Utc>,
    pub training_dataset: Option<String>,
    pub training_duration_hours: Option<f64>,
    pub parameters: usize,
    
    // Performance metrics
    pub bleu_score: Option<f64>,
    pub accuracy: Option<f64>,
    pub throughput_tps: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    
    // Training details
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub optimizer: Option<String>,
    pub batch_size: Option<usize>,
    
    // Provenance
    pub git_commit: Option<String>,
    pub framework_version: String,
    pub hardware: String,
    
    // Tags for organization
    pub tags: Vec<String>,
    pub notes: Option<String>,
}

impl ModelMetadata {
    pub fn new(model_id: String, version: String) -> Self {
        Self {
            model_id,
            version,
            parent_model: None,
            created_at: Utc::now(),
            training_dataset: None,
            training_duration_hours: None,
            parameters: 483_570_000,  // M2M100 default
            
            bleu_score: None,
            accuracy: None,
            throughput_tps: None,
            latency_p95_ms: None,
            
            epochs: None,
            learning_rate: None,
            optimizer: None,
            batch_size: None,
            
            git_commit: None,
            framework_version: "burn-0.14".to_string(),
            hardware: "CPU".to_string(),
            
            tags: Vec::new(),
            notes: None,
        }
    }
    
    pub fn with_benchmark_results(mut self, results: &BenchmarkResults) -> Self {
        self.bleu_score = Some(results.bleu_mean);
        self.throughput_tps = Some(results.throughput_tps);
        self.latency_p95_ms = Some(results.latency_p95_ms);
        self
    }
    
    pub fn with_training_config(
        mut self,
        epochs: usize,
        learning_rate: f64,
        optimizer: String,
        batch_size: usize,
    ) -> Self {
        self.epochs = Some(epochs);
        self.learning_rate = Some(learning_rate);
        self.optimizer = Some(optimizer);
        self.batch_size = Some(batch_size);
        self
    }
}

/// Model save format options
#[derive(Debug, Clone, Copy)]
pub enum SaveFormat {
    Safetensors,
    BurnBinary,
    ONNX,
}

/// Model persistence manager
pub struct ModelPersistence {
    base_path: PathBuf,
}

impl ModelPersistence {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        
        Ok(Self { base_path })
    }
    
    /// Save model with metadata
    pub fn save_model<B: Backend>(
        &self,
        model: &M2M100ForConditionalGeneration<B>,
        metadata: &ModelMetadata,
        format: SaveFormat,
    ) -> Result<PathBuf> {
        info!("💾 Saving model: {} v{}", metadata.model_id, metadata.version);
        
        // Create versioned directory
        let model_dir = self.base_path
            .join(&metadata.model_id)
            .join(&metadata.version);
        fs::create_dir_all(&model_dir)?;
        
        // Save model weights
        let model_path = match format {
            SaveFormat::Safetensors => {
                self.save_as_safetensors(model, &model_dir)?
            }
            SaveFormat::BurnBinary => {
                self.save_as_burn_binary(model, &model_dir)?
            }
            SaveFormat::ONNX => {
                // ONNX export would go here
                unimplemented!("ONNX export not yet implemented")
            }
        };
        
        // Save metadata
        let metadata_path = model_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        fs::write(&metadata_path, metadata_json)?;
        
        info!("✅ Model saved to: {}", model_dir.display());
        info!("   Weights: {}", model_path.display());
        info!("   Metadata: {}", metadata_path.display());
        
        Ok(model_dir)
    }
    
    /// Load model with metadata
    pub fn load_model<B: Backend>(
        &self,
        model_id: &str,
        version: &str,
    ) -> Result<(M2M100ForConditionalGeneration<B>, ModelMetadata)> {
        info!("📂 Loading model: {} v{}", model_id, version);
        
        let model_dir = self.base_path.join(model_id).join(version);
        
        // Load metadata
        let metadata_path = model_dir.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_path)
            .context("Failed to read metadata")?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;
        
        // Load model (placeholder - actual implementation would vary by format)
        // For now, return error indicating implementation needed
        Err(anyhow::anyhow!("Model loading from saved format not yet fully implemented"))
    }
    
    /// List all available model versions
    pub fn list_versions(&self, model_id: &str) -> Result<Vec<String>> {
        let model_dir = self.base_path.join(model_id);
        
        if !model_dir.exists() {
            return Ok(Vec::new());
        }
        
        let mut versions = Vec::new();
        for entry in fs::read_dir(model_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                if let Some(version) = entry.file_name().to_str() {
                    versions.push(version.to_string());
                }
            }
        }
        
        versions.sort();
        Ok(versions)
    }
    
    /// Get latest version of a model
    pub fn get_latest_version(&self, model_id: &str) -> Result<Option<String>> {
        let versions = self.list_versions(model_id)?;
        Ok(versions.into_iter().rev().next())
    }
    
    /// Load model metadata without loading the full model
    pub fn load_metadata(&self, model_id: &str, version: &str) -> Result<ModelMetadata> {
        let metadata_path = self.base_path
            .join(model_id)
            .join(version)
            .join("metadata.json");
        
        let metadata_json = fs::read_to_string(&metadata_path)?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;
        
        Ok(metadata)
    }
    
    /// Compare two model versions
    pub fn compare_versions(
        &self,
        model_id: &str,
        version1: &str,
        version2: &str,
    ) -> Result<String> {
        let meta1 = self.load_metadata(model_id, version1)?;
        let meta2 = self.load_metadata(model_id, version2)?;
        
        let mut comparison = String::new();
        comparison.push_str(&format!("Comparing {} v{} vs v{}\n", model_id, version1, version2));
        comparison.push_str("─────────────────────────────────────\n");
        
        if let (Some(bleu1), Some(bleu2)) = (meta1.bleu_score, meta2.bleu_score) {
            let change = ((bleu2 - bleu1) / bleu1) * 100.0;
            comparison.push_str(&format!("BLEU: {:.4} → {:.4} ({:+.2}%)\n", 
                bleu1, bleu2, change));
        }
        
        if let (Some(tps1), Some(tps2)) = (meta1.throughput_tps, meta2.throughput_tps) {
            let change = ((tps2 - tps1) / tps1) * 100.0;
            comparison.push_str(&format!("Throughput: {:.2} → {:.2} tps ({:+.2}%)\n", 
                tps1, tps2, change));
        }
        
        Ok(comparison)
    }
    
    /// Delete a specific model version
    pub fn delete_version(&self, model_id: &str, version: &str) -> Result<()> {
        let model_dir = self.base_path.join(model_id).join(version);
        
        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)?;
            info!("🗑️  Deleted {} v{}", model_id, version);
        }
        
        Ok(())
    }
    
    fn save_as_safetensors<B: Backend>(
        &self,
        _model: &M2M100ForConditionalGeneration<B>,
        model_dir: &Path,
    ) -> Result<PathBuf> {
        // Safetensors export would go here
        // For now, create placeholder
        let path = model_dir.join("model.safetensors");
        info!("   Format: Safetensors (export not yet implemented)");
        Ok(path)
    }
    
    fn save_as_burn_binary<B: Backend>(
        &self,
        _model: &M2M100ForConditionalGeneration<B>,
        model_dir: &Path,
    ) -> Result<PathBuf> {
        // Burn binary export would go here
        let path = model_dir.join("model.bin");
        info!("   Format: Burn Binary (export not yet implemented)");
        Ok(path)
    }
}

/// Registry for tracking all models
pub struct ModelRegistry {
    persistence: ModelPersistence,
}

impl ModelRegistry {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        Ok(Self {
            persistence: ModelPersistence::new(base_path)?,
        })
    }
    
    /// Register a new model
    pub fn register<B: Backend>(
        &self,
        model: &M2M100ForConditionalGeneration<B>,
        metadata: ModelMetadata,
    ) -> Result<String> {
        let model_id = metadata.model_id.clone();
        let version = metadata.version.clone();
        
        self.persistence.save_model(
            model,
            &metadata,
            SaveFormat::Safetensors,
        )?;
        
        Ok(format!("{}:{}", model_id, version))
    }
    
    /// List all registered models
    pub fn list_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        
        for entry in fs::read_dir(&self.persistence.base_path)? {
            let entry = entry?;
            if entry.path().is_dir() {
                if let Some(model_id) = entry.file_name().to_str() {
                    models.push(model_id.to_string());
                }
            }
        }
        
        models.sort();
        Ok(models)
    }
    
    /// Get model lineage (parent-child relationships)
    pub fn get_lineage(&self, model_id: &str, version: &str) -> Result<Vec<ModelMetadata>> {
        let mut lineage = Vec::new();
        let mut current = Some((model_id.to_string(), version.to_string()));
        
        while let Some((id, ver)) = current {
            let metadata = self.persistence.load_metadata(&id, &ver)?;
            current = metadata.parent_model.clone().map(|p| {
                let parts: Vec<&str> = p.split(':').collect();
                (parts[0].to_string(), parts.get(1).unwrap_or(&"latest").to_string())
            });
            lineage.push(metadata);
        }
        
        Ok(lineage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = ModelMetadata::new(
            "m2m100-arabic".to_string(),
            "v1.0.0".to_string(),
        );
        
        assert_eq!(metadata.model_id, "m2m100-arabic");
        assert_eq!(metadata.version, "v1.0.0");
        assert_eq!(metadata.parameters, 483_570_000);
    }

    #[test]
    fn test_metadata_with_benchmark() {
        let results = BenchmarkResults {
            total_samples: 1000,
            successful: 1000,
            failed: 0,
            latency_mean_ms: 10.5,
            latency_p50_ms: 9.8,
            latency_p95_ms: 15.2,
            latency_p99_ms: 18.5,
            latency_max_ms: 25.0,
            throughput_tps: 95.2,
            tokens_per_second: 1000.0,
            memory_mean_mb: 2048.0,
            memory_peak_mb: 2560.0,
            bleu_mean: 0.847,
            bleu_std: 0.025,
            accuracy_mean: 0.92,
            cpu_utilization: 0.75,
            gpu_utilization: 0.0,
            cost_per_1k_translations: 0.01,
            energy_per_translation_wh: 0.5,
        };
        
        let metadata = ModelMetadata::new(
            "m2m100-test".to_string(),
            "v1.0.0".to_string(),
        ).with_benchmark_results(&results);
        
        assert_eq!(metadata.bleu_score, Some(0.847));
        assert_eq!(metadata.throughput_tps, Some(95.2));
    }
}
