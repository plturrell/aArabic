//! Arabic Translation Embedding Evaluation Framework
//!
//! Tools for comparing embedding models for Arabic-English translation quality,
//! evaluating semantic similarity preservation, and benchmarking performance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod evaluator;
pub mod metrics;
pub mod benchmark;

pub use evaluator::ArabicEvaluator;
pub use metrics::{TranslationEmbeddingMetrics, SemanticSimilarityMetrics};
pub use benchmark::BenchmarkRunner;

/// Model configuration for Arabic translation evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub endpoint: String,
    pub dimension: usize,
    pub model_type: String, // "general" or "financial"
    pub description: String,
    pub supported_languages: Vec<String>,
}

impl ModelConfig {
    /// Multilingual MiniLM - General purpose
    pub fn multilingual_minilm() -> Self {
        Self {
            name: "paraphrase-multilingual-MiniLM-L12-v2".to_string(),
            endpoint: "http://localhost:8007/embed/general".to_string(),
            dimension: 384,
            model_type: "general".to_string(),
            description: "Fast, compact multilingual model (100+ languages)".to_string(),
            supported_languages: vec!["ar".to_string(), "en".to_string()],
        }
    }

    /// CamelBERT - Financial domain
    pub fn camelbert_financial() -> Self {
        Self {
            name: "CamelBERT-Financial".to_string(),
            endpoint: "http://localhost:8007/embed/financial".to_string(),
            dimension: 768,
            model_type: "financial".to_string(),
            description: "Arabic financial domain specialized model".to_string(),
            supported_languages: vec!["ar".to_string(), "en".to_string()],
        }
    }

    /// All-MiniLM-L6-v2 - Baseline
    pub fn all_minilm_l6() -> Self {
        Self {
            name: "all-MiniLM-L6-v2".to_string(),
            endpoint: "http://localhost:8007/embed/general".to_string(),
            dimension: 384,
            model_type: "general".to_string(),
            description: "Baseline compact model for comparison".to_string(),
            supported_languages: vec!["en".to_string()],
        }
    }
}

/// Test case for Arabic translation evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArabicTestCase {
    pub id: String,
    pub arabic_text: String,
    pub english_translation: String,
    pub domain: String, // "general", "financial", "legal", etc.
    pub ground_truth_similar: Vec<String>, // Similar documents (Arabic or English)
    pub ground_truth_dissimilar: Vec<String>, // Dissimilar documents
}

/// Translation pair for semantic similarity testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationPair {
    pub id: String,
    pub arabic: String,
    pub english: String,
    pub domain: String,
    pub similarity_score: f32, // Human-annotated similarity (0-1)
}

/// Model evaluation result for Arabic translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArabicModelEvaluation {
    pub model_name: String,
    pub model_type: String,
    pub dimensions: usize,
    pub test_cases: usize,
    
    // Semantic similarity metrics
    pub semantic_metrics: SemanticSimilarityMetrics,
    
    // Translation embedding metrics
    pub translation_metrics: TranslationEmbeddingMetrics,
    
    // Performance metrics
    pub avg_latency_ms: f64,
    pub throughput_per_sec: f64,
    pub cache_hit_rate: f32,
    
    pub total_time_ms: u64,
}

/// Comparison between models for Arabic translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArabicModelComparison {
    pub models: Vec<String>,
    pub results: Vec<ArabicModelEvaluation>,
    pub best_general: String,
    pub best_financial: String,
    pub best_overall: String,
    pub comparison_date: String,
    pub test_dataset: String,
}

impl ArabicModelComparison {
    pub fn new(test_dataset: String) -> Self {
        Self {
            models: Vec::new(),
            results: Vec::new(),
            best_general: String::new(),
            best_financial: String::new(),
            best_overall: String::new(),
            comparison_date: chrono::Utc::now().to_rfc3339(),
            test_dataset,
        }
    }

    pub fn add_result(&mut self, result: ArabicModelEvaluation) {
        self.models.push(result.model_name.clone());
        self.results.push(result);
    }

    pub fn determine_winners(&mut self) {
        // Best general model
        if let Some(best_gen) = self.results.iter()
            .filter(|r| r.model_type == "general")
            .max_by(|a, b| {
                a.semantic_metrics.cross_lingual_similarity
                    .partial_cmp(&b.semantic_metrics.cross_lingual_similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            self.best_general = best_gen.model_name.clone();
        }

        // Best financial model
        if let Some(best_fin) = self.results.iter()
            .filter(|r| r.model_type == "financial")
            .max_by(|a, b| {
                a.translation_metrics.financial_term_accuracy
                    .partial_cmp(&b.translation_metrics.financial_term_accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            self.best_financial = best_fin.model_name.clone();
        }

        // Best overall (by cross-lingual similarity)
        if let Some(best) = self.results.iter()
            .max_by(|a, b| {
                a.semantic_metrics.cross_lingual_similarity
                    .partial_cmp(&b.semantic_metrics.cross_lingual_similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            self.best_overall = best.model_name.clone();
        }
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::from("# Arabic Translation Embedding Model Comparison\n\n");
        report.push_str(&format!("**Date:** {}\n", self.comparison_date));
        report.push_str(&format!("**Dataset:** {}\n", self.test_dataset));
        report.push_str(&format!("**Models Tested:** {}\n\n", self.models.len()));
        
        report.push_str("## Winners\n\n");
        report.push_str(&format!("- 🏆 **Best Overall:** {}\n", self.best_overall));
        report.push_str(&format!("- 📄 **Best General:** {}\n", self.best_general));
        report.push_str(&format!("- 💰 **Best Financial:** {}\n\n", self.best_financial));
        
        report.push_str("## Detailed Results\n\n");
        report.push_str("| Model | Type | Dims | Cross-Lingual Similarity | Financial Accuracy | Latency (ms) |\n");
        report.push_str("|-------|------|------|--------------------------|-------------------|-------------|\n");
        
        for result in &self.results {
            report.push_str(&format!(
                "| {} | {} | {} | {:.4} | {:.4} | {:.2} |\n",
                result.model_name,
                result.model_type,
                result.dimensions,
                result.semantic_metrics.cross_lingual_similarity,
                result.translation_metrics.financial_term_accuracy,
                result.avg_latency_ms
            ));
        }
        
        report.push_str("\n## Key Metrics Explained\n\n");
        report.push_str("- **Cross-Lingual Similarity**: How well Arabic and English translations align in embedding space (0-1)\n");
        report.push_str("- **Financial Accuracy**: Accuracy of financial term preservation in embeddings (0-1)\n");
        report.push_str("- **Latency**: Average embedding generation time per text\n\n");
        
        report.push_str("## Recommendations\n\n");
        report.push_str(&self.generate_recommendations());
        
        report
    }

    fn generate_recommendations(&self) -> String {
        let mut recs = String::new();
        
        if let Some(best_general) = self.results.iter()
            .find(|r| r.model_name == self.best_general)
        {
            recs.push_str(&format!(
                "### For General Translation\n\nUse **{}** for general Arabic-English documents:\n",
                best_general.model_name
            ));
            recs.push_str(&format!("- Cross-lingual similarity: {:.2}%\n", 
                best_general.semantic_metrics.cross_lingual_similarity * 100.0));
            recs.push_str(&format!("- Latency: {:.2}ms\n", best_general.avg_latency_ms));
            recs.push_str(&format!("- Throughput: {:.0} docs/sec\n\n", best_general.throughput_per_sec));
        }
        
        if let Some(best_fin) = self.results.iter()
            .find(|r| r.model_name == self.best_financial)
        {
            recs.push_str(&format!(
                "### For Financial Documents\n\nUse **{}** for financial Arabic-English documents:\n",
                best_fin.model_name
            ));
            recs.push_str(&format!("- Financial term accuracy: {:.2}%\n", 
                best_fin.translation_metrics.financial_term_accuracy * 100.0));
            recs.push_str(&format!("- Cross-lingual similarity: {:.2}%\n", 
                best_fin.semantic_metrics.cross_lingual_similarity * 100.0));
            recs.push_str(&format!("- Latency: {:.2}ms\n\n", best_fin.avg_latency_ms));
        }
        
        recs
    }
}

impl Default for ArabicModelComparison {
    fn default() -> Self {
        Self::new("default_dataset".to_string())
    }
}
