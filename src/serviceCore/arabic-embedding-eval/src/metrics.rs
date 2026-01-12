//! Metrics for evaluating Arabic translation embedding models

use serde::{Deserialize, Serialize};

/// Semantic similarity metrics for translation pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSimilarityMetrics {
    /// Cross-lingual similarity: How well Arabic and English align (0-1)
    pub cross_lingual_similarity: f32,
    
    /// Monolingual consistency: Similarity between similar Arabic texts (0-1)
    pub arabic_consistency: f32,
    
    /// Monolingual consistency: Similarity between similar English texts (0-1)
    pub english_consistency: f32,
    
    /// Translation invariance: Embedding distance between translation pairs (lower is better)
    pub translation_distance: f32,
    
    /// Correlation with human judgments (Spearman's rho)
    pub human_correlation: f32,
}

impl SemanticSimilarityMetrics {
    pub fn new() -> Self {
        Self {
            cross_lingual_similarity: 0.0,
            arabic_consistency: 0.0,
            english_consistency: 0.0,
            translation_distance: 0.0,
            human_correlation: 0.0,
        }
    }
    
    /// Calculate overall quality score (0-1)
    pub fn overall_score(&self) -> f32 {
        // Weighted average of key metrics
        let similarity_score = self.cross_lingual_similarity * 0.4;
        let consistency_score = (self.arabic_consistency + self.english_consistency) * 0.2;
        let distance_penalty = (1.0 - self.translation_distance.min(1.0)) * 0.2;
        let human_score = self.human_correlation.max(0.0) * 0.2;
        
        similarity_score + consistency_score + distance_penalty + human_score
    }
}

impl Default for SemanticSimilarityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Translation-specific embedding metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationEmbeddingMetrics {
    /// Financial term preservation accuracy (0-1)
    pub financial_term_accuracy: f32,
    
    /// Technical term preservation accuracy (0-1)
    pub technical_term_accuracy: f32,
    
    /// Named entity preservation accuracy (0-1)
    pub named_entity_accuracy: f32,
    
    /// Semantic role preservation: Subject/Object alignment (0-1)
    pub semantic_role_preservation: f32,
    
    /// Contextual coherence: Sentence-level meaning preservation (0-1)
    pub contextual_coherence: f32,
    
    /// Domain adaptation: Performance on domain-specific text (0-1)
    pub domain_adaptation_score: f32,
}

impl TranslationEmbeddingMetrics {
    pub fn new() -> Self {
        Self {
            financial_term_accuracy: 0.0,
            technical_term_accuracy: 0.0,
            named_entity_accuracy: 0.0,
            semantic_role_preservation: 0.0,
            contextual_coherence: 0.0,
            domain_adaptation_score: 0.0,
        }
    }
    
    /// Calculate overall translation quality (0-1)
    pub fn overall_quality(&self) -> f32 {
        // Average of all translation metrics
        let sum = self.financial_term_accuracy
            + self.technical_term_accuracy
            + self.named_entity_accuracy
            + self.semantic_role_preservation
            + self.contextual_coherence
            + self.domain_adaptation_score;
        
        sum / 6.0
    }
}

impl Default for TranslationEmbeddingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance benchmark metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Single text embedding latency (ms)
    pub single_latency_ms: f64,
    
    /// Batch embedding latency (ms) for 32 texts
    pub batch_latency_ms: f64,
    
    /// Throughput (texts per second)
    pub throughput: f64,
    
    /// Memory usage (MB)
    pub memory_mb: f64,
    
    /// Cache hit rate (0-1)
    pub cache_hit_rate: f32,
    
    /// P50 latency (ms)
    pub p50_latency_ms: f64,
    
    /// P95 latency (ms)
    pub p95_latency_ms: f64,
    
    /// P99 latency (ms)
    pub p99_latency_ms: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            single_latency_ms: 0.0,
            batch_latency_ms: 0.0,
            throughput: 0.0,
            memory_mb: 0.0,
            cache_hit_rate: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a * norm_b)
}

/// Calculate Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Spearman's rank correlation
pub fn spearman_correlation(predictions: &[f32], ground_truth: &[f32]) -> f32 {
    use statrs::statistics::OrderStatistics;
    use statrs::statistics::RankStatistics;
    
    if predictions.len() != ground_truth.len() || predictions.is_empty() {
        return 0.0;
    }
    
    // Convert to ranks
    let pred_ranks = rank_data(predictions);
    let truth_ranks = rank_data(ground_truth);
    
    // Calculate correlation between ranks
    pearson_correlation(&pred_ranks, &truth_ranks)
}

/// Calculate Pearson correlation coefficient
pub fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    
    let n = x.len() as f32;
    let mean_x: f32 = x.iter().sum::<f32>() / n;
    let mean_y: f32 = y.iter().sum::<f32>() / n;
    
    let cov: f32 = x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    
    let std_x: f32 = x.iter()
        .map(|xi| (xi - mean_x).powi(2))
        .sum::<f32>()
        .sqrt();
    
    let std_y: f32 = y.iter()
        .map(|yi| (yi - mean_y).powi(2))
        .sum::<f32>()
        .sqrt();
    
    if std_x == 0.0 || std_y == 0.0 {
        return 0.0;
    }
    
    cov / (std_x * std_y)
}

/// Convert values to ranks
fn rank_data(data: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut ranks = vec![0.0; data.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as f32 + 1.0;
    }
    
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        
        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&c, &d)).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_semantic_metrics_overall_score() {
        let metrics = SemanticSimilarityMetrics {
            cross_lingual_similarity: 0.9,
            arabic_consistency: 0.85,
            english_consistency: 0.88,
            translation_distance: 0.1,
            human_correlation: 0.8,
        };
        
        let score = metrics.overall_score();
        assert!(score > 0.7 && score <= 1.0);
    }

    #[test]
    fn test_translation_metrics_quality() {
        let metrics = TranslationEmbeddingMetrics {
            financial_term_accuracy: 0.9,
            technical_term_accuracy: 0.85,
            named_entity_accuracy: 0.88,
            semantic_role_preservation: 0.92,
            contextual_coherence: 0.87,
            domain_adaptation_score: 0.90,
        };
        
        let quality = metrics.overall_quality();
        assert!(quality > 0.8 && quality <= 1.0);
    }
}
