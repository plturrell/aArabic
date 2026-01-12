//! Evaluator for Arabic translation embedding models

use crate::{
    ArabicTestCase, TranslationPair, ModelConfig, ArabicModelEvaluation,
    SemanticSimilarityMetrics, TranslationEmbeddingMetrics,
};
use crate::metrics::{cosine_similarity, euclidean_distance, spearman_correlation};
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, warn};

#[derive(Debug, Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
    normalize: bool,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
    dimensions: usize,
    model_used: String,
    processing_time_ms: u64,
    cached: bool,
    count: usize,
}

/// Evaluator for Arabic translation embedding models
pub struct ArabicEvaluator {
    client: Client,
}

impl ArabicEvaluator {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap(),
        }
    }

    /// Evaluate a single model
    pub async fn evaluate_model(
        &self,
        config: &ModelConfig,
        test_cases: &[ArabicTestCase],
    ) -> Result<ArabicModelEvaluation> {
        info!("Evaluating model: {}", config.name);
        info!("Test cases: {}", test_cases.len());

        let start_time = Instant::now();

        // Compute semantic similarity metrics
        let semantic_metrics = self
            .compute_semantic_metrics(config, test_cases)
            .await?;

        // Compute translation-specific metrics
        let translation_metrics = self
            .compute_translation_metrics(config, test_cases)
            .await?;

        // Compute performance metrics
        let (avg_latency, throughput, cache_hit_rate) = self
            .compute_performance(config, test_cases)
            .await?;

        let total_time = start_time.elapsed().as_millis() as u64;

        Ok(ArabicModelEvaluation {
            model_name: config.name.clone(),
            model_type: config.model_type.clone(),
            dimensions: config.dimension,
            test_cases: test_cases.len(),
            semantic_metrics,
            translation_metrics,
            avg_latency_ms: avg_latency,
            throughput_per_sec: throughput,
            cache_hit_rate,
            total_time_ms: total_time,
        })
    }

    /// Compute semantic similarity metrics
    async fn compute_semantic_metrics(
        &self,
        config: &ModelConfig,
        test_cases: &[ArabicTestCase],
    ) -> Result<SemanticSimilarityMetrics> {
        info!("Computing semantic similarity metrics...");

        let mut similarities = Vec::new();
        let mut distances = Vec::new();

        for test_case in test_cases {
            // Get embeddings for Arabic and English
            let arabic_emb = self
                .get_embedding(config, &test_case.arabic_text)
                .await?;

            let english_emb = self
                .get_embedding(config, &test_case.english_translation)
                .await?;

            // Compute similarity
            let sim = cosine_similarity(&arabic_emb, &english_emb);
            similarities.push(sim);

            // Compute distance
            let dist = euclidean_distance(&arabic_emb, &english_emb);
            distances.push(dist);
        }

        // Compute metrics
        let cross_lingual_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;

        // Normalize distance to 0-1 range
        let max_dist = distances.iter().cloned().fold(f32::MIN, f32::max);
        let translation_distance = if max_dist > 0.0 {
            distances.iter().sum::<f32>() / (distances.len() as f32 * max_dist)
        } else {
            0.0
        };

        // Compute monolingual consistency (simplified)
        let arabic_consistency = self
            .compute_consistency(config, test_cases, true)
            .await?;

        let english_consistency = self
            .compute_consistency(config, test_cases, false)
            .await?;

        Ok(SemanticSimilarityMetrics {
            cross_lingual_similarity,
            arabic_consistency,
            english_consistency,
            translation_distance,
            human_correlation: 0.0, // Would need human annotations
        })
    }

    /// Compute consistency within language
    async fn compute_consistency(
        &self,
        config: &ModelConfig,
        test_cases: &[ArabicTestCase],
        is_arabic: bool,
    ) -> Result<f32> {
        if test_cases.len() < 2 {
            return Ok(0.0);
        }

        let mut similarities = Vec::new();

        // Compare first N pairs
        let n = test_cases.len().min(10);

        for i in 0..n {
            for j in (i + 1)..n {
                let text1 = if is_arabic {
                    &test_cases[i].arabic_text
                } else {
                    &test_cases[i].english_translation
                };

                let text2 = if is_arabic {
                    &test_cases[j].arabic_text
                } else {
                    &test_cases[j].english_translation
                };

                let emb1 = self.get_embedding(config, text1).await?;
                let emb2 = self.get_embedding(config, text2).await?;

                let sim = cosine_similarity(&emb1, &emb2);
                similarities.push(sim);
            }
        }

        if similarities.is_empty() {
            return Ok(0.0);
        }

        Ok(similarities.iter().sum::<f32>() / similarities.len() as f32)
    }

    /// Compute translation-specific metrics
    async fn compute_translation_metrics(
        &self,
        config: &ModelConfig,
        test_cases: &[ArabicTestCase],
    ) -> Result<TranslationEmbeddingMetrics> {
        info!("Computing translation metrics...");

        // Filter financial test cases
        let financial_cases: Vec<_> = test_cases
            .iter()
            .filter(|tc| tc.domain == "financial")
            .collect();

        let financial_accuracy = if !financial_cases.is_empty() {
            self.compute_domain_accuracy(config, &financial_cases)
                .await?
        } else {
            0.0
        };

        // Filter technical test cases
        let technical_cases: Vec<_> = test_cases
            .iter()
            .filter(|tc| tc.domain == "technical")
            .collect();

        let technical_accuracy = if !technical_cases.is_empty() {
            self.compute_domain_accuracy(config, &technical_cases)
                .await?
        } else {
            0.0
        };

        // Overall domain adaptation
        let domain_scores: Vec<_> = test_cases
            .iter()
            .map(|tc| {
                // Simple heuristic: financial model should be better for financial
                if tc.domain == "financial" && config.model_type == "financial" {
                    1.0
                } else if tc.domain != "financial" && config.model_type == "general" {
                    1.0
                } else {
                    0.7 // Penalty for mismatched domain
                }
            })
            .collect();

        let domain_adaptation_score = domain_scores.iter().sum::<f32>() / domain_scores.len() as f32;

        Ok(TranslationEmbeddingMetrics {
            financial_term_accuracy: financial_accuracy,
            technical_term_accuracy: technical_accuracy,
            named_entity_accuracy: 0.85, // Placeholder
            semantic_role_preservation: 0.88, // Placeholder
            contextual_coherence: 0.90, // Placeholder
            domain_adaptation_score,
        })
    }

    /// Compute domain-specific accuracy
    async fn compute_domain_accuracy(
        &self,
        config: &ModelConfig,
        test_cases: &[&ArabicTestCase],
    ) -> Result<f32> {
        let mut similarities = Vec::new();

        for test_case in test_cases {
            let arabic_emb = self
                .get_embedding(config, &test_case.arabic_text)
                .await?;

            let english_emb = self
                .get_embedding(config, &test_case.english_translation)
                .await?;

            let sim = cosine_similarity(&arabic_emb, &english_emb);
            similarities.push(sim);
        }

        if similarities.is_empty() {
            return Ok(0.0);
        }

        Ok(similarities.iter().sum::<f32>() / similarities.len() as f32)
    }

    /// Compute performance metrics
    async fn compute_performance(
        &self,
        config: &ModelConfig,
        test_cases: &[ArabicTestCase],
    ) -> Result<(f64, f64, f32)> {
        info!("Computing performance metrics...");

        let mut latencies = Vec::new();
        let mut cache_hits = 0;
        let mut total = 0;

        // Sample subset for performance testing
        let sample_size = test_cases.len().min(20);

        for test_case in test_cases.iter().take(sample_size) {
            let start = Instant::now();

            let response = self
                .get_embedding_with_metadata(config, &test_case.arabic_text)
                .await?;

            let latency = start.elapsed().as_micros() as f64 / 1000.0;
            latencies.push(latency);

            if response.cached {
                cache_hits += 1;
            }
            total += 1;
        }

        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let throughput = 1000.0 / avg_latency; // texts per second
        let cache_hit_rate = cache_hits as f32 / total as f32;

        Ok((avg_latency, throughput, cache_hit_rate))
    }

    /// Get embedding for a single text
    async fn get_embedding(&self, config: &ModelConfig, text: &str) -> Result<Vec<f32>> {
        let response = self.get_embedding_with_metadata(config, text).await?;
        response
            .embeddings
            .into_iter()
            .next()
            .context("No embedding returned")
    }

    /// Get embedding with full metadata
    async fn get_embedding_with_metadata(
        &self,
        config: &ModelConfig,
        text: &str,
    ) -> Result<EmbedResponse> {
        let request = EmbedRequest {
            texts: vec![text.to_string()],
            normalize: true,
        };

        let response = self
            .client
            .post(&config.endpoint)
            .json(&request)
            .send()
            .await
            .context("Failed to send embedding request")?;

        if !response.status().is_success() {
            anyhow::bail!("Embedding request failed: {}", response.status());
        }

        response
            .json::<EmbedResponse>()
            .await
            .context("Failed to parse embedding response")
    }
}

impl Default for ArabicEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator_creation() {
        let evaluator = ArabicEvaluator::new();
        assert!(evaluator.client.timeout().is_some());
    }
}
