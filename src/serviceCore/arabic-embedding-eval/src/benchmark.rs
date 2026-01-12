//! Performance benchmarking for Arabic embedding models

use crate::{ModelConfig, metrics::PerformanceMetrics};
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::info;

#[derive(Debug, Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
    normalize: bool,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
    processing_time_ms: u64,
    cached: bool,
}

/// Benchmark runner for performance testing
pub struct BenchmarkRunner {
    client: Client,
    warmup_iterations: usize,
    bench_iterations: usize,
}

impl BenchmarkRunner {
    pub fn new(warmup_iterations: usize, bench_iterations: usize) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap(),
            warmup_iterations,
            bench_iterations,
        }
    }

    /// Run complete benchmark suite
    pub async fn benchmark_model(&self, config: &ModelConfig) -> Result<PerformanceMetrics> {
        info!("🔬 Benchmarking: {}", config.name);

        // Warmup
        info!("  Warming up ({} iterations)...", self.warmup_iterations);
        self.warmup(config).await?;

        // Single text benchmark
        info!("  Benchmarking single encoding...");
        let single_latency = self.benchmark_single(config).await?;

        // Batch benchmark
        info!("  Benchmarking batch encoding (32 texts)...");
        let batch_latency = self.benchmark_batch(config, 32).await?;

        // Compute throughput
        let throughput = 1000.0 / single_latency;

        // Percentile measurements
        let latencies = self.measure_latencies(config, 100).await?;
        let (p50, p95, p99) = Self::compute_percentiles(&latencies);

        info!("✅ Benchmark complete!");
        info!("  Single encode: {:.2}ms", single_latency);
        info!("  Batch encode (32): {:.2}ms", batch_latency);
        info!("  Throughput: {:.1} texts/sec", throughput);

        Ok(PerformanceMetrics {
            single_latency_ms: single_latency,
            batch_latency_ms: batch_latency,
            throughput,
            memory_mb: 0.0, // Would need system metrics
            cache_hit_rate: 0.0,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
        })
    }

    /// Warmup phase
    async fn warmup(&self, config: &ModelConfig) -> Result<()> {
        let text = "مرحبا بك في نظام التقييم".to_string();

        for _ in 0..self.warmup_iterations {
            let _ = self.get_embedding(config, &text).await;
        }

        Ok(())
    }

    /// Benchmark single text encoding
    async fn benchmark_single(&self, config: &ModelConfig) -> Result<f64> {
        let text = "فاتورة رقم 12345 بمبلغ 5000 ريال سعودي".to_string();
        let mut latencies = Vec::new();

        for _ in 0..self.bench_iterations {
            let start = Instant::now();
            let _ = self.get_embedding(config, &text).await?;
            let latency = start.elapsed().as_micros() as f64 / 1000.0;
            latencies.push(latency);
        }

        Ok(latencies.iter().sum::<f64>() / latencies.len() as f64)
    }

    /// Benchmark batch encoding
    async fn benchmark_batch(&self, config: &ModelConfig, batch_size: usize) -> Result<f64> {
        let texts: Vec<String> = (0..batch_size)
            .map(|i| format!("نص تجريبي رقم {} للاختبار", i))
            .collect();

        let mut latencies = Vec::new();

        for _ in 0..self.bench_iterations.min(20) {
            let start = Instant::now();
            let _ = self.get_batch_embeddings(config, &texts).await?;
            let latency = start.elapsed().as_micros() as f64 / 1000.0;
            latencies.push(latency);
        }

        Ok(latencies.iter().sum::<f64>() / latencies.len() as f64)
    }

    /// Measure latencies for percentile calculation
    async fn measure_latencies(&self, config: &ModelConfig, n: usize) -> Result<Vec<f64>> {
        let text = "اختبار الأداء للنظام".to_string();
        let mut latencies = Vec::new();

        for _ in 0..n {
            let start = Instant::now();
            let _ = self.get_embedding(config, &text).await?;
            let latency = start.elapsed().as_micros() as f64 / 1000.0;
            latencies.push(latency);
        }

        Ok(latencies)
    }

    /// Compute percentiles
    fn compute_percentiles(latencies: &[f64]) -> (f64, f64, f64) {
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50_idx = (sorted.len() as f64 * 0.50) as usize;
        let p95_idx = (sorted.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted.len() as f64 * 0.99) as usize;

        (
            sorted[p50_idx.min(sorted.len() - 1)],
            sorted[p95_idx.min(sorted.len() - 1)],
            sorted[p99_idx.min(sorted.len() - 1)],
        )
    }

    /// Get single embedding
    async fn get_embedding(&self, config: &ModelConfig, text: &str) -> Result<Vec<f32>> {
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
            .context("Failed to send request")?;

        let embed_response: EmbedResponse = response
            .json()
            .await
            .context("Failed to parse response")?;

        embed_response
            .embeddings
            .into_iter()
            .next()
            .context("No embedding returned")
    }

    /// Get batch embeddings
    async fn get_batch_embeddings(
        &self,
        config: &ModelConfig,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let request = EmbedRequest {
            texts: texts.to_vec(),
            normalize: true,
        };

        let response = self
            .client
            .post(&config.endpoint)
            .json(&request)
            .send()
            .await
            .context("Failed to send batch request")?;

        let embed_response: EmbedResponse = response
            .json()
            .await
            .context("Failed to parse batch response")?;

        Ok(embed_response.embeddings)
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new(10, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentiles() {
        let latencies = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (p50, p95, p99) = BenchmarkRunner::compute_percentiles(&latencies);
        
        assert!((p50 - 5.0).abs() < 1.0);
        assert!((p95 - 9.5).abs() < 1.0);
        assert!((p99 - 9.9).abs() < 1.0);
    }

    #[test]
    fn test_benchmark_runner_creation() {
        let runner = BenchmarkRunner::new(10, 100);
        assert_eq!(runner.warmup_iterations, 10);
        assert_eq!(runner.bench_iterations, 100);
    }
}
