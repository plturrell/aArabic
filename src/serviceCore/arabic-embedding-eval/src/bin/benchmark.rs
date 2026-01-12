//! CLI tool for benchmarking Arabic embedding model performance

use arabic_embedding_eval::{BenchmarkRunner, ModelConfig};
use clap::Parser;
use colored::Colorize;
use tabled::{Table, Tabled};
use anyhow::Result;
use std::fs;

#[derive(Parser)]
#[command(name = "arabic-benchmark")]
#[command(about = "Benchmark Arabic embedding model performance", long_about = None)]
struct Args {
    /// Comma-separated list of models to benchmark
    #[arg(short, long, default_value = "multilingual,camelbert")]
    models: String,

    /// Warmup iterations
    #[arg(short, long, default_value = "10")]
    warmup: usize,

    /// Benchmark iterations
    #[arg(short = 'i', long, default_value = "100")]
    iterations: usize,

    /// Optional JSON output path
    #[arg(short, long)]
    output: Option<String>,
}

#[derive(Tabled)]
struct BenchmarkRow {
    #[tabled(rename = "Model")]
    model: String,
    
    #[tabled(rename = "Single (ms)")]
    single: String,
    
    #[tabled(rename = "Batch/32 (ms)")]
    batch: String,
    
    #[tabled(rename = "Throughput (t/s)")]
    throughput: String,
    
    #[tabled(rename = "P50 (ms)")]
    p50: String,
    
    #[tabled(rename = "P95 (ms)")]
    p95: String,
    
    #[tabled(rename = "P99 (ms)")]
    p99: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let args = Args::parse();

    println!("{}", "\n🔬 Arabic Model Benchmark Tool\n".bright_blue().bold());
    println!("Warmup iterations: {}", args.warmup);
    println!("Benchmark iterations: {}", args.iterations);
    println!();

    // Parse model list
    let model_names: Vec<&str> = args.models.split(',').collect();
    let configs: Vec<ModelConfig> = model_names
        .iter()
        .map(|name| match name.trim() {
            "multilingual" => ModelConfig::multilingual_minilm(),
            "camelbert" => ModelConfig::camelbert_financial(),
            "baseline" => ModelConfig::all_minilm_l6(),
            _ => panic!("Unknown model: {}", name),
        })
        .collect();

    println!("📊 Benchmarking {} models:", configs.len());
    for config in &configs {
        println!("   • {} ({}D, max_len=512)", config.name, config.dimension);
    }
    println!();

    println!("{}", "─".repeat(85).bright_black());

    // Create benchmark runner
    let runner = BenchmarkRunner::new(args.warmup, args.iterations);

    // Collect results
    let mut results = Vec::new();

    // Benchmark each model
    for config in &configs {
        let metrics = runner.benchmark_model(&config).await?;
        
        results.push((config.name.clone(), metrics));
    }

    println!("\n{}", "═".repeat(85).bright_black());
    println!("{}", "\n📊 BENCHMARK RESULTS\n".bright_yellow().bold());

    // Create table
    let rows: Vec<BenchmarkRow> = results
        .iter()
        .map(|(name, metrics)| BenchmarkRow {
            model: name.clone(),
            single: format!("{:.2}", metrics.single_latency_ms),
            batch: format!("{:.2}", metrics.batch_latency_ms),
            throughput: format!("{:.0}", metrics.throughput),
            p50: format!("{:.2}", metrics.p50_latency_ms),
            p95: format!("{:.2}", metrics.p95_latency_ms),
            p99: format!("{:.2}", metrics.p99_latency_ms),
        })
        .collect();

    let table = Table::new(rows).to_string();
    println!("{}", table);

    // Find fastest
    let fastest = results
        .iter()
        .min_by(|a, b| {
            a.1.single_latency_ms
                .partial_cmp(&b.1.single_latency_ms)
                .unwrap()
        })
        .map(|(name, _)| name);

    // Find best throughput
    let best_throughput = results
        .iter()
        .max_by(|a, b| a.1.throughput.partial_cmp(&b.1.throughput).unwrap())
        .map(|(name, _)| name);

    println!("\n🏆 PERFORMANCE WINNERS\n");
    if let Some(name) = fastest {
        println!("   ⚡ Fastest (single): {}", name.bright_green().bold());
    }
    if let Some(name) = best_throughput {
        println!("   🚀 Best throughput: {}", name.bright_cyan().bold());
    }

    // Save JSON output if requested
    if let Some(output_path) = args.output {
        let json_results: Vec<_> = results
            .iter()
            .map(|(name, metrics)| {
                serde_json::json!({
                    "model": name,
                    "single_latency_ms": metrics.single_latency_ms,
                    "batch_latency_ms": metrics.batch_latency_ms,
                    "throughput": metrics.throughput,
                    "p50_ms": metrics.p50_latency_ms,
                    "p95_ms": metrics.p95_latency_ms,
                    "p99_ms": metrics.p99_latency_ms,
                })
            })
            .collect();

        let json_output = serde_json::to_string_pretty(&json_results)?;
        fs::write(&output_path, json_output)?;
        println!("\n✅ Results saved to: {}", output_path.bright_green());
    }

    println!();

    Ok(())
}
