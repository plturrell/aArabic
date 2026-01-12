//! CLI tool for comparing Arabic embedding models

use arabic_embedding_eval::{
    ArabicEvaluator, ArabicTestCase, ArabicModelComparison, ModelConfig,
};
use clap::Parser;
use colored::Colorize;
use std::fs;
use tabled::{Table, Tabled};
use anyhow::Result;

#[derive(Parser)]
#[command(name = "arabic-model-comparison")]
#[command(about = "Compare Arabic translation embedding models", long_about = None)]
struct Args {
    /// Path to test cases JSON file
    #[arg(short, long)]
    test_cases: String,

    /// Comma-separated list of models to compare (multilingual,camelbert)
    #[arg(short, long, default_value = "multilingual,camelbert")]
    models: String,

    /// Output report path
    #[arg(short, long, default_value = "comparison_report.md")]
    output: String,
}

#[derive(Tabled)]
struct ResultRow {
    #[tabled(rename = "Model")]
    model: String,
    
    #[tabled(rename = "Type")]
    model_type: String,
    
    #[tabled(rename = "Dims")]
    dimensions: String,
    
    #[tabled(rename = "Cross-Ling")]
    cross_lingual: String,
    
    #[tabled(rename = "Financial")]
    financial: String,
    
    #[tabled(rename = "Latency (ms)")]
    latency: String,
    
    #[tabled(rename = "Throughput")]
    throughput: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let args = Args::parse();

    println!("{}", "\n🚀 Arabic Model Comparison Tool\n".bright_blue().bold());
    println!("Test cases: {}", args.test_cases);
    println!("Output: {}", args.output);
    println!();

    // Load test cases
    println!("📚 Loading test cases...");
    let test_cases: Vec<ArabicTestCase> = {
        let content = fs::read_to_string(&args.test_cases)?;
        serde_json::from_str(&content)?
    };
    println!("✅ Loaded {} test cases\n", test_cases.len());

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

    println!("🔬 Comparing {} models:", configs.len());
    for config in &configs {
        println!("   • {} ({}D)", config.name, config.dimension);
    }
    println!();

    println!("{}", "─".repeat(65).bright_black());

    // Create evaluator
    let evaluator = ArabicEvaluator::new();

    // Create comparison
    let mut comparison = ArabicModelComparison::new(args.test_cases.clone());

    // Evaluate each model
    for config in &configs {
        println!("\n📊 Evaluating model: {}", config.name.bright_green().bold());
        println!("   Path: {}", config.endpoint);
        println!("   Test cases: {}", test_cases.len());

        let result = evaluator.evaluate_model(config, &test_cases).await?;

        println!("✅ Evaluation complete!");
        println!("   Cross-lingual similarity: {:.4}", result.semantic_metrics.cross_lingual_similarity);
        println!("   Financial accuracy: {:.4}", result.translation_metrics.financial_term_accuracy);
        println!("   Avg latency: {:.2}ms", result.avg_latency_ms);

        comparison.add_result(result);
    }

    println!("\n{}", "═".repeat(65).bright_black());
    println!("{}", "\n📊 COMPARISON RESULTS\n".bright_yellow().bold());

    // Determine winners
    comparison.determine_winners();

    // Create table
    let rows: Vec<ResultRow> = comparison
        .results
        .iter()
        .map(|r| ResultRow {
            model: r.model_name.clone(),
            model_type: r.model_type.clone(),
            dimensions: r.dimensions.to_string(),
            cross_lingual: format!("{:.4}", r.semantic_metrics.cross_lingual_similarity),
            financial: format!("{:.4}", r.translation_metrics.financial_term_accuracy),
            latency: format!("{:.2}", r.avg_latency_ms),
            throughput: format!("{:.0}", r.throughput_per_sec),
        })
        .collect();

    let table = Table::new(rows).to_string();
    println!("{}", table);

    println!("\n🏆 WINNERS\n");
    println!("   Overall: {}", comparison.best_overall.bright_green().bold());
    if !comparison.best_general.is_empty() {
        println!("   General: {}", comparison.best_general.bright_cyan());
    }
    if !comparison.best_financial.is_empty() {
        println!("   Financial: {}", comparison.best_financial.bright_magenta());
    }

    // Generate report
    let report = comparison.generate_report();
    fs::write(&args.output, report)?;

    println!("\n✅ Report saved to: {}", args.output.bright_green());
    println!();

    Ok(())
}
