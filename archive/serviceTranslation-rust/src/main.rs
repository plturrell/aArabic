/// Arabic Translation Training System - CLI
/// High-performance data loader and training pipeline

use anyhow::Result;
use arabic_translation_trainer::*;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "arabic-trainer")]
#[command(about = "High-performance Arabic-English translation training", long_about = None)]
struct Args {
    /// Input dataset path(s)
    #[arg(short, long, value_name = "FILE")]
    input: Vec<PathBuf>,

    /// Dataset format (auto, csv, json, parquet, plain)
    #[arg(short, long, default_value = "auto")]
    format: String,

    /// Arabic column/field name
    #[arg(long, default_value = "arabic")]
    arabic_col: String,

    /// English column/field name
    #[arg(long, default_value = "english")]
    english_col: String,

    /// Maximum number of translation pairs
    #[arg(short, long)]
    max_pairs: Option<usize>,

    /// Minimum Arabic text length
    #[arg(long, default_value = "3")]
    min_arabic_len: usize,

    /// Maximum Arabic text length
    #[arg(long, default_value = "512")]
    max_arabic_len: usize,

    /// Filter duplicate pairs
    #[arg(long, default_value = "true")]
    filter_duplicates: bool,

    /// Shuffle dataset
    #[arg(long, default_value = "true")]
    shuffle: bool,

    /// Validation split ratio
    #[arg(long, default_value = "0.1")]
    validation_split: f32,

    /// Output path for processed data
    #[arg(short, long, default_value = "output/pairs.json")]
    output: PathBuf,

    /// Number of threads (0 = auto)
    #[arg(short, long, default_value = "0")]
    threads: usize,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_max_level(if args.verbose { 
            tracing::Level::DEBUG 
        } else { 
            tracing::Level::INFO 
        })
        .init();

    info!("🚀 Arabic Translation Training System");
    info!("   Rust-powered, blazingly fast data processing");

    // Set thread pool size
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()?;
    }

    // Configure data loader
    let format = match args.format.as_str() {
        "auto" => DatasetFormat::Auto,
        "csv" => DatasetFormat::Csv {
            arabic_col: args.arabic_col.clone(),
            english_col: args.english_col.clone(),
            delimiter: b',',
        },
        "tsv" => DatasetFormat::Csv {
            arabic_col: args.arabic_col.clone(),
            english_col: args.english_col.clone(),
            delimiter: b'\t',
        },
        "json" => DatasetFormat::Json {
            arabic_field: args.arabic_col.clone(),
            english_field: args.english_col.clone(),
        },
        "parquet" => DatasetFormat::Parquet {
            arabic_col: args.arabic_col.clone(),
            english_col: args.english_col.clone(),
        },
        _ => anyhow::bail!("Unsupported format: {}", args.format),
    };

    let config = data_loader::DataLoaderConfig {
        format,
        max_pairs: args.max_pairs,
        min_arabic_len: args.min_arabic_len,
        max_arabic_len: args.max_arabic_len,
        min_english_len: 3,
        max_english_len: 512,
        filter_duplicates: args.filter_duplicates,
        shuffle: args.shuffle,
        validation_split: args.validation_split,
    };

    let loader = DataLoader::new(config);

    // Load all input files
    let mut all_pairs = Vec::new();
    for input_path in &args.input {
        info!("📂 Loading: {}", input_path.display());
        let pairs = loader.load(input_path)?;
        info!("   Loaded {} pairs", pairs.len());
        all_pairs.extend(pairs);
    }

    info!("\n✅ Total loaded: {} pairs", all_pairs.len());

    // Preprocess
    info!("🔧 Preprocessing...");
    let preprocessor = TextPreprocessor::default();
    for pair in &mut all_pairs {
        preprocessor.preprocess(pair);
    }

    // Split
    info!("📊 Splitting train/validation...");
    let (train, val) = loader.split(all_pairs);
    info!("   Train: {} pairs", train.len());
    info!("   Validation: {} pairs", val.len());

    // Save processed data
    info!("💾 Saving to: {}", args.output.display());
    let output_data = serde_json::json!({
        "train": train,
        "validation": val,
        "metadata": {
            "total_pairs": train.len() + val.len(),
            "train_pairs": train.len(),
            "val_pairs": val.len(),
            "validation_split": args.validation_split,
        }
    });

    std::fs::create_dir_all(args.output.parent().unwrap())?;
    std::fs::write(&args.output, serde_json::to_string_pretty(&output_data)?)?;

    info!("\n🎉 Complete! Processed {} total pairs", train.len() + val.len());
    info!("   Output: {}", args.output.display());

    Ok(())
}
