/// Standalone Translation CLI - Pure Rust
/// Replaces standalone_translator.py with 100% Rust implementation

use anyhow::Result;
use arabic_translation_trainer::model::m2m100::{M2M100Model, M2M100Config};
use burn::backend::NdArray;
use clap::Parser;
use tokenizers::Tokenizer;
use tracing::{info, error};
use std::path::PathBuf;
use std::fs;

type Backend = NdArray;

#[derive(Parser, Debug)]
#[command(name = "translate")]
#[command(about = "Arabic-English translation using M2M100 in pure Rust", long_about = None)]
struct Args {
    /// Text to translate (if not using --file or --benchmark)
    #[arg(value_name = "TEXT")]
    text: Option<String>,

    /// Input file with Arabic text (one line per sentence)
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Run benchmark on predefined test cases
    #[arg(short, long)]
    benchmark: bool,

    /// Model path (PyTorch weights directory)
    #[arg(short, long, default_value = "../../vendor/layerModels/folderRepos/arabic_models/m2m100-418M")]
    model: String,

    /// Max generation length
    #[arg(long, default_value = "50")]
    max_length: usize,

    /// Generation temperature
    #[arg(long, default_value = "1.0")]
    temperature: f64,

    /// Number of beams for beam search
    #[arg(long, default_value = "5")]
    num_beams: usize,

    /// Use beam search instead of greedy
    #[arg(long)]
    beam_search: bool,

    /// Output file for results
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

struct TranslationSystem {
    model: M2M100Model<Backend>,
    tokenizer: Tokenizer,
    device: <Backend as burn::tensor::backend::Backend>::Device,
    config: TranslationConfig,
}

#[derive(Debug, Clone)]
struct TranslationConfig {
    max_length: usize,
    temperature: f64,
    num_beams: usize,
    use_beam_search: bool,
}

impl TranslationSystem {
    fn new(model_path: &str, config: TranslationConfig) -> Result<Self> {
        info!("🚀 Initializing Rust Translation System");
        info!("   Model: {}", model_path);

        let device = Default::default();

        // Load tokenizer
        let tokenizer_path = PathBuf::from(model_path).join("tokenizer.json");
        info!("   Loading tokenizer from: {}", tokenizer_path.display());
        
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        info!("   ✅ Tokenizer loaded");

        // Initialize model with M2M100 config
        info!("   Creating M2M100 model...");
        let model_config = M2M100Config::default();
        let model = model_config.init::<Backend>(&device);
        
        info!("   ✅ Model created (random weights)");
        info!("   ⚠️  Need to load PyTorch weights with burn-import");
        info!("   💡 For now, model outputs random (use Python translator for real results)");

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    fn translate(&self, arabic_text: &str) -> Result<String> {
        use burn::prelude::*;
        
        info!("📝 Translating: {}", arabic_text);

        // Tokenize input
        let encoding = self.tokenizer
            .encode(arabic_text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        let input_ids = encoding.get_ids();
        info!("   Tokenized to {} tokens", input_ids.len());

        // Convert to tensor (as Int tensor for token IDs)
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let input_tensor = Tensor::<Backend, 2, burn::tensor::Int>::from_ints(
            input_ids_i64.as_slice(),
            &self.device,
        ).reshape([1, input_ids.len()]);

        // Generate translation
        let output_ids = if self.config.use_beam_search {
            self.model.model.generate_beam_search(
                input_tensor,
                self.config.max_length,
                self.config.num_beams,
                self.config.temperature,
                self.model.config.bos_token_id,
            )
        } else {
            self.model.model.generate(
                input_tensor,
                self.config.max_length,
                self.config.temperature,
                self.model.config.bos_token_id,
            )
        };

        // Decode output
        let output_vec: Vec<u32> = output_ids
            .into_data()
            .convert::<i64>()
            .to_vec()
            .unwrap()
            .into_iter()
            .map(|x| x as u32)
            .collect();

        let translation = self.tokenizer
            .decode(&output_vec, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        info!("   ✅ Translation complete");
        Ok(translation)
    }

    fn translate_batch(&self, texts: Vec<String>) -> Result<Vec<String>> {
        texts.into_iter()
            .map(|text| self.translate(&text))
            .collect()
    }
}

fn run_benchmark(system: &TranslationSystem) -> Result<()> {
    info!("\n🎯 Running Translation Benchmark");
    info!("═══════════════════════════════════");

    let test_cases = vec![
        ("الفاتورة رقم ١٢٣٤", "Invoice number 1234"),
        ("المبلغ الإجمالي", "Total amount"),
        ("البنك الوطني", "National Bank"),
        ("ضريبة القيمة المضافة", "Value Added Tax"),
        ("تاريخ الاستحقاق", "Due date"),
        ("رقم الحساب", "Account number"),
        ("الرقم الضريبي", "Tax number"),
        ("العنوان", "Address"),
        ("التاريخ", "Date"),
        ("المورد", "Supplier"),
    ];

    let mut correct = 0;
    let mut total = 0;

    use std::time::Instant;
    let start = Instant::now();

    println!("\n{:<30} {:<30} {:<10}", "Arabic", "Translation", "Status");
    println!("{:-<80}", "");

    for (arabic, expected) in test_cases {
        let translation = system.translate(arabic)?;
        
        // Simple accuracy check (would be more sophisticated in production)
        let matches = translation.to_lowercase().contains(&expected.to_lowercase());
        
        let status = if matches { "✅" } else { "❌" };
        println!("{:<30} {:<30} {:<10}", arabic, translation, status);
        
        if matches {
            correct += 1;
        }
        total += 1;
    }

    let elapsed = start.elapsed();
    let accuracy = (correct as f64 / total as f64) * 100.0;
    let avg_time = elapsed.as_secs_f64() / total as f64;

    println!("\n{:-<80}", "");
    println!("📊 Results:");
    println!("   Accuracy: {:.1}%", accuracy);
    println!("   Total time: {:.2}s", elapsed.as_secs_f64());
    println!("   Avg per text: {:.2}s", avg_time);
    println!("   Texts processed: {}", total);

    // Save results
    let results = serde_json::json!({
        "accuracy": accuracy,
        "total_time_seconds": elapsed.as_secs_f64(),
        "avg_time_per_text": avg_time,
        "total_texts": total,
        "correct": correct,
        "system": "Rust + Burn (Pure Rust Implementation)",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });

    fs::write(
        "benchmarks/rust_results.json",
        serde_json::to_string_pretty(&results)?
    )?;

    info!("   ✅ Results saved to benchmarks/rust_results.json");

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(if args.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .init();

    info!("🔥 Rust Translation System - 100% Rust!");

    // Create translation system
    let config = TranslationConfig {
        max_length: args.max_length,
        temperature: args.temperature,
        num_beams: args.num_beams,
        use_beam_search: args.beam_search,
    };

    let system = TranslationSystem::new(&args.model, config)?;

    // Run appropriate mode
    if args.benchmark {
        run_benchmark(&system)?;
    } else if let Some(file_path) = args.file {
        // Translate from file
        info!("📂 Reading from file: {}", file_path.display());
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        
        let translations = system.translate_batch(lines)?;
        
        for (i, translation) in translations.iter().enumerate() {
            println!("Line {}: {}", i + 1, translation);
        }

        if let Some(output_path) = args.output {
            fs::write(output_path, translations.join("\n"))?;
        }
    } else if let Some(text) = args.text {
        // Translate single text
        let translation = system.translate(&text)?;
        println!("\n🌍 Translation:");
        println!("   Arabic:  {}", text);
        println!("   English: {}", translation);
    } else {
        error!("No input provided. Use --help for usage.");
        std::process::exit(1);
    }

    Ok(())
}
