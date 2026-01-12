use clap::Parser;
use std::path::PathBuf;
use std::fs::File;
use std::io::Write;
use rand::seq::SliceRandom;
use deepseek_math::data::math_generator::{MathOperationGenerator, MathProblem};
use deepseek_math::data::moirai::MoiraiGenerator;
use deepseek_math::data::lotsa::LotsaLoader;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of samples to generate
    #[arg(short, long, default_value_t = 100)]
    num_samples: usize,

    /// Output file path
    #[arg(short, long, default_value = "data/rust_gen.jsonl")]
    output: PathBuf,

    /// Categories to include
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    categories: Option<Vec<String>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Generating {} samples to {:?}", args.num_samples, args.output);
    
    // Ensure parent dir exists
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let mut file = File::create(&args.output)?;
    let mut math_gen = MathOperationGenerator::new();
    let mut moirai_gen = MoiraiGenerator::new();
    let lotsa_loader = LotsaLoader::new();
    
    let all_cats = vec![
        "arithmetic", "algebra", "geometry", "number_theory", "moirai", "lotsa"
    ];
    let categories = args.categories.unwrap_or(all_cats.iter().map(|s| s.to_string()).collect());
    
    let mut problems: Vec<MathProblem> = Vec::new();
    let mut rng = rand::thread_rng();

    // Determine counts
    // Simple even split for now, or random per sample
    for _ in 0..args.num_samples {
        let cat = categories.choose(&mut rng).unwrap();
        
        let problem = match cat.as_str() {
            "moirai" => Some(moirai_gen.generate("medium")),
            "lotsa" => None, // Batch loaded separately, handled below or simple implementation here
            c => math_gen.generate_problem(c, "medium"),
        };
        
        if let Some(p) = problem {
            problems.push(p);
        }
    }
    
    // Handle LOTSA separately (async batch)
    if categories.contains(&"lotsa".to_string()) {
        // Calculate lotsa share roughly or just append some? 
        // Let's say we want 20% lotsa if requested
        let lotsa_count = std::cmp::max(1, args.num_samples / 5);
        println!("Fetching {} LOTSA samples...", lotsa_count);
        match lotsa_loader.load_samples(lotsa_count).await {
            Ok(lotsa_probs) => problems.extend(lotsa_probs),
            Err(e) => eprintln!("Failed to load LOTSA: {}", e),
        }
    }
    
    // Shuffle and save
    problems.shuffle(&mut rng);
    
    // Trim to requested size if we over-generated logic
    if problems.len() > args.num_samples {
        problems.truncate(args.num_samples);
    }
    
    for p in problems {
        let json = serde_json::to_string(&p)?;
        writeln!(file, "{}", json)?;
    }
    
    println!("Done. output saved to {:?}", args.output);
    
    Ok(())
}
