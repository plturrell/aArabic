use anyhow::Result;
use clap::{Parser, Subcommand};
use lean4_api_client::*;
use std::fs;

#[derive(Parser)]
#[command(name = "lean4-cli")]
#[command(about = "Lean4 Client")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:8080", env = "LEAN4_URL")]
    url: String,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Health,
    Check { code: String },
    CheckFile { path: String },
    Verify { code: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = Lean4Client::new(cli.url);

    match cli.command {
        Commands::Health => {
            if client.health_check()? {
                println!("✅ Lean4 is healthy");
            } else {
                println!("❌ Lean4 health check failed");
            }
        }
        Commands::Check { code } => {
            let result = client.check_proof(&code)?;
            if result.success {
                println!("✅ Proof valid");
            } else {
                println!("❌ Proof failed");
            }
            for msg in result.messages {
                println!("   ℹ️ {}", msg);
            }
            for err in result.errors {
                println!("   ❌ {}", err);
            }
        }
        Commands::CheckFile { path } => {
            let code = fs::read_to_string(path)?;
            let result = client.check_proof(&code)?;
            if result.success {
                println!("✅ Proof valid");
            } else {
                println!("❌ Proof failed");
            }
            for msg in result.messages {
                println!("   ℹ️ {}", msg);
            }
            for err in result.errors {
                println!("   ❌ {}", err);
            }
        }
        Commands::Verify { code } => {
            if client.verify(&code)? {
                println!("✅ Verified");
            } else {
                println!("❌ Verification failed");
            }
        }
    }
    Ok(())
}
