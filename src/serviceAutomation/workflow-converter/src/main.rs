mod models;
mod n8n_to_langflow;
mod langflow_to_n8n;

use clap::{Parser, Subcommand};
use anyhow::{Result, Context};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "workflow-converter")]
#[command(about = "Convert workflows between n8n and Langflow", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert n8n workflow to Langflow flow
    N8nToLangflow {
        /// Input n8n workflow JSON file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output Langflow flow JSON file
        #[arg(short, long)]
        output: PathBuf,
    },
    
    /// Convert Langflow flow to n8n workflow
    LangflowToN8n {
        /// Input Langflow flow JSON file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output n8n workflow JSON file
        #[arg(short, long)]
        output: PathBuf,
    },
    
    /// Convert with auto-detection
    Auto {
        /// Input workflow file (n8n or Langflow)
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output workflow file
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::N8nToLangflow { input, output } => {
            println!("🔄 Converting n8n → Langflow");
            println!("   Input: {}", input.display());
            println!("   Output: {}", output.display());
            
            let n8n_workflow = std::fs::read_to_string(&input)
                .context("Failed to read n8n workflow file")?;
            
            let langflow_flow = n8n_to_langflow::convert(&n8n_workflow)
                .context("Failed to convert n8n workflow to Langflow")?;
            
            std::fs::write(&output, langflow_flow)
                .context("Failed to write Langflow flow file")?;
            
            println!("✅ Conversion complete!");
        }
        
        Commands::LangflowToN8n { input, output } => {
            println!("🔄 Converting Langflow → n8n");
            println!("   Input: {}", input.display());
            println!("   Output: {}", output.display());
            
            let langflow_flow = std::fs::read_to_string(&input)
                .context("Failed to read Langflow flow file")?;
            
            let n8n_workflow = langflow_to_n8n::convert(&langflow_flow)
                .context("Failed to convert Langflow flow to n8n")?;
            
            std::fs::write(&output, n8n_workflow)
                .context("Failed to write n8n workflow file")?;
            
            println!("✅ Conversion complete!");
        }
        
        Commands::Auto { input, output } => {
            println!("🔍 Auto-detecting workflow format...");
            
            let content = std::fs::read_to_string(&input)
                .context("Failed to read input file")?;
            
            // Try to detect format
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) {
                if value.get("nodes").is_some() && value.get("connections").is_some() {
                    // Looks like n8n
                    println!("   Detected: n8n workflow");
                    println!("   Converting: n8n → Langflow");
                    
                    let langflow_flow = n8n_to_langflow::convert(&content)?;
                    std::fs::write(&output, langflow_flow)?;
                } else if value.get("data").is_some() {
                    // Looks like Langflow
                    println!("   Detected: Langflow flow");
                    println!("   Converting: Langflow → n8n");
                    
                    let n8n_workflow = langflow_to_n8n::convert(&content)?;
                    std::fs::write(&output, n8n_workflow)?;
                } else {
                    anyhow::bail!("Unable to detect workflow format");
                }
                
                println!("✅ Conversion complete!");
            } else {
                anyhow::bail!("Invalid JSON format");
            }
        }
    }
    
    Ok(())
}
