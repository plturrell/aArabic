use anyhow::Result;
use clap::{Parser, Subcommand};
use markitdown_api_client::*;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "markitdown-cli")]
#[command(about = "MarkItDown - Convert documents to Markdown\n\nPowered by Replicate AI")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, env = "REPLICATE_API_TOKEN")]
    token: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert document from URL
    ConvertUrl {
        #[arg(short, long)]
        url: String,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Convert document from file
    ConvertFile {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Convert PDF to Markdown
    ConvertPdf {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Convert Word document to Markdown
    ConvertDocx {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Convert PowerPoint to Markdown
    ConvertPptx {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Convert Excel to Markdown
    ConvertXlsx {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Convert image to Markdown (OCR)
    ConvertImage {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Convert HTML to Markdown
    ConvertHtml {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Convert multiple files in batch
    ConvertBatch {
        #[arg(short, long, value_delimiter = ',')]
        files: Vec<PathBuf>,
        #[arg(short, long)]
        output_dir: PathBuf,
    },

    /// Get prediction status
    GetPrediction {
        #[arg(short, long)]
        id: String,
    },

    /// List recent predictions
    ListPredictions,

    /// Cancel prediction
    CancelPrediction {
        #[arg(short, long)]
        id: String,
    },

    /// Get model information
    ModelInfo,

    /// Get account information
    AccountInfo,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = MarkItDownClient::new(cli.token.clone());

    match cli.command {
        Commands::ConvertUrl { url, output } => {
            println!("🔄 Converting from URL: {}", url);
            let result = client.convert_from_url(&url)?;
            
            if let Some(out) = output {
                fs::write(&out, &result.markdown)?;
                println!("✅ Converted to: {:?}", out);
            } else {
                println!("\n📝 Markdown Output:\n");
                println!("{}", result.markdown);
            }
        }

        Commands::ConvertFile { file, output } => {
            println!("🔄 Converting file: {:?}", file);
            let file_str = file.to_str().unwrap();
            let result = client.convert_from_file(file_str)?;
            
            if let Some(out) = output {
                fs::write(&out, &result.markdown)?;
                println!("✅ Converted to: {:?}", out);
            } else {
                println!("\n📝 Markdown Output:\n");
                println!("{}", result.markdown);
            }
        }

        Commands::ConvertPdf { file, output } => {
            println!("📄 Converting PDF: {:?}", file);
            let file_str = file.to_str().unwrap();
            let result = client.convert_pdf(file_str)?;
            
            if let Some(out) = output {
                fs::write(&out, &result.markdown)?;
                println!("✅ Converted to: {:?}", out);
            } else {
                println!("\n📝 Markdown Output:\n");
                println!("{}", result.markdown);
            }
        }

        Commands::ConvertDocx { file, output } => {
            println!("📝 Converting Word: {:?}", file);
            let file_str = file.to_str().unwrap();
            let result = client.convert_docx(file_str)?;
            
            if let Some(out) = output {
                fs::write(&out, &result.markdown)?;
                println!("✅ Converted to: {:?}", out);
            } else {
                println!("\n📝 Markdown Output:\n");
                println!("{}", result.markdown);
            }
        }

        Commands::ConvertPptx { file, output } => {
            println!("📊 Converting PowerPoint: {:?}", file);
            let file_str = file.to_str().unwrap();
            let result = client.convert_pptx(file_str)?;
            
            if let Some(out) = output {
                fs::write(&out, &result.markdown)?;
                println!("✅ Converted to: {:?}", out);
            } else {
                println!("\n📝 Markdown Output:\n");
                println!("{}", result.markdown);
            }
        }

        Commands::ConvertXlsx { file, output } => {
            println!("📊 Converting Excel: {:?}", file);
            let file_str = file.to_str().unwrap();
            let result = client.convert_xlsx(file_str)?;
            
            if let Some(out) = output {
                fs::write(&out, &result.markdown)?;
                println!("✅ Converted to: {:?}", out);
            } else {
                println!("\n📝 Markdown Output:\n");
                println!("{}", result.markdown);
            }
        }

        Commands::ConvertImage { file, output } => {
            println!("🖼️  Converting Image (OCR): {:?}", file);
            let file_str = file.to_str().unwrap();
            let result = client.convert_image(file_str)?;
            
            if let Some(out) = output {
                fs::write(&out, &result.markdown)?;
                println!("✅ Converted to: {:?}", out);
            } else {
                println!("\n📝 Markdown Output:\n");
                println!("{}", result.markdown);
            }
        }

        Commands::ConvertHtml { file, output } => {
            println!("🌐 Converting HTML: {:?}", file);
            let file_str = file.to_str().unwrap();
            let result = client.convert_html(file_str)?;
            
            if let Some(out) = output {
                fs::write(&out, &result.markdown)?;
                println!("✅ Converted to: {:?}", out);
            } else {
                println!("\n📝 Markdown Output:\n");
                println!("{}", result.markdown);
            }
        }

        Commands::ConvertBatch { files, output_dir } => {
            fs::create_dir_all(&output_dir)?;
            println!("📦 Converting {} files...", files.len());
            
            for file in files {
                let file_str = file.to_str().unwrap();
                let filename = file.file_stem().unwrap().to_str().unwrap();
                let output_file = output_dir.join(format!("{}.md", filename));
                
                match client.convert_from_file(file_str) {
                    Ok(result) => {
                        fs::write(&output_file, &result.markdown)?;
                        println!("✅ {:?} → {:?}", file, output_file);
                    }
                    Err(e) => {
                        println!("❌ {:?} - Error: {}", file, e);
                    }
                }
            }
            println!("\n🎉 Batch conversion complete!");
        }

        Commands::GetPrediction { id } => {
            let prediction = client.get_prediction(&id)?;
            println!("📊 Prediction Status:");
            println!("   ID: {}", prediction.id);
            println!("   Status: {}", prediction.status);
            if let Some(output) = prediction.output {
                println!("   Output: {:#?}", output);
            }
            if let Some(error) = prediction.error {
                println!("   Error: {}", error);
            }
        }

        Commands::ListPredictions => {
            let predictions = client.list_predictions()?;
            println!("📋 Recent Predictions ({} total):", predictions.len());
            for pred in predictions {
                println!("   • {} [{}]", pred.id, pred.status);
            }
        }

        Commands::CancelPrediction { id } => {
            let prediction = client.cancel_prediction(&id)?;
            println!("🚫 Cancelled prediction: {}", prediction.id);
        }

        Commands::ModelInfo => {
            let info = client.get_model_info()?;
            println!("📦 Model Information:");
            println!("{:#?}", info);
        }

        Commands::AccountInfo => {
            let info = client.get_account_info()?;
            println!("👤 Account Information:");
            println!("{:#?}", info);
        }
    }

    Ok(())
}
