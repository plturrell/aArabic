use anyhow::Result;
use clap::{Parser, Subcommand};
use shimmy_api_client::*;

#[derive(Parser)]
#[command(name = "shimmy-cli")]
#[command(about = "Shimmy-AI Client\n\nLocal AI inference with OpenAI-compatible API")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:11435", env = "SHIMMY_URL")]
    url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Health & Status
    Health,
    ServerInfo,
    
    // Model operations
    ListModels,
    GetModel { id: String },
    ModelExists { id: String },
    
    // Chat operations (OpenAI-compatible)
    Chat {
        model: String,
        prompt: String,
        #[arg(long)] system: Option<String>,
        #[arg(long)] temperature: Option<f32>,
    },
    
    // Generate operations (Shimmy native)
    Generate {
        prompt: String,
        #[arg(long)] model: Option<String>,
        #[arg(long)] temperature: Option<f32>,
    },
    
    // Streaming operations (WebSocket)
    Stream {
        prompt: String,
        #[arg(long, default_value = "glm4:9b")] model: String,
        #[arg(long)] temperature: Option<f32>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = ShimmyClient::new(cli.url.clone());

    match cli.command {
        Commands::Health => {
            if client.health_check()? {
                println!("✅ Shimmy is healthy");
            } else {
                println!("❌ Shimmy health check failed");
            }
        }

        Commands::ServerInfo => {
            let info = client.get_server_info()?;
            println!("🖥️  Server Info:");
            for (key, value) in info {
                println!("   {}: {}", key, value);
            }
        }

        Commands::ListModels => {
            let models = client.list_models()?;
            println!("🤖 Models ({}):", models.len());
            for model in models {
                println!("   • {} ({})", model.id, model.owned_by);
            }
        }

        Commands::GetModel { id } => {
            let model = client.get_model(&id)?;
            println!("🤖 Model: {}", model.id);
            println!("   Object: {}", model.object);
            println!("   Owner: {}", model.owned_by);
            println!("   Created: {}", model.created);
        }

        Commands::ModelExists { id } => {
            if client.model_exists(&id)? {
                println!("✅ Model exists: {}", id);
            } else {
                println!("❌ Model not found: {}", id);
            }
        }

        Commands::Chat { model, prompt, system, temperature } => {
            let mut messages = Vec::new();
            
            if let Some(sys) = system {
                messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: sys,
                });
            }
            
            messages.push(ChatMessage {
                role: "user".to_string(),
                content: prompt,
            });
            
            println!("💬 Generating response...");
            let response = client.chat(&model, messages, temperature)?;
            println!("\n{}", response);
        }

        Commands::Generate { prompt, model, temperature } => {
            let request = GenerateRequest {
                prompt,
                model,
                temperature,
                max_tokens: None,
            };
            
            println!("⚡ Generating...");
            let response = client.generate(&request)?;
            println!("\n{}", response.response);
            
            if let Some(m) = response.model {
                println!("\n[Model: {}]", m);
            }
        }

        Commands::Stream { prompt, model, temperature } => {
            println!("🌊 Streaming response...\n");
            
            let chunks = stream_chat(&cli.url, &model, &prompt, temperature).await?;
            
            for chunk in chunks {
                print!("{}", chunk);
                use std::io::Write;
                std::io::stdout().flush()?;
            }
            
            println!("\n\n✅ Stream complete");
        }
    }

    Ok(())
}
