use anyhow::Result;
use clap::{Parser, Subcommand};
use memory_api_client::*;

#[derive(Parser)]
#[command(name = "memory-cli")]
#[command(about = "Memory Client")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Get { key: String },
    Set { key: String, value: String },
    SetTtl { key: String, value: String, ttl: u64 },
    Delete { key: String },
    Exists { key: String },
    Keys,
    Clear,
    Len,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = MemoryClient::new();

    match cli.command {
        Commands::Get { key } => {
            if let Some(value) = client.get(&key)? {
                println!("{}", value);
            } else {
                println!("❌ Key not found");
            }
        }
        Commands::Set { key, value } => {
            client.set(&key, &value)?;
            println!("✅ Set: {} = {}", key, value);
        }
        Commands::SetTtl { key, value, ttl } => {
            client.set_with_ttl(&key, &value, ttl)?;
            println!("✅ Set: {} = {} (TTL: {}s)", key, value, ttl);
        }
        Commands::Delete { key } => {
            if client.delete(&key)? {
                println!("✅ Deleted: {}", key);
            } else {
                println!("❌ Key not found");
            }
        }
        Commands::Exists { key } => {
            if client.exists(&key)? {
                println!("✅ Key exists: {}", key);
            } else {
                println!("❌ Key not found: {}", key);
            }
        }
        Commands::Keys => {
            let keys = client.keys()?;
            println!("🔑 Keys ({}):", keys.len());
            for k in keys {
                println!("   • {}", k);
            }
        }
        Commands::Clear => {
            client.clear()?;
            println!("✅ Cleared all keys");
        }
        Commands::Len => {
            let len = client.len()?;
            println!("📊 Total keys: {}", len);
        }
    }
    Ok(())
}
