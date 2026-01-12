use anyhow::Result;
use clap::{Parser, Subcommand};
use filesystem_api_client::*;

#[derive(Parser)]
#[command(name = "fs-cli")]
#[command(about = "Filesystem Client\n\nFile operations")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = ".", env = "FS_BASE_PATH")]
    base: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Read { path: String },
    Write { path: String, content: String },
    Delete { path: String },
    List { path: String },
    Copy { from: String, to: String },
    Rename { from: String, to: String },
    Exists { path: String },
    Mkdir { path: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = FilesystemClient::new(cli.base);

    match cli.command {
        Commands::Read { path } => {
            let content = client.read(&path)?;
            println!("{}", content);
        }
        Commands::Write { path, content } => {
            client.write(&path, &content)?;
            println!("✅ Written: {}", path);
        }
        Commands::Delete { path } => {
            client.delete(&path)?;
            println!("✅ Deleted: {}", path);
        }
        Commands::List { path } => {
            let files = client.list(&path)?;
            println!("📁 Files ({}):", files.len());
            for f in files {
                let icon = if f.is_dir { "📁" } else { "📄" };
                println!("   {} {}", icon, f.path);
            }
        }
        Commands::Copy { from, to } => {
            client.copy(&from, &to)?;
            println!("✅ Copied: {} → {}", from, to);
        }
        Commands::Rename { from, to } => {
            client.rename(&from, &to)?;
            println!("✅ Renamed: {} → {}", from, to);
        }
        Commands::Exists { path } => {
            if client.exists(&path) {
                println!("✅ Exists: {}", path);
            } else {
                println!("❌ Not found: {}", path);
            }
        }
        Commands::Mkdir { path } => {
            client.create_dir(&path)?;
            println!("✅ Created: {}", path);
        }
    }

    Ok(())
}
