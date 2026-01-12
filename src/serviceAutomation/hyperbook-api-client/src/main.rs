use anyhow::Result;
use clap::{Parser, Subcommand};
use hyperbook_api_client::*;
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "hyperbook-cli")]
#[command(about = "Hyperbook Educational Content Client\n\nCreate and manage interactive educational content")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = ".", env = "HYPERBOOK_PATH")]
    path: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Book operations
    Init { name: String, #[arg(default_value = "en")] language: String },
    List,
    LoadConfig { book: String },
    Stats { book: String },
    Build { book: String },
    Serve { book: String, #[arg(default_value = "3000")] port: u16 },
    
    // Chapter operations
    CreateChapter { book: String, name: String, content: String },
    ReadChapter { book: String, name: String },
    UpdateChapter { book: String, name: String, content: String },
    DeleteChapter { book: String, name: String },
    ListChapters { book: String },
    
    // Element creation helpers
    Alert { alert_type: String, content: String, title: Option<String> },
    Collapsible { title: String, content: String, #[arg(long)] open: bool },
    CodeBlock { language: String, code: String, filename: Option<String> },
    Mermaid { diagram: String },
    Math { formula: String, #[arg(long)] inline: bool },
    
    // Glossary operations
    CreateGlossary { book: String, terms_json: String },
    
    // Search operations
    Search { book: String, query: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = HyperbookClient::new(cli.path);

    match cli.command {
        Commands::Init { name, language } => {
            client.init_book(&name, &language)?;
            println!("✅ Initialized book: {} ({})", name, language);
        }

        Commands::List => {
            let books = client.list_books()?;
            println!("📚 Books ({}):", books.len());
            for book in books {
                println!("   • {}", book);
            }
        }

        Commands::LoadConfig { book } => {
            let config = client.load_config(&book)?;
            println!("📖 Book Config:");
            println!("{:#?}", config);
        }

        Commands::Stats { book } => {
            let stats = client.get_stats(&book)?;
            println!("📊 Book Statistics:");
            for (key, value) in stats {
                println!("   {}: {}", key, value);
            }
        }

        Commands::Build { book } => {
            let result = client.build_book(&book)?;
            println!("🔨 {}", result);
        }

        Commands::Serve { book, port } => {
            let result = client.serve_book(&book, port)?;
            println!("🌐 {}", result);
        }

        Commands::CreateChapter { book, name, content } => {
            client.create_chapter(&book, &name, &content)?;
            println!("✅ Created chapter: {}", name);
        }

        Commands::ReadChapter { book, name } => {
            let content = client.read_chapter(&book, &name)?;
            println!("{}", content);
        }

        Commands::UpdateChapter { book, name, content } => {
            client.update_chapter(&book, &name, &content)?;
            println!("✅ Updated chapter: {}", name);
        }

        Commands::DeleteChapter { book, name } => {
            client.delete_chapter(&book, &name)?;
            println!("✅ Deleted chapter: {}", name);
        }

        Commands::ListChapters { book } => {
            let chapters = client.list_chapters(&book)?;
            println!("📄 Chapters in {} ({}):", book, chapters.len());
            for chapter in chapters {
                println!("   • {}", chapter);
            }
        }

        Commands::Alert { alert_type, content, title } => {
            let element = client.create_alert(&alert_type, title.as_deref(), &content);
            println!("{}", element);
        }

        Commands::Collapsible { title, content, open } => {
            let element = client.create_collapsible(&title, &content, open);
            println!("{}", element);
        }

        Commands::CodeBlock { language, code, filename } => {
            let element = client.create_code_block(&language, &code, filename.as_deref());
            println!("{}", element);
        }

        Commands::Mermaid { diagram } => {
            let element = client.create_mermaid(&diagram);
            println!("{}", element);
        }

        Commands::Math { formula, inline } => {
            let element = client.create_math(&formula, inline);
            println!("{}", element);
        }

        Commands::CreateGlossary { book, terms_json } => {
            let terms: Vec<GlossaryTerm> = serde_json::from_str(&terms_json)?;
            let count = terms.len();
            client.create_glossary(&book, terms)?;
            println!("✅ Created glossary with {} terms", count);
        }

        Commands::Search { book, query } => {
            let results = client.search(&book, &query)?;
            println!("🔍 Search Results ({}):", results.len());
            for (chapter, line) in results {
                println!("   {} → {}", chapter, line);
            }
        }
    }

    Ok(())
}
