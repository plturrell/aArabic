use anyhow::Result;
use clap::{Parser, Subcommand};
use glean_api_client::*;

#[derive(Parser)]
#[command(name = "glean-cli")]
#[command(about = "Glean Code Intelligence Client\n\nQuery and index codebases with Glean")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:8080", env = "GLEAN_URL")]
    base_url: String,

    #[arg(short = 'k', long, env = "GLEAN_API_KEY")]
    api_key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Get Glean version
    Version,
    
    /// Health check
    Health,

    // ========================================================================
    // DATABASES
    // ========================================================================
    /// List databases
    ListDatabases,
    
    /// Get database info
    GetDatabase {
        #[arg(short, long)]
        name: String,
    },
    
    /// Create database
    CreateDatabase {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        repo: String,
    },
    
    /// Delete database
    DeleteDatabase {
        #[arg(short, long)]
        name: String,
    },

    // ========================================================================
    // QUERY
    // ========================================================================
    /// Execute Angle query
    Query {
        #[arg(short, long)]
        query: String,
        #[arg(short, long)]
        repo: Option<String>,
        #[arg(short, long)]
        limit: Option<i32>,
    },
    
    /// Query for definitions
    QueryDefinitions {
        #[arg(short, long)]
        symbol: String,
        #[arg(short, long)]
        repo: Option<String>,
    },
    
    /// Query for references
    QueryReferences {
        #[arg(short, long)]
        symbol: String,
        #[arg(short, long)]
        repo: Option<String>,
    },
    
    /// Find symbols in file
    FindSymbols {
        #[arg(short, long)]
        file: String,
        #[arg(short, long)]
        repo: Option<String>,
    },
    
    /// Search code
    SearchCode {
        #[arg(short, long)]
        pattern: String,
        #[arg(short, long)]
        repo: Option<String>,
        #[arg(short = 'f', long)]
        file_pattern: Option<String>,
    },

    // ========================================================================
    // INDEXING
    // ========================================================================
    /// Index Rust codebase
    IndexRust {
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        path: String,
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// Index Go codebase
    IndexGo {
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        path: String,
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// Index Python codebase
    IndexPython {
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        path: String,
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// Get indexing status
    IndexStatus {
        #[arg(short, long)]
        repo: String,
    },

    // ========================================================================
    // CODE NAVIGATION
    // ========================================================================
    /// Go to definition
    GotoDefinition {
        #[arg(short, long)]
        file: String,
        #[arg(short, long)]
        line: i32,
        #[arg(short, long)]
        column: i32,
        #[arg(short, long)]
        repo: Option<String>,
    },
    
    /// Find references
    FindReferences {
        #[arg(short, long)]
        file: String,
        #[arg(short, long)]
        line: i32,
        #[arg(short, long)]
        column: i32,
        #[arg(short, long)]
        repo: Option<String>,
    },
    
    /// Get hover info
    Hover {
        #[arg(short, long)]
        file: String,
        #[arg(short, long)]
        line: i32,
        #[arg(short, long)]
        column: i32,
        #[arg(short, long)]
        repo: Option<String>,
    },

    // ========================================================================
    // STATISTICS
    // ========================================================================
    /// Get repo statistics
    RepoStats {
        #[arg(short, long)]
        repo: String,
    },
    
    /// Get database statistics
    DbStats {
        #[arg(short, long)]
        database: String,
    },
    
    /// Get query statistics
    QueryStats,

    // ========================================================================
    // SHELL
    // ========================================================================
    /// Execute shell command
    ShellExecute {
        #[arg(short, long)]
        command: String,
    },
    
    /// Get shell history
    ShellHistory,
    
    /// Clear shell history
    ShellClear,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = GleanClient::new(cli.base_url.clone(), cli.api_key);

    match cli.command {
        Commands::Version => {
            let version = client.get_version()?;
            println!("📦 Glean Version:");
            println!("{:#?}", version);
        }

        Commands::Health => {
            let health = client.health_check()?;
            println!("💚 Health Status:");
            println!("{:#?}", health);
        }

        Commands::ListDatabases => {
            let databases = client.list_databases()?;
            println!("🗄️  Databases ({} total):", databases.len());
            for db in databases {
                println!("   • {} ({})", db.name, db.repo);
                if let Some(status) = db.status {
                    println!("     Status: {}", status);
                }
            }
        }

        Commands::GetDatabase { name } => {
            let db = client.get_database(&name)?;
            println!("{:#?}", db);
        }

        Commands::CreateDatabase { name, repo } => {
            let db = Database {
                name: name.clone(),
                repo,
                version: None,
                status: None,
            };
            let result = client.create_database(&db)?;
            println!("✅ Created database: {}", result.name);
        }

        Commands::DeleteDatabase { name } => {
            client.delete_database(&name)?;
            println!("✅ Deleted database: {}", name);
        }

        Commands::Query { query, repo, limit } => {
            let q = Query {
                query,
                repo,
                file: None,
                recursive: Some(true),
                limit,
            };
            let result = client.query(&q)?;
            println!("🔍 Query Results ({} found):", result.results.len());
            for (i, res) in result.results.iter().enumerate() {
                println!("   {}. {:#?}", i + 1, res);
            }
            if let Some(stats) = result.stats {
                println!("\n📊 Stats:");
                println!("   Elapsed: {}ms", stats.elapsed_ms.unwrap_or(0));
                println!("   Facts searched: {}", stats.facts_searched.unwrap_or(0));
            }
        }

        Commands::QueryDefinitions { symbol, repo } => {
            let defs = client.query_definitions(&symbol, repo.as_deref())?;
            println!("📍 Definitions ({} found):", defs.len());
            for def in defs {
                println!("   • {} at {}:{}:{}", 
                    def.symbol,
                    def.location.file,
                    def.location.line,
                    def.location.column
                );
                if let Some(sig) = def.signature {
                    println!("     {}", sig);
                }
            }
        }

        Commands::QueryReferences { symbol, repo } => {
            let refs = client.query_references(&symbol, repo.as_deref())?;
            println!("🔗 References ({} found):", refs.len());
            for r in refs {
                println!("   • {} at {}:{}:{}", 
                    r.symbol,
                    r.location.file,
                    r.location.line,
                    r.location.column
                );
            }
        }

        Commands::FindSymbols { file, repo } => {
            let symbols = client.find_symbols(&file, repo.as_deref())?;
            println!("🔤 Symbols in {} ({} found):", file, symbols.len());
            for sym in symbols {
                println!("   • {} ({})", sym.name, sym.kind);
                if let Some(loc) = sym.location {
                    println!("     at {}:{}", loc.line, loc.column);
                }
            }
        }

        Commands::SearchCode { pattern, repo, file_pattern } => {
            let results = client.search_code(&pattern, repo.as_deref(), file_pattern.as_deref())?;
            println!("🔎 Search Results ({} matches):", results.len());
            for (i, res) in results.iter().enumerate() {
                println!("   {}. {:#?}", i + 1, res);
            }
        }

        Commands::IndexRust { repo, path, output } => {
            let req = IndexRequest {
                repo: repo.clone(),
                indexer: "lsif-rust".to_string(),
                input_path: path,
                output_db: output,
            };
            let result = client.index_rust(&req)?;
            if result.success {
                println!("✅ Indexed Rust codebase: {}", repo);
                if let Some(db) = result.database {
                    println!("   Database: {}", db);
                }
                if let Some(stats) = result.stats {
                    println!("   Files: {}", stats.files_indexed.unwrap_or(0));
                    println!("   Facts: {}", stats.facts_generated.unwrap_or(0));
                    println!("   Time: {}ms", stats.elapsed_ms.unwrap_or(0));
                }
            } else {
                println!("❌ Indexing failed");
            }
        }

        Commands::IndexGo { repo, path, output } => {
            let req = IndexRequest {
                repo: repo.clone(),
                indexer: "lsif-go".to_string(),
                input_path: path,
                output_db: output,
            };
            let result = client.index_go(&req)?;
            if result.success {
                println!("✅ Indexed Go codebase: {}", repo);
            } else {
                println!("❌ Indexing failed");
            }
        }

        Commands::IndexPython { repo, path, output } => {
            let req = IndexRequest {
                repo: repo.clone(),
                indexer: "scip-python".to_string(),
                input_path: path,
                output_db: output,
            };
            let result = client.index_python(&req)?;
            if result.success {
                println!("✅ Indexed Python codebase: {}", repo);
            } else {
                println!("❌ Indexing failed");
            }
        }

        Commands::IndexStatus { repo } => {
            let status = client.get_index_status(&repo)?;
            println!("📊 Indexing Status for {}:", repo);
            println!("{:#?}", status);
        }

        Commands::GotoDefinition { file, line, column, repo } => {
            let locations = client.goto_definition(&file, line, column, repo.as_deref())?;
            println!("📍 Definitions ({} found):", locations.len());
            for loc in locations {
                println!("   • {}:{}:{}", loc.file, loc.line, loc.column);
            }
        }

        Commands::FindReferences { file, line, column, repo } => {
            let locations = client.find_references(&file, line, column, repo.as_deref())?;
            println!("🔗 References ({} found):", locations.len());
            for loc in locations {
                println!("   • {}:{}:{}", loc.file, loc.line, loc.column);
            }
        }

        Commands::Hover { file, line, column, repo } => {
            let info = client.get_hover(&file, line, column, repo.as_deref())?;
            println!("💡 Hover Info:");
            println!("{:#?}", info);
        }

        Commands::RepoStats { repo } => {
            let stats = client.get_repo_stats(&repo)?;
            println!("📈 Repository Statistics for {}:", repo);
            println!("{:#?}", stats);
        }

        Commands::DbStats { database } => {
            let stats = client.get_db_stats(&database)?;
            println!("📈 Database Statistics for {}:", database);
            println!("{:#?}", stats);
        }

        Commands::QueryStats => {
            let stats = client.get_query_stats()?;
            println!("📈 Query Performance Statistics:");
            println!("{:#?}", stats);
        }

        Commands::ShellExecute { command } => {
            let result = client.shell_execute(&command)?;
            println!("🐚 Shell Result:");
            println!("{:#?}", result);
        }

        Commands::ShellHistory => {
            let history = client.shell_history()?;
            println!("📜 Shell History ({} entries):", history.len());
            for (i, cmd) in history.iter().enumerate() {
                println!("   {}. {}", i + 1, cmd);
            }
        }

        Commands::ShellClear => {
            client.shell_clear_history()?;
            println!("✅ Cleared shell history");
        }
    }

    Ok(())
}
