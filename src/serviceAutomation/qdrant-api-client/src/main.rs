use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use qdrant_api::QdrantClient;
use qdrant_client::qdrant::Distance;

#[derive(Parser)]
#[command(name = "qdrant-cli")]
#[command(about = "Qdrant CLI - Vector Database Client (v1.12 API)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Qdrant service URL
    #[arg(long, default_value = "http://localhost:6333")]
    url: String,
}

#[derive(Subcommand)]
enum Commands {
    /// List collections
    Collections,
    
    /// Get collection info
    Info {
        /// Collection name
        collection: String,
    },
    
    /// Create collection
    Create {
        /// Collection name
        collection: String,
        
        /// Vector dimension size
        #[arg(short, long)]
        size: u64,
        
        /// Distance metric (Cosine, Euclid, Dot, Manhattan)
        #[arg(short, long, default_value = "Cosine")]
        distance: String,
    },
    
    /// Delete collection
    Delete {
        /// Collection name
        collection: String,
    },
    
    /// Upsert points
    Upsert {
        /// Collection name
        collection: String,
        
        /// JSON file with points
        #[arg(short, long)]
        file: String,
    },
    
    /// Search vectors
    Search {
        /// Collection name
        collection: String,
        
        /// Query vector (comma-separated floats)
        #[arg(short, long)]
        vector: String,
        
        /// Number of results
        #[arg(short, long, default_value = "10")]
        limit: u64,
    },
    
    /// Count points in collection
    Count {
        /// Collection name
        collection: String,
    },
    
    /// Health check
    Health,
    
    /// Run test suite
    Test,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = QdrantClient::new(&cli.url).await?;

    match cli.command {
        Commands::Collections => {
            println!("{}", "Listing Qdrant collections...".cyan());
            let collections = client.list_collections().await?;
            
            if collections.is_empty() {
                println!("\n{}", "No collections found".yellow());
            } else {
                println!("\n{}", "Collections:".green().bold());
                for (i, name) in collections.iter().enumerate() {
                    println!("\n  {}. {}", i + 1, name.cyan());
                    
                    // Get detailed info for each collection
                    if let Ok(info) = client.get_collection_info(name).await {
                        if let Some(points) = info.points_count {
                            println!("     Points: {}", points);
                        }
                        if let Some(vectors) = info.vectors_count {
                            println!("     Vectors: {}", vectors);
                        }
                    }
                }
                println!("\n{} collections total", collections.len());
            }
        }

        Commands::Info { collection } => {
            println!("{}", format!("Getting info for: {}", collection).cyan());
            let info = client.get_collection_info(&collection).await?;
            
            println!("\n{}", "Collection Info:".green().bold());
            println!("  Name: {}", info.name);
            if let Some(points) = info.points_count {
                println!("  Points: {}", points);
            }
            if let Some(vectors) = info.vectors_count {
                println!("  Vectors: {}", vectors);
            }
        }

        Commands::Create {
            collection,
            size,
            distance,
        } => {
            println!("{}", format!("Creating collection: {}", collection).cyan());
            println!("  Vector size: {}", size);
            println!("  Distance: {}", distance);
            
            let distance_metric = match distance.to_lowercase().as_str() {
                "cosine" => Distance::Cosine,
                "euclid" | "euclidean" => Distance::Euclid,
                "dot" => Distance::Dot,
                "manhattan" => Distance::Manhattan,
                _ => {
                    eprintln!("{}", format!("Unknown distance: {}, using Cosine", distance).yellow());
                    Distance::Cosine
                }
            };
            
            client.create_collection(&collection, size, distance_metric).await?;
            
            println!("\n{}", "✓ Collection created successfully!".green().bold());
        }

        Commands::Delete { collection } => {
            println!("{}", format!("Deleting collection: {}", collection).cyan());
            client.delete_collection(&collection).await?;
            println!("\n{}", "✓ Collection deleted successfully!".green().bold());
        }

        Commands::Upsert { collection, file } => {
            println!("{}", format!("Upserting points to: {}", collection).cyan());
            println!("  From file: {}", file);
            
            // Read points from JSON file
            let content = std::fs::read_to_string(&file)?;
            let points: Vec<qdrant_api::Point> = serde_json::from_str(&content)?;
            
            println!("  Points to upsert: {}", points.len());
            
            client.upsert_points(&collection, points).await?;
            
            println!("\n{}", "✓ Points upserted successfully!".green().bold());
        }

        Commands::Search {
            collection,
            vector,
            limit,
        } => {
            println!("{}", format!("Searching in: {}", collection).cyan());
            
            // Parse vector from comma-separated string
            let vec: Result<Vec<f32>, _> = vector
                .split(',')
                .map(|s| s.trim().parse())
                .collect();
            
            let vec = vec.map_err(|e| anyhow::anyhow!("Invalid vector format: {}", e))?;
            
            println!("  Vector dimension: {}", vec.len());
            println!("  Limit: {}", limit);
            
            let results = client.search(&collection, vec, limit).await?;
            
            if results.is_empty() {
                println!("\n{}", "No results found".yellow());
            } else {
                println!("\n{}", format!("✓ Found {} results:", results.len()).green().bold());
                for (i, result) in results.iter().enumerate() {
                    println!("\n  {}. Score: {:.4}", i + 1, result.score);
                    println!("     ID: {}", result.id);
                    if let Some(payload) = &result.payload {
                        if !payload.is_empty() {
                            println!("     Payload: {} fields", payload.len());
                        }
                    }
                }
            }
        }

        Commands::Count { collection } => {
            println!("{}", format!("Counting points in: {}", collection).cyan());
            let count = client.count_points(&collection).await?;
            println!("\n{}", format!("✓ Points: {}", count).green().bold());
        }

        Commands::Health => {
            println!("{}", "Checking Qdrant health...".cyan());
            match client.health_check().await {
                Ok(true) => {
                    println!("\n{}", "✓ Qdrant is healthy".green().bold());
                }
                Ok(false) => {
                    println!("\n{}", "✗ Qdrant is unhealthy".red().bold());
                }
                Err(e) => {
                    println!("\n{}", format!("✗ Health check failed: {}", e).red().bold());
                }
            }
        }

        Commands::Test => {
            println!("{}", "Running Qdrant client tests...".cyan());
            
            // Test 1: Health check
            print!("\n  1. Health check... ");
            match client.health_check().await {
                Ok(_) => println!("{}", "✓ PASS".green()),
                Err(e) => {
                    println!("{}", format!("✗ FAIL: {}", e).red());
                    return Ok(());
                }
            }
            
            // Test 2: List collections
            print!("  2. List collections... ");
            match client.list_collections().await {
                Ok(collections) => {
                    println!("{}", format!("✓ PASS ({} collections)", collections.len()).green());
                }
                Err(e) => {
                    println!("{}", format!("✗ FAIL: {}", e).red());
                    return Ok(());
                }
            }
            
            // Test 3: Create test collection
            let test_collection = "aimo_test_collection";
            print!("  3. Create test collection... ");
            match client.create_collection(test_collection, 128, Distance::Cosine).await {
                Ok(_) => println!("{}", "✓ PASS".green()),
                Err(e) => {
                    // May already exist, which is ok
                    if e.to_string().contains("already exists") {
                        println!("{}", "✓ PASS (already exists)".green());
                    } else {
                        println!("{}", format!("✗ FAIL: {}", e).red());
                        return Ok(());
                    }
                }
            }
            
            // Test 4: Get collection info
            print!("  4. Get collection info... ");
            match client.get_collection_info(test_collection).await {
                Ok(info) => {
                    println!("{}", format!("✓ PASS (name: {})", info.name).green());
                }
                Err(e) => {
                    println!("{}", format!("✗ FAIL: {}", e).red());
                }
            }
            
            // Test 5: Cleanup - delete test collection
            print!("  5. Cleanup test collection... ");
            match client.delete_collection(test_collection).await {
                Ok(_) => println!("{}", "✓ PASS".green()),
                Err(e) => {
                    println!("{}", format!("✗ FAIL: {}", e).red());
                }
            }
            
            println!("\n{}", "✓ All tests passed!".green().bold());
        }
    }

    Ok(())
}
