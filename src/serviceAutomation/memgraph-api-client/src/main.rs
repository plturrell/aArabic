//! Memgraph CLI
//!
//! Command-line interface for interacting with Memgraph graph database.

use anyhow::Result;
use clap::{Parser, Subcommand};
use memgraph_api_client::MemgraphClient;
use serde_json::json;

#[derive(Parser)]
#[command(name = "aimo-memgraph-cli")]
#[command(about = "Memgraph API client CLI", long_about = None)]
struct Cli {
    /// Memgraph base URL
    #[arg(short, long, default_value = "http://localhost:7687")]
    url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a Cypher query
    Query {
        /// Cypher query to execute
        #[arg(short, long)]
        query: String,

        /// Query parameters as JSON
        #[arg(short, long)]
        params: Option<String>,
    },

    /// Get database schema
    Schema,

    /// Get node count
    NodeCount,

    /// Get edge count
    EdgeCount,

    /// Create a node
    CreateNode {
        /// Node labels (comma-separated)
        #[arg(short, long)]
        labels: String,

        /// Node properties as JSON
        #[arg(short, long)]
        properties: String,
    },

    /// Create an edge
    CreateEdge {
        /// Source node ID
        #[arg(long)]
        from: String,

        /// Target node ID
        #[arg(long)]
        to: String,

        /// Edge type
        #[arg(short = 't', long)]
        edge_type: String,

        /// Edge properties as JSON (optional)
        #[arg(short, long)]
        properties: Option<String>,
    },

    /// Clear database (use with caution!)
    Clear {
        /// Confirm deletion
        #[arg(long)]
        confirm: bool,
    },

    /// Health check
    Health,

    /// Test connection
    Test,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = MemgraphClient::new(&cli.url)?;

    match cli.command {
        Commands::Query { query, params } => {
            let parameters = if let Some(p) = params {
                Some(serde_json::from_str(&p)?)
            } else {
                None
            };

            let response = client.execute_query(&query, parameters).await?;
            
            println!("Columns: {:?}", response.columns);
            println!("Rows returned: {}", response.data.len());
            
            for (i, row) in response.data.iter().enumerate() {
                println!("Row {}: {:?}", i + 1, row);
            }

            if let Some(metadata) = response.metadata {
                if let Some(exec_time) = metadata.execution_time {
                    println!("Execution time: {:.3}ms", exec_time);
                }
            }
        }

        Commands::Schema => {
            let response = client.get_schema().await?;
            println!("Schema:");
            println!("{}", serde_json::to_string_pretty(&response.data)?);
        }

        Commands::NodeCount => {
            let count = client.get_node_count().await?;
            println!("Total nodes: {}", count);
        }

        Commands::EdgeCount => {
            let count = client.get_edge_count().await?;
            println!("Total edges: {}", count);
        }

        Commands::CreateNode { labels, properties } => {
            let labels_vec: Vec<&str> = labels.split(',').map(|s| s.trim()).collect();
            let props: serde_json::Value = serde_json::from_str(&properties)?;
            
            let response = client.create_node(&labels_vec, props).await?;
            println!("Node created successfully");
            println!("Result: {:?}", response.data);
        }

        Commands::CreateEdge {
            from,
            to,
            edge_type,
            properties,
        } => {
            let props = if let Some(p) = properties {
                Some(serde_json::from_str(&p)?)
            } else {
                None
            };

            let response = client.create_edge(&from, &to, &edge_type, props).await?;
            println!("Edge created successfully");
            println!("Result: {:?}", response.data);
        }

        Commands::Clear { confirm } => {
            if !confirm {
                eprintln!("ERROR: Must use --confirm flag to clear database");
                eprintln!("This will delete ALL data!");
                std::process::exit(1);
            }

            println!("Clearing database...");
            client.clear_database().await?;
            println!("Database cleared successfully");
        }

        Commands::Health => {
            let health = client.health_check().await?;
            println!("Status: {}", health.status);
            
            if let Some(version) = health.version {
                println!("Version: {}", version);
            }
            
            if let Some(uptime) = health.uptime {
                println!("Uptime: {} seconds", uptime);
            }
        }

        Commands::Test => {
            print!("Testing connection to {}...", cli.url);
            
            match client.test_connection().await {
                Ok(true) => {
                    println!(" ✓ Connected successfully!");
                }
                Ok(false) | Err(_) => {
                    println!(" ✗ Connection failed!");
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}
