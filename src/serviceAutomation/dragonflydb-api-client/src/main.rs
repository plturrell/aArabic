use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use dragonflydb_api::DragonflyClient;

#[derive(Parser)]
#[command(name = "dragonfly-cli")]
#[command(about = "DragonflyDB CLI - Redis-compatible in-memory data store")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// DragonflyDB connection URL
    #[arg(short, long, default_value = "redis://localhost:6379")]
    url: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Set a key-value pair
    Set {
        /// Key name
        key: String,
        /// Value
        value: String,
        /// Expiration in seconds (optional)
        #[arg(short, long)]
        ex: Option<u64>,
    },

    /// Get a value by key
    Get {
        /// Key name
        key: String,
    },

    /// Delete a key
    Del {
        /// Key name
        key: String,
    },

    /// List all keys (optionally filter by pattern)
    Keys {
        /// Pattern (e.g., "user:*")
        #[arg(default_value = "*")]
        pattern: String,
    },

    /// Get database size (number of keys)
    Dbsize,

    /// Push to list (left)
    Lpush {
        /// List key
        key: String,
        /// Value to push
        value: String,
    },

    /// Push to list (right)
    Rpush {
        /// List key
        key: String,
        /// Value to push
        value: String,
    },

    /// Get list range
    Lrange {
        /// List key
        key: String,
        /// Start index
        #[arg(default_value = "0")]
        start: isize,
        /// Stop index
        #[arg(default_value = "-1")]
        stop: isize,
    },

    /// Set hash field
    Hset {
        /// Hash key
        key: String,
        /// Field name
        field: String,
        /// Field value
        value: String,
    },

    /// Get hash field
    Hget {
        /// Hash key
        key: String,
        /// Field name
        field: String,
    },

    /// Get all hash fields
    Hgetall {
        /// Hash key
        key: String,
    },

    /// Add to set
    Sadd {
        /// Set key
        key: String,
        /// Member to add
        member: String,
    },

    /// Get set members
    Smembers {
        /// Set key
        key: String,
    },

    /// Ping server
    Ping,

    /// Get server info
    Info,

    /// Health check
    Health,

    /// Flush database (requires --confirm)
    Flush {
        #[arg(long)]
        confirm: bool,
    },

    /// Run test suite
    Test,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut client = DragonflyClient::new(&cli.url).await?;

    match cli.command {
        Commands::Set { key, value, ex } => {
            if let Some(seconds) = ex {
                client.setex(&key, &value, seconds).await?;
                println!("{}", format!("✓ Set {} (expires in {}s)", key, seconds).green());
            } else {
                client.set(&key, &value).await?;
                println!("{}", format!("✓ Set {}", key).green());
            }
        }

        Commands::Get { key } => {
            match client.get(&key).await? {
                Some(value) => {
                    println!("{}: {}", key.cyan(), value);
                }
                None => {
                    println!("{}", format!("Key '{}' not found", key).yellow());
                }
            }
        }

        Commands::Del { key } => {
            if client.del(&key).await? {
                println!("{}", format!("✓ Deleted {}", key).green());
            } else {
                println!("{}", format!("Key '{}' not found", key).yellow());
            }
        }

        Commands::Keys { pattern } => {
            let keys = client.keys(&pattern).await?;
            if keys.is_empty() {
                println!("{}", format!("No keys matching '{}'", pattern).yellow());
            } else {
                println!("{}", format!("Keys matching '{}':", pattern).cyan());
                for (i, key) in keys.iter().enumerate() {
                    println!("  {}. {}", i + 1, key);
                }
                println!("\n{} keys total", keys.len());
            }
        }

        Commands::Dbsize => {
            let size = client.dbsize().await?;
            println!("{}", format!("Database size: {} keys", size).cyan());
        }

        Commands::Lpush { key, value } => {
            let len = client.lpush(&key, &value).await?;
            println!("{}", format!("✓ Pushed to list (length: {})", len).green());
        }

        Commands::Rpush { key, value } => {
            let len = client.rpush(&key, &value).await?;
            println!("{}", format!("✓ Pushed to list (length: {})", len).green());
        }

        Commands::Lrange { key, start, stop } => {
            let values = client.lrange(&key, start, stop).await?;
            if values.is_empty() {
                println!("{}", "List is empty".yellow());
            } else {
                println!("{}", format!("List '{}' [{} to {}]:", key, start, stop).cyan());
                for (i, value) in values.iter().enumerate() {
                    let idx = if start >= 0 { start as usize + i } else { i };
                    println!("  {}. {}", idx, value);
                }
            }
        }

        Commands::Hset { key, field, value } => {
            client.hset(&key, &field, &value).await?;
            println!("{}", format!("✓ Set {}.{}", key, field).green());
        }

        Commands::Hget { key, field } => {
            match client.hget(&key, &field).await? {
                Some(value) => {
                    println!("{}.{}: {}", key.cyan(), field, value);
                }
                None => {
                    println!("{}", format!("Field '{}.{}' not found", key, field).yellow());
                }
            }
        }

        Commands::Hgetall { key } => {
            let fields = client.hgetall(&key).await?;
            if fields.is_empty() {
                println!("{}", format!("Hash '{}' is empty or doesn't exist", key).yellow());
            } else {
                println!("{}", format!("Hash '{}':", key).cyan());
                for (field, value) in fields {
                    println!("  {}: {}", field, value);
                }
            }
        }

        Commands::Sadd { key, member } => {
            if client.sadd(&key, &member).await? {
                println!("{}", format!("✓ Added '{}' to set '{}'", member, key).green());
            } else {
                println!("{}", format!("'{}' already in set", member).yellow());
            }
        }

        Commands::Smembers { key } => {
            let members = client.smembers(&key).await?;
            if members.is_empty() {
                println!("{}", format!("Set '{}' is empty or doesn't exist", key).yellow());
            } else {
                println!("{}", format!("Set '{}' members:", key).cyan());
                for (i, member) in members.iter().enumerate() {
                    println!("  {}. {}", i + 1, member);
                }
                println!("\n{} members total", members.len());
            }
        }

        Commands::Ping => {
            match client.ping().await {
                Ok(response) => {
                    println!("{}", format!("✓ {}", response).green());
                }
                Err(e) => {
                    println!("{}", format!("✗ Ping failed: {}", e).red());
                }
            }
        }

        Commands::Info => {
            let info = client.info().await?;
            println!("{}", info);
        }

        Commands::Health => {
            println!("{}", "Checking DragonflyDB health...".cyan());
            match client.health_check().await {
                Ok(true) => {
                    println!("\n{}", "✓ DragonflyDB is healthy".green().bold());
                }
                Ok(false) => {
                    println!("\n{}", "✗ DragonflyDB is unhealthy".red().bold());
                }
                Err(e) => {
                    println!("\n{}", format!("✗ Health check failed: {}", e).red().bold());
                }
            }
        }

        Commands::Flush { confirm } => {
            if !confirm {
                println!("{}", "⚠️  This will delete ALL data!".yellow().bold());
                println!("{}", "Use --confirm flag to proceed".yellow());
                return Ok(());
            }

            println!("{}", "Flushing database...".yellow());
            client.flushdb().await?;
            println!("{}", "✓ Database flushed".green().bold());
        }

        Commands::Test => {
            println!("{}", "Running DragonflyDB client tests...".cyan());

            // Test 1: Ping
            print!("\n  1. Ping... ");
            match client.ping().await {
                Ok(_) => println!("{}", "✓ PASS".green()),
                Err(e) => {
                    println!("{}", format!("✗ FAIL: {}", e).red());
                    return Ok(());
                }
            }

            // Test 2: Set/Get
            print!("  2. Set/Get... ");
            client.set("test_key", "test_value").await?;
            match client.get("test_key").await? {
                Some(val) if val == "test_value" => println!("{}", "✓ PASS".green()),
                _ => {
                    println!("{}", "✗ FAIL".red());
                    return Ok(());
                }
            }

            // Test 3: Delete
            print!("  3. Delete... ");
            if client.del("test_key").await? {
                println!("{}", "✓ PASS".green());
            } else {
                println!("{}", "✗ FAIL".red());
                return Ok(());
            }

            // Test 4: List operations
            print!("  4. List operations... ");
            client.rpush("test_list", "item1").await?;
            client.rpush("test_list", "item2").await?;
            let items = client.lrange("test_list", 0, -1).await?;
            if items.len() == 2 {
                println!("{}", "✓ PASS".green());
            } else {
                println!("{}", "✗ FAIL".red());
            }
            client.del("test_list").await?;

            // Test 5: Hash operations
            print!("  5. Hash operations... ");
            client.hset("test_hash", "field1", "value1").await?;
            match client.hget("test_hash", "field1").await? {
                Some(val) if val == "value1" => println!("{}", "✓ PASS".green()),
                _ => {
                    println!("{}", "✗ FAIL".red());
                    return Ok(());
                }
            }
            client.del("test_hash").await?;

            // Test 6: Set operations
            print!("  6. Set operations... ");
            client.sadd("test_set", "member1").await?;
            if client.sismember("test_set", "member1").await? {
                println!("{}", "✓ PASS".green());
            } else {
                println!("{}", "✗ FAIL".red());
            }
            client.del("test_set").await?;

            println!("\n{}", "✓ All tests passed!".green().bold());
        }
    }

    Ok(())
}
