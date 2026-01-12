use anyhow::Result;
use clap::{Parser, Subcommand};
use kafka_api_client::*;

#[derive(Parser)]
#[command(name = "kafka-cli")]
#[command(about = "Apache Kafka Client\n\nMessage streaming and event processing")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "localhost:9092", env = "KAFKA_BROKERS")]
    brokers: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Topic operations
    CreateTopic { name: String, partitions: i32, replication: i32 },
    DeleteTopic { name: String },
    ListTopics,
    GetOffsets { topic: String },
    
    // Producer operations
    Send { topic: String, payload: String, #[arg(long)] key: Option<String> },
    
    // Consumer operations
    Consume { topic: String, #[arg(default_value = "default-group")] group_id: String, #[arg(default_value = "10")] count: usize },
    
    // Group operations
    ListGroups,
    
    // Cluster operations
    ClusterInfo,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = KafkaClient::new(cli.brokers);

    match cli.command {
        Commands::CreateTopic { name, partitions, replication } => {
            client.create_topic(&name, partitions, replication).await?;
            println!("✅ Created topic: {} (partitions: {}, replication: {})", 
                name, partitions, replication);
        }

        Commands::DeleteTopic { name } => {
            client.delete_topic(&name).await?;
            println!("✅ Deleted topic: {}", name);
        }

        Commands::ListTopics => {
            let topics = client.list_topics()?;
            println!("📋 Topics ({}):", topics.len());
            for topic in topics {
                println!("   • {}", topic);
            }
        }

        Commands::GetOffsets { topic } => {
            let offsets = client.get_offsets(&topic)?;
            println!("📊 Offsets for {}:", topic);
            for (partition, low, high) in offsets {
                println!("   Partition {}: {} → {} ({} messages)", 
                    partition, low, high, high - low);
            }
        }

        Commands::Send { topic, payload, key } => {
            let (partition, offset) = client.send_message(&topic, key.as_deref(), &payload).await?;
            println!("✅ Sent to partition {} at offset {}", partition, offset);
        }

        Commands::Consume { topic, group_id, count } => {
            println!("🔄 Consuming {} messages from {}...", count, topic);
            let messages = client.consume_messages(&topic, &group_id, count).await?;
            println!("📨 Received {} messages:", messages.len());
            for msg in messages {
                println!("   [P{} @{}] {}", msg.partition, msg.offset, msg.payload);
            }
        }

        Commands::ListGroups => {
            let groups = client.list_consumer_groups()?;
            println!("👥 Consumer Groups ({}):", groups.len());
            for group in groups {
                println!("   • {}", group);
            }
        }

        Commands::ClusterInfo => {
            let brokers = client.get_cluster_info()?;
            println!("🖥️  Cluster Brokers ({}):", brokers.len());
            for broker in brokers {
                println!("   • {}", broker);
            }
        }
    }

    Ok(())
}
