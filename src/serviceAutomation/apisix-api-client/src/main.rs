use anyhow::Result;
use clap::{Parser, Subcommand};
use apisix_api_client::*;

#[derive(Parser)]
#[command(name = "apisix-cli")]
#[command(about = "Apache APISIX Client\n\nAPI Gateway management")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:9180", env = "APISIX_URL")]
    url: String,

    #[arg(short, long, env = "APISIX_API_KEY")]
    api_key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Health
    Health,
    
    // Route operations
    ListRoutes,
    GetRoute { id: String },
    DeleteRoute { id: String },
    
    // Service operations
    ListServices,
    GetService { id: String },
    DeleteService { id: String },
    
    // Upstream operations
    ListUpstreams,
    GetUpstream { id: String },
    DeleteUpstream { id: String },
    
    // Consumer operations
    ListConsumers,
    GetConsumer { username: String },
    DeleteConsumer { username: String },
    
    // SSL operations
    ListSsl,
    GetSsl { id: String },
    DeleteSsl { id: String },
    
    // Plugin operations
    ListPlugins,
    GetPluginSchema { name: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let client = if let Some(key) = cli.api_key {
        ApisixClient::with_api_key(cli.url, key)
    } else {
        ApisixClient::new(cli.url)
    };

    match cli.command {
        Commands::Health => {
            if client.health_check()? {
                println!("✅ APISIX is healthy");
            } else {
                println!("❌ APISIX health check failed");
            }
        }

        Commands::ListRoutes => {
            let routes = client.list_routes()?;
            println!("🛣️  Routes ({}):", routes.len());
            for route in routes {
                println!("   • {} → {}", route.id.unwrap_or_default(), route.uri);
            }
        }

        Commands::GetRoute { id } => {
            let route = client.get_route(&id)?;
            println!("🛣️  Route: {}", route.uri);
            if let Some(methods) = route.methods {
                println!("   Methods: {:?}", methods);
            }
        }

        Commands::DeleteRoute { id } => {
            client.delete_route(&id)?;
            println!("✅ Deleted route: {}", id);
        }

        Commands::ListServices => {
            let services = client.list_services()?;
            println!("🔧 Services ({}):", services.len());
            for service in services {
                if let Some(name) = service.name {
                    println!("   • {} ({})", name, service.id.unwrap_or_default());
                }
            }
        }

        Commands::GetService { id } => {
            let service = client.get_service(&id)?;
            println!("🔧 Service: {}", service.name.unwrap_or_default());
        }

        Commands::DeleteService { id } => {
            client.delete_service(&id)?;
            println!("✅ Deleted service: {}", id);
        }

        Commands::ListUpstreams => {
            let upstreams = client.list_upstreams()?;
            println!("⬆️  Upstreams ({}):", upstreams.len());
            for upstream in upstreams {
                println!("   • {} nodes", upstream.nodes.len());
            }
        }

        Commands::GetUpstream { id } => {
            let upstream = client.get_upstream(&id)?;
            println!("⬆️  Upstream: {} ({})", upstream.upstream_type, upstream.nodes.len());
        }

        Commands::DeleteUpstream { id } => {
            client.delete_upstream(&id)?;
            println!("✅ Deleted upstream: {}", id);
        }

        Commands::ListConsumers => {
            let consumers = client.list_consumers()?;
            println!("👤 Consumers ({}):", consumers.len());
            for consumer in consumers {
                println!("   • {}", consumer.username);
            }
        }

        Commands::GetConsumer { username } => {
            let consumer = client.get_consumer(&username)?;
            println!("👤 Consumer: {}", consumer.username);
        }

        Commands::DeleteConsumer { username } => {
            client.delete_consumer(&username)?;
            println!("✅ Deleted consumer: {}", username);
        }

        Commands::ListSsl => {
            let ssls = client.list_ssl()?;
            println!("🔒 SSL Certificates ({}):", ssls.len());
            for ssl in ssls {
                println!("   • {} SNI(s)", ssl.snis.len());
            }
        }

        Commands::GetSsl { id } => {
            let ssl = client.get_ssl(&id)?;
            println!("🔒 SSL: {:?}", ssl.snis);
        }

        Commands::DeleteSsl { id } => {
            client.delete_ssl(&id)?;
            println!("✅ Deleted SSL: {}", id);
        }

        Commands::ListPlugins => {
            let plugins = client.list_plugins()?;
            println!("🔌 Available Plugins ({}):", plugins.len());
            for plugin in plugins {
                println!("   • {}", plugin);
            }
        }

        Commands::GetPluginSchema { name } => {
            let schema = client.get_plugin_schema(&name)?;
            println!("🔌 Plugin Schema: {}", name);
            println!("{}", serde_json::to_string_pretty(&schema)?);
        }
    }

    Ok(())
}
