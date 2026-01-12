use anyhow::Result;
use clap::{Parser, Subcommand};
use marquez_api_client::*;

#[derive(Parser)]
#[command(name = "marquez-cli")]
#[command(about = "Marquez Data Lineage Client\n\nMetadata service for data lineage tracking")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:5000", env = "MARQUEZ_URL")]
    url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Namespace commands
    CreateNamespace { name: String, owner: Option<String>, description: Option<String> },
    ListNamespaces,
    GetNamespace { name: String },
    
    // Dataset commands
    CreateDataset { namespace: String, name: String, r#type: String, source: Option<String> },
    ListDatasets { namespace: String },
    GetDataset { namespace: String, name: String },
    GetDatasetVersions { namespace: String, name: String },
    TagDataset { namespace: String, name: String, tag: String },
    
    // Job commands
    CreateJob { namespace: String, name: String, r#type: String, location: Option<String> },
    ListJobs { namespace: String },
    GetJob { namespace: String, name: String },
    GetJobRuns { namespace: String, name: String },
    
    // Run commands
    CreateRun { namespace: String, job: String },
    GetRun { run_id: String },
    StartRun { run_id: String },
    CompleteRun { run_id: String },
    FailRun { run_id: String },
    AbortRun { run_id: String },
    
    // Source commands
    CreateSource { name: String, r#type: String, url: String },
    ListSources,
    GetSource { name: String },
    
    // Tag commands
    CreateTag { name: String, description: Option<String> },
    ListTags,
    
    // Lineage commands
    GetDatasetLineage { namespace: String, name: String, depth: Option<i32> },
    GetJobLineage { namespace: String, name: String, depth: Option<i32> },
    
    // Search command
    Search { query: String, filter: Option<String> },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = MarquezClient::new(cli.url.clone());

    match cli.command {
        Commands::CreateNamespace { name, owner, description } => {
            let ns = Namespace {
                name: name.clone(),
                owner_name: owner,
                description,
                created_at: None,
                updated_at: None,
            };
            let result = client.create_namespace(&ns)?;
            println!("✅ Created namespace: {}", result.name);
        }

        Commands::ListNamespaces => {
            let namespaces = client.list_namespaces()?;
            println!("📁 Namespaces ({} total):", namespaces.len());
            for ns in namespaces {
                println!("   • {}", ns.name);
                if let Some(owner) = ns.owner_name {
                    println!("     Owner: {}", owner);
                }
            }
        }

        Commands::GetNamespace { name } => {
            let ns = client.get_namespace(&name)?;
            println!("{:#?}", ns);
        }

        Commands::CreateDataset { namespace, name, r#type, source } => {
            let dataset = Dataset {
                name: name.clone(),
                namespace: namespace.clone(),
                r#type,
                physical_name: None,
                description: None,
                source_name: source,
                fields: None,
                tags: None,
                facets: None,
            };
            let result = client.create_dataset(&namespace, &dataset)?;
            println!("✅ Created dataset: {}", result.name);
        }

        Commands::ListDatasets { namespace } => {
            let datasets = client.list_datasets(&namespace)?;
            println!("📊 Datasets in {} ({} total):", namespace, datasets.len());
            for ds in datasets {
                println!("   • {} ({})", ds.name, ds.r#type);
            }
        }

        Commands::GetDataset { namespace, name } => {
            let dataset = client.get_dataset(&namespace, &name)?;
            println!("{:#?}", dataset);
        }

        Commands::GetDatasetVersions { namespace, name } => {
            let versions = client.get_dataset_versions(&namespace, &name)?;
            println!("📜 Versions ({} total):", versions.len());
            for (i, v) in versions.iter().enumerate() {
                println!("   {}. {:#?}", i + 1, v);
            }
        }

        Commands::TagDataset { namespace, name, tag } => {
            let dataset = client.tag_dataset(&namespace, &name, &tag)?;
            println!("✅ Tagged dataset: {} with {}", dataset.name, tag);
        }

        Commands::CreateJob { namespace, name, r#type, location } => {
            let job = Job {
                name: name.clone(),
                namespace: namespace.clone(),
                r#type,
                description: None,
                location,
                inputs: None,
                outputs: None,
                context: None,
                facets: None,
            };
            let result = client.create_job(&namespace, &job)?;
            println!("✅ Created job: {}", result.name);
        }

        Commands::ListJobs { namespace } => {
            let jobs = client.list_jobs(&namespace)?;
            println!("⚙️  Jobs in {} ({} total):", namespace, jobs.len());
            for job in jobs {
                println!("   • {} ({})", job.name, job.r#type);
            }
        }

        Commands::GetJob { namespace, name } => {
            let job = client.get_job(&namespace, &name)?;
            println!("{:#?}", job);
        }

        Commands::GetJobRuns { namespace, name } => {
            let runs = client.get_job_runs(&namespace, &name)?;
            println!("🏃 Runs for {} ({} total):", name, runs.len());
            for run in runs {
                println!("   • {} [{}]", 
                    run.id.as_deref().unwrap_or("unknown"),
                    run.state.as_deref().unwrap_or("unknown")
                );
            }
        }

        Commands::CreateRun { namespace, job } => {
            let run = client.create_run(&namespace, &job)?;
            println!("✅ Created run: {}", run.id.as_deref().unwrap_or("unknown"));
        }

        Commands::GetRun { run_id } => {
            let run = client.get_run(&run_id)?;
            println!("{:#?}", run);
        }

        Commands::StartRun { run_id } => {
            let run = client.start_run(&run_id)?;
            println!("▶️  Started run: {}", run.id.as_deref().unwrap_or("unknown"));
        }

        Commands::CompleteRun { run_id } => {
            let run = client.complete_run(&run_id)?;
            println!("✅ Completed run: {}", run.id.as_deref().unwrap_or("unknown"));
        }

        Commands::FailRun { run_id } => {
            let run = client.fail_run(&run_id)?;
            println!("❌ Failed run: {}", run.id.as_deref().unwrap_or("unknown"));
        }

        Commands::AbortRun { run_id } => {
            let run = client.abort_run(&run_id)?;
            println!("🚫 Aborted run: {}", run.id.as_deref().unwrap_or("unknown"));
        }

        Commands::CreateSource { name, r#type, url } => {
            let source = Source {
                name: name.clone(),
                r#type,
                connection_url: url,
                description: None,
            };
            let result = client.create_source(&source)?;
            println!("✅ Created source: {}", result.name);
        }

        Commands::ListSources => {
            let sources = client.list_sources()?;
            println!("🔌 Sources ({} total):", sources.len());
            for source in sources {
                println!("   • {} ({})", source.name, source.r#type);
            }
        }

        Commands::GetSource { name } => {
            let source = client.get_source(&name)?;
            println!("{:#?}", source);
        }

        Commands::CreateTag { name, description } => {
            let tag = Tag {
                name: name.clone(),
                description,
            };
            let result = client.create_tag(&tag)?;
            println!("✅ Created tag: {}", result.name);
        }

        Commands::ListTags => {
            let tags = client.list_tags()?;
            println!("🏷️  Tags ({} total):", tags.len());
            for tag in tags {
                println!("   • {}", tag.name);
            }
        }

        Commands::GetDatasetLineage { namespace, name, depth } => {
            let lineage = client.get_dataset_lineage(&namespace, &name, depth)?;
            println!("🔗 Dataset Lineage ({} nodes):", lineage.graph.len());
            for node in lineage.graph {
                println!("   • {} ({})", node.id, node.r#type);
            }
        }

        Commands::GetJobLineage { namespace, name, depth } => {
            let lineage = client.get_job_lineage(&namespace, &name, depth)?;
            println!("🔗 Job Lineage ({} nodes):", lineage.graph.len());
            for node in lineage.graph {
                println!("   • {} ({})", node.id, node.r#type);
            }
        }

        Commands::Search { query, filter } => {
            let results = client.search(&query, filter.as_deref())?;
            println!("🔍 Search Results ({} found):", results.len());
            for (i, result) in results.iter().enumerate() {
                println!("   {}. {:#?}", i + 1, result);
            }
        }
    }

    Ok(())
}
