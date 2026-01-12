use anyhow::Result;
use clap::{Parser, Subcommand};
use n8n_api_client::*;

#[derive(Parser)]
#[command(name = "n8n-cli")]
#[command(about = "n8n Workflow Automation Client\n\nManage workflows and executions")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, env = "N8N_URL")]
    url: String,

    #[arg(short, long, env = "N8N_API_KEY")]
    api_key: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Workflow operations
    ListWorkflows { #[arg(long)] active: Option<bool> },
    GetWorkflow { id: String },
    DeleteWorkflow { id: String },
    ActivateWorkflow { id: String },
    DeactivateWorkflow { id: String },
    ExecuteWorkflow { id: String },
    WorkflowStats { id: String },
    
    // Execution operations
    ListExecutions { #[arg(long)] workflow_id: Option<String> },
    GetExecution { id: String },
    DeleteExecution { id: String },
    
    // Credential operations
    ListCredentials,
    GetCredential { id: String },
    DeleteCredential { id: String },
    
    // Tag operations
    ListTags,
    GetTag { id: String },
    CreateTag { name: String },
    UpdateTag { id: String, name: String },
    DeleteTag { id: String },
    
    // System operations
    Health,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = N8nClient::new(cli.url, cli.api_key);

    match cli.command {
        Commands::ListWorkflows { active } => {
            let workflows = client.list_workflows(active)?;
            println!("⚡ Workflows ({}):", workflows.len());
            for wf in workflows {
                let status = if wf.active { "🟢" } else { "⚪" };
                println!("   {} {} ({})", status, wf.name, wf.id.unwrap_or_default());
            }
        }

        Commands::GetWorkflow { id } => {
            let wf = client.get_workflow(&id)?;
            println!("⚡ Workflow: {}", wf.name);
            println!("   ID: {}", wf.id.unwrap_or_default());
            println!("   Active: {}", wf.active);
            println!("   Nodes: {}", wf.nodes.len());
        }

        Commands::DeleteWorkflow { id } => {
            client.delete_workflow(&id)?;
            println!("✅ Deleted workflow: {}", id);
        }

        Commands::ActivateWorkflow { id } => {
            let wf = client.activate_workflow(&id)?;
            println!("✅ Activated workflow: {}", wf.name);
        }

        Commands::DeactivateWorkflow { id } => {
            let wf = client.deactivate_workflow(&id)?;
            println!("✅ Deactivated workflow: {}", wf.name);
        }

        Commands::ExecuteWorkflow { id } => {
            let exec = client.execute_workflow(&id)?;
            println!("🚀 Started execution: {}", exec.id);
        }

        Commands::WorkflowStats { id } => {
            let stats = client.get_workflow_stats(&id)?;
            println!("📊 Workflow Statistics:");
            for (key, value) in stats {
                println!("   {}: {}", key, value);
            }
        }

        Commands::ListExecutions { workflow_id } => {
            let execs = client.list_executions(workflow_id.as_deref())?;
            println!("🔄 Executions ({}):", execs.len());
            for exec in execs {
                let status = if exec.finished { "✅" } else { "⏳" };
                println!("   {} {} ({})", status, exec.id, exec.mode);
            }
        }

        Commands::GetExecution { id } => {
            let exec = client.get_execution(&id)?;
            println!("🔄 Execution: {}", exec.id);
            println!("   Workflow: {}", exec.workflow_id);
            println!("   Finished: {}", exec.finished);
            println!("   Mode: {}", exec.mode);
            println!("   Started: {}", exec.started_at);
        }

        Commands::DeleteExecution { id } => {
            client.delete_execution(&id)?;
            println!("✅ Deleted execution: {}", id);
        }

        Commands::ListCredentials => {
            let creds = client.list_credentials()?;
            println!("🔑 Credentials ({}):", creds.len());
            for cred in creds {
                println!("   • {} ({})", cred.name, cred.cred_type);
            }
        }

        Commands::GetCredential { id } => {
            let cred = client.get_credential(&id)?;
            println!("🔑 Credential: {}", cred.name);
            println!("   Type: {}", cred.cred_type);
        }

        Commands::DeleteCredential { id } => {
            client.delete_credential(&id)?;
            println!("✅ Deleted credential: {}", id);
        }

        Commands::ListTags => {
            let tags = client.list_tags()?;
            println!("🏷️  Tags ({}):", tags.len());
            for tag in tags {
                println!("   • {}", tag.name);
            }
        }

        Commands::GetTag { id } => {
            let tag = client.get_tag(&id)?;
            println!("🏷️  Tag: {}", tag.name);
        }

        Commands::CreateTag { name } => {
            let tag = client.create_tag(&name)?;
            println!("✅ Created tag: {}", tag.name);
        }

        Commands::UpdateTag { id, name } => {
            let tag = client.update_tag(&id, &name)?;
            println!("✅ Updated tag: {}", tag.name);
        }

        Commands::DeleteTag { id } => {
            client.delete_tag(&id)?;
            println!("✅ Deleted tag: {}", id);
        }

        Commands::Health => {
            let healthy = client.health_check()?;
            if healthy {
                println!("✅ n8n is healthy");
            } else {
                println!("❌ n8n health check failed");
            }
        }
    }

    Ok(())
}
