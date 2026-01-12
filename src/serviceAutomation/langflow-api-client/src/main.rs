use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use langflow_api_client::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "langflow-cli")]
#[command(about = "Complete Langflow API Client - 100% Coverage\n\nAutomated Langflow operations from Lean4-verified specifications")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:7860", env = "LANGFLOW_URL")]
    base_url: String,

    #[arg(short = 'k', long, env = "LANGFLOW_API_KEY")]
    api_key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // ========================================================================
    // FOLDERS/PROJECTS
    // ========================================================================
    /// Create a new project/folder
    CreateFolder {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        description: Option<String>,
    },
    
    /// List all projects/folders
    ListFolders,
    
    /// Get folder by ID
    GetFolder {
        #[arg(short, long)]
        id: String,
    },
    
    /// Update folder
    UpdateFolder {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        description: Option<String>,
    },
    
    /// Delete folder
    DeleteFolder {
        #[arg(short, long)]
        id: String,
    },

    // ========================================================================
    // FLOWS
    // ========================================================================
    /// Create a new flow
    CreateFlow {
        #[arg(short = 'f', long)]
        folder_name: Option<String>,
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        description: Option<String>,
    },
    
    /// List all flows
    ListFlows,
    
    /// Get flow by ID
    GetFlow {
        #[arg(short, long)]
        id: String,
    },
    
    /// Update flow
    UpdateFlow {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        description: Option<String>,
    },
    
    /// Delete flow
    DeleteFlow {
        #[arg(short, long)]
        id: String,
    },
    
    /// Download flow as JSON
    DownloadFlow {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        output: PathBuf,
    },
    
    /// Upload/Import flow from JSON
    UploadFlow {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short = 'f', long)]
        folder_name: Option<String>,
    },
    
    /// Run a flow
    RunFlow {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        input: Option<String>,
    },

    // ========================================================================
    // COMPONENTS
    // ========================================================================
    /// List all components
    ListComponents,
    
    /// Get component by ID
    GetComponent {
        #[arg(short, long)]
        id: String,
    },
    
    /// Create custom component
    CreateComponent {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        description: Option<String>,
        #[arg(short, long)]
        data_file: PathBuf,
    },
    
    /// Update component
    UpdateComponent {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        description: Option<String>,
    },
    
    /// Delete component
    DeleteComponent {
        #[arg(short, long)]
        id: String,
    },

    // ========================================================================
    // USERS
    // ========================================================================
    /// Get current user info
    Whoami,
    
    /// List all users
    ListUsers,
    
    /// Get user by ID
    GetUser {
        #[arg(short, long)]
        id: String,
    },
    
    /// Update user
    UpdateUser {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        username: Option<String>,
        #[arg(short, long)]
        email: Option<String>,
    },
    
    /// Delete user
    DeleteUser {
        #[arg(short, long)]
        id: String,
    },

    // ========================================================================
    // API KEYS
    // ========================================================================
    /// Create API key
    CreateApiKey {
        #[arg(short, long)]
        name: String,
    },
    
    /// List API keys
    ListApiKeys,
    
    /// Delete API key
    DeleteApiKey {
        #[arg(short, long)]
        id: String,
    },

    // ========================================================================
    // VARIABLES
    // ========================================================================
    /// Create variable
    CreateVariable {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        value: String,
        #[arg(short, long)]
        var_type: Option<String>,
    },
    
    /// List variables
    ListVariables,
    
    /// Get variable by name
    GetVariable {
        #[arg(short, long)]
        name: String,
    },
    
    /// Update variable
    UpdateVariable {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        value: Option<String>,
    },
    
    /// Delete variable
    DeleteVariable {
        #[arg(short, long)]
        id: String,
    },

    // ========================================================================
    // STORE
    // ========================================================================
    /// List store items/templates
    ListStore,
    
    /// Get store item
    GetStoreItem {
        #[arg(short, long)]
        id: String,
    },

    // ========================================================================
    // SYSTEM
    // ========================================================================
    /// Check API health
    Health,
    
    /// Get API version
    Version,
    
    /// Get frontend config
    Config,

    // ========================================================================
    // BATCH OPERATIONS
    // ========================================================================
    /// Import all Lean4-verified flows
    ImportLean4Flows {
        #[arg(short = 'f', long)]
        folder_name: String,
    },
    
    /// Export all flows from a folder
    ExportFolder {
        #[arg(short = 'f', long)]
        folder_name: String,
        #[arg(short, long)]
        output_dir: PathBuf,
    },

    // ========================================================================
    // FILES
    // ========================================================================
    /// Upload file to flow
    UploadFile {
        #[arg(short = 'f', long)]
        flow_id: String,
        #[arg(short, long)]
        file: PathBuf,
    },
    
    /// List files for a flow
    ListFlowFiles {
        #[arg(short = 'f', long)]
        flow_id: String,
    },
    
    /// Download file
    DownloadFile {
        #[arg(short, long)]
        file_id: String,
        #[arg(short, long)]
        output: PathBuf,
    },
    
    /// Delete file
    DeleteFile {
        #[arg(short, long)]
        file_id: String,
    },

    // ========================================================================
    // LOGS
    // ========================================================================
    /// Get logs for a flow
    FlowLogs {
        #[arg(short = 'f', long)]
        flow_id: String,
    },
    
    /// Get logs for a run session
    RunLogs {
        #[arg(short, long)]
        session_id: String,
    },
    
    /// Get all system logs
    SystemLogs,
    
    /// Clear logs for a flow
    ClearFlowLogs {
        #[arg(short = 'f', long)]
        flow_id: String,
    },

    // ========================================================================
    // MONITOR
    // ========================================================================
    /// Get monitor data for a flow
    MonitorFlow {
        #[arg(short = 'f', long)]
        flow_id: String,
    },
    
    /// Get monitor data for all flows
    MonitorAll,
    
    /// Get metrics for a flow
    FlowMetrics {
        #[arg(short = 'f', long)]
        flow_id: String,
    },
    
    /// Get execution history for a flow
    FlowHistory {
        #[arg(short = 'f', long)]
        flow_id: String,
        #[arg(short, long)]
        limit: Option<usize>,
    },

    // ========================================================================
    // BUILD
    // ========================================================================
    /// Build and execute a flow
    BuildFlow {
        #[arg(short = 'f', long)]
        flow_id: String,
        #[arg(short, long)]
        input: Option<String>,
    },
    
    /// Get flow build status
    BuildStatus {
        #[arg(short = 'f', long)]
        flow_id: String,
    },
    
    /// Stream flow events
    BuildStream {
        #[arg(short = 'f', long)]
        flow_id: String,
        #[arg(short, long)]
        input: Option<String>,
        #[arg(short, long, default_value = "true")]
        stream: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = LangflowClient::new(cli.base_url.clone(), cli.api_key);

    match cli.command {
        // ====================================================================
        // FOLDERS
        // ====================================================================
        Commands::CreateFolder { name, description } => {
            let folder = Folder {
                id: None,
                name: name.clone(),
                description,
                parent_id: None,
                components_list: vec![],
            };
            let result = client.create_folder(&folder)?;
            println!("✅ Created folder: {} ({})", result.name, result.id.unwrap());
        }

        Commands::ListFolders => {
            let folders = client.list_folders()?;
            println!("📂 Folders ({} total):", folders.len());
            for folder in folders {
                println!("   • {} ({})", folder.name, folder.id.unwrap());
            }
        }

        Commands::GetFolder { id } => {
            let folder = client.get_folder(id.parse()?)?;
            println!("{:#?}", folder);
        }

        Commands::UpdateFolder { id, name, description } => {
            let mut folder = client.get_folder(id.parse()?)?;
            if let Some(n) = name {
                folder.name = n;
            }
            if let Some(d) = description {
                folder.description = Some(d);
            }
            let result = client.update_folder(id.parse()?, &folder)?;
            println!("✅ Updated folder: {}", result.name);
        }

        Commands::DeleteFolder { id } => {
            client.delete_folder(id.parse()?)?;
            println!("✅ Deleted folder: {}", id);
        }

        // ====================================================================
        // FLOWS
        // ====================================================================
        Commands::CreateFlow { folder_name, name, description } => {
            let folder_id = if let Some(fname) = folder_name {
                let folders = client.list_folders()?;
                folders
                    .iter()
                    .find(|f| f.name == fname)
                    .map(|f| f.id.unwrap())
            } else {
                None
            };

            let flow = Flow {
                id: None,
                name: name.clone(),
                description,
                data: Some(serde_json::json!({
                    "nodes": [],
                    "edges": [],
                    "viewport": {"x": 0, "y": 0, "zoom": 1}
                })),
                folder_id,
                is_component: false,
                updated_at: None,
                gradient: None,
            };
            
            let result = client.create_flow(&flow)?;
            println!("✅ Created flow: {} ({})", result.name, result.id.unwrap());
        }

        Commands::ListFlows => {
            let flows = client.list_flows()?;
            println!("🌊 Flows ({} total):", flows.len());
            for flow in flows {
                println!("   • {} ({})", flow.name, flow.id.unwrap());
            }
        }

        Commands::GetFlow { id } => {
            let flow = client.get_flow(id.parse()?)?;
            println!("{:#?}", flow);
        }

        Commands::UpdateFlow { id, name, description } => {
            let mut flow = client.get_flow(id.parse()?)?;
            if let Some(n) = name {
                flow.name = n;
            }
            if let Some(d) = description {
                flow.description = Some(d);
            }
            let result = client.update_flow(id.parse()?, &flow)?;
            println!("✅ Updated flow: {}", result.name);
        }

        Commands::DeleteFlow { id } => {
            client.delete_flow(id.parse()?)?;
            println!("✅ Deleted flow: {}", id);
        }

        Commands::DownloadFlow { id, output } => {
            let data = client.download_flow(id.parse()?)?;
            fs::write(&output, serde_json::to_string_pretty(&data)?)?;
            println!("✅ Downloaded flow to: {:?}", output);
        }

        Commands::UploadFlow { file, folder_name } => {
            let content = fs::read_to_string(&file)?;
            let data: Value = serde_json::from_str(&content)?;
            
            let folder_id = if let Some(fname) = folder_name {
                let folders = client.list_folders()?;
                folders
                    .iter()
                    .find(|f| f.name == fname)
                    .map(|f| f.id.unwrap())
            } else {
                None
            };

            let result = client.upload_flow(data, folder_id)?;
            println!("✅ Uploaded flow: {} ({})", result.name, result.id.unwrap());
        }

        Commands::RunFlow { id, input } => {
            let mut inputs = HashMap::new();
            if let Some(inp) = input {
                inputs.insert("input".to_string(), serde_json::json!(inp));
            }
            
            let result = client.run_flow(id.parse()?, inputs, None)?;
            println!("✅ Flow execution complete!");
            println!("Session ID: {}", result.session_id);
            println!("Outputs: {:#?}", result.outputs);
        }

        // ====================================================================
        // COMPONENTS
        // ====================================================================
        Commands::ListComponents => {
            let components = client.list_components()?;
            println!("🧩 Components ({} total):", components.len());
            for comp in components {
                println!("   • {} ({})", comp.name, comp.id.unwrap());
            }
        }

        Commands::GetComponent { id } => {
            let component = client.get_component(id.parse()?)?;
            println!("{:#?}", component);
        }

        Commands::CreateComponent { name, description, data_file } => {
            let content = fs::read_to_string(&data_file)?;
            let data: Value = serde_json::from_str(&content)?;
            
            let component = Component {
                id: None,
                name: name.clone(),
                description,
                data,
                is_component: true,
                parent_id: None,
            };
            
            let result = client.create_component(&component)?;
            println!("✅ Created component: {} ({})", result.name, result.id.unwrap());
        }

        Commands::UpdateComponent { id, name, description } => {
            let mut component = client.get_component(id.parse()?)?;
            if let Some(n) = name {
                component.name = n;
            }
            if let Some(d) = description {
                component.description = Some(d);
            }
            let result = client.update_component(id.parse()?, &component)?;
            println!("✅ Updated component: {}", result.name);
        }

        Commands::DeleteComponent { id } => {
            client.delete_component(id.parse()?)?;
            println!("✅ Deleted component: {}", id);
        }

        // ====================================================================
        // USERS
        // ====================================================================
        Commands::Whoami => {
            let user = client.get_current_user()?;
            println!("👤 Current User:");
            println!("   Username: {}", user.username);
            println!("   Email: {:?}", user.email);
            println!("   Active: {}", user.is_active);
            println!("   Superuser: {}", user.is_superuser);
        }

        Commands::ListUsers => {
            let users = client.list_users()?;
            println!("👥 Users ({} total):", users.len());
            for user in users {
                println!("   • {} ({})", user.username, user.id.unwrap());
            }
        }

        Commands::GetUser { id } => {
            let user = client.get_user(id.parse()?)?;
            println!("{:#?}", user);
        }

        Commands::UpdateUser { id, username, email } => {
            let mut user = client.get_user(id.parse()?)?;
            if let Some(u) = username {
                user.username = u;
            }
            if let Some(e) = email {
                user.email = Some(e);
            }
            let result = client.update_user(id.parse()?, &user)?;
            println!("✅ Updated user: {}", result.username);
        }

        Commands::DeleteUser { id } => {
            client.delete_user(id.parse()?)?;
            println!("✅ Deleted user: {}", id);
        }

        // ====================================================================
        // API KEYS
        // ====================================================================
        Commands::CreateApiKey { name } => {
            let result = client.create_api_key(name)?;
            println!("✅ Created API key: {}", result.name);
            if let Some(key) = result.api_key {
                println!("🔑 Key: {}", key);
                println!("⚠️  Save this key - it won't be shown again!");
            }
        }

        Commands::ListApiKeys => {
            let keys = client.list_api_keys()?;
            println!("🔑 API Keys ({} total):", keys.len());
            for key in keys {
                println!("   • {} ({})", key.name, key.id.unwrap());
            }
        }

        Commands::DeleteApiKey { id } => {
            client.delete_api_key(id.parse()?)?;
            println!("✅ Deleted API key: {}", id);
        }

        // ====================================================================
        // VARIABLES
        // ====================================================================
        Commands::CreateVariable { name, value, var_type } => {
            let variable = Variable {
                id: None,
                name: name.clone(),
                value,
                default_value: None,
                r#type: var_type,
            };
            let result = client.create_variable(&variable)?;
            println!("✅ Created variable: {}", result.name);
        }

        Commands::ListVariables => {
            let variables = client.list_variables()?;
            println!("📊 Variables ({} total):", variables.len());
            for var in variables {
                println!("   • {} = {}", var.name, var.value);
            }
        }

        Commands::GetVariable { name } => {
            let variable = client.get_variable(&name)?;
            println!("{:#?}", variable);
        }

        Commands::UpdateVariable { id, name, value } => {
            let mut variable = client.list_variables()?
                .into_iter()
                .find(|v| v.id.unwrap().to_string() == id)
                .context("Variable not found")?;
            
            if let Some(n) = name {
                variable.name = n;
            }
            if let Some(v) = value {
                variable.value = v;
            }
            let result = client.update_variable(id.parse()?, &variable)?;
            println!("✅ Updated variable: {}", result.name);
        }

        Commands::DeleteVariable { id } => {
            client.delete_variable(id.parse()?)?;
            println!("✅ Deleted variable: {}", id);
        }

        // ====================================================================
        // STORE
        // ====================================================================
        Commands::ListStore => {
            let items = client.list_store_items()?;
            println!("🏪 Store Items ({} total):", items.len());
            for item in items {
                if let Some(name) = item.get("name").and_then(|v| v.as_str()) {
                    println!("   • {}", name);
                }
            }
        }

        Commands::GetStoreItem { id } => {
            let item = client.get_store_item(&id)?;
            println!("{:#?}", item);
        }

        // ====================================================================
        // SYSTEM
        // ====================================================================
        Commands::Health => {
            let health = client.health_check()?;
            println!("💚 Health: {:#?}", health);
        }

        Commands::Version => {
            let version = client.get_version()?;
            println!("📦 Version: {:#?}", version);
        }

        Commands::Config => {
            let config = client.get_config()?;
            println!("⚙️  Config: {:#?}", config);
        }

        // ====================================================================
        // BATCH OPERATIONS
        // ====================================================================
        Commands::ImportLean4Flows { folder_name } => {
            println!("🚀 Importing Lean4-verified flows...");
            println!("{}", "=".repeat(60));

            // Find or create folder
            let folders = client.list_folders()?;
            let folder_id = if let Some(folder) = folders.iter().find(|f| f.name == folder_name) {
                folder.id.unwrap()
            } else {
                let new_folder = Folder {
                    id: None,
                    name: folder_name.clone(),
                    description: Some("Lean4-verified workflows".to_string()),
                    parent_id: None,
                    components_list: vec![],
                };
                client.create_folder(&new_folder)?.id.unwrap()
            };

            // Import flows from directory
            let flows_dir = PathBuf::from("src/serviceAutomation/serviceLangflow");
            if flows_dir.exists() {
                for entry in fs::read_dir(flows_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if let Ok(data) = serde_json::from_str::<Value>(&content) {
                                match client.upload_flow(data, Some(folder_id)) {
                                    Ok(flow) => println!("✅ Imported: {} ({})", flow.name, flow.id.unwrap()),
                                    Err(e) => println!("⚠️  Failed to import {:?}: {}", path.file_name(), e),
                                }
                            }
                        }
                    }
                }
            }

            println!("\n🎉 Import complete!");
            println!("   Project: {}", folder_name);
            println!("   Open: {}/flows/folder/{}", cli.base_url, folder_id);
        }

        Commands::ExportFolder { folder_name, output_dir } => {
            println!("📦 Exporting folder: {}", folder_name);
            
            let folders = client.list_folders()?;
            let folder = folders
                .iter()
                .find(|f| f.name == folder_name)
                .context("Folder not found")?;
            
            fs::create_dir_all(&output_dir)?;
            
            let flows = client.list_flows()?;
            let folder_flows: Vec<_> = flows
                .iter()
                .filter(|f| f.folder_id == folder.id)
                .collect();
            
            for flow in &folder_flows {
                let data = client.download_flow(flow.id.unwrap())?;
                let filename = format!("{}.json", flow.name.replace(" ", "_"));
                let path = output_dir.join(filename);
                fs::write(&path, serde_json::to_string_pretty(&data)?)?;
                println!("✅ Exported: {} → {:?}", flow.name, path);
            }
            
            println!("\n🎉 Export complete! ({} flows)", folder_flows.len());
        }

        // ====================================================================
        // FILES
        // ====================================================================
        Commands::UploadFile { flow_id, file } => {
            let file_data = fs::read(&file)?;
            let filename = file.file_name().unwrap().to_str().unwrap();
            let result = client.upload_file(flow_id.parse()?, filename, file_data)?;
            println!("✅ Uploaded file: {:?}", result);
        }

        Commands::ListFlowFiles { flow_id } => {
            let files = client.list_files(flow_id.parse()?)?;
            println!("📁 Files ({} total):", files.len());
            for file in files {
                if let Some(name) = file.get("name").and_then(|v| v.as_str()) {
                    println!("   • {}", name);
                }
            }
        }

        Commands::DownloadFile { file_id, output } => {
            let data = client.download_file(&file_id)?;
            fs::write(&output, data)?;
            println!("✅ Downloaded file to: {:?}", output);
        }

        Commands::DeleteFile { file_id } => {
            client.delete_file(&file_id)?;
            println!("✅ Deleted file: {}", file_id);
        }

        // ====================================================================
        // LOGS
        // ====================================================================
        Commands::FlowLogs { flow_id } => {
            let logs = client.get_flow_logs(flow_id.parse()?)?;
            println!("📋 Flow Logs ({} entries):", logs.len());
            for log in logs {
                println!("   [{}] {} - {}", 
                    log.level.as_deref().unwrap_or("INFO"),
                    log.timestamp.as_deref().unwrap_or("N/A"),
                    log.message
                );
            }
        }

        Commands::RunLogs { session_id } => {
            let logs = client.get_run_logs(&session_id)?;
            println!("📋 Run Logs ({} entries):", logs.len());
            for log in logs {
                println!("   [{}] {}", 
                    log.level.as_deref().unwrap_or("INFO"),
                    log.message
                );
            }
        }

        Commands::SystemLogs => {
            let logs = client.get_system_logs()?;
            println!("📋 System Logs ({} entries):", logs.len());
            for log in logs.iter().take(50) {
                println!("   [{}] {} - {}", 
                    log.level.as_deref().unwrap_or("INFO"),
                    log.timestamp.as_deref().unwrap_or("N/A"),
                    log.message
                );
            }
            if logs.len() > 50 {
                println!("   ... and {} more entries", logs.len() - 50);
            }
        }

        Commands::ClearFlowLogs { flow_id } => {
            client.clear_flow_logs(flow_id.parse()?)?;
            println!("✅ Cleared logs for flow: {}", flow_id);
        }

        // ====================================================================
        // MONITOR
        // ====================================================================
        Commands::MonitorFlow { flow_id } => {
            let monitor = client.get_flow_monitor(flow_id.parse()?)?;
            println!("📊 Flow Monitor:");
            println!("   Status: {}", monitor.status);
            if let Some(metrics) = monitor.metrics {
                println!("   Metrics: {:#?}", metrics);
            }
        }

        Commands::MonitorAll => {
            let monitors = client.get_all_monitors()?;
            println!("📊 All Flow Monitors ({} total):", monitors.len());
            for monitor in monitors {
                println!("   • Flow {} - Status: {}", monitor.flow_id, monitor.status);
            }
        }

        Commands::FlowMetrics { flow_id } => {
            let metrics = client.get_flow_metrics(flow_id.parse()?)?;
            println!("📈 Flow Metrics:");
            println!("{:#?}", metrics);
        }

        Commands::FlowHistory { flow_id, limit } => {
            let history = client.get_flow_history(flow_id.parse()?, limit)?;
            println!("📜 Flow History ({} executions):", history.len());
            for (i, execution) in history.iter().enumerate() {
                println!("   {}. {:#?}", i + 1, execution);
            }
        }

        // ====================================================================
        // BUILD
        // ====================================================================
        Commands::BuildFlow { flow_id, input } => {
            let mut inputs = HashMap::new();
            if let Some(inp) = input {
                inputs.insert("input".to_string(), serde_json::json!(inp));
            }
            
            let result = client.build_flow(flow_id.parse()?, inputs)?;
            println!("🔨 Build complete!");
            println!("{:#?}", result);
        }

        Commands::BuildStatus { flow_id } => {
            let status = client.get_build_status(flow_id.parse()?)?;
            println!("📊 Build Status:");
            println!("{:#?}", status);
        }

        Commands::BuildStream { flow_id, input, stream } => {
            let mut inputs = HashMap::new();
            if let Some(inp) = input {
                inputs.insert("input".to_string(), serde_json::json!(inp));
            }
            
            let events = client.build_flow_stream(flow_id.parse()?, inputs, stream)?;
            println!("🌊 Flow Events (stream={}):", stream);
            println!("{:#?}", events);
        }
    }

    Ok(())
}
