use anyhow::Result;
use clap::{Parser, Subcommand};
use opencanvas_api_client::*;

#[derive(Parser)]
#[command(name = "opencanvas-cli")]
#[command(about = "OpenCanvas Collaborative Editor Client\n\nManage canvases and collaboration")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:3000", env = "OPENCANVAS_URL")]
    url: String,

    #[arg(short, long, env = "OPENCANVAS_API_KEY")]
    api_key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Canvas operations
    List,
    Get { id: String },
    Create { title: String, content: String },
    Update { id: String, title: String, content: String },
    Delete { id: String },
    Duplicate { id: String, new_title: String },
    Search { query: String },
    Stats { id: String },
    
    // Collaboration operations
    Share { canvas_id: String, email: String, permission: String },
    RemoveCollaborator { canvas_id: String, email: String },
    ListCollaborators { canvas_id: String },
    
    // Comment operations
    GetComments { canvas_id: String },
    DeleteComment { canvas_id: String, comment_id: String },
    
    // Version control operations
    GetRevisions { canvas_id: String },
    GetRevision { canvas_id: String, version: i32 },
    RestoreRevision { canvas_id: String, version: i32 },
    
    // Export operations
    ExportMarkdown { canvas_id: String },
    ExportJson { canvas_id: String },
    ExportHtml { canvas_id: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let client = if let Some(key) = cli.api_key {
        OpenCanvasClient::with_api_key(cli.url, key)
    } else {
        OpenCanvasClient::new(cli.url)
    };

    match cli.command {
        Commands::List => {
            let canvases = client.list_canvases()?;
            println!("📝 Canvases ({}):", canvases.len());
            for canvas in canvases {
                println!("   • {} ({})", canvas.title, canvas.id.unwrap_or_default());
            }
        }

        Commands::Get { id } => {
            let canvas = client.get_canvas(&id)?;
            println!("📝 Canvas: {}", canvas.title);
            println!("   ID: {}", canvas.id.unwrap_or_default());
            println!("   Version: {}", canvas.version.unwrap_or(1));
            println!("   Content length: {} chars", canvas.content.len());
        }

        Commands::Create { title, content } => {
            let canvas = Canvas {
                id: None,
                title,
                content,
                version: Some(1),
                created_at: None,
                updated_at: None,
                owner: None,
                collaborators: None,
            };
            let created = client.create_canvas(&canvas)?;
            println!("✅ Created canvas: {}", created.id.unwrap_or_default());
        }

        Commands::Update { id, title, content } => {
            let mut canvas = client.get_canvas(&id)?;
            canvas.title = title;
            canvas.content = content;
            let updated = client.update_canvas(&id, &canvas)?;
            println!("✅ Updated canvas: {}", updated.title);
        }

        Commands::Delete { id } => {
            client.delete_canvas(&id)?;
            println!("✅ Deleted canvas: {}", id);
        }

        Commands::Duplicate { id, new_title } => {
            let duplicated = client.duplicate_canvas(&id, &new_title)?;
            println!("✅ Duplicated as: {}", duplicated.id.unwrap_or_default());
        }

        Commands::Search { query } => {
            let results = client.search_canvases(&query)?;
            println!("🔍 Search Results ({}):", results.len());
            for canvas in results {
                println!("   • {}", canvas.title);
            }
        }

        Commands::Stats { id } => {
            let stats = client.get_canvas_stats(&id)?;
            println!("📊 Canvas Statistics:");
            for (key, value) in stats {
                println!("   {}: {}", key, value);
            }
        }

        Commands::Share { canvas_id, email, permission } => {
            client.share_canvas(&canvas_id, &email, &permission)?;
            println!("✅ Shared with {} ({})", email, permission);
        }

        Commands::RemoveCollaborator { canvas_id, email } => {
            client.remove_collaborator(&canvas_id, &email)?;
            println!("✅ Removed collaborator: {}", email);
        }

        Commands::ListCollaborators { canvas_id } => {
            let collaborators = client.list_collaborators(&canvas_id)?;
            println!("👥 Collaborators ({}):", collaborators.len());
            for collab in collaborators {
                println!("   • {}", collab);
            }
        }

        Commands::GetComments { canvas_id } => {
            let comments = client.get_comments(&canvas_id)?;
            println!("💬 Comments ({}):", comments.len());
            for comment in comments {
                println!("   • {}: {}", comment.author, comment.content);
            }
        }

        Commands::DeleteComment { canvas_id, comment_id } => {
            client.delete_comment(&canvas_id, &comment_id)?;
            println!("✅ Deleted comment: {}", comment_id);
        }

        Commands::GetRevisions { canvas_id } => {
            let revisions = client.get_revisions(&canvas_id)?;
            println!("📜 Revisions ({}):", revisions.len());
            for rev in revisions {
                println!("   • v{} by {} at {}", rev.version, rev.author, rev.timestamp);
            }
        }

        Commands::GetRevision { canvas_id, version } => {
            let revision = client.get_revision(&canvas_id, version)?;
            println!("📜 Revision v{}", revision.version);
            println!("   Author: {}", revision.author);
            println!("   Timestamp: {}", revision.timestamp);
        }

        Commands::RestoreRevision { canvas_id, version } => {
            let canvas = client.restore_revision(&canvas_id, version)?;
            println!("✅ Restored to v{}: {}", version, canvas.title);
        }

        Commands::ExportMarkdown { canvas_id } => {
            let markdown = client.export_markdown(&canvas_id)?;
            println!("{}", markdown);
        }

        Commands::ExportJson { canvas_id } => {
            let json = client.export_json(&canvas_id)?;
            println!("{}", json);
        }

        Commands::ExportHtml { canvas_id } => {
            let html = client.export_html(&canvas_id)?;
            println!("{}", html);
        }
    }

    Ok(())
}
