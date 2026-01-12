use anyhow::Result;
use clap::{Parser, Subcommand};
use git_api_client::*;

#[derive(Parser)]
#[command(name = "git-cli")]
#[command(about = "Git Client")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = ".", env = "GIT_REPO_PATH")]
    path: String,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Clone { url: String },
    Add { files: Vec<String> },
    Commit { message: String },
    Push { remote: String, branch: String },
    Pull { remote: String, branch: String },
    Branch { name: String },
    Branches,
    Status,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = GitClient::new(cli.path);

    match cli.command {
        Commands::Init => {
            client.init()?;
            println!("✅ Initialized repository");
        }
        Commands::Clone { url } => {
            client.clone(&url)?;
            println!("✅ Cloned {}", url);
        }
        Commands::Add { files } => {
            let refs: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
            client.add(&refs)?;
            println!("✅ Added files");
        }
        Commands::Commit { message } => {
            let oid = client.commit(&message)?;
            println!("✅ Committed: {}", oid);
        }
        Commands::Push { remote, branch } => {
            client.push(&remote, &branch)?;
            println!("✅ Pushed to {}/{}", remote, branch);
        }
        Commands::Pull { remote, branch } => {
            client.pull(&remote, &branch)?;
            println!("✅ Pulled from {}/{}", remote, branch);
        }
        Commands::Branch { name } => {
            client.create_branch(&name)?;
            println!("✅ Created branch: {}", name);
        }
        Commands::Branches => {
            let branches = client.list_branches()?;
            println!("🌿 Branches ({}):", branches.len());
            for b in branches {
                println!("   • {}", b);
            }
        }
        Commands::Status => {
            let files = client.status()?;
            println!("📝 Changed files ({}):", files.len());
            for f in files {
                println!("   • {}", f);
            }
        }
    }
    Ok(())
}
