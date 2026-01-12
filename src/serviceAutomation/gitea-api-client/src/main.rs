use anyhow::Result;
use clap::{Parser, Subcommand};
use gitea_api_client::*;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "gitea-cli")]
#[command(about = "Complete Gitea API Client\n\nFull Gitea automation from Rust")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, default_value = "http://localhost:3000", env = "GITEA_URL")]
    base_url: String,

    #[arg(short, long, env = "GITEA_TOKEN")]
    token: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Get Gitea version
    Version,
    
    /// Health check
    Health,

    // ========================================================================
    // USERS
    // ========================================================================
    /// Get current user
    Whoami,
    
    /// List all users
    ListUsers,
    
    /// Get user by username
    GetUser {
        #[arg(short, long)]
        username: String,
    },

    // ========================================================================
    // REPOSITORIES
    // ========================================================================
    /// Create repository
    CreateRepo {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        description: Option<String>,
        #[arg(short, long)]
        private: bool,
    },
    
    /// List user repositories
    ListRepos {
        #[arg(short, long)]
        username: String,
    },
    
    /// Get repository
    GetRepo {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },
    
    /// Delete repository
    DeleteRepo {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },

    // ========================================================================
    // ISSUES
    // ========================================================================
    /// Create issue
    CreateIssue {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        title: String,
        #[arg(short, long)]
        body: Option<String>,
    },
    
    /// List repository issues
    ListIssues {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },
    
    /// Get issue
    GetIssue {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        index: i64,
    },
    
    /// Close issue
    CloseIssue {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        index: i64,
    },

    // ========================================================================
    // PULL REQUESTS
    // ========================================================================
    /// List pull requests
    ListPrs {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },
    
    /// Get pull request
    GetPr {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        index: i64,
    },
    
    /// Merge pull request
    MergePr {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        index: i64,
    },

    // ========================================================================
    // BRANCHES
    // ========================================================================
    /// List branches
    ListBranches {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },
    
    /// Get branch
    GetBranch {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        branch: String,
    },
    
    /// Create branch
    CreateBranch {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        from_ref: String,
    },
    
    /// Delete branch
    DeleteBranch {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        branch: String,
    },

    // ========================================================================
    // COMMITS
    // ========================================================================
    /// List commits
    ListCommits {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },
    
    /// Get commit
    GetCommit {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        sha: String,
    },

    // ========================================================================
    // ORGANIZATIONS
    // ========================================================================
    /// List user organizations
    ListOrgs {
        #[arg(short, long)]
        username: String,
    },
    
    /// Get organization
    GetOrg {
        #[arg(short, long)]
        org: String,
    },

    // ========================================================================
    // TEAMS
    // ========================================================================
    /// List organization teams
    ListTeams {
        #[arg(short, long)]
        org: String,
    },
    
    /// Get team
    GetTeam {
        #[arg(short, long)]
        team_id: i64,
    },

    // ========================================================================
    // RELEASES
    // ========================================================================
    /// Create release
    CreateRelease {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        tag: String,
        #[arg(short, long)]
        name: Option<String>,
        #[arg(short, long)]
        body: Option<String>,
    },
    
    /// List releases
    ListReleases {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },
    
    /// Get release
    GetRelease {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        tag: String,
    },

    // ========================================================================
    // LABELS
    // ========================================================================
    /// Create label
    CreateLabel {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        color: Option<String>,
    },
    
    /// List labels
    ListLabels {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },

    // ========================================================================
    // WEBHOOKS
    // ========================================================================
    /// List webhooks
    ListWebhooks {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
    },

    // ========================================================================
    // FILES
    // ========================================================================
    /// Get file contents
    GetFile {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        path: String,
    },
    
    /// Create or update file
    UpdateFile {
        #[arg(short, long)]
        owner: String,
        #[arg(short, long)]
        repo: String,
        #[arg(short, long)]
        path: String,
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        message: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = GiteaClient::new(cli.base_url.clone(), cli.token);

    match cli.command {
        Commands::Version => {
            let version = client.get_version()?;
            println!("📦 Gitea Version:");
            println!("{:#?}", version);
        }

        Commands::Health => {
            let healthy = client.health_check()?;
            if healthy {
                println!("💚 Gitea is healthy!");
            } else {
                println!("❌ Gitea is not responding");
            }
        }

        Commands::Whoami => {
            let user = client.get_current_user()?;
            println!("👤 Current User:");
            println!("   Username: {}", user.username);
            println!("   Email: {:?}", user.email);
            println!("   Admin: {:?}", user.is_admin);
        }

        Commands::ListUsers => {
            let users = client.list_users()?;
            println!("👥 Users ({} total):", users.len());
            for user in users {
                println!("   • {} ({})", user.username, user.id.unwrap_or(0));
            }
        }

        Commands::GetUser { username } => {
            let user = client.get_user(&username)?;
            println!("{:#?}", user);
        }

        Commands::CreateRepo { name, description, private: is_private } => {
            let repo = Repository {
                id: None,
                name: name.clone(),
                full_name: None,
                description,
                private: is_private,
                fork: None,
                owner: None,
                html_url: None,
                ssh_url: None,
                clone_url: None,
                default_branch: None,
                created_at: None,
                updated_at: None,
            };
            let result = client.create_repo(&repo)?;
            println!("✅ Created repository: {}", result.name);
            if let Some(url) = result.html_url {
                println!("   URL: {}", url);
            }
        }

        Commands::ListRepos { username } => {
            let repos = client.list_user_repos(&username)?;
            println!("📦 Repositories ({} total):", repos.len());
            for repo in repos {
                println!("   • {}", repo.name);
                if let Some(desc) = repo.description {
                    println!("     {}", desc);
                }
            }
        }

        Commands::GetRepo { owner, repo } => {
            let repository = client.get_repo(&owner, &repo)?;
            println!("{:#?}", repository);
        }

        Commands::DeleteRepo { owner, repo } => {
            client.delete_repo(&owner, &repo)?;
            println!("✅ Deleted repository: {}/{}", owner, repo);
        }

        Commands::CreateIssue { owner, repo, title, body } => {
            let issue = Issue {
                id: None,
                number: None,
                title: title.clone(),
                body,
                state: None,
                labels: None,
                user: None,
                assignees: None,
                created_at: None,
                updated_at: None,
            };
            let result = client.create_issue(&owner, &repo, &issue)?;
            println!("✅ Created issue #{}: {}", result.number.unwrap_or(0), result.title);
        }

        Commands::ListIssues { owner, repo } => {
            let issues = client.list_repo_issues(&owner, &repo)?;
            println!("🐛 Issues ({} total):", issues.len());
            for issue in issues {
                println!("   #{} {} [{}]", 
                    issue.number.unwrap_or(0),
                    issue.title,
                    issue.state.as_deref().unwrap_or("unknown")
                );
            }
        }

        Commands::GetIssue { owner, repo, index } => {
            let issue = client.get_issue(&owner, &repo, index)?;
            println!("{:#?}", issue);
        }

        Commands::CloseIssue { owner, repo, index } => {
            let issue = client.close_issue(&owner, &repo, index)?;
            println!("✅ Closed issue #{}", issue.number.unwrap_or(0));
        }

        Commands::ListPrs { owner, repo } => {
            let prs = client.list_pull_requests(&owner, &repo)?;
            println!("🔀 Pull Requests ({} total):", prs.len());
            for pr in prs {
                println!("   #{} {} [{}]",
                    pr.number.unwrap_or(0),
                    pr.title,
                    pr.state.as_deref().unwrap_or("unknown")
                );
            }
        }

        Commands::GetPr { owner, repo, index } => {
            let pr = client.get_pull_request(&owner, &repo, index)?;
            println!("{:#?}", pr);
        }

        Commands::MergePr { owner, repo, index } => {
            let result = client.merge_pull_request(&owner, &repo, index)?;
            println!("✅ Merged PR #{}", index);
            println!("{:#?}", result);
        }

        Commands::ListBranches { owner, repo } => {
            let branches = client.list_branches(&owner, &repo)?;
            println!("🌿 Branches ({} total):", branches.len());
            for branch in branches {
                let protected = if branch.protected.unwrap_or(false) { " 🔒" } else { "" };
                println!("   • {}{}", branch.name, protected);
            }
        }

        Commands::GetBranch { owner, repo, branch } => {
            let b = client.get_branch(&owner, &repo, &branch)?;
            println!("{:#?}", b);
        }

        Commands::CreateBranch { owner, repo, name, from_ref } => {
            let branch = client.create_branch(&owner, &repo, &name, &from_ref)?;
            println!("✅ Created branch: {}", branch.name);
        }

        Commands::DeleteBranch { owner, repo, branch } => {
            client.delete_branch(&owner, &repo, &branch)?;
            println!("✅ Deleted branch: {}", branch);
        }

        Commands::ListCommits { owner, repo } => {
            let commits = client.list_commits(&owner, &repo)?;
            println!("📝 Commits ({} total):", commits.len());
            for commit in commits.iter().take(10) {
                if let Some(sha) = &commit.sha {
                    println!("   {} {}", 
                        &sha[..7], 
                        commit.message.as_deref().unwrap_or("")
                    );
                }
            }
        }

        Commands::GetCommit { owner, repo, sha } => {
            let commit = client.get_commit(&owner, &repo, &sha)?;
            println!("{:#?}", commit);
        }

        Commands::ListOrgs { username } => {
            let orgs = client.list_user_orgs(&username)?;
            println!("🏢 Organizations ({} total):", orgs.len());
            for org in orgs {
                println!("   • {}", org.username);
            }
        }

        Commands::GetOrg { org } => {
            let organization = client.get_org(&org)?;
            println!("{:#?}", organization);
        }

        Commands::ListTeams { org } => {
            let teams = client.list_org_teams(&org)?;
            println!("👥 Teams ({} total):", teams.len());
            for team in teams {
                println!("   • {}", team.name);
            }
        }

        Commands::GetTeam { team_id } => {
            let team = client.get_team(team_id)?;
            println!("{:#?}", team);
        }

        Commands::CreateRelease { owner, repo, tag, name, body } => {
            let release = Release {
                id: None,
                tag_name: tag.clone(),
                target_commitish: None,
                name,
                body,
                draft: None,
                prerelease: None,
            };
            let result = client.create_release(&owner, &repo, &release)?;
            println!("✅ Created release: {}", result.tag_name);
        }

        Commands::ListReleases { owner, repo } => {
            let releases = client.list_releases(&owner, &repo)?;
            println!("🚀 Releases ({} total):", releases.len());
            for release in releases {
                println!("   • {}", release.tag_name);
                if let Some(name) = release.name {
                    println!("     {}", name);
                }
            }
        }

        Commands::GetRelease { owner, repo, tag } => {
            let release = client.get_release(&owner, &repo, &tag)?;
            println!("{:#?}", release);
        }

        Commands::CreateLabel { owner, repo, name, color } => {
            let label = Label {
                id: None,
                name: name.clone(),
                color,
                description: None,
            };
            let result = client.create_label(&owner, &repo, &label)?;
            println!("✅ Created label: {}", result.name);
        }

        Commands::ListLabels { owner, repo } => {
            let labels = client.list_labels(&owner, &repo)?;
            println!("🏷️  Labels ({} total):", labels.len());
            for label in labels {
                println!("   • {} ({})", label.name, label.color.as_deref().unwrap_or(""));
            }
        }

        Commands::ListWebhooks { owner, repo } => {
            let webhooks = client.list_webhooks(&owner, &repo)?;
            println!("🪝 Webhooks ({} total):", webhooks.len());
            for webhook in webhooks {
                let active = if webhook.active { "✅" } else { "❌" };
                println!("   {} {} (ID: {})", active, webhook.r#type, webhook.id.unwrap_or(0));
            }
        }

        Commands::GetFile { owner, repo, path } => {
            let content = client.get_file_contents(&owner, &repo, &path)?;
            println!("📄 File: {}/{}/{}", owner, repo, path);
            println!("{:#?}", content);
        }

        Commands::UpdateFile { owner, repo, path, file, message } => {
            let content = fs::read_to_string(&file)?;
            let result = client.create_or_update_file(&owner, &repo, &path, &content, &message)?;
            println!("✅ Updated file: {}", path);
            println!("{:#?}", result);
        }
    }

    Ok(())
}
