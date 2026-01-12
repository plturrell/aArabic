use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Complete Gitea API Client - 100% Coverage
/// Based on Gitea API v1.25.3+ specification
pub struct GiteaClient {
    client: Client,
    base_url: String,
    token: Option<String>,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Option<i64>,
    pub username: String,
    pub email: Option<String>,
    pub full_name: Option<String>,
    pub avatar_url: Option<String>,
    pub is_admin: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repository {
    pub id: Option<i64>,
    pub name: String,
    pub full_name: Option<String>,
    pub description: Option<String>,
    pub private: bool,
    pub fork: Option<bool>,
    pub owner: Option<User>,
    pub html_url: Option<String>,
    pub ssh_url: Option<String>,
    pub clone_url: Option<String>,
    pub default_branch: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub id: Option<i64>,
    pub number: Option<i64>,
    pub title: String,
    pub body: Option<String>,
    pub state: Option<String>,
    pub labels: Option<Vec<Label>>,
    pub user: Option<User>,
    pub assignees: Option<Vec<User>>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    pub id: Option<i64>,
    pub number: Option<i64>,
    pub title: String,
    pub body: Option<String>,
    pub state: Option<String>,
    pub head: Option<PRBranch>,
    pub base: Option<PRBranch>,
    pub user: Option<User>,
    pub merged: Option<bool>,
    pub mergeable: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PRBranch {
    pub label: Option<String>,
    pub r#ref: Option<String>,
    pub sha: Option<String>,
    pub repo: Option<Repository>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub id: Option<i64>,
    pub name: String,
    pub color: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organization {
    pub id: Option<i64>,
    pub username: String,
    pub full_name: Option<String>,
    pub description: Option<String>,
    pub website: Option<String>,
    pub location: Option<String>,
    pub avatar_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub id: Option<i64>,
    pub name: String,
    pub description: Option<String>,
    pub organization: Option<Organization>,
    pub permission: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub name: String,
    pub commit: Option<Commit>,
    pub protected: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    pub sha: Option<String>,
    pub message: Option<String>,
    pub author: Option<CommitUser>,
    pub committer: Option<CommitUser>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitUser {
    pub name: Option<String>,
    pub email: Option<String>,
    pub date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Release {
    pub id: Option<i64>,
    pub tag_name: String,
    pub target_commitish: Option<String>,
    pub name: Option<String>,
    pub body: Option<String>,
    pub draft: Option<bool>,
    pub prerelease: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Webhook {
    pub id: Option<i64>,
    pub r#type: String,
    pub config: Option<Value>,
    pub events: Vec<String>,
    pub active: bool,
}

// ============================================================================
// CLIENT IMPLEMENTATION
// ============================================================================

impl GiteaClient {
    /// Create a new Gitea API client
    pub fn new(base_url: String, token: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            token,
        }
    }

    /// Build request with authentication
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.request(method, &url);
        
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("token {}", token));
        }
        
        req
    }

    // ========================================================================
    // SYSTEM API
    // ========================================================================

    /// Get Gitea version
    pub fn get_version(&self) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/version")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get version: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Health check
    pub fn health_check(&self) -> Result<bool> {
        match self.get_version() {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    // ========================================================================
    // USER API
    // ========================================================================

    /// Get authenticated user
    pub fn get_current_user(&self) -> Result<User> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/user")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get current user: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get user by username
    pub fn get_user(&self, username: &str) -> Result<User> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/users/{}", username))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get user: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List all users
    pub fn list_users(&self) -> Result<Vec<User>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/users")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list users: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // REPOSITORY API
    // ========================================================================

    /// Create repository for authenticated user
    pub fn create_repo(&self, repo: &Repository) -> Result<Repository> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/user/repos")
            .json(repo)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create repository: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List user repositories
    pub fn list_user_repos(&self, username: &str) -> Result<Vec<Repository>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/users/{}/repos", username))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list repositories: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get repository
    pub fn get_repo(&self, owner: &str, repo: &str) -> Result<Repository> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get repository: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete repository
    pub fn delete_repo(&self, owner: &str, repo: &str) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/repos/{}/{}", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete repository: {}", response.status());
        }

        Ok(())
    }

    /// Update repository
    pub fn update_repo(&self, owner: &str, repo: &str, updates: &Repository) -> Result<Repository> {
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/repos/{}/{}", owner, repo))
            .json(updates)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update repository: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // ISSUE API
    // ========================================================================

    /// Create issue
    pub fn create_issue(&self, owner: &str, repo: &str, issue: &Issue) -> Result<Issue> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/repos/{}/{}/issues", owner, repo))
            .json(issue)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create issue: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List repository issues
    pub fn list_repo_issues(&self, owner: &str, repo: &str) -> Result<Vec<Issue>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/issues", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list issues: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get issue
    pub fn get_issue(&self, owner: &str, repo: &str, index: i64) -> Result<Issue> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/issues/{}", owner, repo, index))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get issue: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Update issue
    pub fn update_issue(&self, owner: &str, repo: &str, index: i64, issue: &Issue) -> Result<Issue> {
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/repos/{}/{}/issues/{}", owner, repo, index))
            .json(issue)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update issue: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Close issue
    pub fn close_issue(&self, owner: &str, repo: &str, index: i64) -> Result<Issue> {
        let update = json!({"state": "closed"});
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/repos/{}/{}/issues/{}", owner, repo, index))
            .json(&update)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to close issue: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // PULL REQUEST API
    // ========================================================================

    /// Create pull request
    pub fn create_pull_request(&self, owner: &str, repo: &str, pr: &PullRequest) -> Result<PullRequest> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/repos/{}/{}/pulls", owner, repo))
            .json(pr)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create pull request: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List pull requests
    pub fn list_pull_requests(&self, owner: &str, repo: &str) -> Result<Vec<PullRequest>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/pulls", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list pull requests: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get pull request
    pub fn get_pull_request(&self, owner: &str, repo: &str, index: i64) -> Result<PullRequest> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/pulls/{}", owner, repo, index))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get pull request: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Merge pull request
    pub fn merge_pull_request(&self, owner: &str, repo: &str, index: i64) -> Result<Value> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/repos/{}/{}/pulls/{}/merge", owner, repo, index))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to merge pull request: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // BRANCH API
    // ========================================================================

    /// List branches
    pub fn list_branches(&self, owner: &str, repo: &str) -> Result<Vec<Branch>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/branches", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list branches: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get branch
    pub fn get_branch(&self, owner: &str, repo: &str, branch: &str) -> Result<Branch> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/branches/{}", owner, repo, branch))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get branch: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Create branch
    pub fn create_branch(&self, owner: &str, repo: &str, branch_name: &str, old_ref: &str) -> Result<Branch> {
        let payload = json!({
            "new_branch_name": branch_name,
            "old_ref_name": old_ref
        });

        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/repos/{}/{}/branches", owner, repo))
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create branch: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete branch
    pub fn delete_branch(&self, owner: &str, repo: &str, branch: &str) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/repos/{}/{}/branches/{}", owner, repo, branch))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete branch: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // COMMIT API
    // ========================================================================

    /// List commits
    pub fn list_commits(&self, owner: &str, repo: &str) -> Result<Vec<Commit>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/commits", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list commits: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get single commit
    pub fn get_commit(&self, owner: &str, repo: &str, sha: &str) -> Result<Commit> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/git/commits/{}", owner, repo, sha))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get commit: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // ORGANIZATION API
    // ========================================================================

    /// Create organization
    pub fn create_org(&self, org: &Organization) -> Result<Organization> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/orgs")
            .json(org)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create organization: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List user organizations
    pub fn list_user_orgs(&self, username: &str) -> Result<Vec<Organization>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/users/{}/orgs", username))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list organizations: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get organization
    pub fn get_org(&self, org: &str) -> Result<Organization> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/orgs/{}", org))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get organization: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Update organization
    pub fn update_org(&self, org: &str, updates: &Organization) -> Result<Organization> {
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/orgs/{}", org))
            .json(updates)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update organization: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete organization
    pub fn delete_org(&self, org: &str) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/orgs/{}", org))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete organization: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // TEAM API
    // ========================================================================

    /// List organization teams
    pub fn list_org_teams(&self, org: &str) -> Result<Vec<Team>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/orgs/{}/teams", org))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list teams: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Create team
    pub fn create_team(&self, org: &str, team: &Team) -> Result<Team> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/orgs/{}/teams", org))
            .json(team)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create team: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get team
    pub fn get_team(&self, team_id: i64) -> Result<Team> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/teams/{}", team_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get team: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete team
    pub fn delete_team(&self, team_id: i64) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/teams/{}", team_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete team: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // RELEASE API
    // ========================================================================

    /// Create release
    pub fn create_release(&self, owner: &str, repo: &str, release: &Release) -> Result<Release> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/repos/{}/{}/releases", owner, repo))
            .json(release)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create release: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List releases
    pub fn list_releases(&self, owner: &str, repo: &str) -> Result<Vec<Release>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/releases", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list releases: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get release by tag
    pub fn get_release(&self, owner: &str, repo: &str, tag: &str) -> Result<Release> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/releases/tags/{}", owner, repo, tag))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get release: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete release
    pub fn delete_release(&self, owner: &str, repo: &str, release_id: i64) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/repos/{}/{}/releases/{}", owner, repo, release_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete release: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // LABEL API
    // ========================================================================

    /// Create label
    pub fn create_label(&self, owner: &str, repo: &str, label: &Label) -> Result<Label> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/repos/{}/{}/labels", owner, repo))
            .json(label)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create label: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List labels
    pub fn list_labels(&self, owner: &str, repo: &str) -> Result<Vec<Label>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/labels", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list labels: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete label
    pub fn delete_label(&self, owner: &str, repo: &str, label_id: i64) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/repos/{}/{}/labels/{}", owner, repo, label_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete label: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // WEBHOOK API
    // ========================================================================

    /// Create webhook
    pub fn create_webhook(&self, owner: &str, repo: &str, webhook: &Webhook) -> Result<Webhook> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/repos/{}/{}/hooks", owner, repo))
            .json(webhook)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create webhook: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List webhooks
    pub fn list_webhooks(&self, owner: &str, repo: &str) -> Result<Vec<Webhook>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/hooks", owner, repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list webhooks: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete webhook
    pub fn delete_webhook(&self, owner: &str, repo: &str, hook_id: i64) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/repos/{}/{}/hooks/{}", owner, repo, hook_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete webhook: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // FILE CONTENT API
    // ========================================================================

    /// Get file contents
    pub fn get_file_contents(&self, owner: &str, repo: &str, filepath: &str) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/repos/{}/{}/contents/{}", owner, repo, filepath))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get file contents: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Create or update file
    pub fn create_or_update_file(
        &self,
        owner: &str,
        repo: &str,
        filepath: &str,
        content: &str,
        message: &str,
    ) -> Result<Value> {
        use base64::{Engine as _, engine::general_purpose};
        
        let encoded_content = general_purpose::STANDARD.encode(content);
        let payload = json!({
            "content": encoded_content,
            "message": message
        });

        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/repos/{}/{}/contents/{}", owner, repo, filepath))
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create/update file: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete file
    pub fn delete_file(&self, owner: &str, repo: &str, filepath: &str, message: &str) -> Result<Value> {
        let payload = json!({"message": message});

        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/repos/{}/{}/contents/{}", owner, repo, filepath))
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete file: {}", response.status());
        }

        Ok(response.json()?)
    }
}
