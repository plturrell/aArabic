/// Gitea API Integration
/// Handles repository creation, commits, and version control

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
pub struct GiteaRepository {
    pub id: i64,
    pub name: String,
    pub full_name: String,
    pub html_url: String,
    pub default_branch: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GiteaCommit {
    pub sha: String,
    pub commit: CommitDetails,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitDetails {
    pub message: String,
    pub author: Author,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Author {
    pub name: String,
    pub email: String,
    pub date: String,
}

pub struct GiteaClient {
    base_url: String,
    token: String,
    org: String,
    client: Client,
}

impl GiteaClient {
    pub fn new(base_url: &str, token: &str, org: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            token: token.to_string(),
            org: org.to_string(),
            client: Client::new(),
        }
    }

    pub fn get_org(&self) -> &str {
        &self.org
    }

    /// Ensure repository exists, create if not
    pub async fn ensure_repository(&self, repo_name: &str) -> Result<GiteaRepository> {
        // Check if repo exists
        let url = format!("{}/api/v1/repos/{}/{}", self.base_url, self.org, repo_name);
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("token {}", self.token))
            .send()
            .await?;

        if response.status().is_success() {
            return Ok(response.json().await?);
        }

        // Create repository
        let create_url = format!("{}/api/v1/orgs/{}/repos", self.base_url, self.org);
        
        let response = self.client
            .post(&create_url)
            .header("Authorization", format!("token {}", self.token))
            .json(&json!({
                "name": repo_name,
                "description": format!("Auto-generated feature: {}", repo_name),
                "private": false,
                "auto_init": true,
                "default_branch": "main"
            }))
            .send()
            .await
            .context("Failed to create repository")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to create repository: {}", error_text);
        }

        Ok(response.json().await?)
    }

    /// Commit multiple files to repository
    pub async fn commit_files(
        &self,
        repo_name: &str,
        branch: &str,
        files: &[(&str, &str)],
        message: &str,
    ) -> Result<String> {
        for (path, content) in files {
            self.commit_file(repo_name, branch, path, content, message).await?;
        }

        // Get latest commit SHA
        let commits = self.get_commits(repo_name, branch).await?;
        Ok(commits.first()
            .map(|c| c.sha.clone())
            .unwrap_or_else(|| "unknown".to_string()))
    }

    /// Commit single file
    async fn commit_file(
        &self,
        repo_name: &str,
        branch: &str,
        file_path: &str,
        content: &str,
        message: &str,
    ) -> Result<()> {
        let url = format!(
            "{}/api/v1/repos/{}/{}/contents/{}",
            self.base_url, self.org, repo_name, file_path
        );

        // Check if file exists to get SHA
        let existing = self.client
            .get(&url)
            .header("Authorization", format!("token {}", self.token))
            .query(&[("ref", branch)])
            .send()
            .await?;

        let sha = if existing.status().is_success() {
            let data: serde_json::Value = existing.json().await?;
            data.get("sha").and_then(|s| s.as_str()).map(|s| s.to_string())
        } else {
            None
        };

        // Base64 encode content
        let encoded = base64::encode(content);

        let mut body = json!({
            "message": message,
            "content": encoded,
            "branch": branch
        });

        if let Some(sha) = sha {
            body["sha"] = json!(sha);
        }

        let response = self.client
            .put(&url)
            .header("Authorization", format!("token {}", self.token))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Failed to commit file {}: {}", file_path, error_text);
        }

        Ok(())
    }

    /// List all repositories in organization
    pub async fn list_repositories(&self) -> Result<Vec<GiteaRepository>> {
        let url = format!("{}/api/v1/orgs/{}/repos", self.base_url, self.org);
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("token {}", self.token))
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list repositories: {}", response.status());
        }

        Ok(response.json().await?)
    }

    /// Get commit history for repository
    pub async fn get_commits(&self, repo_name: &str, branch: &str) -> Result<Vec<GiteaCommit>> {
        let url = format!(
            "{}/api/v1/repos/{}/{}/commits",
            self.base_url, self.org, repo_name
        );
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("token {}", self.token))
            .query(&[("sha", branch)])
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get commits: {}", response.status());
        }

        Ok(response.json().await?)
    }

    /// Get file content at specific commit
    pub async fn get_file_at_commit(
        &self,
        repo_name: &str,
        file_path: &str,
        commit_sha: &str,
    ) -> Result<String> {
        let url = format!(
            "{}/api/v1/repos/{}/{}/contents/{}",
            self.base_url, self.org, repo_name, file_path
        );
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("token {}", self.token))
            .query(&[("ref", commit_sha)])
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get file: {}", response.status());
        }

        let data: serde_json::Value = response.json().await?;
        let encoded = data.get("content")
            .and_then(|c| c.as_str())
            .context("No content in response")?;

        let decoded = base64::decode(encoded.replace("\n", ""))?;
        Ok(String::from_utf8(decoded)?)
    }
}

// Add base64 dependency
use base64::{Engine as _, engine::general_purpose};

fn base64_encode(data: &str) -> String {
    general_purpose::STANDARD.encode(data.as_bytes())
}

fn base64_decode(data: &str) -> Result<Vec<u8>> {
    Ok(general_purpose::STANDARD.decode(data)?)
}

// Update encode/decode calls
impl GiteaClient {
    fn encode_content(content: &str) -> String {
        base64_encode(content)
    }

    fn decode_content(encoded: &str) -> Result<String> {
        let decoded = base64_decode(&encoded.replace("\n", ""))?;
        Ok(String::from_utf8(decoded)?)
    }
}