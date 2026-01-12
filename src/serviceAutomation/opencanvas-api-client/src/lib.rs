use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete OpenCanvas Collaborative Editor API Client
pub struct OpenCanvasClient {
    base_url: String,
    api_key: Option<String>,
    client: Client,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Canvas {
    pub id: Option<String>,
    pub title: String,
    pub content: String,
    pub version: Option<i32>,
    #[serde(rename = "createdAt")]
    pub created_at: Option<String>,
    #[serde(rename = "updatedAt")]
    pub updated_at: Option<String>,
    pub owner: Option<String>,
    pub collaborators: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub id: Option<String>,
    pub canvas_id: String,
    pub author: String,
    pub content: String,
    pub position: Option<Position>,
    #[serde(rename = "createdAt")]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Revision {
    pub id: String,
    pub canvas_id: String,
    pub version: i32,
    pub content: String,
    pub author: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Share {
    pub canvas_id: String,
    pub user_email: String,
    pub permission: String, // "view", "edit", "admin"
}

impl OpenCanvasClient {
    /// Create new OpenCanvas client
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
            client: Client::new(),
        }
    }

    /// Create client with API key
    pub fn with_api_key(base_url: String, api_key: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: Some(api_key),
            client: Client::new(),
        }
    }

    /// Helper to build request
    fn request(&self, method: &str, endpoint: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}/api/{}", self.base_url, endpoint);
        let mut req = match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            "PUT" => self.client.put(&url),
            "PATCH" => self.client.patch(&url),
            "DELETE" => self.client.delete(&url),
            _ => self.client.get(&url),
        };
        
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        
        req
    }

    // ========================================================================
    // CANVAS OPERATIONS
    // ========================================================================

    /// List canvases
    pub fn list_canvases(&self) -> Result<Vec<Canvas>> {
        let response = self.request("GET", "canvases").send()?;
        Ok(response.json()?)
    }

    /// Get canvas
    pub fn get_canvas(&self, id: &str) -> Result<Canvas> {
        let response = self.request("GET", &format!("canvases/{}", id)).send()?;
        Ok(response.json()?)
    }

    /// Create canvas
    pub fn create_canvas(&self, canvas: &Canvas) -> Result<Canvas> {
        let response = self.request("POST", "canvases")
            .json(canvas)
            .send()?;
        Ok(response.json()?)
    }

    /// Update canvas
    pub fn update_canvas(&self, id: &str, canvas: &Canvas) -> Result<Canvas> {
        let response = self.request("PUT", &format!("canvases/{}", id))
            .json(canvas)
            .send()?;
        Ok(response.json()?)
    }

    /// Delete canvas
    pub fn delete_canvas(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("canvases/{}", id)).send()?;
        Ok(())
    }

    /// Duplicate canvas
    pub fn duplicate_canvas(&self, id: &str, new_title: &str) -> Result<Canvas> {
        let canvas = self.get_canvas(id)?;
        let mut new_canvas = canvas;
        new_canvas.id = None;
        new_canvas.title = new_title.to_string();
        new_canvas.version = Some(1);
        self.create_canvas(&new_canvas)
    }

    // ========================================================================
    // COLLABORATION OPERATIONS
    // ========================================================================

    /// Share canvas with user
    pub fn share_canvas(&self, canvas_id: &str, email: &str, permission: &str) -> Result<()> {
        let share = Share {
            canvas_id: canvas_id.to_string(),
            user_email: email.to_string(),
            permission: permission.to_string(),
        };
        self.request("POST", &format!("canvases/{}/share", canvas_id))
            .json(&share)
            .send()?;
        Ok(())
    }

    /// Remove collaborator
    pub fn remove_collaborator(&self, canvas_id: &str, email: &str) -> Result<()> {
        self.request("DELETE", &format!("canvases/{}/collaborators/{}", canvas_id, email))
            .send()?;
        Ok(())
    }

    /// List collaborators
    pub fn list_collaborators(&self, canvas_id: &str) -> Result<Vec<String>> {
        let canvas = self.get_canvas(canvas_id)?;
        Ok(canvas.collaborators.unwrap_or_default())
    }

    // ========================================================================
    // COMMENT OPERATIONS
    // ========================================================================

    /// Add comment
    pub fn add_comment(&self, comment: &Comment) -> Result<Comment> {
        let response = self.request("POST", &format!("canvases/{}/comments", comment.canvas_id))
            .json(comment)
            .send()?;
        Ok(response.json()?)
    }

    /// Get comments
    pub fn get_comments(&self, canvas_id: &str) -> Result<Vec<Comment>> {
        let response = self.request("GET", &format!("canvases/{}/comments", canvas_id)).send()?;
        Ok(response.json()?)
    }

    /// Delete comment
    pub fn delete_comment(&self, canvas_id: &str, comment_id: &str) -> Result<()> {
        self.request("DELETE", &format!("canvases/{}/comments/{}", canvas_id, comment_id))
            .send()?;
        Ok(())
    }

    // ========================================================================
    // VERSION CONTROL OPERATIONS
    // ========================================================================

    /// Get canvas revisions
    pub fn get_revisions(&self, canvas_id: &str) -> Result<Vec<Revision>> {
        let response = self.request("GET", &format!("canvases/{}/revisions", canvas_id)).send()?;
        Ok(response.json()?)
    }

    /// Get specific revision
    pub fn get_revision(&self, canvas_id: &str, version: i32) -> Result<Revision> {
        let response = self.request("GET", &format!("canvases/{}/revisions/{}", canvas_id, version))
            .send()?;
        Ok(response.json()?)
    }

    /// Restore canvas to revision
    pub fn restore_revision(&self, canvas_id: &str, version: i32) -> Result<Canvas> {
        let revision = self.get_revision(canvas_id, version)?;
        let mut canvas = self.get_canvas(canvas_id)?;
        canvas.content = revision.content;
        self.update_canvas(canvas_id, &canvas)
    }

    // ========================================================================
    // EXPORT OPERATIONS
    // ========================================================================

    /// Export canvas as markdown
    pub fn export_markdown(&self, canvas_id: &str) -> Result<String> {
        let canvas = self.get_canvas(canvas_id)?;
        Ok(format!("# {}\n\n{}", canvas.title, canvas.content))
    }

    /// Export canvas as JSON
    pub fn export_json(&self, canvas_id: &str) -> Result<String> {
        let canvas = self.get_canvas(canvas_id)?;
        Ok(serde_json::to_string_pretty(&canvas)?)
    }

    /// Export canvas as HTML
    pub fn export_html(&self, canvas_id: &str) -> Result<String> {
        let canvas = self.get_canvas(canvas_id)?;
        Ok(format!(
            "<!DOCTYPE html>\n<html>\n<head>\n<title>{}</title>\n</head>\n<body>\n{}\n</body>\n</html>",
            canvas.title, canvas.content
        ))
    }

    // ========================================================================
    // SEARCH OPERATIONS
    // ========================================================================

    /// Search canvases
    pub fn search_canvases(&self, query: &str) -> Result<Vec<Canvas>> {
        let all_canvases = self.list_canvases()?;
        Ok(all_canvases
            .into_iter()
            .filter(|c| {
                c.title.to_lowercase().contains(&query.to_lowercase())
                    || c.content.to_lowercase().contains(&query.to_lowercase())
            })
            .collect())
    }

    // ========================================================================
    // STATISTICS OPERATIONS
    // ========================================================================

    /// Get canvas statistics
    pub fn get_canvas_stats(&self, canvas_id: &str) -> Result<HashMap<String, usize>> {
        let canvas = self.get_canvas(canvas_id)?;
        let comments = self.get_comments(canvas_id).unwrap_or_default();
        let revisions = self.get_revisions(canvas_id).unwrap_or_default();
        
        let mut stats = HashMap::new();
        stats.insert("word_count".to_string(), canvas.content.split_whitespace().count());
        stats.insert("char_count".to_string(), canvas.content.len());
        stats.insert("comment_count".to_string(), comments.len());
        stats.insert("revision_count".to_string(), revisions.len());
        stats.insert("collaborator_count".to_string(), canvas.collaborators.unwrap_or_default().len());
        
        Ok(stats)
    }
}
