use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete n8n Workflow Automation API Client
pub struct N8nClient {
    base_url: String,
    api_key: String,
    client: Client,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: Option<String>,
    pub name: String,
    pub active: bool,
    pub nodes: Vec<serde_json::Value>,
    pub connections: HashMap<String, serde_json::Value>,
    pub settings: Option<WorkflowSettings>,
    #[serde(rename = "createdAt")]
    pub created_at: Option<String>,
    #[serde(rename = "updatedAt")]
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSettings {
    #[serde(rename = "saveExecutionProgress")]
    pub save_execution_progress: Option<bool>,
    #[serde(rename = "saveManualExecutions")]
    pub save_manual_executions: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Execution {
    pub id: String,
    pub finished: bool,
    pub mode: String,
    #[serde(rename = "startedAt")]
    pub started_at: String,
    #[serde(rename = "stoppedAt")]
    pub stopped_at: Option<String>,
    #[serde(rename = "workflowId")]
    pub workflow_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credential {
    pub id: Option<String>,
    pub name: String,
    #[serde(rename = "type")]
    pub cred_type: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub id: Option<String>,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowList {
    pub data: Vec<Workflow>,
    pub nextCursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionList {
    pub data: Vec<Execution>,
    pub nextCursor: Option<String>,
}

impl N8nClient {
    /// Create new n8n client
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            client: Client::new(),
        }
    }

    /// Helper to build request
    fn request(&self, method: &str, endpoint: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}/api/v1/{}", self.base_url, endpoint);
        let req = match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            "PUT" => self.client.put(&url),
            "PATCH" => self.client.patch(&url),
            "DELETE" => self.client.delete(&url),
            _ => self.client.get(&url),
        };
        req.header("X-N8N-API-KEY", &self.api_key)
    }

    // ========================================================================
    // WORKFLOW OPERATIONS
    // ========================================================================

    /// List workflows
    pub fn list_workflows(&self, active: Option<bool>) -> Result<Vec<Workflow>> {
        let mut endpoint = "workflows".to_string();
        if let Some(a) = active {
            endpoint.push_str(&format!("?active={}", a));
        }
        
        let response = self.request("GET", &endpoint).send()?;
        let list: WorkflowList = response.json()?;
        Ok(list.data)
    }

    /// Get workflow
    pub fn get_workflow(&self, id: &str) -> Result<Workflow> {
        let response = self.request("GET", &format!("workflows/{}", id)).send()?;
        Ok(response.json()?)
    }

    /// Create workflow
    pub fn create_workflow(&self, workflow: &Workflow) -> Result<Workflow> {
        let response = self.request("POST", "workflows")
            .json(workflow)
            .send()?;
        Ok(response.json()?)
    }

    /// Update workflow
    pub fn update_workflow(&self, id: &str, workflow: &Workflow) -> Result<Workflow> {
        let response = self.request("PUT", &format!("workflows/{}", id))
            .json(workflow)
            .send()?;
        Ok(response.json()?)
    }

    /// Delete workflow
    pub fn delete_workflow(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("workflows/{}", id)).send()?;
        Ok(())
    }

    /// Activate workflow
    pub fn activate_workflow(&self, id: &str) -> Result<Workflow> {
        let response = self.request("PATCH", &format!("workflows/{}/activate", id)).send()?;
        Ok(response.json()?)
    }

    /// Deactivate workflow
    pub fn deactivate_workflow(&self, id: &str) -> Result<Workflow> {
        let response = self.request("PATCH", &format!("workflows/{}/deactivate", id)).send()?;
        Ok(response.json()?)
    }

    // ========================================================================
    // EXECUTION OPERATIONS
    // ========================================================================

    /// List executions
    pub fn list_executions(&self, workflow_id: Option<&str>) -> Result<Vec<Execution>> {
        let mut endpoint = "executions".to_string();
        if let Some(wid) = workflow_id {
            endpoint.push_str(&format!("?workflowId={}", wid));
        }
        
        let response = self.request("GET", &endpoint).send()?;
        let list: ExecutionList = response.json()?;
        Ok(list.data)
    }

    /// Get execution
    pub fn get_execution(&self, id: &str) -> Result<Execution> {
        let response = self.request("GET", &format!("executions/{}", id)).send()?;
        Ok(response.json()?)
    }

    /// Delete execution
    pub fn delete_execution(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("executions/{}", id)).send()?;
        Ok(())
    }

    /// Execute workflow
    pub fn execute_workflow(&self, id: &str) -> Result<Execution> {
        let response = self.request("POST", &format!("workflows/{}/execute", id)).send()?;
        Ok(response.json()?)
    }

    /// Execute workflow with data
    pub fn execute_workflow_with_data(
        &self,
        id: &str,
        data: serde_json::Value,
    ) -> Result<Execution> {
        let response = self.request("POST", &format!("workflows/{}/execute", id))
            .json(&data)
            .send()?;
        Ok(response.json()?)
    }

    // ========================================================================
    // CREDENTIAL OPERATIONS
    // ========================================================================

    /// List credentials
    pub fn list_credentials(&self) -> Result<Vec<Credential>> {
        let response = self.request("GET", "credentials").send()?;
        let data: serde_json::Value = response.json()?;
        Ok(serde_json::from_value(data["data"].clone())?)
    }

    /// Get credential
    pub fn get_credential(&self, id: &str) -> Result<Credential> {
        let response = self.request("GET", &format!("credentials/{}", id)).send()?;
        Ok(response.json()?)
    }

    /// Create credential
    pub fn create_credential(&self, credential: &Credential) -> Result<Credential> {
        let response = self.request("POST", "credentials")
            .json(credential)
            .send()?;
        Ok(response.json()?)
    }

    /// Update credential
    pub fn update_credential(&self, id: &str, credential: &Credential) -> Result<Credential> {
        let response = self.request("PUT", &format!("credentials/{}", id))
            .json(credential)
            .send()?;
        Ok(response.json()?)
    }

    /// Delete credential
    pub fn delete_credential(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("credentials/{}", id)).send()?;
        Ok(())
    }

    // ========================================================================
    // TAG OPERATIONS
    // ========================================================================

    /// List tags
    pub fn list_tags(&self) -> Result<Vec<Tag>> {
        let response = self.request("GET", "tags").send()?;
        let data: serde_json::Value = response.json()?;
        Ok(serde_json::from_value(data["data"].clone())?)
    }

    /// Get tag
    pub fn get_tag(&self, id: &str) -> Result<Tag> {
        let response = self.request("GET", &format!("tags/{}", id)).send()?;
        Ok(response.json()?)
    }

    /// Create tag
    pub fn create_tag(&self, name: &str) -> Result<Tag> {
        let tag = Tag {
            id: None,
            name: name.to_string(),
        };
        let response = self.request("POST", "tags")
            .json(&tag)
            .send()?;
        Ok(response.json()?)
    }

    /// Update tag
    pub fn update_tag(&self, id: &str, name: &str) -> Result<Tag> {
        let tag = Tag {
            id: Some(id.to_string()),
            name: name.to_string(),
        };
        let response = self.request("PUT", &format!("tags/{}", id))
            .json(&tag)
            .send()?;
        Ok(response.json()?)
    }

    /// Delete tag
    pub fn delete_tag(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("tags/{}", id)).send()?;
        Ok(())
    }

    // ========================================================================
    // UTILITY OPERATIONS
    // ========================================================================

    /// Get workflow statistics
    pub fn get_workflow_stats(&self, id: &str) -> Result<HashMap<String, usize>> {
        let executions = self.list_executions(Some(id))?;
        let mut stats = HashMap::new();
        
        stats.insert("total_executions".to_string(), executions.len());
        stats.insert(
            "finished".to_string(),
            executions.iter().filter(|e| e.finished).count(),
        );
        stats.insert(
            "running".to_string(),
            executions.iter().filter(|e| !e.finished).count(),
        );
        
        Ok(stats)
    }

    /// Health check
    pub fn health_check(&self) -> Result<bool> {
        let response = self.client
            .get(&format!("{}/healthz", self.base_url))
            .send()?;
        Ok(response.status().is_success())
    }
}
