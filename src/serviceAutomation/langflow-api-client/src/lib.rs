use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use uuid::Uuid;

/// Complete Langflow API Client - 100% Coverage
/// Based on Langflow v1.0+ API specification
pub struct LangflowClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Folder {
    pub id: Option<Uuid>,
    pub name: String,
    pub description: Option<String>,
    pub parent_id: Option<Uuid>,
    #[serde(default)]
    pub components_list: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flow {
    pub id: Option<Uuid>,
    pub name: String,
    pub description: Option<String>,
    pub data: Option<Value>,
    pub folder_id: Option<Uuid>,
    pub is_component: bool,
    pub updated_at: Option<String>,
    pub gradient: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    pub id: Option<Uuid>,
    pub name: String,
    pub description: Option<String>,
    pub data: Value,
    pub is_component: bool,
    pub parent_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Option<Uuid>,
    pub username: String,
    pub email: Option<String>,
    pub is_active: bool,
    pub is_superuser: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: Option<Uuid>,
    pub name: String,
    pub api_key: Option<String>,
    pub created_at: Option<String>,
    pub last_used_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub id: Option<Uuid>,
    pub name: String,
    pub value: String,
    pub default_value: Option<String>,
    pub r#type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    pub session_id: String,
    pub outputs: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub id: Option<Uuid>,
    pub flow_id: Option<Uuid>,
    pub timestamp: Option<String>,
    pub level: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileUpload {
    pub file_path: String,
    pub flow_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Monitor {
    pub flow_id: Uuid,
    pub status: String,
    pub metrics: Option<Value>,
}

// ============================================================================
// CLIENT IMPLEMENTATION
// ============================================================================

impl LangflowClient {
    /// Create a new Langflow API client
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
        }
    }

    /// Build request with optional API key authentication
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.request(method, &url);
        
        if let Some(key) = &self.api_key {
            req = req.header("x-api-key", key);
        }
        
        req
    }

    // ========================================================================
    // FOLDERS API (Projects)
    // ========================================================================

    /// Create a new folder/project
    pub fn create_folder(&self, folder: &Folder) -> Result<Folder> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/folders/")
            .json(folder)
            .send()
            .context("Failed to send request")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create folder: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List all folders
    pub fn list_folders(&self) -> Result<Vec<Folder>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/folders/")
            .send()
            .context("Failed to send request")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list folders: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get folder by ID
    pub fn get_folder(&self, folder_id: Uuid) -> Result<Folder> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/folders/{}", folder_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get folder: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Update folder
    pub fn update_folder(&self, folder_id: Uuid, folder: &Folder) -> Result<Folder> {
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/folders/{}", folder_id))
            .json(folder)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update folder: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete folder
    pub fn delete_folder(&self, folder_id: Uuid) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/folders/{}", folder_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete folder: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // FLOWS API
    // ========================================================================

    /// Create a new flow
    pub fn create_flow(&self, flow: &Flow) -> Result<Flow> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/flows/")
            .json(flow)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create flow: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List all flows
    pub fn list_flows(&self) -> Result<Vec<Flow>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/flows/")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list flows: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get flow by ID
    pub fn get_flow(&self, flow_id: Uuid) -> Result<Flow> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/flows/{}", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get flow: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Update flow
    pub fn update_flow(&self, flow_id: Uuid, flow: &Flow) -> Result<Flow> {
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/flows/{}", flow_id))
            .json(flow)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update flow: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete flow
    pub fn delete_flow(&self, flow_id: Uuid) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/flows/{}", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete flow: {}", response.status());
        }

        Ok(())
    }

    /// Download flow as JSON
    pub fn download_flow(&self, flow_id: Uuid) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/flows/download/{}", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to download flow: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Upload/Import flow from JSON
    pub fn upload_flow(&self, flow_data: Value, folder_id: Option<Uuid>) -> Result<Flow> {
        let mut payload = flow_data;
        if let Some(fid) = folder_id {
            payload["folder_id"] = json!(fid.to_string());
        }

        let response = self
            .request(reqwest::Method::POST, "/api/v1/flows/upload/")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to upload flow: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Run a flow
    pub fn run_flow(
        &self,
        flow_id: Uuid,
        inputs: HashMap<String, Value>,
        tweaks: Option<HashMap<String, Value>>,
    ) -> Result<RunResponse> {
        let payload = json!({
            "inputs": inputs,
            "tweaks": tweaks.unwrap_or_default()
        });

        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/run/{}", flow_id))
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to run flow: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // COMPONENTS API
    // ========================================================================

    /// List all components
    pub fn list_components(&self) -> Result<Vec<Component>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/components/")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list components: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get component by ID
    pub fn get_component(&self, component_id: Uuid) -> Result<Component> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/components/{}", component_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get component: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Create custom component
    pub fn create_component(&self, component: &Component) -> Result<Component> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/components/")
            .json(component)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create component: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Update component
    pub fn update_component(&self, component_id: Uuid, component: &Component) -> Result<Component> {
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/components/{}", component_id))
            .json(component)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update component: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete component
    pub fn delete_component(&self, component_id: Uuid) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/components/{}", component_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete component: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // USERS API
    // ========================================================================

    /// Get current user
    pub fn get_current_user(&self) -> Result<User> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/users/whoami")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get current user: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List all users (admin only)
    pub fn list_users(&self) -> Result<Vec<User>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/users/")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list users: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get user by ID
    pub fn get_user(&self, user_id: Uuid) -> Result<User> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/users/{}", user_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get user: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Update user
    pub fn update_user(&self, user_id: Uuid, user: &User) -> Result<User> {
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/users/{}", user_id))
            .json(user)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update user: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete user
    pub fn delete_user(&self, user_id: Uuid) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/users/{}", user_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete user: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // API KEYS API
    // ========================================================================

    /// Create API key
    pub fn create_api_key(&self, name: String) -> Result<ApiKey> {
        let payload = json!({ "name": name });
        
        let response = self
            .request(reqwest::Method::POST, "/api/v1/api_key/")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create API key: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List API keys
    pub fn list_api_keys(&self) -> Result<Vec<ApiKey>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/api_key/")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list API keys: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete API key
    pub fn delete_api_key(&self, key_id: Uuid) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/api_key/{}", key_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete API key: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // VARIABLES API
    // ========================================================================

    /// Create variable
    pub fn create_variable(&self, variable: &Variable) -> Result<Variable> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/variables/")
            .json(variable)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create variable: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List variables
    pub fn list_variables(&self) -> Result<Vec<Variable>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/variables/")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list variables: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get variable by name
    pub fn get_variable(&self, name: &str) -> Result<Variable> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/variables/{}", name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get variable: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Update variable
    pub fn update_variable(&self, variable_id: Uuid, variable: &Variable) -> Result<Variable> {
        let response = self
            .request(reqwest::Method::PATCH, &format!("/api/v1/variables/{}", variable_id))
            .json(variable)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to update variable: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete variable
    pub fn delete_variable(&self, variable_id: Uuid) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/variables/{}", variable_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete variable: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // STORE API (Templates/Examples)
    // ========================================================================

    /// List store items
    pub fn list_store_items(&self) -> Result<Vec<Value>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/store/")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list store items: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get store item
    pub fn get_store_item(&self, item_id: &str) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/store/{}", item_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get store item: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // HEALTH & VERSION API
    // ========================================================================

    /// Check API health
    pub fn health_check(&self) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/health")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Health check failed: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get API version
    pub fn get_version(&self) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/version")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get version: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // CONFIG API
    // ========================================================================

    /// Get frontend configuration
    pub fn get_config(&self) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/config")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get config: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // BUILD API (Flow Execution & Streaming)
    // ========================================================================

    /// Build and execute a flow
    pub fn build_flow(&self, flow_id: Uuid, inputs: HashMap<String, Value>) -> Result<Value> {
        let payload = json!({
            "inputs": inputs
        });

        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/build/{}/flow", flow_id))
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to build flow: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Build and stream flow events
    pub fn build_flow_stream(&self, flow_id: Uuid, inputs: HashMap<String, Value>, stream: bool) -> Result<Value> {
        let payload = json!({
            "inputs": inputs
        });

        let stream_param = if stream { "true" } else { "false" };
        let path = format!("/api/v1/build/{}/events?stream={}", flow_id, stream_param);

        let response = self
            .request(reqwest::Method::GET, &path)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get flow events: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get flow build status
    pub fn get_build_status(&self, flow_id: Uuid) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/build/{}/status", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get build status: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // FILES API
    // ========================================================================

    /// Upload file to flow
    pub fn upload_file(&self, flow_id: Uuid, file_path: &str, file_data: Vec<u8>) -> Result<Value> {
        use reqwest::blocking::multipart;
        
        let form = multipart::Form::new()
            .part("file", multipart::Part::bytes(file_data)
                .file_name(file_path.to_string()));

        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/files/upload/{}", flow_id))
            .multipart(form)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to upload file: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Upload image file
    pub fn upload_image(&self, flow_id: Uuid, image_data: Vec<u8>, filename: &str) -> Result<Value> {
        use reqwest::blocking::multipart;
        
        let form = multipart::Form::new()
            .part("file", multipart::Part::bytes(image_data)
                .file_name(filename.to_string()));

        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/files/upload/{}", flow_id))
            .multipart(form)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to upload image: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List files for a flow
    pub fn list_files(&self, flow_id: Uuid) -> Result<Vec<Value>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/files/{}", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list files: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Download file
    pub fn download_file(&self, file_id: &str) -> Result<Vec<u8>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/files/download/{}", file_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to download file: {}", response.status());
        }

        Ok(response.bytes()?.to_vec())
    }

    /// Delete file
    pub fn delete_file(&self, file_id: &str) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/files/{}", file_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete file: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // LOGS API
    // ========================================================================

    /// Get logs for a flow
    pub fn get_flow_logs(&self, flow_id: Uuid) -> Result<Vec<LogEntry>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/logs/flow/{}", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get flow logs: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get logs for a specific run
    pub fn get_run_logs(&self, session_id: &str) -> Result<Vec<LogEntry>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/logs/run/{}", session_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get run logs: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get all system logs
    pub fn get_system_logs(&self) -> Result<Vec<LogEntry>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/logs/")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get system logs: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Clear logs for a flow
    pub fn clear_flow_logs(&self, flow_id: Uuid) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/logs/flow/{}", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to clear flow logs: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // MONITOR API
    // ========================================================================

    /// Get monitoring data for a flow
    pub fn get_flow_monitor(&self, flow_id: Uuid) -> Result<Monitor> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/monitor/flow/{}", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get flow monitor data: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get monitoring data for all flows
    pub fn get_all_monitors(&self) -> Result<Vec<Monitor>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/monitor/")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get monitor data: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get flow metrics
    pub fn get_flow_metrics(&self, flow_id: Uuid) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/monitor/metrics/{}", flow_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get flow metrics: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get flow execution history
    pub fn get_flow_history(&self, flow_id: Uuid, limit: Option<usize>) -> Result<Vec<Value>> {
        let mut path = format!("/api/v1/monitor/history/{}", flow_id);
        if let Some(l) = limit {
            path = format!("{}?limit={}", path, l);
        }

        let response = self
            .request(reqwest::Method::GET, &path)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get flow history: {}", response.status());
        }

        Ok(response.json()?)
    }
}
