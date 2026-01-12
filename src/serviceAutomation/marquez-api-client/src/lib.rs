use anyhow::Result;
use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Complete Marquez API Client
/// Metadata service for data lineage tracking
pub struct MarquezClient {
    client: Client,
    base_url: String,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Namespace {
    pub name: String,
    pub owner_name: Option<String>,
    pub description: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub namespace: String,
    pub r#type: String,
    pub physical_name: Option<String>,
    pub description: Option<String>,
    pub source_name: Option<String>,
    pub fields: Option<Vec<Field>>,
    pub tags: Option<Vec<String>>,
    pub facets: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub r#type: String,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub name: String,
    pub namespace: String,
    pub r#type: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub inputs: Option<Vec<DatasetRef>>,
    pub outputs: Option<Vec<DatasetRef>>,
    pub context: Option<Value>,
    pub facets: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRef {
    pub namespace: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: Option<String>,
    pub state: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub duration_ms: Option<i64>,
    pub args: Option<Value>,
    pub facets: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvent {
    pub event_type: String,
    pub event_time: DateTime<Utc>,
    pub run: Run,
    pub job: Job,
    pub inputs: Option<Vec<Dataset>>,
    pub outputs: Option<Vec<Dataset>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub name: String,
    pub r#type: String,
    pub connection_url: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageGraph {
    pub graph: Vec<LineageNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    pub id: String,
    pub r#type: String,
    pub data: Value,
    pub in_edges: Option<Vec<String>>,
    pub out_edges: Option<Vec<String>>,
}

// ============================================================================
// CLIENT IMPLEMENTATION
// ============================================================================

impl MarquezClient {
    /// Create new Marquez client
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Build request
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        self.client.request(method, &url)
            .header("Content-Type", "application/json")
    }

    // ========================================================================
    // NAMESPACE API
    // ========================================================================

    /// Create namespace
    pub fn create_namespace(&self, namespace: &Namespace) -> Result<Namespace> {
        let response = self
            .request(reqwest::Method::PUT, &format!("/api/v1/namespaces/{}", namespace.name))
            .json(namespace)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create namespace: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List namespaces
    pub fn list_namespaces(&self) -> Result<Vec<Namespace>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/namespaces")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list namespaces: {}", response.status());
        }

        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["namespaces"].clone())?)
    }

    /// Get namespace
    pub fn get_namespace(&self, name: &str) -> Result<Namespace> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/namespaces/{}", name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get namespace: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // DATASET API
    // ========================================================================

    /// Create dataset
    pub fn create_dataset(&self, namespace: &str, dataset: &Dataset) -> Result<Dataset> {
        let response = self
            .request(reqwest::Method::PUT, &format!("/api/v1/namespaces/{}/datasets/{}", namespace, dataset.name))
            .json(dataset)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create dataset: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List datasets
    pub fn list_datasets(&self, namespace: &str) -> Result<Vec<Dataset>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/namespaces/{}/datasets", namespace))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list datasets: {}", response.status());
        }

        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["datasets"].clone())?)
    }

    /// Get dataset
    pub fn get_dataset(&self, namespace: &str, name: &str) -> Result<Dataset> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/namespaces/{}/datasets/{}", namespace, name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get dataset: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get dataset versions
    pub fn get_dataset_versions(&self, namespace: &str, name: &str) -> Result<Vec<Value>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/namespaces/{}/datasets/{}/versions", namespace, name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get dataset versions: {}", response.status());
        }

        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["versions"].clone())?)
    }

    /// Tag dataset
    pub fn tag_dataset(&self, namespace: &str, name: &str, tag: &str) -> Result<Dataset> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/namespaces/{}/datasets/{}/tags/{}", namespace, name, tag))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to tag dataset: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // JOB API
    // ========================================================================

    /// Create job
    pub fn create_job(&self, namespace: &str, job: &Job) -> Result<Job> {
        let response = self
            .request(reqwest::Method::PUT, &format!("/api/v1/namespaces/{}/jobs/{}", namespace, job.name))
            .json(job)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create job: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List jobs
    pub fn list_jobs(&self, namespace: &str) -> Result<Vec<Job>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/namespaces/{}/jobs", namespace))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list jobs: {}", response.status());
        }

        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["jobs"].clone())?)
    }

    /// Get job
    pub fn get_job(&self, namespace: &str, name: &str) -> Result<Job> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/namespaces/{}/jobs/{}", namespace, name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get job: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get job runs
    pub fn get_job_runs(&self, namespace: &str, name: &str) -> Result<Vec<Run>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/namespaces/{}/jobs/{}/runs", namespace, name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get job runs: {}", response.status());
        }

        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["runs"].clone())?)
    }

    // ========================================================================
    // RUN API
    // ========================================================================

    /// Create run
    pub fn create_run(&self, namespace: &str, job_name: &str) -> Result<Run> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/namespaces/{}/jobs/{}/runs", namespace, job_name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create run: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get run
    pub fn get_run(&self, run_id: &str) -> Result<Run> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/runs/{}", run_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get run: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Mark run as started
    pub fn start_run(&self, run_id: &str) -> Result<Run> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/runs/{}/start", run_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to start run: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Mark run as completed
    pub fn complete_run(&self, run_id: &str) -> Result<Run> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/runs/{}/complete", run_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to complete run: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Mark run as failed
    pub fn fail_run(&self, run_id: &str) -> Result<Run> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/runs/{}/fail", run_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to fail run: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Mark run as aborted
    pub fn abort_run(&self, run_id: &str) -> Result<Run> {
        let response = self
            .request(reqwest::Method::POST, &format!("/api/v1/runs/{}/abort", run_id))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to abort run: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // SOURCE API
    // ========================================================================

    /// Create source
    pub fn create_source(&self, source: &Source) -> Result<Source> {
        let response = self
            .request(reqwest::Method::PUT, &format!("/api/v1/sources/{}", source.name))
            .json(source)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create source: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List sources
    pub fn list_sources(&self) -> Result<Vec<Source>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/sources")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list sources: {}", response.status());
        }

        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["sources"].clone())?)
    }

    /// Get source
    pub fn get_source(&self, name: &str) -> Result<Source> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/sources/{}", name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get source: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // TAG API
    // ========================================================================

    /// Create tag
    pub fn create_tag(&self, tag: &Tag) -> Result<Tag> {
        let response = self
            .request(reqwest::Method::PUT, &format!("/api/v1/tags/{}", tag.name))
            .json(tag)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create tag: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List tags
    pub fn list_tags(&self) -> Result<Vec<Tag>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/tags")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list tags: {}", response.status());
        }

        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["tags"].clone())?)
    }

    // ========================================================================
    // LINEAGE API
    // ========================================================================

    /// Get dataset lineage
    pub fn get_dataset_lineage(&self, namespace: &str, name: &str, depth: Option<i32>) -> Result<LineageGraph> {
        let mut url = format!("/api/v1/lineage?nodeId=dataset:{}:{}", namespace, name);
        if let Some(d) = depth {
            url.push_str(&format!("&depth={}", d));
        }

        let response = self
            .request(reqwest::Method::GET, &url)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get lineage: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get job lineage
    pub fn get_job_lineage(&self, namespace: &str, name: &str, depth: Option<i32>) -> Result<LineageGraph> {
        let mut url = format!("/api/v1/lineage?nodeId=job:{}:{}", namespace, name);
        if let Some(d) = depth {
            url.push_str(&format!("&depth={}", d));
        }

        let response = self
            .request(reqwest::Method::GET, &url)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get lineage: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // SEARCH API
    // ========================================================================

    /// Search
    pub fn search(&self, query: &str, filter: Option<&str>) -> Result<Vec<Value>> {
        let mut url = format!("/api/v1/search?q={}", query);
        if let Some(f) = filter {
            url.push_str(&format!("&filter={}", f));
        }

        let response = self
            .request(reqwest::Method::GET, &url)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to search: {}", response.status());
        }

        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["results"].clone())?)
    }

    // ========================================================================
    // EVENTS API
    // ========================================================================

    /// Send run event
    pub fn send_event(&self, event: &RunEvent) -> Result<()> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/events/lineage")
            .json(event)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to send event: {}", response.status());
        }

        Ok(())
    }
}
