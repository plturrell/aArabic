//! Memgraph API Client
//!
//! Rust client library for interacting with Memgraph graph database.
//! Provides methods for executing Cypher queries, managing graph operations,
//! and health monitoring.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Memgraph API client
#[derive(Clone)]
pub struct MemgraphClient {
    base_url: String,
    client: Client,
}

/// Query request payload
#[derive(Debug, Serialize)]
struct QueryRequest {
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

/// Query response from Memgraph
#[derive(Debug, Deserialize)]
pub struct QueryResponse {
    pub columns: Vec<String>,
    pub data: Vec<Vec<serde_json::Value>>,
    #[serde(default)]
    pub metadata: Option<QueryMetadata>,
}

/// Query metadata
#[derive(Debug, Deserialize)]
pub struct QueryMetadata {
    pub execution_time: Option<f64>,
    pub rows_affected: Option<usize>,
}

/// Node in the graph
#[derive(Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: Option<String>,
    pub labels: Vec<String>,
    pub properties: serde_json::Map<String, serde_json::Value>,
}

/// Edge in the graph
#[derive(Debug, Serialize, Deserialize)]
pub struct Edge {
    pub id: Option<String>,
    pub from: String,
    pub to: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    pub properties: serde_json::Map<String, serde_json::Value>,
}

/// Health check response
#[derive(Debug, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub version: Option<String>,
    pub uptime: Option<u64>,
}

impl MemgraphClient {
    /// Create a new Memgraph client
    ///
    /// # Arguments
    /// * `base_url` - Base URL of Memgraph instance (e.g., "http://localhost:7687")
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            base_url: base_url.into(),
            client,
        })
    }

    /// Execute a Cypher query
    ///
    /// # Arguments
    /// * `query` - Cypher query string
    /// * `parameters` - Optional query parameters
    pub async fn execute_query(
        &self,
        query: &str,
        parameters: Option<serde_json::Value>,
    ) -> Result<QueryResponse> {
        let url = format!("{}/query", self.base_url);
        let request = QueryRequest {
            query: query.to_string(),
            parameters,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send query")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Query failed with status {}: {}", status, error_text);
        }

        response
            .json::<QueryResponse>()
            .await
            .context("Failed to parse query response")
    }

    /// Execute a simple query without parameters
    pub async fn query(&self, query: &str) -> Result<QueryResponse> {
        self.execute_query(query, None).await
    }

    /// Create a node in the graph
    ///
    /// # Arguments
    /// * `labels` - Node labels
    /// * `properties` - Node properties as JSON object
    pub async fn create_node(
        &self,
        labels: &[&str],
        properties: serde_json::Value,
    ) -> Result<QueryResponse> {
        let labels_str = labels.join(":");
        let query = format!(
            "CREATE (n:{}) SET n = $props RETURN n",
            labels_str
        );
        
        let params = serde_json::json!({ "props": properties });
        self.execute_query(&query, Some(params)).await
    }

    /// Create an edge between two nodes
    ///
    /// # Arguments
    /// * `from_id` - Source node ID
    /// * `to_id` - Target node ID
    /// * `edge_type` - Type of the relationship
    /// * `properties` - Edge properties as JSON object
    pub async fn create_edge(
        &self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        properties: Option<serde_json::Value>,
    ) -> Result<QueryResponse> {
        let props_clause = if properties.is_some() {
            format!(" SET r = $props")
        } else {
            String::new()
        };

        let query = format!(
            "MATCH (a), (b) WHERE id(a) = $from_id AND id(b) = $to_id \
             CREATE (a)-[r:{}]->(b){} RETURN r",
            edge_type, props_clause
        );

        let mut params = serde_json::json!({
            "from_id": from_id,
            "to_id": to_id
        });

        if let Some(props) = properties {
            params["props"] = props;
        }

        self.execute_query(&query, Some(params)).await
    }

    /// Get graph schema
    pub async fn get_schema(&self) -> Result<QueryResponse> {
        let query = "CALL schema.node_type_properties() YIELD nodeType, nodeLabels, propertyName, propertyTypes";
        self.query(query).await
    }

    /// Get node count
    pub async fn get_node_count(&self) -> Result<usize> {
        let response = self.query("MATCH (n) RETURN count(n) as count").await?;
        
        if let Some(row) = response.data.first() {
            if let Some(count) = row.first() {
                return Ok(count.as_u64().unwrap_or(0) as usize);
            }
        }
        
        Ok(0)
    }

    /// Get edge count
    pub async fn get_edge_count(&self) -> Result<usize> {
        let response = self.query("MATCH ()-[r]->() RETURN count(r) as count").await?;
        
        if let Some(row) = response.data.first() {
            if let Some(count) = row.first() {
                return Ok(count.as_u64().unwrap_or(0) as usize);
            }
        }
        
        Ok(0)
    }

    /// Delete all data (use with caution!)
    pub async fn clear_database(&self) -> Result<QueryResponse> {
        self.query("MATCH (n) DETACH DELETE n").await
    }

    /// Check health status
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let url = format!("{}/health", self.base_url);
        
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to check health")?;

        if !response.status().is_success() {
            anyhow::bail!("Health check failed with status {}", response.status());
        }

        response
            .json::<HealthStatus>()
            .await
            .context("Failed to parse health response")
    }

    /// Test connection
    pub async fn test_connection(&self) -> Result<bool> {
        match self.query("RETURN 1").await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = MemgraphClient::new("http://localhost:7687");
        assert!(client.is_ok());
    }

    #[test]
    fn test_query_request_serialization() {
        let request = QueryRequest {
            query: "MATCH (n) RETURN n".to_string(),
            parameters: None,
        };
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("MATCH"));
    }

    #[test]
    fn test_node_serialization() {
        let node = Node {
            id: Some("1".to_string()),
            labels: vec!["Person".to_string()],
            properties: serde_json::Map::new(),
        };
        
        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains("Person"));
    }
}
