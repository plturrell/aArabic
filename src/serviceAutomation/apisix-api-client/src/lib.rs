use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Complete Apache APISIX API Gateway Client
pub struct ApisixClient {
    base_url: String,
    api_key: Option<String>,
    client: Client,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub methods: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<Upstream>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugins: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Upstream {
    #[serde(rename = "type")]
    pub upstream_type: String,
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub host: String,
    pub port: u16,
    pub weight: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Service {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<Upstream>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugins: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Consumer {
    pub username: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugins: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSL {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub cert: String,
    pub key: String,
    pub snis: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plugin {
    pub name: String,
    pub config: HashMap<String, Value>,
}

impl ApisixClient {
    /// Create new APISIX client
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
        let url = format!("{}/apisix/admin/{}", self.base_url, endpoint.trim_start_matches('/'));
        let mut req = match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            "PUT" => self.client.put(&url),
            "PATCH" => self.client.patch(&url),
            "DELETE" => self.client.delete(&url),
            _ => self.client.get(&url),
        };
        
        if let Some(key) = &self.api_key {
            req = req.header("X-API-KEY", key);
        }
        
        req
    }

    // ========================================================================
    // ROUTE OPERATIONS
    // ========================================================================

    /// List routes
    pub fn list_routes(&self) -> Result<Vec<Route>> {
        let response = self.request("GET", "routes").send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["list"].clone())?)
    }

    /// Get route
    pub fn get_route(&self, id: &str) -> Result<Route> {
        let response = self.request("GET", &format!("routes/{}", id)).send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Create route
    pub fn create_route(&self, route: &Route) -> Result<Route> {
        let response = self.request("POST", "routes")
            .json(route)
            .send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Update route
    pub fn update_route(&self, id: &str, route: &Route) -> Result<Route> {
        let response = self.request("PUT", &format!("routes/{}", id))
            .json(route)
            .send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Delete route
    pub fn delete_route(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("routes/{}", id)).send()?;
        Ok(())
    }

    // ========================================================================
    // SERVICE OPERATIONS
    // ========================================================================

    /// List services
    pub fn list_services(&self) -> Result<Vec<Service>> {
        let response = self.request("GET", "services").send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["list"].clone())?)
    }

    /// Get service
    pub fn get_service(&self, id: &str) -> Result<Service> {
        let response = self.request("GET", &format!("services/{}", id)).send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Create service
    pub fn create_service(&self, service: &Service) -> Result<Service> {
        let response = self.request("POST", "services")
            .json(service)
            .send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Update service
    pub fn update_service(&self, id: &str, service: &Service) -> Result<Service> {
        let response = self.request("PUT", &format!("services/{}", id))
            .json(service)
            .send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Delete service
    pub fn delete_service(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("services/{}", id)).send()?;
        Ok(())
    }

    // ========================================================================
    // UPSTREAM OPERATIONS
    // ========================================================================

    /// List upstreams
    pub fn list_upstreams(&self) -> Result<Vec<Upstream>> {
        let response = self.request("GET", "upstreams").send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["list"].clone())?)
    }

    /// Get upstream
    pub fn get_upstream(&self, id: &str) -> Result<Upstream> {
        let response = self.request("GET", &format!("upstreams/{}", id)).send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Create upstream
    pub fn create_upstream(&self, upstream: &Upstream) -> Result<Upstream> {
        let response = self.request("POST", "upstreams")
            .json(upstream)
            .send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Update upstream
    pub fn update_upstream(&self, id: &str, upstream: &Upstream) -> Result<Upstream> {
        let response = self.request("PUT", &format!("upstreams/{}", id))
            .json(upstream)
            .send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Delete upstream
    pub fn delete_upstream(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("upstreams/{}", id)).send()?;
        Ok(())
    }

    // ========================================================================
    // CONSUMER OPERATIONS
    // ========================================================================

    /// List consumers
    pub fn list_consumers(&self) -> Result<Vec<Consumer>> {
        let response = self.request("GET", "consumers").send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["list"].clone())?)
    }

    /// Get consumer
    pub fn get_consumer(&self, username: &str) -> Result<Consumer> {
        let response = self.request("GET", &format!("consumers/{}", username)).send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Create consumer
    pub fn create_consumer(&self, consumer: &Consumer) -> Result<Consumer> {
        let response = self.request("PUT", &format!("consumers/{}", consumer.username))
            .json(consumer)
            .send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Delete consumer
    pub fn delete_consumer(&self, username: &str) -> Result<()> {
        self.request("DELETE", &format!("consumers/{}", username)).send()?;
        Ok(())
    }

    // ========================================================================
    // SSL OPERATIONS
    // ========================================================================

    /// List SSL certificates
    pub fn list_ssl(&self) -> Result<Vec<SSL>> {
        let response = self.request("GET", "ssls").send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["list"].clone())?)
    }

    /// Get SSL certificate
    pub fn get_ssl(&self, id: &str) -> Result<SSL> {
        let response = self.request("GET", &format!("ssls/{}", id)).send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Create SSL certificate
    pub fn create_ssl(&self, ssl: &SSL) -> Result<SSL> {
        let response = self.request("POST", "ssls")
            .json(ssl)
            .send()?;
        let data: Value = response.json()?;
        Ok(serde_json::from_value(data["node"]["value"].clone())?)
    }

    /// Delete SSL certificate
    pub fn delete_ssl(&self, id: &str) -> Result<()> {
        self.request("DELETE", &format!("ssls/{}", id)).send()?;
        Ok(())
    }

    // ========================================================================
    // PLUGIN OPERATIONS
    // ========================================================================

    /// List available plugins
    pub fn list_plugins(&self) -> Result<Vec<String>> {
        let response = self.request("GET", "plugins/list").send()?;
        Ok(response.json()?)
    }

    /// Get plugin schema
    pub fn get_plugin_schema(&self, name: &str) -> Result<Value> {
        let response = self.request("GET", &format!("schema/plugins/{}", name)).send()?;
        Ok(response.json()?)
    }

    // ========================================================================
    // HEALTH & STATUS
    // ========================================================================

    /// Health check
    pub fn health_check(&self) -> Result<bool> {
        let response = self.request("GET", "routes").send()?;
        Ok(response.status().is_success())
    }
}
