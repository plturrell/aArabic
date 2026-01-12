use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Complete Glean API Client
/// Code intelligence platform for querying and indexing codebases
pub struct GleanClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Database {
    pub name: String,
    pub repo: String,
    pub version: Option<String>,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub query: String,
    pub repo: Option<String>,
    pub file: Option<String>,
    pub recursive: Option<bool>,
    pub limit: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub results: Vec<Value>,
    pub stats: Option<QueryStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    pub elapsed_ms: Option<i64>,
    pub facts_searched: Option<i64>,
    pub bytes_searched: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRequest {
    pub repo: String,
    pub indexer: String,
    pub input_path: String,
    pub output_db: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResult {
    pub success: bool,
    pub database: Option<String>,
    pub stats: Option<IndexStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub files_indexed: Option<i64>,
    pub facts_generated: Option<i64>,
    pub elapsed_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: String,
    pub location: Option<Location>,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub file: String,
    pub line: i32,
    pub column: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub symbol: String,
    pub location: Location,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Definition {
    pub symbol: String,
    pub location: Location,
    pub signature: Option<String>,
    pub documentation: Option<String>,
}

// ============================================================================
// CLIENT IMPLEMENTATION
// ============================================================================

impl GleanClient {
    /// Create new Glean API client
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }

    /// Build request with authentication
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.request(method, &url);
        
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        
        req
    }

    // ========================================================================
    // DATABASE MANAGEMENT
    // ========================================================================

    /// List all databases
    pub fn list_databases(&self) -> Result<Vec<Database>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/databases")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list databases: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get database info
    pub fn get_database(&self, name: &str) -> Result<Database> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/databases/{}", name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get database: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Create new database
    pub fn create_database(&self, db: &Database) -> Result<Database> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/databases")
            .json(db)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to create database: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Delete database
    pub fn delete_database(&self, name: &str) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/api/v1/databases/{}", name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete database: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // QUERY API
    // ========================================================================

    /// Execute Angle query
    pub fn query(&self, query: &Query) -> Result<QueryResult> {
        let response = self
            .request(reqwest::Method::POST, "/api/v1/query")
            .json(query)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to execute query: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Query for symbol definitions
    pub fn query_definitions(&self, symbol: &str, repo: Option<&str>) -> Result<Vec<Definition>> {
        let query_str = format!("src.Definition {{ name = \"{}\" }}", symbol);
        let query = Query {
            query: query_str,
            repo: repo.map(String::from),
            file: None,
            recursive: Some(true),
            limit: Some(100),
        };

        let result = self.query(&query)?;
        
        // Parse results into definitions
        let definitions: Vec<Definition> = result
            .results
            .iter()
            .filter_map(|r| serde_json::from_value(r.clone()).ok())
            .collect();

        Ok(definitions)
    }

    /// Query for symbol references
    pub fn query_references(&self, symbol: &str, repo: Option<&str>) -> Result<Vec<Reference>> {
        let query_str = format!("src.Reference {{ symbol = \"{}\" }}", symbol);
        let query = Query {
            query: query_str,
            repo: repo.map(String::from),
            file: None,
            recursive: Some(true),
            limit: Some(1000),
        };

        let result = self.query(&query)?;
        
        let references: Vec<Reference> = result
            .results
            .iter()
            .filter_map(|r| serde_json::from_value(r.clone()).ok())
            .collect();

        Ok(references)
    }

    /// Find all symbols in a file
    pub fn find_symbols(&self, file: &str, repo: Option<&str>) -> Result<Vec<SymbolInfo>> {
        let query_str = format!("src.Symbol {{ file = \"{}\" }}", file);
        let query = Query {
            query: query_str,
            repo: repo.map(String::from),
            file: Some(file.to_string()),
            recursive: None,
            limit: Some(1000),
        };

        let result = self.query(&query)?;
        
        let symbols: Vec<SymbolInfo> = result
            .results
            .iter()
            .filter_map(|r| serde_json::from_value(r.clone()).ok())
            .collect();

        Ok(symbols)
    }

    /// Search code by pattern
    pub fn search_code(&self, pattern: &str, repo: Option<&str>, file_pattern: Option<&str>) -> Result<Vec<Value>> {
        let query_str = if let Some(fp) = file_pattern {
            format!("src.FileMatch {{ pattern = \"{}\", file = \"{}\" }}", pattern, fp)
        } else {
            format!("src.FileMatch {{ pattern = \"{}\" }}", pattern)
        };

        let query = Query {
            query: query_str,
            repo: repo.map(String::from),
            file: None,
            recursive: Some(true),
            limit: Some(100),
        };

        let result = self.query(&query)?;
        Ok(result.results)
    }

    // ========================================================================
    // INDEXING API
    // ========================================================================

    /// Index Rust codebase using LSIF
    pub fn index_rust(&self, req: &IndexRequest) -> Result<IndexResult> {
        let payload = json!({
            "repo": req.repo,
            "indexer": "lsif-rust",
            "input_path": req.input_path,
            "output_db": req.output_db
        });

        let response = self
            .request(reqwest::Method::POST, "/api/v1/index/rust")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to index Rust code: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Index Go codebase using LSIF
    pub fn index_go(&self, req: &IndexRequest) -> Result<IndexResult> {
        let payload = json!({
            "repo": req.repo,
            "indexer": "lsif-go",
            "input_path": req.input_path,
            "output_db": req.output_db
        });

        let response = self
            .request(reqwest::Method::POST, "/api/v1/index/go")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to index Go code: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Index Python codebase using SCIP
    pub fn index_python(&self, req: &IndexRequest) -> Result<IndexResult> {
        let payload = json!({
            "repo": req.repo,
            "indexer": "scip-python",
            "input_path": req.input_path,
            "output_db": req.output_db
        });

        let response = self
            .request(reqwest::Method::POST, "/api/v1/index/python")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to index Python code: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Generic indexing endpoint
    pub fn index(&self, req: &IndexRequest) -> Result<IndexResult> {
        match req.indexer.as_str() {
            "lsif-rust" => self.index_rust(req),
            "lsif-go" => self.index_go(req),
            "scip-python" => self.index_python(req),
            _ => {
                let response = self
                    .request(reqwest::Method::POST, "/api/v1/index")
                    .json(req)
                    .send()?;

                if !response.status().is_success() {
                    anyhow::bail!("Failed to index: {}", response.status());
                }

                Ok(response.json()?)
            }
        }
    }

    /// Get indexing status
    pub fn get_index_status(&self, repo: &str) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/index/status/{}", repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get index status: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // CODE NAVIGATION
    // ========================================================================

    /// Go to definition
    pub fn goto_definition(&self, file: &str, line: i32, column: i32, repo: Option<&str>) -> Result<Vec<Location>> {
        let payload = json!({
            "file": file,
            "line": line,
            "column": column,
            "repo": repo
        });

        let response = self
            .request(reqwest::Method::POST, "/api/v1/goto/definition")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to go to definition: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Find references
    pub fn find_references(&self, file: &str, line: i32, column: i32, repo: Option<&str>) -> Result<Vec<Location>> {
        let payload = json!({
            "file": file,
            "line": line,
            "column": column,
            "repo": repo
        });

        let response = self
            .request(reqwest::Method::POST, "/api/v1/find/references")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to find references: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get hover information
    pub fn get_hover(&self, file: &str, line: i32, column: i32, repo: Option<&str>) -> Result<Value> {
        let payload = json!({
            "file": file,
            "line": line,
            "column": column,
            "repo": repo
        });

        let response = self
            .request(reqwest::Method::POST, "/api/v1/hover")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get hover info: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // STATISTICS & ANALYTICS
    // ========================================================================

    /// Get repository statistics
    pub fn get_repo_stats(&self, repo: &str) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/stats/repo/{}", repo))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get repo stats: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get database statistics
    pub fn get_db_stats(&self, db_name: &str) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, &format!("/api/v1/stats/db/{}", db_name))
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get database stats: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get query performance stats
    pub fn get_query_stats(&self) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/stats/queries")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get query stats: {}", response.status());
        }

        Ok(response.json()?)
    }

    // ========================================================================
    // SHELL / REPL SUPPORT
    // ========================================================================

    /// Execute shell command
    pub fn shell_execute(&self, command: &str) -> Result<Value> {
        let payload = json!({
            "command": command
        });

        let response = self
            .request(reqwest::Method::POST, "/api/v1/shell/execute")
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to execute shell command: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get shell history
    pub fn shell_history(&self) -> Result<Vec<String>> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/shell/history")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get shell history: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Clear shell history
    pub fn shell_clear_history(&self) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, "/api/v1/shell/history")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to clear shell history: {}", response.status());
        }

        Ok(())
    }

    // ========================================================================
    // SYSTEM
    // ========================================================================

    /// Health check
    pub fn health_check(&self) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/health")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Health check failed: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get version
    pub fn get_version(&self) -> Result<Value> {
        let response = self
            .request(reqwest::Method::GET, "/api/v1/version")
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get version: {}", response.status());
        }

        Ok(response.json()?)
    }
}
