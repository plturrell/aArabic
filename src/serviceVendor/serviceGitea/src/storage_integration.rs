/// Storage Integration Module
/// Distributes artifacts across specialized data stores:
/// - Marquez: Metadata & lineage
/// - Memgraph: SCIP graph structures
/// - Qdrant: Vector embeddings & semantic search

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::fs;

const MARQUEZ_PATH: &str = "/Users/user/Documents/arabic_folder/vendor/layerData/marquez";
const MEMGRAPH_PATH: &str = "/Users/user/Documents/arabic_folder/vendor/layerIntelligence/memgraph-ai-toolkit";
const QDRANT_PATH: &str = "/Users/user/Documents/arabic_folder/vendor/layerData/qdrant";

#[derive(Debug, Serialize, Deserialize)]
pub struct FeatureMetadata {
    pub feature_name: String,
    pub created_at: String,
    pub commit_sha: String,
    pub repository: String,
    pub branch: String,
    pub artifacts: ArtifactMetadata,
    pub lineage: DataLineage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub markdown_spec: ArtifactInfo,
    pub rust_code: ArtifactInfo,
    pub lean4_proofs: ArtifactInfo,
    pub scip_spec: ArtifactInfo,
    pub n8n_workflow: ArtifactInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactInfo {
    pub size_bytes: usize,
    pub hash: String,
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataLineage {
    pub source: String,
    pub transformations: Vec<Transformation>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Transformation {
    pub step: String,
    pub tool: String,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SCIPGraph {
    pub feature_name: String,
    pub nodes: Vec<SCIPNode>,
    pub edges: Vec<SCIPEdge>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SCIPNode {
    pub id: String,
    pub node_type: String,
    pub title: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SCIPEdge {
    pub from: String,
    pub to: String,
    pub relationship: String,
}

pub struct StorageIntegration {
    client: Client,
    marquez_path: String,
    memgraph_path: String,
    qdrant_url: String,
}

impl StorageIntegration {
    pub fn new() -> Self {
        let qdrant_url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6333".to_string());

        Self {
            client: Client::new(),
            marquez_path: MARQUEZ_PATH.to_string(),
            memgraph_path: MEMGRAPH_PATH.to_string(),
            qdrant_url,
        }
    }

    /// Store complete feature data across all storage systems
    pub async fn store_feature(
        &self,
        feature_name: &str,
        artifacts: &super::FeatureArtifacts,
        commit_sha: &str,
        repository: &str,
    ) -> Result<()> {
        // 1. Store metadata in Marquez
        self.store_metadata(feature_name, artifacts, commit_sha, repository).await?;

        // 2. Store SCIP graph in Memgraph
        self.store_scip_graph(feature_name, &artifacts.scip_spec).await?;

        // 3. Store vector embeddings in Qdrant
        self.store_vectors(feature_name, artifacts).await?;

        Ok(())
    }

    /// Store metadata and lineage in Marquez
    async fn store_metadata(
        &self,
        feature_name: &str,
        artifacts: &super::FeatureArtifacts,
        commit_sha: &str,
        repository: &str,
    ) -> Result<()> {
        let metadata = FeatureMetadata {
            feature_name: feature_name.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            commit_sha: commit_sha.to_string(),
            repository: repository.to_string(),
            branch: "main".to_string(),
            artifacts: ArtifactMetadata {
                markdown_spec: Self::artifact_info(&artifacts.markdown_spec, "README.md"),
                rust_code: Self::artifact_info(&artifacts.rust_code, "src/lib.rs"),
                lean4_proofs: Self::artifact_info(&artifacts.lean4_proofs, "proofs/feature.lean"),
                scip_spec: Self::artifact_info(&artifacts.scip_spec, "specs/scip.json"),
                n8n_workflow: Self::artifact_info(&artifacts.n8n_workflow, "workflows/n8n.json"),
            },
            lineage: DataLineage {
                source: "n8n_workflow".to_string(),
                transformations: vec![
                    Transformation {
                        step: "n8n_to_markdown".to_string(),
                        tool: "lean4_parser".to_string(),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                    },
                    Transformation {
                        step: "markdown_to_code".to_string(),
                        tool: "lean4_parser".to_string(),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                    },
                    Transformation {
                        step: "go_to_rust".to_string(),
                        tool: "rust_generator".to_string(),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                    },
                ],
                outputs: vec![
                    "rust_code".to_string(),
                    "lean4_proofs".to_string(),
                    "scip_spec".to_string(),
                ],
            },
        };

        // Ensure directory exists
        fs::create_dir_all(&self.marquez_path).await?;

        // Write metadata file
        let file_path = format!("{}/feature-{}.json", self.marquez_path, feature_name);
        let json = serde_json::to_string_pretty(&metadata)?;
        fs::write(file_path, json).await?;

        log::info!("Stored metadata in Marquez for feature: {}", feature_name);
        Ok(())
    }

    /// Store SCIP graph structure in Memgraph
    async fn store_scip_graph(&self, feature_name: &str, scip_json: &str) -> Result<()> {
        let scip: serde_json::Value = serde_json::from_str(scip_json)?;
        
        let elements = scip.get("elements")
            .and_then(|e| e.as_array())
            .context("Invalid SCIP: missing elements")?;

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create nodes for each SCIP element
        for element in elements {
            let id = element.get("id")
                .and_then(|i| i.as_str())
                .unwrap_or("unknown");
            
            nodes.push(SCIPNode {
                id: id.to_string(),
                node_type: element.get("element_type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("requirement")
                    .to_string(),
                title: element.get("title")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                properties: element.clone(),
            });

            // Create edges for dependencies
            if let Some(deps) = element.get("dependencies").and_then(|d| d.as_array()) {
                for dep in deps {
                    if let Some(dep_id) = dep.as_str() {
                        edges.push(SCIPEdge {
                            from: id.to_string(),
                            to: dep_id.to_string(),
                            relationship: "DEPENDS_ON".to_string(),
                        });
                    }
                }
            }
        }

        let graph = SCIPGraph {
            feature_name: feature_name.to_string(),
            nodes,
            edges,
        };

        // Ensure directory exists
        fs::create_dir_all(&self.memgraph_path).await?;

        // Write graph file
        let file_path = format!("{}/feature-{}-graph.json", self.memgraph_path, feature_name);
        let json = serde_json::to_string_pretty(&graph)?;
        fs::write(&file_path, json).await?;

        // Also write as Cypher queries for Memgraph
        let cypher_path = format!("{}/feature-{}-graph.cypher", self.memgraph_path, feature_name);
        let cypher = self.generate_cypher(&graph)?;
        fs::write(cypher_path, cypher).await?;

        log::info!("Stored SCIP graph in Memgraph for feature: {}", feature_name);
        Ok(())
    }

    /// Generate Cypher queries for Memgraph
    fn generate_cypher(&self, graph: &SCIPGraph) -> Result<String> {
        let mut cypher = String::new();
        
        // Create nodes
        for node in &graph.nodes {
            cypher.push_str(&format!(
                "CREATE (n{}:{} {{id: '{}', title: '{}', properties: '{}'}})\n",
                node.id.replace("-", "_"),
                node.node_type,
                node.id,
                node.title.replace("'", "\\'"),
                serde_json::to_string(&node.properties)?.replace("'", "\\'")
            ));
        }

        cypher.push('\n');

        // Create edges
        for edge in &graph.edges {
            cypher.push_str(&format!(
                "MATCH (a {{id: '{}'}}), (b {{id: '{}'}})\nCREATE (a)-[:{}]->(b)\n",
                edge.from, edge.to, edge.relationship
            ));
        }

        Ok(cypher)
    }

    /// Store vector embeddings in Qdrant
    async fn store_vectors(&self, feature_name: &str, artifacts: &super::FeatureArtifacts) -> Result<()> {
        let collection_name = "features";

        // Ensure collection exists
        self.ensure_qdrant_collection(&collection_name).await?;

        // Generate embeddings for each artifact
        let points = vec![
            self.create_point(feature_name, "markdown", &artifacts.markdown_spec).await?,
            self.create_point(feature_name, "rust", &artifacts.rust_code).await?,
            self.create_point(feature_name, "lean4", &artifacts.lean4_proofs).await?,
            self.create_point(feature_name, "scip", &artifacts.scip_spec).await?,
        ];

        // Upsert points to Qdrant
        let response = self.client
            .put(format!("{}/collections/{}/points", self.qdrant_url, collection_name))
            .json(&json!({
                "points": points
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to store vectors in Qdrant: {}", response.status());
        }

        log::info!("Stored vectors in Qdrant for feature: {}", feature_name);
        Ok(())
    }

    /// Ensure Qdrant collection exists
    async fn ensure_qdrant_collection(&self, name: &str) -> Result<()> {
        let response = self.client
            .put(format!("{}/collections/{}", self.qdrant_url, name))
            .json(&json!({
                "vectors": {
                    "size": 384,
                    "distance": "Cosine"
                }
            }))
            .send()
            .await?;

        // 409 means collection already exists, which is fine
        if !response.status().is_success() && response.status().as_u16() != 409 {
            anyhow::bail!("Failed to create Qdrant collection: {}", response.status());
        }

        Ok(())
    }

    /// Create a Qdrant point with embedding
    async fn create_point(
        &self,
        feature_name: &str,
        artifact_type: &str,
        content: &str,
    ) -> Result<serde_json::Value> {
        // Generate simple embedding (in production, use proper embedding model)
        let embedding = self.generate_embedding(content);

        Ok(json!({
            "id": format!("{}-{}", feature_name, artifact_type),
            "vector": embedding,
            "payload": {
                "feature_name": feature_name,
                "artifact_type": artifact_type,
                "content_preview": &content[..content.len().min(200)],
                "timestamp": chrono::Utc::now().to_rfc3339()
            }
        }))
    }

    /// Generate embedding vector (simplified - use proper model in production)
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        // Simplified: use text hash to generate deterministic embedding
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate 384-dimensional vector
        (0..384)
            .map(|i| ((hash.wrapping_mul(i as u64)) as f32 / u64::MAX as f32) * 2.0 - 1.0)
            .collect()
    }

    fn artifact_info(content: &str, path: &str) -> ArtifactInfo {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);

        ArtifactInfo {
            size_bytes: content.len(),
            hash: format!("{:x}", hasher.finish()),
            path: path.to_string(),
        }
    }
}

impl Default for StorageIntegration {
    fn default() -> Self {
        Self::new()
    }
}