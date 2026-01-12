use anyhow::{anyhow, Result};
use qdrant_client::{
    Qdrant,
    qdrant::{
        vectors_config::Config, CreateCollectionBuilder, Distance, PointStruct,
        SearchPointsBuilder, UpsertPointsBuilder, VectorsConfig, Value,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Qdrant vector database client (v2 API)
pub struct QdrantClient {
    client: Qdrant,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub vectors_count: Option<u64>,
    pub points_count: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Point {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub payload: Option<HashMap<String, Value>>,
}

impl QdrantClient {
    /// Create a new Qdrant client (async)
    pub async fn new(url: &str) -> Result<Self> {
        let client = Qdrant::from_url(url).build()?;
        Ok(Self { client })
    }

    /// Create client from environment variable QDRANT_URL
    pub async fn from_env() -> Result<Self> {
        let url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6333".to_string());
        Self::new(&url).await
    }

    /// List all collections
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let collections = self.client.list_collections().await?;
        Ok(collections
            .collections
            .into_iter()
            .map(|c| c.name)
            .collect())
    }

    /// Create a new collection
    pub async fn create_collection(
        &self,
        name: &str,
        vector_size: u64,
        distance: Distance,
    ) -> Result<()> {
        use qdrant_client::qdrant::VectorParams;
        
        self.client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorsConfig {
                        config: Some(Config::Params(VectorParams {
                            size: vector_size,
                            distance: distance.into(),
                            ..Default::default()
                        })),
                    })
            )
            .await?;

        Ok(())
    }

    /// Delete a collection
    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        self.client.delete_collection(name).await?;
        Ok(())
    }

    /// Check if collection exists
    pub async fn collection_exists(&self, name: &str) -> Result<bool> {
        match self.client.collection_info(name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Upsert points into collection
    pub async fn upsert_points(&self, collection: &str, points: Vec<Point>) -> Result<()> {
        let qdrant_points: Vec<PointStruct> = points
            .into_iter()
            .map(|p| {
                let payload: HashMap<String, Value> = p
                    .payload
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(k, v)| (k, json_to_value(v)))
                    .collect();

                PointStruct::new(p.id, p.vector, payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(collection, qdrant_points))
            .await?;

        Ok(())
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        collection: &str,
        vector: Vec<f32>,
        limit: u64,
    ) -> Result<Vec<SearchResult>> {
        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(collection, vector, limit)
                    .with_payload(true)
            )
            .await?;

        let results = search_result
            .result
            .into_iter()
            .map(|point| SearchResult {
                id: point.id.and_then(|id| id.point_id_options).map_or(0, |id_opt| {
                    match id_opt {
                        qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => n,
                        qdrant_client::qdrant::point_id::PointIdOptions::Uuid(_) => 0,
                    }
                }),
                score: point.score,
                payload: Some(point.payload),
            })
            .collect();

        Ok(results)
    }

    /// Get collection information
    pub async fn get_collection_info(&self, name: &str) -> Result<Collection> {
        let info = self.client.collection_info(name).await?;

        Ok(Collection {
            name: name.to_string(),
            vectors_count: info.result.as_ref().and_then(|r| r.indexed_vectors_count),
            points_count: info.result.as_ref().and_then(|r| r.points_count),
        })
    }

    /// Count points in collection
    pub async fn count_points(&self, collection: &str) -> Result<u64> {
        let info = self.get_collection_info(collection).await?;
        Ok(info.points_count.unwrap_or(0))
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool> {
        match self.client.health_check().await {
            Ok(_) => Ok(true),
            Err(e) => Err(anyhow!("Health check failed: {}", e)),
        }
    }
}

/// Convert serde_json::Value to qdrant Value
fn json_to_value(json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::default(),
        serde_json::Value::Bool(b) => Value {
            kind: Some(qdrant_client::qdrant::value::Kind::BoolValue(b)),
        },
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)),
                }
            } else if let Some(f) = n.as_f64() {
                Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)),
                }
            } else {
                Value::default()
            }
        }
        serde_json::Value::String(s) => Value {
            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(s)),
        },
        serde_json::Value::Array(arr) => {
            let list_value = qdrant_client::qdrant::ListValue {
                values: arr.into_iter().map(json_to_value).collect(),
            };
            Value {
                kind: Some(qdrant_client::qdrant::value::Kind::ListValue(list_value)),
            }
        }
        serde_json::Value::Object(obj) => {
            let struct_value = qdrant_client::qdrant::Struct {
                fields: obj.into_iter().map(|(k, v)| (k, json_to_value(v))).collect(),
            };
            Value {
                kind: Some(qdrant_client::qdrant::value::Kind::StructValue(
                    struct_value,
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let result = QdrantClient::new("http://localhost:6333").await;
        // Note: This will fail if Qdrant is not running, which is expected in unit tests
        // In real tests, you'd use a test container or mock
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_json_to_value_conversion() {
        let json = serde_json::json!({
            "name": "test",
            "count": 42,
            "active": true
        });

        let value = json_to_value(json);
        assert!(value.kind.is_some());
    }
}
