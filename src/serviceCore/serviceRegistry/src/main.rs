use actix_web::{web, App, HttpResponse, HttpServer};
use anyhow::{Context, Result};
use chrono::Utc;
use log::{info, warn};
use neo4rs::{query, Graph};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_postgres::{Client, NoTls};

const DEFAULT_COLLECTION: &str = "services";
const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Clone)]
struct RegistryState {
    db: Arc<Client>,
    config_path: PathBuf,
    memgraph: Option<Graph>,
    qdrant_url: Option<String>,
    qdrant_client: reqwest::Client,
    embedding_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegistryFile {
    services: Vec<ServiceDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServiceDefinition {
    id: String,
    name: String,
    layer: String,
    kind: String,
    vendor_path: Option<String>,
    src_path: Option<String>,
    wrapper: Option<bool>,
    upstream_url: Option<String>,
    health_path: Option<String>,
    version: Option<String>,
    tags: Option<Vec<String>>,
    dependencies: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    postgres: bool,
    memgraph: bool,
    qdrant: bool,
    timestamp: String,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let config_path = env::var("SERVICE_REGISTRY_CONFIG")
        .unwrap_or_else(|_| "config/service_registry.json".to_string());
    let db_url = env::var("SERVICE_REGISTRY_DB_URL")
        .or_else(|_| env::var("SERVICE_REGISTRY_DB"))
        .unwrap_or_else(|_| "postgres://registry:registry@registry-db:5432/service_registry".to_string());
    let embedding_dim = env::var("SERVICE_REGISTRY_EMBEDDING_DIM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_EMBEDDING_DIM);

    let db = init_postgres(&db_url)
        .await
        .expect("failed to initialize postgres");
    let memgraph = connect_memgraph().await;
    let qdrant_url = env::var("QDRANT_URL").ok();

    let qdrant_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("failed to build HTTP client");

    if let Some(url) = &qdrant_url {
        if let Err(error) = ensure_qdrant_collection(&qdrant_client, url, embedding_dim).await {
            warn!("Qdrant collection init failed: {}", error);
        }
    }

    let state = RegistryState {
        db: Arc::new(db),
        config_path: PathBuf::from(config_path),
        memgraph,
        qdrant_url,
        qdrant_client,
        embedding_dim,
    };

    if let Ok(registry_file) = load_registry_file(&state.config_path) {
        for service in &registry_file.services {
            if let Err(error) = persist_service(&state, service).await {
                warn!("Failed to seed registry for {}: {}", service.id, error);
            }
        }
    } else {
        warn!("Registry config not loaded during startup");
    }

    let bind_address = env::var("SERVICE_REGISTRY_BIND")
        .unwrap_or_else(|_| "0.0.0.0:8100".to_string());

    info!("Starting service registry on {}", bind_address);

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .route("/health", web::get().to(health))
            .route("/services", web::get().to(list_services))
            .route("/services/{id}", web::get().to(get_service))
            .route("/services/register", web::post().to(register_service))
            .route("/services/refresh", web::post().to(refresh_services))
    })
    .bind(bind_address)?
    .run()
    .await
}

async fn health(state: web::Data<RegistryState>) -> HttpResponse {
    let postgres_ok = state.db.query_one("SELECT 1", &[]).await.is_ok();
    let memgraph_ok = state.memgraph.is_some();
    let qdrant_ok = match &state.qdrant_url {
        Some(url) => state
            .qdrant_client
            .get(format!("{}/collections/{}", url, DEFAULT_COLLECTION))
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false),
        None => false,
    };

    HttpResponse::Ok().json(HealthResponse {
        status: "healthy".to_string(),
        postgres: postgres_ok,
        memgraph: memgraph_ok,
        qdrant: qdrant_ok,
        timestamp: Utc::now().to_rfc3339(),
    })
}

async fn list_services(state: web::Data<RegistryState>) -> HttpResponse {
    match fetch_services(&state.db).await {
        Ok(services) => HttpResponse::Ok().json(services),
        Err(error) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": error.to_string()
        })),
    }
}

async fn get_service(path: web::Path<String>, state: web::Data<RegistryState>) -> HttpResponse {
    let service_id = path.into_inner();
    match fetch_service(&state.db, &service_id).await {
        Ok(Some(service)) => HttpResponse::Ok().json(service),
        Ok(None) => HttpResponse::NotFound().json(serde_json::json!({
            "error": "service not found"
        })),
        Err(error) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": error.to_string()
        })),
    }
}

async fn register_service(
    state: web::Data<RegistryState>,
    payload: web::Json<ServiceDefinition>,
) -> HttpResponse {
    match persist_service(&state, &payload.into_inner()).await {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({ "status": "registered" })),
        Err(error) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": error.to_string()
        })),
    }
}

async fn refresh_services(state: web::Data<RegistryState>) -> HttpResponse {
    match load_registry_file(&state.config_path) {
        Ok(registry_file) => {
            let mut persisted = 0;
            for service in &registry_file.services {
                if persist_service(&state, service).await.is_ok() {
                    persisted += 1;
                }
            }
            HttpResponse::Ok().json(serde_json::json!({
                "status": "refreshed",
                "services": persisted
            }))
        }
        Err(error) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": error.to_string()
        })),
    }
}

async fn init_postgres(db_url: &str) -> Result<Client> {
    let (client, connection) = tokio_postgres::connect(db_url, NoTls).await?;

    tokio::spawn(async move {
        if let Err(error) = connection.await {
            warn!("Postgres connection error: {}", error);
        }
    });

    client
        .batch_execute(
            "CREATE TABLE IF NOT EXISTS services (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                layer TEXT NOT NULL,
                kind TEXT NOT NULL,
                vendor_path TEXT,
                src_path TEXT,
                wrapper BOOLEAN DEFAULT FALSE,
                upstream_url TEXT,
                health_path TEXT,
                version TEXT,
                tags TEXT,
                dependencies TEXT,
                updated_at TEXT
            );",
        )
        .await?;

    Ok(client)
}

async fn connect_memgraph() -> Option<Graph> {
    let memgraph_uri = env::var("MEMGRAPH_URI").ok().or_else(|| {
        let host = env::var("MEMGRAPH_HOST").ok()?;
        let port = env::var("MEMGRAPH_PORT").ok()?;
        let user = env::var("MEMGRAPH_USERNAME").unwrap_or_default();
        let pass = env::var("MEMGRAPH_PASSWORD").unwrap_or_default();
        let auth = if user.is_empty() {
            "".to_string()
        } else {
            format!("{}:{}@", user, pass)
        };
        Some(format!("bolt://{}{}:{}", auth, host, port))
    });

    match memgraph_uri {
        Some(uri) => match Graph::new(uri).await {
            Ok(graph) => Some(graph),
            Err(error) => {
                warn!("Memgraph connect failed: {}", error);
                None
            }
        },
        None => None,
    }
}

fn load_registry_file(path: &PathBuf) -> Result<RegistryFile> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read registry file: {}", path.display()))?;
    let parsed: RegistryFile = serde_json::from_str(&raw)?;
    Ok(parsed)
}

async fn persist_service(state: &RegistryState, service: &ServiceDefinition) -> Result<()> {
    let normalized = normalize_service(service);
    upsert_postgres(&state.db, &normalized).await?;

    if let Some(graph) = &state.memgraph {
        upsert_memgraph(graph, &normalized).await?;
    }

    if let Some(url) = &state.qdrant_url {
        upsert_qdrant(&state.qdrant_client, url, state.embedding_dim, &normalized).await?;
    }

    Ok(())
}

fn normalize_service(service: &ServiceDefinition) -> ServiceDefinition {
    ServiceDefinition {
        id: service.id.clone(),
        name: service.name.clone(),
        layer: service.layer.clone(),
        kind: service.kind.clone(),
        vendor_path: service.vendor_path.clone(),
        src_path: service.src_path.clone(),
        wrapper: Some(service.wrapper.unwrap_or(false)),
        upstream_url: service.upstream_url.clone(),
        health_path: service.health_path.clone(),
        version: service.version.clone(),
        tags: Some(service.tags.clone().unwrap_or_default()),
        dependencies: Some(service.dependencies.clone().unwrap_or_default()),
    }
}

async fn upsert_postgres(db: &Arc<Client>, service: &ServiceDefinition) -> Result<()> {
    let tags = serde_json::to_string(&service.tags.clone().unwrap_or_default())?;
    let dependencies = serde_json::to_string(&service.dependencies.clone().unwrap_or_default())?;
    let updated_at = Utc::now().to_rfc3339();

    db.execute(
        "INSERT INTO services (
            id, name, layer, kind, vendor_path, src_path, wrapper,
            upstream_url, health_path, version, tags, dependencies, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            layer = EXCLUDED.layer,
            kind = EXCLUDED.kind,
            vendor_path = EXCLUDED.vendor_path,
            src_path = EXCLUDED.src_path,
            wrapper = EXCLUDED.wrapper,
            upstream_url = EXCLUDED.upstream_url,
            health_path = EXCLUDED.health_path,
            version = EXCLUDED.version,
            tags = EXCLUDED.tags,
            dependencies = EXCLUDED.dependencies,
            updated_at = EXCLUDED.updated_at;",
        &[
            &service.id,
            &service.name,
            &service.layer,
            &service.kind,
            &service.vendor_path,
            &service.src_path,
            &service.wrapper.unwrap_or(false),
            &service.upstream_url,
            &service.health_path,
            &service.version,
            &tags,
            &dependencies,
            &updated_at,
        ],
    )
    .await?;
    Ok(())
}

async fn fetch_services(db: &Arc<Client>) -> Result<Vec<ServiceDefinition>> {
    let rows = db
        .query(
            "SELECT id, name, layer, kind, vendor_path, src_path, wrapper,
                    upstream_url, health_path, version, tags, dependencies
             FROM services",
            &[],
        )
        .await?;

    let mut services = Vec::new();
    for row in rows {
        let tags_json: Option<String> = row.get(10);
        let deps_json: Option<String> = row.get(11);
        let tags: Vec<String> = tags_json
            .as_deref()
            .and_then(|value| serde_json::from_str(value).ok())
            .unwrap_or_default();
        let dependencies: Vec<String> = deps_json
            .as_deref()
            .and_then(|value| serde_json::from_str(value).ok())
            .unwrap_or_default();
        let wrapper: bool = row.get::<_, Option<bool>>(6).unwrap_or(false);

        services.push(ServiceDefinition {
            id: row.get(0),
            name: row.get(1),
            layer: row.get(2),
            kind: row.get(3),
            vendor_path: row.get(4),
            src_path: row.get(5),
            wrapper: Some(wrapper),
            upstream_url: row.get(7),
            health_path: row.get(8),
            version: row.get(9),
            tags: Some(tags),
            dependencies: Some(dependencies),
        });
    }

    Ok(services)
}

async fn fetch_service(db: &Arc<Client>, id: &str) -> Result<Option<ServiceDefinition>> {
    let row = db
        .query_opt(
            "SELECT id, name, layer, kind, vendor_path, src_path, wrapper,
                    upstream_url, health_path, version, tags, dependencies
             FROM services WHERE id = $1",
            &[&id],
        )
        .await?;

    let Some(row) = row else {
        return Ok(None);
    };

    let tags_json: Option<String> = row.get(10);
    let deps_json: Option<String> = row.get(11);
    let tags: Vec<String> = tags_json
        .as_deref()
        .and_then(|value| serde_json::from_str(value).ok())
        .unwrap_or_default();
    let dependencies: Vec<String> = deps_json
        .as_deref()
        .and_then(|value| serde_json::from_str(value).ok())
        .unwrap_or_default();
    let wrapper: bool = row.get::<_, Option<bool>>(6).unwrap_or(false);

    Ok(Some(ServiceDefinition {
        id: row.get(0),
        name: row.get(1),
        layer: row.get(2),
        kind: row.get(3),
        vendor_path: row.get(4),
        src_path: row.get(5),
        wrapper: Some(wrapper),
        upstream_url: row.get(7),
        health_path: row.get(8),
        version: row.get(9),
        tags: Some(tags),
        dependencies: Some(dependencies),
    }))
}

async fn upsert_memgraph(graph: &Graph, service: &ServiceDefinition) -> Result<()> {
    let tags = service.tags.clone().unwrap_or_default();
    let dependencies = service.dependencies.clone().unwrap_or_default();

    graph
        .run(
            query(
                "MERGE (s:Service {id: $id})
                 SET s.name = $name,
                     s.layer = $layer,
                     s.kind = $kind,
                     s.vendor_path = $vendor_path,
                     s.src_path = $src_path,
                     s.version = $version,
                     s.upstream_url = $upstream_url,
                     s.health_path = $health_path,
                     s.tags = $tags,
                     s.updated_at = $updated_at
                 MERGE (l:Layer {name: $layer})
                 MERGE (s)-[:IN_LAYER]->(l)",
            )
            .param("id", &service.id)
            .param("name", &service.name)
            .param("layer", &service.layer)
            .param("kind", &service.kind)
            .param("vendor_path", service.vendor_path.clone().unwrap_or_default())
            .param("src_path", service.src_path.clone().unwrap_or_default())
            .param("version", service.version.clone().unwrap_or_default())
            .param("upstream_url", service.upstream_url.clone().unwrap_or_default())
            .param("health_path", service.health_path.clone().unwrap_or_default())
            .param("tags", tags.clone())
            .param("updated_at", Utc::now().to_rfc3339()),
        )
        .await?;

    for dep in dependencies {
        graph
            .run(
                query(
                    "MERGE (d:Service {id: $dep})
                     MERGE (s:Service {id: $id})
                     MERGE (s)-[:DEPENDS_ON]->(d)",
                )
                .param("dep", dep)
                .param("id", &service.id),
            )
            .await?;
    }

    Ok(())
}

async fn ensure_qdrant_collection(
    client: &reqwest::Client,
    url: &str,
    embedding_dim: usize,
) -> Result<()> {
    let response = client
        .put(format!("{}/collections/{}", url, DEFAULT_COLLECTION))
        .json(&serde_json::json!({
            "vectors": {
                "size": embedding_dim,
                "distance": "Cosine"
            }
        }))
        .send()
        .await?;

    if !response.status().is_success() && response.status().as_u16() != 409 {
        anyhow::bail!("Qdrant collection init failed: {}", response.status());
    }

    Ok(())
}

async fn upsert_qdrant(
    client: &reqwest::Client,
    url: &str,
    embedding_dim: usize,
    service: &ServiceDefinition,
) -> Result<()> {
    let text = format!(
        "{} {} {} {} {}",
        service.name,
        service.layer,
        service.kind,
        service.vendor_path.clone().unwrap_or_default(),
        service.upstream_url.clone().unwrap_or_default()
    );
    let embedding = generate_embedding(&text, embedding_dim);
    let payload = serde_json::json!({
        "id": service.id,
        "name": service.name,
        "layer": service.layer,
        "kind": service.kind,
        "vendor_path": service.vendor_path,
        "src_path": service.src_path,
        "upstream_url": service.upstream_url,
        "health_path": service.health_path,
        "version": service.version,
        "tags": service.tags.clone().unwrap_or_default(),
        "dependencies": service.dependencies.clone().unwrap_or_default(),
        "updated_at": Utc::now().to_rfc3339()
    });

    let response = client
        .put(format!("{}/collections/{}/points?wait=true", url, DEFAULT_COLLECTION))
        .json(&serde_json::json!({
            "points": [{
                "id": service.id,
                "vector": embedding,
                "payload": payload
            }]
        }))
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Qdrant upsert failed: {}", response.status());
    }

    Ok(())
}

fn generate_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(dim);
    let mut seed: u64 = 0;

    while values.len() < dim {
        let mut hasher = blake3::Hasher::new();
        hasher.update(text.as_bytes());
        hasher.update(&seed.to_le_bytes());
        let hash = hasher.finalize();
        let bytes = hash.as_bytes();

        for chunk in bytes.chunks(4) {
            if values.len() >= dim {
                break;
            }
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            let raw = u32::from_le_bytes(buf);
            let normalized = (raw as f32 / u32::MAX as f32) * 2.0 - 1.0;
            values.push(normalized);
        }

        seed = seed.wrapping_add(1);
    }

    values
}
