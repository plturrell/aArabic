use actix_web::{web, App, HttpResponse, HttpServer};
use log::{info, warn};
use serde::Serialize;
use std::env;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::time::{timeout, Duration};

#[derive(Clone, Serialize)]
pub struct ServiceConfig {
    pub id: String,
    pub name: String,
    pub layer: String,
    pub kind: String,
    pub vendor_path: String,
    pub src_path: String,
    pub version: String,
    pub port: u16,
    pub upstream_url: Option<String>,
    pub health_path: Option<String>,
    pub tags: Vec<String>,
    pub dependencies: Vec<String>,
    pub registry_url: Option<String>,
}

#[derive(Clone)]
struct AppState {
    config: ServiceConfig,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct HealthStatus {
    status: String,
    service: String,
    version: String,
    upstream: Option<String>,
    upstream_status: String,
}

#[derive(Serialize)]
struct MetadataResponse {
    service: ServiceConfig,
}

impl ServiceConfig {
    pub fn new(
        id: &str,
        name: &str,
        layer: &str,
        kind: &str,
        vendor_path: &str,
        src_path: &str,
        port: u16,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            layer: layer.to_string(),
            kind: kind.to_string(),
            vendor_path: vendor_path.to_string(),
            src_path: src_path.to_string(),
            version: "local".to_string(),
            port,
            upstream_url: None,
            health_path: None,
            tags: Vec::new(),
            dependencies: Vec::new(),
            registry_url: None,
        }
    }

    pub fn with_upstream(mut self, upstream_url: &str, health_path: Option<&str>) -> Self {
        self.upstream_url = Some(upstream_url.to_string());
        self.health_path = health_path.map(|value| value.to_string());
        self
    }

    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.tags = tags.iter().map(|tag| tag.to_string()).collect();
        self
    }

    pub fn with_dependencies(mut self, dependencies: &[&str]) -> Self {
        self.dependencies = dependencies
            .iter()
            .map(|dep| dep.to_string())
            .collect();
        self
    }

    pub fn apply_env(mut self) -> Self {
        if let Ok(id) = env::var("SERVICE_ID") {
            self.id = id;
        }

        if let Ok(name) = env::var("SERVICE_NAME") {
            self.name = name;
        }

        if let Ok(layer) = env::var("SERVICE_LAYER") {
            self.layer = layer;
        }

        if let Ok(kind) = env::var("SERVICE_KIND") {
            self.kind = kind;
        }

        if let Ok(vendor_path) = env::var("SERVICE_VENDOR_PATH") {
            self.vendor_path = vendor_path;
        }

        if let Ok(src_path) = env::var("SERVICE_SRC_PATH") {
            self.src_path = src_path;
        }

        if let Ok(port) = env::var("SERVICE_PORT") {
            if let Ok(parsed) = port.parse::<u16>() {
                self.port = parsed;
            }
        }

        if let Ok(url) = env::var("UPSTREAM_URL") {
            self.upstream_url = Some(url);
        }

        if let Ok(path) = env::var("HEALTH_PATH") {
            self.health_path = Some(path);
        }

        if let Ok(version) = env::var("SERVICE_VERSION") {
            self.version = version;
        }

        if let Ok(registry_url) = env::var("SERVICE_REGISTRY_URL") {
            self.registry_url = Some(registry_url);
        }

        if let Ok(tags) = env::var("SERVICE_TAGS") {
            self.tags = split_csv(&tags);
        }

        if let Ok(deps) = env::var("SERVICE_DEPENDENCIES") {
            self.dependencies = split_csv(&deps);
        }

        self
    }
}

pub async fn run(config: ServiceConfig) -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let bind_address = format!("0.0.0.0:{}", config.port);
    info!("Starting {} on {}", config.name, bind_address);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("failed to build HTTP client");

    let state = Arc::new(AppState {
        config: config.clone(),
        client,
    });

    if let Some(registry_url) = &config.registry_url {
        if let Err(error) = register_with_registry(state.as_ref(), registry_url).await {
            warn!("Registry registration failed: {}", error);
        }
    }

    let shared_state = web::Data::new(state);

    HttpServer::new(move || {
        App::new()
            .app_data(shared_state.clone())
            .route("/health", web::get().to(health))
            .route("/metadata", web::get().to(metadata))
    })
    .bind(bind_address)?
    .run()
    .await
}

async fn health(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let config = &state.config;
    let upstream_status = check_upstream(&state.client, config).await;

    HttpResponse::Ok().json(HealthStatus {
        status: "healthy".to_string(),
        service: config.name.clone(),
        version: config.version.clone(),
        upstream: config.upstream_url.clone(),
        upstream_status,
    })
}

async fn metadata(state: web::Data<Arc<AppState>>) -> HttpResponse {
    HttpResponse::Ok().json(MetadataResponse {
        service: state.config.clone(),
    })
}

async fn check_upstream(client: &reqwest::Client, config: &ServiceConfig) -> String {
    let upstream_url = match &config.upstream_url {
        Some(url) => url,
        None => return "not_configured".to_string(),
    };

    if upstream_url.starts_with("bolt://") {
        return check_tcp_upstream(upstream_url, 7687).await;
    }

    if upstream_url.starts_with("redis://") || upstream_url.starts_with("rediss://") {
        return check_tcp_upstream(upstream_url, 6379).await;
    }

    if upstream_url.starts_with("postgres://") || upstream_url.starts_with("postgresql://") {
        return check_tcp_upstream(upstream_url, 5432).await;
    }

    if !upstream_url.starts_with("http") {
        return "skipped".to_string();
    }

    let health_url = if let Some(path) = &config.health_path {
        if path.starts_with("http") {
            path.to_string()
        } else {
            format!("{}{}", upstream_url.trim_end_matches('/'), path)
        }
    } else {
        upstream_url.to_string()
    };

    match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => "reachable".to_string(),
        Ok(response) => format!("unhealthy({})", response.status()),
        Err(_) => "unreachable".to_string(),
    }
}

async fn register_with_registry(state: &AppState, registry_url: &str) -> Result<(), reqwest::Error> {
    let payload = serde_json::json!({
        "id": state.config.id,
        "name": state.config.name,
        "layer": state.config.layer,
        "kind": state.config.kind,
        "vendor_path": state.config.vendor_path,
        "src_path": state.config.src_path,
        "version": state.config.version,
        "upstream_url": state.config.upstream_url,
        "health_path": state.config.health_path,
        "tags": state.config.tags,
        "dependencies": state.config.dependencies
    });

    state
        .client
        .post(format!("{}/services/register", registry_url.trim_end_matches('/')))
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;

    Ok(())
}

async fn check_tcp_upstream(url: &str, default_port: u16) -> String {
    let trimmed = url
        .trim_start_matches("bolt://")
        .trim_start_matches("redis://")
        .trim_start_matches("rediss://")
        .trim_start_matches("postgres://")
        .trim_start_matches("postgresql://");
    let host_port = trimmed
        .split('@')
        .last()
        .unwrap_or(trimmed)
        .split('/')
        .next()
        .unwrap_or(trimmed);
    let mut parts = host_port.split(':');
    let host = parts.next().unwrap_or("");
    let port = parts
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(default_port);

    if host.is_empty() {
        return "unreachable".to_string();
    }

    let addr = format!("{}:{}", host, port);
    match timeout(Duration::from_secs(2), TcpStream::connect(addr)).await {
        Ok(Ok(_)) => "reachable".to_string(),
        _ => "unreachable".to_string(),
    }
}

fn split_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect()
}
