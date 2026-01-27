//! Service Registry - Service Discovery & Orchestration for serviceCore
//!
//! This service provides:
//! - Service registration and discovery
//! - Health check monitoring
//! - Service metadata management
//! - Configuration distribution

use actix_web::{web, App, HttpResponse, HttpServer, middleware};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tracing::{info, warn, error, Level};
use tracing_subscriber::{fmt, EnvFilter};
use uuid::Uuid;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("Service not found: {0}")]
    ServiceNotFound(String),
    #[error("Invalid service configuration: {0}")]
    InvalidConfig(String),
    #[error("Health check failed for service: {0}")]
    HealthCheckFailed(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

// =============================================================================
// Data Models
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub host: String,
    pub port: u16,
    pub protocol: String,
    pub health_endpoint: String,
    pub status: ServiceStatus,
    pub metadata: serde_json::Value,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Healthy,
    Unhealthy,
    Unknown,
    Draining,
}

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub name: String,
    pub version: String,
    pub host: String,
    pub port: u16,
    #[serde(default = "default_protocol")]
    pub protocol: String,
    #[serde(default = "default_health_endpoint")]
    pub health_endpoint: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_protocol() -> String {
    "http".to_string()
}

fn default_health_endpoint() -> String {
    "/health".to_string()
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub id: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub services_registered: usize,
    pub services_healthy: usize,
}

#[derive(Debug, Serialize)]
pub struct ServiceListResponse {
    pub services: Vec<ServiceInfo>,
    pub total: usize,
}

// =============================================================================
// Application State
// =============================================================================

pub struct AppState {
    services: DashMap<String, ServiceInfo>,
    start_time: DateTime<Utc>,
    config_path: Option<String>,
}

impl AppState {
    pub fn new(config_path: Option<String>) -> Self {
        Self {
            services: DashMap::new(),
            start_time: Utc::now(),
            config_path,
        }
    }

    pub fn register_service(&self, request: RegisterRequest) -> ServiceInfo {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        let service = ServiceInfo {
            id: id.clone(),
            name: request.name,
            version: request.version,
            host: request.host,
            port: request.port,
            protocol: request.protocol,
            health_endpoint: request.health_endpoint,
            status: ServiceStatus::Unknown,
            metadata: request.metadata,
            registered_at: now,
            last_heartbeat: now,
            tags: request.tags,
        };
        
        self.services.insert(id.clone(), service.clone());
        info!("Registered service: {} ({})", service.name, id);
        service
    }

    pub fn get_service(&self, id: &str) -> Option<ServiceInfo> {
        self.services.get(id).map(|s| s.clone())
    }

    pub fn get_services_by_name(&self, name: &str) -> Vec<ServiceInfo> {
        self.services
            .iter()
            .filter(|entry| entry.value().name == name)
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn list_services(&self) -> Vec<ServiceInfo> {
        self.services.iter().map(|entry| entry.value().clone()).collect()
    }

    pub fn deregister_service(&self, id: &str) -> Option<ServiceInfo> {
        self.services.remove(id).map(|(_, s)| {
            info!("Deregistered service: {} ({})", s.name, id);
            s
        })
    }

    pub fn update_heartbeat(&self, id: &str) -> bool {
        if let Some(mut service) = self.services.get_mut(id) {
            service.last_heartbeat = Utc::now();
            service.status = ServiceStatus::Healthy;
            true
        } else {
            false
        }
    }

    pub fn update_service_status(&self, id: &str, status: ServiceStatus) {
        if let Some(mut service) = self.services.get_mut(id) {
            service.status = status;
        }
    }

    pub fn healthy_count(&self) -> usize {
        self.services
            .iter()
            .filter(|entry| entry.value().status == ServiceStatus::Healthy)
            .count()
    }

    pub fn uptime_seconds(&self) -> u64 {
        (Utc::now() - self.start_time).num_seconds() as u64
    }
}

// =============================================================================
// HTTP Handlers
// =============================================================================

/// Health check endpoint
async fn health(data: web::Data<Arc<AppState>>) -> HttpResponse {
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: data.uptime_seconds(),
        services_registered: data.services.len(),
        services_healthy: data.healthy_count(),
    };
    HttpResponse::Ok().json(response)
}

/// Register a new service
async fn register_service(
    data: web::Data<Arc<AppState>>,
    request: web::Json<RegisterRequest>,
) -> HttpResponse {
    let service = data.register_service(request.into_inner());
    HttpResponse::Created().json(RegisterResponse {
        id: service.id,
        message: "Service registered successfully".to_string(),
    })
}

/// Get a specific service by ID
async fn get_service(
    data: web::Data<Arc<AppState>>,
    path: web::Path<String>,
) -> HttpResponse {
    let id = path.into_inner();
    match data.get_service(&id) {
        Some(service) => HttpResponse::Ok().json(service),
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Service not found",
            "id": id
        })),
    }
}

/// List all services or filter by name
async fn list_services(
    data: web::Data<Arc<AppState>>,
    query: web::Query<std::collections::HashMap<String, String>>,
) -> HttpResponse {
    let services = if let Some(name) = query.get("name") {
        data.get_services_by_name(name)
    } else {
        data.list_services()
    };
    
    let total = services.len();
    HttpResponse::Ok().json(ServiceListResponse { services, total })
}

/// Deregister a service
async fn deregister_service(
    data: web::Data<Arc<AppState>>,
    path: web::Path<String>,
) -> HttpResponse {
    let id = path.into_inner();
    match data.deregister_service(&id) {
        Some(_) => HttpResponse::Ok().json(serde_json::json!({
            "message": "Service deregistered successfully",
            "id": id
        })),
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Service not found",
            "id": id
        })),
    }
}

/// Heartbeat endpoint for services to report health
async fn heartbeat(
    data: web::Data<Arc<AppState>>,
    path: web::Path<String>,
) -> HttpResponse {
    let id = path.into_inner();
    if data.update_heartbeat(&id) {
        HttpResponse::Ok().json(serde_json::json!({
            "message": "Heartbeat received",
            "id": id
        }))
    } else {
        HttpResponse::NotFound().json(serde_json::json!({
            "error": "Service not found",
            "id": id
        }))
    }
}

/// Get service discovery information (for clients)
async fn discover(
    data: web::Data<Arc<AppState>>,
    path: web::Path<String>,
) -> HttpResponse {
    let name = path.into_inner();
    let services: Vec<_> = data
        .get_services_by_name(&name)
        .into_iter()
        .filter(|s| s.status == ServiceStatus::Healthy)
        .collect();
    
    if services.is_empty() {
        HttpResponse::NotFound().json(serde_json::json!({
            "error": "No healthy services found",
            "name": name
        }))
    } else {
        HttpResponse::Ok().json(serde_json::json!({
            "name": name,
            "instances": services,
            "count": services.len()
        }))
    }
}

/// Root endpoint
async fn index() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "service": "service-registry",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Service Discovery & Orchestration for serviceCore",
        "endpoints": {
            "health": "/health",
            "register": "POST /api/v1/services",
            "list": "GET /api/v1/services",
            "get": "GET /api/v1/services/{id}",
            "deregister": "DELETE /api/v1/services/{id}",
            "heartbeat": "POST /api/v1/services/{id}/heartbeat",
            "discover": "GET /api/v1/discover/{name}"
        }
    }))
}

// =============================================================================
// Main Entry Point
// =============================================================================

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .json()
        .init();

    // Get configuration from environment
    let bind_addr = std::env::var("SERVICE_REGISTRY_BIND")
        .unwrap_or_else(|_| "0.0.0.0:8100".to_string());
    let config_path = std::env::var("SERVICE_REGISTRY_CONFIG").ok();

    info!("Starting Service Registry on {}", bind_addr);
    
    // Create application state
    let state = Arc::new(AppState::new(config_path));

    // Start HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .wrap(middleware::Logger::default())
            .wrap(middleware::Compress::default())
            .route("/", web::get().to(index))
            .route("/health", web::get().to(health))
            .service(
                web::scope("/api/v1")
                    .route("/services", web::post().to(register_service))
                    .route("/services", web::get().to(list_services))
                    .route("/services/{id}", web::get().to(get_service))
                    .route("/services/{id}", web::delete().to(deregister_service))
                    .route("/services/{id}/heartbeat", web::post().to(heartbeat))
                    .route("/discover/{name}", web::get().to(discover))
            )
    })
    .bind(&bind_addr)?
    .run()
    .await
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_service() {
        let state = AppState::new(None);
        let request = RegisterRequest {
            name: "test-service".to_string(),
            version: "1.0.0".to_string(),
            host: "localhost".to_string(),
            port: 8080,
            protocol: "http".to_string(),
            health_endpoint: "/health".to_string(),
            metadata: serde_json::json!({}),
            tags: vec!["test".to_string()],
        };
        
        let service = state.register_service(request);
        assert_eq!(service.name, "test-service");
        assert_eq!(service.version, "1.0.0");
        assert_eq!(service.status, ServiceStatus::Unknown);
    }

    #[test]
    fn test_heartbeat_updates_status() {
        let state = AppState::new(None);
        let request = RegisterRequest {
            name: "test-service".to_string(),
            version: "1.0.0".to_string(),
            host: "localhost".to_string(),
            port: 8080,
            protocol: "http".to_string(),
            health_endpoint: "/health".to_string(),
            metadata: serde_json::json!({}),
            tags: vec![],
        };
        
        let service = state.register_service(request);
        assert!(state.update_heartbeat(&service.id));
        
        let updated = state.get_service(&service.id).unwrap();
        assert_eq!(updated.status, ServiceStatus::Healthy);
    }

    #[test]
    fn test_deregister_service() {
        let state = AppState::new(None);
        let request = RegisterRequest {
            name: "test-service".to_string(),
            version: "1.0.0".to_string(),
            host: "localhost".to_string(),
            port: 8080,
            protocol: "http".to_string(),
            health_endpoint: "/health".to_string(),
            metadata: serde_json::json!({}),
            tags: vec![],
        };
        
        let service = state.register_service(request);
        let id = service.id.clone();
        
        assert!(state.deregister_service(&id).is_some());
        assert!(state.get_service(&id).is_none());
    }
}