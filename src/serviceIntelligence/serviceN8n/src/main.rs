/// serviceN8n - Meta-Orchestration Service
/// Orchestrates the complete feature engineering workflow

use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, Result};
use chrono::Utc;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use uuid::Uuid;

mod orchestrator;
mod generators;
mod deployers;

use orchestrator::Orchestrator;

/// Application state
struct AppState {
    orchestrator: Mutex<Orchestrator>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OrchestratRequest {
    workflow_json: String,
    feature_name: String,
    #[serde(default)]
    auto_deploy: bool,
}

#[derive(Debug, Serialize)]
struct OrchestratResponse {
    success: bool,
    job_id: String,
    feature_name: String,
    status: String,
    outputs: Option<GeneratedOutputs>,
    error: Option<String>,
    timestamp: chrono::DateTime<Utc>,
}

#[derive(Debug, Serialize)]
struct GeneratedOutputs {
    markdown_spec: String,
    rust_code: String,
    lean4_proofs: String,
    scip_spec: String,
    langflow_workflow: String,
    deployment_paths: DeploymentPaths,
}

#[derive(Debug, Serialize)]
struct DeploymentPaths {
    gitea: Option<String>,
    automation: Option<String>,
    n8n: Option<String>,
}

/// Health check
async fn health() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "serviceN8n",
        "version": env!("CARGO_PKG_VERSION"),
        "port": 8003,
        "timestamp": Utc::now(),
    })))
}

/// Orchestrate complete feature generation workflow
async fn orchestrate_feature(
    data: web::Data<AppState>,
    req: web::Json<OrchestratRequest>,
) -> Result<HttpResponse> {
    let job_id = Uuid::new_v4().to_string();
    
    info!("Starting orchestration job {} for feature: {}", job_id, req.feature_name);
    
    let mut orchestrator = match data.orchestrator.lock() {
        Ok(o) => o,
        Err(e) => {
            error!("Failed to acquire orchestrator lock: {}", e);
            return Ok(HttpResponse::InternalServerError().json(OrchestratResponse {
                success: false,
                job_id,
                feature_name: req.feature_name.clone(),
                status: "error".to_string(),
                outputs: None,
                error: Some("Internal server error".to_string()),
                timestamp: Utc::now(),
            }));
        }
    };
    
    // Execute orchestration workflow
    match orchestrator.orchestrate(
        &req.workflow_json,
        &req.feature_name,
        req.auto_deploy,
    ).await {
        Ok(outputs) => {
            info!("Orchestration job {} completed successfully", job_id);
            
            Ok(HttpResponse::Ok().json(OrchestratResponse {
                success: true,
                job_id,
                feature_name: req.feature_name.clone(),
                status: "completed".to_string(),
                outputs: Some(outputs),
                error: None,
                timestamp: Utc::now(),
            }))
        }
        Err(e) => {
            error!("Orchestration job {} failed: {}", job_id, e);
            
            Ok(HttpResponse::InternalServerError().json(OrchestratResponse {
                success: false,
                job_id,
                feature_name: req.feature_name.clone(),
                status: "failed".to_string(),
                outputs: None,
                error: Some(e.to_string()),
                timestamp: Utc::now(),
            }))
        }
    }
}

#[derive(Debug, Deserialize)]
struct DeployRequest {
    feature_name: String,
    rust_code: String,
    lean4_proofs: String,
    scip_spec: String,
}

/// Deploy to serviceGitea
async fn deploy_to_gitea(
    data: web::Data<AppState>,
    req: web::Json<DeployRequest>,
) -> Result<HttpResponse> {
    info!("Deploying feature '{}' to serviceGitea", req.feature_name);
    
    let mut orchestrator = data.orchestrator.lock().unwrap();
    
    match orchestrator.deploy_to_gitea(
        &req.feature_name,
        &req.rust_code,
        &req.lean4_proofs,
        &req.scip_spec,
    ).await {
        Ok(path) => {
            info!("Deployed to: {}", path);
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "deployment_path": path,
                "timestamp": Utc::now(),
            })))
        }
        Err(e) => {
            error!("Deployment failed: {}", e);
            Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": Utc::now(),
            })))
        }
    }
}

#[derive(Debug, Deserialize)]
struct LangflowRequest {
    scip_json: String,
    feature_name: String,
}

/// Convert SCIP to Langflow workflow
async fn scip_to_langflow(
    data: web::Data<AppState>,
    req: web::Json<LangflowRequest>,
) -> Result<HttpResponse> {
    info!("Converting SCIP to Langflow for feature: {}", req.feature_name);
    
    let mut orchestrator = data.orchestrator.lock().unwrap();
    
    match orchestrator.generate_langflow_workflow(&req.scip_json, &req.feature_name).await {
        Ok(workflow_json) => {
            info!("Langflow workflow generated");
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "workflow": workflow_json,
                "timestamp": Utc::now(),
            })))
        }
        Err(e) => {
            error!("Langflow generation failed: {}", e);
            Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": Utc::now(),
            })))
        }
    }
}

/// Get job status
async fn get_status(job_id: web::Path<String>) -> Result<HttpResponse> {
    info!("Status check for job: {}", job_id);
    
    // TODO: Implement job tracking
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "job_id": job_id.as_str(),
        "status": "in_progress",
        "timestamp": Utc::now(),
    })))
}

/// Service statistics
async fn statistics() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "service": "serviceN8n",
        "version": env!("CARGO_PKG_VERSION"),
        "capabilities": [
            "n8n workflow orchestration",
            "Rust code generation",
            "Langflow workflow generation",
            "Auto-deployment to serviceGitea",
            "Auto-deployment to serviceAutomation"
        ],
        "integrations": {
            "lean4_parser": "http://localhost:8002",
            "service_gitea": "../serviceCore/serviceGitea",
            "service_automation": "../serviceAutomation"
        },
        "timestamp": Utc::now(),
    })))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    
    info!("Starting serviceN8n - Meta-Orchestration Service");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));
    
    // Create application state
    let app_state = web::Data::new(AppState {
        orchestrator: Mutex::new(Orchestrator::new()),
    });
    
    let bind_address = "0.0.0.0:8003";
    info!("Binding to {}", bind_address);
    
    // Start HTTP server
    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);
        
        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .route("/health", web::get().to(health))
            .route("/orchestrate/feature", web::post().to(orchestrate_feature))
            .route("/deploy/gitea", web::post().to(deploy_to_gitea))
            .route("/scip-to-langflow", web::post().to(scip_to_langflow))
            .route("/status/{job_id}", web::get().to(get_status))
            .route("/statistics", web::get().to(statistics))
    })
    .bind(bind_address)?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::test;

    #[actix_web::test]
    async fn test_health_endpoint() {
        let app = test::init_service(
            App::new().route("/health", web::get().to(health))
        ).await;
        
        let req = test::TestRequest::get()
            .uri("/health")
            .to_request();
        
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }
}