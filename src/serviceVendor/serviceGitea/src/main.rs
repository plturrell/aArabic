/// serviceGitea Main Server
/// Aggregates all generated feature services with Gitea integration

use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, Result};
use chrono::Utc;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

mod git_integration;
mod feature_flags;
mod version_control;

use git_integration::GiteaClient;
use feature_flags::FeatureFlagManager;

/// Application state
struct AppState {
    gitea_client: Mutex<GiteaClient>,
    feature_flags: Mutex<FeatureFlagManager>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CommitArtifactRequest {
    feature_name: String,
    artifacts: FeatureArtifacts,
    commit_message: String,
    branch: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FeatureArtifacts {
    markdown_spec: String,
    rust_code: String,
    lean4_proofs: String,
    scip_spec: String,
    n8n_workflow: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct FeatureFlagRequest {
    feature_name: String,
    enabled: bool,
    rollout_percentage: Option<f32>,
    target_users: Option<Vec<String>>,
}

/// Health check
async fn health() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "serviceGitea",
        "version": env!("CARGO_PKG_VERSION"),
        "port": 8004,
        "features": {
            "version_control": true,
            "feature_flags": true,
            "gitea_integration": true
        },
        "timestamp": Utc::now(),
    })))
}

/// Commit all feature artifacts to Gitea
async fn commit_to_gitea(
    data: web::Data<AppState>,
    req: web::Json<CommitArtifactRequest>,
) -> Result<HttpResponse> {
    info!("Committing feature '{}' to Gitea", req.feature_name);
    
    let mut gitea = match data.gitea_client.lock() {
        Ok(g) => g,
        Err(e) => {
            error!("Failed to acquire gitea lock: {}", e);
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Internal server error"
            })));
        }
    };

    // Create repository if not exists
    let repo_name = format!("feature-{}", req.feature_name);
    match gitea.ensure_repository(&repo_name).await {
        Ok(_) => info!("Repository {} ready", repo_name),
        Err(e) => {
            error!("Failed to create repository: {}", e);
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": e.to_string()
            })));
        }
    }

    let branch = req.branch.as_deref().unwrap_or("main");

    // Commit artifacts
    let files = vec![
        ("README.md", req.artifacts.markdown_spec.as_str()),
        ("src/lib.rs", req.artifacts.rust_code.as_str()),
        ("proofs/feature.lean", req.artifacts.lean4_proofs.as_str()),
        ("specs/scip.json", req.artifacts.scip_spec.as_str()),
        ("workflows/n8n.json", req.artifacts.n8n_workflow.as_str()),
    ];

    match gitea.commit_files(&repo_name, branch, &files, &req.commit_message).await {
        Ok(commit_sha) => {
            info!("Successfully committed to {}/{}: {}", repo_name, branch, commit_sha);
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "repository": repo_name,
                "branch": branch,
                "commit_sha": commit_sha,
                "url": format!("http://localhost:3000/{}/{}", gitea.get_org(), repo_name),
                "timestamp": Utc::now(),
            })))
        }
        Err(e) => {
            error!("Failed to commit: {}", e);
            Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": e.to_string()
            })))
        }
    }
}

/// Set feature flag
async fn set_feature_flag(
    data: web::Data<AppState>,
    req: web::Json<FeatureFlagRequest>,
) -> Result<HttpResponse> {
    info!("Setting feature flag for '{}'", req.feature_name);
    
    let mut flags = data.feature_flags.lock().unwrap();
    
    flags.set_flag(
        &req.feature_name,
        req.enabled,
        req.rollout_percentage,
        req.target_users.clone(),
    );

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "feature": req.feature_name,
        "enabled": req.enabled,
        "rollout_percentage": req.rollout_percentage,
        "timestamp": Utc::now(),
    })))
}

/// Get feature flag status
async fn get_feature_flag(
    data: web::Data<AppState>,
    feature_name: web::Path<String>,
) -> Result<HttpResponse> {
    let flags = data.feature_flags.lock().unwrap();
    
    match flags.get_flag(&feature_name) {
        Some(flag) => Ok(HttpResponse::Ok().json(flag)),
        None => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Feature flag not found",
            "feature": feature_name.as_str()
        })))
    }
}

/// Check if feature is enabled for user
async fn check_feature_enabled(
    data: web::Data<AppState>,
    path: web::Path<(String, String)>,
) -> Result<HttpResponse> {
    let (feature_name, user_id) = path.into_inner();
    let flags = data.feature_flags.lock().unwrap();
    
    let enabled = flags.is_enabled(&feature_name, &user_id);
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "feature": feature_name,
        "user_id": user_id,
        "enabled": enabled,
        "timestamp": Utc::now(),
    })))
}

/// List all repositories
async fn list_repositories(
    data: web::Data<AppState>,
) -> Result<HttpResponse> {
    let gitea = data.gitea_client.lock().unwrap();
    
    match gitea.list_repositories().await {
        Ok(repos) => Ok(HttpResponse::Ok().json(serde_json::json!({
            "repositories": repos,
            "count": repos.len(),
            "timestamp": Utc::now(),
        }))),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": e.to_string()
        })))
    }
}

/// Get repository versions
async fn get_versions(
    data: web::Data<AppState>,
    repo_name: web::Path<String>,
) -> Result<HttpResponse> {
    let gitea = data.gitea_client.lock().unwrap();
    
    match gitea.get_commits(&repo_name, "main").await {
        Ok(commits) => Ok(HttpResponse::Ok().json(serde_json::json!({
            "repository": repo_name.as_str(),
            "versions": commits,
            "count": commits.len(),
            "timestamp": Utc::now(),
        }))),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": e.to_string()
        })))
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    
    info!("Starting serviceGitea with Version Control & Feature Flags");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));
    
    // Initialize Gitea client
    let gitea_url = std::env::var("GITEA_URL")
        .unwrap_or_else(|_| "http://localhost:3000".to_string());
    let gitea_token = std::env::var("GITEA_TOKEN")
        .expect("GITEA_TOKEN environment variable required");
    let gitea_org = std::env::var("GITEA_ORG")
        .unwrap_or_else(|_| "generated-features".to_string());
    
    let gitea_client = GiteaClient::new(&gitea_url, &gitea_token, &gitea_org);
    
    // Create application state
    let app_state = web::Data::new(AppState {
        gitea_client: Mutex::new(gitea_client),
        feature_flags: Mutex::new(FeatureFlagManager::new()),
    });
    
    let bind_address = "0.0.0.0:8004";
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
            .route("/gitea/commit", web::post().to(commit_to_gitea))
            .route("/gitea/repositories", web::get().to(list_repositories))
            .route("/gitea/versions/{repo}", web::get().to(get_versions))
            .route("/flags/set", web::post().to(set_feature_flag))
            .route("/flags/{feature}", web::get().to(get_feature_flag))
            .route("/flags/check/{feature}/{user}", web::get().to(check_feature_enabled))
    })
    .bind(bind_address)?
    .run()
    .await
}