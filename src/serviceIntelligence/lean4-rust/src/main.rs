/// Lean4 Parser HTTP Server
/// High-performance Actix-web server for Markdown → Lean4 → SCIP conversion

use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, Result, http::header};
use anyhow::Context as AnyhowContext;
use chrono::Utc;
use lean4_parser::{
    ConversionPipeline, ParseRequest, ParseResult,
    FeatureGenerator, TemplateGenerator, TemplateConfig,
    N8nToMarkdownConverter, CONFIG, is_path_allowed,
};
use log::{error, info, warn};
use std::sync::RwLock;
use serde::Deserialize;

/// Maximum markdown input size (from config)
const MAX_MARKDOWN_SIZE: usize = 5 * 1024 * 1024; // 5MB fallback

/// Application state with pipeline instance (using RwLock for better concurrency)
struct AppState {
    pipeline: RwLock<ConversionPipeline>,
}

/// Input validation error response
fn validation_error(message: &str) -> HttpResponse {
    HttpResponse::BadRequest().json(serde_json::json!({
        "success": false,
        "error": message,
        "timestamp": Utc::now(),
    }))
}

/// Internal error response with context
fn internal_error(context: &str, error: &str) -> HttpResponse {
    error!("{}: {}", context, error);
    HttpResponse::InternalServerError().json(serde_json::json!({
        "success": false,
        "error": format!("{}: {}", context, error),
        "timestamp": Utc::now(),
    }))
}

/// Health check endpoint
async fn health() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "lean4-parser-rust",
        "timestamp": Utc::now(),
        "version": env!("CARGO_PKG_VERSION"),
        "config": {
            "max_body_size": CONFIG.server.max_body_size,
            "cors_allow_any": CONFIG.cors.allow_any_origin,
        }
    })))
}

/// Parse markdown and convert to Lean4 and/or SCIP
async fn parse(
    data: web::Data<AppState>,
    req: web::Json<ParseRequest>,
) -> Result<HttpResponse> {
    // Input validation
    let max_size = CONFIG.filesystem.max_input_size.min(MAX_MARKDOWN_SIZE);
    if req.markdown.len() > max_size {
        return Ok(validation_error(&format!(
            "Markdown input too large. Maximum size: {} bytes, received: {} bytes",
            max_size,
            req.markdown.len()
        )));
    }

    if req.markdown.trim().is_empty() {
        return Ok(validation_error("Markdown input cannot be empty"));
    }

    info!("Processing parse request (markdown length: {})", req.markdown.len());

    // Get pipeline with write lock
    let mut pipeline = match data.pipeline.write() {
        Ok(p) => p,
        Err(e) => {
            return Ok(internal_error("Failed to acquire pipeline lock", &e.to_string()));
        }
    };

    // Process markdown
    match pipeline.process_markdown(&req.markdown) {
        Ok(results) => {
            let stats = results.get_statistics();

            let response = ParseResult {
                success: true,
                lean4_code: if req.generate_lean4 {
                    Some(results.lean4_code)
                } else {
                    None
                },
                lean4_elements_count: results.lean4_elements.len(),
                scip_json: if req.generate_scip {
                    Some(results.scip_json)
                } else {
                    None
                },
                scip_markdown: if req.generate_scip {
                    Some(results.scip_markdown)
                } else {
                    None
                },
                scip_elements_count: results.scip_elements.len(),
                statistics: Some(stats),
                error: None,
                timestamp: Utc::now(),
            };

            info!(
                "Parse successful: {} Lean4 elements, {} SCIP elements",
                results.lean4_elements.len(),
                results.scip_elements.len()
            );

            Ok(HttpResponse::Ok().json(response))
        }
        Err(e) => {
            Ok(internal_error("Parse failed", &e.to_string()))
        }
    }
}

/// Get service statistics
async fn statistics() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "service": "lean4-parser-rust",
        "version": env!("CARGO_PKG_VERSION"),
        "supported_conversions": [
            "markdown → lean4",
            "markdown → scip",
            "markdown → lean4 → scip (full pipeline)"
        ],
        "element_types": {
            "lean4": ["axiom", "theorem", "def", "structure", "inductive"],
            "scip": [
                "requirement",
                "constraint",
                "verification",
                "compliance_rule",
                "control",
                "policy",
                "procedure",
                "evidence"
            ]
        },
        "performance": {
            "language": "Rust",
            "zero_copy_parsing": true,
            "concurrent_processing": true
        },
        "limits": {
            "max_input_size": CONFIG.filesystem.max_input_size,
            "max_body_size": CONFIG.server.max_body_size,
        },
        "timestamp": Utc::now(),
    })))
}

/// Request model for feature generation
#[derive(Debug, Deserialize)]
struct GenerateRequest {
    markdown: String,
    #[serde(default = "default_output_dir")]
    output_dir: String,
}

fn default_output_dir() -> String {
    CONFIG.filesystem.output_dir.clone()
}

/// Generate Gitea feature from markdown specification
async fn generate_feature(req: web::Json<GenerateRequest>) -> Result<HttpResponse> {
    // Input validation
    let max_size = CONFIG.filesystem.max_input_size;
    if req.markdown.len() > max_size {
        return Ok(validation_error(&format!(
            "Markdown input too large. Maximum: {} bytes",
            max_size
        )));
    }

    if req.markdown.trim().is_empty() {
        return Ok(validation_error("Markdown input cannot be empty"));
    }

    // Path validation
    if !is_path_allowed(&req.output_dir) {
        warn!("Blocked path traversal attempt: {}", req.output_dir);
        return Ok(validation_error(&format!(
            "Output directory not allowed: {}. Allowed paths: {:?}",
            req.output_dir,
            CONFIG.filesystem.allowed_paths
        )));
    }

    info!("Processing feature generation request");

    let mut generator = FeatureGenerator::new();

    match generator.parse_feature_spec(&req.markdown) {
        Ok(spec) => {
            // Generate Go code
            let go_code = generator.generate_gitea_code(&spec);

            // Generate Lean4 proofs
            let lean4_proofs = generator.generate_lean4_proofs(&spec);

            // Generate SCIP spec
            let scip_elements = generator.generate_scip_spec(&spec);
            let scip_json = serde_json::to_string_pretty(&serde_json::json!({
                "scip_version": "1.0",
                "feature": spec.name,
                "elements": scip_elements,
            })).context("Failed to serialize SCIP JSON")
              .map_err(|e| {
                  error!("SCIP serialization failed: {}", e);
                  e
              })
              .unwrap_or_else(|_| "{}".to_string());

            info!("Feature generation successful: {}", spec.name);

            Ok(HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "feature_name": spec.name,
                "requirements_count": spec.requirements.len(),
                "api_endpoints_count": spec.api_endpoints.len(),
                "data_models_count": spec.data_models.len(),
                "go_code": go_code,
                "lean4_proofs": lean4_proofs,
                "scip_spec": scip_json,
                "timestamp": Utc::now(),
            })))
        }
        Err(e) => {
            Ok(internal_error("Feature generation failed", &e.to_string()))
        }
    }
}

/// Request model for template generation
#[derive(Debug, Deserialize)]
struct TemplateRequest {
    #[serde(default = "default_feature_name")]
    feature_name: String,
    #[serde(default = "default_template_type")]
    template_type: String, // full, quick, guided, example
    #[serde(default)]
    include_api_section: bool,
    #[serde(default)]
    include_data_model: bool,
    #[serde(default)]
    include_validation: bool,
    #[serde(default)]
    include_security: bool,
    #[serde(default)]
    include_examples: bool,
}

fn default_feature_name() -> String {
    "New Feature".to_string()
}

fn default_template_type() -> String {
    "full".to_string()
}

/// Generate template for feature specification
async fn generate_template(req: web::Json<TemplateRequest>) -> Result<HttpResponse> {
    // Input validation
    if req.feature_name.len() > 256 {
        return Ok(validation_error("Feature name too long. Maximum 256 characters."));
    }

    info!("Processing template generation request: {}", req.template_type);

    let generator = TemplateGenerator::new();

    let template = match req.template_type.as_str() {
        "quick" => generator.generate_quick_template(&req.feature_name),
        "guided" => generator.generate_guided_template(),
        "example" => generator.generate_example_template(),
        "full" | _ => {
            let config = TemplateConfig {
                feature_name: req.feature_name.clone(),
                include_api_section: req.include_api_section,
                include_data_model: req.include_data_model,
                include_validation: req.include_validation,
                include_security: req.include_security,
                include_examples: req.include_examples,
            };
            generator.generate_feature_template(&config)
        }
    };

    info!("Template generation successful: {} ({})", req.feature_name, req.template_type);

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "template_type": req.template_type,
        "feature_name": req.feature_name,
        "template": template,
        "timestamp": Utc::now(),
    })))
}

/// Request model for n8n to markdown conversion
#[derive(Debug, Deserialize)]
struct N8nToMdRequest {
    workflow_json: String,
}

/// Convert n8n workflow JSON to markdown specification
async fn n8n_to_markdown(req: web::Json<N8nToMdRequest>) -> Result<HttpResponse> {
    // Input validation
    let max_size = CONFIG.filesystem.max_input_size;
    if req.workflow_json.len() > max_size {
        return Ok(validation_error(&format!(
            "Workflow JSON too large. Maximum: {} bytes",
            max_size
        )));
    }

    if req.workflow_json.trim().is_empty() {
        return Ok(validation_error("Workflow JSON cannot be empty"));
    }

    // Validate it's actually JSON
    if serde_json::from_str::<serde_json::Value>(&req.workflow_json).is_err() {
        return Ok(validation_error("Invalid JSON format"));
    }

    info!("Processing n8n to markdown conversion");

    let mut converter = N8nToMarkdownConverter::new();

    match converter.convert_to_markdown(&req.workflow_json) {
        Ok(markdown) => {
            info!("n8n conversion successful");

            Ok(HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "markdown": markdown,
                "next_steps": [
                    "Review and refine the generated specification",
                    "Use /generate endpoint to create Gitea feature",
                    "Implement business logic in generated Go code"
                ],
                "timestamp": Utc::now(),
            })))
        }
        Err(e) => {
            Ok(internal_error("n8n conversion failed", &e.to_string()))
        }
    }
}

/// Get example inputs
async fn examples() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "examples": [
            {
                "name": "Simple Requirement",
                "markdown": "# Requirements\n- MUST validate user input",
                "description": "Single requirement with MUST keyword"
            },
            {
                "name": "Multiple Requirements with Constraints",
                "markdown": "# Payment Processing\n\n## Requirements\n- MUST validate payment amount\n- SHALL verify user authentication\n\n## Constraints\n- MUST NOT process amounts over $10,000 without approval\n- Transaction time MUST be under 5 seconds",
                "description": "Requirements and constraints for payment processing"
            },
            {
                "name": "Data Structure Table",
                "markdown": "# Data Model\n\n| Field | Type | Required |\n|-------|------|----------|\n| user_id | string | yes |\n| amount | decimal | yes |\n| timestamp | datetime | yes |",
                "description": "Markdown table converted to Lean4 structures"
            }
        ],
        "timestamp": Utc::now(),
    })))
}

/// Build CORS configuration based on environment
fn build_cors() -> Cors {
    let config = &CONFIG.cors;

    if config.allow_any_origin {
        warn!("CORS is configured to allow any origin - not recommended for production");
        Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(config.max_age_secs as usize)
    } else if config.allowed_origins.is_empty() {
        // No origins configured - restrictive default (same-origin only)
        info!("CORS: No origins configured, using restrictive defaults");
        Cors::default()
            .allowed_methods(vec!["GET", "POST", "PUT", "DELETE"])
            .allowed_headers(vec![header::CONTENT_TYPE, header::AUTHORIZATION])
            .max_age(config.max_age_secs as usize)
    } else {
        // Specific origins configured
        info!("CORS: Allowing origins: {:?}", config.allowed_origins);
        let mut cors = Cors::default()
            .allowed_methods(vec!["GET", "POST", "PUT", "DELETE", "OPTIONS"])
            .allowed_headers(vec![header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
            .max_age(config.max_age_secs as usize);

        for origin in &config.allowed_origins {
            cors = cors.allowed_origin(origin);
        }

        cors
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger with configured level
    let log_level = &CONFIG.log.level;
    env_logger::init_from_env(env_logger::Env::new().default_filter_or(log_level));

    info!("Starting Lean4 Parser Rust Server");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));
    info!("Log level: {}", log_level);

    // Log configuration
    info!("Configuration:");
    info!("  Server: {}:{}", CONFIG.server.host, CONFIG.server.port);
    info!("  Max body size: {} bytes", CONFIG.server.max_body_size);
    info!("  Max input size: {} bytes", CONFIG.filesystem.max_input_size);
    info!("  Output directory: {}", CONFIG.filesystem.output_dir);
    info!("  Allowed paths: {:?}", CONFIG.filesystem.allowed_paths);
    info!("  CORS allow any origin: {}", CONFIG.cors.allow_any_origin);
    if !CONFIG.cors.allowed_origins.is_empty() {
        info!("  CORS allowed origins: {:?}", CONFIG.cors.allowed_origins);
    }

    // Create application state with RwLock for better concurrency
    let app_state = web::Data::new(AppState {
        pipeline: RwLock::new(ConversionPipeline::new()),
    });

    let bind_address = CONFIG.bind_address();
    info!("Binding to {}", bind_address);

    // Start HTTP server
    HttpServer::new(move || {
        App::new()
            .wrap(build_cors())
            .app_data(app_state.clone())
            .app_data(web::JsonConfig::default()
                .limit(CONFIG.server.max_body_size)
                .error_handler(|err, _req| {
                    let message = format!("JSON parsing error: {}", err);
                    actix_web::error::InternalError::from_response(
                        err,
                        HttpResponse::BadRequest().json(serde_json::json!({
                            "success": false,
                            "error": message,
                            "timestamp": Utc::now(),
                        }))
                    ).into()
                })
            )
            .route("/health", web::get().to(health))
            .route("/parse", web::post().to(parse))
            .route("/generate", web::post().to(generate_feature))
            .route("/template", web::post().to(generate_template))
            .route("/n8n-to-md", web::post().to(n8n_to_markdown))
            .route("/statistics", web::get().to(statistics))
            .route("/examples", web::get().to(examples))
    })
    .bind(&bind_address)?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

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

    #[actix_web::test]
    async fn test_statistics_endpoint() {
        let app = test::init_service(
            App::new().route("/statistics", web::get().to(statistics))
        ).await;

        let req = test::TestRequest::get()
            .uri("/statistics")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_examples_endpoint() {
        let app = test::init_service(
            App::new().route("/examples", web::get().to(examples))
        ).await;

        let req = test::TestRequest::get()
            .uri("/examples")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_parse_validation_empty() {
        let app_state = web::Data::new(AppState {
            pipeline: RwLock::new(ConversionPipeline::new()),
        });

        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/parse", web::post().to(parse))
        ).await;

        let req = test::TestRequest::post()
            .uri("/parse")
            .set_json(&serde_json::json!({
                "markdown": "",
                "generate_lean4": true,
                "generate_scip": true
            }))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 400);
    }

    #[actix_web::test]
    async fn test_template_generation() {
        let app = test::init_service(
            App::new().route("/template", web::post().to(generate_template))
        ).await;

        let req = test::TestRequest::post()
            .uri("/template")
            .set_json(&serde_json::json!({
                "feature_name": "Test Feature",
                "template_type": "quick"
            }))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }
}
