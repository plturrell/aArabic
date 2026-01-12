/// Orchestration Engine
/// Coordinates the complete feature engineering workflow

use anyhow::{Context, Result};
use log::{info, error};
use reqwest::Client;
use serde_json::{json, Value};

use crate::{GeneratedOutputs, DeploymentPaths};
use crate::generators::{LangflowGenerator, RustGenerator};
use crate::deployers::{GiteaDeployer, AutomationDeployer};

pub struct Orchestrator {
    client: Client,
    lean4_parser_url: String,
    gitea_deployer: GiteaDeployer,
    automation_deployer: AutomationDeployer,
    langflow_generator: LangflowGenerator,
    rust_generator: RustGenerator,
}

impl Orchestrator {
    pub fn new() -> Self {
        let lean4_parser_url = std::env::var("LEAN4_PARSER_URL")
            .unwrap_or_else(|_| "http://localhost:8002".to_string());
        
        Self {
            client: Client::new(),
            lean4_parser_url,
            gitea_deployer: GiteaDeployer::new(),
            automation_deployer: AutomationDeployer::new(),
            langflow_generator: LangflowGenerator::new(),
            rust_generator: RustGenerator::new(),
        }
    }

    /// Main orchestration workflow
    pub async fn orchestrate(
        &mut self,
        workflow_json: &str,
        feature_name: &str,
        auto_deploy: bool,
    ) -> Result<GeneratedOutputs> {
        info!("Starting orchestration for feature: {}", feature_name);

        // Step 1: Convert n8n workflow to markdown specification
        info!("Step 1: Converting n8n workflow to markdown...");
        let markdown_spec = self.n8n_to_markdown(workflow_json).await
            .context("Failed to convert n8n to markdown")?;

        // Step 2: Generate code from specification
        info!("Step 2: Generating code from markdown...");
        let generation_result = self.generate_from_markdown(&markdown_spec).await
            .context("Failed to generate code")?;

        // Step 3: Convert to Rust (from Go if needed)
        info!("Step 3: Converting to Rust...");
        let rust_code = self.rust_generator.convert_go_to_rust(&generation_result.go_code)
            .context("Failed to convert to Rust")?;

        // Step 4: Generate Langflow workflow from SCIP
        info!("Step 4: Generating Langflow workflow...");
        let langflow_workflow = self.langflow_generator
            .generate_from_scip(&generation_result.scip_spec)
            .context("Failed to generate Langflow workflow")?;

        // Step 5: Deploy if auto_deploy is enabled
        let mut deployment_paths = DeploymentPaths {
            gitea: None,
            automation: None,
            n8n: None,
        };

        if auto_deploy {
            info!("Step 5: Auto-deploying...");

            // Deploy to serviceGitea
            match self.gitea_deployer.deploy(
                feature_name,
                &rust_code,
                &generation_result.lean4_proofs,
                &generation_result.scip_spec,
            ).await {
                Ok(path) => {
                    info!("Deployed to serviceGitea: {}", path);
                    deployment_paths.gitea = Some(path);
                }
                Err(e) => error!("Failed to deploy to serviceGitea: {}", e),
            }

            // Deploy to serviceAutomation
            match self.automation_deployer.deploy(
                feature_name,
                &langflow_workflow,
            ).await {
                Ok(path) => {
                    info!("Deployed to serviceAutomation: {}", path);
                    deployment_paths.automation = Some(path);
                }
                Err(e) => error!("Failed to deploy to serviceAutomation: {}", e),
            }

            // Store n8n workflow
            match self.store_n8n_workflow(feature_name, workflow_json).await {
                Ok(path) => {
                    info!("Stored n8n workflow: {}", path);
                    deployment_paths.n8n = Some(path);
                }
                Err(e) => error!("Failed to store n8n workflow: {}", e),
            }
        }

        Ok(GeneratedOutputs {
            markdown_spec,
            rust_code,
            lean4_proofs: generation_result.lean4_proofs,
            scip_spec: generation_result.scip_spec,
            langflow_workflow,
            deployment_paths,
        })
    }

    /// Convert n8n workflow JSON to markdown specification
    async fn n8n_to_markdown(&self, workflow_json: &str) -> Result<String> {
        let response = self.client
            .post(format!("{}/n8n-to-md", self.lean4_parser_url))
            .json(&json!({
                "workflow_json": workflow_json
            }))
            .send()
            .await
            .context("Failed to send request to Lean4 parser")?;

        if !response.status().is_success() {
            anyhow::bail!("Lean4 parser returned error: {}", response.status());
        }

        let result: Value = response.json().await
            .context("Failed to parse response from Lean4 parser")?;

        result.get("markdown")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .context("Response missing markdown field")
    }

    /// Generate code from markdown specification
    async fn generate_from_markdown(&self, markdown: &str) -> Result<GenerationResult> {
        let response = self.client
            .post(format!("{}/generate", self.lean4_parser_url))
            .json(&json!({
                "markdown": markdown
            }))
            .send()
            .await
            .context("Failed to send generate request")?;

        if !response.status().is_success() {
            anyhow::bail!("Generation failed: {}", response.status());
        }

        let result: Value = response.json().await
            .context("Failed to parse generation response")?;

        Ok(GenerationResult {
            go_code: result.get("go_code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            lean4_proofs: result.get("lean4_proofs")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            scip_spec: result.get("scip_spec")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string(),
        })
    }

    /// Deploy to serviceGitea
    pub async fn deploy_to_gitea(
        &mut self,
        feature_name: &str,
        rust_code: &str,
        lean4_proofs: &str,
        scip_spec: &str,
    ) -> Result<String> {
        self.gitea_deployer.deploy(feature_name, rust_code, lean4_proofs, scip_spec).await
    }

    /// Generate Langflow workflow from SCIP
    pub async fn generate_langflow_workflow(
        &mut self,
        scip_json: &str,
        feature_name: &str,
    ) -> Result<String> {
        self.langflow_generator.generate_from_scip(scip_json)
    }

    /// Store n8n workflow
    async fn store_n8n_workflow(&self, feature_name: &str, workflow_json: &str) -> Result<String> {
        use tokio::fs;

        let workflows_dir = "n8n-workflows";
        fs::create_dir_all(workflows_dir).await
            .context("Failed to create n8n-workflows directory")?;

        let file_path = format!("{}/{}-workflow.json", workflows_dir, feature_name);
        fs::write(&file_path, workflow_json).await
            .context("Failed to write n8n workflow file")?;

        Ok(file_path)
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

struct GenerationResult {
    go_code: String,
    lean4_proofs: String,
    scip_spec: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = Orchestrator::new();
        assert!(orchestrator.lean4_parser_url.contains("localhost"));
    }
}