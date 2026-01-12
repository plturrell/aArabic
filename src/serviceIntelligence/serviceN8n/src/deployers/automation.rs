/// Automation Service Deployer
/// Deploys Langflow workflows to serviceAutomation

use anyhow::{Context, Result};
use tokio::fs;

pub struct AutomationDeployer {
    base_path: String,
}

impl AutomationDeployer {
    pub fn new() -> Self {
        let base_path = std::env::var("AUTOMATION_PATH")
            .unwrap_or_else(|_| "../../serviceAutomation".to_string());
        
        Self { base_path }
    }

    /// Deploy Langflow workflow to serviceAutomation
    pub async fn deploy(
        &self,
        feature_name: &str,
        langflow_workflow: &str,
    ) -> Result<String> {
        let workflow_dir = format!("{}/workflows/{}", self.base_path, feature_name);
        
        // Create directory structure
        fs::create_dir_all(&workflow_dir).await
            .context("Failed to create workflow directory")?;

        // Write Langflow workflow JSON
        fs::write(format!("{}/workflow.json", workflow_dir), langflow_workflow).await
            .context("Failed to write Langflow workflow")?;

        // Create README
        let readme = format!(
            "# {} - Langflow Workflow\n\nAuto-generated from SCIP specification\n\n## Usage\n\n1. Import `workflow.json` into Langflow\n2. Configure API keys if needed\n3. Run the workflow\n",
            feature_name
        );
        fs::write(format!("{}/README.md", workflow_dir), readme).await?;

        Ok(workflow_dir)
    }
}

impl Default for AutomationDeployer {
    fn default() -> Self {
        Self::new()
    }
}