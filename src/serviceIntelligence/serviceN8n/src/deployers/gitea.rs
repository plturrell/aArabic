/// Gitea Service Deployer
/// Deploys generated Rust code to serviceGitea

use anyhow::{Context, Result};
use tokio::fs;

pub struct GiteaDeployer {
    base_path: String,
}

impl GiteaDeployer {
    pub fn new() -> Self {
        let base_path = std::env::var("GITEA_SERVICE_PATH")
            .unwrap_or_else(|_| "../../serviceCore/serviceGitea".to_string());
        
        Self { base_path }
    }

    /// Deploy feature to serviceGitea
    pub async fn deploy(
        &self,
        feature_name: &str,
        rust_code: &str,
        lean4_proofs: &str,
        scip_spec: &str,
    ) -> Result<String> {
        let feature_dir = format!("{}/features/{}", self.base_path, feature_name);
        
        // Create directory structure
        fs::create_dir_all(&feature_dir).await
            .context("Failed to create feature directory")?;
        
        fs::create_dir_all(format!("{}/src", feature_dir)).await?;
        fs::create_dir_all(format!("{}/proofs", feature_dir)).await?;
        fs::create_dir_all(format!("{}/specs", feature_dir)).await?;
        fs::create_dir_all(format!("{}/tests", feature_dir)).await?;

        // Write Rust code
        fs::write(format!("{}/src/lib.rs", feature_dir), rust_code).await
            .context("Failed to write Rust code")?;

        // Write Lean4 proofs
        fs::write(format!("{}/proofs/feature.lean", feature_dir), lean4_proofs).await
            .context("Failed to write Lean4 proofs")?;

        // Write SCIP spec
        fs::write(format!("{}/specs/scip.json", feature_dir), scip_spec).await
            .context("Failed to write SCIP spec")?;

        // Create Cargo.toml for feature
        let cargo_toml = format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[dependencies]
actix-web = "4.4"
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
tokio = {{ version = "1.35", features = ["full"] }}
anyhow = "1.0"
"#,
            feature_name
        );
        fs::write(format!("{}/Cargo.toml", feature_dir), cargo_toml).await?;

        // Create README
        let readme = format!(
            "# {}\n\nAuto-generated Gitea feature\n\n## Files\n- `src/lib.rs` - Rust implementation\n- `proofs/feature.lean` - Lean4 formal proofs\n- `specs/scip.json` - SCIP compliance spec\n",
            feature_name
        );
        fs::write(format!("{}/README.md", feature_dir), readme).await?;

        Ok(feature_dir)
    }
}

impl Default for GiteaDeployer {
    fn default() -> Self {
        Self::new()
    }
}