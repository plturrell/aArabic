/// Pipeline orchestrator for Markdown → Lean4 → SCIP conversion
/// High-performance pipeline with error handling

use crate::config::is_path_allowed;
use crate::models::PipelineResults;
use crate::md_parser::{generate_lean4_code, MarkdownParser};
use crate::scip_converter::{generate_scip_json, generate_scip_markdown, SCIPConverter};
use anyhow::{bail, Context, Result};

pub struct ConversionPipeline {
    md_parser: MarkdownParser,
    scip_converter: SCIPConverter,
}

impl ConversionPipeline {
    pub fn new() -> Self {
        Self {
            md_parser: MarkdownParser::new(),
            scip_converter: SCIPConverter::new(),
        }
    }

    /// Process markdown through complete pipeline
    pub fn process_markdown(&mut self, markdown: &str) -> Result<PipelineResults> {
        // Step 1: Parse Markdown → Lean4 Elements
        let lean4_elements = self
            .md_parser
            .parse_to_lean4(markdown);

        // Step 2: Generate Lean4 Code
        let lean4_code = generate_lean4_code(&lean4_elements);

        // Step 3: Convert Lean4 → SCIP Elements
        let scip_elements = self.scip_converter.convert(&lean4_elements);

        // Step 4: Generate SCIP outputs
        let scip_json = generate_scip_json(&scip_elements)
            .context("Failed to generate SCIP JSON")?;

        let scip_markdown = generate_scip_markdown(&scip_elements);

        Ok(PipelineResults {
            markdown_input: markdown.to_string(),
            lean4_elements,
            lean4_code,
            scip_elements,
            scip_json,
            scip_markdown,
        })
    }

    /// Process markdown from file
    /// Security: Validates the path is within allowed directories
    pub fn process_file(&mut self, path: &str) -> Result<PipelineResults> {
        // Security check: Validate path is within allowed directories
        if !is_path_allowed(path) {
            bail!(
                "Access denied: Path '{}' is outside allowed directories",
                path
            );
        }

        let markdown = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path))?;

        self.process_markdown(&markdown)
    }

    /// Save results to files
    /// Security: Validates output directory and sanitizes filenames
    pub fn save_results(
        &self,
        results: &PipelineResults,
        output_dir: &str,
        base_name: &str,
    ) -> Result<()> {
        use std::fs;
        use std::path::Path;

        // Security check: Validate output directory is allowed
        if !is_path_allowed(output_dir) {
            bail!(
                "Access denied: Output directory '{}' is outside allowed directories",
                output_dir
            );
        }

        // Sanitize base_name to prevent path traversal
        let safe_base_name: String = base_name
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect();

        if safe_base_name.is_empty() {
            bail!("Invalid base name: must contain alphanumeric characters");
        }

        let output_path = Path::new(output_dir);
        fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

        // Save Lean4 code
        let lean4_file = output_path.join(format!("{}.lean", safe_base_name));
        fs::write(&lean4_file, &results.lean4_code)
            .with_context(|| format!("Failed to write Lean4 file: {:?}", lean4_file))?;

        // Save SCIP JSON
        let scip_json_file = output_path.join(format!("{}_scip.json", safe_base_name));
        fs::write(&scip_json_file, &results.scip_json)
            .with_context(|| format!("Failed to write SCIP JSON file: {:?}", scip_json_file))?;

        // Save SCIP Markdown
        let scip_md_file = output_path.join(format!("{}_scip.md", safe_base_name));
        fs::write(&scip_md_file, &results.scip_markdown)
            .with_context(|| format!("Failed to write SCIP markdown file: {:?}", scip_md_file))?;

        // Save summary
        let summary = serde_json::json!({
            "input_length": results.markdown_input.len(),
            "lean4_elements_count": results.lean4_elements.len(),
            "scip_elements_count": results.scip_elements.len(),
            "output_files": [
                format!("{}.lean", safe_base_name),
                format!("{}_scip.json", safe_base_name),
                format!("{}_scip.md", safe_base_name),
            ]
        });

        let summary_file = output_path.join(format!("{}_summary.json", safe_base_name));
        fs::write(&summary_file, serde_json::to_string_pretty(&summary)?)
            .with_context(|| format!("Failed to write summary file: {:?}", summary_file))?;

        Ok(())
    }
}

impl Default for ConversionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline() {
        let markdown = r#"
# Payment Processing

## Requirements
- MUST validate payment amount
- SHALL verify user authentication

## Constraints
- MUST NOT process amounts over $10,000
"#;

        let mut pipeline = ConversionPipeline::new();
        let results = pipeline.process_markdown(markdown).unwrap();

        assert!(!results.lean4_elements.is_empty());
        assert!(!results.lean4_code.is_empty());
        assert!(!results.scip_elements.is_empty());
        assert!(!results.scip_json.is_empty());
        assert!(!results.scip_markdown.is_empty());
    }

    #[test]
    fn test_statistics() {
        let markdown = r#"
# Test
- MUST validate
- MUST NOT exceed
"#;

        let mut pipeline = ConversionPipeline::new();
        let results = pipeline.process_markdown(markdown).unwrap();
        let stats = results.get_statistics();

        assert_eq!(stats.input.markdown_length, markdown.len());
        assert!(stats.lean4.elements_count > 0);
        assert!(stats.scip.elements_count > 0);
    }
}