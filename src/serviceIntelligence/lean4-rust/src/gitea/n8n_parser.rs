/// n8n Workflow to Markdown Specification Parser
/// Converts n8n workflow JSON into structured markdown specifications
/// that can then be processed into Gitea features, Lean4 proofs, and SCIP specs

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct N8nWorkflow {
    pub name: String,
    pub nodes: Vec<N8nNode>,
    pub connections: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub settings: WorkflowSettings,
}

#[derive(Debug, Clone, Deserialize)]
pub struct N8nNode {
    pub name: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub parameters: Option<Value>,
    pub position: Option<[f64; 2]>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WorkflowSettings {
    #[serde(default)]
    pub timezone: String,
}

#[derive(Debug, Clone)]
pub struct WorkflowRequirement {
    pub id: String,
    pub text: String,
    pub requirement_type: String,
    pub source_node: String,
    pub category: String,
}

#[derive(Debug, Clone)]
pub struct WorkflowEndpoint {
    pub method: String,
    pub path: String,
    pub description: String,
    pub trigger_node: String,
}

#[derive(Debug, Clone)]
pub struct WorkflowDataFlow {
    pub from_node: String,
    pub to_node: String,
    pub data_type: String,
    pub description: String,
}

pub struct N8nToMarkdownConverter {
    counter: usize,
}

impl N8nToMarkdownConverter {
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    /// Parse n8n workflow JSON and convert to markdown specification
    pub fn convert_to_markdown(&mut self, workflow_json: &str) -> Result<String> {
        let workflow: N8nWorkflow = serde_json::from_str(workflow_json)
            .context("Failed to parse n8n workflow JSON")?;

        let mut md = Vec::new();

        // Generate header
        md.push(format!("# {}", workflow.name));
        md.push("".to_string());
        md.push("> Auto-generated from n8n workflow design".to_string());
        md.push("> This specification can be used to generate Gitea features, Lean4 proofs, and SCIP specs".to_string());
        md.push("".to_string());

        // Generate overview
        md.push("## Overview".to_string());
        md.push("".to_string());
        md.push(format!("**Workflow Name:** {}", workflow.name));
        md.push(format!("**Total Nodes:** {}", workflow.nodes.len()));
        md.push(format!("**Node Types:** {}", self.get_unique_node_types(&workflow)));
        md.push("".to_string());

        // Extract and generate requirements
        let requirements = self.extract_requirements(&workflow);
        if !requirements.is_empty() {
            md.push("## Requirements".to_string());
            md.push("".to_string());
            md.push("### Functional Requirements".to_string());
            md.push("".to_string());
            
            for req in &requirements {
                md.push(format!("- {} {} (from node: `{}`)", 
                    req.requirement_type, 
                    req.text,
                    req.source_node
                ));
            }
            md.push("".to_string());
        }

        // Extract API endpoints from HTTP/Webhook nodes
        let endpoints = self.extract_api_endpoints(&workflow);
        if !endpoints.is_empty() {
            md.push("## API Endpoints".to_string());
            md.push("".to_string());
            
            for endpoint in &endpoints {
                md.push(format!("- {} {} - {} (trigger: `{}`)",
                    endpoint.method,
                    endpoint.path,
                    endpoint.description,
                    endpoint.trigger_node
                ));
            }
            md.push("".to_string());
        }

        // Extract data model from Set/Function nodes
        let data_model = self.extract_data_model(&workflow);
        if !data_model.is_empty() {
            md.push("## Data Model".to_string());
            md.push("".to_string());
            md.push("| Field | Type | Required | Source Node |".to_string());
            md.push("|-------|------|----------|-------------|".to_string());
            
            for (field, info) in data_model {
                md.push(format!("| {} | {} | {} | {} |",
                    field,
                    info.get("type").unwrap_or(&"string".to_string()),
                    info.get("required").unwrap_or(&"no".to_string()),
                    info.get("source").unwrap_or(&"unknown".to_string())
                ));
            }
            md.push("".to_string());
        }

        // Extract validation rules
        let validations = self.extract_validations(&workflow);
        if !validations.is_empty() {
            md.push("## Validation Rules".to_string());
            md.push("".to_string());
            
            for validation in validations {
                md.push(format!("- {}", validation));
            }
            md.push("".to_string());
        }

        // Generate workflow diagram
        md.push("## Workflow Diagram".to_string());
        md.push("".to_string());
        md.push("```mermaid".to_string());
        md.push(self.generate_mermaid_diagram(&workflow));
        md.push("```".to_string());
        md.push("".to_string());

        // Node details
        md.push("## Node Details".to_string());
        md.push("".to_string());
        
        for node in &workflow.nodes {
            md.push(format!("### {} (`{}`)", node.name, node.node_type));
            md.push("".to_string());
            
            if !node.notes.is_empty() {
                md.push(format!("**Notes:** {}", node.notes));
                md.push("".to_string());
            }
            
            if let Some(params) = &node.parameters {
                if !params.is_null() {
                    md.push("**Parameters:**".to_string());
                    md.push("```json".to_string());
                    md.push(serde_json::to_string_pretty(params).unwrap_or_default());
                    md.push("```".to_string());
                    md.push("".to_string());
                }
            }
        }

        // Implementation notes
        md.push("## Implementation Notes".to_string());
        md.push("".to_string());
        md.push("### From n8n Workflow".to_string());
        md.push("".to_string());
        md.push("This specification was auto-generated from an n8n workflow design.".to_string());
        md.push("".to_string());
        md.push("**Next Steps:**".to_string());
        md.push("1. Review and refine the requirements above".to_string());
        md.push("2. Add any missing validation rules or constraints".to_string());
        md.push("3. Use `/generate` endpoint to create Gitea feature scaffolding".to_string());
        md.push("4. Implement business logic in generated Go code".to_string());
        md.push("5. Verify with generated Lean4 proofs".to_string());
        md.push("".to_string());

        // Metadata
        md.push("---".to_string());
        md.push("".to_string());
        md.push(format!("**Generated:** {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        md.push(format!("**Source:** n8n workflow `{}`", workflow.name));
        md.push("**Status:** Draft - Needs Review".to_string());

        Ok(md.join("\n"))
    }

    fn get_unique_node_types(&self, workflow: &N8nWorkflow) -> String {
        let mut types: Vec<String> = workflow.nodes
            .iter()
            .map(|n| n.node_type.clone())
            .collect();
        types.sort();
        types.dedup();
        types.join(", ")
    }

    fn extract_requirements(&mut self, workflow: &N8nWorkflow) -> Vec<WorkflowRequirement> {
        let mut requirements = Vec::new();

        for node in &workflow.nodes {
            // Look for validation nodes
            if node.node_type.to_lowercase().contains("if") 
                || node.node_type.to_lowercase().contains("switch")
                || node.node_type.to_lowercase().contains("filter") {
                
                self.counter += 1;
                requirements.push(WorkflowRequirement {
                    id: format!("REQ-{:03}", self.counter),
                    text: format!("validate conditions in {}", node.name),
                    requirement_type: "MUST".to_string(),
                    source_node: node.name.clone(),
                    category: "validation".to_string(),
                });
            }

            // Look for HTTP request nodes
            if node.node_type == "n8n-nodes-base.httpRequest" {
                self.counter += 1;
                requirements.push(WorkflowRequirement {
                    id: format!("REQ-{:03}", self.counter),
                    text: format!("handle HTTP request from {}", node.name),
                    requirement_type: "MUST".to_string(),
                    source_node: node.name.clone(),
                    category: "api".to_string(),
                });
            }

            // Look for database nodes
            if node.node_type.to_lowercase().contains("postgres") 
                || node.node_type.to_lowercase().contains("mysql")
                || node.node_type.to_lowercase().contains("mongo") {
                
                self.counter += 1;
                requirements.push(WorkflowRequirement {
                    id: format!("REQ-{:03}", self.counter),
                    text: format!("persist data from {}", node.name),
                    requirement_type: "MUST".to_string(),
                    source_node: node.name.clone(),
                    category: "data".to_string(),
                });
            }

            // Extract from notes
            if !node.notes.is_empty() {
                if node.notes.to_lowercase().contains("must") 
                    || node.notes.to_lowercase().contains("required") {
                    self.counter += 1;
                    requirements.push(WorkflowRequirement {
                        id: format!("REQ-{:03}", self.counter),
                        text: node.notes.clone(),
                        requirement_type: "MUST".to_string(),
                        source_node: node.name.clone(),
                        category: "business_rule".to_string(),
                    });
                }
            }
        }

        requirements
    }

    fn extract_api_endpoints(&self, workflow: &N8nWorkflow) -> Vec<WorkflowEndpoint> {
        let mut endpoints = Vec::new();

        for node in &workflow.nodes {
            // Webhook trigger
            if node.node_type == "n8n-nodes-base.webhook" {
                if let Some(params) = &node.parameters {
                    let method = params.get("httpMethod")
                        .and_then(|v| v.as_str())
                        .unwrap_or("POST")
                        .to_uppercase();
                    
                    let path = params.get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("/webhook")
                        .to_string();

                    endpoints.push(WorkflowEndpoint {
                        method,
                        path,
                        description: format!("Webhook trigger for {}", node.name),
                        trigger_node: node.name.clone(),
                    });
                }
            }

            // HTTP Request node (as endpoint design)
            if node.node_type == "n8n-nodes-base.httpRequest" {
                if let Some(params) = &node.parameters {
                    let method = params.get("method")
                        .and_then(|v| v.as_str())
                        .unwrap_or("GET")
                        .to_uppercase();
                    
                    let url = params.get("url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("/api/endpoint");

                    // Extract path from URL
                    let path = if let Some(path_start) = url.find("/api/") {
                        url[path_start..].split('?').next().unwrap_or("/api/endpoint")
                    } else {
                        "/api/endpoint"
                    };

                    endpoints.push(WorkflowEndpoint {
                        method,
                        path: path.to_string(),
                        description: format!("API call from {}", node.name),
                        trigger_node: node.name.clone(),
                    });
                }
            }
        }

        endpoints
    }

    fn extract_data_model(&self, workflow: &N8nWorkflow) -> HashMap<String, HashMap<String, String>> {
        let mut model = HashMap::new();

        for node in &workflow.nodes {
            // Set node - defines data structure
            if node.node_type == "n8n-nodes-base.set" {
                if let Some(params) = &node.parameters {
                    if let Some(values) = params.get("values") {
                        if let Some(values_obj) = values.as_object() {
                            for (key, value) in values_obj {
                                let mut field_info = HashMap::new();
                                field_info.insert("type".to_string(), self.infer_type(value));
                                field_info.insert("required".to_string(), "yes".to_string());
                                field_info.insert("source".to_string(), node.name.clone());
                                model.insert(key.clone(), field_info);
                            }
                        }
                    }
                }
            }

            // Function node - data transformation hints
            if node.node_type == "n8n-nodes-base.function" {
                if let Some(params) = &node.parameters {
                    if let Some(code) = params.get("functionCode").and_then(|v| v.as_str()) {
                        // Extract variable assignments as potential fields
                        let var_regex = regex::Regex::new(r"(?m)^\s*(?:const|let|var)\s+(\w+)\s*=").unwrap();
                        for cap in var_regex.captures_iter(code) {
                            if let Some(var_name) = cap.get(1) {
                                if !model.contains_key(var_name.as_str()) {
                                    let mut field_info = HashMap::new();
                                    field_info.insert("type".to_string(), "string".to_string());
                                    field_info.insert("required".to_string(), "no".to_string());
                                    field_info.insert("source".to_string(), node.name.clone());
                                    model.insert(var_name.as_str().to_string(), field_info);
                                }
                            }
                        }
                    }
                }
            }
        }

        model
    }

    fn extract_validations(&self, workflow: &N8nWorkflow) -> Vec<String> {
        let mut validations = Vec::new();

        for node in &workflow.nodes {
            // IF node - conditional validation
            if node.node_type == "n8n-nodes-base.if" {
                if let Some(params) = &node.parameters {
                    if let Some(_conditions) = params.get("conditions") {
                        let validation = format!("{}: MUST satisfy condition from IF node", node.name);
                        validations.push(validation);
                    }
                }
            }

            // Switch node - multiple validations
            if node.node_type == "n8n-nodes-base.switch" {
                validations.push(format!("{}: MUST match one of the defined cases", node.name));
            }

            // Validation from notes
            if node.notes.to_lowercase().contains("validate") 
                || node.notes.to_lowercase().contains("check") {
                validations.push(format!("{}: {}", node.name, node.notes));
            }
        }

        validations
    }

    fn generate_mermaid_diagram(&self, workflow: &N8nWorkflow) -> String {
        let mut diagram = vec!["graph TD".to_string()];

        // Add nodes
        for (idx, node) in workflow.nodes.iter().enumerate() {
            let node_id = format!("N{}", idx);
            let node_label = format!("{}[{}]", node_id, node.name);
            diagram.push(format!("    {}", node_label));
        }

        // Add connections (if available)
        if let Some(connections) = &workflow.connections {
            for (source, _targets) in connections {
                // Simplified connection representation
                diagram.push(format!("    {} --> ...", source));
            }
        } else {
            // Create simple sequential flow
            for i in 0..workflow.nodes.len().saturating_sub(1) {
                diagram.push(format!("    N{} --> N{}", i, i + 1));
            }
        }

        diagram.join("\n")
    }

    fn infer_type(&self, value: &Value) -> String {
        match value {
            Value::String(_) => "string",
            Value::Number(_) => "number",
            Value::Bool(_) => "boolean",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
            Value::Null => "null",
        }
        .to_string()
    }
}

impl Default for N8nToMarkdownConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to convert n8n JSON file to markdown
pub fn convert_n8n_file_to_markdown(json_path: &str, output_path: &str) -> Result<()> {
    let json_content = std::fs::read_to_string(json_path)
        .with_context(|| format!("Failed to read n8n file: {}", json_path))?;

    let mut converter = N8nToMarkdownConverter::new();
    let markdown = converter.convert_to_markdown(&json_content)?;

    std::fs::write(output_path, markdown)
        .with_context(|| format!("Failed to write markdown: {}", output_path))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_workflow() {
        let json = r#"{
            "name": "Test Workflow",
            "nodes": [
                {
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "/api/test"
                    },
                    "notes": "MUST validate input"
                },
                {
                    "name": "Validate",
                    "type": "n8n-nodes-base.if",
                    "parameters": {},
                    "notes": ""
                }
            ]
        }"#;

        let mut converter = N8nToMarkdownConverter::new();
        let result = converter.convert_to_markdown(json);

        assert!(result.is_ok());
        let md = result.unwrap();
        assert!(md.contains("# Test Workflow"));
        assert!(md.contains("## Requirements"));
        assert!(md.contains("MUST"));
    }

    #[test]
    fn test_infer_type() {
        let converter = N8nToMarkdownConverter::new();
        
        assert_eq!(converter.infer_type(&Value::String("test".into())), "string");
        assert_eq!(converter.infer_type(&Value::Number(42.into())), "number");
        assert_eq!(converter.infer_type(&Value::Bool(true)), "boolean");
    }
}