/// Langflow Workflow Generator
/// Converts SCIP specifications to Langflow workflow JSON

use anyhow::Result;
use serde_json::{json, Value};

pub struct LangflowGenerator;

impl LangflowGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Langflow workflow from SCIP specification
    pub fn generate_from_scip(&self, scip_json: &str) -> Result<String> {
        let scip: Value = serde_json::from_str(scip_json)?;
        
        let elements = scip.get("elements")
            .and_then(|e| e.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid SCIP: missing elements"))?;

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_id = 0;

        // Create start node
        nodes.push(json!({
            "id": format!("node_{}", node_id),
            "type": "Start",
            "position": {"x": 100, "y": 100},
            "data": {
                "label": "Start",
                "description": "Workflow entry point"
            }
        }));
        let prev_id = node_id;
        node_id += 1;

        // Create nodes for each SCIP element
        for element in elements {
            let elem_type = element.get("element_type")
                .and_then(|t| t.as_str())
                .unwrap_or("requirement");
            
            let title = element.get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("Untitled");

            let node_type = match elem_type {
                "requirement" => "LLMChain",
                "verification" => "Validator",
                "control" => "Condition",
                _ => "Custom",
            };

            nodes.push(json!({
                "id": format!("node_{}", node_id),
                "type": node_type,
                "position": {"x": 100 + (node_id * 200), "y": 100},
                "data": {
                    "label": title,
                    "description": element.get("description").and_then(|d| d.as_str()).unwrap_or(""),
                    "severity": element.get("severity").and_then(|s| s.as_str()).unwrap_or("medium")
                }
            }));

            // Create edge from previous node
            edges.push(json!({
                "id": format!("edge_{}_{}", prev_id, node_id),
                "source": format!("node_{}", prev_id),
                "target": format!("node_{}", node_id),
                "type": "default"
            }));

            node_id += 1;
        }

        // Create end node
        nodes.push(json!({
            "id": format!("node_{}", node_id),
            "type": "End",
            "position": {"x": 100 + (node_id * 200), "y": 100},
            "data": {
                "label": "End",
                "description": "Workflow completion"
            }
        }));

        edges.push(json!({
            "id": format!("edge_{}_{}", node_id - 1, node_id),
            "source": format!("node_{}", node_id - 1),
            "target": format!("node_{}", node_id),
            "type": "default"
        }));

        let workflow = json!({
            "name": scip.get("feature").and_then(|f| f.as_str()).unwrap_or("Generated Workflow"),
            "description": "Auto-generated from SCIP specification",
            "version": "1.0",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "generated_from": "SCIP",
                "scip_version": scip.get("scip_version").and_then(|v| v.as_str()).unwrap_or("1.0")
            }
        });

        Ok(serde_json::to_string_pretty(&workflow)?)
    }
}

impl Default for LangflowGenerator {
    fn default() -> Self {
        Self::new()
    }
}