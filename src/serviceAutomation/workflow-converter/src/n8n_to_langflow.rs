use crate::models::*;
use anyhow::{Result, Context};
use std::collections::HashMap;
use uuid::Uuid;

pub fn convert(n8n_json: &str) -> Result<String> {
    let n8n_workflow: N8nWorkflow = serde_json::from_str(n8n_json)
        .context("Failed to parse n8n workflow JSON")?;
    
    let langflow_flow = convert_workflow(&n8n_workflow)?;
    
    serde_json::to_string_pretty(&langflow_flow)
        .context("Failed to serialize Langflow flow")
}

fn convert_workflow(n8n_workflow: &N8nWorkflow) -> Result<LangflowFlow> {
    // Convert nodes
    let mut langflow_nodes = Vec::new();
    for n8n_node in &n8n_workflow.nodes {
        langflow_nodes.push(convert_node(n8n_node)?);
    }
    
    // Convert connections to edges
    let mut langflow_edges = Vec::new();
    for (source_node, connections) in &n8n_workflow.connections {
        for main_connections in &connections.main {
            for conn in main_connections {
                langflow_edges.push(convert_connection(source_node, conn));
            }
        }
    }
    
    Ok(LangflowFlow {
        name: n8n_workflow.name.clone(),
        description: format!("Converted from n8n workflow: {}", n8n_workflow.name),
        data: LangflowData {
            nodes: langflow_nodes,
            edges: langflow_edges,
        },
        is_component: false,
    })
}

fn convert_node(n8n_node: &N8nNode) -> Result<LangflowNode> {
    let component_type = NodeTypeMapper::n8n_to_langflow(&n8n_node.node_type);
    let node_id = n8n_node.id.clone();
    
    // Convert parameters to template fields
    let mut template = HashMap::new();
    for (key, value) in &n8n_node.parameters {
        template.insert(
            key.clone(),
            LangflowTemplateField {
                field_type: infer_field_type(value),
                required: false,
                placeholder: String::new(),
                list: value.is_array(),
                show: true,
                multiline: value.is_string() && value.as_str().map(|s| s.len() > 100).unwrap_or(false),
                value: value.clone(),
            },
        );
    }
    
    Ok(LangflowNode {
        id: node_id.clone(),
        data: LangflowNodeData {
            component_type: component_type.to_string(),
            node: LangflowNodeConfig {
                template,
                description: format!("Converted from n8n node: {}", n8n_node.name),
                base_classes: vec!["Component".to_string()],
                name: n8n_node.name.clone(),
                display_name: n8n_node.name.clone(),
                documentation: format!("Original n8n type: {}", n8n_node.node_type),
            },
            id: node_id,
        },
        position: LangflowPosition {
            x: n8n_node.position[0],
            y: n8n_node.position[1],
        },
        node_type: "genericNode".to_string(),
    })
}

fn convert_connection(source_node: &str, conn: &N8nConnection) -> LangflowEdge {
    LangflowEdge {
        source: source_node.to_string(),
        target: conn.node.clone(),
        source_handle: format!("{}-output-{}", source_node, conn.index),
        target_handle: format!("{}-input-{}", conn.node, conn.index),
        id: Uuid::new_v4().to_string(),
    }
}

fn infer_field_type(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(_) => "str".to_string(),
        serde_json::Value::Number(_) => "int".to_string(),
        serde_json::Value::Bool(_) => "bool".to_string(),
        serde_json::Value::Array(_) => "list".to_string(),
        serde_json::Value::Object(_) => "dict".to_string(),
        serde_json::Value::Null => "Any".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_conversion() {
        let n8n_json = r#"{
            "name": "Test Workflow",
            "nodes": [
                {
                    "id": "node1",
                    "name": "HTTP Request",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 1.0,
                    "position": [250, 300],
                    "parameters": {
                        "url": "https://api.example.com",
                        "method": "GET"
                    }
                }
            ],
            "connections": {}
        }"#;
        
        let result = convert(n8n_json);
        assert!(result.is_ok());
    }
}
