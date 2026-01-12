use crate::models::*;
use anyhow::{Result, Context};
use std::collections::HashMap;

pub fn convert(langflow_json: &str) -> Result<String> {
    let langflow_flow: LangflowFlow = serde_json::from_str(langflow_json)
        .context("Failed to parse Langflow flow JSON")?;
    
    let n8n_workflow = convert_flow(&langflow_flow)?;
    
    serde_json::to_string_pretty(&n8n_workflow)
        .context("Failed to serialize n8n workflow")
}

fn convert_flow(langflow_flow: &LangflowFlow) -> Result<N8nWorkflow> {
    // Convert nodes
    let mut n8n_nodes = Vec::new();
    for langflow_node in &langflow_flow.data.nodes {
        n8n_nodes.push(convert_node(langflow_node)?);
    }
    
    // Convert edges to connections
    let connections = convert_edges(&langflow_flow.data.edges);
    
    Ok(N8nWorkflow {
        name: langflow_flow.name.clone(),
        nodes: n8n_nodes,
        connections,
        settings: N8nSettings {
            executionOrder: "v1".to_string(),
        },
        staticData: None,
        active: false,
    })
}

fn convert_node(langflow_node: &LangflowNode) -> Result<N8nNode> {
    let n8n_type = NodeTypeMapper::langflow_to_n8n(&langflow_node.data.component_type);
    
    // Convert template fields to parameters
    let mut parameters = HashMap::new();
    for (key, field) in &langflow_node.data.node.template {
        if !field.value.is_null() {
            parameters.insert(key.clone(), field.value.clone());
        }
    }
    
    Ok(N8nNode {
        id: langflow_node.id.clone(),
        name: langflow_node.data.node.name.clone(),
        node_type: n8n_type.to_string(),
        type_version: 1.0,
        position: [
            langflow_node.position.x,
            langflow_node.position.y,
        ],
        parameters,
        credentials: None,
    })
}

fn convert_edges(edges: &[LangflowEdge]) -> HashMap<String, N8nConnections> {
    let mut connections: HashMap<String, N8nConnections> = HashMap::new();
    
    for edge in edges {
        let entry = connections.entry(edge.source.clone())
            .or_insert_with(|| N8nConnections {
                main: vec![vec![]],
            });
        
        // Ensure we have enough connection arrays
        while entry.main.len() <= 0 {
            entry.main.push(vec![]);
        }
        
        entry.main[0].push(N8nConnection {
            node: edge.target.clone(),
            connection_type: "main".to_string(),
            index: 0,
        });
    }
    
    connections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_conversion() {
        let langflow_json = r#"{
            "name": "Test Flow",
            "description": "Test",
            "data": {
                "nodes": [
                    {
                        "id": "node1",
                        "data": {
                            "type": "HTTPRequest",
                            "node": {
                                "template": {},
                                "description": "Test",
                                "baseClasses": ["Component"],
                                "name": "HTTP Request",
                                "displayName": "HTTP Request",
                                "documentation": ""
                            },
                            "id": "node1"
                        },
                        "position": {
                            "x": 250,
                            "y": 300
                        },
                        "type": "genericNode"
                    }
                ],
                "edges": []
            },
            "is_component": false
        }"#;
        
        let result = convert(langflow_json);
        assert!(result.is_ok());
    }
}
