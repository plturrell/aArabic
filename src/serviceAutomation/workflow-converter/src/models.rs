use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// n8n Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct N8nWorkflow {
    pub name: String,
    pub nodes: Vec<N8nNode>,
    pub connections: HashMap<String, N8nConnections>,
    #[serde(default)]
    pub settings: N8nSettings,
    #[serde(default)]
    pub staticData: Option<serde_json::Value>,
    #[serde(default)]
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct N8nNode {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(rename = "typeVersion")]
    pub type_version: f64,
    pub position: [f64; 2],
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub credentials: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct N8nConnections {
    pub main: Vec<Vec<N8nConnection>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct N8nConnection {
    pub node: String,
    #[serde(rename = "type")]
    pub connection_type: String,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct N8nSettings {
    #[serde(default)]
    pub executionOrder: String,
}

// ============================================================================
// Langflow Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangflowFlow {
    pub name: String,
    pub description: String,
    pub data: LangflowData,
    #[serde(default)]
    pub is_component: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangflowData {
    pub nodes: Vec<LangflowNode>,
    pub edges: Vec<LangflowEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangflowNode {
    pub id: String,
    pub data: LangflowNodeData,
    pub position: LangflowPosition,
    #[serde(rename = "type")]
    pub node_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangflowNodeData {
    #[serde(rename = "type")]
    pub component_type: String,
    pub node: LangflowNodeConfig,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangflowNodeConfig {
    pub template: HashMap<String, LangflowTemplateField>,
    pub description: String,
    #[serde(rename = "baseClasses")]
    pub base_classes: Vec<String>,
    pub name: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
    #[serde(default)]
    pub documentation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangflowTemplateField {
    #[serde(rename = "type")]
    pub field_type: String,
    pub required: bool,
    #[serde(default)]
    pub placeholder: String,
    #[serde(default)]
    pub list: bool,
    #[serde(default)]
    pub show: bool,
    #[serde(default)]
    pub multiline: bool,
    #[serde(default)]
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangflowPosition {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangflowEdge {
    pub source: String,
    pub target: String,
    #[serde(rename = "sourceHandle")]
    pub source_handle: String,
    #[serde(rename = "targetHandle")]
    pub target_handle: String,
    pub id: String,
}

// ============================================================================
// Node Type Mappings
// ============================================================================

pub struct NodeTypeMapper;

impl NodeTypeMapper {
    /// Map n8n node type to Langflow component type
    pub fn n8n_to_langflow(n8n_type: &str) -> &str {
        match n8n_type {
            // HTTP/API nodes
            "n8n-nodes-base.httpRequest" => "HTTPRequest",
            "n8n-nodes-base.webhook" => "Webhook",
            
            // Data transformation
            "n8n-nodes-base.set" => "DataProcessor",
            "n8n-nodes-base.function" => "PythonFunction",
            "n8n-nodes-base.code" => "CodeComponent",
            
            // LLM nodes
            "n8n-nodes-base.openAi" => "OpenAI",
            "n8n-nodes-base.anthropic" => "AnthropicLLM",
            
            // Database
            "n8n-nodes-base.postgres" => "PostgresComponent",
            "n8n-nodes-base.mysql" => "SQLDatabase",
            
            // Utilities
            "n8n-nodes-base.if" => "ConditionalRouter",
            "n8n-nodes-base.switch" => "Router",
            "n8n-nodes-base.merge" => "Combiner",
            
            // Default fallback
            _ => "CustomComponent",
        }
    }
    
    /// Map Langflow component type to n8n node type
    pub fn langflow_to_n8n(langflow_type: &str) -> &str {
        match langflow_type {
            // HTTP/API
            "HTTPRequest" => "n8n-nodes-base.httpRequest",
            "Webhook" => "n8n-nodes-base.webhook",
            
            // Data transformation
            "DataProcessor" => "n8n-nodes-base.set",
            "PythonFunction" => "n8n-nodes-base.function",
            "CodeComponent" => "n8n-nodes-base.code",
            
            // LLM
            "OpenAI" | "ChatOpenAI" => "n8n-nodes-base.openAi",
            "AnthropicLLM" | "ChatAnthropic" => "n8n-nodes-base.anthropic",
            
            // Database
            "PostgresComponent" => "n8n-nodes-base.postgres",
            "SQLDatabase" => "n8n-nodes-base.mysql",
            
            // Utilities
            "ConditionalRouter" => "n8n-nodes-base.if",
            "Router" => "n8n-nodes-base.switch",
            "Combiner" => "n8n-nodes-base.merge",
            
            // Default fallback
            _ => "n8n-nodes-base.function",
        }
    }
}
