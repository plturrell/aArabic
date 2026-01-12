/// Data models for Lean4 parser
/// Defines structures for Markdown, Lean4, and SCIP elements

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

// =============================================================================
// Markdown Element Models
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MarkdownElementType {
    Heading,
    ListItem,
    CodeBlock,
    Paragraph,
    Table,
    Definition,
    Requirement,
    Constraint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownElement {
    pub element_type: MarkdownElementType,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<MarkdownElement>>,
}

// =============================================================================
// Lean4 Element Models
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Lean4ElementType {
    Theorem,
    Axiom,
    Def,
    Inductive,
    Structure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lean4Element {
    pub name: String,
    pub element_type: Lean4ElementType,
    pub statement: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

// =============================================================================
// SCIP Element Models
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SCIPElementType {
    Requirement,
    Constraint,
    Verification,
    ComplianceRule,
    Control,
    Policy,
    Procedure,
    Evidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SCIPSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SCIPStatus {
    Draft,
    Review,
    Approved,
    Implemented,
    Verified,
    Deprecated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCIPElement {
    pub id: String,
    pub element_type: SCIPElementType,
    pub title: String,
    pub description: String,
    pub severity: SCIPSeverity,
    pub status: SCIPStatus,
    
    // Formal specification (from Lean4)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formal_spec: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_status: Option<String>,
    
    // Compliance metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub references: Option<Vec<String>>,
    
    // Verification data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification_method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_cases: Option<Vec<HashMap<String, serde_json::Value>>>,
    
    // Lineage
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<Vec<String>>,
    
    // Timestamps
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// =============================================================================
// API Request/Response Models
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct ParseRequest {
    pub markdown: String,
    #[serde(default = "default_true")]
    pub generate_lean4: bool,
    #[serde(default = "default_true")]
    pub generate_scip: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct ParseResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lean4_code: Option<String>,
    pub lean4_elements_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scip_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scip_markdown: Option<String>,
    pub scip_elements_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistics: Option<ConversionStatistics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
pub struct ConversionStatistics {
    pub input: InputStats,
    pub lean4: Lean4Stats,
    pub scip: SCIPStats,
}

#[derive(Debug, Serialize)]
pub struct InputStats {
    pub markdown_length: usize,
}

#[derive(Debug, Serialize)]
pub struct Lean4Stats {
    pub elements_count: usize,
    pub types: HashMap<String, usize>,
}

#[derive(Debug, Serialize)]
pub struct SCIPStats {
    pub elements_count: usize,
    pub types: HashMap<String, usize>,
    pub severities: HashMap<String, usize>,
}

// =============================================================================
// Pipeline Results
// =============================================================================

#[derive(Debug)]
pub struct PipelineResults {
    pub markdown_input: String,
    pub lean4_elements: Vec<Lean4Element>,
    pub lean4_code: String,
    pub scip_elements: Vec<SCIPElement>,
    pub scip_json: String,
    pub scip_markdown: String,
}

impl PipelineResults {
    pub fn get_statistics(&self) -> ConversionStatistics {
        let mut lean4_types: HashMap<String, usize> = HashMap::new();
        for element in &self.lean4_elements {
            let type_str = format!("{:?}", element.element_type).to_lowercase();
            *lean4_types.entry(type_str).or_insert(0) += 1;
        }

        let mut scip_types: HashMap<String, usize> = HashMap::new();
        let mut scip_severities: HashMap<String, usize> = HashMap::new();
        for element in &self.scip_elements {
            let type_str = format!("{:?}", element.element_type).to_lowercase();
            let severity_str = format!("{:?}", element.severity).to_lowercase();
            *scip_types.entry(type_str).or_insert(0) += 1;
            *scip_severities.entry(severity_str).or_insert(0) += 1;
        }

        ConversionStatistics {
            input: InputStats {
                markdown_length: self.markdown_input.len(),
            },
            lean4: Lean4Stats {
                elements_count: self.lean4_elements.len(),
                types: lean4_types,
            },
            scip: SCIPStats {
                elements_count: self.scip_elements.len(),
                types: scip_types,
                severities: scip_severities,
            },
        }
    }
}