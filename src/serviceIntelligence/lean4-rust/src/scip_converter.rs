/// Lean4 to SCIP Converter (Rust implementation)
/// Converts Lean4 elements to SCIP compliance elements

use crate::models::{
    Lean4Element, Lean4ElementType, SCIPElement, SCIPElementType,
    SCIPSeverity, SCIPStatus,
};
use chrono::Utc;
use std::collections::HashMap;

pub struct SCIPConverter {
    counter: usize,
}

impl SCIPConverter {
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    /// Convert Lean4 elements to SCIP elements
    pub fn convert(&mut self, lean4_elements: &[Lean4Element]) -> Vec<SCIPElement> {
        lean4_elements
            .iter()
            .filter_map(|element| self.convert_element(element))
            .collect()
    }

    fn convert_element(&mut self, element: &Lean4Element) -> Option<SCIPElement> {
        self.counter += 1;
        
        let scip_type = self.determine_scip_type(element);
        let severity = self.determine_severity(element);
        let title = element.name.replace('_', " ").to_uppercase();
        let description = self.extract_description(element);
        let scip_id = format!("SCIP-{:04}", self.counter);
        let proof_status = if element.proof.is_some() {
            "proven"
        } else {
            "axiom"
        };
        let tags = self.extract_tags(element);
        let category = self.extract_category(element);
        let verification_method = self.determine_verification_method(element);
        
        let now = Utc::now();
        
        Some(SCIPElement {
            id: scip_id,
            element_type: scip_type,
            title,
            description,
            severity,
            status: SCIPStatus::Draft,
            formal_spec: Some(element.statement.clone()),
            proof_status: Some(proof_status.to_string()),
            category: Some(category),
            tags: Some(tags),
            references: element.dependencies.clone(),
            verification_method: Some(verification_method),
            test_cases: None,
            source: Some("lean4_parser_rust".to_string()),
            derived_from: Some(vec![element.name.clone()]),
            created_at: now,
            updated_at: now,
        })
    }

    fn determine_scip_type(&self, element: &Lean4Element) -> SCIPElementType {
        if let Some(metadata) = &element.metadata {
            if let Some(element_type) = metadata.get("type") {
                return match element_type.as_str() {
                    "requirement" => SCIPElementType::Requirement,
                    "constraint" => SCIPElementType::Constraint,
                    _ => SCIPElementType::Verification,
                };
            }
        }
        
        let name_lower = element.name.to_lowercase();
        if name_lower.contains("compliance") {
            SCIPElementType::ComplianceRule
        } else if name_lower.contains("control") {
            SCIPElementType::Control
        } else if name_lower.contains("policy") {
            SCIPElementType::Policy
        } else if name_lower.contains("procedure") {
            SCIPElementType::Procedure
        } else {
            SCIPElementType::Verification
        }
    }

    fn determine_severity(&self, element: &Lean4Element) -> SCIPSeverity {
        if let Some(metadata) = &element.metadata {
            if let Some(original) = metadata.get("original") {
                let original_lower = original.to_lowercase();
                
                if original_lower.contains("critical") || original_lower.contains("must not") {
                    return SCIPSeverity::Critical;
                } else if original_lower.contains("must") 
                    || original_lower.contains("shall") 
                    || original_lower.contains("required") {
                    return SCIPSeverity::High;
                } else if original_lower.contains("should") 
                    || original_lower.contains("recommended") {
                    return SCIPSeverity::Medium;
                } else if original_lower.contains("may") 
                    || original_lower.contains("optional") {
                    return SCIPSeverity::Low;
                }
            }
        }
        
        // Default based on Lean4 type
        match element.element_type {
            Lean4ElementType::Axiom => SCIPSeverity::High,
            Lean4ElementType::Theorem => SCIPSeverity::Medium,
            _ => SCIPSeverity::Low,
        }
    }

    fn extract_description(&self, element: &Lean4Element) -> String {
        if let Some(metadata) = &element.metadata {
            if let Some(description) = metadata.get("description") {
                return description.clone();
            } else if let Some(original) = metadata.get("original") {
                return original.clone();
            }
        }
        element.statement.clone()
    }

    fn extract_category(&self, element: &Lean4Element) -> String {
        if let Some(metadata) = &element.metadata {
            if let Some(category) = metadata.get("category") {
                return category.clone();
            }
            if let Some(element_type) = metadata.get("type") {
                return element_type.clone();
            }
        }
        
        let name_lower = element.name.to_lowercase();
        if name_lower.contains("payment") || name_lower.contains("financial") {
            "financial".to_string()
        } else if name_lower.contains("auth") || name_lower.contains("security") {
            "security".to_string()
        } else if name_lower.contains("data") {
            "data_management".to_string()
        } else if name_lower.contains("compliance") {
            "compliance".to_string()
        } else {
            "general".to_string()
        }
    }

    fn extract_tags(&self, element: &Lean4Element) -> Vec<String> {
        let mut tags = Vec::new();
        
        // Add type as tag
        tags.push(format!("{:?}", element.element_type).to_lowercase());
        
        // Add metadata type
        if let Some(metadata) = &element.metadata {
            if let Some(element_type) = metadata.get("type") {
                tags.push(element_type.clone());
            }
        }
        
        // Add category
        tags.push(self.extract_category(element));
        
        // Add based on name patterns
        let name_lower = element.name.to_lowercase();
        if name_lower.starts_with("req_") {
            tags.push("requirement".to_string());
        }
        if name_lower.starts_with("constraint_") {
            tags.push("constraint".to_string());
        }
        
        // Remove duplicates
        tags.sort();
        tags.dedup();
        tags
    }

    fn determine_verification_method(&self, element: &Lean4Element) -> String {
        if element.proof.is_some() {
            "formal_proof".to_string()
        } else {
            match element.element_type {
                Lean4ElementType::Axiom => "assertion".to_string(),
                Lean4ElementType::Theorem => "mathematical_proof".to_string(),
                Lean4ElementType::Structure => "type_checking".to_string(),
                _ => "manual_review".to_string(),
            }
        }
    }
}

/// Generate SCIP JSON output
pub fn generate_scip_json(elements: &[SCIPElement]) -> Result<String, serde_json::Error> {
    let data = serde_json::json!({
        "scip_version": "1.0",
        "generated_at": Utc::now(),
        "elements": elements,
    });
    serde_json::to_string_pretty(&data)
}

/// Generate SCIP Markdown documentation
pub fn generate_scip_markdown(elements: &[SCIPElement]) -> String {
    let mut lines = Vec::new();
    
    lines.push("# SCIP Compliance Elements".to_string());
    lines.push("".to_string());
    lines.push(format!("Generated: {}", Utc::now()));
    lines.push(format!("Total Elements: {}", elements.len()));
    lines.push("".to_string());
    lines.push("---".to_string());
    lines.push("".to_string());
    
    // Group by type
    let mut by_type: HashMap<String, Vec<&SCIPElement>> = HashMap::new();
    for element in elements {
        let type_key = format!("{:?}", element.element_type);
        by_type.entry(type_key).or_insert_with(Vec::new).push(element);
    }
    
    // Generate sections
    for (scip_type, type_elements) in by_type.iter() {
        lines.push(format!("## {} ({})", 
            scip_type.replace('_', " "), 
            type_elements.len()
        ));
        lines.push("".to_string());
        
        for element in type_elements {
            lines.push(format!("### {}: {}", element.id, element.title));
            lines.push("".to_string());
            lines.push(format!("**Severity:** {}", format!("{:?}", element.severity).to_uppercase()));
            lines.push(format!("**Status:** {:?}", element.status));
            lines.push(format!("**Category:** {}", element.category.as_ref().unwrap_or(&"N/A".to_string())));
            lines.push("".to_string());
            lines.push("**Description:**".to_string());
            lines.push(element.description.clone());
            lines.push("".to_string());
            
            if let Some(formal_spec) = &element.formal_spec {
                lines.push("**Formal Specification:**".to_string());
                lines.push("```lean".to_string());
                lines.push(formal_spec.clone());
                lines.push("```".to_string());
                lines.push("".to_string());
            }
            
            if let Some(tags) = &element.tags {
                lines.push(format!("**Tags:** {}", tags.join(", ")));
                lines.push("".to_string());
            }
            
            if let Some(verification_method) = &element.verification_method {
                lines.push(format!("**Verification Method:** {}", verification_method));
                lines.push("".to_string());
            }
            
            lines.push("---".to_string());
            lines.push("".to_string());
        }
    }
    
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_severity() {
        let converter = SCIPConverter::new();
        
        let mut metadata = HashMap::new();
        metadata.insert("original".to_string(), "MUST validate input".to_string());
        
        let element = Lean4Element {
            name: "test_requirement".to_string(),
            element_type: Lean4ElementType::Axiom,
            statement: "axiom test : True".to_string(),
            proof: None,
            dependencies: None,
            metadata: Some(metadata),
        };
        
        let severity = converter.determine_severity(&element);
        assert_eq!(severity, SCIPSeverity::High);
    }

    #[test]
    fn test_extract_category() {
        let converter = SCIPConverter::new();
        
        let element = Lean4Element {
            name: "payment_validation".to_string(),
            element_type: Lean4ElementType::Axiom,
            statement: "axiom test : True".to_string(),
            proof: None,
            dependencies: None,
            metadata: None,
        };
        
        let category = converter.extract_category(&element);
        assert_eq!(category, "financial");
    }
}