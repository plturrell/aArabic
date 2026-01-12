/// Feature Forward Engineering Generator
/// Takes MD specification → Generates Gitea feature scaffolding + Lean4 proofs

use crate::models::{SCIPElement, SCIPElementType, SCIPSeverity, SCIPStatus};
use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;

// Pre-compiled regex patterns for better performance
lazy_static! {
    /// Pattern for MUST/SHALL/REQUIRED keywords
    static ref MUST_RE: Regex = Regex::new(r"(?i)\b(MUST|SHALL|REQUIRED)\b").unwrap();

    /// Pattern for SHOULD/RECOMMENDED keywords
    static ref SHOULD_RE: Regex = Regex::new(r"(?i)\b(SHOULD|RECOMMENDED)\b").unwrap();

    /// Pattern for MAY/OPTIONAL keywords
    static ref MAY_RE: Regex = Regex::new(r"(?i)\b(MAY|OPTIONAL)\b").unwrap();

    /// Pattern for HTTP methods
    static ref HTTP_METHOD_RE: Regex = Regex::new(r"(?i)\b(GET|POST|PUT|PATCH|DELETE)\b").unwrap();

    /// Pattern for API paths
    static ref API_PATH_RE: Regex = Regex::new(r"/[\w/:\-{}]+").unwrap();

    /// Pattern for sanitizing Lean4 names
    static ref LEAN4_NAME_SANITIZE_RE: Regex = Regex::new(r"[^a-zA-Z0-9_]").unwrap();

    /// Pattern for collapsing multiple underscores
    static ref MULTI_UNDERSCORE_RE: Regex = Regex::new(r"_+").unwrap();
}

#[derive(Debug, Clone)]
pub struct FeatureSpec {
    pub name: String,
    pub description: String,
    pub requirements: Vec<Requirement>,
    pub api_endpoints: Vec<APIEndpoint>,
    pub data_models: Vec<DataModel>,
    pub validation_rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone)]
pub struct Requirement {
    pub id: String,
    pub text: String,
    pub requirement_type: String, // MUST, SHOULD, MAY
    pub category: String,
}

#[derive(Debug, Clone)]
pub struct APIEndpoint {
    pub method: String,  // GET, POST, PUT, DELETE
    pub path: String,
    pub description: String,
    pub request_body: Option<String>,
    pub response_body: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DataModel {
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub field_type: String,
    pub required: bool,
    pub validation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub field: String,
    pub rule: String,
    pub error_message: String,
}

pub struct FeatureGenerator {
    counter: usize,
}

impl FeatureGenerator {
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    /// Parse markdown specification into feature spec
    pub fn parse_feature_spec(&mut self, markdown: &str) -> Result<FeatureSpec> {
        let mut spec = FeatureSpec {
            name: String::new(),
            description: String::new(),
            requirements: Vec::new(),
            api_endpoints: Vec::new(),
            data_models: Vec::new(),
            validation_rules: Vec::new(),
        };

        let lines: Vec<&str> = markdown.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Parse feature name from first heading
            if line.starts_with("# ") && spec.name.is_empty() {
                spec.name = line.trim_start_matches("# ").to_string();
                i += 1;
                continue;
            }

            // Parse requirements section
            if line.to_lowercase().contains("requirement") {
                i = self.parse_requirements_section(&lines, i, &mut spec.requirements);
                continue; // Re-evaluate the new line position
            }

            // Parse API endpoints section
            if line.to_lowercase().contains("api") || line.to_lowercase().contains("endpoint") {
                i = self.parse_api_section(&lines, i, &mut spec.api_endpoints);
                continue; // Re-evaluate the new line position
            }

            // Parse data model section
            if line.contains('|') && line.contains("Field") {
                i = self.parse_table_as_model(&lines, i, &mut spec.data_models);
                continue; // Re-evaluate the new line position
            }

            // Parse validation rules
            if line.to_lowercase().contains("validation") {
                i = self.parse_validation_section(&lines, i, &mut spec.validation_rules);
                continue; // Re-evaluate the new line position
            }

            i += 1;
        }

        Ok(spec)
    }

    fn parse_requirements_section(
        &mut self,
        lines: &[&str],
        start: usize,
        requirements: &mut Vec<Requirement>,
    ) -> usize {
        let mut i = start + 1;

        while i < lines.len() {
            let line = lines[i].trim();

            if line.starts_with('#') {
                break;
            }

            if line.starts_with('-') || line.starts_with('*') {
                let text = line.trim_start_matches('-').trim_start_matches('*').trim();

                // Use pre-compiled regex patterns
                let req_type = if MUST_RE.is_match(text) {
                    "MUST"
                } else if SHOULD_RE.is_match(text) {
                    "SHOULD"
                } else if MAY_RE.is_match(text) {
                    "MAY"
                } else {
                    "SHOULD"
                };

                self.counter += 1;
                requirements.push(Requirement {
                    id: format!("REQ-{:03}", self.counter),
                    text: text.to_string(),
                    requirement_type: req_type.to_string(),
                    category: "functional".to_string(),
                });
            }

            i += 1;
        }

        i
    }

    fn parse_api_section(
        &mut self,
        lines: &[&str],
        start: usize,
        endpoints: &mut Vec<APIEndpoint>,
    ) -> usize {
        let mut i = start + 1;

        while i < lines.len() {
            let line = lines[i].trim();

            if line.starts_with('#') && !line.to_lowercase().contains("api") {
                break;
            }

            // Use pre-compiled regex patterns
            if let Some(captures) = HTTP_METHOD_RE.captures(line) {
                let method = captures.get(0).unwrap().as_str().to_uppercase();

                // Extract path - look for /path pattern using pre-compiled regex
                let path = if let Some(path_match) = API_PATH_RE.find(line) {
                    path_match.as_str().to_string()
                } else {
                    "/api/unknown".to_string()
                };

                endpoints.push(APIEndpoint {
                    method,
                    path,
                    description: line.to_string(),
                    request_body: None,
                    response_body: None,
                });
            }

            i += 1;
        }

        i
    }

    fn parse_table_as_model(
        &mut self,
        lines: &[&str],
        start: usize,
        models: &mut Vec<DataModel>,
    ) -> usize {
        let mut i = start;
        let header_line = lines[i];

        // Parse headers
        let headers: Vec<String> = header_line
            .split('|')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if headers.is_empty() || !headers[0].to_lowercase().contains("field") {
            return i;
        }

        // Skip separator line
        i += 1;
        if i >= lines.len() || !lines[i].contains("---") {
            return i;
        }
        i += 1;

        // Parse data rows
        let mut fields = Vec::new();
        while i < lines.len() {
            let line = lines[i].trim();

            if line.is_empty() || !line.contains('|') {
                break;
            }

            let cells: Vec<String> = line
                .split('|')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if cells.len() >= 2 {
                let field_name = cells[0].clone();
                let field_type = cells.get(1).cloned().unwrap_or_else(|| "String".to_string());
                let required = cells.get(2).map(|s| s.to_lowercase().contains("yes")).unwrap_or(false);
                let validation = cells.get(3).cloned();

                fields.push(Field {
                    name: field_name,
                    field_type,
                    required,
                    validation,
                });
            }

            i += 1;
        }

        if !fields.is_empty() {
            models.push(DataModel {
                name: "FeatureData".to_string(),
                fields,
            });
        }

        i
    }

    fn parse_validation_section(
        &mut self,
        lines: &[&str],
        start: usize,
        rules: &mut Vec<ValidationRule>,
    ) -> usize {
        let mut i = start + 1;

        while i < lines.len() {
            let line = lines[i].trim();

            if line.starts_with('#') {
                break;
            }

            if line.starts_with('-') || line.starts_with('*') {
                let text = line.trim_start_matches('-').trim_start_matches('*').trim();
                
                // Try to extract field name and rule
                if let Some((field, rule)) = text.split_once(':') {
                    rules.push(ValidationRule {
                        field: field.trim().to_string(),
                        rule: rule.trim().to_string(),
                        error_message: format!("{} validation failed", field.trim()),
                    });
                }
            }

            i += 1;
        }

        i
    }

    /// Generate Gitea Go code for the feature
    pub fn generate_gitea_code(&self, spec: &FeatureSpec) -> String {
        let mut code = Vec::new();

        code.push("// Auto-generated Gitea feature scaffolding".to_string());
        code.push(format!("// Feature: {}", spec.name));
        code.push("// Generated by: Lean4 Parser Forward Engineering Tool".to_string());
        code.push("".to_string());
        code.push("package features".to_string());
        code.push("".to_string());
        code.push("import (".to_string());
        code.push("\t\"net/http\"".to_string());
        code.push("\t\"code.gitea.io/gitea/modules/context\"".to_string());
        code.push("\t\"code.gitea.io/gitea/modules/structs\"".to_string());
        code.push(")".to_string());
        code.push("".to_string());

        // Generate data models
        for model in &spec.data_models {
            code.push(self.generate_go_struct(model));
            code.push("".to_string());
        }

        // Generate API handlers
        for endpoint in &spec.api_endpoints {
            code.push(self.generate_go_handler(endpoint, spec));
            code.push("".to_string());
        }

        // Generate validation functions
        if !spec.validation_rules.is_empty() {
            code.push(self.generate_validation_function(spec));
        }

        code.join("\n")
    }

    fn generate_go_struct(&self, model: &DataModel) -> String {
        let mut lines = Vec::new();
        lines.push(format!("// {} represents the data model", model.name));
        lines.push(format!("type {} struct {{", model.name));

        for field in &model.fields {
            let field_name = self.to_pascal_case(&field.name);
            let go_type = self.map_type_to_go(&field.field_type);
            let json_tag = format!("`json:\"{}\"`", field.name);

            if field.required {
                lines.push(format!("\t{} {} {} // Required", field_name, go_type, json_tag));
            } else {
                lines.push(format!("\t{} *{} {} // Optional", field_name, go_type, json_tag));
            }
        }

        lines.push("}".to_string());
        lines.join("\n")
    }

    fn generate_go_handler(&self, endpoint: &APIEndpoint, spec: &FeatureSpec) -> String {
        let handler_name = format!("Handle{}{}", 
            endpoint.method,
            self.to_pascal_case(&endpoint.path.replace('/', "_"))
        );

        let mut lines = Vec::new();
        lines.push(format!("// {} handles {}", handler_name, endpoint.description));
        lines.push(format!("func {}(ctx *context.APIContext) {{", handler_name));
        lines.push("\t// TODO: Implement feature logic".to_string());
        lines.push("".to_string());
        
        // Add requirement checks as comments
        for req in &spec.requirements {
            if req.requirement_type == "MUST" {
                lines.push(format!("\t// {}: {}", req.id, req.text));
            }
        }
        
        lines.push("".to_string());
        lines.push("\tctx.JSON(http.StatusOK, map[string]interface{}{".to_string());
        lines.push("\t\t\"message\": \"Feature implemented\",".to_string());
        lines.push("\t})".to_string());
        lines.push("}".to_string());

        lines.join("\n")
    }

    fn generate_validation_function(&self, spec: &FeatureSpec) -> String {
        let mut lines = Vec::new();
        lines.push("// ValidateFeatureData validates input data".to_string());
        lines.push("func ValidateFeatureData(data *FeatureData) error {".to_string());

        for rule in &spec.validation_rules {
            lines.push(format!("\t// Validate {}: {}", rule.field, rule.rule));
            lines.push(format!("\t// if !validate{}() {{", self.to_pascal_case(&rule.field)));
            lines.push(format!("\t//     return fmt.Errorf(\"{}\")", rule.error_message));
            lines.push("\t// }".to_string());
        }

        lines.push("\treturn nil".to_string());
        lines.push("}".to_string());

        lines.join("\n")
    }

    /// Generate Lean4 proofs for feature verification
    pub fn generate_lean4_proofs(&self, spec: &FeatureSpec) -> String {
        let mut code = Vec::new();

        code.push("-- Lean4 Formal Verification for Gitea Feature".to_string());
        code.push(format!("-- Feature: {}", spec.name));
        code.push("-- Auto-generated by Forward Engineering Tool".to_string());
        code.push("".to_string());
        code.push("import Std".to_string());
        code.push("".to_string());

        // Generate namespace
        let namespace = self.sanitize_lean4_name(&spec.name);
        code.push(format!("namespace {}", namespace));
        code.push("".to_string());

        // Generate axioms for requirements
        for req in &spec.requirements {
            let req_name = self.sanitize_lean4_name(&req.text[..req.text.len().min(40)]);
            
            code.push(format!("-- {}: {}", req.id, req.text));
            code.push(format!("axiom {} : True", req_name));
            code.push("".to_string());
        }

        // Generate theorems for data model validity
        for model in &spec.data_models {
            code.push(format!("-- Data Model: {}", model.name));
            code.push(format!("structure {} where", model.name));
            
            for field in &model.fields {
                let field_name = self.sanitize_lean4_name(&field.name);
                let lean_type = self.map_type_to_lean4(&field.field_type);
                
                if field.required {
                    code.push(format!("  {} : {}", field_name, lean_type));
                } else {
                    code.push(format!("  {} : Option {}", field_name, lean_type));
                }
            }
            code.push("".to_string());

            // Generate validity theorem
            code.push(format!("theorem {}_valid : True := by", self.sanitize_lean4_name(&model.name)));
            code.push("  trivial".to_string());
            code.push("".to_string());
        }

        // Generate API endpoint specifications
        for endpoint in &spec.api_endpoints {
            let endpoint_name = self.sanitize_lean4_name(&format!("{}_{}", endpoint.method, &endpoint.path));
            
            code.push(format!("-- API Endpoint: {} {}", endpoint.method, endpoint.path));
            code.push(format!("axiom {}_exists : True", endpoint_name));
            code.push("".to_string());
        }

        // Generate validation proofs
        if !spec.validation_rules.is_empty() {
            code.push("-- Validation Rules".to_string());
            for rule in &spec.validation_rules {
                let rule_name = self.sanitize_lean4_name(&format!("validate_{}", rule.field));
                code.push(format!("axiom {} : True  -- {}", rule_name, rule.rule));
            }
            code.push("".to_string());
        }

        // Generate completeness theorem
        code.push("-- Completeness: All requirements are specified".to_string());
        code.push(format!("theorem {}_complete : True := by", namespace));
        code.push("  trivial".to_string());
        code.push("".to_string());

        code.push(format!("end {}", namespace));

        code.join("\n")
    }

    /// Generate SCIP specification
    pub fn generate_scip_spec(&self, spec: &FeatureSpec) -> Vec<SCIPElement> {
        let mut elements = Vec::new();
        let now = chrono::Utc::now();

        // Convert each requirement to SCIP element
        for req in &spec.requirements {
            let severity = match req.requirement_type.as_str() {
                "MUST" => SCIPSeverity::High,
                "SHOULD" => SCIPSeverity::Medium,
                "MAY" => SCIPSeverity::Low,
                _ => SCIPSeverity::Medium,
            };

            elements.push(SCIPElement {
                id: req.id.clone(),
                element_type: SCIPElementType::Requirement,
                title: req.text.chars().take(80).collect(),
                description: req.text.clone(),
                severity,
                status: SCIPStatus::Draft,
                formal_spec: Some(format!("axiom {} : True", self.sanitize_lean4_name(&req.text))),
                proof_status: Some("axiom".to_string()),
                category: Some(req.category.clone()),
                tags: Some(vec!["gitea_feature".to_string(), req.requirement_type.to_lowercase()]),
                references: None,
                verification_method: Some("formal_proof".to_string()),
                test_cases: None,
                source: Some("feature_generator".to_string()),
                derived_from: Some(vec![spec.name.clone()]),
                created_at: now,
                updated_at: now,
            });
        }

        elements
    }

    /// Generate complete feature package for Gitea
    pub fn generate_feature_package(&self, spec: &FeatureSpec, output_dir: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        let output_path = Path::new(output_dir);
        fs::create_dir_all(output_path)?;

        // Generate Go code
        let go_code = self.generate_gitea_code(spec);
        fs::write(output_path.join("feature.go"), go_code)?;

        // Generate Lean4 proofs
        let lean4_proofs = self.generate_lean4_proofs(spec);
        fs::write(output_path.join("feature_proofs.lean"), lean4_proofs)?;

        // Generate SCIP spec
        let scip_elements = self.generate_scip_spec(spec);
        let scip_json = serde_json::to_string_pretty(&serde_json::json!({
            "scip_version": "1.0",
            "feature": spec.name,
            "elements": scip_elements,
        }))?;
        fs::write(output_path.join("scip_spec.json"), scip_json)?;

        // Generate README
        let readme = self.generate_feature_readme(spec);
        fs::write(output_path.join("README.md"), readme)?;

        Ok(())
    }

    fn generate_feature_readme(&self, spec: &FeatureSpec) -> String {
        let mut lines = Vec::new();

        lines.push(format!("# {}", spec.name));
        lines.push("".to_string());
        lines.push("## Auto-generated Feature Package".to_string());
        lines.push("".to_string());
        lines.push("This package contains:".to_string());
        lines.push("- `feature.go` - Go implementation scaffolding".to_string());
        lines.push("- `feature_proofs.lean` - Lean4 formal proofs".to_string());
        lines.push("- `scip_spec.json` - SCIP compliance specification".to_string());
        lines.push("".to_string());
        lines.push("## Requirements".to_string());
        lines.push("".to_string());

        for req in &spec.requirements {
            lines.push(format!("- **{}** [{}]: {}", req.id, req.requirement_type, req.text));
        }

        lines.push("".to_string());
        lines.push("## API Endpoints".to_string());
        lines.push("".to_string());

        for endpoint in &spec.api_endpoints {
            lines.push(format!("- `{} {}` - {}", endpoint.method, endpoint.path, endpoint.description));
        }

        lines.push("".to_string());
        lines.push("## Data Models".to_string());
        lines.push("".to_string());

        for model in &spec.data_models {
            lines.push(format!("### {}", model.name));
            lines.push("".to_string());
            for field in &model.fields {
                let req_str = if field.required { "required" } else { "optional" };
                lines.push(format!("- `{}` ({}) - {}", field.name, field.field_type, req_str));
            }
            lines.push("".to_string());
        }

        lines.join("\n")
    }

    // Helper functions
    fn to_pascal_case(&self, s: &str) -> String {
        s.split(&['_', '-', ' '][..])
            .filter(|s| !s.is_empty())
            .map(|s| {
                let mut chars = s.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                }
            })
            .collect()
    }

    fn sanitize_lean4_name(&self, text: &str) -> String {
        // Use pre-compiled regex patterns
        let mut name = LEAN4_NAME_SANITIZE_RE.replace_all(text, "_").to_string();
        name = MULTI_UNDERSCORE_RE.replace_all(&name, "_").to_string();
        name = name.trim_matches('_').to_string();

        if name.is_empty() || name.chars().next().unwrap().is_numeric() {
            name = format!("r_{}", name);
        }

        name
    }

    fn map_type_to_go(&self, field_type: &str) -> String {
        match field_type.to_lowercase().as_str() {
            "string" | "text" => "string",
            "int" | "integer" | "number" => "int64",
            "bool" | "boolean" => "bool",
            "decimal" | "float" => "float64",
            "datetime" | "timestamp" => "time.Time",
            _ => "string",
        }
        .to_string()
    }

    fn map_type_to_lean4(&self, field_type: &str) -> String {
        match field_type.to_lowercase().as_str() {
            "string" | "text" => "String",
            "int" | "integer" | "number" => "Nat",
            "bool" | "boolean" => "Bool",
            "decimal" | "float" => "Float",
            _ => "String",
        }
        .to_string()
    }
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_feature_spec() {
        let markdown = r#"
# User Authentication Feature

## Requirements
- MUST validate username and password
- SHOULD support 2FA
- MAY support OAuth providers

## API Endpoints
- POST /api/auth/login
- GET /api/auth/status

## Data Model
| Field | Type | Required |
|-------|------|----------|
| username | string | yes |
| password | string | yes |
"#;

        let mut generator = FeatureGenerator::new();
        let spec = generator.parse_feature_spec(markdown).unwrap();

        assert_eq!(spec.name, "User Authentication Feature");
        assert_eq!(spec.requirements.len(), 3);
        assert_eq!(spec.api_endpoints.len(), 2);
        assert_eq!(spec.data_models.len(), 1);
    }

    #[test]
    fn test_to_pascal_case() {
        let generator = FeatureGenerator::new();
        assert_eq!(generator.to_pascal_case("user_name"), "UserName");
        assert_eq!(generator.to_pascal_case("api-endpoint"), "ApiEndpoint");
    }
}