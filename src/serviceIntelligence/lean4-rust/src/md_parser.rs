/// Markdown to Lean4 Parser (Rust implementation)
/// High-performance parser with zero-copy where possible

use crate::models::{Lean4Element, Lean4ElementType};
use lazy_static::lazy_static;
use pulldown_cmark::{Event, Parser, Tag};
use regex::Regex;
use std::collections::HashMap;

// Pre-compiled regex patterns for better performance
lazy_static! {
    /// Pattern for sanitizing names (removes non-alphanumeric chars except underscore)
    static ref NAME_SANITIZE_RE: Regex = Regex::new(r"[^a-zA-Z0-9_]").unwrap();

    /// Pattern for collapsing multiple underscores
    static ref MULTI_UNDERSCORE_RE: Regex = Regex::new(r"_+").unwrap();

    /// Pattern for detecting MUST/SHALL/REQUIRED keywords (requirements)
    static ref REQUIREMENT_RE: Regex = Regex::new(r"(?i)(MUST|SHALL|REQUIRED|MANDATORY)").unwrap();

    /// Pattern for detecting CONSTRAINT/LIMIT/MUST NOT keywords (constraints)
    static ref CONSTRAINT_RE: Regex = Regex::new(r"(?i)(CONSTRAINT|LIMIT|MUST NOT)").unwrap();
}

pub struct MarkdownParser {
    counter: usize,
}

impl MarkdownParser {
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    /// Parse markdown text into Lean4 elements
    pub fn parse_to_lean4(&mut self, markdown: &str) -> Vec<Lean4Element> {
        let mut elements = Vec::new();
        let parser = Parser::new(markdown);
        
        let mut current_text = String::new();
        let mut in_heading = false;
        let mut heading_level = 0;
        let mut in_code_block = false;
        let mut code_content = String::new();
        let mut in_list = false;
        let mut table_rows: Vec<Vec<String>> = Vec::new();
        let mut in_table = false;
        
        for event in parser {
            match event {
                Event::Start(Tag::Heading(level, _, _)) => {
                    in_heading = true;
                    heading_level = level as u32;
                    current_text.clear();
                }
                Event::End(Tag::Heading(..)) => {
                    if in_heading {
                        elements.push(self.heading_to_lean4(&current_text, heading_level));
                        in_heading = false;
                        current_text.clear();
                    }
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    in_code_block = true;
                    code_content.clear();
                }
                Event::End(Tag::CodeBlock(_)) => {
                    if in_code_block {
                        elements.push(self.code_to_lean4(&code_content));
                        in_code_block = false;
                    }
                }
                Event::Start(Tag::List(_)) => {
                    in_list = true;
                }
                Event::End(Tag::List(_)) => {
                    in_list = false;
                }
                Event::Start(Tag::Item) => {
                    current_text.clear();
                }
                Event::End(Tag::Item) => {
                    if in_list && !current_text.is_empty() {
                        if let Some(element) = self.list_item_to_lean4(&current_text) {
                            elements.push(element);
                        }
                        current_text.clear();
                    }
                }
                Event::Start(Tag::Table(_)) => {
                    in_table = true;
                    table_rows.clear();
                }
                Event::End(Tag::Table(_)) => {
                    if in_table {
                        elements.extend(self.table_to_lean4(&table_rows));
                        in_table = false;
                    }
                }
                Event::Start(Tag::TableRow) => {
                    table_rows.push(Vec::new());
                }
                Event::Start(Tag::TableCell) => {
                    current_text.clear();
                }
                Event::End(Tag::TableCell) => {
                    if let Some(row) = table_rows.last_mut() {
                        row.push(current_text.trim().to_string());
                    }
                    current_text.clear();
                }
                Event::Text(text) => {
                    if in_code_block {
                        code_content.push_str(&text);
                    } else {
                        current_text.push_str(&text);
                    }
                }
                Event::Code(code) => {
                    current_text.push('`');
                    current_text.push_str(&code);
                    current_text.push('`');
                }
                _ => {}
            }
        }
        
        elements
    }

    fn sanitize_name(&self, text: &str) -> String {
        // Use pre-compiled regex for better performance
        let mut name = NAME_SANITIZE_RE.replace_all(text, "_").to_string();

        // Remove multiple underscores using pre-compiled regex
        name = MULTI_UNDERSCORE_RE.replace_all(&name, "_").to_string();

        // Trim underscores
        name = name.trim_matches('_').to_string();

        // Ensure starts with letter
        if name.is_empty() || name.chars().next().unwrap().is_numeric() {
            name = format!("r_{}", name);
        }

        if name.is_empty() {
            name = format!("element_{}", self.counter);
        }

        name
    }

    fn heading_to_lean4(&mut self, content: &str, level: u32) -> Lean4Element {
        self.counter += 1;
        let name = self.sanitize_name(content);
        
        let mut metadata = HashMap::new();
        metadata.insert("original".to_string(), content.to_string());
        metadata.insert("level".to_string(), level.to_string());
        metadata.insert("description".to_string(), format!("Section: {}", content));
        
        Lean4Element {
            name: name.clone(),
            element_type: Lean4ElementType::Axiom,
            statement: format!("axiom {}_exists : True", name),
            proof: None,
            dependencies: None,
            metadata: Some(metadata),
        }
    }

    fn list_item_to_lean4(&mut self, content: &str) -> Option<Lean4Element> {
        self.counter += 1;

        // Check if it's a requirement or constraint using pre-compiled regex
        let (element_type_str, prefix) = if CONSTRAINT_RE.is_match(content) {
            ("constraint", "constraint_")
        } else if REQUIREMENT_RE.is_match(content) {
            ("requirement", "req_")
        } else {
            return None; // Skip non-requirement/constraint list items
        };
        
        let name = self.sanitize_name(&content[..content.len().min(50)]);
        let full_name = format!("{}{}", prefix, name);
        
        let mut metadata = HashMap::new();
        metadata.insert("original".to_string(), content.to_string());
        metadata.insert("type".to_string(), element_type_str.to_string());
        metadata.insert("description".to_string(), content.to_string());
        
        Some(Lean4Element {
            name: full_name.clone(),
            element_type: Lean4ElementType::Axiom,
            statement: format!("axiom {}_required : True", name),
            proof: None,
            dependencies: None,
            metadata: Some(metadata),
        })
    }

    fn code_to_lean4(&mut self, content: &str) -> Lean4Element {
        self.counter += 1;
        
        // Check if it's already Lean4 code
        if content.contains("theorem") || content.contains("def") || content.contains("axiom") {
            let mut metadata = HashMap::new();
            metadata.insert("original".to_string(), content.to_string());
            metadata.insert("type".to_string(), "embedded_lean4".to_string());
            
            return Lean4Element {
                name: format!("embedded_code_{}", self.counter),
                element_type: Lean4ElementType::Def,
                statement: content.to_string(),
                proof: None,
                dependencies: None,
                metadata: Some(metadata),
            };
        }
        
        // Otherwise, create a comment
        let commented = content
            .lines()
            .map(|line| format!("-- {}", line))
            .collect::<Vec<_>>()
            .join("\n");
        
        let mut metadata = HashMap::new();
        metadata.insert("original".to_string(), content.to_string());
        metadata.insert("type".to_string(), "code_reference".to_string());
        
        Lean4Element {
            name: format!("code_block_{}", self.counter),
            element_type: Lean4ElementType::Axiom,
            statement: format!("-- Code block:\n{}", commented),
            proof: None,
            dependencies: None,
            metadata: Some(metadata),
        }
    }

    fn table_to_lean4(&mut self, rows: &[Vec<String>]) -> Vec<Lean4Element> {
        if rows.len() < 2 {
            return Vec::new();
        }
        
        let headers = &rows[0];
        let data_rows = &rows[1..]; // Skip separator row if any
        
        let mut elements = Vec::new();
        
        for (idx, row) in data_rows.iter().enumerate() {
            if row.is_empty() {
                continue;
            }
            
            self.counter += 1;
            let row_name = self.sanitize_name(&row.get(0).map(|s| s.as_str()).unwrap_or(&format!("row_{}", idx)));
            
            // Create structure fields
            let fields: Vec<String> = headers
                .iter()
                .map(|header| {
                    let field_name = self.sanitize_name(header);
                    format!("  {} : String", field_name)
                })
                .collect();
            
            let statement = format!(
                "structure TableRow_{} where\n{}",
                row_name,
                fields.join("\n")
            );
            
            let mut metadata = HashMap::new();
            metadata.insert("original".to_string(), row.join(" | "));
            metadata.insert("type".to_string(), "table_row".to_string());
            metadata.insert("headers".to_string(), headers.join(","));
            metadata.insert("values".to_string(), row.join(","));
            
            elements.push(Lean4Element {
                name: format!("TableRow_{}", row_name),
                element_type: Lean4ElementType::Structure,
                statement,
                proof: None,
                dependencies: None,
                metadata: Some(metadata),
            });
        }
        
        elements
    }
}

/// Generate Lean4 code from elements
pub fn generate_lean4_code(elements: &[Lean4Element]) -> String {
    let mut code = Vec::new();
    
    code.push("-- Auto-generated from Markdown".to_string());
    code.push("-- Parser: Rust lean4-parser".to_string());
    code.push("".to_string());
    code.push("import Std".to_string());
    code.push("".to_string());
    
    for element in elements {
        // Add comment with metadata
        if let Some(metadata) = &element.metadata {
            if let Some(description) = metadata.get("description") {
                code.push(format!("-- {}", description));
            }
        }
        
        // Add the statement
        code.push(element.statement.clone());
        
        // Add proof if exists
        if let Some(proof) = &element.proof {
            code.push(proof.clone());
        }
        
        code.push("".to_string()); // Empty line
    }
    
    code.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_requirements() {
        let markdown = r#"
# Requirements
- MUST validate input
- SHALL verify authentication
"#;
        
        let mut parser = MarkdownParser::new();
        let elements = parser.parse_to_lean4(markdown);
        
        assert!(!elements.is_empty());
        assert!(elements.iter().any(|e| e.metadata
            .as_ref()
            .and_then(|m| m.get("type"))
            .map(|t| t == "requirement")
            .unwrap_or(false)));
    }

    #[test]
    fn test_sanitize_name() {
        let parser = MarkdownParser::new();
        
        assert_eq!(parser.sanitize_name("Hello World!"), "Hello_World");
        assert_eq!(parser.sanitize_name("123test"), "r_123test");
        assert_eq!(parser.sanitize_name("test___name"), "test_name");
    }
}