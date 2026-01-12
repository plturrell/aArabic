use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Complete Hyperbook API Client
/// Educational content management system
pub struct HyperbookClient {
    base_path: String,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Book {
    pub name: String,
    pub language: Option<String>,
    pub src: Option<String>,
    pub sections: Option<Vec<Section>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    pub name: String,
    pub path: Option<String>,
    pub sections: Option<Vec<Section>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbookConfig {
    pub book: Book,
    pub build: Option<BuildConfig>,
    pub preprocessor: Option<HashMap<String, PreprocessorConfig>>,
    pub output: Option<HashMap<String, OutputConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    pub build_dir: Option<String>,
    pub create_missing: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    pub command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub optional: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chapter {
    pub name: String,
    pub path: String,
    pub content: String,
    pub sub_items: Vec<Chapter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    pub element_type: String,
    pub title: Option<String>,
    pub content: String,
    pub collapsible: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossaryTerm {
    pub term: String,
    pub definition: String,
    pub references: Vec<String>,
}

impl HyperbookClient {
    /// Create new Hyperbook client
    pub fn new(base_path: String) -> Self {
        Self { base_path }
    }

    // ========================================================================
    // BOOK OPERATIONS
    // ========================================================================

    /// Initialize new book
    pub fn init_book(&self, name: &str, language: &str) -> Result<()> {
        let book_path = Path::new(&self.base_path).join(name);
        fs::create_dir_all(&book_path)?;
        
        let config = HyperbookConfig {
            book: Book {
                name: name.to_string(),
                language: Some(language.to_string()),
                src: Some("src".to_string()),
                sections: Some(vec![]),
            },
            build: Some(BuildConfig {
                build_dir: Some("book".to_string()),
                create_missing: Some(true),
            }),
            preprocessor: None,
            output: Some(HashMap::from([(
                "html".to_string(),
                OutputConfig { optional: Some(false) },
            )])),
        };

        let config_path = book_path.join("book.toml");
        let toml_string = toml::to_string_pretty(&config)?;
        fs::write(config_path, toml_string)?;

        // Create src directory
        let src_path = book_path.join("src");
        fs::create_dir_all(&src_path)?;

        // Create initial README
        fs::write(src_path.join("README.md"), "# Introduction\n\nWelcome to your Hyperbook!")?;

        Ok(())
    }

    /// Load book config
    pub fn load_config(&self, book_name: &str) -> Result<HyperbookConfig> {
        let config_path = Path::new(&self.base_path)
            .join(book_name)
            .join("book.toml");
        
        let content = fs::read_to_string(config_path)?;
        Ok(toml::from_str(&content)?)
    }

    /// Save book config
    pub fn save_config(&self, book_name: &str, config: &HyperbookConfig) -> Result<()> {
        let config_path = Path::new(&self.base_path)
            .join(book_name)
            .join("book.toml");
        
        let toml_string = toml::to_string_pretty(config)?;
        fs::write(config_path, toml_string)?;
        Ok(())
    }

    /// List books
    pub fn list_books(&self) -> Result<Vec<String>> {
        let mut books = Vec::new();
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            if entry.path().is_dir() {
                if entry.path().join("book.toml").exists() {
                    if let Some(name) = entry.file_name().to_str() {
                        books.push(name.to_string());
                    }
                }
            }
        }
        Ok(books)
    }

    // ========================================================================
    // CHAPTER OPERATIONS
    // ========================================================================

    /// Create chapter
    pub fn create_chapter(
        &self,
        book_name: &str,
        chapter_name: &str,
        content: &str,
    ) -> Result<()> {
        let chapter_path = Path::new(&self.base_path)
            .join(book_name)
            .join("src")
            .join(format!("{}.md", chapter_name));
        
        fs::write(chapter_path, content)?;
        Ok(())
    }

    /// Read chapter
    pub fn read_chapter(&self, book_name: &str, chapter_name: &str) -> Result<String> {
        let chapter_path = Path::new(&self.base_path)
            .join(book_name)
            .join("src")
            .join(format!("{}.md", chapter_name));
        
        Ok(fs::read_to_string(chapter_path)?)
    }

    /// Update chapter
    pub fn update_chapter(
        &self,
        book_name: &str,
        chapter_name: &str,
        content: &str,
    ) -> Result<()> {
        self.create_chapter(book_name, chapter_name, content)
    }

    /// Delete chapter
    pub fn delete_chapter(&self, book_name: &str, chapter_name: &str) -> Result<()> {
        let chapter_path = Path::new(&self.base_path)
            .join(book_name)
            .join("src")
            .join(format!("{}.md", chapter_name));
        
        fs::remove_file(chapter_path)?;
        Ok(())
    }

    /// List chapters
    pub fn list_chapters(&self, book_name: &str) -> Result<Vec<String>> {
        let src_path = Path::new(&self.base_path).join(book_name).join("src");
        
        let mut chapters = Vec::new();
        for entry in fs::read_dir(src_path)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("md") {
                if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    chapters.push(name.to_string());
                }
            }
        }
        Ok(chapters)
    }

    // ========================================================================
    // ELEMENT OPERATIONS
    // ========================================================================

    /// Create alert element
    pub fn create_alert(&self, alert_type: &str, title: Option<&str>, content: &str) -> String {
        let title_str = title.map(|t| format!("title=\"{}\"", t)).unwrap_or_default();
        format!(":::alert{{type=\"{}\" {}}}\n{}\n:::", alert_type, title_str, content)
    }

    /// Create collapsible element
    pub fn create_collapsible(&self, title: &str, content: &str, open: bool) -> String {
        let open_str = if open { " open" } else { "" };
        format!(":::collapsible{{title=\"{}\"{}}}\n{}\n:::", title, open_str, content)
    }

    /// Create tabs element
    pub fn create_tabs(&self, tabs: Vec<(&str, &str)>) -> String {
        let mut result = ":::tabs\n".to_string();
        for (title, content) in tabs {
            result.push_str(&format!(":::tab{{title=\"{}\"}}\n{}\n:::\n", title, content));
        }
        result.push_str(":::");
        result
    }

    /// Create code block with syntax highlighting
    pub fn create_code_block(&self, language: &str, code: &str, filename: Option<&str>) -> String {
        let filename_str = filename.map(|f| format!(" title=\"{}\"", f)).unwrap_or_default();
        format!("```{}{}\n{}\n```", language, filename_str, code)
    }

    /// Create mermaid diagram
    pub fn create_mermaid(&self, diagram: &str) -> String {
        format!("```mermaid\n{}\n```", diagram)
    }

    /// Create math block
    pub fn create_math(&self, formula: &str, inline: bool) -> String {
        if inline {
            format!("${}", formula)
        } else {
            format!("$$\n{}\n$$", formula)
        }
    }

    // ========================================================================
    // GLOSSARY OPERATIONS
    // ========================================================================

    /// Create glossary
    pub fn create_glossary(&self, book_name: &str, terms: Vec<GlossaryTerm>) -> Result<()> {
        let mut content = String::from("# Glossary\n\n");
        
        for term in terms {
            content.push_str(&format!("## {}\n\n", term.term));
            content.push_str(&format!("{}\n\n", term.definition));
            if !term.references.is_empty() {
                content.push_str("**References:**\n");
                for reference in term.references {
                    content.push_str(&format!("- {}\n", reference));
                }
                content.push('\n');
            }
        }

        let glossary_path = Path::new(&self.base_path)
            .join(book_name)
            .join("src")
            .join("GLOSSARY.md");
        
        fs::write(glossary_path, content)?;
        Ok(())
    }

    // ========================================================================
    // ASSET OPERATIONS
    // ========================================================================

    /// Add image
    pub fn add_image(&self, book_name: &str, image_name: &str, image_data: &[u8]) -> Result<()> {
        let images_path = Path::new(&self.base_path)
            .join(book_name)
            .join("src")
            .join("images");
        
        fs::create_dir_all(&images_path)?;
        fs::write(images_path.join(image_name), image_data)?;
        Ok(())
    }

    /// Create image reference
    pub fn image_reference(&self, alt_text: &str, image_path: &str, title: Option<&str>) -> String {
        if let Some(t) = title {
            format!("![{}]({} \"{}\")", alt_text, image_path, t)
        } else {
            format!("![{}]({})", alt_text, image_path)
        }
    }

    // ========================================================================
    // METADATA OPERATIONS
    // ========================================================================

    /// Add frontmatter to chapter
    pub fn add_frontmatter(&self, chapter_content: &str, metadata: HashMap<String, String>) -> String {
        let mut frontmatter = String::from("---\n");
        for (key, value) in metadata {
            frontmatter.push_str(&format!("{}: {}\n", key, value));
        }
        frontmatter.push_str("---\n\n");
        frontmatter.push_str(chapter_content);
        frontmatter
    }

    // ========================================================================
    // BUILD OPERATIONS
    // ========================================================================

    /// Build book (simulated - would call hyperbook CLI)
    pub fn build_book(&self, book_name: &str) -> Result<String> {
        let book_path = Path::new(&self.base_path).join(book_name);
        
        // This would normally call: hyperbook build
        // For now, we'll just verify the structure exists
        if !book_path.exists() {
            anyhow::bail!("Book does not exist");
        }
        
        if !book_path.join("book.toml").exists() {
            anyhow::bail!("book.toml not found");
        }
        
        Ok(format!("Book '{}' is ready to build", book_name))
    }

    /// Serve book (simulated)
    pub fn serve_book(&self, book_name: &str, port: u16) -> Result<String> {
        // This would normally call: hyperbook serve
        Ok(format!("Book '{}' would be served on http://localhost:{}", book_name, port))
    }

    // ========================================================================
    // SEARCH OPERATIONS
    // ========================================================================

    /// Search content
    pub fn search(&self, book_name: &str, query: &str) -> Result<Vec<(String, String)>> {
        let src_path = Path::new(&self.base_path).join(book_name).join("src");
        let mut results = Vec::new();
        
        for entry in fs::read_dir(src_path)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("md") {
                let content = fs::read_to_string(entry.path())?;
                if content.to_lowercase().contains(&query.to_lowercase()) {
                    if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                        // Find matching line
                        for line in content.lines() {
                            if line.to_lowercase().contains(&query.to_lowercase()) {
                                results.push((name.to_string(), line.to_string()));
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }

    // ========================================================================
    // UTILITY OPERATIONS
    // ========================================================================

    /// Get book statistics
    pub fn get_stats(&self, book_name: &str) -> Result<HashMap<String, usize>> {
        let src_path = Path::new(&self.base_path).join(book_name).join("src");
        let mut stats = HashMap::new();
        
        let mut total_chapters = 0;
        let mut total_words = 0;
        let mut total_lines = 0;
        
        for entry in fs::read_dir(src_path)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("md") {
                total_chapters += 1;
                let content = fs::read_to_string(entry.path())?;
                total_words += content.split_whitespace().count();
                total_lines += content.lines().count();
            }
        }
        
        stats.insert("chapters".to_string(), total_chapters);
        stats.insert("words".to_string(), total_words);
        stats.insert("lines".to_string(), total_lines);
        
        Ok(stats)
    }
}
