/// Lean4 Parser Library
/// High-performance Markdown → Lean4 → SCIP conversion

pub mod config;
pub mod models;
pub mod md_parser;
pub mod scip_converter;
pub mod pipeline;

// Service-specific modules
pub mod gitea;

// Re-export main types for convenience
pub use models::{
    Lean4Element, Lean4ElementType,
    SCIPElement, SCIPElementType, SCIPSeverity, SCIPStatus,
    ParseRequest, ParseResult, PipelineResults,
};

pub use md_parser::{MarkdownParser, generate_lean4_code};
pub use scip_converter::{SCIPConverter, generate_scip_json, generate_scip_markdown};
pub use pipeline::ConversionPipeline;

// Re-export Gitea-specific types
pub use gitea::{
    FeatureGenerator, FeatureSpec,
    TemplateGenerator, TemplateConfig,
    N8nToMarkdownConverter, N8nWorkflow,
};

// Re-export configuration
pub use config::{AppConfig, CONFIG, is_path_allowed, sanitize_path};
