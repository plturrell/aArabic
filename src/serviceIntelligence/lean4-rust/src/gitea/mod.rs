/// Gitea-specific code generation and templates
/// For generating Gitea features, Go code, and specifications

pub mod feature_generator;
pub mod template_generator;
pub mod n8n_parser;

pub use feature_generator::{FeatureGenerator, FeatureSpec};
pub use template_generator::{TemplateGenerator, TemplateConfig};
pub use n8n_parser::{N8nToMarkdownConverter, N8nWorkflow};
