/// M2M100 Model Implementation in Burn
/// Complete Rust implementation for Arabic-English translation

pub mod attention;
pub mod embedding;
pub mod encoder;
pub mod decoder;
pub mod m2m100;

pub use m2m100::M2M100ForConditionalGeneration;
