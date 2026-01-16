/// Flexible Data Loader for Arabic-English Translation Datasets
/// Supports: CSV, JSON, Parquet, Plain Text, and custom formats

use anyhow::{Context, Result};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationPair {
    pub arabic: String,
    pub english: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum DatasetFormat {
    /// CSV with configurable column names
    Csv {
        arabic_col: String,
        english_col: String,
        delimiter: u8,
    },
    /// JSON/JSONL with flexible field paths
    Json {
        arabic_field: String,
        english_field: String,
    },
    /// Parquet files
    Parquet {
        arabic_col: String,
        english_col: String,
    },
    /// Plain text files (parallel corpus)
    PlainText {
        arabic_file: PathBuf,
        english_file: PathBuf,
    },
    /// Auto-detect format
    Auto,
}

pub struct DataLoader {
    config: DataLoaderConfig,
}

#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub format: DatasetFormat,
    pub max_pairs: Option<usize>,
    pub min_arabic_len: usize,
    pub max_arabic_len: usize,
    pub min_english_len: usize,
    pub max_english_len: usize,
    pub filter_duplicates: bool,
    pub shuffle: bool,
    pub validation_split: f32,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            format: DatasetFormat::Auto,
            max_pairs: None,
            min_arabic_len: 3,
            max_arabic_len: 512,
            min_english_len: 3,
            max_english_len: 512,
            filter_duplicates: true,
            shuffle: true,
            validation_split: 0.1,
        }
    }
}

impl DataLoader {
    pub fn new(config: DataLoaderConfig) -> Self {
        Self { config }
    }

    /// Load dataset from any supported format
    pub fn load(&self, path: &Path) -> Result<Vec<TranslationPair>> {
        info!("Loading dataset from: {}", path.display());

        let format = match &self.config.format {
            DatasetFormat::Auto => self.detect_format(path)?,
            format => format.clone(),
        };

        let mut pairs = match format {
            DatasetFormat::Csv {
                arabic_col,
                english_col,
                delimiter,
            } => self.load_csv(path, &arabic_col, &english_col, delimiter)?,
            DatasetFormat::Json {
                arabic_field,
                english_field,
            } => self.load_json(path, &arabic_field, &english_field)?,
            DatasetFormat::Parquet {
                arabic_col,
                english_col,
            } => self.load_parquet(path, &arabic_col, &english_col)?,
            DatasetFormat::PlainText {
                arabic_file,
                english_file,
            } => self.load_plain_text(&arabic_file, &english_file)?,
            DatasetFormat::Auto => unreachable!(),
        };

        info!("Loaded {} raw pairs", pairs.len());

        // Apply filters
        pairs = self.filter_pairs(pairs)?;
        info!("After filtering: {} pairs", pairs.len());

        // Limit pairs if specified
        if let Some(max) = self.config.max_pairs {
            pairs.truncate(max);
            info!("Limited to {} pairs", pairs.len());
        }

        // Shuffle if requested
        if self.config.shuffle {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            pairs.shuffle(&mut thread_rng());
            info!("Shuffled dataset");
        }

        Ok(pairs)
    }

    /// Auto-detect dataset format
    fn detect_format(&self, path: &Path) -> Result<DatasetFormat> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .context("No file extension")?;

        match ext.to_lowercase().as_str() {
            "csv" | "tsv" => Ok(DatasetFormat::Csv {
                arabic_col: "arabic".to_string(),
                english_col: "english".to_string(),
                delimiter: if ext == "tsv" { b'\t' } else { b',' },
            }),
            "json" | "jsonl" => Ok(DatasetFormat::Json {
                arabic_field: "arabic".to_string(),
                english_field: "english".to_string(),
            }),
            "parquet" => Ok(DatasetFormat::Parquet {
                arabic_col: "arabic".to_string(),
                english_col: "english".to_string(),
            }),
            _ => anyhow::bail!("Unsupported format: {}", ext),
        }
    }

    /// Load CSV dataset
    fn load_csv(
        &self,
        path: &Path,
        arabic_col: &str,
        english_col: &str,
        delimiter: u8,
    ) -> Result<Vec<TranslationPair>> {
        let file = std::fs::File::open(path)?;
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_parse_options(
                CsvParseOptions::default()
                    .with_separator(delimiter)
            )
            .into_reader_with_file_handle(file)
            .finish()?;

        self.extract_pairs_from_df(&df, arabic_col, english_col)
    }

    /// Load JSON/JSONL dataset
    fn load_json(
        &self,
        path: &Path,
        arabic_field: &str,
        english_field: &str,
    ) -> Result<Vec<TranslationPair>> {
        let file = std::fs::File::open(path)?;
        let df = JsonReader::new(file).finish()?;
        self.extract_pairs_from_df(&df, arabic_field, english_field)
    }

    /// Load Parquet dataset
    fn load_parquet(
        &self,
        path: &Path,
        arabic_col: &str,
        english_col: &str,
    ) -> Result<Vec<TranslationPair>> {
        // For now, skip parquet - focus on CSV/JSON which are more common
        anyhow::bail!("Parquet support coming soon. Please convert to CSV or JSON for now.")
    }

    /// Load parallel text files
    fn load_plain_text(
        &self,
        arabic_file: &Path,
        english_file: &Path,
    ) -> Result<Vec<TranslationPair>> {
        use std::fs;

        let arabic_lines: Vec<String> = fs::read_to_string(arabic_file)?
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let english_lines: Vec<String> = fs::read_to_string(english_file)?
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if arabic_lines.len() != english_lines.len() {
            warn!(
                "Mismatched line counts: {} Arabic, {} English",
                arabic_lines.len(),
                english_lines.len()
            );
        }

        Ok(arabic_lines
            .into_iter()
            .zip(english_lines)
            .map(|(arabic, english)| TranslationPair {
                arabic,
                english,
                metadata: None,
            })
            .collect())
    }

    /// Extract translation pairs from DataFrame
    fn extract_pairs_from_df(
        &self,
        df: &DataFrame,
        arabic_col: &str,
        english_col: &str,
    ) -> Result<Vec<TranslationPair>> {
        // Try multiple possible column name variations
        let arabic_variations = vec![
            arabic_col,
            "ar",
            "arabic_text",
            "source",
            "src",
            "text_ar",
        ];
        let english_variations = vec![
            english_col,
            "en",
            "english_text",
            "target",
            "tgt",
            "text_en",
            "translation",
        ];

        let arabic_series = arabic_variations
            .iter()
            .find_map(|&col| df.column(col).ok())
            .context("Arabic column not found")?;

        let english_series = english_variations
            .iter()
            .find_map(|&col| df.column(col).ok())
            .context("English column not found")?;

        let arabic_vec = arabic_series.str()?.into_iter().collect::<Vec<_>>();
        let english_vec = english_series.str()?.into_iter().collect::<Vec<_>>();

        Ok(arabic_vec
            .into_iter()
            .zip(english_vec)
            .filter_map(|(ar, en)| {
                ar.and_then(|a| {
                    en.map(|e| TranslationPair {
                        arabic: a.to_string(),
                        english: e.to_string(),
                        metadata: None,
                    })
                })
            })
            .collect())
    }

    /// Filter translation pairs based on configuration
    fn filter_pairs(&self, pairs: Vec<TranslationPair>) -> Result<Vec<TranslationPair>> {
        let filtered: Vec<_> = pairs
            .into_par_iter()
            .filter(|pair| {
                let ar_len = pair.arabic.chars().count();
                let en_len = pair.english.chars().count();

                ar_len >= self.config.min_arabic_len
                    && ar_len <= self.config.max_arabic_len
                    && en_len >= self.config.min_english_len
                    && en_len <= self.config.max_english_len
                    && self.is_valid_arabic(&pair.arabic)
                    && self.is_valid_english(&pair.english)
            })
            .collect();

        let result = if self.config.filter_duplicates {
            self.deduplicate(filtered)
        } else {
            filtered
        };

        Ok(result)
    }

    /// Check if text contains Arabic characters
    fn is_valid_arabic(&self, text: &str) -> bool {
        text.chars()
            .any(|c| ('\u{0600}'..='\u{06FF}').contains(&c))
    }

    /// Check if text contains English characters
    fn is_valid_english(&self, text: &str) -> bool {
        text.chars().any(|c| c.is_ascii_alphabetic())
    }

    /// Remove duplicate pairs
    fn deduplicate(&self, pairs: Vec<TranslationPair>) -> Vec<TranslationPair> {
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        pairs
            .into_iter()
            .filter(|pair| {
                let key = (pair.arabic.clone(), pair.english.clone());
                seen.insert(key)
            })
            .collect()
    }

    /// Split dataset into train/validation
    pub fn split(
        &self,
        pairs: Vec<TranslationPair>,
    ) -> (Vec<TranslationPair>, Vec<TranslationPair>) {
        let split_idx = (pairs.len() as f32 * (1.0 - self.config.validation_split)) as usize;
        let (train, val) = pairs.split_at(split_idx);
        (train.to_vec(), val.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arabic_detection() {
        let loader = DataLoader::new(DataLoaderConfig::default());
        assert!(loader.is_valid_arabic("مرحبا"));
        assert!(!loader.is_valid_arabic("hello"));
    }

    #[test]
    fn test_english_detection() {
        let loader = DataLoader::new(DataLoaderConfig::default());
        assert!(loader.is_valid_english("hello"));
        assert!(!loader.is_valid_english("مرحبا"));
    }
}
