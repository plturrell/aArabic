use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Filesystem API Client
pub struct FilesystemClient {
    base_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub size: u64,
    pub is_dir: bool,
}

impl FilesystemClient {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self { base_path: base_path.into() }
    }

    pub fn read(&self, path: &str) -> Result<String> {
        Ok(fs::read_to_string(self.base_path.join(path))?)
    }

    pub fn write(&self, path: &str, content: &str) -> Result<()> {
        let full_path = self.base_path.join(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(full_path, content)?;
        Ok(())
    }

    pub fn delete(&self, path: &str) -> Result<()> {
        let full_path = self.base_path.join(path);
        if full_path.is_dir() {
            fs::remove_dir_all(full_path)?;
        } else {
            fs::remove_file(full_path)?;
        }
        Ok(())
    }

    pub fn list(&self, path: &str) -> Result<Vec<FileInfo>> {
        let full_path = self.base_path.join(path);
        let mut files = Vec::new();
        
        for entry in fs::read_dir(full_path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            files.push(FileInfo {
                path: entry.path().display().to_string(),
                size: metadata.len(),
                is_dir: metadata.is_dir(),
            });
        }
        Ok(files)
    }

    pub fn copy(&self, from: &str, to: &str) -> Result<()> {
        fs::copy(self.base_path.join(from), self.base_path.join(to))?;
        Ok(())
    }

    pub fn rename(&self, from: &str, to: &str) -> Result<()> {
        fs::rename(self.base_path.join(from), self.base_path.join(to))?;
        Ok(())
    }

    pub fn exists(&self, path: &str) -> bool {
        self.base_path.join(path).exists()
    }

    pub fn create_dir(&self, path: &str) -> Result<()> {
        fs::create_dir_all(self.base_path.join(path))?;
        Ok(())
    }
}
