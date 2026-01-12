use anyhow::Result;
use git2::{Repository, Signature, IndexAddOption, BranchType, Oid};
use std::path::Path;

pub struct GitClient {
    repo_path: String,
}

impl GitClient {
    pub fn new(path: impl Into<String>) -> Self {
        Self { repo_path: path.into() }
    }

    pub fn init(&self) -> Result<()> {
        Repository::init(&self.repo_path)?;
        Ok(())
    }

    pub fn clone(&self, url: &str) -> Result<()> {
        Repository::clone(url, &self.repo_path)?;
        Ok(())
    }

    pub fn add(&self, patterns: &[&str]) -> Result<()> {
        let repo = Repository::open(&self.repo_path)?;
        let mut index = repo.index()?;
        index.add_all(patterns.iter(), IndexAddOption::DEFAULT, None)?;
        index.write()?;
        Ok(())
    }

    pub fn commit(&self, message: &str) -> Result<String> {
        let repo = Repository::open(&self.repo_path)?;
        let sig = Signature::now("User", "user@example.com")?;
        let tree_id = repo.index()?.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let parent = repo.head()?.peel_to_commit()?;
        let oid = repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &[&parent])?;
        Ok(oid.to_string())
    }

    pub fn push(&self, remote: &str, branch: &str) -> Result<()> {
        let repo = Repository::open(&self.repo_path)?;
        let mut remote = repo.find_remote(remote)?;
        remote.push(&[&format!("refs/heads/{}", branch)], None)?;
        Ok(())
    }

    pub fn pull(&self, remote: &str, branch: &str) -> Result<()> {
        let repo = Repository::open(&self.repo_path)?;
        let mut remote = repo.find_remote(remote)?;
        remote.fetch(&[branch], None, None)?;
        Ok(())
    }

    pub fn create_branch(&self, name: &str) -> Result<()> {
        let repo = Repository::open(&self.repo_path)?;
        let head = repo.head()?.peel_to_commit()?;
        repo.branch(name, &head, false)?;
        Ok(())
    }

    pub fn list_branches(&self) -> Result<Vec<String>> {
        let repo = Repository::open(&self.repo_path)?;
        let branches = repo.branches(Some(BranchType::Local))?;
        Ok(branches.filter_map(|b| b.ok().and_then(|(b, _)| b.name().ok().flatten().map(String::from))).collect())
    }

    pub fn status(&self) -> Result<Vec<String>> {
        let repo = Repository::open(&self.repo_path)?;
        let statuses = repo.statuses(None)?;
        Ok(statuses.iter().filter_map(|s| s.path().map(String::from)).collect())
    }
}
