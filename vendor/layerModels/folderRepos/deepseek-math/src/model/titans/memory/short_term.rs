//! Short-term memory buffer for recent interactions.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Entry in short-term memory buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortTermEntry {
    /// Original query/problem
    pub query: String,
    /// Tools that were used
    pub tools_used: Vec<String>,
    /// Outcome/result
    pub outcome: Option<String>,
    /// Confidence of the result
    pub confidence: f64,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Domain classification
    pub domain: String,
}

impl ShortTermEntry {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            tools_used: Vec::new(),
            outcome: None,
            confidence: 0.0,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            domain: "unknown".to_string(),
        }
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools_used = tools;
        self
    }

    pub fn with_outcome(mut self, outcome: impl Into<String>, confidence: f64) -> Self {
        self.outcome = Some(outcome.into());
        self.confidence = confidence;
        self
    }

    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self
    }
}

/// Short-term memory buffer with fixed capacity.
#[derive(Debug)]
pub struct ShortTermBuffer {
    entries: VecDeque<ShortTermEntry>,
    capacity: usize,
}

impl ShortTermBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add an entry, evicting oldest if at capacity.
    pub fn push(&mut self, entry: ShortTermEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Query for similar recent entries.
    pub fn query_similar(&self, query: &str, max_results: usize) -> Vec<&ShortTermEntry> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut matches: Vec<_> = self.entries.iter()
            .filter_map(|entry| {
                let entry_lower = entry.query.to_lowercase();
                let entry_words: Vec<&str> = entry_lower.split_whitespace().collect();
                
                // Simple word overlap score
                let overlap = query_words.iter()
                    .filter(|w| entry_words.contains(w))
                    .count();
                
                if overlap > query_words.len() / 4 {
                    Some((entry, overlap))
                } else {
                    None
                }
            })
            .collect();

        // Sort by overlap (descending)
        matches.sort_by(|a, b| b.1.cmp(&a.1));
        
        matches.into_iter()
            .take(max_results)
            .map(|(entry, _)| entry)
            .collect()
    }

    /// Get entries by domain.
    pub fn by_domain(&self, domain: &str) -> Vec<&ShortTermEntry> {
        self.entries.iter()
            .filter(|e| e.domain == domain)
            .collect()
    }

    /// Get recent successful entries.
    pub fn recent_successes(&self, min_confidence: f64) -> Vec<&ShortTermEntry> {
        self.entries.iter()
            .filter(|e| e.confidence >= min_confidence)
            .collect()
    }

    /// Get all entries.
    pub fn all(&self) -> impl Iterator<Item = &ShortTermEntry> {
        self.entries.iter()
    }

    /// Current size.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
