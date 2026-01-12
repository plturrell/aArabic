//! Memory-backed tool routing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// use crate::tools::ToolRegistry; // Removed dependency

/// Score for a tool based on memory and heuristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolScore {
    pub tool_name: String,
    pub score: f32,
    pub memory_score: f32,
    pub keyword_score: f32,
    pub cost_penalty: f32,
}

impl ToolScore {
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            score: 0.0,
            memory_score: 0.0,
            keyword_score: 0.0,
            cost_penalty: 0.0,
        }
    }

    /// Compute final score.
    pub fn compute_final(&mut self, cost_weight: f32) {
        self.score = self.memory_score + self.keyword_score - (cost_weight * self.cost_penalty);
    }
}

/// Decision from the memory router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Ranked tool scores
    pub tool_scores: Vec<ToolScore>,
    /// Confidence from memory retrieval
    pub memory_confidence: f32,
    /// Surprise signal
    pub surprise: f32,
    /// Domain classification
    pub domain: String,
}

/// Memory-backed router for tool selection.
pub struct MemoryRouter {
    /// Tool success history: tool_name -> (successes, total)
    tool_history: HashMap<String, (u32, u32)>,
    /// Domain -> tool preferences
    domain_preferences: HashMap<String, Vec<(String, f32)>>,
    /// Keyword -> tool mappings
    keyword_tools: HashMap<String, Vec<String>>,
}

impl MemoryRouter {
    pub fn new() -> Self {
        let mut router = Self {
            tool_history: HashMap::new(),
            domain_preferences: HashMap::new(),
            keyword_tools: HashMap::new(),
        };
        router.init_keyword_mappings();
        router
    }

    fn init_keyword_mappings(&mut self) {
        // Search-related keywords
        for kw in ["search", "find", "lookup", "query", "retrieve"] {
            self.keyword_tools.insert(kw.to_string(), vec![
                "gateway_search".to_string(),
                "search_semantic".to_string(),
            ]);
        }

        // Verification keywords
        for kw in ["verify", "prove", "theorem", "lean", "proof", "check"] {
            self.keyword_tools.insert(kw.to_string(), vec![
                "lean4_verify".to_string(),
            ]);
        }

        // Chat/reasoning keywords
        for kw in ["explain", "reason", "think", "analyze", "chat"] {
            self.keyword_tools.insert(kw.to_string(), vec![
                "localai_chat".to_string(),
                "gateway_ai_chat".to_string(),
            ]);
        }

        // Embedding keywords
        for kw in ["embed", "embedding", "vector", "similarity"] {
            self.keyword_tools.insert(kw.to_string(), vec![
                "model2vec_embed".to_string(),
            ]);
        }

        // Graph keywords
        for kw in ["graph", "network", "node", "edge", "spacetime"] {
            self.keyword_tools.insert(kw.to_string(), vec![
                "gnn_spacetime".to_string(),
            ]);
        }
    }

    /// Route a query to tools.
    /// Adapted to take a list of available tools instead of ToolRegistry
    pub fn route(&self, query: &str, available_tools: &[String]) -> RoutingDecision {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();
        let domain = self.detect_domain(&query_lower);

        let mut scores: HashMap<String, ToolScore> = HashMap::new();

        // Initialize scores for all available tools
        for name in available_tools {
            scores.insert(name.to_string(), ToolScore::new(name));
        }

        // Keyword-based scoring
        for word in &words {
            if let Some(tools) = self.keyword_tools.get(*word) {
                for tool in tools {
                    if let Some(score) = scores.get_mut(tool) {
                        score.keyword_score += 0.3;
                    }
                }
            }
        }

        // Domain preference scoring
        if let Some(prefs) = self.domain_preferences.get(&domain) {
            for (tool, pref_score) in prefs {
                if let Some(score) = scores.get_mut(tool) {
                    score.memory_score += pref_score;
                }
            }
        }

        // Historical success rate scoring
        for (tool, (successes, total)) in &self.tool_history {
            if *total > 0 {
                let success_rate = *successes as f32 / *total as f32;
                if let Some(score) = scores.get_mut(tool) {
                    score.memory_score += success_rate * 0.2;
                }
            }
        }

        // Cost penalty - SIMULATED since we don't have registry specs
        // ...

        // Compute final scores
        let mut tool_scores: Vec<ToolScore> = scores.into_values().collect();
        for score in &mut tool_scores {
            score.compute_final(0.2);
        }

        // Sort by score descending
        tool_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let memory_confidence = tool_scores.first()
            .map(|s| s.memory_score.min(1.0))
            .unwrap_or(0.0);

        RoutingDecision {
            tool_scores,
            memory_confidence,
            surprise: 0.5, // Will be computed after execution
            domain,
        }
    }

    /// Update router with execution results.
    pub fn update(&mut self, tool: &str, success: bool, domain: &str) {
        let entry = self.tool_history.entry(tool.to_string()).or_insert((0, 0));
        if success {
            entry.0 += 1;
        }
        entry.1 += 1;

        // Update domain preferences
        let prefs = self.domain_preferences.entry(domain.to_string()).or_default();
        if let Some(pref) = prefs.iter_mut().find(|(t, _)| t == tool) {
            if success {
                pref.1 = (pref.1 + 0.1).min(1.0);
            } else {
                pref.1 = (pref.1 - 0.05).max(0.0);
            }
        } else if success {
            prefs.push((tool.to_string(), 0.5));
        }
    }

    fn detect_domain(&self, query: &str) -> String {
        if query.contains("triangle") || query.contains("angle") || query.contains("circle") {
            "geometry".to_string()
        } else if query.contains("prime") || query.contains("divisible") || query.contains("gcd") {
            "number_theory".to_string()
        } else if query.contains("equation") || query.contains("solve") || query.contains("polynomial") {
            "algebra".to_string()
        } else if query.contains("count") || query.contains("ways") || query.contains("permut") {
            "combinatorics".to_string()
        } else if query.contains("search") || query.contains("find") {
            "search".to_string()
        } else if query.contains("verify") || query.contains("prove") {
            "verification".to_string()
        } else {
            "general".to_string()
        }
    }
}

impl Default for MemoryRouter {
    fn default() -> Self {
        Self::new()
    }
}
