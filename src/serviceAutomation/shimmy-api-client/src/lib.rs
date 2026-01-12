use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// WebSocket streaming module
pub mod websocket;
pub use websocket::stream_chat;

/// Complete Shimmy-AI API Client (OpenAI-compatible local inference)
pub struct ShimmyClient {
    base_url: String,
    client: Client,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl ShimmyClient {
    /// Create new Shimmy client
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    /// Helper to build request
    fn request(&self, method: &str, endpoint: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}/{}", self.base_url, endpoint.trim_start_matches('/'));
        match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            "DELETE" => self.client.delete(&url),
            _ => self.client.get(&url),
        }
    }

    // ========================================================================
    // HEALTH & STATUS
    // ========================================================================

    /// Health check
    pub fn health_check(&self) -> Result<bool> {
        let response = self.request("GET", "/health").send()?;
        Ok(response.status().is_success())
    }

    // ========================================================================
    // MODEL OPERATIONS (OpenAI-compatible)
    // ========================================================================

    /// List available models
    pub fn list_models(&self) -> Result<Vec<Model>> {
        let response = self.request("GET", "/v1/models").send()?;
        let models_response: ModelsResponse = response.json()?;
        Ok(models_response.data)
    }

    /// Get model details
    pub fn get_model(&self, model_id: &str) -> Result<Model> {
        let response = self.request("GET", &format!("/v1/models/{}", model_id)).send()?;
        Ok(response.json()?)
    }

    // ========================================================================
    // CHAT COMPLETIONS (OpenAI-compatible)
    // ========================================================================

    /// Create chat completion
    pub fn chat_completion(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let response = self.request("POST", "/v1/chat/completions")
            .json(request)
            .send()?;
        Ok(response.json()?)
    }

    /// Simple chat helper
    pub fn chat(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
    ) -> Result<String> {
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            temperature,
            max_tokens: None,
            stream: Some(false),
        };
        
        let response = self.chat_completion(&request)?;
        
        if let Some(choice) = response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(anyhow::anyhow!("No response from model"))
        }
    }

    // ========================================================================
    // NATIVE GENERATE API
    // ========================================================================

    /// Generate completion (Shimmy native API)
    pub fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let response = self.request("POST", "/api/generate")
            .json(request)
            .send()?;
        Ok(response.json()?)
    }

    /// Simple generate helper
    pub fn generate_simple(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Result<String> {
        let request = GenerateRequest {
            prompt: prompt.to_string(),
            model: model.map(|s| s.to_string()),
            temperature: None,
            max_tokens: None,
        };
        
        let response = self.generate(&request)?;
        Ok(response.response)
    }

    // ========================================================================
    // UTILITY OPERATIONS
    // ========================================================================

    /// Get server info
    pub fn get_server_info(&self) -> Result<HashMap<String, serde_json::Value>> {
        let response = self.request("GET", "/").send()?;
        Ok(response.json()?)
    }

    /// Check if model exists
    pub fn model_exists(&self, model_id: &str) -> Result<bool> {
        match self.get_model(model_id) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

// ============================================================================
// BUILDER PATTERN FOR CHAT REQUESTS
// ============================================================================

pub struct ChatRequestBuilder {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

impl ChatRequestBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
        }
    }

    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "system".to_string(),
            content: content.into(),
        });
        self
    }

    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: content.into(),
        });
        self
    }

    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: content.into(),
        });
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: i32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn build(self) -> ChatRequest {
        ChatRequest {
            model: self.model,
            messages: self.messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: Some(false),
        }
    }
}
