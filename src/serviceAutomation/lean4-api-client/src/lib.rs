use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

pub struct Lean4Client {
    base_url: String,
    client: Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRequest {
    pub code: String,
    pub timeout: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResponse {
    pub success: bool,
    pub messages: Vec<String>,
    pub errors: Vec<String>,
}

impl Lean4Client {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    fn request(&self, method: &str, endpoint: &str) -> reqwest::blocking::RequestBuilder {
        let url = format!("{}/{}", self.base_url, endpoint.trim_start_matches('/'));
        match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            _ => self.client.get(&url),
        }
    }

    pub fn check_proof(&self, code: &str) -> Result<ProofResponse> {
        let req = ProofRequest {
            code: code.to_string(),
            timeout: Some(30),
        };
        let response = self.request("POST", "check").json(&req).send()?;
        Ok(response.json()?)
    }

    pub fn verify(&self, code: &str) -> Result<bool> {
        let result = self.check_proof(code)?;
        Ok(result.success)
    }

    pub fn health_check(&self) -> Result<bool> {
        let response = self.request("GET", "health").send()?;
        Ok(response.status().is_success())
    }
}
