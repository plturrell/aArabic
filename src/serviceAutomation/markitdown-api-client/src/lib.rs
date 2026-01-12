use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::thread;
use std::time::Duration;

/// MarkItDown API Client for Replicate
/// Convert documents to Markdown using AI
pub struct MarkItDownClient {
    client: Client,
    api_token: String,
    model_version: String,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub id: String,
    pub status: String,
    pub output: Option<Value>,
    pub error: Option<String>,
    pub logs: Option<String>,
    pub metrics: Option<PredictionMetrics>,
    pub urls: Option<PredictionUrls>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetrics {
    pub predict_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionUrls {
    pub get: String,
    pub cancel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionRequest {
    pub file_url: Option<String>,
    pub file_data: Option<String>, // base64 encoded
    pub output_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionResult {
    pub markdown: String,
    pub metadata: Option<Value>,
    pub prediction_id: String,
}

// ============================================================================
// CLIENT IMPLEMENTATION
// ============================================================================

impl MarkItDownClient {
    /// Create new MarkItDown client
    pub fn new(api_token: String) -> Self {
        Self {
            client: Client::new(),
            api_token,
            model_version: "cuuupid/markitdown".to_string(),
        }
    }

    /// Create client with custom model version
    pub fn with_model_version(api_token: String, model_version: String) -> Self {
        Self {
            client: Client::new(),
            api_token,
            model_version,
        }
    }

    /// Build authenticated request
    fn request(&self, method: reqwest::Method, url: &str) -> reqwest::blocking::RequestBuilder {
        self.client
            .request(method, url)
            .header("Authorization", format!("Token {}", self.api_token))
            .header("Content-Type", "application/json")
    }

    // ========================================================================
    // PREDICTION API
    // ========================================================================

    /// Create a new prediction
    pub fn create_prediction(&self, input: Value) -> Result<Prediction> {
        let url = format!("https://api.replicate.com/v1/predictions");
        
        let payload = json!({
            "version": self.model_version,
            "input": input
        });

        let response = self
            .request(reqwest::Method::POST, &url)
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text()?;
            anyhow::bail!("Failed to create prediction ({}): {}", status, error_text);
        }

        Ok(response.json()?)
    }

    /// Get prediction status
    pub fn get_prediction(&self, prediction_id: &str) -> Result<Prediction> {
        let url = format!("https://api.replicate.com/v1/predictions/{}", prediction_id);

        let response = self
            .request(reqwest::Method::GET, &url)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get prediction: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Wait for prediction to complete
    pub fn wait_for_prediction(&self, prediction_id: &str, max_wait_seconds: u64) -> Result<Prediction> {
        let start = std::time::Instant::now();
        let max_duration = Duration::from_secs(max_wait_seconds);

        loop {
            let prediction = self.get_prediction(prediction_id)?;

            match prediction.status.as_str() {
                "succeeded" => return Ok(prediction),
                "failed" | "canceled" => {
                    anyhow::bail!(
                        "Prediction {} - Error: {:?}",
                        prediction.status,
                        prediction.error
                    );
                }
                _ => {
                    // Still processing
                    if start.elapsed() > max_duration {
                        anyhow::bail!("Prediction timed out after {} seconds", max_wait_seconds);
                    }
                    thread::sleep(Duration::from_secs(2));
                }
            }
        }
    }

    /// Cancel a prediction
    pub fn cancel_prediction(&self, prediction_id: &str) -> Result<Prediction> {
        let url = format!("https://api.replicate.com/v1/predictions/{}/cancel", prediction_id);

        let response = self
            .request(reqwest::Method::POST, &url)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to cancel prediction: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// List predictions
    pub fn list_predictions(&self) -> Result<Vec<Prediction>> {
        let url = "https://api.replicate.com/v1/predictions";

        let response = self
            .request(reqwest::Method::GET, url)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list predictions: {}", response.status());
        }

        let data: Value = response.json()?;
        let predictions: Vec<Prediction> = serde_json::from_value(
            data.get("results").unwrap_or(&json!([])).clone()
        )?;

        Ok(predictions)
    }

    // ========================================================================
    // DOCUMENT CONVERSION
    // ========================================================================

    /// Convert document from URL to Markdown
    pub fn convert_from_url(&self, url: &str) -> Result<ConversionResult> {
        let input = json!({
            "file_url": url
        });

        let prediction = self.create_prediction(input)?;
        let final_prediction = self.wait_for_prediction(&prediction.id, 300)?;

        if let Some(output) = final_prediction.output {
            let markdown = self.extract_markdown(&output)?;
            Ok(ConversionResult {
                markdown,
                metadata: Some(output),
                prediction_id: final_prediction.id,
            })
        } else {
            anyhow::bail!("No output from prediction");
        }
    }

    /// Convert document from file path
    pub fn convert_from_file(&self, file_path: &str) -> Result<ConversionResult> {
        use base64::{Engine as _, engine::general_purpose};
        
        let file_data = std::fs::read(file_path)
            .context("Failed to read file")?;
        
        let encoded = general_purpose::STANDARD.encode(&file_data);
        
        let input = json!({
            "file_data": format!("data:application/octet-stream;base64,{}", encoded)
        });

        let prediction = self.create_prediction(input)?;
        let final_prediction = self.wait_for_prediction(&prediction.id, 300)?;

        if let Some(output) = final_prediction.output {
            let markdown = self.extract_markdown(&output)?;
            Ok(ConversionResult {
                markdown,
                metadata: Some(output),
                prediction_id: final_prediction.id,
            })
        } else {
            anyhow::bail!("No output from prediction");
        }
    }

    /// Convert document with custom options
    pub fn convert_with_options(&self, request: &ConversionRequest) -> Result<ConversionResult> {
        let mut input = json!({});

        if let Some(url) = &request.file_url {
            input["file_url"] = json!(url);
        }

        if let Some(data) = &request.file_data {
            input["file_data"] = json!(data);
        }

        if let Some(format) = &request.output_format {
            input["output_format"] = json!(format);
        }

        let prediction = self.create_prediction(input)?;
        let final_prediction = self.wait_for_prediction(&prediction.id, 300)?;

        if let Some(output) = final_prediction.output {
            let markdown = self.extract_markdown(&output)?;
            Ok(ConversionResult {
                markdown,
                metadata: Some(output),
                prediction_id: final_prediction.id,
            })
        } else {
            anyhow::bail!("No output from prediction");
        }
    }

    /// Convert multiple documents in batch
    pub fn convert_batch(&self, urls: Vec<String>) -> Result<Vec<ConversionResult>> {
        let mut results = Vec::new();

        for url in urls {
            match self.convert_from_url(&url) {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Failed to convert {}: {}", url, e);
                }
            }
        }

        Ok(results)
    }

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    /// Extract markdown from prediction output
    fn extract_markdown(&self, output: &Value) -> Result<String> {
        // Output can be string or object with markdown field
        if let Some(markdown) = output.as_str() {
            Ok(markdown.to_string())
        } else if let Some(markdown) = output.get("markdown").and_then(|v| v.as_str()) {
            Ok(markdown.to_string())
        } else if let Some(text) = output.get("text").and_then(|v| v.as_str()) {
            Ok(text.to_string())
        } else {
            // Return entire output as JSON string
            Ok(serde_json::to_string_pretty(output)?)
        }
    }

    // ========================================================================
    // DOCUMENT TYPE SPECIFIC CONVERSIONS
    // ========================================================================

    /// Convert PDF to Markdown
    pub fn convert_pdf(&self, file_path: &str) -> Result<ConversionResult> {
        self.convert_from_file(file_path)
    }

    /// Convert Word document to Markdown
    pub fn convert_docx(&self, file_path: &str) -> Result<ConversionResult> {
        self.convert_from_file(file_path)
    }

    /// Convert PowerPoint to Markdown
    pub fn convert_pptx(&self, file_path: &str) -> Result<ConversionResult> {
        self.convert_from_file(file_path)
    }

    /// Convert Excel to Markdown
    pub fn convert_xlsx(&self, file_path: &str) -> Result<ConversionResult> {
        self.convert_from_file(file_path)
    }

    /// Convert image to Markdown (OCR)
    pub fn convert_image(&self, file_path: &str) -> Result<ConversionResult> {
        self.convert_from_file(file_path)
    }

    /// Convert HTML to Markdown
    pub fn convert_html(&self, file_path: &str) -> Result<ConversionResult> {
        self.convert_from_file(file_path)
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    /// Get model information
    pub fn get_model_info(&self) -> Result<Value> {
        let url = format!("https://api.replicate.com/v1/models/{}", self.model_version);

        let response = self
            .request(reqwest::Method::GET, &url)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get model info: {}", response.status());
        }

        Ok(response.json()?)
    }

    /// Get account information
    pub fn get_account_info(&self) -> Result<Value> {
        let url = "https://api.replicate.com/v1/account";

        let response = self
            .request(reqwest::Method::GET, url)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get account info: {}", response.status());
        }

        Ok(response.json()?)
    }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Quick conversion from URL
pub fn convert_url_to_markdown(api_token: &str, url: &str) -> Result<String> {
    let client = MarkItDownClient::new(api_token.to_string());
    let result = client.convert_from_url(url)?;
    Ok(result.markdown)
}

/// Quick conversion from file
pub fn convert_file_to_markdown(api_token: &str, file_path: &str) -> Result<String> {
    let client = MarkItDownClient::new(api_token.to_string());
    let result = client.convert_from_file(file_path)?;
    Ok(result.markdown)
}
