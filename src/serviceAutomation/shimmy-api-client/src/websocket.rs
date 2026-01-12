// WebSocket streaming implementation for Shimmy
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use url::Url;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub content: String,
    pub done: bool,
}

/// Stream chat completion via WebSocket
pub async fn stream_chat(
    base_url: &str,
    model: &str,
    prompt: &str,
    temperature: Option<f32>,
) -> Result<Vec<String>> {
    let ws_url = base_url
        .replace("http://", "ws://")
        .replace("https://", "wss://");
    let url = Url::parse(&format!("{}/ws/generate", ws_url))?;
    
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();
    
    // Send request
    let request = StreamRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        temperature,
        max_tokens: None,
    };
    
    let msg = Message::Text(serde_json::to_string(&request)?);
    write.send(msg).await?;
    
    // Collect chunks
    let mut chunks = Vec::new();
    
    while let Some(msg) = read.next().await {
        match msg? {
            Message::Text(text) => {
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(&text) {
                    if !chunk.content.is_empty() {
                        chunks.push(chunk.content);
                    }
                    if chunk.done {
                        break;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    
    Ok(chunks)
}
