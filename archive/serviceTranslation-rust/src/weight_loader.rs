/// Weight Loader for M2M100 Model
/// Loads PyTorch safetensors weights into Burn model

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::Data;
use std::path::Path;
use std::collections::HashMap;
use tracing::{info, warn, debug};
use safetensors::SafeTensors;

/// Load and convert a single tensor from safetensors to Burn format
fn load_tensor_f32<B: Backend>(
    tensors: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Result<Option<Tensor<B, 2>>> {
    match tensors.tensor(name) {
        Ok(view) => {
            let shape = view.shape();
            debug!("Loading tensor: {} with shape {:?}", name, shape);
            
            // Get raw bytes
            let data_bytes = view.data();
            
            // Convert to f32 based on dtype
            let dtype = view.dtype();
            let floats: Vec<f32> = match dtype {
                safetensors::Dtype::F32 => {
                    // Already f32, just cast
                    data_bytes.chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                }
                safetensors::Dtype::F16 => {
                    // Convert from f16 to f32
                    data_bytes.chunks_exact(2)
                        .map(|chunk| {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect()
                }
                _ => bail!("Unsupported dtype: {:?}", dtype),
            };
            
            // Create Burn tensor
            if shape.len() == 2 {
                // 2D tensor - direct reshape
                let tensor = Tensor::<B, 1>::from_floats(floats.as_slice(), device)
                    .reshape([shape[0], shape[1]]);
                Ok(Some(tensor))
            } else if shape.len() == 1 {
                // 1D tensor (bias) - create as 1D then unsqueeze
                let tensor_1d = Tensor::<B, 1>::from_floats(floats.as_slice(), device);
                let tensor_2d = tensor_1d.unsqueeze_dim(0); // Add batch dimension
                Ok(Some(tensor_2d))
            } else {
                warn!("Skipping tensor {} with unsupported shape: {:?}", name, shape);
                Ok(None)
            }
        }
        Err(_) => {
            debug!("Tensor {} not found", name);
            Ok(None)
        }
    }
}

/// Load embedding weights
pub fn load_embedding_weights<B: Backend>(
    tensors: &SafeTensors,
    prefix: &str,
    device: &B::Device,
) -> Result<Option<Tensor<B, 2>>> {
    let weight_name = format!("{}.weight", prefix);
    load_tensor_f32(tensors, &weight_name, device)
}

/// Load linear layer weights
pub fn load_linear_weights<B: Backend>(
    tensors: &SafeTensors,
    prefix: &str,
    device: &B::Device,
) -> Result<(Option<Tensor<B, 2>>, Option<Tensor<B, 2>>)> {
    let weight_name = format!("{}.weight", prefix);
    let bias_name = format!("{}.bias", prefix);
    
    let weight = load_tensor_f32(tensors, &weight_name, device)?;
    let bias = load_tensor_f32(tensors, &bias_name, device)?;
    
    Ok((weight, bias))
}

/// Load layer norm weights
pub fn load_layer_norm_weights<B: Backend>(
    tensors: &SafeTensors,
    prefix: &str,
    device: &B::Device,
) -> Result<(Option<Tensor<B, 2>>, Option<Tensor<B, 2>>)> {
    let weight_name = format!("{}.weight", prefix);
    let bias_name = format!("{}.bias", prefix);
    
    let weight = load_tensor_f32(tensors, &weight_name, device)?;
    let bias = load_tensor_f32(tensors, &bias_name, device)?;
    
    Ok((weight, bias))
}

/// Create parameter name mapping from PyTorch to our Burn structure
fn get_parameter_mapping() -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    
    // Shared embeddings
    mapping.insert(
        "model.shared.weight".to_string(),
        "shared_embedding".to_string()
    );
    
    // Encoder layers (0-11)
    for i in 0..12 {
        let enc_prefix = format!("model.encoder.layers.{}", i);
        
        // Self attention
        mapping.insert(
            format!("{}.self_attn.q_proj", enc_prefix),
            format!("encoder.layers.{}.self_attn.q_proj", i)
        );
        mapping.insert(
            format!("{}.self_attn.k_proj", enc_prefix),
            format!("encoder.layers.{}.self_attn.k_proj", i)
        );
        mapping.insert(
            format!("{}.self_attn.v_proj", enc_prefix),
            format!("encoder.layers.{}.self_attn.v_proj", i)
        );
        mapping.insert(
            format!("{}.self_attn.out_proj", enc_prefix),
            format!("encoder.layers.{}.self_attn.out_proj", i)
        );
        
        // Feed forward
        mapping.insert(
            format!("{}.fc1", enc_prefix),
            format!("encoder.layers.{}.feed_forward.linear1", i)
        );
        mapping.insert(
            format!("{}.fc2", enc_prefix),
            format!("encoder.layers.{}.feed_forward.linear2", i)
        );
        
        // Layer norms
        mapping.insert(
            format!("{}.self_attn_layer_norm", enc_prefix),
            format!("encoder.layers.{}.norm1", i)
        );
        mapping.insert(
            format!("{}.final_layer_norm", enc_prefix),
            format!("encoder.layers.{}.norm2", i)
        );
    }
    
    // Decoder layers (0-11)
    for i in 0..12 {
        let dec_prefix = format!("model.decoder.layers.{}", i);
        
        // Self attention
        mapping.insert(
            format!("{}.self_attn.q_proj", dec_prefix),
            format!("decoder.layers.{}.self_attn.q_proj", i)
        );
        mapping.insert(
            format!("{}.self_attn.k_proj", dec_prefix),
            format!("decoder.layers.{}.self_attn.k_proj", i)
        );
        mapping.insert(
            format!("{}.self_attn.v_proj", dec_prefix),
            format!("decoder.layers.{}.self_attn.v_proj", i)
        );
        mapping.insert(
            format!("{}.self_attn.out_proj", dec_prefix),
            format!("decoder.layers.{}.self_attn.out_proj", i)
        );
        
        // Cross attention
        mapping.insert(
            format!("{}.encoder_attn.q_proj", dec_prefix),
            format!("decoder.layers.{}.cross_attn.q_proj", i)
        );
        mapping.insert(
            format!("{}.encoder_attn.k_proj", dec_prefix),
            format!("decoder.layers.{}.cross_attn.k_proj", i)
        );
        mapping.insert(
            format!("{}.encoder_attn.v_proj", dec_prefix),
            format!("decoder.layers.{}.cross_attn.v_proj", i)
        );
        mapping.insert(
            format!("{}.encoder_attn.out_proj", dec_prefix),
            format!("decoder.layers.{}.cross_attn.out_proj", i)
        );
        
        // Feed forward
        mapping.insert(
            format!("{}.fc1", dec_prefix),
            format!("decoder.layers.{}.feed_forward.linear1", i)
        );
        mapping.insert(
            format!("{}.fc2", dec_prefix),
            format!("decoder.layers.{}.feed_forward.linear2", i)
        );
        
        // Layer norms
        mapping.insert(
            format!("{}.self_attn_layer_norm", dec_prefix),
            format!("decoder.layers.{}.norm1", i)
        );
        mapping.insert(
            format!("{}.encoder_attn_layer_norm", dec_prefix),
            format!("decoder.layers.{}.norm2", i)
        );
        mapping.insert(
            format!("{}.final_layer_norm", dec_prefix),
            format!("decoder.layers.{}.norm3", i)
        );
    }
    
    // Final layer norms
    mapping.insert(
        "model.encoder.layer_norm".to_string(),
        "encoder.norm".to_string()
    );
    mapping.insert(
        "model.decoder.layer_norm".to_string(),
        "decoder.norm".to_string()
    );
    
    // LM head
    mapping.insert(
        "lm_head".to_string(),
        "lm_head".to_string()
    );
    
    mapping
}

/// Load all model weights from safetensors file
pub fn load_model_weights<B: Backend>(
    safetensors_path: &Path,
    device: &B::Device,
) -> Result<HashMap<String, (Option<Tensor<B, 2>>, Option<Tensor<B, 2>>)>> {
    info!("🔥 Loading M2M100 weights from safetensors");
    info!("   Path: {}", safetensors_path.display());
    
    // Read file
    let buffer = std::fs::read(safetensors_path)
        .context("Failed to read safetensors file")?;
    
    info!("   Size: {:.2} GB", buffer.len() as f64 / 1e9);
    
    // Parse safetensors
    let tensors = SafeTensors::deserialize(&buffer)
        .context("Failed to parse safetensors")?;
    
    info!("   Found {} tensors", tensors.len());
    
    // Load all weights
    let mut weights = HashMap::new();
    let mapping = get_parameter_mapping();
    
    let mut loaded = 0;
    let mut skipped = 0;
    
    for (pytorch_name, burn_name) in &mapping {
        match load_linear_weights(&tensors, pytorch_name, device) {
            Ok((weight, bias)) => {
                if weight.is_some() || bias.is_some() {
                    debug!("Loaded: {} -> {}", pytorch_name, burn_name);
                    weights.insert(burn_name.clone(), (weight, bias));
                    loaded += 1;
                } else {
                    skipped += 1;
                }
            }
            Err(e) => {
                warn!("Failed to load {}: {}", pytorch_name, e);
                skipped += 1;
            }
        }
    }
    
    info!("✅ Loaded {} parameter sets, skipped {}", loaded, skipped);
    
    Ok(weights)
}

/// Simple function to just list what's in the file
pub fn list_model_parameters(safetensors_path: &Path) -> Result<Vec<(String, Vec<usize>)>> {
    let buffer = std::fs::read(safetensors_path)?;
    let tensors = SafeTensors::deserialize(&buffer)?;
    
    let mut params = Vec::new();
    for name in tensors.names() {
        if let Ok(view) = tensors.tensor(&name) {
            params.push((name.clone(), view.shape().to_vec()));
        }
    }
    
    params.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_mapping() {
        let mapping = get_parameter_mapping();
        
        // Test encoder mapping
        assert!(mapping.contains_key("model.encoder.layers.0.self_attn.q_proj"));
        assert_eq!(
            mapping.get("model.encoder.layers.0.self_attn.q_proj").unwrap(),
            "encoder.layers.0.self_attn.q_proj"
        );
        
        // Test decoder mapping
        assert!(mapping.contains_key("model.decoder.layers.5.encoder_attn.k_proj"));
        assert_eq!(
            mapping.get("model.decoder.layers.5.encoder_attn.k_proj").unwrap(),
            "decoder.layers.5.cross_attn.k_proj"
        );
    }
}
