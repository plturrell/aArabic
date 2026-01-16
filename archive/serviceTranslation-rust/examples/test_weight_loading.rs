/// Test Weight Loading from Safetensors
/// Verifies that we can successfully load M2M100 weights into Burn tensors

use anyhow::Result;
use arabic_translation_trainer::weight_loader::load_model_weights;
use burn::backend::NdArray;
use std::path::PathBuf;
use tracing::{info, Level};

type Backend = NdArray;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🔥 Testing M2M100 Weight Loading");
    info!("══════════════════════════════════\n");

    let model_path = PathBuf::from("../../vendor/layerModels/folderRepos/arabic_models/m2m100-418M/model.safetensors");
    
    info!("📂 Model path: {}", model_path.display());
    
    if !model_path.exists() {
        eprintln!("❌ Model file not found!");
        eprintln!("   Expected: {}", model_path.display());
        return Ok(());
    }
    
    // Initialize device
    let device = Default::default();
    info!("🖥️  Device: NdArray (CPU)\n");
    
    // Load weights
    info!("⏳ Loading weights...");
    let weights = load_model_weights::<Backend>(&model_path, &device)?;
    
    info!("\n📊 Weight Loading Results:");
    info!("══════════════════════════════════");
    info!("   Total parameter sets loaded: {}", weights.len());
    
    // Analyze what we loaded
    let mut with_weights = 0;
    let mut with_bias = 0;
    let mut total_params = 0;
    
    for (name, (weight, bias)) in &weights {
        if let Some(w) = weight {
            with_weights += 1;
            let shape = w.dims();
            let param_count = shape.iter().product::<usize>();
            total_params += param_count;
            
            if name.contains("layers.0") {
                // Show first layer details
                info!("   ✓ {}: shape {:?} ({} params)", 
                    name, shape, format_params(param_count));
            }
        }
        if bias.is_some() {
            with_bias += 1;
        }
    }
    
    info!("\n📈 Summary:");
    info!("   Weight tensors:  {}", with_weights);
    info!("   Bias tensors:    {}", with_bias);
    info!("   Total params:    {}", format_params(total_params));
    info!("   Expected params: ~484M");
    
    let coverage = (total_params as f64 / 484_000_000.0) * 100.0;
    info!("   Coverage:        {:.1}%", coverage);
    
    if coverage > 90.0 {
        info!("\n✅ SUCCESS! Weight loading is working!");
        info!("   Ready to use for inference.");
    } else {
        info!("\n⚠️  Partial loading - some weights missing");
        info!("   May need to adjust parameter mapping.");
    }
    
    info!("\n💡 Next steps:");
    info!("   1. Integrate weight loading into model initialization");
    info!("   2. Test forward pass with loaded weights");
    info!("   3. Verify translation accuracy");
    
    Ok(())
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
