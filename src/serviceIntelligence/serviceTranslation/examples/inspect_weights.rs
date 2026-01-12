/// Inspect M2M100 safetensors weights
/// Shows tensor names and shapes to understand the model structure

use anyhow::Result;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;

fn main() -> Result<()> {
    println!("🔍 Inspecting M2M100 Model Weights");
    println!("═══════════════════════════════════\n");

    let model_path = "../../vendor/layerModels/folderRepos/arabic_models/m2m100-418M/model.safetensors";
    
    println!("📂 Loading: {}", model_path);
    
    // Read file
    let mut file = File::open(model_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    println!("✅ File loaded: {:.2} MB\n", buffer.len() as f64 / 1_000_000.0);
    
    // Parse safetensors
    let tensors = SafeTensors::deserialize(&buffer)?;
    
    println!("📊 Model Structure:");
    println!("─────────────────────────────────────────────────────────────");
    
    let mut names = tensors.names();
    names.sort();
    
    // Group by component
    let mut encoder_params = 0;
    let mut decoder_params = 0;
    let mut embedding_params = 0;
    let mut other_params = 0;
    
    for name in &names {
        let tensor_view = tensors.tensor(name)?;
        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();
        
        let param_count: usize = shape.iter().product();
        
        // Categorize
        if name.starts_with("model.encoder") {
            encoder_params += param_count;
        } else if name.starts_with("model.decoder") {
            decoder_params += param_count;
        } else if name.contains("embed") {
            embedding_params += param_count;
        } else {
            other_params += param_count;
        }
        
        println!("{:<60} {:?} {:>15} params", name, shape, format_number(param_count));
    }
    
    println!("─────────────────────────────────────────────────────────────");
    println!("\n📈 Parameter Breakdown:");
    println!("   Encoder:    {:>15} params ({:.1}%)", 
        format_number(encoder_params),
        encoder_params as f64 / (encoder_params + decoder_params + embedding_params + other_params) as f64 * 100.0
    );
    println!("   Decoder:    {:>15} params ({:.1}%)", 
        format_number(decoder_params),
        decoder_params as f64 / (encoder_params + decoder_params + embedding_params + other_params) as f64 * 100.0
    );
    println!("   Embeddings: {:>15} params ({:.1}%)", 
        format_number(embedding_params),
        embedding_params as f64 / (encoder_params + decoder_params + embedding_params + other_params) as f64 * 100.0
    );
    println!("   Other:      {:>15} params ({:.1}%)", 
        format_number(other_params),
        other_params as f64 / (encoder_params + decoder_params + embedding_params + other_params) as f64 * 100.0
    );
    
    let total = encoder_params + decoder_params + embedding_params + other_params;
    println!("   ─────────────────────────────────");
    println!("   TOTAL:      {:>15} params\n", format_number(total));
    
    println!("✅ Inspection complete!");
    println!("\n💡 Next steps:");
    println!("   1. Map these parameter names to Burn model structure");
    println!("   2. Load weights into Burn tensors");
    println!("   3. Test forward pass");
    
    Ok(())
}

fn format_number(n: usize) -> String {
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
