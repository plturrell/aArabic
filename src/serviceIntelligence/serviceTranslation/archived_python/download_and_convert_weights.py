#!/usr/bin/env python3
"""
Download M2M100 weights from HuggingFace and convert to Burn format
"""

import os
import sys
from pathlib import Path
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from safetensors.torch import save_file
import json

def download_model(model_name: str, save_path: str):
    """Download M2M100 model and tokenizer from HuggingFace"""
    print(f"📥 Downloading {model_name} from HuggingFace...")
    print(f"💾 Save path: {save_path}")
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    # Download model
    print("⏳ Downloading model (1.8GB)...")
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    
    # Download tokenizer
    print("⏳ Downloading tokenizer...")
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    
    # Save model in PyTorch format
    print("💾 Saving PyTorch model...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Also save as safetensors for easier loading
    print("💾 Converting to safetensors format...")
    from safetensors.torch import save_model
    safetensors_path = os.path.join(save_path, "model.safetensors")
    save_model(model, safetensors_path)
    
    # Save config in JSON
    config = model.config.to_dict()
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Download complete!")
    print(f"📁 Files saved to {save_path}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    return model, tokenizer

def export_to_onnx(model, tokenizer, output_path: str):
    """Export model to ONNX format for Burn conversion"""
    print("\n🔄 Exporting to ONNX format...")
    
    try:
        import torch.onnx
        
        # Create dummy inputs
        dummy_input_ids = torch.randint(0, 128112, (1, 10))
        dummy_decoder_input_ids = torch.randint(0, 128112, (1, 10))
        
        # Export
        onnx_path = os.path.join(output_path, "model.onnx")
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_decoder_input_ids),
            onnx_path,
            input_names=['input_ids', 'decoder_input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'decoder_input_ids': {0: 'batch', 1: 'sequence'},
                'logits': {0: 'batch', 1: 'sequence'}
            },
            opset_version=14
        )
        
        print(f"✅ ONNX export complete: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}")
        print("   This is optional - you can still use safetensors format")
        return None

def create_weight_mapping(save_path: str):
    """Create mapping file for PyTorch -> Burn weight conversion"""
    print("\n📝 Creating weight mapping file...")
    
    mapping = {
        "info": "M2M100 PyTorch to Burn weight mapping",
        "model": "facebook/m2m100_418M",
        "mappings": {
            "encoder.embed_tokens.weight": "encoder_embedding.embedding.weight",
            "decoder.embed_tokens.weight": "decoder_embedding.embedding.weight",
            # Add more mappings as needed
        }
    }
    
    mapping_path = os.path.join(save_path, "weight_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Mapping saved: {mapping_path}")

def main():
    model_name = "facebook/m2m100_418M"
    save_path = "../../../vendor/layerModels/folderRepos/arabic_models/m2m100-418M"
    
    # Make path absolute
    script_dir = Path(__file__).parent
    save_path = (script_dir / save_path).resolve()
    
    print("🔥 M2M100 Weight Download & Conversion")
    print("=" * 50)
    
    # Download model
    model, tokenizer = download_model(model_name, str(save_path))
    
    # Export to ONNX (optional)
    export_to_onnx(model, tokenizer, str(save_path))
    
    # Create weight mapping
    create_weight_mapping(str(save_path))
    
    print("\n✅ All done!")
    print("\nNext steps:")
    print("1. Use safetensors file for direct loading in Rust")
    print("2. Or use ONNX + burn-import for automatic conversion")
    print(f"\nFiles location: {save_path}")

if __name__ == "__main__":
    main()
