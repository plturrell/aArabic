#!/usr/bin/env python3
"""
Standalone Arabic-English Translator
Loads M2M100 models directly from disk - no services needed!

Perfect for:
- Quick testing
- Development
- Benchmarking without Docker
- Financial/banking translation
"""

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple

# ============================================================================
# Configuration
# ============================================================================

MODELS_DIR = Path("../../../vendor/layerModels/folderRepos/arabic_models")

print("🚀 Standalone Arabic-English Translator")
print("="*60)

# ============================================================================
# Standalone Translator
# ============================================================================

class StandaloneTranslator:
    """Direct M2M100 translation without services"""
    
    def __init__(self, model_size: str = "418M"):
        """
        Initialize translator
        
        Args:
            model_size: "418M" (fast) or "1.2B" (quality)
        """
        self.model_size = model_size
        self.model_path = MODELS_DIR / f"m2m100-{model_size}"
        
        print(f"\n📦 Loading M2M100-{model_size} from disk...")
        print(f"   Path: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float32  # Use FP32 for CPU
        )
        
        # Load tokenizer
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            str(self.model_path)
        )
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Parameters: ~{model_size}")
        print(f"   Memory: ~{self._get_model_size_mb():.0f} MB")
    
    def _get_model_size_mb(self) -> float:
        """Estimate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def translate(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
        max_length: int = 256
    ) -> Tuple[str, float]:
        """
        Translate text
        
        Returns:
            (translated_text, time_ms)
        """
        start_time = time.time()
        
        # Set source language
        self.tokenizer.src_lang = source_lang
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
                max_length=max_length,
                num_beams=5,  # Beam search for quality
                early_stopping=True
            )
        
        # Decode
        translated = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return translated, elapsed_ms
    
    def batch_translate(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Dict]:
        """Translate multiple texts"""
        results = []
        
        print(f"\n🌐 Translating {len(texts)} texts...")
        
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}/{len(texts)}] {text[:50]}...")
            
            translated, time_ms = self.translate(text, **kwargs)
            
            result = {
                "source": text,
                "translation": translated,
                "time_ms": time_ms,
                "model": f"m2m100-{self.model_size}"
            }
            
            results.append(result)
            
            print(f"   → {translated}")
            print(f"   Time: {time_ms:.0f}ms")
        
        return results

# ============================================================================
# Quick Benchmark
# ============================================================================

def run_quick_benchmark(translator: StandaloneTranslator, test_pairs: List[Tuple[str, str]]):
    """Run quick benchmark on test pairs"""
    print("\n" + "="*60)
    print(" QUICK BENCHMARK ".center(60))
    print("="*60)
    
    results = []
    total_time = 0
    
    for i, (arabic, expected) in enumerate(test_pairs, 1):
        print(f"\n[{i}/{len(test_pairs)}] {arabic}")
        
        translated, time_ms = translator.translate(arabic)
        total_time += time_ms
        
        # Simple accuracy check
        matches = sum(1 for word in expected.lower().split() if word in translated.lower())
        accuracy = matches / len(expected.split()) if expected.split() else 0
        
        result = {
            "arabic": arabic,
            "expected": expected,
            "translated": translated,
            "time_ms": time_ms,
            "accuracy": accuracy
        }
        results.append(result)
        
        print(f"   Expected:   {expected}")
        print(f"   Translated: {translated}")
        print(f"   Accuracy:   {accuracy:.1%}")
        print(f"   Time:       {time_ms:.0f}ms")
    
    # Summary
    avg_time = total_time / len(test_pairs)
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    
    print("\n" + "="*60)
    print(" BENCHMARK RESULTS ".center(60))
    print("="*60)
    print(f"\n   Total texts:     {len(test_pairs)}")
    print(f"   Avg time:        {avg_time:.0f}ms")
    print(f"   Avg accuracy:    {avg_accuracy:.1%}")
    print(f"   Total time:      {total_time/1000:.1f}s")
    print(f"   Throughput:      {len(test_pairs)/(total_time/1000):.1f} texts/sec")
    
    # Save results
    output_file = Path("benchmarks/standalone_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": f"m2m100-{translator.model_size}",
            "test_pairs": len(test_pairs),
            "avg_time_ms": avg_time,
            "avg_accuracy": avg_accuracy,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Standalone Arabic-English Translator (No Services Required)"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Arabic text to translate"
    )
    parser.add_argument(
        "--model",
        choices=["418M", "1.2B"],
        default="418M",
        help="Model size (default: 418M for speed)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on test pairs"
    )
    parser.add_argument(
        "--file",
        help="File with Arabic texts (one per line)"
    )
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = StandaloneTranslator(model_size=args.model)
    
    if args.benchmark:
        # Run benchmark
        test_pairs = [
            ("الفاتورة رقم ١٢٣٤", "Invoice number 1234"),
            ("المبلغ الإجمالي ١٠٠٠ ريال سعودي", "Total amount 1000 Saudi Riyals"),
            ("تاريخ الاستحقاق", "Due date"),
            ("الرقم الضريبي", "Tax identification number"),
            ("اسم الشركة", "Company name"),
            ("البنك الوطني", "National Bank"),
            ("رقم الحساب", "Account number"),
            ("ضريبة القيمة المضافة", "Value Added Tax"),
            ("شروط الدفع", "Payment terms"),
            ("العنوان", "Address"),
        ]
        
        run_quick_benchmark(translator, test_pairs)
    
    elif args.file:
        # Translate from file
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = translator.batch_translate(texts)
        
        # Save results
        output_file = args.file.replace('.txt', '_translated.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Translations saved to {output_file}")
    
    elif args.text:
        # Single translation
        print(f"\n🌐 Translating: {args.text}")
        translated, time_ms = translator.translate(args.text)
        print(f"\n   → {translated}")
        print(f"   Time: {time_ms:.0f}ms")
    
    else:
        # Interactive mode
        print("\n=== Interactive Translation ===")
        print("Enter Arabic text (Ctrl+D to finish):\n")
        
        import sys
        text = sys.stdin.read().strip()
        
        if text:
            translated, time_ms = translator.translate(text)
            print(f"\nTranslation: {translated}")
            print(f"Time: {time_ms:.0f}ms")

if __name__ == "__main__":
    main()
