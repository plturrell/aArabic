#!/usr/bin/env python3
"""
Quick benchmark runner - works with actual models

This script:
1. Checks model availability
2. Creates test data
3. Runs benchmarks
4. Reports results
"""

import sys
from pathlib import Path
import json

print("🔍 Checking system setup...")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Check model directory
MODELS_DIR = Path("../../../vendor/layerModels/folderRepos/arabic_models")
print(f"\n📁 Checking models directory: {MODELS_DIR}")

if MODELS_DIR.exists():
    print("✅ Models directory found")
    models = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
    print(f"   Available models: {', '.join(models)}")
else:
    print(f"⚠️  Models directory not found at {MODELS_DIR}")
    print("   Using fallback mode (metrics only, no translation)")

# Create test data
print("\n📝 Creating test dataset...")

test_pairs = [
    ("الفاتورة رقم ١٢٣٤", "Invoice number 1234"),
    ("المبلغ الإجمالي ١٠٠٠ ريال سعودي", "Total amount 1000 Saudi Riyals"),
    ("تاريخ الاستحقاق ٣١ يناير ٢٠٢٥", "Due date January 31, 2025"),
    ("الرقم الضريبي للمورد", "Supplier tax identification number"),
    ("اسم الشركة: شركة التقنية المتقدمة", "Company name: Advanced Technology Company"),
    ("البنك: البنك الوطني", "Bank: National Bank"),
    ("رقم الحساب البنكي", "Bank account number"),
    ("ضريبة القيمة المضافة", "Value Added Tax"),
    ("شروط الدفع: ثلاثون يوماً", "Payment terms: thirty days"),
    ("العنوان: الرياض، المملكة العربية السعودية", "Address: Riyadh, Saudi Arabia"),
]

print(f"✅ Created {len(test_pairs)} test pairs")

# Create CSV for benchmark
test_file = Path("data/translation_training/test_set.csv")
test_file.parent.mkdir(parents=True, exist_ok=True)

import csv
with open(test_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['arabic', 'english', 'dialect', 'domain'])
    for arabic, english in test_pairs:
        # Infer domain
        domain = 'financial' if any(term in arabic for term in ['فاتورة', 'ضريبة', 'ريال']) else 'general'
        writer.writerow([arabic, english, 'msa', domain])

print(f"✅ Saved test set to {test_file}")

# Check if we can actually run translation
print("\n🔧 Checking translation capabilities...")

try:
    # Try importing translation system
    from translation_system import ArabicTranslationSystem, TranslationInput
    
    print("✅ Translation system imported")
    
    # Try to initialize (this will check services)
    print("\n🚀 Initializing translation system...")
    print("   (This will check CamelBERT, LocalAI, and Lean4 services)")
    print("   Note: Services may not be running - that's OK for metrics testing")
    
    system = ArabicTranslationSystem()
    print("✅ System initialized")
    
    CAN_TRANSLATE = True
    
except Exception as e:
    print(f"⚠️  Translation system unavailable: {e}")
    print("   Will run metrics-only mode")
    CAN_TRANSLATE = False

# Check benchmark system
print("\n📊 Checking benchmark system...")

try:
    from metrics_benchmarks import (
        TranslationBenchmarkSuite,
        TranslationMetricsCalculator,
        BenchmarkVisualizer
    )
    print("✅ Benchmark system imported")
    
    # Initialize
    suite = TranslationBenchmarkSuite()
    print("✅ Benchmark suite initialized")
    
    metrics_calc = TranslationMetricsCalculator()
    print("✅ Metrics calculator initialized")
    
    CAN_BENCHMARK = True
    
except Exception as e:
    print(f"❌ Benchmark system error: {e}")
    CAN_BENCHMARK = False
    sys.exit(1)

# Decision point
print("\n" + "="*70)
print(" SYSTEM STATUS ".center(70))
print("="*70)
print(f"\n   Translation: {'✅ Ready' if CAN_TRANSLATE else '⚠️  Unavailable (services not running)'}")
print(f"   Benchmarking: {'✅ Ready' if CAN_BENCHMARK else '❌ Failed'}")
print(f"   Test Data: ✅ Ready ({len(test_pairs)} pairs)")
print(f"   Models: {'✅ Found' if MODELS_DIR.exists() else '⚠️  Not found'}")

if not CAN_BENCHMARK:
    print("\n❌ Cannot run benchmarks - fix errors above")
    sys.exit(1)

# Run appropriate mode
print("\n" + "="*70)

if CAN_TRANSLATE:
    print(" RUNNING FULL BENCHMARK (Translation + Metrics) ".center(70))
    print("="*70)
    
    print("\n⚠️  Note: This requires CamelBERT and LocalAI services")
    print("   If services are not running, benchmark will use fallback mode")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        # Run full benchmark
        result = suite.run_benchmark(
            test_pairs=test_pairs,
            benchmark_name="production_benchmark"
        )
        
        print("\n✅ Benchmark completed successfully!")
        print(f"\n📊 Results: BLEU={result.avg_bleu:.1f}, Confidence={result.avg_confidence:.1%}")
        
        # Try to visualize
        try:
            print("\n📊 Generating visualizations...")
            viz = BenchmarkVisualizer()
            viz.plot_quality_distribution(result)
            viz.plot_model_comparison(result)
            print("✅ Visualizations saved")
        except Exception as e:
            print(f"⚠️  Visualization failed (optional): {e}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

else:
    print(" RUNNING METRICS-ONLY MODE ".center(70))
    print("="*70)
    
    print("\n📊 Computing metrics on test translations...")
    print("   (Using reference translations as 'system output' for demo)")
    
    all_metrics = []
    
    for i, (arabic, english) in enumerate(test_pairs, 1):
        print(f"\n[{i}/{len(test_pairs)}] {arabic[:40]}...")
        
        # Calculate metrics (using reference as both hypothesis and reference)
        metrics = metrics_calc.calculate_all_metrics(
            source_arabic=arabic,
            translated_english=english,  # Using reference as translation
            reference_english=english,
            dialect_confidence=0.9,
            grammar_score=0.85,
            model_used="reference",
            dialect_detected="msa",
            overall_quality="high",
            overall_confidence=0.95,
            translation_time_ms=0
        )
        
        all_metrics.append(metrics)
        
        print(f"   BLEU: {metrics.bleu_score:.1f}")
        print(f"   Financial Accuracy: {metrics.financial_term_accuracy:.1%}")
    
    # Calculate averages
    avg_bleu = sum(m.bleu_score for m in all_metrics) / len(all_metrics)
    avg_meteor = sum(m.meteor_score for m in all_metrics) / len(all_metrics)
    avg_financial = sum(m.financial_term_accuracy for m in all_metrics) / len(all_metrics)
    
    print("\n" + "="*70)
    print(" METRICS-ONLY RESULTS ".center(70))
    print("="*70)
    print(f"\n   Average BLEU: {avg_bleu:.1f}")
    print(f"   Average METEOR: {avg_meteor:.3f}")
    print(f"   Average Financial Accuracy: {avg_financial:.1%}")
    
    # Save results
    results_file = Path("benchmarks/translation/metrics_only_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'metrics_only',
            'test_pairs': len(test_pairs),
            'avg_bleu': avg_bleu,
            'avg_meteor': avg_meteor,
            'avg_financial_accuracy': avg_financial,
            'note': 'Reference translations used as system output (demo mode)'
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")

print("\n" + "="*70)
print(" COMPLETE ".center(70))
print("="*70)

print("\n📝 Next Steps:")
print("   1. To run with real models, start CamelBERT and LocalAI services")
print("   2. To add more test data, edit data/translation_training/test_set.csv")
print("   3. To view results, check benchmarks/translation/ directory")
print("   4. To visualize, run with --visualize flag")

print("\n✅ Benchmark system is ready!")
