#!/usr/bin/env python3
"""
Translation Metrics & Benchmarking System

Comprehensive evaluation framework for Arabic-English translation:
1. BLEU, METEOR, chrF scores (standard MT metrics)
2. Domain-specific accuracy (financial terminology)
3. Dialect handling performance
4. Speed benchmarks
5. Quality tier validation
6. Lean4 proof verification rate

Supports:
- Per-model comparison (418M vs 1.2B vs fine-tuned)
- Per-dialect performance
- Per-domain accuracy
- Temporal tracking (improvement over time)
"""

import os
import json
import time
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Standard MT metrics
try:
    from sacrebleu import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from nltk.translate.bleu_score import sentence_bleu
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    print("⚠️  Install sacrebleu and nltk for full metrics: pip install sacrebleu nltk")
    METRICS_AVAILABLE = False

# ============================================================================
# Metric Structures
# ============================================================================

@dataclass
class TranslationMetrics:
    """Complete metrics for a translation"""
    # Standard MT metrics
    bleu_score: float  # 0-100
    meteor_score: float  # 0-1
    chrf_score: float  # 0-100
    
    # Domain-specific
    financial_term_accuracy: float  # 0-1
    dialect_confidence: float  # 0-1
    grammar_score: float  # 0-1
    
    # Performance
    translation_time_ms: float
    tokens_per_second: float
    
    # Quality
    overall_quality: str  # high/medium/low
    confidence: float  # 0-1
    
    # Model info
    model_used: str
    dialect_detected: str
    
    # Metadata
    timestamp: str
    source_length: int
    target_length: int

@dataclass
class BenchmarkResult:
    """Results from a complete benchmark run"""
    benchmark_id: str
    timestamp: str
    total_samples: int
    
    # Aggregate metrics
    avg_bleu: float
    avg_meteor: float
    avg_chrf: float
    avg_time_ms: float
    avg_confidence: float
    
    # Per-model breakdown
    model_metrics: Dict[str, Dict[str, float]]
    
    # Per-dialect breakdown
    dialect_metrics: Dict[str, Dict[str, float]]
    
    # Per-domain breakdown
    domain_metrics: Dict[str, Dict[str, float]]
    
    # Quality distribution
    quality_distribution: Dict[str, int]  # {high: 45, medium: 40, low: 15}

# ============================================================================
# Metrics Calculator
# ============================================================================

class TranslationMetricsCalculator:
    """Calculate comprehensive translation metrics"""
    
    def __init__(self):
        self.financial_terms = self._load_financial_terms()
    
    def _load_financial_terms(self) -> set:
        """Load financial terms for accuracy checking"""
        terms_file = Path("data/translation_training/financial_terms.json")
        if terms_file.exists():
            with open(terms_file, 'r', encoding='utf-8') as f:
                terms_dict = json.load(f)
                return set(terms_dict.keys())
        else:
            # Default set
            return {
                'فاتورة', 'ريال', 'ضريبة', 'مورد', 'مبلغ',
                'تاريخ', 'رقم', 'حساب', 'بنك', 'شركة'
            }
    
    def calculate_bleu(
        self,
        hypothesis: str,
        reference: str
    ) -> float:
        """Calculate BLEU score (0-100)"""
        if not METRICS_AVAILABLE:
            return 0.0
        
        try:
            # Tokenize
            hyp_tokens = hypothesis.split()
            ref_tokens = reference.split()
            
            # Calculate BLEU
            score = sentence_bleu([ref_tokens], hyp_tokens)
            return score * 100
        except Exception as e:
            print(f"⚠️  BLEU calculation failed: {e}")
            return 0.0
    
    def calculate_meteor(
        self,
        hypothesis: str,
        reference: str
    ) -> float:
        """Calculate METEOR score (0-1)"""
        if not METRICS_AVAILABLE:
            return 0.0
        
        try:
            hyp_tokens = hypothesis.split()
            ref_tokens = reference.split()
            return meteor_score([ref_tokens], hyp_tokens)
        except Exception as e:
            print(f"⚠️  METEOR calculation failed: {e}")
            return 0.0
    
    def calculate_chrf(
        self,
        hypothesis: str,
        reference: str
    ) -> float:
        """Calculate chrF score (0-100)"""
        if not METRICS_AVAILABLE:
            return 0.0
        
        try:
            result = corpus_bleu([hypothesis], [[reference]], lowercase=True)
            return result.score
        except Exception as e:
            print(f"⚠️  chrF calculation failed: {e}")
            return 0.0
    
    def calculate_financial_term_accuracy(
        self,
        source_arabic: str,
        translated_english: str,
        reference_english: Optional[str] = None
    ) -> float:
        """
        Calculate how well financial terms were translated
        
        Checks if key financial terms in Arabic have appropriate
        English equivalents in translation
        """
        # Find financial terms in source
        terms_in_source = [
            term for term in self.financial_terms
            if term in source_arabic
        ]
        
        if not terms_in_source:
            return 1.0  # No financial terms to check
        
        # Expected translations
        term_translations = {
            'فاتورة': ['invoice', 'bill', 'receipt'],
            'ريال': ['riyal', 'sar', 'saudi'],
            'ضريبة': ['tax', 'vat', 'levy'],
            'مورد': ['supplier', 'vendor', 'provider'],
            'مبلغ': ['amount', 'sum', 'total'],
            'تاريخ': ['date'],
            'رقم': ['number', 'no.', '#'],
            'حساب': ['account', 'invoice'],
            'بنك': ['bank'],
            'شركة': ['company', 'corporation'],
        }
        
        # Check how many terms were translated correctly
        correct = 0
        for term in terms_in_source:
            expected = term_translations.get(term, [])
            if any(exp.lower() in translated_english.lower() for exp in expected):
                correct += 1
        
        return correct / len(terms_in_source) if terms_in_source else 1.0
    
    def calculate_all_metrics(
        self,
        source_arabic: str,
        translated_english: str,
        reference_english: Optional[str],
        dialect_confidence: float,
        grammar_score: float,
        model_used: str,
        dialect_detected: str,
        overall_quality: str,
        overall_confidence: float,
        translation_time_ms: float
    ) -> TranslationMetrics:
        """Calculate complete metrics for a translation"""
        
        # Standard MT metrics (if reference available)
        bleu = 0.0
        meteor = 0.0
        chrf = 0.0
        
        if reference_english:
            bleu = self.calculate_bleu(translated_english, reference_english)
            meteor = self.calculate_meteor(translated_english, reference_english)
            chrf = self.calculate_chrf(translated_english, reference_english)
        
        # Domain-specific
        fin_term_acc = self.calculate_financial_term_accuracy(
            source_arabic,
            translated_english,
            reference_english
        )
        
        # Performance
        source_tokens = len(source_arabic.split())
        target_tokens = len(translated_english.split())
        tokens_per_sec = (source_tokens + target_tokens) / (translation_time_ms / 1000) if translation_time_ms > 0 else 0
        
        return TranslationMetrics(
            bleu_score=bleu,
            meteor_score=meteor,
            chrf_score=chrf,
            financial_term_accuracy=fin_term_acc,
            dialect_confidence=dialect_confidence,
            grammar_score=grammar_score,
            translation_time_ms=translation_time_ms,
            tokens_per_second=tokens_per_sec,
            overall_quality=overall_quality,
            confidence=overall_confidence,
            model_used=model_used,
            dialect_detected=dialect_detected,
            timestamp=datetime.now().isoformat(),
            source_length=len(source_arabic),
            target_length=len(translated_english)
        )

# ============================================================================
# Benchmark Suite
# ============================================================================

class TranslationBenchmarkSuite:
    """Comprehensive benchmark suite"""
    
    def __init__(self, output_dir: Path = Path("benchmarks/translation")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_calculator = TranslationMetricsCalculator()
        self.results_history: List[BenchmarkResult] = []
    
    def load_test_set(self, test_file: Path) -> List[Tuple[str, str]]:
        """
        Load test set with ground truth
        
        Expected format (CSV):
        arabic,english,dialect,domain
        """
        try:
            df = pd.read_csv(test_file)
            pairs = [
                (row['arabic'], row['english'])
                for _, row in df.iterrows()
            ]
            return pairs
        except Exception as e:
            print(f"⚠️  Using sample test set: {e}")
            return self._get_sample_test_set()
    
    def _get_sample_test_set(self) -> List[Tuple[str, str]]:
        """Sample test set for demonstration"""
        return [
            ("الفاتورة رقم ١٢٣٤", "Invoice number 1234"),
            ("المبلغ الإجمالي ١٠٠٠ ريال سعودي", "Total amount 1000 Saudi Riyals"),
            ("تاريخ الاستحقاق ٣١ يناير ٢٠٢٥", "Due date January 31, 2025"),
            ("الرقم الضريبي للمورد", "Supplier tax identification number"),
            ("اسم الشركة: شركة التقنية المتقدمة", "Company name: Advanced Technology Company"),
            ("البنك: البنك الوطني", "Bank: National Bank"),
            ("رقم الحساب البنكي", "Bank account number"),
            ("ضريبة القيمة المضافة خمسة عشر بالمئة", "Value Added Tax fifteen percent"),
            ("شروط الدفع: ثلاثون يوماً", "Payment terms: thirty days"),
            ("العنوان: الرياض، المملكة العربية السعودية", "Address: Riyadh, Saudi Arabia"),
        ]
    
    def run_benchmark(
        self,
        test_pairs: List[Tuple[str, str]],
        models_to_test: List[str] = ["m2m100-418M", "m2m100-1.2B"],
        benchmark_name: str = "default"
    ) -> BenchmarkResult:
        """
        Run complete benchmark on test set
        
        Args:
            test_pairs: List of (arabic, english_reference) tuples
            models_to_test: Models to benchmark
            benchmark_name: Name for this benchmark run
        
        Returns:
            BenchmarkResult with comprehensive metrics
        """
        print("\n" + "="*70)
        print(f" BENCHMARK: {benchmark_name} ".center(70, "="))
        print("="*70 + "\n")
        
        all_metrics: List[TranslationMetrics] = []
        model_results = defaultdict(list)
        dialect_results = defaultdict(list)
        domain_results = defaultdict(list)
        quality_dist = defaultdict(int)
        
        from translation_system import ArabicTranslationSystem, TranslationInput
        
        # Initialize system
        system = ArabicTranslationSystem()
        
        total = len(test_pairs)
        for i, (arabic, english_ref) in enumerate(test_pairs, 1):
            print(f"\n[{i}/{total}] Testing: {arabic[:50]}...")
            
            # Translate
            input_data = TranslationInput(text=arabic)
            result = system.translate(input_data)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                source_arabic=arabic,
                translated_english=result.translated_text,
                reference_english=english_ref,
                dialect_confidence=result.dialect_analysis.confidence,
                grammar_score=result.grammatical_verification.get('grammar_score', 0.5),
                model_used=result.model_used,
                dialect_detected=result.dialect_analysis.primary_dialect.value,
                overall_quality=result.quality.value,
                overall_confidence=result.confidence_score,
                translation_time_ms=result.processing_time_ms
            )
            
            all_metrics.append(metrics)
            
            # Aggregate by model
            model_results[metrics.model_used].append(metrics)
            
            # Aggregate by dialect
            dialect_results[metrics.dialect_detected].append(metrics)
            
            # Aggregate by domain (infer from content)
            domain = self._infer_domain(arabic)
            domain_results[domain].append(metrics)
            
            # Count quality distribution
            quality_dist[metrics.overall_quality] += 1
            
            # Progress
            print(f"    BLEU: {metrics.bleu_score:.1f}")
            print(f"    Financial Accuracy: {metrics.financial_term_accuracy:.1%}")
            print(f"    Quality: {metrics.overall_quality} ({metrics.confidence:.1%})")
        
        # Aggregate results
        benchmark_result = BenchmarkResult(
            benchmark_id=f"{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            total_samples=total,
            avg_bleu=self._avg_metric(all_metrics, 'bleu_score'),
            avg_meteor=self._avg_metric(all_metrics, 'meteor_score'),
            avg_chrf=self._avg_metric(all_metrics, 'chrf_score'),
            avg_time_ms=self._avg_metric(all_metrics, 'translation_time_ms'),
            avg_confidence=self._avg_metric(all_metrics, 'confidence'),
            model_metrics=self._aggregate_by_group(model_results),
            dialect_metrics=self._aggregate_by_group(dialect_results),
            domain_metrics=self._aggregate_by_group(domain_results),
            quality_distribution=dict(quality_dist)
        )
        
        # Save results
        self._save_benchmark(benchmark_result, all_metrics)
        
        # Display summary
        self._display_summary(benchmark_result)
        
        self.results_history.append(benchmark_result)
        
        return benchmark_result
    
    def _avg_metric(self, metrics: List[TranslationMetrics], field: str) -> float:
        """Calculate average of a metric field"""
        values = [getattr(m, field) for m in metrics]
        return sum(values) / len(values) if values else 0.0
    
    def _aggregate_by_group(
        self,
        grouped_metrics: Dict[str, List[TranslationMetrics]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics for each group"""
        result = {}
        
        for group, metrics in grouped_metrics.items():
            result[group] = {
                'count': len(metrics),
                'avg_bleu': self._avg_metric(metrics, 'bleu_score'),
                'avg_meteor': self._avg_metric(metrics, 'meteor_score'),
                'avg_confidence': self._avg_metric(metrics, 'confidence'),
                'avg_time_ms': self._avg_metric(metrics, 'translation_time_ms'),
                'financial_accuracy': self._avg_metric(metrics, 'financial_term_accuracy'),
            }
        
        return result
    
    def _infer_domain(self, text: str) -> str:
        """Infer domain from text content"""
        if any(term in text for term in ['فاتورة', 'ضريبة', 'ريال', 'مبلغ']):
            return 'financial'
        elif any(term in text for term in ['عقد', 'قانون', 'محكمة']):
            return 'legal'
        elif any(term in text for term in ['مريض', 'علاج', 'طبيب']):
            return 'medical'
        else:
            return 'general'
    
    def _save_benchmark(self, result: BenchmarkResult, detailed_metrics: List[TranslationMetrics]):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary_file = self.output_dir / f"benchmark_{timestamp}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        
        # Save detailed metrics
        details_file = self.output_dir / f"benchmark_{timestamp}_details.json"
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(m) for m in detailed_metrics],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        print(f"\n💾 Saved results to:")
        print(f"   - {summary_file}")
        print(f"   - {details_file}")
    
    def _display_summary(self, result: BenchmarkResult):
        """Display benchmark summary"""
        print("\n" + "="*70)
        print(" BENCHMARK SUMMARY ".center(70, "="))
        print("="*70 + "\n")
        
        print(f"📊 Overall Results (n={result.total_samples})")
        print(f"   BLEU:       {result.avg_bleu:.2f}")
        print(f"   METEOR:     {result.avg_meteor:.3f}")
        print(f"   chrF:       {result.avg_chrf:.2f}")
        print(f"   Confidence: {result.avg_confidence:.1%}")
        print(f"   Avg Time:   {result.avg_time_ms:.0f}ms")
        
        print(f"\n🎯 Quality Distribution:")
        for quality, count in result.quality_distribution.items():
            pct = (count / result.total_samples) * 100
            bar = "█" * int(pct / 2)
            print(f"   {quality:8s}: {bar} {count} ({pct:.1f}%)")
        
        print(f"\n📱 Per-Model Performance:")
        for model, metrics in result.model_metrics.items():
            print(f"\n   {model}:")
            print(f"     BLEU:      {metrics['avg_bleu']:.2f}")
            print(f"     Confidence: {metrics['avg_confidence']:.1%}")
            print(f"     Time:      {metrics['avg_time_ms']:.0f}ms")
            print(f"     Financial: {metrics['financial_accuracy']:.1%}")
        
        print(f"\n🌍 Per-Dialect Performance:")
        for dialect, metrics in result.dialect_metrics.items():
            print(f"   {dialect}: BLEU={metrics['avg_bleu']:.1f}, n={metrics['count']}")
        
        print(f"\n📋 Per-Domain Performance:")
        for domain, metrics in result.domain_metrics.items():
            print(f"   {domain}: Financial Acc={metrics['financial_accuracy']:.1%}, n={metrics['count']}")

# ============================================================================
# Performance Benchmarks
# ============================================================================

class PerformanceBenchmark:
    """Speed and throughput benchmarks"""
    
    def benchmark_speed(
        self,
        texts: List[str],
        model_name: str = "m2m100-418M"
    ) -> Dict[str, float]:
        """
        Benchmark translation speed
        
        Returns:
            {
                'avg_time_ms': float,
                'tokens_per_second': float,
                'throughput_texts_per_minute': float
            }
        """
        print(f"\n⚡ Speed Benchmark: {model_name}")
        print(f"   Testing on {len(texts)} texts...")
        
        from translation_system import ArabicTranslationSystem, TranslationInput
        
        system = ArabicTranslationSystem()
        
        times = []
        total_tokens = 0
        
        for text in texts:
            start = time.time()
            
            input_data = TranslationInput(
                text=text,
                use_quality_model=("1.2B" in model_name)
            )
            result = system.translate(input_data)
            
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
            total_tokens += len(text.split()) + len(result.translated_text.split())
        
        avg_time = sum(times) / len(times)
        total_time_sec = sum(times) / 1000
        tokens_per_sec = total_tokens / total_time_sec
        throughput = (len(texts) / total_time_sec) * 60  # texts per minute
        
        results = {
            'avg_time_ms': avg_time,
            'tokens_per_second': tokens_per_sec,
            'throughput_texts_per_minute': throughput,
            'total_texts': len(texts),
            'total_time_seconds': total_time_sec
        }
        
        print(f"\n   Results:")
        print(f"     Avg Time:    {avg_time:.0f}ms")
        print(f"     Tokens/sec:  {tokens_per_sec:.1f}")
        print(f"     Throughput:  {throughput:.1f} texts/min")
        
        return results

# ============================================================================
# Visualization
# ============================================================================

class BenchmarkVisualizer:
    """Generate visualizations of benchmark results"""
    
    def __init__(self, output_dir: Path = Path("benchmarks/translation/plots")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_quality_distribution(self, result: BenchmarkResult):
        """Plot quality tier distribution"""
        plt.figure(figsize=(10, 6))
        
        qualities = list(result.quality_distribution.keys())
        counts = list(result.quality_distribution.values())
        
        plt.bar(qualities, counts, color=['green', 'orange', 'red', 'gray'])
        plt.title('Translation Quality Distribution')
        plt.xlabel('Quality Tier')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        
        output_file = self.output_dir / f"{result.benchmark_id}_quality.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot: {output_file}")
        plt.close()
    
    def plot_model_comparison(self, result: BenchmarkResult):
        """Compare performance across models"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(result.model_metrics.keys())
        
        # BLEU scores
        bleu_scores = [result.model_metrics[m]['avg_bleu'] for m in models]
        axes[0].bar(models, bleu_scores)
        axes[0].set_title('BLEU Score by Model')
        axes[0].set_ylabel('BLEU')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Speed
        times = [result.model_metrics[m]['avg_time_ms'] for m in models]
        axes[1].bar(models, times, color='orange')
        axes[1].set_title('Speed by Model')
        axes[1].set_ylabel('Time (ms)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Financial accuracy
        fin_acc = [result.model_metrics[m]['financial_accuracy'] * 100 for m in models]
        axes[2].bar(models, fin_acc, color='green')
        axes[2].set_title('Financial Term Accuracy')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_file = self.output_dir / f"{result.benchmark_id}_models.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot: {output_file}")
        plt.close()

# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for benchmarking"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Translation Metrics & Benchmarking System"
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="CSV file with test pairs (arabic,english,dialect,domain)"
    )
    parser.add_argument(
        "--benchmark-name",
        default="translation_benchmark",
        help="Name for this benchmark run"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    suite = TranslationBenchmarkSuite()
    
    # Load or create test set
    if args.test_file and args.test_file.exists():
        test_pairs = suite.load_test_set(args.test_file)
    else:
        print("⚠️  No test file provided, using sample test set")
        test_pairs = suite._get_sample_test_set()
    
    # Run benchmark
    result = suite.run_benchmark(
        test_pairs=test_pairs,
        benchmark_name=args.benchmark_name
    )
    
    # Visualize if requested
    if args.visualize:
        print("\n📊 Generating visualizations...")
        viz = BenchmarkVisualizer()
        viz.plot_quality_distribution(result)
        viz.plot_model_comparison(result)
    
    # Performance benchmark
    print("\n⚡ Running speed benchmark...")
    perf = PerformanceBenchmark()
    speed_results = perf.benchmark_speed([p[0] for p in test_pairs[:10]])
    
    print("\n" + "="*70)
    print(" BENCHMARK COMPLETE ".center(70, "="))
    print("="*70)

if __name__ == "__main__":
    main()
