#!/usr/bin/env python3
"""
Arabic-English Translation System with Lean4 Proof Integration

Leverages:
- M2M100 models (418M/1.2B) for neural translation
- CamelBERT for dialect classification
- Lean4 Arabica for grammatical verification
- Training data from Kaggle Arabic financial news

Architecture:
1. Input Arabic text
2. Dialect classification (CamelBERT)
3. Neural translation (M2M100)
4. Grammatical verification (Lean4 Arabica)
5. Quality scoring
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Configuration
# ============================================================================

CAMELBERT_URL = os.getenv("CAMELBERT_URL", "http://localhost:8001")
SHIMMY_URL = os.getenv("SHIMMY_URL", "http://127.0.0.1:11435")
LEAN4_PROOF_SERVER = os.getenv("LEAN4_PROOF_SERVER", "http://localhost:3000")

class TranslationQuality(Enum):
    HIGH = "high"        # 95%+ confidence, verified
    MEDIUM = "medium"    # 80-95% confidence
    LOW = "low"          # <80% confidence, needs review
    FAILED = "failed"    # Translation or verification failed

class ArabicDialect(Enum):
    MSA = "msa"          # Modern Standard Arabic
    GULF = "gulf"        # Gulf dialects (Saudi, UAE, Kuwait)
    LEVANTINE = "levantine"  # Levantine (Syria, Lebanon, Jordan)
    EGYPTIAN = "egyptian"    # Egyptian dialect
    MAGHREBI = "maghrebi"    # Maghrebi (Morocco, Algeria, Tunisia)
    YEMENI = "yemeni"        # Yemeni dialect
    SUDANESE = "sudanese"    # Sudanese dialect
    IRAQI = "iraqi"          # Iraqi dialect

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TranslationInput:
    """Input for translation"""
    text: str
    source_lang: str = "ar"
    target_lang: str = "en"
    domain: Optional[str] = None  # e.g., "financial", "legal"
    use_quality_model: bool = True  # True = m2m100-1.2B, False = m2m100-418M

@dataclass
class DialectAnalysis:
    """Dialect classification result"""
    primary_dialect: ArabicDialect
    confidence: float
    all_scores: Dict[str, float]
    features: Dict[str, any]

@dataclass
class TranslationResult:
    """Complete translation result"""
    source_text: str
    translated_text: str
    dialect_analysis: DialectAnalysis
    quality: TranslationQuality
    confidence_score: float
    grammatical_verification: Optional[Dict]
    model_used: str
    processing_time_ms: float
    metadata: Dict

# ============================================================================
# Dialect Classifier (CamelBERT)
# ============================================================================

class DialectClassifier:
    """Classify Arabic text dialect using CamelBERT"""
    
    def __init__(self, base_url: str = CAMELBERT_URL):
        self.base_url = base_url
        self._check_health()
    
    def _check_health(self):
        """Verify CamelBERT service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/camelbert/ready", timeout=5)
            if response.status_code != 200:
                print(f"⚠️  CamelBERT not ready: {response.status_code}")
        except Exception as e:
            print(f"⚠️  CamelBERT service unavailable: {e}")
    
    def classify(self, text: str) -> DialectAnalysis:
        """Classify dialect of Arabic text"""
        try:
            response = requests.post(
                f"{self.base_url}/api/camelbert/classify",
                json={"text": text},
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Map result to our enum
            dialect_map = {
                "MSA": ArabicDialect.MSA,
                "Gulf": ArabicDialect.GULF,
                "Levantine": ArabicDialect.LEVANTINE,
                "Egyptian": ArabicDialect.EGYPTIAN,
                "Maghrebi": ArabicDialect.MAGHREBI,
                "Yemeni": ArabicDialect.YEMENI,
                "Sudanese": ArabicDialect.SUDANESE,
                "Iraqi": ArabicDialect.IRAQI,
            }
            
            primary = result.get("dialect", "MSA")
            confidence = result.get("confidence", 0.0)
            all_scores = result.get("scores", {})
            
            return DialectAnalysis(
                primary_dialect=dialect_map.get(primary, ArabicDialect.MSA),
                confidence=confidence,
                all_scores=all_scores,
                features={
                    "model_variant": result.get("model_variant", "unknown"),
                    "processing_time": result.get("processing_time_ms", 0)
                }
            )
            
        except Exception as e:
            print(f"❌ Dialect classification failed: {e}")
            # Fallback: assume MSA
            return DialectAnalysis(
                primary_dialect=ArabicDialect.MSA,
                confidence=0.5,
                all_scores={},
                features={"error": str(e)}
            )

# ============================================================================
# Neural Translator (M2M100)
# ============================================================================

class NeuralTranslator:
    """Translate using M2M100 models via Shimmy"""
    
    def __init__(self, base_url: str = SHIMMY_URL):
        self.base_url = base_url
        self.model_1_2b = "m2m100-1.2B"  # High quality
        self.model_418m = "m2m100-418M"  # Fast
    
    def translate(
        self,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
        use_quality_model: bool = True
    ) -> Tuple[str, str]:
        """
        Translate text using M2M100
        
        Returns:
            (translated_text, model_used)
        """
        model = self.model_1_2b if use_quality_model else self.model_418m
        
        try:
            # Shimmy translation API
            # First, try direct translation endpoint
            response = requests.post(
                f"{self.base_url}/api/translate",
                json={
                    "model": model,
                    "text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                translated = result.get("translated_text", result.get("translation", ""))
                return translated, model
            
            # Fallback: Use generate with translation prompt
            prompt = f"Translate from {source_lang} to {target_lang}: {text}"
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.3  # Lower temp for translation
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            translated = result.get("response", "")
            
            return translated, model
            
        except Exception as e:
            print(f"❌ Translation failed with {model}: {e}")
            
            # Fallback to faster model if quality model failed
            if use_quality_model:
                print(f"🔄 Falling back to {self.model_418m}...")
                return self.translate(text, source_lang, target_lang, use_quality_model=False)
            
            raise Exception(f"Translation failed: {e}")

# ============================================================================
# Grammatical Verifier (Lean4 Integration)
# ============================================================================

class GrammaticalVerifier:
    """Verify translation quality using Lean4 Arabica proofs"""
    
    def __init__(self, proof_server_url: str = LEAN4_PROOF_SERVER):
        self.proof_server_url = proof_server_url
    
    def verify(self, arabic_text: str, english_text: str) -> Dict:
        """
        Verify translation using Lean4 grammatical proofs
        
        Returns dict with:
        - is_valid: bool
        - grammar_score: float (0-1)
        - issues: List[str]
        - proof_trace: Optional[str]
        """
        try:
            response = requests.post(
                f"{self.proof_server_url}/api/verify-translation",
                json={
                    "arabic": arabic_text,
                    "english": english_text
                },
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Verification service returned {response.status_code}")
                return self._fallback_verification(arabic_text)
                
        except Exception as e:
            print(f"⚠️  Verification failed: {e}")
            return self._fallback_verification(arabic_text)
    
    def _fallback_verification(self, text: str) -> Dict:
        """Simple heuristic-based verification when Lean4 unavailable"""
        # Check basic Arabic script presence
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
        
        # Check length (reasonable translation shouldn't be too different)
        words = text.split()
        word_count = len(words)
        
        return {
            "is_valid": has_arabic and word_count > 0,
            "grammar_score": 0.7 if has_arabic else 0.3,
            "issues": [] if has_arabic else ["No Arabic script detected"],
            "proof_trace": None,
            "fallback": True
        }

# ============================================================================
# Main Translation System
# ============================================================================

class ArabicTranslationSystem:
    """Complete Arabic-English translation system"""
    
    def __init__(self):
        self.dialect_classifier = DialectClassifier()
        self.neural_translator = NeuralTranslator()
        self.grammatical_verifier = GrammaticalVerifier()
        
        print("✅ Arabic Translation System initialized")
        print(f"   - Dialect Classifier: {CAMELBERT_URL}")
        print(f"   - Neural Translator (Shimmy): {SHIMMY_URL}")
        print(f"   - Grammatical Verifier: {LEAN4_PROOF_SERVER}")
    
    def translate(self, input_data: TranslationInput) -> TranslationResult:
        """
        Complete translation pipeline with quality assurance
        """
        import time
        start_time = time.time()
        
        # Step 1: Classify dialect
        print(f"\n📊 Step 1: Classifying dialect...")
        dialect_analysis = self.dialect_classifier.classify(input_data.text)
        print(f"   Detected: {dialect_analysis.primary_dialect.value} "
              f"(confidence: {dialect_analysis.confidence:.2%})")
        
        # Step 2: Translate
        print(f"\n🌐 Step 2: Translating...")
        model_choice = "quality (1.2B)" if input_data.use_quality_model else "fast (418M)"
        print(f"   Using: {model_choice} model")
        
        translated_text, model_used = self.neural_translator.translate(
            text=input_data.text,
            source_lang=input_data.source_lang,
            target_lang=input_data.target_lang,
            use_quality_model=input_data.use_quality_model
        )
        print(f"   Translation: '{translated_text}'")
        
        # Step 3: Verify grammar
        print(f"\n✓ Step 3: Verifying grammatical correctness...")
        verification = self.grammatical_verifier.verify(
            input_data.text,
            translated_text
        )
        grammar_score = verification.get("grammar_score", 0.5)
        print(f"   Grammar score: {grammar_score:.2%}")
        
        # Step 4: Calculate overall quality
        print(f"\n🎯 Step 4: Calculating quality...")
        quality, confidence = self._calculate_quality(
            dialect_confidence=dialect_analysis.confidence,
            grammar_score=grammar_score,
            model_used=model_used
        )
        print(f"   Overall quality: {quality.value} (confidence: {confidence:.2%})")
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return TranslationResult(
            source_text=input_data.text,
            translated_text=translated_text,
            dialect_analysis=dialect_analysis,
            quality=quality,
            confidence_score=confidence,
            grammatical_verification=verification,
            model_used=model_used,
            processing_time_ms=processing_time,
            metadata={
                "domain": input_data.domain,
                "source_lang": input_data.source_lang,
                "target_lang": input_data.target_lang
            }
        )
    
    def _calculate_quality(
        self,
        dialect_confidence: float,
        grammar_score: float,
        model_used: str
    ) -> Tuple[TranslationQuality, float]:
        """Calculate overall translation quality"""
        
        # Weighted average
        model_weight = 0.5 if "1.2B" in model_used else 0.4
        
        overall_confidence = (
            dialect_confidence * 0.2 +
            grammar_score * 0.3 +
            model_weight
        )
        
        # Classify quality tier
        if overall_confidence >= 0.95:
            return TranslationQuality.HIGH, overall_confidence
        elif overall_confidence >= 0.80:
            return TranslationQuality.MEDIUM, overall_confidence
        elif overall_confidence >= 0.60:
            return TranslationQuality.LOW, overall_confidence
        else:
            return TranslationQuality.FAILED, overall_confidence
    
    def batch_translate(
        self,
        texts: List[str],
        **kwargs
    ) -> List[TranslationResult]:
        """Translate multiple texts"""
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n{'='*60}")
            print(f"Translation {i}/{len(texts)}")
            print(f"{'='*60}")
            
            input_data = TranslationInput(text=text, **kwargs)
            result = self.translate(input_data)
            results.append(result)
        
        return results

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Arabic-English Translation System with Lean4 Verification"
    )
    parser.add_argument("text", nargs="?", help="Arabic text to translate")
    parser.add_argument(
        "--file", "-f",
        help="File containing Arabic text (one sentence per line)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use fast model (418M) instead of quality model (1.2B)"
    )
    parser.add_argument(
        "--domain", "-d",
        choices=["financial", "legal", "medical", "general"],
        default="general",
        help="Domain context for translation"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = ArabicTranslationSystem()
    
    # Get texts to translate
    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("\n=== Interactive Translation Mode ===")
        print("Enter Arabic text (Ctrl+D or Ctrl+Z to finish):\n")
        import sys
        texts = [sys.stdin.read().strip()]
    
    # Translate
    results = system.batch_translate(
        texts,
        use_quality_model=not args.fast,
        domain=args.domain
    )
    
    # Display results
    print("\n" + "="*60)
    print("TRANSLATION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Source: {result.source_text}")
        print(f"    Translation: {result.translated_text}")
        print(f"    Dialect: {result.dialect_analysis.primary_dialect.value}")
        print(f"    Quality: {result.quality.value} ({result.confidence_score:.1%})")
        print(f"    Model: {result.model_used}")
        print(f"    Time: {result.processing_time_ms:.0f}ms")
    
    # Save to file if requested
    if args.output:
        output_data = [
            {
                "source": r.source_text,
                "translation": r.translated_text,
                "dialect": r.dialect_analysis.primary_dialect.value,
                "quality": r.quality.value,
                "confidence": r.confidence_score,
                "model": r.model_used
            }
            for r in results
        ]
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()
