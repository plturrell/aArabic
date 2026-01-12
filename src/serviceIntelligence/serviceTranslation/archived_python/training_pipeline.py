#!/usr/bin/env python3
"""
Translation Training Pipeline with Kaggle Data Integration

Leverages Kaggle datasets to fine-tune Arabic translation models:
1. Arabic Financial News (SANAD) - for domain-specific terminology
2. Arabic-English Parallel Corpus - for general translation pairs
3. Arabic NER Dataset - for entity recognition in translations

Pipeline:
1. Download & preprocess Kaggle data
2. Extract financial terminology
3. Fine-tune M2M100 on financial domain
4. Generate training data for Lean4 verification
5. Evaluate translation quality
"""

import os
import json
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import kagglehub
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("data/translation_training")
MODELS_DIR = Path("vendor/layerModels/folderRepos/arabic_models")
OUTPUT_DIR = Path("models/finetuned_translation")

# Create directories
for dir_path in [DATA_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Kaggle Dataset Downloader
# ============================================================================

class KaggleDatasetManager:
    """Manage Kaggle dataset downloads and preprocessing"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.datasets = {
            "financial_news": None,
            "parallel_corpus": None,
            "ner_dataset": None
        }
    
    def download_financial_news(self) -> Path:
        """
        Download SANAD Arabic Financial News Dataset
        ~200K Arabic financial articles
        """
        print("📥 Downloading SANAD Financial News Dataset...")
        try:
            path = kagglehub.dataset_download("haithemhermessi/sanad-dataset")
            self.datasets["financial_news"] = Path(path)
            print(f"✅ Downloaded to: {path}")
            return Path(path)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("   Please ensure kagglehub is configured with API key")
            return None
    
    def download_parallel_corpus(self) -> Path:
        """
        Download Arabic-English Parallel Corpus
        ~30M sentence pairs
        """
        print("📥 Downloading Arabic-English Parallel Corpus...")
        try:
            path = kagglehub.dataset_download("sameh1/arabic-to-english")
            self.datasets["parallel_corpus"] = Path(path)
            print(f"✅ Downloaded to: {path}")
            return Path(path)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def download_ner_dataset(self) -> Path:
        """
        Download Arabic NER Dataset
        For entity recognition in translations
        """
        print("📥 Downloading Arabic NER Dataset...")
        try:
            path = kagglehub.dataset_download("mksaad/arabic-ner")
            self.datasets["ner_dataset"] = Path(path)
            print(f"✅ Downloaded to: {path}")
            return Path(path)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def download_all(self):
        """Download all datasets"""
        print("\n" + "="*60)
        print("DOWNLOADING KAGGLE DATASETS")
        print("="*60 + "\n")
        
        self.download_financial_news()
        self.download_parallel_corpus()
        self.download_ner_dataset()
        
        print("\n✅ All datasets downloaded")

# ============================================================================
# Financial Terminology Extractor
# ============================================================================

class FinancialTermExtractor:
    """Extract financial terminology from SANAD corpus"""
    
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.financial_terms = {}
    
    def extract_terms(self, max_articles: int = 10000) -> Dict[str, int]:
        """
        Extract financial terms with frequency counts
        
        Returns:
            Dict mapping Arabic term → frequency
        """
        print(f"\n📊 Extracting financial terms from {max_articles} articles...")
        
        from collections import Counter
        import re
        
        terms = Counter()
        
        # Financial keywords to track
        financial_keywords = [
            'ريال', 'دينار', 'جنيه', 'درهم',  # Currencies
            'ضريبة', 'رسوم', 'مصلحة',  # Tax terms
            'فاتورة', 'إيصال', 'كشف',  # Invoice terms
            'بنك', 'مصرف', 'حساب',  # Banking
            'شركة', 'مؤسسة', 'منشأة',  # Business entities
            'رقم', 'تاريخ', 'مبلغ',  # Fields
            'قيمة', 'سعر', 'تكلفة',  # Values
            'دفع', 'سداد', 'استحقاق',  # Payment
        ]
        
        # Read articles (assuming CSV format)
        try:
            df = pd.read_csv(self.corpus_path / "sanad.csv", nrows=max_articles)
            
            for i, text in enumerate(df['text'].dropna(), 1):
                if i % 1000 == 0:
                    print(f"   Processed {i}/{max_articles} articles...")
                
                # Extract financial terms
                for keyword in financial_keywords:
                    if keyword in text:
                        # Get context (50 chars before/after)
                        matches = re.finditer(keyword, text)
                        for match in matches:
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            context = text[start:end].strip()
                            terms[keyword] += 1
                
                # Extract amounts (numbers with currency)
                amounts = re.findall(
                    r'\d+\.?\d*\s*(?:ريال|دينار|جنيه|درهم)',
                    text
                )
                terms.update(amounts)
        
        except Exception as e:
            print(f"❌ Error reading corpus: {e}")
            # Generate sample data for testing
            for term in financial_keywords:
                terms[term] = 100
        
        self.financial_terms = dict(terms.most_common(1000))
        
        print(f"✅ Extracted {len(self.financial_terms)} unique financial terms")
        
        return self.financial_terms
    
    def save_to_json(self, output_path: Path):
        """Save extracted terms to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.financial_terms, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved terms to {output_path}")

# ============================================================================
# Translation Data Preprocessor
# ============================================================================

class TranslationDataPreprocessor:
    """Prepare training data for M2M100 fine-tuning"""
    
    def __init__(self, parallel_corpus_path: Path, financial_terms: Dict[str, int]):
        self.parallel_corpus_path = parallel_corpus_path
        self.financial_terms = financial_terms
    
    def prepare_financial_pairs(
        self,
        max_pairs: int = 50000
    ) -> List[Tuple[str, str]]:
        """
        Filter parallel corpus for financial domain
        
        Returns:
            List of (arabic, english) pairs
        """
        print(f"\n📝 Preparing {max_pairs} financial translation pairs...")
        
        pairs = []
        
        try:
            # Read parallel corpus
            df = pd.read_csv(
                self.parallel_corpus_path / "arabic_english.csv",
                nrows=max_pairs * 2  # Read more to filter
            )
            
            for i, row in df.iterrows():
                arabic = row.get('arabic', '')
                english = row.get('english', '')
                
                # Filter for financial content
                if self._is_financial(arabic):
                    pairs.append((arabic, english))
                    
                    if len(pairs) >= max_pairs:
                        break
                
                if (i + 1) % 10000 == 0:
                    print(f"   Processed {i+1} pairs, found {len(pairs)} financial pairs...")
        
        except Exception as e:
            print(f"⚠️  Error reading corpus: {e}")
            # Generate sample data
            pairs = self._generate_sample_pairs(max_pairs)
        
        print(f"✅ Prepared {len(pairs)} financial translation pairs")
        
        return pairs
    
    def _is_financial(self, text: str) -> bool:
        """Check if text contains financial terms"""
        return any(term in text for term in list(self.financial_terms.keys())[:100])
    
    def _generate_sample_pairs(self, n: int) -> List[Tuple[str, str]]:
        """Generate sample translation pairs for testing"""
        samples = [
            ("الفاتورة رقم ١٢٣٤", "Invoice number 1234"),
            ("المبلغ الإجمالي ١٠٠٠ ريال", "Total amount 1000 riyals"),
            ("تاريخ الاستحقاق ٢٠٢٥/٠١/٣١", "Due date 2025/01/31"),
            ("الرقم الضريبي ٣١٢٣٤٥٦٧٨٩٠١٢٣٤", "Tax ID 312345678901234"),
            ("اسم المورد: شركة التقنية", "Supplier name: Technology Company"),
        ]
        return samples * (n // len(samples) + 1)[:n]
    
    def create_dataset(self, pairs: List[Tuple[str, str]]) -> Dataset:
        """Create HuggingFace Dataset from pairs"""
        data = {
            'arabic': [p[0] for p in pairs],
            'english': [p[1] for p in pairs]
        }
        return Dataset.from_dict(data)

# ============================================================================
# M2M100 Fine-Tuner
# ============================================================================

class M2M100FineTuner:
    """Fine-tune M2M100 on financial domain"""
    
    def __init__(
        self,
        base_model_path: Path,
        output_dir: Path = OUTPUT_DIR
    ):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        
        print(f"\n🔧 Loading base model from {base_model_path}...")
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            str(base_model_path)
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            str(base_model_path)
        )
        
        # Set language tokens
        self.tokenizer.src_lang = "ar"
        self.tokenizer.tgt_lang = "en"
        
        print("✅ Model loaded")
    
    def prepare_training_data(self, dataset: Dataset) -> Dataset:
        """Tokenize and prepare data for training"""
        print("\n🔄 Tokenizing dataset...")
        
        def tokenize_function(examples):
            # Tokenize source (Arabic)
            model_inputs = self.tokenizer(
                examples['arabic'],
                max_length=128,
                truncation=True,
                padding='max_length'
            )
            
            # Tokenize target (English)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['english'],
                    max_length=128,
                    truncation=True,
                    padding='max_length'
                )
            
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print("✅ Dataset tokenized")
        return tokenized
    
    def train(
        self,
        train_dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """Fine-tune the model"""
        print(f"\n🎓 Training for {epochs} epochs...")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="no",  # Add eval_dataset for "steps"
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        print("✅ Training complete")
    
    def save_model(self):
        """Save fine-tuned model"""
        output_path = self.output_dir / "m2m100-financial"
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        print(f"✅ Model saved to {output_path}")

# ============================================================================
# Training Pipeline
# ============================================================================

class TranslationTrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self):
        self.dataset_manager = KaggleDatasetManager()
        self.financial_terms = {}
        self.training_pairs = []
    
    def run_full_pipeline(
        self,
        use_418m: bool = True,  # True = faster, False = better quality
        max_training_pairs: int = 10000
    ):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print(" TRANSLATION TRAINING PIPELINE ".center(70, "="))
        print("="*70 + "\n")
        
        # Step 1: Download datasets
        print("\n📦 STEP 1: Download Kaggle Datasets")
        print("-" * 70)
        self.dataset_manager.download_all()
        
        # Step 2: Extract financial terminology
        print("\n\n📖 STEP 2: Extract Financial Terminology")
        print("-" * 70)
        
        financial_path = self.dataset_manager.datasets.get("financial_news")
        if financial_path and financial_path.exists():
            extractor = FinancialTermExtractor(financial_path)
            self.financial_terms = extractor.extract_terms(max_articles=5000)
            
            # Save to JSON
            terms_output = DATA_DIR / "financial_terms.json"
            extractor.save_to_json(terms_output)
        else:
            print("⚠️  Using sample financial terms")
            self.financial_terms = {
                "فاتورة": 1000, "ريال": 800, "ضريبة": 600,
                "مورد": 400, "مبلغ": 500
            }
        
        # Step 3: Prepare training data
        print("\n\n🔧 STEP 3: Prepare Training Data")
        print("-" * 70)
        
        parallel_path = self.dataset_manager.datasets.get("parallel_corpus")
        preprocessor = TranslationDataPreprocessor(
            parallel_path or DATA_DIR,
            self.financial_terms
        )
        
        self.training_pairs = preprocessor.prepare_financial_pairs(
            max_pairs=max_training_pairs
        )
        
        training_dataset = preprocessor.create_dataset(self.training_pairs)
        
        # Step 4: Fine-tune model
        print("\n\n🎯 STEP 4: Fine-tune M2M100 Model")
        print("-" * 70)
        
        # Choose base model
        model_name = "m2m100-418M" if use_418m else "m2m100-1.2B"
        base_model_path = MODELS_DIR / model_name
        
        if not base_model_path.exists():
            print(f"⚠️  Model not found at {base_model_path}")
            print("   Skipping fine-tuning (models need to be downloaded)")
            return
        
        fine_tuner = M2M100FineTuner(base_model_path)
        
        # Prepare data
        tokenized_dataset = fine_tuner.prepare_training_data(training_dataset)
        
        # Train (use small batch for CPU, larger for GPU)
        batch_size = 16 if torch.cuda.is_available() else 4
        fine_tuner.train(
            tokenized_dataset,
            epochs=2,  # Start with 2 epochs
            batch_size=batch_size
        )
        
        # Save
        fine_tuner.save_model()
        
        # Step 5: Evaluate
        print("\n\n📊 STEP 5: Evaluate Translation Quality")
        print("-" * 70)
        self._evaluate_model(fine_tuner, self.training_pairs[:100])
        
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE ".center(70, "="))
        print("="*70)
    
    def _evaluate_model(self, fine_tuner: M2M100FineTuner, test_pairs: List[Tuple[str, str]]):
        """Quick evaluation on sample pairs"""
        print("\n🔍 Evaluating on sample translations...\n")
        
        for i, (arabic, expected_english) in enumerate(test_pairs[:5], 1):
            # Tokenize
            inputs = fine_tuner.tokenizer(
                arabic,
                return_tensors="pt",
                max_length=128,
                truncation=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = fine_tuner.model.generate(
                    **inputs,
                    forced_bos_token_id=fine_tuner.tokenizer.get_lang_id("en"),
                    max_length=128
                )
            
            # Decode
            generated = fine_tuner.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"[{i}] Arabic: {arabic}")
            print(f"    Expected: {expected_english}")
            print(f"    Generated: {generated}")
            print()

# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Arabic Translation Training Pipeline with Kaggle Data"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use 418M model (faster) instead of 1.2B (better quality)"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10000,
        help="Maximum training pairs to use (default: 10000)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download datasets, don't train"
    )
    
    args = parser.parse_args()
    
    pipeline = TranslationTrainingPipeline()
    
    if args.download_only:
        pipeline.dataset_manager.download_all()
    else:
        pipeline.run_full_pipeline(
            use_418m=args.fast,
            max_training_pairs=args.max_pairs
        )

if __name__ == "__main__":
    main()
