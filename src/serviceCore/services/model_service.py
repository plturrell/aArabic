"""
Model service for loading and managing ML models
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)
import torch
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML models"""
    
    def __init__(self):
        self.camel_tokenizer: Optional[AutoTokenizer] = None
        self.camel_model: Optional[AutoModelForSequenceClassification] = None
        self.m2m_tokenizer: Optional[M2M100Tokenizer] = None
        self.m2m_model: Optional[M2M100ForConditionalGeneration] = None
        self._models_loaded = False
    
    def load_models(self) -> Tuple[bool, bool]:
        """
        Load all ML models
        
        Returns:
            Tuple of (camelbert_loaded, m2m100_loaded)
        """
        if self._models_loaded:
            return self.camel_model is not None, self.m2m_model is not None
        
        logger.info("Loading ML models...")
        
        # Load CamelBERT
        camel_loaded = self._load_camelbert()
        
        # Load M2M100
        m2m_loaded = self._load_m2m100()
        
        self._models_loaded = True
        logger.info(f"Models loaded - CamelBERT: {camel_loaded}, M2M100: {m2m_loaded}")
        
        return camel_loaded, m2m_loaded
    
    def _load_camelbert(self) -> bool:
        """Load CamelBERT model"""
        try:
            camelbert_path = Path(settings.camelbert_path)
            if not camelbert_path.exists():
                logger.warning(f"CamelBERT path does not exist: {camelbert_path}")
                return False
            
            self.camel_tokenizer = AutoTokenizer.from_pretrained(str(camelbert_path))
            self.camel_model = AutoModelForSequenceClassification.from_pretrained(
                str(camelbert_path),
                num_labels=2
            )
            logger.info("✅ CamelBERT Financial loaded")
            return True
        except Exception as e:
            logger.error(f"⚠️ Could not load CamelBERT: {e}", exc_info=True)
            return False
    
    def _load_m2m100(self) -> bool:
        """Load M2M100 model"""
        try:
            m2m_path = Path(settings.m2m100_path)
            if not m2m_path.exists():
                logger.warning(f"M2M100 path does not exist: {m2m_path}")
                return False
            
            self.m2m_tokenizer = M2M100Tokenizer.from_pretrained(str(m2m_path))
            self.m2m_model = M2M100ForConditionalGeneration.from_pretrained(str(m2m_path))
            logger.info("✅ M2M100 loaded")
            return True
        except Exception as e:
            logger.error(f"⚠️ Could not load M2M100: {e}", exc_info=True)
            return False
    
    def translate(self, text: str, source_lang: str = "ar", target_lang: str = "en") -> dict:
        """
        Translate text using M2M100 model
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translation result dictionary
        """
        if not self.m2m_model or not self.m2m_tokenizer:
            logger.warning("M2M100 model not loaded, returning mock translation")
            return {
                "original": text,
                "translated_text": f"[MOCK] Translation of: {text}",
                "model": "m2m100-418M",
                "mock": True
            }
        
        try:
            self.m2m_tokenizer.src_lang = source_lang
            encoded = self.m2m_tokenizer(text, return_tensors="pt")
            generated_tokens = self.m2m_model.generate(
                **encoded,
                forced_bos_token_id=self.m2m_tokenizer.get_lang_id(target_lang)
            )
            translated = self.m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            return {
                "original": text,
                "translated_text": translated,
                "model": "m2m100-418M",
                "source_lang": source_lang,
                "target_lang": target_lang
            }
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            raise
    
    def analyze(self, text: str) -> dict:
        """
        Analyze invoice text using CamelBERT model
        
        Args:
            text: Text to analyze
        
        Returns:
            Analysis result dictionary
        """
        if not self.camel_model or not self.camel_tokenizer:
            logger.warning("CamelBERT model not loaded, returning mock analysis")
            return {
                "classification": "Full Tax Invoice",
                "confidence": 0.98,
                "compliance_checks": {"vat_id": True, "date_format": True},
                "model": "camelbert-dialect-financial",
                "mock": True
            }
        
        try:
            inputs = self.camel_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.camel_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).tolist()[0]
            
            score = probs[1]  # Assume class 1 is "Valid/Positive"
            
            return {
                "classification": "Full Tax Invoice" if score > 0.5 else "Simplified Invoice",
                "confidence": round(score, 4),
                "raw_logits": logits.tolist(),
                "model": "camelbert-dialect-financial"
            }
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            raise
    
    @property
    def is_ready(self) -> bool:
        """Check if models are loaded"""
        return self._models_loaded and (self.camel_model is not None or self.m2m_model is not None)


# Global model service instance
model_service = ModelService()

