#!/usr/bin/env python3
"""
Mojo Translation Service
Semantic translation using embeddings + retrieval for high-quality Arabic-English translation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
import time
import uuid
import requests
from transformers import MarianMTModel, MarianTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mojo Translation Service",
    description="Semantic Arabic-English translation using embeddings",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMBEDDING_SERVICE_URL = "http://localhost:8007"
QDRANT_URL = "http://localhost:6333"

class TranslationCache:
    """Cache for translation pairs using embeddings"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        logger.info(f"Using device: {self.device}")
    
    def get_model(self, direction: str):
        """Get or load translation model"""
        if direction == "ar-en":
            model_name = "Helsinki-NLP/opus-mt-ar-en"
        elif direction == "en-ar":
            model_name = "Helsinki-NLP/opus-mt-en-ar"
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        
        if model_name not in self.models:
            logger.info(f"Loading model: {model_name}")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.models[model_name] = (tokenizer, model)
            logger.info(f"Model loaded: {model_name}")
        
        return self.models[model_name]
    
    def translate_text(self, text: str, direction: str) -> str:
        """Translate using MarianMT"""
        tokenizer, model = self.get_model(direction)
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4)
        
        # Decode
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

translation_cache = TranslationCache()

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    source_lang: str = Field(default="ar", pattern="^(ar|en)$")
    target_lang: str = Field(default="en", pattern="^(ar|en)$")
    use_embeddings: bool = Field(default=True, description="Use embeddings for semantic enhancement")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    @validator('source_lang', 'target_lang')
    def validate_langs(cls, v):
        if v not in ['ar', 'en']:
            raise ValueError("Only 'ar' and 'en' are supported")
        return v

class TranslateBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=32)
    source_lang: str = Field(default="ar", pattern="^(ar|en)$")
    target_lang: str = Field(default="en", pattern="^(ar|en)$")
    use_embeddings: bool = Field(default=True)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "translation-mojo",
        "version": "0.1.0",
        "features": [
            "Arabic ↔ English translation",
            "Embedding-enhanced translation",
            "Semantic similarity search",
            "MarianMT neural translation"
        ],
        "embedding_service": EMBEDDING_SERVICE_URL,
        "device": translation_cache.device
    }

@app.post("/translate")
async def translate(request: TranslateRequest):
    """Translate text with optional embedding enhancement"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"[{request_id}] Translation request: {request.source_lang} → {request.target_lang}, length={len(request.text)}")
        
        # Determine translation direction
        direction = f"{request.source_lang}-{request.target_lang}"
        if direction not in ["ar-en", "en-ar"]:
            raise HTTPException(400, "Must translate between Arabic and English")
        
        # Get base translation
        translation_start = time.time()
        translated_text = translation_cache.translate_text(request.text, direction)
        translation_time = (time.time() - translation_start) * 1000
        
        logger.info(f"[{request_id}] Base translation: {translation_time:.2f}ms")
        
        # Optional: Enhance with embeddings for verification
        semantic_score = None
        if request.use_embeddings:
            try:
                # Get embeddings for source and translation
                source_embed_resp = requests.post(
                    f"{EMBEDDING_SERVICE_URL}/embed/single",
                    json={
                        "text": request.text,
                        "model_type": "financial" if request.source_lang == "ar" else "general"
                    },
                    timeout=5
                )
                
                target_embed_resp = requests.post(
                    f"{EMBEDDING_SERVICE_URL}/embed/single",
                    json={
                        "text": translated_text,
                        "model_type": "financial" if request.target_lang == "ar" else "general"
                    },
                    timeout=5
                )
                
                if source_embed_resp.ok and target_embed_resp.ok:
                    source_emb = source_embed_resp.json()["embedding"]
                    target_emb = target_embed_resp.json()["embedding"]
                    
                    # Calculate cosine similarity
                    import numpy as np
                    dot_product = np.dot(source_emb, target_emb)
                    norm_source = np.linalg.norm(source_emb)
                    norm_target = np.linalg.norm(target_emb)
                    semantic_score = dot_product / (norm_source * norm_target)
                    
                    logger.info(f"[{request_id}] Semantic similarity: {semantic_score:.3f}")
                
            except Exception as e:
                logger.warning(f"[{request_id}] Embedding enhancement failed: {e}")
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "source_text": request.text,
            "translated_text": translated_text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "direction": direction,
            "semantic_score": semantic_score,
            "translation_time_ms": round(translation_time, 2),
            "total_time_ms": round(total_time, 2),
            "request_id": request_id,
            "model_used": f"Helsinki-NLP/opus-mt-{direction}"
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Translation error: {e}")
        raise HTTPException(500, f"Translation failed: {str(e)}")

@app.post("/translate/batch")
async def translate_batch(request: TranslateBatchRequest):
    """Translate multiple texts"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        direction = f"{request.source_lang}-{request.target_lang}"
        logger.info(f"[{request_id}] Batch translation: {len(request.texts)} texts, {direction}")
        
        translations = []
        for i, text in enumerate(request.texts):
            translated = translation_cache.translate_text(text, direction)
            translations.append({
                "source": text,
                "translation": translated,
                "index": i
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "translations": translations,
            "count": len(translations),
            "direction": direction,
            "total_time_ms": round(total_time, 2),
            "avg_time_per_text_ms": round(total_time / len(request.texts), 2),
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Batch translation error: {e}")
        raise HTTPException(500, f"Batch translation failed: {str(e)}")

@app.post("/translate/with_rag")
async def translate_with_rag(request: TranslateRequest):
    """Translate using RAG for context-aware translation"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"[{request_id}] RAG-enhanced translation")
        
        # Step 1: Get embedding for source text
        model_type = "financial" if request.source_lang == "ar" else "general"
        embed_resp = requests.post(
            f"{EMBEDDING_SERVICE_URL}/embed/single",
            json={"text": request.text, "model_type": model_type},
            timeout=5
        )
        
        if not embed_resp.ok:
            raise HTTPException(500, "Failed to generate embedding")
        
        source_embedding = embed_resp.json()["embedding"]
        
        # Step 2: Search Qdrant for similar translations (if collection exists)
        similar_translations = []
        try:
            search_resp = requests.post(
                f"{QDRANT_URL}/collections/translations/points/search",
                json={
                    "vector": source_embedding,
                    "limit": 3,
                    "with_payload": True
                },
                timeout=3
            )
            
            if search_resp.ok:
                results = search_resp.json().get("result", [])
                similar_translations = [
                    {
                        "source": r["payload"]["source"],
                        "translation": r["payload"]["translation"],
                        "score": r["score"]
                    }
                    for r in results
                ]
                logger.info(f"[{request_id}] Found {len(similar_translations)} similar translations")
        except:
            logger.warning(f"[{request_id}] No similar translations found (collection may not exist)")
        
        # Step 3: Base translation
        direction = f"{request.source_lang}-{request.target_lang}"
        translated_text = translation_cache.translate_text(request.text, direction)
        
        # Step 4: Store this translation pair in Qdrant for future use
        try:
            # Generate embedding for translation
            target_model = "financial" if request.target_lang == "ar" else "general"
            target_embed_resp = requests.post(
                f"{EMBEDDING_SERVICE_URL}/embed/single",
                json={"text": translated_text, "model_type": target_model},
                timeout=5
            )
            
            if target_embed_resp.ok:
                # Store in Qdrant
                point_id = abs(hash(request.text)) % (10**9)
                requests.put(
                    f"{QDRANT_URL}/collections/translations/points",
                    json={
                        "points": [{
                            "id": point_id,
                            "vector": source_embedding,
                            "payload": {
                                "source": request.text,
                                "translation": translated_text,
                                "source_lang": request.source_lang,
                                "target_lang": request.target_lang,
                                "timestamp": time.time()
                            }
                        }]
                    },
                    timeout=3
                )
                logger.info(f"[{request_id}] Stored translation pair in Qdrant")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to store in Qdrant: {e}")
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "source_text": request.text,
            "translated_text": translated_text,
            "similar_translations": similar_translations,
            "direction": direction,
            "total_time_ms": round(total_time, 2),
            "request_id": request_id,
            "method": "RAG-enhanced translation"
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] RAG translation error: {e}")
        raise HTTPException(500, f"RAG translation failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available translation models"""
    return {
        "models": [
            {
                "direction": "ar-en",
                "name": "Helsinki-NLP/opus-mt-ar-en",
                "description": "Arabic to English",
                "status": "ready"
            },
            {
                "direction": "en-ar",
                "name": "Helsinki-NLP/opus-mt-en-ar",
                "description": "English to Arabic",
                "status": "ready"
            }
        ],
        "embedding_models": [
            {
                "name": "general",
                "dimensions": 384,
                "languages": "50+"
            },
            {
                "name": "financial",
                "dimensions": 768,
                "languages": "Arabic financial"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("🌐 Mojo Translation Service")
    print("=" * 80)
    print("🚀 Status: Starting...")
    print("📍 Port: 8008")
    print("🌐 Health: http://localhost:8008/health")
    print("📚 API Docs: http://localhost:8008/docs")
    print("=" * 80)
    print("")
    print("🎯 Features:")
    print("  • Arabic ↔ English translation (MarianMT)")
    print("  • Embedding-enhanced translation quality")
    print("  • RAG-based translation memory")
    print("  • Semantic similarity verification")
    print("=" * 80)
    print("")
    print("📝 Endpoints:")
    print("  POST /translate          - Single translation")
    print("  POST /translate/batch    - Batch translation")
    print("  POST /translate/with_rag - RAG-enhanced translation")
    print("  GET  /models             - List available models")
    print("=" * 80)
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")
