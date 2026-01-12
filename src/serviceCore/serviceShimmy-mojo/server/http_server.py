#!/usr/bin/env python3
"""
Shimmy-Mojo HTTP Server
OpenAI-compatible API for Pure Mojo LLM inference
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncIterator
import uvicorn
import time
import json
import asyncio
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

HOST = "0.0.0.0"
PORT = 11434  # Ollama-compatible port
MODEL_PATH = "./models"
MAX_CONTEXT = 4096

# ============================================================================
# Request/Response Models (OpenAI Compatible)
# ============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    max_tokens: int = Field(100, ge=1, le=4096)
    stream: bool = Field(False, description="Enable streaming")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    max_tokens: int = Field(100, ge=1, le=4096)
    stream: bool = False
    stop: Optional[List[str]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "shimmy-mojo"
    permission: List = []
    root: str
    parent: Optional[str] = None

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Shimmy-Mojo API",
    description="OpenAI-compatible API for Pure Mojo LLM inference",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

class ModelManager:
    """Manages loaded models"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.available_models = [
            "phi-3-mini",
            "llama-3.2-1b",
            "llama-3.2-3b"
        ]
    
    def get_model(self, model_name: str):
        """Get or load model"""
        if model_name not in self.models:
            # TODO: Load actual Mojo model
            print(f"📦 Loading model: {model_name}")
            self.models[model_name] = {
                "name": model_name,
                "loaded": True,
                "config": {}
            }
        return self.models[model_name]
    
    def list_models(self) -> List[str]:
        """List available models"""
        return self.available_models

model_manager = ModelManager()

# ============================================================================
# Helper Functions
# ============================================================================

def generate_id(prefix: str = "chatcmpl") -> str:
    """Generate unique ID"""
    timestamp = int(time.time() * 1000)
    return f"{prefix}-{timestamp}"

def format_chat_messages(messages: List[ChatMessage]) -> str:
    """Format chat messages into prompt"""
    prompt_parts = []
    for msg in messages:
        role = msg.role
        content = msg.content
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)

async def mock_generate(prompt: str, config: Dict[str, Any]) -> str:
    """Mock text generation (replace with actual Mojo call)"""
    # Simulate generation delay
    await asyncio.sleep(0.1)
    
    # Mock response
    return f"This is a mock response to: {prompt[:50]}..."

async def mock_generate_stream(prompt: str, config: Dict[str, Any]) -> AsyncIterator[str]:
    """Mock streaming generation"""
    response = f"This is a mock streaming response to: {prompt[:50]}..."
    words = response.split()
    
    for word in words:
        await asyncio.sleep(0.05)
        yield word + " "

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Shimmy-Mojo API",
        "version": "1.0.0",
        "status": "running",
        "models": model_manager.list_models()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_manager.models)
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models = model_manager.list_models()
    
    model_objects = []
    for model_id in models:
        model_objects.append(
            ModelInfo(
                id=model_id,
                created=int(time.time()),
                root=model_id
            ).dict()
        )
    
    return {
        "object": "list",
        "data": model_objects
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint (OpenAI compatible)"""
    
    try:
        # Get model
        model = model_manager.get_model(request.model)
        
        # Format prompt
        prompt = format_chat_messages(request.messages)
        
        # Generation config
        config = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_tokens": request.max_tokens,
            "stop": request.stop or []
        }
        
        # Handle streaming
        if request.stream:
            async def generate_stream():
                completion_id = generate_id()
                
                # Stream start
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
                
                # Stream tokens
                async for token in mock_generate_stream(prompt, config):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Stream end
                end_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(end_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        start_time = time.time()
        response_text = await mock_generate(prompt, config)
        duration = time.time() - start_time
        
        # Calculate tokens (mock)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())
        
        return ChatCompletionResponse(
            id=generate_id(),
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Completions endpoint (OpenAI compatible)"""
    
    try:
        # Get model
        model = model_manager.get_model(request.model)
        
        # Generation config
        config = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_tokens": request.max_tokens,
            "stop": request.stop or []
        }
        
        # Handle streaming
        if request.stream:
            async def generate_stream():
                completion_id = generate_id("cmpl")
                
                async for token in mock_generate_stream(request.prompt, config):
                    chunk = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "text": token,
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # End chunk
                end_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(end_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        # Non-streaming
        response_text = await mock_generate(request.prompt, config)
        
        return {
            "id": generate_id("cmpl"),
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(request.prompt.split()) + len(response_text.split())
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tags")
async def ollama_tags():
    """Ollama-compatible models list"""
    models = model_manager.list_models()
    
    return {
        "models": [
            {
                "name": model,
                "modified_at": datetime.now().isoformat(),
                "size": 0,
                "digest": f"sha256:{model}",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"]
                }
            }
            for model in models
        ]
    }

# ============================================================================
# Server Startup
# ============================================================================

def start_server(host: str = HOST, port: int = PORT):
    """Start the HTTP server"""
    print("=" * 80)
    print("🚀 Shimmy-Mojo HTTP Server")
    print("=" * 80)
    print()
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  URL: http://{host}:{port}")
    print()
    print("  Endpoints:")
    print(f"    GET  /                          - Server info")
    print(f"    GET  /health                    - Health check")
    print(f"    GET  /v1/models                 - List models")
    print(f"    POST /v1/chat/completions       - Chat completions")
    print(f"    POST /v1/completions            - Text completions")
    print(f"    GET  /api/tags                  - Ollama-compatible")
    print()
    print("  Available models:")
    for model in model_manager.list_models():
        print(f"    - {model}")
    print()
    print("=" * 80)
    print()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
