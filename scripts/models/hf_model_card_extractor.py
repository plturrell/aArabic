#!/usr/bin/env python3
"""
HuggingFace Model Card Extractor and Mapper
Extracts model card metadata and maps to orchestration task categories
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

# Mapping from HuggingFace tags/pipeline_tag to our orchestration categories
HF_TO_ORCHESTRATION_MAPPING = {
    # Math & Reasoning
    "math": ["math"],
    "reasoning": ["reasoning", "math"],
    "question-answering": ["reasoning"],
    
    # Code
    "text-generation": ["code"],  # Many text-gen models do code
    "code": ["code"],
    "code-generation": ["code"],
    
    # Summarization & Document Processing
    "summarization": ["summarization"],
    "document-question-answering": ["ocr_extraction", "reasoning"],
    "document": ["ocr_extraction"],
    
    # Translation (important for Arabic)
    "translation": ["relational"],  # Cross-lingual relationships
    "multilingual": ["relational"],
    
    # Time Series
    "time-series": ["time_series"],
    "forecasting": ["time_series"],
    
    # Vector Search & Embeddings
    "feature-extraction": ["vector_search"],
    "sentence-similarity": ["vector_search"],
    "embeddings": ["vector_search"],
    
    # Graph & Relational
    "table-question-answering": ["relational", "graph"],
    "tabular": ["relational"],
    
    # OCR & Extraction
    "image-to-text": ["ocr_extraction"],
    "ocr": ["ocr_extraction"],
}

# Infer categories from model name/description
MODEL_NAME_PATTERNS = {
    "math": ["gsm", "math", "calc"],
    "code": ["code", "coder", "starcoder", "codegen"],
    "reasoning": ["reason", "think", "chain-of-thought", "cot"],
    "summarization": ["summar", "abstract"],
    "time_series": ["forecast", "timeseries", "time-series"],
    "relational": ["sql", "table", "tabular"],
    "graph": ["graph", "cypher", "neo4j"],
    "vector_search": ["embed", "retriev", "rag"],
    "ocr_extraction": ["ocr", "document", "vision", "extract"],
}

# Language-specific enhancements
LANGUAGE_MAPPINGS = {
    "ar": ["relational"],  # Arabic models benefit from cross-lingual understanding
    "multilingual": ["relational"],
}


def fetch_model_card(repo_id: str) -> Optional[Dict[str, Any]]:
    """Fetch model card from HuggingFace API"""
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Warning: Could not fetch model card for {repo_id}: {e}", file=sys.stderr)
        return None


def map_to_orchestration_categories(model_data: Dict[str, Any]) -> List[str]:
    """Map HuggingFace model metadata to orchestration categories"""
    categories = set()
    
    # 1. Map from pipeline_tag
    pipeline_tag = model_data.get("pipeline_tag", "")
    if pipeline_tag in HF_TO_ORCHESTRATION_MAPPING:
        categories.update(HF_TO_ORCHESTRATION_MAPPING[pipeline_tag])
    
    # 2. Map from tags
    tags = model_data.get("tags", [])
    for tag in tags:
        if tag in HF_TO_ORCHESTRATION_MAPPING:
            categories.update(HF_TO_ORCHESTRATION_MAPPING[tag])
    
    # 3. Infer from model name
    model_id = model_data.get("modelId", "").lower()
    for category, patterns in MODEL_NAME_PATTERNS.items():
        if any(pattern in model_id for pattern in patterns):
            categories.add(category)
    
    # 4. Check language tags
    for tag in tags:
        if tag.startswith("language:"):
            lang = tag.split(":")[1]
            if lang in LANGUAGE_MAPPINGS:
                categories.update(LANGUAGE_MAPPINGS[lang])
    
    # 5. Check for multilingual
    if any("multilingual" in tag for tag in tags):
        categories.update(LANGUAGE_MAPPINGS["multilingual"])
    
    return sorted(list(categories))


def extract_model_metadata(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant metadata from HuggingFace model card"""
    
    metadata = {
        "model_id": model_data.get("modelId", ""),
        "name": model_data.get("modelId", "").split("/")[-1],
        "author": model_data.get("author", ""),
        "pipeline_tag": model_data.get("pipeline_tag", ""),
        "tags": model_data.get("tags", []),
        "downloads": model_data.get("downloads", 0),
        "likes": model_data.get("likes", 0),
        "library_name": model_data.get("library_name", ""),
        "license": model_data.get("license", ""),
        "languages": [],
        "datasets": [],
        "metrics": {},
    }
    
    # Extract languages
    for tag in metadata["tags"]:
        if tag.startswith("language:"):
            metadata["languages"].append(tag.split(":")[1])
        elif tag.startswith("dataset:"):
            metadata["datasets"].append(tag.split(":")[1])
    
    # Extract card data if available
    if "cardData" in model_data:
        card = model_data["cardData"]
        metadata["base_model"] = card.get("base_model", "")
        metadata["metrics"] = card.get("model-index", [{}])[0].get("results", [])
    
    # Map to orchestration categories
    metadata["orchestration_categories"] = map_to_orchestration_categories(model_data)
    
    # Determine agent types
    metadata["agent_types"] = determine_agent_types(metadata["orchestration_categories"])
    
    # Extract benchmark info if available
    metadata["benchmarks"] = extract_benchmarks(model_data)
    
    return metadata


def determine_agent_types(categories: List[str]) -> List[str]:
    """Determine which agent types this model supports"""
    agent_types = set()
    
    # Inference agents (all models)
    agent_types.add("inference")
    
    # Tool agents (for specific categories)
    tool_categories = {"vector_search", "ocr_extraction", "relational", "graph"}
    if any(cat in tool_categories for cat in categories):
        agent_types.add("tool")
    
    # Orchestrator agents (for reasoning/summarization)
    orchestrator_categories = {"reasoning", "summarization"}
    if any(cat in orchestrator_categories for cat in categories):
        agent_types.add("orchestrator")
    
    return sorted(list(agent_types))


def extract_benchmarks(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract benchmark results from model card"""
    benchmarks = {}
    
    if "cardData" in model_data and "model-index" in model_data["cardData"]:
        for entry in model_data["cardData"]["model-index"]:
            if "results" in entry:
                for result in entry["results"]:
                    dataset = result.get("dataset", {}).get("name", "unknown")
                    metrics = result.get("metrics", [])
                    if metrics:
                        benchmarks[dataset] = {
                            metric["name"]: metric["value"]
                            for metric in metrics
                        }
    
    return benchmarks


def enrich_model_registry(registry_path: str, model_entries: List[Dict[str, Any]]) -> None:
    """Enrich MODEL_REGISTRY.json with HuggingFace model card data"""
    
    # Load existing registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Enrich each model
    for i, model in enumerate(registry["models"]):
        hf_repo = model.get("hf_repo")
        if not hf_repo:
            continue
        
        print(f"Fetching model card for: {hf_repo}")
        model_data = fetch_model_card(hf_repo)
        
        if model_data:
            metadata = extract_model_metadata(model_data)
            
            # Add orchestration metadata
            model["orchestration_categories"] = metadata["orchestration_categories"]
            model["agent_types"] = metadata["agent_types"]
            model["benchmarks"] = metadata["benchmarks"]
            model["hf_metadata"] = {
                "downloads": metadata["downloads"],
                "likes": metadata["likes"],
                "license": metadata["license"],
                "languages": metadata["languages"],
                "datasets": metadata["datasets"],
                "pipeline_tag": metadata["pipeline_tag"],
            }
            
            print(f"  ✓ Categories: {', '.join(metadata['orchestration_categories'])}")
            print(f"  ✓ Agent Types: {', '.join(metadata['agent_types'])}")
            if metadata["benchmarks"]:
                print(f"  ✓ Benchmarks: {list(metadata['benchmarks'].keys())}")
        else:
            print(f"  ✗ Could not fetch model card")
    
    # Save enriched registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✓ Registry enriched and saved to {registry_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python hf_model_card_extractor.py <path_to_MODEL_REGISTRY.json>")
        print("\nOr to test a single model:")
        print("  python hf_model_card_extractor.py --test <hf_repo_id>")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        # Test mode - fetch and display single model
        if len(sys.argv) < 3:
            print("Error: Please provide HuggingFace repo ID")
            sys.exit(1)
        
        repo_id = sys.argv[2]
        print(f"Fetching model card for: {repo_id}\n")
        
        model_data = fetch_model_card(repo_id)
        if model_data:
            metadata = extract_model_metadata(model_data)
            print(json.dumps(metadata, indent=2))
        else:
            print("Failed to fetch model card")
            sys.exit(1)
    else:
        # Enrich registry mode
        registry_path = sys.argv[1]
        
        if not Path(registry_path).exists():
            print(f"Error: Registry file not found: {registry_path}")
            sys.exit(1)
        
        print("Enriching MODEL_REGISTRY.json with HuggingFace model card data...")
        print("This may take a minute...\n")
        
        enrich_model_registry(registry_path, [])
        
        print("\n✅ Model registry enriched successfully!")
        print("\nThe registry now includes:")
        print("  • Orchestration categories (math, code, reasoning, etc.)")
        print("  • Agent types (inference, tool, orchestrator)")
        print("  • Benchmark results from HuggingFace")
        print("  • Language and dataset information")
        print("  • Download counts and popularity metrics")


if __name__ == "__main__":
    main()
