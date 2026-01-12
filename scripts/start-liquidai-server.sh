#!/bin/bash
# Start vLLM server for LiquidAI/LFM2.5-1.2B-Base model

set -e

MODEL_PATH="/Users/user/Documents/arabic_folder/vendor/layerModels/huggingFace/LiquidAI-LFM2.5-1.2B-Base"
PORT=8000
HOST="0.0.0.0"

echo "Starting vLLM server for LiquidAI/LFM2.5-1.2B-Base..."
echo "Model path: $MODEL_PATH"
echo "Server will be available at: http://localhost:$PORT"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the model first using:"
    echo "  huggingface-cli download LiquidAI/LFM2.5-1.2B-Base --local-dir $MODEL_PATH"
    exit 1
fi

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --served-model-name "LiquidAI/LFM2.5-1.2B-Base" \
    --trust-remote-code

echo "vLLM server stopped"
