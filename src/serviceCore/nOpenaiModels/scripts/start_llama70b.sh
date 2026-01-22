#!/bin/bash
# Start Llama 3.3 70B Instruct with SSD tiering
#
# This script:
# 1. Waits for the model download to complete
# 2. Validates the GGUF file
# 3. Starts the server with tiering enabled

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/layerModels"
MODEL_FILE="$MODEL_DIR/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
CONFIG_FILE="$SCRIPT_DIR/../config.llama70b.json"
EXPECTED_SIZE_GB=42

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Llama 3.3 70B Instruct - SSD Tiered Server             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if model exists and is complete
check_model() {
    if [ ! -f "$MODEL_FILE" ]; then
        echo "❌ Model file not found: $MODEL_FILE"
        echo "   Run: hf download bartowski/Llama-3.3-70B-Instruct-GGUF --include '*Q4_K_M*' --local-dir layerModels"
        return 1
    fi
    
    # Check file size (should be ~42GB)
    SIZE_BYTES=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null)
    SIZE_GB=$((SIZE_BYTES / 1073741824))
    
    if [ "$SIZE_GB" -lt "$EXPECTED_SIZE_GB" ]; then
        echo "⏳ Model download in progress: ${SIZE_GB}GB / ${EXPECTED_SIZE_GB}GB"
        return 1
    fi
    
    echo "✅ Model file ready: ${SIZE_GB}GB"
    return 0
}

# Wait for download with progress
wait_for_download() {
    echo "⏳ Waiting for model download to complete..."
    while ! check_model; do
        sleep 10
    done
}

# Validate GGUF magic bytes
validate_gguf() {
    echo "🔍 Validating GGUF format..."
    MAGIC=$(xxd -l 4 "$MODEL_FILE" | awk '{print $2$3}')
    if [ "$MAGIC" != "4747554603000000" ] && [ "$MAGIC" != "47475546" ]; then
        # GGUF magic is "GGUF" = 0x46554747 (little endian)
        echo "⚠️  Warning: Unexpected GGUF magic bytes: $MAGIC"
        echo "   Expected: GGUF (0x47475546)"
    else
        echo "✅ Valid GGUF format"
    fi
}

# Create tiering cache directory
setup_tiering() {
    CACHE_DIR="$MODEL_DIR/.cache/tiering"
    mkdir -p "$CACHE_DIR"
    echo "✅ Tiering cache: $CACHE_DIR"
}

# Memory check
check_memory() {
    echo "🧠 System Memory Check:"
    if [ "$(uname)" = "Darwin" ]; then
        TOTAL_MB=$(($(sysctl -n hw.memsize) / 1048576))
        echo "   Total RAM: ${TOTAL_MB}MB"
        
        if [ "$TOTAL_MB" -lt 16384 ]; then
            echo "   ⚠️  Less than 16GB RAM - heavy SSD tiering will be used"
        elif [ "$TOTAL_MB" -lt 32768 ]; then
            echo "   ℹ️  16-32GB RAM - moderate SSD tiering"
        else
            echo "   ✅ 32GB+ RAM - optimal for 70B model"
        fi
    fi
}

# Main
main() {
    check_memory
    echo ""
    
    if ! check_model; then
        wait_for_download
    fi
    
    validate_gguf
    setup_tiering
    echo ""
    
    echo "🚀 Starting server..."
    echo "   Config: $CONFIG_FILE"
    echo "   Model: $MODEL_FILE"
    echo ""
    
    # Set environment
    export SHIMMY_CONFIG="$CONFIG_FILE"
    export SHIMMY_MODEL_PATH="$MODEL_FILE"
    export SHIMMY_TIERING_ENABLED=1
    export SHIMMY_MAX_RAM_MB=24576
    export SHIMMY_SSD_CACHE="$MODEL_DIR/.cache/tiering"
    
    cd "$SCRIPT_DIR/.."
    
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Server starting on http://0.0.0.0:11434                         ║"
    echo "║  API: http://localhost:11434/v1/chat/completions                 ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Start the Mojo server
    if command -v mojo &> /dev/null; then
        mojo main.mojo
    else
        echo "❌ Mojo not found. Please install Mojo SDK."
        echo "   See: https://docs.modular.com/mojo/manual/get-started/"
        exit 1
    fi
}

main "$@"

