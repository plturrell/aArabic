#!/bin/bash
# Download Models Directly on Brev Instance from HuggingFace
# Usage: Run this ON the Brev instance after cloning the repo
# ./scripts/download_models_on_brev.sh [model-name]

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

MODEL_NAME="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/vendor/layerModels"

echo "=== Model Downloader for Brev Instance ==="
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    print_step "Installing HuggingFace CLI..."
    pip install -q huggingface_hub[cli]
    print_success "HuggingFace CLI installed"
fi

# Function to download a model
download_model() {
    local name=$1
    local repo=$2
    local files=$3
    
    print_step "Downloading $name from $repo"
    
    mkdir -p "$MODELS_DIR/$name"
    cd "$MODELS_DIR/$name"
    
    if [ -n "$files" ]; then
        # Download specific files
        IFS=',' read -ra FILE_ARRAY <<< "$files"
        for file in "${FILE_ARRAY[@]}"; do
            print_step "Downloading $file..."
            huggingface-cli download "$repo" "$file" --local-dir . --local-dir-use-symlinks False
        done
    else
        # Download entire repo
        huggingface-cli download "$repo" --local-dir . --local-dir-use-symlinks False
    fi
    
    print_success "$name downloaded successfully"
    du -sh "$MODELS_DIR/$name"
    echo ""
}

# If no model specified, show menu
if [ -z "$MODEL_NAME" ]; then
    echo "Available models for download:"
    echo ""
    echo "1. google-gemma-3-270m-it        (540 MB  - Testing/Dev)"
    echo "2. LFM2.5-1.2B-Instruct-GGUF     (1.2 GB  - Production)"
    echo "3. microsoft-phi-2               (5.2 GB  - General Purpose)"
    echo "4. HY-MT1.5-7B                   (4-8 GB  - Arabic Translation)"
    echo "5. all-testing                   (Downloads all testing models)"
    echo "6. all-t4-recommended            (Downloads all T4-recommended models)"
    echo ""
    read -p "Enter model number or name: " choice
    
    case $choice in
        1|google-gemma-3-270m-it)
            MODEL_NAME="google-gemma-3-270m-it"
            ;;
        2|LFM2.5-1.2B-Instruct-GGUF)
            MODEL_NAME="LFM2.5-1.2B-Instruct-GGUF"
            ;;
        3|microsoft-phi-2)
            MODEL_NAME="microsoft-phi-2"
            ;;
        4|HY-MT1.5-7B)
            MODEL_NAME="HY-MT1.5-7B"
            ;;
        5|all-testing)
            MODEL_NAME="all-testing"
            ;;
        6|all-t4-recommended)
            MODEL_NAME="all-t4-recommended"
            ;;
        *)
            MODEL_NAME="$choice"
            ;;
    esac
fi

# Download based on model name
case $MODEL_NAME in
    google-gemma-3-270m-it)
        download_model "google-gemma-3-270m-it" "google/gemma-3-270m-it" ""
        ;;
    
    LFM2.5-1.2B-Instruct-GGUF)
        download_model "LFM2.5-1.2B-Instruct-GGUF" "LiquidAI/LFM-2.5-1.2B-Instruct-GGUF" "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
        ;;
    
    microsoft-phi-2)
        download_model "microsoft-phi-2" "microsoft/phi-2" ""
        ;;
    
    HY-MT1.5-7B)
        print_warning "Note: This model requires HuggingFace token for access"
        print_warning "Run: huggingface-cli login"
        read -p "Press enter after logging in..."
        download_model "HY-MT1.5-7B" "Helsinki-NLP/opus-mt-ar-en" ""
        ;;
    
    all-testing)
        print_step "Downloading all testing models..."
        download_model "google-gemma-3-270m-it" "google/gemma-3-270m-it" ""
        download_model "LFM2.5-1.2B-Instruct-GGUF" "LiquidAI/LFM-2.5-1.2B-Instruct-GGUF" "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
        ;;
    
    all-t4-recommended)
        print_step "Downloading all T4-recommended models..."
        download_model "google-gemma-3-270m-it" "google/gemma-3-270m-it" ""
        download_model "LFM2.5-1.2B-Instruct-GGUF" "LiquidAI/LFM-2.5-1.2B-Instruct-GGUF" "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
        download_model "microsoft-phi-2" "microsoft/phi-2" ""
        ;;
    
    *)
        print_error "Unknown model: $MODEL_NAME"
        print_error "Use: google-gemma-3-270m-it, LFM2.5-1.2B-Instruct-GGUF, microsoft-phi-2, or HY-MT1.5-7B"
        exit 1
        ;;
esac

print_success "Model download complete!"
echo ""
print_step "Models installed in: $MODELS_DIR"
ls -lh "$MODELS_DIR"
echo ""
print_step "Next steps:"
echo "1. Run tests: ./scripts/test_t4_gpu.sh"
echo "2. Start inference: cd src/serviceCore/nOpenaiServer && zig build run"
echo ""
