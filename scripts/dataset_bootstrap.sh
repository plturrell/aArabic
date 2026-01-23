#!/bin/bash
# Dataset Bootstrap Script
# Downloads essential benchmark datasets for nLocalModels agent categories

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ORCHESTRATION_DIR="$PROJECT_ROOT/src/serviceCore/nLocalModels/orchestration"
DATA_DIR="$PROJECT_ROOT/data/benchmarks"

echo "=========================================="
echo "Dataset Bootstrap for nLocalModels"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check for zig
    if ! command -v zig &> /dev/null; then
        echo -e "${RED}✗ Zig compiler not found${NC}"
        echo "  Install from: https://ziglang.org/download/"
        exit 1
    fi
    echo -e "${GREEN}✓ Zig compiler found${NC}"
    
    # Check for huggingface-cli
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}⚠ HuggingFace CLI not found${NC}"
        echo "  Installing huggingface_hub..."
        pip install -q huggingface_hub
        if ! command -v huggingface-cli &> /dev/null; then
            echo -e "${RED}✗ Failed to install huggingface-cli${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}✓ HuggingFace CLI found${NC}"
    
    # Check for jq (optional but helpful)
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}⚠ jq not found (optional)${NC}"
    else
        echo -e "${GREEN}✓ jq found${NC}"
    fi
    
    echo ""
}

# Build dataset loader
build_loader() {
    echo "Building dataset loader..."
    cd "$ORCHESTRATION_DIR"
    
    if ! zig build 2>&1 | grep -v "warning"; then
        echo -e "${RED}✗ Build failed${NC}"
        exit 1
    fi
    
    if [ ! -f "zig-out/bin/dataset_loader" ]; then
        echo -e "${RED}✗ dataset_loader binary not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Dataset loader built successfully${NC}"
    echo ""
}

# Download dataset
download_dataset() {
    local dataset_name=$1
    local category=$2
    local benchmark=$3
    local split=$4
    
    echo -e "${YELLOW}Downloading ${dataset_name} (${split} split)...${NC}"
    
    cd "$ORCHESTRATION_DIR"
    if zig build run-dataset-loader -- download-hf "$dataset_name" "$category" "$benchmark" "$split" 2>&1; then
        echo -e "${GREEN}✓ Downloaded ${dataset_name} ${split}${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to download ${dataset_name} ${split}${NC}"
        return 1
    fi
}

# Download essential datasets
download_essential() {
    echo "=========================================="
    echo "Downloading Essential Datasets"
    echo "=========================================="
    echo ""
    echo "This will download small, essential datasets:"
    echo "  - GSM8K (Math): ~3.5 MB"
    echo "  - HumanEval (Code): ~0.5 MB"
    echo "  Total: ~4 MB"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    
    echo ""
    
    # Math category - GSM8K test split
    download_dataset "gsm8k" "math" "gsm8k" "test" || true
    
    # Code category - HumanEval
    download_dataset "openai_humaneval" "code" "humaneval" "test" || true
    
    echo ""
    echo -e "${GREEN}Essential datasets downloaded${NC}"
    echo ""
}

# Download standard datasets
download_standard() {
    echo "=========================================="
    echo "Downloading Standard Datasets"
    echo "=========================================="
    echo ""
    echo "This will download commonly used datasets:"
    echo "  - GSM8K (Math): ~3.5 MB"
    echo "  - MATH (Math): ~50 MB"
    echo "  - HumanEval (Code): ~0.5 MB"
    echo "  - MBPP (Code): ~2 MB"
    echo "  - ARC-Challenge (Reasoning): ~2.5 MB"
    echo "  - Winogrande (Reasoning): ~5 MB"
    echo "  Total: ~65 MB"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    
    echo ""
    
    # Math
    download_dataset "gsm8k" "math" "gsm8k" "test" || true
    download_dataset "competition_math" "math" "math" "test" || true
    
    # Code
    download_dataset "openai_humaneval" "code" "humaneval" "test" || true
    download_dataset "mbpp" "code" "mbpp" "test" || true
    
    # Reasoning
    download_dataset "ai2_arc" "reasoning" "arc_challenge" "test" || true
    download_dataset "winogrande" "reasoning" "winogrande" "test" || true
    
    echo ""
    echo -e "${GREEN}Standard datasets downloaded${NC}"
    echo ""
}

# Download full datasets
download_full() {
    echo "=========================================="
    echo "Downloading Full Dataset Suite"
    echo "=========================================="
    echo ""
    echo "This will download all benchmark datasets:"
    echo "  - Math: GSM8K, MATH"
    echo "  - Code: HumanEval, MBPP"
    echo "  - Reasoning: ARC, HellaSwag, MMLU, Winogrande"
    echo "  - Summarization: SummScreen, GovReport"
    echo "  Total: ~900 MB"
    echo ""
    echo -e "${YELLOW}Warning: This will take several minutes${NC}"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    
    echo ""
    
    # Math
    download_dataset "gsm8k" "math" "gsm8k" "test" || true
    download_dataset "competition_math" "math" "math" "test" || true
    
    # Code
    download_dataset "openai_humaneval" "code" "humaneval" "test" || true
    download_dataset "mbpp" "code" "mbpp" "test" || true
    
    # Reasoning
    download_dataset "ai2_arc" "reasoning" "arc_challenge" "test" || true
    download_dataset "hellaswag" "reasoning" "hellaswag" "test" || true
    download_dataset "cais/mmlu" "reasoning" "mmlu" "test" || true
    download_dataset "winogrande" "reasoning" "winogrande" "test" || true
    
    # Summarization
    download_dataset "tau/sled" "summarization" "summscreen" "test" || true
    download_dataset "ccdv/govreport-summarization" "summarization" "govreport" "test" || true
    
    echo ""
    echo -e "${GREEN}Full dataset suite downloaded${NC}"
    echo ""
}

# List downloaded datasets
list_datasets() {
    echo "=========================================="
    echo "Downloaded Datasets"
    echo "=========================================="
    echo ""
    
    cd "$ORCHESTRATION_DIR"
    zig build run-dataset-loader -- list
    
    echo ""
}

# Validate datasets
validate_all() {
    echo "=========================================="
    echo "Validating Datasets"
    echo "=========================================="
    echo ""
    
    cd "$ORCHESTRATION_DIR"
    
    # Check if catalog exists
    if [ ! -f "$DATA_DIR/dataset_catalog.json" ]; then
        echo -e "${YELLOW}No datasets downloaded yet${NC}"
        return
    fi
    
    # Extract dataset IDs if jq is available
    if command -v jq &> /dev/null; then
        dataset_ids=$(jq -r '.datasets | keys[]' "$DATA_DIR/dataset_catalog.json" 2>/dev/null || echo "")
        
        if [ -z "$dataset_ids" ]; then
            echo -e "${YELLOW}No datasets found in catalog${NC}"
            return
        fi
        
        for dataset_id in $dataset_ids; do
            echo "Validating $dataset_id..."
            if zig build run-dataset-loader -- validate "$dataset_id" 2>&1 | grep -q "validation passed"; then
                echo -e "${GREEN}✓ $dataset_id valid${NC}"
            else
                echo -e "${RED}✗ $dataset_id validation failed${NC}"
            fi
        done
    else
        echo -e "${YELLOW}jq not available, skipping validation${NC}"
    fi
    
    echo ""
}

# Show help
show_help() {
    echo "Dataset Bootstrap Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  essential   Download essential datasets (~4 MB)"
    echo "  standard    Download standard datasets (~65 MB)"
    echo "  full        Download full dataset suite (~900 MB)"
    echo "  list        List downloaded datasets"
    echo "  validate    Validate all downloaded datasets"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 essential   # Quick start"
    echo "  $0 standard    # Recommended for most users"
    echo "  $0 full        # Complete benchmark suite"
    echo ""
}

# Main execution
main() {
    local command=${1:-essential}
    
    case $command in
        essential)
            check_prerequisites
            build_loader
            download_essential
            list_datasets
            ;;
        standard)
            check_prerequisites
            build_loader
            download_standard
            list_datasets
            ;;
        full)
            check_prerequisites
            build_loader
            download_full
            list_datasets
            ;;
        list)
            build_loader
            list_datasets
            ;;
        validate)
            build_loader
            validate_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: $command${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
