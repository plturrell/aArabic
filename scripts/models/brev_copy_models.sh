#!/bin/bash
# Copy Models to Brev Instance using brev copy
# Usage: ./scripts/models/brev_copy_models.sh [brev-instance-name]

set -e

BREV_INSTANCE="${1:-awesome-gpu-nucleus}"
LOCAL_PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_PROJECT_ROOT="/home/ubuntu/arabic_folder"

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

echo "=== Copy Models to Brev Instance using brev copy ==="
echo "Instance: $BREV_INSTANCE"
echo "Local:    $LOCAL_PROJECT_ROOT/vendor/layerModels"
echo "Remote:   $REMOTE_PROJECT_ROOT/vendor/layerModels"
echo ""

# Step 1: Check if models exist locally
print_step "Step 1: Checking local models"
if [ ! -d "$LOCAL_PROJECT_ROOT/vendor/layerModels" ]; then
    print_error "Local models directory not found: $LOCAL_PROJECT_ROOT/vendor/layerModels"
    exit 1
fi

MODEL_COUNT=$(find "$LOCAL_PROJECT_ROOT/vendor/layerModels" -type f \( -name "*.gguf" -o -name "*.safetensors" -o -name "*.bin" -o -name "*.model" -o -name "*.json" \) 2>/dev/null | wc -l | tr -d ' ')
print_success "Found $MODEL_COUNT model files locally"
echo ""

# Step 2: List models to copy
print_step "Step 2: Models to copy:"
du -sh "$LOCAL_PROJECT_ROOT/vendor/layerModels"/* 2>/dev/null | while read -r size dir; do
    echo "  $size  $(basename "$dir")"
done
echo ""

# Step 3: Confirm copy
read -p "Continue with copy? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Copy cancelled"
    exit 0
fi
echo ""

# Step 4: Copy models using brev copy
print_step "Step 4: Copying models (this may take several minutes...)"
echo ""

cd "$LOCAL_PROJECT_ROOT"

# Copy entire layerModels directory
print_step "Executing: brev copy vendor/layerModels $BREV_INSTANCE:$REMOTE_PROJECT_ROOT/vendor/"
brev copy vendor/layerModels "$BREV_INSTANCE:$REMOTE_PROJECT_ROOT/vendor/"

if [ $? -eq 0 ]; then
    print_success "Models copied successfully!"
else
    print_error "Copy failed. Check brev connection and try again."
    exit 1
fi
echo ""

# Step 5: Verify on remote
print_step "Step 5: Verifying files on remote instance..."
echo ""
brev shell "$BREV_INSTANCE" -- "ls -lh $REMOTE_PROJECT_ROOT/vendor/layerModels/"
echo ""

print_success "Model copy complete!"
echo ""
print_step "Next steps:"
echo "1. Connect to Brev: brev shell $BREV_INSTANCE"
echo "2. Navigate to project: cd $REMOTE_PROJECT_ROOT"
echo "3. Run tests: ./scripts/gpu/test_t4_gpu.sh"
echo ""
