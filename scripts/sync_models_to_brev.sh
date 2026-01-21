#!/bin/bash
# Sync Large Model Files to Brev Instance
# Usage: ./scripts/sync_models_to_brev.sh [brev-instance-name]

set -e

BREV_INSTANCE="${1:-awesome-gpu-nucleus}"
LOCAL_PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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

echo "=== Model Sync to Brev Instance ==="
echo "Instance: $BREV_INSTANCE"
echo "Local:    $LOCAL_PROJECT_ROOT"
echo "Remote:   $REMOTE_PROJECT_ROOT"
echo ""

# Step 1: Check if models exist locally
print_step "Step 1: Checking local models"
if [ ! -d "$LOCAL_PROJECT_ROOT/vendor/layerModels" ]; then
    print_error "Local models directory not found: $LOCAL_PROJECT_ROOT/vendor/layerModels"
    exit 1
fi

MODEL_COUNT=$(find "$LOCAL_PROJECT_ROOT/vendor/layerModels" -type f \( -name "*.gguf" -o -name "*.safetensors" -o -name "*.bin" \) 2>/dev/null | wc -l)
print_success "Found $MODEL_COUNT model files locally"
echo ""

# Step 2: Check Brev CLI
print_step "Step 2: Checking Brev CLI"
if ! command -v brev &> /dev/null; then
    print_error "Brev CLI not found. Install from: https://brev.dev"
    exit 1
fi
print_success "Brev CLI installed"
echo ""

# Step 3: Test connection to Brev instance
print_step "Step 3: Testing connection to Brev instance"
if ! brev ls | grep -q "$BREV_INSTANCE"; then
    print_error "Brev instance '$BREV_INSTANCE' not found"
    echo "Available instances:"
    brev ls
    exit 1
fi
print_success "Brev instance accessible"
echo ""

# Step 4: Create remote directory structure
print_step "Step 4: Creating remote directory structure"
brev shell "$BREV_INSTANCE" -- "mkdir -p $REMOTE_PROJECT_ROOT/vendor/layerModels"
print_success "Remote directories created"
echo ""

# Step 5: Sync models using rsync
print_step "Step 5: Syncing models (this may take a while...)"
echo ""

# Get Brev instance SSH config
BREV_HOST=$(brev ls | grep "$BREV_INSTANCE" | awk '{print $1}')

print_warning "Starting rsync transfer..."
echo "This will sync all model files to the Brev instance."
echo "Large models may take several minutes."
echo ""

# Rsync with progress
rsync -avz --progress \
    --include='*/' \
    --include='*.gguf' \
    --include='*.safetensors' \
    --include='*.bin' \
    --include='*.model' \
    --include='*.json' \
    --include='config.json' \
    --include='tokenizer*' \
    --exclude='*' \
    "$LOCAL_PROJECT_ROOT/vendor/layerModels/" \
    "brev-$BREV_HOST:$REMOTE_PROJECT_ROOT/vendor/layerModels/"

print_success "Model sync completed!"
echo ""

# Step 6: Verify sync
print_step "Step 6: Verifying sync on remote"
REMOTE_COUNT=$(brev shell "$BREV_INSTANCE" -- "find $REMOTE_PROJECT_ROOT/vendor/layerModels -type f \( -name '*.gguf' -o -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l")
print_success "Remote model count: $REMOTE_COUNT"

if [ "$MODEL_COUNT" -eq "$REMOTE_COUNT" ]; then
    print_success "All models synced successfully!"
else
    print_warning "Model count mismatch. Local: $MODEL_COUNT, Remote: $REMOTE_COUNT"
fi
echo ""

# Step 7: Display model summary
print_step "Step 7: Model Summary on Remote"
brev shell "$BREV_INSTANCE" -- "cd $REMOTE_PROJECT_ROOT && du -sh vendor/layerModels/* 2>/dev/null | sort -h"
echo ""

print_success "Model sync complete!"
echo ""
echo "Next steps:"
echo "1. Connect to Brev: brev shell $BREV_INSTANCE"
echo "2. Navigate to project: cd $REMOTE_PROJECT_ROOT"
echo "3. Run tests: ./scripts/test_t4_gpu.sh"
echo ""
