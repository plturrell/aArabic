#!/bin/bash

################################################################################
# FastSpeech2 Training Launcher - Day 19 (Steps 0-25k)
# AudioLabShimmy - TTS Training Infrastructure
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================="
echo "  FastSpeech2 Training - Day 19"
echo "  Steps: 0 → 25,000"
echo "  Expected Duration: ~24 hours"
echo "  AudioLabShimmy TTS System"
echo -e "==================================================${NC}"
echo ""

# Check Mojo installation
if ! command -v mojo &> /dev/null; then
    echo -e "${RED}Error: Mojo not found${NC}"
    echo "Please install Mojo first:"
    echo "  bash scripts/install_mojo.sh"
    exit 1
fi

echo -e "${GREEN}✓ Mojo installed${NC}"

# Check dataset
MANIFEST_PATH="$PROJECT_ROOT/data/datasets/ljspeech_processed/training_manifest.json"
if [ ! -f "$MANIFEST_PATH" ]; then
    echo -e "${RED}Error: Training manifest not found${NC}"
    echo "Expected: $MANIFEST_PATH"
    echo ""
    echo "Please complete preprocessing first (Days 16-18):"
    echo "  bash scripts/download_ljspeech.sh"
    echo "  python3 scripts/preprocess_ljspeech.py"
    echo "  python3 scripts/extract_features_day17.py"
    echo "  python3 scripts/convert_phonemes_day18.py"
    exit 1
fi

echo -e "${GREEN}✓ Training manifest found${NC}"

# Check configuration
CONFIG_PATH="$PROJECT_ROOT/config/training_config.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Training configuration not found${NC}"
    echo "Expected: $CONFIG_PATH"
    exit 1
fi

echo -e "${GREEN}✓ Training configuration found${NC}"

# Create output directories
CHECKPOINT_DIR="$PROJECT_ROOT/data/models/fastspeech2/checkpoints"
LOG_DIR="$PROJECT_ROOT/data/models/fastspeech2/logs"
SAMPLE_DIR="$PROJECT_ROOT/data/models/fastspeech2/samples"
TB_DIR="$PROJECT_ROOT/data/models/fastspeech2/tensorboard"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$SAMPLE_DIR"
mkdir -p "$TB_DIR"

echo -e "${GREEN}✓ Output directories created${NC}"
echo ""

# Check system resources
echo -e "${BLUE}System Resources:${NC}"
echo "  CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'unknown')"
echo "  Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $0/1024/1024/1024 " GB"}' || free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo 'unknown')"

# Check available disk space
AVAILABLE_SPACE=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
echo "  Available Disk Space: $AVAILABLE_SPACE"
echo ""

# Check for existing checkpoints
LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "checkpoint_*.mojo" 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    STEP=$(basename "$LATEST_CHECKPOINT" | grep -oP '\d+' | head -1)
    echo -e "${YELLOW}Found existing checkpoint at step $STEP${NC}"
    echo "Resume from checkpoint? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        RESUME_FLAG="--resume $LATEST_CHECKPOINT"
        echo -e "${GREEN}Will resume from step $STEP${NC}"
    else
        RESUME_FLAG=""
        echo -e "${YELLOW}Starting fresh training${NC}"
    fi
else
    RESUME_FLAG=""
    echo -e "${GREEN}Starting fresh training from step 0${NC}"
fi
echo ""

# Training settings
echo -e "${BLUE}Training Settings:${NC}"
echo "  Model: FastSpeech2"
echo "  Dataset: LJSpeech-1.1 (13,100 samples)"
echo "  Max Steps: 200,000"
echo "  Today's Target: Steps 0 → 25,000"
echo "  Batch Size: 16 (effective: 32 with gradient accumulation)"
echo "  Learning Rate: 1e-4 (with warmup)"
echo "  Checkpoints: Every 5,000 steps"
echo "  Validation: Every 1,000 steps"
echo ""

# Confirm training start
echo -e "${YELLOW}Ready to start training. This will run for ~24 hours.${NC}"
echo "Continue? (y/n)"
read -r confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo -e "${RED}Training cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo ""

# Create log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_day19_$TIMESTAMP.log"

# Launch training with nohup for background execution
cd "$PROJECT_ROOT"

echo "Training log: $LOG_FILE"
echo ""
echo -e "${BLUE}Launching training process...${NC}"
echo "You can monitor progress with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop training, use:"
echo "  pkill -f train_fastspeech2.mojo"
echo ""

# Launch training
nohup mojo run mojo/train_fastspeech2.mojo \
    --config "$CONFIG_PATH" \
    --target-step 25000 \
    $RESUME_FLAG \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo -e "${GREEN}✓ Training started (PID: $TRAIN_PID)${NC}"
echo ""

# Wait a moment to check if process started successfully
sleep 3

if ps -p $TRAIN_PID > /dev/null; then
    echo -e "${GREEN}✓ Training process running successfully${NC}"
    echo ""
    echo "Monitoring initial output..."
    echo ""
    
    # Show initial log output
    tail -20 "$LOG_FILE"
    
    echo ""
    echo -e "${BLUE}=================================================="
    echo "  Training is now running in the background"
    echo "  Log file: $LOG_FILE"
    echo "  Process ID: $TRAIN_PID"
    echo -e "==================================================${NC}"
    echo ""
    echo "Useful commands:"
    echo "  Monitor logs:      tail -f $LOG_FILE"
    echo "  Watch progress:    python3 scripts/monitor_training_day19.py"
    echo "  Stop training:     kill $TRAIN_PID"
    echo "  Check status:      ps -p $TRAIN_PID"
    echo ""
    
    # Save PID for later reference
    echo "$TRAIN_PID" > "$LOG_DIR/training_day19.pid"
    
    echo -e "${GREEN}Day 19 training launch complete!${NC}"
    echo "Expected completion: $(date -v+24H '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -d '+24 hours' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '~24 hours from now')"
else
    echo -e "${RED}Error: Training process failed to start${NC}"
    echo "Check the log file for details:"
    echo "  cat $LOG_FILE"
    exit 1
fi
