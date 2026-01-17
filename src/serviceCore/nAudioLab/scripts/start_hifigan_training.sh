#!/bin/bash
# Start HiFiGAN Training - Day 27
# AudioLabShimmy Neural Vocoder Training Script

set -e  # Exit on error

echo "========================================"
echo "  HiFiGAN Training Launcher - Day 27"
echo "  AudioLabShimmy TTS System"
echo "========================================"
echo ""

# Configuration
CONFIG_FILE="config/hifigan_training_config.yaml"
CHECKPOINT_DIR="data/models/hifigan/checkpoints"
LOG_DIR="data/models/hifigan/logs"
RESUME_CHECKPOINT=""
TARGET_STEP=50000  # Day 27 target: 50k steps

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --target-step)
            TARGET_STEP="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config PATH          Training config file (default: config/hifigan_training_config.yaml)"
            echo "  --resume PATH          Resume from checkpoint"
            echo "  --target-step NUM      Target training step (default: 50000)"
            echo "  --help                 Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if FastSpeech2 checkpoint exists
FASTSPEECH2_CHECKPOINT="data/models/fastspeech2/checkpoints/checkpoint_200000.mojo"
if [ ! -f "$FASTSPEECH2_CHECKPOINT" ]; then
    echo "Error: FastSpeech2 checkpoint not found: $FASTSPEECH2_CHECKPOINT"
    echo "Please complete FastSpeech2 training first (Days 19-26)"
    exit 1
fi

# Check if preprocessed data exists
DATA_DIR="data/datasets/ljspeech_processed"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Preprocessed data not found: $DATA_DIR"
    echo "Please run preprocessing first (Days 16-18)"
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "data/models/hifigan/samples"
echo "✓ Directories created"
echo ""

# Check system resources
echo "System Information:"
echo "  CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'unknown')"
echo "  Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $0/1024/1024/1024 " GB"}' || free -h 2>/dev/null | awk '/Mem:/ {print $2}' || echo 'unknown')"
echo "  Platform: $(uname -s) $(uname -m)"
echo ""

# Display training configuration
echo "Training Configuration:"
echo "  Config File: $CONFIG_FILE"
echo "  Data Directory: $DATA_DIR"
echo "  Checkpoint Directory: $CHECKPOINT_DIR"
echo "  Target Step: $TARGET_STEP"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  Resume From: $RESUME_CHECKPOINT"
else
    echo "  Resume From: (starting fresh)"
fi
echo ""

# Timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_day27_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/training_day27.pid"

echo "Log file: $LOG_FILE"
echo ""

# Build command
MOJO_CMD="mojo run mojo/train_hifigan.mojo --config $CONFIG_FILE --target-step $TARGET_STEP"
if [ -n "$RESUME_CHECKPOINT" ]; then
    MOJO_CMD="$MOJO_CMD --resume $RESUME_CHECKPOINT"
fi

# Confirm before starting
echo "========================================"
echo "  Ready to Start Training"
echo "========================================"
echo ""
echo "This will train HiFiGAN for ~24 hours to reach $TARGET_STEP steps."
echo ""
read -p "Start training now? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "========================================"
echo "  Starting HiFiGAN Training"
echo "========================================"
echo ""
echo "Training started at: $(date)"
echo "Command: $MOJO_CMD"
echo ""
echo "To monitor progress in real-time:"
echo "  python3 scripts/monitor_hifigan_training.py"
echo ""
echo "To stop training:"
echo "  kill \$(cat $PID_FILE)"
echo ""

# Start training in background and save PID
$MOJO_CMD 2>&1 | tee "$LOG_FILE" &
TRAINING_PID=$!
echo $TRAINING_PID > "$PID_FILE"

echo "Training process PID: $TRAINING_PID"
echo "PID saved to: $PID_FILE"
echo ""

# Wait a few seconds to check if process started successfully
sleep 3
if ps -p $TRAINING_PID > /dev/null; then
    echo "✓ Training started successfully!"
    echo ""
    echo "Training is running in the background."
    echo "Check progress with: tail -f $LOG_FILE"
else
    echo "✗ Training failed to start. Check the log file for errors:"
    echo "  $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

echo ""
echo "========================================"
echo "  Day 27 Training Schedule"
echo "========================================"
echo ""
echo "Target: 0 → 50,000 steps (~24 hours)"
echo ""
echo "Expected Checkpoints:"
echo "  - 10,000 steps (~5 hours)"
echo "  - 20,000 steps (~10 hours)"
echo "  - 30,000 steps (~15 hours)"
echo "  - 40,000 steps (~20 hours)"
echo "  - 50,000 steps (~24 hours)"
echo ""
echo "Training will continue until target reached."
echo "Monitor with: python3 scripts/monitor_hifigan_training.py"
echo ""

exit 0
