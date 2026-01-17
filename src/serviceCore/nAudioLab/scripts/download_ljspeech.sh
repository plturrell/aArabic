#!/bin/bash
"""
Download LJSpeech Dataset
========================

Downloads and extracts the LJSpeech-1.1 dataset for TTS training.

Dataset: LJSpeech-1.1
Size: 2.6 GB compressed
Samples: 13,100 audio clips
Quality: 22.05 kHz, 16-bit mono
Speaker: Single female speaker (Linda Johnson)
"""

set -e  # Exit on error

# Configuration
DATASET_URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DOWNLOAD_DIR="data/datasets"
DATASET_NAME="LJSpeech-1.1"
ARCHIVE_NAME="LJSpeech-1.1.tar.bz2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "  LJSpeech Dataset Download"
echo "  AudioLabShimmy - Day 16"
echo "=================================================="
echo ""

# Create download directory
echo "Creating download directory..."
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Check if dataset already exists
if [ -d "$DATASET_NAME" ]; then
    echo -e "${YELLOW}Warning: Dataset directory already exists: $DATASET_NAME${NC}"
    read -p "Do you want to re-download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download. Using existing dataset."
        exit 0
    fi
    echo "Removing existing dataset..."
    rm -rf "$DATASET_NAME"
fi

# Download dataset
if [ -f "$ARCHIVE_NAME" ]; then
    echo -e "${YELLOW}Archive already exists. Skipping download.${NC}"
else
    echo "Downloading LJSpeech-1.1 dataset (2.6 GB)..."
    echo "This may take several minutes depending on your connection..."
    
    if command -v wget &> /dev/null; then
        wget -c "$DATASET_URL" -O "$ARCHIVE_NAME"
    elif command -v curl &> /dev/null; then
        curl -L "$DATASET_URL" -o "$ARCHIVE_NAME"
    else
        echo -e "${RED}Error: Neither wget nor curl is available${NC}"
        echo "Please install wget or curl and try again"
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Download failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Download complete${NC}"
fi

# Extract dataset
echo ""
echo "Extracting dataset..."
tar -xjf "$ARCHIVE_NAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Extraction failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Extraction complete${NC}"

# Verify dataset structure
echo ""
echo "Verifying dataset structure..."

if [ ! -d "$DATASET_NAME" ]; then
    echo -e "${RED}Error: Dataset directory not found${NC}"
    exit 1
fi

if [ ! -d "$DATASET_NAME/wavs" ]; then
    echo -e "${RED}Error: Audio directory (wavs) not found${NC}"
    exit 1
fi

if [ ! -f "$DATASET_NAME/metadata.csv" ]; then
    echo -e "${RED}Error: Metadata file not found${NC}"
    exit 1
fi

# Count audio files
NUM_FILES=$(ls "$DATASET_NAME/wavs"/*.wav 2>/dev/null | wc -l)
echo "Found $NUM_FILES audio files"

if [ "$NUM_FILES" -ne 13100 ]; then
    echo -e "${YELLOW}Warning: Expected 13,100 files, found $NUM_FILES${NC}"
fi

# Display dataset statistics
echo ""
echo "Dataset Statistics:"
echo "==================="
echo "Location: $(pwd)/$DATASET_NAME"
echo "Audio files: $NUM_FILES"
echo "Sample rate: 22.05 kHz"
echo "Bit depth: 16-bit"
echo "Channels: Mono"
echo "Total duration: ~24 hours"

# Calculate disk usage
DATASET_SIZE=$(du -sh "$DATASET_NAME" | cut -f1)
echo "Disk usage: $DATASET_SIZE"

echo ""
echo -e "${GREEN}✓ Dataset ready for preprocessing${NC}"
echo ""
echo "Next steps:"
echo "  1. Run preprocessing script:"
echo "     ./scripts/preprocess_ljspeech.sh"
echo "  2. This will convert audio to 48kHz, extract features"
echo "  3. Preprocessing will take several hours"
echo ""
echo "=================================================="
