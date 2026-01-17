#!/usr/bin/env python3
"""
LJSpeech Preprocessing Pipeline
================================

Orchestrates the preprocessing of LJSpeech dataset:
1. Convert audio to 48kHz stereo
2. Extract mel-spectrograms (128 bins)
3. Extract F0 contours (YIN algorithm)
4. Extract energy values
5. Convert text to phonemes
6. Prepare for forced alignment

This script uses existing Mojo/Zig components built in Days 1-5.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List
import multiprocessing
from tqdm import tqdm

# Configuration
INPUT_DIR = "data/datasets/LJSpeech-1.1"
OUTPUT_DIR = "data/datasets/ljspeech_processed"
TARGET_SR = 48000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
NUM_WORKERS = multiprocessing.cpu_count()

def print_header(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def create_directory_structure():
    """Create output directory structure."""
    print("Creating directory structure...")
    
    dirs = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/audio",          # 48kHz stereo audio
        f"{OUTPUT_DIR}/mels",            # Mel-spectrograms
        f"{OUTPUT_DIR}/f0",              # F0 contours
        f"{OUTPUT_DIR}/energy",          # Energy values
        f"{OUTPUT_DIR}/phonemes",        # Phoneme sequences
        f"{OUTPUT_DIR}/metadata",        # Metadata files
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created {len(dirs)} directories")

def load_metadata() -> List[Dict]:
    """Load LJSpeech metadata."""
    print("Loading metadata...")
    
    metadata_path = Path(INPUT_DIR) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    samples = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                samples.append({
                    'id': parts[0],
                    'text': parts[1],
                    'normalized_text': parts[2] if len(parts) > 2 else parts[1]
                })
    
    print(f"✓ Loaded {len(samples)} samples")
    return samples

def convert_audio_to_48k(input_path: str, output_path: str) -> bool:
    """Convert audio to 48kHz stereo using Sox or ffmpeg."""
    try:
        # Try Sox first (faster)
        if subprocess.run(['which', 'sox'], capture_output=True).returncode == 0:
            cmd = [
                'sox', input_path, '-r', '48000', '-c', '2', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        # Fall back to ffmpeg
        elif subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode == 0:
            cmd = [
                'ffmpeg', '-i', input_path, '-ar', '48000', '-ac', '2',
                '-y', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True, stderr=subprocess.DEVNULL)
            return True
        else:
            print("Error: Neither sox nor ffmpeg found")
            return False
    except subprocess.CalledProcessError:
        return False

def process_single_sample(args) -> Dict:
    """Process a single audio sample."""
    sample_id, sample_data = args
    
    input_audio = Path(INPUT_DIR) / "wavs" / f"{sample_id}.wav"
    output_audio = Path(OUTPUT_DIR) / "audio" / f"{sample_id}.wav"
    
    if not input_audio.exists():
        return {'id': sample_id, 'status': 'error', 'message': 'Input not found'}
    
    # Convert audio to 48kHz stereo
    if not convert_audio_to_48k(str(input_audio), str(output_audio)):
        return {'id': sample_id, 'status': 'error', 'message': 'Conversion failed'}
    
    # Extract features using Mojo preprocessor (would call actual Mojo code)
    # For now, we'll create placeholder files
    
    # Create metadata entry
    metadata = {
        'id': sample_id,
        'text': sample_data['text'],
        'normalized_text': sample_data['normalized_text'],
        'audio_path': str(output_audio.relative_to(OUTPUT_DIR)),
        'sample_rate': TARGET_SR,
        'n_mels': N_MELS
    }
    
    # Save metadata
    metadata_path = Path(OUTPUT_DIR) / "metadata" / f"{sample_id}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {'id': sample_id, 'status': 'success'}

def process_audio_files(samples: List[Dict]):
    """Process all audio files with multiprocessing."""
    print_header("Processing Audio Files")
    print(f"Converting {len(samples)} audio files to 48kHz stereo...")
    print(f"Using {NUM_WORKERS} workers")
    
    # Prepare arguments
    args_list = [(s['id'], s) for s in samples]
    
    # Process with progress bar
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_sample, args_list),
            total=len(samples),
            desc="Processing"
        ))
    
    # Count successes/failures
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = len(results) - successes
    
    print(f"\n✓ Processed {successes} samples successfully")
    if failures > 0:
        print(f"✗ {failures} samples failed")
    
    return results

def extract_features_batch(samples: List[Dict]):
    """Extract mel-spectrograms, F0, and energy for all samples."""
    print_header("Extracting Features")
    print("This step would call Mojo feature extraction...")
    print("Features to extract:")
    print("  • Mel-spectrograms (128 bins)")
    print("  • F0 contours (YIN algorithm)")
    print("  • Energy values")
    print("")
    print("Note: This requires running the Mojo preprocessing pipeline")
    print("      which uses the components built in Days 1-5:")
    print("      - audio_io.zig (audio loading)")
    print("      - mel_features.mojo (mel extraction)")
    print("      - f0_extractor.mojo (F0 extraction)")
    print("      - prosody.mojo (energy extraction)")
    print("")
    print("Command to run:")
    print("  mojo run mojo/training/preprocessor.mojo \\")
    print("    --input data/datasets/ljspeech_processed/audio \\")
    print("    --output data/datasets/ljspeech_processed")

def convert_text_to_phonemes(samples: List[Dict]):
    """Convert text to phonemes for all samples."""
    print_header("Converting Text to Phonemes")
    print("This step would call Mojo text processing...")
    print("Processing steps:")
    print("  • Text normalization (normalizer.mojo)")
    print("  • Phoneme conversion (phoneme.mojo)")
    print("  • CMU dictionary lookup (cmu_dict.mojo)")
    print("")
    print("Note: Uses components from Days 4-5")

def generate_statistics(results: List[Dict]):
    """Generate preprocessing statistics."""
    print_header("Preprocessing Statistics")
    
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total - successful
    
    print(f"Total samples: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    # Calculate storage
    audio_dir = Path(OUTPUT_DIR) / "audio"
    if audio_dir.exists():
        total_size = sum(f.stat().st_size for f in audio_dir.glob("*.wav"))
        print(f"\nAudio storage: {total_size / 1024**3:.2f} GB")
    
    print("\nNext steps:")
    print("  1. Run Mojo feature extraction")
    print("  2. Run phoneme conversion")
    print("  3. Run forced alignment (Montreal Forced Aligner)")
    print("  4. Verify preprocessing quality")

def main():
    """Main preprocessing pipeline."""
    print_header("LJSpeech Preprocessing Pipeline")
    print("AudioLabShimmy - Day 16")
    
    # Check if input dataset exists
    if not Path(INPUT_DIR).exists():
        print(f"Error: Input dataset not found: {INPUT_DIR}")
        print("Please run download_ljspeech.sh first")
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Load metadata
    samples = load_metadata()
    
    # Process audio files
    results = process_audio_files(samples)
    
    # Extract features (placeholder - would call Mojo)
    extract_features_batch(samples)
    
    # Convert text to phonemes (placeholder - would call Mojo)
    convert_text_to_phonemes(samples)
    
    # Generate statistics
    generate_statistics(results)
    
    print_header("Preprocessing Stage 1 Complete")
    print("Audio files converted to 48kHz stereo")
    print("Ready for feature extraction (Day 17-18)")

if __name__ == "__main__":
    main()
