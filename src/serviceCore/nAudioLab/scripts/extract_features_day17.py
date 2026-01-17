#!/usr/bin/env python3
"""
Day 17: Feature Extraction (Part 1) - Mel-Spectrograms & F0
AudioLabShimmy - LJSpeech Feature Extraction Pipeline

This script orchestrates the extraction of:
1. Mel-spectrograms (128 bins, 48kHz)
2. F0 contours (YIN algorithm)
3. Energy values

Uses Mojo implementations from Days 2-3 for high-performance extraction.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Configuration
INPUT_DIR = "data/datasets/ljspeech_processed"
AUDIO_DIR = os.path.join(INPUT_DIR, "audio")
MEL_DIR = os.path.join(INPUT_DIR, "mels")
F0_DIR = os.path.join(INPUT_DIR, "f0")
ENERGY_DIR = os.path.join(INPUT_DIR, "energy")
METADATA_DIR = os.path.join(INPUT_DIR, "metadata")

# Feature extraction parameters
SAMPLE_RATE = 48000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048

# Processing
NUM_WORKERS = max(1, cpu_count() - 1)  # Leave one core free


def print_header():
    """Print script header"""
    print("=" * 60)
    print("  LJSpeech Feature Extraction (Day 17)")
    print("  AudioLabShimmy - Mel + F0 + Energy")
    print("=" * 60)
    print()


def create_directories():
    """Create output directories"""
    print("Creating output directories...")
    dirs = [MEL_DIR, F0_DIR, ENERGY_DIR]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {len(dirs)} output directories")
    print()


def load_metadata():
    """Load metadata for all samples"""
    print("Loading metadata...")
    metadata_files = sorted(Path(METADATA_DIR).glob("*.json"))
    
    samples = []
    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            sample = json.load(f)
            samples.append(sample)
    
    print(f"✓ Loaded {len(samples)} samples")
    print()
    return samples


def extract_mel_spectrogram(audio_path, output_path):
    """
    Extract mel-spectrogram using Mojo implementation
    Calls: mojo/audio/mel_features.mojo
    """
    try:
        # Call Mojo mel extraction
        cmd = [
            "mojo", "run",
            "mojo/audio/mel_features.mojo",
            "--input", audio_path,
            "--output", output_path,
            "--sample-rate", str(SAMPLE_RATE),
            "--n-mels", str(N_MELS),
            "--n-fft", str(N_FFT),
            "--hop-length", str(HOP_LENGTH)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def extract_f0_contour(audio_path, output_path):
    """
    Extract F0 contour using Mojo implementation
    Calls: mojo/audio/f0_extractor.mojo (YIN algorithm)
    """
    try:
        # Call Mojo F0 extraction
        cmd = [
            "mojo", "run",
            "mojo/audio/f0_extractor.mojo",
            "--input", audio_path,
            "--output", output_path,
            "--sample-rate", str(SAMPLE_RATE),
            "--method", "yin"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def extract_energy(audio_path, output_path):
    """
    Extract frame-level energy using Mojo implementation
    Calls: mojo/audio/prosody.mojo
    """
    try:
        # Call Mojo energy extraction
        cmd = [
            "mojo", "run",
            "mojo/audio/prosody.mojo",
            "--input", audio_path,
            "--output", output_path,
            "--sample-rate", str(SAMPLE_RATE),
            "--feature", "energy"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def process_sample(sample):
    """
    Process a single sample: extract mel, F0, and energy
    """
    sample_id = sample['id']
    audio_path = os.path.join(INPUT_DIR, sample['audio_path'])
    
    # Output paths
    mel_path = os.path.join(MEL_DIR, f"{sample_id}.npy")
    f0_path = os.path.join(F0_DIR, f"{sample_id}.npy")
    energy_path = os.path.join(ENERGY_DIR, f"{sample_id}.npy")
    
    # Skip if already exists
    if os.path.exists(mel_path) and os.path.exists(f0_path) and os.path.exists(energy_path):
        return {
            'id': sample_id,
            'success': True,
            'skipped': True,
            'error': None
        }
    
    # Extract mel-spectrogram
    mel_success, mel_error = extract_mel_spectrogram(audio_path, mel_path)
    if not mel_success:
        return {
            'id': sample_id,
            'success': False,
            'skipped': False,
            'error': f"Mel extraction failed: {mel_error}"
        }
    
    # Extract F0 contour
    f0_success, f0_error = extract_f0_contour(audio_path, f0_path)
    if not f0_success:
        return {
            'id': sample_id,
            'success': False,
            'skipped': False,
            'error': f"F0 extraction failed: {f0_error}"
        }
    
    # Extract energy
    energy_success, energy_error = extract_energy(audio_path, energy_path)
    if not energy_success:
        return {
            'id': sample_id,
            'success': False,
            'skipped': False,
            'error': f"Energy extraction failed: {energy_error}"
        }
    
    # Update metadata
    sample['mel_path'] = f"mels/{sample_id}.npy"
    sample['f0_path'] = f"f0/{sample_id}.npy"
    sample['energy_path'] = f"energy/{sample_id}.npy"
    
    # Save updated metadata
    metadata_path = os.path.join(METADATA_DIR, f"{sample_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(sample, f, indent=2)
    
    return {
        'id': sample_id,
        'success': True,
        'skipped': False,
        'error': None
    }


def process_samples_parallel(samples):
    """
    Process all samples in parallel
    """
    print("=" * 60)
    print("  Extracting Features (Mel + F0 + Energy)")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Workers: {NUM_WORKERS}")
    print()
    
    start_time = time.time()
    
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_sample, samples),
            total=len(samples),
            desc="Processing",
            unit="sample"
        ))
    
    elapsed = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if r['success'] and not r['skipped'])
    skipped = sum(1 for r in results if r['skipped'])
    failed = sum(1 for r in results if not r['success'])
    
    print()
    print("✓ Feature extraction complete")
    print()
    
    return {
        'total': len(samples),
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'elapsed': elapsed,
        'results': results
    }


def print_statistics(stats, samples):
    """Print extraction statistics"""
    print("=" * 60)
    print("  Feature Extraction Statistics")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Skipped (already processed): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {100 * (stats['successful'] + stats['skipped']) / stats['total']:.1f}%")
    print()
    
    # Time statistics
    print(f"Processing time: {stats['elapsed']:.1f}s")
    print(f"Per sample: {stats['elapsed'] / stats['total']:.2f}s")
    print(f"Throughput: {stats['total'] / stats['elapsed']:.1f} samples/sec")
    print()
    
    # Storage statistics
    mel_size = sum(os.path.getsize(os.path.join(MEL_DIR, f)) 
                   for f in os.listdir(MEL_DIR) if f.endswith('.npy'))
    f0_size = sum(os.path.getsize(os.path.join(F0_DIR, f)) 
                  for f in os.listdir(F0_DIR) if f.endswith('.npy'))
    energy_size = sum(os.path.getsize(os.path.join(ENERGY_DIR, f)) 
                      for f in os.listdir(ENERGY_DIR) if f.endswith('.npy'))
    
    total_size = mel_size + f0_size + energy_size
    
    print("Storage usage:")
    print(f"  Mel-spectrograms: {mel_size / 1e9:.2f} GB")
    print(f"  F0 contours: {f0_size / 1e9:.2f} GB")
    print(f"  Energy values: {energy_size / 1e9:.2f} GB")
    print(f"  Total: {total_size / 1e9:.2f} GB")
    print()
    
    # Show failures if any
    if stats['failed'] > 0:
        print("Failed samples:")
        for result in stats['results']:
            if not result['success']:
                print(f"  {result['id']}: {result['error']}")
        print()
    
    print("Next steps:")
    print("  1. Verify feature quality with validation script")
    print("  2. Continue to Day 18 (phoneme conversion & alignment)")
    print("  3. Create training manifest")
    print()


def main():
    """Main execution"""
    print_header()
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory not found: {INPUT_DIR}")
        print("Please run Day 16 preprocessing first:")
        print("  ./scripts/download_ljspeech.sh")
        print("  python3 scripts/preprocess_ljspeech.py")
        sys.exit(1)
    
    # Check if audio directory exists
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Error: Audio directory not found: {AUDIO_DIR}")
        print("Please run Day 16 preprocessing first.")
        sys.exit(1)
    
    # Create output directories
    create_directories()
    
    # Load metadata
    samples = load_metadata()
    
    if len(samples) == 0:
        print("❌ Error: No samples found in metadata directory")
        sys.exit(1)
    
    # Process samples
    stats = process_samples_parallel(samples)
    
    # Print statistics
    print_statistics(stats, samples)
    
    # Exit code
    if stats['failed'] > 0:
        print(f"⚠️  Warning: {stats['failed']} samples failed")
        sys.exit(1)
    else:
        print("✓ All features extracted successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
