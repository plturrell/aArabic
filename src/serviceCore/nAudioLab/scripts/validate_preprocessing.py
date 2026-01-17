#!/usr/bin/env python3
"""
Validate Preprocessing
=====================

Validates the preprocessed LJSpeech dataset to ensure quality.
"""

import sys
from pathlib import Path
import json

def print_header(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_directory_structure():
    """Check if directory structure is correct."""
    print("Checking directory structure...")
    
    required_dirs = [
        "data/datasets/ljspeech_processed",
        "data/datasets/ljspeech_processed/audio",
        "data/datasets/ljspeech_processed/metadata",
    ]
    
    for d in required_dirs:
        if not Path(d).exists():
            print(f"✗ Missing: {d}")
            return False
        print(f"✓ Found: {d}")
    
    return True

def validate_audio_files():
    """Validate audio files."""
    print("\nValidating audio files...")
    
    audio_dir = Path("data/datasets/ljspeech_processed/audio")
    if not audio_dir.exists():
        print("✗ Audio directory not found")
        return False
    
    wav_files = list(audio_dir.glob("*.wav"))
    print(f"✓ Found {len(wav_files)} audio files")
    
    if len(wav_files) == 0:
        print("✗ No audio files found")
        return False
    
    return True

def validate_metadata():
    """Validate metadata files."""
    print("\nValidating metadata...")
    
    metadata_dir = Path("data/datasets/ljspeech_processed/metadata")
    if not metadata_dir.exists():
        print("✗ Metadata directory not found")
        return False
    
    json_files = list(metadata_dir.glob("*.json"))
    print(f"✓ Found {len(json_files)} metadata files")
    
    # Check a sample metadata file
    if json_files:
        sample = json_files[0]
        try:
            with open(sample) as f:
                data = json.load(f)
            print(f"✓ Sample metadata structure valid")
            print(f"  Keys: {list(data.keys())}")
        except Exception as e:
            print(f"✗ Error reading metadata: {e}")
            return False
    
    return True

def main():
    """Main validation."""
    print_header("Preprocessing Validation")
    print("AudioLabShimmy - Day 16")
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Audio Files", validate_audio_files),
        ("Metadata", validate_metadata),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Validation Summary")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ Preprocessing validation complete!")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
