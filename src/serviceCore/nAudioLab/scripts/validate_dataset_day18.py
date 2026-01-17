#!/usr/bin/env python3
"""
Day 18: Final Dataset Validation
AudioLabShimmy - Complete Preprocessing Validation

Validates the complete preprocessed dataset:
1. All audio files present and valid
2. All features extracted (mel, F0, energy)
3. All phoneme sequences created
4. All alignments complete
5. Training manifest valid
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DIR = "data/datasets/ljspeech_processed"
MANIFEST_PATH = os.path.join(INPUT_DIR, "training_manifest.json")


def print_header():
    """Print script header"""
    print("=" * 60)
    print("  Complete Dataset Validation (Day 18)")
    print("  AudioLabShimmy - Final QA Check")
    print("=" * 60)
    print()


def load_manifest():
    """Load training manifest"""
    print("Loading training manifest...")
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌ Error: Training manifest not found: {MANIFEST_PATH}")
        return None
    
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    print(f"✓ Loaded manifest with {len(manifest['samples'])} samples")
    print()
    return manifest


def validate_sample(sample):
    """Validate a single sample has all required components"""
    sample_id = sample['id']
    issues = []
    
    # Check audio file
    audio_path = os.path.join(INPUT_DIR, sample['audio_path'])
    if not os.path.exists(audio_path):
        issues.append(f"Missing audio: {audio_path}")
    
    # Check mel-spectrogram
    mel_path = os.path.join(INPUT_DIR, sample['mel_path'])
    if not os.path.exists(mel_path):
        issues.append(f"Missing mel: {mel_path}")
    else:
        try:
            mel = np.load(mel_path)
            if mel.shape[1] != 128:
                issues.append(f"Invalid mel shape: {mel.shape}, expected [time, 128]")
        except Exception as e:
            issues.append(f"Error loading mel: {str(e)}")
    
    # Check F0
    f0_path = os.path.join(INPUT_DIR, sample['f0_path'])
    if not os.path.exists(f0_path):
        issues.append(f"Missing F0: {f0_path}")
    else:
        try:
            f0 = np.load(f0_path)
            if len(f0.shape) != 1:
                issues.append(f"Invalid F0 shape: {f0.shape}, expected [time]")
        except Exception as e:
            issues.append(f"Error loading F0: {str(e)}")
    
    # Check energy
    energy_path = os.path.join(INPUT_DIR, sample['energy_path'])
    if not os.path.exists(energy_path):
        issues.append(f"Missing energy: {energy_path}")
    else:
        try:
            energy = np.load(energy_path)
            if len(energy.shape) != 1:
                issues.append(f"Invalid energy shape: {energy.shape}, expected [time]")
        except Exception as e:
            issues.append(f"Error loading energy: {str(e)}")
    
    # Check phoneme sequence
    phoneme_path = os.path.join(INPUT_DIR, sample['phoneme_path'])
    if not os.path.exists(phoneme_path):
        issues.append(f"Missing phonemes: {phoneme_path}")
    else:
        try:
            with open(phoneme_path, 'r') as f:
                phonemes = f.read().strip()
            if not phonemes:
                issues.append("Empty phoneme sequence")
        except Exception as e:
            issues.append(f"Error loading phonemes: {str(e)}")
    
    # Check alignment
    alignment_path = os.path.join(INPUT_DIR, sample['alignment_path'])
    if not os.path.exists(alignment_path):
        issues.append(f"Missing alignment: {alignment_path}")
    else:
        try:
            durations = np.load(alignment_path)
            if len(durations.shape) != 1:
                issues.append(f"Invalid duration shape: {durations.shape}, expected [phonemes]")
        except Exception as e:
            issues.append(f"Error loading durations: {str(e)}")
    
    return {
        'id': sample_id,
        'valid': len(issues) == 0,
        'issues': issues
    }


def validate_dataset(manifest):
    """Validate all samples in the dataset"""
    print("=" * 60)
    print("  Validating Complete Dataset")
    print("=" * 60)
    print()
    
    samples = manifest['samples']
    results = []
    
    for sample in tqdm(samples, desc="Validating"):
        result = validate_sample(sample)
        results.append(result)
    
    # Analyze results
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count
    
    print()
    print(f"✓ Validation complete")
    print(f"  Valid samples: {valid_count}/{len(results)}")
    print(f"  Invalid samples: {invalid_count}")
    print()
    
    # Show first 10 invalid samples
    if invalid_count > 0:
        print("Sample issues (showing first 10):")
        count = 0
        for result in results:
            if not result['valid']:
                print(f"\n  {result['id']}:")
                for issue in result['issues']:
                    print(f"    - {issue}")
                count += 1
                if count >= 10:
                    break
        print()
    
    return valid_count, invalid_count, results


def check_feature_alignment(manifest):
    """Check that features are properly aligned"""
    print("=" * 60)
    print("  Checking Feature Alignment")
    print("=" * 60)
    print()
    
    alignment_issues = 0
    
    # Sample a few files to check alignment
    samples = manifest['samples'][:min(100, len(manifest['samples']))]
    
    for sample in tqdm(samples, desc="Checking alignment"):
        try:
            # Load features
            mel = np.load(os.path.join(INPUT_DIR, sample['mel_path']))
            f0 = np.load(os.path.join(INPUT_DIR, sample['f0_path']))
            energy = np.load(os.path.join(INPUT_DIR, sample['energy_path']))
            durations = np.load(os.path.join(INPUT_DIR, sample['alignment_path']))
            
            # Check time dimensions match
            mel_frames = mel.shape[0]
            f0_frames = len(f0)
            energy_frames = len(energy)
            
            if not (mel_frames == f0_frames == energy_frames):
                alignment_issues += 1
        
        except Exception:
            alignment_issues += 1
    
    print()
    if alignment_issues == 0:
        print("✓ All features properly aligned")
    else:
        print(f"⚠️  Found {alignment_issues} alignment issues")
    print()
    
    return alignment_issues


def calculate_statistics(manifest):
    """Calculate dataset statistics"""
    print("=" * 60)
    print("  Dataset Statistics")
    print("=" * 60)
    print()
    
    total_samples = len(manifest['samples'])
    total_duration = 0.0
    
    # Calculate total duration (sample ~100 files)
    sample_size = min(100, total_samples)
    samples_to_check = manifest['samples'][:sample_size]
    
    durations = []
    for sample in samples_to_check:
        try:
            mel = np.load(os.path.join(INPUT_DIR, sample['mel_path']))
            # Each frame is 512 samples at 48kHz = 0.0107 seconds
            duration = mel.shape[0] * 512 / 48000
            durations.append(duration)
        except:
            continue
    
    if durations:
        avg_duration = np.mean(durations)
        estimated_total = avg_duration * total_samples
    else:
        estimated_total = 0
    
    print(f"Total samples: {total_samples}")
    print(f"Sample rate: {manifest['sample_rate']} Hz")
    print(f"Mel bins: {manifest['n_mels']}")
    print(f"Estimated total duration: {estimated_total / 3600:.1f} hours")
    print()
    
    # Storage usage
    print("Storage usage:")
    dirs_to_check = ['audio', 'mels', 'f0', 'energy', 'phonemes', 'alignments']
    total_size = 0
    
    for dir_name in dirs_to_check:
        dir_path = os.path.join(INPUT_DIR, dir_name)
        if os.path.exists(dir_path):
            size = sum(os.path.getsize(os.path.join(dir_path, f))
                      for f in os.listdir(dir_path))
            total_size += size
            print(f"  {dir_name}: {size / 1e9:.2f} GB")
    
    print(f"  Total: {total_size / 1e9:.2f} GB")
    print()


def generate_report(valid_count, invalid_count, alignment_issues, manifest):
    """Generate final validation report"""
    print("=" * 60)
    print("  Final Validation Report")
    print("=" * 60)
    print()
    
    total_samples = len(manifest['samples'])
    
    print(f"Dataset: {manifest['dataset']}")
    print(f"Total samples: {total_samples}")
    print()
    
    print("Validation Results:")
    print(f"  ✓ Valid samples: {valid_count} ({100*valid_count/total_samples:.1f}%)")
    if invalid_count > 0:
        print(f"  ✗ Invalid samples: {invalid_count} ({100*invalid_count/total_samples:.1f}%)")
    print()
    
    if alignment_issues > 0:
        print(f"  ⚠️  Alignment issues: {alignment_issues}")
        print()
    
    # Overall status
    all_valid = (invalid_count == 0 and alignment_issues == 0)
    
    print("=" * 60)
    if all_valid:
        print("✓ DATASET READY FOR TRAINING")
        print("All samples validated successfully!")
    else:
        print("⚠️  DATASET HAS ISSUES")
        print("Some samples need attention before training.")
    print("=" * 60)
    print()
    
    # Save report
    report_path = os.path.join(INPUT_DIR, "validation_report_day18.txt")
    with open(report_path, 'w') as f:
        f.write("Dataset Validation Report - Day 18\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {manifest['dataset']}\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Valid samples: {valid_count}\n")
        f.write(f"Invalid samples: {invalid_count}\n")
        f.write(f"Alignment issues: {alignment_issues}\n\n")
        
        if all_valid:
            f.write("Status: READY FOR TRAINING\n")
        else:
            f.write("Status: HAS ISSUES\n")
    
    print(f"Report saved to: {report_path}")
    print()
    
    return all_valid


def main():
    """Main execution"""
    print_header()
    
    # Load manifest
    manifest = load_manifest()
    if manifest is None:
        sys.exit(1)
    
    # Validate all samples
    valid_count, invalid_count, results = validate_dataset(manifest)
    
    # Check feature alignment
    alignment_issues = check_feature_alignment(manifest)
    
    # Calculate statistics
    calculate_statistics(manifest)
    
    # Generate report
    all_valid = generate_report(valid_count, invalid_count, alignment_issues, manifest)
    
    # Exit code
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
