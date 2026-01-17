#!/usr/bin/env python3
"""
Day 17: Feature Validation Script
AudioLabShimmy - Validate extracted mel-spectrograms, F0, and energy

This script validates:
1. Feature file existence
2. Feature dimensions and shapes
3. Value ranges and statistics
4. Data quality checks
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = "data/datasets/ljspeech_processed"
MEL_DIR = os.path.join(INPUT_DIR, "mels")
F0_DIR = os.path.join(INPUT_DIR, "f0")
ENERGY_DIR = os.path.join(INPUT_DIR, "energy")
METADATA_DIR = os.path.join(INPUT_DIR, "metadata")
OUTPUT_DIR = "data/datasets/ljspeech_processed/validation_reports"

# Expected parameters
EXPECTED_N_MELS = 128
MIN_F0 = 50.0    # Hz
MAX_F0 = 500.0   # Hz
MIN_ENERGY = 0.0
MAX_ENERGY = 100.0


def print_header():
    """Print script header"""
    print("=" * 60)
    print("  Feature Validation (Day 17)")
    print("  AudioLabShimmy - Quality Assurance")
    print("=" * 60)
    print()


def create_output_dir():
    """Create output directory for validation reports"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_metadata():
    """Load all metadata files"""
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


def validate_file_existence(samples):
    """Check if all feature files exist"""
    print("Validating file existence...")
    
    missing_mels = []
    missing_f0s = []
    missing_energies = []
    
    for sample in samples:
        sample_id = sample['id']
        
        # Check mel file
        mel_path = os.path.join(INPUT_DIR, sample.get('mel_path', f"mels/{sample_id}.npy"))
        if not os.path.exists(mel_path):
            missing_mels.append(sample_id)
        
        # Check F0 file
        f0_path = os.path.join(INPUT_DIR, sample.get('f0_path', f"f0/{sample_id}.npy"))
        if not os.path.exists(f0_path):
            missing_f0s.append(sample_id)
        
        # Check energy file
        energy_path = os.path.join(INPUT_DIR, sample.get('energy_path', f"energy/{sample_id}.npy"))
        if not os.path.exists(energy_path):
            missing_energies.append(sample_id)
    
    # Report
    total = len(samples)
    mel_present = total - len(missing_mels)
    f0_present = total - len(missing_f0s)
    energy_present = total - len(missing_energies)
    
    print(f"Mel-spectrograms: {mel_present}/{total} ({100*mel_present/total:.1f}%)")
    print(f"F0 contours: {f0_present}/{total} ({100*f0_present/total:.1f}%)")
    print(f"Energy values: {energy_present}/{total} ({100*energy_present/total:.1f}%)")
    print()
    
    all_present = len(missing_mels) == 0 and len(missing_f0s) == 0 and len(missing_energies) == 0
    
    return {
        'all_present': all_present,
        'missing_mels': missing_mels,
        'missing_f0s': missing_f0s,
        'missing_energies': missing_energies
    }


def validate_feature_shapes(samples, num_samples=100):
    """Validate feature dimensions and shapes"""
    print(f"Validating feature shapes (sampling {num_samples} files)...")
    
    issues = []
    mel_shapes = []
    f0_shapes = []
    energy_shapes = []
    
    # Sample random files
    import random
    sampled = random.sample(samples, min(num_samples, len(samples)))
    
    for sample in sampled:
        sample_id = sample['id']
        
        try:
            # Load mel
            mel_path = os.path.join(INPUT_DIR, sample.get('mel_path', f"mels/{sample_id}.npy"))
            if os.path.exists(mel_path):
                mel = np.load(mel_path)
                mel_shapes.append(mel.shape)
                
                # Check mel bins
                if len(mel.shape) != 2:
                    issues.append(f"{sample_id}: Mel has wrong dimensions {mel.shape}")
                elif mel.shape[1] != EXPECTED_N_MELS:
                    issues.append(f"{sample_id}: Mel has {mel.shape[1]} bins, expected {EXPECTED_N_MELS}")
            
            # Load F0
            f0_path = os.path.join(INPUT_DIR, sample.get('f0_path', f"f0/{sample_id}.npy"))
            if os.path.exists(f0_path):
                f0 = np.load(f0_path)
                f0_shapes.append(f0.shape)
                
                if len(f0.shape) != 1:
                    issues.append(f"{sample_id}: F0 has wrong dimensions {f0.shape}")
            
            # Load energy
            energy_path = os.path.join(INPUT_DIR, sample.get('energy_path', f"energy/{sample_id}.npy"))
            if os.path.exists(energy_path):
                energy = np.load(energy_path)
                energy_shapes.append(energy.shape)
                
                if len(energy.shape) != 1:
                    issues.append(f"{sample_id}: Energy has wrong dimensions {energy.shape}")
        
        except Exception as e:
            issues.append(f"{sample_id}: Error loading features - {str(e)}")
    
    print(f"Validated {len(sampled)} samples")
    if len(issues) == 0:
        print("✓ All feature shapes are correct")
    else:
        print(f"⚠️  Found {len(issues)} shape issues")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
    print()
    
    return {
        'issues': issues,
        'mel_shapes': mel_shapes,
        'f0_shapes': f0_shapes,
        'energy_shapes': energy_shapes
    }


def validate_feature_ranges(samples, num_samples=100):
    """Validate feature value ranges"""
    print(f"Validating feature value ranges (sampling {num_samples} files)...")
    
    mel_stats = defaultdict(list)
    f0_stats = defaultdict(list)
    energy_stats = defaultdict(list)
    
    issues = []
    
    # Sample random files
    import random
    sampled = random.sample(samples, min(num_samples, len(samples)))
    
    for sample in sampled:
        sample_id = sample['id']
        
        try:
            # Load and analyze mel
            mel_path = os.path.join(INPUT_DIR, sample.get('mel_path', f"mels/{sample_id}.npy"))
            if os.path.exists(mel_path):
                mel = np.load(mel_path)
                mel_stats['min'].append(np.min(mel))
                mel_stats['max'].append(np.max(mel))
                mel_stats['mean'].append(np.mean(mel))
                mel_stats['std'].append(np.std(mel))
                
                # Check for NaN/Inf
                if np.any(np.isnan(mel)) or np.any(np.isinf(mel)):
                    issues.append(f"{sample_id}: Mel contains NaN/Inf values")
            
            # Load and analyze F0
            f0_path = os.path.join(INPUT_DIR, sample.get('f0_path', f"f0/{sample_id}.npy"))
            if os.path.exists(f0_path):
                f0 = np.load(f0_path)
                f0_nonzero = f0[f0 > 0]  # Exclude unvoiced frames
                
                if len(f0_nonzero) > 0:
                    f0_stats['min'].append(np.min(f0_nonzero))
                    f0_stats['max'].append(np.max(f0_nonzero))
                    f0_stats['mean'].append(np.mean(f0_nonzero))
                    f0_stats['std'].append(np.std(f0_nonzero))
                
                # Check for unreasonable values
                if np.any((f0_nonzero < MIN_F0) | (f0_nonzero > MAX_F0)):
                    issues.append(f"{sample_id}: F0 out of range [{MIN_F0}, {MAX_F0}] Hz")
                
                if np.any(np.isnan(f0)) or np.any(np.isinf(f0)):
                    issues.append(f"{sample_id}: F0 contains NaN/Inf values")
            
            # Load and analyze energy
            energy_path = os.path.join(INPUT_DIR, sample.get('energy_path', f"energy/{sample_id}.npy"))
            if os.path.exists(energy_path):
                energy = np.load(energy_path)
                energy_stats['min'].append(np.min(energy))
                energy_stats['max'].append(np.max(energy))
                energy_stats['mean'].append(np.mean(energy))
                energy_stats['std'].append(np.std(energy))
                
                # Check for negative values
                if np.any(energy < 0):
                    issues.append(f"{sample_id}: Energy has negative values")
                
                if np.any(np.isnan(energy)) or np.any(np.isinf(energy)):
                    issues.append(f"{sample_id}: Energy contains NaN/Inf values")
        
        except Exception as e:
            issues.append(f"{sample_id}: Error analyzing features - {str(e)}")
    
    # Print statistics
    print("\nMel-spectrogram statistics:")
    for key in ['min', 'max', 'mean', 'std']:
        if mel_stats[key]:
            print(f"  {key}: {np.mean(mel_stats[key]):.4f} ± {np.std(mel_stats[key]):.4f}")
    
    print("\nF0 statistics (Hz, voiced frames only):")
    for key in ['min', 'max', 'mean', 'std']:
        if f0_stats[key]:
            print(f"  {key}: {np.mean(f0_stats[key]):.2f} ± {np.std(f0_stats[key]):.2f}")
    
    print("\nEnergy statistics:")
    for key in ['min', 'max', 'mean', 'std']:
        if energy_stats[key]:
            print(f"  {key}: {np.mean(energy_stats[key]):.4f} ± {np.std(energy_stats[key]):.4f}")
    
    print()
    if len(issues) == 0:
        print("✓ All feature values are in expected ranges")
    else:
        print(f"⚠️  Found {len(issues)} value range issues")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
    print()
    
    return {
        'mel_stats': mel_stats,
        'f0_stats': f0_stats,
        'energy_stats': energy_stats,
        'issues': issues
    }


def create_visualization(samples, num_samples=5):
    """Create visualization of sample features"""
    print(f"Creating visualizations for {num_samples} samples...")
    
    import random
    sampled = random.sample(samples, min(num_samples, len(samples)))
    
    for sample in sampled:
        sample_id = sample['id']
        
        try:
            # Load features
            mel_path = os.path.join(INPUT_DIR, sample.get('mel_path', f"mels/{sample_id}.npy"))
            f0_path = os.path.join(INPUT_DIR, sample.get('f0_path', f"f0/{sample_id}.npy"))
            energy_path = os.path.join(INPUT_DIR, sample.get('energy_path', f"energy/{sample_id}.npy"))
            
            mel = np.load(mel_path) if os.path.exists(mel_path) else None
            f0 = np.load(f0_path) if os.path.exists(f0_path) else None
            energy = np.load(energy_path) if os.path.exists(energy_path) else None
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot mel-spectrogram
            if mel is not None:
                im = axes[0].imshow(mel.T, aspect='auto', origin='lower', cmap='viridis')
                axes[0].set_title(f'Mel-Spectrogram - {sample_id}')
                axes[0].set_xlabel('Time Frame')
                axes[0].set_ylabel('Mel Bin')
                plt.colorbar(im, ax=axes[0])
            
            # Plot F0
            if f0 is not None:
                time_axis = np.arange(len(f0))
                axes[1].plot(time_axis, f0, linewidth=0.5)
                axes[1].set_title('F0 Contour (Fundamental Frequency)')
                axes[1].set_xlabel('Time Frame')
                axes[1].set_ylabel('F0 (Hz)')
                axes[1].set_ylim([0, 400])
                axes[1].grid(True, alpha=0.3)
            
            # Plot energy
            if energy is not None:
                time_axis = np.arange(len(energy))
                axes[2].plot(time_axis, energy, linewidth=0.5, color='green')
                axes[2].set_title('Frame Energy')
                axes[2].set_xlabel('Time Frame')
                axes[2].set_ylabel('Energy')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(OUTPUT_DIR, f"{sample_id}_features.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
        
        except Exception as e:
            print(f"  Error creating visualization for {sample_id}: {str(e)}")
    
    print(f"✓ Saved visualizations to {OUTPUT_DIR}")
    print()


def generate_report(existence_result, shape_result, range_result, samples):
    """Generate validation report"""
    print("=" * 60)
    print("  Validation Report")
    print("=" * 60)
    print()
    
    total = len(samples)
    
    # File existence
    print("File Existence:")
    print(f"  Total samples: {total}")
    print(f"  Missing mel files: {len(existence_result['missing_mels'])}")
    print(f"  Missing F0 files: {len(existence_result['missing_f0s'])}")
    print(f"  Missing energy files: {len(existence_result['missing_energies'])}")
    
    if existence_result['all_present']:
        print("  ✓ PASS: All feature files present")
    else:
        print("  ✗ FAIL: Some feature files missing")
    print()
    
    # Shape validation
    print("Feature Shapes:")
    if len(shape_result['issues']) == 0:
        print("  ✓ PASS: All features have correct shapes")
    else:
        print(f"  ✗ FAIL: {len(shape_result['issues'])} shape issues found")
    print()
    
    # Value range validation
    print("Feature Values:")
    if len(range_result['issues']) == 0:
        print("  ✓ PASS: All feature values in expected ranges")
    else:
        print(f"  ✗ FAIL: {len(range_result['issues'])} value range issues found")
    print()
    
    # Overall result
    all_passed = (existence_result['all_present'] and
                  len(shape_result['issues']) == 0 and
                  len(range_result['issues']) == 0)
    
    print("=" * 60)
    if all_passed:
        print("✓ VALIDATION PASSED")
        print("All features are correctly extracted and validated!")
    else:
        print("✗ VALIDATION FAILED")
        print("Some issues found. Please review the details above.")
    print("=" * 60)
    print()
    
    # Save report to file
    report_path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    with open(report_path, 'w') as f:
        f.write("Feature Validation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Missing mel files: {len(existence_result['missing_mels'])}\n")
        f.write(f"Missing F0 files: {len(existence_result['missing_f0s'])}\n")
        f.write(f"Missing energy files: {len(existence_result['missing_energies'])}\n\n")
        f.write(f"Shape issues: {len(shape_result['issues'])}\n")
        f.write(f"Value range issues: {len(range_result['issues'])}\n\n")
        
        if all_passed:
            f.write("Result: PASSED\n")
        else:
            f.write("Result: FAILED\n")
    
    print(f"Report saved to: {report_path}")
    print()
    
    return all_passed


def main():
    """Main execution"""
    print_header()
    
    # Check if directories exist
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    # Create output directory
    create_output_dir()
    
    # Load metadata
    samples = load_metadata()
    
    if len(samples) == 0:
        print("❌ Error: No samples found")
        sys.exit(1)
    
    # Run validations
    existence_result = validate_file_existence(samples)
    shape_result = validate_feature_shapes(samples, num_samples=100)
    range_result = validate_feature_ranges(samples, num_samples=100)
    
    # Create visualizations
    create_visualization(samples, num_samples=5)
    
    # Generate report
    all_passed = generate_report(existence_result, shape_result, range_result, samples)
    
    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
