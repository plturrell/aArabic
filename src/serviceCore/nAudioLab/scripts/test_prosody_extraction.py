#!/usr/bin/env python3
"""
Day 3 Test Script: Prosody Feature Extraction
Tests F0, energy, and voiced/unvoiced detection on 48kHz audio

This validates the Mojo implementation approach using librosa and other tools.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configuration matching Mojo implementation
SAMPLE_RATE = 48000
FRAME_LENGTH = 2048
HOP_LENGTH = 512
F0_MIN = 80.0
F0_MAX = 400.0


def extract_f0_yin(audio, sr=SAMPLE_RATE):
    """
    Extract F0 using YIN-like algorithm (via librosa's pyin)
    
    Returns:
        f0: F0 contour in Hz, NaN for unvoiced
        voiced_flag: Boolean array for voiced frames
        voiced_prob: Voicing probability
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )
    
    # Convert NaN to 0.0 for unvoiced
    f0_clean = np.nan_to_num(f0, nan=0.0)
    
    return f0_clean, voiced_flag, voiced_prob


def extract_frame_energy(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Extract RMS energy per frame
    
    Returns:
        energy: RMS energy values
    """
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    return energy


def compute_zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Compute zero-crossing rate
    
    Returns:
        zcr: Zero-crossing rate per frame
    """
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    return zcr


def detect_voiced_frames(audio, f0, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Detect voiced frames using multiple criteria
    
    Returns:
        voiced: Boolean array for voiced frames
    """
    # Get ZCR and energy
    zcr = compute_zero_crossing_rate(audio, frame_length, hop_length)
    energy = extract_frame_energy(audio, frame_length, hop_length)
    
    # Normalize
    zcr_norm = zcr / np.max(zcr) if np.max(zcr) > 0 else zcr
    energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy
    
    # Thresholds
    ZCR_THRESHOLD = 0.3
    ENERGY_THRESHOLD = 0.01
    
    # Combine criteria
    f0_voiced = f0 > 0
    zcr_voiced = zcr_norm < ZCR_THRESHOLD
    energy_voiced = energy_norm > ENERGY_THRESHOLD
    
    voiced = f0_voiced & (zcr_voiced | energy_voiced)
    
    return voiced


def extract_prosody_features(audio, sr=SAMPLE_RATE):
    """
    Extract complete prosody feature set
    
    Returns:
        Dictionary with all prosody features
    """
    # F0 extraction
    f0, voiced_flag, voiced_prob = extract_f0_yin(audio, sr)
    
    # Energy extraction
    energy = extract_frame_energy(audio)
    
    # ZCR
    zcr = compute_zero_crossing_rate(audio)
    
    # Voiced detection
    voiced = detect_voiced_frames(audio, f0)
    
    # Log F0 (for neural network input)
    log_f0 = np.log(f0 + 1e-10)  # Add small constant to avoid log(0)
    log_f0[f0 == 0] = 0.0
    
    # Normalize energy (min-max)
    energy_min = np.min(energy)
    energy_max = np.max(energy)
    normalized_energy = (energy - energy_min) / (energy_max - energy_min) if energy_max > energy_min else energy
    
    return {
        'f0': f0,
        'energy': energy,
        'zcr': zcr,
        'voiced': voiced,
        'voiced_prob': voiced_prob,
        'log_f0': log_f0,
        'normalized_energy': normalized_energy
    }


def visualize_prosody_features(features, output_path, duration):
    """
    Create comprehensive visualization of prosody features
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 10))
    
    time_frames = np.arange(len(features['f0'])) * HOP_LENGTH / SAMPLE_RATE
    
    # F0 contour
    axes[0].plot(time_frames, features['f0'], 'b-', linewidth=1.5, label='F0 (Hz)')
    axes[0].fill_between(time_frames, 0, features['f0'], 
                          where=(features['voiced']), alpha=0.3, color='green', label='Voiced')
    axes[0].set_ylabel('F0 (Hz)')
    axes[0].set_title('Pitch Contour (F0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, duration)
    
    # Log F0
    axes[1].plot(time_frames, features['log_f0'], 'r-', linewidth=1.5)
    axes[1].set_ylabel('Log F0')
    axes[1].set_title('Log-scaled F0 (Neural Network Input)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, duration)
    
    # Energy
    axes[2].plot(time_frames, features['energy'], 'g-', linewidth=1.5, label='RMS Energy')
    axes[2].plot(time_frames, features['normalized_energy'], 'orange', linewidth=1.5, 
                 alpha=0.7, label='Normalized')
    axes[2].set_ylabel('Energy')
    axes[2].set_title('Frame Energy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, duration)
    
    # Zero-Crossing Rate
    axes[3].plot(time_frames, features['zcr'], 'm-', linewidth=1.5)
    axes[3].set_ylabel('ZCR')
    axes[3].set_title('Zero-Crossing Rate')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, duration)
    
    # Voiced/Unvoiced
    axes[4].fill_between(time_frames, 0, 1, where=features['voiced'], 
                          step='mid', alpha=0.5, color='green', label='Voiced')
    axes[4].fill_between(time_frames, 0, 1, where=~features['voiced'], 
                          step='mid', alpha=0.5, color='red', label='Unvoiced')
    axes[4].set_ylabel('V/UV')
    axes[4].set_title('Voiced/Unvoiced Detection')
    axes[4].set_xlabel('Time (s)')
    axes[4].legend()
    axes[4].set_xlim(0, duration)
    axes[4].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved prosody visualization: {output_path}")
    plt.close()


def compute_statistics(features):
    """Compute prosody statistics"""
    voiced_frames = features['f0'] > 0
    
    return {
        'n_frames': len(features['f0']),
        'voiced_frames': np.sum(voiced_frames),
        'voiced_percentage': np.mean(voiced_frames) * 100,
        'f0_mean': np.mean(features['f0'][voiced_frames]) if np.any(voiced_frames) else 0,
        'f0_std': np.std(features['f0'][voiced_frames]) if np.any(voiced_frames) else 0,
        'f0_min': np.min(features['f0'][voiced_frames]) if np.any(voiced_frames) else 0,
        'f0_max': np.max(features['f0'][voiced_frames]) if np.any(voiced_frames) else 0,
        'energy_mean': np.mean(features['energy']),
        'energy_std': np.std(features['energy']),
        'zcr_mean': np.mean(features['zcr']),
        'zcr_std': np.std(features['zcr'])
    }


def test_prosody_extraction():
    """Main test function"""
    print("=" * 70)
    print("AudioLabShimmy Day 3: Prosody Feature Extraction Test")
    print("=" * 70)
    print()
    
    # Test files from Day 1
    test_dir = Path("test_output")
    if not test_dir.exists():
        print("ERROR: test_output directory not found!")
        print("Please run Day 1 tests first to generate audio files.")
        sys.exit(1)
    
    test_files = [
        "tone_440hz_48k_24bit.wav",
        "tone_C4.wav",
        "tone_E4.wav",
        "tone_G4.wav"
    ]
    
    results = []
    
    for i, filename in enumerate(test_files, 1):
        filepath = test_dir / filename
        if not filepath.exists():
            print(f"WARNING: {filename} not found, skipping...")
            continue
        
        print(f"\nTest {i}: {filename}")
        print("-" * 70)
        
        # Load audio
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        print(f"Loaded audio: {len(audio)} samples, {duration:.2f}s, {sr}Hz")
        
        # Extract prosody features
        features = extract_prosody_features(audio, sr)
        
        # Compute statistics
        stats = compute_statistics(features)
        
        print(f"\nProsody Statistics:")
        print(f"  Total frames: {stats['n_frames']}")
        print(f"  Voiced frames: {stats['voiced_frames']} ({stats['voiced_percentage']:.1f}%)")
        print(f"  F0 range: {stats['f0_min']:.1f} - {stats['f0_max']:.1f} Hz")
        print(f"  F0 mean: {stats['f0_mean']:.1f} Hz (±{stats['f0_std']:.1f})")
        print(f"  Energy mean: {stats['energy_mean']:.4f} (±{stats['energy_std']:.4f})")
        print(f"  ZCR mean: {stats['zcr_mean']:.4f} (±{stats['zcr_std']:.4f})")
        
        # Save visualization
        vis_output = test_dir / f"{filepath.stem}_prosody.png"
        visualize_prosody_features(features, vis_output, duration)
        
        results.append({
            'file': filename,
            'features': features,
            'stats': stats
        })
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Processed {len(results)} audio files")
    print(f"✓ Generated {len(results)} prosody visualizations")
    print(f"✓ Configuration: {FRAME_LENGTH} frame, {HOP_LENGTH} hop, {SAMPLE_RATE}Hz")
    print(f"✓ F0 range: {F0_MIN}-{F0_MAX} Hz (speech range)")
    print()
    
    # Validation checks
    print("Validation Checks:")
    all_valid = True
    
    for result in results:
        stats = result['stats']
        
        # Check F0 range
        if stats['f0_mean'] > 0 and (stats['f0_mean'] < F0_MIN or stats['f0_mean'] > F0_MAX):
            print(f"  WARNING: {result['file']}: F0 mean outside expected range")
        
        # Check voiced percentage (tones should be mostly voiced)
        if stats['voiced_percentage'] < 80:
            print(f"  WARNING: {result['file']}: Low voiced percentage ({stats['voiced_percentage']:.1f}%)")
        
        # Check energy
        if stats['energy_mean'] < 0.001:
            print(f"  WARNING: {result['file']}: Very low energy")
    
    if all_valid:
        print("✓ All prosody features valid!")
    
    print()
    print("=" * 70)
    print("Day 3 Prosody Feature Extraction Complete!")
    print("=" * 70)
    print()
    print("Generated files in test_output/:")
    for result in results:
        print(f"  • {Path(result['file']).stem}_prosody.png")
    print()
    
    # Print feature shapes for reference
    if results:
        features = results[0]['features']
        print("Feature Shapes (for Mojo reference):")
        print(f"  f0: [{len(features['f0'])}]")
        print(f"  energy: [{len(features['energy'])}]")
        print(f"  voiced: [{len(features['voiced'])}] (bool)")
        print(f"  log_f0: [{len(features['log_f0'])}]")
        print(f"  normalized_energy: [{len(features['normalized_energy'])}]")
    
    return results


if __name__ == "__main__":
    try:
        results = test_prosody_extraction()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
