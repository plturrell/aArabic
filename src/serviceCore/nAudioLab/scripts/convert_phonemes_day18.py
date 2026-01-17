#!/usr/bin/env python3
"""
Day 18: Phoneme Conversion & Forced Alignment
AudioLabShimmy - Text to Phoneme Pipeline

This script:
1. Converts text transcriptions to phoneme sequences
2. Prepares data for Montreal Forced Aligner (MFA)
3. Runs forced alignment to extract phoneme durations
4. Creates final training manifest

Uses Mojo implementations from Days 4-5 for text normalization and phonemization.
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
PHONEME_DIR = os.path.join(INPUT_DIR, "phonemes")
ALIGNMENT_DIR = os.path.join(INPUT_DIR, "alignments")
METADATA_DIR = os.path.join(INPUT_DIR, "metadata")
MFA_CORPUS_DIR = os.path.join(INPUT_DIR, "mfa_corpus")
MFA_OUTPUT_DIR = os.path.join(INPUT_DIR, "mfa_output")

# Processing
NUM_WORKERS = max(1, cpu_count() - 1)


def print_header():
    """Print script header"""
    print("=" * 60)
    print("  Phoneme Conversion & Forced Alignment (Day 18)")
    print("  AudioLabShimmy - Text Processing Pipeline")
    print("=" * 60)
    print()


def create_directories():
    """Create output directories"""
    print("Creating output directories...")
    dirs = [PHONEME_DIR, ALIGNMENT_DIR, MFA_CORPUS_DIR, MFA_OUTPUT_DIR]
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


def normalize_and_phonemize_text(text, sample_id):
    """
    Normalize text and convert to phonemes using Mojo implementations
    Calls: mojo/text/normalizer.mojo + mojo/text/phoneme.mojo
    """
    try:
        # Call Mojo text normalization and phonemization
        cmd = [
            "mojo", "run",
            "mojo/text/phoneme.mojo",
            "--text", text,
            "--output", os.path.join(PHONEME_DIR, f"{sample_id}.txt"),
            "--normalize"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Read generated phoneme sequence
            phoneme_file = os.path.join(PHONEME_DIR, f"{sample_id}.txt")
            if os.path.exists(phoneme_file):
                with open(phoneme_file, 'r') as f:
                    phonemes = f.read().strip()
                return True, phonemes, None
            else:
                return False, None, "Phoneme file not created"
        else:
            return False, None, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, None, "Timeout"
    except Exception as e:
        return False, None, str(e)


def process_sample_phonemes(sample):
    """
    Process a single sample: normalize text and convert to phonemes
    """
    sample_id = sample['id']
    text = sample.get('normalized_text', sample.get('text', ''))
    
    # Skip if already processed
    phoneme_path = os.path.join(PHONEME_DIR, f"{sample_id}.txt")
    if os.path.exists(phoneme_path):
        with open(phoneme_path, 'r') as f:
            phonemes = f.read().strip()
        return {
            'id': sample_id,
            'success': True,
            'skipped': True,
            'phonemes': phonemes,
            'error': None
        }
    
    # Normalize and phonemize
    success, phonemes, error = normalize_and_phonemize_text(text, sample_id)
    
    if not success:
        return {
            'id': sample_id,
            'success': False,
            'skipped': False,
            'phonemes': None,
            'error': f"Phonemization failed: {error}"
        }
    
    # Update metadata
    sample['phoneme_path'] = f"phonemes/{sample_id}.txt"
    sample['phoneme_sequence'] = phonemes
    
    # Save updated metadata
    metadata_path = os.path.join(METADATA_DIR, f"{sample_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(sample, f, indent=2)
    
    return {
        'id': sample_id,
        'success': True,
        'skipped': False,
        'phonemes': phonemes,
        'error': None
    }


def process_phonemes_parallel(samples):
    """
    Process all samples in parallel for phoneme conversion
    """
    print("=" * 60)
    print("  Converting Text to Phonemes")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Workers: {NUM_WORKERS}")
    print()
    
    start_time = time.time()
    
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_sample_phonemes, samples),
            total=len(samples),
            desc="Converting",
            unit="sample"
        ))
    
    elapsed = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if r['success'] and not r['skipped'])
    skipped = sum(1 for r in results if r['skipped'])
    failed = sum(1 for r in results if not r['success'])
    
    print()
    print("✓ Phoneme conversion complete")
    print()
    
    return {
        'total': len(samples),
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'elapsed': elapsed,
        'results': results
    }


def prepare_mfa_corpus(samples):
    """
    Prepare corpus for Montreal Forced Aligner
    MFA expects:
    - Audio files in corpus directory
    - Text transcription files with same name as audio
    """
    print("=" * 60)
    print("  Preparing MFA Corpus")
    print("=" * 60)
    print()
    
    for sample in tqdm(samples, desc="Preparing corpus"):
        sample_id = sample['id']
        
        # Copy/link audio file
        src_audio = os.path.join(INPUT_DIR, sample['audio_path'])
        dst_audio = os.path.join(MFA_CORPUS_DIR, f"{sample_id}.wav")
        
        if not os.path.exists(dst_audio):
            if os.path.exists(src_audio):
                # Create symlink to save space
                os.symlink(os.path.abspath(src_audio), dst_audio)
        
        # Create transcription file (phoneme sequence)
        phoneme_path = os.path.join(PHONEME_DIR, f"{sample_id}.txt")
        trans_path = os.path.join(MFA_CORPUS_DIR, f"{sample_id}.lab")
        
        if os.path.exists(phoneme_path):
            with open(phoneme_path, 'r') as f:
                phonemes = f.read().strip()
            
            # Write transcription file for MFA
            with open(trans_path, 'w') as f:
                f.write(phonemes)
    
    print(f"✓ Prepared {len(samples)} files for MFA")
    print()


def run_mfa_alignment():
    """
    Run Montreal Forced Aligner to get phoneme durations
    """
    print("=" * 60)
    print("  Running Montreal Forced Aligner")
    print("=" * 60)
    print()
    
    print("Checking MFA installation...")
    
    # Check if MFA is installed
    try:
        result = subprocess.run(
            ["mfa", "version"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ MFA version: {result.stdout.strip()}")
        else:
            print("❌ MFA not found. Installing...")
            install_mfa()
    except FileNotFoundError:
        print("❌ MFA not found. Installing...")
        install_mfa()
    
    print()
    print("Running forced alignment...")
    print("This may take 1-2 hours for 13,100 samples...")
    print()
    
    # Run MFA alignment
    cmd = [
        "mfa", "align",
        MFA_CORPUS_DIR,  # Corpus directory
        "english_us_arpa",  # Pretrained acoustic model
        "english_us_arpa",  # Pretrained dictionary
        MFA_OUTPUT_DIR,  # Output directory
        "--clean"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print()
            print("✓ Forced alignment complete")
            return True
        else:
            print()
            print(f"❌ MFA alignment failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running MFA: {str(e)}")
        return False


def install_mfa():
    """Install Montreal Forced Aligner"""
    print("Installing Montreal Forced Aligner via conda...")
    
    cmd = [
        "conda", "install", "-c", "conda-forge",
        "montreal-forced-aligner", "-y"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ MFA installed successfully")
    except Exception as e:
        print(f"❌ Failed to install MFA: {str(e)}")
        print("\nManual installation:")
        print("  conda install -c conda-forge montreal-forced-aligner")
        sys.exit(1)


def extract_durations_from_textgrids(samples):
    """
    Extract phoneme durations from MFA TextGrid output files
    """
    print("=" * 60)
    print("  Extracting Phoneme Durations")
    print("=" * 60)
    print()
    
    import textgrid  # pip install textgrid
    
    successful = 0
    failed = 0
    
    for sample in tqdm(samples, desc="Extracting durations"):
        sample_id = sample['id']
        
        # TextGrid file from MFA
        textgrid_path = os.path.join(MFA_OUTPUT_DIR, f"{sample_id}.TextGrid")
        
        if not os.path.exists(textgrid_path):
            failed += 1
            continue
        
        try:
            # Parse TextGrid
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # Extract phone tier (phoneme alignments)
            phone_tier = None
            for tier in tg:
                if tier.name.lower() == 'phones':
                    phone_tier = tier
                    break
            
            if phone_tier is None:
                failed += 1
                continue
            
            # Extract durations
            phonemes = []
            durations = []
            
            for interval in phone_tier:
                if interval.mark.strip():  # Skip empty intervals
                    phonemes.append(interval.mark)
                    duration = interval.maxTime - interval.minTime
                    durations.append(duration)
            
            # Save duration file
            duration_path = os.path.join(ALIGNMENT_DIR, f"{sample_id}.npy")
            import numpy as np
            np.save(duration_path, np.array(durations, dtype=np.float32))
            
            # Update metadata
            sample['alignment_path'] = f"alignments/{sample_id}.npy"
            sample['phoneme_durations'] = durations
            
            # Save updated metadata
            metadata_path = os.path.join(METADATA_DIR, f"{sample_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(sample, f, indent=2)
            
            successful += 1
            
        except Exception as e:
            failed += 1
            continue
    
    print()
    print(f"✓ Extracted durations for {successful}/{len(samples)} samples")
    if failed > 0:
        print(f"⚠️  Failed to extract {failed} samples")
    print()
    
    return successful, failed


def create_training_manifest(samples):
    """
    Create final training manifest JSON file
    """
    print("=" * 60)
    print("  Creating Training Manifest")
    print("=" * 60)
    print()
    
    manifest = {
        'dataset': 'LJSpeech-1.1',
        'total_samples': len(samples),
        'sample_rate': 48000,
        'n_mels': 128,
        'samples': []
    }
    
    valid_samples = 0
    
    for sample in samples:
        # Check if sample has all required features
        required_paths = ['audio_path', 'mel_path', 'f0_path', 'energy_path',
                         'phoneme_path', 'alignment_path']
        
        if all(key in sample for key in required_paths):
            manifest['samples'].append({
                'id': sample['id'],
                'text': sample['text'],
                'normalized_text': sample['normalized_text'],
                'audio_path': sample['audio_path'],
                'mel_path': sample['mel_path'],
                'f0_path': sample['f0_path'],
                'energy_path': sample['energy_path'],
                'phoneme_path': sample['phoneme_path'],
                'alignment_path': sample['alignment_path']
            })
            valid_samples += 1
    
    # Save manifest
    manifest_path = os.path.join(INPUT_DIR, "training_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created training manifest: {manifest_path}")
    print(f"  Valid samples: {valid_samples}/{len(samples)}")
    print()
    
    return manifest_path, valid_samples


def print_statistics(phoneme_stats, alignment_success, alignment_failed, manifest_path, valid_samples, samples):
    """Print final statistics"""
    print("=" * 60)
    print("  Day 18 Completion Statistics")
    print("=" * 60)
    print()
    
    print("Phoneme Conversion:")
    print(f"  Total: {phoneme_stats['total']}")
    print(f"  Successful: {phoneme_stats['successful']}")
    print(f"  Skipped: {phoneme_stats['skipped']}")
    print(f"  Failed: {phoneme_stats['failed']}")
    print(f"  Time: {phoneme_stats['elapsed']:.1f}s")
    print()
    
    print("Forced Alignment:")
    print(f"  Successful: {alignment_success}")
    print(f"  Failed: {alignment_failed}")
    print()
    
    print("Training Manifest:")
    print(f"  Location: {manifest_path}")
    print(f"  Valid samples: {valid_samples}/{len(samples)}")
    print()
    
    # Storage statistics
    phoneme_size = sum(os.path.getsize(os.path.join(PHONEME_DIR, f))
                      for f in os.listdir(PHONEME_DIR) if f.endswith('.txt'))
    alignment_size = sum(os.path.getsize(os.path.join(ALIGNMENT_DIR, f))
                        for f in os.listdir(ALIGNMENT_DIR) if f.endswith('.npy'))
    
    print("Storage:")
    print(f"  Phoneme files: {phoneme_size / 1e6:.1f} MB")
    print(f"  Alignment files: {alignment_size / 1e6:.1f} MB")
    print()
    
    print("Next steps:")
    print("  1. Validate training manifest")
    print("  2. Begin FastSpeech2 training (Day 19)")
    print("  3. Monitor training progress")
    print()


def main():
    """Main execution"""
    print_header()
    
    # Check prerequisites
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory not found: {INPUT_DIR}")
        print("Please complete Days 16-17 first.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Load metadata
    samples = load_metadata()
    
    if len(samples) == 0:
        print("❌ Error: No samples found")
        sys.exit(1)
    
    # Step 1: Convert text to phonemes
    phoneme_stats = process_phonemes_parallel(samples)
    
    if phoneme_stats['failed'] > 0:
        print(f"⚠️  Warning: {phoneme_stats['failed']} samples failed phoneme conversion")
    
    # Step 2: Prepare MFA corpus
    prepare_mfa_corpus(samples)
    
    # Step 3: Run forced alignment
    alignment_success = run_mfa_alignment()
    
    if not alignment_success:
        print("❌ Forced alignment failed. Please check MFA installation and try again.")
        sys.exit(1)
    
    # Step 4: Extract durations from TextGrids
    alignment_success, alignment_failed = extract_durations_from_textgrids(samples)
    
    # Step 5: Create training manifest
    manifest_path, valid_samples = create_training_manifest(samples)
    
    # Print final statistics
    print_statistics(phoneme_stats, alignment_success, alignment_failed,
                    manifest_path, valid_samples, samples)
    
    # Exit code
    if valid_samples == len(samples):
        print("✓ Day 18 complete! Dataset ready for training.")
        sys.exit(0)
    else:
        print(f"⚠️  Warning: Only {valid_samples}/{len(samples)} samples are valid")
        sys.exit(1)


if __name__ == "__main__":
    main()
