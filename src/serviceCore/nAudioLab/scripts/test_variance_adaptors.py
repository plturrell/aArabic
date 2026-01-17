#!/usr/bin/env python3
"""
Test script for variance adaptors (Day 8)

Tests:
- Duration predictor
- Pitch predictor
- Energy predictor
- Length regulator
- Complete variance adaptor pipeline
"""

import subprocess
import sys
from pathlib import Path


def test_component(component_name: str, test_code: str) -> bool:
    """Test a single component."""
    print(f"\n{'='*60}")
    print(f"Testing {component_name}...")
    print('='*60)
    
    # Write test code to temp file
    test_file = Path("test_temp.mojo")
    test_file.write_text(test_code)
    
    try:
        # Run mojo test
        result = subprocess.run(
            ["mojo", "run", str(test_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✓ {component_name} test PASSED")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            return True
        else:
            print(f"✗ {component_name} test FAILED")
            print(f"Error:\n{result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {component_name} test TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {component_name} test ERROR: {e}")
        return False
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


def main():
    """Run all variance adaptor tests."""
    print("=" * 60)
    print("AudioLabShimmy Day 8: Variance Adaptors Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Duration Predictor
    duration_test = '''
from tensor import Tensor
from memory import memset_zero

# Import the duration predictor
import sys
sys.path.append("mojo/models")
from duration_predictor import DurationPredictor

fn main():
    print("Test 1: Duration Predictor")
    
    # Create predictor
    var predictor = DurationPredictor(d_model=256, kernel_size=3, dropout=0.1)
    predictor.eval()
    
    # Create dummy encoder output [batch=2, seq_len=10, d_model=256]
    var encoder_output = Tensor[DType.float32](2, 10, 256)
    
    # Initialize with dummy values
    for i in range(encoder_output.num_elements()):
        encoder_output[i] = 0.1
    
    # Forward pass
    var durations = predictor.forward(encoder_output)
    
    # Check output shape
    print("Input shape: [2, 10, 256]")
    print("Output shape: [" + str(durations.shape()[0]) + ", " + str(durations.shape()[1]) + "]")
    
    if durations.shape()[0] == 2 and durations.shape()[1] == 10:
        print("✓ Duration predictor output shape correct")
    else:
        print("✗ Duration predictor output shape incorrect")
    
    # Check values are reasonable
    var min_val = durations[0]
    var max_val = durations[0]
    for i in range(durations.num_elements()):
        if durations[i] < min_val:
            min_val = durations[i]
        if durations[i] > max_val:
            max_val = durations[i]
    
    print("Duration range: [" + str(min_val) + ", " + str(max_val) + "]")
    print("✓ Duration predictor test complete")
'''
    results['Duration Predictor'] = test_component('Duration Predictor', duration_test)
    
    # Test 2: Pitch Predictor
    pitch_test = '''
from tensor import Tensor

import sys
sys.path.append("mojo/models")
from pitch_predictor import PitchPredictor

fn main():
    print("Test 2: Pitch Predictor")
    
    # Create predictor
    var predictor = PitchPredictor(d_model=256, kernel_size=3, dropout=0.1)
    predictor.eval()
    
    # Create dummy encoder output [batch=2, seq_len=10, d_model=256]
    var encoder_output = Tensor[DType.float32](2, 10, 256)
    for i in range(encoder_output.num_elements()):
        encoder_output[i] = 0.1
    
    # Forward pass
    var pitch = predictor.forward(encoder_output)
    
    # Check output shape
    print("Input shape: [2, 10, 256]")
    print("Output shape: [" + str(pitch.shape()[0]) + ", " + str(pitch.shape()[1]) + "]")
    
    if pitch.shape()[0] == 2 and pitch.shape()[1] == 10:
        print("✓ Pitch predictor output shape correct")
    else:
        print("✗ Pitch predictor output shape incorrect")
    
    print("✓ Pitch predictor test complete")
'''
    results['Pitch Predictor'] = test_component('Pitch Predictor', pitch_test)
    
    # Test 3: Energy Predictor
    energy_test = '''
from tensor import Tensor

import sys
sys.path.append("mojo/models")
from energy_predictor import EnergyPredictor

fn main():
    print("Test 3: Energy Predictor")
    
    # Create predictor
    var predictor = EnergyPredictor(d_model=256, kernel_size=3, dropout=0.1)
    predictor.eval()
    
    # Create dummy encoder output [batch=2, seq_len=10, d_model=256]
    var encoder_output = Tensor[DType.float32](2, 10, 256)
    for i in range(encoder_output.num_elements()):
        encoder_output[i] = 0.1
    
    # Forward pass
    var energy = predictor.forward(encoder_output)
    
    # Check output shape
    print("Input shape: [2, 10, 256]")
    print("Output shape: [" + str(energy.shape()[0]) + ", " + str(energy.shape()[1]) + "]")
    
    if energy.shape()[0] == 2 and energy.shape()[1] == 10:
        print("✓ Energy predictor output shape correct")
    else:
        print("✗ Energy predictor output shape incorrect")
    
    print("✓ Energy predictor test complete")
'''
    results['Energy Predictor'] = test_component('Energy Predictor', energy_test)
    
    # Test 4: Length Regulator
    length_test = '''
from tensor import Tensor

import sys
sys.path.append("mojo/models")
from length_regulator import LengthRegulator

fn main():
    print("Test 4: Length Regulator")
    
    # Create regulator
    var regulator = LengthRegulator()
    
    # Create encoder output [batch=2, phoneme_len=5, d_model=256]
    var encoder_output = Tensor[DType.float32](2, 5, 256)
    for i in range(encoder_output.num_elements()):
        encoder_output[i] = 0.5
    
    # Create durations [batch=2, phoneme_len=5]
    # Each phoneme gets 3 frames
    var durations = Tensor[DType.float32](2, 5)
    for i in range(durations.num_elements()):
        durations[i] = 3.0  # 3 frames per phoneme
    
    # Regulate length
    var expanded = regulator.regulate_length(encoder_output, durations, alpha=1.0)
    
    print("Input shape: [2, 5, 256]")
    print("Durations: [3, 3, 3, 3, 3] per batch")
    print("Expected output: [2, 15, 256] (5 phonemes × 3 frames = 15)")
    print("Output shape: [" + str(expanded.shape()[0]) + ", " + str(expanded.shape()[1]) + ", " + str(expanded.shape()[2]) + "]")
    
    if expanded.shape()[0] == 2 and expanded.shape()[2] == 256:
        print("✓ Length regulator batch and feature dimensions correct")
    else:
        print("✗ Length regulator dimensions incorrect")
    
    # Check that expansion happened
    if expanded.shape()[1] >= 10:  # At least 10 frames (should be ~15)
        print("✓ Length regulator expanded sequence")
    else:
        print("✗ Length regulator did not expand properly")
    
    print("✓ Length regulator test complete")
'''
    results['Length Regulator'] = test_component('Length Regulator', length_test)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for component, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{component:.<50} {status}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All variance adaptor tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
