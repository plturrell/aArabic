#!/usr/bin/env python3

"""
Training Infrastructure Test - Day 19
AudioLabShimmy - Quick validation run to verify training setup
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

class TrainingInfrastructureTest:
    """Test the training infrastructure with a small run"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_path = project_root / "config" / "training_config.yaml"
        self.manifest_path = project_root / "data" / "datasets" / "ljspeech_processed" / "training_manifest.json"
        self.test_output_dir = project_root / "data" / "models" / "fastspeech2" / "test_run"
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print("=" * 70)
        print("  Training Infrastructure Test")
        print("  Verifying Prerequisites")
        print("=" * 70)
        print()
        
        checks_passed = True
        
        # Check Mojo installation
        print("1. Checking Mojo installation...")
        try:
            result = subprocess.run(
                ["mojo", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"   ✓ Mojo installed: {result.stdout.strip()}")
            else:
                print("   ✗ Mojo not properly configured")
                checks_passed = False
        except FileNotFoundError:
            print("   ✗ Mojo not found in PATH")
            checks_passed = False
        except Exception as e:
            print(f"   ✗ Error checking Mojo: {e}")
            checks_passed = False
        
        # Check configuration file
        print("\n2. Checking configuration file...")
        if self.config_path.exists():
            print(f"   ✓ Configuration found: {self.config_path}")
        else:
            print(f"   ✗ Configuration not found: {self.config_path}")
            checks_passed = False
        
        # Check training manifest
        print("\n3. Checking training manifest...")
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    manifest = json.load(f)
                    num_samples = len(manifest.get('samples', []))
                    print(f"   ✓ Manifest found with {num_samples} samples")
            except Exception as e:
                print(f"   ✗ Error reading manifest: {e}")
                checks_passed = False
        else:
            print(f"   ✗ Manifest not found: {self.manifest_path}")
            print("   Note: Run Days 16-18 preprocessing first")
            checks_passed = False
        
        # Check model code
        print("\n4. Checking model implementations...")
        model_files = [
            "mojo/models/fastspeech2.mojo",
            "mojo/models/fastspeech2_encoder.mojo",
            "mojo/models/fastspeech2_decoder.mojo",
            "mojo/training/trainer.mojo",
            "mojo/train_fastspeech2.mojo"
        ]
        
        all_models_exist = True
        for model_file in model_files:
            if (self.project_root / model_file).exists():
                print(f"   ✓ {model_file}")
            else:
                print(f"   ✗ {model_file} missing")
                all_models_exist = False
        
        if not all_models_exist:
            print("   Note: Model code from Days 6-15 required")
            checks_passed = False
        
        print()
        return checks_passed
    
    def run_quick_test(self) -> bool:
        """Run a quick training test (10 steps)"""
        print("=" * 70)
        print("  Quick Training Test (10 steps)")
        print("=" * 70)
        print()
        
        # Create test output directory
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Running 10-step test...")
        print("This validates:")
        print("  - Data loading works")
        print("  - Model forward pass works")
        print("  - Loss calculation works")
        print("  - Backward pass works")
        print("  - Optimizer step works")
        print()
        
        # Create test configuration
        test_config = {
            "test_mode": True,
            "max_steps": 10,
            "batch_size": 2,
            "log_every": 1,
            "output_dir": str(self.test_output_dir)
        }
        
        test_config_path = self.test_output_dir / "test_config.json"
        with open(test_config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        # Run test
        try:
            cmd = [
                "mojo", "run",
                str(self.project_root / "mojo" / "train_fastspeech2.mojo"),
                "--config", str(self.config_path),
                "--test-mode",
                "--max-steps", "10",
                "--batch-size", "2",
                "--output-dir", str(self.test_output_dir)
            ]
            
            print(f"Command: {' '.join(cmd)}")
            print()
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=False,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print()
                print("✓ Test completed successfully!")
                return True
            else:
                print()
                print("✗ Test failed with return code:", result.returncode)
                return False
                
        except subprocess.TimeoutExpired:
            print()
            print("✗ Test timed out (>5 minutes)")
            return False
        except Exception as e:
            print()
            print(f"✗ Test error: {e}")
            return False
    
    def check_test_output(self) -> bool:
        """Verify test produced expected output"""
        print()
        print("=" * 70)
        print("  Verifying Test Output")
        print("=" * 70)
        print()
        
        # Check for log file
        log_files = list(self.test_output_dir.glob("*.log"))
        if log_files:
            print(f"✓ Found log file: {log_files[0].name}")
            
            # Parse log for key metrics
            with open(log_files[0]) as f:
                log_content = f.read()
                
            if "Step 10" in log_content:
                print("✓ Reached step 10")
            else:
                print("✗ Did not reach step 10")
                return False
            
            if "Loss:" in log_content:
                print("✓ Loss calculated successfully")
            else:
                print("✗ No loss values found")
                return False
        else:
            print("✗ No log file found")
            return False
        
        # Check for test checkpoint
        checkpoint_files = list(self.test_output_dir.glob("checkpoint_*.mojo"))
        if checkpoint_files:
            print(f"✓ Found checkpoint: {checkpoint_files[0].name}")
        else:
            print("⚠ No checkpoint created (may be expected for short test)")
        
        print()
        print("=" * 70)
        print("✓ Training infrastructure validated successfully!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Review test output in:", self.test_output_dir)
        print("  2. Launch full training with:")
        print("     bash scripts/start_training_day19.sh")
        print()
        
        return True
    
    def run(self) -> int:
        """Main test execution"""
        print()
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 15 + "TRAINING INFRASTRUCTURE TEST" + " " * 25 + "║")
        print("║" + " " * 20 + "AudioLabShimmy - Day 19" + " " * 25 + "║")
        print("╚" + "═" * 68 + "╝")
        print()
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            print()
            print("=" * 70)
            print("✗ Prerequisites check failed")
            print("Please complete missing steps before running training")
            print("=" * 70)
            return 1
        
        print("=" * 70)
        print("✓ All prerequisites met!")
        print("=" * 70)
        print()
        
        # Step 2: Run quick test
        input("Press Enter to run 10-step training test...")
        print()
        
        if not self.run_quick_test():
            print()
            print("=" * 70)
            print("✗ Quick test failed")
            print("Please review errors above and fix before full training")
            print("=" * 70)
            return 1
        
        # Step 3: Verify output
        if not self.check_test_output():
            print()
            print("=" * 70)
            print("✗ Output verification failed")
            print("=" * 70)
            return 1
        
        return 0


def main():
    """Main entry point"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    tester = TrainingInfrastructureTest(project_root)
    
    try:
        return tester.run()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
