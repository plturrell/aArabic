#!/usr/bin/env python3
"""
Test HiFiGAN Training Infrastructure - Day 27
Validates training setup before starting the long training run
"""

import os
import sys
from pathlib import Path
import yaml

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(text):
    """Print formatted header"""
    print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")


def print_test(name, passed, details=""):
    """Print test result"""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} {name}")
    if details:
        print(f"      {details}")


def test_config_file():
    """Test configuration file exists and is valid"""
    print_header("Testing Configuration File")
    
    config_path = Path("config/hifigan_training_config.yaml")
    
    # Check if file exists
    if not config_path.exists():
        print_test("Config file exists", False, f"Not found: {config_path}")
        return False
    
    print_test("Config file exists", True, str(config_path))
    
    # Try to parse YAML
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print_test("Config file is valid YAML", True)
    except Exception as e:
        print_test("Config file is valid YAML", False, str(e))
        return False
    
    # Check required fields
    required_fields = [
        ("model", "n_mels"),
        ("training", "max_steps"),
        ("training", "batch_size"),
        ("paths", "data_dir"),
        ("paths", "checkpoint_dir"),
    ]
    
    all_present = True
    for field_path in required_fields:
        current = config
        field_name = " -> ".join(field_path)
        try:
            for key in field_path:
                current = current[key]
            print_test(f"Config has {field_name}", True, f"Value: {current}")
        except KeyError:
            print_test(f"Config has {field_name}", False, "Missing")
            all_present = False
    
    return all_present


def test_data_availability():
    """Test if required data is available"""
    print_header("Testing Data Availability")
    
    all_available = True
    
    # Check preprocessed dataset
    data_dir = Path("data/datasets/ljspeech_processed")
    if data_dir.exists():
        print_test("Preprocessed dataset directory", True, str(data_dir))
        
        # Count files
        wav_files = list(data_dir.glob("**/*.wav"))
        mel_files = list(data_dir.glob("**/*.npy"))
        
        print_test(f"Audio files found", len(wav_files) > 0, f"{len(wav_files)} files")
        print_test(f"Mel files found", len(mel_files) > 0, f"{len(mel_files)} files")
        
        if len(wav_files) == 0 or len(mel_files) == 0:
            all_available = False
    else:
        print_test("Preprocessed dataset directory", False, "Not found")
        all_available = False
    
    # Check FastSpeech2 checkpoint
    fs2_checkpoint = Path("data/models/fastspeech2/checkpoints/checkpoint_200000.mojo")
    if fs2_checkpoint.exists():
        size_mb = fs2_checkpoint.stat().st_size / (1024 * 1024)
        print_test("FastSpeech2 checkpoint", True, f"{size_mb:.1f} MB")
    else:
        print_test("FastSpeech2 checkpoint", False, "checkpoint_200000.mojo not found")
        all_available = False
    
    return all_available


def test_output_directories():
    """Test output directories can be created"""
    print_header("Testing Output Directories")
    
    all_ok = True
    
    directories = [
        "data/models/hifigan/checkpoints",
        "data/models/hifigan/logs",
        "data/models/hifigan/samples",
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print_test(f"Directory: {dir_path}", True, "Created/exists")
        except Exception as e:
            print_test(f"Directory: {dir_path}", False, str(e))
            all_ok = False
    
    return all_ok


def test_mojo_installation():
    """Test if Mojo is installed"""
    print_header("Testing Mojo Installation")
    
    # Check if mojo command exists
    import subprocess
    
    try:
        result = subprocess.run(
            ["mojo", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print_test("Mojo installed", True, version)
            return True
        else:
            print_test("Mojo installed", False, "Command failed")
            return False
    except FileNotFoundError:
        print_test("Mojo installed", False, "mojo command not found")
        return False
    except Exception as e:
        print_test("Mojo installed", False, str(e))
        return False


def test_training_script():
    """Test if training script exists"""
    print_header("Testing Training Script")
    
    script_path = Path("mojo/train_hifigan.mojo")
    
    if script_path.exists():
        size_kb = script_path.stat().st_size / 1024
        print_test("Training script exists", True, f"{size_kb:.1f} KB")
        
        # Check if it has main function
        try:
            with open(script_path) as f:
                content = f.read()
                has_main = 'def main()' in content or 'fn main()' in content
                has_trainer = 'HiFiGANTrainer' in content
                
                print_test("Has main function", has_main)
                print_test("Has HiFiGANTrainer class", has_trainer)
                
                return has_main and has_trainer
        except Exception as e:
            print_test("Script is readable", False, str(e))
            return False
    else:
        print_test("Training script exists", False, str(script_path))
        return False


def test_model_components():
    """Test if model component files exist"""
    print_header("Testing Model Components")
    
    components = [
        "mojo/models/hifigan_generator.mojo",
        "mojo/models/hifigan_discriminator.mojo",
        "mojo/models/hifigan_blocks.mojo",
        "mojo/training/losses.mojo",
        "mojo/training/dataset.mojo",
        "mojo/training/cpu_optimizer.mojo",
    ]
    
    all_present = True
    for component in components:
        path = Path(component)
        exists = path.exists()
        print_test(f"Component: {component}", exists)
        if not exists:
            all_present = False
    
    return all_present


def test_system_resources():
    """Test system resources"""
    print_header("Testing System Resources")
    
    import psutil
    
    # CPU cores
    cpu_count = psutil.cpu_count()
    cpu_ok = cpu_count >= 4
    print_test(f"CPU cores (need ≥4)", cpu_ok, f"{cpu_count} cores")
    
    # Memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_ok = memory_gb >= 8
    print_test(f"Memory (need ≥8 GB)", memory_ok, f"{memory_gb:.1f} GB")
    
    # Disk space
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / (1024**3)
    disk_ok = disk_free_gb >= 10
    print_test(f"Disk space (need ≥10 GB)", disk_ok, f"{disk_free_gb:.1f} GB free")
    
    return cpu_ok and memory_ok and disk_ok


def test_scripts_executable():
    """Test if scripts are executable"""
    print_header("Testing Scripts")
    
    scripts = [
        "scripts/start_hifigan_training.sh",
        "scripts/monitor_hifigan_training.py",
    ]
    
    all_ok = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            # Check if executable (on Unix systems)
            is_executable = os.access(path, os.X_OK)
            if not is_executable and not sys.platform.startswith('win'):
                # Try to make it executable
                try:
                    os.chmod(path, 0o755)
                    is_executable = True
                    print_test(f"Script: {script}", True, "Made executable")
                except:
                    print_test(f"Script: {script}", False, "Not executable")
                    all_ok = False
            else:
                print_test(f"Script: {script}", True)
        else:
            print_test(f"Script: {script}", False, "Not found")
            all_ok = False
    
    return all_ok


def run_all_tests():
    """Run all tests and return summary"""
    
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}HiFiGAN Training Infrastructure Test Suite - Day 27{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    
    tests = [
        ("Configuration", test_config_file),
        ("Data Availability", test_data_availability),
        ("Output Directories", test_output_directories),
        ("Mojo Installation", test_mojo_installation),
        ("Training Script", test_training_script),
        ("Model Components", test_model_components),
        ("System Resources", test_system_resources),
        ("Scripts", test_scripts_executable),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n{RED}Error in {name}: {e}{RESET}")
            results.append((name, False))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}✓{RESET}" if result else f"{RED}✗{RESET}"
        print(f"{status} {name}")
    
    print(f"\n{BOLD}Results: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}{BOLD}✓ All tests passed! Ready to start training.{RESET}")
        return True
    else:
        print(f"\n{YELLOW}{BOLD}⚠ Some tests failed. Please fix issues before training.{RESET}")
        return False


def main():
    """Main entry point"""
    try:
        # Change to project root if needed
        if not Path("mojo").exists():
            print(f"{YELLOW}Note: Run this script from the nAudioLab root directory{RESET}")
        
        success = run_all_tests()
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Test interrupted by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
