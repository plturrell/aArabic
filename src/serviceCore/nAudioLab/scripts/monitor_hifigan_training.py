#!/usr/bin/env python3
"""
HiFiGAN Training Monitor - Day 27
Real-time training progress monitoring with metrics dashboard
"""

import os
import sys
import time
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
import re

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
CYAN = "\033[96m"


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def format_time(seconds):
    """Format seconds as HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))


def format_number(num):
    """Format number with thousands separator"""
    return f"{num:,}"


def parse_log_file(log_file):
    """Parse training log file for metrics"""
    metrics = {
        'current_step': 0,
        'loss_g': [],
        'loss_d': [],
        'loss_stft': [],
        'loss_adv': [],
        'loss_fm': [],
        'start_time': None,
        'last_update': None,
        'checkpoints': []
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # Find start time
        for line in lines:
            if 'Training started at:' in line:
                metrics['start_time'] = line.split('Training started at:')[1].strip()
                break
        
        # Parse training steps
        for line in lines:
            # Step format: "Step 100 | G: 15.2345 | D: 2.1234 | STFT: 10.1234 | Adv: 1.2345 | FM: 2.3456"
            match = re.search(r'Step\s+(\d+)\s+\|\s+G:\s+([\d.]+)\s+\|\s+D:\s+([\d.]+)\s+\|\s+STFT:\s+([\d.]+)\s+\|\s+Adv:\s+([\d.]+)\s+\|\s+FM:\s+([\d.]+)', line)
            if match:
                step = int(match.group(1))
                metrics['current_step'] = step
                metrics['loss_g'].append(float(match.group(2)))
                metrics['loss_d'].append(float(match.group(3)))
                metrics['loss_stft'].append(float(match.group(4)))
                metrics['loss_adv'].append(float(match.group(5)))
                metrics['loss_fm'].append(float(match.group(6)))
                metrics['last_update'] = datetime.now()
            
            # Parse checkpoint saves
            if 'Saving checkpoint:' in line and 'checkpoint_' in line:
                ckpt_match = re.search(r'checkpoint_(\d+)\.mojo', line)
                if ckpt_match:
                    metrics['checkpoints'].append(int(ckpt_match.group(1)))
    
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error parsing log: {e}")
    
    return metrics


def get_latest_log_file():
    """Find the most recent training log file"""
    log_dir = Path("data/models/hifigan/logs")
    if not log_dir.exists():
        return None
    
    log_files = list(log_dir.glob("training_day27_*.log"))
    if not log_files:
        return None
    
    return max(log_files, key=os.path.getmtime)


def calculate_eta(current_step, target_step, start_time_str):
    """Calculate estimated time to completion"""
    if not start_time_str or current_step == 0:
        return "Calculating..."
    
    try:
        # Parse start time
        start_time = datetime.strptime(start_time_str, "%a %b %d %H:%M:%S %Z %Y")
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Calculate steps per second
        steps_per_sec = current_step / elapsed if elapsed > 0 else 0
        
        # Calculate remaining time
        remaining_steps = target_step - current_step
        if steps_per_sec > 0:
            remaining_seconds = remaining_steps / steps_per_sec
            return format_time(remaining_seconds)
        else:
            return "Unknown"
    except:
        return "Unknown"


def get_recent_average(values, window=10):
    """Get average of last N values"""
    if not values:
        return 0.0
    recent = values[-window:]
    return sum(recent) / len(recent)


def draw_progress_bar(current, target, width=40):
    """Draw a text-based progress bar"""
    progress = current / target if target > 0 else 0
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percentage = progress * 100
    return f"[{bar}] {percentage:5.1f}%"


def display_dashboard(metrics, target_step=50000):
    """Display real-time training dashboard"""
    clear_screen()
    
    # Header
    print(f"{BOLD}{CYAN}╔{'═' * 78}╗{RESET}")
    print(f"{BOLD}{CYAN}║{' ' * 20}HiFiGAN Training Monitor - Day 27{' ' * 24}║{RESET}")
    print(f"{BOLD}{CYAN}╚{'═' * 78}╝{RESET}")
    print()
    
    # Training status
    if metrics['current_step'] > 0:
        status = f"{GREEN}● TRAINING{RESET}"
    else:
        status = f"{YELLOW}● STARTING{RESET}"
    
    print(f"{BOLD}Status:{RESET} {status}")
    print(f"{BOLD}Start Time:{RESET} {metrics['start_time'] or 'Not started'}")
    
    if metrics['last_update']:
        elapsed = (datetime.now() - metrics['last_update']).total_seconds()
        if elapsed < 60:
            update_status = f"{GREEN}Updated {int(elapsed)}s ago{RESET}"
        elif elapsed < 300:  # 5 minutes
            update_status = f"{YELLOW}Updated {int(elapsed/60)}m ago{RESET}"
        else:
            update_status = f"{RED}Stale - {int(elapsed/60)}m ago{RESET}"
        print(f"{BOLD}Last Update:{RESET} {update_status}")
    print()
    
    # Progress
    print(f"{BOLD}═══ Training Progress ═══{RESET}")
    print(f"Current Step: {BOLD}{GREEN}{format_number(metrics['current_step'])}{RESET} / {format_number(target_step)}")
    print(draw_progress_bar(metrics['current_step'], target_step))
    
    # ETA
    eta = calculate_eta(metrics['current_step'], target_step, metrics['start_time'])
    print(f"Estimated Time Remaining: {BOLD}{CYAN}{eta}{RESET}")
    print()
    
    # Loss metrics
    print(f"{BOLD}═══ Loss Metrics (Last 10 Steps Average) ═══{RESET}")
    if metrics['loss_g']:
        print(f"Generator Loss:      {BOLD}{get_recent_average(metrics['loss_g']):8.4f}{RESET}")
        print(f"Discriminator Loss:  {BOLD}{get_recent_average(metrics['loss_d']):8.4f}{RESET}")
        print(f"STFT Loss:           {BOLD}{get_recent_average(metrics['loss_stft']):8.4f}{RESET}")
        print(f"Adversarial Loss:    {BOLD}{get_recent_average(metrics['loss_adv']):8.4f}{RESET}")
        print(f"Feature Match Loss:  {BOLD}{get_recent_average(metrics['loss_fm']):8.4f}{RESET}")
    else:
        print(f"{YELLOW}No loss data yet...{RESET}")
    print()
    
    # Loss trends
    if len(metrics['loss_g']) >= 100:
        recent_100 = metrics['loss_g'][-100:]
        first_20 = sum(recent_100[:20]) / 20
        last_20 = sum(recent_100[-20:]) / 20
        trend = "↓ Improving" if last_20 < first_20 else "↑ Increasing"
        trend_color = GREEN if last_20 < first_20 else YELLOW
        print(f"{BOLD}Loss Trend (last 100 steps):{RESET} {trend_color}{trend}{RESET}")
        print()
    
    # Checkpoints
    print(f"{BOLD}═══ Checkpoints ═══{RESET}")
    if metrics['checkpoints']:
        print(f"Saved Checkpoints: {BOLD}{len(metrics['checkpoints'])}{RESET}")
        print("Latest:", ", ".join([format_number(c) for c in sorted(metrics['checkpoints'])[-5:]]))
    else:
        print(f"{YELLOW}No checkpoints saved yet{RESET}")
    print()
    
    # Day 27 milestones
    print(f"{BOLD}═══ Day 27 Milestones ═══{RESET}")
    milestones = [10000, 20000, 30000, 40000, 50000]
    for milestone in milestones:
        if metrics['current_step'] >= milestone:
            status_icon = f"{GREEN}✓{RESET}"
        else:
            status_icon = f"{YELLOW}○{RESET}"
        print(f"{status_icon} {format_number(milestone):>6} steps")
    print()
    
    # Expected performance
    print(f"{BOLD}═══ Expected Performance ═══{RESET}")
    if metrics['current_step'] < 50000:
        print("Initial Training (0-50k steps):")
        print("  • Generator Loss: ~15-8")
        print("  • Discriminator Loss: ~2-1.5")
        print("  • STFT Loss: ~10-5")
    print()
    
    # Instructions
    print(f"{BOLD}═══ Commands ═══{RESET}")
    print("Press Ctrl+C to exit monitor")
    print(f"Log file: {CYAN}data/models/hifigan/logs/training_day27_*.log{RESET}")
    print(f"Checkpoints: {CYAN}data/models/hifigan/checkpoints/{RESET}")
    print()
    
    # Footer
    print(f"{CYAN}{'─' * 80}{RESET}")
    print(f"Refreshing every 5 seconds... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main monitoring loop"""
    print(f"{BOLD}{BLUE}HiFiGAN Training Monitor{RESET}")
    print("Searching for training logs...")
    
    target_step = 50000  # Day 27 target
    
    try:
        while True:
            log_file = get_latest_log_file()
            
            if not log_file:
                clear_screen()
                print(f"{YELLOW}No training logs found.{RESET}")
                print()
                print("Expected log location: data/models/hifigan/logs/training_day27_*.log")
                print()
                print(f"Waiting for training to start... {datetime.now().strftime('%H:%M:%S')}")
                time.sleep(5)
                continue
            
            # Parse log and display dashboard
            metrics = parse_log_file(log_file)
            display_dashboard(metrics, target_step)
            
            # Check if training is complete
            if metrics['current_step'] >= target_step:
                print(f"\n{BOLD}{GREEN}✓ Day 27 Training Complete!{RESET}")
                print(f"Final step: {format_number(metrics['current_step'])}")
                break
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print(f"\n\n{BOLD}Monitor stopped by user.{RESET}")
        print("Training is still running in the background.")
        sys.exit(0)
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
