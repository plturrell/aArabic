#!/usr/bin/env python3

"""
FastSpeech2 Training Monitor - Day 19
AudioLabShimmy - Real-time Training Progress Dashboard
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not found. Install with: pip install rich")
    print("Falling back to basic text output...")

class TrainingMonitor:
    """Real-time training progress monitor"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.log_dir = project_root / "data" / "models" / "fastspeech2" / "logs"
        self.checkpoint_dir = project_root / "data" / "models" / "fastspeech2" / "checkpoints"
        
        if RICH_AVAILABLE:
            self.console = Console()
        
        self.target_step = 25000
        self.start_time = None
        self.last_step = 0
        self.loss_history = []
        self.step_times = []
    
    def find_latest_log(self) -> Optional[Path]:
        """Find the most recent training log file"""
        log_files = list(self.log_dir.glob("training_day19_*.log"))
        if not log_files:
            return None
        return max(log_files, key=lambda p: p.stat().st_mtime)
    
    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a training log line"""
        # Match patterns like: "Step 1000 | Loss: 2.3456 | Duration: 0.12s | ..."
        step_match = re.search(r'Step\s+(\d+)', line)
        loss_match = re.search(r'Loss:\s+([\d.]+)', line)
        duration_match = re.search(r'Duration:\s+([\d.]+)s', line)
        
        if step_match and loss_match:
            return {
                'step': int(step_match.group(1)),
                'loss': float(loss_match.group(1)),
                'duration': float(duration_match.group(1)) if duration_match else None,
                'timestamp': datetime.now()
            }
        return None
    
    def get_checkpoint_info(self) -> List[Tuple[int, Path]]:
        """Get list of saved checkpoints"""
        checkpoints = []
        for ckpt in self.checkpoint_dir.glob("checkpoint_*.mojo"):
            step_match = re.search(r'checkpoint_(\d+)', ckpt.name)
            if step_match:
                checkpoints.append((int(step_match.group(1)), ckpt))
        return sorted(checkpoints, key=lambda x: x[0])
    
    def calculate_eta(self, current_step: int, steps_per_sec: float) -> str:
        """Calculate estimated time to completion"""
        remaining_steps = self.target_step - current_step
        if steps_per_sec <= 0:
            return "calculating..."
        
        remaining_seconds = remaining_steps / steps_per_sec
        eta = timedelta(seconds=int(remaining_seconds))
        return str(eta)
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def create_dashboard_rich(self, stats: Dict) -> Layout:
        """Create rich dashboard layout"""
        layout = Layout()
        
        # Header
        header = Panel(
            Text("FastSpeech2 Training - Day 19", justify="center", style="bold blue"),
            subtitle=f"Target: {self.target_step:,} steps"
        )
        
        # Progress section
        progress_table = Table(show_header=False, box=None, padding=(0, 2))
        progress_table.add_column("Metric", style="cyan")
        progress_table.add_column("Value", style="green")
        
        current_step = stats.get('current_step', 0)
        progress_pct = (current_step / self.target_step * 100) if self.target_step > 0 else 0
        
        progress_table.add_row("Current Step", f"{current_step:,} / {self.target_step:,}")
        progress_table.add_row("Progress", f"{progress_pct:.1f}%")
        progress_table.add_row("Current Loss", f"{stats.get('current_loss', 0.0):.4f}")
        progress_table.add_row("Avg Loss (100)", f"{stats.get('avg_loss_100', 0.0):.4f}")
        progress_table.add_row("Steps/sec", f"{stats.get('steps_per_sec', 0.0):.2f}")
        progress_table.add_row("ETA", stats.get('eta', 'calculating...'))
        
        progress_panel = Panel(progress_table, title="📊 Progress", border_style="blue")
        
        # Checkpoints section
        checkpoint_table = Table(show_header=True, box=None)
        checkpoint_table.add_column("Step", style="cyan")
        checkpoint_table.add_column("Time", style="yellow")
        
        for step, ckpt in stats.get('checkpoints', [])[-5:]:
            mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
            checkpoint_table.add_row(f"{step:,}", mtime.strftime("%H:%M:%S"))
        
        checkpoint_panel = Panel(checkpoint_table, title="💾 Checkpoints", border_style="green")
        
        # System info
        system_table = Table(show_header=False, box=None, padding=(0, 2))
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="yellow")
        
        system_table.add_row("Elapsed Time", self.format_duration(stats.get('elapsed_time', 0)))
        system_table.add_row("Log File", stats.get('log_file', 'N/A'))
        system_table.add_row("Last Update", stats.get('last_update', 'N/A'))
        
        system_panel = Panel(system_table, title="⚙️  System", border_style="yellow")
        
        # Layout
        layout.split_column(
            Layout(header, size=3),
            Layout(progress_panel),
            Layout(checkpoint_panel),
            Layout(system_panel, size=6)
        )
        
        return layout
    
    def display_dashboard_basic(self, stats: Dict):
        """Display basic text dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 70)
        print("  FastSpeech2 Training Monitor - Day 19")
        print(f"  Target: {self.target_step:,} steps")
        print("=" * 70)
        print()
        
        current_step = stats.get('current_step', 0)
        progress_pct = (current_step / self.target_step * 100) if self.target_step > 0 else 0
        
        print("📊 Progress:")
        print(f"  Current Step:    {current_step:,} / {self.target_step:,}")
        print(f"  Progress:        {progress_pct:.1f}%")
        print(f"  Current Loss:    {stats.get('current_loss', 0.0):.4f}")
        print(f"  Avg Loss (100):  {stats.get('avg_loss_100', 0.0):.4f}")
        print(f"  Steps/sec:       {stats.get('steps_per_sec', 0.0):.2f}")
        print(f"  ETA:             {stats.get('eta', 'calculating...')}")
        print()
        
        print("💾 Recent Checkpoints:")
        for step, ckpt in stats.get('checkpoints', [])[-5:]:
            mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
            print(f"  Step {step:,}: {mtime.strftime('%H:%M:%S')}")
        print()
        
        print("⚙️  System:")
        print(f"  Elapsed Time:    {self.format_duration(stats.get('elapsed_time', 0))}")
        print(f"  Log File:        {stats.get('log_file', 'N/A')}")
        print(f"  Last Update:     {stats.get('last_update', 'N/A')}")
        print()
        print("=" * 70)
        print("Press Ctrl+C to exit")
    
    def monitor(self, refresh_interval: int = 5):
        """Main monitoring loop"""
        log_file = self.find_latest_log()
        
        if not log_file:
            print("Error: No training log file found")
            print(f"Expected location: {self.log_dir}/training_day19_*.log")
            return
        
        print(f"Monitoring: {log_file}")
        print(f"Refresh interval: {refresh_interval}s")
        print()
        
        if RICH_AVAILABLE:
            self._monitor_rich(log_file, refresh_interval)
        else:
            self._monitor_basic(log_file, refresh_interval)
    
    def _monitor_rich(self, log_file: Path, refresh_interval: int):
        """Rich terminal monitoring"""
        with Live(auto_refresh=False, console=self.console) as live:
            try:
                with open(log_file, 'r') as f:
                    # Seek to end
                    f.seek(0, 2)
                    
                    while True:
                        line = f.readline()
                        if line:
                            log_data = self.parse_log_line(line)
                            if log_data:
                                self.last_step = log_data['step']
                                self.loss_history.append(log_data['loss'])
                                if log_data['duration']:
                                    self.step_times.append(log_data['duration'])
                                
                                if not self.start_time:
                                    self.start_time = datetime.now()
                        else:
                            # No new lines, update display
                            stats = self._gather_stats(log_file)
                            dashboard = self.create_dashboard_rich(stats)
                            live.update(dashboard, refresh=True)
                            time.sleep(refresh_interval)
                            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Monitoring stopped[/yellow]")
    
    def _monitor_basic(self, log_file: Path, refresh_interval: int):
        """Basic text monitoring"""
        try:
            with open(log_file, 'r') as f:
                # Seek to end
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        log_data = self.parse_log_line(line)
                        if log_data:
                            self.last_step = log_data['step']
                            self.loss_history.append(log_data['loss'])
                            if log_data['duration']:
                                self.step_times.append(log_data['duration'])
                            
                            if not self.start_time:
                                self.start_time = datetime.now()
                    else:
                        # No new lines, update display
                        stats = self._gather_stats(log_file)
                        self.display_dashboard_basic(stats)
                        time.sleep(refresh_interval)
                        
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    
    def _gather_stats(self, log_file: Path) -> Dict:
        """Gather current training statistics"""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        steps_per_sec = self.last_step / elapsed if elapsed > 0 else 0
        
        # Calculate average loss
        avg_loss_100 = sum(self.loss_history[-100:]) / len(self.loss_history[-100:]) if self.loss_history else 0.0
        
        # Get checkpoints
        checkpoints = self.get_checkpoint_info()
        
        # Calculate ETA
        eta = self.calculate_eta(self.last_step, steps_per_sec)
        
        return {
            'current_step': self.last_step,
            'current_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'avg_loss_100': avg_loss_100,
            'steps_per_sec': steps_per_sec,
            'eta': eta,
            'elapsed_time': elapsed,
            'checkpoints': checkpoints,
            'log_file': log_file.name,
            'last_update': datetime.now().strftime("%H:%M:%S")
        }


def main():
    """Main entry point"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("=" * 70)
    print("  FastSpeech2 Training Monitor")
    print("  AudioLabShimmy - Day 19")
    print("=" * 70)
    print()
    
    # Check if training is running
    pid_file = project_root / "data" / "models" / "fastspeech2" / "logs" / "training_day19.pid"
    if pid_file.exists():
        with open(pid_file) as f:
            pid = f.read().strip()
        print(f"Training Process ID: {pid}")
        
        # Check if process is running
        try:
            os.kill(int(pid), 0)
            print("✓ Training process is running")
        except (OSError, ValueError):
            print("⚠ Warning: Training process may not be running")
    else:
        print("⚠ Warning: No PID file found")
    
    print()
    
    # Start monitoring
    monitor = TrainingMonitor(project_root)
    
    try:
        monitor.monitor(refresh_interval=5)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
