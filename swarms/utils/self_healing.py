import os
import sys
import time
import psutil
import threading
import traceback
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from swarms.utils.terminal_output import terminal

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    timestamp: datetime

class HealthCheck:
    """System health monitoring and self-healing capabilities"""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.error_count: Dict[str, int] = {}
        self.recovery_actions: Dict[str, Callable] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Default thresholds
        self.thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "error_threshold": 3
        }

    def register_recovery_action(self, error_type: str, action: Callable):
        """Register a recovery action for a specific error type"""
        self.recovery_actions[error_type] = action
        terminal.status_panel(f"Registered recovery action for {error_type}", "info")

    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            metrics = SystemMetrics(
                cpu_percent=cpu,
                memory_percent=memory,
                disk_usage_percent=disk,
                timestamp=datetime.now()
            )
            
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:  # Keep last 100 readings
                self.metrics_history.pop(0)
                
            return metrics
        
        except Exception as e:
            terminal.status_panel(f"Error collecting metrics: {str(e)}", "error")
            return None

    def check_system_health(self) -> bool:
        """Check if system metrics are within acceptable thresholds"""
        metrics = self.collect_metrics()
        if not metrics:
            return False
            
        issues = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            issues.append(f"High CPU usage: {metrics.cpu_percent}%")
            
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            issues.append(f"High memory usage: {metrics.memory_percent}%")
            
        if metrics.disk_usage_percent > self.thresholds["disk_usage_percent"]:
            issues.append(f"High disk usage: {metrics.disk_usage_percent}%")
            
        if issues:
            terminal.status_panel("\n".join(issues), "warning")
            return False
            
        return True

    def handle_error(self, error: Exception, context: str = ""):
        """Handle errors and attempt recovery"""
        error_type = type(error).__name__
        
        # Increment error count
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1
        
        terminal.status_panel(
            f"Error occurred in {context}: {str(error)}\n{traceback.format_exc()}", 
            "error"
        )
        
        # Check if we need to take recovery action
        if self.error_count[error_type] >= self.thresholds["error_threshold"]:
            self.attempt_recovery(error_type, error)

    def attempt_recovery(self, error_type: str, error: Exception):
        """Attempt to recover from an error"""
        terminal.status_panel(f"Attempting recovery for {error_type}", "info")
        
        if error_type in self.recovery_actions:
            try:
                self.recovery_actions[error_type](error)
                terminal.status_panel(f"Recovery action completed for {error_type}", "success")
                self.error_count[error_type] = 0  # Reset error count after successful recovery
            except Exception as e:
                terminal.status_panel(
                    f"Recovery action failed for {error_type}: {str(e)}", 
                    "error"
                )
        else:
            terminal.status_panel(
                f"No recovery action registered for {error_type}", 
                "warning"
            )

    def start_monitoring(self):
        """Start continuous system monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
            
        def monitor():
            while not self.stop_monitoring.is_set():
                healthy = self.check_system_health()
                if healthy:
                    terminal.status_panel("System health check passed", "success")
                time.sleep(60)  # Check every minute
                
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        terminal.status_panel("System monitoring started", "info")

    def stop_monitoring(self):
        """Stop system monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join()
            terminal.status_panel("System monitoring stopped", "info")

    def get_health_report(self) -> dict:
        """Generate a health report"""
        if not self.metrics_history:
            return {"status": "No metrics collected yet"}
            
        latest = self.metrics_history[-1]
        avg_metrics = {
            "cpu_percent": sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history),
            "memory_percent": sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history),
            "disk_usage_percent": sum(m.disk_usage_percent for m in self.metrics_history) / len(self.metrics_history)
        }
        
        return {
            "current_metrics": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_usage_percent": latest.disk_usage_percent,
                "timestamp": latest.timestamp.isoformat()
            },
            "average_metrics": avg_metrics,
            "error_counts": self.error_count,
            "status": "Healthy" if self.check_system_health() else "Issues Detected"
        }

# Create singleton instance
health_monitor = HealthCheck()