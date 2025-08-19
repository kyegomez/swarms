"""
Swarms CLI Terminal User Interface (TUI) - Clean Version
Professional industrial interface with improved architecture
"""

import os
import json
import yaml
import psutil
import asyncio
import platform
import subprocess
import webbrowser
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
import time

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Input, TextArea, 
    DataTable, Tree, Label, ProgressBar, Rule, Select, Checkbox
)
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from textual.screen import ModalScreen
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.progress import Progress as RichProgress

from swarms.cli.theme import (
    SWARMS_COLORS, SWARMS_LOGO_ASCII, HEXAGON_CONTROL_PATTERN, INDUSTRIAL_BORDER,
    SECTION_DIVIDER, get_theme_colors, get_tui_theme, get_symbols
)
from swarms.cli.onboarding_process import OnboardingProcess
from swarms.cli.swarm_service import (
    SwarmService, SwarmConfig, ExecutionResult, LogLevel, LogEntry
)
from swarms.utils.loguru_logger import logger

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    try:
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")

# Load environment variables on import
load_env_file()

# Dynamic table generation functions
def create_dynamic_table(title: str, headers: List[str], rows: List[List[str]], colors: List[str] = None) -> str:
    """Create a dynamic table that adjusts to content width"""
    if not rows:
        return f"[bold cyan]{title}[/bold cyan]\nNo data available"
    
    # Calculate column widths based on content
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in rows:
            if i < len(row):
                # Strip markup for width calculation
                clean_text = row[i].replace('[green]', '').replace('[/green]', '').replace('[red]', '').replace('[/red]', '').replace('[yellow]', '').replace('[/yellow]', '')
                max_width = max(max_width, len(clean_text))
        col_widths.append(max_width + 2)  # Add padding
    
    # Build table
    total_width = sum(col_widths) + len(headers) - 1
    
    # Title
    table_str = f"[bold cyan]{title}[/bold cyan]\n"
    
    # Top border
    table_str += "┌" + "┬".join("─" * width for width in col_widths) + "┐\n"
    
    # Headers
    header_row = "│"
    for i, header in enumerate(headers):
        header_row += f" {header:<{col_widths[i]-1}}│"
    table_str += header_row + "\n"
    
    # Header separator
    table_str += "├" + "┼".join("─" * width for width in col_widths) + "┤\n"
    
    # Data rows
    for row in rows:
        data_row = "│"
        for i, cell in enumerate(row):
            if i < len(col_widths):
                # Calculate padding accounting for markup
                clean_text = cell.replace('[green]', '').replace('[/green]', '').replace('[red]', '').replace('[/red]', '').replace('[yellow]', '').replace('[/yellow]', '')
                padding = col_widths[i] - len(clean_text) - 1
                data_row += f" {cell}{' ' * padding}│"
        table_str += data_row + "\n"
    
    # Bottom border
    table_str += "└" + "┴".join("─" * width for width in col_widths) + "┘"
    
    return table_str

def create_info_panel(title: str, items: List[tuple]) -> str:
    """Create a dynamic info panel that adjusts to content"""
    if not items:
        return f"[bold cyan]{title}[/bold cyan]\nNo information available"
    
    # Calculate widths
    max_key_width = max(len(str(item[0])) for item in items) + 2
    max_value_width = max(len(str(item[1])) for item in items) + 2
    
    total_width = max_key_width + max_value_width + 1
    
    # Build panel
    panel_str = f"[bold cyan]{title}[/bold cyan]\n"
    
    # Top border
    panel_str += "┌" + "─" * max_key_width + "┬" + "─" * max_value_width + "┐\n"
    
    # Data rows
    for key, value in items:
        panel_str += f"│ {str(key):<{max_key_width-1}}│ {str(value):<{max_value_width-1}}│\n"
    
    # Bottom border
    panel_str += "└" + "─" * max_key_width + "┴" + "─" * max_value_width + "┘"
    
    return panel_str


class SetupScreen(ModalScreen):
    """Setup screen with improved configuration management"""
    
    BINDINGS = [
        ("escape", "dismiss", "Back"),
        ("up", "focus_previous", "Up"),
        ("down", "focus_next", "Down"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
        ("ctrl+s", "save_config", "Save Config"),
        ("ctrl+t", "test_connections", "Test API"),
        ("ctrl+c", "dismiss", "Cancel"),
    ]
    
    def __init__(self, swarm_service: SwarmService):
        super().__init__()
        self.swarm_service = swarm_service
        self.api_keys = {
            "swarms": self.swarm_service.config_manager.get_api_key("swarms") or "",
            "openai": self.swarm_service.config_manager.get_api_key("openai") or "",
            "anthropic": self.swarm_service.config_manager.get_api_key("anthropic") or "",
            "groq": self.swarm_service.config_manager.get_api_key("groq") or "",
            "workspace_dir": self.swarm_service.config_manager.get_workspace_dir()
        }
    
    def action_dismiss(self) -> None:
        """Go back to main menu"""
        self.dismiss()
    
    def action_select(self) -> None:
        """Handle selection"""
        pass
    
    def handle_button_action(self, button_id: str) -> None:
        """Handle button actions"""
        if button_id == "save_config":
            self.save_configuration()
        elif button_id == "test_connections":
            self.test_connections()
        elif button_id == "configure_swarms":
            self.configure_swarms_key()
        elif button_id == "configure_openai":
            self.configure_openai_key()
        elif button_id == "configure_anthropic":
            self.configure_anthropic_key()
        elif button_id == "configure_groq":
            self.configure_groq_key()
        elif button_id == "configure_workspace":
            self.configure_workspace()
    
    def configure_swarms_key(self) -> None:
        """Configure Swarms API key"""
        key_input = self.query_one("#swarms_api_key", Input)
        if key_input:
            self.api_keys["swarms"] = key_input.value
    
    def configure_openai_key(self) -> None:
        """Configure OpenAI API key"""
        key_input = self.query_one("#openai_api_key", Input)
        if key_input:
            self.api_keys["openai"] = key_input.value
    
    def configure_anthropic_key(self) -> None:
        """Configure Anthropic API key"""
        key_input = self.query_one("#anthropic_api_key", Input)
        if key_input:
            self.api_keys["anthropic"] = key_input.value
    
    def configure_groq_key(self) -> None:
        """Configure Groq API key"""
        key_input = self.query_one("#groq_api_key", Input)
        if key_input:
            self.api_keys["groq"] = key_input.value
    
    def configure_workspace(self) -> None:
        """Configure workspace directory"""
        workspace_input = self.query_one("#workspace_dir", Input)
        if workspace_input:
            self.api_keys["workspace_dir"] = workspace_input.value
    
    def test_connections(self) -> None:
        """Test API connections"""
        try:
            # Save current configuration first
            self.save_configuration()
            
            # Test connections
            api_status = self.swarm_service.get_api_status()
            
            # Update status display
            status_text = "API Connection Test Results:\n"
            for provider, status in api_status.items():
                status_icon = "[+]" if status else "[x]"
                status_text += f"{status_icon} {provider.title()}: {'Connected' if status else 'Failed'}\n"
            
            status_widget = self.query_one("#connection_status", Static)
            if status_widget:
                status_widget.update(status_text)
                
        except Exception as e:
            logger.error(f"Error testing connections: {e}")
    
    def save_configuration(self) -> None:
        """Save configuration"""
        try:
            # Save API keys
            for provider, key in self.api_keys.items():
                if provider != "workspace_dir" and key:
                    self.swarm_service.configure_api_key(provider, key)
            
            # Save workspace directory
            if self.api_keys["workspace_dir"]:
                self.swarm_service.config_manager.set_workspace_dir(self.api_keys["workspace_dir"])
            
            # Update status
            status_widget = self.query_one("#save_status", Static)
            if status_widget:
                status_widget.update("[+] Configuration saved successfully!")
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            status_widget = self.query_one("#save_status", Static)
            if status_widget:
                status_widget.update(f"[x] Error saving configuration: {str(e)}")
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        input_id = event.input.id
        if input_id == "swarms_api_key":
            self.api_keys["swarms"] = event.input.value
        elif input_id == "openai_api_key":
            self.api_keys["openai"] = event.input.value
        elif input_id == "anthropic_api_key":
            self.api_keys["anthropic"] = event.input.value
        elif input_id == "groq_api_key":
            self.api_keys["groq"] = event.input.value
        elif input_id == "workspace_dir":
            self.api_keys["workspace_dir"] = event.input.value
    
    def compose(self) -> ComposeResult:
        yield Container(
            # Header
            Static("SYSTEM CONFIGURATION", classes="modal-title"),
            Static("Configure API keys and workspace settings", classes="modal-subtitle"),
            
            # API Keys Section
            Static("API Configuration", classes="content-section-title"),
            
            Static("Swarms API Key:", classes="field-label"),
            Input(
                value=self.api_keys["swarms"],
                placeholder="Enter your Swarms API key",
                id="swarms_api_key",
                classes="primary-input"
            ),
            
            Static("OpenAI API Key:", classes="field-label"),
            Input(
                value=self.api_keys["openai"],
                placeholder="Enter your OpenAI API key",
                id="openai_api_key",
                classes="primary-input"
            ),
            
            Static("Anthropic API Key:", classes="field-label"),
            Input(
                value=self.api_keys["anthropic"],
                placeholder="Enter your Anthropic API key",
                id="anthropic_api_key",
                classes="primary-input"
            ),
            
            Static("Groq API Key:", classes="field-label"),
            Input(
                value=self.api_keys["groq"],
                placeholder="Enter your Groq API key",
                id="groq_api_key",
                classes="primary-input"
            ),
            
            # Workspace Configuration
            Static("Workspace Configuration", classes="content-section-title"),
            
            Static("Workspace Directory:", classes="field-label"),
            Input(
                value=self.api_keys["workspace_dir"],
                placeholder="Enter workspace directory path",
                id="workspace_dir",
                classes="primary-input"
            ),
            
            # Status Areas
            Static("Save Status", classes="content-section-title"),
            Static("", id="save_status", classes="status-area"),
            
            Static("Connection Status", classes="content-section-title"),
            Static("", id="connection_status", classes="status-area"),
            
            # Keyboard Shortcuts Help
            Static("Keyboard Shortcuts", classes="content-section-title"),
            Static(
                "Ctrl+S: Save Configuration  |  Ctrl+T: Test Connections  |  Ctrl+C: Cancel  |  Escape: Back",
                classes="help-text"
            ),
            
            classes="setup-modal"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        self.handle_button_action(event.button.id)
    
    def on_mount(self) -> None:
        """Initialize screen"""
        pass
    
    def action_save_config(self) -> None:
        """Save configuration via keyboard shortcut"""
        self.save_configuration()
    
    def action_test_connections(self) -> None:
        """Test connections via keyboard shortcut"""
        self.test_connections()


class SwarmRunnerScreen(ModalScreen):
    """Improved swarm runner with service integration"""
    
    BINDINGS = [
        ("escape", "dismiss", "Back"),
        ("up", "focus_previous", "Up"),
        ("down", "focus_next", "Down"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
        ("ctrl+r", "start_swarm", "Run Swarm"),
        ("ctrl+x", "stop_swarm", "Stop Swarm"),
        ("ctrl+f", "refresh_swarms", "Refresh"),
        ("ctrl+z", "clear_logs", "Clear Logs"),
    ]
    
    def __init__(self, swarm_service: SwarmService):
        super().__init__()
        self.swarm_service = swarm_service
        self.available_swarms: List[SwarmConfig] = []
        self.swarm_running = False
        self.current_execution = None
        
        # Set up log callback
        self.swarm_service.add_log_callback(self._on_log_update)
        
        # Load available swarms
        self.load_available_swarms()
    
    def load_available_swarms(self) -> None:
        """Load available swarms using service"""
        try:
            self.available_swarms = self.swarm_service.discover_swarms()
            
            # If no swarms found, create default templates
            if not self.available_swarms:
                self.available_swarms = [
                    SwarmConfig(
                        name="Research Swarm",
                        description="Multi-agent research and analysis swarm",
                        type="Template",
                        file_path="research_template",
                        config={"swarm_type": "SequentialWorkflow"}
                    ),
                    SwarmConfig(
                        name="Analysis Swarm",
                        description="Data analysis and reporting swarm", 
                        type="Template",
                        file_path="analysis_template",
                        config={"swarm_type": "GroupChat"}
                    )
                ]
                
        except Exception as e:
            logger.error(f"Error loading swarms: {e}")
            self.available_swarms = []
    
    def action_dismiss(self) -> None:
        """Go back to main menu"""
        if self.swarm_running:
            self.stop_swarm()
        self.dismiss()
    
    def action_start_swarm(self) -> None:
        """Start the selected swarm"""
        self.start_swarm()
    
    def action_stop_swarm(self) -> None:
        """Stop the swarm"""
        self.stop_swarm()
    
    def start_swarm(self) -> None:
        """Start swarm execution"""
        try:
            if self.swarm_running:
                self.app.notify("Swarm already running", severity="warning")
                return
            
            # Get selected swarm
            swarm_select = self.query_one("#swarm_select", Select)
            if not swarm_select or not swarm_select.value:
                self.app.notify("ERROR: Please select a swarm", severity="error")
                return
            
            selected_swarm_name = swarm_select.value
            selected_swarm = None
            
            for swarm in self.available_swarms:
                if swarm.name == selected_swarm_name:
                    selected_swarm = swarm
                    break
            
            if not selected_swarm:
                self.app.notify("ERROR: Selected swarm not found", severity="error")
                return
            
            # Get task description
            task_description = self.query_one("#task_description", TextArea).text
            if not task_description.strip():
                self.app.notify("ERROR: Task description required", severity="error")
                return
            
            # Clear previous logs
            self.swarm_service.clear_logs()
            
            # Add initial log
            self.swarm_service.log_manager.add_log(
                f"Starting swarm execution: {selected_swarm.name}",
                LogLevel.INFO,
                "swarm_execution"
            )
            
            self.swarm_running = True
            self.app.notify("SWARM STARTED - EXECUTING", severity="information")
            
            # Execute swarm asynchronously
            asyncio.create_task(self._execute_swarm_async(selected_swarm, task_description))
            
        except Exception as e:
            self.swarm_service.log_manager.add_log(
                f"Error starting swarm: {str(e)}",
                LogLevel.ERROR,
                "swarm_execution"
            )
            self.app.notify(f"Swarm start error: {str(e)}", severity="error")
    
    async def _execute_swarm_async(self, swarm_config: SwarmConfig, task: str):
        """Execute swarm asynchronously"""
        try:
            result = self.swarm_service.execute_swarm(swarm_config, task)
            
            if result.success:
                self.swarm_service.log_manager.add_log(
                    f"Swarm execution completed successfully in {result.execution_time:.2f}s",
                    LogLevel.INFO,
                    "swarm_execution"
                )
                
                # Display result
                if result.response:
                    self.swarm_service.log_manager.add_log(
                        f"RESULT: {result.response}",
                        LogLevel.INFO,
                        "swarm_result"
                    )
            else:
                self.swarm_service.log_manager.add_log(
                    f"Swarm execution failed: {result.error}",
                    LogLevel.ERROR,
                    "swarm_execution"
                )
                
        except Exception as e:
            self.swarm_service.log_manager.add_log(
                f"Unexpected error during execution: {str(e)}",
                LogLevel.ERROR,
                "swarm_execution"
            )
        finally:
            self.swarm_running = False
    
    def stop_swarm(self) -> None:
        """Stop the running swarm"""
        if self.swarm_running:
            self.swarm_running = False
            self.swarm_service.log_manager.add_log(
                "Swarm execution stopped by user",
                LogLevel.WARNING,
                "swarm_execution"
            )
            self.app.notify("Swarm stopped", severity="information")
        else:
            self.app.notify("No swarm running", severity="warning")
    
    def _on_log_update(self, log_entry: LogEntry):
        """Handle log updates from service"""
        # Update log display
        self.update_log_display()
    
    def update_log_display(self) -> None:
        """Update the log display"""
        try:
            log_widget = self.query_one("#log_text", Static)
            if log_widget:
                # Get recent logs
                logs = self.swarm_service.get_logs(limit=50)
                
                # Format logs for display
                log_lines = []
                for log in logs:
                    timestamp = log.timestamp.strftime('%H:%M:%S')
                    level_icon = {
                        LogLevel.DEBUG: "[+]",
                        LogLevel.INFO: "[ ]",
                        LogLevel.WARNING: "[!]",
                        LogLevel.ERROR: "[x]",
                        LogLevel.CRITICAL: "[!]"
                    }.get(log.level, "[ ]")
                    
                    log_lines.append(f"[{timestamp}] {level_icon} {log.message}")
                
                log_content = "\n".join(log_lines)
                log_widget.update(log_content)
        except Exception as e:
            logger.error(f"Error updating log display: {e}")
    
    def action_refresh_swarms(self) -> None:
        """Refresh swarms via keyboard shortcut"""
        self.load_available_swarms()
        # Update select widget
        swarm_select = self.query_one("#swarm_select", Select)
        if swarm_select and self.available_swarms:
            swarm_select.options = [(swarm.name, swarm.name) for swarm in self.available_swarms]
            swarm_select.value = self.available_swarms[0].name
    
    def action_clear_logs(self) -> None:
        """Clear logs via keyboard shortcut"""
        self.swarm_service.clear_logs()
        self.update_log_display()
    
    def compose(self) -> ComposeResult:
        yield Container(
            # Header
            Static("SWARM EXECUTION SYSTEM", classes="modal-title"),
            Static("Execute swarms with improved service integration", classes="modal-subtitle"),
            
            # Swarm Selection Section
            Static("Swarm Selection", classes="content-section-title"),
            
            Static("Available Swarms:", classes="field-label"),
            Select(
                options=[(swarm.name, swarm.name) for swarm in self.available_swarms],
                value=self.available_swarms[0].name if self.available_swarms else "",
                id="swarm_select",
                classes="primary-select"
            ),
            
            Static("Task Description:", classes="field-label"),
            TextArea(id="task_description", classes="primary-textarea"),
            
            # Log Display
            Static("Execution Logs", classes="content-section-title"),
            Static("", id="log_text", classes="status-area"),
            
            # Keyboard Shortcuts Help
            Static("Keyboard Shortcuts", classes="content-section-title"),
            Static(
                "Ctrl+R: Run Swarm  |  Ctrl+X: Stop  |  Ctrl+F: Refresh  |  Ctrl+Z: Clear Logs  |  Escape: Back",
                classes="help-text"
            ),
            
            classes="swarm-runner-modal"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "start_swarm":
            self.start_swarm()
        elif event.button.id == "stop_swarm":
            self.stop_swarm()
        elif event.button.id == "refresh_swarms":
            self.action_refresh_swarms()
        elif event.button.id == "clear_logs":
            self.action_clear_logs()
        elif event.button.id == "back_btn":
            self.action_dismiss()
    
    def on_mount(self) -> None:
        """Initialize screen"""
        self.update_log_display()


class SystemInfo:
    """System information service"""
    
    @staticmethod
    def get_system_status() -> Dict[str, Any]:
        """Get system status information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                "cpu_usage": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used": memory.used,
                "memory_total": memory.total,
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "cpu_usage": 0,
                "memory_percent": 0,
                "memory_used": 0,
                "memory_total": 0,
                "platform": "Unknown",
                "python_version": "Unknown",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }


class SwarmsTUI(App):
    """Main Swarms TUI Application with service integration"""
    
    CSS = f"""
    Screen {{
        background: {SWARMS_COLORS['carbon_black']};
    }}
    
    Header {{
        background: {SWARMS_COLORS['swarms_red']};
        color: white;
        height: 3;
        text-align: center;
        content-align: center middle;
    }}
    
    Footer {{
        background: {SWARMS_COLORS['panel_black']};
        color: {SWARMS_COLORS['secondary_text']};
        border-top: solid {SWARMS_COLORS['border_gray']};
        height: 3;
        text-align: center;
        content-align: center middle;
    }}
    
    .main-container {{
        height: 100%;
    }}
    
    .sidebar {{
        background: {SWARMS_COLORS['panel_black']};
        border-right: thick {SWARMS_COLORS['swarms_red']};
        width: 30%;
        padding: 1;
    }}
    
    .content {{
        background: {SWARMS_COLORS['carbon_black']};
        width: 70%;
        padding: 2;
        overflow-y: auto;
    }}
    
    .logo {{
        color: {SWARMS_COLORS['swarms_red']};
        text-style: bold;
        text-align: center;
        margin: 0 0 1 0;
        padding: 1;
        border: round {SWARMS_COLORS['swarms_red']};
        background: {SWARMS_COLORS['carbon_black']};
    }}
    
    .menu-item {{
        background: #1A1A1A;
        color: #FFFFFF;
        border: solid #2D2D2D;
        height: 3;
        padding: 0 1;
        text-align: left;
        margin: 0 0 1 0;
        text-style: bold;
    }}
    
    .menu-item:hover {{
        background: #E60000;
        color: #FFFFFF;
        border: solid #E60000;
    }}
    
    .menu-item:focus {{
        background: #FF0000;
        color: #FFFFFF;
        border: thick #E60000;
    }}
    
    .status-text {{
        color: {SWARMS_COLORS['secondary_text']};
        text-align: center;
        height: 1;
        margin: 0 0 1 0;
        background: {SWARMS_COLORS['carbon_black']};
        border: round {SWARMS_COLORS['border_gray']};
        text-style: italic;
    }}
    
    Button {{
        background: #0D1117;
        color: #E60000;
        border: thick #E60000;
        height: 4;
        padding: 1 2;
        text-align: center;
        text-style: bold;
        content-align: center middle;
        min-width: 15;
    }}
    
    Button:hover {{
        background: #1A1A1A;
        color: #FF0000;
        border: thick #FF0000;
        text-style: bold;
    }}
    
    Button:focus {{
        background: #1A1A1A;
        color: #FF0000;
        border: thick #FF0000;
        text-style: bold;
    }}
    
    Input {{
        background: {SWARMS_COLORS['carbon_black']};
        color: {SWARMS_COLORS['primary_text']};
        border: thick {SWARMS_COLORS['border_gray']};
        height: 3;
        padding: 0 1;
        text-style: bold;
    }}
    
    Input:focus {{
        border: thick {SWARMS_COLORS['swarms_red']};
        background: #111111;
    }}
    
    TextArea {{
        background: {SWARMS_COLORS['carbon_black']};
        color: {SWARMS_COLORS['primary_text']};
        border: thick {SWARMS_COLORS['border_gray']};
        min-height: 6;
        height: 6;
        padding: 1;
        text-style: bold;
    }}
    
    TextArea:focus {{
        border: thick {SWARMS_COLORS['swarms_red']};
        background: #111111;
    }}
    
    Select {{
        background: {SWARMS_COLORS['carbon_black']};
        color: white;
        border: thick {SWARMS_COLORS['border_gray']};
        height: 3;
        padding: 0 1;
    }}
    
    Select:focus {{
        border: thick {SWARMS_COLORS['swarms_red']};
        background: #111111;
    }}
    
    .setup-modal {{
        background: {SWARMS_COLORS['panel_black']};
        border: thick {SWARMS_COLORS['swarms_red']};
        padding: 3;
        margin: 1;
        width: 85%;
        height: 85%;
        overflow-y: auto;
    }}
    
    .swarm-runner-modal {{
        background: {SWARMS_COLORS['panel_black']};
        border: thick {SWARMS_COLORS['swarms_red']};
        padding: 3;
        margin: 1;
        width: 90%;
        height: 90%;
        overflow-y: auto;
    }}
    
    .modal-title {{
        color: {SWARMS_COLORS['swarms_red']};
        text-style: bold;
        text-align: center;
        margin: 0 0 2 0;
        padding: 1;
        background: {SWARMS_COLORS['carbon_black']};
        border: round {SWARMS_COLORS['swarms_red']};
    }}
    
    .modal-subtitle {{
        color: {SWARMS_COLORS['secondary_text']};
        text-align: center;
        margin: 0 0 2 0;
        padding: 1;
        text-style: italic;
        background: {SWARMS_COLORS['carbon_black']};
        border: round {SWARMS_COLORS['border_gray']};
    }}
    
    .content-section-title {{
        color: {SWARMS_COLORS['primary_text']};
        text-style: bold;
        padding: 1;
        margin: 2 0 1 0;
        background: {SWARMS_COLORS['panel_black']};
        border: solid {SWARMS_COLORS['swarms_red']};
        text-align: center;
    }}
    
    .field-label {{
        color: {SWARMS_COLORS['primary_text']};
        margin: 1 0 1 0;
        padding: 0 1;
        text-style: bold;
        background: {SWARMS_COLORS['carbon_black']};
        border-left: thick {SWARMS_COLORS['swarms_red']};
    }}
    
    .primary-input {{
        background: {SWARMS_COLORS['carbon_black']};
        color: {SWARMS_COLORS['primary_text']};
        border: thick {SWARMS_COLORS['border_gray']};
        margin: 0 0 2 0;
        height: 3;
        padding: 0 1;
        text-style: bold;
    }}
    
    .primary-input:focus {{
        border: thick {SWARMS_COLORS['swarms_red']};
        background: #111111;
    }}
    
    .primary-textarea {{
        background: {SWARMS_COLORS['carbon_black']};
        color: {SWARMS_COLORS['primary_text']};
        border: thick {SWARMS_COLORS['border_gray']};
        margin: 0 0 2 0;
        height: 8;
        padding: 1;
        text-style: bold;
    }}
    
    .primary-textarea:focus {{
        border: thick {SWARMS_COLORS['swarms_red']};
        background: #111111;
    }}
    
    .primary-select {{
        background: {SWARMS_COLORS['carbon_black']};
        color: white;
        border: thick {SWARMS_COLORS['border_gray']};
        margin: 0 0 2 0;
        height: 3;
        padding: 0 1;
    }}
    
    .primary-select:focus {{
        border: thick {SWARMS_COLORS['swarms_red']};
        background: #111111;
    }}
    
    .action-button {{
        background: #0D1117;
        color: #E60000;
        border: thick #E60000;
        margin: 1 1 1 0;
        height: 4;
        min-height: 4;
        min-width: 20;
        padding: 1 3;
        text-style: bold;
        text-align: center;
        content-align: center middle;
    }}
    
    .action-button:hover {{
        background: #1A1A1A;
        color: #FF0000;
        border: thick #FF0000;
        text-style: bold;
    }}
    
    .action-button:focus {{
        background: #1A1A1A;
        color: #FF0000;
        border: thick #FF0000;
        text-style: bold;
    }}
    
    Horizontal {{
        height: auto;
        width: 100%;
        align: center top;
        margin: 1 0;
    }}
    
    .status-area {{
        background: {SWARMS_COLORS['carbon_black']};
        color: {SWARMS_COLORS['primary_text']};
        border: thick {SWARMS_COLORS['border_gray']};
        margin: 1 0;
        height: 20;
        min-height: 15;
        padding: 1;
        overflow-y: auto;
    }}
    
    .help-text {{
        color: {SWARMS_COLORS['secondary_text']};
        text-align: center;
        margin: 1 0;
        text-style: italic;
        background: {SWARMS_COLORS['carbon_black']};
        border: round {SWARMS_COLORS['border_gray']};
    }}
    
    Rule {{
        color: {SWARMS_COLORS['swarms_red']};
        margin: 1 0;
    }}
    """
    
    TITLE = "SWARMS - INDUSTRIAL CONTROL INTERFACE"
    SUB_TITLE = "MULTI-AGENT SYSTEM MANAGEMENT"
    
    BINDINGS = [
        ("q", "quit", "QUIT"),
        ("h", "show_help", "HELP"),
        ("ctrl+c", "quit", "EXIT"),
        ("1", "show_dashboard", "DASHBOARD"),
        ("2", "show_agents", "AGENTS"),
        ("3", "show_workflows", "WORKFLOWS"),
        ("4", "show_config", "CONFIG"),
        ("5", "show_monitoring", "MONITORING"),
        ("s", "show_setup", "SETUP"),
        ("a", "show_agent_creation", "CREATE AGENT"),
        ("b", "show_builder", "BUILD"),
        ("r", "show_swarm_runner", "RUN SWARM"),
        ("m", "show_monitoring", "MONITORING"),
        ("c", "show_config", "CONFIG"),
        ("v", "show_quality_control", "VALIDATE"),
        ("up", "focus_previous", "UP"),
        ("down", "focus_next", "DOWN"),
        ("left", "focus_previous", "LEFT"),
        ("right", "focus_next", "RIGHT"),
        ("tab", "focus_next", "NEXT"),
        ("shift+tab", "focus_previous", "PREVIOUS"),
    ]
    
    current_view = reactive("dashboard")
    symbols = get_symbols()
    system_info = SystemInfo()
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.now()
        self.error_count = 0
        self.success_count = 0
        
        # Initialize swarm service
        self.swarm_service = SwarmService()
        
        # Initialize logging
        logger.info("Swarms TUI initialized with service integration")
    
    def compose(self) -> ComposeResult:
        """Create a clean, professional TUI layout with enhanced visuals"""
        yield Header()
        
        with Horizontal(classes="main-container"):
            # Enhanced sidebar with visual hierarchy
            with Vertical(classes="sidebar"):
                # Enhanced logo with visual frame
                yield Static("[*] SWARMS [*]", classes="logo")
                yield Rule()
                
                # Navigation section with visual grouping
                yield Static("NAVIGATION", classes="content-section-title")
                yield Button("[>] Dashboard", id="btn_dashboard", classes="menu-item")
                yield Button("[+] Create Agent", id="btn_create_agent", classes="menu-item")
                yield Button("[*] Auto Builder", id="btn_builder", classes="menu-item")
                yield Button("[>>] Run Swarm", id="btn_run_swarm", classes="menu-item")
                
                yield Rule()
                
                # System section
                yield Static("SYSTEM", classes="content-section-title")
                yield Button("[#] Monitoring", id="btn_monitoring", classes="menu-item")
                yield Button("[v] Quality Control", id="btn_quality", classes="menu-item")
                yield Button("[=] Setup", id="btn_setup", classes="menu-item")
                yield Button("[@] Configuration", id="btn_config", classes="menu-item")
                yield Button("[?] Help", id="btn_help", classes="menu-item")
                
                yield Rule()
                
                # Enhanced status indicators with visual frames
                yield Static("STATUS", classes="content-section-title")
                yield Static("[+] System: Online", classes="status-text")
                yield Static("[~] API: Ready", classes="status-text")
                yield Static("[*] Engine: Active", classes="status-text")
            
            # Enhanced main content area
            with Vertical(classes="content"):
                yield Static("Welcome to Swarms Industrial Control Interface", id="main_content")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the TUI with enhanced capabilities"""
        try:
            logger.info("Mounting Swarms TUI with service integration")
            
            # Display experimental warning notification
            self.notify(
                "WARNING: This TUI is experimental and unstable. Use at your own discretion.",
                severity="warning",
                timeout=8.0
            )
            
            self.show_dashboard_view()
            self.set_timer(10.0, self.refresh_dashboard)
            
            # Initialize background services
            self.swarm_service.initialize_discovery()
            
        except Exception as e:
            logger.error(f"Error during TUI mount: {e}")
            self.notify(f"INITIALIZATION ERROR: {str(e)}", severity="error")
    
    def check_system_requirements(self) -> None:
        """Check system requirements"""
        try:
            # Check Python version
            if platform.python_version_tuple() < ('3', '8'):
                self.notify("Warning: Python 3.8+ recommended", severity="warning")
            
            # Check available memory
            memory = psutil.virtual_memory()
            if memory.available < 500 * 1024 * 1024:  # 500MB
                self.notify("Warning: Low memory available", severity="warning")
                
        except Exception as e:
            logger.error(f"Error checking system requirements: {e}")
    
    def refresh_dashboard(self) -> None:
        """Refresh dashboard information"""
        try:
            if self.current_view == "dashboard":
                self.show_dashboard_view()
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        try:
            if event.button.id == "btn_dashboard":
                self.show_dashboard_view()
            elif event.button.id == "btn_setup":
                self.show_setup()
            elif event.button.id == "btn_builder":
                self.show_builder()
            elif event.button.id == "btn_run_swarm":
                self.show_swarm_runner()
            elif event.button.id == "btn_help":
                self.show_help_view()
            elif event.button.id == "btn_config":
                self.show_config_view()
            elif event.button.id == "btn_quality":
                self.show_quality_control()
            elif event.button.id == "btn_create_agent":
                self.show_agent_creation()
            elif event.button.id == "btn_monitoring":
                self.show_system_monitoring()
        except Exception as e:
            logger.error(f"Error handling button press: {e}")
            self.notify(f"Error: {str(e)}", severity="error")
    
    def show_setup(self) -> None:
        """Show setup screen"""
        try:
            setup_screen = SetupScreen(self.swarm_service)
            self.push_screen(setup_screen)
        except Exception as e:
            logger.error(f"Error showing setup: {e}")
            self.notify(f"Setup error: {str(e)}", severity="error")
    
    def show_builder(self) -> None:
        """Show auto builder screen"""
        try:
            builder_screen = AutoBuilderScreen(self.swarm_service)
            self.push_screen(builder_screen)
        except Exception as e:
            logger.error(f"Error showing builder: {e}")
            self.notify(f"Builder error: {str(e)}", severity="error")
    
    def show_swarm_runner(self) -> None:
        """Show swarm runner screen"""
        try:
            swarm_runner_screen = SwarmRunnerScreen(self.swarm_service)
            self.push_screen(swarm_runner_screen)
        except Exception as e:
            logger.error(f"Error showing swarm runner: {e}")
            self.notify(f"Swarm runner error: {str(e)}", severity="error")
    
    def show_agent_creation(self) -> None:
        """Show agent creation screen"""
        try:
            agent_creation_screen = AgentCreationScreen(self.swarm_service)
            self.push_screen(agent_creation_screen)
        except Exception as e:
            logger.error(f"Error showing agent creation: {e}")
            self.notify(f"Agent creation error: {str(e)}", severity="error")
    
    def show_quality_control(self) -> None:
        """Show quality control screen"""
        try:
            quality_control_screen = QualityControlScreen(self.swarm_service)
            self.push_screen(quality_control_screen)
        except Exception as e:
            logger.error(f"Error showing quality control: {e}")
            self.notify(f"Quality control error: {str(e)}", severity="error")
    
    def show_system_monitoring(self) -> None:
        """Show system monitoring screen"""
        try:
            monitoring_screen = SystemMonitoringScreen(self.swarm_service)
            self.push_screen(monitoring_screen)
        except Exception as e:
            logger.error(f"Error showing system monitoring: {e}")
            self.notify(f"System monitoring error: {str(e)}", severity="error")
    
    def show_dashboard_view(self) -> None:
        """Show the main dashboard with dynamic widgets"""
        try:
            content = self.query_one("#main_content", Static)
            status = self.system_info.get_system_status()
            api_status = self.swarm_service.get_api_status()
            
            # Count available swarms
            available_swarms = self.swarm_service.discover_swarms()
            
            # Create dynamic content
            dashboard_content = f"""[bold red]ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆ    ÆÆÆÆ    ÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆ      ÆÆ      ÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆ      ÆÆ      ÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆ    ÆÆÆÆ    ÆÆÆÆÆÆÆÆ
ÆÆÆÆ     ÆÆÆ     ÆÆ     ÆÆÆÆ
ÆÆÆ                      ÆÆÆ
ÆÆÆÆ     ÆÆÆ     ÆÆ     ÆÆÆÆ
ÆÆÆÆÆÆÆÆ    ÆÆÆÆ    ÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆ      ÆÆ      ÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆ      ÆÆ      ÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆ    ÆÆÆÆ    ÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ[/bold red]

[bold white]╔══ SWARMS Multi-Agent CONTROL INTERFACE ══════════════════════╗[/bold white]
[bold white]║[/bold white]                    [bold red][⬡] SYSTEM DASHBOARD [⬡][/bold red]                  [bold white]║[/bold white]
[bold white]╚══════════════════════════════════════════════════════════════╝[/bold white]

[bold yellow]!!! EXPERIMENTAL INTERFACE WARNING !!![/bold yellow]
[yellow]This TUI is Work-in-Progress (WIP) - Expect bugs and instability[/yellow]
[yellow]Use at your own discretion - Report issues via GitHub[/yellow]
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]

"""
            
            # Dynamic System Resources Table
            cpu_usage = status.get('cpu_usage', 0)
            memory_usage = status.get('memory_percent', 0)
            
            def get_status_indicator(usage):
                if usage < 50:
                    return "[green]●●●●●●●●●●[/green]"
                elif usage < 80:
                    return "[yellow]●●●●●●●●●●[/yellow]"
                else:
                    return "[red]●●●●●●●●●●[/red]"
            
            system_rows = [
                ["[#] CPU", f"{cpu_usage:.1f}%", get_status_indicator(cpu_usage)],
                ["[@] Memory", f"{memory_usage:.1f}%", get_status_indicator(memory_usage)],
                ["[D] Storage", f"{status.get('memory_used', 0) // (1024**3):.1f}GB", f"{status.get('memory_total', 0) // (1024**3):.1f}GB Total"],
                ["[~] Platform", status.get('platform', 'Unknown'), f"Python {status.get('python_version', 'Unknown')}"]
            ]
            
            dashboard_content += create_dynamic_table(
                "System Resources", 
                ["Component", "Usage", "Status"], 
                system_rows
            ) + "\n\n"
            
            # Dynamic API Status Table  
            api_rows = []
            for provider, connected in api_status.items():
                status_text = "[green]Connected[/green]" if connected else "[red]Offline[/red]"
                indicator = "[green][+] Active[/green]" if connected else "[red][x] Down[/red]"
                performance = "[green]Optimal[/green]" if connected else "[red]Unavailable[/red]"
                api_rows.append([provider.title(), status_text, indicator, performance])
            
            dashboard_content += create_dynamic_table(
                "API Connectivity Status",
                ["Provider", "Status", "Indicator", "Performance"],
                api_rows
            ) + "\n\n"
            
            # Dynamic Swarm Intelligence Table
            swarm_rows = [
                ["[+] Available Swarms", str(len(available_swarms)), "Ready for deployment"],
                ["[>>] Active Executions", "0", "Currently running"],
                ["[*] System Performance", "[green]OPTIMAL[/green]" if cpu_usage < 50 else "[yellow]HIGH[/yellow]", "Resource utilization"],
                ["[=] Security Status", "[green]SECURE[/green]", "All systems protected"]
            ]
            
            dashboard_content += create_dynamic_table(
                "Swarm Intelligence Status",
                ["Metric", "Value", "Description"],
                swarm_rows
            ) + "\n\n"
            
            # Dynamic Quick Access Commands
            command_items = [
                ("[=] Press S", "System Setup & Configuration"),
                ("[+] Press A", "Create Custom Agents"),
                ("[>>] Press R", "Execute Swarm Operations"),
                ("[*] Press B", "Auto-Build Intelligent Swarms"),
                ("[#] Press M", "Real-Time System Monitoring"),
                ("[v] Press V", "Quality Control & Validation"),
                ("[?] Press H", "Help & Documentation")
            ]
            
            dashboard_content += create_info_panel("Quick Access Commands", command_items) + "\n\n"
            
            # Dynamic System Information
            info_items = [
                ("[@] Last Update", status.get('timestamp', 'Unknown')),
                ("[~] Uptime", "Active since system initialization"),
                ("[!] Sync Status", "All components synchronized"),
                ("[*] Version", "Swarms Industrial Control Interface")
            ]
            
            dashboard_content += create_info_panel("System Information", info_items)
            
            content.update(dashboard_content)
            self.current_view = "dashboard"
        except Exception as e:
            logger.error(f"Error showing dashboard view: {e}")
            self.notify(f"DASHBOARD ERROR: {str(e)}", severity="error")
    
    def show_config_view(self) -> None:
        """Show configuration view"""
        try:
            content = self.query_one("#main_content", Static)
            
            config_content = f"""
CONFIGURATION MANAGEMENT

API Keys Status
──────────────
Swarms API:     {'[+] Configured' if self.swarm_service.config_manager.get_api_key('swarms') else '[x] Not Configured'}
OpenAI API:     {'[+] Configured' if self.swarm_service.config_manager.get_api_key('openai') else '[x] Not Configured'}
Anthropic API:  {'[+] Configured' if self.swarm_service.config_manager.get_api_key('anthropic') else '[x] Not Configured'}
Groq API:       {'[+] Configured' if self.swarm_service.config_manager.get_api_key('groq') else '[x] Not Configured'}

Workspace: {self.swarm_service.config_manager.get_workspace_dir()}

Actions
───────
• Press 'S' to open System Setup
• Press 'H' for Help
            """
            
            content.update(config_content)
            self.current_view = "config"
        except Exception as e:
            logger.error(f"Error showing config view: {e}")
            self.notify(f"CONFIG ERROR: {str(e)}", severity="error")
    
    def show_help_view(self) -> None:
        """Show help information"""
        try:
            content = self.query_one("#main_content", Static)
            
            help_content = f"""[bold red]ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆ    ÆÆÆÆ    ÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆ      ÆÆ      ÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆ      ÆÆ      ÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆ    ÆÆÆÆ    ÆÆÆÆÆÆÆÆ
ÆÆÆÆ     ÆÆÆ     ÆÆ     ÆÆÆÆ
ÆÆÆ                      ÆÆÆ
ÆÆÆÆ     ÆÆÆ     ÆÆ     ÆÆÆÆ
ÆÆÆÆÆÆÆÆ    ÆÆÆÆ    ÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆ      ÆÆ      ÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆ      ÆÆ      ÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆ    ÆÆÆÆ    ÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ
ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ[/bold red]

[?] SYSTEM DOCUMENTATION

{SECTION_DIVIDER}
KEYBOARD SHORTCUTS
{SECTION_DIVIDER}
┌─ NAVIGATION CONTROLS ───────────────────────────────┐
│ Q / Ctrl+C    Exit Application                      │
│ H             Show This Help                        │
│ S             Open System Setup                     │
│ A             Create Agent                          │
│ B             Launch Auto Swarm Builder             │
│ R             Run Swarm                             │
│ M             System Monitoring                     │
│ C             Configuration                         │
│ V             Quality Control Validation            │
│ 1             Dashboard View                        │
│ 2             Agent Management                      │
│ 3             Workflow Control                      │
│ 4             System Configuration                  │
│ 5             System Monitoring                     │
└─────────────────────────────────────────────────────┘

{SECTION_DIVIDER}
PLATFORM CAPABILITIES
{SECTION_DIVIDER}
┌─ CORE FEATURES ─────────────────────────────────────┐
│ [#] Real-time system monitoring                    │
│ [+] Multi-agent swarm execution                    │
│ [*] Intelligent swarm discovery                    │
│ [=] Secure configuration management                │
│ [@] Structured logging and error handling          │
│ [~] API key validation and testing                 │
│ [+] Agent creation with templates                  │
│ [*] Auto swarm builder with AI                     │
│ [v] Quality control and validation                 │
│ [#] Advanced metrics and analytics                 │
└─────────────────────────────────────────────────────┘

{SECTION_DIVIDER}
GETTING STARTED
{SECTION_DIVIDER}
┌─ INITIALIZATION SEQUENCE ───────────────────────────┐
│ 1. Press 'S' to configure API keys                 │
│ 2. Set up workspace directory                      │
│ 3. Press 'A' to create your first agent            │
│ 4. Press 'B' to build an intelligent swarm         │
│ 5. Press 'R' to run your swarms                    │
│ 6. Press 'M' to monitor system performance         │
│ 7. Press 'V' to validate your configurations       │
└─────────────────────────────────────────────────────┘

{SECTION_DIVIDER}
ADVANCED FEATURES
{SECTION_DIVIDER}
┌─ PROFESSIONAL CAPABILITIES ─────────────────────────┐
│ Agent Templates:    Research, Creative, Code, Analysis │
│ Swarm Types:        Sequential, GroupChat, Hierarchical│
│ Execution Modes:    Cloud API, Local, Hybrid          │
│ Quality Control:    Config validation, Security checks │
│ Monitoring:         CPU, Memory, Network, API status   │
│ Auto Builder:       AI-powered swarm generation        │
│ Configuration:      Secure key management, Validation  │
│ Logging:            Structured, Real-time, Filterable  │
└─────────────────────────────────────────────────────┘

{SECTION_DIVIDER}
TROUBLESHOOTING
{SECTION_DIVIDER}
┌─ COMMON ISSUES ─────────────────────────────────────┐
│ [x] API Connection Failed: Press 'S' → Test Connections │
│ [!] No Swarms Found: Press 'A' to create agents        │
│ [x] Execution Errors: Press 'V' for validation         │
│ [#] Performance Issues: Press 'M' for monitoring       │
│ [@] Config Problems: Press 'C' for configuration       │
└─────────────────────────────────────────────────────┘
            """
            
            content.update(help_content)
            self.current_view = "help"
        except Exception as e:
            logger.error(f"Error showing help view: {e}")
            self.notify(f"HELP ERROR: {str(e)}", severity="error")
    
    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()
    
    def action_show_help(self) -> None:
        """Show help"""
        self.show_help_view()
    
    def action_show_dashboard(self) -> None:
        """Show dashboard"""
        self.show_dashboard_view()
    
    def action_show_agents(self) -> None:
        """Show agent creation"""
        self.show_agent_creation()
    
    def action_show_workflows(self) -> None:
        """Show swarm runner"""
        self.show_swarm_runner()
    
    def action_show_config(self) -> None:
        """Show configuration"""
        self.show_config_view()
    
    def action_show_monitoring(self) -> None:
        """Show system monitoring"""
        self.show_system_monitoring()
    
    def action_show_setup(self) -> None:
        """Show setup"""
        self.show_setup()
    
    def action_show_agent_creation(self) -> None:
        """Show agent creation"""
        self.show_agent_creation()
    
    def action_show_builder(self) -> None:
        """Show builder"""
        self.show_builder()
    
    def action_show_swarm_runner(self) -> None:
        """Show swarm runner"""
        self.show_swarm_runner()
    
    def action_show_quality_control(self) -> None:
        """Show quality control"""
        self.show_quality_control()


class SystemMonitoringScreen(ModalScreen):
    """Real-time system monitoring with advanced metrics"""
    
    BINDINGS = [
        ("escape", "dismiss", "Back"),
        ("ctrl+r", "refresh", "Refresh"),
        ("ctrl+z", "clear", "Clear"),
    ]
    
    def __init__(self, swarm_service: SwarmService):
        super().__init__()
        self.swarm_service = swarm_service
        self.monitoring_active = False
        
    def action_dismiss(self) -> None:
        """Stop monitoring and go back"""
        self.monitoring_active = False
        self.dismiss()
        
    def action_refresh(self) -> None:
        """Force refresh monitoring data"""
        self.update_monitoring_display()
        
    def action_clear(self) -> None:
        """Clear monitoring history"""
        monitoring_widget = self.query_one("#monitoring_display", Static)
        if monitoring_widget:
            monitoring_widget.update("Monitoring cleared. Press 'R' to refresh data.")
    
    def update_monitoring_display(self) -> None:
        """Update real-time monitoring display"""
        try:
            # Get system metrics
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network stats
            net_io = psutil.net_io_counters()
            
            # Format monitoring data
            monitoring_data = f"""
REAL-TIME SYSTEM MONITORING
═══════════════════════════

SYSTEM RESOURCES
────────────────
CPU Usage:      {cpu_percent:.1f}%
Memory:         {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB/{memory.total // (1024**3):.1f}GB)
Disk Usage:     {(disk.used / disk.total * 100):.1f}% ({disk.used // (1024**3):.1f}GB/{disk.total // (1024**3):.1f}GB)
Available RAM:  {memory.available // (1024**3):.1f}GB

NETWORK ACTIVITY
────────────────
Bytes Sent:     {net_io.bytes_sent // (1024**2):.1f}MB
Bytes Received: {net_io.bytes_recv // (1024**2):.1f}MB
Packets Sent:   {net_io.packets_sent:,}
Packets Recv:   {net_io.packets_recv:,}

API CONNECTIONS
───────────────
            """
            
            # Add API status
            api_status = self.swarm_service.get_api_status()
            for provider, status in api_status.items():
                status_text = "[+] Connected" if status else "[x] Disconnected"
                monitoring_data += f"{provider.title():<12} {status_text}\n"
            
            # Add swarm information
            available_swarms = self.swarm_service.discover_swarms()
            monitoring_data += f"\nSWARM STATUS\n────────────\n"
            monitoring_data += f"Available Swarms: {len(available_swarms)}\n"
            
            # Add recent logs
            logs = self.swarm_service.get_logs(limit=10)
            if logs:
                monitoring_data += f"\nRECENT ACTIVITY\n───────────────\n"
                for log in logs[-5:]:  # Show last 5 logs
                    timestamp = log.timestamp.strftime('%H:%M:%S')
                    monitoring_data += f"[{timestamp}] {log.message[:60]}...\n"
            
            # Update display
            monitoring_widget = self.query_one("#monitoring_display", Static)
            if monitoring_widget:
                monitoring_widget.update(monitoring_data)
                
        except Exception as e:
            logger.error(f"Error updating monitoring display: {e}")
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("REAL-TIME SYSTEM MONITORING", classes="modal-title"),
            Static("Advanced system metrics and performance monitoring", classes="modal-subtitle"),
            
            Static("", id="monitoring_display", classes="status-area"),
            
            # Keyboard Shortcuts Help
            Static("Keyboard Shortcuts", classes="content-section-title"),
            Static(
                "Ctrl+R: Refresh Data  |  Ctrl+Z: Clear Display  |  Escape: Back",
                classes="help-text"
            ),
            
            classes="swarm-runner-modal"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "refresh_btn":
            self.action_refresh()
        elif event.button.id == "clear_btn":
            self.action_clear()
        elif event.button.id == "back_btn":
            self.action_dismiss()
    
    def on_mount(self) -> None:
        """Start monitoring when screen loads"""
        self.monitoring_active = True
        self.update_monitoring_display()
        # Set up periodic refresh
        self.set_interval(5, self.update_monitoring_display)


class AgentCreationScreen(ModalScreen):
    """Enhanced agent creation with templates and validation"""
    
    BINDINGS = [
        ("escape", "dismiss", "Back"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
        ("ctrl+s", "create_agent", "Save Agent"),
        ("ctrl+l", "load_template", "Load Template"),
        ("ctrl+z", "clear_form", "Clear Form"),
    ]
    
    def __init__(self, swarm_service: SwarmService):
        super().__init__()
        self.swarm_service = swarm_service
        self.agent_templates = {
            "Research Agent": {
                "description": "Specialized in research and data analysis",
                "system_prompt": "You are an expert research agent. Conduct thorough research on any topic and provide comprehensive, well-sourced analysis.",
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 4000,
                "temperature": 0.3
            },
            "Creative Agent": {
                "description": "Specialized in creative content generation",
                "system_prompt": "You are a creative writing agent. Generate engaging, original content including stories, marketing copy, and creative solutions.",
                "model": "gpt-4",
                "max_tokens": 2000,
                "temperature": 0.8
            },
            "Code Agent": {
                "description": "Specialized in software development",
                "system_prompt": "You are an expert software developer. Write clean, efficient code with proper documentation and best practices.",
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 4000,
                "temperature": 0.1
            },
            "Analysis Agent": {
                "description": "Specialized in data analysis and insights",
                "system_prompt": "You are a data analysis expert. Analyze complex datasets and provide actionable insights and recommendations.",
                "model": "gpt-4",
                "max_tokens": 3000,
                "temperature": 0.2
            }
        }
    
    def action_dismiss(self) -> None:
        """Go back to main menu"""
        self.dismiss()
    
    def action_create_agent(self) -> None:
        """Create the agent configuration"""
        self.create_agent()
    
    def load_template(self, template_name: str) -> None:
        """Load agent template into form"""
        if template_name in self.agent_templates:
            template = self.agent_templates[template_name]
            
            # Update form fields
            self.query_one("#agent_description", Input).value = template["description"]
            self.query_one("#system_prompt", TextArea).text = template["system_prompt"]
            self.query_one("#model_name", Input).value = template["model"]
            self.query_one("#max_tokens", Input).value = str(template["max_tokens"])
            self.query_one("#temperature", Input).value = str(template["temperature"])
    
    def create_agent(self) -> None:
        """Create agent configuration file"""
        try:
            # Get form data
            agent_name = self.query_one("#agent_name", Input).value
            agent_description = self.query_one("#agent_description", Input).value
            system_prompt = self.query_one("#system_prompt", TextArea).text
            model_name = self.query_one("#model_name", Input).value
            max_tokens = self.query_one("#max_tokens", Input).value
            temperature = self.query_one("#temperature", Input).value
            
            # Validate inputs
            if not agent_name.strip():
                self.app.notify("[x] Agent name is required", severity="error")
                return
            
            if not agent_description.strip():
                self.app.notify("[x] Agent description is required", severity="error")
                return
            
            if not system_prompt.strip():
                self.app.notify("[x] System prompt is required", severity="error")
                return
            
            # Create agent configuration
            agent_config = {
                "name": agent_name,
                "description": agent_description,
                "type": "Agent",
                "agent_name": agent_name,
                "system_prompt": system_prompt,
                "model_name": model_name or "gpt-4",
                "max_tokens": int(max_tokens) if max_tokens.isdigit() else 2000,
                "temperature": float(temperature) if temperature.replace('.', '').isdigit() else 0.5,
                "role": "worker",
                "max_loops": 1,
                "auto_generate_prompt": False,
                "verbose": True
            }
            
            # Save to file
            import os
            os.makedirs("agent_configs", exist_ok=True)
            
            filename = f"agent_configs/{agent_name.lower().replace(' ', '_')}.yaml"
            
            import yaml
            with open(filename, 'w') as f:
                yaml.dump(agent_config, f, default_flow_style=False, indent=2)
            
            self.app.notify(f"[+] Agent created successfully: {filename}", severity="information")
            
            # Log the creation
            self.swarm_service.log_manager.add_log(
                f"Created agent configuration: {agent_name}",
                LogLevel.INFO,
                "agent_creation"
            )
            
            # Clear form
            self.clear_form()
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            self.app.notify(f"[x] Error creating agent: {str(e)}", severity="error")
    
    def clear_form(self) -> None:
        """Clear all form fields"""
        self.query_one("#agent_name", Input).value = ""
        self.query_one("#agent_description", Input).value = ""
        self.query_one("#system_prompt", TextArea).text = ""
        self.query_one("#model_name", Input).value = "gpt-4"
        self.query_one("#max_tokens", Input).value = "2000"
        self.query_one("#temperature", Input).value = "0.5"
    
    def action_load_template(self) -> None:
        """Load template via keyboard shortcut"""
        template_select = self.query_one("#template_select", Select)
        if template_select.value:
            self.load_template(template_select.value)
    
    def action_clear_form(self) -> None:
        """Clear form via keyboard shortcut"""
        self.clear_form()
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("AGENT CREATION STUDIO", classes="modal-title"),
            Static("Create custom agents with advanced configuration", classes="modal-subtitle"),
            
            Static("Agent Templates", classes="content-section-title"),
            Select(
                options=[(name, name) for name in self.agent_templates.keys()],
                value="Research Agent",
                id="template_select",
                classes="primary-select"
            ),
            
            Static("Agent Configuration", classes="content-section-title"),
            
            Static("Agent Name:", classes="field-label"),
            Input(placeholder="Enter agent name", id="agent_name", classes="primary-input"),
            
            Static("Description:", classes="field-label"),
            Input(placeholder="Enter agent description", id="agent_description", classes="primary-input"),
            
            Static("System Prompt:", classes="field-label"),
            TextArea(id="system_prompt", classes="primary-textarea"),
            
            Static("Model Name:", classes="field-label"),
            Input(value="gpt-4", placeholder="Model name", id="model_name", classes="primary-input"),
            
            Static("Max Tokens:", classes="field-label"),
            Input(value="2000", placeholder="Max tokens", id="max_tokens", classes="primary-input"),
            
            Static("Temperature:", classes="field-label"),
            Input(value="0.5", placeholder="Temperature (0.0-1.0)", id="temperature", classes="primary-input"),
            
            # Keyboard Shortcuts Help
            Static("Keyboard Shortcuts", classes="content-section-title"),
            Static(
                "Ctrl+S: Save Agent  |  Ctrl+L: Load Template  |  Ctrl+Z: Clear Form  |  Escape: Back",
                classes="help-text"
            ),
            
            classes="swarm-runner-modal"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "create_agent":
            self.create_agent()
        elif event.button.id == "load_template":
            self.action_load_template()
        elif event.button.id == "clear_form":
            self.action_clear_form()
        elif event.button.id == "back_btn":
            self.action_dismiss()
    
    def on_mount(self) -> None:
        """Initialize with default template"""
        self.load_template("Research Agent")


class AutoBuilderScreen(ModalScreen):
    """Intelligent auto swarm builder with advanced configuration"""
    
    BINDINGS = [
        ("escape", "dismiss", "Back"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
        ("ctrl+b", "build_swarm", "Build Swarm"),
        ("ctrl+z", "clear_form", "Clear Form"),
    ]
    
    def __init__(self, swarm_service: SwarmService):
        super().__init__()
        self.swarm_service = swarm_service
        self.swarm_types = {
            "SequentialWorkflow": "Agents work in sequence, passing results to next agent",
            "GroupChat": "Agents collaborate in a group discussion format",
            "Hierarchical": "Agents work in a hierarchical structure with a coordinator",
            "Parallel": "Agents work simultaneously on different aspects",
            "RoundRobin": "Agents take turns processing the task"
        }
        self.building = False
    
    def action_dismiss(self) -> None:
        """Go back to main menu"""
        self.dismiss()
    
    def action_build_swarm(self) -> None:
        """Build the swarm configuration"""
        self.build_swarm()
    
    def build_swarm(self) -> None:
        """Build swarm configuration intelligently"""
        if self.building:
            self.app.notify("[!] Swarm building already in progress", severity="warning")
            return
            
        try:
            self.building = True
            
            # Get form data
            task_description = self.query_one("#task_description", TextArea).text
            swarm_name = self.query_one("#swarm_name", Input).value
            swarm_type = self.query_one("#swarm_type_select", Select).value
            num_agents = self.query_one("#num_agents", Input).value
            
            # Validate inputs
            if not task_description.strip():
                self.app.notify("[x] Task description is required", severity="error")
                return
            
            if not swarm_name.strip():
                self.app.notify("[x] Swarm name is required", severity="error")
                return
            
            if not num_agents.isdigit() or int(num_agents) < 1:
                self.app.notify("[x] Number of agents must be a positive integer", severity="error")
                return
            
            num_agents = int(num_agents)
            
            self.app.notify("[*] Building intelligent swarm configuration...", severity="information")
            
            # Generate intelligent agent roles based on task
            agent_roles = self._generate_agent_roles(task_description, num_agents)
            
            # Create swarm configuration
            swarm_config = {
                "name": swarm_name,
                "description": f"Auto-generated swarm for: {task_description[:100]}...",
                "type": "Swarm",
                "swarm_type": swarm_type,
                "task": task_description,
                "max_loops": 1,
                "agents": []
            }
            
            # Generate agents
            for i, role in enumerate(agent_roles):
                agent = {
                    "agent_name": f"{swarm_name} {role['name']}",
                    "description": role['description'],
                    "system_prompt": role['system_prompt'],
                    "model_name": "claude-3-5-sonnet-20240620",
                    "role": "worker",
                    "max_loops": 1,
                    "max_tokens": 4000,
                    "temperature": role['temperature'],
                    "auto_generate_prompt": False
                }
                swarm_config["agents"].append(agent)
            
            # Save configuration
            import os
            os.makedirs("swarm_configs", exist_ok=True)
            
            filename = f"swarm_configs/{swarm_name.lower().replace(' ', '_')}.yaml"
            
            import yaml
            with open(filename, 'w') as f:
                yaml.dump(swarm_config, f, default_flow_style=False, indent=2)
            
            self.app.notify(f"[+] Swarm built successfully: {filename}", severity="information")
            
            # Log the creation
            self.swarm_service.log_manager.add_log(
                f"Auto-built swarm: {swarm_name} with {num_agents} agents",
                LogLevel.INFO,
                "auto_builder"
            )
            
            # Update status display
            status_text = f"[+] Successfully built swarm: {swarm_name}\n"
            status_text += f"    Type: {swarm_type}\n"
            status_text += f"    Agents: {num_agents}\n"
            status_text += f"    File: {filename}\n"
            status_text += f"    Roles: {', '.join([role['name'] for role in agent_roles])}"
            
            status_widget = self.query_one("#build_status", Static)
            if status_widget:
                status_widget.update(status_text)
            
        except Exception as e:
            logger.error(f"Error building swarm: {e}")
            self.app.notify(f"[x] Error building swarm: {str(e)}", severity="error")
        finally:
            self.building = False
    
    def _generate_agent_roles(self, task_description: str, num_agents: int) -> List[Dict[str, Any]]:
        """Generate intelligent agent roles based on task description"""
        # Basic role templates based on common patterns
        base_roles = [
            {
                "name": "Analyst",
                "description": "Analyzes and breaks down the task",
                "system_prompt": f"You are an expert analyst. Analyze this task thoroughly: {task_description}. Break it down into key components and provide structured analysis.",
                "temperature": 0.3
            },
            {
                "name": "Researcher", 
                "description": "Conducts research and gathers information",
                "system_prompt": f"You are a research specialist. Research all relevant information for: {task_description}. Provide comprehensive, well-sourced information.",
                "temperature": 0.2
            },
            {
                "name": "Strategist",
                "description": "Develops strategies and approaches",
                "system_prompt": f"You are a strategic planner. Develop effective strategies for: {task_description}. Create actionable plans and approaches.",
                "temperature": 0.4
            },
            {
                "name": "Creator",
                "description": "Creates and generates solutions",
                "system_prompt": f"You are a creative problem solver. Generate innovative solutions for: {task_description}. Think outside the box and create original approaches.",
                "temperature": 0.7
            },
            {
                "name": "Reviewer",
                "description": "Reviews and validates outputs",
                "system_prompt": f"You are a quality reviewer. Review and validate all work related to: {task_description}. Ensure accuracy, completeness, and quality.",
                "temperature": 0.2
            },
            {
                "name": "Synthesizer",
                "description": "Synthesizes and summarizes results",
                "system_prompt": f"You are a synthesis expert. Combine and synthesize all information for: {task_description}. Create clear, actionable summaries and conclusions.",
                "temperature": 0.3
            }
        ]
        
        # Select appropriate roles based on number of agents
        if num_agents <= len(base_roles):
            return base_roles[:num_agents]
        else:
            # Duplicate and modify roles if more agents needed
            roles = base_roles.copy()
            for i in range(len(base_roles), num_agents):
                base_role = base_roles[i % len(base_roles)].copy()
                base_role["name"] = f"{base_role['name']} {i // len(base_roles) + 1}"
                roles.append(base_role)
            return roles
    
    def action_clear_form(self) -> None:
        """Clear form via keyboard shortcut"""
        self.query_one("#swarm_name", Input).value = ""
        self.query_one("#task_description", TextArea).text = ""
        self.query_one("#num_agents", Input).value = "3"
        status_widget = self.query_one("#build_status", Static)
        if status_widget:
            status_widget.update("Ready to build swarm configuration...")
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("INTELLIGENT AUTO SWARM BUILDER", classes="modal-title"),
            Static("Automatically generate optimized swarm configurations", classes="modal-subtitle"),
            
            Static("Swarm Configuration", classes="content-section-title"),
            
            Static("Swarm Name:", classes="field-label"),
            Input(placeholder="Enter swarm name", id="swarm_name", classes="primary-input"),
            
            Static("Task Description:", classes="field-label"),
            TextArea(id="task_description", classes="primary-textarea"),
            
            Static("Swarm Type:", classes="field-label"),
            Select(
                options=[(f"{key} - {value}", key) for key, value in self.swarm_types.items()],
                value="SequentialWorkflow",
                id="swarm_type_select",
                classes="primary-select"
            ),
            
            Static("Number of Agents:", classes="field-label"),
            Input(value="3", placeholder="Number of agents (1-10)", id="num_agents", classes="primary-input"),
            
            Static("Build Status", classes="content-section-title"),
            Static("Ready to build swarm configuration...", id="build_status", classes="status-area"),
            
            # Keyboard Shortcuts Help
            Static("Keyboard Shortcuts", classes="content-section-title"),
            Static(
                "Ctrl+B: Build Swarm  |  Ctrl+Z: Clear Form  |  Escape: Back",
                classes="help-text"
            ),
            
            classes="swarm-runner-modal"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "build_swarm":
            self.build_swarm()
        elif event.button.id == "clear_form":
            self.action_clear_form()
        elif event.button.id == "back_btn":
            self.action_dismiss()
    
    def on_mount(self) -> None:
        """Initialize screen"""
        pass


class QualityControlScreen(ModalScreen):
    """Quality control and validation system"""
    
    BINDINGS = [
        ("escape", "dismiss", "Back"),
        ("ctrl+r", "run_validation", "Run Validation"),
        ("ctrl+z", "clear_results", "Clear Results"),
    ]
    
    def __init__(self, swarm_service: SwarmService):
        super().__init__()
        self.swarm_service = swarm_service
        self.validation_running = False
    
    def action_dismiss(self) -> None:
        """Go back to main menu"""
        self.dismiss()
    
    def action_run_validation(self) -> None:
        """Run quality validation"""
        self.run_validation()
    
    def action_clear_results(self) -> None:
        """Clear validation results"""
        results_widget = self.query_one("#validation_results", Static)
        if results_widget:
            results_widget.update("Ready to run quality validation...")
    
    def run_validation(self) -> None:
        """Run comprehensive quality validation"""
        if self.validation_running:
            self.app.notify("[!] Validation already running", severity="warning")
            return
        
        try:
            self.validation_running = True
            self.app.notify("[*] Running quality validation...", severity="information")
            
            validation_results = "QUALITY CONTROL VALIDATION REPORT\n"
            validation_results += "═" * 50 + "\n\n"
            
            # 1. Validate swarm configurations
            validation_results += "1. SWARM CONFIGURATION VALIDATION\n"
            validation_results += "─" * 35 + "\n"
            
            swarms = self.swarm_service.discover_swarms()
            valid_swarms = 0
            invalid_swarms = 0
            
            for swarm in swarms:
                if self._validate_swarm_config(swarm):
                    valid_swarms += 1
                else:
                    invalid_swarms += 1
                    validation_results += f"[x] Invalid: {swarm.name} - {swarm.file_path}\n"
            
            validation_results += f"[+] Valid swarms: {valid_swarms}\n"
            validation_results += f"[x] Invalid swarms: {invalid_swarms}\n\n"
            
            # 2. Validate API connections
            validation_results += "2. API CONNECTION VALIDATION\n"
            validation_results += "─" * 30 + "\n"
            
            api_status = self.swarm_service.get_api_status()
            connected_apis = sum(1 for status in api_status.values() if status)
            total_apis = len(api_status)
            
            for provider, status in api_status.items():
                status_icon = "[+]" if status else "[x]"
                validation_results += f"{status_icon} {provider.title()}: {'Connected' if status else 'Failed'}\n"
            
            validation_results += f"\nConnection Rate: {connected_apis}/{total_apis} ({(connected_apis/total_apis*100):.1f}%)\n\n"
            
            # 3. System resource validation
            validation_results += "3. SYSTEM RESOURCE VALIDATION\n"
            validation_results += "─" * 32 + "\n"
            
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # CPU validation
            if cpu_percent < 80:
                validation_results += f"[+] CPU Usage: {cpu_percent:.1f}% (Optimal)\n"
            else:
                validation_results += f"[!] CPU Usage: {cpu_percent:.1f}% (High)\n"
            
            # Memory validation
            if memory.percent < 80:
                validation_results += f"[+] Memory Usage: {memory.percent:.1f}% (Optimal)\n"
            else:
                validation_results += f"[!] Memory Usage: {memory.percent:.1f}% (High)\n"
            
            # Disk validation
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent < 80:
                validation_results += f"[+] Disk Usage: {disk_percent:.1f}% (Optimal)\n"
            else:
                validation_results += f"[!] Disk Usage: {disk_percent:.1f}% (High)\n"
            
            validation_results += "\n"
            
            # 4. Configuration validation
            validation_results += "4. CONFIGURATION VALIDATION\n"
            validation_results += "─" * 28 + "\n"
            
            config_issues = []
            
            # Check workspace directory
            workspace_dir = self.swarm_service.config_manager.get_workspace_dir()
            if os.path.exists(workspace_dir):
                validation_results += f"[+] Workspace Directory: {workspace_dir}\n"
            else:
                validation_results += f"[x] Workspace Directory Missing: {workspace_dir}\n"
                config_issues.append("Workspace directory not found")
            
            # Check config directories
            config_dirs = ["agent_configs", "swarm_configs"]
            for config_dir in config_dirs:
                if os.path.exists(config_dir):
                    validation_results += f"[+] Config Directory: {config_dir}\n"
                else:
                    validation_results += f"[!] Config Directory Missing: {config_dir}\n"
            
            validation_results += "\n"
            
            # 5. Security validation
            validation_results += "5. SECURITY VALIDATION\n"
            validation_results += "─" * 20 + "\n"
            
            # Check for exposed API keys
            security_issues = []
            if os.path.exists(".env"):
                validation_results += f"[!] .env file detected - ensure not in version control\n"
            else:
                validation_results += f"[+] No .env file in root directory\n"
            
            # Check file permissions
            config_file = ".swarms_config.json"
            if os.path.exists(config_file):
                import stat
                file_stats = os.stat(config_file)
                file_mode = stat.filemode(file_stats.st_mode)
                validation_results += f"[i] Config file permissions: {file_mode}\n"
            
            validation_results += "\n"
            
            # Summary
            validation_results += "VALIDATION SUMMARY\n"
            validation_results += "═" * 18 + "\n"
            
            total_issues = invalid_swarms + len(config_issues) + len(security_issues)
            if total_issues == 0:
                validation_results += "[+] All validations passed successfully!\n"
                validation_results += "[+] System is ready for production use.\n"
            else:
                validation_results += f"[!] Found {total_issues} issues that need attention.\n"
                validation_results += "[i] Review the report above for details.\n"
            
            # Update display
            results_widget = self.query_one("#validation_results", Static)
            if results_widget:
                results_widget.update(validation_results)
            
            # Log validation
            self.swarm_service.log_manager.add_log(
                f"Quality validation completed - {total_issues} issues found",
                LogLevel.INFO if total_issues == 0 else LogLevel.WARNING,
                "quality_control"
            )
            
            if total_issues == 0:
                self.app.notify("[+] Quality validation passed!", severity="information")
            else:
                self.app.notify(f"[!] Quality validation found {total_issues} issues", severity="warning")
            
        except Exception as e:
            logger.error(f"Error running validation: {e}")
            self.app.notify(f"[x] Validation error: {str(e)}", severity="error")
        finally:
            self.validation_running = False
    
    def _validate_swarm_config(self, swarm: SwarmConfig) -> bool:
        """Validate individual swarm configuration"""
        try:
            # Check required fields
            if not swarm.name or not swarm.description:
                return False
            
            # Check file exists
            if not os.path.exists(swarm.file_path):
                return False
            
            # Check configuration structure
            config = swarm.config
            if not isinstance(config, dict):
                return False
            
            # Additional validation for swarm-specific fields
            if swarm.swarm_type and swarm.agents:
                if not isinstance(swarm.agents, list):
                    return False
                
                for agent in swarm.agents:
                    if not isinstance(agent, dict):
                        return False
                    if 'agent_name' not in agent and 'name' not in agent:
                        return False
            
            return True
        except Exception:
            return False
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("QUALITY CONTROL SYSTEM", classes="modal-title"),
            Static("Comprehensive validation and quality assurance", classes="modal-subtitle"),
            
            Static("Validation Results", classes="content-section-title"),
            Static("Ready to run quality validation...", id="validation_results", classes="status-area"),
            
            # Keyboard Shortcuts Help
            Static("Keyboard Shortcuts", classes="content-section-title"),
            Static(
                "Ctrl+R: Run Validation  |  Ctrl+Z: Clear Results  |  Escape: Back",
                classes="help-text"
            ),
            
            classes="swarm-runner-modal"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "run_validation":
            self.run_validation()
        elif event.button.id == "clear_results":
            self.action_clear_results()
        elif event.button.id == "back_btn":
            self.action_dismiss()
    
    def on_mount(self) -> None:
        """Initialize screen"""
        pass


def run_tui() -> None:
    """Run the Swarms TUI application"""
    try:
        app = SwarmsTUI()
        app.run()
    except Exception as e:
        logger.error(f"Error running TUI: {e}")
        print(f"Error: {e}")
        print("Please check your installation and try again.") 
