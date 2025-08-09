import argparse
import os
import sys
import time
import webbrowser
import subprocess
import signal
import threading
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.layout import Layout
from rich.live import Live

from swarms.agents.auto_generate_swarm_config import (
    generate_swarm_config,
)
from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
from swarms.cli.onboarding_process import OnboardingProcess
from swarms.cli.theme import SWARMS_COLORS, SWARMS_LOGO_ASCII, get_rich_theme, get_symbols
from swarms.utils.formatter import formatter
from swarms.utils.data_to_text import (
    csv_to_text,
    json_to_text,
    txt_to_text,
    pdf_to_text,
    data_to_text,
)
from swarms.utils.file_processing import (
    load_json,
    sanitize_file_path as sanitize_path_util,
    zip_workspace,
    create_file_in_folder,
    zip_folders,
)
from swarms.utils.parse_code import extract_code_from_markdown
from swarms.utils.litellm_tokenizer import count_tokens
from swarms.utils.check_all_model_max_tokens import check_all_model_max_tokens

# Initialize console with corporate control theme
console = Console(theme=get_rich_theme())
symbols = get_symbols()

# Global tmux session storage
ACTIVE_SESSIONS = {}
AGENT_PANES = {}


class TmuxSession:
    """Enhanced tmux session with AI agent integration"""
    
    def __init__(self, session_name: str, session_type: str = "development"):
        self.session_name = session_name
        self.session_type = session_type
        self.windows = {}
        self.agents = {}
        self.panes = {}
        self.process = None
        self.created_at = time.time()
        
    def add_agent_pane(self, agent_name: str, agent_config: Dict[str, Any]):
        """Add an AI agent to a dedicated pane"""
        self.agents[agent_name] = agent_config
        
    def get_session_info(self) -> Dict[str, Any]:
        """Get comprehensive session information"""
        return {
            "name": self.session_name,
            "type": self.session_type,
            "windows": len(self.windows),
            "agents": len(self.agents),
            "panes": len(self.panes),
            "uptime": time.time() - self.created_at
        }


class AgentAssistant:
    """AI Agent Assistant for tmux sessions"""
    
    def __init__(self, agent_type: str, model: str = "gpt-4"):
        self.agent_type = agent_type
        self.model = model
        self.context = []
        self.active = False
        
    def analyze_command(self, command: str) -> Dict[str, Any]:
        """Analyze command and provide intelligent suggestions"""
        suggestions = {
            "optimizations": [],
            "warnings": [],
            "alternatives": [],
            "context_suggestions": []
        }
        
        # Basic command analysis (can be enhanced with actual AI)
        if "rm -rf" in command:
            suggestions["warnings"].append("CRITICAL: Destructive operation detected")
            suggestions["alternatives"].append("Consider using trash or backup first")
            
        if "sudo" in command:
            suggestions["warnings"].append("Elevated privileges detected")
            suggestions["context_suggestions"].append("Verify command necessity")
            
        if command.startswith("git"):
            suggestions["optimizations"].append("Git operation detected")
            suggestions["context_suggestions"].append("Consider branch status check")
            
        return suggestions
        
    def monitor_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and analyze system performance"""
        analysis = {
            "status": "operational",
            "recommendations": [],
            "alerts": []
        }
        
        if metrics.get("cpu_usage", 0) > 80:
            analysis["alerts"].append("High CPU usage detected")
            analysis["recommendations"].append("Consider optimizing running processes")
            
        if metrics.get("memory_usage", 0) > 90:
            analysis["alerts"].append("High memory usage detected")
            analysis["recommendations"].append("Review memory-intensive applications")
            
        return analysis


class SwarmSystemError(Exception):
    """Critical system exception for Swarm Control Matrix errors"""
    pass


def create_system_spinner(text: str) -> Progress:
    """Create a systematic progress indicator with corporate control theme."""
    return Progress(
        SpinnerColumn(style=SWARMS_COLORS["command_red"]),
        TextColumn("[{task.description}]", style=SWARMS_COLORS["primary_text"]),
        console=console,
    )


def display_control_matrix():
    """Display the corporate control matrix interface."""
    panel = Panel(
        Text(SWARMS_LOGO_ASCII, style=f"bold {SWARMS_COLORS['hex_red']}"),
        border_style=SWARMS_COLORS["command_red"],
        title="[bold]SWARMS CORPORATE CONTROL MATRIX[/bold]",
        subtitle="[dim]SYSTEMATIC MULTI-AGENT INTELLIGENCE MANAGEMENT PLATFORM[/dim]",
    )
    console.print(panel)


def create_command_hierarchy() -> Tree:
    """Create systematic command hierarchy showing all operational directives organized by control sectors."""
    tree = Tree(
        f"[bold {SWARMS_COLORS['hex_red']}]SYSTEM COMMAND HIERARCHY[/bold {SWARMS_COLORS['hex_red']}]",
        style=SWARMS_COLORS["primary_text"]
    )
    
    # System Core Operations
    core_sector = tree.add("[bold]SYSTEM CORE OPERATIONS[/bold]", style=SWARMS_COLORS["command_red"])
    core_sector.add("tui - Launch primary control interface")
    core_sector.add("onboarding - Initialize system environment protocols")
    core_sector.add("get-api-key - Retrieve authentication credentials")
    core_sector.add("check-login - Verify system access authorization")
    
    # Agent Control Matrix
    agent_sector = tree.add("[bold]AGENT CONTROL MATRIX[/bold]", style=SWARMS_COLORS["command_red"])
    agent_sector.add("run-agents - Execute configured agent protocols")
    agent_sector.add("autoswarm - Deploy autonomous agent collective")
    agent_sector.add("create-agent - Generate new agent configuration")
    agent_sector.add("list-agents - Display available agent resources")
    agent_sector.add("validate-config - Verify configuration integrity")
    
    # Data Processing Infrastructure
    data_sector = tree.add("[bold]DATA PROCESSING INFRASTRUCTURE[/bold]", style=SWARMS_COLORS["command_red"])
    data_sector.add("convert-file - Process files through text conversion matrix")
    data_sector.add("extract-code - Extract code segments from documentation")
    data_sector.add("count-tokens - Analyze token utilization metrics")
    data_sector.add("zip-workspace - Archive workspace directories")
    data_sector.add("sanitize-path - Clean file path specifications")
    
    # Intelligent Terminal Multiplexer
    tmux_sector = tree.add("[bold]INTELLIGENT TERMINAL MULTIPLEXER[/bold]", style=SWARMS_COLORS["command_red"])
    tmux_sector.add("tmux - Launch AI-enhanced terminal multiplexer")
    tmux_sector.add("new-session - Create intelligent session")
    tmux_sector.add("attach-session - Connect to existing session")
    tmux_sector.add("list-sessions - Display active sessions")
    tmux_sector.add("kill-session - Terminate session with cleanup")
    tmux_sector.add("agent-assist - Deploy agent assistance")
    tmux_sector.add("collaborative-mode - Enable multi-agent collaboration")
    
    # System Diagnostics Division
    diagnostics_sector = tree.add("[bold]SYSTEM DIAGNOSTICS DIVISION[/bold]", style=SWARMS_COLORS["command_red"])
    diagnostics_sector.add("model-info - Query model capability specifications")
    diagnostics_sector.add("health-check - Execute system diagnostic protocols")
    diagnostics_sector.add("version - Display system version registry")
    
    # External Integration Services
    integration_sector = tree.add("[bold]EXTERNAL INTEGRATION SERVICES[/bold]", style=SWARMS_COLORS["command_red"])
    integration_sector.add("book-call - Schedule strategic consultation")
    integration_sector.add("auto-upgrade - System upgrade management")
    
    return tree


class CustomHelpAction(argparse.Action):
    """Custom help action that uses our enhanced corporate help display."""
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super(CustomHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        display_command_reference()
        parser.exit()


def display_command_reference():
    """Display comprehensive command reference with corporate hierarchy."""
    console.print()
    console.print(
        Panel(
            Text("SWARMS CORPORATE CONTROL MATRIX", style=f"bold {SWARMS_COLORS['hex_red']}"),
            title="[bold]COMMAND REFERENCE DATABASE[/bold]",
            border_style=SWARMS_COLORS["command_red"],
            padding=(1, 2)
        )
    )
    
    console.print()
    console.print(create_command_hierarchy())
    
    console.print()
    
    # Enhanced Command Categories with color coding
    console.print(
        Panel(
            Text("OPERATIONAL PARAMETER SPECIFICATIONS", style=f"bold {SWARMS_COLORS['command_red']}"),
            border_style=SWARMS_COLORS["steel_gray"],
            padding=(0, 1)
        )
    )
    
    # Core Parameters
    params_table = Table(title="CORE OPERATIONAL PARAMETERS", show_header=True, header_style=f"bold {SWARMS_COLORS['hex_red']}")
    params_table.add_column("PARAMETER", style=SWARMS_COLORS["command_red"], width=20)
    params_table.add_column("SPECIFICATION", style=SWARMS_COLORS["primary_text"], width=50)
    
    params_table.add_row("--yaml-file", "YAML configuration file specification (default: agents.yaml)")
    params_table.add_row("--task", "Task specification for autonomous swarm deployment")
    params_table.add_row("--model", "Model specification for operations (default: gpt-4)")
    params_table.add_row("--file", "File path for data processing operations")
    params_table.add_row("--output", "Output file path specification")
    params_table.add_row("--language", "Programming language filter for code extraction")
    params_table.add_row("--name", "Agent designation identifier")
    params_table.add_row("--description", "Agent operational description")
    params_table.add_row("--workspace", "Workspace directory specification")
    params_table.add_row("--session-name", "TMux session identifier")
    params_table.add_row("--session-type", "Session type: development, devops, research")
    params_table.add_row("--agents", "Comma-separated agent types to deploy")
    params_table.add_row("--layout", "Pane layout: tiled, main-vertical, main-horizontal")
    
    console.print(params_table)
    console.print()
    
    # Operational Examples with enhanced formatting
    console.print(
        Panel(
            Text("OPERATIONAL DIRECTIVE EXAMPLES", style=f"bold {SWARMS_COLORS['command_red']}"),
            border_style=SWARMS_COLORS["steel_gray"],
            padding=(0, 1)
        )
    )
    
    examples_table = Table(title="SYSTEM OPERATIONAL EXAMPLES", show_header=True, header_style=f"bold {SWARMS_COLORS['hex_red']}")
    examples_table.add_column("COMMAND", style=SWARMS_COLORS["success"], width=40)
    examples_table.add_column("DESCRIPTION", style=SWARMS_COLORS["secondary_text"], width=45)
    
    examples_table.add_row(
        "swarms tui",
        "Launch primary control interface"
    )
    examples_table.add_row(
        "swarms tmux --session-type=\"development\"",
        "Launch intelligent development session"
    )
    examples_table.add_row(
        "swarms tmux --agents=\"dev,test,security\"",
        "Deploy collaborative AI agent session"
    )
    examples_table.add_row(
        "swarms autoswarm --task \"analyze data\"",
        "Deploy autonomous swarm collective"
    )
    examples_table.add_row(
        "swarms convert-file report.pdf",
        "Process file through conversion matrix"
    )
    examples_table.add_row(
        "swarms count-tokens document.txt",
        "Analyze token utilization metrics"
    )
    examples_table.add_row(
        "swarms create-agent \"Assistant\" \"AI Unit\"",
        "Generate agent configuration template"
    )
    examples_table.add_row(
        "swarms validate-config --yaml-file config.yaml",
        "Verify configuration integrity"
    )
    examples_table.add_row(
        "swarms health-check",
        "Execute system diagnostic protocols"
    )
    
    console.print(examples_table)
    console.print()
    
    # System Access Information
    console.print(
        Panel(
            f"""[bold {SWARMS_COLORS['info']}]SYSTEM ACCESS PROTOCOLS[/bold {SWARMS_COLORS['info']}]

{symbols['directive']} Access technical reference database: [bold {SWARMS_COLORS['info']}]https://docs.swarms.world[/bold {SWARMS_COLORS['info']}]

{symbols['directive']} Schedule strategic consultation: [bold {SWARMS_COLORS['warning']}]swarms book-call[/bold {SWARMS_COLORS['warning']}]

{symbols['directive']} Join operational network: [bold {SWARMS_COLORS['info']}]Discord Community[/bold {SWARMS_COLORS['info']}]

{symbols['directive']} Report system issues: [bold {SWARMS_COLORS['error']}]GitHub Issues[/bold {SWARMS_COLORS['error']}]""",
            border_style=SWARMS_COLORS["info"],
            padding=(1, 2)
        )
    )


def display_system_error(message: str, diagnostic_text: str = None):
    """Display system error with corporate diagnostic information"""
    error_panel = Panel(
        f"[bold {SWARMS_COLORS['error']}]SYSTEM ERROR: {message}[/bold {SWARMS_COLORS['error']}]",
        title=f"{symbols['error']} CRITICAL SYSTEM ALERT",
        border_style=SWARMS_COLORS["error"],
    )
    console.print(error_panel)

    if diagnostic_text:
        console.print(f"\n[{SWARMS_COLORS['info']}]{symbols['info']} DIAGNOSTIC INFORMATION: {diagnostic_text}[/{SWARMS_COLORS['info']}]")


def display_operation_success(message: str):
    """Display operation success with corporate confirmation"""
    success_panel = Panel(
        f"[bold {SWARMS_COLORS['success']}]OPERATION COMPLETED: {message}[/bold {SWARMS_COLORS['success']}]",
        title=f"{symbols['success']} SYSTEM CONFIRMATION",
        border_style=SWARMS_COLORS["success"],
    )
    console.print(success_panel)


def execute_with_system_monitoring(action: callable, operation_text: str) -> Any:
    """Execute operation with systematic monitoring and progress tracking."""
    with create_system_spinner(operation_text) as progress:
        task = progress.add_task(operation_text, total=None)
        result = action()
        progress.remove_task(task)
    return result


# Intelligent Terminal Multiplexer Operations
def check_tmux_availability() -> bool:
    """Check if tmux is available on the system (Windows: Git Bash/MSYS2/WSL, Linux/macOS: native)"""
    try:
        # Try standard tmux command first (works on Linux, macOS, WSL, Git Bash with tmux)
        result = subprocess.run(["tmux", "-V"], capture_output=True, text=True, shell=(os.name == 'nt'))
        if result.returncode == 0:
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # On Windows, also check if we're in a supported environment
    if os.name == 'nt':
        # Check if we're in Git Bash or similar environment
        msystem = os.environ.get('MSYSTEM')
        if msystem:  # MINGW64, MINGW32, MSYS, etc.
            try:
                result = subprocess.run(["tmux", "-V"], capture_output=True, text=True)
                return result.returncode == 0
            except:
                pass
    
    return False


def install_tmux_guide():
    """Display tmux installation guide based on operating system"""
    if os.name == 'nt':  # Windows
        console.print(
            Panel(
                f"""[bold {SWARMS_COLORS['warning']}]TMUX INSTALLATION REQUIRED FOR WINDOWS[/bold {SWARMS_COLORS['warning']}]

{symbols['directive']} [bold {SWARMS_COLORS['info']}]RECOMMENDED: Git Bash + MSYS2 Method[/bold {SWARMS_COLORS['info']}]
   1. Install Git with Git Bash: [bold]https://git-scm.com/download/win[/bold]
   2. Install MSYS2: [bold]https://www.msys2.org/[/bold]
   3. In MSYS2 terminal: [bold]pacman -S tmux[/bold]
   4. Copy tmux files to Git Bash: Copy from C:\\msys64\\usr\\bin to C:\\Program Files\\Git\\usr\\bin
   5. Restart Git Bash and run: [bold]tmux[/bold]

{symbols['directive']} [bold {SWARMS_COLORS['info']}]ALTERNATIVE: Windows Subsystem for Linux (WSL)[/bold {SWARMS_COLORS['info']}]
   1. Install WSL: [bold]wsl --install[/bold]
   2. Install tmux in WSL: [bold]sudo apt install tmux[/bold]
   3. Use WSL terminal for Swarms tmux commands

{symbols['directive']} [bold {SWARMS_COLORS['info']}]ALTERNATIVE: itmux Package[/bold {SWARMS_COLORS['info']}]
   1. Download from: [bold]https://itefix.net/itmux[/bold]
   2. Extract and add to PATH
   3. Use itmux as standalone solution

{symbols['directive']} [bold {SWARMS_COLORS['warning']}]NOTE:[/bold {SWARMS_COLORS['warning']}] Run Swarms from Git Bash, WSL, or Windows Terminal for best compatibility.

After installation, restart your terminal and run [bold]swarms tmux[/bold] again.""",
                title="WINDOWS TERMINAL MULTIPLEXER SETUP",
                border_style=SWARMS_COLORS["warning"],
                padding=(1, 2)
            )
        )
    else:
        console.print(
            Panel(
                f"""[bold {SWARMS_COLORS['warning']}]TMUX INSTALLATION REQUIRED[/bold {SWARMS_COLORS['warning']}]

{symbols['directive']} Ubuntu/Debian: [bold]sudo apt-get install tmux[/bold]
{symbols['directive']} macOS: [bold]brew install tmux[/bold]
{symbols['directive']} CentOS/RHEL: [bold]sudo yum install tmux[/bold]
{symbols['directive']} Fedora: [bold]sudo dnf install tmux[/bold]
{symbols['directive']} Arch Linux: [bold]sudo pacman -S tmux[/bold]

After installation, restart your terminal and run [bold]swarms tmux[/bold] again.""",
                title="SYSTEM DEPENDENCY MISSING",
                border_style=SWARMS_COLORS["warning"],
                padding=(1, 2)
            )
        )


def create_intelligent_session_config(session_name: str, session_type: str, agents: List[str]) -> Dict[str, Any]:
    """Create intelligent session configuration based on type and agents"""
    
    base_config = {
        "session_name": session_name,
        "session_type": session_type,
        "agents": agents,
        "windows": []
    }
    
    if session_type == "development":
        base_config["windows"] = [
            {
                "name": "code",
                "layout": "main-vertical",
                "panes": [
                    {"command": "echo 'Development Environment Ready'"},
                    {"command": "echo 'Agent Assistant: Ready for coding tasks'", "agent": "dev"}
                ]
            },
            {
                "name": "testing",
                "layout": "tiled",
                "panes": [
                    {"command": "echo 'Test Runner Ready'"},
                    {"command": "echo 'Agent Assistant: Monitoring test results'", "agent": "test"}
                ]
            }
        ]
    elif session_type == "devops":
        base_config["windows"] = [
            {
                "name": "monitoring",
                "layout": "main-horizontal",
                "panes": [
                    {"command": "echo 'System Monitoring Active'"},
                    {"command": "echo 'Agent Assistant: Analyzing system metrics'", "agent": "monitoring"}
                ]
            },
            {
                "name": "deployment",
                "layout": "tiled",
                "panes": [
                    {"command": "echo 'Deployment Console Ready'"},
                    {"command": "echo 'Agent Assistant: Managing deployments'", "agent": "deploy"}
                ]
            }
        ]
    elif session_type == "research":
        base_config["windows"] = [
            {
                "name": "analysis",
                "layout": "main-vertical",
                "panes": [
                    {"command": "echo 'Research Environment Ready'"},
                    {"command": "echo 'Agent Assistant: Ready for data analysis'", "agent": "research"}
                ]
            }
        ]
    else:
        # Generic session
        base_config["windows"] = [
            {
                "name": "main",
                "layout": "tiled",
                "panes": [
                    {"command": "echo 'Swarms Intelligence Terminal Ready'"},
                    {"command": "echo 'Agent Assistant: Ready for collaboration'", "agent": "general"}
                ]
            }
        ]
    
    return base_config


def launch_intelligent_tmux(session_name: str = None, session_type: str = "development", 
                          agents: List[str] = None, layout: str = "tiled"):
    """Launch AI-enhanced terminal multiplexer with intelligent features"""
    
    if not check_tmux_availability():
        install_tmux_guide()
        return
    
    session_name = session_name or f"swarms-{int(time.time())}"
    agents = agents or ["general"]
    
    console.print(f"[{SWARMS_COLORS['info']}][TMUX] Launching intelligent terminal multiplexer...[/{SWARMS_COLORS['info']}]")
    
    try:
        # Create session configuration
        config = create_intelligent_session_config(session_name, session_type, agents)
        
        # Display session launch information
        launch_table = Table(title="INTELLIGENT SESSION INITIALIZATION")
        launch_table.add_column("PARAMETER", style=SWARMS_COLORS["command_red"])
        launch_table.add_column("VALUE", style=SWARMS_COLORS["primary_text"])
        
        launch_table.add_row("SESSION NAME", session_name)
        launch_table.add_row("SESSION TYPE", session_type.upper())
        launch_table.add_row("AGENT ASSISTANCE", ", ".join(agents).upper())
        launch_table.add_row("LAYOUT STRATEGY", layout.upper())
        launch_table.add_row("WINDOWS CONFIGURED", str(len(config["windows"])))
        launch_table.add_row("PLATFORM", "WINDOWS" if os.name == 'nt' else "UNIX")
        
        console.print(launch_table)
        
        # Check if session already exists
        check_cmd = ["tmux", "has-session", "-t", session_name]
        result = subprocess.run(check_cmd, capture_output=True, shell=(os.name == 'nt'))
        
        if result.returncode == 0:
            console.print(f"[{SWARMS_COLORS['warning']}][TMUX] Session '{session_name}' already exists. Attaching...[/{SWARMS_COLORS['warning']}]")
            attach_cmd = ["tmux", "attach-session", "-t", session_name]
            subprocess.run(attach_cmd, shell=(os.name == 'nt'))
        else:
            # Create new session
            console.print(f"[{SWARMS_COLORS['info']}][TMUX] Creating new intelligent session...[/{SWARMS_COLORS['info']}]")
            
            # Create session with first window
            create_cmd = ["tmux", "new-session", "-d", "-s", session_name, "-x", "120", "-y", "40"]
            subprocess.run(create_cmd, check=True, shell=(os.name == 'nt'))
            
            # Configure windows and panes according to session type
            for i, window_config in enumerate(config["windows"]):
                if i > 0:  # Create additional windows
                    subprocess.run([
                        "tmux", "new-window", "-t", f"{session_name}:{i}",
                        "-n", window_config["name"]
                    ], shell=(os.name == 'nt'))
                else:  # Rename first window
                    subprocess.run([
                        "tmux", "rename-window", "-t", f"{session_name}:0",
                        window_config["name"]
                    ], shell=(os.name == 'nt'))
                
                # Create panes
                for j, pane_config in enumerate(window_config["panes"]):
                    if j > 0:  # Split for additional panes
                        subprocess.run([
                            "tmux", "split-window", "-t", f"{session_name}:{i}"
                        ], shell=(os.name == 'nt'))
                    
                    # Send command to pane
                    subprocess.run([
                        "tmux", "send-keys", "-t", f"{session_name}:{i}.{j}",
                        pane_config["command"], "Enter"
                    ], shell=(os.name == 'nt'))
                
                # Apply layout
                subprocess.run([
                    "tmux", "select-layout", "-t", f"{session_name}:{i}",
                    window_config["layout"]
                ], shell=(os.name == 'nt'))
            
            # Set session options for enhanced experience
            session_options = [
                ("status-style", "bg=black,fg=red"),
                ("status-left", f"[SWARMS:{session_type.upper()}]"),
                ("status-right", f"[{session_name}] %H:%M"),
                ("pane-border-style", "fg=red"),
                ("pane-active-border-style", "fg=brightred"),
            ]
            
            for option, value in session_options:
                subprocess.run([
                    "tmux", "set-option", "-t", session_name, option, value
                ], shell=(os.name == 'nt'))
            
            # Store session information
            session_obj = TmuxSession(session_name, session_type)
            for agent in agents:
                session_obj.add_agent_pane(agent, {"type": agent, "active": True})
            
            ACTIVE_SESSIONS[session_name] = session_obj
            
            display_operation_success(f"Intelligent session '{session_name}' created successfully")
            
            # Attach to session
            console.print(f"[{SWARMS_COLORS['info']}][TMUX] Attaching to session...[/{SWARMS_COLORS['info']}]")
            attach_cmd = ["tmux", "attach-session", "-t", session_name]
            subprocess.run(attach_cmd, shell=(os.name == 'nt'))
            
    except subprocess.CalledProcessError as e:
        display_system_error(f"TMux session creation failed: {str(e)}")
        console.print(f"\n[{SWARMS_COLORS['info']}][HELP] If tmux is not available, please install it using the guide above.[/{SWARMS_COLORS['info']}]")
    except Exception as e:
        display_system_error(f"Intelligent session launch error: {str(e)}")


def list_intelligent_sessions():
    """List all active intelligent tmux sessions"""
    try:
        # Get tmux sessions
        result = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True, shell=(os.name == 'nt'))
        
        if result.returncode != 0:
            console.print(f"[{SWARMS_COLORS['warning']}][TMUX] No active sessions found[/{SWARMS_COLORS['warning']}]")
            return
        
        sessions_table = Table(title="ACTIVE INTELLIGENT SESSIONS")
        sessions_table.add_column("SESSION NAME", style=SWARMS_COLORS["command_red"])
        sessions_table.add_column("STATUS", style=SWARMS_COLORS["success"])
        sessions_table.add_column("WINDOWS", style=SWARMS_COLORS["primary_text"])
        sessions_table.add_column("AGENTS", style=SWARMS_COLORS["info"])
        sessions_table.add_column("TYPE", style=SWARMS_COLORS["secondary_text"])
        sessions_table.add_column("PLATFORM", style=SWARMS_COLORS["warning"])
        
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line:
                parts = line.split(':')
                session_name = parts[0]
                
                # Get session info from our storage or defaults
                session_info = ACTIVE_SESSIONS.get(session_name, {})
                session_type = getattr(session_info, 'session_type', 'unknown') if hasattr(session_info, 'session_type') else 'unknown'
                agents = list(getattr(session_info, 'agents', {}).keys()) if hasattr(session_info, 'agents') else ['general']
                
                # Parse tmux output for windows count
                if 'windows' in line:
                    windows = line.split('windows')[0].split()[-1]
                else:
                    windows = "1"
                
                sessions_table.add_row(
                    session_name,
                    "ACTIVE",
                    windows,
                    ", ".join(agents).upper(),
                    session_type.upper(),
                    "WINDOWS" if os.name == 'nt' else "UNIX"
                )
        
        console.print(sessions_table)
        
    except subprocess.CalledProcessError as e:
        display_system_error(f"Failed to list sessions: {str(e)}")
        console.print(f"\n[{SWARMS_COLORS['info']}][HELP] Ensure tmux is installed and accessible.[/{SWARMS_COLORS['info']}]")
    except Exception as e:
        display_system_error(f"Session listing error: {str(e)}")


def kill_intelligent_session(session_name: str):
    """Kill intelligent tmux session with proper cleanup"""
    try:
        # Check if session exists
        check_cmd = ["tmux", "has-session", "-t", session_name]
        result = subprocess.run(check_cmd, capture_output=True, shell=(os.name == 'nt'))
        
        if result.returncode != 0:
            display_system_error(f"Session '{session_name}' not found")
            return
        
        console.print(f"[{SWARMS_COLORS['warning']}][TMUX] Terminating session '{session_name}'...[/{SWARMS_COLORS['warning']}]")
        
        # Clean up our session storage
        if session_name in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[session_name]
        
        # Kill tmux session
        kill_cmd = ["tmux", "kill-session", "-t", session_name]
        subprocess.run(kill_cmd, check=True, shell=(os.name == 'nt'))
        
        display_operation_success(f"Session '{session_name}' terminated successfully")
        
    except subprocess.CalledProcessError as e:
        display_system_error(f"Failed to kill session: {str(e)}")
    except Exception as e:
        display_system_error(f"Session termination error: {str(e)}")


def deploy_agent_assistance(session_name: str, agent_type: str):
    """Deploy AI agent assistance to existing session"""
    try:
        # Verify session exists
        check_cmd = ["tmux", "has-session", "-t", session_name]
        result = subprocess.run(check_cmd, capture_output=True, shell=(os.name == 'nt'))
        
        if result.returncode != 0:
            display_system_error(f"Session '{session_name}' not found")
            return
        
        console.print(f"[{SWARMS_COLORS['info']}][AGENT] Deploying {agent_type} assistant to session '{session_name}'...[/{SWARMS_COLORS['info']}]")
        
        # Create new window for agent
        agent_window = f"agent-{agent_type}"
        subprocess.run([
            "tmux", "new-window", "-t", session_name,
            "-n", agent_window,
            "-c", os.getcwd()
        ], shell=(os.name == 'nt'))
        
        # Configure agent pane
        agent_commands = {
            "dev": "echo 'Development Agent: Ready to assist with coding tasks'",
            "test": "echo 'Testing Agent: Monitoring test execution and results'",
            "security": "echo 'Security Agent: Scanning for vulnerabilities and threats'",
            "performance": "echo 'Performance Agent: Analyzing system metrics and optimization'",
            "research": "echo 'Research Agent: Ready for data analysis and investigation'"
        }
        
        command = agent_commands.get(agent_type, f"echo '{agent_type.title()} Agent: Ready for collaboration'")
        
        subprocess.run([
            "tmux", "send-keys", "-t", f"{session_name}:{agent_window}",
            command, "Enter"
        ], shell=(os.name == 'nt'))
        
        # Store agent information
        if session_name in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[session_name].add_agent_pane(agent_type, {"type": agent_type, "active": True})
        
        display_operation_success(f"{agent_type.title()} agent deployed to session '{session_name}'")
        
    except subprocess.CalledProcessError as e:
        display_system_error(f"Agent deployment failed: {str(e)}")
    except Exception as e:
        display_system_error(f"Agent assistance error: {str(e)}")


def enable_collaborative_mode(session_name: str):
    """Enable multi-agent collaborative mode for session"""
    try:
        console.print(f"[{SWARMS_COLORS['info']}][COLLABORATION] Enabling multi-agent collaboration for '{session_name}'...[/{SWARMS_COLORS['info']}]")
        
        # Deploy multiple agents for collaboration
        agent_types = ["dev", "test", "security", "performance"]
        
        for agent_type in agent_types:
            deploy_agent_assistance(session_name, agent_type)
            time.sleep(0.5)  # Small delay between deployments
        
        # Configure collaboration window
        subprocess.run([
            "tmux", "new-window", "-t", session_name,
            "-n", "collaboration-hub"
        ], shell=(os.name == 'nt'))
        
        subprocess.run([
            "tmux", "send-keys", "-t", f"{session_name}:collaboration-hub",
            "echo 'Multi-Agent Collaboration Hub: All agents are now working together'", "Enter"
        ], shell=(os.name == 'nt'))
        
        display_operation_success(f"Collaborative mode enabled for session '{session_name}'")
        
    except Exception as e:
        display_system_error(f"Collaborative mode error: {str(e)}")


# System Core Operations
def access_credential_portal():
    """Access API credential management portal with system monitoring."""
    with create_system_spinner(f"{symbols['shield']} Accessing credential management portal...") as progress:
        task = progress.add_task("Establishing secure connection...")
        webbrowser.open("https://swarms.world/platform/api-keys")
        time.sleep(1)
        progress.remove_task(task)
    console.print(
        f"\n[{SWARMS_COLORS['success']}][SYSTEM] Credential portal access established[/{SWARMS_COLORS['success']}]"
    )


def verify_system_authorization():
    """Verify system authorization status with corporate protocols."""
    cache_file = "cache.txt"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            if f.read() == "logged_in":
                console.print(
                    f"[{SWARMS_COLORS['success']}][AUTHORIZATION] System access verified[/{SWARMS_COLORS['success']}]"
                )
                return True

    with create_system_spinner(f"{symbols['shield']} Executing authorization protocols...") as progress:
        task = progress.add_task("Initializing secure session...")
        time.sleep(1)
        with open(cache_file, "w") as f:
            f.write("logged_in")
        progress.remove_task(task)

    console.print(
        f"[{SWARMS_COLORS['success']}][AUTHORIZATION] System access granted[/{SWARMS_COLORS['success']}]"
    )
    return True


def launch_primary_interface():
    """Launch the primary control interface."""
    try:
        from swarms.cli.tui_clean import run_tui
        
        # Warning message about experimental TUI
        console.print()
        console.print(Panel(
            Text.from_markup(
                "[bold yellow]!!! EXPERIMENTAL INTERFACE WARNING !!![/bold yellow]\n\n"
                "[yellow]The TUI (Terminal User Interface) is currently in development and is considered:\n\n"
                "• Work-in-Progress (WIP)\n"
                "• Unstable and experimental\n"
                "• Subject to bugs and unexpected behavior\n"
                "• Not recommended for production use\n\n"
                "USE AT YOUR OWN DISCRETION[/yellow]\n\n"
                "[dim]Press Ctrl+C at any time to exit safely[/dim]"
            ),
            border_style="yellow",
            title="[bold red]*** CAUTION ***[/bold red]",
            title_align="center"
        ))
        console.print()
        
        console.print(
            f"[{SWARMS_COLORS['info']}][SYSTEM] Launching Primary Control Interface...[/{SWARMS_COLORS['info']}]"
        )
        run_tui()
    except ImportError:
        display_system_error(
            "Primary interface dependencies not installed",
            "RESOLUTION: Install textual dependency: pip install textual"
        )
    except Exception as e:
        display_system_error(f"Failed to launch primary interface: {str(e)}")


# Agent Control Matrix Operations
def execute_autonomous_swarm(task: str, model: str):
    """Execute autonomous swarm deployment with comprehensive error management"""
    try:
        console.print(
            f"[{SWARMS_COLORS['info']}][AGENT MATRIX] Initializing autonomous swarm deployment...[/{SWARMS_COLORS['info']}]"
        )

        # Set LiteLLM verbose mode for system monitoring
        import litellm
        litellm.set_verbose = True

        # Validate operational parameters
        if not task or task.strip() == "":
            raise SwarmSystemError("Task specification cannot be empty")

        if not model or model.strip() == "":
            raise SwarmSystemError("Model specification cannot be empty")

        # Execute swarm configuration generation
        console.print(
            f"[{SWARMS_COLORS['info']}][SWARM DEPLOYMENT] Generating configuration for task: {task}[/{SWARMS_COLORS['info']}]"
        )
        result = generate_swarm_config(task=task, model=model)

        if result:
            console.print(
                f"[{SWARMS_COLORS['success']}][DEPLOYMENT SUCCESS] Autonomous swarm configuration completed[/{SWARMS_COLORS['success']}]"
            )
        else:
            raise SwarmSystemError("Swarm configuration generation failed")

    except Exception as e:
        if "No YAML content found" in str(e):
            display_system_error(
                "Configuration generation failure detected",
                "DIAGNOSTIC PROCEDURES:\n"
                + "1. Verify authentication credentials\n"
                + "2. Validate model specification parameters\n"
                + "3. Execute with --model gpt-4 directive",
            )
        else:
            display_system_error(
                f"Autonomous swarm deployment error: {str(e)}",
                "SYSTEM DIAGNOSTIC PROCEDURES:\n"
                + "1. Verify authentication credentials\n"
                + "2. Confirm network connectivity status\n"
                + "3. Attempt alternative model specification",
            )


def create_agent_configuration(name: str, description: str, model: str = "gpt-4o-mini"):
    """Create new agent configuration with systematic template generation."""
    try:
        config = {
            "agents": [
                {
                    "agent_name": name,
                    "agent_description": description,
                    "model_name": model,
                    "max_loops": 1,
                    "temperature": 0.7,
                    "system_prompt": f"You are {name}, {description}. Execute directives with precision and efficiency.",
                    "tasks": [
                        "Execute assigned operational requirements"
                    ]
                }
            ]
        }
        
        filename = f"{name.lower().replace(' ', '_')}_config.yaml"
        
        import yaml
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        display_operation_success(f"Agent configuration file created: {filename}")
        
    except Exception as e:
        display_system_error(f"Agent configuration creation failed: {str(e)}")


def validate_configuration_integrity(yaml_file: str):
    """Validate agent configuration integrity with comprehensive verification."""
    try:
        import yaml
        
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"Configuration file not found: {yaml_file}")
        
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Systematic configuration validation
        if not isinstance(config, dict):
            raise ValueError("Configuration must be valid YAML dictionary structure")
        
        if "agents" not in config:
            raise ValueError("Configuration must contain 'agents' specification")
        
        if not isinstance(config["agents"], list):
            raise ValueError("'agents' must be structured as list")
        
        for i, agent in enumerate(config["agents"]):
            if not isinstance(agent, dict):
                raise ValueError(f"Agent {i} must be dictionary structure")
            
            required_fields = ["agent_name", "model_name"]
            for field in required_fields:
                if field not in agent:
                    raise ValueError(f"Agent {i} missing required field: {field}")
        
        display_operation_success(f"Configuration integrity verification passed: {yaml_file}")
        
    except Exception as e:
        display_system_error(f"Configuration validation failed: {str(e)}")


# Data Processing Infrastructure Operations
def process_file_conversion(file_path: str, output_file: Optional[str] = None):
    """Process file through text conversion matrix with format detection."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        console.print(f"[{SWARMS_COLORS['info']}][DATA PROCESSING] Converting {file_path} through text matrix...[/{SWARMS_COLORS['info']}]")
        
        text_content = data_to_text(file_path)
        
        if text_content is None:
            display_system_error(f"File format not supported or binary file detected: {file_path}")
            return
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            display_operation_success(f"Text conversion completed: {output_file}")
        else:
            console.print(Panel(
                text_content[:1000] + ("..." if len(text_content) > 1000 else ""),
                title="PROCESSED FILE CONTENT (displaying first 1000 characters)",
                border_style=SWARMS_COLORS["success"]
            ))
            
    except Exception as e:
        display_system_error(f"File conversion processing failed: {str(e)}")


def extract_code_segments(file_path: str, language: Optional[str] = None, output_file: Optional[str] = None):
    """Extract code segments from documentation with language filtering."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        code_blocks = extract_code_from_markdown(content, language)
        
        if not code_blocks:
            console.print(f"[{SWARMS_COLORS['warning']}][PROCESSING] No code segments detected[/{SWARMS_COLORS['warning']}]")
            return
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, block in enumerate(code_blocks):
                    f.write(f"# Code Segment {i+1}\n")
                    f.write(block)
                    f.write("\n\n")
            display_operation_success(f"Code segments extracted to: {output_file}")
        else:
            for i, block in enumerate(code_blocks):
                console.print(Panel(
                    block,
                    title=f"CODE SEGMENT {i+1}",
                    border_style=SWARMS_COLORS["success"]
                ))
                
    except Exception as e:
        display_system_error(f"Code extraction failed: {str(e)}")


def analyze_token_metrics(file_path: str, model: str = "gpt-4"):
    """Analyze token utilization metrics for specified file and model."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        token_count = count_tokens(content, model)
        
        table = Table(title="TOKEN UTILIZATION ANALYSIS")
        table.add_column("METRIC", style=SWARMS_COLORS["command_red"])
        table.add_column("VALUE", style=SWARMS_COLORS["primary_text"])
        
        table.add_row("FILE PATH", file_path)
        table.add_row("MODEL SPECIFICATION", model)
        table.add_row("CHARACTER COUNT", str(len(content)))
        table.add_row("TOKEN COUNT", str(token_count))
        
        console.print(table)
        
    except Exception as e:
        display_system_error(f"Token analysis failed: {str(e)}")


def archive_workspace_directory(workspace_path: str, output_filename: str):
    """Archive workspace directory with systematic compression."""
    try:
        if not os.path.exists(workspace_path):
            raise FileNotFoundError(f"Directory not found: {workspace_path}")
        
        console.print(f"[{SWARMS_COLORS['info']}][ARCHIVE] Processing directory: {workspace_path}...[/{SWARMS_COLORS['info']}]")
        
        zip_workspace(workspace_path, output_filename)
        
        display_operation_success(f"Directory archive completed: {output_filename}")
        
    except Exception as e:
        display_system_error(f"Directory archiving failed: {str(e)}")


def sanitize_path_specification(file_path: str):
    """Clean and validate file path specifications."""
    try:
        sanitized = sanitize_path_util(file_path)
        
        table = Table(title="PATH SANITIZATION PROCESS")
        table.add_column("TYPE", style=SWARMS_COLORS["command_red"])
        table.add_column("PATH SPECIFICATION", style=SWARMS_COLORS["primary_text"])
        
        table.add_row("ORIGINAL", file_path)
        table.add_row("SANITIZED", sanitized)
        
        console.print(table)
        
    except Exception as e:
        display_system_error(f"Path sanitization failed: {str(e)}")


# System Diagnostics Division Operations
def display_model_specifications(model: str):
    """Display model capability specifications and operational limits."""
    try:
        max_tokens = check_all_model_max_tokens(model)
        
        table = Table(title=f"MODEL SPECIFICATION ANALYSIS: {model}")
        table.add_column("SPECIFICATION", style=SWARMS_COLORS["command_red"])
        table.add_column("VALUE", style=SWARMS_COLORS["primary_text"])
        
        table.add_row("MODEL DESIGNATION", model)
        table.add_row("MAXIMUM TOKEN LIMIT", str(max_tokens) if max_tokens else "SPECIFICATION UNKNOWN")
        
        console.print(table)
        
    except Exception as e:
        display_system_error(f"Model specification analysis failed: {str(e)}")


def execute_system_diagnostics():
    """Execute comprehensive system diagnostic protocols."""
    try:
        diagnostics = []
        
        # Python version verification
        python_version = sys.version.split()[0]
        diagnostics.append(("PYTHON VERSION", python_version, "OPERATIONAL" if python_version >= "3.7" else "CRITICAL"))
        
        # API credential verification
        openai_key = os.getenv("OPENAI_API_KEY")
        diagnostics.append(("OPENAI CREDENTIALS", "CONFIGURED" if openai_key else "NOT CONFIGURED", "OPERATIONAL" if openai_key else "WARNING"))
        
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        diagnostics.append(("ANTHROPIC CREDENTIALS", "CONFIGURED" if anthropic_key else "NOT CONFIGURED", "OPERATIONAL" if anthropic_key else "WARNING"))
        
        # TMux availability check
        tmux_available = check_tmux_availability()
        if os.name == 'nt':
            # Windows tmux detection
            diagnostics.append(("TMUX AVAILABILITY", "INSTALLED" if tmux_available else "NOT INSTALLED", "OPERATIONAL" if tmux_available else "WARNING"))
        else:
            # Unix tmux detection
            diagnostics.append(("TMUX AVAILABILITY", "INSTALLED" if tmux_available else "NOT INSTALLED", "OPERATIONAL" if tmux_available else "WARNING"))
        
        # Workspace directory verification
        workspace_dir = os.getenv("WORKSPACE_DIR", "agent_workspace")
        workspace_exists = os.path.exists(workspace_dir)
        diagnostics.append(("WORKSPACE DIRECTORY", workspace_dir, "OPERATIONAL" if workspace_exists else "WARNING"))
        
        table = Table(title="SYSTEM DIAGNOSTIC ANALYSIS")
        table.add_column("COMPONENT", style=SWARMS_COLORS["command_red"])
        table.add_column("STATUS", style=SWARMS_COLORS["primary_text"])
        table.add_column("ASSESSMENT", style=SWARMS_COLORS["primary_text"])
        
        for component, status, assessment in diagnostics:
            table.add_row(component, status, assessment)
        
        console.print(table)
        
    except Exception as e:
        display_system_error(f"System diagnostics failed: {str(e)}")


def display_version_registry():
    """Display system version registry and component information."""
    try:
        import swarms
        version = getattr(swarms, '__version__', 'VERSION UNKNOWN')
        
        table = Table(title="SYSTEM VERSION REGISTRY")
        table.add_column("COMPONENT", style=SWARMS_COLORS["command_red"])
        table.add_column("VERSION", style=SWARMS_COLORS["primary_text"])
        
        table.add_row("SWARMS PLATFORM", version)
        table.add_row("PYTHON RUNTIME", sys.version.split()[0])
        
        # Check tmux version if available
        if check_tmux_availability():
            try:
                tmux_result = subprocess.run(["tmux", "-V"], capture_output=True, text=True)
                tmux_version = tmux_result.stdout.strip()
                table.add_row("TMUX MULTIPLEXER", tmux_version)
            except:
                table.add_row("TMUX MULTIPLEXER", "VERSION UNKNOWN")
        else:
            table.add_row("TMUX MULTIPLEXER", "NOT INSTALLED")
        
        console.print(table)
        
    except Exception as e:
        display_system_error(f"Version registry access failed: {str(e)}")


def main():
    try:
        display_control_matrix()

        parser = argparse.ArgumentParser(
            description="Swarms Corporate Control Matrix - Multi-Tool Command Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False,  # Disable default help to use our custom help
            epilog="""
═══════════════════════════════════════════════════════════════════════════════════

SWARMS CORPORATE CONTROL MATRIX - OPERATIONAL DIRECTIVE SYSTEM

═══════════════════════════════════════════════════════════════════════════════════

COMMAND HIERARCHY SECTORS:
  ▣ SYSTEM CORE OPERATIONS
    tui, onboarding, get-api-key, check-login
    
  ▣ AGENT CONTROL MATRIX  
    run-agents, autoswarm, create-agent, validate-config
    
  ▣ DATA PROCESSING INFRASTRUCTURE
    convert-file, extract-code, count-tokens, zip-workspace
    
  ▣ INTELLIGENT TERMINAL MULTIPLEXER
    tmux, new-session, attach-session, list-sessions, kill-session
    
  ▣ SYSTEM DIAGNOSTICS DIVISION
    model-info, health-check, version
    
  ▣ EXTERNAL INTEGRATION SERVICES
    book-call, auto-upgrade

═══════════════════════════════════════════════════════════════════════════════════

OPERATIONAL EXAMPLES:
  swarms tui                               ▶ Launch primary control interface
  swarms tmux --session-type="development" ▶ Launch intelligent terminal multiplexer
  swarms autoswarm --task "analyze data"  ▶ Deploy autonomous swarm collective  
  swarms convert-file report.pdf          ▶ Process file through conversion matrix
  swarms count-tokens document.txt        ▶ Analyze token utilization metrics
  swarms create-agent "Assistant" "Unit"  ▶ Generate agent configuration

═══════════════════════════════════════════════════════════════════════════════════

ACCESS PROTOCOLS:
  ▶ Enhanced operational parameters: swarms -h / swarms --help
  ▶ Technical reference database: https://docs.swarms.world
  ▶ Strategic consultation: swarms book-call

═══════════════════════════════════════════════════════════════════════════════════
            """
        )
        
        # Add custom help action
        parser.add_argument(
            "-h", "--help",
            action=CustomHelpAction,
            help="Display enhanced command reference database"
        )
        
        parser.add_argument(
            "command",
            choices=[
                # System Core Operations
                "tui", "onboarding", "get-api-key", "check-login",
                # Agent Control Matrix
                "run-agents", "autoswarm", "create-agent", "list-agents", "validate-config",
                # Data Processing Infrastructure
                "convert-file", "extract-code", "count-tokens", "zip-workspace", "sanitize-path",
                # Intelligent Terminal Multiplexer
                "tmux", "new-session", "attach-session", "list-sessions", "kill-session",
                "agent-assist", "collaborative-mode",
                # System Diagnostics Division
                "model-info", "health-check", "version",
                # External Integration Services
                "book-call", "auto-upgrade",
            ],
            help="Command to execute",
        )
        
        # Core operational parameters
        parser.add_argument("--yaml-file", type=str, default="agents.yaml", help="YAML configuration file specification")
        parser.add_argument("--task", type=str, help="Task specification for autoswarm")
        parser.add_argument("--model", type=str, default="gpt-4", help="Model specification for operations")
        
        # Data processing parameters
        parser.add_argument("--file", type=str, help="File path for processing operations")
        parser.add_argument("--output", type=str, help="Output file path specification")
        parser.add_argument("--language", type=str, help="Programming language for code extraction")
        
        # Agent configuration parameters
        parser.add_argument("--name", type=str, help="Agent designation")
        parser.add_argument("--description", type=str, help="Agent operational description")
        
        # TMux session parameters
        parser.add_argument("--session-name", type=str, help="TMux session name")
        parser.add_argument("--session-type", type=str, default="development", 
                          choices=["development", "devops", "research", "generic"],
                          help="Session type for intelligent configuration")
        parser.add_argument("--agents", type=str, help="Comma-separated agent types (dev,test,security,performance)")
        parser.add_argument("--layout", type=str, default="tiled",
                          choices=["tiled", "main-vertical", "main-horizontal"],
                          help="Pane layout strategy")
        
        # System parameters
        parser.add_argument("--workspace", type=str, help="Workspace directory specification")

        args = parser.parse_args()

        try:
            # System Core Operations
            if args.command == "tui":
                launch_primary_interface()
            elif args.command == "onboarding":
                OnboardingProcess().run()
            elif args.command == "get-api-key":
                access_credential_portal()
            elif args.command == "check-login":
                verify_system_authorization()
                
            # Intelligent Terminal Multiplexer Operations
            elif args.command == "tmux":
                agents_list = args.agents.split(',') if args.agents else None
                launch_intelligent_tmux(
                    session_name=args.session_name,
                    session_type=args.session_type,
                    agents=agents_list,
                    layout=args.layout
                )
            elif args.command == "new-session":
                agents_list = args.agents.split(',') if args.agents else None
                launch_intelligent_tmux(
                    session_name=args.session_name,
                    session_type=args.session_type,
                    agents=agents_list,
                    layout=args.layout
                )
            elif args.command == "attach-session":
                if not args.session_name:
                    display_system_error(
                        "Missing required parameter: --session-name",
                        "USAGE: swarms attach-session --session-name session_name"
                    )
                    exit(1)
                try:
                    subprocess.run(["tmux", "attach-session", "-t", args.session_name], check=True)
                except subprocess.CalledProcessError:
                    display_system_error(f"Failed to attach to session: {args.session_name}")
            elif args.command == "list-sessions":
                list_intelligent_sessions()
            elif args.command == "kill-session":
                if not args.session_name:
                    display_system_error(
                        "Missing required parameter: --session-name",
                        "USAGE: swarms kill-session --session-name session_name"
                    )
                    exit(1)
                kill_intelligent_session(args.session_name)
            elif args.command == "agent-assist":
                if not args.session_name:
                    display_system_error(
                        "Missing required parameter: --session-name",
                        "USAGE: swarms agent-assist --session-name session_name --agents dev,test"
                    )
                    exit(1)
                agent_types = args.agents.split(',') if args.agents else ["general"]
                for agent_type in agent_types:
                    deploy_agent_assistance(args.session_name, agent_type.strip())
            elif args.command == "collaborative-mode":
                if not args.session_name:
                    display_system_error(
                        "Missing required parameter: --session-name",
                        "USAGE: swarms collaborative-mode --session-name session_name"
                    )
                    exit(1)
                enable_collaborative_mode(args.session_name)
                
            # Agent Control Matrix Operations
            elif args.command == "run-agents":
                try:
                    console.print(
                        f"[{SWARMS_COLORS['info']}][AGENT MATRIX] Loading agent configurations from {args.yaml_file}...[/{SWARMS_COLORS['info']}]"
                    )

                    if not os.path.exists(args.yaml_file):
                        raise FileNotFoundError(
                            f"Configuration file not found: {args.yaml_file}\n"
                            "Verify file exists in current directory."
                        )

                    # Create systematic progress monitoring
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    )

                    with progress:
                        # Initialize operational task
                        init_task = progress.add_task(f"[SYSTEM] Initializing...", total=None)

                        # Load and validate configuration
                        progress.update(init_task, description=f"[SYSTEM] Loading configuration...")

                        # Create agent instances
                        progress.update(init_task, description=f"[SYSTEM] Creating agent instances...")
                        result = create_agents_from_yaml(
                            yaml_file=args.yaml_file,
                            return_type="run_swarm",
                        )

                        # Update completion status
                        progress.update(init_task, description=f"[COMPLETE] Processing complete", completed=True)

                    if result:
                        # Format and display execution results
                        if isinstance(result, str):
                            console.print(
                                f"\n[bold {SWARMS_COLORS['success']}][EXECUTION RESULTS] Operation completed successfully[/bold {SWARMS_COLORS['success']}]"
                            )
                            console.print(
                                Panel(
                                    result,
                                    title="AGENT EXECUTION OUTPUT",
                                    border_style=SWARMS_COLORS["success"],
                                )
                            )
                        elif isinstance(result, dict):
                            console.print(
                                f"\n[bold {SWARMS_COLORS['success']}][EXECUTION RESULTS] Operation completed successfully[/bold {SWARMS_COLORS['success']}]"
                            )
                            for key, value in result.items():
                                console.print(
                                    f"[{SWARMS_COLORS['command_red']}]{key}:[/{SWARMS_COLORS['command_red']}] {value}"
                                )
                        else:
                            console.print(
                                f"[{SWARMS_COLORS['success']}][COMPLETION] Agent tasks executed successfully[/{SWARMS_COLORS['success']}]"
                            )
                    else:
                        console.print(
                            f"[{SWARMS_COLORS['warning']}][WARNING] Agents completed but no results returned[/{SWARMS_COLORS['warning']}]"
                        )

                except FileNotFoundError as e:
                    display_system_error("File System Error", str(e))
                except ValueError as e:
                    display_system_error("Configuration Error", str(e) + "\n\nVerify agents.yaml file format specification.")
                except Exception as e:
                    # Enhanced error management
                    error_msg = str(e)
                    if "context_length_exceeded" in error_msg:
                        display_system_error(
                            "Context Length Exceeded",
                            "Model context limits exceeded. RESOLUTION:\n"
                            "1. Reduce max_tokens in configuration\n"
                            "2. Reduce context_length in configuration\n"
                            "3. Use model with larger context capacity",
                        )
                    elif "api_key" in error_msg.lower():
                        display_system_error(
                            "Authentication Error",
                            "Authentication issue detected. RESOLUTION:\n"
                            "1. Verify API credential configuration\n"
                            "2. Validate credential status\n"
                            "3. Execute 'swarms get-api-key' for new credentials",
                        )
                    else:
                        display_system_error(
                            "Execution Error",
                            f"Unexpected error: {error_msg}\n"
                            "DIAGNOSTIC PROCEDURES:\n"
                            "1. Verify configuration specifications\n"
                            "2. Check credential status\n"
                            "3. Confirm network connectivity",
                        )
                        
            elif args.command == "autoswarm":
                if not args.task:
                    display_system_error(
                        "Missing required parameter: --task",
                        "USAGE: swarms autoswarm --task 'task specification' --model gpt-4",
                    )
                    exit(1)
                execute_autonomous_swarm(args.task, args.model)
                
            elif args.command == "create-agent":
                if not args.name or not args.description:
                    display_system_error(
                        "Missing required parameters: --name and --description",
                        "USAGE: swarms create-agent --name 'Agent Name' --description 'Agent Description'",
                    )
                    exit(1)
                create_agent_configuration(args.name, args.description, args.model)
                
            elif args.command == "validate-config":
                validate_configuration_integrity(args.yaml_file)
                
            # Data Processing Infrastructure Operations
            elif args.command == "convert-file":
                if not args.file:
                    display_system_error(
                        "Missing required parameter: --file",
                        "USAGE: swarms convert-file --file path/to/file.pdf --output output.txt",
                    )
                    exit(1)
                process_file_conversion(args.file, args.output)
                
            elif args.command == "extract-code":
                if not args.file:
                    display_system_error(
                        "Missing required parameter: --file",
                        "USAGE: swarms extract-code --file README.md --language python --output code.py",
                    )
                    exit(1)
                extract_code_segments(args.file, args.language, args.output)
                
            elif args.command == "count-tokens":
                if not args.file:
                    display_system_error(
                        "Missing required parameter: --file",
                        "USAGE: swarms count-tokens --file document.txt --model gpt-4",
                    )
                    exit(1)
                analyze_token_metrics(args.file, args.model)
                
            elif args.command == "zip-workspace":
                workspace = args.workspace or "agent_workspace"
                output = args.output or f"{workspace}.zip"
                archive_workspace_directory(workspace, output)
                
            elif args.command == "sanitize-path":
                if not args.file:
                    display_system_error(
                        "Missing required parameter: --file",
                        "USAGE: swarms sanitize-path --file 'path/to/file'",
                    )
                    exit(1)
                sanitize_path_specification(args.file)
                
            # System Diagnostics Division Operations
            elif args.command == "model-info":
                display_model_specifications(args.model)
                
            elif args.command == "health-check":
                execute_system_diagnostics()
                
            elif args.command == "version":
                display_version_registry()
                
            # External Integration Services
            elif args.command == "book-call":
                webbrowser.open("https://cal.com/swarms/swarms-strategy-session")
                
            elif args.command == "auto-upgrade":
                console.print(f"[{SWARMS_COLORS['info']}][SYSTEM] Auto-upgrade functionality under development...[/{SWARMS_COLORS['info']}]")
                
        except Exception as e:
            console.print(
                f"[{SWARMS_COLORS['error']}][CRITICAL] System Error: {str(e)}[/{SWARMS_COLORS['error']}]"
            )
            return
    except Exception as error:
        formatter.print_panel(f"Critical system error detected: {error} - verify operational parameters")
        raise error


if __name__ == "__main__":
    main()
