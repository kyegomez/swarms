import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from swarms.agents.auto_generate_swarm_config import (
    generate_swarm_config,
)
from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)

from swarms.structs.agent import Agent
from swarms.structs.agent_loader import AgentLoader
from swarms.structs.llm_council import LLMCouncil
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.utils.formatter import formatter
from swarms.utils.workspace_utils import get_workspace_dir

load_dotenv()

# Initialize console with custom styling
console = Console()


class SwarmCLIError(Exception):
    """Custom exception for Swarm CLI errors"""

    pass


# Color scheme
COLORS = {
    "primary": "red",
    "secondary": "#FF6B6B",
    "accent": "#4A90E2",
    "success": "#2ECC71",
    "warning": "#F1C40F",
    "error": "#E74C3C",
    "text": "#FFFFFF",
}

ASCII_ART = r"""
  █████████  █████   ███   █████   █████████   ███████████   ██████   ██████  █████████ 
 ███░░░░░███░░███   ░███  ░░███   ███░░░░░███ ░░███░░░░░███ ░░██████ ██████  ███░░░░░███
░███    ░░░  ░███   ░███   ░███  ░███    ░███  ░███    ░███  ░███░█████░███ ░███    ░░░ 
░░█████████  ░███   ░███   ░███  ░███████████  ░██████████   ░███░░███ ░███ ░░█████████ 
 ░░░░░░░░███ ░░███  █████  ███   ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░░░░░░░░███
 ███    ░███  ░░░█████░█████░    ░███    ░███  ░███    ░███  ░███      ░███  ███    ░███
░░█████████     ░░███ ░░███      █████   █████ █████   █████ █████     █████░░█████████ 
 ░░░░░░░░░       ░░░   ░░░      ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░  ░░░░░░░░░                                 
"""


def create_spinner(text: str) -> Progress:
    """Create a custom spinner with the given text."""
    return Progress(
        SpinnerColumn(style=COLORS["primary"]),
        TextColumn("[{task.description}]", style=COLORS["text"]),
        console=console,
    )


def show_ascii_art():
    """Display the ASCII art with a glowing effect."""
    panel = Panel(
        Text(ASCII_ART, style=f"bold {COLORS['primary']}"),
        border_style=COLORS["secondary"],
        title="[bold]Swarms CLI[/bold]",
    )

    console.print(panel)

    formatter.print_panel(
        "Access the full Swarms CLI documentation and API guide at https://docs.swarms.world/en/latest/swarms/cli/cli_reference/. For help with a specific command, use swarms <command> --help to unlock the full power of Swarms CLI.",
        title="Documentation and Assistance",
        style="red",
    )


def check_workspace_dir() -> tuple[bool, str, str]:
    """Check if WORKSPACE_DIR environment variable is set."""
    try:
        workspace_dir = get_workspace_dir()
    except ValueError:
        workspace_dir = None
    if workspace_dir:
        path = Path(workspace_dir)
        if path.exists():
            return (
                True,
                "✓",
                f"WORKSPACE_DIR is set to: {workspace_dir}",
            )
        else:
            return (
                False,
                "⚠",
                f"WORKSPACE_DIR is set but path doesn't exist: {workspace_dir}",
            )
    else:
        return (
            False,
            "✗",
            "WORKSPACE_DIR environment variable is not set",
        )


def check_env_file() -> tuple[bool, str, str]:
    """Check if .env file exists and has content."""
    env_path = Path(".env")
    if env_path.exists():
        try:
            content = env_path.read_text().strip()
            if content:
                # Count API keys
                api_keys = [
                    line
                    for line in content.split("\n")
                    if "API_KEY" in line and not line.startswith("#")
                ]
                return (
                    True,
                    "✓",
                    f".env file exists with {len(api_keys)} API key(s)",
                )
            else:
                return False, "⚠", ".env file exists but is empty"
        except Exception as e:
            return (
                False,
                "✗",
                f".env file exists but cannot be read: {str(e)}",
            )
    else:
        return False, "✗", ".env file not found in current directory"


def check_swarms_version(
    verbose: bool = False,
) -> tuple[bool, str, str, str]:
    """Check if swarms is at the latest version."""
    try:
        # Get current version using multiple methods
        current_version = "Unknown"

        if verbose:
            console.print(
                "[dim]🔍 Attempting to detect Swarms version...[/dim]"
            )

        # Method 1: Try importlib.metadata (Python 3.8+)
        try:
            import importlib.metadata

            current_version = importlib.metadata.version("swarms")
            if verbose:
                console.print(
                    f"[dim]  ✓ Method 1 (importlib.metadata): {current_version}[/dim]"
                )
        except ImportError:
            if verbose:
                console.print(
                    "[dim]  ✗ Method 1 (importlib.metadata): Not available[/dim]"
                )
            pass

        # Method 2: Try pkg_resources (older method)
        if current_version == "Unknown":
            try:
                import pkg_resources

                current_version = pkg_resources.get_distribution(
                    "swarms"
                ).version
                if verbose:
                    console.print(
                        f"[dim]  ✓ Method 2 (pkg_resources): {current_version}[/dim]"
                    )
            except ImportError:
                if verbose:
                    console.print(
                        "[dim]  ✗ Method 2 (pkg_resources): Not available[/dim]"
                    )
                pass

        # Method 3: Try direct attribute access
        if current_version == "Unknown":
            try:
                import swarms

                current_version = getattr(
                    swarms, "__version__", "Unknown"
                )
                if verbose:
                    console.print(
                        f"[dim]  ✓ Method 3 (direct attribute): {current_version}[/dim]"
                    )
            except ImportError:
                if verbose:
                    console.print(
                        "[dim]  ✗ Method 3 (direct attribute): Import failed[/dim]"
                    )
                pass

        # Method 4: Try to get from pyproject.toml or setup.py
        if current_version == "Unknown":
            try:
                import subprocess

                result = subprocess.run(
                    ["pip", "show", "swarms"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.startswith("Version:"):
                            current_version = line.split(":", 1)[
                                1
                            ].strip()
                            if verbose:
                                console.print(
                                    f"[dim]  ✓ Method 4 (pip show): {current_version}[/dim]"
                                )
                            break
            except Exception:
                if verbose:
                    console.print(
                        "[dim]  ✗ Method 4 (pip show): Failed[/dim]"
                    )
                pass

        # Method 5: Try to read from __init__.py file
        if current_version == "Unknown":
            try:
                import swarms

                swarms_path = swarms.__file__
                if swarms_path:
                    init_file = os.path.join(
                        os.path.dirname(swarms_path), "__init__.py"
                    )
                    if os.path.exists(init_file):
                        with open(init_file, "r") as f:
                            content = f.read()
                            # Look for version patterns like __version__ = "8.1.1"
                            import re

                            version_match = re.search(
                                r'__version__\s*=\s*["\']([^"\']+)["\']',
                                content,
                            )
                            if version_match:
                                current_version = version_match.group(
                                    1
                                )
                                if verbose:
                                    console.print(
                                        f"[dim]  ✓ Method 5 (__init__.py): {current_version}[/dim]"
                                    )
            except Exception:
                if verbose:
                    console.print(
                        "[dim]  ✗ Method 5 (__init__.py): Failed[/dim]"
                    )
                pass

        # Method 6: Try to read from pyproject.toml
        if current_version == "Unknown":
            try:
                import swarms

                swarms_path = swarms.__file__
                if swarms_path:
                    # Go up to find pyproject.toml
                    current_dir = os.path.dirname(swarms_path)
                    for _ in range(5):  # Go up max 5 levels
                        pyproject_path = os.path.join(
                            current_dir, "pyproject.toml"
                        )
                        if os.path.exists(pyproject_path):
                            with open(pyproject_path, "r") as f:
                                content = f.read()
                                # Look for version in pyproject.toml
                                import re

                                version_match = re.search(
                                    r'version\s*=\s*["\']([^"\']+)["\']',
                                    content,
                                )
                                if version_match:
                                    current_version = (
                                        version_match.group(1)
                                    )
                                    break
                        current_dir = os.path.dirname(current_dir)
                        if current_dir == os.path.dirname(
                            current_dir
                        ):  # Reached root
                            break
            except Exception:
                pass

        if verbose:
            console.print(
                f"[dim]🎯 Final detected version: {current_version}[/dim]\n"
            )

        # Try to get latest version from PyPI
        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    "https://pypi.org/pypi/swarms/json"
                )
                if response.status_code == 200:
                    latest_version = response.json()["info"][
                        "version"
                    ]
                    is_latest = current_version == latest_version
                    if is_latest:
                        return (
                            True,
                            "✓",
                            f"Current version: {current_version}",
                            latest_version,
                        )
                    else:
                        return (
                            False,
                            "⚠",
                            f"Current version: {current_version}",
                            latest_version,
                        )
                else:
                    return (
                        True,
                        "✓",
                        f"Current version: {current_version}",
                        "Unknown (PyPI unreachable)",
                    )
        except ImportError:
            return (
                True,
                "✓",
                f"Current version: {current_version}",
                "Unknown (httpx not available)",
            )
        except Exception:
            return (
                True,
                "✓",
                f"Current version: {current_version}",
                "Unknown (PyPI check failed)",
            )

        # If we still don't have a version, try one more method
        if current_version == "Unknown":
            try:
                # Try to get from environment variable (sometimes set during build)
                current_version = os.getenv(
                    "SWARMS_VERSION", "Unknown"
                )
                if verbose and current_version != "Unknown":
                    console.print(
                        f"[dim]  ✓ Method 7 (env var): {current_version}[/dim]"
                    )
            except Exception:
                if verbose:
                    console.print(
                        "[dim]  ✗ Method 7 (env var): Failed[/dim]"
                    )
                pass

    except Exception as e:
        return (
            False,
            "✗",
            f"Error checking version: {str(e)}",
            "Unknown",
        )


def check_python_version() -> tuple[bool, str, str]:
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return (
            True,
            "✓",
            f"Python {version.major}.{version.minor}.{version.micro}",
        )
    else:
        return (
            False,
            "✗",
            f"Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)",
        )


def check_api_keys() -> tuple[bool, str, str]:
    """
    Check if at least one common API key is set in the environment variables.

    Returns:
        tuple: (True, "✓", message) if at least one API key is set,
               (False, "✗", message) otherwise.
    """

    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
    }

    # At least one key must be present and non-empty
    if any(value for value in api_keys.values()):
        present_keys = [
            key for key, value in api_keys.items() if value
        ]
        return (
            True,
            "✓",
            f"At least one API key found: {', '.join(present_keys)}",
        )
    else:
        return (
            False,
            "✗",
            "No API keys found in environment variables",
        )


def check_dependencies() -> tuple[bool, str, str]:
    """Check if key dependencies are available."""
    required_deps = ["torch", "transformers", "litellm", "rich"]
    missing_deps = []

    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    if not missing_deps:
        return True, "✓", "All required dependencies available"
    else:
        return (
            False,
            "⚠",
            f"Missing dependencies: {', '.join(missing_deps)}",
        )


def run_setup_check(verbose: bool = False):
    """Run comprehensive setup check with beautiful formatting."""
    console.print(
        "\n[bold blue]🔍 Running Swarms Environment Setup Check[/bold blue]\n"
    )

    if verbose:
        console.print(
            "[dim]Debug mode enabled - showing detailed version detection steps[/dim]\n"
        )

    # Create results table
    table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["secondary"],
        title="Environment Check Results",
        padding=(0, 2),
    )

    table.add_column("Status", style="bold", width=8)
    table.add_column("Check", style="bold white", width=25)
    table.add_column("Details", style="dim white")

    # Run all checks
    checks = [
        ("Python Version", check_python_version()),
        ("Swarms Version", check_swarms_version(verbose)),
        ("API Keys", check_api_keys()),
        ("Dependencies", check_dependencies()),
        ("Environment File", check_env_file()),
        ("Workspace Directory", check_workspace_dir()),
    ]

    all_passed = True

    for check_name, (passed, status_icon, details, *extra) in checks:
        if not passed:
            all_passed = False

        # Color code the status
        if passed:
            status_style = f"bold {COLORS['success']}"
        elif status_icon == "⚠":
            status_style = f"bold {COLORS['warning']}"
        else:
            status_style = f"bold {COLORS['error']}"

        table.add_row(
            f"[{status_style}]{status_icon}[/{status_style}]",
            check_name,
            details,
        )

    console.print(table)

    # Show summary
    if all_passed:
        summary_panel = Panel(
            Align.center(
                "[bold green]🎉 All checks passed! Your environment is ready for Swarms.[/bold green]"
            ),
            border_style=COLORS["success"],
            title="[bold]Setup Check Complete[/bold]",
            padding=(1, 2),
        )
    else:
        summary_panel = Panel(
            Align.center(
                "[bold yellow]⚠️ Some checks failed. Please review the issues above.[/bold yellow]"
            ),
            border_style=COLORS["warning"],
            title="[bold]Setup Check Complete[/bold]",
            padding=(1, 2),
        )

    console.print(summary_panel)

    # Show recommendations
    if not all_passed:
        console.print("\n[bold blue]💡 Recommendations:[/bold blue]")

        recommendations = []

        # Check specific failures and provide recommendations
        for check_name, (
            passed,
            status_icon,
            details,
            *extra,
        ) in checks:
            if not passed:
                if "WORKSPACE_DIR" in check_name:
                    recommendations.append(
                        "Set WORKSPACE_DIR environment variable: export WORKSPACE_DIR=/path/to/your/workspace"
                    )
                elif "API Keys" in check_name:
                    recommendations.append(
                        "Set API keys: export OPENAI_API_KEY='your-key-here'"
                    )
                elif "Environment File" in check_name:
                    recommendations.append(
                        "Create .env file with your API keys"
                    )
                elif (
                    "Swarms Version" in check_name and len(extra) > 0
                ):
                    latest_version = extra[0]
                    if latest_version != "Unknown":
                        recommendations.append(
                            f"Update Swarms: pip install --upgrade swarms (latest: {latest_version})"
                        )
                elif "Dependencies" in check_name:
                    recommendations.append(
                        "Install missing dependencies: pip install -r requirements.txt"
                    )

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                console.print(f"  {i}. [yellow]{rec}[/yellow]")

        console.print(
            "\n[dim]Run 'swarms setup-check' again after making changes to verify.[/dim]"
        )

    return all_passed


def create_command_table() -> Table:
    """Create a beautifully formatted table of commands."""
    table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["secondary"],
        title="Available Commands",
        padding=(0, 2),
    )

    table.add_column("Command", style="bold white")
    table.add_column("Description", style="dim white")

    commands = [
        (
            "onboarding",
            "Run environment setup check (same as setup-check)",
        ),
        ("help", "Display this help message"),
        ("get-api-key", "Retrieve your API key from the platform"),
        ("check-login", "Verify login status and initialize cache"),
        ("run-agents", "Execute agents from your YAML configuration"),
        (
            "load-markdown",
            "Load agents from markdown files with YAML frontmatter",
        ),
        (
            "agent",
            "Create and run a custom agent with specified parameters",
        ),
        ("auto-upgrade", "Update Swarms to the latest version"),
        ("book-call", "Schedule a strategy session with our team"),
        ("autoswarm", "Generate and execute an autonomous swarm"),
        (
            "setup-check",
            "Run a comprehensive environment setup check",
        ),
        (
            "llm-council",
            "Run the LLM Council with multiple agents collaborating on a task",
        ),
        (
            "heavy-swarm",
            "Run HeavySwarm with specialized agents for complex task analysis",
        ),
        (
            "features",
            "Display all available features and actions in a comprehensive table",
        ),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    return table


def create_detailed_command_table() -> Table:
    """Create a comprehensive table of all available commands with detailed information."""
    table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["secondary"],
        title="🚀Swarms CLI - Complete Command Reference",
        title_style=f"bold {COLORS['primary']}",
        padding=(0, 1),
        show_lines=True,
        expand=True,
    )

    # Add columns with consistent widths and better styles
    table.add_column(
        "Command",
        style=f"bold {COLORS['accent']}",
        width=16,
        no_wrap=True,
    )
    table.add_column(
        "Category", style="bold cyan", width=12, justify="center"
    )
    table.add_column(
        "Description", style="white", width=45, no_wrap=False
    )
    table.add_column(
        "Usage Example", style="dim yellow", width=50, no_wrap=False
    )
    table.add_column(
        "Key Args", style="dim magenta", width=20, no_wrap=False
    )

    commands = [
        {
            "cmd": "onboarding",
            "category": "Setup",
            "desc": "Run environment setup check (same as setup-check)",
            "usage": "swarms onboarding [--verbose]",
            "args": "--verbose",
        },
        {
            "cmd": "help",
            "category": "Info",
            "desc": "Display this comprehensive help message",
            "usage": "swarms help",
            "args": "None",
        },
        {
            "cmd": "get-api-key",
            "category": "Setup",
            "desc": "Open browser to retrieve API keys from platform",
            "usage": "swarms get-api-key",
            "args": "None",
        },
        {
            "cmd": "check-login",
            "category": "Auth",
            "desc": "Verify authentication status and cache",
            "usage": "swarms check-login",
            "args": "None",
        },
        {
            "cmd": "run-agents",
            "category": "Execution",
            "desc": "Execute agents from YAML configuration",
            "usage": "swarms run-agents --yaml-file agents.yaml",
            "args": "--yaml-file",
        },
        {
            "cmd": "load-markdown",
            "category": "Loading",
            "desc": "Load agents from markdown files",
            "usage": "swarms load-markdown --markdown-path ./agents/",
            "args": "--markdown-path",
        },
        {
            "cmd": "agent",
            "category": "Creation",
            "desc": "Create and run a custom agent",
            "usage": "swarms agent --name 'Agent' --task 'Analyze data'",
            "args": "--name, --task",
        },
        {
            "cmd": "auto-upgrade",
            "category": "Maintenance",
            "desc": "Update Swarms to latest version",
            "usage": "swarms auto-upgrade",
            "args": "None",
        },
        {
            "cmd": "book-call",
            "category": "Support",
            "desc": "Schedule a strategy session",
            "usage": "swarms book-call",
            "args": "None",
        },
        {
            "cmd": "autoswarm",
            "category": "AI Gen",
            "desc": "Generate autonomous swarm config",
            "usage": "swarms autoswarm --task 'analyze data' --model gpt-4",
            "args": "--task, --model",
        },
        {
            "cmd": "setup-check",
            "category": "Diagnostics",
            "desc": "Run environment setup checks",
            "usage": "swarms setup-check [--verbose]",
            "args": "--verbose",
        },
        {
            "cmd": "llm-council",
            "category": "Collaboration",
            "desc": "Run LLM Council with multiple agents",
            "usage": "swarms llm-council --task 'Your question here' [--verbose]",
            "args": "--task, --verbose",
        },
        {
            "cmd": "heavy-swarm",
            "category": "Execution",
            "desc": "Run HeavySwarm with specialized agents",
            "usage": "swarms heavy-swarm --task 'Your task here' [--loops-per-agent 1] [--question-agent-model-name gpt-4o-mini] [--worker-model-name gpt-4o-mini] [--random-loops-per-agent] [--verbose]",
            "args": "--task, --loops-per-agent, --question-agent-model-name, --worker-model-name, --random-loops-per-agent, --verbose",
        },
        {
            "cmd": "features",
            "category": "Info",
            "desc": "Display all available features and actions",
            "usage": "swarms features",
            "args": "None",
        },
    ]

    for cmd_info in commands:
        table.add_row(
            cmd_info["cmd"],
            cmd_info["category"],
            cmd_info["desc"],
            cmd_info["usage"],
            cmd_info["args"],
        )

    return table


def show_features():
    """
    Display all available CLI features and actions in a comprehensive table.
    """
    console.print(
        "\n[bold]🚀 Swarms CLI - All Available Features[/bold]\n",
        style=COLORS["primary"],
    )

    # Create main features table
    features_table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["secondary"],
        title="✨ Complete Feature Reference",
        title_style=f"bold {COLORS['primary']}",
        padding=(0, 1),
        show_lines=True,
        expand=True,
    )

    # Add columns
    features_table.add_column(
        "Feature",
        style=f"bold {COLORS['accent']}",
        width=20,
        no_wrap=True,
    )
    features_table.add_column(
        "Category",
        style="bold cyan",
        width=15,
        justify="center",
    )
    features_table.add_column(
        "Description",
        style="white",
        width=50,
        no_wrap=False,
    )
    features_table.add_column(
        "Command",
        style="dim yellow",
        width=35,
        no_wrap=False,
    )
    features_table.add_column(
        "Key Parameters",
        style="dim magenta",
        width=30,
        no_wrap=False,
    )

    # Define all features
    features = [
        {
            "feature": "Environment Setup",
            "category": "Setup",
            "desc": "Check and verify your Swarms environment configuration",
            "command": "swarms setup-check [--verbose]",
            "params": "--verbose",
        },
        {
            "feature": "Onboarding",
            "category": "Setup",
            "desc": "Run environment setup check (alias for setup-check)",
            "command": "swarms onboarding [--verbose]",
            "params": "--verbose",
        },
        {
            "feature": "API Key Management",
            "category": "Setup",
            "desc": "Retrieve API keys from the Swarms platform",
            "command": "swarms get-api-key",
            "params": "None",
        },
        {
            "feature": "Authentication",
            "category": "Auth",
            "desc": "Verify login status and initialize authentication cache",
            "command": "swarms check-login",
            "params": "None",
        },
        {
            "feature": "YAML Agent Execution",
            "category": "Execution",
            "desc": "Execute agents from YAML configuration files",
            "command": "swarms run-agents --yaml-file agents.yaml",
            "params": "--yaml-file",
        },
        {
            "feature": "Markdown Agent Loading",
            "category": "Loading",
            "desc": "Load agents from markdown files with YAML frontmatter",
            "command": "swarms load-markdown --markdown-path ./agents/",
            "params": "--markdown-path, --concurrent",
        },
        {
            "feature": "Custom Agent Creation",
            "category": "Creation",
            "desc": "Create and run a custom agent with specified parameters",
            "command": "swarms agent --name 'Agent' --task 'Task' --system-prompt 'Prompt'",
            "params": "--name, --task, --system-prompt, --model-name, --temperature, --max-loops, --verbose",
        },
        {
            "feature": "Auto Swarm Generation",
            "category": "AI Generation",
            "desc": "Automatically generate and execute an autonomous swarm configuration",
            "command": "swarms autoswarm --task 'analyze data' --model gpt-4",
            "params": "--task, --model",
        },
        {
            "feature": "LLM Council",
            "category": "Collaboration",
            "desc": "Run LLM Council with multiple agents collaborating and evaluating responses",
            "command": "swarms llm-council --task 'Your question' [--verbose]",
            "params": "--task, --verbose",
        },
        {
            "feature": "HeavySwarm",
            "category": "Execution",
            "desc": "Run HeavySwarm with specialized agents for complex task analysis",
            "command": "swarms heavy-swarm --task 'Your task' [options]",
            "params": "--task, --loops-per-agent, --question-agent-model-name, --worker-model-name, --random-loops-per-agent, --verbose",
        },
        {
            "feature": "Package Upgrade",
            "category": "Maintenance",
            "desc": "Update Swarms to the latest version",
            "command": "swarms auto-upgrade",
            "params": "None",
        },
        {
            "feature": "Support Booking",
            "category": "Support",
            "desc": "Schedule a strategy session with the Swarms team",
            "command": "swarms book-call",
            "params": "None",
        },
        {
            "feature": "Help Documentation",
            "category": "Info",
            "desc": "Display comprehensive help message with all commands",
            "command": "swarms help",
            "params": "None",
        },
        {
            "feature": "Features List",
            "category": "Info",
            "desc": "Display all available features and actions in a table",
            "command": "swarms features",
            "params": "None",
        },
    ]

    # Add rows to table
    for feat in features:
        features_table.add_row(
            feat["feature"],
            feat["category"],
            feat["desc"],
            feat["command"],
            feat["params"],
        )

    console.print(features_table)

    # Add category summary
    console.print("\n[bold cyan]📊 Feature Categories:[/bold cyan]\n")

    category_table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["secondary"],
        padding=(0, 2),
    )

    category_table.add_column("Category", style="bold cyan", width=20)
    category_table.add_column(
        "Count", style="bold white", justify="center", width=10
    )
    category_table.add_column("Features", style="dim white", width=60)

    # Count features by category
    categories = {}
    for feat in features:
        cat = feat["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(feat["feature"])

    for category, feature_list in sorted(categories.items()):
        category_table.add_row(
            category,
            str(len(feature_list)),
            ", ".join(feature_list),
        )

    console.print(category_table)

    # Add usage tips
    tips_panel = Panel(
        "[bold cyan]💡 Quick Tips:[/bold cyan]\n"
        "• Use [yellow]swarms features[/yellow] to see this table anytime\n"
        "• Use [yellow]swarms help[/yellow] for detailed command documentation\n"
        "• Use [yellow]swarms setup-check --verbose[/yellow] for detailed diagnostics\n"
        "• Most commands support [yellow]--verbose[/yellow] for detailed output\n"
        "• Use [yellow]swarms <command> --help[/yellow] for command-specific help",
        title="📚 Usage Tips",
        border_style=COLORS["success"],
        padding=(1, 2),
    )
    console.print(tips_panel)

    console.print(
        "\n[dim]For more information, visit: https://docs.swarms.world[/dim]"
    )


def show_help():
    """Display a beautifully formatted help message with comprehensive command reference."""
    console.print(
        "\n[bold]Swarms CLI - Command Reference[/bold]\n",
        style=COLORS["primary"],
    )

    # Add a quick usage panel with consistent sizing
    usage_panel = Panel(
        "[bold cyan]Quick Start Commands:[/bold cyan]\n"
        "• [yellow]swarms onboarding[/yellow] - Environment setup check\n"
        "• [yellow]swarms setup-check[/yellow] - Check your environment\n"
        "• [yellow]swarms agent --name 'MyAgent' --task 'Hello World'[/yellow] - Create agent\n"
        "• [yellow]swarms autoswarm --task 'analyze data' --model gpt-4[/yellow] - Auto-generate swarm\n"
        "• [yellow]swarms llm-council --task 'Your question'[/yellow] - Run LLM Council\n"
        "• [yellow]swarms heavy-swarm --task 'Your task'[/yellow] - Run HeavySwarm\n"
        "• [yellow]swarms features[/yellow] - View all available features",
        title="⚡ Quick Usage Guide",
        border_style=COLORS["secondary"],
        padding=(1, 2),
        expand=False,
        width=140,
    )
    console.print(usage_panel)
    console.print("\n")

    console.print(create_detailed_command_table())

    # Add additional help panels with consistent sizing
    docs_panel = Panel(
        "📚 [bold]Documentation:[/bold] https://docs.swarms.world\n"
        "🐛 [bold]Support:[/bold] https://github.com/kyegomez/swarms/issues\n"
        "💬 [bold]Community:[/bold] https://discord.gg/EamjgSaEQf",
        title="🔗 Useful Links",
        border_style=COLORS["success"],
        padding=(1, 2),
        expand=False,
        width=140,
    )
    console.print(docs_panel)

    console.print(
        "\n[dim]💡 Tip: Use [bold]swarms setup-check --verbose[/bold] for detailed environment diagnostics[/dim]"
    )


def show_error(message: str, help_text: str = None):
    """Display error message in a formatted panel"""
    error_panel = Panel(
        f"[bold red]{message}[/bold red]",
        title="Error",
        border_style="red",
    )
    console.print(error_panel)

    if help_text:
        console.print(f"\n[yellow]ℹ️ {help_text}[/yellow]")


def execute_with_spinner(action: callable, text: str) -> None:
    """Execute an action with a spinner animation."""
    with create_spinner(text) as progress:
        task = progress.add_task(text, total=None)
        result = action()
        progress.remove_task(task)
    return result


def get_api_key():
    """Retrieve API key with visual feedback."""
    with create_spinner("Opening API key portal...") as progress:
        task = progress.add_task("Opening browser...")
        webbrowser.open("https://swarms.world/platform/api-keys")
        time.sleep(1)
        progress.remove_task(task)
    console.print(
        f"\n[{COLORS['success']}]✓ API key page opened in your browser[/{COLORS['success']}]"
    )


def check_login():
    """Verify login status with enhanced visual feedback."""
    cache_file = "cache.txt"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            if f.read() == "logged_in":
                console.print(
                    f"[{COLORS['success']}]✓ Authentication verified[/{COLORS['success']}]"
                )
                return True

    with create_spinner("Authenticating...") as progress:
        task = progress.add_task("Initializing session...")
        time.sleep(1)
        with open(cache_file, "w") as f:
            f.write("logged_in")
        progress.remove_task(task)

    console.print(
        f"[{COLORS['success']}]✓ Login successful![/{COLORS['success']}]"
    )
    return True


def run_autoswarm(task: str, model: str):
    """Run autoswarm with enhanced error handling"""
    try:
        console.print(
            "[yellow]Initializing autoswarm configuration...[/yellow]"
        )

        # Validate inputs
        if not task or task.strip() == "":
            raise SwarmCLIError("Task cannot be empty")

        if not model or model.strip() == "":
            raise SwarmCLIError("Model name cannot be empty")

        # Attempt to generate swarm configuration
        console.print(
            f"[yellow]Generating swarm for task: {task}[/yellow]"
        )
        result = generate_swarm_config(task=task, model=model)

        if result:
            console.print(
                "[green]✓ Swarm configuration generated successfully![/green]"
            )
        else:
            raise SwarmCLIError(
                "Failed to generate swarm configuration"
            )

    except Exception as e:
        if "No YAML content found" in str(e):
            show_error(
                "Failed to generate YAML configuration",
                "This might be due to an API key issue or invalid model configuration.\n"
                + "1. Check if your OpenAI API key is set correctly\n"
                + "2. Verify the model name is valid\n"
                + "3. Try running with --model gpt-4",
            )
        else:
            show_error(
                f"Error during autoswarm execution: {str(e)}",
                "For debugging, try:\n"
                + "1. Check your API keys are set correctly\n"
                + "2. Verify your network connection\n"
                + "3. Try a different model",
            )


def load_markdown_agents(
    file_path: str, concurrent: bool = True, **kwargs
):
    """Load agents from markdown files with enhanced visual feedback."""
    try:
        console.print(
            f"[yellow]Loading agents from markdown: {file_path}[/yellow]"
        )

        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )

        with progress:
            # Add initial task
            init_task = progress.add_task(
                "Initializing markdown loader...", total=None
            )

            # Initialize agent loader
            progress.update(
                init_task,
                description="Initializing agent loader...",
            )
            loader = AgentLoader()

            # Load agents
            progress.update(
                init_task,
                description="Loading agents from markdown...",
            )

            if os.path.isdir(file_path):
                agents = loader.load_multiple_agents(
                    file_path, concurrent=concurrent, **kwargs
                )
            else:
                agents = [
                    loader.load_single_agent(file_path, **kwargs)
                ]

            # Update progress on completion
            progress.update(
                init_task,
                description="Processing complete!",
                completed=True,
            )

        # Display results
        if agents:
            console.print(
                f"\n[bold green]✓ Successfully loaded {len(agents)} agents![/bold green]"
            )

            # Create a table to display loaded agents
            agent_table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                border_style=COLORS["secondary"],
                title="Loaded Agents",
                padding=(0, 2),
            )

            agent_table.add_column("Name", style="bold white")
            agent_table.add_column("Model", style="dim white")
            agent_table.add_column("Description", style="dim white")

            for agent in agents:
                name = getattr(agent, "agent_name", "Unknown")
                model = getattr(agent, "model_name", "Unknown")
                description = getattr(
                    agent, "agent_description", "No description"
                )

                agent_table.add_row(name, model, description)

            console.print(agent_table)

            return agents
        else:
            console.print(
                "[yellow]⚠ No agents were loaded from the markdown files.[/yellow]"
            )
            return []

    except FileNotFoundError:
        show_error(
            "File Error",
            f"Markdown file/directory not found: {file_path}\n"
            "Please make sure the path exists and you're in the correct directory.",
        )
        return []
    except ValueError as e:
        show_error(
            "Configuration Error",
            f"Error parsing markdown: {str(e)}\n\n"
            "Please check that your markdown files use the correct YAML frontmatter format:\n"
            "---\n"
            "name: Agent Name\n"
            "description: Agent Description\n"
            "model_name: gpt-4\n"
            "temperature: 0.1\n"
            "---\n"
            "System prompt content here...",
        )
        return []
    except Exception as e:
        show_error(
            "Execution Error",
            f"An unexpected error occurred: {str(e)}\n"
            "1. Check your markdown file format\n"
            "2. Verify your API keys are set\n"
            "3. Check network connectivity",
        )
        return []


def run_heavy_swarm(
    task: str,
    loops_per_agent: int = 1,
    question_agent_model_name: str = "gpt-4o-mini",
    worker_model_name: str = "gpt-4o-mini",
    random_loops_per_agent: bool = False,
    verbose: bool = False,
):
    """
    Run the HeavySwarm with a given task.

    Args:
        task: The task/query for the HeavySwarm to process
        loops_per_agent: Number of execution loops each agent should perform
        question_agent_model_name: Model name for question generation
        worker_model_name: Model name for specialized worker agents
        random_loops_per_agent: Enable random number of loops per agent (1-10 range)
        verbose: Whether to show verbose output
    """
    try:
        console.print(
            "[yellow]🚀 Initializing HeavySwarm...[/yellow]"
        )

        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )

        with progress:
            # Add initial task
            init_task = progress.add_task(
                "Initializing swarm...", total=None
            )

            # Create HeavySwarm
            progress.update(
                init_task,
                description="Creating HeavySwarm with specialized agents...",
            )
            swarm = HeavySwarm(
                loops_per_agent=loops_per_agent,
                question_agent_model_name=question_agent_model_name,
                worker_model_name=worker_model_name,
                random_loops_per_agent=random_loops_per_agent,
                verbose=verbose,
            )

            # Update progress
            progress.update(
                init_task,
                description="Swarm initialized! Processing task...",
            )

            # Run the swarm
            result = swarm.run(task=task)

            # Update progress on completion
            progress.update(
                init_task,
                description="Task completed!",
                completed=True,
            )

        # Display results
        if result:
            console.print(
                "\n[bold green]✓ HeavySwarm completed successfully![/bold green]"
            )

            # Display result in a panel
            result_panel = Panel(
                str(result),
                title="HeavySwarm Final Response",
                border_style="green",
                padding=(1, 2),
            )
            console.print(result_panel)

            return result
        else:
            console.print(
                "[yellow]⚠ HeavySwarm completed but returned no results.[/yellow]"
            )
            return None

    except Exception as e:
        show_error(
            "HeavySwarm Error",
            f"Failed to run HeavySwarm: {str(e)}\n\n"
            "Please check:\n"
            "1. Your API keys are set correctly\n"
            "2. You have network connectivity\n"
            "3. The task is properly formatted",
        )
        return None


def run_llm_council(task: str, verbose: bool = True):
    """
    Run the LLM Council with a given task.

    Args:
        task: The task/query for the LLM Council to process
        verbose: Whether to show verbose output
    """
    try:
        console.print(
            "[yellow]🏛️  Initializing LLM Council...[/yellow]"
        )

        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )

        with progress:
            # Add initial task
            init_task = progress.add_task(
                "Initializing council...", total=None
            )

            # Create LLM Council
            progress.update(
                init_task,
                description="Creating LLM Council with default members...",
            )
            council = LLMCouncil(verbose=verbose)

            # Update progress
            progress.update(
                init_task,
                description="Council initialized! Processing task...",
            )

            # Run the council
            result = council.run(query=task)

            # Update progress on completion
            progress.update(
                init_task,
                description="Task completed!",
                completed=True,
            )

        # Display results
        if result:
            console.print(
                "\n[bold green]✓ LLM Council completed successfully![/bold green]"
            )

            # Display result in a panel
            result_panel = Panel(
                str(result),
                title="LLM Council Final Response",
                border_style="green",
                padding=(1, 2),
            )
            console.print(result_panel)

            return result
        else:
            console.print(
                "[yellow]⚠ LLM Council completed but returned no results.[/yellow]"
            )
            return None

    except Exception as e:
        show_error(
            "LLM Council Error",
            f"Failed to run LLM Council: {str(e)}\n\n"
            "Please check:\n"
            "1. Your API keys are set correctly\n"
            "2. You have network connectivity\n"
            "3. The task is properly formatted",
        )
        return None


def create_swarm_agent(
    name: str,
    description: str,
    system_prompt: str,
    model_name: str,
    task: str,
    **kwargs,
):
    """Create and run a custom agent with the specified parameters."""
    try:
        console.print(
            f"[yellow]Creating custom agent: {name}[/yellow]"
        )

        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )

        with progress:
            # Add initial task
            init_task = progress.add_task(
                "Initializing agent...", total=None
            )

            # Create agent
            progress.update(
                init_task,
                description="Creating agent with specified parameters...",
            )

            # Build agent configuration
            agent_config = {
                "agent_name": name,
                "agent_description": description,
                "system_prompt": system_prompt,
                "model_name": model_name,
                **kwargs,
            }

            # Remove None values to use defaults
            agent_config = {
                k: v for k, v in agent_config.items() if v is not None
            }

            agent = Agent(**agent_config)

            # Update progress
            progress.update(
                init_task,
                description="Agent created successfully! Running task...",
            )

            # Run the agent with the specified task
            progress.update(
                init_task,
                description=f"Executing task: {task[:50]}...",
            )

            result = agent.run(task)

            # Update progress on completion
            progress.update(
                init_task,
                description="Task completed!",
                completed=True,
            )

        # Display results
        if result:
            console.print(
                f"\n[bold green]✓ Agent '{name}' completed the task successfully![/bold green]"
            )

            # Display agent info
            agent_info = Panel(
                f"[bold]Agent Name:[/bold] {name}\n"
                f"[bold]Model:[/bold] {model_name}\n"
                f"[bold]Task:[/bold] {task}\n"
                f"[bold]Result:[/bold]\n{result}",
                title="Agent Execution Results",
                border_style="green",
                padding=(1, 2),
            )
            console.print(agent_info)

            return result
        else:
            console.print(
                f"[yellow]⚠ Agent '{name}' completed but returned no results.[/yellow]"
            )
            return None

    except Exception as e:
        show_error(
            "Agent Creation Error",
            f"Failed to create or run agent: {str(e)}\n\n"
            "Please check:\n"
            "1. Your API keys are set correctly\n"
            "2. The model name is valid\n"
            "3. All required parameters are provided\n"
            "4. Your system prompt is properly formatted",
        )
        return None


def main():
    try:

        show_ascii_art()

        parser = argparse.ArgumentParser(
            description="Swarms Cloud CLI"
        )
        parser.add_argument(
            "command",
            choices=[
                "onboarding",
                "help",
                "get-api-key",
                "check-login",
                "run-agents",
                "load-markdown",
                "agent",
                "auto-upgrade",
                "book-call",
                "autoswarm",
                "setup-check",
                "llm-council",
                "heavy-swarm",
                "features",
            ],
            help="Command to execute",
        )
        parser.add_argument(
            "--yaml-file",
            type=str,
            default="agents.yaml",
            help="YAML configuration file path",
        )
        parser.add_argument(
            "--markdown-path",
            type=str,
            help="Path to markdown file or directory containing markdown files",
        )
        parser.add_argument(
            "--concurrent",
            action="store_true",
            default=True,
            help="Enable concurrent processing for multiple markdown files (default: True)",
        )
        # Swarm agent specific arguments
        parser.add_argument(
            "--name",
            type=str,
            help="Name of the custom agent",
        )
        parser.add_argument(
            "--description",
            type=str,
            help="Description of the custom agent",
        )
        parser.add_argument(
            "--system-prompt",
            type=str,
            help="System prompt for the custom agent",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default="gpt-4",
            help="Model name for the custom agent (default: gpt-4)",
        )
        parser.add_argument(
            "--task",
            type=str,
            help="Task for the custom agent to execute",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            help="Temperature setting for the agent (0.0-2.0)",
        )
        parser.add_argument(
            "--max-loops",
            type=int,
            help="Maximum number of loops for the agent",
        )
        parser.add_argument(
            "--auto-generate-prompt",
            action="store_true",
            help="Enable auto-generation of prompts",
        )
        parser.add_argument(
            "--dynamic-temperature-enabled",
            action="store_true",
            help="Enable dynamic temperature adjustment",
        )
        parser.add_argument(
            "--dynamic-context-window",
            action="store_true",
            help="Enable dynamic context window",
        )
        parser.add_argument(
            "--output-type",
            type=str,
            help="Output type for the agent (e.g., 'str', 'json')",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose mode for the agent",
        )
        parser.add_argument(
            "--streaming-on",
            action="store_true",
            help="Enable streaming mode for the agent",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            help="Context length for the agent",
        )
        parser.add_argument(
            "--retry-attempts",
            type=int,
            help="Number of retry attempts for the agent",
        )
        parser.add_argument(
            "--return-step-meta",
            action="store_true",
            help="Return step metadata from the agent",
        )
        parser.add_argument(
            "--dashboard",
            action="store_true",
            help="Enable dashboard for the agent",
        )
        parser.add_argument(
            "--autosave",
            action="store_true",
            help="Enable autosave for the agent",
        )
        parser.add_argument(
            "--saved-state-path",
            type=str,
            help="Path for saving agent state",
        )
        parser.add_argument(
            "--user-name",
            type=str,
            help="Username for the agent",
        )
        parser.add_argument(
            "--mcp-url",
            type=str,
            help="MCP URL for the agent",
        )
        parser.add_argument(
            "--marketplace-prompt-id",
            type=str,
            help="Fetch system prompt from Swarms marketplace using this prompt ID",
        )
        # HeavySwarm specific arguments
        parser.add_argument(
            "--loops-per-agent",
            type=int,
            default=1,
            help="Number of execution loops each agent should perform (default: 1)",
        )
        parser.add_argument(
            "--question-agent-model-name",
            type=str,
            default="gpt-4o-mini",
            help="Model name for question generation agent (default: gpt-4o-mini)",
        )
        parser.add_argument(
            "--worker-model-name",
            type=str,
            default="gpt-4o-mini",
            help="Model name for specialized worker agents (default: gpt-4o-mini)",
        )
        parser.add_argument(
            "--random-loops-per-agent",
            action="store_true",
            help="Enable random number of loops per agent (1-10 range)",
        )

        args = parser.parse_args()

        try:
            if args.command == "onboarding":
                # For compatibility, redirect onboarding to setup-check
                console.print(
                    "[yellow]Note: 'swarms onboarding' now runs the same checks as 'swarms setup-check'[/yellow]"
                )
                run_setup_check(verbose=args.verbose)
            elif args.command == "help":
                show_help()
            elif args.command == "features":
                show_features()
            elif args.command == "get-api-key":
                get_api_key()
            elif args.command == "check-login":
                check_login()
            elif args.command == "run-agents":
                try:
                    console.print(
                        f"[yellow]Loading agents from {args.yaml_file}...[/yellow]"
                    )

                    if not os.path.exists(args.yaml_file):
                        raise FileNotFoundError(
                            f"YAML file not found: {args.yaml_file}\n"
                            "Please make sure the file exists and you're in the correct directory."
                        )

                    # Create progress display
                    progress = Progress(
                        SpinnerColumn(),
                        TextColumn(
                            "[progress.description]{task.description}"
                        ),
                        console=console,
                    )

                    with progress:
                        # Add initial task
                        init_task = progress.add_task(
                            "Initializing...", total=None
                        )

                        # Load and validate YAML
                        progress.update(
                            init_task,
                            description="Loading YAML configuration...",
                        )

                        # Create agents
                        progress.update(
                            init_task,
                            description="Creating agents...",
                        )
                        result = create_agents_from_yaml(
                            yaml_file=args.yaml_file,
                            return_type="run_swarm",
                        )

                        # Update progress on completion
                        progress.update(
                            init_task,
                            description="Processing complete!",
                            completed=True,
                        )

                    if result:
                        # Format and display the results
                        if isinstance(result, str):
                            console.print(
                                "\n[bold green]Results:[/bold green]"
                            )
                            console.print(
                                Panel(
                                    result,
                                    title="Agent Output",
                                    border_style="green",
                                )
                            )
                        elif isinstance(result, dict):
                            console.print(
                                "\n[bold green]Results:[/bold green]"
                            )
                            for key, value in result.items():
                                console.print(
                                    f"[cyan]{key}:[/cyan] {value}"
                                )
                        else:
                            console.print(
                                "[green]✓ Agents completed their tasks successfully![/green]"
                            )
                    else:
                        console.print(
                            "[yellow]⚠ Agents completed but returned no results.[/yellow]"
                        )

                except FileNotFoundError as e:
                    show_error("File Error", str(e))
                except ValueError as e:
                    show_error(
                        "Configuration Error",
                        str(e)
                        + "\n\nPlease check your agents.yaml file format.",
                    )
                except Exception as e:
                    # Enhanced error handling
                    error_msg = str(e)
                    if "context_length_exceeded" in error_msg:
                        show_error(
                            "Context Length Error",
                            "The model's context length was exceeded. Try:\n"
                            "1. Reducing max_tokens in your YAML config\n"
                            "2. Reducing context_length in your YAML config\n"
                            "3. Using a model with larger context window",
                        )
                    elif "api_key" in error_msg.lower():
                        show_error(
                            "API Key Error",
                            "There seems to be an issue with the API key. Please:\n"
                            "1. Check if your API key is set correctly\n"
                            "2. Verify the API key is valid\n"
                            "3. Run 'swarms get-api-key' to get a new key",
                        )
                    else:
                        show_error(
                            "Execution Error",
                            f"An unexpected error occurred: {error_msg}\n"
                            "1. Check your YAML configuration\n"
                            "2. Verify your API keys are set\n"
                            "3. Check network connectivity",
                        )
            elif args.command == "load-markdown":
                if not args.markdown_path:
                    show_error(
                        "Missing required argument: --markdown-path",
                        "Example usage: python cli.py load-markdown --markdown-path ./agents/",
                    )
                    exit(1)

                # Load agents from markdown
                agents = load_markdown_agents(
                    args.markdown_path, concurrent=args.concurrent
                )

                if agents:
                    console.print(
                        f"\n[bold green]Ready to use {len(agents)} agents![/bold green]\n"
                        "[dim]You can now use these agents in your code or run them interactively.[/dim]"
                    )
            elif args.command == "agent":
                # Validate required arguments
                # system_prompt not required if marketplace_prompt_id provided
                required_args = ["name", "description", "task"]
                if not getattr(args, "marketplace_prompt_id", None):
                    required_args.append("system_prompt")

                missing_args = [
                    arg
                    for arg in required_args
                    if not getattr(args, arg)
                ]

                if missing_args:
                    show_error(
                        "Missing required arguments",
                        f"Required arguments: {', '.join(missing_args)}\n\n"
                        "Example usage:\n"
                        "python cli.py agent \\\n"
                        "  --name 'Trading Agent' \\\n"
                        "  --description 'Advanced trading agent' \\\n"
                        "  --system-prompt 'You are an expert trader...' \\\n"
                        "  --task 'Analyze market trends' \\\n"
                        "  --model-name 'gpt-4' \\\n"
                        "  --temperature 0.1",
                    )
                    exit(1)

                # Build kwargs for additional parameters
                additional_params = {}
                param_mapping = {
                    "temperature": "temperature",
                    "max_loops": "max_loops",
                    "auto_generate_prompt": "auto_generate_prompt",
                    "dynamic_temperature_enabled": "dynamic_temperature_enabled",
                    "dynamic_context_window": "dynamic_context_window",
                    "output_type": "output_type",
                    "verbose": "verbose",
                    "streaming_on": "streaming_on",
                    "context_length": "context_length",
                    "retry_attempts": "retry_attempts",
                    "return_step_meta": "return_step_meta",
                    "dashboard": "dashboard",
                    "autosave": "autosave",
                    "saved_state_path": "saved_state_path",
                    "user_name": "user_name",
                    "mcp_url": "mcp_url",
                    "marketplace_prompt_id": "marketplace_prompt_id",
                }

                for cli_arg, agent_param in param_mapping.items():
                    value = getattr(args, cli_arg)
                    if value is not None:
                        additional_params[agent_param] = value

                # Create and run the custom agent
                result = create_swarm_agent(
                    name=args.name,
                    description=args.description,
                    system_prompt=args.system_prompt,
                    model_name=args.model_name,
                    task=args.task,
                    **additional_params,
                )

                if result:
                    console.print(
                        f"\n[bold green]Agent '{args.name}' executed successfully![/bold green]"
                    )
            elif args.command == "book-call":
                webbrowser.open(
                    "https://cal.com/swarms/swarms-strategy-session"
                )
            elif args.command == "autoswarm":
                if not args.task:
                    show_error(
                        "Missing required argument: --task",
                        "Example usage: python cli.py autoswarm --task 'analyze this data' --model gpt-4",
                    )
                    exit(1)
                run_autoswarm(args.task, args.model)
            elif args.command == "setup-check":
                run_setup_check(verbose=args.verbose)
            elif args.command == "llm-council":
                if not args.task:
                    show_error(
                        "Missing required argument: --task",
                        "Example usage: swarms llm-council --task 'What is the best approach to solve this problem?'",
                    )
                    exit(1)
                run_llm_council(task=args.task, verbose=args.verbose)
            elif args.command == "heavy-swarm":
                if not args.task:
                    show_error(
                        "Missing required argument: --task",
                        "Example usage: swarms heavy-swarm --task 'Analyze the current market trends'",
                    )
                    exit(1)
                run_heavy_swarm(
                    task=args.task,
                    loops_per_agent=args.loops_per_agent,
                    question_agent_model_name=args.question_agent_model_name,
                    worker_model_name=args.worker_model_name,
                    random_loops_per_agent=args.random_loops_per_agent,
                    verbose=args.verbose,
                )
        except Exception as e:
            console.print(
                f"[{COLORS['error']}]Error: {str(e)}[/{COLORS['error']}]"
            )
            return
    except Exception as error:
        formatter.print_panel(
            f"Error detected: {error} check your args"
        )
        raise error


if __name__ == "__main__":
    main()
