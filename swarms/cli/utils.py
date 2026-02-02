"""
Utility functions for the Swarms CLI.

This module contains helper functions for display, validation, and formatting
used throughout the CLI application.
"""

import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from swarms.utils.formatter import formatter
from swarms.utils.workspace_utils import get_workspace_dir

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
    """
    Create a custom spinner with the given text.

    Args:
        text: The text to display in the spinner

    Returns:
        Progress: A Rich Progress instance configured as a spinner
    """
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


def check_workspace_dir() -> Tuple[bool, str, str]:
    """
    Check if WORKSPACE_DIR environment variable is set.

    Returns:
        Tuple containing (success, status_icon, message)
    """
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


def check_env_file() -> Tuple[bool, str, str]:
    """
    Check if .env file exists and has content.

    Returns:
        Tuple containing (success, status_icon, message)
    """
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


def check_swarms_version(verbose: bool = False) -> str:
    """
    Check if swarms is at the latest version using only the 'importlib.metadata' package.

    Args:
        verbose: Whether to show verbose output

    Returns:
        str: Current version string
    """
    try:
        import importlib.metadata

        current_version = importlib.metadata.version("swarms")
    except Exception as e:
        if verbose:
            console.print(
                f"[dim]Error getting current version: {str(e)}[/dim]"
            )
        return "Unknown"

    if verbose:
        console.print(
            f"[dim]Detected swarms version: {current_version}[/dim]"
        )

    return current_version


def check_python_version() -> Tuple[bool, str, str]:
    """
    Check Python version compatibility.

    Returns:
        Tuple containing (success, status_icon, message)
    """
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


def check_api_keys() -> Tuple[bool, str, str]:
    """
    Check if at least one common API key is set in the environment variables.

    Returns:
        Tuple containing (success, status_icon, message)
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


def check_dependencies() -> Tuple[bool, str, str]:
    """
    Check if key dependencies are available.

    Returns:
        Tuple containing (success, status_icon, message)
    """
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
    """
    Run comprehensive setup check with beautiful formatting.

    Args:
        verbose: Whether to show verbose output
    """
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
        (
            "Swarms Version",
            (True, "✓", check_swarms_version(verbose)),
        ),
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
    """
    Create a beautifully formatted table of commands.

    Returns:
        Table: Rich table with available commands
    """
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
        (
            "chat",
            "Start an interactive chat agent with optimized defaults",
        ),
        ("upgrade", "Update Swarms to the latest version"),
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
    """
    Create a comprehensive table of all available commands with detailed information.

    Returns:
        Table: Rich table with detailed command information
    """
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
            "desc": "Create and run a custom agent (task is optional, interactive mode on by default)",
            "usage": "swarms agent --name 'Agent' [--task 'Analyze data']",
            "args": "--name, --task (optional), --interactive (default: True)",
        },
        {
            "cmd": "chat",
            "category": "Chat",
            "desc": "Start an interactive chat agent with optimized defaults for conversation (max_loops='auto')",
            "usage": "swarms chat [--task 'Hello'] [--name 'Chat Agent'] [--system-prompt 'You are helpful']",
            "args": "--task (optional), --name, --description, --system-prompt",
        },
        {
            "cmd": "upgrade",
            "category": "Maintenance",
            "desc": "Update Swarms to latest version",
            "usage": "swarms upgrade",
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
            "desc": "Create and run a custom agent with specified parameters (task is optional, interactive mode on by default)",
            "command": "swarms agent --name 'Agent' [--task 'Task'] --system-prompt 'Prompt'",
            "params": "--name, --task (optional), --system-prompt, --model-name, --temperature, --max-loops, --interactive (default: True), --verbose",
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
            "command": "swarms upgrade",
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


def create_commands_parameters_table() -> Table:
    """
    Create a comprehensive table showing all commands with their required and optional parameters.

    Returns:
        Table: Rich table with commands and their parameters
    """
    table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["secondary"],
        title="📋 Commands & Parameters Reference",
        title_style=f"bold {COLORS['primary']}",
        padding=(0, 1),
        show_lines=True,
        expand=True,
    )

    table.add_column(
        "Command",
        style=f"bold {COLORS['accent']}",
        min_width=15,
        no_wrap=False,
    )
    table.add_column(
        "Required Parameters",
        style="bold red",
        no_wrap=False,
    )
    table.add_column(
        "Optional Parameters",
        style="dim yellow",
        no_wrap=False,
    )
    table.add_column(
        "Description",
        style="white",
        no_wrap=False,
    )

    commands_params = [
        {
            "cmd": "onboarding",
            "required": "None",
            "optional": "--verbose",
            "desc": "Run environment setup check",
        },
        {
            "cmd": "help",
            "required": "None",
            "optional": "None",
            "desc": "Display comprehensive help message",
        },
        {
            "cmd": "get-api-key",
            "required": "None",
            "optional": "None",
            "desc": "Open browser to retrieve API keys",
        },
        {
            "cmd": "check-login",
            "required": "None",
            "optional": "None",
            "desc": "Verify authentication status",
        },
        {
            "cmd": "run-agents",
            "required": "None",
            "optional": "--yaml-file (default: agents.yaml)",
            "desc": "Execute agents from YAML config",
        },
        {
            "cmd": "load-markdown",
            "required": "--markdown-path",
            "optional": "--concurrent (default: True)",
            "desc": "Load agents from markdown files",
        },
        {
            "cmd": "agent",
            "required": "--name, --description, --system-prompt (or --marketplace-prompt-id)",
            "optional": "--task, --model-name, --temperature, --max-loops, --interactive, --verbose, --streaming-on, --context-length, --retry-attempts, --return-step-meta, --dashboard, --autosave, --saved-state-path, --user-name, --mcp-url, --marketplace-prompt-id, --auto-generate-prompt, --dynamic-temperature-enabled, --dynamic-context-window, --output-type",
            "desc": "Create and run a custom agent",
        },
        {
            "cmd": "chat",
            "required": "None",
            "optional": "--task, --name, --description, --system-prompt",
            "desc": "Start an interactive chat agent with optimized defaults (max_loops='auto')",
        },
        {
            "cmd": "autoswarm",
            "required": "--task, --model",
            "optional": "None",
            "desc": "Generate autonomous swarm config",
        },
        {
            "cmd": "setup-check",
            "required": "None",
            "optional": "--verbose",
            "desc": "Run environment setup checks",
        },
        {
            "cmd": "llm-council",
            "required": "--task",
            "optional": "--verbose",
            "desc": "Run LLM Council with multiple agents",
        },
        {
            "cmd": "heavy-swarm",
            "required": "--task",
            "optional": "--loops-per-agent (default: 1), --question-agent-model-name (default: gpt-4o-mini), --worker-model-name (default: gpt-4o-mini), --random-loops-per-agent, --verbose",
            "desc": "Run HeavySwarm with specialized agents",
        },
        {
            "cmd": "features",
            "required": "None",
            "optional": "None",
            "desc": "Display all available features",
        },
        {
            "cmd": "upgrade",
            "required": "None",
            "optional": "None",
            "desc": "Update Swarms to latest version",
        },
    ]

    for cmd_info in commands_params:
        table.add_row(
            cmd_info["cmd"],
            cmd_info["required"],
            cmd_info["optional"],
            cmd_info["desc"],
        )

    return table


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
        "• [yellow]swarms chat[/yellow] or [yellow]swarms chat --task 'Hello'[/yellow] - Start interactive chat agent\n"
        "• [yellow]swarms agent --name 'MyAgent' [--task 'Hello World'][/yellow] - Create agent (task optional, interactive by default)\n"
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

    # Show commands and parameters table
    console.print(create_commands_parameters_table())
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


def show_error(message: str, help_text: Optional[str] = None):
    """
    Display error message in a formatted panel.

    Args:
        message: The error message to display
        help_text: Optional help text to display below the error
    """
    error_panel = Panel(
        f"[bold red]{message}[/bold red]",
        title="Error",
        border_style="red",
    )
    console.print(error_panel)

    if help_text:
        console.print(f"\n[yellow]ℹ️ {help_text}[/yellow]")


def execute_with_spinner(action: callable, text: str) -> None:
    """
    Execute an action with a spinner animation.

    Args:
        action: The callable to execute
        text: The text to display in the spinner
    """
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
    """
    Verify login status with enhanced visual feedback.

    Returns:
        bool: True if login is successful
    """
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
