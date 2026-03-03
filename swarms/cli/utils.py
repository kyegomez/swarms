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

import random

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from swarms.utils.workspace_utils import get_workspace_dir

# Initialize console with custom styling
console = Console()


class SwarmCLIError(Exception):
    """Custom exception for Swarm CLI errors"""

    pass


# Color scheme
COLORS = {
    "primary": "red",
    "secondary": "red",
    "accent": "white",
    "success": "white",
    "warning": "white",
    "error": "red",
    "text": "#FFFFFF",
}


def _detect_active_provider() -> str:
    """Return a label for the active AI provider(s), like Claude Code's model line."""
    detected = []
    if os.getenv("OPENAI_API_KEY"):
        detected.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        detected.append("Anthropic")
    if os.getenv("GROQ_API_KEY"):
        detected.append("Groq")
    if os.getenv("GOOGLE_API_KEY"):
        detected.append("Google")
    if os.getenv("COHERE_API_KEY"):
        detected.append("Cohere")
    if os.getenv("MISTRAL_API_KEY"):
        detected.append("Mistral")
    if os.getenv("TOGETHER_API_KEY"):
        detected.append("Together AI")

    if not detected:
        return (
            "No API key found — run [bold]swarms setup-check[/bold]"
        )
    primary = detected[0]
    extras = len(detected) - 1
    suffix = f" +{extras} more" if extras else ""
    return f"{primary}{suffix} · Multi-Agent Framework"


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
    """Display a compact Claude Code-style CLI header."""
    version = check_swarms_version()
    cwd = str(Path.cwd()).replace(str(Path.home()), "~")
    provider = _detect_active_provider()

    # ── Pre-header startup tip ───────────────────────────────────────────────
    startup_tips = [
        # Commands
        "Start chatting instantly with [bold]swarms chat[/bold]",
        "Verify your setup anytime with [bold]swarms setup-check[/bold]",
        "See every command with [bold]swarms --help[/bold]",
        "Auto-build a swarm with [bold]swarms autoswarm --task '...'[/bold]",
        "Deep multi-agent analysis with [bold]swarms heavy-swarm --task '...'[/bold]",
        "Run a multi-model debate with [bold]swarms llm-council --task '...'[/bold]",
        "Load agents from YAML with [bold]swarms run-agents --yaml-file agents.yaml[/bold]",
        "Load agents from markdown files with [bold]swarms load-markdown --markdown-path ./agents/[/bold]",
        "Run a one-shot agent task with [bold]swarms agent --name '...' --task '...'[/bold]",
        "Upgrade to the latest version with [bold]swarms upgrade[/bold]",
        # Agent tips
        "Pass [bold]--max-loops auto[/bold] to let an agent decide when it's done",
        "Use [bold]--system-prompt[/bold] to give your agent a custom persona or role",
        "Use [bold]--model-name[/bold] to switch models, e.g. [bold]gpt-4o[/bold], [bold]claude-3-5-sonnet[/bold]",
        "Use [bold]--temperature 0.1[/bold] for more deterministic, factual agent responses",
        "Use [bold]--temperature 0.9[/bold] for more creative, varied agent responses",
        "Use [bold]--verbose[/bold] to see every step an agent takes in real time",
        "Use [bold]--streaming-on[/bold] to stream agent output token by token",
        "Use [bold]--context-length[/bold] to control how much history an agent retains",
        "Save and resume agent state with [bold]--autosave --saved-state-path ./state.json[/bold]",
        "Fetch a pre-built system prompt with [bold]--marketplace-prompt-id[/bold]",
        # Swarm tips
        "HeavySwarm spawns specialist sub-agents — great for research or code review",
        "LLM Council runs the same task across multiple models and aggregates answers",
        "AutoSwarm auto-generates the right swarm topology for your task",
        "Combine [bold]--loops-per-agent[/bold] with [bold]--random-loops-per-agent[/bold] for non-deterministic swarms",
        "Use [bold]--worker-model-name[/bold] to choose which model powers HeavySwarm workers",
        # General
        "Store your API keys in a [bold].env[/bold] file — swarms loads it automatically",
        "Set [bold]WORKSPACE_DIR[/bold] to control where agents read and write files",
        "Run [bold]swarms setup-check --verbose[/bold] to diagnose environment issues",
        "Star the repo and contribute at [bold]https://github.com/kyegomez/swarms[/bold]",
        "Join the community at [bold]https://discord.gg/EamjgSaEQf[/bold]",
        "Full docs at [bold]https://docs.swarms.world[/bold]",
    ]
    # ── Pixel-art alien icon (👾) ─────────────────────────────────────────────
    icon = Text()
    icon.append("▄     ▄\n", style="bold red")
    icon.append("▀█████▀\n", style="bold red")
    icon.append("█▀███▀█\n", style="bold red")
    icon.append("███████\n", style="bold red")
    icon.append("▀█   █▀", style="bold red")

    # ── Info block ────────────────────────────────────────────────────────────
    info = Text()
    info.append("Swarms", style="bold white")
    info.append(f"  v{version}\n", style="dim white")
    info.append(provider + "\n", style="dim white")
    info.append(cwd + "\n", style="dim white")
    info.append(
        "https://github.com/kyegomez/swarms", style="dim white"
    )

    header = Table.grid(padding=(0, 1))
    header.add_column(width=9, vertical="top")
    header.add_column(vertical="top")
    header.add_row(icon, info)

    # ── Rotating command tip ──────────────────────────────────────────────────
    tips = [
        "[bold white]swarms chat[/bold white] — interactive autonomous agent",
        "[bold white]swarms agent --name '...' --task '...'[/bold white] — one-shot agent",
        "[bold white]swarms autoswarm --task '...'[/bold white] — auto-generate a swarm",
        "[bold white]swarms heavy-swarm --task '...'[/bold white] — deep multi-agent analysis",
        "[bold white]swarms llm-council --task '...'[/bold white] — multi-model debate",
        "[bold white]swarms load-markdown --markdown-path ./agents/[/bold white] — load agents",
        "[bold white]swarms run-agents --yaml-file agents.yaml[/bold white] — run from YAML",
        "[bold white]swarms upgrade[/bold white] — update to the latest version",
        "[bold white]swarms setup-check --verbose[/bold white] — diagnose your environment",
        "[bold white]--max-loops auto[/bold white] — let an agent decide when it's done",
        "[bold white]--verbose[/bold white] — see every step an agent takes in real time",
        "[bold white]--streaming-on[/bold white] — stream agent output token by token",
        "[bold white]--model-name gpt-4o[/bold white] — switch models on any agent",
        "[bold white]--temperature 0.1[/bold white] — more deterministic responses",
        "[bold white]--autosave --saved-state-path ./state.json[/bold white] — save agent state",
        "Store API keys in [bold white].env[/bold white] — swarms loads it automatically",
        "Set [bold white]WORKSPACE_DIR[/bold white] to control agent file access",
    ]

    tip_line = Text.from_markup(
        f"[bold red] ⚡[/bold red]  [dim white]{random.choice(tips)}[/dim white]"
    )

    startup_tip = Text.from_markup(
        f"[bold red] Tip:[/bold red]  [white]{random.choice(startup_tips)}[/white]"
    )

    # ── Panel ─────────────────────────────────────────────────────────────────
    panel_content = Group(
        header,
        Text(""),
        Rule(style="dim red"),
        tip_line,
    )

    console.print(
        Panel(
            panel_content,
            border_style="red",
            title="[bold red] 👾 Swarms [/bold red]",
            title_align="left",
            subtitle="[dim white] swarms --help [/dim white]",
            subtitle_align="right",
            padding=(0, 2),
        )
    )
    console.print(startup_tip)
    console.print()


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
        "\n[bold red]🔍 Running Swarms Environment Setup Check[/bold red]\n"
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
                "[bold white]🎉 All checks passed! Your environment is ready for Swarms.[/bold white]"
            ),
            border_style=COLORS["success"],
            title="[bold]Setup Check Complete[/bold]",
            padding=(1, 2),
        )
    else:
        summary_panel = Panel(
            Align.center(
                "[bold white]⚠️ Some checks failed. Please review the issues above.[/bold white]"
            ),
            border_style=COLORS["warning"],
            title="[bold]Setup Check Complete[/bold]",
            padding=(1, 2),
        )

    console.print(summary_panel)

    # Show recommendations
    if not all_passed:
        console.print("\n[bold red]💡 Recommendations:[/bold red]")

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
                console.print(f"  {i}. [white]{rec}[/white]")

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
        "Category", style="bold white", width=12, justify="center"
    )
    table.add_column(
        "Description", style="white", width=45, no_wrap=False
    )
    table.add_column(
        "Usage Example", style="dim white", width=50, no_wrap=False
    )
    table.add_column(
        "Key Args", style="dim white", width=20, no_wrap=False
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
        style="dim white",
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
        "[bold red]Quick Start Commands:[/bold red]\n"
        "• [white]swarms onboarding[/white] - Environment setup check\n"
        "• [white]swarms setup-check[/white] - Check your environment\n"
        "• [white]swarms chat[/white] or [white]swarms chat --task 'Hello'[/white] - Start interactive chat agent\n"
        "• [white]swarms agent --name 'MyAgent' [--task 'Hello World'][/white] - Create agent (task optional, interactive by default)\n"
        "• [white]swarms autoswarm --task 'analyze data' --model gpt-4[/white] - Auto-generate swarm\n"
        "• [white]swarms llm-council --task 'Your question'[/white] - Run LLM Council\n"
        "• [white]swarms heavy-swarm --task 'Your task'[/white] - Run HeavySwarm",
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
        console.print(f"\n[white]ℹ️ {help_text}[/white]")


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
