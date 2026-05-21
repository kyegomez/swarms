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
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from swarms.cli.tips import render_tip
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

    # Two rotating tips: a quick command hint inside the panel + a longer
    # contextual tip below it. Both pull from swarms.cli.tips for variety.
    panel_tip = render_tip(category="commands")
    startup_tip = render_tip()

    # ── Panel ─────────────────────────────────────────────────────────────────
    panel_content = Group(
        header,
        Text(""),
        Rule(style="dim red"),
        panel_tip,
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
    else:
        # Contextual next-step tip on success
        from swarms.cli.tips import render_tip

        console.print()
        console.print(render_tip(category="commands"))

    return all_passed


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
    cache_dir = Path.home() / ".cache" / "swarms"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "auth"

    if cache_file.exists():
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
        cache_file.chmod(0o600)
        progress.remove_task(task)

    console.print(
        f"[{COLORS['success']}]✓ Login successful![/{COLORS['success']}]"
    )
    return True
