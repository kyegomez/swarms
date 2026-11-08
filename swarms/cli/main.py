"""
Entry point for the swarms command-line interface.

Parses the command line and routes each command to its handler in
swarms.cli.commands. The parser lives in swarms.cli.parser and the
error screen in swarms.cli.errors.
"""

import argparse
import difflib
import subprocess
import sys
import traceback
from typing import Any, Dict, Optional

from swarms.cli.commands import (
    handle_agent,
    handle_autoswarm,
    handle_chat,
    handle_heavy_swarm,
    handle_init,
    handle_llm_council,
    handle_load_markdown,
    handle_models,
    handle_onboarding,
    handle_run_agents,
    handle_tips,
)
from swarms.cli.errors import _show_command_error
from swarms.cli.parser import COMMAND_CHOICES, setup_argument_parser
from swarms.cli.utils import (
    check_login,
    get_api_key,
    run_setup_check,
    show_ascii_art,
    show_error,
)
from swarms.env import load_swarms_env
from swarms.utils.formatter import formatter

load_swarms_env()


def _suggest_command(typed: str) -> Optional[str]:
    """Return the closest valid command, or None if nothing is close enough."""
    matches = difflib.get_close_matches(
        typed, COMMAND_CHOICES, n=1, cutoff=0.5
    )
    return matches[0] if matches else None


def route_command(args: argparse.Namespace) -> None:
    """
    Route the command to the appropriate handler function.

    Maps CLI commands to their respective handler functions and executes them.
    If a command is not recognized, displays an error message with helpful
    suggestions.

    Args:
        args: Parsed command line arguments containing:
            - command: The command name to execute
            - Additional command-specific arguments

    Note:
        The command routing dictionary maps command names to their handler
        functions. Some commands use lambda functions for simple operations
        like opening URLs or displaying help.
    """
    command_handlers: Dict[str, Any] = {
        "init": handle_init,
        "onboarding": handle_onboarding,
        "get-api-key": lambda args: get_api_key(),
        "check-login": lambda args: check_login(),
        "run-agents": handle_run_agents,
        "load-markdown": handle_load_markdown,
        "agent": handle_agent,
        "chat": handle_chat,
        "upgrade": lambda args: subprocess.run(
            ["pip", "install", "--upgrade", "swarms"], check=True
        ),
        "autoswarm": handle_autoswarm,
        "setup-check": lambda args: run_setup_check(
            verbose=args.verbose
        ),
        "llm-council": handle_llm_council,
        "heavy-swarm": handle_heavy_swarm,
        "tips": handle_tips,
        "models": handle_models,
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        show_error(
            "Unknown command",
            f"Command '{args.command}' is not recognized. Use 'swarms help' to see available commands.",
        )


def main() -> None:
    """
    Main entry point for the Swarms CLI.

    This function serves as the primary entry point for the command-line
    interface. It handles:
    1. Displaying the ASCII art banner
    2. Setting up and parsing command-line arguments
    3. Routing commands to appropriate handlers
    4. Error handling and user-friendly error messages

    The function provides comprehensive error handling at multiple levels:
    - Command execution errors are caught and displayed with troubleshooting tips
    - Critical errors (argument parsing, etc.) are displayed with formatted panels
    - All errors include traceback information for debugging

    Raises:
        Exception: Re-raises critical errors after displaying formatted error
            messages. A failed command is displayed and exits with status 1.

    Note:
        The function uses try-except blocks to ensure graceful error handling
        and user-friendly error messages. Tracebacks are included for debugging
        purposes while maintaining a clean user experience.
    """
    try:
        show_ascii_art()

        # Catch a mistyped command before argparse's generic "invalid choice"
        if (
            len(sys.argv) > 1
            and not sys.argv[1].startswith("-")
            and sys.argv[1] not in COMMAND_CHOICES
        ):
            typed = sys.argv[1]
            suggestion = _suggest_command(typed)
            hint = (
                f"\nDid you mean [bold white]swarms {suggestion}[/bold white]?"
                if suggestion
                else ""
            )
            show_error(
                f"Unknown command '{typed}'",
                f"Available commands: {', '.join(COMMAND_CHOICES)}{hint}\n"
                f"Run [bold white]swarms --help[/bold white] for full usage.",
            )
            sys.exit(2)

        parser = setup_argument_parser()
        args = parser.parse_args()

        try:
            route_command(args)
        except Exception as e:
            _show_command_error(e)
            sys.exit(1)
    except Exception as error:
        formatter.print_panel(
            f"Critical error detected: {error}\n\n"
            "Your command could not be processed due to the above error.\n"
            "👉 Please review your arguments, environment settings, and try again.\n"
            "For more information, run 'swarms help' or visit the documentation:\n"
            "https://docs.swarms.world/cli/overview\n\n"
            f"[dim]Traceback:[/dim]\n{traceback.format_exc()}",
            title="Fatal Error",
            style="red",
        )
        raise error


if __name__ == "__main__":
    main()
