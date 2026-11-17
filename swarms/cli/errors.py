"""Error rendering for commands that fail inside the swarms CLI."""

import traceback
from typing import List

from rich.markup import escape as rich_escape

from swarms.cli.utils import COLORS, console


def _classify_command_error(message: str) -> List[str]:
    """
    Inspect an exception message and return a list of targeted recovery hints.

    Matches are case-insensitive and additive: multiple hints can fire for a
    single error (e.g. a missing-key error that also mentions a stale model).
    """
    err = message.lower()
    hints: List[str] = []

    if (
        "401" in err
        or "unauthorized" in err
        or "authenticationerror" in err
        or "invalid api key" in err
        or "missing api key" in err
    ):
        hints.append(
            "Authentication failed — set your provider key in [bold]swarms init[/bold] "
            "or get one via [bold]swarms get-api-key[/bold]"
        )

    if (
        "model not found" in err
        or "model_not_found" in err
        or ("model" in err and "does not exist" in err)
    ):
        hints.append(
            "Find a valid model with [bold]swarms models --search <name>[/bold] "
            "or list them all with [bold]swarms models[/bold]"
        )

    if "workspace_dir" in err or "workspace directory" in err:
        hints.append(
            "Workspace not configured — run [bold]swarms init[/bold] to scaffold one"
        )

    if "rate limit" in err or "ratelimit" in err or "429" in err:
        hints.append(
            "Rate-limited by provider — slow down, use a smaller model with "
            "[bold]--model-name[/bold], or retry in a minute"
        )

    if (
        "connection" in err
        or "timeout" in err
        or "timed out" in err
        or "network" in err
    ):
        hints.append(
            "Network issue — check connectivity, then run [bold]swarms setup-check --verbose[/bold]"
        )

    if "modulenotfounderror" in err or "no module named" in err:
        hints.append(
            "Missing dependency — try [bold]swarms upgrade[/bold] or "
            "[bold]pip install -U swarms[/bold]"
        )

    return hints


def _show_command_error(error: Exception) -> None:
    """Render a command-execution error with classified recovery hints."""
    hints = _classify_command_error(str(error))

    hint_block = ""
    if hints:
        hint_block = (
            "[bold white]Suggested next steps:[/bold white]\n"
            + "\n".join(f"  • {h}" for h in hints)
        )
    else:
        hint_block = (
            "[bold white]Troubleshooting tips:[/bold white]\n"
            "- Double-check your arguments and the command structure\n"
            "- Try [bold]swarms --help[/bold] for command details\n"
            "- Run [bold]swarms setup-check --verbose[/bold] to validate your environment\n"
            "- Report bugs at https://github.com/kyegomez/swarms/issues"
        )

    console.print(
        f"\n[{COLORS['error']}]Oops! An unexpected error occurred while running your command:[/{COLORS['error']}]\n"
        f"[bold]{rich_escape(str(error))}[/bold]\n\n"
        f"{hint_block}\n\n"
        f"[dim]Traceback:[/dim]\n{rich_escape(traceback.format_exc())}"
    )
