"""Argument parser and help screen for the swarms CLI."""

import argparse
import shutil
import textwrap
from typing import Any, Dict, List, Optional

from swarms.cli.utils import check_swarms_version

DEFAULT_HEAVY_SWARM_MODEL: str = "gpt-5.4"


# Single source of truth: argparse validates against it, the help screen lists it and main() suggests from it.
COMMANDS: Dict[str, str] = {
    "init": "Create a .env file and workspace directory for a new project",
    "onboarding": "Run the environment setup check (same as setup-check)",
    "get-api-key": "Open the Swarms platform API key page in your browser",
    "run-agents": "Execute agents from a YAML configuration file",
    "load-markdown": "Load agents from markdown files with YAML frontmatter",
    "agent": "Create and run a custom agent",
    "chat": "Start an interactive chat agent",
    "upgrade": "Update Swarms to the latest version",
    "autoswarm": "Generate and execute an autonomous swarm",
    "setup-check": "Run a comprehensive environment setup check",
    "llm-council": "Run LLM Council with multiple collaborating agents",
    "heavy-swarm": "Run HeavySwarm with specialized agents for complex tasks",
    "tips": "Show tips and tricks for the CLI",
    "models": "List, search and inspect available models",
}

COMMAND_CHOICES: List[str] = list(COMMANDS)

HELP_EXAMPLES: List[str] = [
    "chat",
    "chat --model-name gpt-5.4",
    "chat --model-name claude-opus-4-6",
    "chat --model-name gpt-4o",
    "chat --model-name gemini/gemini-2.0-flash",
    "agent --name 'MyAgent' --task 'Summarize this'",
    "autoswarm --task 'analyze data' --model gpt-4",
    "heavy-swarm --task 'complex analysis'",
    "llm-council --task 'What is AGI?'",
    "setup-check --verbose",
    "models --search claude",
    "tips --category agents",
]

_FLAG_WIDTH = 30


def _option_lines(action: argparse.Action, width: int) -> List[str]:
    """
    Help lines for one option: its flags, then its wrapped description.

    Args:
        action: The argparse action for the option.
        width: Maximum width of the description column.

    Returns:
        The lines to print for this option.
    """
    flag = ", ".join(action.option_strings)
    if action.nargs != 0:
        flag += f" {action.metavar or action.dest.upper()}"

    desc = action.help or ""
    default = action.default
    if (
        default is not None
        and default is not argparse.SUPPRESS
        and not isinstance(default, bool)
        and "default" not in desc.lower()
    ):
        desc += f" (default: {default})"

    pad = " " * (_FLAG_WIDTH + 4)
    wrapped = textwrap.wrap(desc, width, break_on_hyphens=False) or [
        ""
    ]
    if len(flag) <= _FLAG_WIDTH:
        lines = [f"  {flag:<{_FLAG_WIDTH}}  {wrapped[0]}"]
    else:
        lines = [f"  {flag}", pad + wrapped[0]]
    return lines + [pad + line for line in wrapped[1:]]


def format_help(parser: argparse.ArgumentParser) -> str:
    """
    Help screen built from the command registry and the parser's options.

    Args:
        parser: The CLI argument parser.

    Returns:
        The full help text.
    """
    prog = parser.prog or "swarms"
    width = max(
        40, shutil.get_terminal_size().columns - _FLAG_WIDTH - 6
    )

    lines = [
        f"usage: {prog} COMMAND [OPTIONS]",
        "",
        "Swarms Cloud CLI - Multi-Agent Framework",
        "",
        "Commands:",
    ]
    lines += [
        f"  {name:<20}  {desc}" for name, desc in COMMANDS.items()
    ]

    for group in parser._action_groups:
        actions = [
            a for a in group._group_actions if a.option_strings
        ]
        if not actions:
            continue
        title = (
            "Options" if group is parser._optionals else group.title
        )
        lines += ["", f"{title}:"]
        if group.description:
            lines += [f"  {group.description}"]
        for action in actions:
            lines += _option_lines(action, width)

    lines += ["", "Examples:"]
    lines += [f"  {prog} {example}" for example in HELP_EXAMPLES]
    lines += [
        "",
        "Documentation:  https://docs.swarms.world",
        "Support:        https://github.com/kyegomez/swarms/issues",
        "Community:      https://discord.gg/EamjgSaEQf",
        "",
    ]
    return "\n".join(lines)


class CustomHelpAction(argparse.Action):
    """Help action that prints the generated help screen and exits."""

    def __init__(
        self,
        option_strings: List[str],
        dest: str = argparse.SUPPRESS,
        default: Any = argparse.SUPPRESS,
        help: Optional[str] = None,
    ) -> None:
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        """
        Print the help screen and exit.

        Args:
            parser: The argument parser.
            namespace: The parsed namespace, unused.
            values: The option values, unused.
            option_string: The option that triggered the action.
        """
        print(format_help(parser))
        parser.exit()


def _max_loops(value: str) -> Any:
    """Parse --max-loops as an integer or "auto"."""
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be an integer or 'auto', got {value!r}"
        )


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for every swarms command.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="swarms",
        description="Swarms Cloud CLI",
        add_help=False,
    )
    files = parser.add_argument_group(
        "run-agents and load-markdown options"
    )
    agent = parser.add_argument_group(
        "agent options",
        "chat reads only --name, --description, --system-prompt and --model-name.",
    )
    heavy = parser.add_argument_group("heavy-swarm options")
    autoswarm = parser.add_argument_group("autoswarm options")
    init = parser.add_argument_group("init options")
    tips = parser.add_argument_group("tips options")
    models = parser.add_argument_group("models options")

    parser.add_argument(
        "-h",
        "--help",
        action=CustomHelpAction,
        help="Show this help message and exit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"swarms {check_swarms_version()}",
        help="Print the installed swarms version and exit",
    )
    parser.add_argument(
        "command",
        nargs="?",
        metavar="COMMAND",
        choices=COMMAND_CHOICES,
        help=f"Command to execute. Available commands: {', '.join(COMMAND_CHOICES)}",
    )
    files.add_argument(
        "--yaml-file",
        type=str,
        default="agents.yaml",
        metavar="FILE",
        help="YAML configuration file path",
    )
    files.add_argument(
        "--markdown-path",
        type=str,
        metavar="PATH",
        help="Path to markdown file or directory containing markdown files",
    )
    files.add_argument(
        "--concurrent",
        action="store_true",
        default=True,
        help="Enable concurrent processing for multiple markdown files (default: True)",
    )
    agent.add_argument(
        "--name",
        type=str,
        metavar="NAME",
        help="Name of the custom agent",
    )
    agent.add_argument(
        "--description",
        type=str,
        metavar="TEXT",
        help="Description of the custom agent",
    )
    agent.add_argument(
        "--system-prompt",
        type=str,
        metavar="PROMPT",
        help="System prompt for the custom agent",
    )
    agent.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="MODEL",
        help="Model name to use (e.g. gpt-5.4, gpt-4o, claude-opus-4-6). Defaults to gpt-5.4 for agent, gpt-5.1 for chat.",
    )
    parser.add_argument(
        "--task",
        type=str,
        metavar="TASK",
        help="Task to run (agent, chat, autoswarm, llm-council, heavy-swarm)",
    )
    agent.add_argument(
        "--interactive",
        action="store_true",
        help="Keep prompting for input after the first response (default: off)",
    )
    agent.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive mode for the agent",
    )
    agent.add_argument(
        "--temperature",
        type=float,
        metavar="FLOAT",
        help="Temperature setting for the agent (0.0-2.0)",
    )
    agent.add_argument(
        "--max-loops",
        type=_max_loops,
        metavar="N",
        help="Maximum number of loops for the agent (integer or 'auto' for autonomous loops)",
    )
    agent.add_argument(
        "--auto-generate-prompt",
        action="store_true",
        help="Enable auto-generation of prompts",
    )
    agent.add_argument(
        "--dynamic-temperature-enabled",
        action="store_true",
        help="Enable dynamic temperature adjustment",
    )
    agent.add_argument(
        "--dynamic-context-window",
        action="store_true",
        help="Enable dynamic context window",
    )
    agent.add_argument(
        "--output-type",
        type=str,
        metavar="TYPE",
        help="Output type for the agent (e.g., 'str', 'json')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (agent, setup-check, onboarding, llm-council, heavy-swarm)",
    )
    agent.add_argument(
        "--streaming-on",
        action="store_true",
        help="Enable streaming mode for the agent",
    )
    agent.add_argument(
        "--context-length",
        type=int,
        metavar="N",
        help="Context length for the agent",
    )
    agent.add_argument(
        "--retry-attempts",
        type=int,
        metavar="N",
        help="Number of retry attempts for the agent",
    )
    agent.add_argument(
        "--return-step-meta",
        action="store_true",
        help="Return step metadata from the agent",
    )
    agent.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable dashboard for the agent",
    )
    agent.add_argument(
        "--autosave",
        action="store_true",
        help="Enable autosave for the agent",
    )
    agent.add_argument(
        "--saved-state-path",
        type=str,
        metavar="PATH",
        help="Path for saving agent state",
    )
    agent.add_argument(
        "--user-name",
        type=str,
        metavar="NAME",
        help="Username for the agent",
    )
    agent.add_argument(
        "--mcp-url",
        type=str,
        metavar="URL",
        help="MCP URL for the agent",
    )
    agent.add_argument(
        "--marketplace-prompt-id",
        type=str,
        metavar="ID",
        help="Fetch system prompt from Swarms marketplace using this prompt ID",
    )
    heavy.add_argument(
        "--loops-per-agent",
        type=int,
        default=1,
        metavar="N",
        help="Number of execution loops each agent should perform (default: 1)",
    )
    heavy.add_argument(
        "--question-agent-model-name",
        type=str,
        default=DEFAULT_HEAVY_SWARM_MODEL,
        metavar="MODEL",
        help=f"Model name for question generation agent (default: {DEFAULT_HEAVY_SWARM_MODEL})",
    )
    heavy.add_argument(
        "--worker-model-name",
        type=str,
        default=DEFAULT_HEAVY_SWARM_MODEL,
        metavar="MODEL",
        help=f"Model name for specialized worker agents (default: {DEFAULT_HEAVY_SWARM_MODEL})",
    )
    heavy.add_argument(
        "--random-loops-per-agent",
        action="store_true",
        help="Enable random number of loops per agent (1-10 range)",
    )
    autoswarm.add_argument(
        "--model",
        type=str,
        metavar="MODEL",
        help="Model name for autoswarm command",
    )
    autoswarm.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Output file path for the generated Python script (autoswarm command)",
    )
    autoswarm.add_argument(
        "-d",
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to create the generated Python script in (autoswarm command)",
    )
    autoswarm.add_argument(
        "--no-run",
        action="store_true",
        help="Only write the generated Python file, do not execute the swarm (autoswarm command)",
    )
    init.add_argument(
        "--dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory for 'swarms init' (default: prompted interactively)",
    )
    tips.add_argument(
        "--count",
        type=int,
        default=1,
        metavar="N",
        help="Number of tips to show (used with 'swarms tips')",
    )
    tips.add_argument(
        "--category",
        type=str,
        default=None,
        metavar="CATEGORY",
        help="Restrict tips to a category: commands, agents, swarms, models, pro, trivia, env, community",
    )
    tips.add_argument(
        "--all",
        dest="all_tips",
        action="store_true",
        help="Show every tip in the selected category (or every category when no --category is given)",
    )
    models.add_argument(
        "--search",
        type=str,
        default=None,
        metavar="PATTERN",
        help="Fuzzy-search the model list (used with 'swarms models')",
    )
    models.add_argument(
        "--info",
        type=str,
        default=None,
        metavar="MODEL",
        help="Show full metadata for a single model (used with 'swarms models')",
    )
    models.add_argument(
        "--provider",
        type=str,
        default=None,
        metavar="PROVIDER",
        help="Restrict 'swarms models' to one provider (e.g. openai, anthropic)",
    )
    return parser
