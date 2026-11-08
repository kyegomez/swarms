"""Argument parser and help screen for the swarms CLI."""

import argparse
from typing import Any, List, Optional


DEFAULT_HEAVY_SWARM_MODEL: str = "gpt-5.4"


# Single source of truth: argparse validates against it and main() suggests the closest match.
COMMAND_CHOICES: List[str] = [
    "init",
    "onboarding",
    "get-api-key",
    "check-login",
    "run-agents",
    "load-markdown",
    "agent",
    "chat",
    "upgrade",
    "autoswarm",
    "setup-check",
    "llm-council",
    "heavy-swarm",
    "tips",
    "models",
]


class CustomHelpAction(argparse.Action):
    """
    Custom help action that displays a formatted commands/parameters table.

    This action overrides the default argparse help behavior to provide a
    more user-friendly, formatted display of available commands and their
    parameters using Rich tables.

    Attributes:
        Inherits all attributes from argparse.Action.
    """

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
        Execute the custom help action.

        Displays a plain-text usage message and command reference
        instead of the default argparse help output.

        Args:
            parser: The argument parser instance.
            namespace: The namespace object (unused in this implementation).
            values: The option values (unused for help action).
            option_string: The option string that triggered this action.
        """
        prog_name: str = (
            parser.prog
            if hasattr(parser, "prog") and parser.prog
            else "swarms"
        )
        print(f"usage: {prog_name} COMMAND [OPTIONS]")
        print()
        print("Swarms Cloud CLI - Multi-Agent Framework")
        print()
        print("Commands:")
        commands = [
            (
                "init",
                "Scaffold a new project with .env, agents.yaml, and workspace",
            ),
            ("onboarding", "Run environment setup check"),
            (
                "get-api-key",
                "Open browser to retrieve API keys from the platform",
            ),
            (
                "check-login",
                "Verify authentication status and initialize cache",
            ),
            (
                "run-agents",
                "Execute agents from a YAML configuration file",
            ),
            (
                "load-markdown",
                "Load agents from markdown files with YAML frontmatter",
            ),
            ("agent", "Create and run a custom agent"),
            ("chat", "Start an interactive chat agent"),
            ("upgrade", "Update Swarms to the latest version"),
            ("autoswarm", "Generate and execute an autonomous swarm"),
            (
                "setup-check",
                "Run a comprehensive environment setup check",
            ),
            (
                "llm-council",
                "Run LLM Council with multiple collaborating agents",
            ),
            (
                "heavy-swarm",
                "Run HeavySwarm with specialized agents for complex tasks",
            ),
        ]
        for cmd, desc in commands:
            print(f"  {cmd:<20}  {desc}")
        print()
        print("Options:")
        options = [
            ("-h, --help", "Show this help message and exit"),
            (
                "--dir PATH",
                "Project directory for 'swarms init' (default: prompted)",
            ),
            (
                "--yaml-file FILE",
                "YAML configuration file path (default: agents.yaml)",
            ),
            (
                "--markdown-path PATH",
                "Path to markdown file or directory",
            ),
            ("--task TASK", "Task for the agent to execute"),
            ("--name NAME", "Name of the custom agent"),
            ("--description DESC", "Description of the custom agent"),
            (
                "--system-prompt PROMPT",
                "System prompt for the custom agent",
            ),
            (
                "--model-name MODEL",
                "Model name for agent (default: gpt-4)",
            ),
            ("--model MODEL", "Model name for autoswarm command"),
            (
                "--temperature FLOAT",
                "Temperature setting for the agent (0.0-2.0)",
            ),
            (
                "--max-loops N",
                "Maximum number of loops (integer or 'auto')",
            ),
            (
                "--loops-per-agent N",
                "Loops per agent for heavy-swarm (default: 1)",
            ),
            (
                "--question-agent-model-name MODEL",
                f"Model for question generation agent (default: {DEFAULT_HEAVY_SWARM_MODEL})",
            ),
            (
                "--worker-model-name MODEL",
                f"Model for worker agents (default: {DEFAULT_HEAVY_SWARM_MODEL})",
            ),
            ("--context-length N", "Context length for the agent"),
            ("--retry-attempts N", "Number of retry attempts"),
            (
                "--output-type TYPE",
                "Output type (e.g., 'str', 'json')",
            ),
            (
                "--saved-state-path PATH",
                "Path for saving agent state",
            ),
            ("--user-name NAME", "Username for the agent"),
            ("--mcp-url URL", "MCP URL for the agent"),
            (
                "--marketplace-prompt-id ID",
                "Fetch system prompt from Swarms marketplace",
            ),
            ("--verbose", "Enable verbose output"),
            ("--interactive", "Enable interactive mode (default)"),
            ("--no-interactive", "Disable interactive mode"),
            ("--streaming-on", "Enable streaming mode"),
            (
                "--concurrent",
                "Enable concurrent processing for markdown files",
            ),
            (
                "--random-loops-per-agent",
                "Enable random number of loops per agent",
            ),
            (
                "--auto-generate-prompt",
                "Enable auto-generation of prompts",
            ),
            (
                "--dynamic-temperature-enabled",
                "Enable dynamic temperature adjustment",
            ),
            (
                "--dynamic-context-window",
                "Enable dynamic context window",
            ),
            (
                "--return-step-meta",
                "Return step metadata from the agent",
            ),
            ("--dashboard", "Enable agent dashboard"),
            ("--autosave", "Enable autosave for the agent"),
        ]
        for flag, desc in options:
            if len(flag) <= 28:
                print(f"  {flag:<30}  {desc}")
            else:
                print(f"  {flag}")
                print(f"  {'':30}  {desc}")
        print()
        print("Examples:")
        print(f"  {prog_name} chat")
        print(f"  {prog_name} chat --model-name gpt-5.4")
        print(f"  {prog_name} chat --model-name claude-opus-4-6")
        print(f"  {prog_name} chat --model-name gpt-4o")
        print(
            f"  {prog_name} chat --model-name gemini/gemini-2.0-flash"
        )
        print(
            f"  {prog_name} agent --name 'MyAgent' --task 'Summarize this'"
        )
        print(
            f"  {prog_name} autoswarm --task 'analyze data' --model gpt-4"
        )
        print(f"  {prog_name} heavy-swarm --task 'complex analysis'")
        print(f"  {prog_name} llm-council --task 'What is AGI?'")
        print(f"  {prog_name} setup-check --verbose")
        print()
        print("Documentation:  https://docs.swarms.world")
        print(
            "Support:        https://github.com/kyegomez/swarms/issues"
        )
        print("Community:      https://discord.gg/EamjgSaEQf")
        print()
        parser.exit()


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up and configure the argument parser for the CLI.

    Creates and configures an ArgumentParser with all available commands and
    their associated arguments. Uses a custom help action to provide formatted
    command/parameter tables.

    Returns:
        argparse.ArgumentParser: A fully configured argument parser with:
            - Custom help action (CustomHelpAction)
            - Command choices (onboarding, help, get-api-key, etc.)
            - Command-specific arguments (--yaml-file, --task, --model, etc.)
            - Agent configuration arguments (--name, --description, etc.)
            - Swarm-specific arguments (--loops-per-agent, etc.)

    Note:
        The parser uses add_help=False to allow custom help formatting via
        CustomHelpAction. All commands and their arguments are defined here,
        making this the central configuration point for CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Swarms Cloud CLI",
        add_help=False,  # We'll add custom help action
    )
    parser.add_argument(
        "-h",
        "--help",
        action=CustomHelpAction,
        help="Show this help message and exit",
    )
    parser.add_argument(
        "command",
        metavar="COMMAND",
        choices=COMMAND_CHOICES,
        help=f"Command to execute. Available commands: {', '.join(COMMAND_CHOICES)}",
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
        default=None,
        help="Model name to use (e.g. gpt-5.4, gpt-4o, claude-opus-4-6). Defaults to gpt-5.4 for chat, gpt-4 for agent.",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task for the custom agent to execute (optional)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode for the agent (default: True)",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive mode for the agent",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature setting for the agent (0.0-2.0)",
    )
    parser.add_argument(
        "--max-loops",
        type=str,
        help="Maximum number of loops for the agent (integer or 'auto' for autonomous loops)",
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
        default=DEFAULT_HEAVY_SWARM_MODEL,
        help=f"Model name for question generation agent (default: {DEFAULT_HEAVY_SWARM_MODEL})",
    )
    parser.add_argument(
        "--worker-model-name",
        type=str,
        default=DEFAULT_HEAVY_SWARM_MODEL,
        help=f"Model name for specialized worker agents (default: {DEFAULT_HEAVY_SWARM_MODEL})",
    )
    parser.add_argument(
        "--random-loops-per-agent",
        action="store_true",
        help="Enable random number of loops per agent (1-10 range)",
    )
    # Autoswarm specific arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Model name for autoswarm command",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path for the generated Python script (autoswarm command)",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to create the generated Python script in (autoswarm command)",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only write the generated Python file, do not execute the swarm (autoswarm command)",
    )
    # Init specific arguments
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory for 'swarms init' (default: prompted interactively)",
    )
    # Tips command arguments
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of tips to show (used with 'swarms tips')",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Restrict tips to a category: commands, agents, swarms, models, pro, trivia, env, community",
    )
    parser.add_argument(
        "--all",
        dest="all_tips",
        action="store_true",
        help="Show every tip in the selected category (or every category when no --category is given)",
    )
    # Models command arguments
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Fuzzy-search the model list (used with 'swarms models')",
    )
    parser.add_argument(
        "--info",
        type=str,
        default=None,
        help="Show full metadata for a single model (used with 'swarms models')",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Restrict 'swarms models' to one provider (e.g. openai, anthropic)",
    )
    return parser
