"""
Main CLI module for the Swarms command-line interface.

This module provides the entry point and command routing for the Swarms CLI,
handling argument parsing, command dispatching, and execution of various
Swarms operations including agent creation, swarm management, and system checks.

The CLI supports multiple commands:
    - Agent management (create, run, load from YAML/markdown)
    - Swarm operations (autoswarm, heavy-swarm, llm-council)
    - System utilities (setup-check, onboarding, authentication)
    - Interactive chat agent
    - Help and documentation

All commands are routed through handler functions that provide enhanced
error handling, progress feedback, and user-friendly output formatting.
"""

import argparse
import os
import webbrowser
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from litellm import traceback
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from swarms.agents.auto_generate_swarm_config import (
    generate_swarm_config,
)
from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
from swarms.agents.auto_chat_agent import auto_chat_agent

from swarms.structs.agent import Agent
from swarms.structs.agent_loader import AgentLoader
from swarms.structs.llm_council import LLMCouncil
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.utils.formatter import formatter

from swarms.cli.utils import (
    COLORS,
    SwarmCLIError,
    check_login,
    console,
    create_commands_parameters_table,
    get_api_key,
    run_setup_check,
    show_ascii_art,
    show_error,
    show_features,
    show_help,
)

load_dotenv()


def run_autoswarm(task: str, model: str) -> None:
    """
    Generate and execute an autonomous swarm configuration.

    This function validates inputs, generates a swarm configuration using
    the specified model, and handles errors with user-friendly messages.

    Args:
        task: The task description for the swarm to execute. Must be non-empty.
        model: The model name to use for swarm generation (e.g., 'gpt-4').
            Must be non-empty.

    Raises:
        SwarmCLIError: If task or model is empty or invalid.
        Exception: If swarm generation fails, displays formatted error message.

    Note:
        The function provides enhanced error handling with specific messages
        for common issues like missing API keys or invalid model configurations.
    """
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
    file_path: str, concurrent: bool = True, **kwargs: Any
) -> List[Agent]:
    """
    Load agents from markdown files with enhanced visual feedback.

    Supports loading from either a single markdown file or a directory
    containing multiple markdown files. Each markdown file should contain
    YAML frontmatter with agent configuration.

    Args:
        file_path: Path to a markdown file or directory containing markdown files.
            Can be a relative or absolute path.
        concurrent: Whether to load multiple agents concurrently when processing
            a directory. Defaults to True for better performance.
        **kwargs: Additional keyword arguments to pass to the agent loader.

    Returns:
        List[Agent]: A list of loaded Agent instances. Returns an empty list
            if no agents were loaded or if an error occurred.

    Raises:
        FileNotFoundError: If the specified file or directory doesn't exist.
        ValueError: If the markdown file format is invalid or missing required
            YAML frontmatter.

    Note:
        The function displays a formatted table of loaded agents including
        their names, models, and descriptions upon successful loading.
    """
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
) -> Optional[Any]:
    """
    Run the HeavySwarm with a given task.

    HeavySwarm uses specialized agents to analyze complex tasks by breaking
    them down into questions and having worker agents process them.

    Args:
        task: The task or query for the HeavySwarm to process. Must be non-empty.
        loops_per_agent: Number of execution loops each agent should perform.
            Defaults to 1. Higher values allow more iterative refinement.
        question_agent_model_name: Model name for the question generation agent.
            Defaults to "gpt-4o-mini".
        worker_model_name: Model name for specialized worker agents that process
            the questions. Defaults to "gpt-4o-mini".
        random_loops_per_agent: If True, enables random number of loops per agent
            in the range 1-10. Defaults to False.
        verbose: Whether to show verbose output during execution. Defaults to False.

    Returns:
        Optional[Any]: The result from the HeavySwarm execution, or None if
            execution failed or returned no results.

    Raises:
        Exception: If HeavySwarm initialization or execution fails. Errors are
            displayed with formatted messages and troubleshooting tips.
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


def run_llm_council(task: str, verbose: bool = True) -> Optional[Any]:
    """
    Run the LLM Council with a given task.

    The LLM Council uses multiple agents to collaborate on a task, with each
    agent providing different perspectives and evaluating responses.

    Args:
        task: The task or query for the LLM Council to process. Must be non-empty.
        verbose: Whether to show verbose output during execution. Defaults to True
            to provide detailed information about the council's deliberations.

    Returns:
        Optional[Any]: The final response from the LLM Council, or None if
            execution failed or returned no results.

    Raises:
        Exception: If LLM Council initialization or execution fails. Errors are
            displayed with formatted messages and troubleshooting tips.
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
    task: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Union[Agent, Any]]:
    """
    Create and optionally run a custom agent with the specified parameters.

    This function creates a new Agent instance with the provided configuration.
    If a task is provided, the agent will execute it immediately. Otherwise,
    the agent is created in interactive mode, ready for user input.

    Args:
        name: The name of the agent. Used for identification and display.
        description: A description of the agent's purpose and capabilities.
        system_prompt: The system prompt that defines the agent's behavior
            and instructions.
        model_name: The LLM model to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
        task: Optional task for the agent to execute immediately. If None,
            the agent is created in interactive mode. Defaults to None.
        **kwargs: Additional keyword arguments passed to the Agent constructor.
            Common options include:
            - temperature: Float between 0.0 and 2.0
            - max_loops: Integer or "auto" for autonomous loops
            - interactive: Boolean to enable interactive mode
            - verbose: Boolean for verbose output
            - context_length: Integer for context window size
            - And other Agent configuration parameters

    Returns:
        Optional[Union[Agent, Any]]:
            - If task is provided: Returns the task execution result (Any)
            - If task is None: Returns the created Agent instance
            - Returns None if agent creation or execution fails

    Raises:
        Exception: If agent creation or execution fails. Errors are displayed
            with formatted messages and troubleshooting tips.

    Note:
        The function displays progress indicators and formatted results panels
        for better user experience. Agent information is shown in a formatted
        table upon successful creation.
    """
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
                description="Agent created successfully!",
            )

            # Only run the agent if a task is provided
            if task:
                progress.update(
                    init_task,
                    description=f"Running task: {task[:50]}...",
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
            else:
                # No task provided, just create the agent
                progress.update(
                    init_task,
                    description="Agent created! (No task provided - use interactive mode)",
                    completed=True,
                )
                console.print(
                    f"\n[bold green]✓ Agent '{name}' created successfully![/bold green]"
                )
                console.print(
                    "[yellow]ℹ️  No task provided. Agent is ready for interactive use.[/yellow]"
                )

                # Display agent info
                agent_info = Panel(
                    f"[bold]Agent Name:[/bold] {name}\n"
                    f"[bold]Model:[/bold] {model_name}\n"
                    f"[bold]Interactive Mode:[/bold] Enabled\n"
                    f"[bold]Status:[/bold] Ready for use",
                    title="Agent Created",
                    border_style="green",
                    padding=(1, 2),
                )
                console.print(agent_info)

                return agent

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

        Displays a formatted usage message and commands/parameters table
        instead of the default argparse help output.

        Args:
            parser: The argument parser instance.
            namespace: The namespace object (unused in this implementation).
            values: The option values (unused for help action).
            option_string: The option string that triggered this action.
        """
        # Show simplified usage
        prog_name: str = (
            parser.prog
            if hasattr(parser, "prog") and parser.prog
            else "swarms"
        )
        console.print("\n[bold cyan]Usage:[/bold cyan]")
        console.print(f"  {prog_name} COMMAND [OPTIONS]\n")

        # Show the commands/parameters table prominently
        console.print(create_commands_parameters_table())

        # Show a note about detailed options
        console.print(
            "\n[dim]💡 Tip: For detailed information about all available options, "
            "use a specific command or see the full documentation at "
            "https://docs.swarms.world[/dim]\n"
        )
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
    command_choices = [
        "onboarding",
        "help",
        "get-api-key",
        "check-login",
        "run-agents",
        "load-markdown",
        "agent",
        "chat",
        "auto-upgrade",
        "book-call",
        "autoswarm",
        "hierarchical-auto",
        "setup-check",
        "llm-council",
        "heavy-swarm",
        "features",
    ]
    parser.add_argument(
        "command",
        metavar="COMMAND",
        choices=command_choices,
        help=f"Command to execute. Available commands: {', '.join(command_choices)}",
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
    # Autoswarm specific arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Model name for autoswarm command",
    )
    parser.add_argument(
        "--auto-build",
        action="store_true",
        help="Auto-build agents from the provided task prompt before running the hierarchical swarm",
    )
    parser.add_argument(
        "--department",
        type=str,
        help="Department name to add auto-built agents to (optional)",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel execution of orders",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers when executing orders",
    )
    return parser


def handle_onboarding(args: argparse.Namespace) -> None:
    """
    Handle the onboarding command.

    Runs the same environment setup checks as the setup-check command,
    verifying Python version, Swarms installation, API keys, dependencies,
    environment files, and workspace directory configuration.

    Args:
        args: Parsed command line arguments containing:
            - verbose: Optional boolean flag for verbose output

    Note:
        This command is an alias for 'setup-check' and provides the same
        functionality for backward compatibility.
    """
    console.print(
        "[yellow]Note: 'swarms onboarding' now runs the same checks as 'swarms setup-check'[/yellow]"
    )
    run_setup_check(verbose=args.verbose)


def handle_run_agents(args: argparse.Namespace) -> None:
    """
    Handle the run-agents command.

    Loads and executes agents from a YAML configuration file. The YAML file
    should contain agent definitions with their configurations and tasks.

    Args:
        args: Parsed command line arguments containing:
            - yaml_file: Path to the YAML configuration file (default: "agents.yaml")
            - verbose: Optional boolean flag for verbose output

    Raises:
        FileNotFoundError: If the specified YAML file doesn't exist.
        ValueError: If the YAML file format is invalid or malformed.
        Exception: For other execution errors, with enhanced error messages
            for common issues like context length exceeded or API key problems.

    Note:
        The function displays progress indicators and formatted results.
        Results can be strings, dictionaries, or other types depending on
        the agent configuration.
    """
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
            TextColumn("[progress.description]{task.description}"),
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
                console.print("\n[bold green]Results:[/bold green]")
                console.print(
                    Panel(
                        result,
                        title="Agent Output",
                        border_style="green",
                    )
                )
            elif isinstance(result, dict):
                console.print("\n[bold green]Results:[/bold green]")
                for key, value in result.items():
                    console.print(f"[cyan]{key}:[/cyan] {value}")
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
            str(e) + "\n\nPlease check your agents.yaml file format.",
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


def handle_load_markdown(args: argparse.Namespace) -> None:
    """
    Handle the load-markdown command.

    Loads agents from markdown files with YAML frontmatter. Supports loading
    from a single file or a directory containing multiple markdown files.

    Args:
        args: Parsed command line arguments containing:
            - markdown_path: Required path to markdown file or directory
            - concurrent: Optional boolean flag for concurrent processing
                (default: True)

    Exits:
        Exits with code 1 if --markdown-path is not provided.

    Note:
        Displays a formatted table of loaded agents upon successful completion.
        Agents are ready for use in code or interactive execution.
    """
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


def handle_agent(args: argparse.Namespace) -> None:
    """
    Handle the agent command for creating and running custom agents.

    Validates required arguments, maps CLI parameters to agent configuration,
    and creates/runs the agent with the specified parameters.

    Args:
        args: Parsed command line arguments containing:
            - name: Required agent name
            - description: Required agent description
            - system_prompt: Required unless marketplace_prompt_id is provided
            - marketplace_prompt_id: Optional prompt ID from Swarms marketplace
            - task: Optional task to execute immediately
            - model_name: Model to use (default: "gpt-4")
            - interactive: Enable interactive mode (default: True)
            - temperature: Temperature setting (0.0-2.0)
            - max_loops: Maximum loops (integer or "auto")
            - And other optional agent configuration parameters

    Exits:
        Exits with code 1 if required arguments are missing or if max_loops
        value is invalid.

    Note:
        The task parameter is optional. If not provided, the agent is created
        in interactive mode, ready for user input. The function handles conversion
        of max_loops from string to int or "auto" as appropriate.
    """
    # Validate required arguments
    # system_prompt not required if marketplace_prompt_id provided
    # task is now optional
    required_args = ["name", "description"]
    if not getattr(args, "marketplace_prompt_id", None):
        required_args.append("system_prompt")

    missing_args = [
        arg for arg in required_args if not getattr(args, arg)
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
            "  --temperature 0.1\n\n"
            "Note: --task is optional. If not provided, agent will be created in interactive mode.",
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

    # Set interactive to True by default
    additional_params["interactive"] = getattr(
        args, "interactive", True
    )

    for cli_arg, agent_param in param_mapping.items():
        value = getattr(args, cli_arg, None)
        if value is not None:
            # Handle max_loops: convert to int if numeric, keep "auto" as string
            if cli_arg == "max_loops":
                if value.lower() == "auto":
                    additional_params[agent_param] = "auto"
                else:
                    try:
                        additional_params[agent_param] = int(value)
                    except ValueError:
                        show_error(
                            "Invalid max-loops value",
                            f"max-loops must be an integer or 'auto', got: {value}",
                        )
                        exit(1)
            else:
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


def handle_autoswarm(args: argparse.Namespace) -> None:
    """
    Handle the autoswarm command.

    Generates and executes an autonomous swarm configuration based on the
    provided task and model.

    Args:
        args: Parsed command line arguments containing:
            - task: Required task description for the swarm
            - model: Required model name for swarm generation

    Exits:
        Exits with code 1 if --task is not provided.

    Note:
        The autoswarm command automatically generates a swarm configuration
        optimized for the given task using the specified model.
    """
    if not args.task:
        show_error(
            "Missing required argument: --task",
            "Example usage: python cli.py autoswarm --task 'analyze this data' --model gpt-4",
        )
        exit(1)
    run_autoswarm(args.task, args.model)


def handle_llm_council(args: argparse.Namespace) -> None:
    """
    Handle the llm-council command.

    Runs the LLM Council with multiple agents collaborating on a task.

    Args:
        args: Parsed command line arguments containing:
            - task: Required task or question for the council to process
            - verbose: Optional boolean flag for verbose output

    Exits:
        Exits with code 1 if --task is not provided.

    Note:
        The LLM Council uses multiple agents with different perspectives to
        collaborate and evaluate responses, providing comprehensive analysis.
    """
    if not args.task:
        show_error(
            "Missing required argument: --task",
            "Example usage: swarms llm-council --task 'What is the best approach to solve this problem?'",
        )
        exit(1)
    run_llm_council(task=args.task, verbose=args.verbose)


def handle_heavy_swarm(args: argparse.Namespace) -> None:
    """
    Handle the heavy-swarm command.

    Runs HeavySwarm with specialized agents for complex task analysis.

    Args:
        args: Parsed command line arguments containing:
            - task: Required task for HeavySwarm to process
            - loops_per_agent: Number of loops per agent (default: 1)
            - question_agent_model_name: Model for question generation
                (default: "gpt-4o-mini")
            - worker_model_name: Model for worker agents (default: "gpt-4o-mini")
            - random_loops_per_agent: Enable random loops (1-10 range)
            - verbose: Optional boolean flag for verbose output

    Exits:
        Exits with code 1 if --task is not provided.

    Note:
        HeavySwarm breaks down complex tasks into questions and uses specialized
        worker agents to process them, allowing for deep analysis.
    """
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


def handle_hierarchical_auto(args: argparse.Namespace) -> None:
    """
    Handle the hierarchical-auto command.

    Optionally auto-builds agents from the task prompt using AutoSwarmBuilder,
    then runs `HierarchicalSwarm` with the generated/available agents.
    """
    if not args.task:
        show_error(
            "Missing required argument: --task",
            "Example usage: swarms hierarchical-auto --task 'Analyze market trends' --auto-build --department Market",
        )
        exit(1)

    # Initialize swarm
    swarm = HierarchicalSwarm(
        name="hierarchical-auto",
        description="Auto-built hierarchical swarm via CLI",
        max_loops=1 if not args.max_loops else int(args.max_loops) if args.max_loops.isdigit() else 1,
        interactive=False,
        use_parallel_execution=getattr(args, "parallel", True),
        max_workers=getattr(args, "max_workers", None),
    )

    # Auto-build if requested
    if args.auto_build:
        try:
            swarm.auto_build_agents_from_prompt(args.task, department_name=getattr(args, "department", None))
            console.print("[green]✓ Auto-built agents and added to swarm[/green]")
        except Exception as e:
            show_error("Auto-build failed", str(e))
            return

    # Run the swarm
    try:
        console.print(f"[yellow]Running hierarchical swarm for task: {args.task}[/yellow]")
        result = swarm.run(task=args.task)
        console.print(Panel(result or "No result", title="Hierarchical Swarm Result", border_style="green"))
    except Exception as e:
        show_error("Hierarchical swarm run failed", str(e))


def handle_chat(args: argparse.Namespace) -> Optional[Agent]:
    """
    Handle the chat command for interactive chat agent.

    Initializes and runs an interactive chat agent with optimized defaults
    for conversation. The agent is configured for interactive use with
    autonomous loops enabled.

    Args:
        args: Parsed command line arguments containing:
            - name: Optional agent name (default: "Swarms Agent")
            - description: Optional agent description
            - system_prompt: Optional custom system prompt
            - interactive: Enable interactive mode (default: True)

    Returns:
        Optional[Agent]: The initialized chat agent instance, or None if
            initialization failed.

    Raises:
        Exception: If chat agent initialization fails. Errors are displayed
            with formatted messages and troubleshooting tips.

    Note:
        The chat agent is optimized for conversation with dynamic context
        window and temperature enabled, and uses autonomous loops for
        continuous interaction.
    """
    try:
        console.print(
            "[yellow]💬 Initializing chat agent...[/yellow]"
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
                "Initializing chat agent...", total=None
            )

            # Get parameters with defaults
            name = getattr(args, "name", "Swarms Agent")
            description = getattr(
                args,
                "description",
                "A Swarms agent that can chat with the user.",
            )
            system_prompt = getattr(args, "system_prompt", None)
            interactive = getattr(args, "interactive", True)

            # Update progress
            progress.update(
                init_task,
                description="Creating chat agent...",
            )

            # Create and run the chat agent
            result = auto_chat_agent(
                name=name,
                description=description,
                system_prompt=system_prompt,
                interactive=interactive,
            )

            # Update progress on completion
            progress.update(
                init_task,
                description="Chat agent ready!",
                completed=True,
            )

        if result:
            console.print(
                "\n[bold green]✓ Chat agent initialized successfully![/bold green]"
            )
            return result
        else:
            console.print(
                "[yellow]⚠ Chat agent initialized but returned no result.[/yellow]"
            )
            return None

    except Exception as e:
        show_error(
            "Chat Agent Error",
            f"Failed to initialize chat agent: {str(e)}\n\n"
            "Please check:\n"
            "1. Your API keys are set correctly\n"
            "2. You have network connectivity\n"
            "3. All parameters are properly formatted",
        )
        return None


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
        "onboarding": handle_onboarding,
        "help": lambda args: show_help(),
        "features": lambda args: show_features(),
        "get-api-key": lambda args: get_api_key(),
        "check-login": lambda args: check_login(),
        "run-agents": handle_run_agents,
        "load-markdown": handle_load_markdown,
        "agent": handle_agent,
        "chat": handle_chat,
        "book-call": lambda args: webbrowser.open(
            "https://cal.com/swarms/swarms-strategy-session"
        ),
        "autoswarm": handle_autoswarm,
        "hierarchical-auto": handle_hierarchical_auto,
        "setup-check": lambda args: run_setup_check(
            verbose=args.verbose
        ),
        "llm-council": handle_llm_council,
        "heavy-swarm": handle_heavy_swarm,
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
            messages. Command execution errors are caught and displayed without
            raising.

    Note:
        The function uses try-except blocks to ensure graceful error handling
        and user-friendly error messages. Tracebacks are included for debugging
        purposes while maintaining a clean user experience.
    """
    try:
        show_ascii_art()

        parser = setup_argument_parser()
        args = parser.parse_args()

        try:
            route_command(args)
        except Exception as e:
            console.print(
                f"\n[{COLORS['error']}]Oops! An unexpected error occurred while running your command:[/{COLORS['error']}]\n"
                f"[bold]{str(e)}[/bold]\n\n"
                "[bold yellow]Troubleshooting tips:[/bold yellow]\n"
                "- Double-check your arguments and the command structure\n"
                "- Try 'swarms help' for command details and examples\n"
                "- If the issue persists, please report it at https://github.com/OpenAgentsInc/swarms/issues\n\n"
                f"[dim]Traceback:[/dim]\n{traceback.format_exc()}"
            )
            return
    except Exception as error:
        formatter.print_panel(
            f"Critical error detected: {error}\n\n"
            "Your command could not be processed due to the above error.\n"
            "👉 Please review your arguments, environment settings, and try again.\n"
            "For more information, run 'swarms help' or visit the documentation:\n"
            "https://docs.swarms.world/en/latest/swarms/cli/cli_reference/\n\n"
            f"[dim]Traceback:[/dim]\n{traceback.format_exc()}",
            title="Fatal Error",
            style="red",
        )
        raise error


if __name__ == "__main__":
    main()
