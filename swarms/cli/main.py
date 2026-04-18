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
import getpass
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from litellm import traceback
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from swarms.agents.auto_chat_agent import auto_chat_agent
from swarms.agents.auto_generate_swarm_config import (
    generate_swarm_config,
)
from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
from swarms.cli.utils import (
    COLORS,
    SwarmCLIError,
    check_login,
    console,
    get_api_key,
    run_setup_check,
    show_ascii_art,
    show_error,
)
from swarms.env import load_swarms_env
from swarms.structs.agent import Agent
from swarms.structs.agent_loader import AgentLoader
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.llm_council import LLMCouncil
from swarms.utils.formatter import formatter

load_swarms_env()


def run_autoswarm(
    task: str,
    model: str,
    output_path: str = None,
    output_dir: str = None,
    no_run: bool = False,
) -> None:
    """
    Generate and execute an autonomous swarm configuration.

    Always writes a ready-to-run Python file to disk. By default also
    executes the swarm immediately. Use --no-run to skip execution.

    Args:
        task: The task description for the swarm to execute. Must be non-empty.
        model: The model name to use for swarm generation (e.g., 'gpt-4').
            Must be non-empty.
        output_path: Optional file path for the generated Python script.
        output_dir: Optional directory to create the generated file in.
        no_run: If True, only write the file without executing the swarm.

    Raises:
        SwarmCLIError: If task or model is empty or invalid.
        Exception: If swarm generation fails, displays formatted error message.
    """
    from swarms.agents.auto_generate_swarm_config import (
        write_autoswarm_file,
    )

    try:
        console.print(
            "[white]Initializing autoswarm configuration...[/white]"
        )

        # Validate inputs
        if not task or task.strip() == "":
            raise SwarmCLIError("Task cannot be empty")

        if not model or model.strip() == "":
            raise SwarmCLIError("Model name cannot be empty")

        # Step 1: Generate the config dict via LLM
        console.print(
            f"[white]Generating swarm for task: {task}[/white]"
        )
        config = generate_swarm_config(
            task=task,
            model_name=model,
        )

        if not config:
            raise SwarmCLIError(
                "Failed to generate swarm configuration"
            )

        agents_count = len(config.get("agents", []))
        console.print(
            "[white]✓ Swarm configuration generated[/white]"
        )
        console.print(
            f"[white]✓ Parsed {agents_count} agents from config[/white]"
        )

        # Step 2: Write the Python file (always)
        written_path = write_autoswarm_file(
            config=config,
            task=task,
            output_path=output_path,
            output_dir=output_dir,
        )
        console.print(f"[white]✓ Written to: {written_path}[/white]")

        # Step 3: Optionally run the swarm
        if not no_run:
            import yaml

            console.print("[white]Running swarm...[/white]")
            yaml_string = yaml.dump(config, default_flow_style=False)
            create_agents_from_yaml(
                yaml_string=yaml_string,
                return_type="run_swarm",
            )
            console.print(
                "[white]✓ Swarm executed successfully![/white]"
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
            f"[white]Loading agents from markdown: {file_path}[/white]"
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
                f"\n[bold white]✓ Successfully loaded {len(agents)} agents![/bold white]"
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
                "[white]⚠ No agents were loaded from the markdown files.[/white]"
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
    question_agent_model_name: str = "gpt-5.4",
    worker_model_name: str = "gpt-5.4",
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
            Defaults to "gpt-5.4".
        worker_model_name: Model name for specialized worker agents that process
            the questions. Defaults to "gpt-5.4".
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
        console.print("[white]🚀 Initializing HeavySwarm...[/white]")

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
                "\n[bold white]✓ HeavySwarm completed successfully![/bold white]"
            )

            # Display result in a panel
            result_panel = Panel(
                str(result),
                title="HeavySwarm Final Response",
                border_style="red",
                padding=(1, 2),
            )
            console.print(result_panel)

            return result
        else:
            console.print(
                "[white]⚠ HeavySwarm completed but returned no results.[/white]"
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
        console.print("[white]🏛️  Initializing LLM Council...[/white]")

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
                "\n[bold white]✓ LLM Council completed successfully![/bold white]"
            )

            # Display result in a panel
            result_panel = Panel(
                str(result),
                title="LLM Council Final Response",
                border_style="red",
                padding=(1, 2),
            )
            console.print(result_panel)

            return result
        else:
            console.print(
                "[white]⚠ LLM Council completed but returned no results.[/white]"
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
        console.print(f"[white]Creating custom agent: {name}[/white]")

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
                        f"\n[bold white]✓ Agent '{name}' completed the task successfully![/bold white]"
                    )

                    # Display agent info
                    agent_info = Panel(
                        f"[bold]Agent Name:[/bold] {name}\n"
                        f"[bold]Model:[/bold] {model_name}\n"
                        f"[bold]Task:[/bold] {task}\n"
                        f"[bold]Result:[/bold]\n{result}",
                        title="Agent Execution Results",
                        border_style="red",
                        padding=(1, 2),
                    )
                    console.print(agent_info)

                    return result
                else:
                    console.print(
                        f"[white]⚠ Agent '{name}' completed but returned no results.[/white]"
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
                    f"\n[bold white]✓ Agent '{name}' created successfully![/bold white]"
                )
                console.print(
                    "[white]ℹ️  No task provided. Agent is ready for interactive use.[/white]"
                )

                # Display agent info
                agent_info = Panel(
                    f"[bold]Agent Name:[/bold] {name}\n"
                    f"[bold]Model:[/bold] {model_name}\n"
                    f"[bold]Interactive Mode:[/bold] Enabled\n"
                    f"[bold]Status:[/bold] Ready for use",
                    title="Agent Created",
                    border_style="red",
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
                "Model for question generation agent (default: gpt-4o-mini)",
            ),
            (
                "--worker-model-name MODEL",
                "Model for worker agents (default: gpt-4o-mini)",
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
    command_choices = [
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
        default="gpt-5.4",
        help="Model name for question generation agent (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--worker-model-name",
        type=str,
        default="gpt-5.4",
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
        "[white]Note: 'swarms onboarding' now runs the same checks as 'swarms setup-check'[/white]"
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
            f"[white]Loading agents from {args.yaml_file}...[/white]"
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
                console.print("\n[bold white]Results:[/bold white]")
                console.print(
                    Panel(
                        result,
                        title="Agent Output",
                        border_style="red",
                    )
                )
            elif isinstance(result, dict):
                console.print("\n[bold white]Results:[/bold white]")
                for key, value in result.items():
                    console.print(f"[white]{key}:[/white] {value}")
            else:
                console.print(
                    "[white]✓ Agents completed their tasks successfully![/white]"
                )
        else:
            console.print(
                "[white]⚠ Agents completed but returned no results.[/white]"
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
            f"\n[bold white]Ready to use {len(agents)} agents![/bold white]\n"
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
            f"\n[bold white]Agent '{args.name}' executed successfully![/bold white]"
        )


def handle_autoswarm(args: argparse.Namespace) -> None:
    """
    Handle the autoswarm command.

    Generates and executes an autonomous swarm configuration based on the
    provided task and model. Writes a ready-to-run Python file to disk.

    Args:
        args: Parsed command line arguments containing:
            - task: Required task description for the swarm
            - model: Required model name for swarm generation
            - output: Optional output file path for the generated script
            - no_run: Optional flag to skip execution and only write the file

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
    run_autoswarm(
        task=args.task,
        model=args.model,
        output_path=getattr(args, "output", None),
        output_dir=getattr(args, "output_dir", None),
        no_run=getattr(args, "no_run", False),
    )


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
                (default: "gpt-5.4")
            - worker_model_name: Model for worker agents (default: "gpt-5.4")
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


def handle_chat(args: argparse.Namespace) -> Optional[Agent]:
    """
    Handle the chat command for interactive chat agent.

    Initializes and runs an interactive chat agent with optimized defaults
    for conversation. The agent is configured for interactive use with
    autonomous loops enabled (max_loops="auto").

    Args:
        args: Parsed command line arguments containing:
            - name: Optional agent name (default: "Swarms Agent")
            - description: Optional agent description
            - system_prompt: Optional custom system prompt
            - task: Optional initial task to start the conversation

    Returns:
        Optional[Agent]: The initialized chat agent instance, or None if
            initialization failed.

    Raises:
        Exception: If chat agent initialization fails. Errors are displayed
            with formatted messages and troubleshooting tips.

    Note:
        The chat agent is optimized for conversation with dynamic context
        window and temperature enabled, and uses autonomous loops for
        continuous interaction. If no initial task is provided, the agent
        will prompt for input interactively.
    """
    try:
        # Get parameters with defaults - handle None values
        name = getattr(args, "name", None) or "Swarms Agent"
        description = (
            getattr(args, "description", None)
            or "A Swarms agent that can chat with the user."
        )
        system_prompt = getattr(args, "system_prompt", None)
        model_name = getattr(args, "model_name", None) or "gpt-5.1"
        task = getattr(args, "task", None)

        console.print(
            f"\n[bold white]{name}[/bold white]  [dim white]({model_name})[/dim white]"
        )
        console.print(
            "[dim white]  Switch model: [/dim white][white]swarms chat --model-name gpt-5.4[/white]"
            "[dim white]  ·  other options: claude-opus-4-6, gpt-4o, gemini/gemini-2.0-flash[/dim white]\n"
        )

        # Create and run the chat agent (no progress spinner to avoid blocking input)
        result = auto_chat_agent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            model_name=model_name,
            task=task,
        )

        if result:
            console.print(
                "\n[bold white]✓ Chat session completed![/bold white]"
            )
            return result
        else:
            console.print("[white]⚠ Chat session ended.[/white]")
            return None

    except Exception as e:
        console.print(
            f"\n[bold red]Chat Agent Error:[/bold red] {str(e)}"
        )
        console.print(
            "[white]Please check:\n"
            "1. Your API keys are set correctly\n"
            "2. You have network connectivity\n"
            "3. All parameters are properly formatted[/white]"
        )
        return None


def handle_init(args: argparse.Namespace) -> None:
    """
    Handle the `swarms init` command — interactive project scaffolding wizard.

    Prompts the user for a project directory, workspace directory, and LLM
    provider API keys, then writes a .env file into the chosen directory.
    Finishes by validating the environment is ready.

    Args:
        args: Parsed command line arguments containing:
            - dir: Optional path for the new project (default: prompted)
    """
    from rich.rule import Rule

    console.print()
    console.print(
        Panel(
            "[bold white]Swarms Project Initialization Wizard[/bold white]\n"
            "[dim white]We'll create a .env file and workspace directory.[/dim white]",
            border_style="red",
            title="[bold red] 🚀 swarms init [/bold red]",
            title_align="left",
            padding=(0, 2),
        )
    )
    console.print()

    # ── Step 1: Project directory ────────────────────────────────────────────
    console.print(
        Rule(
            "[bold red]Step 1 — Project Directory[/bold red]",
            style="dim red",
        )
    )
    console.print()

    if getattr(args, "dir", None):
        project_dir = Path(args.dir).expanduser().resolve()
    else:
        default_dir = str(Path.cwd())
        console.print(
            "  [dim white]Where should the project files be written?[/dim white]\n"
            "  [dim white](Press Enter to use the current directory)[/dim white]"
        )
        raw = console.input(
            f"  [bold white]Project directory[/bold white] [dim white]\\[{rich_escape(default_dir)}][/dim white]: "
        ).strip()
        project_dir = (
            Path(raw).expanduser().resolve()
            if raw
            else Path(default_dir)
        )

    project_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"  [white]✓ Project directory:[/white] [bold white]{project_dir}[/bold white]\n"
    )

    # ── Step 2: Workspace directory ──────────────────────────────────────────
    console.print(
        Rule(
            "[bold red]Step 2 — Workspace Directory[/bold red]",
            style="dim red",
        )
    )
    console.print()
    console.print(
        "  [dim white]WORKSPACE_DIR is where agents read and write files.[/dim white]"
    )
    default_workspace = str(project_dir / "workspace")
    raw = console.input(
        f"  [bold white]Workspace directory[/bold white] [dim white]\\[{rich_escape(default_workspace)}][/dim white]: "
    ).strip()
    workspace_dir = (
        Path(raw).expanduser().resolve()
        if raw
        else Path(default_workspace)
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"  [white]✓ Workspace directory:[/white] [bold white]{workspace_dir}[/bold white]\n"
    )

    # ── Step 3: LLM provider API keys ────────────────────────────────────────
    console.print(
        Rule(
            "[bold red]Step 3 — LLM Provider API Keys[/bold red]",
            style="dim red",
        )
    )
    console.print()

    providers = [
        ("OPENAI_API_KEY", "OpenAI", "gpt-4o, gpt-5.x"),
        (
            "ANTHROPIC_API_KEY",
            "Anthropic",
            "claude-opus-4-6, claude-sonnet-4-6",
        ),
        ("GOOGLE_API_KEY", "Google", "gemini-2.0-flash, gemini-pro"),
        (
            "GROQ_API_KEY",
            "Groq",
            "llama-3.x, mixtral — fast inference",
        ),
        ("MISTRAL_API_KEY", "Mistral", "mistral-large, codestral"),
        (
            "TOGETHER_API_KEY",
            "Together AI",
            "llama, qwen — open weights",
        ),
        ("COHERE_API_KEY", "Cohere", "command-r-plus"),
        ("XAI_API_KEY", "xAI / Grok", "grok-4, grok-beta"),
        (
            "OPENROUTER_API_KEY",
            "OpenRouter",
            "any model via openrouter.ai",
        ),
    ]

    table = Table(
        show_header=True,
        header_style="bold red",
        border_style="dim red",
        padding=(0, 2),
        title="Available Providers",
    )
    table.add_column("#", style="bold white", width=4)
    table.add_column("Provider", style="bold white", width=14)
    table.add_column("Models / Notes", style="dim white")
    for idx, (_, name, note) in enumerate(providers, 1):
        table.add_row(str(idx), name, note)
    console.print(table)
    console.print()

    console.print(
        "  [dim white]Enter the numbers of the providers you want to configure,[/dim white]\n"
        "  [dim white]separated by commas. Type [bold white]all[/bold white][dim white] to configure every provider,[/dim white]\n"
        "  [dim white]or press [bold white]Enter[/bold white][dim white] to skip this step.[/dim white]"
    )
    selection_raw = (
        console.input(
            "  [bold white]Providers[/bold white] [dim white](e.g. 1,2 or all)[/dim white]: "
        )
        .strip()
        .lower()
    )

    selected_providers: List[tuple] = []
    if selection_raw == "all":
        selected_providers = providers
    elif selection_raw:
        selected_indices = {
            int(x.strip())
            for x in selection_raw.split(",")
            if x.strip().isdigit()
        }
        selected_providers = [
            p
            for i, p in enumerate(providers, 1)
            if i in selected_indices
        ]

    console.print()
    collected_keys: Dict[str, str] = {}

    for env_var, name, _ in selected_providers:
        # Check if already set in environment
        existing = os.environ.get(env_var, "")
        if existing:
            console.print(
                f"  [white]✓ {name}[/white] [dim white]({env_var} already set in environment — "
                "press Enter to keep, or type a new key)[/dim white]"
            )
        else:
            console.print(
                f"  [bold white]{name} API Key[/bold white] [dim white]({env_var})[/dim white]"
            )

        try:
            key = getpass.getpass(prompt="    Key (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""

        if key:
            collected_keys[env_var] = key
            console.print(f"  [white]  ✓ {env_var} saved[/white]")
        elif existing:
            console.print(
                f"  [dim white]  → keeping existing {env_var}[/dim white]"
            )
        else:
            console.print(
                f"  [dim white]  → {env_var} skipped[/dim white]"
            )
        console.print()

    # ── Step 4: Swarms API key (optional) ────────────────────────────────────
    console.print(
        Rule(
            "[bold red]Step 4 — Swarms API Key (optional)[/bold red]",
            style="dim red",
        )
    )
    console.print()
    console.print(
        "  [dim white]Required for [bold white]--marketplace-prompt-id[/bold white][dim white] and the Swarms platform.[/dim white]\n"
        "  [dim white]Get one at: [bold white]https://swarms.world/platform/api-keys[/bold white][dim white][/dim white]"
    )
    try:
        swarms_key = getpass.getpass(
            prompt="    SWARMS_API_KEY (hidden, Enter to skip): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        swarms_key = ""
    if swarms_key:
        collected_keys["SWARMS_API_KEY"] = swarms_key
        console.print("  [white]  ✓ SWARMS_API_KEY saved[/white]")
    else:
        console.print(
            "  [dim white]  → SWARMS_API_KEY skipped[/dim white]"
        )
    console.print()

    # ── Step 4: Write .env ───────────────────────────────────────────────────
    console.print(
        Rule(
            "[bold red]Step 4 — Writing Project Files[/bold red]",
            style="dim red",
        )
    )
    console.print()

    env_path = project_dir / ".env"
    env_lines = [
        "# Swarms environment — generated by `swarms init`",
        "# Add your API keys below. Never commit this file to version control.",
        "",
        f"WORKSPACE_DIR={workspace_dir}",
        "",
        "# ── LLM Provider Keys ───────────────────────────────────────────────",
    ]
    all_provider_vars = [p[0] for p in providers] + ["SWARMS_API_KEY"]
    for env_var in all_provider_vars:
        value = collected_keys.get(env_var, "")
        if value:
            env_lines.append(f"{env_var}={value}")
        else:
            env_lines.append(f"# {env_var}=")
    env_lines += [
        "",
        "# ── Optional Settings ───────────────────────────────────────────────",
        "# SWARMS_VERBOSE=false",
        "# SWARMS_LOG_LEVEL=INFO",
    ]

    env_path.write_text("\n".join(env_lines) + "\n")
    console.print(f"  [white]✓ .env[/white]               {env_path}")
    console.print(
        f"  [white]✓ workspace/[/white]         {workspace_dir}"
    )
    console.print()

    # ── Step 5: Validate ─────────────────────────────────────────────────────
    console.print(
        Rule(
            "[bold red]Step 5 — Validating Setup[/bold red]",
            style="dim red",
        )
    )
    console.print()

    issues: List[str] = []

    # Check at least one LLM key
    all_llm_vars = [p[0] for p in providers]
    active_llm_keys = [
        v
        for v in all_llm_vars
        if collected_keys.get(v) or os.environ.get(v)
    ]
    if active_llm_keys:
        console.print(
            f"  [white]✓ LLM keys configured:[/white] [dim white]{', '.join(active_llm_keys)}[/dim white]"
        )
    else:
        console.print(
            "  [bold red]✗ No LLM provider key set.[/bold red] "
            "[dim white]Add at least one key to .env before running agents.[/dim white]"
        )
        issues.append("No LLM provider API key configured.")

    # Check workspace dir
    if workspace_dir.exists():
        console.print(
            f"  [white]✓ Workspace directory exists:[/white] [dim white]{workspace_dir}[/dim white]"
        )
    else:
        console.print(
            f"  [bold red]✗ Workspace directory missing:[/bold red] {workspace_dir}"
        )
        issues.append(
            f"Workspace directory not found: {workspace_dir}"
        )

    # Check .env readable
    if env_path.exists() and env_path.stat().st_size > 0:
        console.print(
            "  [white]✓ .env file written successfully[/white]"
        )
    else:
        issues.append(".env file is missing or empty.")

    console.print()

    # ── Summary ──────────────────────────────────────────────────────────────
    if not issues:
        console.print(
            Panel(
                f"[bold white]Your environment is ready![/bold white]\n\n"
                f"  [dim white]cd[/dim white] [white]{project_dir}[/white]\n"
                f"  [dim white]then try:[/dim white]\n\n"
                f"  [bold white]swarms chat[/bold white]                            [dim white]# interactive agent[/dim white]\n"
                f"  [bold white]swarms heavy-swarm --task '...'[/bold white]        [dim white]# deep analysis[/dim white]\n"
                f"  [bold white]swarms setup-check[/bold white]                     [dim white]# verify environment[/dim white]",
                border_style="white",
                title="[bold white] ✓ Done [/bold white]",
                title_align="left",
                padding=(1, 2),
            )
        )
    else:
        issue_text = "\n".join(f"  • {i}" for i in issues)
        console.print(
            Panel(
                f"[bold white]Project created with warnings:[/bold white]\n\n"
                f"[bold red]{issue_text}[/bold red]\n\n"
                f"[dim white]Edit [bold white]{env_path}[/bold white][dim white] to add missing keys, then run "
                f"[bold white]swarms setup-check[/bold white][dim white] to verify.[/dim white]",
                border_style="red",
                title="[bold red] ⚠ Completed with issues [/bold red]",
                title_align="left",
                padding=(1, 2),
            )
        )


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
                f"[bold]{rich_escape(str(e))}[/bold]\n\n"
                "[bold white]Troubleshooting tips:[/bold white]\n"
                "- Double-check your arguments and the command structure\n"
                "- Try 'swarms help' for command details and examples\n"
                "- If the issue persists, please report it at https://github.com/OpenAgentsInc/swarms/issues\n\n"
                f"[dim]Traceback:[/dim]\n{rich_escape(traceback.format_exc())}"
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
