"""Command handlers for the swarms CLI, one handle_ function per command."""

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import set_key
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from swarms.agents.auto_chat_agent import auto_chat_agent
from swarms.agents.auto_generate_swarm_config import (
    generate_swarm_config,
)
from swarms.cli.parser import DEFAULT_HEAVY_SWARM_MODEL
from swarms.cli.utils import (
    COLORS,
    SwarmCLIError,
    console,
    show_error,
)
from swarms.structs.agent import Agent
from swarms.structs.agent_loader import AgentLoader
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.llm_council import LLMCouncil


def _print_result(
    result: Any,
    label: str,
    title: str,
    body: Optional[str] = None,
    success: Optional[str] = None,
) -> Optional[Any]:
    """
    Print a command's result in a red panel and return it, or print a
    warning and return None when it is empty.

    ``body`` defaults to ``str(result)``, and ``success`` to
    "<label> completed successfully!".
    """
    if not result:
        console.print(
            f"[white]⚠ {label} completed but returned no results.[/white]"
        )
        return None
    success = success or f"{label} completed successfully!"
    console.print(f"\n[bold white]✓ {success}[/bold white]")
    console.print(
        Panel(
            str(result) if body is None else body,
            title=title,
            border_style="red",
            padding=(1, 2),
        )
    )
    return result


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

    if env_path.exists():
        for env_var, value in {
            "WORKSPACE_DIR": str(workspace_dir),
            **collected_keys,
        }.items():
            set_key(env_path, env_var, value, quote_mode="never")
    else:
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

    # Contextual next-step tip
    from swarms.cli.tips import render_tip

    console.print()
    console.print(render_tip(category="commands"))
    console.print()


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
            result = AgentLoader().load_agents_from_yaml(
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
        sys.exit(1)
    except ValueError as e:
        show_error(
            "Configuration Error",
            str(e) + "\n\nPlease check your agents.yaml file format.",
        )
        sys.exit(1)
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
        sys.exit(1)


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
            if no agents were loaded.

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

            agents = loader.load_agents_from_markdown(
                file_path, concurrent=concurrent, **kwargs
            )

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
        sys.exit(1)
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
        sys.exit(1)
    except Exception as e:
        show_error(
            "Execution Error",
            f"An unexpected error occurred: {str(e)}\n"
            "1. Check your markdown file format\n"
            "2. Verify your API keys are set\n"
            "3. Check network connectivity",
        )
        sys.exit(1)


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


def create_swarm_agent(
    name: str,
    description: str,
    system_prompt: str,
    model_name: str,
    task: str,
    **kwargs: Any,
) -> Optional[Any]:
    """
    Create a custom agent with the specified parameters and run it on a task.

    Args:
        name: The name of the agent. Used for identification and display.
        description: A description of the agent's purpose and capabilities.
        system_prompt: The system prompt that defines the agent's behavior
            and instructions.
        model_name: The LLM model to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
        task: The task for the agent to run.
        **kwargs: Additional keyword arguments passed to the Agent constructor.
            Common options include:
            - temperature: Float between 0.0 and 2.0
            - max_loops: Integer or "auto" for autonomous loops
            - interactive: Boolean to enable interactive mode
            - verbose: Boolean for verbose output
            - context_length: Integer for context window size
            - And other Agent configuration parameters

    Returns:
        The task's result, or None if the task returned no results.

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

                return _print_result(
                    result,
                    label=f"Agent '{name}'",
                    title="Agent Execution Results",
                    success=f"Agent '{name}' completed the task successfully!",
                    body=f"[bold]Agent Name:[/bold] {name}\n"
                    f"[bold]Model:[/bold] {model_name}\n"
                    f"[bold]Task:[/bold] {task}\n"
                    f"[bold]Result:[/bold]\n{result}",
                )

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
        sys.exit(1)


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
            - task: Task to run (required)
            - model_name: Model to use (default: "gpt-4")
            - interactive: Keep prompting after the first response (default: False)
            - temperature: Temperature setting (0.0-2.0)
            - max_loops: Maximum loops (integer or "auto")
            - And other optional agent configuration parameters

    Exits:
        Exits with code 1 if required arguments are missing or if max_loops
        value is invalid.

    Note:
        For an interactive agent without a task, use `swarms chat`. The function
        converts max_loops from a string to an int or "auto" as appropriate.
    """
    required_args = ["name", "description"]
    if not getattr(args, "marketplace_prompt_id", None):
        required_args.append("system_prompt")
    required_args.append("task")

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
            "Note: --task is required. For an interactive agent, use `swarms chat`.",
        )
        exit(1)

    # Build kwargs for additional parameters
    additional_params = {}
    param_mapping = {
        "interactive": "interactive",
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
            the session returned nothing.

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
        sys.exit(1)


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
            AgentLoader().load_agents_from_yaml(
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
        sys.exit(1)


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
            it returned no results.

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
            result = council.run(task=task)

            # Update progress on completion
            progress.update(
                init_task,
                description="Task completed!",
                completed=True,
            )

        return _print_result(
            result,
            label="LLM Council",
            title="LLM Council Final Response",
        )

    except Exception as e:
        show_error(
            "LLM Council Error",
            f"Failed to run LLM Council: {str(e)}\n\n"
            "Please check:\n"
            "1. Your API keys are set correctly\n"
            "2. You have network connectivity\n"
            "3. The task is properly formatted",
        )
        sys.exit(1)


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


def run_heavy_swarm(
    task: str,
    loops_per_agent: int = 1,
    question_agent_model_name: str = DEFAULT_HEAVY_SWARM_MODEL,
    worker_model_name: str = DEFAULT_HEAVY_SWARM_MODEL,
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
            it returned no results.

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

        return _print_result(
            result,
            label="HeavySwarm",
            title="HeavySwarm Final Response",
        )

    except Exception as e:
        show_error(
            "HeavySwarm Error",
            f"Failed to run HeavySwarm: {str(e)}\n\n"
            "Please check:\n"
            "1. Your API keys are set correctly\n"
            "2. You have network connectivity\n"
            "3. The task is properly formatted",
        )
        sys.exit(1)


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


def handle_tips(args: argparse.Namespace) -> None:
    """
    Display random tips & tricks for the Swarms CLI.

    Flags:
        --count N       Show N random tips (default: 1).
        --category CAT  Restrict to one category (e.g. agents, swarms, pro).
        --all           Show every tip in the (optionally filtered) pool.
    """
    from swarms.cli.tips import (
        TIP_CATEGORIES,
        list_categories,
        render_random_tips,
        render_tip,
    )

    category = getattr(args, "category", None)
    show_all = getattr(args, "all_tips", False)
    n = max(1, getattr(args, "count", 1) or 1)

    try:
        if show_all:
            categories = [category] if category else list_categories()
            for cat in categories:
                console.print(f"\n[bold red]── {cat} ──[/bold red]")
                for tip_body in TIP_CATEGORIES[cat]:
                    console.print(render_tip(body=tip_body))
            return

        for line in render_random_tips(n=n, category=category):
            console.print(line)
    except ValueError as e:
        show_error(
            "Unknown tip category",
            f"{e}\nRun 'swarms tips' with no --category to see a random tip from any pool.",
        )
        sys.exit(1)


def handle_models(args: argparse.Namespace) -> None:
    """
    Discover and inspect LLM models available via LiteLLM.

    Flags:
        --search PATTERN   Fuzzy-match against the model list.
        --info NAME        Show full metadata for one model.
        --provider NAME    Filter the list to one provider.

    With no flag, prints every model grouped by provider.
    """
    from swarms.cli.models import (
        list_models,
        search_models,
        show_model_info,
    )

    info_name = getattr(args, "info", None)
    search_pattern = getattr(args, "search", None)
    provider = getattr(args, "provider", None)

    if info_name:
        show_model_info(console, info_name)
        return
    if search_pattern:
        search_models(console, search_pattern)
        return
    list_models(console, provider=provider)
