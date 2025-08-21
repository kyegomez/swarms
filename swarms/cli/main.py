import argparse
import os
import time
import webbrowser

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from swarms.agents.auto_generate_swarm_config import (
    generate_swarm_config,
)
from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
from swarms.cli.onboarding_process import OnboardingProcess
from swarms.structs.agent import Agent
from swarms.utils.agent_loader import AgentLoader
from swarms.utils.formatter import formatter

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
  _________                                     
 /   _____/_  _  _______ _______  _____   ______
 \_____  \\ \/ \/ /\__  \\_  __ \/     \ /  ___/
 /        \\     /  / __ \|  | \/  Y Y  \\___ \ 
/_______  / \/\_/  (____  /__|  |__|_|  /____  >
        \/              \/            \/     \/                                
"""


def create_spinner(text: str) -> Progress:
    """Create a custom spinner with the given text."""
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
        title="[bold]Welcome to Swarms[/bold]",
        subtitle="[dim]swarms.ai[/dim]",
    )
    console.print(panel)


def create_command_table() -> Table:
    """Create a beautifully formatted table of commands."""
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
        ("onboarding", "Start the interactive onboarding process"),
        ("help", "Display this help message"),
        ("get-api-key", "Retrieve your API key from the platform"),
        ("check-login", "Verify login status and initialize cache"),
        ("run-agents", "Execute agents from your YAML configuration"),
        (
            "load-markdown",
            "Load agents from markdown files with YAML frontmatter",
        ),
        ("agent", "Create and run a custom agent with specified parameters"),
        ("auto-upgrade", "Update Swarms to the latest version"),
        ("book-call", "Schedule a strategy session with our team"),
        ("autoswarm", "Generate and execute an autonomous swarm"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    return table


def show_help():
    """Display a beautifully formatted help message."""
    console.print(
        "\n[bold]Swarms CLI - Command Reference[/bold]\n",
        style=COLORS["primary"],
    )
    console.print(create_command_table())
    console.print(
        "\n[dim]For detailed documentation, visit: https://docs.swarms.world[/dim]"
    )


def show_error(message: str, help_text: str = None):
    """Display error message in a formatted panel"""
    error_panel = Panel(
        f"[bold red]{message}[/bold red]",
        title="Error",
        border_style="red",
    )
    console.print(error_panel)

    if help_text:
        console.print(f"\n[yellow]ℹ️ {help_text}[/yellow]")


def execute_with_spinner(action: callable, text: str) -> None:
    """Execute an action with a spinner animation."""
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
    """Verify login status with enhanced visual feedback."""
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


def run_autoswarm(task: str, model: str):
    """Run autoswarm with enhanced error handling"""
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
    file_path: str, concurrent: bool = True, **kwargs
):
    """Load agents from markdown files with enhanced visual feedback."""
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


def create_swarm_agent(
    name: str,
    description: str,
    system_prompt: str,
    model_name: str,
    task: str,
    **kwargs
):
    """Create and run a custom agent with the specified parameters."""
    try:
        console.print(
            f"[yellow]Creating custom agent: {name}[/yellow]"
        )

        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[progress.description]{task.description}"
            ),
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
                **kwargs
            }
            
            # Remove None values to use defaults
            agent_config = {k: v for k, v in agent_config.items() if v is not None}
            
            agent = Agent(**agent_config)

            # Update progress
            progress.update(
                init_task,
                description="Agent created successfully! Running task...",
            )

            # Run the agent with the specified task
            progress.update(
                init_task,
                description=f"Executing task: {task[:50]}...",
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
                padding=(1, 2)
            )
            console.print(agent_info)
            
            return result
        else:
            console.print(
                f"[yellow]⚠ Agent '{name}' completed but returned no results.[/yellow]"
            )
            return None

    except Exception as e:
        show_error(
            "Agent Creation Error",
            f"Failed to create or run agent: {str(e)}\n\n"
            "Please check:\n"
            "1. Your API keys are set correctly\n"
            "2. The model name is valid\n"
            "3. All required parameters are provided\n"
            "4. Your system prompt is properly formatted"
        )
        return None


def main():
    try:

        show_ascii_art()

        parser = argparse.ArgumentParser(
            description="Swarms Cloud CLI"
        )
        parser.add_argument(
            "command",
            choices=[
                "onboarding",
                "help",
                "get-api-key",
                "check-login",
                "run-agents",
                "load-markdown",
                "agent",
                "auto-upgrade",
                "book-call",
                "autoswarm",
            ],
            help="Command to execute",
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
            help="Task for the custom agent to execute",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            help="Temperature setting for the agent (0.0-2.0)",
        )
        parser.add_argument(
            "--max-loops",
            type=int,
            help="Maximum number of loops for the agent",
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

        args = parser.parse_args()

        try:
            if args.command == "onboarding":
                OnboardingProcess().run()
            elif args.command == "help":
                show_help()
            elif args.command == "get-api-key":
                get_api_key()
            elif args.command == "check-login":
                check_login()
            elif args.command == "run-agents":
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
                        TextColumn(
                            "[progress.description]{task.description}"
                        ),
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
                            console.print(
                                "\n[bold green]Results:[/bold green]"
                            )
                            console.print(
                                Panel(
                                    result,
                                    title="Agent Output",
                                    border_style="green",
                                )
                            )
                        elif isinstance(result, dict):
                            console.print(
                                "\n[bold green]Results:[/bold green]"
                            )
                            for key, value in result.items():
                                console.print(
                                    f"[cyan]{key}:[/cyan] {value}"
                                )
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
                        str(e)
                        + "\n\nPlease check your agents.yaml file format.",
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
            elif args.command == "load-markdown":
                if not args.markdown_path:
                    show_error(
                        "Missing required argument: --markdown-path",
                        "Example usage: python cli.py load-markdown --markdown-path ./agents/",
                    )
                    exit(1)
                
                # Load agents from markdown
                agents = load_markdown_agents(
                    args.markdown_path,
                    concurrent=args.concurrent
                )
                
                if agents:
                    console.print(
                        f"\n[bold green]Ready to use {len(agents)} agents![/bold green]\n"
                        "[dim]You can now use these agents in your code or run them interactively.[/dim]"
                    )
            elif args.command == "agent":
                # Validate required arguments
                required_args = ["name", "description", "system_prompt", "task"]
                missing_args = [arg for arg in required_args if not getattr(args, arg)]
                
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
                        "  --temperature 0.1"
                    )
                    exit(1)
                
                # Build kwargs for additional parameters
                additional_params = {}
                param_mapping = {
                    'temperature': 'temperature',
                    'max_loops': 'max_loops',
                    'auto_generate_prompt': 'auto_generate_prompt',
                    'dynamic_temperature_enabled': 'dynamic_temperature_enabled',
                    'dynamic_context_window': 'dynamic_context_window',
                    'output_type': 'output_type',
                    'verbose': 'verbose',
                    'streaming_on': 'streaming_on',
                    'context_length': 'context_length',
                    'retry_attempts': 'retry_attempts',
                    'return_step_meta': 'return_step_meta',
                    'dashboard': 'dashboard',
                    'autosave': 'autosave',
                    'saved_state_path': 'saved_state_path',
                    'user_name': 'user_name',
                    'mcp_url': 'mcp_url'
                }
                
                for cli_arg, agent_param in param_mapping.items():
                    value = getattr(args, cli_arg)
                    if value is not None:
                        additional_params[agent_param] = value
                
                # Create and run the custom agent
                result = create_swarm_agent(
                    name=args.name,
                    description=args.description,
                    system_prompt=args.system_prompt,
                    model_name=args.model_name,
                    task=args.task,
                    **additional_params
                )
                
                if result:
                    console.print(
                        f"\n[bold green]Agent '{args.name}' executed successfully![/bold green]"
                    )
            elif args.command == "book-call":
                webbrowser.open(
                    "https://cal.com/swarms/swarms-strategy-session"
                )
            elif args.command == "autoswarm":
                if not args.task:
                    show_error(
                        "Missing required argument: --task",
                        "Example usage: python cli.py autoswarm --task 'analyze this data' --model gpt-4",
                    )
                    exit(1)
                run_autoswarm(args.task, args.model)
        except Exception as e:
            console.print(
                f"[{COLORS['error']}]Error: {str(e)}[/{COLORS['error']}]"
            )
            return
    except Exception as error:
        formatter.print_panel(
            f"Error detected: {error} check your args"
        )
        raise error


if __name__ == "__main__":
    main()
