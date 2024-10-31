import argparse
import os
import time

from rich.console import Console
from rich.text import Text
from swarms.cli.onboarding_process import OnboardingProcess
from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
import subprocess

console = Console()


ASCII_ART = """
  _________                                     
 /   _____/_  _  _______ _______  _____   ______
 \_____  \\ \/ \/ /\__  \\_  __ \/     \ /  ___/
 /        \\     /  / __ \|  | \/  Y Y  \\___ \ 
/_______  / \/\_/  (____  /__|  |__|_|  /____  >
        \/              \/            \/     \/ 

"""


# Function to display the ASCII art in red
def show_ascii_art():
    text = Text(ASCII_ART, style="bold cyan")
    console.print(text)


# Help command
def show_help():
    console.print(
        """
    [bold cyan]Swarms CLI - Help[/bold cyan]

    [bold magenta]Commands:[/bold magenta]
    [bold white]onboarding[/bold white]    : Starts the onboarding process
    [bold white]help[/bold white]          : Shows this help message
    [bold white]get-api-key[/bold white]   : Retrieves your API key from the platform
    [bold white]check-login[/bold white]   : Checks if you're logged in and starts the cache
    [bold white]read-docs[/bold white]     : Redirects you to swarms cloud documentation!
    [bold white]run-agents[/bold white]    : Run your Agents from your specified yaml file. Specify the yaml file with path the `--yaml-file` arg. Example: `--yaml-file agents.yaml`
    [bold white]generate-prompt[/bold white]    : Generate a prompt through automated prompt engineering. Requires an OPENAI Key in your `.env` Example: --prompt "Generate a prompt for an agent to analyze legal docs"
    [bold white]auto-upgrade[/bold white]   : Automatically upgrades Swarms to the latest version
    [bold white]book-call[/bold white]     : Book a strategy session with our team to discuss your use case and get personalized guidance

    For more details, visit: https://docs.swarms.world
    """
    )

    # [bold white]add-agent[/bold white]   : Add an agent to the marketplace under your name. Must have a Dockerfile + your agent.yaml to publish. Learn more Here: https://docs.swarms.world/en/latest/swarms_cloud/vision/


# Fetch API key from platform
def get_api_key():
    console.print(
        "[bold yellow]Opening the API key retrieval page...[/bold yellow]"
    )
    # Simulating API key retrieval process by opening the website
    import webbrowser

    webbrowser.open("https://swarms.world/platform/api-keys")
    time.sleep(2)
    console.print(
        "[bold green]Your API key is available on the dashboard.[/bold green]"
    )


# Redirect to docs
def redirect_to_docs():
    console.print(
        "[bold yellow]Opening the Docs page...[/bold yellow]"
    )
    # Simulating API key retrieval process by opening the website
    import webbrowser

    webbrowser.open("https://docs.swarms.world")
    time.sleep(2)


# Redirect to docs
def redirect_to_call():
    console.print(
        "[bold yellow]Opening the Call page...[/bold yellow]"
    )
    # Simulating API key retrieval process by opening the website
    import webbrowser

    webbrowser.open("https://cal.com/swarms/swarms-strategy-session")
    time.sleep(2)


# Check and start cache (login system simulation)
def check_login():
    cache_file = "cache.txt"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_content = f.read()
        if cache_content == "logged_in":
            console.print(
                "[bold green]You are already logged in.[/bold green]"
            )
        else:
            console.print(
                "[bold red]You are not logged in.[/bold red]"
            )
    else:
        console.print("[bold yellow]Logging in...[/bold yellow]")
        time.sleep(2)
        with open(cache_file, "w") as f:
            f.write("logged_in")
        console.print("[bold green]Login successful![/bold green]")


def check_and_upgrade_version():
    console.print(
        "[bold yellow]Checking for Swarms updates...[/bold yellow]"
    )
    try:
        # Check for updates using pip
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=freeze"],
            capture_output=True,
            text=True,
        )
        outdated_packages = result.stdout.splitlines()

        # Check if Swarms is outdated
        for package in outdated_packages:
            if package.startswith("swarms=="):
                console.print(
                    "[bold magenta]New version available! Upgrading...[/bold magenta]"
                )
                subprocess.run(
                    ["pip", "install", "--upgrade", "swarms"],
                    check=True,
                )
                console.print(
                    "[bold green]Swarms upgraded successfully![/bold green]"
                )
                return

        console.print(
            "[bold green]Swarms is up-to-date.[/bold green]"
        )
    except Exception as e:
        console.print(
            f"[bold red]Error checking for updates: {e}[/bold red]"
        )


# Main CLI handler
def main():
    parser = argparse.ArgumentParser(description="Swarms Cloud CLI")

    # Adding arguments for different commands
    parser.add_argument(
        "command",
        choices=[
            "onboarding",
            "help",
            "get-api-key",
            "check-login",
            "run-agents",
            "generate-prompt",  # Added new command for generating prompts
            "auto-upgrade",  # Added new command for auto-upgrade,
            "book-call",
        ],
        help="Command to run",
    )
    parser.add_argument(
        "--yaml-file",
        type=str,
        default="agents.yaml",
        help="Specify the YAML file for running agents",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Specify the task for generating a prompt",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=1,
        help="Specify the number of loops for generating a prompt",
    )
    parser.add_argument(
        "--autosave",
        action="store_true",
        help="Enable autosave for the prompt generator",
    )
    parser.add_argument(
        "--save-to-yaml",
        action="store_true",
        help="Save the generated prompt to a YAML file",
    )

    args = parser.parse_args()

    show_ascii_art()

    # Determine which command to run
    if args.command == "onboarding":
        OnboardingProcess().run()
    elif args.command == "help":
        show_help()
    elif args.command == "get-api-key":
        get_api_key()
    elif args.command == "check-login":
        check_login()
    elif args.command == "run-agents":
        create_agents_from_yaml(
            yaml_file=args.yaml_file, return_type="tasks"
        )
    # elif args.command == "generate-prompt":
    #     if (
    #         args.prompt
    #     ):  # Corrected from args.prompt_task to args.prompt
    #         generate_prompt(
    #             num_loops=args.num_loops,
    #             autosave=args.autosave,
    #             save_to_yaml=args.save_to_yaml,
    #             prompt=args.prompt,  # Corrected from args.prompt_task to args.prompt
    #         )
    #     else:
    #         console.print(
    #             "[bold red]Please specify a task for generating a prompt using '--prompt'.[/bold red]"
    #         )
    elif args.command == "auto-upgrade":
        check_and_upgrade_version()
    elif args.command == "book-call":
        redirect_to_call()
    else:
        console.print(
            "[bold red]Unknown command! Type 'help' for usage.[/bold red]"
        )


if __name__ == "__main__":
    main()
