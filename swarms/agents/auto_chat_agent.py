from swarms.structs.agent import Agent
from swarms.utils.formatter import formatter
from typing import Optional

EXIT_COMMANDS = {"exit", "quit", "bye", "q"}


def auto_chat_agent(
    name: str = "Swarms Agent",
    description: str = "A Swarms agent that can chat with the user.",
    system_prompt: Optional[str] = None,
    model_name: str = "gpt-4.1",
    task: Optional[str] = None,
) -> Agent:
    """
    Create an auto chat agent.

    Runs the autonomous loop on each task, then prompts for the next task
    when done. Continues until the user types an exit command.
    """
    temperature = 1.0

    agent = Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=system_prompt,
        model_name=model_name,
        dynamic_context_window=True,
        dynamic_temperature_enabled=True,
        max_loops="auto",
        temperature=temperature,
    )

    current_task = task

    while True:
        if not current_task:
            formatter.console.print()
            current_task = formatter.console.input(
                "[bold cyan]You[/bold cyan] [bold green]❯[/bold green] "
            ).strip()

        if not current_task or current_task.lower() in EXIT_COMMANDS:
            formatter.console.print(
                "[yellow]Exiting chat session.[/yellow]"
            )
            break

        agent.run(task=current_task)
        current_task = None  # reset so next iteration prompts again

    return agent
