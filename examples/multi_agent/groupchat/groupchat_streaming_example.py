from rich.console import Console
from rich.theme import Theme

from swarms import RESPOND_TOOL, Agent, GroupChat
from swarms.utils.formatter import formatter

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
})

console = Console(theme=custom_theme)


def streaming_callback(sender: str, content: str, is_final: bool):
    if is_final:
        console.print("\n[dim][[Stream complete]][/dim]")
        return
    if sender and content:
        style = "bold green" if sender == "User" else "bold blue"
        formatter.print_panel(
            content,
            title=f"[{style}]{sender}[/{style}]",
            border_style=style,
        )


a1 = Agent(
    agent_name="Researcher",
    system_prompt="You are a research-minded agent who values evidence.",
    model_name="gpt-4o-mini",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
    output_type="final",
)

a2 = Agent(
    agent_name="Skeptic",
    system_prompt="You push back on weak claims and ask sharp questions.",
    model_name="gpt-4.1",
    max_loops=1,
    tools_list_dictionary=[RESPOND_TOOL],
    persistent_memory=False,
    output_type="final",
)

a3 = Agent(
    agent_name="Builder",
    system_prompt="You turn ideas into concrete next steps.",
    model_name="gpt-4o-mini",
    max_loops=1,
    tools_list_dictionary=[RESPOND_TOOL],
    persistent_memory=False,
    output_type="final",
)

chat = GroupChat(
    agents=[a1, a2, a3],
    max_loops=4,
    threshold=0.6,
    output_type="dict",
    auto_equip=False,
)

console.print("\n[info]Starting GroupChat with streaming callback...[/info]\n")

result = chat.run(
    "Should we use vector databases or knowledge graphs for agent memory?",
    streaming_callback=streaming_callback,
)

console.print("\n[info]=== Final Result ===[/info]")
console.print(result)
