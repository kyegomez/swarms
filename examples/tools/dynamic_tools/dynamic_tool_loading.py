"""
Dynamic tools with an autonomous agent.

Tool schemas are re-sent on every request, so a big tool set is paid for on
every call. With `dynamic_tools=True` the agent starts with almost none of
them: it gets `tool_search`, finds what it needs, and the tools it loads
become callable on its next turn.

    python3 dynamic_tools_example.py

Needs a provider key (OPENAI_API_KEY here), read from .env or the environment.
"""

import json

from dotenv import load_dotenv

from swarms import Agent

load_dotenv()


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: 21C, clear skies"


def convert_currency(amount: float, source: str, target: str) -> str:
    """Convert an amount of money from one currency to another.

    Args:
        amount: The amount to convert.
        source: Source currency code, e.g. USD.
        target: Target currency code, e.g. EUR.
    """
    return f"{amount} {source} = {amount * 0.92:.2f} {target}"


agent = Agent(
    agent_name="TravelAgent",
    model_name="gpt-5.4",
    max_loops="auto",
    tools=[get_weather, convert_currency],
    dynamic_tools=True,  # <- defer every tool behind tool_search
    print_on=False,
    reasoning_effort=None,
)

# Before the run only `tool_search` is offered for the user's tools. Once the
# autonomous loop starts it adds its own tools to the catalog too, keeping just
# the control tools (create_plan, subtask_done, complete_task, ...) always on.
print(
    f"exposed before the run: {[t['function']['name'] for t in agent.tools_list_dictionary]}"
)

result = agent.run(
    "What is the weather in Tokyo, and what is 100 USD in EUR?"
)

exposed = [t["function"]["name"] for t in agent.tools_list_dictionary]
loader = agent.tool_loader


print(
    f"Agent full history: {agent.short_memory.return_messages_as_dictionary()[0]}"
)

print(
    f"\nexposed after the run ({len(exposed)}): {', '.join(sorted(exposed))}"
)
print(f"loaded on demand:  {', '.join(loader.loaded_names)}")
print(
    f"never loaded ({len(loader.deferred_names)}): {', '.join(loader.deferred_names)}"
)

# What this saved: an eager agent sends every schema on every request.
eager = json.dumps(
    [t.schema for t in loader._catalog.values()]
    + agent.tools_list_dictionary
)
actual = json.dumps(agent.tools_list_dictionary)
print(
    f"\nschema bytes per request - eager: {len(eager):,}  "
    f"dynamic: {len(actual):,}  "
    f"saved: {100 * (len(eager) - len(actual)) // len(eager)}%"
)

# `output_type` defaults to the whole conversation, so pull out the final
# summary rather than printing a truncated tail of the history.
print("\n--- answer ---")
print(str(result).split("Subtask Breakdown:")[0].strip()[-700:])
