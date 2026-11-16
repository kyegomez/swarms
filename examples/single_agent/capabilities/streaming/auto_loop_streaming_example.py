from dotenv import load_dotenv

from swarms import Agent

load_dotenv()


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city.

    Returns:
        A short weather report.
    """
    reports = {
        "Paris": "18C and sunny",
        "London": "12C and raining",
        "Tokyo": "22C and humid",
    }
    return reports.get(city, "15C and cloudy")


agent = Agent(
    agent_name="Weather-Agent",
    agent_description="Compares the weather across cities",
    model_name="gpt-5.4-mini",
    max_loops="auto",
    tools=[get_weather],
    persistent_memory=False,
    print_on=False,
)

# Every part of the run streams live: the plan, each tool call and its arguments, and the final summary.
for token in agent.run_stream(
    "Get the weather in Paris, London and Tokyo, then tell me which city is warmest."
):
    print(token, end="", flush=True)
print()
