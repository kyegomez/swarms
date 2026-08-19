"""
Agent structured output (output_schema).

Pass a Pydantic model to the Agent and every LLM response is:
1. requested as JSON conforming to the model (via response_format),
2. validated with model_validate,
3. retried automatically when the schema does not match,
4. returned as a validated model instance from agent.run().

Requires an OPENAI_API_KEY (or any LiteLLM-supported provider key).
"""

from pydantic import BaseModel, Field

from swarms import Agent


class WeatherReport(BaseModel):
    """The exact shape the agent must return."""

    city: str = Field(description="The city the report is about")
    temperature_c: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="e.g. sunny, cloudy, rainy")


agent = Agent(
    agent_name="WeatherAgent",
    model_name="gpt-4o",
    max_loops=1,
    output_schema=WeatherReport,
)

result = agent.run("What is the weather in Paris today?")

# result is a validated WeatherReport instance
print(result.city, result.temperature_c, result.condition)
print(result.model_dump())