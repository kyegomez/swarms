from swarms import Agent
from swarms import llama3Hosted
from weather_swarm.prompts import GLOSSARY_PROMPTS
from weather_swarm.prompts import (
    FEW_SHORT_PROMPTS,
    WEATHER_AGENT_SYSTEM_PROMPT,
)


# Purpose = To generate weather information for the user and send API requests to the Baron Weather API
agent = Agent(
    agent_name="WeatherMan Agent",
    system_prompt=WEATHER_AGENT_SYSTEM_PROMPT,
    sop_list=[GLOSSARY_PROMPTS, FEW_SHORT_PROMPTS],
    # sop=list_tool_schemas_json,
    llm=llama3Hosted(
        max_tokens=2000,
        temperature=0.1,
    ),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    interactive=True,
)

# Run the agent to generate the person's information
generated_data = agent.run(
    "Based on the current humidity in Huntsville, how frizzy will my"
    " hair get?"
)

# Print the generated data
# print(f"Generated data: {generated_data}")
