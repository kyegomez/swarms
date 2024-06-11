from dotenv import load_dotenv
from swarms import Agent, OpenAIChat

from weather_swarm.prompts import (
    FEW_SHORT_PROMPTS,
    GLOSSARY_PROMPTS,
    WEATHER_AGENT_SYSTEM_PROMPT,
)
from weather_swarm.tools.tools import (
    point_query,
    request_ndfd_basic,
    request_ndfd_hourly,
)

# Load the environment variables
load_dotenv()


# Purpose = To generate weather information for the user and send API requests to the Baron Weather API
agent = Agent(
    agent_name="WeatherMan Agent",
    system_prompt=WEATHER_AGENT_SYSTEM_PROMPT,
    sop_list=[GLOSSARY_PROMPTS, FEW_SHORT_PROMPTS],
    # sop=list_tool_schemas_json,
    llm=OpenAIChat(),
    max_loops=1,
    # interactive=True,
    dynamic_temperature_enabled=True,
    verbose=True,
    # Set the output type to the tool schema which is a BaseMode
    output_type=str,  # or dict, or str
    tools=[
        # request_metar_nearest,
        point_query,
        request_ndfd_basic,
        # point_query_region,
        request_ndfd_hourly,
    ],
    docs_folder="datasets",  # Add every document in the datasets folder
    metadata="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
)

# Run the agent to generate the person's information
# Run the agent to generate the person's information
output = agent.run("Are there any chances of rain today in Huntsville?")
# # Write the output to a new file
# with open('output.txt', 'w') as f:
#     f.write(str(output))