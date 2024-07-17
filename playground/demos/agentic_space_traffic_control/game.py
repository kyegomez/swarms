from swarms import (
    Agent,
    llama3Hosted,
    AgentRearrange,
)
from playground.demos.agentic_space_traffic_control.prompts import (
    WEATHER_ANALYST_SYSTEM_PROMPT,
    SPACE_TRAFFIC_CONTROLLER_SYS_PROMPT,
)
from tools import (
    fetch_weather_data,
)
from swarms.tools import get_openai_function_schema_from_func


def prep_weather_tool_prompt(city: str = "Melbourne, Fl") -> str:
    out = get_openai_function_schema_from_func(
        fetch_weather_data,
        name="Fetch Weather Data by City",
        description="Fetch near real-time weather data for a city using wttr.in. Provide the name of the city (e.g., 'Austin, Tx') and state, as input.",
    )
    return out


# Purpose = To generate weather information for the user and send API requests to the Baron Weather API
agent = Agent(
    agent_name="Weather Analyst Agent",
    system_prompt=WEATHER_ANALYST_SYSTEM_PROMPT,
    llm=llama3Hosted(),
    max_loops=1,
    # autosave=True,
    dashboard=False,
    verbose=True,
    # sop=list_base_models_json,
    # sop_list=[
    #     prep_weather_tool_prompt
    # ],  # Set the output type to the tool schema which is a BaseModel
    # output_type=str,  # or dict, or str
    # metadata_output_type="json",
    # # List of schemas that the agent can handle
    # function_calling_format_type="OpenAI",
    # function_calling_type="json",  # or soon yaml
    # sop=fetch_weather_data,
)


# Purpose = To manage the trajectories and communication of spacecraft
agent2 = Agent(
    agent_name="Space Traffic Controller Agent",
    system_prompt=SPACE_TRAFFIC_CONTROLLER_SYS_PROMPT,
    # sop=list_base_models_json,
    llm=llama3Hosted(),
    max_loops=1,
    # autosave=True,
    dashboard=False,
    verbose=True,
    # Set the output type to the tool schema which is a BaseModel
    # output_type=str,  # or dict, or str
    # metadata_output_type="json",
    # # List of schemas that the agent can handle
    # function_calling_format_type="OpenAI",
    # function_calling_type="json",  # or soon yaml
)

# Rearrange
flow = AgentRearrange(
    agents=[agent, agent2],
    flow="Weather Analyst Agent -> Space Traffic Controller Agent",
    max_loops=3,
)
# Run the flow
flow.run(
    "We're preparing for a launch in Cape canveral, let's begin the launch process, whats the weather like?"
)
