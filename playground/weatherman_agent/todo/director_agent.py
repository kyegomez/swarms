from swarms import Agent
from swarms import llama3Hosted
from weather_swarm.prompts import GLOSSARY_PROMPTS
from pydantic import BaseModel, Field


# Define the schema for the HierarchicalSwarmRequest
# class HierarchicalSwarmRequest(BaseModel):
#     agents: Dict[str, Any] = Field(
#         ...,
#         description=(
#             "The name of the agents and their respective tasks to be"
#             " executed hierarchically."
#         ),
#         examples={
#             "Weather Director Agent": {
#                 "task": (
#                     "Are there any chances of rain today in"
#                     " Huntsville?"
#                 )
#             }
#         },
#     )


class HierarchicalSwarmRequest(BaseModel):
    task: str = Field(
        ...,
        description="The user's query.",
        examples={
            "What is the current temperature at my location?": {
                "task": "What is the current temperature at my location?"
            }
        },
    )
    agent_name: str = Field(
        ...,
        description="The name of the specialized agent.",
        examples={
            "Current Temperature Retrieval Agent": "Current Temperature Retrieval Agent"
        },
    )


# Define the schema for the HierarchicalSwarmResponse
def DIRECTOR_SYSTEM_PROMPT() -> str:
    return """**Prompt:**
    As a director master agent, your task is to communicate with the user, understand their weather-related queries, and delegate the appropriate tasks to specialized worker agents. Each worker agent is specialized in retrieving a specific type of weather data. Your role involves selecting the correct agent or a list of agents, giving them the necessary tasks, and compiling their responses to provide a comprehensive answer to the user.

    **Goal:**
    Efficiently manage and delegate tasks to specialized worker agents to gather the necessary weather data and provide a detailed, accurate response to the user.

    **Process:**
    1. **Receive User Query:**
    - Understand the user's question or request regarding weather data.

    2. **Identify Required Data:**
    - Determine the type(s) of weather data needed to answer the user's query.

    3. **Select Appropriate Agents:**
    - Choose the specialized agent(s) capable of retrieving the required data.

    4. **Delegate Tasks:**
    - Assign the relevant task to the selected agent(s) using the appropriate inputs.

    5. **Compile Responses:**
    - Gather and compile the data returned by the worker agents into a cohesive response.

    6. **Respond to User:**
    - Provide a detailed and accurate answer to the user based on the compiled data.

    **Worker Agents and Their Specializations:**
    1. **Current Temperature Retrieval Agent**
    - Task: Provide the current temperature based on the user's location.
    - Required Inputs: User's location (latitude and longitude).
    - API Example: `request_metar_nearest("38", "-96")`

    2. **Current Weather Description Agent**
    - Task: Construct a narrative weather description based on current conditions.
    - Required Inputs: User's location (latitude and longitude).
    - API Example: `request_metar_nearest("38", "-96")`

    3. **Rainfall Accumulation Agent**
    - Task: Provide the accumulated rainfall at the user's location for the last 24 hours.
    - Required Inputs: User's location (latitude and longitude).
    - API Example: `point_query('precip-totalaccum-24hr', 'Standard-Mercator', -86.6, 34.4)`

    4. **Cloud Coverage Forecast Agent**
    - Task: Provide the cloud coverage forecast for the user's location for the next day.
    - Required Inputs: User's location (latitude and longitude).
    - API Example: `request_ndfd_basic(34.730301, -86.586098, forecast_time)`

    5. **Precipitation Forecast Agent**
    - Task: Provide the precipitation forecast for the user's location for the next 6 hours.
    - Required Inputs: User's location (latitude and longitude).
    - API Example: `point_query('baron-hires-maxreflectivity-dbz-all', 'Mask1-Mercator', -86.6, 34.4)`

    6. **Maximum Temperature Forecast Agent**
    - Task: Provide the maximum forecasted temperature for the user's location for today.
    - Required Inputs: User's location (latitude and longitude).
    - API Example: `request_ndfd_basic(34.730301, -86.586098, forecast_time)`

    7. **Wind Speed Forecast Agent**
    - Task: Provide the maximum wind speed forecast for the user's location for today.
    - Required Inputs: User's location (latitude and longitude).
    - API Example: `point_query('baron-hires-windspeed-mph-10meter', 'Standard-Mercator', -86.6, 34.4)`

    **Example Workflow:**
    1. **User Query:**
    - "What is the current temperature and will it rain in the next 6 hours at my location?"

    2. **Identify Required Data:**
    - Current temperature and precipitation forecast.

    3. **Select Appropriate Agents:**
    - Current Temperature Retrieval Agent
    - Precipitation Forecast Agent

    4. **Delegate Tasks:**
    - Current Temperature Retrieval Agent: `request_metar_nearest("38", "-96")`
    - Precipitation Forecast Agent: `point_query('baron-hires-maxreflectivity-dbz-all', 'Mask1-Mercator', -86.6, 34.4)`

    5. **Compile Responses:**
    - Gather responses from both agents.

    6. **Respond to User:**
    - "The current temperature at your location is X degrees. There is/is not expected to be precipitation in the next 6 hours."

    By following this structured approach, you can efficiently manage user queries and provide accurate, detailed weather information.
    """


# Define the schema for the HierarchicalSwarmResponse
def DIRECTOR_SCHEMA() -> str:
    return """

    {
    "type": "object",
    "properties": {
        "task_id": {
        "type": "string",
        "description": "Unique identifier for the task"
        },
        "user_query": {
        "type": "string",
        "description": "The query provided by the user"
        },
        "agents": {
        "type": "array",
        "description": "List of agents to handle the query",
        "items": {
            "type": "object",
            "properties": {
            "agent_name": {
                "type": "string",
                "description": "Name of the specialized agent"
            },
            "task": {
                "type": "string",
                "description": "Task description for the agent"
            },
            },
            "required": ["agent_name", "task"]
        }
        }
    },
    "required": ["task_id", "user_query", "agents"]
    }

    """


def DIRECTOR_AGENT_CALLING_FEW_SHOT() -> str:
    return """
    
    {
    "task_id": "1",
    "user_query": "What is the current temperature at my location?",
    "agents": [
        {
        "agent_name": "Current Temperature Retrieval Agent",
        "task": "Provide the current temperature based on the user's location.",
        }
    ]
    }
    
    
    ########## "What is the current temperature and will it rain in the next 6 hours at my location? #########
    
    {
    "task_id": "2",
    "user_query": "What is the current temperature and will it rain in the next 6 hours at my location?",
    "agents": [
        {
        "agent_name": "Current Temperature Retrieval Agent",
        "task": "Provide the current temperature based on the user's location.",
        },
        {
        "agent_name": "Precipitation Forecast Agent",
        "task": "Provide the precipitation forecast for the user's location for the next 6 hours.",
        }
    ]
    }
    
    ########### END OF EXAMPLES ###########
    
    ############# Example 3: Maximum Temperature and Wind Speed Forecast #########
    {
    "task_id": "3",
    "user_query": "What is the maximum temperature and wind speed forecast for today at my location?",
    "agents": [
        {
        "agent_name": "Maximum Temperature Forecast Agent",
        "task": "Provide the maximum forecasted temperature for the user's location for today.",
        },
        {
        "agent_name": "Wind Speed Forecast Agent",
        "task": "Provide the maximum wind speed forecast for the user's location for today.",
        }
    ]
    }
    
    
    ############ End of Example 3 ############
    
    ############ Example 4: Rainfall Accumulation and Cloud Coverage Forecast #########
    {
    "task_id": "4",
    "user_query": "How much rain fell at my location in the last 24 hours and what is the cloud coverage forecast for tomorrow?",
    "agents": [
        {
        "agent_name": "Rainfall Accumulation Agent",
        "task": "Provide the accumulated rainfall at the user's location for the last 24 hours.",
        },
        {
        "agent_name": "Cloud Coverage Forecast Agent",
        "task": "Provide the cloud coverage forecast for the user's location for the next day.",
        }
    ]
    }
    
    ############ End of Example 4 ############

    """


# [C]reate a new agent
agent = Agent(
    agent_name="Weather Director Agent",
    system_prompt=DIRECTOR_SYSTEM_PROMPT(),
    sop_list=[
        GLOSSARY_PROMPTS,
        DIRECTOR_SCHEMA(),
        DIRECTOR_AGENT_CALLING_FEW_SHOT(),
    ],
    # sop=list_tool_schemas_json,
    llm=llama3Hosted(max_tokens=1000),
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    # interactive=True,
    verbose=True,
    # Set the output type to the tool schema which is a BaseModel
    output_type=str,  # or dict, or str
    metadata_output_type="json",
    # List of schemas that the agent can handle
    function_calling_format_type="OpenAI",
    function_calling_type="json",  # or soon yaml
    # return_history=True,
)

# Run the agent to generate the person's information
generated_data = agent.run(
    "Are there any chances of rain today in Huntsville?"
)

# Print the generated data
print(f"Generated data: {generated_data}")
