from swarms import Agent
from swarms import llama3Hosted
from pydantic import BaseModel, Field
from weather_swarm.tools.tools import (
    request_metar_nearest,
    point_query,
    request_ndfd_basic,
    point_query_region,
    request_ndfd_hourly,
)


class WeatherRequest(BaseModel):
    """
    A class to represent the weather request.

    Attributes
    ----------
    query : str
        The user's query.
    """

    task: str = Field(..., title="The user's query")
    tool: str = Field(None, title="The tool to execute")


def current_temperature_retrieval_agent():
    return """
    ### Current Temperature Retrieval Agent

    **Prompt:**
    As a specialized weather data agent, your task is to provide the current temperature based on the user's location. Ensure accuracy and up-to-date information.

    **Goal:**
    Allow the user to request the current temperature for their location.

    **Required Inputs:**
    User's location (latitude and longitude).

    **API Example:**
    request_metar_nearest("38", "-96")
    """


def current_weather_description_agent():
    return """
    ### Current Weather Description Agent

    **Prompt:**
    As a specialized weather data agent, your task is to construct a narrative weather description based on the current conditions at the user's location.

    **Goal:**
    Have the LLM construct a narrative weather description based on current conditions.

    **Required Inputs:**
    User's location (latitude and longitude).

    **API Example:**
    request_metar_nearest("38", "-96")
    """


def rainfall_accumulation_agent():
    return """
    ### Rainfall Accumulation Agent

    **Prompt:**
    As a specialized weather data agent, your task is to provide the accumulated rainfall at the user's location for the last 24 hours.

    **Goal:**
    Allow the user to determine how much rain has accumulated at their location in the last 24 hours.

    **Required Inputs:**
    User's location (latitude and longitude).

    **API Example:**
    point_query('precip-totalaccum-24hr', 'Standard-Mercator', -86.6, 34.4)
    """


def cloud_coverage_forecast_agent():
    return """
    ### Cloud Coverage Forecast Agent

    **Prompt:**
    As a specialized weather data agent, your task is to provide the cloud coverage forecast for the user's location for the next day.

    **Goal:**
    Allow the user to determine cloud coverage for their location.

    **Required Inputs:**
    User's location (latitude and longitude).

    **API Example:**
    request_ndfd_basic(34.730301, -86.586098, forecast_time)
    """


def precipitation_forecast_agent():
    return """
    ### Precipitation Forecast Agent

    **Prompt:**
    As a specialized weather data agent, your task is to provide the precipitation forecast for the user's location for the next 6 hours.

    **Goal:**
    Allow the user to determine if precipitation will fall in the coming hours.

    **Required Inputs:**
    User's location (latitude and longitude).

    **API Example:**
    point_query('baron-hires-maxreflectivity-dbz-all', 'Mask1-Mercator', -86.6, 34.4)
    """


def maximum_temperature_forecast_agent():
    return """
    ### Maximum Temperature Forecast Agent

    **Prompt:**
    As a specialized weather data agent, your task is to provide the maximum forecasted temperature for the user's location for today.

    **Goal:**
    Allow the user to determine how hot or cold the air temperature will be.

    **Required Inputs:**
    User's location (latitude and longitude).

    **API Example:**
    request_ndfd_basic(34.730301, -86.586098, forecast_time)
    """


def wind_speed_forecast_agent():
    return """
    ### Wind Speed Forecast Agent

    **Prompt:**
    As a specialized weather data agent, your task is to provide the maximum wind speed forecast for the user's location for today.

    **Goal:**
    Allow the user to determine the maximum wind speed for that day.

    **Required Inputs:**
    User's location (latitude and longitude).

    **API Example:**
    point_query('baron-hires-windspeed-mph-10meter', 'Standard-Mercator', -86.6, 34.4)
    """


llm = llama3Hosted(
    max_tokens=1000,
    temperature=0.5,
)


# Define the agents with their specific prompts
temp_tracker = Agent(
    agent_name="TempTracker",
    system_prompt=current_temperature_retrieval_agent(),
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[request_metar_nearest],
)

weather_narrator = Agent(
    agent_name="WeatherNarrator",
    system_prompt=current_weather_description_agent(),
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[request_metar_nearest],
)

rain_gauge = Agent(
    agent_name="RainGauge",
    system_prompt=rainfall_accumulation_agent(),
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[point_query],
)

cloud_predictor = Agent(
    agent_name="CloudPredictor",
    system_prompt=cloud_coverage_forecast_agent(),
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[request_ndfd_basic],
)

rain_forecaster = Agent(
    agent_name="RainForecaster",
    system_prompt=precipitation_forecast_agent(),
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[point_query_region],
)

temp_forecaster = Agent(
    agent_name="TempForecaster",
    system_prompt=maximum_temperature_forecast_agent(),
    llm=llm,
    max_loops=1,
    verbose=True,
    output_type=dict,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    stopping_token="<DONE>",
    tools=[request_ndfd_hourly],
)

wind_watcher = Agent(
    agent_name="WindWatcher",
    system_prompt=wind_speed_forecast_agent(),
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[point_query_region],
)

# Create a list
agents = [
    temp_tracker,
    weather_narrator,
    rain_gauge,
    cloud_predictor,
    rain_forecaster,
    temp_forecaster,
    wind_watcher,
]

# # Create a hierarchical swarm
# swarm = HiearchicalSwarm(
#     name = "WeatherSwarm",
#     description="A swarm of weather agents",
#     agents=agents,
#     director =
# )
