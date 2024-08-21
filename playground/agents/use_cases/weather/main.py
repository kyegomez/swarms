import os
import asyncio
from swarms.models.popular_llms import OpenAIChatLLM
from weather_agent_example import WeatherAgent

# Set the OpenAI environment to use vLLM
api_key = os.getenv("OPENAI_API_KEY") or "EMPTY" # for vllm
api_base = os.getenv("OPENAI_API_BASE") or "http://localhost:8000/v1" # for vllm
weather_api_key= "af6ef989b5c50a91ca068cc00df125b7",  # Replace with your weather API key

# Create an instance of the OpenAIChat class
llm = OpenAIChatLLM(
    base_url=api_base, 
    api_key=api_key, 
    model="NousResearch/Meta-Llama-3-8B-Instruct", 
    temperature=0, 
    streaming=False
)

agent = WeatherAgent(
        weather_api_key=weather_api_key,
        city_name="Nepean, Ontario", 
        agent_name="Weather Reporting Agent",
        system_prompt="You are an exciting and professional weather reporting agent.  Given a city you will summarize the weather every hour.",
        llm=llm,
        max_loops=3,
        autosave=True,
        dynamic_temperature_enabled=True,
        dashboard=False,
        verbose=True,
        streaming_on=False,
        saved_state_path="weather_agent_state.json",
        user_name="RAH@EntangleIT.com",
        retry_attempts=3,
        context_length=200000,
    )

agent.run("Summarize the following weather_data JSON for the average human.  Translate the UNIX system time ('dt' in the JSON) to UTC.  Note the temperature is listed in Kelvin in the JSON, so please translate Kelvin to Farheinheit and Celcius.  Base your report only on the JSON and output the details in Markdown. \n")