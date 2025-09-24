import os

from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

# Initialize the agent for quantitative trading using the Grok LLM API
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="openrouter/qwen/qwen3-vl-235b-a22b-instruct",  # Use the correct Grok model name
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    streaming_on=True,
    llm_base_url="https://openrouter.ai/api/v1",  # Grok API base URL
    llm_api_key=os.getenv(
        "OPENROUTER_API_KEY"
    ),  # Use the correct Grok API key environment variable
)

# Run the agent on the specified task
out = agent.run(
    task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
)

print(out)
