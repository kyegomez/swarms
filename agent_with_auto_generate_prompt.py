import os
from swarms import Agent
from swarm_models import OpenAIChat

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Initialize the model for OpenAI Chat
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)

# Initialize the agent with automated prompt engineering enabled
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=None,  # System prompt is dynamically generated
    agent_description=None,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=False,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="Human:",
    return_step_meta=False,
    output_type="string",
    streaming_on=False,
    auto_generate_prompt=True,  # Enable automated prompt engineering
)

# Run the agent with a task description and specify the device
agent.run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria",
    ## Will design a system prompt based on the task if description and system prompt are None
    device="cpu",
)

# Print the dynamically generated system prompt
print(agent.system_prompt)
