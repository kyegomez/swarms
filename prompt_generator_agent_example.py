import os
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.prompts.prompt_generator_optimizer import (
    prompt_generator_sys_prompt,
)
from dotenv import load_dotenv
from swarms.agents.prompt_generator_agent import PromptGeneratorAgent

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    openai_api_key=api_key,
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=2000,
)

# Initialize the agent
agent = Agent(
    agent_name="Prompt-Optimizer",
    system_prompt=prompt_generator_sys_prompt.get_prompt(),
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="optimizer_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
    # output_type="json",
    output_type="string",
)


# Main Class
prompt_generator = PromptGeneratorAgent(agent)

# Run the agent
prompt_generator.run(
    "Generate an amazing prompt for analyzing healthcare insurance documents"
)
