import os
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Define custom system prompts for QR code generation
QR_CODE_AGENT_1_SYS_PROMPT = """
You are a Python coding expert. Your task is to write a Python script to generate a QR code for the link: https://lu.ma/jjc1b2bo. The code should save the QR code as an image file.
"""

QR_CODE_AGENT_2_SYS_PROMPT = """
You are a Python coding expert. Your task is to write a Python script to generate a QR code for the link: https://github.com/The-Swarm-Corporation/Cookbook. The code should save the QR code as an image file.
"""

# Example usage:
api_key = os.getenv("OPENAI_API_KEY")

# Model
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize your agents for QR code generation
agents = [
    Agent(
        agent_name="QR-Code-Generator-Agent-Luma",
        system_prompt=QR_CODE_AGENT_1_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="qr_code_agent_luma.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
    Agent(
        agent_name="QR-Code-Generator-Agent-Cookbook",
        system_prompt=QR_CODE_AGENT_2_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="qr_code_agent_cookbook.json",
        user_name="swarms_corp",
        retry_attempts=1,
    ),
]

# Create a Swarm with the list of agents
swarm = SpreadSheetSwarm(
    name="QR-Code-Generation-Swarm",
    description="A swarm that generates Python scripts to create QR codes for specific links.",
    agents=agents,
    autosave_on=True,
    save_file_path="qr_code_generation_results.csv",
    run_all_agents=False,
    max_loops=1,
)

# Run the swarm
swarm.run(
    task="Generate Python scripts to create QR codes for the provided links and save them as image files."
)
