import os

from dotenv import load_dotenv

from swarms.models.gpt4_vision_api import GPT4VisionAPI
from swarms.prompts.multi_modal_autonomous_instruction_prompt import (
    MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
)
from swarms.structs import Agent

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

llm = GPT4VisionAPI(
    openai_api_key=api_key,
)

task = "What is the color of the object?"
img = "images/swarms.jpeg"

## Initialize the workflow
agent = Agent(
    llm=llm,
    max_loops="auto",
    sop=MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
    autosave=True,
    dashboard=True,
)

out = agent.run(task=task, img=img)
print(out)
