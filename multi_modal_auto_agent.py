from swarms.structs import Flow
from swarms.models.gpt4_vision_api import GPT4VisionAPI
from swarms.prompts.multi_modal_autonomous_instruction_prompt import (
    MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
)


llm = GPT4VisionAPI()

task = "What is the color of the object?"
img = "images/swarms.jpeg"

## Initialize the workflow
flow = Flow(
    llm=llm,
    sop=MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
    max_loops="auto",
)

flow.run(task=task, img=img)
