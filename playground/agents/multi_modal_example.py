import os
from dotenv import load_dotenv
from swarms import GPT4VisionAPI, Agent

# Load the environment variables
load_dotenv()


# Initialize the language model
llm = GPT4VisionAPI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    max_tokens=500,
)

# Initialize the task
task = (
    "Analyze this image of an assembly line and identify any issues such as"
    " misaligned parts, defects, or deviations from the standard assembly"
    " process. IF there is anything unsafe in the image, explain why it is"
    " unsafe and how it could be improved."
)
img = "assembly_line.jpg"

## Initialize the workflow
agent = Agent(
    agent_name="Multi-ModalAgent",
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=True,
    multi_modal=True,
)

# Run the workflow on a task
agent.run(task, img)
