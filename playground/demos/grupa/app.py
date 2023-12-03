import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from swarms.models import OpenAIChat
from swarms.prompts.code_interpreter import CODE_INTERPRETER
from swarms.structs import Agent


class AgentInput(BaseModel):
    feature: str
    codebase: str


app = FastAPI()

load_dotenv()

# Load the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the language agent
llm = OpenAIChat(
    model_name="gpt-4",
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=2000,
)

# Product Manager Agent init
product_manager_agent = Agent(
    llm=llm, max_loops=1, sop=CODE_INTERPRETER, autosave=True
)

# Initialize the agent with the language agent
feature_implementer_frontend = Agent(
    llm=llm, max_loops=1, sop=CODE_INTERPRETER, autosave=True
)

# Create another agent for a different task
feature_implementer_backend = Agent(
    llm=llm, max_loops=1, sop=CODE_INTERPRETER, autosave=True
)


# ##################### FastAPI #####################


def feature_codebase_product_agentprompt(
    feature: str, codebase: str
) -> str:
    prompt = (
        "Create an algorithmic pseudocode for an all-new feature:"
        f" {feature} based on this codebase: {codebase}"
    )
    return prompt


# @app.post("/agent/")
# async def run_feature_implementer_frontend(item: AgentInput):
#     agent1_out = feature_implementer_frontend.run(
#         f"Create the backend code for {item.feature} in markdown"
#         " based off of this algorithmic pseudocode:"
#         f" {product_manager_agent.run(feature_codebase_product_agentprompt(item.feature, item.codebase))} write"
#         f" the logic based on the following codebase: {item.codebase}"
#     )
#     return {"output": agent1_out}

def software_gpt(feature: str, codebase: str) -> str:
    agent1_out = feature_implementer_frontend.run(
        f"Create the backend code for {feature} in markdown"
        " based off of this algorithmic pseudocode:"
        f" {product_manager_agent.run(feature_codebase_product_agentprompt(feature, codebase))} write"
        f" the logic based on the following codebase: {codebase}"
    )
    print(agent1_out)