import asyncio
import os
from typing import List

import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from swarms import Agent, Anthropic, GPT4o, GPT4VisionAPI, OpenAIChat
from swarms.utils.loguru_logger import logger

from swarms_cloud.schema.cog_vlm_schemas import (
    ChatCompletionResponse,
    UsageInfo,
)


# Define the input model using Pydantic
class AgentInput(BaseModel):
    agent_name: str = "Swarm Agent"
    system_prompt: str = None
    agent_description: str = None
    model_name: str = "OpenAIChat"
    max_loops: int = 1
    autosave: bool = False
    dynamic_temperature_enabled: bool = False
    dashboard: bool = False
    verbose: bool = False
    streaming_on: bool = True
    saved_state_path: str = None
    sop: str = None
    sop_list: List[str] = None
    user_name: str = "User"
    retry_attempts: int = 3
    context_length: int = 8192
    task: str = None


# Define the input model using Pydantic
class AgentOutput(BaseModel):
    agent: AgentInput
    completions: ChatCompletionResponse


async def count_tokens(
    text: str,
):
    try:
        # Get the encoding for the specific model
        encoding = tiktoken.get_encoding("gpt-4o")

        # Encode the text
        tokens = encoding.encode(text)

        # Count the tokens
        token_count = len(tokens)

        return token_count
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def model_router(model_name: str):
    """
    Function to switch to the specified model.

    Parameters:
    - model_name (str): The name of the model to switch to.

    Returns:
    - None

    Raises:
    - None

    """
    # Logic to switch to the specified model
    if model_name == "OpenAIChat":
        # Switch to OpenAIChat model
        llm = OpenAIChat()
    elif model_name == "GPT4o":
        # Switch to GPT4o model
        llm = GPT4o(openai_api_key=os.getenv("OPENAI_API_KEY"))
    elif model_name == "GPT4VisionAPI":
        # Switch to GPT4VisionAPI model
        llm = GPT4VisionAPI()
    elif model_name == "Anthropic":
        # Switch to Anthropic model
        llm = Anthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        # Invalid model name
        pass

    return llm


# Create a FastAPI app
app = FastAPI(debug=True)

# Load the middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.get("/v1/models", response_model=ModelList)
# async def list_models():
#     """
#     An endpoint to list available models. It returns a list of model cards.
#     This is useful for clients to query and understand what models are available for use.
#     """
#     model_card = ModelCard(
#         id="cogvlm-chat-17b"
#     )  # can be replaced by your model id like cogagent-chat-18b
#     return ModelList(data=[model_card])


@app.post("v1/agent/completions", response_model=AgentOutput)
async def agent_completions(agent_input: AgentInput):
    try:
        logger.info(f"Received request: {agent_input}")
        llm = model_router(agent_input.model_name)

        agent = Agent(
            agent_name=agent_input.agent_name,
            system_prompt=agent_input.system_prompt,
            agent_description=agent_input.agent_description,
            llm=llm,
            max_loops=agent_input.max_loops,
            autosave=agent_input.autosave,
            dynamic_temperature_enabled=agent_input.dynamic_temperature_enabled,
            dashboard=agent_input.dashboard,
            verbose=agent_input.verbose,
            streaming_on=agent_input.streaming_on,
            saved_state_path=agent_input.saved_state_path,
            sop=agent_input.sop,
            sop_list=agent_input.sop_list,
            user_name=agent_input.user_name,
            retry_attempts=agent_input.retry_attempts,
            context_length=agent_input.context_length,
        )

        # Run the agent
        logger.info(f"Running agent with task: {agent_input.task}")
        completions = await agent.run(agent_input.task)

        logger.info(f"Completions: {completions}")
        all_input_tokens, output_tokens = await asyncio.gather(
            count_tokens(agent.short_memory.return_history_as_string()),
            count_tokens(completions),
        )

        logger.info(f"Token counts: {all_input_tokens}, {output_tokens}")

        out = AgentOutput(
            agent=agent_input,
            completions=ChatCompletionResponse(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": agent_input.agent_name,
                            "content": completions,
                            "name": None,
                        },
                    }
                ],
                stream_choices=None,
                usage_info=UsageInfo(
                    prompt_tokens=all_input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=all_input_tokens + output_tokens,
                ),
            ),
        )

        return out.json()

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(
#         app, host="0.0.0.0", port=8000, use_colors=True, log_level="info"
#     )
