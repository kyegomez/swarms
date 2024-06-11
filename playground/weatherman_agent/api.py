import os
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from swarms import Agent, OpenAIChat
from swarms.utils.loguru_logger import logger

from weather_swarm.prompts import (
    FEW_SHORT_PROMPTS,
    GLOSSARY_PROMPTS,
    WEATHER_AGENT_SYSTEM_PROMPT,
)
from weather_swarm.tools.tools import (
    point_query,
    request_ndfd_basic,
    request_ndfd_hourly,
)

load_dotenv()

logger.info("Starting the API server..")
app = FastAPI(debug=True)

# Load the middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]


@app.get("/v1/health")
async def health_check():
    return {"status": "ok"}


@app.get("/v1/models")
async def get_models():
    return {"models": ["WeatherMan Agent"]}


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    if request.model != "WeatherMan Agent":
        raise HTTPException(status_code=400, detail="Model not found")

    # Initialize the WeatherMan Agent
    agent = Agent(
        agent_name="WeatherMan Agent",
        system_prompt=WEATHER_AGENT_SYSTEM_PROMPT,
        sop_list=[GLOSSARY_PROMPTS, FEW_SHORT_PROMPTS],
        llm=OpenAIChat(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ),
        max_loops=1,
        # dynamic_temperature_enabled=True,
        # verbose=True,
        output_type=str,
        metadata_output_type="json",
        function_calling_format_type="OpenAI",
        function_calling_type="json",
        tools=[point_query, request_ndfd_basic, request_ndfd_hourly],
    )

    # Response from the agent

    try:
        response = agent.run(request.prompt)
        return {
            "id": uuid.uuid4(),
            "object": "text_completion",
            "created": int(os.times().system),
            "model": agent.agent_name,
            "choices": [{"text": response}],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(request.prompt.split())
                + len(response.split()),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example of how to run the FastAPI app
def deploy_app(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


# Run the FastAPI app
if __name__ == "__main__":
    deploy_app()
