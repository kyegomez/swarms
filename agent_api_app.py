from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from reflection_tuner import ReflectionTuner
from token_cache_and_adaptive_factory import TokenCache, AdaptiveAgentFactory
from swarms import Agent
from swarm_models import OpenAIChat
from swarms_memory import ChromaDB
import os

# Initialize FastAPI application
app = FastAPI()

# Define the token cache and model
token_cache = TokenCache(cache_duration_minutes=30)
model = OpenAIChat(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini", temperature=0.1)
memory = ChromaDB(metric="cosine", output_dir="api_memory")

# Agent creation factory
adaptive_factory = AdaptiveAgentFactory(model, token_cache)

# Input model for API request
class AgentRequest(BaseModel):
    agent_name: str
    system_prompt: str
    task: str
    reflection_steps: Optional[int] = 2

# Endpoint for creating and running an agent with Reflection-Tuning
@app.post("/run_agent")
async def run_agent(request: AgentRequest):
    # Create or retrieve the agent from cache
    agent = adaptive_factory.create_agent(
        agent_name=request.agent_name,
        system_prompt=request.system_prompt,
        task=request.task,
        memory=memory
    )
    
    # Initialize Reflection-Tuning
    reflection_tuner = ReflectionTuner(agent, reflection_steps=request.reflection_steps)
    response = reflection_tuner.reflect_and_tune(request.task)
    return {"response": response}

# Endpoint for running an existing agent without creating a new one
@app.post("/run_existing_agent")
async def run_existing_agent(agent_name: str, task: str):
    # Retrieve agent from cache
    agent_token = token_cache.get_token(agent_name)
    if not agent_token:
        raise HTTPException(status_code=404, detail="Agent not found in cache. Create a new agent instead.")
    
    # Run the agent
    response = agent_token.run(task)
    return {"response": response}

# Endpoint to clear cache for a specific agent
@app.delete("/clear_cache/{agent_name}")
async def clear_cache(agent_name: str):
    token_cache.token_cache.pop(agent_name, None)
    return {"detail": f"Cache for agent {agent_name} cleared."}
