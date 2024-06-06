import time
import uuid

from fastapi import FastAPI, HTTPException

from swarms import Agent, OpenAIChat
from swarms.schemas.assistants_api import (
    AssistantRequest,
    AssistantResponse,
)

# Create an instance of the FastAPI application
app = FastAPI(debug=True, title="Assistant API", version="1.0")

# In-memory store for assistants
assistants_db = {}


# Health check endpoint
@app.get("/v1/health")
def health():
    return {"status": "healthy"}


# Create an agent endpoint
@app.post("/v1/agents")
def create_agent(request: AssistantRequest):
    try:
        # Example initialization, in practice, you'd pass in more parameters
        agent = Agent(
            agent_name=request.name,
            agent_description=request.description,
            system_prompt=request.instructions,
            llm=OpenAIChat(),
            max_loops="auto",
            autosave=True,
            verbose=True,
            # long_term_memory=memory,
            stopping_condition="finish",
            temperature=request.temperature,
            # output_type="json_object"
        )

        # Simulating running a task
        task = ("What are the symptoms of COVID-19?",)
        out = agent.run(task)

        return {
            "status": "Agent created and task run successfully",
            "output": out,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Create an assistant endpoint
@app.post("/v1/assistants", response_model=AssistantResponse)
def create_assistant(request: AssistantRequest):
    assistant_id = str(uuid.uuid4())
    assistant_data = request.dict()
    assistant_data.update(
        {
            "id": assistant_id,
            "object": "assistant",
            "created_at": int(time.time()),
        }
    )
    assistants_db[assistant_id] = assistant_data
    return AssistantResponse(**assistant_data)


# Get assistant by ID endpoint
@app.get("/v1/assistants/{assistant_id}", response_model=AssistantResponse)
def get_assistant(assistant_id: str):
    assistant = assistants_db.get(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return AssistantResponse(**assistant)
