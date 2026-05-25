# FastAPI + Uvicorn

FastAPI is a straightforward way to expose a Swarms agent through an HTTP API. It works well for internal tools, product backends, prototypes, and services that need a predictable request-response interface.

## Install

```bash
pip install -U swarms fastapi uvicorn
```

Set the model provider credentials required by your agent:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Example API

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from swarms import Agent


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1)


class AgentResponse(BaseModel):
    result: str


agent = Agent(
    agent_name="API-Agent",
    agent_description="Handles API requests with a bounded agent loop.",
    model_name="gpt-4.1",
    max_loops=1,
)

app = FastAPI(title="Swarms Agent API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=AgentResponse)
def run_agent(request: AgentRequest) -> AgentResponse:
    result = agent.run(request.task)
    return AgentResponse(result=str(result))
```

Run the service locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then call it:

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Summarize the latest customer feedback themes."}'
```

## Deployment Notes

- Create the agent at startup instead of rebuilding it on every request.
- Keep `max_loops` low for synchronous APIs.
- Use background jobs for long-running workflows.
- Add authentication before exposing the API publicly.
- Log errors without printing secrets or full provider responses.
- Put provider keys, model names, and service settings in environment variables.
