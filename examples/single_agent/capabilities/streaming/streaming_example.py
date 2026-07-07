"""
FastAPI server that streams agent tokens in real time over HTTP.

Three routes — all stream tokens as plain text via StreamingResponse:

  POST /stream/multi       — multi-loop agent (max_loops=3) with a tool
  POST /stream/auto        — autonomous agent (max_loops="auto") with a tool
  POST /stream/sequential  — pipeline of agents, streamed in order

The /stream/sequential route uses SequentialWorkflow.arun_stream() which
streams tokens from each agent in turn — every agent's full output becomes
the next agent's input, same hand-off as the non-streaming run().

Run:
    pip install fastapi uvicorn
    export OPENAI_API_KEY=sk-...
    uvicorn streaming_example:app --reload

Test (the -N flag disables curl buffering so tokens appear as they arrive):
    curl -N -X POST http://localhost:8000/stream/multi \\
        -H 'content-type: application/json' \\
        -d '{"task": "Use add to compute 17 + 25, then state the result."}'

    curl -N -X POST http://localhost:8000/stream/auto \\
        -H 'content-type: application/json' \\
        -d '{"task": "Use add to compute 99 + 1, then briefly explain."}'

    curl -N -X POST http://localhost:8000/stream/sequential \\
        -H 'content-type: application/json' \\
        -d '{"task": "Impact of rising interest rates on tech stocks."}'
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from swarms import Agent, SequentialWorkflow


def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


class TaskRequest(BaseModel):
    task: str


app = FastAPI()


def build_agent(max_loops):
    """Fresh agent per request so concurrent calls don't share conversation state."""
    return Agent(
        agent_name="Streamer",
        model_name="gpt-5.4-mini",
        max_loops=max_loops,
        tools=[add],
        persistent_memory=False,
        print_on=False,
    )


@app.post("/stream/multi")
async def stream_multi(req: TaskRequest):
    """Multi-loop streaming — tokens flow through tool-call and synthesis turns."""
    agent = build_agent(max_loops=3)

    async def token_stream():
        async for token in agent.arun_stream(req.task):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/stream/auto")
async def stream_auto(req: TaskRequest):
    """Autonomous streaming — tokens flow through plan, execute, and final summary."""
    agent = build_agent(max_loops="auto")

    async def token_stream():
        async for token in agent.arun_stream(req.task):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


def build_pipeline_agent(name: str, system_prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=system_prompt,
        model_name="gpt-5.4-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
    )


@app.post("/stream/sequential")
async def stream_sequential(req: TaskRequest):
    """Sequential pipeline streaming via SequentialWorkflow.arun_stream()."""
    workflow = SequentialWorkflow(
        agents=[
            build_pipeline_agent(
                "Researcher",
                "You research the topic and produce a concise factual brief.",
            ),
            build_pipeline_agent(
                "Analyst",
                "You take a research brief and produce sharp analytical insights.",
            ),
            build_pipeline_agent(
                "Writer",
                "You take analysis and produce a polished, reader-friendly summary.",
            ),
        ],
        autosave=False,
    )

    async def token_stream():
        async for token in workflow.arun_stream(req.task):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")
