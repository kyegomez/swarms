# Output Formats & Swarm Architectures

This guide explains the available output formats in `swarms` (see `swarms/utils/output_types.py`) and how the different swarm architectures affect outputs, streaming, and structured responses. It also includes short, copyable examples using common multi-agent architectures.

**Overview**

- Swarms exposes many output customization options to control the shape and type of an agent or swarm response. Output formats affect whether you receive raw text, structured dictionaries, a pydantic model, streaming chunks, or the final synthesized result.

**Available Output Formats (summary)**

The framework supports the following common formats (aliases shown):

- `list` — conversation as a list of message dicts. Use when you need full, ordered message history.
- `dict` / `dictionary` — conversation as a dict keyed by message metadata. Use for programmatic inspection.
- `string` / `str` / `all` — full conversation rendered as one string. Good for logging or human display.
- `final` / `last` — content of the final message only. Use when you only care about the last reply.
- `json` — conversation as a JSON string. Use when passing response to JSON consumers or APIs.
- `yaml` / `xml` — conversation serialized as YAML or XML strings for integration with tooling that requires those formats.
- `dict-all-except-first` / `str-all-except-first` — omit the very first message (often the system prompt) and return the rest. Useful when initial system instruction should be excluded.
- `basemodel` — returns a validated `pydantic.BaseModel` (or similar) instance when a schema is provided. Use for strict typed outputs.
- `dict-final` / `list-final` — final message returned as a `dict` or `list` respectively. Useful when the final message contains structured content.

When to use each format: pick structured types (`dict`, `basemodel`, `dict-final`) when downstream code must parse reliably. Use `final`/`string` for human-facing outputs or when you only need the synthesized answer.

**Swarm Architectures and Their Outputs**

- `SequentialWorkflow` (default output: `dict`) — agents execute in a fixed sequence, passing messages from one to the next. Outputs commonly include the entire message chain as a `dict` or `list` so you can inspect intermediate step results.
- `ConcurrentWorkflow` (default output: `dict-all-except-first`) — agents run in parallel (threaded). Designed for dashboards and streaming; returns combined results from parallel workers. Streaming callbacks often emit partial results as they arrive.
- `SwarmRouter` (default output: `dict-all-except-first`) — routes tasks to sub-swarms/agents based on rules. Useful for dynamic dispatch and multi-model routing; output contains per-route results.
- `AgentRearrange` (default output: `all`) — flexible custom flows and rearrangements; outputs depend on the configured flow and often include full histories.
- `MixtureOfAgents` (default output: `final`) — a multi-layer aggregator that synthesizes several agent outputs into a final answer. Use when you want an aggregator synthesis rather than full histories.

Note: other specialized architectures exist in the repo; the above are common patterns and their default output tendencies. Check the specific struct's source file for exact behavior.

**Streaming Capabilities & Callbacks**

- Many multi-agent workflows support streaming. Streaming typically emits partial tokens or chunked partial messages as workers generate them.
- Typical parameters you will pass to the run/execute method:
  - `stream=True` — enable streaming mode.
  - `on_stream=callable` — callback invoked for each streamed chunk. The callback signature commonly receives the chunk text and optional metadata.
- Streaming is especially useful for `ConcurrentWorkflow` where partial results arrive at different times and you want to show them in a dashboard or forward them to a client via websocket.

Streaming pattern example:

```py
def on_chunk(chunk, meta=None):
    # meta may include agent id, token index, or partial score
    print('STREAM:', chunk)

result = swarm.run('Summarize the dataset', stream=True, on_stream=on_chunk, output_format='json')
```

**Response Schemas & Structured Outputs**

- Use `basemodel` (or `dict-final`) when you want schema validation. Provide a `pydantic.BaseModel` subclass describing the expected fields; the agent/swarm will parse and validate the final result.
- `json` and `yaml` are good when integrating with external services; `dict`/`list` are best for internal programmatic use.

Pydantic schema example:

```py
from pydantic import BaseModel

class SummarySchema(BaseModel):
    summary: str
    confidence: float

# If the API supports a `schema` or `response_schema` argument:
resp = agent.run('Summarize X', output_format='basemodel', schema=SummarySchema)
print(resp.summary, resp.confidence)
```

If your agent or swarm doesn't have a dedicated `schema` param, you can still request `json` or `dict-final` and validate manually with Pydantic:

```py
raw = swarm.run('Analyze data', output_format='json')
data = SummarySchema.parse_raw(raw)
```

**Practical Examples**

1. Sequential workflow (inspect intermediate results)

```py
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms import Agent

# Build simple agents (pseudocode)
a1 = Agent.from_config('researcher')
a2 = Agent.from_config('synthesizer')

sw = SequentialWorkflow([a1, a2])
# Get the full dict with intermediate messages
out = sw.run('Investigate X', output_format='dict')
print(out)  # contains messages from a1 and a2
```

2. Concurrent workflow with streaming (dashboard-friendly)

```py
from swarms.structs.concurrent_workflow import ConcurrentWorkflow

def on_stream(chunk, meta=None):
    print('partial:', meta.get('agent'), chunk)

sw = ConcurrentWorkflow([a1, a2, a3])
res = sw.run('Compare strategies', stream=True, on_stream=on_stream, output_format='json')
print('final aggregated:', res)
```

3. Router example (route to specialized agents)

```py
from swarms.structs.swarm_router import SwarmRouter

router = SwarmRouter(routes={'finance': finance_swarm, 'legal': legal_swarm})
# Default returns per-route dicts; choose `final` if you only want synthesised answer
resp = router.run('Provide compliance summary', output_format='dict-all-except-first')
```

**Tips & Best Practices**

- For programmatic integrations (APIs, tests, pipelines): prefer `dict`, `dict-final`, or `basemodel`.
- For user-facing UI or chat: prefer `final` or `string` to minimize noise.
- Use streaming (`stream=True`) with `on_stream` for responsive UIs and websockets; use `ConcurrentWorkflow` when you expect multiple parallel partial results.
- When aggregating many agents (MixtureOfAgents), request `final` unless you need intermediate opinions.

**Where to look in the codebase**

- Output type definitions: `swarms/utils/output_types.py`
- Sequential workflow implementation: `swarms/structs/sequential_workflow.py`
- Concurrent workflow implementation: `swarms/structs/concurrent_workflow.py`
- Router implementation: `swarms/structs/swarm_router.py`

---

If you'd like, I can also add a short runnable example under `examples/` that demonstrates `stream=True` + a small concurrent swarm and a README with commands to run it locally. Want me to add that example too?
