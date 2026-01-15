# Output Formats & Swarm Architectures

This guide documents the output formats supported by the framework and explains how different swarm architectures produce and stream outputs. It includes usage patterns and concise `agent.py` examples.

## Output formats (supported values)

The framework exposes the following `OutputType` values (see `swarms/utils/output_types.py`):

- `list` — conversation as a list of message dicts. Use when you need full message metadata.
- `dict` / `dictionary` — keyed conversation object. Useful for programmatic access.
- `string` / `str` / `all` — flattened conversation as a string. Good for human-readable logs or prompts.
- `final` / `last` — content of the final message only. Use for single-answer workflows.
- `json` — JSON string of the conversation. Use for APIs that expect JSON payloads.
- `yaml` — YAML string.
- `xml` — XML string.
- `dict-all-except-first` — dictionary containing all messages except the first (system prompt), helpful when ignoring initial context.
- `str-all-except-first` — string of all messages except the first.
- `basemodel` — return as a Pydantic `BaseModel`/schema instance when structured outputs are required.
- `dict-final` — final message returned as a dict.
- `list-final` — final message returned as a list.

When to use each format
- Use `list`/`dict` for internal processing, pipelines, or when preserving role/metadata is important.
- Use `string`/`final` for quick human consumption or downstream prompt feeding.
- Use `basemodel`/`dict-final` when you use Pydantic schemas or structured outputs to validate/parsing.
- Use `json`/`yaml` for external integrations (APIs, config, storage).

## Streaming and callbacks

`Agent` supports two streaming modes (see `swarms/structs/agent.py` constructor args):
- `streaming_on` (simple formatted streaming panels)
- `stream` (detailed token-by-token streaming with metadata)

Both accept a `streaming_callback` callable to receive real-time tokens or fragments. Example callback signature:

```py
def my_stream_callback(token: str):
    # token arrives incrementally; append to buffer or display
    print(token, end="", flush=True)
```

Example usage with `agent.py`:

```py
from swarms.structs.agent import Agent

def stream_cb(token: str):
    print(token, end="", flush=True)

agent = Agent(
    llm="gpt-4o-mini",
    streaming_on=True,
    stream=True,
    streaming_callback=stream_cb,
    output_type="final",
)

resp = agent.run("Summarize the following document...")
print("\nFinal response (per output_type):", resp)
```

Notes on streaming:
- When `stream` is enabled you will receive token-level updates.
- For structured/basemodel outputs, prefer collecting the final result then parsing via the schema (streams may be interleaved metadata).

## Swarm Architectures and their outputs

Below is a concise mapping of common architectures, their default outputs, execution model, and streaming behavior.

- **SequentialWorkflow**
  - Default output: `dict` (full history/dict)
  - Execution: sequential agent-to-agent passing
  - Streaming: Typically non-parallel but can stream from individual agents; `streaming_callback` works per-agent
  - Usage: multi-step pipelines where one agent's output becomes the next agent's input

- **ConcurrentWorkflow**
  - Default output: `dict-all-except-first` (merged outputs from concurrent agents)
  - Execution: threaded / parallel
  - Streaming: supports dashboard-style streaming and per-agent streaming callbacks
  - Usage: parallel workers producing pieces of work to be aggregated

- **SwarmRouter**
  - Default output: `dict-all-except-first`
  - Execution: routed — decisions route inputs to different swarm types
  - Streaming: depends on routed swarm; router surfaces aggregated results
  - Usage: choose sub-swarms or specialized pipelines based on input

- **AgentRearrange / SwarmRearrange**
  - Default output: `all` / `list` (custom flows)
  - Execution: flexible custom sequencing
  - Streaming: custom; you control when agents emit stream tokens
  - Usage: dynamic reorder or conditional step flows

- **MixtureOfAgents (MoA)**
  - Default output: `final`
  - Execution: multi-layer agents + aggregator synthesis
  - Streaming: aggregator may stream synthesized answer after collecting sub-agent outputs
  - Usage: ensemble-style reasoning and synthesis

- **RoundRobin / MajorityVoting / DebateWithJudge / Council/LLM Council**
  - Default output: `dict` or `final` depending on the implementation
  - Execution: voting or debate patterns
  - Streaming: usually collect responses then stream final adjudicated result
  - Usage: reliability, self-consistency, redundancy

- **GraphWorkflow / BatchedGridWorkflow / GroupChat**
  - Default output: `dict` / `list`
  - Execution: graph-based dependencies or batched execution
  - Streaming: supports per-node/per-agent streaming; aggregator gathers node outputs

- **Hierarchical Swarms**
  - Default output: varies (often `dict` or `dict-all-except-first`)
  - Execution: hierarchical orchestration — parent swarms summarize/route to children
  - Streaming: parent can stream summaries while children stream detailed outputs

If you need architecture-specific output control, pass `output_type` to the top-level orchestration or to individual agents inside the swarm.

## Response schemas and structured outputs

- For validated structured responses use `output_type='basemodel'` and provide a `BaseModel` schema in your agent/tooling. This ensures results are parsed and validated.
- For function/tool-calling flows, tool outputs are returned as dicts or `BaseModel`s depending on `tool_schema` and `output_type`.

Example: Pydantic structured output pattern

```py
from pydantic import BaseModel
from swarms.structs.agent import Agent

class InvoiceSchema(BaseModel):
    invoice_id: str
    total: float
    vendor: str

agent = Agent(..., output_type="basemodel")
resp: InvoiceSchema = agent.run("Extract invoice info:")
print(resp.invoice_id, resp.total)
```

## Quick architecture examples

SequentialWorkflow (pattern):

```py
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.agent import Agent

agent_a = Agent(..., output_type="dict-final")
agent_b = Agent(..., output_type="final")

sw = SequentialWorkflow([agent_a, agent_b])
out = sw.run("Process this user request...")
print(out)
```

ConcurrentWorkflow (pattern):

```py
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.agent import Agent

workers = [Agent(..., output_type="dict") for _ in range(3)]
cw = ConcurrentWorkflow(workers)
result = cw.run("Analyze dataset A and return findings")
print(result)
```

## Where to find the code

- `swarms/utils/output_types.py` — list of supported formats
- `swarms/structs/agent.py` — agent API and streaming options
- Swarm implementation files (examples): `swarms/structs/sequential_workflow.py`, `swarms/structs/concurrent_workflow.py`, `swarms/structs/swarm_router.py`

## Examples script
See `examples/output_formats_architectures_example.py` for a compact example demonstrating `Agent` output types and small Sequential/Concurrent usage patterns.

---
If you'd like, I can also add a short runnable demo that spins up minimal mock agents so the `examples/` script can be executed in CI or locally without real LLM keys. Want that?
