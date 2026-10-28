# Broadcast Examples

One agent speaks, many agents respond. The sender answers the task first, then every receiver reads the shared conversation and replies in turn. Receivers can be a flat list of agents or a list of lists.

- [broadcast_class_example.py](broadcast_class_example.py) - `Broadcast`, a reusable group with a synchronous `run()` (Kimi K3 via OpenRouter)
- [broadcast_function_example.py](broadcast_function_example.py) - `broadcast()`, the async functional form (GLM 5.3 via OpenRouter)

```python
import asyncio
from swarms import Agent, Broadcast, broadcast

group = Broadcast(sender=announcer, receivers=departments, output_type="dict")
history = group.run("Announce the change.")

# or, inside an event loop
history = asyncio.run(broadcast(announcer, departments, "Announce the change."))
```

Both return the conversation history in the format named by `output_type`. See `swarms/structs/broadcast.py`.

Each example names its model with a plain LiteLLM string; swap it for any provider you have a key for. OpenRouter models need `OPENROUTER_API_KEY`.
