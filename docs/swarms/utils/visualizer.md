# Visualizer Utility

`SwarmVisualizationRich` is a live dashboard built with Rich that lets you watch agents work while monitoring system resources.

## Starting the Visualizer

```python
from swarms.utils.visualizer import SwarmVisualizationRich, SwarmMetadata
from swarms import Agent

# create agents
agent = Agent(name="DemoAgent")

metadata = SwarmMetadata(name="Demo Swarm", description="Example swarm")

viz = SwarmVisualizationRich(swarm_metadata=metadata, agents=[agent])

import asyncio
asyncio.run(viz.start())
```

The `start()` method runs an asynchronous event loop that continually refreshes the console layout.

## CPU and GPU Metrics

When `update_resources=True`, the visualizer displays current CPU core count, memory usage, and GPU utilization. GPU statistics rely on `pynvml`; if no GPU is found the panel shows `No GPU detected`.

- **CPU** – logical core count returned by `psutil.cpu_count()`.
- **Memory** – used/total RAM with percentage from `psutil.virtual_memory()`.
- **GPU** – per‑device memory consumption via NVIDIA's NVML.

## Embedding in Custom Workflows

You can stream agent output into the visualizer from any workflow:

```python
viz.log_agent_output(agent, "Task started...")
result = await some_coroutine()
viz.log_agent_output(agent, f"Result: {result}")
```

Combine this with your swarms or structured workflows to obtain a live view of progress and system health.
