# FairySwarm Examples

Examples demonstrating the FairySwarm multi-agent coordination system, inspired by tldraw's "fairies" feature.

## Examples

| File | Description |
|------|-------------|
| `basic_fairy_swarm.py` | Simple landing page wireframe task |
| `selected_fairies.py` | Run with subset of fairies (orchestrator election) |
| `custom_tools.py` | Add custom tools like color palette generator |
| `standalone_tools.py` | Use canvas/todo tools without full swarm |
| `multi_page_wireframe.py` | Complex multi-page coordination task |
| `analytical_research.py` | Research-heavy dashboard design |
| `custom_fairy.py` | Create custom fairy agents |

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="your-key"

# Run an example
python fairy_swarm_examples/basic_fairy_swarm.py
```

## Key Concepts

1. **Orchestrator Fairy** - Plans and delegates tasks
2. **Worker Fairies** - Creative, Operational, Analytical, Harmonizer
3. **Shared Todo List** - Coordination between agents
4. **Canvas Tools** - Functions for manipulating shared state
5. **Context Refresh** - Mid-work updates when needed

