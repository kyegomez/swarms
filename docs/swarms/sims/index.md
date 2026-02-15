# Sims

The `sims` module contains large-scale simulation primitives for multi-agent experimentation.

## Available Exports

- `SenatorAssembly`
- `_create_senator_agents`

## SenatorAssembly Overview

`SenatorAssembly` provides a role-based simulation around US Senate participants with predefined policy perspectives and communication flows.

Use cases:

- Debate and consensus simulation
- Policy narrative stress-testing
- Multi-agent communication benchmarking

## Quickstart

```python
from swarms.sims import SenatorAssembly

assembly = SenatorAssembly(
    name="Budget Resolution Simulation",
)

result = assembly.run(
    "Debate a bipartisan budget package with constraints on defense and healthcare spending."
)

print(result)
```

## Performance Considerations

- Simulation scale can be high due to many participating agents.
- Tune model selection and token budgets before running full-size scenarios.
- Start with smaller test prompts to validate orchestration behavior.
