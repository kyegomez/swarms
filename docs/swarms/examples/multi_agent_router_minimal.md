# MultiAgentRouter Minimal Example

This example shows how to route a task to the most suitable agent using `SwarmRouter` with `swarm_type="MultiAgentRouter"`.

```python
from swarms import Agent
from swarms.structs.swarm_router import SwarmRouter

agents = [
    Agent(
        agent_name="Researcher",
        system_prompt="Answer questions briefly.",
        model_name="gpt-4o-mini",
    ),
    Agent(
        agent_name="Coder",
        system_prompt="Write small Python functions.",
        model_name="gpt-4o-mini",
    ),
]

router = SwarmRouter(
    name="multi-agent-router-demo",
    description="Routes tasks to the most suitable agent",
    agents=agents,
    swarm_type="MultiAgentRouter"
)

result = router.run("Write a function that adds two numbers")
print(result)
```

View the source on [GitHub](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/mar/multi_agent_router_minimal.py).