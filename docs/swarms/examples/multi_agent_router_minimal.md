# MultiAgentRouter Minimal Example

This example shows how to route a task to the most suitable agent using `MultiAgentRouter`.

```python
from swarms import Agent, MultiAgentRouter

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

router = MultiAgentRouter(agents=agents)

result = router.route_task("Write a function that adds two numbers")
```

View the source on [GitHub](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/mar/multi_agent_router_minimal.py).
