#!/usr/bin/env python3

from swarms import Agent
from swarms.structs.aop import AOP

# Create a simple agent
agent = Agent(
    agent_name="persistence_agent",
    agent_description="An agent for persistence demo",
    system_prompt="You are a helpful assistant.",
)

# Create AOP with persistence enabled
aop = AOP(
    server_name="Persistent AOP Server",
    description="A persistent AOP server that auto-restarts",
    agents=[agent],
    port=8001,
    persistence=True,  # Enable persistence
    max_restart_attempts=5,  # Allow up to 5 restarts
    restart_delay=3.0,  # Wait 3 seconds between restarts
    verbose=True,
)

# Show persistence status
status = aop.get_persistence_status()

# Run with persistence enabled
try:
    aop.run()
except KeyboardInterrupt:
    pass
except Exception:
    pass
