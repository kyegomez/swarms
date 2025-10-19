from swarms import Agent
from swarms.structs.aop import AOP

# Create a simple agent
agent = Agent(
    agent_name="network_test_agent",
    agent_description="An agent for testing network error handling",
    system_prompt="You are a helpful assistant for network testing.",
)

# Create AOP with network monitoring enabled
aop = AOP(
    server_name="Network Resilient AOP Server",
    description="An AOP server with network error handling and retry logic",
    agents=[agent],
    port=8003,
    host="localhost",
    persistence=True,  # Enable persistence for automatic restart
    max_restart_attempts=3,
    restart_delay=2.0,
    network_monitoring=True,  # Enable network monitoring
    max_network_retries=5,  # Allow up to 5 network retries
    network_retry_delay=3.0,  # Wait 3 seconds between network retries
    network_timeout=10.0,  # 10 second network timeout
    verbose=True,
)

# Show initial network status
network_status = aop.get_network_status()

# Show persistence status
persistence_status = aop.get_persistence_status()

# Run with network monitoring enabled
try:
    aop.run()
except KeyboardInterrupt:
    pass
except Exception:
    pass
