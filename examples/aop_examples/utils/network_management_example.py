import time
import threading
from swarms import Agent
from swarms.structs.aop import AOP

# Create a simple agent
agent = Agent(
    agent_name="network_monitor_agent",
    agent_description="An agent for network monitoring demo",
    system_prompt="You are a helpful assistant for network monitoring.",
)

# Create AOP with comprehensive network monitoring
aop = AOP(
    server_name="Network Managed AOP Server",
    description="An AOP server with comprehensive network management",
    agents=[agent],
    port=8004,
    host="localhost",
    persistence=True,
    max_restart_attempts=5,
    restart_delay=3.0,
    network_monitoring=True,
    max_network_retries=10,
    network_retry_delay=2.0,
    network_timeout=5.0,
    verbose=True,
)

# Show initial configuration
server_name = aop.server_name
host = aop.host
port = aop.port
persistence = aop.persistence
network_monitoring = aop.network_monitoring
max_network_retries = aop.max_network_retries
network_timeout = aop.network_timeout


# Start monitoring in background
def monitor_network_status(aop_instance):
    while True:
        try:
            network_status = aop_instance.get_network_status()
            persistence_status = aop_instance.get_persistence_status()

            # Check if we should stop monitoring
            if (
                persistence_status["shutdown_requested"]
                and not persistence_status["persistence_enabled"]
            ):
                break

            time.sleep(5)  # Update every 5 seconds

        except Exception:
            time.sleep(5)


monitor_thread = threading.Thread(
    target=monitor_network_status, args=(aop,), daemon=True
)
monitor_thread.start()

# Run the server
try:
    aop.run()
except KeyboardInterrupt:
    pass
except Exception:
    pass
finally:
    # Clean shutdown
    aop.disable_persistence()
    aop.request_shutdown()
