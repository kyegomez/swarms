import time
import threading
from swarms import Agent
from swarms.structs.aop import AOP

# Create a simple agent
agent = Agent(
    agent_name="management_agent",
    agent_description="An agent for persistence management demo",
    system_prompt="You are a helpful assistant for testing persistence.",
)

# Create AOP with persistence initially disabled
aop = AOP(
    server_name="Managed AOP Server",
    description="An AOP server with runtime persistence management",
    agents=[agent],
    port=8002,
    persistence=False,  # Start with persistence disabled
    max_restart_attempts=3,
    restart_delay=2.0,
    verbose=True,
)

# Show initial status
status = aop.get_persistence_status()


# Start monitoring in background
def monitor_persistence(aop_instance):
    while True:
        try:
            status = aop_instance.get_persistence_status()

            # Check if we should stop monitoring
            if (
                status["shutdown_requested"]
                and not status["persistence_enabled"]
            ):
                break

            time.sleep(10)  # Check every 10 seconds

        except Exception:
            time.sleep(10)


monitor_thread = threading.Thread(
    target=monitor_persistence, args=(aop,), daemon=True
)
monitor_thread.start()

# Demonstrate persistence management
# Enable persistence
aop.enable_persistence()

# Get updated status
updated_status = aop.get_persistence_status()

# Request shutdown
aop.request_shutdown()

# Disable persistence
aop.disable_persistence()

# Reset restart count
aop.reset_restart_count()

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
