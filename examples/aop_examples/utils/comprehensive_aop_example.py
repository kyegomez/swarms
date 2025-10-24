#!/usr/bin/env python3


import time
import threading
from swarms import Agent
from swarms.structs.aop import AOP

# Create multiple agents for comprehensive testing
agent1 = Agent(
    agent_name="primary_agent",
    agent_description="Primary agent for comprehensive testing",
    system_prompt="You are the primary assistant for comprehensive testing.",
)

agent2 = Agent(
    agent_name="secondary_agent",
    agent_description="Secondary agent for comprehensive testing",
    system_prompt="You are the secondary assistant for comprehensive testing.",
)

agent3 = Agent(
    agent_name="monitoring_agent",
    agent_description="Agent for monitoring and status reporting",
    system_prompt="You are a monitoring assistant for system status.",
)

# Create AOP with all features enabled
aop = AOP(
    server_name="Comprehensive AOP Server",
    description="A comprehensive AOP server with all features enabled",
    agents=[agent1, agent2, agent3],
    port=8005,
    host="localhost",
    transport="streamable-http",
    verbose=True,
    traceback_enabled=True,
    queue_enabled=True,  # Enable queue-based execution
    max_workers_per_agent=2,
    max_queue_size_per_agent=100,
    processing_timeout=30,
    retry_delay=1.0,
    persistence=True,  # Enable persistence
    max_restart_attempts=10,
    restart_delay=5.0,
    network_monitoring=True,  # Enable network monitoring
    max_network_retries=8,
    network_retry_delay=3.0,
    network_timeout=15.0,
    log_level="INFO",
)

# Get comprehensive server information
server_info = aop.get_server_info()

# Get persistence status
persistence_status = aop.get_persistence_status()

# Get network status
aop.get_network_status()

# Get queue statistics
aop.get_queue_stats()

# List all agents
agent_list = aop.list_agents()

# Get detailed agent information
agent_info = {}
for agent_name in agent_list:
    agent_info[agent_name] = aop.get_agent_info(agent_name)


# Start comprehensive monitoring
def comprehensive_monitor(aop_instance):
    while True:
        try:
            # Monitor all aspects
            persistence_status = aop_instance.get_persistence_status()
            aop_instance.get_network_status()
            aop_instance.get_queue_stats()

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
    target=comprehensive_monitor, args=(aop,), daemon=True
)
monitor_thread.start()

# Demonstrate various management operations
# Enable persistence
aop.enable_persistence()

# Pause all queues
pause_results = aop.pause_all_queues()

# Resume all queues
resume_results = aop.resume_all_queues()

# Clear all queues
clear_results = aop.clear_all_queues()

# Reset restart count
aop.reset_restart_count()

# Reset network retry count
aop.reset_network_retry_count()

# Request shutdown
aop.request_shutdown()

# Disable persistence
aop.disable_persistence()

# Run the comprehensive server
try:
    aop.run()
except KeyboardInterrupt:
    pass
except Exception:
    pass
finally:
    # Comprehensive cleanup
    aop.disable_persistence()
    aop.request_shutdown()

    # Pause all queues
    aop.pause_all_queues()

    # Clear all queues
    aop.clear_all_queues()
