#!/usr/bin/env python3
"""
Example demonstrating AOP persistence management methods.

This example shows how to control persistence mode at runtime,
including enabling/disabling persistence and monitoring status.
"""

import time
import threading
from swarms import Agent
from swarms.structs.aop import AOP


def monitor_persistence(aop_instance):
    """Monitor persistence status in a separate thread."""
    while True:
        status = aop_instance.get_persistence_status()
        print("\n[Monitor] Persistence Status:")
        print(f"  - Enabled: {status['persistence_enabled']}")
        print(
            f"  - Shutdown Requested: {status['shutdown_requested']}"
        )
        print(f"  - Restart Count: {status['restart_count']}")
        print(
            f"  - Remaining Restarts: {status['remaining_restarts']}"
        )
        print(
            f"  - Max Restart Attempts: {status['max_restart_attempts']}"
        )
        print(f"  - Restart Delay: {status['restart_delay']}s")

        if status["shutdown_requested"]:
            break

        time.sleep(10)  # Check every 10 seconds


def main():
    """Demonstrate AOP persistence management."""

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

    print("AOP Persistence Management Demo")
    print("=" * 40)
    print()

    # Show initial status
    print("Initial persistence status:")
    status = aop.get_persistence_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    # Start monitoring in background
    monitor_thread = threading.Thread(
        target=monitor_persistence, args=(aop,), daemon=True
    )
    monitor_thread.start()

    print("Available commands:")
    print("  'enable' - Enable persistence mode")
    print("  'disable' - Disable persistence mode")
    print("  'shutdown' - Request graceful shutdown")
    print("  'reset' - Reset restart counter")
    print("  'status' - Show current status")
    print("  'start' - Start the server")
    print("  'quit' - Exit the program")
    print()

    try:
        while True:
            command = input("Enter command: ").strip().lower()

            if command == "enable":
                aop.enable_persistence()
                print("Persistence enabled!")

            elif command == "disable":
                aop.disable_persistence()
                print("Persistence disabled!")

            elif command == "shutdown":
                aop.request_shutdown()
                print("Shutdown requested!")

            elif command == "reset":
                aop.reset_restart_count()
                print("Restart counter reset!")

            elif command == "status":
                status = aop.get_persistence_status()
                print("Current status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")

            elif command == "start":
                print(
                    "Starting server... (Press Ctrl+C to test restart)"
                )
                try:
                    aop.run()
                except KeyboardInterrupt:
                    print("Server interrupted!")

            elif command == "quit":
                print("Exiting...")
                break

            else:
                print(
                    "Unknown command. Try: enable, disable, shutdown, reset, status, start, quit"
                )

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean shutdown
        aop.disable_persistence()
        aop.request_shutdown()


if __name__ == "__main__":
    main()
