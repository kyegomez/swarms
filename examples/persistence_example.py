#!/usr/bin/env python3
"""
Example demonstrating the AOP persistence feature.

This example shows how to use the persistence mode to create a server
that automatically restarts when stopped, with failsafe protection.
"""

from swarms import Agent
from swarms.structs.aop import AOP


def main():
    """Demonstrate AOP persistence functionality."""

    # Create a simple agent
    agent = Agent(
        agent_name="example_agent",
        agent_description="An example agent for persistence demo",
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

    print("Starting persistent AOP server...")
    print("Press Ctrl+C to test the restart functionality")
    print("The server will restart automatically up to 5 times")
    print("After 5 failed restarts, it will shut down permanently")
    print()

    # Show persistence status
    status = aop.get_persistence_status()
    print(f"Persistence Status: {status}")
    print()

    try:
        # This will run with persistence enabled
        aop.run()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
        print(
            "In persistence mode, the server would normally restart"
        )
        print(
            "To disable persistence and shut down gracefully, call:"
        )
        print("  aop.disable_persistence()")
        print("  aop.request_shutdown()")


if __name__ == "__main__":
    main()
