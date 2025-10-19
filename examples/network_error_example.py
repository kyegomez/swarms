#!/usr/bin/env python3
"""
Example demonstrating the AOP network error handling feature.

This example shows how the AOP server handles network connection issues
with custom error messages and automatic retry logic.
"""

from swarms import Agent
from swarms.structs.aop import AOP


def main():
    """Demonstrate AOP network error handling functionality."""

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

    print("AOP Network Error Handling Demo")
    print("=" * 40)
    print()

    # Show initial network status
    print("Initial network status:")
    network_status = aop.get_network_status()
    for key, value in network_status.items():
        print(f"  {key}: {value}")
    print()

    # Show persistence status
    print("Persistence status:")
    persistence_status = aop.get_persistence_status()
    for key, value in persistence_status.items():
        print(f"  {key}: {value}")
    print()

    print("Network error handling features:")
    print("✅ Custom error messages with emojis")
    print("✅ Automatic network connectivity testing")
    print("✅ Configurable retry attempts and delays")
    print("✅ Network error detection and classification")
    print("✅ Graceful degradation and recovery")
    print()

    print("To test network error handling:")
    print("1. Start the server (it will run on localhost:8003)")
    print("2. Simulate network issues by:")
    print("   - Disconnecting your network")
    print("   - Blocking the port with firewall")
    print("   - Stopping the network service")
    print("3. Watch the custom error messages and retry attempts")
    print("4. Reconnect and see automatic recovery")
    print()

    try:
        print("Starting server with network monitoring...")
        print("Press Ctrl+C to stop the demo")
        print()

        # This will run with network monitoring enabled
        aop.run()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        print("Network status at shutdown:")
        network_status = aop.get_network_status()
        for key, value in network_status.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("This demonstrates how non-network errors are handled")


def simulate_network_issues():
    """
    Simulate various network issues for testing.

    This function can be used to test the network error handling
    in a controlled environment.
    """
    print("Network Issue Simulation:")
    print("1. Connection Refused - Server not running")
    print("2. Connection Reset - Server closed connection")
    print("3. Timeout - Server not responding")
    print("4. Host Resolution Failed - Invalid hostname")
    print("5. Network Unreachable - No route to host")
    print()
    print("The AOP server will detect these errors and:")
    print("- Display custom error messages with emojis")
    print("- Attempt automatic reconnection")
    print("- Test network connectivity before retry")
    print("- Give up after max retry attempts")


if __name__ == "__main__":
    main()
    print("\n" + "=" * 40)
    simulate_network_issues()
