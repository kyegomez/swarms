#!/usr/bin/env python3
"""
Example demonstrating AOP network management and monitoring.

This example shows how to monitor and manage network connectivity
in an AOP server with real-time status updates.
"""

import time
import threading
from swarms import Agent
from swarms.structs.aop import AOP


def monitor_network_status(aop_instance):
    """Monitor network status in a separate thread."""
    while True:
        try:
            network_status = aop_instance.get_network_status()
            persistence_status = aop_instance.get_persistence_status()

            print(f"\n{'='*60}")
            print(
                f"📊 REAL-TIME STATUS MONITOR - {time.strftime('%H:%M:%S')}"
            )
            print(f"{'='*60}")

            # Network Status
            print("🌐 NETWORK STATUS:")
            print(
                f"  Monitoring: {'✅ Enabled' if network_status['network_monitoring_enabled'] else '❌ Disabled'}"
            )
            print(
                f"  Connected: {'✅ Yes' if network_status['network_connected'] else '❌ No'}"
            )
            print(
                f"  Retry Count: {network_status['network_retry_count']}/{network_status['max_network_retries']}"
            )
            print(
                f"  Remaining Retries: {network_status['remaining_network_retries']}"
            )
            print(
                f"  Host: {network_status['host']}:{network_status['port']}"
            )
            print(f"  Timeout: {network_status['network_timeout']}s")
            print(
                f"  Retry Delay: {network_status['network_retry_delay']}s"
            )

            if network_status["last_network_error"]:
                print(
                    f"  Last Error: {network_status['last_network_error']}"
                )

            # Persistence Status
            print("\n🔄 PERSISTENCE STATUS:")
            print(
                f"  Enabled: {'✅ Yes' if persistence_status['persistence_enabled'] else '❌ No'}"
            )
            print(
                f"  Shutdown Requested: {'❌ Yes' if persistence_status['shutdown_requested'] else '✅ No'}"
            )
            print(
                f"  Restart Count: {persistence_status['restart_count']}/{persistence_status['max_restart_attempts']}"
            )
            print(
                f"  Remaining Restarts: {persistence_status['remaining_restarts']}"
            )
            print(
                f"  Restart Delay: {persistence_status['restart_delay']}s"
            )

            # Connection Health
            if network_status["network_connected"]:
                print("\n💚 CONNECTION HEALTH: Excellent")
            elif network_status["network_retry_count"] == 0:
                print("\n🟡 CONNECTION HEALTH: Unknown")
            elif network_status["remaining_network_retries"] > 0:
                print(
                    f"\n🟠 CONNECTION HEALTH: Recovering ({network_status['remaining_network_retries']} retries left)"
                )
            else:
                print(
                    "\n🔴 CONNECTION HEALTH: Critical (No retries left)"
                )

            print(f"{'='*60}")

            # Check if we should stop monitoring
            if (
                persistence_status["shutdown_requested"]
                and not persistence_status["persistence_enabled"]
            ):
                print("🛑 Shutdown requested, stopping monitor...")
                break

            time.sleep(5)  # Update every 5 seconds

        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(5)


def main():
    """Demonstrate AOP network management."""

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

    print("AOP Network Management Demo")
    print("=" * 50)
    print()

    # Show initial configuration
    print("Initial Configuration:")
    print(f"  Server: {aop.server_name}")
    print(f"  Host: {aop.host}:{aop.port}")
    print(f"  Persistence: {aop.persistence}")
    print(f"  Network Monitoring: {aop.network_monitoring}")
    print(f"  Max Network Retries: {aop.max_network_retries}")
    print(f"  Network Timeout: {aop.network_timeout}s")
    print()

    # Start monitoring in background
    print("Starting network status monitor...")
    monitor_thread = threading.Thread(
        target=monitor_network_status, args=(aop,), daemon=True
    )
    monitor_thread.start()

    print("Available commands:")
    print("  'start' - Start the server")
    print("  'status' - Show current status")
    print("  'reset_network' - Reset network retry counter")
    print("  'disable_network' - Disable network monitoring")
    print("  'enable_network' - Enable network monitoring")
    print("  'shutdown' - Request graceful shutdown")
    print("  'quit' - Exit the program")
    print()

    try:
        while True:
            command = input("Enter command: ").strip().lower()

            if command == "start":
                print(
                    "Starting server... (Press Ctrl+C to test network error handling)"
                )
                try:
                    aop.run()
                except KeyboardInterrupt:
                    print("Server interrupted!")

            elif command == "status":
                print("\nCurrent Status:")
                network_status = aop.get_network_status()
                persistence_status = aop.get_persistence_status()

                print("Network:")
                for key, value in network_status.items():
                    print(f"  {key}: {value}")

                print("\nPersistence:")
                for key, value in persistence_status.items():
                    print(f"  {key}: {value}")

            elif command == "reset_network":
                aop.reset_network_retry_count()
                print("Network retry counter reset!")

            elif command == "disable_network":
                aop.network_monitoring = False
                print("Network monitoring disabled!")

            elif command == "enable_network":
                aop.network_monitoring = True
                print("Network monitoring enabled!")

            elif command == "shutdown":
                aop.request_shutdown()
                print("Shutdown requested!")

            elif command == "quit":
                print("Exiting...")
                break

            else:
                print(
                    "Unknown command. Try: start, status, reset_network, disable_network, enable_network, shutdown, quit"
                )

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean shutdown
        aop.disable_persistence()
        aop.request_shutdown()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
