"""
OpenTelemetry Tracing Example for Swarms

This example demonstrates how to enable OpenTelemetry tracing
for agent and multi-agent workflow executions.

Prerequisites:
    pip install swarms[otel]
    # or
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

Configuration:
    Set these environment variables before running:
    - SWARMS_OTEL_ENABLED=true        # Enable tracing
    - OTEL_SERVICE_NAME=my-app        # Your service name (default: swarms)
    - OTEL_EXPORTER_OTLP_ENDPOINT=... # OTLP endpoint (optional)

Running with Jaeger:
    # Start Jaeger
    docker run -d --name jaeger \
        -p 16686:16686 \
        -p 4317:4317 \
        jaegertracing/all-in-one:latest

    # Set environment and run
    export SWARMS_OTEL_ENABLED=true
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    python otel_tracing_example.py

    # View traces at http://localhost:16686
"""

import os

os.environ["SWARMS_OTEL_ENABLED"] = "true"
os.environ["OTEL_SERVICE_NAME"] = "swarms-example"

from swarms import Agent
from swarms.structs.swarm_router import SwarmRouter
from swarms.telemetry import is_otel_enabled, otel_available


def main():
    print("OpenTelemetry Integration Example")
    print("=" * 50)

    print(f"OTEL Available: {otel_available()}")
    print(f"OTEL Enabled: {is_otel_enabled()}")
    print()

    if not otel_available():
        print(
            "OpenTelemetry packages not installed. Install with:"
        )
        print("  pip install swarms[otel]")
        print()

    analyst = Agent(
        agent_name="Financial-Analyst",
        system_prompt="You are a financial analyst. Provide brief, concise analysis.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    researcher = Agent(
        agent_name="Market-Researcher",
        system_prompt="You are a market researcher. Provide brief insights.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    print("Running single agent (traced)...")
    result = analyst.run(
        "What are the key factors affecting tech stock prices?"
    )
    print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
    print()

    print("Running multi-agent workflow (traced)...")
    router = SwarmRouter(
        name="analysis-team",
        agents=[analyst, researcher],
        swarm_type="SequentialWorkflow",
        max_loops=1,
    )

    workflow_result = router.run(
        "Analyze the current state of AI chip market"
    )
    print("Workflow completed!")
    print()

    if is_otel_enabled():
        print("Traces have been recorded.")
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            print(f"Check your OTLP backend at: {endpoint}")
        else:
            print(
                "No OTLP endpoint configured - traces stored in memory only."
            )
    else:
        print(
            "OTEL not enabled. Set SWARMS_OTEL_ENABLED=true to enable tracing."
        )


if __name__ == "__main__":
    main()
