# Telemetry Examples

This directory contains examples demonstrating telemetry and monitoring capabilities for agents.

## Overview

Telemetry examples demonstrate how to add monitoring, logging, and observability to agents. These examples show how to track agent performance, log operations, and monitor agent behavior using decorators and class methods.

## OpenTelemetry

Swarms can emit OpenTelemetry traces for agent and multi-agent
execution. Tracing is off by default.

Enable OTLP export with environment variables:

```bash
export SWARMS_OTEL_ENABLED=true
export OTEL_SERVICE_NAME=swarms
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

For local debugging without an OTLP collector:

```bash
export SWARMS_OTEL_ENABLED=true
export SWARMS_OTEL_EXPORTER=console
```

