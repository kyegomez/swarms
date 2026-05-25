# Deployment Solutions

Swarms agents can run in notebooks, scripts, API services, scheduled jobs, and cloud platforms. This section collects deployment patterns for moving an agent or swarm from a local experiment into a repeatable production workflow.

Start with the smallest deployment that gives your users a stable interface. A single FastAPI service is often enough for request-response workflows, while scheduled jobs and cloud workers are better for background tasks, monitoring, and event-driven automation.

## Deployment Options

| Pattern | Best for |
| --- | --- |
| FastAPI service | HTTP APIs, internal tools, dashboards, and application backends |
| Cron job | Scheduled research, reporting, and recurring agent runs |
| MCP server | Exposing an agent or tool to MCP-compatible clients |
| Cloud Run | Containerized APIs with managed scaling |
| Cloudflare Workers | Lightweight edge workflows and low-latency request handling |
| Phala | Trusted execution and confidential workloads |

## Production Checklist

- Keep API keys and provider credentials in environment variables.
- Set request timeouts for tools and model calls.
- Log request IDs, agent names, and high-level run status.
- Add health checks before putting an agent behind a load balancer.
- Bound agent loop counts and tool calls for predictable cost.
- Validate inputs before sending them to the agent.
- Store long-running task state outside the process when reliability matters.

## Minimal Service Shape

Most deployments follow the same shape:

1. Load configuration from the environment.
2. Create the agent or swarm once during process startup.
3. Validate incoming user input.
4. Run the agent with bounded settings.
5. Return a structured response to the caller.

For a concrete HTTP example, see [FastAPI + Uvicorn](fastapi_agent_api.md). For scheduled jobs, see the [CronJob reference](../swarms/structs/cron_job.md).
