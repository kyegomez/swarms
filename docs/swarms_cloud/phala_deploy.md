# Deploy on Phala

Phala can be used for trusted execution environment (TEE) deployments where a Swarms workload needs stronger isolation guarantees. This is useful for confidential agent workflows, private prompts, or workloads that process sensitive external inputs.

Use this page as a deployment checklist and adapt the exact commands to the Phala environment and template you are using.

## Good Fit

- Confidential agent workflows
- Private data processing
- Services that need verifiable execution properties
- Workloads where secrets should stay isolated from the host environment

## Service Shape

Package the Swarms application as a containerized HTTP service:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

The service should expose a health check and one or more narrow endpoints that call the agent with validated inputs.

## Deployment Checklist

- Keep provider API keys and secrets out of source code.
- Pass secrets through the deployment environment or secret manager.
- Validate request payloads before invoking the agent.
- Bound loops, retries, and tool calls to avoid unexpected cost.
- Avoid logging private prompts, secrets, or raw confidential data.
- Document what data enters and exits the trusted environment.

## Runtime Notes

For production workloads, prefer small explicit APIs over broad generic agent endpoints. This makes it easier to audit what the deployment is allowed to do and what data leaves the environment.
