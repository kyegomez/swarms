# Deploy on Google Cloud Run

Google Cloud Run is a good target for containerized Swarms APIs because it provides managed HTTPS, autoscaling, and simple environment-variable configuration.

Use this pattern when you have a FastAPI, Flask, or similar HTTP service that wraps an agent or swarm.

## Prerequisites

- Google Cloud project with billing enabled
- `gcloud` CLI installed and authenticated
- Dockerfile for the Swarms service
- Required model provider keys stored as environment variables

## Minimal Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Cloud Run injects the `PORT` environment variable, but `8080` is the default expected port and works for most deployments.

## Deploy

```bash
gcloud run deploy swarms-agent-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY="$OPENAI_API_KEY"
```

For private services, remove `--allow-unauthenticated` and configure IAM access for callers.

## Operational Notes

- Keep synchronous agent APIs bounded with low `max_loops` values.
- Use Cloud Run minimum instances only when cold starts are a problem.
- Store sensitive values in Secret Manager for production deployments.
- Add a `/health` route so uptime checks can verify the service.
- Log request IDs and high-level agent status, not raw secrets or long prompts.
- Move long-running workflows to background jobs or queues instead of blocking an HTTP request.
