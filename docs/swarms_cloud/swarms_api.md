# Swarms Cloud API

The Swarms Cloud API route is referenced by contributor documentation as the hosted API reference entry point. This page gives users a stable local documentation page for API setup, credentials, and integration guidance.

For exact endpoint definitions, request and response schemas, and current hosted API behavior, use the official hosted API documentation for your Swarms Cloud environment.

## Authentication

Use an API key from the Swarms Cloud dashboard or your deployment environment. Keep it in an environment variable:

```bash
export SWARMS_API_KEY="your-api-key"
```

Applications should pass credentials through server-side configuration. Do not expose API keys in frontend code.

## Request Pattern

```python
import os
import requests


def call_swarms_cloud(task: str) -> str:
    api_key = os.getenv("SWARMS_API_KEY")
    response = requests.post(
        "https://api.swarms.world/v1/run",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"task": task},
        timeout=60,
    )
    response.raise_for_status()
    return response.text
```

Adjust the URL and payload shape to the hosted API endpoint you are using.

## Operational Notes

- Validate request payloads before sending them to the API.
- Set explicit timeouts for all network calls.
- Keep retries bounded and avoid retrying non-idempotent operations blindly.
- Log status codes and request IDs, not secrets.
- Use server-side proxy routes when calling the API from a web application.

## Related Docs

- [Swarms Cloud Python Client](python_client.md)
- [Environment Setup](../swarms/install/env.md)
- [Contributor Environment Setup](../contributors/environment_setup.md)
