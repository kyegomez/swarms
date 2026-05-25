# Swarms Cloud Python Client

The Swarms Cloud Python client is the recommended interface for Python applications that need to call hosted Swarms infrastructure. The ecosystem overview links to this route as the Python SDK documentation entry point.

Use this page as a starting point for client setup, environment configuration, and integration patterns.

## Install

```bash
pip install -U swarms-sdk
```

## Configuration

Store credentials in environment variables instead of source code:

```bash
export SWARMS_API_KEY="your-api-key"
```

For applications with multiple environments, keep development, staging, and production keys separate.

## Integration Pattern

Most Python client integrations follow this shape:

1. Load the API key from the environment.
2. Create a small wrapper around the hosted operation you need.
3. Validate inputs before sending requests.
4. Return compact results to the rest of your application.
5. Log request status without logging secrets or sensitive payloads.

## Production Checklist

- Keep API keys out of notebooks, commits, and screenshots.
- Add retries only for idempotent operations.
- Set request timeouts in application code.
- Preserve request IDs or trace IDs when debugging production issues.
- Avoid sending unnecessary private data to hosted endpoints.

## Related Docs

- [Swarms Cloud API](swarms_api.md)
- [Environment Setup](../swarms/install/env.md)
- [Contributor Environment Setup](../contributors/environment_setup.md)
