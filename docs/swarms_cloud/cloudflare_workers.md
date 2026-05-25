# Deploy on Cloudflare Workers

Cloudflare Workers are useful for lightweight edge workflows around Swarms systems, especially request routing, simple preprocessing, webhook intake, and calls to a hosted Swarms API.

Workers are not a full Python runtime for running the Swarms Python package directly. Use a Worker as an edge adapter when your agent is hosted behind an HTTP API such as Cloud Run, a VM, or another container platform.

## Good Fit

- Routing requests to a hosted Swarms API
- Validating lightweight payloads at the edge
- Receiving webhooks and forwarding compact tasks
- Adding simple authentication checks before traffic reaches the agent service

## Worker Example

```javascript
export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const payload = await request.json();
    const response = await fetch(env.SWARMS_API_URL + "/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${env.SWARMS_API_TOKEN}`,
      },
      body: JSON.stringify({ task: payload.task }),
    });

    return new Response(await response.text(), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  },
};
```

## Configuration

Set the hosted Swarms service URL and token as Worker environment variables or secrets:

```bash
wrangler secret put SWARMS_API_TOKEN
wrangler secret put SWARMS_API_URL
```

## Operational Notes

- Keep the Worker focused on edge concerns and call the Swarms runtime over HTTP.
- Validate payload size and required fields before forwarding.
- Do not store provider keys in client-side code.
- Add rate limiting or authentication before public endpoints.
- Return compact errors so callers can retry safely.
