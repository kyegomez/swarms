# A2A examples

Serve a Swarms agent over the [A2A](https://a2a-protocol.org) JSON-RPC 1.0 protocol, and call any A2A agent as if it were local.

| File | What it shows |
|---|---|
| `a2a_server_example.py` | Deploy an agent as an A2A server with an API key |
| `a2a_client_example.py` | Read a remote agent's card and send it a task |
| `a2a_server_and_client_example.py` | Start a server in the background and call it, in one script |
| `a2a_handoff_example.py` | Give a local agent a remote A2A agent as a handoff |

## Run

Set a model key, for example `OPENAI_API_KEY`, then:

```bash
# One script, nothing else to start
python examples/a2a/a2a_server_and_client_example.py

# Or two terminals
python examples/a2a/a2a_server_example.py
python examples/a2a/a2a_client_example.py
```

## What the server exposes

- `GET /.well-known/agent-card.json`: the public agent card.
- `POST /a2a`: the `SendMessage` endpoint. It needs the `A2A-Version: 1.0` header and the API key, sent as `x-api-key` or `Authorization: Bearer`.

`A2AServer` is an `MCPDeployer`, so the same process also serves the agent as an MCP tool and takes the same options (`host`, `port`, `api_keys`, custom auth).
