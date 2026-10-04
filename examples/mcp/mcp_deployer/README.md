# MCPDeployer Examples

`MCPDeployer` turns one `Agent`, any swarm with a `run()` method, or a plain callable into an MCP server with an auth layer in front of it. Each target becomes one tool, named after it, taking `task` and an optional `img`. Pass a list, or a dict of tool name to target, to serve several agents and swarms from one server; `add_tool()` registers more before start.

```python
from swarms import Agent, MCPDeployer

agent = Agent(agent_name="Researcher", model_name="gpt-5.4", max_loops=1)
MCPDeployer(agent, api_keys=["sk-local-dev"], port=8000).run()
```

Another agent then connects with `Agent(mcp_url=MCPConnection(url="http://127.0.0.1:8000/mcp", api_key="sk-local-dev"))`.

| Example | Target | Auth | Transport | Runs on its own? |
|---|---|---|---|---|
| [single_agent_api_key.py](single_agent_api_key.py) | One `Agent` | static `api_keys` | streamable HTTP | serves until stopped |
| [sequential_workflow_as_tool.py](sequential_workflow_as_tool.py) | `SequentialWorkflow` | static `api_keys` | streamable HTTP | serves until stopped |
| [multiple_agents_one_server.py](multiple_agents_one_server.py) | two Agents, a `SequentialWorkflow` and two functions, each its own tool | static `api_keys` | streamable HTTP | serves until stopped |
| [custom_auth_per_tenant.py](custom_auth_per_tenant.py) | One `Agent` | async `auth` callable reading `x-tenant` | streamable HTTP | serves until stopped |
| [owner_key_or_tenant_auth.py](owner_key_or_tenant_auth.py) | One `Agent` | sync `auth` callable: an owner key from the environment, or an allow-listed `x-tenant` | streamable HTTP | serves until stopped |
| [token_verifier_with_scopes.py](token_verifier_with_scopes.py) | One `Agent` | `TokenVerifier` with `required_scopes` | streamable HTTP | serves until stopped |
| [env_keys_and_extra_tools.py](env_keys_and_extra_tools.py) | One `Agent` plus two plain functions | `api_key_env` | streamable HTTP | serves until stopped |
| [background_server_and_client_agent.py](background_server_and_client_agent.py) | One `Agent` | static `api_keys` | streamable HTTP | yes: serves, calls, stops |
| [plain_function_json_response.py](plain_function_json_response.py) | A plain function, no LLM | static `api_keys` | streamable HTTP, JSON replies | yes, no LLM key needed |
| [sse_transport.py](sse_transport.py) | One `Agent` | static `api_keys` | SSE | serves until stopped |
| [stdio_transport.py](stdio_transport.py) | One `Agent` | none (host is the boundary) | stdio | launched by an MCP host |

## Auth, in order of precedence

1. `auth=callable(credential, headers)`: your own check, sync or async. Truthy admits, a dict is kept as the request's claims, falsy or an exception refuses.
2. `token_verifier=`: the `mcp` package's `TokenVerifier` protocol. `required_scopes` and expiry are enforced.
3. `api_keys=[...]` and `api_key_env="VAR"`: static keys compared in constant time.
4. `allow_anonymous=True`: opt in explicitly. With nothing configured the constructor refuses to build.

Clients may send the key as `x-api-key` (rename with `api_key_header`) or as `Authorization: Bearer`, which is what `MCPManager` sends by default. Refused requests get a 401 with `WWW-Authenticate: Bearer`. `/health` is always public.

## Lifecycle

`run()` blocks. `start()` and `stop()`, or `with MCPDeployer(...) as d:`, run the server on a background thread, which is what the background_server_and_client_agent and plain_function_json_response examples do. `timeout` bounds one tool call; `extra_tools` exposes more plain functions beside the main tool.

Each example names its model with a plain LiteLLM string; swap it for any provider you have a key for.
