# Codex CLI backend

Use an existing local Codex login from Swarms without writing a custom LLM adapter.
The optional backend uses the official `codex exec` command. Swarms runs locally;
model requests still go to Codex and use that account's access and usage limits.

## Setup

Install Swarms and [Codex CLI](https://learn.chatgpt.com/docs/cli), then log in:

```sh
npm install -g @openai/codex
codex login
codex login status
```

Codex CLI 0.147.0 or newer is required for the flags used here. Authentication is
managed by Codex; Swarms does not read or copy login tokens. Keep authentication
on your own machine. The automated tests use a fake CLI and require no credentials.

## Python

```python
from swarms import Agent

agent = Agent(
    agent_name="LocalAssistant",
    system_prompt="Answer clearly and concisely.",
    llm_backend="codex",
    model_name=None,  # Use the Codex CLI default model.
    codex_config={"timeout": 180},
    max_loops=1,
    retry_attempts=1,
    output_type="final",
)
print(agent.run("Explain how to combine CSV files with pandas."))
```

Set `model_name` to a model supported by your Codex account to select it explicitly.
Omitting `model_name` in Python retains Agent's usual model default; pass `None`
to use the CLI default. `codex_config` currently accepts only `timeout`, a positive,
finite number of seconds per CLI invocation. Unknown keys fail rather than being
silently ignored. Existing LiteLLM agents and custom `llm` objects work as before.

## YAML

The existing YAML loader forwards these options to Agent:

```yaml
agents:
  - agent_name: LocalAssistant
    system_prompt: Answer clearly and concisely.
    llm_backend: codex
    model_name: null
    max_loops: 1
    retry_attempts: 1
    codex_config:
      timeout: 180
```

Load it with `create_agents_from_yaml(yaml_file="agents.yaml", return_type="agents")`.
See [the example notebook](codex_backend.ipynb) for a two-agent SequentialWorkflow.

## Execution behavior and limits

- Each invocation starts an ephemeral Codex session in its own temporary directory.
  Swarms forwards the current system prompt and text message history. Memory
  compression also uses this backend instead of making a separate LiteLLM request.
- The CLI runs with a read-only sandbox. Shell tools, web search, Codex multi-agent
  delegation, apps, plugins, and project instructions are disabled for these text
  requests. This is not a repository-editing backend.
- `--ignore-user-config` prevents loading the user's Codex `config.toml` for this
  invocation. Saved authentication is still used. No global settings are changed.
- The final-message file supplies the response; progress logs are not treated as
  model output. Timeouts and keyboard cancellation terminate the CLI and its child
  processes, and temporary files are cleaned up.
- The first version rejects Swarms tools/MCP, image inputs, streaming, and
  `max_loops="auto"`. Use bounded text agents and workflows. Provider sampling knobs
  such as `temperature` and `max_tokens` are not forwarded to Codex, and Codex token
  usage is not yet included in `agent.usage`.
- For authentication or model errors, check `codex login status` and the installed
  CLI version. Retry behavior still follows the Agent's `retry_attempts` setting.

Reference: [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode).
