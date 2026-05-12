# Swarms Quickstart

This quickstart walks through the shortest path from a clean Python environment to a running Swarms `Agent`. Use it when you want to verify your installation, confirm your API key setup, and run a small first task before moving into multi-agent workflows.

## 1. Create a Python Environment

Swarms supports Python 3.10 or higher. Start from a fresh virtual environment so package versions and local experiments stay isolated from the rest of your system.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If you prefer `uv`, create and activate the environment in the same project folder:

```bash
uv venv
source .venv/bin/activate
```

## 2. Install Swarms

Install the latest published package with `pip`:

```bash
pip install -U swarms
```

For faster installs, use `uv`:

```bash
uv pip install -U swarms
```

To work on Swarms itself, clone the repository and install it in editable mode:

```bash
git clone https://github.com/kyegomez/swarms.git
cd swarms
pip install -e .
```

## 3. Configure Environment Variables

Most examples use an LLM provider key plus a workspace directory. Create a `.env` file in your project root:

```bash
OPENAI_API_KEY="your-openai-api-key"
WORKSPACE_DIR="agent_workspace"
SWARMS_VERBOSE_GLOBAL="False"
```

Keep `.env` out of version control:

```bash
echo ".env" >> .gitignore
```

Swarms can also work with other provider keys, including `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, and additional tool provider keys documented in the environment setup guide. Start with one working provider key, confirm the first agent run, and add more providers only when your workflow needs them.

## 4. Run Your First Agent

Create a file named `quickstart_agent.py`:

```python
from swarms import Agent


agent = Agent(
    agent_name="Quickstart-Agent",
    agent_description="A small agent used to verify a new Swarms setup.",
    system_prompt=(
        "You are a concise technical assistant. "
        "Return practical answers with clear next steps."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    max_tokens=1024,
    temperature=0.3,
    output_type="str",
    safety_prompt_on=True,
)

response = agent.run(
    "Give me a three-step checklist for evaluating a new automation idea."
)

print(response)
```

Run it from the same environment where you installed Swarms:

```bash
python quickstart_agent.py
```

If everything is configured correctly, the script prints a short checklist. That confirms the package import, model provider configuration, agent loop, and output handling are all working.

## 5. Troubleshoot Common Setup Issues

### `ModuleNotFoundError: No module named 'swarms'`

Confirm your virtual environment is active and reinstall the package:

```bash
source .venv/bin/activate
pip install -U swarms
```

### Missing API Key

Check that the variable name matches the provider used by your model. For the example above, `OPENAI_API_KEY` must be available in your shell or loaded from `.env`.

```bash
echo "$OPENAI_API_KEY"
```

If your project uses `python-dotenv`, load the file before creating the agent:

```python
from dotenv import load_dotenv

load_dotenv()
```

### Workspace Problems

If the agent cannot write logs or artifacts, set `WORKSPACE_DIR` to a folder your process can create and modify:

```bash
export WORKSPACE_DIR="agent_workspace"
mkdir -p "$WORKSPACE_DIR"
```

## 6. Next Steps

After the first agent runs successfully, continue with the deeper guides:

- [Installation Guide](swarms/install/install.md) for more package manager and development setup options.
- [Environment Variables](swarms/install/env.md) for provider keys, workspace configuration, and security practices.
- [Agents Introduction](swarms/agents/index.md) for the core `Agent` parameters and patterns.
- [Basic Agent Example](swarms/examples/basic_agent.md) for a more complete single-agent walkthrough.

From there, move into workflows, routers, tools, and multi-agent architectures when your project needs coordination across several specialized agents.
