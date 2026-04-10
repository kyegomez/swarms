# CLAUDE.md — Swarms Codebase Guide

## Project Overview

**Swarms** is an enterprise-grade, production-ready multi-agent orchestration
framework (v9.0.2). It provides infrastructure to compose, orchestrate, and
deploy AI agents for real-world automation tasks.

- **License:** MIT
- **Python:** >=3.10,<4.0
- **Build system:** Poetry
- **PyPI:** `swarms`
- **CLI entrypoint:** `swarms` (maps to `swarms.cli.main:main`)
- **Docs:** https://docs.swarms.world

---

## Repository Layout

```
swarms/                    # Main source package
  __init__.py              # Re-exports everything; runs telemetry bootup
  agents/                  # Specialized agent implementations
  structs/                 # Swarm orchestration structures (60+ files)
  tools/                   # Tool management and function-calling helpers
  prompts/                 # Domain-specific system prompt templates
  schemas/                 # Pydantic models for shared data contracts
  utils/                   # Utility helpers (logging, formatting, LLM wrappers)
  artifacts/               # Artifact versioning (main_artifact.py)
  telemetry/               # Startup hooks and telemetry collection
  cli/                     # CLI commands (main.py + utils.py)
  sims/                    # Simulation examples (senator_assembly.py)
tests/                     # Mirrors swarms/ directory structure
  structs/                 # ~25 test files for swarm structures
  agents/                  # Agent tests
  utils/                   # Utility tests
  tools/                   # Tool tests
  benchmarks/              # Performance benchmarks
examples/                  # Runnable example scripts and notebooks
docs/                      # MkDocs documentation source
scripts/                   # Maintenance and release scripts
.github/workflows/         # CI/CD pipeline definitions
```

---

## Key Modules

### `swarms/structs/` — Orchestration Primitives

The heart of the framework. Public API exported via `structs/__init__.py`.

| Class/Function | File | Purpose |
|---|---|---|
| `Agent` | `agent.py` | Core agent with LLM, tools, memory, conversation |
| `SequentialWorkflow` | `sequential_workflow.py` | Agents run in sequence |
| `ConcurrentWorkflow` | `concurrent_workflow.py` | Agents run in parallel via ThreadPoolExecutor |
| `AgentRearrange` | `agent_rearrange.py` | Dynamic agent topology from a flow string |
| `HierarchicalSwarm` | `hiearchical_swarm.py` | Director + worker agent hierarchy |
| `MixtureOfAgents` | `mixture_of_agents.py` | Ensemble pattern with aggregator |
| `GraphWorkflow` | `graph_workflow.py` | DAG-based execution (Node/Edge/NodeType) |
| `GroupChat` | `groupchat.py` | Multi-agent chat with speaker strategies |
| `SwarmRouter` | `swarm_router.py` | Auto-routes tasks to appropriate swarm type |
| `MultiAgentRouter` | `multi_agent_router.py` | Routes to individual agents |
| `SpreadSheetSwarm` | `spreadsheet_swarm.py` | Grid-based parallel execution |
| `HierarchicalSwarm` | `hiearchical_swarm.py` | Includes agent-as-judge scoring, async `arun()` |
| `RoundRobinSwarm` | `round_robin.py` | Cyclic agent selection |
| `MajorityVoting` | `majority_voting.py` | Consensus via voting |
| `CouncilAsAJudge` | `council_as_judge.py` | Council-based evaluation |
| `DebateWithJudge` | `debate_with_judge.py` | Structured debate pattern |
| `HeavySwarm` | `heavy_swarm.py` | Heavyweight swarm variant |
| `LLMCouncil` | `llm_council.py` | Multi-model council |
| `AutoSwarmBuilder` | `auto_swarm_builder.py` | Dynamically constructs swarms |
| `AOP` | `aop.py` | Agent Orchestration Protocol |
| `Conversation` | `conversation.py` | Conversation/message-history management |
| `CronJob` | `cron_job.py` | Scheduled agent execution |
| `BatchedGridWorkflow` | `batched_grid_workflow.py` | Batched grid execution |
| `run_agents_concurrently` | `multi_agent_exec.py` | Standalone concurrent execution helpers |

Speaker strategies for `GroupChat`: `round_robin_speaker`, `random_speaker`,
`expertise_based`, `priority_speaker`, `random_dynamic_speaker`.

Swarming topology helpers: `broadcast`, `circular_swarm`, `grid_swarm`,
`mesh_swarm`, `one_to_one`, `pyramid_swarm`, `star_swarm`.

### `swarms/agents/` — Specialized Agents

| Class | File | Purpose |
|---|---|---|
| `Agent` | `structs/agent.py` | Base production agent (imported via structs) |
| `AgentJudge` | `agent_judge.py` | Evaluation/scoring agent |
| `SelfConsistencyAgent` | `consistency_agent.py` | Self-consistency sampling |
| `ReasoningDuo` | `reasoning_duo.py` | Two-model reasoning pair |
| `ReasoningAgentRouter` | `reasoning_agent_router.py` | Routes to specialized reasoners |
| `ReflexionAgent` | `flexion_agent.py` | Reflexion-style iterative refinement |
| `IterativeReflectiveExpansion` | `i_agent.py` | Deep iterative expansion |
| `GKPAgent` | `gkp_agent.py` | Generated knowledge prompting |
| `create_agents_from_yaml` | `create_agents_from_yaml.py` | YAML-driven agent factory |

### `swarms/tools/` — Tool Layer

- `BaseTool` / `ToolRegistry` — base class and registry for tools
- `py_func_to_openai_func_str.py` — converts Python callables to OpenAI
  function-calling schema
- `mcp_client_tools.py` — MCP (Model Context Protocol) tool integration
- `tool_parse_exec.py` — tool invocation and result parsing
- `handoffs_tool.py` / `create_agent_tool.py` — agent handoff patterns

### `swarms/utils/` — Utilities

- `litellm_wrapper.py` — unified LLM provider interface (via LiteLLM)
- `litellm_tokenizer.py` — token counting
- `loguru_logger.py` — structured logger (use this, not stdlib `logging`)
- `formatter.py` / `history_output_formatter.py` — output formatting
- `data_to_text.py`, `pdf_to_text.py`, `file_processing.py` — data ingestion
- `dynamic_context_window.py` — adaptive context management
- `output_types.py` — output type constants and helpers

### `swarms/schemas/` — Pydantic Models

- `base_schemas.py` — shared base models
- `agent_class_schema.py` — agent configuration schema
- `conversation_schema.py` — message/conversation models
- `mcp_schemas.py`, `agent_mcp_errors.py` — MCP data contracts
- `handoffs_schema.py` — agent handoff data models
- `swarms_api_schemas.py` — API-layer schemas

---

## Development Workflow

### Installation (from source)

```bash
# Preferred: Poetry
poetry install

# With test dependencies
poetry install --with test

# With lint tools
poetry install --with lint

# All groups
poetry install --with test,lint,dev

# Alternative: pip editable install
pip install -e .
```

### Environment Setup

Copy `.env.example` to `.env` and fill in the required keys:

```bash
cp .env.example .env
```

Key environment variables:

```bash
# Framework
WORKSPACE_DIR="agent_workspace"        # where agents save files
SWARMS_VERBOSE_GLOBAL="False"
SWARMS_TELEMETRY_ON="false"
SWARMS_OUTPUT_FORMATTING_MARKDOWN_ENABLED=False

# Model providers (set at least one)
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GEMINI_API_KEY=""
GROQ_API_KEY=""

# Optional tool integrations
TAVILY_API_KEY=""
BRAVESEARCH_API_KEY=""
EXA_API_KEY=""
```

### Running Tests

```bash
# Full test suite
poetry run pytest tests/ -v

# Specific test file
poetry run pytest tests/structs/test_sequential_workflow.py -v

# With coverage
poetry run pytest tests/ --cov=swarms
```

Tests mirror the `swarms/` package layout. New features require corresponding
tests in `tests/<subpackage>/test_<module>.py`.

### Linting and Formatting

```bash
# Format code
black .

# Check formatting without modifying
black . --check --diff

# Lint
ruff check .

# Fix auto-fixable lint issues
ruff check . --fix
```

CI runs both `black --check` and `ruff check` on every push and PR.

**Line length is 70 characters** (configured in both `[tool.black]` and
`[tool.ruff]` in `pyproject.toml`).

---

## Coding Conventions

### Code Style

- **PEP 8** compliance enforced via Black + Ruff
- **Line length: 70 characters** — strictly enforced by CI
- **Formatter:** Black (target Python 3.8+ syntax via `target-version = ["py38"]`)
- **Linter:** Ruff

### Naming

| Construct | Convention | Example |
|---|---|---|
| Classes | PascalCase | `SequentialWorkflow`, `BaseSwarm` |
| Functions/methods | snake_case | `run()`, `add_agent()`, `run_async()` |
| Constants | UPPER_SNAKE_CASE | `WORKSPACE_DIR` |
| Private members | leading underscore | `_internal_state` |
| Base classes | `Base` prefix or `base_` filename | `BaseSwarm`, `base_tool.py` |

### Type Annotations

All public functions and methods **must** have full type annotations:

```python
def run(self, task: str) -> str:
    ...
```

### Docstrings

Every public class, function, and method requires a docstring in either
Google or NumPy style:

```python
def run(self, task: str) -> str:
    """
    Run the agent on a task.

    Args:
        task (str): The task description to execute.

    Returns:
        str: The agent's final response.

    Raises:
        ValueError: If task is empty.
    """
```

### Logging

Use `loguru_logger` from `swarms.utils.loguru_logger`, **not** Python's
stdlib `logging`:

```python
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="my_module")
logger.info("Agent started")
```

### Error Handling

- Define custom exceptions in `swarms/schemas/` (e.g., `AgentMCPConnectionError`)
- Only validate at system boundaries (user input, external APIs)
- Do not add defensive checks for conditions that cannot occur internally

---

## Architecture Patterns

### Adding a New Swarm Architecture

1. Create `swarms/structs/<your_swarm>.py`
2. Subclass `BaseSwarm` (from `base_swarm.py`) or `BaseStructure`
3. Implement at minimum: `run(task: str) -> Any`
4. Export from `swarms/structs/__init__.py` and add to `__all__`
5. Add tests in `tests/structs/test_<your_swarm>.py`
6. Add docs page in `docs/`

### Adding a New Agent Type

1. Create `swarms/agents/<your_agent>.py`
2. Export from `swarms/agents/__init__.py` and add to `__all__`
3. Add tests in `tests/agents/test_<your_agent>.py`

### Adding a New Tool

1. Create tool logic in `swarms/tools/`
2. Use `py_func_to_openai_func_str` to expose Python callables as OpenAI tools
3. Register via `ToolRegistry` if it needs discovery
4. Define schemas in `swarms/schemas/` if needed

### Adding Prompts

Add domain-specific system prompts as constants or functions in
`swarms/prompts/<domain>_prompt.py`. Follow the existing naming pattern
(e.g., `finance_agent_prompt.py`, `legal_agent_prompt.py`).

---

## CI/CD Pipelines

Located in `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `lint.yml` | push, PR | Black format check + Ruff lint |
| `tests.yml` | push/PR to master | `poetry run pytest tests/ -v` |
| `python-package.yml` | push | Package build validation |
| `test-main-features.yml` | push | Core feature regression tests |
| `codeql.yml` | scheduled + push | CodeQL security analysis |
| `trivy.yml` | scheduled + push | Container/dependency security scan |
| `pysa.yml` | scheduled | Python static analysis (Pysa) |
| `docs.yml` | push | MkDocs documentation build |

All CI checks must pass before merging to `master`.

---

## Public API Surface

The top-level `swarms` package re-exports everything from all subpackages via
wildcard imports in `swarms/__init__.py`. All public names should be listed
in the relevant subpackage's `__all__`.

Primary public classes users interact with:

```python
from swarms import (
    Agent,                      # Core agent
    SequentialWorkflow,         # Sequential multi-agent
    ConcurrentWorkflow,         # Parallel multi-agent
    AgentRearrange,             # Dynamic topology
    HierarchicalSwarm,          # Director + workers
    MixtureOfAgents,            # Ensemble
    GraphWorkflow,              # DAG workflow
    GroupChat,                  # Multi-agent chat
    SwarmRouter,                # Auto-routing swarm
    MultiAgentRouter,           # Agent router
    SpreadSheetSwarm,           # Grid execution
    ReasoningAgentRouter,       # Reasoning specialization
    create_agents_from_yaml,    # YAML agent factory
)
```

---

## Important Files Reference

| File | Purpose |
|---|---|
| `pyproject.toml` | Poetry config, dependencies, Black/Ruff settings |
| `.env.example` | All supported environment variables |
| `swarms/__init__.py` | Package entry; runs `bootup()` then wildcard imports |
| `swarms/structs/agent.py` | Core `Agent` class — primary production agent |
| `swarms/structs/__init__.py` | Struct public API (`__all__`) |
| `swarms/agents/__init__.py` | Agents public API (`__all__`) |
| `swarms/utils/loguru_logger.py` | Logging utility |
| `swarms/utils/litellm_wrapper.py` | Multi-provider LLM interface |
| `swarms/telemetry/bootup.py` | Startup/telemetry initialization |
| `CONTRIBUTING.md` | Full contributor guide |
| `docs/mkdocs.yml` | Documentation site configuration |

---

## Notes for AI Assistants

- **Do not modify `__init__.py` wildcard imports** without also updating `__all__`
  in the relevant subpackage.
- **Line length is 70 chars** — this is stricter than the PEP 8 default of 79;
  Black will reformat to this limit automatically.
- **Telemetry runs at import time** via `bootup()` in `swarms/__init__.py`;
  set `SWARMS_TELEMETRY_ON=false` in `.env` to disable during development.
- **LiteLLM** is the unified LLM backend; model strings follow LiteLLM format
  (e.g., `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`, `"gemini/gemini-pro"`).
- **Pydantic v2** is used throughout (`pydantic>=2.12.5`); use `model_validator`,
  `field_validator`, and `model_dump()` (not `.dict()`).
- **`uvloop`** is installed on Linux/macOS for async performance; use
  `run_agents_concurrently_uvloop` when maximum async throughput is needed.
- Tests in `tests/` mirror `swarms/` structure; always add tests alongside
  new code.
- The `swarms` CLI is defined in `swarms/cli/main.py` and exposed as the
  `swarms` console script.
