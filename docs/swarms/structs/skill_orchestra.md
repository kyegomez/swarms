# SkillOrchestra Documentation

SkillOrchestra is a skill-aware agent orchestration system based on the paper ["SkillOrchestra: Learning to Route Agents via Skill Transfer"](https://arxiv.org/abs/2602.19672). Instead of end-to-end RL routing, it maintains a **Skill Handbook** that profiles each agent on fine-grained skills, infers which skills a task requires via LLM, and matches agents to tasks via explicit competence-cost scoring.

## Table of Contents
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Key Components](#key-components)
- [Arguments](#arguments)
- [Methods](#methods)
- [Architecture](#architecture)
- [Best Practices](#best-practices)

## Installation

```bash
pip install swarms
```

## How It Works

SkillOrchestra routes tasks through a 5-step pipeline:

```
Task → Skill Inference → Agent Scoring → Agent Selection → Execution → Learning
```

1. **Skill Inference** — An LLM analyzes the incoming task and identifies which fine-grained skills are required (e.g., `python_coding`, `data_analysis`, `technical_writing`), each with an importance weight.
2. **Agent Scoring** — Each agent is scored using a weighted competence-cost formula against the required skills. This step is pure math — no LLM calls.
3. **Agent Selection** — The top-k agents with the highest scores are selected.
4. **Execution** — Selected agents execute the task. Multiple agents run concurrently via `ThreadPoolExecutor`.
5. **Learning** (optional) — An LLM evaluates the output quality, and agent skill profiles are updated via exponential moving average (EMA).

### Scoring Formula

For each agent, the score is computed as:

```
score = Σ (competence_weight × competence_i × importance_i + cost_weight × normalized_cost_i × importance_i) / total_importance
```

Where:
- `competence_i` is the agent's estimated probability of success on skill `i`
- `normalized_cost_i` is `1 - (cost - min_cost) / (max_cost - min_cost)` (lower cost = higher score)
- `importance_i` is how important the skill is for the task

## Key Components

### Data Models

| Model | Description |
|-------|-------------|
| `SkillDefinition` | A fine-grained skill with name, description, and optional category |
| `AgentSkillProfile` | An agent's competence (0-1) and cost on a specific skill, with execution statistics |
| `AgentProfile` | Complete skill profile for a single agent |
| `SkillHandbook` | Central data structure mapping all skills to all agent profiles |
| `TaskSkillInference` | LLM output: skills required by a given task with importance weights |
| `AgentSelectionResult` | Result of agent scoring with name, score, and reasoning |
| `ExecutionFeedback` | Post-execution quality assessment for updating skill profiles |

### Arguments Table

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `name` | `str` | `"SkillOrchestra"` | Name identifier for the orchestrator |
| `description` | `str` | `"Skill-aware agent orchestration..."` | Description of the orchestrator's purpose |
| `agents` | `List[Union[Agent, Callable]]` | `None` | List of agents to orchestrate (required, at least 1) |
| `max_loops` | `int` | `1` | Maximum execution-feedback loops per task |
| `output_type` | `OutputType` | `"dict"` | Output format: `"dict"`, `"str"`, `"json"`, `"final"`, etc. |
| `model` | `str` | `"gpt-4o-mini"` | LLM model for skill inference and evaluation |
| `temperature` | `float` | `0.1` | LLM temperature for inference calls |
| `skill_handbook` | `Optional[SkillHandbook]` | `None` | Pre-built skill handbook. If `None`, auto-generated from agent descriptions |
| `auto_generate_skills` | `bool` | `True` | Whether to auto-generate handbook when none is provided |
| `cost_weight` | `float` | `0.3` | Weight for cost component in scoring (0-1) |
| `competence_weight` | `float` | `0.7` | Weight for competence component in scoring (0-1) |
| `top_k_agents` | `int` | `1` | Number of agents to select per task |
| `learning_enabled` | `bool` | `True` | Whether to update skill profiles after execution via EMA |
| `learning_rate` | `float` | `0.1` | EMA learning rate for profile updates |
| `autosave` | `bool` | `True` | Whether to save conversation history and handbook to disk |
| `verbose` | `bool` | `False` | Whether to log detailed information |
| `print_on` | `bool` | `True` | Whether to print panels to console |

### Methods Table

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `run` | `task: str, img: Optional[str], imgs: Optional[List[str]]` | `Any` | Run the full pipeline on a single task |
| `__call__` | `task: str, *args, **kwargs` | `Any` | Callable interface — delegates to `run()` |
| `batch_run` | `tasks: List[str]` | `List[Any]` | Run multiple tasks sequentially |
| `concurrent_batch_run` | `tasks: List[str]` | `List[Any]` | Run multiple tasks concurrently |
| `get_handbook` | — | `dict` | Return the current skill handbook as a dictionary |
| `update_handbook` | `handbook: SkillHandbook` | `None` | Replace the skill handbook |

## Architecture

### Pipeline Flow

```mermaid
flowchart TD
    A["Incoming Task"] --> B["1. Skill Inference (LLM)"]
    B --> C["2. Agent Scoring (Math)"]
    C --> D["3. Select Top-K Agents"]
    D --> E["4. Execute Agents"]
    E --> F{Learning Enabled?}
    F -- Yes --> G["5. Evaluate & Learn (LLM + EMA)"]
    G --> H{More Loops?}
    H -- Yes --> B
    H -- No --> I["Return Output"]
    F -- No --> I

    subgraph Skill Handbook
        S1["Skills: python_coding, api_design, technical_writing, ..."]
        S2["Agent Profiles: competence + cost per skill"]
    end

    B -. reads .-> S1
    C -. reads .-> S2
    G -. updates .-> S2
```

### Scoring & Selection

```mermaid
flowchart LR
    subgraph Task Skills
        TS1["python_coding (importance: 0.9)"]
        TS2["api_design (importance: 0.5)"]
    end

    subgraph Agent Profiles
        AP1["CodeExpert\npython: 0.95, api: 0.90"]
        AP2["TechWriter\npython: 0.30, api: 0.50"]
        AP3["Researcher\npython: 0.60, api: 0.30"]
    end

    Task Skills --> SCORE["Scoring Formula\nscore = sum(w_c * competence * importance\n+ w_cost * norm_cost * importance)\n/ total_importance"]
    Agent Profiles --> SCORE

    SCORE --> R1["CodeExpert: 0.68"]
    SCORE --> R2["TechWriter: 0.31"]
    SCORE --> R3["Researcher: 0.42"]

    R1 --> SEL["Select Top-K"]
    R2 --> SEL
    R3 --> SEL
    SEL --> WIN["CodeExpert selected"]
```

### Execution Modes

```mermaid
flowchart TD
    SEL["Selected Agents"] --> CHECK{top_k_agents}
    CHECK -- "k = 1" --> SINGLE["Direct Execution\nagent.run(task)"]
    CHECK -- "k > 1" --> MULTI["Concurrent Execution\nThreadPoolExecutor"]
    MULTI --> A1["Agent 1"]
    MULTI --> A2["Agent 2"]
    MULTI --> A3["Agent N"]
    A1 --> COLLECT["Collect Results"]
    A2 --> COLLECT
    A3 --> COLLECT
    SINGLE --> OUTPUT["Output"]
    COLLECT --> OUTPUT
```

## Best Practices

### Agent Design

- **Write descriptive agent descriptions** — The auto-generated skill handbook is only as good as your agent descriptions. Be specific about what each agent can do.
- **Use distinct specializations** — Agents with overlapping skills reduce the effectiveness of skill-based routing. Make each agent clearly specialized.
- **Keep system prompts focused** — System prompts should reinforce the agent's specialization, not try to make the agent a generalist.

### Tuning Weights

- **Default (0.7 competence / 0.3 cost)** — Good for most use cases where quality matters more than cost.
- **High competence weight (0.9 / 0.1)** — Use when quality is critical and cost is not a concern.
- **Balanced (0.5 / 0.5)** — Use when you want a balance between quality and cost efficiency.
- **High cost weight (0.3 / 0.7)** — Use for high-volume, cost-sensitive workloads where "good enough" is acceptable.

### Learning Configuration

- **`learning_rate=0.1`** (default) — Slow adaptation, stable profiles. Good for production.
- **`learning_rate=0.3`** — Faster adaptation. Good for initial calibration of a new team.
- **`max_loops=1`** — Single pass, no refinement. Best for simple tasks.
- **`max_loops=2-3`** — Execute, evaluate, refine. Good for complex tasks that benefit from iterative improvement.

### Error Handling

```python
try:
    result = orchestra.run(task)
except ValueError as e:
    # Configuration errors (no agents, invalid weights)
    print(f"Configuration error: {e}")
except Exception as e:
    # Execution errors (LLM failures, agent errors)
    print(f"Execution error: {e}")
```

### Inspecting Routing Decisions

Enable `verbose=True` and `print_on=True` to see detailed routing information:

```python
orchestra = SkillOrchestra(
    agents=agents,
    verbose=True,    # Logs skill inference and scoring details
    print_on=True,   # Prints formatted panels to console
)
```

### Saving and Loading Handbooks

```python
import json
from swarms.structs.skill_orchestra import SkillHandbook

# Save a tuned handbook
handbook_dict = orchestra.get_handbook()
with open("my_handbook.json", "w") as f:
    json.dump(handbook_dict, f, indent=2)

# Load and reuse later
with open("my_handbook.json") as f:
    data = json.load(f)

handbook = SkillHandbook.model_validate(data)
orchestra = SkillOrchestra(
    agents=agents,
    skill_handbook=handbook,
    auto_generate_skills=False,
)
```
