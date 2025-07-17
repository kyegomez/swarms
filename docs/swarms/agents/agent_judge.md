# AgentJudge

A specialized agent for evaluating and judging outputs from other agents or systems. Acts as a quality control mechanism providing objective assessments and feedback.

Based on the research paper: **"Agent-as-a-Judge: Evaluate Agents with Agents"** - [arXiv:2410.10934](https://arxiv.org/abs/2410.10934)

## Overview

The AgentJudge is designed to evaluate and critique outputs from other AI agents, providing structured feedback on quality, accuracy, and areas for improvement. It supports both single-shot evaluations and iterative refinement through multiple evaluation loops with context building.

Key capabilities:

- **Quality Assessment**: Evaluates correctness, clarity, and completeness of agent outputs

- **Structured Feedback**: Provides detailed critiques with strengths, weaknesses, and suggestions

- **Multimodal Support**: Can evaluate text outputs alongside images

- **Context Building**: Maintains evaluation context across multiple iterations

- **Batch Processing**: Efficiently processes multiple evaluations

## Architecture

```mermaid
graph TD
    A[Input Task] --> B[AgentJudge]
    B --> C{Evaluation Mode}

    C -->|step()| D[Single Eval]
    C -->|run()| E[Iterative Eval]
    C -->|run_batched()| F[Batch Eval]

    D --> G[Agent Core]
    E --> G
    F --> G

    G --> H[LLM Model]
    H --> I[Quality Analysis]
    I --> J[Feedback & Output]

    subgraph "Feedback Details"
        N[Strengths]
        O[Weaknesses]
        P[Improvements]
        Q[Accuracy Check]
    end

    J --> N
    J --> O
    J --> P
    J --> Q

```

## Class Reference

### Constructor

```python
AgentJudge(
    id: str = str(uuid.uuid4()),
    agent_name: str = "Agent Judge",
    description: str = "You're an expert AI agent judge...",
    system_prompt: str = AGENT_JUDGE_PROMPT,
    model_name: str = "openai/o1",
    max_loops: int = 1,
    verbose: bool = False,
    *args,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | `str` | `str(uuid.uuid4())` | Unique identifier for the judge instance |
| `agent_name` | `str` | `"Agent Judge"` | Name of the agent judge |
| `description` | `str` | `"You're an expert AI agent judge..."` | Description of the agent's role |
| `system_prompt` | `str` | `AGENT_JUDGE_PROMPT` | System instructions for evaluation |
| `model_name` | `str` | `"openai/o1"` | LLM model for evaluation |
| `max_loops` | `int` | `1` | Maximum evaluation iterations |
| `verbose` | `bool` | `False` | Enable verbose logging |

### Methods

#### step()

```python
step(
    task: str = None,
    tasks: Optional[List[str]] = None,
    img: Optional[str] = None
) -> str
```

Processes a single task or list of tasks and returns evaluation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `None` | Single task/output to evaluate |
| `tasks` | `List[str]` | `None` | List of tasks/outputs to evaluate |
| `img` | `str` | `None` | Path to image for multimodal evaluation |

**Returns:** `str` - Detailed evaluation response

#### run()

```python
run(
    task: str = None,
    tasks: Optional[List[str]] = None,
    img: Optional[str] = None
) -> List[str]
```

Executes evaluation in multiple iterations with context building.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `None` | Single task/output to evaluate |
| `tasks` | `List[str]` | `None` | List of tasks/outputs to evaluate |
| `img` | `str` | `None` | Path to image for multimodal evaluation |

**Returns:** `List[str]` - List of evaluation responses from each iteration

#### run_batched()

```python
run_batched(
    tasks: Optional[List[str]] = None,
    imgs: Optional[List[str]] = None
) -> List[List[str]]
```

Executes batch evaluation of multiple tasks with corresponding images.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `List[str]` | `None` | List of tasks/outputs to evaluate |
| `imgs` | `List[str]` | `None` | List of image paths (same length as tasks) |

**Returns:** `List[List[str]]` - Evaluation responses for each task

## Examples

### Basic Usage

```python
from swarms import AgentJudge

# Initialize with default settings
judge = AgentJudge()

# Single task evaluation
result = judge.step(task="The capital of France is Paris.")
print(result)
```

### Custom Configuration

```python
from swarms import AgentJudge

# Custom judge configuration
judge = AgentJudge(
    agent_name="content-evaluator",
    model_name="gpt-4",
    max_loops=3,
    verbose=True
)

# Evaluate multiple outputs
outputs = [
    "Agent CalculusMaster: The integral of x^2 + 3x + 2 is (1/3)x^3 + (3/2)x^2 + 2x + C",
    "Agent DerivativeDynamo: The derivative of sin(x) is cos(x)",
    "Agent LimitWizard: The limit of sin(x)/x as x approaches 0 is 1"
]

evaluation = judge.step(tasks=outputs)
print(evaluation)
```

### Iterative Evaluation with Context

```python
from swarms import AgentJudge

# Multiple iterations with context building
judge = AgentJudge(max_loops=3)

# Each iteration builds on previous context
evaluations = judge.run(task="Agent output: 2+2=5")
for i, eval_result in enumerate(evaluations):
    print(f"Iteration {i+1}: {eval_result}\n")
```

### Multimodal Evaluation

```python
from swarms import AgentJudge

judge = AgentJudge()

# Evaluate with image
evaluation = judge.step(
    task="Describe what you see in this image",
    img="path/to/image.jpg"
)
print(evaluation)
```

### Batch Processing

```python
from swarms import AgentJudge

judge = AgentJudge()

# Batch evaluation with images
tasks = [
    "Describe this chart",
    "What's the main trend?",
    "Any anomalies?"
]
images = [
    "chart1.png",
    "chart2.png", 
    "chart3.png"
]

# Each task evaluated independently
evaluations = judge.run_batched(tasks=tasks, imgs=images)
for i, task_evals in enumerate(evaluations):
    print(f"Task {i+1} evaluations: {task_evals}")
```

## Reference

```bibtex
@misc{zhuge2024agentasajudgeevaluateagentsagents,
    title={Agent-as-a-Judge: Evaluate Agents with Agents}, 
    author={Mingchen Zhuge and Changsheng Zhao and Dylan Ashley and Wenyi Wang and Dmitrii Khizbullin and Yunyang Xiong and Zechun Liu and Ernie Chang and Raghuraman Krishnamoorthi and Yuandong Tian and Yangyang Shi and Vikas Chandra and Jürgen Schmidhuber},
    year={2024},
    eprint={2410.10934},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2410.10934}
}
```