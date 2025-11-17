# AgentJudge

A specialized agent for evaluating and judging outputs from other agents or systems. Acts as a quality control mechanism providing objective assessments and feedback.

Based on the research paper: **"Agent-as-a-Judge: Evaluate Agents with Agents"** - [arXiv:2410.10934](https://arxiv.org/abs/2410.10934)

## Overview

The AgentJudge is designed to evaluate and critique outputs from other AI agents, providing structured feedback on quality, accuracy, and areas for improvement. It supports both single-shot evaluations and iterative refinement through multiple evaluation loops with context building.

Key capabilities:

| Capability                   | Description                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------------------|
| **Quality Assessment**        | Evaluates correctness, clarity, and completeness of agent outputs                            |
| **Structured Feedback**       | Provides detailed critiques with strengths, weaknesses, and suggestions                      |
| **Multimodal Support**        | Can evaluate text outputs alongside images                                                   |
| **Context Building**          | Maintains evaluation context across multiple iterations                                      |
| **Custom Evaluation Criteria**| Supports weighted evaluation criteria for domain-specific assessments                        |
| **Batch Processing**          | Efficiently processes multiple evaluations                                                   |

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
    system_prompt: str = None,
    model_name: str = "openai/o1",
    max_loops: int = 1,
    verbose: bool = False,
    evaluation_criteria: Optional[Dict[str, float]] = None,
    return_score: bool = False,
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
| `system_prompt` | `str` | `None` | Custom system instructions (uses default if None) |
| `model_name` | `str` | `"openai/o1"` | LLM model for evaluation |
| `max_loops` | `int` | `1` | Maximum evaluation iterations |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `evaluation_criteria` | `Optional[Dict[str, float]]` | `None` | Dictionary of evaluation criteria and weights |
| `return_score` | `bool` | `False` | Whether to return a numerical score instead of full conversation |

### Methods

#### step()

```python
step(
    task: str = None,
    img: Optional[str] = None
) -> str
```

Processes a single task and returns the agent's evaluation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `None` | Single task/output to evaluate |
| `img` | `Optional[str]` | `None` | Path to image for multimodal evaluation |

**Returns:** `str` - Detailed evaluation response

**Raises:** `ValueError` - If no task is provided

#### run()

```python
run(
    task: str = None,
    img: Optional[str] = None
) -> Union[str, int]
```

Executes evaluation in multiple iterations with context building.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `None` | Single task/output to evaluate |
| `img` | `Optional[str]` | `None` | Path to image for multimodal evaluation |

**Returns:** 
- `str` - Full conversation context if `return_score=False` (default)
- `int` - Numerical reward score if `return_score=True`

#### run_batched()

```python
run_batched(
    tasks: Optional[List[str]] = None
) -> List[Union[str, int]]
```

Executes batch evaluation of multiple tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tasks` | `Optional[List[str]]` | `None` | List of tasks/outputs to evaluate |

**Returns:** `List[Union[str, int]]` - Evaluation responses for each task

## Examples

### Basic Evaluation

```python
from swarms.agents.agent_judge import AgentJudge

# Initialize the agent judge
judge = AgentJudge(
    agent_name="quality-judge",
    model_name="gpt-4",
    max_loops=2
)

# Example agent output to evaluate
agent_output = "The capital of France is Paris. The city is known for its famous Eiffel Tower and delicious croissants. The population is approximately 2.1 million people."

# Run evaluation with context building
evaluations = judge.run(task=agent_output)
```

### Technical Evaluation with Custom Criteria

```python
from swarms.agents.agent_judge import AgentJudge

# Initialize the agent judge with custom evaluation criteria
judge = AgentJudge(
    agent_name="technical-judge",
    model_name="gpt-4",
    max_loops=1,
    evaluation_criteria={
        "accuracy": 0.4,
        "completeness": 0.3,
        "clarity": 0.2,
        "logic": 0.1,
    },
)

# Example technical agent output to evaluate
technical_output = "To solve the quadratic equation x² + 5x + 6 = 0, we can use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a. Here, a=1, b=5, c=6. Substituting: x = (-5 ± √(25 - 24)) / 2 = (-5 ± √1) / 2 = (-5 ± 1) / 2. So x = -2 or x = -3."

# Run evaluation with context building
evaluations = judge.run(task=technical_output)
```

### Creative Content Evaluation

```python
from swarms.agents.agent_judge import AgentJudge

# Initialize the agent judge for creative content evaluation
judge = AgentJudge(
    agent_name="creative-judge",
    model_name="gpt-4",
    max_loops=2,
    evaluation_criteria={
        "creativity": 0.4,
        "originality": 0.3,
        "engagement": 0.2,
        "coherence": 0.1,
    },
)

# Example creative agent output to evaluate
creative_output = "The moon hung like a silver coin in the velvet sky, casting shadows that danced with the wind. Ancient trees whispered secrets to the stars, while time itself seemed to pause in reverence of this magical moment. The world held its breath, waiting for the next chapter of the eternal story."

# Run evaluation with context building
evaluations = judge.run(task=creative_output)
```

### Single Task Evaluation

```python
from swarms.agents.agent_judge import AgentJudge

# Initialize with default settings
judge = AgentJudge()

# Single task evaluation
result = judge.step(task="The answer is 42.")
```

### Multimodal Evaluation

```python
from swarms.agents.agent_judge import AgentJudge

judge = AgentJudge()

# Evaluate with image
evaluation = judge.step(
    task="Describe what you see in this image",
    img="path/to/image.jpg"
)
```

### Batch Processing

```python
from swarms.agents.agent_judge import AgentJudge

judge = AgentJudge()

# Batch evaluation
tasks = [
    "The capital of France is Paris.",
    "2 + 2 = 4",
    "The Earth is flat."
]

# Each task evaluated independently
evaluations = judge.run_batched(tasks=tasks)
```

### Scoring Mode

```python
from swarms.agents.agent_judge import AgentJudge

# Initialize with scoring enabled
judge = AgentJudge(
    agent_name="scoring-judge",
    model_name="gpt-4",
    max_loops=2,
    return_score=True
)

# Get numerical score instead of full conversation
score = judge.run(task="This is a correct and well-explained answer.")
# Returns: 1 (if positive keywords found) or 0
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