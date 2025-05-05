# SwarmMatcher

SwarmMatcher is a tool for automatically matching tasks to the most appropriate swarm type based on their semantic similarity.

## Overview

The SwarmMatcher utilizes transformer-based embeddings to determine the best swarm architecture for a given task. By analyzing the semantic meaning of task descriptions and comparing them to known swarm types, it can intelligently select the optimal swarm configuration for any task.

## Installation

SwarmMatcher is included in the Swarms package. To use it, simply import it from the library:

```python
from swarms.structs.swarm_matcher import SwarmMatcher, SwarmMatcherConfig, SwarmType
```

## Basic Usage

```python
from swarms.structs.swarm_matcher import swarm_matcher

# Use the simplified function to match a task to a swarm type
swarm_type = swarm_matcher("Analyze this dataset and create visualizations")
print(f"Selected swarm type: {swarm_type}")
```

## Advanced Usage

For more control over the matching process, you can create and configure your own SwarmMatcher instance:

```python
from swarms.structs.swarm_matcher import SwarmMatcher, SwarmMatcherConfig, SwarmType, initialize_swarm_types

# Create a configuration
config = SwarmMatcherConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=512
)

# Initialize the matcher
matcher = SwarmMatcher(config)

# Add default swarm types
initialize_swarm_types(matcher)

# Add a custom swarm type
custom_swarm = SwarmType(
    name="CustomSwarm",
    description="A specialized swarm for handling specific domain tasks with expert knowledge."
)
matcher.add_swarm_type(custom_swarm)

# Find the best match for a task
best_match, score = matcher.find_best_match("Process natural language and extract key insights")
print(f"Best match: {best_match}, Score: {score}")

# Auto-select a swarm type
selected_swarm = matcher.auto_select_swarm("Create data visualizations from this CSV file")
print(f"Selected swarm: {selected_swarm}")
```

## Available Swarm Types

SwarmMatcher comes with several pre-defined swarm types:

| Swarm Type | Description |
| ---------- | ----------- |
| AgentRearrange | Optimize agent order and rearrange flow for multi-step tasks, ensuring efficient task allocation and minimizing bottlenecks. |
| MixtureOfAgents | Combine diverse expert agents for comprehensive analysis, fostering a collaborative approach to problem-solving and leveraging individual strengths. |
| SpreadSheetSwarm | Collaborative data processing and analysis in a spreadsheet-like environment, facilitating real-time data sharing and visualization. |
| SequentialWorkflow | Execute tasks in a step-by-step, sequential process workflow, ensuring a logical and methodical approach to task execution. |
| ConcurrentWorkflow | Process multiple tasks or data sources concurrently in parallel, maximizing productivity and reducing processing time. |

## API Reference

### SwarmType

A class representing a type of swarm with its name and description.

```python
class SwarmType(BaseModel):
    name: str
    description: str
    embedding: Optional[List[float]] = Field(default=None, exclude=True)
```

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| name | str | The name of the swarm type |
| description | str | A detailed description of the swarm type's capabilities and ideal use cases |
| embedding | Optional[List[float]] | The generated embedding vector for this swarm type (auto-populated) |

### SwarmMatcherConfig

Configuration settings for the SwarmMatcher.

```python
class SwarmMatcherConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 512
```

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| model_name | str | "sentence-transformers/all-MiniLM-L6-v2" | The transformer model to use for embeddings |
| embedding_dim | int | 512 | The dimension of the embedding vectors |

### SwarmMatcher

The main class for matching tasks to swarm types.

```python
class SwarmMatcher:
    def __init__(self, config: SwarmMatcherConfig)
    def get_embedding(self, text: str) -> np.ndarray
    def add_swarm_type(self, swarm_type: SwarmType)
    def find_best_match(self, task: str) -> Tuple[str, float]
    def auto_select_swarm(self, task: str) -> str
    def run_multiple(self, tasks: List[str]) -> List[str]
    def save_swarm_types(self, filename: str)
    def load_swarm_types(self, filename: str)
```

#### Methods

##### `__init__(config: SwarmMatcherConfig)`

Initializes the SwarmMatcher with a configuration.

##### `get_embedding(text: str) -> np.ndarray`

Generates an embedding vector for a given text using the configured model.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| text | str | The text to embed |
| Returns | np.ndarray | The embedding vector |

##### `add_swarm_type(swarm_type: SwarmType)`

Adds a swarm type to the matcher, generating an embedding for its description.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| swarm_type | SwarmType | The swarm type to add |

##### `find_best_match(task: str) -> Tuple[str, float]`

Finds the best matching swarm type for a given task.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| task | str | The task description |
| Returns | Tuple[str, float] | The name of the best matching swarm type and the similarity score |

##### `auto_select_swarm(task: str) -> str`

Automatically selects the best swarm type for a given task.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| task | str | The task description |
| Returns | str | The name of the selected swarm type |

##### `run_multiple(tasks: List[str]) -> List[str]`

Matches multiple tasks to swarm types in batch.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| tasks | List[str] | A list of task descriptions |
| Returns | List[str] | A list of selected swarm type names |

##### `save_swarm_types(filename: str)`

Saves the registered swarm types to a JSON file.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| filename | str | Path where the swarm types will be saved |

##### `load_swarm_types(filename: str)`

Loads swarm types from a JSON file.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| filename | str | Path to the JSON file containing swarm types |

## Examples

### Simple Matching

```python
from swarms.structs.swarm_matcher import swarm_matcher

# Match tasks to swarm types
tasks = [
    "Analyze this dataset and create visualizations",
    "Coordinate multiple agents to tackle different aspects of a problem",
    "Process these 10 PDF files in sequence",
    "Handle these data processing tasks in parallel"
]

for task in tasks:
    swarm_type = swarm_matcher(task)
    print(f"Task: {task}")
    print(f"Selected swarm: {swarm_type}\n")
```

### Custom Swarm Types

```python
from swarms.structs.swarm_matcher import SwarmMatcher, SwarmMatcherConfig, SwarmType

# Create configuration and matcher
config = SwarmMatcherConfig()
matcher = SwarmMatcher(config)

# Define custom swarm types
swarm_types = [
    SwarmType(
        name="DataAnalysisSwarm",
        description="Specialized in processing and analyzing large datasets, performing statistical analysis, and extracting insights from complex data."
    ),
    SwarmType(
        name="CreativeWritingSwarm",
        description="Optimized for creative content generation, storytelling, and producing engaging written material with consistent style and tone."
    ),
    SwarmType(
        name="ResearchSwarm",
        description="Focused on deep research tasks, synthesizing information from multiple sources, and producing comprehensive reports on complex topics."
    )
]

# Add swarm types
for swarm_type in swarm_types:
    matcher.add_swarm_type(swarm_type)

# Save the swarm types for future use
matcher.save_swarm_types("custom_swarm_types.json")

# Use the matcher
task = "Research quantum computing advances in the last 5 years"
best_match = matcher.auto_select_swarm(task)
print(f"Selected swarm type: {best_match}")
```

## How It Works

SwarmMatcher uses a transformer-based model to generate embeddings (vector representations) of both the task descriptions and the swarm type descriptions. It then calculates the similarity between these embeddings to determine which swarm type is most semantically similar to the given task.

The matching process follows these steps:

1. The task description is converted to an embedding vector
2. Each swarm type's description is converted to an embedding vector
3. The similarity between the task embedding and each swarm type embedding is calculated
4. The swarm type with the highest similarity score is selected

This approach ensures that the matcher can understand the semantic meaning of tasks, not just keyword matching, resulting in more accurate swarm type selection.
