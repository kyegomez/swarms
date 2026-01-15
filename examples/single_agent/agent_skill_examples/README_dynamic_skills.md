# Dynamic Skills Loader

The Dynamic Skills Loader provides intelligent skill loading for agents based on task similarity. Instead of loading all available skills into memory, it dynamically selects and loads only the skills that are most relevant to the current task.

## How It Works

1. **Task Specification**: Provide a `task` parameter when creating the agent to enable dynamic skills loading
2. **Initialization**: During agent initialization, `handle_skills(task)` is called
3. **Task Tokenization**: The task description is tokenized and embedded using term frequency
4. **Skill Matching**: Each available skill's description is compared to the task using cosine similarity
5. **Selective Loading**: Only skills with similarity scores above the threshold are loaded into the agent

## Architecture

- **`handle_skills(task=None)`**: Main skills handler that takes an optional task parameter
- **`_load_static_skills()`**: Loads all skills at initialization when no task is provided
- **`_load_dynamic_skills_for_task(task)`**: Loads relevant skills based on task similarity
- **`DynamicSkillsLoader`**: Core similarity calculation engine

## Features

- **Efficient Memory Usage**: Only loads relevant skills instead of all available skills
- **Task-Aware**: Skills are selected based on semantic similarity to the task
- **Configurable Threshold**: Adjust similarity threshold for more or less selective loading
- **Cosine Similarity**: Uses mathematical cosine similarity calculation with the math library

## Usage

### Basic Usage

```python
from swarms import Agent

# Create agent with dynamic skills loading
agent = Agent(
    agent_name="Dynamic Agent",
    model_name="gpt-4o",
    skills_dir="./skills",  # Directory containing skill folders
    task="analyze financial data and create charts",  # Load skills relevant to this task
)

# Skills are loaded during initialization based on the task
# Run tasks normally
response = agent.run("Calculate ROI and profit margins")
```

### Advanced Configuration

```python
from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader

# Create custom loader with specific threshold
loader = DynamicSkillsLoader(
    skills_dir="./skills",
    similarity_threshold=0.4  # Higher = more selective
)

# Check which skills would be loaded for a task
relevant_skills = loader.load_relevant_skills("Create data visualizations")
print([skill["name"] for skill in relevant_skills])
```

## Skill Format

Skills must be stored in SKILL.md files with YAML frontmatter:

```markdown
---
name: financial-analysis
description: Create financial analysis reports and projections
---

# Financial Analysis Skill

Your instructions here...
```

## Parameters

- `skills_dir`: Path to directory containing skill folders
- `similarity_threshold`: Minimum similarity score (0-1) for skill loading (default: 0.3)
- `dynamic_skills_loading`: Enable dynamic loading in Agent (default: False)

## Similarity Calculation

The system uses cosine similarity between term frequency vectors:

1. Text is tokenized (lowercased, punctuation removed)
2. Term frequency embeddings are created
3. Cosine similarity is calculated using the math library
4. Skills above threshold are loaded

## Example Output

```
Task: "analyze financial data and create charts"
Financial analysis similarity: 0.500
Data visualization similarity: 0.309
Loaded skills: ['financial-analysis', 'data-visualization']
```

## Benefits

- **Performance**: Faster startup and lower memory usage
- **Relevance**: Only contextually appropriate skills are loaded
- **Scalability**: Can handle large numbers of skills efficiently
- **Flexibility**: Easy to adjust similarity thresholds