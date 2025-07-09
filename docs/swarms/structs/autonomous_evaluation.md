# Autonomous Evaluation for AutoSwarmBuilder

## Overview

The Autonomous Evaluation feature enhances the AutoSwarmBuilder with iterative improvement capabilities. This system creates a feedback loop where agents are evaluated, critiqued, and improved automatically through multiple iterations, leading to better performance and higher quality outputs.

## Key Features

- **Iterative Improvement**: Automatically improves agent performance across multiple iterations
- **Multi-dimensional Evaluation**: Evaluates agents on accuracy, helpfulness, coherence, and instruction adherence
- **Autonomous Feedback Loop**: Uses AI judges and critics to provide detailed feedback
- **Performance Tracking**: Tracks improvement metrics across iterations
- **Configurable Evaluation**: Customizable evaluation parameters and thresholds

## Architecture

The autonomous evaluation system consists of several key components:

### 1. Evaluation Judges
- **CouncilAsAJudge**: Multi-agent evaluation system that assesses performance across dimensions
- **Improvement Strategist**: Analyzes feedback and suggests specific improvements

### 2. Feedback Loop
1. **Build Agents** → Create initial agent configuration
2. **Execute Task** → Run the swarm on the given task
3. **Evaluate Output** → Judge performance across multiple dimensions
4. **Generate Feedback** → Create detailed improvement suggestions
5. **Improve Agents** → Build enhanced agents based on feedback
6. **Repeat** → Continue until convergence or max iterations

### 3. Performance Tracking
- Dimension scores (0.0 to 1.0 scale)
- Strengths and weaknesses identification
- Improvement suggestions
- Best iteration tracking

## Usage

### Basic Usage with Evaluation

```python
from swarms.structs.auto_swarm_builder import (
    AutoSwarmBuilder,
    IterativeImprovementConfig,
)

# Configure evaluation parameters
eval_config = IterativeImprovementConfig(
    max_iterations=3,
    improvement_threshold=0.1,
    evaluation_dimensions=["accuracy", "helpfulness", "coherence"],
    use_judge_agent=True,
    store_all_iterations=True,
)

# Create AutoSwarmBuilder with evaluation enabled
swarm = AutoSwarmBuilder(
    name="SmartResearchSwarm",
    description="A self-improving research swarm",
    enable_evaluation=True,
    evaluation_config=eval_config,
)

# Run with autonomous evaluation
task = "Research the latest developments in quantum computing"
result = swarm.run(task)

# Access evaluation results
evaluation_history = swarm.get_evaluation_results()
best_iteration = swarm.get_best_iteration()
```

### Configuration Options

#### IterativeImprovementConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 3 | Maximum number of improvement iterations |
| `improvement_threshold` | float | 0.1 | Minimum improvement required to continue |
| `evaluation_dimensions` | List[str] | ["accuracy", "helpfulness", "coherence", "instruction_adherence"] | Dimensions to evaluate |
| `use_judge_agent` | bool | True | Whether to use CouncilAsAJudge for evaluation |
| `store_all_iterations` | bool | True | Whether to store results from all iterations |

#### AutoSwarmBuilder Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_evaluation` | bool | False | Enable autonomous evaluation |
| `evaluation_config` | IterativeImprovementConfig | None | Evaluation configuration |

## Evaluation Dimensions

### Accuracy
Evaluates factual correctness and reliability of information:
- Cross-references factual claims
- Identifies inconsistencies
- Detects technical inaccuracies
- Flags unsupported assertions

### Helpfulness
Assesses practical value and problem-solving efficacy:
- Alignment with user intent
- Solution feasibility
- Inclusion of essential context
- Proactive addressing of follow-ups

### Coherence
Analyzes structural integrity and logical flow:
- Information hierarchy
- Transition effectiveness
- Logical argument structure
- Clear connections between ideas

### Instruction Adherence
Measures compliance with requirements:
- Coverage of prompt requirements
- Adherence to constraints
- Output format compliance
- Scope appropriateness

## Examples

### Research Task with Evaluation

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder, IterativeImprovementConfig

# Configure for research tasks
config = IterativeImprovementConfig(
    max_iterations=4,
    improvement_threshold=0.15,
    evaluation_dimensions=["accuracy", "helpfulness", "coherence"],
)

swarm = AutoSwarmBuilder(
    name="ResearchSwarm",
    description="Advanced research analysis swarm",
    enable_evaluation=True,
    evaluation_config=config,
)

task = """
Analyze the current state of renewable energy technology,
including market trends, technological breakthroughs,
and policy implications for the next decade.
"""

result = swarm.run(task)

# Print evaluation summary
for i, eval_result in enumerate(swarm.get_evaluation_results()):
    score = sum(eval_result.evaluation_scores.values()) / len(eval_result.evaluation_scores)
    print(f"Iteration {i+1}: Overall Score = {score:.3f}")
```

### Content Creation with Evaluation

```python
config = IterativeImprovementConfig(
    max_iterations=3,
    evaluation_dimensions=["helpfulness", "coherence", "instruction_adherence"],
)

swarm = AutoSwarmBuilder(
    name="ContentCreationSwarm",
    enable_evaluation=True,
    evaluation_config=config,
)

task = """
Create a comprehensive marketing plan for a new SaaS product
targeting small businesses, including market analysis,
positioning strategy, and go-to-market tactics.
"""

result = swarm.run(task)
```

## Evaluation Results

### EvaluationResult Model

```python
class EvaluationResult(BaseModel):
    iteration: int                           # Iteration number
    task: str                               # Original task
    output: str                             # Swarm output
    evaluation_scores: Dict[str, float]     # Dimension scores (0.0-1.0)
    feedback: str                           # Detailed feedback
    strengths: List[str]                    # Identified strengths
    weaknesses: List[str]                   # Identified weaknesses
    suggestions: List[str]                  # Improvement suggestions
```

### Accessing Results

```python
# Get all evaluation results
evaluations = swarm.get_evaluation_results()

# Get best performing iteration
best = swarm.get_best_iteration()

# Print detailed results
for eval_result in evaluations:
    print(f"Iteration {eval_result.iteration}:")
    print(f"  Overall Score: {sum(eval_result.evaluation_scores.values()):.3f}")
    
    for dimension, score in eval_result.evaluation_scores.items():
        print(f"  {dimension}: {score:.3f}")
    
    print(f"  Strengths: {len(eval_result.strengths)}")
    print(f"  Weaknesses: {len(eval_result.weaknesses)}")
    print(f"  Suggestions: {len(eval_result.suggestions)}")
```

## Best Practices

### 1. Task Complexity Matching
- Simple tasks: 1-2 iterations
- Medium tasks: 2-3 iterations
- Complex tasks: 3-5 iterations

### 2. Evaluation Dimension Selection
- **Research tasks**: accuracy, helpfulness, coherence
- **Creative tasks**: helpfulness, coherence, instruction_adherence
- **Analysis tasks**: accuracy, coherence, instruction_adherence
- **All-purpose**: All four dimensions

### 3. Threshold Configuration
- **Conservative**: 0.05-0.10 (more iterations)
- **Balanced**: 0.10-0.15 (moderate iterations)
- **Aggressive**: 0.15-0.25 (fewer iterations)

### 4. Performance Monitoring
```python
# Track improvement across iterations
scores = []
for eval_result in swarm.get_evaluation_results():
    overall_score = sum(eval_result.evaluation_scores.values()) / len(eval_result.evaluation_scores)
    scores.append(overall_score)

# Calculate improvement
if len(scores) > 1:
    improvement = scores[-1] - scores[0]
    print(f"Total improvement: {improvement:.3f}")
```

## Advanced Configuration

### Custom Evaluation Dimensions

```python
custom_config = IterativeImprovementConfig(
    max_iterations=3,
    evaluation_dimensions=["accuracy", "creativity", "practicality"],
    improvement_threshold=0.12,
)

# Note: Custom dimensions require corresponding keywords
# in the evaluation system
```

### Disabling Judge Agent (Performance Mode)

```python
performance_config = IterativeImprovementConfig(
    max_iterations=2,
    use_judge_agent=False,  # Faster but less detailed evaluation
    evaluation_dimensions=["helpfulness", "coherence"],
)
```

## Troubleshooting

### Common Issues

1. **High iteration count without improvement**
   - Lower the improvement threshold
   - Reduce max_iterations
   - Check evaluation dimension relevance

2. **Evaluation system errors**
   - Verify OpenAI API key configuration
   - Check network connectivity
   - Ensure proper model access

3. **Inconsistent scoring**
   - Use more evaluation dimensions
   - Increase iteration count
   - Review task complexity

### Performance Optimization

1. **Reduce evaluation overhead**
   - Set `use_judge_agent=False` for faster evaluation
   - Limit evaluation dimensions
   - Reduce max_iterations

2. **Improve convergence**
   - Adjust improvement threshold
   - Add more specific evaluation dimensions
   - Enhance task clarity

## Integration Examples

### With Existing Workflows

```python
def research_pipeline(topic: str):
    """Research pipeline with autonomous evaluation"""
    
    config = IterativeImprovementConfig(
        max_iterations=3,
        evaluation_dimensions=["accuracy", "helpfulness"],
    )
    
    swarm = AutoSwarmBuilder(
        name=f"Research-{topic}",
        enable_evaluation=True,
        evaluation_config=config,
    )
    
    result = swarm.run(f"Research {topic}")
    
    # Return both result and evaluation metrics
    best_iteration = swarm.get_best_iteration()
    return {
        "result": result,
        "quality_score": sum(best_iteration.evaluation_scores.values()),
        "iterations": len(swarm.get_evaluation_results()),
    }
```

### Batch Processing with Evaluation

```python
def batch_process_with_evaluation(tasks: List[str]):
    """Process multiple tasks with evaluation tracking"""
    
    results = []
    for task in tasks:
        swarm = AutoSwarmBuilder(
            enable_evaluation=True,
            evaluation_config=IterativeImprovementConfig(max_iterations=2)
        )
        
        result = swarm.run(task)
        best = swarm.get_best_iteration()
        
        results.append({
            "task": task,
            "result": result,
            "quality": sum(best.evaluation_scores.values()) if best else 0,
        })
    
    return results
```

## Future Enhancements

- **Custom evaluation metrics**: User-defined evaluation criteria
- **Evaluation dataset integration**: Benchmark-based evaluation
- **Real-time feedback**: Live evaluation during execution
- **Ensemble evaluation**: Multiple evaluation models
- **Performance prediction**: ML-based iteration outcome prediction

## Conclusion

The Autonomous Evaluation feature transforms the AutoSwarmBuilder into a self-improving system that automatically enhances agent performance through iterative feedback loops. This leads to higher quality outputs, better task completion, and more reliable AI agent performance across diverse use cases.