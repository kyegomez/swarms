# Hierarchical Structured Communication Framework

The Hierarchical Structured Communication Framework implements the "Talk Structurally, Act Hierarchically" approach for LLM multi-agent systems, based on the research paper arXiv:2502.11098.

## Overview

This framework provides:
- **Structured Communication Protocol** with Message (M_ij), Background (B_ij), and Intermediate Output (I_ij)
- **Hierarchical Evaluation System** with supervisor coordination
- **Specialized Agent Classes** for different roles
- **Main Swarm Orchestrator** for workflow management

## Key Components

### Agent Classes
- `HierarchicalStructuredCommunicationGenerator` - Creates initial content
- `HierarchicalStructuredCommunicationEvaluator` - Evaluates content quality
- `HierarchicalStructuredCommunicationRefiner` - Improves content based on feedback
- `HierarchicalStructuredCommunicationSupervisor` - Coordinates workflow

### Main Framework
- `HierarchicalStructuredCommunicationFramework` - Main orchestrator class
- `HierarchicalStructuredCommunicationSwarm` - Convenience alias

## Quick Start

```python
from swarms.structs.hierarchical_structured_communication_framework import (
    HierarchicalStructuredCommunicationFramework,
    HierarchicalStructuredCommunicationGenerator,
    HierarchicalStructuredCommunicationEvaluator,
    HierarchicalStructuredCommunicationRefiner,
    HierarchicalStructuredCommunicationSupervisor
)

# Create specialized agents
generator = HierarchicalStructuredCommunicationGenerator(
    agent_name="ContentGenerator"
)

evaluator = HierarchicalStructuredCommunicationEvaluator(
    agent_name="QualityEvaluator"
)

refiner = HierarchicalStructuredCommunicationRefiner(
    agent_name="ContentRefiner"
)

supervisor = HierarchicalStructuredCommunicationSupervisor(
    agent_name="WorkflowSupervisor"
)

# Create the framework
framework = HierarchicalStructuredCommunicationFramework(
    name="MyFramework",
    supervisor=supervisor,
    generators=[generator],
    evaluators=[evaluator],
    refiners=[refiner],
    max_loops=3
)

# Run the workflow
result = framework.run("Create a comprehensive analysis of AI trends in 2024")
```

## Basic Usage

```python
from swarms.structs.hierarchical_structured_communication_framework import (
    HierarchicalStructuredCommunicationFramework,
    HierarchicalStructuredCommunicationGenerator,
    HierarchicalStructuredCommunicationEvaluator,
    HierarchicalStructuredCommunicationRefiner
)

# Create agents with custom names
generator = HierarchicalStructuredCommunicationGenerator(agent_name="ContentGenerator")
evaluator = HierarchicalStructuredCommunicationEvaluator(agent_name="QualityEvaluator")
refiner = HierarchicalStructuredCommunicationRefiner(agent_name="ContentRefiner")

# Create framework with default supervisor
framework = HierarchicalStructuredCommunicationFramework(
    generators=[generator],
    evaluators=[evaluator],
    refiners=[refiner],
    max_loops=3,
    verbose=True
)

# Execute task
result = framework.run("Write a detailed report on renewable energy technologies")
print(result["final_result"])
```

## Advanced Configuration

```python
from swarms.structs.hierarchical_structured_communication_framework import (
    HierarchicalStructuredCommunicationFramework
)

# Create framework with custom configuration
framework = HierarchicalStructuredCommunicationFramework(
    name="AdvancedFramework",
    max_loops=5,
    enable_structured_communication=True,
    enable_hierarchical_evaluation=True,
    shared_memory=True,
    model_name="gpt-4o-mini",
    verbose=True
)

# Run with custom parameters
result = framework.run(
    "Analyze the impact of climate change on global agriculture",
    max_loops=3
)
```

## Integration with Other Swarms

```python
from swarms.structs.hierarchical_structured_communication_framework import (
    HierarchicalStructuredCommunicationFramework
)
from swarms.structs import AutoSwarmBuilder

# Use HierarchicalStructuredCommunicationFramework for content generation
framework = HierarchicalStructuredCommunicationFramework(
    max_loops=2,
    verbose=True
)

# Integrate with AutoSwarmBuilder
builder = AutoSwarmBuilder()
swarm = builder.create_swarm(
    swarm_type="HierarchicalStructuredCommunicationFramework",
    task="Generate a comprehensive business plan"
)
```

## API Reference

### HierarchicalStructuredCommunicationFramework

The main orchestrator class that implements the complete framework.

#### Parameters
- `name` (str): Name of the framework
- `supervisor`: Main supervisor agent
- `generators` (List): List of generator agents
- `evaluators` (List): List of evaluator agents
- `refiners` (List): List of refiner agents
- `max_loops` (int): Maximum refinement loops
- `enable_structured_communication` (bool): Enable structured protocol
- `enable_hierarchical_evaluation` (bool): Enable hierarchical evaluation
- `verbose` (bool): Enable verbose logging

#### Methods
- `run(task)`: Execute complete workflow
- `step(task)`: Execute single workflow step
- `send_structured_message()`: Send structured communication
- `run_hierarchical_evaluation()`: Run evaluation system

## Contributing

Contributions to improve the Hierarchical Structured Communication Framework are welcome! Please:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure all imports use the correct module paths

## License

This framework is part of the Swarms project and follows the same licensing terms. 
